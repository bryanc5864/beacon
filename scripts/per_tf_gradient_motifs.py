#!/usr/bin/env python3
"""
Per-TF gradient-based motif extraction for BEACON.

Instead of per-slot gradients (which fail due to single-slot dominance),
this computes gradient of the model's TF classification output w.r.t. input
for each TF class. For each test sample labeled as TF X, compute gradient
of the logit for class X. Aggregate across samples to build TF-specific
importance maps and extract PWMs.

This works regardless of how many slots are active.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]

# JASPAR matrix IDs for our TFs
JASPAR_IDS = {
    "CTCF": "MA0139",
    "GATA1": "MA0035",
    "TAL1": "MA0140",
    "MYC": "MA0147",
    "MAX": "MA0058",
    "SPI1": "MA0080",
    "CEBPB": "MA0466",
}


def load_jaspar_pfm(pfm_path):
    """Load JASPAR PFMs from pfm_vertebrates.txt format."""
    motifs = {}
    current_id = None
    current_name = None
    current_rows = []

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous
                if current_name and current_rows:
                    # PFM format: each line is one base (A, C, G, T)
                    # Transpose to get [positions, 4]
                    matrix = np.array(current_rows)
                    if matrix.shape[0] == 4:
                        matrix = matrix.T  # [L, 4]
                    # Normalize rows
                    row_sums = matrix.sum(axis=1, keepdims=True)
                    row_sums = np.maximum(row_sums, 1e-8)
                    matrix = matrix / row_sums
                    motifs[current_name] = matrix

                # Parse header: >MA0139.1 CTCF
                parts = line[1:].split()
                current_id = parts[0] if parts else ""
                current_name = parts[1] if len(parts) > 1 else parts[0]
                current_rows = []
            elif line and current_id:
                try:
                    values = [float(x) for x in line.split()]
                    if values:
                        current_rows.append(values)
                except ValueError:
                    pass

    # Save last
    if current_name and current_rows:
        matrix = np.array(current_rows)
        if matrix.shape[0] == 4:
            matrix = matrix.T
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-8)
        matrix = matrix / row_sums
        motifs[current_name] = matrix

    return motifs


def load_jaspar_meme(meme_path):
    """Load JASPAR motifs from MEME format."""
    motifs = {}
    current_name = None
    current_matrix = []
    in_matrix = False

    with open(meme_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if current_name and current_matrix:
                    motifs[current_name] = np.array(current_matrix)
                parts = line.split()
                current_name = parts[2] if len(parts) > 2 else parts[1]
                current_matrix = []
                in_matrix = False
            elif line.startswith("letter-probability"):
                in_matrix = True
                current_matrix = []
            elif in_matrix and line:
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 4:
                        current_matrix.append(values)
                except ValueError:
                    in_matrix = False
            elif not line:
                in_matrix = False

    if current_name and current_matrix:
        motifs[current_name] = np.array(current_matrix)

    return motifs


def compute_input_x_gradient(model, sequence, tf_idx, device):
    """
    Compute input × gradient for a specific TF class.

    Uses the max TF logit across all slots for the target TF,
    then backpropagates to get per-base importance.
    """
    seq = sequence.unsqueeze(0).to(device).requires_grad_(True)

    outputs = model(seq)
    tf_logits = outputs["tf_logits"][0]  # [K, N_TFs]
    occupancy = outputs["occupancy"][0, :, 0]  # [K]

    # Occupancy-weighted TF logit for target class across all slots
    tf_score = (tf_logits[:, tf_idx] * occupancy).sum()

    tf_score.backward()

    # Input × gradient
    importance = (seq.grad[0] * seq.data[0]).detach().cpu().numpy()  # [L, 4]
    return importance


def extract_motif_at_peak(importance, center, window=30):
    """Extract PWM from importance scores around a known peak position."""
    half_w = window // 2
    start = max(0, center - half_w)
    end = min(len(importance), center + half_w)

    region = importance[start:end]

    # Convert importance to PWM via positive values
    pos_imp = np.maximum(region, 0)
    row_sums = pos_imp.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pwm = pos_imp / row_sums

    # Trim low-information positions (IC = log2(4) + sum_b p(b)*log2(p(b)))
    # IC ranges from 0 (uniform) to 2 (perfectly conserved)
    info_content = np.log2(4) + np.sum(pwm * np.log2(pwm + 1e-8), axis=1)
    informative = info_content > 0.3

    if informative.sum() < 4:
        return pwm

    first = np.argmax(informative)
    last = len(informative) - 1 - np.argmax(informative[::-1])
    return pwm[first:last+1]


def extract_motif_denovo(importance, window=20, top_k=10):
    """Extract PWM from importance by finding top peaks de novo."""
    total_imp = np.abs(importance).sum(axis=1)

    # Smooth
    kernel = np.ones(3) / 3
    smoothed = np.convolve(total_imp, kernel, mode='same')

    # Find peaks above 90th percentile
    threshold = np.percentile(smoothed, 90)
    peaks = []
    for i in range(2, len(smoothed) - 2):
        if (smoothed[i] > threshold and
            smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i+1] and
            smoothed[i] >= smoothed[i-2] and smoothed[i] >= smoothed[i+2]):
            peaks.append((smoothed[i], i))

    peaks.sort(reverse=True)
    peaks = peaks[:top_k]

    if not peaks:
        return None

    half_w = window // 2
    all_windows = []
    weights = []

    for score, pos in peaks:
        start = max(0, pos - half_w)
        end = min(len(importance), pos + half_w)
        w = importance[start:end]
        if len(w) == window:
            all_windows.append(w)
            weights.append(score)

    if not all_windows:
        return None

    # Weighted average of windows
    weights = np.array(weights)
    weights /= weights.sum()

    avg_window = np.zeros((window, 4))
    for w, weight in zip(all_windows, weights):
        avg_window += w * weight

    # To PWM
    pos_imp = np.maximum(avg_window, 0)
    row_sums = pos_imp.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pwm = pos_imp / row_sums

    # Trim (IC = log2(4) + sum_b p(b)*log2(p(b)), range 0-2 bits)
    info_content = np.log2(4) + np.sum(pwm * np.log2(pwm + 1e-8), axis=1)
    informative = info_content > 0.3
    if informative.sum() < 4:
        return pwm

    first = np.argmax(informative)
    last = len(informative) - 1 - np.argmax(informative[::-1])
    return pwm[first:last+1]


def compare_motifs(discovered, reference):
    """Best-alignment Pearson correlation between two PWMs."""
    if discovered is None or reference is None:
        return 0.0

    d_len, r_len = len(discovered), len(reference)
    best_r = 0.0

    for offset in range(-(d_len - 4), r_len - 3):
        d_start = max(0, -offset)
        d_end = min(d_len, r_len - offset)
        r_start = max(0, offset)
        r_end = min(r_len, offset + d_len)

        overlap = d_end - d_start
        if overlap < 4:
            continue

        d_flat = discovered[d_start:d_end].flatten()
        r_flat = reference[r_start:r_end].flatten()

        min_len = min(len(d_flat), len(r_flat))
        d_flat, r_flat = d_flat[:min_len], r_flat[:min_len]

        d_c = d_flat - d_flat.mean()
        r_c = r_flat - r_flat.mean()
        denom = np.sqrt((d_c**2).sum() * (r_c**2).sum())
        if denom < 1e-8:
            continue

        r_val = (d_c * r_c).sum() / denom

        # Also try reverse complement
        d_rc = discovered[d_start:d_end][::-1, ::-1].flatten()[:min_len]
        d_rc_c = d_rc - d_rc.mean()
        denom_rc = np.sqrt((d_rc_c**2).sum() * (r_c**2).sum())
        if denom_rc > 1e-8:
            r_val = max(r_val, (d_rc_c * r_c).sum() / denom_rc)

        best_r = max(best_r, r_val)

    return best_r


def main():
    parser = argparse.ArgumentParser(description="Per-TF gradient motif extraction")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--jaspar", type=Path,
                        default=Path("data/raw/jaspar/meme_vertebrates.txt"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    jaspar_path = args.jaspar
    if not jaspar_path.is_absolute():
        jaspar_path = base_dir / jaspar_path

    # Find checkpoint
    checkpoint_path = None
    for subdir in sorted(experiment_dir.iterdir()):
        if subdir.is_dir():
            for nested in sorted(subdir.iterdir()):
                if nested.is_dir():
                    bp = nested / "best_model.pt"
                    if bp.exists():
                        checkpoint_path = bp
                        break
            if checkpoint_path is None:
                bp = subdir / "best_model.pt"
                if bp.exists():
                    checkpoint_path = bp
        if checkpoint_path:
            break

    if not checkpoint_path:
        print(f"ERROR: No best_model.pt in {experiment_dir}")
        return 1

    print("=" * 60)
    print("Per-TF Gradient Motif Extraction")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = checkpoint_path.parent / "per_tf_motifs"
    elif not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = checkpoint_path.parent / "config.json"
    config = {}
    for p in [config_path, checkpoint_path.parent.parent / "config.json"]:
        if p.exists():
            with open(p) as f:
                config = json.load(f)
            break

    n_tfs = config.get("n_tfs", config.get("num_tfs", 7))
    seq_len = config.get("seq_len", config.get("seq_length", 2000))
    n_slots = config.get("n_slots", config.get("num_slots", 16))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\n--- Loading Model ---")
    model = BEACON(
        seq_len=seq_len, input_channels=4, backbone_type="dilated",
        backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=n_slots, slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=n_tfs, position_mode="gaussian", dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params on {device}")

    # Load test data
    print(f"\n--- Loading Test Data ---")
    test_path = data_dir / "test.h5"
    dataset = BEACONDataset(str(test_path), seq_length=seq_len,
                            augment=False, n_slots=n_slots, extract_peaks=True)
    print(f"  {len(dataset)} test samples")

    # Organize by TF
    tf_samples = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        tf_idx = sample["tf_index"].item() if "tf_index" in sample else -1
        if tf_idx >= 0 and len(tf_samples[tf_idx]) < args.max_samples:
            tf_samples[tf_idx].append(sample)

    for tf_idx in sorted(tf_samples.keys()):
        print(f"  {TF_NAMES[tf_idx]}: {len(tf_samples[tf_idx])} samples")

    # Load JASPAR
    print(f"\n--- Loading JASPAR ---")
    jaspar_motifs = {}
    if jaspar_path.exists():
        if "meme" in jaspar_path.name:
            all_motifs = load_jaspar_meme(str(jaspar_path))
        else:
            all_motifs = load_jaspar_pfm(str(jaspar_path))

        for tf in TF_NAMES:
            if tf in all_motifs:
                jaspar_motifs[tf] = all_motifs[tf]
                print(f"  {tf}: {all_motifs[tf].shape[0]}bp motif")
            else:
                # Try case variations
                for name, motif in all_motifs.items():
                    if tf.upper() in name.upper():
                        jaspar_motifs[tf] = motif
                        print(f"  {tf} (matched as {name}): {motif.shape[0]}bp motif")
                        break
                else:
                    print(f"  {tf}: not found in JASPAR")

    # Compute per-TF gradient motifs
    print(f"\n--- Computing Per-TF Gradient Motifs ---")
    results = {}

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx]
        samples = tf_samples[tf_idx]
        print(f"\n  {tf_name} ({len(samples)} samples):")

        # Accumulate importance maps
        all_importance = []
        for sample in samples:
            try:
                model.zero_grad()
                imp = compute_input_x_gradient(model, sample["sequence"], tf_idx, device)
                all_importance.append(imp)
            except Exception as e:
                continue

        if not all_importance:
            print(f"    No valid gradients")
            continue

        print(f"    Computed {len(all_importance)} gradient maps")

        # Average importance
        avg_importance = np.mean(all_importance, axis=0)  # [L, 4]

        # Extract motif de novo from averaged importance
        motif = extract_motif_denovo(avg_importance, window=24, top_k=15)

        if motif is None:
            print(f"    Failed to extract motif")
            continue

        print(f"    Extracted {motif.shape[0]}bp motif")

        # Compare to JASPAR
        correlation = 0.0
        if tf_name in jaspar_motifs:
            correlation = compare_motifs(motif, jaspar_motifs[tf_name])
            print(f"    JASPAR correlation: r = {correlation:.3f}")
        else:
            print(f"    No JASPAR reference")

        # Information content (IC per position: 0 = uniform, 2 = perfectly conserved)
        ic = np.log2(4) + np.sum(motif * np.log2(motif + 1e-8), axis=1)
        print(f"    Mean IC: {ic.mean():.3f} bits, max IC: {ic.max():.3f} bits")

        results[tf_name] = {
            "tf_name": tf_name,
            "tf_idx": tf_idx,
            "n_samples": len(all_importance),
            "motif_length": int(motif.shape[0]),
            "jaspar_correlation": float(correlation),
            "mean_ic": float(ic.mean()),
            "max_ic": float(ic.max()),
            "motif_pwm": motif.tolist(),
        }

        # Save motif
        np.save(output_dir / f"{tf_name}_gradient_motif.npy", motif)

        # Also save raw average importance
        np.save(output_dir / f"{tf_name}_avg_importance.npy", avg_importance)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    correlations = [r["jaspar_correlation"] for r in results.values() if r["jaspar_correlation"] > 0]

    print(f"\n{'TF':<8} {'Length':>6} {'JASPAR r':>10} {'Mean IC':>8} {'Samples':>8}")
    print("-" * 45)
    for tf in TF_NAMES:
        if tf in results:
            r = results[tf]
            print(f"{tf:<8} {r['motif_length']:>6} {r['jaspar_correlation']:>10.3f} "
                  f"{r['mean_ic']:>8.3f} {r['n_samples']:>8}")

    if correlations:
        print(f"\nMean JASPAR correlation: {np.mean(correlations):.3f}")
        print(f"Max JASPAR correlation:  {np.max(correlations):.3f}")
        print(f"TFs with r > 0.5: {sum(1 for r in correlations if r > 0.5)}/{len(correlations)}")
        print(f"TFs with r > 0.6: {sum(1 for r in correlations if r > 0.6)}/{len(correlations)}")
        print(f"\nPrevious method (attention-weighted): mean r = 0.457")
        print(f"This method (per-TF gradient):        mean r = {np.mean(correlations):.3f}")

    # Save
    save_data = {
        "method": "per_tf_gradient_motif_extraction",
        "description": "Gradient of occupancy-weighted TF logit w.r.t. input, averaged per TF",
        "per_tf": results,
        "summary": {
            "mean_correlation": float(np.mean(correlations)) if correlations else None,
            "max_correlation": float(np.max(correlations)) if correlations else None,
            "n_tfs_evaluated": len(results),
            "n_tfs_with_jaspar": len(correlations),
        },
    }
    with open(output_dir / "per_tf_motif_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
