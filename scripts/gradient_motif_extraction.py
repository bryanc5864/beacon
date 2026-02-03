#!/usr/bin/env python3
"""
Per-slot gradient-based motif extraction for BEACON.

Instead of crude attention-weighted nucleotide frequencies (r=0.457),
this computes gradient of each slot's occupancy w.r.t. the input sequence
to identify which bases drive each slot's activation. Then builds PWMs
from high-importance regions and compares to JASPAR.

Target: motif correlation r > 0.6
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


def load_jaspar_motifs(jaspar_path):
    """Load JASPAR PFMs from meme format."""
    motifs = {}
    current_tf = None
    current_matrix = []

    with open(jaspar_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if current_tf and current_matrix:
                    motifs[current_tf] = np.array(current_matrix)
                parts = line.split()
                motif_id = parts[1] if len(parts) > 1 else ""
                motif_name = parts[2] if len(parts) > 2 else parts[1]
                # Match to our TF names
                for tf in TF_NAMES:
                    if tf.upper() in motif_name.upper():
                        current_tf = tf
                        break
                else:
                    current_tf = motif_name
                current_matrix = []
            elif line.startswith("letter-probability"):
                current_matrix = []
            elif current_tf and line and not line.startswith("URL") and not line.startswith("ALPHABET"):
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 4:
                        current_matrix.append(values)
                except ValueError:
                    pass

    if current_tf and current_matrix:
        motifs[current_tf] = np.array(current_matrix)

    return motifs


def compute_slot_gradients(model, sequence, slot_idx, device):
    """
    Compute gradient of slot occupancy w.r.t. input sequence.

    Returns importance scores [L, 4] (gradient * input).
    """
    sequence = sequence.unsqueeze(0).to(device).requires_grad_(True)

    outputs = model(sequence)
    occupancy = outputs["occupancy"][0, slot_idx, 0]  # scalar

    occupancy.backward()

    # Gradient * input gives per-base importance
    importance = (sequence.grad[0] * sequence.data[0]).detach().cpu().numpy()  # [L, 4]
    return importance


def compute_tf_logit_gradients(model, sequence, slot_idx, tf_idx, device):
    """
    Compute gradient of slot's TF logit w.r.t. input sequence.

    More specific than occupancy — tells us which bases make this slot
    predict a specific TF.
    """
    sequence = sequence.unsqueeze(0).to(device).requires_grad_(True)

    outputs = model(sequence)
    tf_logit = outputs["tf_logits"][0, slot_idx, tf_idx]

    tf_logit.backward()

    importance = (sequence.grad[0] * sequence.data[0]).detach().cpu().numpy()
    return importance


def extract_motif_from_importance(importance, window=20, top_k=5):
    """
    Extract PWM from importance scores.

    1. Find top-k importance peaks
    2. Extract windows around peaks
    3. Build PWM from importance-weighted nucleotide preferences
    """
    # Sum importance across channels for peak finding
    total_importance = np.abs(importance).sum(axis=1)  # [L]

    # Smooth for peak finding
    kernel = np.ones(5) / 5
    smoothed = np.convolve(total_importance, kernel, mode='same')

    # Find peaks (local maxima above threshold)
    threshold = np.percentile(smoothed, 95)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            peaks.append((smoothed[i], i))

    peaks.sort(reverse=True)
    peaks = peaks[:top_k]

    if not peaks:
        return None

    # Extract windows and build weighted PWM
    half_w = window // 2
    pwm_accumulator = np.zeros((window, 4))
    weight_sum = 0

    for score, pos in peaks:
        start = max(0, pos - half_w)
        end = min(len(importance), pos + half_w)
        actual_start = half_w - (pos - start)
        actual_end = actual_start + (end - start)

        # Use absolute importance as weight
        window_importance = np.abs(importance[start:end])
        pwm_accumulator[actual_start:actual_end] += window_importance * score
        weight_sum += score

    if weight_sum > 0:
        pwm_accumulator /= weight_sum

    # Convert to probability (softmax-like normalization)
    # Use positive importance values
    pwm_pos = np.maximum(pwm_accumulator, 0)
    row_sums = pwm_pos.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pwm = pwm_pos / row_sums

    # Trim uninformative positions (near-uniform)
    info_content = np.sum(pwm * np.log2(pwm + 1e-8) + np.log2(4), axis=1)
    informative = info_content > 0.1
    if informative.sum() < 4:
        return pwm  # Return full if too few informative positions

    # Find contiguous informative region
    first = np.argmax(informative)
    last = len(informative) - 1 - np.argmax(informative[::-1])
    return pwm[first:last+1]


def compare_motifs(discovered, reference):
    """
    Compare discovered motif to reference using best-alignment Pearson correlation.
    """
    if discovered is None or reference is None:
        return 0.0, 0

    d_len = len(discovered)
    r_len = len(reference)

    best_r = 0.0
    best_offset = 0

    # Slide shorter along longer
    for offset in range(-(d_len - 4), r_len - 4 + 1):
        # Overlap region
        d_start = max(0, -offset)
        d_end = min(d_len, r_len - offset)
        r_start = max(0, offset)
        r_end = min(r_len, offset + d_len)

        overlap = d_end - d_start
        if overlap < 4:
            continue

        d_slice = discovered[d_start:d_end].flatten()
        r_slice = reference[r_start:r_end].flatten()

        if len(d_slice) != len(r_slice):
            min_len = min(len(d_slice), len(r_slice))
            d_slice = d_slice[:min_len]
            r_slice = r_slice[:min_len]

        # Pearson correlation
        d_centered = d_slice - d_slice.mean()
        r_centered = r_slice - r_slice.mean()
        denom = np.sqrt((d_centered ** 2).sum() * (r_centered ** 2).sum())
        if denom < 1e-8:
            continue
        r_val = (d_centered * r_centered).sum() / denom

        # Also try reverse complement
        d_rc = discovered[d_start:d_end][::-1, ::-1].flatten()
        d_rc_centered = d_rc - d_rc.mean()
        denom_rc = np.sqrt((d_rc_centered ** 2).sum() * (r_centered ** 2).sum())
        if denom_rc > 1e-8:
            r_rc = (d_rc_centered * r_centered).sum() / denom_rc
            r_val = max(r_val, r_rc)

        if r_val > best_r:
            best_r = r_val
            best_offset = offset

    return best_r, best_offset


def main():
    parser = argparse.ArgumentParser(description="Gradient-based motif extraction")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--jaspar", type=Path,
                        default=Path("data/raw/jaspar/meme_vertebrates.txt"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max samples per TF for gradient computation")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Resolve paths
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

    if checkpoint_path is None:
        print(f"ERROR: No best_model.pt found in {experiment_dir}")
        return 1

    print("=" * 60)
    print("Gradient-Based Motif Extraction")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    # Output dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = checkpoint_path.parent / "gradient_motifs"
    elif not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Try parent
        for p in [checkpoint_path.parent.parent / "config.json"]:
            if p.exists():
                with open(p) as f:
                    config = json.load(f)
                break
        else:
            config = {}

    n_tfs = config.get("n_tfs", config.get("num_tfs", 7))
    seq_len = config.get("seq_len", config.get("seq_length", 2000))
    n_slots = config.get("n_slots", config.get("num_slots", 16))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\n--- Loading Model ---")
    model = BEACON(
        seq_len=seq_len, input_channels=4,
        backbone_type="dilated",
        backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=n_slots,
        slot_dim=config.get("slot_dim", 128),
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
    print(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} params")

    # Load test data
    print(f"\n--- Loading Test Data ---")
    test_path = data_dir / "test.h5"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found")
        return 1

    dataset = BEACONDataset(str(test_path), seq_length=seq_len,
                            augment=False, n_slots=n_slots, extract_peaks=True)
    print(f"  {len(dataset)} test samples")

    # Load JASPAR motifs
    print(f"\n--- Loading JASPAR Motifs ---")
    if jaspar_path.exists():
        jaspar_motifs = load_jaspar_motifs(str(jaspar_path))
        print(f"  Loaded motifs for: {list(jaspar_motifs.keys())}")
    else:
        print(f"  WARNING: {jaspar_path} not found, skipping comparison")
        jaspar_motifs = {}

    # Step 1: Determine dominant TF per slot
    print(f"\n--- Determining Slot-TF Assignments ---")
    slot_tf_counts = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for i in range(min(len(dataset), 500)):
            sample = dataset[i]
            seq = sample["sequence"].unsqueeze(0).to(device)
            outputs = model(seq)

            occupancy = outputs["occupancy"][0, :, 0].cpu().numpy()
            tf_preds = outputs["tf_logits"][0].argmax(dim=-1).cpu().numpy()
            target_tf = sample["tf_index"].item() if "tf_index" in sample else -1

            for k in range(n_slots):
                if occupancy[k] > 0.3:
                    slot_tf_counts[k][tf_preds[k]] += 1

    # Find dominant TF per slot
    slot_dominant_tf = {}
    for k in range(n_slots):
        if slot_tf_counts[k]:
            dominant = max(slot_tf_counts[k].items(), key=lambda x: x[1])
            slot_dominant_tf[k] = dominant[0]
            total = sum(slot_tf_counts[k].values())
            print(f"  Slot {k}: dominant TF = {TF_NAMES[dominant[0]]} "
                  f"({dominant[1]}/{total} = {dominant[1]/total:.1%})")
        else:
            print(f"  Slot {k}: inactive")

    # Step 2: Collect samples per TF
    print(f"\n--- Collecting Samples Per TF ---")
    tf_samples = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        tf_idx = sample["tf_index"].item() if "tf_index" in sample else -1
        if tf_idx >= 0 and len(tf_samples[tf_idx]) < args.max_samples:
            tf_samples[tf_idx].append(sample["sequence"])

    for tf_idx, samples in tf_samples.items():
        print(f"  {TF_NAMES[tf_idx]}: {len(samples)} samples")

    # Step 3: Compute gradients for each slot's dominant TF
    print(f"\n--- Computing Gradient Motifs ---")
    slot_motifs = {}

    for k, tf_idx in slot_dominant_tf.items():
        tf_name = TF_NAMES[tf_idx]
        samples = tf_samples.get(tf_idx, [])
        if not samples:
            print(f"  Slot {k} ({tf_name}): no samples, skipping")
            continue

        print(f"  Slot {k} ({tf_name}): computing gradients over {len(samples)} samples...")

        # Accumulate importance maps
        all_importance = []
        n_computed = 0

        for seq_tensor in samples[:args.max_samples]:
            try:
                model.zero_grad()
                # Use TF-logit gradient (more specific than occupancy)
                importance = compute_tf_logit_gradients(
                    model, seq_tensor, k, tf_idx, device
                )
                all_importance.append(importance)
                n_computed += 1
            except Exception as e:
                continue

        if not all_importance:
            print(f"    No valid gradients")
            continue

        # Average importance across samples
        avg_importance = np.mean(all_importance, axis=0)

        # Extract motif from averaged importance
        motif = extract_motif_from_importance(avg_importance, window=24, top_k=10)

        if motif is not None:
            slot_motifs[k] = {
                "motif": motif,
                "tf_name": tf_name,
                "tf_idx": tf_idx,
                "n_samples": n_computed,
            }
            print(f"    Extracted motif: {motif.shape[0]}bp")
        else:
            print(f"    Failed to extract motif")

    # Step 4: Compare to JASPAR
    print(f"\n--- Comparing to JASPAR ---")
    results = {}

    for k, info in slot_motifs.items():
        tf_name = info["tf_name"]
        discovered = info["motif"]

        if tf_name in jaspar_motifs:
            reference = jaspar_motifs[tf_name]
            correlation, offset = compare_motifs(discovered, reference)
            results[k] = {
                "slot": k,
                "tf_name": tf_name,
                "correlation": float(correlation),
                "offset": offset,
                "discovered_length": len(discovered),
                "reference_length": len(reference),
                "n_samples": info["n_samples"],
            }
            print(f"  Slot {k} ({tf_name}): r = {correlation:.3f} "
                  f"(discovered {len(discovered)}bp vs reference {len(reference)}bp)")
        else:
            print(f"  Slot {k} ({tf_name}): no JASPAR reference found")
            results[k] = {
                "slot": k,
                "tf_name": tf_name,
                "correlation": None,
                "n_samples": info["n_samples"],
            }

    # Summary
    print(f"\n--- Summary ---")
    valid_correlations = [r["correlation"] for r in results.values()
                         if r["correlation"] is not None]

    if valid_correlations:
        mean_r = np.mean(valid_correlations)
        max_r = np.max(valid_correlations)
        print(f"  Mean correlation: {mean_r:.3f}")
        print(f"  Max correlation:  {max_r:.3f}")
        print(f"  Slots with r > 0.5: {sum(1 for r in valid_correlations if r > 0.5)}")
        print(f"  Slots with r > 0.6: {sum(1 for r in valid_correlations if r > 0.6)}")

        # Improvement over baseline
        print(f"\n  Previous method (attention-weighted): mean r = 0.457")
        print(f"  Gradient method:                      mean r = {mean_r:.3f}")
        if mean_r > 0.457:
            print(f"  Improvement: +{mean_r - 0.457:.3f}")

    # Save results
    save_results = {
        "method": "gradient_motif_extraction",
        "description": "Per-slot gradient of TF logit w.r.t. input, averaged over samples",
        "per_slot": {},
        "summary": {
            "mean_correlation": float(mean_r) if valid_correlations else None,
            "max_correlation": float(max_r) if valid_correlations else None,
            "n_slots_evaluated": len(results),
        },
    }
    for k, r in results.items():
        save_results["per_slot"][str(k)] = r
        # Save motif as list for JSON
        if k in slot_motifs:
            save_results["per_slot"][str(k)]["motif_pwm"] = slot_motifs[k]["motif"].tolist()

    with open(output_dir / "gradient_motif_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Save motifs as numpy
    for k, info in slot_motifs.items():
        np.save(output_dir / f"slot_{k}_{info['tf_name']}_motif.npy", info["motif"])

    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
