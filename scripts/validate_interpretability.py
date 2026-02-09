#!/usr/bin/env python3
"""
A4 Validation: Interpretability vs Gradient Attribution.

Compares three interpretability methods for BEACON:
  1. BEACON attention weights (free from forward pass)
  2. Input x gradient of profile reconstruction loss (one backward pass)
  3. Per-TF gradient of occupancy-weighted TF logit (existing method)

For each method, extracts motif PWMs per TF, compares to JASPAR ground truth,
and measures computational cost. Produces a comparison table and bar chart.
"""

import sys
import json
import time
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


# ---------------------------------------------------------------------------
# JASPAR loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Motif comparison
# ---------------------------------------------------------------------------

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
        denom = np.sqrt((d_c ** 2).sum() * (r_c ** 2).sum())
        if denom < 1e-8:
            continue

        r_val = (d_c * r_c).sum() / denom

        # Reverse complement
        d_rc = discovered[d_start:d_end][::-1, ::-1].flatten()[:min_len]
        d_rc_c = d_rc - d_rc.mean()
        denom_rc = np.sqrt((d_rc_c ** 2).sum() * (r_c ** 2).sum())
        if denom_rc > 1e-8:
            r_val = max(r_val, (d_rc_c * r_c).sum() / denom_rc)

        best_r = max(best_r, r_val)

    return best_r


# ---------------------------------------------------------------------------
# Motif extraction helpers
# ---------------------------------------------------------------------------

def importance_to_pwm(importance, window=20, top_k=10):
    """Extract a PWM from an importance map by finding high-importance peaks."""
    total_imp = np.abs(importance).sum(axis=1)

    kernel = np.ones(3) / 3
    smoothed = np.convolve(total_imp, kernel, mode="same")

    threshold = np.percentile(smoothed, 90)
    peaks = []
    for i in range(2, len(smoothed) - 2):
        if (
            smoothed[i] > threshold
            and smoothed[i] >= smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
            and smoothed[i] >= smoothed[i - 2]
            and smoothed[i] >= smoothed[i + 2]
        ):
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

    weights = np.array(weights)
    weights /= weights.sum()

    avg_window = np.zeros((window, 4))
    for w, weight in zip(all_windows, weights):
        avg_window += w * weight

    pos_imp = np.maximum(avg_window, 0)
    row_sums = pos_imp.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pwm = pos_imp / row_sums

    info_content = np.log2(4) + np.sum(pwm * np.log2(pwm + 1e-8), axis=1)
    informative = info_content > 0.3

    if informative.sum() < 4:
        return pwm

    first = np.argmax(informative)
    last = len(informative) - 1 - np.argmax(informative[::-1])
    return pwm[first : last + 1]


def compute_info_content(pwm):
    """Compute mean information content of a PWM in bits."""
    if pwm is None:
        return 0.0
    ic = np.log2(4) + np.sum(pwm * np.log2(pwm + 1e-8), axis=1)
    return float(ic.mean())


# ---------------------------------------------------------------------------
# Method 1: BEACON attention weights
# ---------------------------------------------------------------------------

def attention_importance(model, sequence, tf_idx, device):
    """
    Extract importance map from slot attention weights.

    Uses the attention pattern of the highest-occupancy slot whose TF
    prediction matches tf_idx.  The importance at each position is the
    attention weight multiplied by the one-hot sequence, giving an
    attention-weighted sequence matrix.
    """
    seq = sequence.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(seq)

    attention = outputs["attention"][0].detach().cpu().numpy()   # [K, L]
    occupancy = outputs["occupancy"][0, :, 0].detach().cpu().numpy()  # [K]
    tf_logits = outputs["tf_logits"][0].detach().cpu().numpy()   # [K, N_TFs]

    seq_np = sequence.cpu().numpy()  # [L, 4]

    # Find the highest-occupancy slot that predicts the target TF
    tf_preds = tf_logits.argmax(axis=1)  # [K]
    matching_slots = np.where(tf_preds == tf_idx)[0]

    if len(matching_slots) == 0:
        # Fall back to the overall highest-occupancy slot
        best_slot = int(np.argmax(occupancy))
    else:
        occ_vals = occupancy[matching_slots]
        best_slot = matching_slots[int(np.argmax(occ_vals))]

    attn_weights = attention[best_slot]  # [L]

    # Weight the one-hot sequence by attention to form an importance map
    importance = seq_np * attn_weights[:, None]  # [L, 4]
    return importance


# ---------------------------------------------------------------------------
# Method 2: Input x gradient of profile loss
# ---------------------------------------------------------------------------

def profile_gradient_importance(model, sequence, target_profile, device):
    """
    Compute input x gradient using the profile reconstruction loss.

    The gradient of the MSE between predicted and target profile w.r.t.
    the input sequence gives a task-agnostic importance map.
    """
    seq = sequence.unsqueeze(0).to(device).requires_grad_(True)
    target = target_profile.unsqueeze(0).to(device)

    outputs = model(seq)
    pred_profile = outputs["profile"]  # [1, L]

    loss = F.mse_loss(pred_profile, target)
    loss.backward()

    importance = (seq.grad[0] * seq.data[0]).detach().cpu().numpy()  # [L, 4]
    return importance


# ---------------------------------------------------------------------------
# Method 3: Per-TF gradient of occupancy-weighted TF logit
# ---------------------------------------------------------------------------

def per_tf_gradient_importance(model, sequence, tf_idx, device):
    """
    Compute input x gradient for the occupancy-weighted TF logit.

    This is the existing method from per_tf_gradient_motifs.py.
    """
    seq = sequence.unsqueeze(0).to(device).requires_grad_(True)

    outputs = model(seq)
    tf_logits = outputs["tf_logits"][0]  # [K, N_TFs]
    occupancy = outputs["occupancy"][0, :, 0]  # [K]

    tf_score = (tf_logits[:, tf_idx] * occupancy).sum()
    tf_score.backward()

    importance = (seq.grad[0] * seq.data[0]).detach().cpu().numpy()  # [L, 4]
    return importance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A4 Validation: Interpretability vs Gradient Attribution"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/phase10_slot_ablation/slots_16/"
            "beacon_20260203_231900/best_model.pt"
        ),
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/multi_tf_k562/test.h5"),
        help="Path to test HDF5 file",
    )
    parser.add_argument(
        "--jaspar",
        type=Path,
        default=Path("data/raw/jaspar/meme_vertebrates.txt"),
        help="Path to JASPAR MEME file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/a4_interpretability_validation"),
        help="Directory for output files",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum samples per TF",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Resolve relative paths
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path

    test_data_path = args.test_data
    if not test_data_path.is_absolute():
        test_data_path = base_dir / test_data_path

    jaspar_path = args.jaspar
    if not jaspar_path.is_absolute():
        jaspar_path = base_dir / jaspar_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("A4 Validation: Interpretability vs Gradient Attribution")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Test data  : {test_data_path}")
    print(f"JASPAR     : {jaspar_path}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {device}")
    print(f"Max samples: {args.max_samples}")

    print("\n--- Loading Model ---")
    model = BEACON(
        seq_len=2000,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=16,
        slot_dim=128,
        n_iterations=3,
        n_tfs=7,
        position_mode="gaussian",
        dropout=0.0,
        attention_mode="independent",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if unexpected:
        print(f"  Skipped {len(unexpected)} unexpected keys (Phase 1 heads not used here)")
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} parameters on {device}")

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    print("\n--- Loading Test Data ---")
    dataset = BEACONDataset(
        str(test_data_path),
        seq_length=2000,
        augment=False,
        n_slots=16,
        extract_peaks=True,
    )
    print(f"  {len(dataset)} test samples")

    # Organize samples by TF
    tf_samples = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        tf_idx = sample["tf_index"].item() if "tf_index" in sample else -1
        if 0 <= tf_idx < len(TF_NAMES) and len(tf_samples[tf_idx]) < args.max_samples:
            tf_samples[tf_idx].append(sample)

    for tf_idx in sorted(tf_samples.keys()):
        print(f"  {TF_NAMES[tf_idx]}: {len(tf_samples[tf_idx])} samples")

    # ------------------------------------------------------------------
    # 3. Load JASPAR reference motifs
    # ------------------------------------------------------------------
    print("\n--- Loading JASPAR Reference Motifs ---")
    jaspar_motifs = {}
    if jaspar_path.exists():
        all_motifs = load_jaspar_meme(str(jaspar_path))
        for tf in TF_NAMES:
            if tf in all_motifs:
                jaspar_motifs[tf] = all_motifs[tf]
                print(f"  {tf}: {all_motifs[tf].shape[0]}bp motif")
            else:
                for name, motif in all_motifs.items():
                    if tf.upper() in name.upper():
                        jaspar_motifs[tf] = motif
                        print(f"  {tf} (matched as {name}): {motif.shape[0]}bp motif")
                        break
                else:
                    print(f"  {tf}: not found in JASPAR")
    else:
        print(f"  WARNING: JASPAR file not found at {jaspar_path}")

    # ------------------------------------------------------------------
    # 4. Run all three methods
    # ------------------------------------------------------------------
    method_names = [
        "attention",
        "profile_gradient",
        "per_tf_gradient",
    ]
    method_labels = [
        "BEACON Attention",
        "Profile Gradient",
        "Per-TF Gradient",
    ]

    # Storage: method -> tf_name -> list of importance maps
    all_importance = {m: defaultdict(list) for m in method_names}
    # Timing: method -> list of per-sample durations
    timing = {m: [] for m in method_names}

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx]
        samples = tf_samples[tf_idx]
        n = len(samples)
        print(f"\n--- Processing {tf_name} ({n} samples) ---")

        for si, sample in enumerate(samples):
            if (si + 1) % 50 == 0 or si == 0:
                print(f"  sample {si + 1}/{n}")

            seq = sample["sequence"]  # [L, 4]
            profile = sample.get("profile", None)

            # Method 1: Attention (forward only, no grad)
            model.zero_grad()
            t0 = time.perf_counter()
            imp_attn = attention_importance(model, seq, tf_idx, device)
            t1 = time.perf_counter()
            timing["attention"].append((t1 - t0) * 1000)
            all_importance["attention"][tf_name].append(imp_attn)

            # Method 2: Profile gradient (needs target profile)
            model.zero_grad()
            t0 = time.perf_counter()
            if profile is not None:
                imp_prof = profile_gradient_importance(model, seq, profile, device)
            else:
                # If no profile available, use predicted profile as pseudo-target
                with torch.no_grad():
                    pseudo = model(seq.unsqueeze(0).to(device))["profile"][0].cpu()
                imp_prof = profile_gradient_importance(model, seq, pseudo, device)
            t1 = time.perf_counter()
            timing["profile_gradient"].append((t1 - t0) * 1000)
            all_importance["profile_gradient"][tf_name].append(imp_prof)

            # Method 3: Per-TF gradient
            model.zero_grad()
            t0 = time.perf_counter()
            imp_tf = per_tf_gradient_importance(model, seq, tf_idx, device)
            t1 = time.perf_counter()
            timing["per_tf_gradient"].append((t1 - t0) * 1000)
            all_importance["per_tf_gradient"][tf_name].append(imp_tf)

    # ------------------------------------------------------------------
    # 5. Extract motifs and compute metrics
    # ------------------------------------------------------------------
    print("\n--- Extracting Motifs and Computing Metrics ---")

    # results[method][tf_name] = { pearson_r, ic, motif_len, ... }
    results = {m: {} for m in method_names}

    for method in method_names:
        for tf_name in TF_NAMES:
            imp_list = all_importance[method].get(tf_name, [])
            if not imp_list:
                continue

            avg_imp = np.mean(imp_list, axis=0)  # [L, 4]
            motif = importance_to_pwm(avg_imp, window=24, top_k=15)

            if motif is None:
                results[method][tf_name] = {
                    "pearson_r": 0.0,
                    "info_content": 0.0,
                    "motif_length": 0,
                    "n_samples": len(imp_list),
                }
                continue

            pearson_r = 0.0
            if tf_name in jaspar_motifs:
                pearson_r = compare_motifs(motif, jaspar_motifs[tf_name])

            ic = compute_info_content(motif)

            results[method][tf_name] = {
                "pearson_r": float(pearson_r),
                "info_content": float(ic),
                "motif_length": int(motif.shape[0]),
                "n_samples": len(imp_list),
            }

    # ------------------------------------------------------------------
    # 6. Compute summary statistics
    # ------------------------------------------------------------------
    summary = {}
    for mi, method in enumerate(method_names):
        rs = [
            results[method][tf]["pearson_r"]
            for tf in TF_NAMES
            if tf in results[method] and results[method][tf]["pearson_r"] > 0
        ]
        ics = [
            results[method][tf]["info_content"]
            for tf in TF_NAMES
            if tf in results[method]
        ]
        t_list = timing[method]
        summary[method] = {
            "label": method_labels[mi],
            "mean_pearson_r": float(np.mean(rs)) if rs else 0.0,
            "median_pearson_r": float(np.median(rs)) if rs else 0.0,
            "max_pearson_r": float(np.max(rs)) if rs else 0.0,
            "n_tfs_above_0.5": int(sum(1 for r in rs if r > 0.5)),
            "n_tfs_above_0.6": int(sum(1 for r in rs if r > 0.6)),
            "mean_ic": float(np.mean(ics)) if ics else 0.0,
            "mean_time_ms": float(np.mean(t_list)) if t_list else 0.0,
            "median_time_ms": float(np.median(t_list)) if t_list else 0.0,
            "n_tfs_evaluated": len(rs),
        }

    # ------------------------------------------------------------------
    # 7. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RESULTS: Interpretability Method Comparison")
    print("=" * 90)

    # Per-TF table
    header = f"{'TF':<8}"
    for mi, method in enumerate(method_names):
        header += f" | {method_labels[mi]:>18s}"
    print(f"\n{header}")
    print("-" * len(header))

    print("JASPAR Pearson r:")
    for tf in TF_NAMES:
        row = f"  {tf:<6}"
        for method in method_names:
            if tf in results[method]:
                r = results[method][tf]["pearson_r"]
                row += f" | {r:>18.3f}"
            else:
                row += f" | {'N/A':>18s}"
        print(row)

    print(f"\nInformation Content (bits):")
    for tf in TF_NAMES:
        row = f"  {tf:<6}"
        for method in method_names:
            if tf in results[method]:
                ic = results[method][tf]["info_content"]
                row += f" | {ic:>18.3f}"
            else:
                row += f" | {'N/A':>18s}"
        print(row)

    # Summary table
    print(f"\n{'':=<90}")
    print("SUMMARY")
    print(f"{'':=<90}")
    print(f"{'Metric':<25}", end="")
    for mi, method in enumerate(method_names):
        print(f" | {method_labels[mi]:>18s}", end="")
    print()
    print("-" * 85)

    metrics_to_show = [
        ("Mean Pearson r", "mean_pearson_r"),
        ("Median Pearson r", "median_pearson_r"),
        ("Max Pearson r", "max_pearson_r"),
        ("TFs with r > 0.5", "n_tfs_above_0.5"),
        ("TFs with r > 0.6", "n_tfs_above_0.6"),
        ("Mean IC (bits)", "mean_ic"),
        ("Mean time (ms/sample)", "mean_time_ms"),
        ("Median time (ms/sample)", "median_time_ms"),
    ]

    for label, key in metrics_to_show:
        row = f"  {label:<23}"
        for method in method_names:
            val = summary[method][key]
            if isinstance(val, int):
                row += f" | {val:>18d}"
            else:
                row += f" | {val:>18.3f}"
        print(row)

    # Speed-up relative to per-TF gradient
    base_time = summary["per_tf_gradient"]["mean_time_ms"]
    if base_time > 0:
        print(f"\nSpeedup vs Per-TF Gradient:")
        for method in method_names:
            t = summary[method]["mean_time_ms"]
            if t > 0:
                speedup = base_time / t
                print(f"  {summary[method]['label']}: {speedup:.1f}x")

    # ------------------------------------------------------------------
    # 8. Save results JSON
    # ------------------------------------------------------------------
    save_data = {
        "validation": "A4_interpretability_vs_gradient_attribution",
        "checkpoint": str(checkpoint_path),
        "test_data": str(test_data_path),
        "max_samples_per_tf": args.max_samples,
        "methods": method_labels,
        "per_tf_results": {},
        "summary": summary,
    }

    for tf in TF_NAMES:
        tf_data = {}
        for method in method_names:
            if tf in results[method]:
                tf_data[method] = results[method][tf]
        save_data["per_tf_results"][tf] = tf_data

    results_path = output_dir / "a4_interpretability_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 9. Create matplotlib visualization
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: JASPAR Pearson r per TF per method ---
    ax = axes[0]
    x = np.arange(len(TF_NAMES))
    width = 0.25
    colors = ["#4e79a7", "#e15759", "#59a14f"]

    for mi, method in enumerate(method_names):
        vals = []
        for tf in TF_NAMES:
            if tf in results[method]:
                vals.append(results[method][tf]["pearson_r"])
            else:
                vals.append(0.0)
        ax.bar(x + mi * width, vals, width, label=method_labels[mi], color=colors[mi])

    ax.set_xlabel("Transcription Factor")
    ax.set_ylabel("JASPAR Pearson r")
    ax.set_title("Motif Recovery (JASPAR Correlation)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(TF_NAMES, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # --- Panel B: Mean Pearson r and timing ---
    ax = axes[1]
    mean_rs = [summary[m]["mean_pearson_r"] for m in method_names]
    bar_positions = np.arange(len(method_names))
    bars = ax.bar(bar_positions, mean_rs, color=colors)
    ax.set_ylabel("Mean JASPAR Pearson r")
    ax.set_title("Mean Motif Recovery")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([l.replace(" ", "\n") for l in method_labels], fontsize=9)
    ax.set_ylim(0, max(mean_rs) * 1.3 if max(mean_rs) > 0 else 1.0)
    for bar, val in zip(bars, mean_rs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- Panel C: Timing comparison ---
    ax = axes[2]
    mean_times = [summary[m]["mean_time_ms"] for m in method_names]
    bars = ax.bar(bar_positions, mean_times, color=colors)
    ax.set_ylabel("Time (ms / sample)")
    ax.set_title("Computational Cost")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([l.replace(" ", "\n") for l in method_labels], fontsize=9)
    for bar, val in zip(bars, mean_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_times) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "A4 Validation: Interpretability vs Gradient Attribution",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig_path = output_dir / "a4_interpretability_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
