#!/usr/bin/env python3
"""
A1 Validation: Per-Base Attribution Head Speed and Accuracy

Compares three per-base importance methods for BEACON:
  1. BEACON AttributionHead (single forward pass, no grad)
  2. Gradient * input importance (backprop from profile loss, DeepSHAP approx)
  3. Per-TF gradient importance (existing method from per_tf_gradient_motifs.py)

For each method, measures wall-clock time per sample and Pearson correlation
with the per-TF gradient (c) as reference ground truth.  If the attribution
head is trained, also reports correlation between (a) and (b).

Produces a comparison table, results JSON, and a 3-panel matplotlib figure.
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
# Correlation helper
# ---------------------------------------------------------------------------

def pearson_correlation(a, b):
    """Compute Pearson correlation between two flat arrays."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((a_c * b_c).sum() / denom)


# ---------------------------------------------------------------------------
# Method (a): BEACON AttributionHead (single forward pass, no grad)
# ---------------------------------------------------------------------------

def attribution_head_importance(model, sequence, device):
    """
    Extract per-base importance from the AttributionHead.

    Uses a single forward pass with no gradients.  The attribution output
    is [B, K, L] per-slot importance.  We weight each slot by its occupancy
    to obtain a total importance [B, L], then multiply by the one-hot
    sequence to get [L, 4].
    """
    seq = sequence.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(seq)

    attribution = outputs["attribution"][0].detach().cpu().numpy()  # [K, L]
    occupancy = outputs["occupancy"][0, :, 0].detach().cpu().numpy()  # [K]
    seq_np = sequence.cpu().numpy()  # [L, 4]

    # Occupancy-weighted sum across slots -> [L]
    total_importance = np.einsum("k,kl->l", occupancy, attribution)

    # Multiply by one-hot sequence -> [L, 4]
    importance = seq_np * total_importance[:, None]
    return importance


# ---------------------------------------------------------------------------
# Method (b): Gradient * input importance (profile loss backprop)
# ---------------------------------------------------------------------------

def gradient_input_importance(model, sequence, target_profile, device):
    """
    Compute input x gradient using the profile reconstruction loss.

    Enable gradients on the input, forward pass, compute MSE loss between
    predicted and target profile, backward, importance = seq.grad * seq.data.
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
# Method (c): Per-TF gradient importance (reference / ground truth)
# ---------------------------------------------------------------------------

def per_tf_gradient_importance(model, sequence, tf_idx, device):
    """
    Compute input x gradient for the occupancy-weighted TF logit.

    This is the existing method from per_tf_gradient_motifs.py, used as
    the reference ground truth.
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
        description="A1 Validation: Per-Base Attribution Head Speed and Accuracy"
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
        "--output-dir",
        type=Path,
        default=Path("outputs/a1_attribution_validation"),
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
    print("A1 Validation: Per-Base Attribution Head Speed and Accuracy")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Test data  : {test_data_path}")
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
        use_attribution_head=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Check whether the checkpoint contains attribution head weights
    attribution_head_trained = any(
        k.startswith("attribution_head.") for k in state_dict.keys()
    )

    if attribution_head_trained:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("  Attribution head weights found in checkpoint (trained).")
        if unexpected:
            print(f"  Skipped {len(unexpected)} unexpected keys (e.g. motif_embedding_head)")
    else:
        # Load with strict=False -- attribution head stays randomly initialized
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("  Attribution head NOT in checkpoint -- randomly initialized.")
        print(f"  Missing keys : {len(missing)}")
        print(f"  Unexpected   : {len(unexpected)}")

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} parameters on {device}")
    print(f"  Attribution head trained: {attribution_head_trained}")

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
    # 3. Run all three methods
    # ------------------------------------------------------------------
    method_names = [
        "attribution_head",
        "gradient_input",
        "per_tf_gradient",
    ]
    method_labels = [
        "Attribution Head",
        "Gradient * Input",
        "Per-TF Gradient",
    ]

    # Storage: method -> tf_name -> list of importance maps [L, 4]
    all_importance = {m: defaultdict(list) for m in method_names}
    # Timing: method -> list of per-sample durations (ms)
    timing = {m: [] for m in method_names}

    # Keep one sample per TF for the scatter plot (Panel C)
    scatter_samples = {}

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

            # Method (a): Attribution Head (forward only, no grad)
            model.zero_grad()
            t0 = time.perf_counter()
            imp_attr = attribution_head_importance(model, seq, device)
            t1 = time.perf_counter()
            timing["attribution_head"].append((t1 - t0) * 1000)
            all_importance["attribution_head"][tf_name].append(imp_attr)

            # Method (b): Gradient * input (profile loss backprop)
            model.zero_grad()
            t0 = time.perf_counter()
            if profile is not None:
                imp_grad = gradient_input_importance(model, seq, profile, device)
            else:
                with torch.no_grad():
                    pseudo = model(seq.unsqueeze(0).to(device))["profile"][0].cpu()
                imp_grad = gradient_input_importance(model, seq, pseudo, device)
            t1 = time.perf_counter()
            timing["gradient_input"].append((t1 - t0) * 1000)
            all_importance["gradient_input"][tf_name].append(imp_grad)

            # Method (c): Per-TF gradient (reference ground truth)
            model.zero_grad()
            t0 = time.perf_counter()
            imp_tf = per_tf_gradient_importance(model, seq, tf_idx, device)
            t1 = time.perf_counter()
            timing["per_tf_gradient"].append((t1 - t0) * 1000)
            all_importance["per_tf_gradient"][tf_name].append(imp_tf)

            # Store the first sample per TF for scatter plot
            if si == 0:
                scatter_samples[tf_name] = {
                    "attribution_head": imp_attr,
                    "gradient_input": imp_grad,
                    "per_tf_gradient": imp_tf,
                }

    # ------------------------------------------------------------------
    # 4. Compute per-sample Pearson correlations
    # ------------------------------------------------------------------
    print("\n--- Computing Correlations ---")

    # corr_data[tf_name] = {
    #   "attr_vs_ref": [list of per-sample correlations],
    #   "grad_vs_ref": [list of per-sample correlations],
    #   "attr_vs_grad": [list of per-sample correlations],
    # }
    corr_data = {}

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx]
        n = len(all_importance["per_tf_gradient"][tf_name])
        attr_vs_ref = []
        grad_vs_ref = []
        attr_vs_grad = []

        for si in range(n):
            imp_attr = all_importance["attribution_head"][tf_name][si]
            imp_grad = all_importance["gradient_input"][tf_name][si]
            imp_ref = all_importance["per_tf_gradient"][tf_name][si]

            attr_vs_ref.append(pearson_correlation(imp_attr, imp_ref))
            grad_vs_ref.append(pearson_correlation(imp_grad, imp_ref))
            attr_vs_grad.append(pearson_correlation(imp_attr, imp_grad))

        corr_data[tf_name] = {
            "attr_vs_ref": attr_vs_ref,
            "grad_vs_ref": grad_vs_ref,
            "attr_vs_grad": attr_vs_grad,
        }
        print(
            f"  {tf_name}: attr-vs-ref={np.mean(attr_vs_ref):.3f}, "
            f"grad-vs-ref={np.mean(grad_vs_ref):.3f}, "
            f"attr-vs-grad={np.mean(attr_vs_grad):.3f}"
        )

    # ------------------------------------------------------------------
    # 5. Compute summary statistics
    # ------------------------------------------------------------------
    summary = {}
    for mi, method in enumerate(method_names):
        t_list = timing[method]
        summary[method] = {
            "label": method_labels[mi],
            "mean_time_ms": float(np.mean(t_list)) if t_list else 0.0,
            "median_time_ms": float(np.median(t_list)) if t_list else 0.0,
            "std_time_ms": float(np.std(t_list)) if t_list else 0.0,
        }

    # Aggregate correlation stats
    all_attr_vs_ref = []
    all_grad_vs_ref = []
    all_attr_vs_grad = []
    per_tf_corr_summary = {}

    for tf_name in TF_NAMES:
        if tf_name not in corr_data:
            continue
        cd = corr_data[tf_name]
        per_tf_corr_summary[tf_name] = {
            "mean_attr_vs_ref": float(np.mean(cd["attr_vs_ref"])),
            "mean_grad_vs_ref": float(np.mean(cd["grad_vs_ref"])),
            "mean_attr_vs_grad": float(np.mean(cd["attr_vs_grad"])),
            "n_samples": len(cd["attr_vs_ref"]),
        }
        all_attr_vs_ref.extend(cd["attr_vs_ref"])
        all_grad_vs_ref.extend(cd["grad_vs_ref"])
        all_attr_vs_grad.extend(cd["attr_vs_grad"])

    global_corr = {
        "mean_attr_vs_ref": float(np.mean(all_attr_vs_ref)) if all_attr_vs_ref else 0.0,
        "mean_grad_vs_ref": float(np.mean(all_grad_vs_ref)) if all_grad_vs_ref else 0.0,
        "mean_attr_vs_grad": float(np.mean(all_attr_vs_grad)) if all_attr_vs_grad else 0.0,
        "std_attr_vs_ref": float(np.std(all_attr_vs_ref)) if all_attr_vs_ref else 0.0,
        "std_grad_vs_ref": float(np.std(all_grad_vs_ref)) if all_grad_vs_ref else 0.0,
        "std_attr_vs_grad": float(np.std(all_attr_vs_grad)) if all_attr_vs_grad else 0.0,
    }

    # ------------------------------------------------------------------
    # 6. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RESULTS: Per-Base Attribution Head Validation")
    print("=" * 90)

    # Per-TF correlation table
    header = f"{'TF':<8} | {'Attr vs Ref':>14s} | {'Grad vs Ref':>14s} | {'Attr vs Grad':>14s} | {'N':>5s}"
    print(f"\nPearson Correlation (per-TF mean):")
    print(header)
    print("-" * len(header))

    for tf_name in TF_NAMES:
        if tf_name not in per_tf_corr_summary:
            print(f"  {tf_name:<6} | {'N/A':>14s} | {'N/A':>14s} | {'N/A':>14s} | {'N/A':>5s}")
            continue
        s = per_tf_corr_summary[tf_name]
        print(
            f"  {tf_name:<6} | {s['mean_attr_vs_ref']:>14.3f} | "
            f"{s['mean_grad_vs_ref']:>14.3f} | "
            f"{s['mean_attr_vs_grad']:>14.3f} | "
            f"{s['n_samples']:>5d}"
        )

    # Summary
    print(f"\n{'':=<90}")
    print("SUMMARY")
    print(f"{'':=<90}")

    print(f"\nCorrelation (all samples):")
    print(f"  Attr Head vs Per-TF Gradient (ref) : {global_corr['mean_attr_vs_ref']:.4f} +/- {global_corr['std_attr_vs_ref']:.4f}")
    print(f"  Grad*Input vs Per-TF Gradient (ref): {global_corr['mean_grad_vs_ref']:.4f} +/- {global_corr['std_grad_vs_ref']:.4f}")
    print(f"  Attr Head vs Grad*Input            : {global_corr['mean_attr_vs_grad']:.4f} +/- {global_corr['std_attr_vs_grad']:.4f}")

    # Timing table
    print(f"\nTiming (ms / sample):")
    print(f"  {'Method':<20s} | {'Mean':>10s} | {'Median':>10s} | {'Std':>10s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for method in method_names:
        s = summary[method]
        print(
            f"  {s['label']:<20s} | {s['mean_time_ms']:>10.2f} | "
            f"{s['median_time_ms']:>10.2f} | {s['std_time_ms']:>10.2f}"
        )

    # Speed-up relative to per-TF gradient
    base_time = summary["per_tf_gradient"]["mean_time_ms"]
    if base_time > 0:
        print(f"\nSpeedup vs Per-TF Gradient:")
        for method in method_names:
            t = summary[method]["mean_time_ms"]
            if t > 0:
                speedup = base_time / t
                print(f"  {summary[method]['label']}: {speedup:.1f}x")

    print(f"\nAttribution head trained: {attribution_head_trained}")

    # ------------------------------------------------------------------
    # 7. Save results JSON
    # ------------------------------------------------------------------
    save_data = {
        "validation": "A1_per_base_attribution_head",
        "checkpoint": str(checkpoint_path),
        "test_data": str(test_data_path),
        "max_samples_per_tf": args.max_samples,
        "attribution_head_trained": attribution_head_trained,
        "methods": method_labels,
        "per_tf_correlations": per_tf_corr_summary,
        "global_correlations": global_corr,
        "timing": summary,
    }

    results_path = output_dir / "a1_attribution_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 8. Create matplotlib visualization (3 panels)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ["#4e79a7", "#e15759", "#59a14f"]

    # --- Panel A: Per-TF correlation bars ---
    ax = axes[0]
    tfs_with_data = [tf for tf in TF_NAMES if tf in per_tf_corr_summary]
    x = np.arange(len(tfs_with_data))
    width = 0.25

    vals_attr = [per_tf_corr_summary[tf]["mean_attr_vs_ref"] for tf in tfs_with_data]
    vals_grad = [per_tf_corr_summary[tf]["mean_grad_vs_ref"] for tf in tfs_with_data]
    vals_cross = [per_tf_corr_summary[tf]["mean_attr_vs_grad"] for tf in tfs_with_data]

    ax.bar(x - width, vals_attr, width, label="Attr Head vs Ref", color=colors[0])
    ax.bar(x, vals_grad, width, label="Grad*Input vs Ref", color=colors[1])
    ax.bar(x + width, vals_cross, width, label="Attr Head vs Grad", color=colors[2])

    ax.set_xlabel("Transcription Factor")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("A) Per-TF Correlation with Reference")
    ax.set_xticks(x)
    ax.set_xticklabels(tfs_with_data, rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # --- Panel B: Speedup comparison ---
    ax = axes[1]
    mean_times = [summary[m]["mean_time_ms"] for m in method_names]
    bar_positions = np.arange(len(method_names))
    bars = ax.bar(bar_positions, mean_times, color=colors)
    ax.set_ylabel("Time (ms / sample)")
    ax.set_title("B) Computational Cost")
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
    if base_time > 0 and summary["attribution_head"]["mean_time_ms"] > 0:
        speedup = base_time / summary["attribution_head"]["mean_time_ms"]
        ax.annotate(
            f"{speedup:.1f}x speedup",
            xy=(0, summary["attribution_head"]["mean_time_ms"]),
            xytext=(0.5, max(mean_times) * 0.8),
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black"),
            ha="center",
        )

    # --- Panel C: Scatter plot of attribution vs gradient for a sample ---
    ax = axes[2]
    # Pick the first TF that has scatter data
    scatter_tf = None
    for tf_name in TF_NAMES:
        if tf_name in scatter_samples:
            scatter_tf = tf_name
            break

    if scatter_tf is not None:
        imp_attr_flat = scatter_samples[scatter_tf]["attribution_head"].flatten()
        imp_ref_flat = scatter_samples[scatter_tf]["per_tf_gradient"].flatten()

        # Sub-sample for readability
        n_pts = len(imp_attr_flat)
        if n_pts > 2000:
            idx = np.random.choice(n_pts, 2000, replace=False)
            imp_attr_flat = imp_attr_flat[idx]
            imp_ref_flat = imp_ref_flat[idx]

        ax.scatter(
            imp_ref_flat,
            imp_attr_flat,
            alpha=0.15,
            s=4,
            color=colors[0],
            rasterized=True,
        )
        r = pearson_correlation(imp_attr_flat, imp_ref_flat)
        ax.set_xlabel("Per-TF Gradient (reference)")
        ax.set_ylabel("Attribution Head")
        ax.set_title(f"C) Attr vs Gradient ({scatter_tf}, r={r:.3f})")

        # Fit line
        if np.std(imp_ref_flat) > 1e-12 and np.std(imp_attr_flat) > 1e-12:
            z = np.polyfit(imp_ref_flat, imp_attr_flat, 1)
            p = np.poly1d(z)
            x_line = np.linspace(imp_ref_flat.min(), imp_ref_flat.max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=1.5, alpha=0.7)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C) Attr vs Gradient (no data)")

    trained_tag = "trained" if attribution_head_trained else "untrained (random init)"
    fig.suptitle(
        f"A1 Validation: Per-Base Attribution Head ({trained_tag})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig_path = output_dir / "a1_attribution_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
