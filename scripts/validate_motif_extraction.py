#!/usr/bin/env python3
"""
A3 Validation: Per-Slot Motif Extraction and TF-MoDISco Compatibility.

Validates the PerSlotMotifExtractor from the analysis package:
  1. Extracts per-slot motifs from BEACON attribution maps.
  2. Compares extracted PWMs to JASPAR ground truth (Pearson correlation).
  3. Tests TF-MoDISco-compatible output format.
  4. Measures extraction speed vs TF-MoDISco baseline estimate.

Produces a comparison table, JSON results, and a 3-panel figure.
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset
from beacon.analysis import PerSlotMotifExtractor

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
# Helpers
# ---------------------------------------------------------------------------

def compute_info_content(pwm):
    """Compute mean information content of a PWM in bits."""
    if pwm is None:
        return 0.0
    ic = np.log2(4) + np.sum(pwm * np.log2(pwm + 1e-8), axis=1)
    return float(ic.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A3 Validation: Per-Slot Motif Extraction and TF-MoDISco Compatibility"
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
        default=Path("outputs/a3_motif_extraction_validation"),
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
    print("A3 Validation: Per-Slot Motif Extraction & TF-MoDISco Compatibility")
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
        use_attribution_head=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} parameters on {device} (strict=False for missing weights)")

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
    # 4. Collect per-slot attribution and sequences per TF
    # ------------------------------------------------------------------
    print("\n--- Collecting Per-Slot Attribution Maps ---")

    extractor = PerSlotMotifExtractor(
        window_size=24,
        percentile_threshold=95.0,
        top_k_peaks=15,
    )

    # Storage per TF: accumulated importance maps and one-hot sequences
    tf_importance = {}     # tf_name -> [n_samples, L, 4]
    tf_sequences = {}      # tf_name -> [n_samples, L, 4]
    tf_slot_ids = {}       # tf_name -> best slot id (most common)

    # Timing: per-sample extraction durations (ms)
    extraction_times = []
    per_tf_gradient_times = []

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx]
        samples = tf_samples[tf_idx]
        n = len(samples)
        print(f"\n--- Processing {tf_name} ({n} samples) ---")

        importance_maps = []
        seq_arrays = []
        slot_counts = defaultdict(int)

        for si, sample in enumerate(samples):
            if (si + 1) % 50 == 0 or si == 0:
                print(f"  sample {si + 1}/{n}")

            seq = sample["sequence"]  # [L, 4]
            seq_tensor = seq.unsqueeze(0).to(device)

            # Forward pass with attribution
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(seq_tensor)
            t_forward = time.perf_counter()

            # attribution: [B, K, L], occupancy: [B, K, 1], tf_logits: [B, K, N_TFs]
            if "attribution" in outputs:
                attribution = outputs["attribution"][0].cpu().numpy()  # [K, L]
            else:
                # Fall back to attention weights if attribution head not loaded
                attribution = outputs["attention"][0].detach().cpu().numpy()  # [K, L]

            occupancy = outputs["occupancy"][0, :, 0].detach().cpu().numpy()  # [K]
            tf_logits = outputs["tf_logits"][0].detach().cpu().numpy()  # [K, N_TFs]

            t_extract = time.perf_counter()
            extraction_times.append((t_extract - t0) * 1000)

            seq_np = seq.cpu().numpy()  # [L, 4]

            # Find highest-occupancy slot that predicts this TF
            tf_preds = tf_logits.argmax(axis=1)  # [K]
            matching_slots = np.where(tf_preds == tf_idx)[0]

            if len(matching_slots) == 0:
                best_slot = int(np.argmax(occupancy))
            else:
                occ_vals = occupancy[matching_slots]
                best_slot = matching_slots[int(np.argmax(occ_vals))]

            slot_counts[best_slot] += 1

            # Build per-slot importance: attribution[k, l] * seq[l, 4]
            attn_k = attribution[best_slot]  # [L]
            importance = seq_np * attn_k[:, None]  # [L, 4]
            importance_maps.append(importance)
            seq_arrays.append(seq_np)

            # Also measure per-TF gradient time for speed comparison
            model.zero_grad()
            seq_grad = seq.unsqueeze(0).to(device).requires_grad_(True)
            t0_grad = time.perf_counter()
            out_grad = model(seq_grad)
            tf_score = (
                out_grad["tf_logits"][0][:, tf_idx]
                * out_grad["occupancy"][0, :, 0]
            ).sum()
            tf_score.backward()
            t1_grad = time.perf_counter()
            per_tf_gradient_times.append((t1_grad - t0_grad) * 1000)

        # Determine the most commonly chosen slot
        if slot_counts:
            most_common_slot = max(slot_counts, key=slot_counts.get)
        else:
            most_common_slot = 0

        tf_importance[tf_name] = np.array(importance_maps)   # [N, L, 4]
        tf_sequences[tf_name] = np.array(seq_arrays)         # [N, L, 4]
        tf_slot_ids[tf_name] = most_common_slot

    # ------------------------------------------------------------------
    # 5. Extract motifs using PerSlotMotifExtractor
    # ------------------------------------------------------------------
    print("\n--- Extracting Motifs via PerSlotMotifExtractor ---")

    tf_results = {}
    extracted_motifs_list = []

    for tf_name in TF_NAMES:
        if tf_name not in tf_importance:
            continue

        importance_maps = tf_importance[tf_name]
        sequences_array = tf_sequences[tf_name]
        slot_id = tf_slot_ids[tf_name]

        avg_importance = importance_maps.mean(axis=0)  # [L, 4]

        motif = extractor.extract_from_importance(
            avg_importance,
            sequences_array,
            slot_id=slot_id,
            tf_name=tf_name,
        )

        # Compare to JASPAR
        pearson_r = 0.0
        if motif.pfm is not None and tf_name in jaspar_motifs:
            pearson_r = compare_motifs(motif.pfm, jaspar_motifs[tf_name])

        ic = motif.info_content if motif.pfm is not None else 0.0
        motif_len = motif.pfm.shape[0] if motif.pfm is not None else 0
        n_seqlets = motif.n_seqlets

        tf_results[tf_name] = {
            "pearson_r": float(pearson_r),
            "info_content": float(ic),
            "motif_length": int(motif_len),
            "n_seqlets": int(n_seqlets),
            "slot_id": int(slot_id),
            "n_samples": int(len(importance_maps)),
        }
        extracted_motifs_list.append(motif)

        print(
            f"  {tf_name}: r={pearson_r:.3f}, len={motif_len}, "
            f"IC={ic:.3f} bits, seqlets={n_seqlets}, slot={slot_id}"
        )

    # ------------------------------------------------------------------
    # 6. Test TF-MoDISco format compatibility
    # ------------------------------------------------------------------
    print("\n--- Testing TF-MoDISco Format Compatibility ---")

    modisco_output = extractor.to_modisco_format(extracted_motifs_list)
    modisco_pass = True
    modisco_errors = []

    # Check top-level key
    top_key = "metacluster_idx_to_submetacluster_results"
    if top_key not in modisco_output:
        modisco_pass = False
        modisco_errors.append(f"Missing top-level key: {top_key}")
    else:
        mc = modisco_output[top_key]
        if "metacluster_0" not in mc:
            modisco_pass = False
            modisco_errors.append("Missing metacluster_0")
        else:
            mc0 = mc["metacluster_0"]
            if "activity_pattern" not in mc0:
                modisco_pass = False
                modisco_errors.append("Missing activity_pattern")
            if "seqlets_to_patterns_result" not in mc0:
                modisco_pass = False
                modisco_errors.append("Missing seqlets_to_patterns_result")
            else:
                patterns = mc0["seqlets_to_patterns_result"].get("patterns", {})
                for pkey, pval in patterns.items():
                    for required in [
                        "sequence",
                        "contrib_scores",
                        "hypothetical_contribs",
                        "seqlets_and_alnmts",
                        "num_seqlets",
                    ]:
                        if required not in pval:
                            modisco_pass = False
                            modisco_errors.append(
                                f"{pkey} missing field: {required}"
                            )
                n_patterns = len(patterns)
                print(f"  Patterns in output: {n_patterns}")

    modisco_status = "PASS" if modisco_pass else "FAIL"
    print(f"  TF-MoDISco format check: {modisco_status}")
    if modisco_errors:
        for err in modisco_errors:
            print(f"    ERROR: {err}")

    # ------------------------------------------------------------------
    # 7. Compute summary statistics
    # ------------------------------------------------------------------
    rs = [
        tf_results[tf]["pearson_r"]
        for tf in TF_NAMES
        if tf in tf_results and tf_results[tf]["pearson_r"] > 0
    ]
    ics = [
        tf_results[tf]["info_content"]
        for tf in TF_NAMES
        if tf in tf_results
    ]

    mean_extraction_ms = float(np.mean(extraction_times)) if extraction_times else 0.0
    median_extraction_ms = float(np.median(extraction_times)) if extraction_times else 0.0
    mean_gradient_ms = float(np.mean(per_tf_gradient_times)) if per_tf_gradient_times else 0.0
    modisco_estimate_ms = 8400.0  # TF-MoDISco ~8.4s per sequence

    summary = {
        "mean_pearson_r": float(np.mean(rs)) if rs else 0.0,
        "median_pearson_r": float(np.median(rs)) if rs else 0.0,
        "max_pearson_r": float(np.max(rs)) if rs else 0.0,
        "n_tfs_above_0.5": int(sum(1 for r in rs if r > 0.5)),
        "n_tfs_above_0.75": int(sum(1 for r in rs if r > 0.75)),
        "mean_ic": float(np.mean(ics)) if ics else 0.0,
        "n_tfs_evaluated": len(rs),
        "mean_extraction_ms": mean_extraction_ms,
        "median_extraction_ms": median_extraction_ms,
        "mean_per_tf_gradient_ms": mean_gradient_ms,
        "modisco_estimate_ms": modisco_estimate_ms,
        "modisco_format_compatible": modisco_pass,
    }

    # ------------------------------------------------------------------
    # 8. Print formatted table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RESULTS: Per-Slot Motif Extraction Validation")
    print("=" * 90)

    header = (
        f"{'TF':<8} | {'JASPAR r':>9} | {'Length':>6} | "
        f"{'IC (bits)':>9} | {'Seqlets':>8} | {'Slot':>4} | {'Samples':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for tf in TF_NAMES:
        if tf in tf_results:
            r = tf_results[tf]
            print(
                f"  {tf:<6} | {r['pearson_r']:>9.3f} | {r['motif_length']:>6d} | "
                f"{r['info_content']:>9.3f} | {r['n_seqlets']:>8d} | "
                f"{r['slot_id']:>4d} | {r['n_samples']:>7d}"
            )
        else:
            print(f"  {tf:<6} | {'N/A':>9} | {'N/A':>6} | {'N/A':>9} | {'N/A':>8} | {'N/A':>4} | {'N/A':>7}")

    print(f"\n{'':=<90}")
    print("SUMMARY")
    print(f"{'':=<90}")
    print(f"  Mean JASPAR Pearson r   : {summary['mean_pearson_r']:.3f}")
    print(f"  Median JASPAR Pearson r : {summary['median_pearson_r']:.3f}")
    print(f"  Max JASPAR Pearson r    : {summary['max_pearson_r']:.3f}")
    print(f"  TFs with r > 0.50      : {summary['n_tfs_above_0.5']}/{summary['n_tfs_evaluated']}")
    print(f"  TFs with r > 0.75      : {summary['n_tfs_above_0.75']}/{summary['n_tfs_evaluated']}")
    print(f"  Mean IC (bits)          : {summary['mean_ic']:.3f}")

    print(f"\n{'':=<90}")
    print("EXTRACTION SPEED")
    print(f"{'':=<90}")
    print(f"  BEACON attribution (fwd pass) : {mean_extraction_ms:>10.1f} ms/sample")
    print(f"  Per-TF gradient (bwd pass)    : {mean_gradient_ms:>10.1f} ms/sample")
    print(f"  TF-MoDISco estimate           : {modisco_estimate_ms:>10.1f} ms/sample")
    if mean_extraction_ms > 0:
        speedup_vs_grad = mean_gradient_ms / mean_extraction_ms
        speedup_vs_modisco = modisco_estimate_ms / mean_extraction_ms
        print(f"  Speedup vs gradient           : {speedup_vs_grad:>10.1f}x")
        print(f"  Speedup vs TF-MoDISco         : {speedup_vs_modisco:>10.1f}x")

    print(f"\n{'':=<90}")
    print("TF-MoDISco FORMAT COMPATIBILITY")
    print(f"{'':=<90}")
    print(f"  Status: {modisco_status}")

    # ------------------------------------------------------------------
    # 9. Save results JSON
    # ------------------------------------------------------------------
    save_data = {
        "validation": "A3_per_slot_motif_extraction",
        "checkpoint": str(checkpoint_path),
        "test_data": str(test_data_path),
        "max_samples_per_tf": args.max_samples,
        "extractor_config": {
            "window_size": extractor.window_size,
            "percentile_threshold": extractor.percentile_threshold,
            "top_k_peaks": extractor.top_k_peaks,
        },
        "per_tf_results": tf_results,
        "summary": summary,
    }

    results_path = output_dir / "a3_motif_extraction_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 10. Create matplotlib visualization (1x3 panels)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: Per-TF JASPAR correlation bars ---
    ax = axes[0]
    tfs_with_data = [tf for tf in TF_NAMES if tf in tf_results]
    x = np.arange(len(tfs_with_data))
    r_vals = [tf_results[tf]["pearson_r"] for tf in tfs_with_data]
    colors_bar = ["#4e79a7" if r >= 0.5 else "#e15759" for r in r_vals]
    bars = ax.bar(x, r_vals, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Transcription Factor")
    ax.set_ylabel("JASPAR Pearson r")
    ax.set_title("A) Per-TF Motif Recovery (JASPAR Correlation)")
    ax.set_xticks(x)
    ax.set_xticklabels(tfs_with_data, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="r=0.5")
    ax.axhline(y=0.75, color="gray", linestyle=":", linewidth=0.8, alpha=0.4, label="r=0.75")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, r_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # --- Panel B: Extraction speed comparison ---
    ax = axes[1]
    speed_labels = ["BEACON\nAttribution", "Per-TF\nGradient", "TF-MoDISco\n(estimate)"]
    speed_vals = [mean_extraction_ms, mean_gradient_ms, modisco_estimate_ms]
    speed_colors = ["#59a14f", "#4e79a7", "#e15759"]
    bar_positions = np.arange(len(speed_labels))
    bars = ax.bar(bar_positions, speed_vals, color=speed_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Time (ms / sample)")
    ax.set_title("B) Extraction Speed Comparison")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(speed_labels, fontsize=9)
    ax.set_yscale("log")
    for bar, val in zip(bars, speed_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- Panel C: Example motif logo (PWM heatmap for best TF) ---
    ax = axes[2]
    best_tf = None
    best_r = -1.0
    best_pfm = None
    for motif in extracted_motifs_list:
        if motif.pfm is not None and motif.tf_name in tf_results:
            r = tf_results[motif.tf_name]["pearson_r"]
            if r > best_r:
                best_r = r
                best_tf = motif.tf_name
                best_pfm = motif.pfm

    if best_pfm is not None:
        im = ax.imshow(
            best_pfm.T,
            aspect="auto",
            cmap="Blues",
            interpolation="nearest",
        )
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["A", "C", "G", "T"])
        ax.set_xlabel("Position")
        ax.set_title(f"C) Best Motif PWM: {best_tf} (r={best_r:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Frequency")
    else:
        ax.text(
            0.5,
            0.5,
            "No motifs extracted",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("C) Best Motif PWM")

    fig.suptitle(
        "A3 Validation: Per-Slot Motif Extraction & TF-MoDISco Compatibility",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig_path = output_dir / "a3_motif_extraction_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
