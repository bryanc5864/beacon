#!/usr/bin/env python3
"""
A5: Compile Validation Results

Aggregates results from A1-A4 validation experiments into a single summary
table for the paper.

Expected result locations:
  A1+A2: outputs/validation/multi_tf_cobinding/  (multi-site detection, position accuracy)
  A3:    outputs/validation/cross_cell_type/      (cross-cell-type transfer)
  A4:    outputs/validation/interpretability/      (JASPAR correlation comparison)

Outputs:
  - compiled_results.json   (machine-readable aggregate)
  - validation_summary.md   (formatted markdown table)
  - validation_summary.png  (single-page summary figure)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

base_dir = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    """Load a JSON file, returning None if it does not exist or is malformed."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _find_json_files(directory, patterns=None):
    """Return a dict mapping filename stems to loaded JSON data.

    If *patterns* is given (list of glob patterns), only files matching at
    least one pattern are considered.  Otherwise every ``*.json`` in
    *directory* (non-recursive) is loaded.
    """
    results = {}
    if not directory.exists():
        return results
    if patterns is None:
        patterns = ["*.json"]
    for pattern in patterns:
        for fp in sorted(directory.glob(pattern)):
            data = _load_json(fp)
            if data is not None:
                results[fp.stem] = data
    return results


def _fmt(value, fmt=".3f"):
    """Format a numeric value, returning 'N/A' for None."""
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def _pct(value, fmt=".1f"):
    """Format a fraction as a percentage string."""
    if value is None:
        return "N/A"
    try:
        return f"{value * 100:{fmt}}%"
    except (ValueError, TypeError):
        return str(value)


# ---------------------------------------------------------------------------
# A1 + A2 loading (multi-TF co-binding validation)
# ---------------------------------------------------------------------------

def load_a1_a2(cobinding_dir):
    """Load results from the multi-TF co-binding validation (A1 + A2).

    Expected JSON files in *cobinding_dir*:
      - multi_site_detection_results.json  (or any *detection*.json)
      - position_accuracy_results.json     (or any *position*.json)
      - cobinding_results.json / cobinding_summary.json
      - Any other *.json produced by validate_multi_tf_cobinding.py

    Returns a dict with parsed A1 and A2 metrics, or a dict with
    ``status: "pending"`` if the directory is absent or empty.
    """
    all_data = _find_json_files(cobinding_dir)
    if not all_data:
        return {"status": "pending", "reason": f"No result files in {cobinding_dir}"}

    a1 = {}
    a2 = {}

    # ------------------------------------------------------------------
    # Try to find the combined / summary file first.
    # ------------------------------------------------------------------
    summary = (
        all_data.get("cobinding_results")
        or all_data.get("cobinding_summary")
        or all_data.get("multi_tf_cobinding_results")
        or all_data.get("validation_results")
    )

    # If there is no obvious summary, merge all loaded files.
    if summary is None:
        summary = {}
        for v in all_data.values():
            if isinstance(v, dict):
                summary.update(v)

    # ------------------------------------------------------------------
    # A1 metrics: multi-site detection rate, per-site TF accuracy,
    #             position MAE by number of sites.
    # ------------------------------------------------------------------
    a1["multi_site_detection_rate"] = (
        summary.get("multi_site_detection_rate")
        or summary.get("detection_rate")
        or summary.get("site_detection_rate")
    )
    a1["per_site_tf_accuracy"] = (
        summary.get("per_site_tf_accuracy")
        or summary.get("tf_accuracy")
        or summary.get("per_site_accuracy")
    )

    # Position MAE by number of sites (dict: n_sites -> MAE)
    position_mae_by_sites = (
        summary.get("position_mae_by_n_sites")
        or summary.get("position_mae_by_sites")
        or summary.get("mae_by_num_sites")
    )
    if position_mae_by_sites is not None:
        # Normalise keys to ints for downstream use.
        a1["position_mae_by_n_sites"] = {
            str(k): float(v) for k, v in position_mae_by_sites.items()
        }
    else:
        a1["position_mae_by_n_sites"] = None

    a1["overall_position_mae"] = (
        summary.get("overall_position_mae")
        or summary.get("position_mae")
        or summary.get("mean_position_mae")
    )

    # ------------------------------------------------------------------
    # A2 metrics: position MAE at various true positions, position bias.
    # ------------------------------------------------------------------
    position_mae_by_true_pos = (
        summary.get("position_mae_by_true_position")
        or summary.get("mae_by_true_position")
        or summary.get("position_mae_by_position")
    )
    if position_mae_by_true_pos is not None:
        a2["position_mae_by_true_position"] = {
            str(k): float(v) for k, v in position_mae_by_true_pos.items()
        }
    else:
        a2["position_mae_by_true_position"] = None

    a2["position_bias"] = (
        summary.get("position_bias")
        or summary.get("mean_position_bias")
        or summary.get("bias")
    )
    a2["position_bias_std"] = summary.get("position_bias_std")
    a2["position_mae_overall"] = (
        summary.get("position_mae_overall")
        or summary.get("overall_position_mae")
        or a1.get("overall_position_mae")
    )

    status = "complete"
    # If every extracted value is None, mark as pending.
    if all(v is None for v in list(a1.values()) + list(a2.values())):
        status = "partial"

    return {"status": status, "a1": a1, "a2": a2, "raw_files": list(all_data.keys())}


# ---------------------------------------------------------------------------
# A3 loading (cross-cell-type transfer)
# ---------------------------------------------------------------------------

def load_a3(cross_cell_dir):
    """Load cross-cell-type transfer results (A3).

    Expected JSON files:
      - cross_cell_results.json
      - transfer_results.json
    """
    all_data = _find_json_files(cross_cell_dir)
    if not all_data:
        return {"status": "pending", "reason": f"No result files in {cross_cell_dir}"}

    summary = (
        all_data.get("cross_cell_results")
        or all_data.get("transfer_results")
        or all_data.get("cross_cell_transfer_results")
    )
    if summary is None:
        summary = {}
        for v in all_data.values():
            if isinstance(v, dict):
                summary.update(v)

    a3 = {}

    # Source / target cell line results
    source = summary.get("k562") or summary.get("source") or {}
    target = summary.get("hepg2") or summary.get("target") or {}
    transfer = summary.get("transfer_efficiency") or summary.get("transfer") or {}

    a3["source_cell"] = summary.get("source_cell", "K562")
    a3["target_cell"] = summary.get("target_cell", "HepG2")

    a3["source_tf_accuracy"] = source.get("tf_accuracy")
    a3["target_tf_accuracy"] = target.get("tf_accuracy")
    a3["source_site_detection"] = source.get("site_detection")
    a3["target_site_detection"] = target.get("site_detection")
    a3["source_profile_pearson"] = source.get("profile_pearson")
    a3["target_profile_pearson"] = target.get("profile_pearson")

    a3["transfer_tf_accuracy"] = transfer.get("tf_accuracy")
    a3["transfer_site_detection"] = transfer.get("site_detection")
    a3["transfer_profile_pearson"] = transfer.get("profile_pearson")

    # Per-TF transfer
    common_tfs = summary.get("common_tfs")
    a3["common_tfs"] = common_tfs

    # Look for per-TF breakdown
    per_tf = summary.get("per_tf_transfer") or summary.get("per_tf") or {}
    a3["per_tf_transfer"] = per_tf if per_tf else None

    # Overall transfer percentage
    transfer_values = [
        v for v in [
            a3.get("transfer_tf_accuracy"),
            a3.get("transfer_site_detection"),
            a3.get("transfer_profile_pearson"),
        ]
        if v is not None
    ]
    a3["overall_transfer_pct"] = (
        float(np.mean(transfer_values)) if transfer_values else None
    )

    a3["interpretation"] = summary.get("interpretation")

    status = "complete" if transfer_values else "pending"
    return {"status": status, "a3": a3, "raw_files": list(all_data.keys())}


# ---------------------------------------------------------------------------
# A4 loading (interpretability / JASPAR correlation)
# ---------------------------------------------------------------------------

def load_a4(interp_dir):
    """Load interpretability validation results (A4).

    Expected JSON files:
      - gradient_motif_results.json
      - attention_results.json
      - motif_discovery_results.json
      - interpretability_results.json
    """
    all_data = _find_json_files(interp_dir)
    if not all_data:
        return {"status": "pending", "reason": f"No result files in {interp_dir}"}

    a4 = {}

    # ------------------------------------------------------------------
    # Method 1: Attention-weighted motif extraction
    # ------------------------------------------------------------------
    attn = all_data.get("attention_results") or all_data.get("attention_motif_results") or {}
    a4["attention_overall_corr"] = attn.get("overall_mean_correlation")
    attn_per_tf = attn.get("attention_profile_correlation") or {}
    a4["attention_per_tf"] = {
        tf: info.get("mean") if isinstance(info, dict) else info
        for tf, info in attn_per_tf.items()
    } if attn_per_tf else None

    # ------------------------------------------------------------------
    # Method 2: Gradient-based motif extraction (per-TF)
    # ------------------------------------------------------------------
    grad = (
        all_data.get("gradient_motif_results")
        or all_data.get("per_tf_gradient_motifs")
        or all_data.get("gradient_results")
        or {}
    )
    per_tf_gradient = grad.get("per_tf") or grad.get("per_tf_results") or {}
    a4["gradient_per_tf"] = {}
    for tf, info in per_tf_gradient.items():
        if isinstance(info, dict):
            a4["gradient_per_tf"][tf] = info.get("jaspar_correlation", info.get("correlation"))
        else:
            a4["gradient_per_tf"][tf] = info
    a4["gradient_mean_corr"] = grad.get("mean_jaspar_correlation") or grad.get("mean_correlation")
    if not a4["gradient_mean_corr"] and a4["gradient_per_tf"]:
        vals = [v for v in a4["gradient_per_tf"].values() if v is not None]
        a4["gradient_mean_corr"] = float(np.mean(vals)) if vals else None

    # ------------------------------------------------------------------
    # Method 3: Slot-level motif discovery (Phase 4.1-style)
    # ------------------------------------------------------------------
    motif = all_data.get("motif_discovery_results") or all_data.get("slot_motif_results") or {}
    a4["slot_motif_mean_corr"] = motif.get("mean_correlation") or motif.get("mean_jaspar_correlation")
    a4["slot_motif_per_slot"] = motif.get("per_slot") or motif.get("slot_correlations")

    # Summary across all three methods
    method_corrs = {}
    if a4.get("attention_overall_corr") is not None:
        method_corrs["attention"] = a4["attention_overall_corr"]
    if a4.get("gradient_mean_corr") is not None:
        method_corrs["gradient"] = a4["gradient_mean_corr"]
    if a4.get("slot_motif_mean_corr") is not None:
        method_corrs["slot_motif"] = a4["slot_motif_mean_corr"]
    a4["method_comparison"] = method_corrs if method_corrs else None

    status = "complete" if method_corrs else "pending"
    return {"status": status, "a4": a4, "raw_files": list(all_data.keys())}


# ---------------------------------------------------------------------------
# Markdown summary generation
# ---------------------------------------------------------------------------

def generate_markdown(compiled, output_path):
    """Write a nicely formatted markdown summary."""
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# BEACON Validation Summary")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- A1: Multi-site detection ----
    lines.append("## A1: Multi-Site Co-Binding Detection")
    lines.append("")
    a1_data = compiled.get("a1", {})
    a1_status = a1_data.get("status", "pending")
    if a1_status == "pending":
        lines.append("**Status:** PENDING -- results not yet available.")
        reason = a1_data.get("reason", "")
        if reason:
            lines.append(f"  Reason: {reason}")
    else:
        a1 = a1_data.get("a1", {})
        lines.append("| Metric                        | Value   |")
        lines.append("|-------------------------------|---------|")
        lines.append(f"| Multi-site detection rate      | {_pct(a1.get('multi_site_detection_rate'))} |")
        lines.append(f"| Per-site TF accuracy           | {_pct(a1.get('per_site_tf_accuracy'))} |")
        lines.append(f"| Overall position MAE (bp)      | {_fmt(a1.get('overall_position_mae'), '.1f')} |")
        lines.append("")

        mae_by_sites = a1.get("position_mae_by_n_sites")
        if mae_by_sites:
            lines.append("### Position MAE by Number of Binding Sites")
            lines.append("")
            lines.append("| N Sites | Position MAE (bp) |")
            lines.append("|---------|-------------------|")
            for n_sites in sorted(mae_by_sites.keys(), key=lambda x: int(x)):
                lines.append(f"| {n_sites:>7} | {_fmt(mae_by_sites[n_sites], '.2f'):>17} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- A2: Position accuracy ----
    lines.append("## A2: Position Accuracy Analysis")
    lines.append("")
    a2_data = compiled.get("a2", {})
    a2_status = a2_data.get("status", a1_status)
    if a2_status == "pending":
        lines.append("**Status:** PENDING -- results not yet available.")
    else:
        a2 = a1_data.get("a2", {}) if "a2" in a1_data else a2_data.get("a2", {})
        lines.append("| Metric                 | Value   |")
        lines.append("|------------------------|---------|")
        lines.append(f"| Position MAE (overall)  | {_fmt(a2.get('position_mae_overall'), '.2f')} |")
        lines.append(f"| Position bias (mean)    | {_fmt(a2.get('position_bias'), '.2f')} |")
        lines.append(f"| Position bias (std)     | {_fmt(a2.get('position_bias_std'), '.2f')} |")
        lines.append("")

        mae_by_pos = a2.get("position_mae_by_true_position")
        if mae_by_pos:
            lines.append("### Position MAE by True Position")
            lines.append("")
            lines.append("| True Position (bp) | MAE (bp) |")
            lines.append("|--------------------|----------|")
            for pos in sorted(mae_by_pos.keys(), key=lambda x: float(x)):
                lines.append(f"| {pos:>18} | {_fmt(mae_by_pos[pos], '.2f'):>8} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- A3: Cross-cell-type transfer ----
    lines.append("## A3: Cross-Cell-Type Transfer")
    lines.append("")
    a3_block = compiled.get("a3", {})
    a3_status = a3_block.get("status", "pending")
    if a3_status == "pending":
        lines.append("**Status:** PENDING -- results not yet available.")
        reason = a3_block.get("reason", "")
        if reason:
            lines.append(f"  Reason: {reason}")
    else:
        a3 = a3_block.get("a3", {})
        src = a3.get("source_cell", "K562")
        tgt = a3.get("target_cell", "HepG2")
        lines.append(f"Transfer: {src} -> {tgt}")
        lines.append("")

        common = a3.get("common_tfs")
        if common:
            lines.append(f"Shared TFs: {', '.join(common)}")
            lines.append("")

        lines.append(f"| Metric              | {src:>8} | {tgt:>8} | Transfer % |")
        lines.append(f"|---------------------|----------|----------|------------|")
        lines.append(
            f"| TF Accuracy         | {_pct(a3.get('source_tf_accuracy')):>8} "
            f"| {_pct(a3.get('target_tf_accuracy')):>8} "
            f"| {_fmt(a3.get('transfer_tf_accuracy'), '.1f'):>10} |"
        )
        lines.append(
            f"| Site Detection      | {_pct(a3.get('source_site_detection')):>8} "
            f"| {_pct(a3.get('target_site_detection')):>8} "
            f"| {_fmt(a3.get('transfer_site_detection'), '.1f'):>10} |"
        )
        lines.append(
            f"| Profile Pearson     | {_fmt(a3.get('source_profile_pearson')):>8} "
            f"| {_fmt(a3.get('target_profile_pearson')):>8} "
            f"| {_fmt(a3.get('transfer_profile_pearson'), '.1f'):>10} |"
        )
        lines.append("")
        lines.append(f"Overall transfer: {_fmt(a3.get('overall_transfer_pct'), '.1f')}%")

        per_tf = a3.get("per_tf_transfer")
        if per_tf and isinstance(per_tf, dict):
            lines.append("")
            lines.append("### Per-TF Transfer Efficiency")
            lines.append("")
            lines.append("| TF       | Source Acc | Target Acc | Transfer % |")
            lines.append("|----------|-----------|------------|------------|")
            for tf_name, info in per_tf.items():
                if isinstance(info, dict):
                    lines.append(
                        f"| {tf_name:<8} "
                        f"| {_pct(info.get('source_accuracy', info.get('k562_accuracy'))):>9} "
                        f"| {_pct(info.get('target_accuracy', info.get('hepg2_accuracy'))):>10} "
                        f"| {_fmt(info.get('transfer_pct', info.get('transfer')), '.1f'):>10} |"
                    )
                else:
                    lines.append(f"| {tf_name:<8} | {'N/A':>9} | {'N/A':>10} | {_fmt(info, '.1f'):>10} |")

        interp = a3.get("interpretation")
        if interp:
            lines.append("")
            lines.append(f"**Interpretation:** {interp}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- A4: Interpretability / JASPAR correlation ----
    lines.append("## A4: Interpretability -- JASPAR Motif Correlation")
    lines.append("")
    a4_block = compiled.get("a4", {})
    a4_status = a4_block.get("status", "pending")
    if a4_status == "pending":
        lines.append("**Status:** PENDING -- results not yet available.")
        reason = a4_block.get("reason", "")
        if reason:
            lines.append(f"  Reason: {reason}")
    else:
        a4 = a4_block.get("a4", {})

        method_comp = a4.get("method_comparison", {})
        if method_comp:
            lines.append("### Method Comparison (Mean JASPAR Correlation)")
            lines.append("")
            lines.append("| Method               | Mean r |")
            lines.append("|----------------------|--------|")
            for method, corr in sorted(method_comp.items(), key=lambda x: -(x[1] or 0)):
                lines.append(f"| {method:<20} | {_fmt(corr):>6} |")
            lines.append("")

        # Gradient per-TF breakdown
        grad_per_tf = a4.get("gradient_per_tf", {})
        if grad_per_tf:
            lines.append("### Per-TF JASPAR Correlation (Gradient Method)")
            lines.append("")
            lines.append("| TF       | JASPAR r |")
            lines.append("|----------|----------|")
            for tf_name in sorted(grad_per_tf.keys()):
                lines.append(f"| {tf_name:<8} | {_fmt(grad_per_tf[tf_name]):>8} |")
            lines.append("")

        # Attention per-TF
        attn_per_tf = a4.get("attention_per_tf", {})
        if attn_per_tf:
            lines.append("### Per-TF Attention-Profile Correlation")
            lines.append("")
            lines.append("| TF       | Mean r |")
            lines.append("|----------|--------|")
            for tf_name in sorted(attn_per_tf.keys()):
                lines.append(f"| {tf_name:<8} | {_fmt(attn_per_tf[tf_name]):>6} |")
            lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Overall status ----
    lines.append("## Overall Validation Status")
    lines.append("")
    lines.append("| Experiment | Status |")
    lines.append("|------------|--------|")
    for label, key in [
        ("A1: Multi-site detection", "a1"),
        ("A2: Position accuracy", "a2"),
        ("A3: Cross-cell transfer", "a3"),
        ("A4: Interpretability", "a4"),
    ]:
        block = compiled.get(key, {})
        st = block.get("status", "pending")
        status_str = "COMPLETE" if st == "complete" else ("PARTIAL" if st == "partial" else "PENDING")
        lines.append(f"| {label:<30} | {status_str:<6} |")

    lines.append("")

    md_text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md_text)

    return md_text


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def create_summary_figure(compiled, output_path):
    """Create a single-page summary figure with key metrics from A1-A4."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("BEACON Validation Summary", fontsize=16, fontweight="bold", y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.95, top=0.92, bottom=0.06)

    status_colors = {"complete": "#2ecc71", "partial": "#f39c12", "pending": "#e74c3c"}

    # ---- Panel A: A1 -- Position MAE by number of sites ----
    ax_a1 = fig.add_subplot(gs[0, 0])
    a1_block = compiled.get("a1", {})
    a1_status = a1_block.get("status", "pending")

    if a1_status != "pending":
        a1 = a1_block.get("a1", {})
        mae_by_sites = a1.get("position_mae_by_n_sites", {})
        if mae_by_sites:
            n_sites_keys = sorted(mae_by_sites.keys(), key=lambda x: int(x))
            x_vals = [int(k) for k in n_sites_keys]
            y_vals = [mae_by_sites[k] for k in n_sites_keys]
            bars = ax_a1.bar(x_vals, y_vals, color="steelblue", alpha=0.8, edgecolor="black")
            for bar, yv in zip(bars, y_vals):
                ax_a1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(y_vals) * 0.02,
                    f"{yv:.1f}",
                    ha="center", va="bottom", fontsize=9,
                )
            ax_a1.set_xlabel("Number of Binding Sites")
            ax_a1.set_ylabel("Position MAE (bp)")
            ax_a1.set_title("A1: Position MAE by Site Count", fontweight="bold")
        else:
            # Show summary metrics as text
            det_rate = a1.get("multi_site_detection_rate")
            tf_acc = a1.get("per_site_tf_accuracy")
            overall_mae = a1.get("overall_position_mae")
            text_lines = ["A1: Multi-Site Detection", ""]
            if det_rate is not None:
                text_lines.append(f"Detection rate: {det_rate*100:.1f}%")
            if tf_acc is not None:
                text_lines.append(f"Per-site TF acc: {tf_acc*100:.1f}%")
            if overall_mae is not None:
                text_lines.append(f"Position MAE: {overall_mae:.1f} bp")
            if len(text_lines) == 2:
                text_lines.append("No detailed data available")
            ax_a1.text(
                0.5, 0.5, "\n".join(text_lines),
                transform=ax_a1.transAxes, ha="center", va="center",
                fontsize=12, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
            ax_a1.set_title("A1: Multi-Site Detection", fontweight="bold")
            ax_a1.axis("off")
    else:
        ax_a1.text(
            0.5, 0.5, "A1: Multi-Site Detection\n\nPENDING",
            transform=ax_a1.transAxes, ha="center", va="center",
            fontsize=14, color=status_colors["pending"],
            fontweight="bold",
        )
        ax_a1.axis("off")
        ax_a1.set_title("A1: Multi-Site Detection", fontweight="bold")

    # ---- Panel B: A2 -- Position MAE by true position ----
    ax_a2 = fig.add_subplot(gs[0, 1])
    a2_block = compiled.get("a2", compiled.get("a1", {}))
    a2_status = a2_block.get("status", "pending")

    a2_data = a2_block.get("a2", {})
    if a2_status != "pending" and a2_data:
        mae_by_pos = a2_data.get("position_mae_by_true_position", {})
        if mae_by_pos:
            pos_keys = sorted(mae_by_pos.keys(), key=lambda x: float(x))
            x_vals = [float(k) for k in pos_keys]
            y_vals = [mae_by_pos[k] for k in pos_keys]
            ax_a2.plot(x_vals, y_vals, "o-", color="coral", linewidth=2, markersize=5)
            bias = a2_data.get("position_bias")
            if bias is not None:
                ax_a2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax_a2.text(
                    0.95, 0.95, f"Bias: {bias:.2f} bp",
                    transform=ax_a2.transAxes, ha="right", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            ax_a2.set_xlabel("True Position (bp)")
            ax_a2.set_ylabel("Position MAE (bp)")
            ax_a2.set_title("A2: Position Accuracy by Location", fontweight="bold")
        else:
            text_lines = ["A2: Position Accuracy", ""]
            mae_overall = a2_data.get("position_mae_overall")
            bias = a2_data.get("position_bias")
            if mae_overall is not None:
                text_lines.append(f"MAE (overall): {mae_overall:.2f} bp")
            if bias is not None:
                text_lines.append(f"Position bias: {bias:.2f} bp")
            if len(text_lines) == 2:
                text_lines.append("No positional breakdown available")
            ax_a2.text(
                0.5, 0.5, "\n".join(text_lines),
                transform=ax_a2.transAxes, ha="center", va="center",
                fontsize=12, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
            ax_a2.set_title("A2: Position Accuracy", fontweight="bold")
            ax_a2.axis("off")
    else:
        ax_a2.text(
            0.5, 0.5, "A2: Position Accuracy\n\nPENDING",
            transform=ax_a2.transAxes, ha="center", va="center",
            fontsize=14, color=status_colors["pending"],
            fontweight="bold",
        )
        ax_a2.axis("off")
        ax_a2.set_title("A2: Position Accuracy", fontweight="bold")

    # ---- Panel C: A3 -- Cross-cell-type transfer ----
    ax_a3 = fig.add_subplot(gs[1, 0])
    a3_block = compiled.get("a3", {})
    a3_status = a3_block.get("status", "pending")

    if a3_status != "pending":
        a3 = a3_block.get("a3", {})
        src = a3.get("source_cell", "K562")
        tgt = a3.get("target_cell", "HepG2")

        metric_names = []
        source_vals = []
        target_vals = []

        metric_map = [
            ("TF Accuracy", "source_tf_accuracy", "target_tf_accuracy"),
            ("Site Detection", "source_site_detection", "target_site_detection"),
            ("Profile r", "source_profile_pearson", "target_profile_pearson"),
        ]
        for label, src_key, tgt_key in metric_map:
            sv = a3.get(src_key)
            tv = a3.get(tgt_key)
            if sv is not None and tv is not None:
                metric_names.append(label)
                source_vals.append(sv * 100)
                target_vals.append(tv * 100)

        if metric_names:
            x = np.arange(len(metric_names))
            width = 0.35
            bars_src = ax_a3.bar(
                x - width / 2, source_vals, width,
                label=f"{src} (source)", color="steelblue", alpha=0.8,
            )
            bars_tgt = ax_a3.bar(
                x + width / 2, target_vals, width,
                label=f"{tgt} (transfer)", color="coral", alpha=0.8,
            )
            ax_a3.set_xticks(x)
            ax_a3.set_xticklabels(metric_names)
            ax_a3.set_ylabel("Performance (%)")
            ax_a3.set_ylim(0, 110)
            ax_a3.legend(loc="upper right", fontsize=9)
            for bar, v in zip(bars_src, source_vals):
                ax_a3.text(bar.get_x() + bar.get_width() / 2, v + 2,
                           f"{v:.0f}", ha="center", fontsize=9)
            for bar, v in zip(bars_tgt, target_vals):
                ax_a3.text(bar.get_x() + bar.get_width() / 2, v + 2,
                           f"{v:.0f}", ha="center", fontsize=9)
        else:
            ax_a3.text(
                0.5, 0.5, "A3: Cross-Cell Transfer\n\nNo numeric data parsed",
                transform=ax_a3.transAxes, ha="center", va="center",
                fontsize=12, color=status_colors["partial"],
            )
            ax_a3.axis("off")

        ax_a3.set_title(f"A3: Cross-Cell Transfer ({src} -> {tgt})", fontweight="bold")
    else:
        ax_a3.text(
            0.5, 0.5, "A3: Cross-Cell Transfer\n\nPENDING",
            transform=ax_a3.transAxes, ha="center", va="center",
            fontsize=14, color=status_colors["pending"],
            fontweight="bold",
        )
        ax_a3.axis("off")
        ax_a3.set_title("A3: Cross-Cell Transfer", fontweight="bold")

    # ---- Panel D: A4 -- JASPAR correlation comparison ----
    ax_a4 = fig.add_subplot(gs[1, 1])
    a4_block = compiled.get("a4", {})
    a4_status = a4_block.get("status", "pending")

    if a4_status != "pending":
        a4 = a4_block.get("a4", {})
        method_comp = a4.get("method_comparison", {})

        if method_comp:
            methods = sorted(method_comp.keys(), key=lambda m: -(method_comp[m] or 0))
            corrs = [method_comp[m] for m in methods]
            colors = ["#2ecc71", "#3498db", "#e67e22"]
            bar_colors = colors[: len(methods)] + ["gray"] * max(0, len(methods) - len(colors))
            bars = ax_a4.barh(
                methods[::-1], corrs[::-1],
                color=bar_colors[::-1], alpha=0.8, edgecolor="black",
            )
            for bar, c in zip(bars, corrs[::-1]):
                ax_a4.text(
                    bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{c:.3f}", va="center", fontsize=10,
                )
            ax_a4.set_xlabel("Mean JASPAR Correlation (r)")
            ax_a4.set_xlim(0, max(corrs) * 1.25 if corrs else 1.0)
            ax_a4.set_title("A4: Motif Recovery -- Method Comparison", fontweight="bold")
        else:
            # Fall back to gradient per-TF if available
            grad_per_tf = a4.get("gradient_per_tf", {})
            if grad_per_tf:
                tf_names = sorted(grad_per_tf.keys())
                corr_vals = [grad_per_tf[t] for t in tf_names]
                valid_vals = [(t, c) for t, c in zip(tf_names, corr_vals) if c is not None]
                if valid_vals:
                    tf_labels, cv = zip(*valid_vals)
                    bar_colors = [
                        "#2ecc71" if c > 0.5 else "#f39c12" if c > 0.3 else "#e74c3c"
                        for c in cv
                    ]
                    ax_a4.barh(
                        list(tf_labels)[::-1], list(cv)[::-1],
                        color=bar_colors[::-1], alpha=0.8, edgecolor="black",
                    )
                    ax_a4.set_xlabel("JASPAR Correlation (r)")
                    ax_a4.set_title("A4: Per-TF Gradient Motif Recovery", fontweight="bold")
                else:
                    ax_a4.text(
                        0.5, 0.5, "A4: Interpretability\n\nNo valid correlations",
                        transform=ax_a4.transAxes, ha="center", va="center",
                        fontsize=12,
                    )
                    ax_a4.axis("off")
            else:
                ax_a4.text(
                    0.5, 0.5, "A4: Interpretability\n\nData loaded but no method comparison",
                    transform=ax_a4.transAxes, ha="center", va="center",
                    fontsize=12, color=status_colors["partial"],
                )
                ax_a4.axis("off")
                ax_a4.set_title("A4: Interpretability", fontweight="bold")
    else:
        ax_a4.text(
            0.5, 0.5, "A4: Interpretability\n\nPENDING",
            transform=ax_a4.transAxes, ha="center", va="center",
            fontsize=14, color=status_colors["pending"],
            fontweight="bold",
        )
        ax_a4.axis("off")
        ax_a4.set_title("A4: Interpretability", fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(compiled):
    """Print a nicely aligned summary to stdout."""
    sep = "=" * 72

    print()
    print(sep)
    print("  BEACON Validation Results -- Compiled Summary")
    print(sep)
    print()

    # -- A1 --
    print("  A1: Multi-Site Co-Binding Detection")
    print("  " + "-" * 42)
    a1_block = compiled.get("a1", {})
    if a1_block.get("status") == "pending":
        print("    Status: PENDING")
        reason = a1_block.get("reason", "")
        if reason:
            print(f"    Reason: {reason}")
    else:
        a1 = a1_block.get("a1", {})
        print(f"    Multi-site detection rate : {_pct(a1.get('multi_site_detection_rate'))}")
        print(f"    Per-site TF accuracy      : {_pct(a1.get('per_site_tf_accuracy'))}")
        print(f"    Overall position MAE      : {_fmt(a1.get('overall_position_mae'), '.1f')} bp")
        mae_by_sites = a1.get("position_mae_by_n_sites")
        if mae_by_sites:
            print()
            print("    Position MAE by number of sites:")
            for n in sorted(mae_by_sites.keys(), key=lambda x: int(x)):
                print(f"      {n} site(s) : {mae_by_sites[n]:.2f} bp")
    print()

    # -- A2 --
    print("  A2: Position Accuracy Analysis")
    print("  " + "-" * 42)
    a2_data = a1_block.get("a2", compiled.get("a2", {}).get("a2", {}))
    a2_status = a1_block.get("status", compiled.get("a2", {}).get("status", "pending"))
    if a2_status == "pending":
        print("    Status: PENDING")
    else:
        print(f"    Position MAE (overall)  : {_fmt(a2_data.get('position_mae_overall'), '.2f')} bp")
        print(f"    Position bias (mean)    : {_fmt(a2_data.get('position_bias'), '.2f')} bp")
        print(f"    Position bias (std)     : {_fmt(a2_data.get('position_bias_std'), '.2f')} bp")
        mae_by_pos = a2_data.get("position_mae_by_true_position")
        if mae_by_pos:
            print()
            print("    Position MAE at various true positions:")
            for pos in sorted(mae_by_pos.keys(), key=lambda x: float(x)):
                print(f"      Position {pos:>6} bp : MAE {mae_by_pos[pos]:.2f} bp")
    print()

    # -- A3 --
    print("  A3: Cross-Cell-Type Transfer")
    print("  " + "-" * 42)
    a3_block = compiled.get("a3", {})
    if a3_block.get("status") == "pending":
        print("    Status: PENDING")
        reason = a3_block.get("reason", "")
        if reason:
            print(f"    Reason: {reason}")
    else:
        a3 = a3_block.get("a3", {})
        src = a3.get("source_cell", "K562")
        tgt = a3.get("target_cell", "HepG2")
        print(f"    Transfer: {src} -> {tgt}")
        common = a3.get("common_tfs")
        if common:
            print(f"    Shared TFs: {', '.join(common)}")
        print()
        header = f"    {'Metric':<22} {src:>8}   {tgt:>8}   Transfer"
        print(header)
        print("    " + "-" * len(header.strip()))
        for label, sk, tk, trk in [
            ("TF Accuracy", "source_tf_accuracy", "target_tf_accuracy", "transfer_tf_accuracy"),
            ("Site Detection", "source_site_detection", "target_site_detection", "transfer_site_detection"),
            ("Profile Pearson", "source_profile_pearson", "target_profile_pearson", "transfer_profile_pearson"),
        ]:
            sv = a3.get(sk)
            tv = a3.get(tk)
            tr = a3.get(trk)
            print(
                f"    {label:<22} "
                f"{_pct(sv):>8}   "
                f"{_pct(tv):>8}   "
                f"{_fmt(tr, '.1f') + '%' if tr is not None else 'N/A':>8}"
            )
        overall = a3.get("overall_transfer_pct")
        if overall is not None:
            print(f"\n    Overall transfer: {overall:.1f}%")
        interp = a3.get("interpretation")
        if interp:
            print(f"    Interpretation: {interp}")
    print()

    # -- A4 --
    print("  A4: Interpretability -- JASPAR Motif Correlation")
    print("  " + "-" * 42)
    a4_block = compiled.get("a4", {})
    if a4_block.get("status") == "pending":
        print("    Status: PENDING")
        reason = a4_block.get("reason", "")
        if reason:
            print(f"    Reason: {reason}")
    else:
        a4 = a4_block.get("a4", {})
        method_comp = a4.get("method_comparison", {})
        if method_comp:
            print()
            print("    Method comparison (mean JASPAR r):")
            for method in sorted(method_comp.keys(), key=lambda m: -(method_comp[m] or 0)):
                print(f"      {method:<20} : {_fmt(method_comp[method])}")

        grad_per_tf = a4.get("gradient_per_tf", {})
        if grad_per_tf:
            print()
            print("    Per-TF JASPAR correlation (gradient method):")
            for tf in sorted(grad_per_tf.keys()):
                print(f"      {tf:<8} : {_fmt(grad_per_tf[tf])}")

        attn_per_tf = a4.get("attention_per_tf", {})
        if attn_per_tf:
            print()
            print("    Per-TF attention-profile correlation:")
            for tf in sorted(attn_per_tf.keys()):
                print(f"      {tf:<8} : {_fmt(attn_per_tf[tf])}")
    print()

    # -- Overall status --
    print(sep)
    print("  Validation Status")
    print(sep)
    for label, key in [
        ("A1: Multi-site detection", "a1"),
        ("A2: Position accuracy   ", "a2"),
        ("A3: Cross-cell transfer ", "a3"),
        ("A4: Interpretability    ", "a4"),
    ]:
        block = compiled.get(key, {})
        st = block.get("status", "pending")
        marker = "[COMPLETE]" if st == "complete" else ("[PARTIAL]" if st == "partial" else "[PENDING]")
        print(f"    {label}  {marker}")
    print()
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A5: Compile validation results from A1-A4 experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Expected directory layout under --validation-dir:
  multi_tf_cobinding/   A1+A2 results (*.json)
  cross_cell_type/      A3 results (*.json)
  interpretability/     A4 results (*.json)

Example:
  python scripts/compile_validation_results.py
  python scripts/compile_validation_results.py --validation-dir outputs/validation
  python scripts/compile_validation_results.py --output-dir results/paper_tables
""",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Root directory containing A1-A4 subdirectories "
             "(default: <project>/outputs/validation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for compiled outputs "
             "(default: same as --validation-dir)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root.
    validation_dir = args.validation_dir
    if validation_dir is None:
        validation_dir = base_dir / "outputs" / "validation"
    elif not validation_dir.is_absolute():
        validation_dir = base_dir / validation_dir

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = validation_dir
    elif not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    cobinding_dir = validation_dir / "multi_tf_cobinding"
    cross_cell_dir = validation_dir / "cross_cell_type"
    interp_dir = validation_dir / "interpretability"

    print("=" * 72)
    print("  A5: Compile Validation Results")
    print("=" * 72)
    print()
    print(f"  Validation directory : {validation_dir}")
    print(f"  Output directory     : {output_dir}")
    print()
    print(f"  A1+A2 source : {cobinding_dir}")
    print(f"    exists     : {cobinding_dir.exists()}")
    print(f"  A3 source    : {cross_cell_dir}")
    print(f"    exists     : {cross_cell_dir.exists()}")
    print(f"  A4 source    : {interp_dir}")
    print(f"    exists     : {interp_dir.exists()}")
    print()

    # ------------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------------
    print("Loading A1+A2 (multi-TF co-binding)...")
    a1_a2_results = load_a1_a2(cobinding_dir)
    print(f"  Status: {a1_a2_results['status']}")
    if "raw_files" in a1_a2_results:
        print(f"  Files loaded: {', '.join(a1_a2_results['raw_files'])}")

    print()
    print("Loading A3 (cross-cell-type transfer)...")
    a3_results = load_a3(cross_cell_dir)
    print(f"  Status: {a3_results['status']}")
    if "raw_files" in a3_results:
        print(f"  Files loaded: {', '.join(a3_results['raw_files'])}")

    print()
    print("Loading A4 (interpretability)...")
    a4_results = load_a4(interp_dir)
    print(f"  Status: {a4_results['status']}")
    if "raw_files" in a4_results:
        print(f"  Files loaded: {', '.join(a4_results['raw_files'])}")
    print()

    # ------------------------------------------------------------------
    # Build compiled structure.  A1 and A2 come from the same source but
    # are reported separately.
    # ------------------------------------------------------------------
    compiled = {
        "timestamp": datetime.now().isoformat(),
        "validation_dir": str(validation_dir),
        "a1": a1_a2_results,
        "a2": {
            "status": a1_a2_results.get("status", "pending"),
            "a2": a1_a2_results.get("a2", {}),
        },
        "a3": a3_results,
        "a4": a4_results,
    }

    # ------------------------------------------------------------------
    # Print to stdout
    # ------------------------------------------------------------------
    print_summary(compiled)

    # ------------------------------------------------------------------
    # Save compiled JSON
    # ------------------------------------------------------------------
    json_path = output_dir / "compiled_results.json"

    # Prepare a serialisable copy (remove sets and non-JSON types)
    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(json_path, "w") as f:
        json.dump(_make_serialisable(compiled), f, indent=2)
    print(f"  Compiled JSON saved to: {json_path}")

    # ------------------------------------------------------------------
    # Save markdown summary
    # ------------------------------------------------------------------
    md_path = output_dir / "validation_summary.md"
    md_text = generate_markdown(compiled, md_path)
    print(f"  Markdown summary saved to: {md_path}")

    # ------------------------------------------------------------------
    # Create summary figure
    # ------------------------------------------------------------------
    fig_path = output_dir / "validation_summary.png"
    create_summary_figure(compiled, fig_path)
    print(f"  Summary figure saved to: {fig_path}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
