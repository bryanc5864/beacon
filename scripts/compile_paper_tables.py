#!/usr/bin/env python3
"""
Compile all evaluation results into publication-quality tables (Markdown + LaTeX).

Reads JSON outputs from the other evaluation scripts and produces a
unified set of tables suitable for the ISMB paper.

Usage:
    python scripts/compile_paper_tables.py
    python scripts/compile_paper_tables.py --eval-dir outputs/evaluation
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
BPNET_AGG = {
    "K562-7tf": 0.616, "K562-fulltf": 0.564,
    "HepG2-7tf": 0.625, "HepG2-fulltf": 0.612,
}


def _load(dir, name):
    p = dir / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"  (missing: {p.name})")
    return None


def _ci(d, fmt=".3f"):
    if d is None:
        return "N/A"
    return f"{d['mean']:{fmt}} [{d.get('ci_lower',0):{fmt}}, {d.get('ci_upper',0):{fmt}}]"


# ── Markdown tables ───────────────────────────────────────────────────
def table_main(stats, metrics):
    lines = ["## Table 1: Main Results (BEACON vs BPNet)", "",
             "| Dataset | Model | Profile r [95% CI] | JSD | MNLL |",
             "|---------|-------|--------------------|-----|------|"]
    datasets = ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]
    for ds in datasets:
        r_ci = jsd = mnll = "N/A"
        if metrics and ds in metrics:
            r_ci = _ci(metrics[ds].get("pearson"))
            jsd = _ci(metrics[ds].get("jsd"), ".4f")
            mnll = _ci(metrics[ds].get("mnll"), ".3f")
        elif stats and "bootstrap_cis" in stats and ds in stats["bootstrap_cis"]:
            r_ci = _ci(stats["bootstrap_cis"][ds].get("profile_pearson"))
        lines.append(f"| {ds} | **BEACON** | {r_ci} | {jsd} | {mnll} |")
        lines.append(f"| | BPNet | {BPNET_AGG.get(ds, 0):.3f} | --- | --- |")

    if stats and "beacon_vs_bpnet" in stats:
        lines += ["", "### Significance (BEACON > BPNet)", "",
                   "| Dataset | p (Bonf.) | Cohen's d |",
                   "|---------|-----------|-----------|"]
        for ds in datasets:
            bvb = stats["beacon_vs_bpnet"].get(ds, {})
            p = bvb.get("wilcoxon_p_bonf")
            d = bvb.get("cohens_d")
            if p is not None:
                lines.append(f"| {ds} | {p:.2e} | {d:.3f} |")
    return "\n".join(lines)


def table_per_tf(stats, metrics):
    lines = ["## Table 2: Per-TF Profile Pearson r [95% CI]", ""]
    if stats and "bootstrap_cis" in stats:
        for model, data in stats["bootstrap_cis"].items():
            lines += [f"### {model}", "",
                      "| TF | r [95% CI] | n |", "|----|-----------|---|"]
            for tf, ci in data.get("per_tf", {}).items():
                lines.append(f"| {tf} | {_ci(ci)} | {ci.get('n_samples',0)} |")
            lines.append("")
    return "\n".join(lines)


def table_ablation(stats):
    lines = ["## Table 3: Ablation Study (K562-7tf)", "",
             "| Variant | Profile r [95% CI] | Δ vs Baseline | p (Bonf.) |",
             "|---------|-------------------|---------------|-----------|"]
    if not stats or "ablation_significance" not in stats:
        return "\n".join(lines + ["| *(not yet computed)* | | | |"])
    ab = stats["ablation_significance"]
    order = ["baseline", "slot_losses", "slot_dropout", "pcgrad", "gradnorm", "full"]
    labels = {"baseline": "Baseline", "slot_losses": "+Slot losses",
              "slot_dropout": "+Slot dropout", "pcgrad": "+PCGrad",
              "gradnorm": "+GradNorm", "full": "Full"}
    for v in order:
        ci = ab.get("per_variant_ci", {}).get(v)
        if v == "baseline":
            lines.append(f"| {labels[v]} | {_ci(ci)} | --- | --- |")
        else:
            pw = ab.get("pairwise", {}).get(f"{v}_vs_baseline", {})
            d = pw.get("delta", 0)
            p = pw.get("p_bonf", 1)
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            lines.append(f"| {labels[v]} | {_ci(ci)} | {d:+.4f} | {p:.2e} {sig} |")
    return "\n".join(lines)


def table_ism(ism):
    lines = ["## Table 4: Enhanced In-Silico Mutagenesis", "",
             "| TF | Motif Δprofile | Random Δprofile | Ratio | p (M-W) |",
             "|----|---------------|----------------|-------|---------|"]
    if not ism:
        return "\n".join(lines + ["| *(not yet computed)* | | | | |"])
    comp = ism.get("comparison", {})
    for tf in ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]:
        c = comp.get(tf)
        if not c:
            continue
        lines.append(f"| {tf} | {c['motif_mean']:.4f} | {c['random_mean']:.4f} | "
                     f"{c['ratio']:.1f}x | {c['mann_whitney_p']:.2e} |")
    return "\n".join(lines)


def table_error(err):
    lines = ["## Table 5: Error Analysis", ""]
    if not err:
        return "\n".join(lines + ["*(not yet computed)*"])
    for model, data in err.items():
        lines.append(f"### {model}")
        strat = data.get("signal_stratification", {}).get("overall", {})
        if strat:
            lines += ["", "| Quartile | n | Mean r |", "|----------|---|--------|"]
            for q, d in strat.items():
                lines.append(f"| {q} | {d['n']} | {d['mean_r']:.4f} |")
        cal = data.get("calibration", {})
        if cal:
            lines.append(f"\nCalibration ECE = {cal.get('ece', 0):.4f}")
        fail = data.get("failure_analysis", {})
        if fail:
            lines.append(f"\nBottom 10% mean r = {fail.get('mean_r', 0):.4f}")
        lines.append("")
    return "\n".join(lines)


def table_seeds(seeds):
    lines = ["## Table 6: Multi-Seed Consistency (K562-7tf)", "",
             "| Seed | Pearson r [95% CI] | JSD | MNLL |",
             "|------|-------------------|-----|------|"]
    if not seeds or "per_seed" not in seeds:
        return "\n".join(lines + ["| *(not yet computed)* | | | |"])
    for s, d in seeds["per_seed"].items():
        r_ci = _ci(d.get("pearson"))
        jsd = d.get("jsd", {}).get("mean", 0)
        mnll = d.get("mnll", {}).get("mean", 0)
        lines.append(f"| {s} | {r_ci} | {jsd:.4f} | {mnll:.3f} |")
    cs = seeds.get("cross_seed", {})
    if cs:
        lines.append(f"\n**Across seeds:** {cs.get('mean_r',0):.4f} ± {cs.get('std_r',0):.4f}")
    return "\n".join(lines)


def table_coop(coop):
    lines = ["## Table 7: Cooperative Binding (Cross-Sample)", ""]
    if not coop:
        return "\n".join(lines + ["*(not yet computed)*"])

    attn = coop.get("attention_patterns", {})
    if attn:
        lines += ["### Attention Similarity for Known TF Pairs", "",
                   "| Pair | Attn cosine | vs Control | Biology |",
                   "|------|------------|-----------|---------|"]
        for p, d in attn.items():
            cos = d.get("mean_attention_cosine")
            diff = d.get("partner_vs_control_diff")
            bio = d.get("biology", "")
            cos_str = f"{cos:.3f}" if cos is not None else "N/A"
            diff_str = f"{diff:+.3f}" if diff is not None else "N/A"
            lines.append(f"| {p} | {cos_str} | {diff_str} | {bio} |")

    shapes = coop.get("profile_shapes", {})
    if shapes:
        lines += ["", "### Profile Shape Similarity", "",
                   "| Pair | Profile r | Control mean r | More similar? |",
                   "|------|----------|---------------|--------------|"]
        for p, d in shapes.items():
            r = d.get("mean_profile_pearson", 0)
            ctrl = d.get("control_pearson", {}).get("mean", 0)
            more = d.get("partner_more_similar", False)
            lines.append(f"| {p} | {r:.3f} | {ctrl:.3f} | {'Yes' if more else 'No'} |")

    return "\n".join(lines)


def table_position(pos):
    lines = ["## Table 8: Position Prediction Accuracy", "",
             "| Dataset | MAE (bp) | MedAE (bp) | n matched |",
             "|---------|----------|------------|-----------|"]
    if not pos:
        return "\n".join(lines + ["| *(not yet computed)* | | | |"])
    for ds in ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]:
        d = pos.get(ds)
        if not d or d.get("n_matched", 0) == 0:
            continue
        lines.append(f"| {ds} | {d['overall_mae']:.3f} | {d['overall_medae']:.3f} | {d['n_matched']} |")

    # Per-TF detail for K562-7tf
    k562 = pos.get("K562-7tf", {})
    per_tf = k562.get("per_tf", {})
    if per_tf:
        lines += ["", "### Per-TF Position Accuracy (K562-7tf)", "",
                   "| TF | MAE (bp) | MedAE (bp) | Signed bias | n |",
                   "|----|----------|------------|-------------|---|"]
        for tf, d in per_tf.items():
            if d.get("mae") is not None:
                bias = d.get("signed_mean", 0)
                lines.append(f"| {tf} | {d['mae']:.3f} | {d['medae']:.3f} | "
                             f"{bias:+.3f} | {d['n']} |")

    return "\n".join(lines)


def table_saliency(sal):
    lines = ["## Table 9: Gradient Saliency Analysis", ""]
    if not sal:
        return "\n".join(lines + ["*(not yet computed)*"])
    for model, data in sal.items():
        lines.append(f"### {model}")
        ag = data.get("attention_vs_gradient", {})
        if ag and "mean" in ag:
            ci = ag.get("ci", {})
            ci_str = f" [{ci.get('ci_lower',0):.3f}, {ci.get('ci_upper',0):.3f}]" if ci else ""
            lines.append(f"- Attention-gradient correlation: {ag['mean']:.3f}{ci_str}")
            lines.append(f"- Positive correlations: {ag.get('n_positive',0)}/{ag.get('n_samples',0)}")
        enr = data.get("motif_enrichment", {})
        if enr:
            lines.append(f"- Gradient at motifs vs random: {enr.get('motif_vs_random_ratio',0):.2f}x")
            lines.append(f"- Gradient at motifs vs flanking: {enr.get('motif_vs_flanking_ratio',0):.2f}x")
        lines.append("")
    return "\n".join(lines)


def table_slot_embeddings(emb):
    lines = ["## Table 10: Slot Embedding Analysis", ""]
    if not emb:
        return "\n".join(lines + ["*(not yet computed)*"])

    for model in ["K562-7tf", "K562-fulltf"]:
        d = emb.get(model)
        if not d:
            continue
        lines.append(f"### {model}")
        probe = d.get("linear_probe", {})
        if probe:
            lines.append(f"- Linear probe accuracy: {probe['mean_accuracy']:.3f} ± {probe['std_accuracy']:.3f}")
        clust = d.get("tf_family_clustering", {})
        if clust and clust.get("silhouette_score") is not None:
            lines.append(f"- Silhouette score: {clust['silhouette_score']:.3f}")
            lines.append(f"- Adjusted Rand Index: {clust['adjusted_rand_index']:.3f}")
        lines.append("")
    return "\n".join(lines)


def table_denovo(motifs):
    lines = ["## Table 11: De Novo Motif Discovery (Gradient-Based)", ""]
    if not motifs:
        return "\n".join(lines + ["*(not yet computed)*"])

    per_tf = motifs.get("per_tf", {})
    if per_tf:
        lines += ["| TF | JASPAR Pearson | IC (bits) | Motif length |",
                   "|----|--------------|-----------|-------------|"]
        for tf in ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]:
            d = per_tf.get(tf)
            if not d:
                continue
            jp = d.get("jaspar_pearson")
            ic = d.get("info_content", 0)
            ml = d.get("motif_length", 0)
            jp_str = f"{jp:.3f}" if jp is not None else "N/A"
            lines.append(f"| {tf} | {jp_str} | {ic:.2f} | {ml} |")

    mean_r = motifs.get("mean_jaspar_pearson")
    if mean_r is not None:
        lines.append(f"\n**Mean JASPAR Pearson: {mean_r:.3f}** (attention-based: 0.574)")

    comp = motifs.get("comparison_to_attention", {})
    if comp and comp.get("improvement") is not None:
        lines.append(f"Improvement: {comp['improvement']:+.3f}")

    return "\n".join(lines)


def table_throughput(tp):
    lines = ["## Table 12: Inference Throughput", ""]
    if not tp:
        return "\n".join(lines + ["*(not yet computed)*"])

    for model_name in ["beacon", "bpnet"]:
        d = tp.get(model_name)
        if not d:
            continue
        label = "BEACON" if model_name == "beacon" else "BPNet"
        params = d.get("parameters", {})
        lines.append(f"### {label} ({params.get('total_params', 0):,} params)")
        br = d.get("batch_results", {})
        if br:
            lines += ["", "| Batch | Samples/sec | Latency (ms) | Peak mem (MB) |",
                       "|-------|-------------|-------------|---------------|"]
            for bs, r in br.items():
                if "error" in r:
                    lines.append(f"| {bs} | OOM | | |")
                else:
                    lines.append(f"| {bs} | {r.get('samples_per_sec',0):.0f} | "
                                 f"{r.get('mean_latency_ms',0):.1f} | "
                                 f"{r.get('peak_memory_mb',0):.0f} |")
        lines.append("")
    return "\n".join(lines)


def table_attention_localization(attn):
    lines = ["## Table 13: Attention Localization", "",
             "| Dataset | Mean dist (bp) | Median dist (bp) | Within 50bp | Enrichment | n |",
             "|---------|---------------|------------------|-------------|------------|---|"]
    if not attn:
        return "\n".join(lines + ["| *(not yet computed)* | | | | | |"])
    for ds in ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]:
        d = attn.get(ds, {}).get("localization")
        if not d:
            continue
        ci = d.get("distance_ci", {})
        ci_str = f" [{ci.get('ci_lower',0):.1f}, {ci.get('ci_upper',0):.1f}]" if ci else ""
        lines.append(f"| {ds} | {d['mean_distance_bp']:.1f}{ci_str} | "
                     f"{d['median_distance_bp']:.0f} | "
                     f"{d['within_50bp']:.1%} | "
                     f"{d['attention_enrichment']:.1f}x | "
                     f"{d['n_sites']} |")

    # Per-TF detail for K562-7tf
    k562_tf = attn.get("K562-7tf", {}).get("per_tf_localization", {})
    if k562_tf:
        lines += ["", "### Per-TF Attention Localization (K562-7tf)", "",
                   "| TF | Mean dist (bp) | Within 50bp | n |",
                   "|----|---------------|-------------|---|"]
        for tf, d in k562_tf.items():
            if 'mean_distance_bp' in d:
                lines.append(f"| {tf} | {d['mean_distance_bp']:.1f} | "
                             f"{d['within_50bp']:.1%} | {d['n']} |")

    # Slot utilization
    lines += ["", "### Slot Utilization", "",
               "| Dataset | Mean active | Effective slots | Ever active / Total |",
               "|---------|------------|----------------|---------------------|"]
    for ds in ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]:
        u = attn.get(ds, {}).get("slot_utilization")
        if not u:
            continue
        lines.append(f"| {ds} | {u['mean_active_slots']:.2f} | "
                     f"{u['effective_n_slots']:.2f} | "
                     f"{u['n_slots_ever_active']}/{len(u['slot_activation_freq'])} |")

    return "\n".join(lines)


def table_robustness(rob):
    lines = ["## Table 14: Robustness Analysis", ""]
    if not rob:
        return "\n".join(lines + ["*(not yet computed)*"])

    # Robustness curve
    beacon_rob = rob.get("beacon_robustness", {})
    if beacon_rob:
        lines += ["### BEACON Robustness to Sequence Perturbation", "",
                   "| Mutation rate | Profile r | Retained |",
                   "|--------------|-----------|----------|"]
        for rate in sorted(beacon_rob.keys(), key=float):
            d = beacon_rob[rate]
            lines.append(f"| {float(rate)*100:.0f}% | {d['mean_r']:.4f} | {d['relative_retained']:.1%} |")

    # BPNet comparison
    bpnet_rob = rob.get("bpnet_robustness", {})
    if bpnet_rob:
        lines += ["", "### BPNet Robustness", "",
                   "| Mutation rate | Profile r | Retained |",
                   "|--------------|-----------|----------|"]
        for rate in sorted(bpnet_rob.keys(), key=float):
            d = bpnet_rob[rate]
            lines.append(f"| {float(rate)*100:.0f}% | {d['mean_r']:.4f} | {d['relative_retained']:.1%} |")

    # Multi-site test
    multi = rob.get("multi_site", {})
    if multi and "error" not in multi:
        lines += ["", "### Multi-Site Synthetic Test", "",
                   f"- Pairs tested: {multi.get('n_pairs', 0)}",
                   f"- Mean active slots (combined): {multi.get('mean_active_slots_combined', 0):.2f}",
                   f"- Mean active slots (single): {multi.get('mean_active_slots_single', 0):.2f}",
                   f"- 2+ slots activated: {multi.get('two_plus_slots_rate', 0):.1%}",
                   f"- Both TFs detected: {multi.get('both_tfs_detected_rate', 0):.1%}"]

    # Fair BPNet comparison
    fair = rob.get("fair_bpnet_comparison", {})
    if fair:
        lines += ["", "### Fair BPNet Per-TF Comparison", "",
                   "| TF | BEACON r | BPNet r | Delta |",
                   "|----|---------|---------|-------|"]
        for tf, d in fair.items():
            if 'beacon_r' in d:
                lines.append(f"| {tf} | {d['beacon_r']:.3f} | {d['bpnet_r']:.3f} | {d.get('delta', 0):+.3f} |")

    return "\n".join(lines)


def table_cross_cell(cc):
    lines = ["## Table 15: Extended Cross-Cell Analysis", ""]
    if not cc:
        return "\n".join(lines + ["*(not yet computed)*"])

    mt = cc.get("motif_transfer", {})
    if mt:
        lines += ["### Motif-Level Transfer (K562 → HepG2)", "",
                   "| TF | Gradient correlation |",
                   "|----|-------------------|"]
        for tf, d in mt.items():
            lines.append(f"| {tf} | {d['importance_correlation']:.3f} |")

    at = cc.get("attention_transfer", {})
    if at:
        lines += ["", "### Attention Pattern Transfer", "",
                   "| TF | K562 entropy | HepG2 entropy | K562 peaks | HepG2 peaks |",
                   "|----|-------------|---------------|-----------|-------------|"]
        for tf, d in at.items():
            lines.append(f"| {tf} | {d['k562_entropy']:.3f} | {d['hepg2_entropy']:.3f} | "
                         f"{d['k562_peak_count']:.1f} | {d['hepg2_peak_count']:.1f} |")

    fa = cc.get("feature_analysis", {})
    if fa:
        lines += ["", "### Variance Decomposition",
                   f"- R² cell type: {fa.get('r2_cell_type', 0):.3f}",
                   f"- R² TF identity: {fa.get('r2_tf_identity', 0):.3f}",
                   f"- R² combined: {fa.get('r2_combined', 0):.3f}"]

    return "\n".join(lines)


# ── LaTeX snippets ────────────────────────────────────────────────────
def latex_main(stats, metrics):
    lines = [
        "% Table 1: Main Results",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Profile prediction. 95\\% bootstrap CIs.}",
        "\\label{tab:main}", "\\begin{tabular}{llccc}", "\\toprule",
        "Dataset & Model & Pearson $r$ & JSD & MNLL \\\\", "\\midrule",
    ]
    for ds in ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]:
        r = jsd = mnll = "---"
        if metrics and ds in metrics:
            r = _ci(metrics[ds].get("pearson"))
            jsd = _ci(metrics[ds].get("jsd"), ".4f")
            mnll = _ci(metrics[ds].get("mnll"), ".3f")
        lines.append(f"{ds} & \\textbf{{BEACON}} & {r} & {jsd} & {mnll} \\\\")
        lines.append(f" & BPNet & {BPNET_AGG.get(ds,0):.3f} & --- & --- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compile paper tables")
    parser.add_argument("--eval-dir", default="outputs/evaluation")
    args = parser.parse_args()

    edir = BASE_DIR / args.eval_dir
    tdir = edir / "paper_tables"
    tdir.mkdir(parents=True, exist_ok=True)

    stats = _load(edir, "statistical_evaluation.json")
    metrics = stats.get("profile_metrics") if stats else None
    ism = _load(edir, "enhanced_ism.json")
    err = _load(edir, "error_analysis.json")
    seeds = _load(edir, "multi_seed.json")
    coop = _load(edir, "cooperative_binding.json")
    pos = _load(edir, "position_accuracy.json")
    sal = _load(edir, "saliency_analysis.json")
    emb = _load(edir, "slot_embedding_analysis.json")
    tp = _load(edir, "throughput_benchmark.json")
    cc = _load(edir, "cross_cell_analysis.json")
    attn_loc = _load(edir, "attention_localization.json")
    rob = _load(edir, "robustness_analysis.json")

    md_tables = {
        "table1_main.md": table_main(stats, metrics),
        "table2_per_tf.md": table_per_tf(stats, metrics),
        "table3_ablation.md": table_ablation(stats),
        "table4_ism.md": table_ism(ism),
        "table5_error.md": table_error(err),
        "table6_seeds.md": table_seeds(seeds),
        "table7_cooperative.md": table_coop(coop),
        "table8_position.md": table_position(pos),
        "table9_saliency.md": table_saliency(sal),
        "table10_slot_embeddings.md": table_slot_embeddings(emb),
        "table11_throughput.md": table_throughput(tp),
        "table12_cross_cell.md": table_cross_cell(cc),
        "table13_attention_localization.md": table_attention_localization(attn_loc),
        "table14_robustness.md": table_robustness(rob),
    }
    for name, content in md_tables.items():
        (tdir / name).write_text(content + "\n")
        print(f"  {name}")

    (tdir / "table1_main.tex").write_text(latex_main(stats, metrics) + "\n")
    print(f"  table1_main.tex")

    # Combined
    combined = [f"# Evaluation Results", f"*{datetime.now():%Y-%m-%d %H:%M}*", ""]
    for content in md_tables.values():
        combined += [content, "", "---", ""]
    (tdir / "all_tables.md").write_text("\n".join(combined))
    print(f"  all_tables.md")

    all_sources = [stats, ism, err, seeds, coop, pos, sal, emb, tp, cc, attn_loc, rob]
    n_loaded = sum(1 for x in all_sources if x)
    print(f"\nCompiled {len(md_tables)} tables ({n_loaded}/{len(all_sources)} data sources)")
    print(f"Output: {tdir}")


if __name__ == "__main__":
    sys.exit(main())
