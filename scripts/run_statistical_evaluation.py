#!/usr/bin/env python3
"""
Statistical evaluation: bootstrap CIs, significance tests, profile metrics.

Covers Phase 1 (statistical rigor) and Phase 2 (JSD / MNLL / AUPRC) of
the comprehensive evaluation pipeline.

Usage:
    python scripts/run_statistical_evaluation.py --device cuda:2
    python scripts/run_statistical_evaluation.py --device cuda:2 --phases 1a,1d
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_model, load_test_data, run_model_batch, per_sample_pearson,
    hungarian_accuracy_per_site, json_serialize,
    bootstrap_ci, paired_wilcoxon, bonferroni, cohens_d,
    per_sample_jsd, per_sample_mnll, site_auprc,
    DATASET_CONFIGS, MODEL_CHECKPOINTS, ABLATION_CHECKPOINTS, BASE_DIR,
)


# ── Phase 1A: Bootstrap CIs for all models ────────────────────────────
def phase_1a(device, n_boot):
    print("=" * 70)
    print("Phase 1A: Bootstrap 95% CIs for main metrics")
    print("=" * 70)

    results = {}
    for name, cfg in DATASET_CONFIGS.items():
        ckpt = BASE_DIR / MODEL_CHECKPOINTS[name]
        test_path = BASE_DIR / cfg["data_dir"] / "test.h5"
        if not ckpt.exists() or not test_path.exists():
            print(f"  SKIP {name}: missing files")
            continue

        print(f"\n  {name}:")
        model = load_model(ckpt, cfg["n_tfs"], device)
        seqs, profs, tf_idx, bs = load_test_data(test_path)
        out = run_model_batch(model, seqs, device)

        sample_rs = per_sample_pearson(out['profile'], profs)
        r_ci = bootstrap_ci(sample_rs, n_boot)
        print(f"    Profile r: {r_ci['mean']:.4f} [{r_ci['ci_lower']:.4f}, {r_ci['ci_upper']:.4f}]")

        # Per-TF CIs
        per_tf = {}
        for ti, tn in enumerate(cfg["tf_names"]):
            mask = tf_idx == ti
            if mask.sum() < 5:
                continue
            ci = bootstrap_ci(sample_rs[mask], n_boot)
            per_tf[tn] = ci
            print(f"      {tn:>8}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] (n={ci['n_samples']})")

        # TF accuracy CI
        tf_correct = hungarian_accuracy_per_site(out, bs, cfg["n_tfs"])
        tf_ci = bootstrap_ci(tf_correct, n_boot) if len(tf_correct) > 0 else None
        if tf_ci:
            print(f"    TF acc:    {tf_ci['mean']:.4f} [{tf_ci['ci_lower']:.4f}, {tf_ci['ci_upper']:.4f}]")

        results[name] = {
            "profile_pearson": r_ci,
            "tf_accuracy": tf_ci,
            "per_tf": per_tf,
            "n_samples": len(seqs),
        }
        del model; torch.cuda.empty_cache()
    return results


# ── Phase 1B: BEACON vs BPNet significance ────────────────────────────
def phase_1b(device, n_boot):
    print("\n" + "=" * 70)
    print("Phase 1B: BEACON vs BPNet significance tests")
    print("=" * 70)

    n_datasets = len(DATASET_CONFIGS)
    results = {}

    for name, cfg in DATASET_CONFIGS.items():
        ckpt = BASE_DIR / MODEL_CHECKPOINTS[name]
        test_path = BASE_DIR / cfg["data_dir"] / "test.h5"
        if not ckpt.exists() or not test_path.exists():
            continue

        print(f"\n  {name}:")
        model = load_model(ckpt, cfg["n_tfs"], device)
        seqs, profs, *_ = load_test_data(test_path)
        out = run_model_batch(model, seqs, device)
        beacon_rs = per_sample_pearson(out['profile'], profs)
        beacon_ci = bootstrap_ci(beacon_rs, n_boot)

        bpnet_rs = _eval_bpnet(name, seqs, profs, device)
        if bpnet_rs is not None:
            n_common = min(len(beacon_rs), len(bpnet_rs))
            w = paired_wilcoxon(beacon_rs[:n_common], bpnet_rs[:n_common])
            d = cohens_d(beacon_rs[:n_common], bpnet_rs[:n_common])
            p_corr = bonferroni(w["pvalue"], n_datasets)
            bpnet_ci = bootstrap_ci(bpnet_rs[:n_common], n_boot)
            print(f"    BEACON: {beacon_ci['mean']:.4f}  BPNet: {bpnet_ci['mean']:.4f}")
            print(f"    p={w['pvalue']:.2e}  corrected={p_corr:.2e}  d={d:.3f}")
            results[name] = {
                "beacon_ci": beacon_ci, "bpnet_ci": bpnet_ci,
                "wilcoxon_p": w["pvalue"], "wilcoxon_p_bonf": p_corr,
                "cohens_d": d, "n_common": n_common,
            }
        else:
            agg = _bpnet_aggregate(name)
            print(f"    BEACON: {beacon_ci['mean']:.4f}  BPNet (agg): {agg:.4f}" if agg else
                  f"    BEACON: {beacon_ci['mean']:.4f}  BPNet: N/A")
            results[name] = {"beacon_ci": beacon_ci, "bpnet_aggregate": agg}

        del model; torch.cuda.empty_cache()
    return results


def _eval_bpnet(dataset_name, seqs, profs, device):
    """Try to evaluate BPNet per-sample. Returns None if unavailable."""
    try:
        from bpnetlite import BPNet
    except ImportError:
        return None
    paths = [BASE_DIR / "bpnet.64.8.final.torch",
             BASE_DIR / f"outputs/bpnet_benchmark/bpnet_{dataset_name}.torch"]
    model_path = next((p for p in paths if p.exists()), None)
    if model_path is None:
        return None
    try:
        loaded = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, BPNet):
            bpnet = loaded
        else:
            bpnet = BPNet(n_filters=64, n_layers=8, n_outputs=1,
                           n_control_tracks=0, trimming=500, verbose=False)
            bpnet.load_state_dict(loaded)
        bpnet = bpnet.to(device).eval()
        X = seqs.transpose(0, 2, 1)
        trim = 500
        corrs = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                bx = torch.from_numpy(X[i:i+32]).float().to(device)
                pred = bpnet(bx)
                pp = (pred[0] if isinstance(pred, tuple) else pred).cpu().numpy()
                gt = profs[i:i+32, trim:seqs.shape[1]-trim]
                for j in range(len(bx)):
                    p, g = pp[j].flatten(), gt[j].flatten()
                    if p.std() > 0 and g.std() > 0:
                        r = np.corrcoef(p, g)[0, 1]
                        corrs.append(r if not np.isnan(r) else 0.0)
                    else:
                        corrs.append(0.0)
        del bpnet; torch.cuda.empty_cache()
        return np.array(corrs)
    except Exception as e:
        print(f"    BPNet eval failed: {e}")
        return None


def _bpnet_aggregate(name):
    p = BASE_DIR / f"outputs/bpnet_benchmark/bpnet_{name}_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f).get("mean_profile_pearson")
    return None


# ── Phase 1D: Ablation significance ───────────────────────────────────
def phase_1d(device, n_boot):
    print("\n" + "=" * 70)
    print("Phase 1D: Ablation significance tests")
    print("=" * 70)

    test_path = BASE_DIR / "data/processed/multi_tf_k562/test.h5"
    if not test_path.exists():
        return {}
    seqs, profs, *_ = load_test_data(test_path)

    variant_rs = {}
    for vname, rel in ABLATION_CHECKPOINTS.items():
        ckpt = BASE_DIR / rel
        if not ckpt.exists():
            print(f"  SKIP {vname}")
            continue
        model = load_model(ckpt, 7, device)
        out = run_model_batch(model, seqs, device)
        rs = per_sample_pearson(out['profile'], profs)
        variant_rs[vname] = rs
        ci = bootstrap_ci(rs, n_boot)
        print(f"  {vname:>14}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        del model; torch.cuda.empty_cache()

    pairs = [("baseline", v) for v in ["slot_losses", "slot_dropout", "pcgrad", "gradnorm", "full"]
             if v in variant_rs]
    pairs += [("slot_losses", "slot_dropout"), ("slot_dropout", "full")]
    pairs = [(a, b) for a, b in pairs if a in variant_rs and b in variant_rs]
    n_tests = len(pairs)

    pairwise = {}
    print(f"\n  Pairwise (Bonferroni, {n_tests} tests):")
    for a, b in pairs:
        w = paired_wilcoxon(variant_rs[b], variant_rs[a])
        d = cohens_d(variant_rs[b], variant_rs[a])
        p_corr = bonferroni(w["pvalue"], n_tests)
        delta = float(np.mean(variant_rs[b]) - np.mean(variant_rs[a]))
        print(f"    {b} vs {a}: Δ={delta:+.4f}  p={p_corr:.2e}  d={d:.3f}")
        pairwise[f"{b}_vs_{a}"] = {
            "delta": delta, "p": w["pvalue"], "p_bonf": p_corr, "d": d,
        }

    return {
        "per_variant_ci": {v: bootstrap_ci(rs, n_boot) for v, rs in variant_rs.items()},
        "pairwise": pairwise,
    }


# ── Phase 2: Additional profile metrics ───────────────────────────────
def phase_2(device, n_boot):
    print("\n" + "=" * 70)
    print("Phase 2: Additional profile metrics (JSD, MNLL, AUPRC)")
    print("=" * 70)

    results = {}
    for name, cfg in DATASET_CONFIGS.items():
        ckpt = BASE_DIR / MODEL_CHECKPOINTS[name]
        test_path = BASE_DIR / cfg["data_dir"] / "test.h5"
        if not ckpt.exists() or not test_path.exists():
            continue

        print(f"\n  {name}:")
        model = load_model(ckpt, cfg["n_tfs"], device)
        seqs, profs, tf_idx, bs = load_test_data(test_path)
        out = run_model_batch(model, seqs, device)
        pred = out['profile']

        jsds = per_sample_jsd(pred, profs)
        mnlls = per_sample_mnll(pred, profs)
        rs = per_sample_pearson(pred, profs)
        auprc = site_auprc(pred, bs)

        jsd_ci = bootstrap_ci(jsds, n_boot)
        mnll_ci = bootstrap_ci(mnlls, n_boot)
        r_ci = bootstrap_ci(rs, n_boot)
        print(f"    r:    {r_ci['mean']:.4f} [{r_ci['ci_lower']:.4f}, {r_ci['ci_upper']:.4f}]")
        print(f"    JSD:  {jsd_ci['mean']:.4f} [{jsd_ci['ci_lower']:.4f}, {jsd_ci['ci_upper']:.4f}]")
        print(f"    MNLL: {mnll_ci['mean']:.4f} [{mnll_ci['ci_lower']:.4f}, {mnll_ci['ci_upper']:.4f}]")
        for k, v in auprc.items():
            print(f"    AUPRC@{k}: {v['auprc']:.4f}")

        # Per-TF
        per_tf = {}
        for ti, tn in enumerate(cfg["tf_names"]):
            mask = tf_idx == ti
            if mask.sum() < 5:
                continue
            per_tf[tn] = {
                "pearson": bootstrap_ci(rs[mask], n_boot),
                "jsd": bootstrap_ci(jsds[mask], n_boot),
                "mnll": bootstrap_ci(mnlls[mask], n_boot),
                "n": int(mask.sum()),
            }

        results[name] = {
            "pearson": r_ci, "jsd": jsd_ci, "mnll": mnll_ci,
            "auprc": auprc, "per_tf": per_tf, "n": len(seqs),
        }
        del model; torch.cuda.empty_cache()

    # BPNet comparison on K562-7tf
    bpnet_metrics = _bpnet_metrics(device, n_boot)
    if bpnet_metrics:
        results["BPNet_K562-7tf"] = bpnet_metrics

    return results


def _bpnet_metrics(device, n_boot):
    """Compute JSD/MNLL for BPNet on K562-7tf if available."""
    try:
        from bpnetlite import BPNet
    except ImportError:
        return None
    bp_path = BASE_DIR / "bpnet.64.8.final.torch"
    test_path = BASE_DIR / "data/processed/multi_tf_k562/test.h5"
    if not bp_path.exists() or not test_path.exists():
        return None
    try:
        seqs, profs, *_ = load_test_data(test_path)
        loaded = torch.load(bp_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, BPNet):
            bpnet = loaded
        else:
            bpnet = BPNet(n_filters=64, n_layers=8, n_outputs=1,
                           n_control_tracks=0, trimming=500, verbose=False)
            bpnet.load_state_dict(loaded)
        bpnet = bpnet.to(device).eval()
        X = seqs.transpose(0, 2, 1)
        trim = 500
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                bx = torch.from_numpy(X[i:i+32]).float().to(device)
                pred = bpnet(bx)
                preds.append((pred[0] if isinstance(pred, tuple) else pred).cpu().numpy())
        preds = np.concatenate(preds, axis=0).reshape(len(seqs), -1)
        gt_trim = profs[:, trim:seqs.shape[1]-trim]
        jsds = per_sample_jsd(preds, gt_trim)
        mnlls = per_sample_mnll(preds, gt_trim)
        rs = per_sample_pearson(preds, gt_trim)
        del bpnet; torch.cuda.empty_cache()
        return {
            "pearson": bootstrap_ci(rs, n_boot),
            "jsd": bootstrap_ci(jsds, n_boot),
            "mnll": bootstrap_ci(mnlls, n_boot),
            "n": len(seqs),
            "note": "BPNet outputs 1000bp (trimmed); metrics on trimmed region only",
        }
    except Exception as e:
        print(f"    BPNet metrics failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Statistical evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--phases", default="1a,1b,1d,2",
                        help="Comma-separated: 1a, 1b, 1d, 2")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = {p.strip() for p in args.phases.split(",")}
    nb = args.n_bootstrap
    results = {}

    if "1a" in phases:
        results["bootstrap_cis"] = phase_1a(device, nb)
    if "1b" in phases:
        results["beacon_vs_bpnet"] = phase_1b(device, nb)
    if "1d" in phases:
        results["ablation_significance"] = phase_1d(device, nb)
    if "2" in phases:
        results["profile_metrics"] = phase_2(device, nb)

    path = out_dir / "statistical_evaluation.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
