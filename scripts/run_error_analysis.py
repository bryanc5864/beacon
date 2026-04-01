#!/usr/bin/env python3
"""
Error analysis: performance stratification, TF confusion, failure cases, calibration.

Usage:
    python scripts/run_error_analysis.py --device cuda:2
    python scripts/run_error_analysis.py --device cuda:2 --models K562-7tf
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_model, load_test_data, run_model_batch,
    per_sample_pearson, hungarian_confusion_matrix, gc_content, json_serialize,
    DATASET_CONFIGS, MODEL_CHECKPOINTS, TF_FAMILIES, BASE_DIR,
)


def stratify_by_signal(pred, gt, tf_idx, tf_names):
    """Performance by ground-truth signal-strength quartiles."""
    rs = per_sample_pearson(pred, gt)
    strength = np.array([np.abs(gt[i]).sum() for i in range(len(gt))])
    edges = np.percentile(strength, [0, 25, 50, 75, 100])
    labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]

    overall = {}
    for q in range(4):
        mask = (strength >= edges[q]) & (strength < edges[q+1]) if q < 3 else strength >= edges[q]
        if mask.sum() == 0:
            continue
        overall[labels[q]] = {
            "n": int(mask.sum()),
            "mean_r": float(np.mean(rs[mask])),
            "std_r": float(np.std(rs[mask])),
            "signal_range": [float(strength[mask].min()), float(strength[mask].max())],
        }
    return {"overall": overall}


def tf_family_analysis(confusion, tf_names):
    """Within-family vs between-family confusion."""
    n_tfs = len(tf_names)
    tf_to_fam = {}
    for fam, members in TF_FAMILIES.items():
        for m in members:
            tf_to_fam[m] = fam

    within, between = 0, 0
    for i in range(n_tfs):
        for j in range(n_tfs):
            if i == j:
                continue
            fi = tf_to_fam.get(tf_names[i], "?")
            fj = tf_to_fam.get(tf_names[j], "?")
            if fi == fj:
                within += confusion[i, j]
            else:
                between += confusion[i, j]
    total_err = confusion.sum() - np.trace(confusion)
    return {
        "within_family_errors": int(within),
        "between_family_errors": int(between),
        "within_pct": float(within / total_err * 100) if total_err else 0,
        "per_tf_accuracy": {
            tf_names[i]: float(confusion[i, i] / confusion[i].sum())
            if confusion[i].sum() > 0 else 0
            for i in range(n_tfs)
        },
    }


def failure_analysis(seqs, pred, gt, tf_idx, tf_names):
    """Characterize bottom-10% worst samples."""
    rs = per_sample_pearson(pred, gt)
    n = len(rs)
    bottom_idx = np.argsort(rs)[:max(1, n // 10)]

    fail_tf = defaultdict(int)
    all_tf = defaultdict(int)
    for i in range(n):
        t = int(tf_idx[i])
        if t < len(tf_names):
            all_tf[tf_names[t]] += 1
    for i in bottom_idx:
        t = int(tf_idx[i])
        if t < len(tf_names):
            fail_tf[tf_names[t]] += 1

    enrichment = {}
    for tn in tf_names:
        fp = fail_tf.get(tn, 0) / len(bottom_idx) * 100
        ap = all_tf.get(tn, 0) / n * 100
        enrichment[tn] = {
            "failure_pct": float(fp), "overall_pct": float(ap),
            "enrichment": float(fp / ap) if ap > 0 else 0,
        }

    fail_sig = np.array([np.abs(gt[i]).sum() for i in bottom_idx])
    all_sig = np.array([np.abs(gt[i]).sum() for i in range(n)])
    fail_gc = np.array([gc_content(seqs[i]) for i in bottom_idx])
    all_gc = np.array([gc_content(seqs[i]) for i in range(n)])

    return {
        "n_failures": len(bottom_idx),
        "threshold_r": float(rs[bottom_idx[-1]]),
        "mean_r": float(np.mean(rs[bottom_idx])),
        "tf_enrichment": enrichment,
        "signal": {"fail": float(np.mean(fail_sig)), "all": float(np.mean(all_sig))},
        "gc": {"fail": float(np.mean(fail_gc)), "all": float(np.mean(all_gc))},
    }


def calibration_analysis(outputs, binding_sites):
    """Reliability diagram for occupancy scores."""
    occ = outputs['occupancy']
    n_samples, n_slots = occ.shape
    n_bins = 10
    bins = {i: {"pred": [], "actual": []} for i in range(n_bins)}

    for b in range(n_samples):
        n_gt = sum(1 for s in range(binding_sites.shape[1]) if binding_sites[b, s, 2] > 0.5)
        top_k = np.argsort(-occ[b])[:max(n_gt, 1)]
        for k in range(n_slots):
            bi = min(int(occ[b, k] * n_bins), n_bins - 1)
            bins[bi]["pred"].append(float(occ[b, k]))
            bins[bi]["actual"].append(1.0 if k in top_k else 0.0)

    curve = []
    ece = 0
    total = sum(len(bins[i]["pred"]) for i in range(n_bins))
    for i in range(n_bins):
        if not bins[i]["pred"]:
            continue
        mp = np.mean(bins[i]["pred"])
        fa = np.mean(bins[i]["actual"])
        curve.append({"bin": f"[{i/n_bins:.1f}, {(i+1)/n_bins:.1f})",
                       "n": len(bins[i]["pred"]),
                       "mean_pred": float(mp), "frac_active": float(fa)})
        ece += len(bins[i]["pred"]) / total * abs(mp - fa)
    return {"curve": curve, "ece": float(ece)}


def main():
    parser = argparse.ArgumentParser(description="Error analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="K562-7tf,K562-fulltf")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",")]
    results = {}

    for name in model_names:
        if name not in DATASET_CONFIGS:
            continue
        cfg = DATASET_CONFIGS[name]
        ckpt = BASE_DIR / MODEL_CHECKPOINTS[name]
        test_path = BASE_DIR / cfg["data_dir"] / "test.h5"
        if not ckpt.exists() or not test_path.exists():
            print(f"SKIP {name}")
            continue

        print(f"\n{'='*70}\nError analysis: {name}\n{'='*70}")
        model = load_model(ckpt, cfg["n_tfs"], device)
        seqs, profs, tf_idx, bs = load_test_data(test_path)
        out = run_model_batch(model, seqs, device)
        pred = out['profile']

        r = {}
        r["signal_stratification"] = stratify_by_signal(pred, profs, tf_idx, cfg["tf_names"])

        confusion = hungarian_confusion_matrix(out, bs, cfg["n_tfs"])
        r["confusion_matrix"] = confusion.tolist()
        r["tf_family"] = tf_family_analysis(confusion, cfg["tf_names"])

        r["failure_analysis"] = failure_analysis(seqs, pred, profs, tf_idx, cfg["tf_names"])
        r["calibration"] = calibration_analysis(out, bs)

        results[name] = r
        del model; torch.cuda.empty_cache()

    path = out_dir / "error_analysis.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
