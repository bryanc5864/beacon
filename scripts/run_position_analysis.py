#!/usr/bin/env python3
"""
Position prediction accuracy analysis across all BEACON datasets.

Measures how accurately BEACON predicts binding site positions — a
unique capability vs BPNet which only predicts profile shape.

Usage:
    python scripts/run_position_analysis.py --device cuda:2
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_named_model, load_test_data, get_test_path,
    run_model_batch, json_serialize,
    DATASET_CONFIGS, BASE_DIR,
)
from beacon.analysis.position_accuracy import (
    hungarian_position_errors, per_tf_position_stats,
    position_error_by_signal, position_error_by_gc,
    systematic_offset_analysis,
)


def analyze_dataset(name, device):
    """Run full position analysis on one dataset."""
    print(f"\n--- {name} ---")
    model, cfg = load_named_model(name, device)
    test_path = get_test_path(name)
    seqs, profs, tf_idx, bs = load_test_data(test_path)
    tf_names = cfg["tf_names"]

    outputs = run_model_batch(model, seqs, device)

    print("  Computing position errors...")
    errors = hungarian_position_errors(outputs, bs)
    n_matched = len(errors['abs_errors'])
    print(f"  Matched sites: {n_matched}")

    if n_matched == 0:
        del model
        torch.cuda.empty_cache()
        return {'n_matched': 0}

    abs_arr = errors['abs_errors']
    import numpy as np
    print(f"  MAE: {np.mean(abs_arr):.4f} bp")
    print(f"  MedAE: {np.median(abs_arr):.4f} bp")

    print("  Per-TF statistics...")
    per_tf = per_tf_position_stats(errors, tf_names)
    for tf, stats in per_tf.items():
        if stats.get('mae') is not None:
            print(f"    {tf}: MAE={stats['mae']:.4f}, MedAE={stats['medae']:.4f} (n={stats['n']})")

    print("  Signal stratification...")
    by_signal = position_error_by_signal(errors, profs)
    for q, stats in by_signal.items():
        print(f"    {q}: MAE={stats['mae']:.4f} (n={stats['n']})")

    print("  GC stratification...")
    by_gc = position_error_by_gc(errors, seqs)
    for q, stats in by_gc.items():
        print(f"    {q}: MAE={stats['mae']:.4f} (n={stats['n']})")

    print("  Systematic offset analysis...")
    offsets = systematic_offset_analysis(errors, tf_names)
    for tf, stats in offsets.items():
        if stats.get('mean_signed_error') is not None:
            biased = " *BIASED*" if stats.get('is_biased') else ""
            print(f"    {tf}: mean offset={stats['mean_signed_error']:+.4f} bp{biased}")

    result = {
        'n_matched': n_matched,
        'overall_mae': float(np.mean(abs_arr)),
        'overall_medae': float(np.median(abs_arr)),
        'overall_mae_ci': {
            'mean': float(np.mean(abs_arr)),
            'std': float(np.std(abs_arr)),
        },
        'per_tf': per_tf,
        'by_signal_quartile': by_signal,
        'by_gc_quartile': by_gc,
        'systematic_offsets': offsets,
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Position prediction accuracy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Position Prediction Accuracy Analysis")
    print("=" * 70)

    results = {}
    for name in ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]:
        try:
            results[name] = analyze_dataset(name, device)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")

    path = out_dir / "position_accuracy.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
