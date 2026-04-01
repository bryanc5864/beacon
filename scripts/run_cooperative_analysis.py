#!/usr/bin/env python3
"""
Cooperative binding analysis: cross-sample attention patterns, slot usage,
position distributions, and profile shape similarity.

Since each test sample contains a single binding site, cooperative binding
is analysed across samples (e.g., do MYC and MAX samples share similar
attention patterns, since both bind the E-box?).

Usage:
    python scripts/run_cooperative_analysis.py --device cuda:2
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_model, load_test_data, run_model_batch, json_serialize,
    attention_pair_analysis, slot_utilization_analysis,
    position_distribution_analysis, profile_shape_similarity,
    DATASET_CONFIGS, MODEL_CHECKPOINTS, BASE_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="Cooperative binding analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_CONFIGS["K562-7tf"]
    ckpt = BASE_DIR / MODEL_CHECKPOINTS["K562-7tf"]
    test_path = BASE_DIR / cfg["data_dir"] / "test.h5"

    print("=" * 70)
    print("Cooperative binding analysis (K562-7tf)")
    print("=" * 70)

    model = load_model(ckpt, cfg["n_tfs"], device)
    seqs, profs, tf_idx, bs = load_test_data(test_path)
    outputs = run_model_batch(model, seqs, device, return_attention=True)

    tf_names = cfg["tf_names"]
    results = {}

    print("\n--- Cross-sample attention similarity ---")
    results["attention_patterns"] = attention_pair_analysis(
        outputs, tf_names, tf_idx=tf_idx)
    for pair, data in results["attention_patterns"].items():
        cos = data.get("mean_attention_cosine")
        diff = data.get("partner_vs_control_diff")
        cos_str = f"{cos:.3f}" if cos is not None else "N/A"
        diff_str = f"{diff:+.3f}" if diff is not None else "N/A"
        print(f"  {pair}: attention cosine={cos_str}, "
              f"vs control={diff_str}")

    print("\n--- Slot utilization similarity ---")
    results["slot_utilization"] = slot_utilization_analysis(
        outputs, tf_names, tf_idx=tf_idx)
    for pair, data in results["slot_utilization"].items():
        if pair == "slot_similarity_matrix":
            continue
        jsd = data.get("jsd")
        overlap = data.get("top3_slot_overlap")
        print(f"  {pair}: slot JSD={jsd:.4f}, "
              f"top-3 overlap={overlap}")

    print("\n--- Position distribution analysis ---")
    results["position_distributions"] = position_distribution_analysis(
        outputs, tf_names, tf_idx=tf_idx)
    for pair, data in results["position_distributions"].items():
        print(f"  {pair}: mean pos A={data['mean_pos_a']:.0f}bp, "
              f"B={data['mean_pos_b']:.0f}bp, "
              f"KS p={data['ks_pvalue']:.2e}")

    print("\n--- Profile shape similarity ---")
    results["profile_shapes"] = profile_shape_similarity(
        outputs, profs, tf_names, tf_idx)
    for pair, data in results["profile_shapes"].items():
        r = data.get("mean_profile_pearson", 0)
        ctrl = data.get("control_pearson", {}).get("mean", 0)
        print(f"  {pair}: profile r={r:.3f}, "
              f"control mean={ctrl:.3f}")

    path = out_dir / "cooperative_binding.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
