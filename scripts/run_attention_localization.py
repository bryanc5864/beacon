#!/usr/bin/env python3
"""
Attention localization and slot utilization analysis.

Tests the core interpretability claim: do attention peaks align with
actual binding sites? Also analyzes effective slot utilization.

Usage:
    python scripts/run_attention_localization.py --device cuda:1
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
from beacon.analysis.attention_localization import (
    attention_peak_at_binding_site, per_tf_attention_localization,
    attention_entropy_analysis, effective_slot_utilization,
)


def analyze_dataset(name, device):
    """Run attention localization on one dataset."""
    print(f"\n--- {name} ---")
    model, cfg = load_named_model(name, device)
    test_path = get_test_path(name)
    seqs, profs, tf_idx, bs = load_test_data(test_path)
    tf_names = cfg["tf_names"]

    outputs = run_model_batch(model, seqs, device, return_attention=True)

    # Attention localization
    print("  Attention peak localization...")
    localization = attention_peak_at_binding_site(outputs, bs)
    print(f"    Mean distance to site: {localization['mean_distance_bp']:.1f} bp")
    print(f"    Median distance: {localization['median_distance_bp']:.1f} bp")
    print(f"    Within 50bp: {localization['within_50bp']:.1%}")
    print(f"    Within 100bp: {localization['within_100bp']:.1%}")
    print(f"    Attention enrichment at sites: {localization['attention_enrichment']:.2f}x")

    # Per-TF localization
    print("  Per-TF localization...")
    per_tf_loc = per_tf_attention_localization(outputs, bs, tf_names)
    for tf, data in per_tf_loc.items():
        if 'mean_distance_bp' in data:
            print(f"    {tf}: mean={data['mean_distance_bp']:.1f}bp, "
                  f"within 50bp={data['within_50bp']:.1%} (n={data['n']})")

    # Attention entropy
    print("  Attention entropy analysis...")
    entropy = attention_entropy_analysis(outputs, bs, tf_names)
    print(f"    Mean entropy: {entropy['mean_entropy']:.3f} / {entropy['max_possible_entropy']:.3f}")
    print(f"    Normalized entropy: {entropy['normalized_entropy']:.3f}")

    # Slot utilization
    print("  Effective slot utilization...")
    utilization = effective_slot_utilization(outputs, tf_names, bs)
    print(f"    Mean active slots: {utilization['mean_active_slots']:.2f}")
    print(f"    Effective n slots: {utilization['effective_n_slots']:.2f}")
    print(f"    Slots ever active: {utilization['n_slots_ever_active']}/{outputs['occupancy'].shape[1]}")
    print(f"    Mean slot specificity: {utilization['mean_slot_specificity']:.3f}")

    result = {
        'localization': localization,
        'per_tf_localization': per_tf_loc,
        'entropy': entropy,
        'slot_utilization': utilization,
    }

    del model; torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Attention localization analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Attention Localization & Slot Utilization Analysis")
    print("=" * 70)

    results = {}
    for name in ["K562-7tf", "K562-fulltf", "HepG2-7tf", "HepG2-fulltf"]:
        try:
            results[name] = analyze_dataset(name, device)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")

    path = out_dir / "attention_localization.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
