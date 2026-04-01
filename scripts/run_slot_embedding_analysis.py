#!/usr/bin/env python3
"""
Slot embedding analysis for BEACON.

Analyzes learned slot representations: cross-seed consistency (RSA),
linear probing for TF identity, and TF family clustering metrics.

Usage:
    python scripts/run_slot_embedding_analysis.py --device cuda:5
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
    load_named_model, load_test_data, get_test_path,
    run_model_batch, json_serialize,
    DATASET_CONFIGS, TF_FAMILIES, BASE_DIR,
)
from beacon.analysis.slot_embeddings import (
    extract_slot_embeddings, cross_seed_embedding_consistency,
    linear_probe_cv, tf_family_clustering_metrics,
)

# Multi-seed checkpoints
SEED_CHECKPOINTS = {
    "seed_42": "outputs/multi_seed/seed_42/beacon_seed42_20260329_181458/best_model.pt",
    "seed_123": "outputs/multi_seed/seed_123/beacon_seed123_20260329_181123/best_model.pt",
    "seed_456": "outputs/multi_seed/seed_456/beacon_seed456_20260329_181123/best_model.pt",
}


def get_active_slot_data(emb_dict, tf_idx, tf_names, occ_threshold=0.3):
    """Extract embeddings and labels for active slots matched to ground truth TFs.

    Returns:
        (embeddings [M, D], tf_labels [M] as ints, tf_name_labels [M] as strings)
    """
    occupancy = emb_dict['occupancy']  # [N, K]
    tf_logits = emb_dict['tf_logits']  # [N, K, T]
    slot_emb = emb_dict['slot_embeddings']  # [N, K, D]

    embeddings = []
    tf_int_labels = []
    tf_name_labels = []

    for i in range(len(occupancy)):
        for k in range(occupancy.shape[1]):
            if occupancy[i, k] > occ_threshold:
                pred_tf = int(tf_logits[i, k].argmax())
                if pred_tf < len(tf_names):
                    embeddings.append(slot_emb[i, k])
                    tf_int_labels.append(pred_tf)
                    tf_name_labels.append(tf_names[pred_tf])

    return (np.array(embeddings), np.array(tf_int_labels),
            np.array(tf_name_labels))


def main():
    parser = argparse.ArgumentParser(description="Slot embedding analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Slot Embedding Analysis")
    print("=" * 70)

    results = {}

    # ── Cross-seed consistency (K562-7tf) ───────────────────────────────
    print("\n--- Cross-seed RSA (K562-7tf) ---")
    cfg = DATASET_CONFIGS["K562-7tf"]
    test_path = get_test_path("K562-7tf")
    seqs, profs, tf_idx, bs = load_test_data(test_path, max_samples=args.max_samples)

    seed_ckpts = {}
    for name, rel_path in SEED_CHECKPOINTS.items():
        full_path = BASE_DIR / rel_path
        if full_path.exists():
            seed_ckpts[name] = full_path
            print(f"  Found: {name}")
        else:
            print(f"  Missing: {name} ({full_path})")

    if len(seed_ckpts) >= 2:
        print(f"  Computing RSA across {len(seed_ckpts)} seeds...")
        rsa_results = cross_seed_embedding_consistency(
            seed_ckpts, seqs, cfg["n_tfs"], device, max_samples=args.max_samples
        )
        results["cross_seed_rsa"] = rsa_results
        print(f"  Mean RSA: {rsa_results['mean_rsa']:.3f} ± {rsa_results['std_rsa']:.3f}")
        for pair, data in rsa_results["pairwise"].items():
            print(f"    {pair}: {data['rsa_correlation']:.3f}")
    else:
        print("  Not enough seed checkpoints for RSA analysis")
        results["cross_seed_rsa"] = {"error": "insufficient checkpoints"}

    # ── Linear probe and clustering (K562-7tf) ──────────────────────────
    for name in ["K562-7tf", "K562-fulltf"]:
        print(f"\n--- Linear probe & clustering ({name}) ---")
        try:
            model, cfg = load_named_model(name, device)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        test_path = get_test_path(name)
        seqs, profs, tf_idx, bs = load_test_data(test_path, max_samples=args.max_samples)
        tf_names = cfg["tf_names"]

        emb_dict = extract_slot_embeddings(model, seqs, device)

        embeddings, tf_int_labels, tf_name_labels = get_active_slot_data(
            emb_dict, tf_idx, tf_names
        )
        print(f"  Active slots: {len(embeddings)}")

        if len(embeddings) < 20:
            print("  Too few active slots for analysis")
            del model; torch.cuda.empty_cache()
            continue

        # Linear probe
        print("  Linear probe (5-fold CV)...")
        probe = linear_probe_cv(embeddings, tf_int_labels)
        print(f"    Accuracy: {probe['mean_accuracy']:.3f} ± {probe['std_accuracy']:.3f}")

        # TF family clustering
        print("  TF family clustering...")
        clustering = tf_family_clustering_metrics(embeddings, tf_name_labels, TF_FAMILIES)
        if clustering.get('silhouette_score') is not None:
            print(f"    Silhouette: {clustering['silhouette_score']:.3f}")
            print(f"    ARI: {clustering['adjusted_rand_index']:.3f}")

        results[name] = {
            'n_active_slots': len(embeddings),
            'linear_probe': probe,
            'tf_family_clustering': clustering,
        }

        del model; torch.cuda.empty_cache()

    path = out_dir / "slot_embedding_analysis.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
