#!/usr/bin/env python3
"""
Extended cross-cell-type generalization analysis.

Deepens K562->HepG2 transfer results by analyzing which features
transfer: attention patterns, gradient importance, and embeddings.

Usage:
    python scripts/run_cross_cell_analysis.py --device cuda:5
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
    DATASET_CONFIGS, BASE_DIR,
)
from beacon.analysis.saliency import compute_gradient_x_input
from beacon.analysis.denovo_motifs import compute_per_tf_gradient_importance
from beacon.analysis.cross_cell import (
    motif_level_transfer, attention_pattern_transfer,
    cell_specific_vs_shared_features,
)

# Shared TFs between K562-7tf and HepG2-7tf
SHARED_TFS = ["CTCF", "MYC", "MAX", "CEBPB"]
K562_TF_MAP = {name: i for i, name in enumerate(DATASET_CONFIGS["K562-7tf"]["tf_names"])}
HEPG2_TF_MAP = {name: i for i, name in enumerate(DATASET_CONFIGS["HepG2-7tf"]["tf_names"])}


def get_active_embeddings(outputs, tf_names, occ_threshold=0.3):
    """Extract embeddings and labels for active slots."""
    occupancy = outputs['occupancy']
    tf_logits = outputs['tf_logits']
    slot_emb = outputs['slot_embeddings']

    embeddings = []
    tf_labels = []

    for i in range(len(occupancy)):
        for k in range(occupancy.shape[1]):
            if occupancy[i, k] > occ_threshold:
                pred_tf = int(tf_logits[i, k].argmax())
                if pred_tf < len(tf_names):
                    embeddings.append(slot_emb[i, k])
                    tf_labels.append(tf_names[pred_tf])

    return np.array(embeddings), np.array(tf_labels)


def main():
    parser = argparse.ArgumentParser(description="Cross-cell generalization analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extended Cross-Cell Generalization Analysis")
    print("=" * 70)

    # Load K562-7tf model (used for both cell types)
    model, k562_cfg = load_named_model("K562-7tf", device)
    k562_tf_names = k562_cfg["tf_names"]

    # Load test data
    k562_test = get_test_path("K562-7tf")
    hepg2_test = get_test_path("HepG2-7tf")

    # Load ALL test data first, then subsample per TF to ensure coverage
    k562_seqs_all, k562_profs_all, k562_tf_idx_all, k562_bs_all = load_test_data(k562_test)
    hepg2_seqs_all, hepg2_profs_all, hepg2_tf_idx_all, hepg2_bs_all = load_test_data(hepg2_test)

    # Subsample: take up to max_samples_per_tf from each shared TF
    max_per_tf = args.max_samples // len(SHARED_TFS)
    rng = np.random.RandomState(42)

    def subsample_shared(seqs, profs, tf_idx, bs, tf_map):
        indices = []
        for tf in SHARED_TFS:
            t_idx = tf_map.get(tf)
            if t_idx is None:
                continue
            tf_mask = np.where(tf_idx == t_idx)[0]
            if len(tf_mask) > max_per_tf:
                tf_mask = rng.choice(tf_mask, max_per_tf, replace=False)
            indices.extend(tf_mask.tolist())
        indices = sorted(indices)
        return seqs[indices], profs[indices], tf_idx[indices], bs[indices]

    k562_seqs, k562_profs, k562_tf_idx, k562_bs = subsample_shared(
        k562_seqs_all, k562_profs_all, k562_tf_idx_all, k562_bs_all, K562_TF_MAP)
    hepg2_seqs, hepg2_profs, hepg2_tf_idx, hepg2_bs = subsample_shared(
        hepg2_seqs_all, hepg2_profs_all, hepg2_tf_idx_all, hepg2_bs_all, HEPG2_TF_MAP)

    print(f"  K562 samples: {len(k562_seqs)}")
    print(f"  HepG2 samples: {len(hepg2_seqs)}")

    results = {}

    # ── Attention pattern transfer ──────────────────────────────────────
    print("\n--- Attention Pattern Transfer ---")
    k562_outputs = run_model_batch(model, k562_seqs, device, return_attention=True)
    hepg2_outputs = run_model_batch(model, hepg2_seqs, device, return_attention=True)

    # For HepG2, we need to remap TF indices to K562 space
    hepg2_tf_names = DATASET_CONFIGS["HepG2-7tf"]["tf_names"]
    hepg2_to_k562_map = {}
    for tf in SHARED_TFS:
        h_idx = HEPG2_TF_MAP.get(tf)
        k_idx = K562_TF_MAP.get(tf)
        if h_idx is not None and k_idx is not None:
            hepg2_to_k562_map[h_idx] = k_idx

    # Remap HepG2 TF indices to K562 space for shared TFs
    hepg2_tf_idx_remapped = np.full_like(hepg2_tf_idx, -1)
    for h_idx, k_idx in hepg2_to_k562_map.items():
        mask = hepg2_tf_idx == h_idx
        hepg2_tf_idx_remapped[mask] = k_idx

    attn_transfer = attention_pattern_transfer(
        k562_outputs, hepg2_outputs,
        k562_tf_idx, hepg2_tf_idx_remapped,
        SHARED_TFS, K562_TF_MAP, K562_TF_MAP,  # Both in K562 space after remapping
    )
    results['attention_transfer'] = attn_transfer
    for tf, data in attn_transfer.items():
        print(f"  {tf}: K562 entropy={data['k562_entropy']:.3f}, "
              f"HepG2 entropy={data['hepg2_entropy']:.3f}")

    # ── Motif-level transfer (gradient importance) ──────────────────────
    print("\n--- Motif-Level Transfer (Gradient) ---")
    k562_grad = compute_per_tf_gradient_importance(
        model, k562_seqs, k562_tf_idx, device, max_per_tf=30
    )

    # For HepG2, use remapped indices
    shared_mask = hepg2_tf_idx_remapped >= 0
    if shared_mask.any():
        hepg2_grad = compute_per_tf_gradient_importance(
            model, hepg2_seqs[shared_mask], hepg2_tf_idx_remapped[shared_mask],
            device, max_per_tf=30
        )
    else:
        hepg2_grad = {}

    motif_transfer = motif_level_transfer(
        k562_grad, hepg2_grad, SHARED_TFS, K562_TF_MAP, K562_TF_MAP
    )
    results['motif_transfer'] = motif_transfer
    for tf, data in motif_transfer.items():
        print(f"  {tf}: importance correlation = {data['importance_correlation']:.3f}")

    # ── Cell-specific vs shared features ────────────────────────────────
    print("\n--- Cell-Specific vs Shared Features ---")
    if 'slot_embeddings' in k562_outputs and 'slot_embeddings' in hepg2_outputs:
        k562_emb, k562_labels = get_active_embeddings(k562_outputs, k562_tf_names)
        hepg2_emb, hepg2_labels = get_active_embeddings(hepg2_outputs, k562_tf_names)

        if len(k562_emb) > 10 and len(hepg2_emb) > 10:
            feature_analysis = cell_specific_vs_shared_features(
                k562_emb, hepg2_emb, k562_labels, hepg2_labels
            )
            results['feature_analysis'] = feature_analysis
            print(f"  R² cell type: {feature_analysis['r2_cell_type']:.3f}")
            print(f"  R² TF identity: {feature_analysis['r2_tf_identity']:.3f}")
            print(f"  R² combined: {feature_analysis['r2_combined']:.3f}")
        else:
            print("  Too few active slots for feature analysis")
    else:
        print("  No slot embeddings available")

    path = out_dir / "cross_cell_analysis.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
