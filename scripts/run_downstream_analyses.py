#!/usr/bin/env python3
"""
Downstream analyses for BEACON paper.

1. Co-binding decomposition: Multi-TF slot decomposition analysis
2. ISM at K562 motif sites: In-silico mutagenesis of known motifs
3. Scaling analysis: 7→14 TF performance comparison
4. Slot specialization: Slot-TF assignment consistency
5. Per-TF profile breakdown: Detailed per-TF metrics across all models
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from beacon.models import BEACON

base_dir = Path(__file__).parent.parent

TF_NAMES_7 = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
TF_NAMES_14 = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "CEBPB",
                "REST", "YY1", "NRF1", "JUND", "FOS", "ATF3", "ELF1", "GABPA"]

# JASPAR consensus motifs (core sequences for each TF)
JASPAR_CONSENSUS = {
    "CTCF": "CCGCGNGGNGGCAG",
    "GATA1": "AGATAA",
    "TAL1": "CAGCTG",
    "MYC": "CACGTG",
    "MAX": "CACGTG",
    "SPI1": "AAAGAGGAAGTG",
    "CEBPB": "TTGCGCAA",
}

NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}


def load_model(checkpoint_path, n_tfs, device):
    """Load BEACON model from checkpoint."""
    model = BEACON(
        seq_len=2000, input_channels=4, backbone_type="dilated",
        backbone_dim=128, backbone_layers=4, n_slots=16, slot_dim=128,
        n_iterations=3, n_tfs=n_tfs, position_mode="gaussian",
        dropout=0.0, attention_mode="independent",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_test_data(data_path, max_samples=None):
    """Load test sequences, profiles, tf_indices, binding_sites."""
    with h5py.File(data_path, 'r') as f:
        n = f['sequences'].shape[0]
        if max_samples and max_samples < n:
            n = max_samples
        sequences = f['sequences'][:n]
        profiles = f['profiles'][:n]
        tf_indices = f['tf_indices'][:n]
        binding_sites = f['binding_sites'][:n]
    return sequences, profiles, tf_indices, binding_sites


def run_model_batch(model, sequences, device, batch_size=32):
    """Run BEACON on batch, return all outputs."""
    all_outputs = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[i:i+batch_size], dtype=torch.float32
            ).to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(batch)
            for k in ['profile', 'occupancy', 'tf_logits', 'positions']:
                if k in outputs:
                    all_outputs[k].append(outputs[k].cpu().numpy())

    result = {}
    for k, v in all_outputs.items():
        arr = np.concatenate(v, axis=0)
        # Normalize shapes: profile [N,1,L]->[N,L], occupancy [N,K,1]->[N,K]
        if k == 'profile' and arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif k == 'occupancy' and arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], arr.shape[1])
        result[k] = arr
    return result


# =========================================================================
# Experiment 1: Co-binding Decomposition
# =========================================================================
def experiment_cobinding(model, sequences, tf_indices, device, tf_names):
    """Analyze multi-TF slot decomposition."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Co-binding Decomposition Analysis")
    print("=" * 70)

    outputs = run_model_batch(model, sequences, device)
    occupancy = outputs['occupancy']  # [N, K]
    tf_logits = outputs['tf_logits']  # [N, K, n_tfs]
    tf_preds = tf_logits.argmax(axis=-1)  # [N, K]

    occ_threshold = 0.3
    n_samples = len(sequences)

    # For each sample, count how many distinct TFs are predicted in active slots
    multi_tf_samples = []
    tf_pair_counts = defaultdict(int)

    for i in range(n_samples):
        active_mask = occupancy[i] > occ_threshold
        active_tf_preds = tf_preds[i][active_mask]
        unique_tfs = set(active_tf_preds.tolist())

        if len(unique_tfs) >= 2:
            multi_tf_samples.append({
                'idx': i,
                'gt_tf': int(tf_indices[i]),
                'predicted_tfs': sorted(unique_tfs),
                'n_active_slots': int(active_mask.sum()),
                'n_unique_tfs': len(unique_tfs),
            })
            # Count TF pairs
            tfs_sorted = sorted(unique_tfs)
            for a in range(len(tfs_sorted)):
                for b in range(a + 1, len(tfs_sorted)):
                    pair = (tfs_sorted[a], tfs_sorted[b])
                    tf_pair_counts[pair] += 1

    pct_multi = len(multi_tf_samples) / n_samples * 100
    print(f"\n  Samples with multi-TF predictions: {len(multi_tf_samples)}/{n_samples} ({pct_multi:.1f}%)")

    # Distribution of number of unique TFs per sample
    n_unique_dist = defaultdict(int)
    for s in multi_tf_samples:
        n_unique_dist[s['n_unique_tfs']] += 1
    print(f"  Distribution of unique TFs per sample:")
    for k in sorted(n_unique_dist):
        print(f"    {k} TFs: {n_unique_dist[k]} samples")

    # Most common co-occurring TF pairs
    print(f"\n  Top co-occurring TF pairs:")
    sorted_pairs = sorted(tf_pair_counts.items(), key=lambda x: -x[1])[:10]
    for (a, b), count in sorted_pairs:
        name_a = tf_names[a] if a < len(tf_names) else f"TF{a}"
        name_b = tf_names[b] if b < len(tf_names) else f"TF{b}"
        print(f"    {name_a} + {name_b}: {count} samples ({count/n_samples*100:.1f}%)")

    # Slot occupancy statistics
    mean_active = (occupancy > occ_threshold).sum(axis=1).mean()
    print(f"\n  Mean active slots per sample: {mean_active:.1f}")

    # For GT TF, check if model assigns primary slot correctly
    correct_primary = 0
    for i in range(n_samples):
        gt_tf = int(tf_indices[i])
        active_mask = occupancy[i] > occ_threshold
        if active_mask.any():
            # Highest-occupancy slot
            best_slot = occupancy[i].argmax()
            if tf_preds[i][best_slot] == gt_tf:
                correct_primary += 1
    primary_acc = correct_primary / n_samples * 100
    print(f"  Primary slot predicts GT TF: {primary_acc:.1f}%")

    results = {
        'n_samples': n_samples,
        'pct_multi_tf': pct_multi,
        'mean_active_slots': float(mean_active),
        'primary_slot_accuracy': primary_acc,
        'n_unique_dist': {str(k): v for k, v in n_unique_dist.items()},
        'top_pairs': [
            {'tf_a': tf_names[a], 'tf_b': tf_names[b], 'count': c}
            for (a, b), c in sorted_pairs
        ],
    }
    return results


# =========================================================================
# Experiment 2: ISM at Known Motif Sites
# =========================================================================
def experiment_ism(model, sequences, profiles, tf_indices, device, tf_names):
    """In-silico mutagenesis at known JASPAR motif sites."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: In-Silico Mutagenesis at Known Motif Sites")
    print("=" * 70)

    results = {}

    for tf_idx, tf_name in enumerate(tf_names):
        if tf_name not in JASPAR_CONSENSUS:
            continue

        consensus = JASPAR_CONSENSUS[tf_name]
        # Get samples for this TF
        mask = tf_indices == tf_idx
        tf_seqs = sequences[mask]
        tf_profiles = profiles[mask]

        if len(tf_seqs) == 0:
            continue

        # Take up to 200 samples
        n = min(200, len(tf_seqs))
        tf_seqs = tf_seqs[:n]
        tf_profiles = tf_profiles[:n]

        # Run ref predictions
        ref_outputs = run_model_batch(model, tf_seqs, device)
        ref_occ = ref_outputs['occupancy']  # [N, K]
        ref_profile = ref_outputs['profile']  # [N, L]
        ref_tf_preds = ref_outputs['tf_logits'].argmax(axis=-1)  # [N, K]

        # Mutate the center of each sequence (where the motif should be)
        center = 1000  # sequences are centered on binding site
        motif_len = len(consensus)
        start = center - motif_len // 2
        end = start + motif_len

        # Create mutant: shuffle the motif region
        mut_seqs = tf_seqs.copy()
        for i in range(n):
            # Replace motif region with random sequence
            np.random.seed(42 + i)
            for pos in range(start, end):
                mut_seqs[i, pos] = 0.0
                random_nuc = np.random.randint(0, 4)
                mut_seqs[i, pos, random_nuc] = 1.0

        # Run mutant predictions
        mut_outputs = run_model_batch(model, mut_seqs, device)
        mut_occ = mut_outputs['occupancy']
        mut_profile = mut_outputs['profile']
        mut_tf_preds = mut_outputs['tf_logits'].argmax(axis=-1)

        # Measure disruption
        # 1. Profile change at motif site
        window = 50  # +/- 50bp around motif
        ps, pe = max(0, center - window), min(2000, center + window)
        ref_local = ref_profile[:, ps:pe].sum(axis=1)
        mut_local = mut_profile[:, ps:pe].sum(axis=1)
        profile_drop = (ref_local - mut_local) / (ref_local + 1e-8)

        # 2. Occupancy change for slots predicted as this TF
        ref_tf_occ = []
        mut_tf_occ = []
        for i in range(n):
            tf_slots = ref_tf_preds[i] == tf_idx
            ref_tf_occ.append(ref_occ[i][tf_slots].sum() if tf_slots.any() else 0)
            tf_slots_mut = mut_tf_preds[i] == tf_idx
            mut_tf_occ.append(mut_occ[i][tf_slots_mut].sum() if tf_slots_mut.any() else 0)
        ref_tf_occ = np.array(ref_tf_occ)
        mut_tf_occ = np.array(mut_tf_occ)
        occ_drop = ref_tf_occ - mut_tf_occ

        # 3. Profile Pearson correlation before/after
        profile_corrs = []
        for i in range(n):
            r, _ = pearsonr(ref_profile[i], mut_profile[i])
            profile_corrs.append(r)
        profile_corrs = np.array(profile_corrs)

        mean_profile_drop = float(profile_drop.mean())
        mean_occ_drop = float(occ_drop.mean())
        mean_profile_corr = float(profile_corrs.mean())
        pct_disrupted = float((occ_drop > 0.1).mean() * 100)

        print(f"\n  {tf_name} (motif: {consensus}, n={n}):")
        print(f"    Profile signal drop at motif: {mean_profile_drop*100:.1f}%")
        print(f"    Occupancy drop (TF slots):    {mean_occ_drop:.3f}")
        print(f"    Profile correlation ref vs mut: {mean_profile_corr:.3f}")
        print(f"    Samples with occupancy disrupted (>0.1): {pct_disrupted:.1f}%")

        results[tf_name] = {
            'n_samples': n,
            'consensus': consensus,
            'mean_profile_drop_pct': mean_profile_drop * 100,
            'mean_occ_drop': mean_occ_drop,
            'mean_profile_corr': mean_profile_corr,
            'pct_disrupted': pct_disrupted,
        }

    # Summary
    print(f"\n  ISM Summary:")
    mean_drop = np.mean([r['mean_occ_drop'] for r in results.values()])
    mean_pct = np.mean([r['pct_disrupted'] for r in results.values()])
    print(f"    Mean occupancy drop: {mean_drop:.3f}")
    print(f"    Mean % disrupted: {mean_pct:.1f}%")

    return results


# =========================================================================
# Experiment 3: Scaling Analysis (7→14 TFs)
# =========================================================================
def experiment_scaling(device):
    """Compare per-TF performance between 7-TF and 14-TF models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Scaling Analysis (7 → 14 TFs)")
    print("=" * 70)

    # Load K562-7tf model
    model_7tf_path = base_dir / "outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt"
    model_14tf_path = base_dir / "outputs/k562_20tf/K562-20tf_baseline/beacon_20260205_190727/best_model.pt"

    if not model_7tf_path.exists() or not model_14tf_path.exists():
        print("  ERROR: Missing model checkpoints")
        return {}

    # Load test data
    test_7tf = base_dir / "data/processed/multi_tf_k562/test.h5"
    test_14tf = base_dir / "data/processed/multi_tf_k562_20tf/test.h5"

    model_7tf = load_model(model_7tf_path, n_tfs=7, device=device)
    model_14tf = load_model(model_14tf_path, n_tfs=14, device=device)

    # Evaluate 7TF model on 7TF test
    seqs_7, profs_7, tfs_7, _ = load_test_data(test_7tf)
    outputs_7 = run_model_batch(model_7tf, seqs_7, device)

    # Evaluate 14TF model on 14TF test (only for the first 7 shared TFs)
    seqs_14, profs_14, tfs_14, _ = load_test_data(test_14tf)
    outputs_14 = run_model_batch(model_14tf, seqs_14, device)

    # Compare per-TF Pearson for the 7 shared TFs
    shared_tfs = ["CTCF", "GATA1", "TAL1", "MYC", "MAX"]
    # Note: SPI1 is in 7tf (idx 5) but not in 14tf. CEBPB is idx 6 in 7tf, idx 5 in 14tf
    # 14tf has: CTCF(0), GATA1(1), TAL1(2), MYC(3), MAX(4), CEBPB(5), ...

    results = {}
    print(f"\n  {'TF':<8} {'7-TF r':>10} {'14-TF r':>10} {'Delta':>10} {'Retained':>10}")
    print(f"  {'-'*48}")

    for tf_name in ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "CEBPB"]:
        tf_idx_7 = TF_NAMES_7.index(tf_name)
        tf_idx_14 = TF_NAMES_14.index(tf_name)

        # 7TF model: filter samples for this TF
        mask_7 = tfs_7 == tf_idx_7
        pred_7 = outputs_7['profile'][mask_7]
        gt_7 = profs_7[mask_7]

        # 14TF model: filter samples for this TF
        mask_14 = tfs_14 == tf_idx_14
        pred_14 = outputs_14['profile'][mask_14]
        gt_14 = profs_14[mask_14]

        # Per-sample Pearson
        rs_7 = [pearsonr(gt_7[i], pred_7[i])[0] for i in range(len(gt_7)) if np.std(gt_7[i]) > 0]
        rs_14 = [pearsonr(gt_14[i], pred_14[i])[0] for i in range(len(gt_14)) if np.std(gt_14[i]) > 0]

        mean_7 = float(np.mean(rs_7)) if rs_7 else 0.0
        mean_14 = float(np.mean(rs_14)) if rs_14 else 0.0
        delta = mean_14 - mean_7
        retained = mean_14 / mean_7 * 100 if mean_7 > 0 else 0.0

        print(f"  {tf_name:<8} {mean_7:>10.3f} {mean_14:>10.3f} {delta:>+10.3f} {retained:>9.1f}%")

        results[tf_name] = {
            'r_7tf': mean_7, 'r_14tf': mean_14,
            'delta': delta, 'retained_pct': retained,
            'n_7tf': len(rs_7), 'n_14tf': len(rs_14),
        }

    # Overall means
    mean_7_all = np.mean([r['r_7tf'] for r in results.values()])
    mean_14_all = np.mean([r['r_14tf'] for r in results.values()])
    print(f"  {'Mean':<8} {mean_7_all:>10.3f} {mean_14_all:>10.3f} {mean_14_all-mean_7_all:>+10.3f} {mean_14_all/mean_7_all*100:>9.1f}%")

    results['_summary'] = {
        'mean_7tf': float(mean_7_all), 'mean_14tf': float(mean_14_all),
        'delta': float(mean_14_all - mean_7_all),
        'retained_pct': float(mean_14_all / mean_7_all * 100),
    }

    del model_7tf, model_14tf
    torch.cuda.empty_cache()
    return results


# =========================================================================
# Experiment 4: Slot Specialization
# =========================================================================
def experiment_slot_specialization(model, sequences, tf_indices, device, tf_names):
    """Analyze slot-TF assignment patterns."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Slot Specialization Analysis")
    print("=" * 70)

    outputs = run_model_batch(model, sequences, device)
    occupancy = outputs['occupancy']  # [N, K]
    tf_logits = outputs['tf_logits']  # [N, K, n_tfs]
    tf_preds = tf_logits.argmax(axis=-1)  # [N, K]

    n_samples, n_slots = occupancy.shape
    n_tfs = len(tf_names)
    occ_threshold = 0.3

    # Build slot-TF assignment matrix: slot_tf_matrix[k][t] = count of times slot k predicts TF t
    slot_tf_matrix = np.zeros((n_slots, n_tfs))
    slot_occ_by_tf = np.zeros((n_slots, n_tfs))

    for i in range(n_samples):
        gt_tf = int(tf_indices[i])
        for k in range(n_slots):
            if occupancy[i, k] > occ_threshold:
                pred_tf = int(tf_preds[i, k])
                if pred_tf < n_tfs:
                    slot_tf_matrix[k, pred_tf] += 1
                    slot_occ_by_tf[k, pred_tf] += occupancy[i, k]

    # Slot purity: for each slot, what fraction of assignments go to the dominant TF
    slot_purity = []
    slot_dominant_tf = []
    for k in range(n_slots):
        total = slot_tf_matrix[k].sum()
        if total > 0:
            dominant = slot_tf_matrix[k].max()
            purity = dominant / total
            dom_tf = int(slot_tf_matrix[k].argmax())
        else:
            purity = 0.0
            dom_tf = -1
        slot_purity.append(purity)
        slot_dominant_tf.append(dom_tf)

    # Slot utilization: fraction of samples where each slot is active
    slot_util = (occupancy > occ_threshold).mean(axis=0)

    print(f"\n  {'Slot':>4} {'Dominant TF':>12} {'Purity':>8} {'Utilization':>12} {'Assignments':>12}")
    print(f"  {'-'*52}")
    active_slots = 0
    for k in range(n_slots):
        total = int(slot_tf_matrix[k].sum())
        if total > 10:  # Only show slots with meaningful activity
            active_slots += 1
            dom_name = tf_names[slot_dominant_tf[k]] if slot_dominant_tf[k] >= 0 else "N/A"
            print(f"  {k:>4} {dom_name:>12} {slot_purity[k]:>7.1%} {slot_util[k]:>11.1%} {total:>12}")

    # Overall metrics
    active_slot_purities = [p for p, u in zip(slot_purity, slot_util) if u > 0.05]
    mean_purity = float(np.mean(active_slot_purities)) if active_slot_purities else 0.0

    # TF coverage: how many TFs have at least one dedicated slot
    tf_has_slot = set()
    for k in range(n_slots):
        if slot_purity[k] > 0.5 and slot_util[k] > 0.05:
            tf_has_slot.add(slot_dominant_tf[k])
    tf_coverage = len(tf_has_slot)

    # Slot entropy (how spread out are TF assignments for each slot)
    slot_entropies = []
    for k in range(n_slots):
        total = slot_tf_matrix[k].sum()
        if total > 0:
            probs = slot_tf_matrix[k] / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            slot_entropies.append(entropy)

    mean_entropy = float(np.mean(slot_entropies)) if slot_entropies else 0.0
    max_entropy = np.log2(n_tfs)

    print(f"\n  Active slots (>10 assignments): {active_slots}/{n_slots}")
    print(f"  Mean slot purity: {mean_purity:.1%}")
    print(f"  Mean slot entropy: {mean_entropy:.2f} / {max_entropy:.2f} (max)")
    print(f"  TFs with dedicated slot (purity >50%): {tf_coverage}/{n_tfs}")
    print(f"  TFs covered: {[tf_names[t] for t in sorted(tf_has_slot)]}")

    results = {
        'n_active_slots': active_slots,
        'mean_purity': mean_purity,
        'mean_entropy': mean_entropy,
        'max_entropy': float(max_entropy),
        'tf_coverage': tf_coverage,
        'tfs_with_slots': [tf_names[t] for t in sorted(tf_has_slot)],
        'slot_details': [
            {
                'slot': k,
                'dominant_tf': tf_names[slot_dominant_tf[k]] if slot_dominant_tf[k] >= 0 else None,
                'purity': float(slot_purity[k]),
                'utilization': float(slot_util[k]),
                'assignments': int(slot_tf_matrix[k].sum()),
            }
            for k in range(n_slots) if slot_tf_matrix[k].sum() > 0
        ],
    }
    return results


# =========================================================================
# Experiment 5: Per-TF Profile Breakdown
# =========================================================================
def experiment_per_tf_breakdown(device):
    """Detailed per-TF profile Pearson for all models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Per-TF Profile Breakdown Across All Models")
    print("=" * 70)

    models_config = [
        {
            'name': 'K562-7tf',
            'path': base_dir / "outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt",
            'data': base_dir / "data/processed/multi_tf_k562/test.h5",
            'n_tfs': 7,
            'tf_names': TF_NAMES_7,
        },
        {
            'name': 'K562-fulltf',
            'path': base_dir / "outputs/k562_20tf/K562-20tf_baseline/beacon_20260205_190727/best_model.pt",
            'data': base_dir / "data/processed/multi_tf_k562_20tf/test.h5",
            'n_tfs': 14,
            'tf_names': TF_NAMES_14,
        },
        {
            'name': 'HepG2-7tf',
            'path': base_dir / "outputs/hepg2_7tf",
            'data': base_dir / "data/processed/multi_tf_hepg2_7tf/test.h5",
            'n_tfs': 7,
            'tf_names': ["CTCF", "MYC", "MAX", "CEBPB", "REST", "YY1", "NRF1"],
        },
        {
            'name': 'HepG2-fulltf',
            'path': base_dir / "outputs/hepg2_20tf",
            'data': base_dir / "data/processed/multi_tf_hepg2_20tf/test.h5",
            'n_tfs': 12,
            'tf_names': ["CTCF", "MYC", "MAX", "CEBPB", "REST", "YY1", "NRF1",
                         "ELF1", "FOXA2", "HNF4A", "MAFK", "NFE2L2"],
        },
    ]

    all_results = {}

    for cfg in models_config:
        name = cfg['name']
        print(f"\n  --- {name} ---")

        # Find checkpoint
        ckpt = cfg['path']
        if ckpt.is_dir():
            # Find best_model.pt in subdirectories
            candidates = list(ckpt.glob("**/best_model.pt"))
            if not candidates:
                print(f"  SKIP: No checkpoint found in {ckpt}")
                continue
            ckpt = candidates[0]

        if not ckpt.exists():
            print(f"  SKIP: Checkpoint not found: {ckpt}")
            continue

        if not cfg['data'].exists():
            print(f"  SKIP: Test data not found: {cfg['data']}")
            continue

        model = load_model(ckpt, cfg['n_tfs'], device)
        seqs, profs, tfs, _ = load_test_data(cfg['data'])
        outputs = run_model_batch(model, seqs, device)
        pred_profiles = outputs['profile']

        tf_results = {}
        print(f"  {'TF':<8} {'n':>6} {'Mean r':>8} {'Median r':>10} {'Std':>8}")
        print(f"  {'-'*42}")

        for tf_idx, tf_name in enumerate(cfg['tf_names']):
            mask = tfs == tf_idx
            if mask.sum() == 0:
                continue
            gt = profs[mask]
            pred = pred_profiles[mask]

            rs = []
            for j in range(len(gt)):
                if np.std(gt[j]) > 0:
                    r, _ = pearsonr(gt[j], pred[j])
                    rs.append(r)

            if rs:
                mean_r = float(np.mean(rs))
                median_r = float(np.median(rs))
                std_r = float(np.std(rs))
                print(f"  {tf_name:<8} {len(rs):>6} {mean_r:>8.3f} {median_r:>10.3f} {std_r:>8.3f}")
                tf_results[tf_name] = {
                    'n': len(rs), 'mean_r': mean_r,
                    'median_r': median_r, 'std_r': std_r,
                }

        overall_mean = np.mean([r['mean_r'] for r in tf_results.values()])
        print(f"  {'OVERALL':<8} {sum(r['n'] for r in tf_results.values()):>6} {overall_mean:>8.3f}")

        all_results[name] = {
            'per_tf': tf_results,
            'overall_mean_r': float(overall_mean),
            'checkpoint': str(ckpt),
        }

        del model
        torch.cuda.empty_cache()

    return all_results


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="BEACON Downstream Analyses")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="outputs/downstream_analyses")
    parser.add_argument("--experiments", type=str, default="1,2,3,4,5",
                        help="Comma-separated experiment numbers to run")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [int(x) for x in args.experiments.split(',')]

    all_results = {'timestamp': datetime.now().isoformat()}

    # Load primary model (K562-7tf slot_dropout) for experiments 1, 2, 4
    primary_ckpt = base_dir / "outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt"
    primary_data = base_dir / "data/processed/multi_tf_k562/test.h5"

    if any(e in experiments for e in [1, 2, 4]):
        print("Loading primary model (K562-7tf slot_dropout)...")
        model = load_model(primary_ckpt, n_tfs=7, device=device)
        seqs, profs, tfs, bs = load_test_data(primary_data)
        print(f"  Loaded {len(seqs)} test samples\n")

    if 1 in experiments:
        all_results['cobinding'] = experiment_cobinding(
            model, seqs, tfs, device, TF_NAMES_7)

    if 2 in experiments:
        all_results['ism'] = experiment_ism(
            model, seqs, profs, tfs, device, TF_NAMES_7)

    if 4 in experiments:
        all_results['slot_specialization'] = experiment_slot_specialization(
            model, seqs, tfs, device, TF_NAMES_7)

    # Free primary model before loading others
    if any(e in experiments for e in [1, 2, 4]):
        del model
        torch.cuda.empty_cache()

    if 3 in experiments:
        all_results['scaling'] = experiment_scaling(device)

    if 5 in experiments:
        all_results['per_tf_breakdown'] = experiment_per_tf_breakdown(device)

    # Save results
    results_path = output_dir / "downstream_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
