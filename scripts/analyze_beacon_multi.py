#!/usr/bin/env python3
"""
BEACON-multi Characterization Analysis

Priority #1 analyses:
  1a. Per-slot quality: TF accuracy by slot rank (highest-occ, 2nd-highest, etc.)
  1b. Compositional co-binding validation: MYC+MAX, GATA1+TAL1 enrichment
  1c. Performance breakdown by number of ground-truth TFs per sequence
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
TF_FAMILIES = {
    "CTCF": "Zinc Finger", "GATA1": "GATA", "TAL1": "bHLH",
    "MYC": "bHLH", "MAX": "bHLH", "SPI1": "ETS", "CEBPB": "bZIP",
}


def load_model(config, checkpoint_path, device):
    """Load BEACON model from checkpoint."""
    model = BEACON(
        seq_len=config.get("seq_len", 2000), input_channels=4,
        backbone_type="dilated", backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16), slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 7), position_mode="gaussian",
        attention_mode=config.get("attention_mode", "independent"),
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, loader, device):
    """Run model on all test data, collect predictions and ground truth."""
    all_pred_occupancy = []
    all_pred_tf_logits = []
    all_pred_positions = []
    all_gt_binding_sites = []
    all_profiles = []
    all_pred_profiles = []

    with torch.no_grad():
        for batch in loader:
            sequences = batch['sequence'].to(device)
            outputs = model(sequences)

            occ = outputs['occupancy'].squeeze(-1).cpu().numpy()  # [B, K]
            tf_logits = outputs['tf_logits'].cpu().numpy()         # [B, K, 7]
            positions = outputs['positions'].cpu().numpy()          # [B, K, 2]
            pred_profile = outputs['profile'].cpu().numpy()         # [B, L]

            all_pred_occupancy.append(occ)
            all_pred_tf_logits.append(tf_logits)
            all_pred_positions.append(positions)
            all_pred_profiles.append(pred_profile)
            all_gt_binding_sites.append(batch['binding_sites'].numpy())
            all_profiles.append(batch['profile'].numpy())

    return {
        'pred_occupancy': np.concatenate(all_pred_occupancy),
        'pred_tf_logits': np.concatenate(all_pred_tf_logits),
        'pred_positions': np.concatenate(all_pred_positions),
        'pred_profiles': np.concatenate(all_pred_profiles),
        'gt_binding_sites': np.concatenate(all_gt_binding_sites),
        'gt_profiles': np.concatenate(all_profiles),
    }


def hungarian_match(pred_positions, pred_tf, pred_occ, gt_sites):
    """Match predicted slots to ground truth sites using Hungarian algorithm.

    gt_sites: [K, 3] where each row is [position, tf_id, occupancy]
    Returns list of (pred_slot_idx, gt_site_idx) pairs.
    """
    # Find valid ground truth sites
    gt_valid = gt_sites[:, 2] > 0.1  # occupancy > 0.1
    gt_indices = np.where(gt_valid)[0]
    n_gt = len(gt_indices)

    if n_gt == 0:
        return [], gt_indices

    # Find active predicted slots (occupancy > 0.3)
    pred_active = pred_occ > 0.3
    pred_indices = np.where(pred_active)[0]
    n_pred = len(pred_indices)

    if n_pred == 0:
        return [], gt_indices

    # Build cost matrix [n_pred, n_gt]
    cost = np.zeros((n_pred, n_gt))
    for i, pi in enumerate(pred_indices):
        for j, gi in enumerate(gt_indices):
            # Position cost (normalized 0-1)
            pos_cost = abs(pred_positions[pi, 0] - gt_sites[gi, 0])
            # TF cost (0 if match, 1 if not)
            tf_cost = 0.0 if pred_tf[pi] == int(gt_sites[gi, 1]) else 1.0
            cost[i, j] = pos_cost + tf_cost

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = [(pred_indices[r], gt_indices[c]) for r, c in zip(row_ind, col_ind)]
    return matches, gt_indices


def analysis_1a_per_slot_quality(data):
    """Per-slot TF accuracy by slot rank (highest occupancy first)."""
    print("\n" + "=" * 60)
    print("Analysis 1a: Per-Slot Quality by Rank")
    print("=" * 60)

    pred_occ = data['pred_occupancy']      # [N, K]
    pred_tf_logits = data['pred_tf_logits']  # [N, K, 7]
    pred_positions = data['pred_positions']  # [N, K, 2]
    gt_sites = data['gt_binding_sites']      # [N, K, 3]

    n_samples = len(pred_occ)
    pred_tf = pred_tf_logits.argmax(axis=-1)  # [N, K]

    # Track per-rank statistics
    rank_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'occupancies': [],
                                       'tf_distribution': Counter()})

    # Track matched vs unmatched
    total_gt_sites = 0
    total_matched = 0
    slot_usage_counts = Counter()

    for i in range(n_samples):
        matches, gt_indices = hungarian_match(
            pred_positions[i], pred_tf[i], pred_occ[i], gt_sites[i]
        )
        n_gt = len(gt_indices)
        total_gt_sites += n_gt
        total_matched += len(matches)

        # Count active slots
        n_active = (pred_occ[i] > 0.3).sum()
        slot_usage_counts[int(n_active)] += 1

        # Rank active slots by occupancy (descending)
        active_slots = np.where(pred_occ[i] > 0.3)[0]
        if len(active_slots) == 0:
            continue

        ranked_slots = active_slots[np.argsort(-pred_occ[i][active_slots])]

        # Check each ranked slot against its match
        matched_pred_slots = {m[0]: m[1] for m in matches}

        for rank, slot_idx in enumerate(ranked_slots):
            rank_stats[rank]['occupancies'].append(pred_occ[i][slot_idx])
            pred_tf_id = pred_tf[i][slot_idx]
            rank_stats[rank]['tf_distribution'][TF_NAMES[pred_tf_id]] += 1

            if slot_idx in matched_pred_slots:
                gt_idx = matched_pred_slots[slot_idx]
                gt_tf_id = int(gt_sites[i][gt_idx][1])
                if pred_tf_id == gt_tf_id:
                    rank_stats[rank]['correct'] += 1
                rank_stats[rank]['total'] += 1
            else:
                # Active slot with no match — false positive
                rank_stats[rank]['total'] += 1

    print(f"\nTotal ground-truth binding sites: {total_gt_sites}")
    print(f"Total matched (Hungarian): {total_matched}")
    print(f"Match rate: {total_matched/max(total_gt_sites,1)*100:.1f}%")

    print(f"\nSlot usage distribution:")
    for n_slots in sorted(slot_usage_counts.keys()):
        count = slot_usage_counts[n_slots]
        print(f"  {n_slots} active slots: {count} samples ({count/n_samples*100:.1f}%)")

    print(f"\n{'Rank':<6} {'TF Acc':<10} {'N Matched':<12} {'Mean Occ':<10} {'Top TFs'}")
    print("-" * 70)

    results = {}
    for rank in sorted(rank_stats.keys()):
        stats = rank_stats[rank]
        acc = stats['correct'] / max(stats['total'], 1) * 100
        mean_occ = np.mean(stats['occupancies']) if stats['occupancies'] else 0
        top_tfs = stats['tf_distribution'].most_common(3)
        top_tfs_str = ', '.join(f"{tf}({c})" for tf, c in top_tfs)

        print(f"  {rank:<4} {acc:>6.1f}%    {stats['total']:<10} {mean_occ:.3f}     {top_tfs_str}")

        results[f'rank_{rank}'] = {
            'tf_accuracy': acc,
            'n_matched': stats['total'],
            'mean_occupancy': float(mean_occ),
            'top_tfs': dict(stats['tf_distribution']),
        }

    # Per-TF accuracy for matched slots
    print(f"\n--- Per-TF Accuracy (matched slots only) ---")
    tf_correct = defaultdict(int)
    tf_total = defaultdict(int)

    for i in range(n_samples):
        matches, gt_indices = hungarian_match(
            pred_positions[i], pred_tf[i], pred_occ[i], gt_sites[i]
        )
        for pred_idx, gt_idx in matches:
            gt_tf_id = int(gt_sites[i][gt_idx][1])
            pred_tf_id = pred_tf[i][pred_idx]
            tf_total[gt_tf_id] += 1
            if pred_tf_id == gt_tf_id:
                tf_correct[gt_tf_id] += 1

    print(f"\n{'TF':<8} {'Accuracy':<10} {'Correct':<10} {'Total'}")
    print("-" * 40)
    for tf_idx in range(7):
        total = tf_total[tf_idx]
        correct = tf_correct[tf_idx]
        acc = correct / max(total, 1) * 100
        print(f"  {TF_NAMES[tf_idx]:<6} {acc:>6.1f}%    {correct:<8} {total}")
        results[f'per_tf_{TF_NAMES[tf_idx]}'] = {'accuracy': acc, 'correct': correct, 'total': total}

    overall_correct = sum(tf_correct.values())
    overall_total = sum(tf_total.values())
    overall_acc = overall_correct / max(overall_total, 1) * 100
    print(f"  {'Overall':<6} {overall_acc:>6.1f}%    {overall_correct:<8} {overall_total}")
    results['overall_matched_accuracy'] = overall_acc

    return results


def analysis_1b_co_binding(data):
    """Compositional co-binding validation."""
    print("\n" + "=" * 60)
    print("Analysis 1b: Compositional Co-Binding Validation")
    print("=" * 60)

    pred_occ = data['pred_occupancy']
    pred_tf_logits = data['pred_tf_logits']
    pred_positions = data['pred_positions']
    gt_sites = data['gt_binding_sites']

    n_samples = len(pred_occ)
    pred_tf = pred_tf_logits.argmax(axis=-1)

    # --- Ground truth co-occurrence ---
    gt_pair_counts = Counter()
    gt_pair_spacings = defaultdict(list)
    gt_tf_counts = Counter()
    gt_n_tfs_per_sample = []

    for i in range(n_samples):
        gt_valid = gt_sites[i][:, 2] > 0.1
        gt_active = gt_sites[i][gt_valid]
        gt_tfs = set()
        for site in gt_active:
            tf_id = int(site[1])
            gt_tfs.add(tf_id)
            gt_tf_counts[tf_id] += 1

        gt_n_tfs_per_sample.append(len(gt_tfs))

        for j in range(len(gt_active)):
            for k in range(j + 1, len(gt_active)):
                tf_a = int(gt_active[j][1])
                tf_b = int(gt_active[k][1])
                pair = tuple(sorted([tf_a, tf_b]))
                gt_pair_counts[pair] += 1
                spacing = abs(gt_active[j][0] - gt_active[k][0]) * 2000  # Convert to bp
                gt_pair_spacings[pair].append(spacing)

    # --- Predicted co-occurrence ---
    pred_pair_counts = Counter()
    pred_pair_spacings = defaultdict(list)
    pred_tf_counts = Counter()
    pred_n_tfs_per_sample = []

    for i in range(n_samples):
        active_slots = np.where(pred_occ[i] > 0.3)[0]
        pred_tfs = set()
        active_sites = []

        for s in active_slots:
            tf_id = int(pred_tf[i][s])
            pred_tfs.add(tf_id)
            pred_tf_counts[tf_id] += 1
            active_sites.append((s, tf_id, pred_positions[i][s][0]))

        pred_n_tfs_per_sample.append(len(pred_tfs))

        for j in range(len(active_sites)):
            for k in range(j + 1, len(active_sites)):
                tf_a = active_sites[j][1]
                tf_b = active_sites[k][1]
                pair = tuple(sorted([tf_a, tf_b]))
                pred_pair_counts[pair] += 1
                spacing = abs(active_sites[j][2] - active_sites[k][2]) * 2000
                pred_pair_spacings[pair].append(spacing)

    # --- Print results ---
    print(f"\nGround Truth TF Co-occurrence:")
    print(f"  Samples with 1 TF: {sum(1 for x in gt_n_tfs_per_sample if x == 1)} ({sum(1 for x in gt_n_tfs_per_sample if x == 1)/n_samples*100:.1f}%)")
    print(f"  Samples with 2+ TFs: {sum(1 for x in gt_n_tfs_per_sample if x >= 2)} ({sum(1 for x in gt_n_tfs_per_sample if x >= 2)/n_samples*100:.1f}%)")
    print(f"  Mean unique TFs per sample: {np.mean(gt_n_tfs_per_sample):.2f}")

    print(f"\nPredicted TF Co-occurrence:")
    print(f"  Samples with 1 predicted TF: {sum(1 for x in pred_n_tfs_per_sample if x == 1)}")
    print(f"  Samples with 2+ predicted TFs: {sum(1 for x in pred_n_tfs_per_sample if x >= 2)}")
    print(f"  Mean unique predicted TFs per sample: {np.mean(pred_n_tfs_per_sample):.2f}")

    # Top co-occurring pairs
    print(f"\n--- Top Ground Truth TF Pairs ---")
    print(f"{'Pair':<20} {'Count':<8} {'Mean Spacing (bp)'}")
    print("-" * 50)
    for pair, count in gt_pair_counts.most_common(15):
        tf_a, tf_b = TF_NAMES[pair[0]], TF_NAMES[pair[1]]
        mean_sp = np.mean(gt_pair_spacings[pair]) if gt_pair_spacings[pair] else 0
        print(f"  {tf_a}-{tf_b:<12} {count:<8} {mean_sp:.0f}")

    print(f"\n--- Top Predicted TF Pairs ---")
    print(f"{'Pair':<20} {'Count':<8} {'Mean Spacing (bp)'}")
    print("-" * 50)
    for pair, count in pred_pair_counts.most_common(15):
        tf_a, tf_b = TF_NAMES[pair[0]], TF_NAMES[pair[1]]
        mean_sp = np.mean(pred_pair_spacings[pair]) if pred_pair_spacings[pair] else 0
        print(f"  {tf_a}-{tf_b:<12} {count:<8} {mean_sp:.0f}")

    # --- Enrichment analysis ---
    # Expected co-occurrence under independence
    total_pred_pairs = sum(pred_pair_counts.values())

    print(f"\n--- Co-Binding Enrichment (Predicted vs Expected) ---")
    print(f"{'Pair':<20} {'Observed':<10} {'Expected':<10} {'Enrichment':<12} {'Biological?'}")
    print("-" * 70)

    results = {}

    if total_pred_pairs > 0:
        total_pred_tfs = sum(pred_tf_counts.values())
        for pair, obs_count in pred_pair_counts.most_common(15):
            tf_a, tf_b = pair
            # Expected under independence
            freq_a = pred_tf_counts[tf_a] / total_pred_tfs
            freq_b = pred_tf_counts[tf_b] / total_pred_tfs
            expected = freq_a * freq_b * total_pred_pairs * (2 if tf_a != tf_b else 1)
            enrichment = obs_count / max(expected, 0.01)

            pair_name = f"{TF_NAMES[tf_a]}-{TF_NAMES[tf_b]}"

            # Known biological interactions
            bio_known = ""
            if set(pair) == {3, 4}:  # MYC-MAX
                bio_known = "YES (heterodimer)"
            elif set(pair) == {1, 2}:  # GATA1-TAL1
                bio_known = "YES (erythroid)"
            elif set(pair) == {3, 2} or set(pair) == {4, 2}:  # MYC/MAX-TAL1
                bio_known = "bHLH family"

            print(f"  {pair_name:<18} {obs_count:<10} {expected:<10.1f} {enrichment:<12.2f} {bio_known}")

            mean_sp = np.mean(pred_pair_spacings[pair]) if pred_pair_spacings[pair] else 0
            results[pair_name] = {
                'observed': int(obs_count),
                'expected': float(expected),
                'enrichment': float(enrichment),
                'mean_spacing_bp': float(mean_sp),
                'biological': bio_known,
            }

    # --- Biologically important pairs ---
    print(f"\n--- Key Biological Pairs ---")

    key_pairs = [
        ((3, 4), "MYC-MAX", "Obligate heterodimer, share E-box (CACGTG)"),
        ((1, 2), "GATA1-TAL1", "Erythroid co-regulatory program"),
        ((0, 5), "CTCF-SPI1", "Insulator + myeloid (should NOT co-occur)"),
    ]

    for pair, name, description in key_pairs:
        gt_count = gt_pair_counts.get(tuple(sorted(pair)), 0)
        pred_count = pred_pair_counts.get(tuple(sorted(pair)), 0)
        gt_spacing = gt_pair_spacings.get(tuple(sorted(pair)), [])
        pred_spacing = pred_pair_spacings.get(tuple(sorted(pair)), [])

        print(f"\n  {name} ({description}):")
        print(f"    Ground truth: {gt_count} co-occurrences" +
              (f", mean spacing {np.mean(gt_spacing):.0f}bp" if gt_spacing else ""))
        print(f"    Predicted:    {pred_count} co-occurrences" +
              (f", mean spacing {np.mean(pred_spacing):.0f}bp" if pred_spacing else ""))

    return results


def analysis_1c_by_complexity(data):
    """Performance breakdown by number of ground-truth TFs per sequence."""
    print("\n" + "=" * 60)
    print("Analysis 1c: Performance by Sequence Complexity")
    print("=" * 60)

    pred_occ = data['pred_occupancy']
    pred_tf_logits = data['pred_tf_logits']
    pred_positions = data['pred_positions']
    pred_profiles = data['pred_profiles']
    gt_sites = data['gt_binding_sites']
    gt_profiles = data['gt_profiles']

    n_samples = len(pred_occ)
    pred_tf = pred_tf_logits.argmax(axis=-1)

    # Group samples by number of GT binding sites
    complexity_groups = defaultdict(lambda: {
        'indices': [], 'n_active_pred': [], 'tf_correct': [], 'tf_total': [],
        'site_detected': [], 'site_total': [], 'profile_pearson': [],
    })

    for i in range(n_samples):
        gt_valid = gt_sites[i][:, 2] > 0.1
        n_gt = gt_valid.sum()

        # Determine unique GT TFs
        gt_active = gt_sites[i][gt_valid]
        gt_tfs = set(int(s[1]) for s in gt_active)
        n_gt_tfs = len(gt_tfs)

        # Group label
        if n_gt <= 1:
            group = "1 site"
        elif n_gt == 2:
            group = "2 sites"
        elif n_gt == 3:
            group = "3 sites"
        else:
            group = "4+ sites"

        g = complexity_groups[group]
        g['indices'].append(i)

        # Count active predicted slots
        n_active_pred = (pred_occ[i] > 0.3).sum()
        g['n_active_pred'].append(n_active_pred)

        # Hungarian matching for TF accuracy
        matches, gt_indices = hungarian_match(
            pred_positions[i], pred_tf[i], pred_occ[i], gt_sites[i]
        )

        correct = 0
        for pred_idx, gt_idx in matches:
            gt_tf_id = int(gt_sites[i][gt_idx][1])
            pred_tf_id = pred_tf[i][pred_idx]
            if pred_tf_id == gt_tf_id:
                correct += 1

        g['tf_correct'].append(correct)
        g['tf_total'].append(int(n_gt))

        # Site detection rate
        g['site_detected'].append(len(matches))
        g['site_total'].append(int(n_gt))

        # Profile Pearson correlation
        p = pred_profiles[i]
        g_prof = gt_profiles[i]
        if np.std(p) > 0 and np.std(g_prof) > 0:
            r = np.corrcoef(p, g_prof)[0, 1]
        else:
            r = 0.0
        g['profile_pearson'].append(r)

    # Print results table
    print(f"\n{'Complexity':<12} {'N':<8} {'Avg Slots':<12} {'Site Det%':<12} {'TF Acc%':<10} {'Profile r'}")
    print("-" * 70)

    results = {}
    for group in ["1 site", "2 sites", "3 sites", "4+ sites"]:
        if group not in complexity_groups:
            continue
        g = complexity_groups[group]
        n = len(g['indices'])
        avg_slots = np.mean(g['n_active_pred'])

        total_detected = sum(g['site_detected'])
        total_sites = sum(g['site_total'])
        site_det = total_detected / max(total_sites, 1) * 100

        total_correct = sum(g['tf_correct'])
        total_tf = sum(g['tf_total'])
        tf_acc = total_correct / max(total_tf, 1) * 100

        mean_profile_r = np.mean(g['profile_pearson'])

        print(f"  {group:<10} {n:<8} {avg_slots:<12.2f} {site_det:<12.1f} {tf_acc:<10.1f} {mean_profile_r:.3f}")

        results[group] = {
            'n_samples': n,
            'avg_slots_predicted': float(avg_slots),
            'site_detection_rate': float(site_det),
            'tf_accuracy': float(tf_acc),
            'profile_pearson': float(mean_profile_r),
        }

    # Overall
    all_detected = sum(sum(g['site_detected']) for g in complexity_groups.values())
    all_sites = sum(sum(g['site_total']) for g in complexity_groups.values())
    all_correct = sum(sum(g['tf_correct']) for g in complexity_groups.values())
    all_tf = sum(sum(g['tf_total']) for g in complexity_groups.values())
    all_r = np.mean([r for g in complexity_groups.values() for r in g['profile_pearson']])
    all_slots = np.mean([s for g in complexity_groups.values() for s in g['n_active_pred']])

    print(f"  {'Overall':<10} {n_samples:<8} {all_slots:<12.2f} {all_detected/max(all_sites,1)*100:<12.1f} {all_correct/max(all_tf,1)*100:<10.1f} {all_r:.3f}")

    results['overall'] = {
        'n_samples': n_samples,
        'avg_slots_predicted': float(all_slots),
        'site_detection_rate': float(all_detected / max(all_sites, 1) * 100),
        'tf_accuracy': float(all_correct / max(all_tf, 1) * 100),
        'profile_pearson': float(all_r),
    }

    return results


def create_visualizations(data, results_1a, results_1b, results_1c, output_dir):
    """Create summary visualization figures."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Per-rank TF accuracy bar chart
    ax = axes[0, 0]
    ranks = []
    accs = []
    for rank in range(6):
        key = f'rank_{rank}'
        if key in results_1a:
            ranks.append(f"Rank {rank}")
            accs.append(results_1a[key]['tf_accuracy'])
    if ranks:
        bars = ax.bar(ranks, accs, color='steelblue', alpha=0.8, edgecolor='black')
        ax.axhline(y=14.3, color='red', linestyle='--', label='Chance (14.3%)')
        ax.set_ylabel('TF Accuracy (%)')
        ax.set_title('TF Accuracy by Slot Rank')
        ax.legend()
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', fontsize=9)

    # 2. Per-TF accuracy
    ax = axes[0, 1]
    tf_accs = []
    tf_names_plot = []
    for tf in TF_NAMES:
        key = f'per_tf_{tf}'
        if key in results_1a:
            tf_names_plot.append(tf)
            tf_accs.append(results_1a[key]['accuracy'])
    if tf_names_plot:
        colors = ['#e41a1c' if TF_FAMILIES[tf] == 'Zinc Finger' else
                  '#377eb8' if TF_FAMILIES[tf] == 'GATA' else
                  '#4daf4a' if TF_FAMILIES[tf] == 'bHLH' else
                  '#984ea3' if TF_FAMILIES[tf] == 'ETS' else
                  '#ff7f00' for tf in tf_names_plot]
        bars = ax.bar(tf_names_plot, tf_accs, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(y=14.3, color='red', linestyle='--', label='Chance')
        ax.set_ylabel('TF Accuracy (%)')
        ax.set_title('Per-TF Accuracy (Matched Slots)')
        ax.legend()
        for bar, acc in zip(bars, tf_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', fontsize=8)

    # 3. Slot usage histogram
    ax = axes[0, 2]
    n_active = (data['pred_occupancy'] > 0.3).sum(axis=1)
    bins = np.arange(-0.5, max(n_active) + 1.5, 1)
    ax.hist(n_active, bins=bins, color='coral', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Active Slots per Sample')
    ax.set_ylabel('Count')
    ax.set_title(f'Slot Usage Distribution (mean={np.mean(n_active):.2f})')
    ax.axvline(x=np.mean(n_active), color='red', linestyle='--', label=f'Mean={np.mean(n_active):.2f}')
    ax.legend()

    # 4. Performance by complexity
    ax = axes[1, 0]
    groups = ["1 site", "2 sites", "3 sites", "4+ sites"]
    groups_present = [g for g in groups if g in results_1c]
    if groups_present:
        x = np.arange(len(groups_present))
        width = 0.25
        site_det = [results_1c[g]['site_detection_rate'] for g in groups_present]
        tf_acc = [results_1c[g]['tf_accuracy'] for g in groups_present]
        prof_r = [results_1c[g]['profile_pearson'] * 100 for g in groups_present]

        ax.bar(x - width, site_det, width, label='Site Det %', color='steelblue', alpha=0.8)
        ax.bar(x, tf_acc, width, label='TF Acc %', color='coral', alpha=0.8)
        ax.bar(x + width, prof_r, width, label='Profile r×100', color='mediumseagreen', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups_present)
        ax.set_ylabel('Percentage / Score')
        ax.set_title('Performance by Sequence Complexity')
        ax.legend(fontsize=8)

    # 5. Predicted vs GT slots per sample
    ax = axes[1, 1]
    gt_n = []
    pred_n = []
    for i in range(len(data['pred_occupancy'])):
        gt_valid = data['gt_binding_sites'][i][:, 2] > 0.1
        gt_n.append(gt_valid.sum())
        pred_n.append((data['pred_occupancy'][i] > 0.3).sum())

    gt_n = np.array(gt_n)
    pred_n = np.array(pred_n)

    # Jitter for visibility
    jitter = np.random.normal(0, 0.1, len(gt_n))
    ax.scatter(gt_n + jitter, pred_n + jitter, alpha=0.1, s=5, c='steelblue')
    ax.plot([0, 7], [0, 7], 'r--', label='Perfect')
    ax.set_xlabel('Ground Truth # Sites')
    ax.set_ylabel('Predicted # Active Slots')
    ax.set_title('Slot Count: Predicted vs Ground Truth')
    ax.legend()

    # Compute mean predicted per GT count
    for n in range(1, 7):
        mask = gt_n == n
        if mask.sum() > 10:
            mean_pred = pred_n[mask].mean()
            ax.plot(n, mean_pred, 'ro', markersize=8)
            ax.annotate(f'{mean_pred:.1f}', (n + 0.1, mean_pred + 0.1), fontsize=8, color='red')

    # 6. Co-binding enrichment
    ax = axes[1, 2]
    if results_1b:
        pairs = list(results_1b.keys())[:10]
        enrichments = [results_1b[p]['enrichment'] for p in pairs]
        bio_colors = ['gold' if results_1b[p].get('biological', '') else 'lightgray' for p in pairs]

        y_pos = np.arange(len(pairs))
        ax.barh(y_pos, enrichments, color=bio_colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=8)
        ax.axvline(x=1.0, color='red', linestyle='--', label='Expected')
        ax.set_xlabel('Enrichment (Obs/Expected)')
        ax.set_title('Co-Binding Enrichment')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "beacon_multi_characterization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {output_dir / 'beacon_multi_characterization.png'}")


def main():
    parser = argparse.ArgumentParser(description="BEACON-multi Characterization")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/beacon_multi"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/beacon_multi"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir

    # Find latest run
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    # Find checkpoint
    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"
    if not checkpoint_path.exists():
        # Try inner subdir
        for d in experiment_dir.iterdir():
            if d.is_dir():
                cp = d / "best_model.pt"
                if cp.exists():
                    checkpoint_path = cp
                    break

    # Load config
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        config_path = experiment_dir.parent / "config.json"
    if not config_path.exists():
        # Try all levels
        for p in [experiment_dir / experiment_dir.name / "config.json",
                   checkpoint_path.parent / "config.json"]:
            if p.exists():
                config_path = p
                break

    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = args.output_dir or experiment_dir / "characterization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEACON-multi Characterization Analysis")
    print("=" * 60)

    # Load model
    print(f"\nLoading model...")
    device = torch.device(args.device)
    model = load_model(config, checkpoint_path, device)

    # Load data
    test_path = data_dir / "test.h5"
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")

    # Run inference
    print("Running inference...")
    data = run_inference(model, loader, device)
    print(f"  Predictions: occupancy {data['pred_occupancy'].shape}, "
          f"tf_logits {data['pred_tf_logits'].shape}")

    # Run analyses
    results_1a = analysis_1a_per_slot_quality(data)
    results_1b = analysis_1b_co_binding(data)
    results_1c = analysis_1c_by_complexity(data)

    # Create visualizations
    create_visualizations(data, results_1a, results_1b, results_1c, output_dir)

    # Save results
    all_results = {
        'per_slot_quality': results_1a,
        'co_binding': results_1b,
        'by_complexity': results_1c,
    }

    with open(output_dir / "characterization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
