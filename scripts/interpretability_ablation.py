#!/usr/bin/env python3
"""
Slot Ablation Study: Which slots are necessary for which TFs?

For each slot:
1. Zero it out (set occupancy=0)
2. Measure performance drop per TF
3. Test necessity (removing hurts) and sufficiency (alone predicts)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import copy

import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


def evaluate_with_mask(model, data_loader, device, slot_mask=None):
    """
    Evaluate model with optional slot masking.
    slot_mask: boolean array [K] - True = keep slot, False = zero it out
    """
    model.eval()

    per_tf_profiles = defaultdict(lambda: {'preds': [], 'trues': []})
    per_tf_correct = defaultdict(int)
    per_tf_total = defaultdict(int)
    all_sites_detected = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            true_profiles = batch['profile'].numpy()
            tf_indices = batch['tf_index'].numpy()

            outputs = model(sequences)

            occupancy = outputs['occupancy'].squeeze(-1)  # [B, K]
            tf_logits = outputs['tf_logits']  # [B, K, n_tfs]
            pred_profiles = outputs['profile'].squeeze(-1).cpu().numpy()

            # Apply slot mask to occupancy
            if slot_mask is not None:
                mask_tensor = torch.tensor(slot_mask, dtype=torch.float32, device=device)
                occupancy = occupancy * mask_tensor.unsqueeze(0)

            occupancy_np = occupancy.cpu().numpy()
            tf_logits_np = tf_logits.cpu().numpy()

            for b in range(sequences.shape[0]):
                tf_idx = int(tf_indices[b])
                total_samples += 1
                per_tf_total[tf_idx] += 1

                # TF classification from highest occupancy slot
                best_slot = occupancy_np[b].argmax()
                pred_tf = tf_logits_np[b, best_slot].argmax()
                if pred_tf == tf_idx:
                    per_tf_correct[tf_idx] += 1

                # Site detection
                if occupancy_np[b].max() > 0.5:
                    all_sites_detected += 1

                # Profile correlation
                true_p = true_profiles[b]
                pred_p = pred_profiles[b]
                if true_p.std() > 0 and pred_p.std() > 0:
                    r = pearsonr(true_p.flatten(), pred_p.flatten())[0]
                else:
                    r = 0.0
                per_tf_profiles[tf_idx]['preds'].append(r)

    results = {
        'overall_tf_acc': sum(per_tf_correct.values()) / sum(per_tf_total.values()),
        'site_detection': all_sites_detected / total_samples,
        'per_tf_acc': {},
        'per_tf_profile_r': {},
    }

    for tf_idx in range(len(TF_NAMES)):
        if per_tf_total[tf_idx] > 0:
            results['per_tf_acc'][TF_NAMES[tf_idx]] = per_tf_correct[tf_idx] / per_tf_total[tf_idx]
            results['per_tf_profile_r'][TF_NAMES[tf_idx]] = float(np.mean(
                per_tf_profiles[tf_idx]['preds'])) if per_tf_profiles[tf_idx]['preds'] else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Slot Ablation Study")
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = args.output_dir or experiment_dir / "ablation_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Slot Ablation Study")
    print("=" * 60)

    # Load config
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        config_path = experiment_dir / experiment_dir.name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    n_slots = config.get("n_slots", 16)

    # Load model
    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    print(f"Loading model from {checkpoint_path}...")
    model = BEACON(
        seq_len=config.get("seq_len", 2000), input_channels=4,
        backbone_type="dilated", backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=n_slots, slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 7), position_mode="gaussian",
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Load data
    test_path = data_dir / "test.h5"
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=n_slots)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")

    # 1. Baseline (no ablation)
    print("\n--- Baseline (all slots active) ---")
    baseline = evaluate_with_mask(model, loader, device, slot_mask=None)
    print(f"  TF Accuracy: {baseline['overall_tf_acc']*100:.1f}%")
    print(f"  Site Detection: {baseline['site_detection']*100:.1f}%")

    # 2. Single-slot ablation (NECESSITY)
    print("\n--- Single-Slot Ablation (Necessity) ---")
    print("Removing one slot at a time...\n")

    necessity_results = {}
    for slot_idx in range(n_slots):
        mask = np.ones(n_slots, dtype=bool)
        mask[slot_idx] = False

        result = evaluate_with_mask(model, loader, device, slot_mask=mask)

        tf_acc_drop = baseline['overall_tf_acc'] - result['overall_tf_acc']
        site_drop = baseline['site_detection'] - result['site_detection']

        per_tf_drops = {}
        for tf in TF_NAMES:
            drop = baseline['per_tf_acc'].get(tf, 0) - result['per_tf_acc'].get(tf, 0)
            per_tf_drops[tf] = drop

        necessity_results[slot_idx] = {
            'tf_acc_drop': tf_acc_drop,
            'site_detection_drop': site_drop,
            'per_tf_drops': per_tf_drops,
            'overall_tf_acc': result['overall_tf_acc'],
        }

        # Find most affected TF
        most_affected = max(per_tf_drops.items(), key=lambda x: x[1])
        if tf_acc_drop > 0.01:
            print(f"  Slot {slot_idx:2d}: TF acc drop={tf_acc_drop*100:+.1f}%, "
                  f"most affected: {most_affected[0]} ({most_affected[1]*100:+.1f}%)")

    # 3. Single-slot sufficiency
    print("\n--- Single-Slot Sufficiency ---")
    print("Keeping only one slot active...\n")

    sufficiency_results = {}
    for slot_idx in range(n_slots):
        mask = np.zeros(n_slots, dtype=bool)
        mask[slot_idx] = True

        result = evaluate_with_mask(model, loader, device, slot_mask=mask)

        sufficiency_results[slot_idx] = {
            'tf_acc': result['overall_tf_acc'],
            'site_detection': result['site_detection'],
            'per_tf_acc': result['per_tf_acc'],
        }

        if result['overall_tf_acc'] > 0.2:
            print(f"  Slot {slot_idx:2d}: TF acc={result['overall_tf_acc']*100:.1f}%, "
                  f"site detect={result['site_detection']*100:.1f}%")

    # 4. Summary: Identify critical slots
    print("\n" + "=" * 60)
    print("CRITICAL SLOTS ANALYSIS")
    print("=" * 60)

    # Most necessary slots (largest drop when removed)
    sorted_necessity = sorted(necessity_results.items(),
                              key=lambda x: x[1]['tf_acc_drop'], reverse=True)

    print("\nMost Necessary Slots (largest accuracy drop when removed):")
    for slot_idx, data in sorted_necessity[:5]:
        print(f"  Slot {slot_idx}: {data['tf_acc_drop']*100:+.1f}% overall drop")
        top_tfs = sorted(data['per_tf_drops'].items(), key=lambda x: x[1], reverse=True)[:2]
        for tf, drop in top_tfs:
            if drop > 0.01:
                print(f"    -> {tf}: {drop*100:+.1f}%")

    # Most sufficient slots (highest accuracy alone)
    sorted_sufficiency = sorted(sufficiency_results.items(),
                                key=lambda x: x[1]['tf_acc'], reverse=True)

    print("\nMost Sufficient Slots (highest accuracy when used alone):")
    for slot_idx, data in sorted_sufficiency[:5]:
        print(f"  Slot {slot_idx}: {data['tf_acc']*100:.1f}% TF accuracy alone")
        top_tfs = sorted(data['per_tf_acc'].items(), key=lambda x: x[1], reverse=True)[:2]
        for tf, acc in top_tfs:
            if acc > 0.2:
                print(f"    -> {tf}: {acc*100:.1f}%")

    # 5. Per-TF critical slot map
    print("\n" + "=" * 60)
    print("PER-TF CRITICAL SLOTS")
    print("=" * 60)

    for tf in TF_NAMES:
        # Find most necessary slot for this TF
        drops = [(s, d['per_tf_drops'][tf]) for s, d in necessity_results.items()]
        drops.sort(key=lambda x: x[1], reverse=True)
        best_nec = drops[0]

        # Find most sufficient slot for this TF
        suffs = [(s, d['per_tf_acc'].get(tf, 0)) for s, d in sufficiency_results.items()]
        suffs.sort(key=lambda x: x[1], reverse=True)
        best_suf = suffs[0]

        print(f"\n  {tf}:")
        print(f"    Most necessary: Slot {best_nec[0]} (removing drops {best_nec[1]*100:.1f}%)")
        print(f"    Most sufficient: Slot {best_suf[0]} ({best_suf[1]*100:.1f}% alone)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Necessity heatmap (slot x TF)
    ax = axes[0, 0]
    necessity_matrix = np.zeros((n_slots, len(TF_NAMES)))
    for slot_idx in range(n_slots):
        for j, tf in enumerate(TF_NAMES):
            necessity_matrix[slot_idx, j] = necessity_results[slot_idx]['per_tf_drops'][tf]

    im = ax.imshow(necessity_matrix.T * 100, cmap='Reds', aspect='auto')
    ax.set_xticks(range(n_slots))
    ax.set_yticks(range(len(TF_NAMES)))
    ax.set_xticklabels(range(n_slots))
    ax.set_yticklabels(TF_NAMES)
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('TF')
    ax.set_title('Necessity: Accuracy Drop When Slot Removed (%)')
    plt.colorbar(im, ax=ax, label='Drop (%)')

    # 2. Sufficiency heatmap
    ax = axes[0, 1]
    sufficiency_matrix = np.zeros((n_slots, len(TF_NAMES)))
    for slot_idx in range(n_slots):
        for j, tf in enumerate(TF_NAMES):
            sufficiency_matrix[slot_idx, j] = sufficiency_results[slot_idx]['per_tf_acc'].get(tf, 0)

    im = ax.imshow(sufficiency_matrix.T * 100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n_slots))
    ax.set_yticks(range(len(TF_NAMES)))
    ax.set_xticklabels(range(n_slots))
    ax.set_yticklabels(TF_NAMES)
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('TF')
    ax.set_title('Sufficiency: Accuracy With Only This Slot (%)')
    plt.colorbar(im, ax=ax, label='Accuracy (%)')

    # 3. Overall necessity bar chart
    ax = axes[1, 0]
    drops = [necessity_results[s]['tf_acc_drop'] * 100 for s in range(n_slots)]
    colors = ['red' if d > 2 else 'orange' if d > 0.5 else 'gray' for d in drops]
    ax.bar(range(n_slots), drops, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('TF Accuracy Drop (%)')
    ax.set_title('Overall Slot Necessity')
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='>2% critical')
    ax.legend()

    # 4. Overall sufficiency bar chart
    ax = axes[1, 1]
    accs = [sufficiency_results[s]['tf_acc'] * 100 for s in range(n_slots)]
    chance = 100 / len(TF_NAMES)
    colors = ['green' if a > chance * 2 else 'orange' if a > chance else 'gray' for a in accs]
    ax.bar(range(n_slots), accs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=chance, color='red', linestyle='--', alpha=0.5, label=f'Chance ({chance:.0f}%)')
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('TF Accuracy (%)')
    ax.set_title('Overall Slot Sufficiency')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "slot_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {
        'baseline': {
            'tf_acc': baseline['overall_tf_acc'],
            'site_detection': baseline['site_detection'],
            'per_tf_acc': baseline['per_tf_acc'],
        },
        'necessity': {str(k): {
            'tf_acc_drop': v['tf_acc_drop'],
            'site_detection_drop': v['site_detection_drop'],
            'per_tf_drops': v['per_tf_drops'],
        } for k, v in necessity_results.items()},
        'sufficiency': {str(k): {
            'tf_acc': v['tf_acc'],
            'site_detection': v['site_detection'],
            'per_tf_acc': v['per_tf_acc'],
        } for k, v in sufficiency_results.items()},
    }

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
