#!/usr/bin/env python3
"""
Phase 5B: Cross-Cell-Line Transfer

Tests whether BEACON trained on K562 can generalize to HepG2.

Key questions:
1. Does profile prediction transfer? (BEACON learns sequence → profile)
2. Does TF classification transfer? (BEACON learns motifs, not cell-specific)
3. Does binding site detection transfer?
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


def evaluate_model(model, data_loader, device, tf_names, occupancy_threshold=0.5):
    """Evaluate BEACON on a dataset."""
    model.eval()

    all_true_tfs = []
    all_pred_tfs = []
    all_occupancies = []
    all_profile_corrs = []
    all_sites_detected = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            true_profiles = batch['profile'].numpy()
            true_tfs = batch['tf_index'].numpy()

            outputs = model(sequences)

            occupancy = outputs['occupancy'].squeeze(-1).cpu().numpy()
            tf_logits = outputs['tf_logits'].cpu().numpy()
            pred_profiles = outputs['profile'].squeeze(-1).cpu().numpy()

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                # TF classification (from highest occupancy slot)
                best_slot = occupancy[b].argmax()
                pred_tf = tf_logits[b, best_slot].argmax()
                all_true_tfs.append(true_tfs[b])
                all_pred_tfs.append(pred_tf)

                # Occupancy
                max_occ = occupancy[b].max()
                all_occupancies.append(max_occ)

                # Site detection
                site_detected = max_occ > occupancy_threshold
                all_sites_detected.append(site_detected)

                # Profile correlation
                true_prof = true_profiles[b]
                pred_prof = pred_profiles[b]
                if true_prof.std() > 0 and pred_prof.std() > 0:
                    corr = pearsonr(true_prof.flatten(), pred_prof.flatten())[0]
                else:
                    corr = 0.0
                all_profile_corrs.append(corr)

    # Compute metrics
    results = {
        'tf_accuracy': accuracy_score(all_true_tfs, all_pred_tfs),
        'tf_f1_macro': f1_score(all_true_tfs, all_pred_tfs, average='macro'),
        'site_detection_rate': np.mean(all_sites_detected),
        'mean_occupancy': np.mean(all_occupancies),
        'profile_pearson_mean': np.mean(all_profile_corrs),
        'profile_pearson_median': np.median(all_profile_corrs),
        'n_samples': len(all_true_tfs),
        'confusion_matrix': confusion_matrix(all_true_tfs, all_pred_tfs).tolist(),
        'all_true_tfs': all_true_tfs,
        'all_pred_tfs': all_pred_tfs,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 5B: Cross-Cell Transfer")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--k562-data", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--hepg2-data", type=Path,
                        default=Path("data/processed/multi_tf_hepg2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Setup paths
    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = base_dir / model_dir

    if model_dir.exists():
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if subdirs:
            model_dir = sorted(subdirs)[-1]

    k562_data = args.k562_data
    if not k562_data.is_absolute():
        k562_data = base_dir / k562_data

    hepg2_data = args.hepg2_data
    if not hepg2_data.is_absolute():
        hepg2_data = base_dir / hepg2_data

    output_dir = args.output_dir or model_dir / "cross_cell_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 5B: Cross-Cell-Line Transfer (K562 → HepG2)")
    print("=" * 60)
    print(f"Model: {model_dir}")
    print(f"K562 data: {k562_data}")
    print(f"HepG2 data: {hepg2_data}")
    print()

    # Load model config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path = model_dir / model_dir.name / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    k562_tf_names = config.get("tf_names", ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"])

    # Load HepG2 config
    with open(hepg2_data / "config.json") as f:
        hepg2_config = json.load(f)
    hepg2_tf_names = hepg2_config["tf_names"]

    # Find common TFs
    common_tfs = [tf for tf in hepg2_tf_names if tf in k562_tf_names]
    print(f"K562 TFs: {k562_tf_names}")
    print(f"HepG2 TFs: {hepg2_tf_names}")
    print(f"Common TFs: {common_tfs}")

    if len(common_tfs) < 2:
        print("ERROR: Need at least 2 common TFs")
        return 1

    # Create TF index mapping (HepG2 → K562)
    tf_mapping = {hepg2_tf_names.index(tf): k562_tf_names.index(tf)
                  for tf in common_tfs}
    print(f"TF mapping (HepG2 idx → K562 idx): {tf_mapping}")

    # Load model
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = model_dir / model_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    print(f"\nLoading K562-trained model from {checkpoint_path}...")

    model = BEACON(
        seq_len=config.get("seq_len", 2000),
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16),
        slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 7),
        position_mode="gaussian",
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Load datasets
    print("\nLoading datasets...")

    k562_test = BEACONDataset(k562_data / "test.h5",
                               seq_length=config.get("seq_len", 2000),
                               augment=False, n_slots=config.get("n_slots", 16))

    hepg2_test = BEACONDataset(hepg2_data / "test.h5",
                                seq_length=config.get("seq_len", 2000),
                                augment=False, n_slots=config.get("n_slots", 16))

    k562_loader = DataLoader(k562_test, batch_size=32, shuffle=False, num_workers=0)
    hepg2_loader = DataLoader(hepg2_test, batch_size=32, shuffle=False, num_workers=0)

    print(f"K562 test samples: {len(k562_test)}")
    print(f"HepG2 test samples: {len(hepg2_test)}")

    # Evaluate on K562 (baseline)
    print("\n" + "=" * 60)
    print("Evaluating on K562 (same cell line - baseline)")
    print("=" * 60)

    k562_results = evaluate_model(model, k562_loader, device, k562_tf_names)

    print(f"\nK562 Results:")
    print(f"  TF Accuracy: {k562_results['tf_accuracy']*100:.1f}%")
    print(f"  TF F1 (macro): {k562_results['tf_f1_macro']:.3f}")
    print(f"  Site Detection: {k562_results['site_detection_rate']*100:.1f}%")
    print(f"  Profile Pearson (mean): {k562_results['profile_pearson_mean']:.3f}")

    # Evaluate on HepG2 (transfer)
    print("\n" + "=" * 60)
    print("Evaluating on HepG2 (cross-cell transfer)")
    print("=" * 60)

    hepg2_results = evaluate_model(model, hepg2_loader, device, k562_tf_names)

    # Remap TF indices for accuracy calculation (only count common TFs)
    hepg2_true_mapped = []
    hepg2_pred_list = hepg2_results['all_pred_tfs']

    for i, true_tf in enumerate(hepg2_results['all_true_tfs']):
        if true_tf in tf_mapping:
            hepg2_true_mapped.append(tf_mapping[true_tf])
        else:
            hepg2_true_mapped.append(-1)  # TF not in K562 model

    # Filter to common TFs only
    valid_mask = [t != -1 for t in hepg2_true_mapped]
    hepg2_true_valid = [t for t, v in zip(hepg2_true_mapped, valid_mask) if v]
    hepg2_pred_valid = [p for p, v in zip(hepg2_pred_list, valid_mask) if v]

    hepg2_tf_accuracy = accuracy_score(hepg2_true_valid, hepg2_pred_valid)
    hepg2_tf_f1 = f1_score(hepg2_true_valid, hepg2_pred_valid, average='macro')

    print(f"\nHepG2 Results (transfer):")
    print(f"  TF Accuracy: {hepg2_tf_accuracy*100:.1f}%")
    print(f"  TF F1 (macro): {hepg2_tf_f1:.3f}")
    print(f"  Site Detection: {hepg2_results['site_detection_rate']*100:.1f}%")
    print(f"  Profile Pearson (mean): {hepg2_results['profile_pearson_mean']:.3f}")

    # Compute transfer efficiency
    print("\n" + "=" * 60)
    print("TRANSFER EFFICIENCY")
    print("=" * 60)

    tf_transfer = hepg2_tf_accuracy / k562_results['tf_accuracy'] * 100
    site_transfer = hepg2_results['site_detection_rate'] / k562_results['site_detection_rate'] * 100
    profile_transfer = hepg2_results['profile_pearson_mean'] / k562_results['profile_pearson_mean'] * 100

    print(f"\n| Metric | K562 | HepG2 | Transfer % |")
    print(f"|--------|------|-------|------------|")
    print(f"| TF Accuracy | {k562_results['tf_accuracy']*100:.1f}% | {hepg2_tf_accuracy*100:.1f}% | {tf_transfer:.1f}% |")
    print(f"| Site Detection | {k562_results['site_detection_rate']*100:.1f}% | {hepg2_results['site_detection_rate']*100:.1f}% | {site_transfer:.1f}% |")
    print(f"| Profile Pearson | {k562_results['profile_pearson_mean']:.3f} | {hepg2_results['profile_pearson_mean']:.3f} | {profile_transfer:.1f}% |")

    # Per-TF analysis
    print("\n" + "=" * 60)
    print("PER-TF TRANSFER ANALYSIS")
    print("=" * 60)

    for hepg2_idx, k562_idx in tf_mapping.items():
        tf_name = hepg2_tf_names[hepg2_idx]

        # K562 performance for this TF
        k562_mask = [t == k562_idx for t in k562_results['all_true_tfs']]
        k562_correct = sum(1 for t, p, m in zip(k562_results['all_true_tfs'],
                                                  k562_results['all_pred_tfs'], k562_mask)
                          if m and t == p)
        k562_total = sum(k562_mask)
        k562_tf_acc = k562_correct / k562_total if k562_total > 0 else 0

        # HepG2 performance for this TF
        hepg2_mask = [t == k562_idx for t in hepg2_true_mapped]
        hepg2_correct = sum(1 for t, p, m in zip(hepg2_true_mapped,
                                                   hepg2_pred_list, hepg2_mask)
                           if m and t == p)
        hepg2_total = sum(hepg2_mask)
        hepg2_tf_acc = hepg2_correct / hepg2_total if hepg2_total > 0 else 0

        transfer_pct = hepg2_tf_acc / k562_tf_acc * 100 if k562_tf_acc > 0 else 0

        print(f"\n{tf_name}:")
        print(f"  K562: {k562_tf_acc*100:.1f}% ({k562_correct}/{k562_total})")
        print(f"  HepG2: {hepg2_tf_acc*100:.1f}% ({hepg2_correct}/{hepg2_total})")
        print(f"  Transfer: {transfer_pct:.1f}%")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if tf_transfer > 80:
        interpretation = "EXCELLENT transfer - model learns TF motifs, not cell-specific patterns"
    elif tf_transfer > 60:
        interpretation = "GOOD transfer - model generalizes well across cell types"
    elif tf_transfer > 40:
        interpretation = "MODERATE transfer - some cell-type-specific learning"
    else:
        interpretation = "POOR transfer - model may be overfitting to cell-specific patterns"

    print(f"\n{interpretation}")
    print(f"\nTF classification transfer: {tf_transfer:.1f}%")
    print(f"Site detection transfer: {site_transfer:.1f}%")
    print(f"Profile prediction transfer: {profile_transfer:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Transfer comparison bar chart
    ax = axes[0, 0]
    metrics = ['TF Accuracy', 'Site Detection', 'Profile r']
    k562_vals = [k562_results['tf_accuracy']*100,
                  k562_results['site_detection_rate']*100,
                  k562_results['profile_pearson_mean']*100]
    hepg2_vals = [hepg2_tf_accuracy*100,
                   hepg2_results['site_detection_rate']*100,
                   hepg2_results['profile_pearson_mean']*100]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, k562_vals, width, label='K562 (train cell)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, hepg2_vals, width, label='HepG2 (transfer)', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Performance (%)')
    ax.set_title('Cross-Cell Transfer: K562 → HepG2')
    ax.legend()
    ax.set_ylim(0, 105)

    for i, (k, h) in enumerate(zip(k562_vals, hepg2_vals)):
        ax.text(i - width/2, k + 2, f'{k:.0f}', ha='center', fontsize=9)
        ax.text(i + width/2, h + 2, f'{h:.0f}', ha='center', fontsize=9)

    # 2. Per-TF transfer
    ax = axes[0, 1]
    tf_names_common = [hepg2_tf_names[i] for i in tf_mapping.keys()]
    tf_transfers = []

    for hepg2_idx, k562_idx in tf_mapping.items():
        k562_mask = [t == k562_idx for t in k562_results['all_true_tfs']]
        k562_correct = sum(1 for t, p, m in zip(k562_results['all_true_tfs'],
                                                  k562_results['all_pred_tfs'], k562_mask)
                          if m and t == p)
        k562_total = sum(k562_mask)
        k562_tf_acc = k562_correct / k562_total if k562_total > 0 else 0

        hepg2_mask = [t == k562_idx for t in hepg2_true_mapped]
        hepg2_correct = sum(1 for t, p, m in zip(hepg2_true_mapped,
                                                   hepg2_pred_list, hepg2_mask)
                           if m and t == p)
        hepg2_total = sum(hepg2_mask)
        hepg2_tf_acc = hepg2_correct / hepg2_total if hepg2_total > 0 else 0

        transfer = hepg2_tf_acc / k562_tf_acc * 100 if k562_tf_acc > 0 else 0
        tf_transfers.append(transfer)

    colors = ['green' if t > 80 else 'orange' if t > 50 else 'red' for t in tf_transfers]
    bars = ax.bar(tf_names_common, tf_transfers, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=100, color='blue', linestyle='--', alpha=0.5, label='100% transfer')
    ax.axhline(y=60, color='green', linestyle=':', alpha=0.5, label='60% threshold')
    ax.set_ylabel('Transfer Efficiency (%)')
    ax.set_xlabel('Transcription Factor')
    ax.set_title('Per-TF Transfer Efficiency')
    ax.legend()

    for bar, t in zip(bars, tf_transfers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{t:.0f}%', ha='center', fontsize=9)

    # 3. Confusion matrix - K562
    ax = axes[1, 0]
    cm_k562 = np.array(k562_results['confusion_matrix'])
    # Normalize by row
    cm_k562_norm = cm_k562.astype(float) / cm_k562.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_k562_norm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(k562_tf_names)))
    ax.set_yticks(range(len(k562_tf_names)))
    ax.set_xticklabels(k562_tf_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(k562_tf_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('K562 Confusion Matrix (%)')
    plt.colorbar(im, ax=ax)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
CROSS-CELL TRANSFER SUMMARY
===========================

Training cell line: K562
Test cell line: HepG2
Common TFs: {', '.join(common_tfs)}

TRANSFER EFFICIENCY:
  TF Classification: {tf_transfer:.1f}%
  Site Detection: {site_transfer:.1f}%
  Profile Prediction: {profile_transfer:.1f}%

CONCLUSION:
{interpretation}

KEY INSIGHT:
BEACON learns TF binding motifs from sequence,
which transfer across cell types because motifs
are conserved while chromatin accessibility differs.
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_dir / "cross_cell_transfer.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {
        'k562': {
            'tf_accuracy': float(k562_results['tf_accuracy']),
            'tf_f1': float(k562_results['tf_f1_macro']),
            'site_detection': float(k562_results['site_detection_rate']),
            'profile_pearson': float(k562_results['profile_pearson_mean']),
            'n_samples': int(k562_results['n_samples']),
        },
        'hepg2': {
            'tf_accuracy': float(hepg2_tf_accuracy),
            'tf_f1': float(hepg2_tf_f1),
            'site_detection': float(hepg2_results['site_detection_rate']),
            'profile_pearson': float(hepg2_results['profile_pearson_mean']),
            'n_samples': int(hepg2_results['n_samples']),
        },
        'transfer_efficiency': {
            'tf_accuracy': float(tf_transfer),
            'site_detection': float(site_transfer),
            'profile_pearson': float(profile_transfer),
        },
        'common_tfs': common_tfs,
        'interpretation': interpretation,
    }

    with open(output_dir / "cross_cell_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
