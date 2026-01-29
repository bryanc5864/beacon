#!/usr/bin/env python3
"""
Analyze Phase 2B Multi-TF BEACON results.

Generates:
1. TF confusion matrix
2. Per-TF accuracy breakdown
3. Attention patterns by TF class
4. Slot specialization analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


def load_model(checkpoint_path, config):
    """Load model from checkpoint."""
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

    # Remove 'module.' prefix if present (DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


def collect_predictions(model, data_loader, device, n_samples=None):
    """Run inference and collect all predictions."""
    model.eval()

    all_tf_pred = []
    all_tf_true = []
    all_occupancy = []
    all_attention = []
    all_positions = []
    all_slot_embeddings = []

    n_collected = 0
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            tf_true = batch['tf_index']

            outputs = model(sequences)

            # Get TF predictions from slots with highest occupancy
            occupancy = outputs['occupancy'].squeeze(-1)  # [B, K]
            tf_logits = outputs['tf_logits']  # [B, K, n_tfs]

            # For each sample, get the TF prediction from the most occupied slot
            best_slot = occupancy.argmax(dim=1)  # [B]
            batch_size = sequences.shape[0]
            tf_pred = tf_logits[torch.arange(batch_size), best_slot].argmax(dim=1)

            all_tf_pred.append(tf_pred.cpu())
            all_tf_true.append(tf_true)
            all_occupancy.append(occupancy.cpu())

            if outputs.get('attention') is not None:
                all_attention.append(outputs['attention'].cpu())
            if outputs.get('positions') is not None:
                all_positions.append(outputs['positions'].cpu())
            if outputs.get('slot_embeddings') is not None:
                all_slot_embeddings.append(outputs['slot_embeddings'].cpu())

            n_collected += batch_size
            if n_samples and n_collected >= n_samples:
                break

    results = {
        'tf_pred': torch.cat(all_tf_pred, dim=0).numpy(),
        'tf_true': torch.cat(all_tf_true, dim=0).numpy(),
        'occupancy': torch.cat(all_occupancy, dim=0).numpy(),
    }

    if all_attention:
        results['attention'] = torch.cat(all_attention, dim=0).numpy()
    if all_positions:
        results['positions'] = torch.cat(all_positions, dim=0).numpy()
    if all_slot_embeddings:
        results['slot_embeddings'] = torch.cat(all_slot_embeddings, dim=0).numpy()

    return results


def plot_confusion_matrix(y_true, y_pred, tf_names, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tf_names, yticklabels=tf_names, ax=ax)
    ax.set_xlabel('Predicted TF')
    ax.set_ylabel('True TF')
    ax.set_title('Confusion Matrix (Counts)')

    # Normalized
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tf_names, yticklabels=tf_names, ax=ax)
    ax.set_xlabel('Predicted TF')
    ax.set_ylabel('True TF')
    ax.set_title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    save_path = output_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return cm, cm_normalized


def plot_per_tf_accuracy(y_true, y_pred, tf_names, output_dir):
    """Plot per-TF accuracy breakdown."""
    n_tfs = len(tf_names)

    # Calculate per-class metrics
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i in range(n_tfs):
        mask_true = y_true == i
        mask_pred = y_pred == i

        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        tn = np.sum(~mask_true & ~mask_pred)

        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(np.sum(mask_true))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(n_tfs)
    width = 0.6

    # Precision
    ax = axes[0, 0]
    bars = ax.bar(x, precisions, width, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(tf_names, rotation=45, ha='right')
    ax.set_ylabel('Precision')
    ax.set_title('Per-TF Precision')
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(precisions), color='r', linestyle='--', label=f'Mean: {np.mean(precisions):.3f}')
    ax.legend()
    for bar, val in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Recall
    ax = axes[0, 1]
    bars = ax.bar(x, recalls, width, color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(tf_names, rotation=45, ha='right')
    ax.set_ylabel('Recall')
    ax.set_title('Per-TF Recall')
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(recalls), color='r', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
    ax.legend()
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # F1
    ax = axes[1, 0]
    bars = ax.bar(x, f1s, width, color='forestgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(tf_names, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-TF F1 Score')
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(f1s), color='r', linestyle='--', label=f'Mean: {np.mean(f1s):.3f}')
    ax.legend()
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Support (sample counts)
    ax = axes[1, 1]
    bars = ax.bar(x, supports, width, color='purple')
    ax.set_xticks(x)
    ax.set_xticklabels(tf_names, rotation=45, ha='right')
    ax.set_ylabel('Sample Count')
    ax.set_title('Per-TF Test Samples')
    for bar, val in zip(bars, supports):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(val)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "per_tf_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Return metrics dict
    metrics = {tf_names[i]: {
        'precision': precisions[i],
        'recall': recalls[i],
        'f1': f1s[i],
        'support': int(supports[i])
    } for i in range(n_tfs)}

    return metrics


def plot_attention_by_tf(results, tf_names, output_dir, n_examples=3):
    """Plot attention patterns grouped by TF class."""
    attention = results.get('attention')
    if attention is None:
        print("No attention data available")
        return

    tf_true = results['tf_true']
    n_tfs = len(tf_names)

    fig, axes = plt.subplots(n_tfs, n_examples, figsize=(4*n_examples, 2.5*n_tfs))

    for tf_idx in range(n_tfs):
        # Get samples for this TF
        mask = tf_true == tf_idx
        tf_attention = attention[mask]

        for ex_idx in range(min(n_examples, len(tf_attention))):
            ax = axes[tf_idx, ex_idx] if n_tfs > 1 else axes[ex_idx]

            attn = tf_attention[ex_idx]  # [K, L]
            im = ax.imshow(attn, aspect='auto', cmap='viridis')

            if ex_idx == 0:
                ax.set_ylabel(f'{tf_names[tf_idx]}\nSlot')
            if tf_idx == n_tfs - 1:
                ax.set_xlabel('Position')
            if tf_idx == 0:
                ax.set_title(f'Example {ex_idx + 1}')

    plt.suptitle('Slot Attention Patterns by TF Class', fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "attention_by_tf.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_slot_specialization(results, tf_names, output_dir):
    """Analyze which slots specialize for which TFs."""
    occupancy = results['occupancy']  # [N, K]
    tf_true = results['tf_true']  # [N]

    n_tfs = len(tf_names)
    n_slots = occupancy.shape[1]

    # For each slot, compute average occupancy per TF class
    slot_tf_occupancy = np.zeros((n_slots, n_tfs))

    for tf_idx in range(n_tfs):
        mask = tf_true == tf_idx
        slot_tf_occupancy[:, tf_idx] = occupancy[mask].mean(axis=0)

    # Normalize per slot
    slot_tf_normalized = slot_tf_occupancy / (slot_tf_occupancy.sum(axis=1, keepdims=True) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw occupancy
    ax = axes[0]
    im = ax.imshow(slot_tf_occupancy, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_tfs))
    ax.set_xticklabels(tf_names, rotation=45, ha='right')
    ax.set_yticks(range(n_slots))
    ax.set_ylabel('Slot Index')
    ax.set_xlabel('TF Class')
    ax.set_title('Avg Occupancy per Slot-TF')
    plt.colorbar(im, ax=ax, label='Avg Occupancy')

    # Normalized (slot specialization)
    ax = axes[1]
    im = ax.imshow(slot_tf_normalized, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_tfs))
    ax.set_xticklabels(tf_names, rotation=45, ha='right')
    ax.set_yticks(range(n_slots))
    ax.set_ylabel('Slot Index')
    ax.set_xlabel('TF Class')
    ax.set_title('Slot TF Specialization (Normalized)')
    plt.colorbar(im, ax=ax, label='TF Proportion')

    plt.tight_layout()
    save_path = output_dir / "slot_specialization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Find most specialized slots
    specialization = {}
    for slot_idx in range(n_slots):
        max_tf = np.argmax(slot_tf_normalized[slot_idx])
        max_prop = slot_tf_normalized[slot_idx, max_tf]
        if max_prop > 0.2:  # At least 20% specialized
            specialization[f"slot_{slot_idx}"] = {
                "primary_tf": tf_names[max_tf],
                "proportion": float(max_prop)
            }

    return specialization


def plot_tf_family_analysis(cm_normalized, tf_names, output_dir):
    """Analyze confusion within TF families."""
    # TF family mapping
    tf_families = {
        'CTCF': 'Zinc Finger',
        'GATA1': 'GATA',
        'TAL1': 'bHLH',
        'MYC': 'bHLH',
        'MAX': 'bHLH',
        'SPI1': 'ETS',
        'CEBPB': 'bZIP',
    }

    # Get family for each TF in our list
    families = [tf_families.get(tf, 'Unknown') for tf in tf_names]
    unique_families = list(set(families))

    # Calculate within-family vs between-family confusion
    n_tfs = len(tf_names)
    within_family_confusion = []
    between_family_confusion = []

    for i in range(n_tfs):
        for j in range(n_tfs):
            if i != j:
                if families[i] == families[j]:
                    within_family_confusion.append(cm_normalized[i, j])
                else:
                    between_family_confusion.append(cm_normalized[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram comparison
    ax = axes[0]
    ax.hist(within_family_confusion, bins=20, alpha=0.7, label='Same Family', color='red')
    ax.hist(between_family_confusion, bins=20, alpha=0.7, label='Different Family', color='blue')
    ax.set_xlabel('Confusion Rate')
    ax.set_ylabel('Count')
    ax.set_title('Confusion: Same vs Different TF Family')
    ax.legend()

    within_mean = np.mean(within_family_confusion) if within_family_confusion else 0
    between_mean = np.mean(between_family_confusion) if between_family_confusion else 0
    ax.axvline(within_mean, color='red', linestyle='--', label=f'Within mean: {within_mean:.3f}')
    ax.axvline(between_mean, color='blue', linestyle='--', label=f'Between mean: {between_mean:.3f}')

    # bHLH subfamily analysis (MYC, MAX, TAL1)
    ax = axes[1]
    bhlh_tfs = [i for i, tf in enumerate(tf_names) if tf in ['MYC', 'MAX', 'TAL1']]
    if len(bhlh_tfs) >= 2:
        bhlh_cm = cm_normalized[np.ix_(bhlh_tfs, bhlh_tfs)]
        bhlh_names = [tf_names[i] for i in bhlh_tfs]

        sns.heatmap(bhlh_cm, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=bhlh_names, yticklabels=bhlh_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('bHLH Family Confusion\n(MYC, MAX, TAL1 - Same Family)')
    else:
        ax.text(0.5, 0.5, 'Not enough bHLH TFs', ha='center', va='center')

    plt.tight_layout()
    save_path = output_dir / "tf_family_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return {
        'within_family_mean_confusion': float(within_mean),
        'between_family_mean_confusion': float(between_mean),
        'ratio': float(within_mean / between_mean) if between_mean > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 2B Multi-TF results")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples (None = all)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    # Find experiment directory
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = Path(__file__).parent.parent / experiment_dir

    # Find the actual run directory
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]  # Most recent

    output_dir = args.output_dir or experiment_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2B Multi-TF Analysis")
    print("=" * 60)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load config
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    tf_names = config.get("tf_names", ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"])
    print(f"TFs: {tf_names}")
    print()

    # Load model
    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        # Try nested structure
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config)
    device = torch.device(args.device)
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load data
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir

    test_path = data_dir / "test.h5"
    if not test_path.exists():
        print(f"Test data not found: {test_path}")
        return 1

    print(f"\nLoading test data from {test_path}...")
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")

    # Collect predictions
    print("\nRunning inference...")
    results = collect_predictions(model, loader, device, n_samples=args.n_samples)
    print(f"Collected {len(results['tf_pred'])} predictions")

    # Generate analyses
    print("\n" + "=" * 60)
    print("Generating Analyses")
    print("=" * 60)

    # 1. Confusion matrix
    print("\n1. Confusion Matrix...")
    cm, cm_normalized = plot_confusion_matrix(
        results['tf_true'], results['tf_pred'], tf_names, output_dir
    )

    # 2. Per-TF metrics
    print("\n2. Per-TF Metrics...")
    per_tf_metrics = plot_per_tf_accuracy(
        results['tf_true'], results['tf_pred'], tf_names, output_dir
    )

    # 3. Attention by TF
    print("\n3. Attention Patterns by TF...")
    plot_attention_by_tf(results, tf_names, output_dir)

    # 4. Slot specialization
    print("\n4. Slot Specialization...")
    specialization = plot_slot_specialization(results, tf_names, output_dir)

    # 5. TF family analysis
    print("\n5. TF Family Analysis...")
    family_analysis = plot_tf_family_analysis(cm_normalized, tf_names, output_dir)

    # Save summary
    summary = {
        'overall_accuracy': float(np.mean(results['tf_pred'] == results['tf_true'])),
        'per_tf_metrics': per_tf_metrics,
        'slot_specialization': specialization,
        'family_analysis': family_analysis,
        'n_samples': len(results['tf_pred']),
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nOverall TF Accuracy: {summary['overall_accuracy']:.1%}")
    print(f"\nPer-TF F1 Scores:")
    for tf, metrics in per_tf_metrics.items():
        print(f"  {tf}: {metrics['f1']:.3f} (n={metrics['support']})")

    print(f"\nTF Family Confusion:")
    print(f"  Within-family mean: {family_analysis['within_family_mean_confusion']:.3f}")
    print(f"  Between-family mean: {family_analysis['between_family_mean_confusion']:.3f}")
    print(f"  Ratio (within/between): {family_analysis['ratio']:.2f}x")

    if specialization:
        print(f"\nSlot Specialization:")
        for slot, info in specialization.items():
            print(f"  {slot}: {info['primary_tf']} ({info['proportion']:.1%})")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
