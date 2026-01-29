#!/usr/bin/env python3
"""
Analyze Phase 2A BEACON training results.

Visualizes:
1. Training curves (loss, metrics over epochs)
2. Slot attention patterns
3. Learned slot representations (t-SNE/UMAP)
4. Profile reconstruction quality
5. Position prediction distributions
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def load_history(experiment_dir):
    """Load training history."""
    history_path = experiment_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def plot_training_curves(history, output_dir):
    """Plot training and validation curves."""
    if not history:
        print("No history found, skipping training curves")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract data
    train_data = history.get('train', [])
    val_data = history.get('val', [])

    if not train_data:
        print("No training data in history")
        return

    epochs = [d['epoch'] for d in train_data]

    # Loss
    ax = axes[0, 0]
    train_loss = [d.get('loss', d.get('total_loss', 0)) for d in train_data]
    ax.plot(epochs, train_loss, 'b-', label='Train')
    if val_data:
        val_epochs = [d['epoch'] for d in val_data]
        val_loss = [d.get('loss', d.get('total_loss', 0)) for d in val_data]
        ax.plot(val_epochs, val_loss, 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Profile Pearson
    ax = axes[0, 1]
    if 'profile_pearson' in train_data[0]:
        train_pearson = [d['profile_pearson'] for d in train_data]
        ax.plot(epochs, train_pearson, 'b-', label='Train')
        if val_data and 'profile_pearson' in val_data[0]:
            val_pearson = [d['profile_pearson'] for d in val_data]
            ax.plot(val_epochs, val_pearson, 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pearson R')
    ax.set_title('Profile Pearson Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Target')

    # Site F1
    ax = axes[0, 2]
    if 'site_f1' in train_data[0]:
        train_f1 = [d['site_f1'] for d in train_data]
        ax.plot(epochs, train_f1, 'b-', label='Train')
        if val_data and 'site_f1' in val_data[0]:
            val_f1 = [d['site_f1'] for d in val_data]
            ax.plot(val_epochs, val_f1, 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Site Detection F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target')

    # Slot utilization
    ax = axes[1, 0]
    if 'slot_utilization' in train_data[0]:
        train_util = [d['slot_utilization'] for d in train_data]
        ax.plot(epochs, train_util, 'b-', label='Train')
        if val_data and 'slot_utilization' in val_data[0]:
            val_util = [d['slot_utilization'] for d in val_data]
            ax.plot(val_epochs, val_util, 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Utilization')
    ax.set_title('Slot Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Avg occupancy
    ax = axes[1, 1]
    if 'avg_occupancy' in train_data[0]:
        train_occ = [d['avg_occupancy'] for d in train_data]
        ax.plot(epochs, train_occ, 'b-', label='Train')
        if val_data and 'avg_occupancy' in val_data[0]:
            val_occ = [d['avg_occupancy'] for d in val_data]
            ax.plot(val_epochs, val_occ, 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg Occupancy')
    ax.set_title('Average Slot Occupancy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate (if available)
    ax = axes[1, 2]
    if 'lr' in train_data[0]:
        lrs = [d['lr'] for d in train_data]
        ax.plot(epochs, lrs, 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'LR not logged', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Learning Rate Schedule')

    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def analyze_model_outputs(model, data_loader, device, n_samples=100):
    """Run model and collect outputs for analysis."""
    model.eval()

    all_outputs = {
        'profiles_pred': [],
        'profiles_true': [],
        'attention': [],
        'slot_embeddings': [],
        'positions': [],
        'occupancy': [],
        'tf_logits': [],
    }

    n_collected = 0
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            profiles_true = batch['profile'].to(device)

            outputs = model(sequences)

            all_outputs['profiles_pred'].append(outputs['profile'].cpu())
            all_outputs['profiles_true'].append(profiles_true.cpu())

            if 'attention' in outputs and outputs['attention'] is not None:
                all_outputs['attention'].append(outputs['attention'].cpu())
            if 'slot_embeddings' in outputs and outputs['slot_embeddings'] is not None:
                all_outputs['slot_embeddings'].append(outputs['slot_embeddings'].cpu())
            if 'positions' in outputs:
                all_outputs['positions'].append(outputs['positions'].cpu())
            if 'occupancy' in outputs:
                all_outputs['occupancy'].append(outputs['occupancy'].cpu())
            if 'tf_logits' in outputs:
                all_outputs['tf_logits'].append(outputs['tf_logits'].cpu())

            n_collected += sequences.shape[0]
            if n_collected >= n_samples:
                break

    # Concatenate
    for key in all_outputs:
        if all_outputs[key]:
            all_outputs[key] = torch.cat(all_outputs[key], dim=0)[:n_samples]
        else:
            all_outputs[key] = None

    return all_outputs


def plot_attention_patterns(outputs, output_dir, n_examples=5):
    """Visualize slot attention patterns."""
    attention = outputs.get('attention')
    if attention is None:
        print("No attention weights available")
        return

    n_slots = attention.shape[1]
    seq_len = attention.shape[2]

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples))
    if n_examples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= attention.shape[0]:
            break

        attn = attention[i].numpy()  # [K, L]

        # Plot heatmap
        im = ax.imshow(attn, aspect='auto', cmap='viridis',
                       extent=[0, seq_len, n_slots, 0])
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Slot')
        ax.set_title(f'Sample {i+1}: Slot Attention Patterns')
        plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    save_path = output_dir / "attention_heatmaps.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Also plot attention distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Attention entropy per slot
    ax = axes[0, 0]
    attn_flat = attention.numpy()
    entropy_per_slot = -np.sum(attn_flat * np.log(attn_flat + 1e-10), axis=2).mean(axis=0)
    ax.bar(range(n_slots), entropy_per_slot)
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('Avg Entropy')
    ax.set_title('Attention Entropy per Slot\n(Higher = more diffuse)')
    ax.axhline(y=np.log(seq_len), color='r', linestyle='--', alpha=0.5, label='Max entropy')

    # Max attention per slot
    ax = axes[0, 1]
    max_attn_per_slot = attn_flat.max(axis=2).mean(axis=0)
    ax.bar(range(n_slots), max_attn_per_slot)
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('Avg Max Attention')
    ax.set_title('Peak Attention per Slot\n(Higher = more focused)')

    # Attention overlap between slots
    ax = axes[1, 0]
    # Compute pairwise cosine similarity
    attn_norm = attn_flat / (np.linalg.norm(attn_flat, axis=2, keepdims=True) + 1e-10)
    overlap = np.zeros((n_slots, n_slots))
    for s1 in range(n_slots):
        for s2 in range(n_slots):
            overlap[s1, s2] = np.mean(np.sum(attn_norm[:, s1, :] * attn_norm[:, s2, :], axis=1))
    im = ax.imshow(overlap, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Slot')
    ax.set_title('Slot Attention Similarity\n(Diagonal should be 1)')
    plt.colorbar(im, ax=ax)

    # Active slots histogram
    ax = axes[1, 1]
    occupancy = outputs.get('occupancy')
    if occupancy is not None:
        occ = occupancy.squeeze(-1).numpy()
        ax.hist(occ.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Occupancy')
        ax.set_ylabel('Count')
        ax.set_title('Occupancy Distribution\n(Should have peaks at 0 and 1)')
        ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = output_dir / "attention_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_slot_embeddings(outputs, output_dir):
    """Visualize learned slot embeddings with dimensionality reduction."""
    embeddings = outputs.get('slot_embeddings')
    if embeddings is None:
        print("No slot embeddings available")
        return

    # Reshape: [B, K, D] -> [B*K, D]
    B, K, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D).numpy()

    # Try UMAP first, fall back to t-SNE
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
        emb_2d = reducer.fit_transform(emb_flat)
        method = 'UMAP'
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_flat)-1))
        emb_2d = reducer.fit_transform(emb_flat)
        method = 't-SNE'

    # Color by slot index
    slot_indices = np.tile(np.arange(K), B)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colored by slot
    ax = axes[0]
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=slot_indices,
                         cmap='tab20', alpha=0.5, s=10)
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(f'Slot Embeddings ({method})\nColored by Slot Index')
    plt.colorbar(scatter, ax=ax, label='Slot Index')

    # Colored by sample
    ax = axes[1]
    sample_indices = np.repeat(np.arange(B), K)
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=sample_indices,
                         cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(f'Slot Embeddings ({method})\nColored by Sample Index')
    plt.colorbar(scatter, ax=ax, label='Sample Index')

    plt.tight_layout()
    save_path = output_dir / "slot_embeddings.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_profile_reconstruction(outputs, output_dir, n_examples=6):
    """Compare predicted vs true profiles."""
    pred = outputs['profiles_pred'].numpy()
    true = outputs['profiles_true'].numpy()

    n_cols = 2
    n_rows = (n_examples + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten()

    for i in range(min(n_examples, len(pred))):
        ax = axes[i]

        x = np.arange(len(pred[i]))
        ax.fill_between(x, true[i], alpha=0.5, label='True', color='blue')
        ax.plot(x, pred[i], 'r-', linewidth=1, label='Predicted', alpha=0.8)

        # Compute correlation
        corr = np.corrcoef(pred[i], true[i])[0, 1]

        ax.set_xlabel('Position')
        ax.set_ylabel('Signal')
        ax.set_title(f'Sample {i+1}: Pearson r = {corr:.4f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_examples, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_path = output_dir / "profile_reconstruction.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Overall correlation histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    correlations = [np.corrcoef(pred[i], true[i])[0, 1] for i in range(len(pred))]
    ax.hist(correlations, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(correlations), color='r', linestyle='--',
               label=f'Mean: {np.mean(correlations):.4f}')
    ax.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='Target: 0.5')
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Profile Reconstruction Quality Distribution')
    ax.legend()

    save_path = output_dir / "profile_correlation_dist.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_position_predictions(outputs, output_dir):
    """Analyze position predictions."""
    positions = outputs.get('positions')
    if positions is None:
        print("No position predictions available")
        return

    # positions shape: [B, K, 2] for Gaussian (mean, std) or [B, K, 1] for continuous
    if positions.shape[-1] == 2:
        means = positions[:, :, 0].numpy()
        stds = positions[:, :, 1].numpy()
    else:
        means = positions.squeeze(-1).numpy()
        stds = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position mean distribution
    ax = axes[0, 0]
    ax.hist(means.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Position (normalized 0-1)')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Position Distribution')
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

    # Position std distribution (if Gaussian)
    ax = axes[0, 1]
    if stds is not None:
        ax.hist(stds.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Position Std (uncertainty)')
        ax.set_ylabel('Count')
        ax.set_title('Position Uncertainty Distribution')
    else:
        ax.text(0.5, 0.5, 'No uncertainty\n(non-Gaussian mode)',
                ha='center', va='center', transform=ax.transAxes)

    # Position per slot
    ax = axes[1, 0]
    n_slots = means.shape[1]
    for s in range(n_slots):
        ax.scatter([s]*len(means), means[:, s], alpha=0.3, s=5)
    ax.boxplot([means[:, s] for s in range(n_slots)], positions=range(n_slots))
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('Position')
    ax.set_title('Position Distribution per Slot')

    # Occupancy vs position (if available)
    ax = axes[1, 1]
    occupancy = outputs.get('occupancy')
    if occupancy is not None:
        occ = occupancy.squeeze(-1).numpy()
        ax.scatter(means.flatten(), occ.flatten(), alpha=0.2, s=5)
        ax.set_xlabel('Position')
        ax.set_ylabel('Occupancy')
        ax.set_title('Occupancy vs Position')
    else:
        ax.text(0.5, 0.5, 'No occupancy data',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    save_path = output_dir / "position_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 2A BEACON results")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/ctcf_k562/ctcf_k562_bigwig_v1"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/ctcf_k562_bigwig"))
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples for visualization")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    # Handle nested experiment directory
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = Path(__file__).parent.parent / experiment_dir

    # Check for nested structure
    nested_dir = experiment_dir / experiment_dir.name
    if nested_dir.exists():
        experiment_dir = nested_dir

    output_dir = args.output_dir or experiment_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2A BEACON Analysis")
    print("=" * 60)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load training history
    print("Loading training history...")
    history = load_history(experiment_dir)

    # Plot training curves
    print("\n1. Plotting training curves...")
    plot_training_curves(history, output_dir)

    # Load model
    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Skipping model analysis")
        return 0

    print(f"\n2. Loading model from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path)

    # Load config
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "seq_len": 2000,
            "n_slots": 16,
            "backbone_dim": 128,
            "backbone_layers": 4,
            "slot_dim": 128,
            "n_iterations": 3,
            "n_tfs": 1,
        }

    # Create model
    model = BEACON(
        seq_len=config.get("seq_len", 2000),
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16),
        slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 1),
        position_mode="gaussian",
    )

    # Load weights (handle DataParallel prefix)
    state_dict = checkpoint['model_state_dict']
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load data
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir

    test_path = data_dir / "test.h5"
    if not test_path.exists():
        test_path = data_dir / "val.h5"

    if not test_path.exists():
        print(f"No test data found at {data_dir}")
        return 1

    print(f"\n3. Loading test data from {test_path}...")
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Run inference
    print(f"\n4. Running inference on {args.n_samples} samples...")
    outputs = analyze_model_outputs(model, loader, device, n_samples=args.n_samples)

    # Generate visualizations
    print("\n5. Generating visualizations...")

    print("  - Attention patterns...")
    plot_attention_patterns(outputs, output_dir)

    print("  - Slot embeddings...")
    plot_slot_embeddings(outputs, output_dir)

    print("  - Profile reconstruction...")
    plot_profile_reconstruction(outputs, output_dir)

    print("  - Position predictions...")
    plot_position_predictions(outputs, output_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
