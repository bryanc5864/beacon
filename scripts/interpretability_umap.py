#!/usr/bin/env python3
"""
Representation Geometry: UMAP/t-SNE of slot embeddings.

Visualizes how BEACON organizes TF representations internally.
Expected: TFs from same family cluster together.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
TF_FAMILIES = {
    "CTCF": "Zinc Finger",
    "GATA1": "GATA",
    "TAL1": "bHLH",
    "MYC": "bHLH",
    "MAX": "bHLH",
    "SPI1": "ETS",
    "CEBPB": "bZIP",
}

FAMILY_COLORS = {
    "Zinc Finger": "#e41a1c",
    "GATA": "#377eb8",
    "bHLH": "#4daf4a",
    "ETS": "#984ea3",
    "bZIP": "#ff7f00",
}


def extract_representations(model, data_loader, device):
    """Extract slot embeddings and other layer representations."""
    model.eval()

    slot_embeddings = []  # After slot attention
    backbone_features = []  # After backbone
    tf_labels = []
    occupancies = []
    tf_logits_all = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            tf_indices = batch['tf_index'].numpy()

            # Forward pass with intermediate outputs
            features = model.encode_sequence(sequences)  # [B, L, D]

            # Run slot attention (returns tuple, not dict)
            slot_out = model.slot_attention(features, return_attention=True)
            if len(slot_out) == 3:  # PositionAwareSlotAttention
                slots, _, _ = slot_out
            else:
                slots, _ = slot_out  # [B, K, D], [B, K, L]

            # Get predictions
            outputs = model(sequences)
            occupancy = outputs['occupancy'].squeeze(-1).cpu().numpy()
            tf_logits = outputs['tf_logits'].cpu().numpy()

            # Extract per-sample representations
            for b in range(sequences.shape[0]):
                # Use the highest-occupancy slot's embedding
                best_slot = occupancy[b].argmax()
                slot_emb = slots[b, best_slot].cpu().numpy()
                slot_embeddings.append(slot_emb)

                # Global backbone features (avg pool)
                backbone_avg = features[b].mean(dim=0).cpu().numpy()
                backbone_features.append(backbone_avg)

                tf_labels.append(int(tf_indices[b]))
                occupancies.append(occupancy[b])
                tf_logits_all.append(tf_logits[b])

    return {
        'slot_embeddings': np.array(slot_embeddings),
        'backbone_features': np.array(backbone_features),
        'tf_labels': np.array(tf_labels),
        'occupancies': np.array(occupancies),
        'tf_logits': np.array(tf_logits_all),
    }


def compute_rsa(embeddings, tf_labels, tf_names):
    """
    Representational Similarity Analysis.
    Compare model's TF similarity to biological TF family similarity.
    """
    n_tfs = len(tf_names)

    # Compute model similarity (mean embedding per TF)
    tf_centroids = []
    for tf_idx in range(n_tfs):
        mask = tf_labels == tf_idx
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
        else:
            centroid = np.zeros(embeddings.shape[1])
        tf_centroids.append(centroid)

    tf_centroids = np.array(tf_centroids)

    # Model similarity matrix
    model_sim = np.zeros((n_tfs, n_tfs))
    for i in range(n_tfs):
        for j in range(n_tfs):
            if np.linalg.norm(tf_centroids[i]) > 0 and np.linalg.norm(tf_centroids[j]) > 0:
                model_sim[i, j] = 1 - cosine(tf_centroids[i], tf_centroids[j])
            else:
                model_sim[i, j] = 0

    # Biological similarity (same family = 1, different = 0)
    bio_sim = np.zeros((n_tfs, n_tfs))
    for i in range(n_tfs):
        for j in range(n_tfs):
            if TF_FAMILIES[tf_names[i]] == TF_FAMILIES[tf_names[j]]:
                bio_sim[i, j] = 1.0
            else:
                bio_sim[i, j] = 0.0

    # RSA correlation (upper triangle only, excluding diagonal)
    mask = np.triu(np.ones((n_tfs, n_tfs), dtype=bool), k=1)
    model_flat = model_sim[mask]
    bio_flat = bio_sim[mask]

    rsa_r, rsa_p = pearsonr(model_flat, bio_flat)

    return model_sim, bio_sim, rsa_r, rsa_p


def linear_probe(features, labels, feature_name=""):
    """Train linear probe and report accuracy."""
    # Simple train/test split (80/20)
    n = len(labels)
    perm = np.random.permutation(n)
    train_idx = perm[:int(0.8 * n)]
    test_idx = perm[int(0.8 * n):]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features[train_idx], labels[train_idx])
    acc = clf.score(features[test_idx], labels[test_idx])

    return acc


def main():
    parser = argparse.ArgumentParser(description="Representation Geometry Analysis")
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

    output_dir = args.output_dir or experiment_dir / "representation_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Representation Geometry Analysis")
    print("=" * 60)

    # Load model
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        config_path = experiment_dir / experiment_dir.name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    print(f"Loading model...")
    model = BEACON(
        seq_len=config.get("seq_len", 2000), input_channels=4,
        backbone_type="dilated", backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16), slot_dim=config.get("slot_dim", 128),
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
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")

    # Extract representations
    print("\nExtracting representations...")
    reps = extract_representations(model, loader, device)

    slot_emb = reps['slot_embeddings']
    backbone_feat = reps['backbone_features']
    labels = reps['tf_labels']

    print(f"  Slot embeddings: {slot_emb.shape}")
    print(f"  Backbone features: {backbone_feat.shape}")

    # 1. t-SNE visualization
    print("\n--- t-SNE Visualization ---")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    slot_tsne = tsne.fit_transform(slot_emb)

    backbone_tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    backbone_2d = backbone_tsne.fit_transform(backbone_feat)

    # 2. PCA
    print("--- PCA ---")
    pca = PCA(n_components=2)
    slot_pca = pca.fit_transform(slot_emb)
    pca_var = pca.explained_variance_ratio_
    print(f"  PCA variance explained: {pca_var[0]:.3f}, {pca_var[1]:.3f}")

    # 3. RSA
    print("\n--- Representational Similarity Analysis ---")
    model_sim, bio_sim, rsa_r, rsa_p = compute_rsa(slot_emb, labels, TF_NAMES)
    print(f"  RSA correlation: r={rsa_r:.3f} (p={rsa_p:.4f})")

    # 4. Linear probing
    print("\n--- Linear Probing ---")
    slot_probe_acc = linear_probe(slot_emb, labels, "slot")
    backbone_probe_acc = linear_probe(backbone_feat, labels, "backbone")
    print(f"  Slot embeddings: {slot_probe_acc*100:.1f}% accuracy")
    print(f"  Backbone features: {backbone_probe_acc*100:.1f}% accuracy")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. t-SNE of slot embeddings
    ax = axes[0, 0]
    for tf_idx, tf_name in enumerate(TF_NAMES):
        mask = labels == tf_idx
        family = TF_FAMILIES[tf_name]
        color = FAMILY_COLORS[family]
        ax.scatter(slot_tsne[mask, 0], slot_tsne[mask, 1],
                  c=color, label=f"{tf_name} ({family})",
                  alpha=0.6, s=20)
    ax.legend(fontsize=7, loc='best')
    ax.set_title('t-SNE of Slot Embeddings')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # 2. t-SNE of backbone features
    ax = axes[0, 1]
    for tf_idx, tf_name in enumerate(TF_NAMES):
        mask = labels == tf_idx
        family = TF_FAMILIES[tf_name]
        color = FAMILY_COLORS[family]
        ax.scatter(backbone_2d[mask, 0], backbone_2d[mask, 1],
                  c=color, label=f"{tf_name}",
                  alpha=0.6, s=20)
    ax.legend(fontsize=7, loc='best')
    ax.set_title('t-SNE of Backbone Features')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # 3. PCA of slot embeddings
    ax = axes[0, 2]
    for tf_idx, tf_name in enumerate(TF_NAMES):
        mask = labels == tf_idx
        family = TF_FAMILIES[tf_name]
        color = FAMILY_COLORS[family]
        ax.scatter(slot_pca[mask, 0], slot_pca[mask, 1],
                  c=color, label=tf_name,
                  alpha=0.6, s=20)
    ax.legend(fontsize=7, loc='best')
    ax.set_title(f'PCA of Slot Embeddings ({pca_var[0]:.1%}+{pca_var[1]:.1%} var)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # 4. Model similarity matrix
    ax = axes[1, 0]
    im = ax.imshow(model_sim, cmap='coolwarm', vmin=-0.5, vmax=1)
    ax.set_xticks(range(len(TF_NAMES)))
    ax.set_yticks(range(len(TF_NAMES)))
    ax.set_xticklabels(TF_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(TF_NAMES, fontsize=8)
    ax.set_title(f'Model TF Similarity\n(RSA r={rsa_r:.3f}, p={rsa_p:.3f})')
    plt.colorbar(im, ax=ax)

    for i in range(len(TF_NAMES)):
        for j in range(len(TF_NAMES)):
            ax.text(j, i, f'{model_sim[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(model_sim[i,j]) > 0.5 else 'black')

    # 5. Biological similarity
    ax = axes[1, 1]
    im = ax.imshow(bio_sim, cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xticks(range(len(TF_NAMES)))
    ax.set_yticks(range(len(TF_NAMES)))
    ax.set_xticklabels(TF_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(TF_NAMES, fontsize=8)
    ax.set_title('Biological TF Family Similarity')
    plt.colorbar(im, ax=ax)

    # 6. Linear probing comparison
    ax = axes[1, 2]
    layers = ['Backbone\n(avg pool)', 'Slot\nEmbeddings', 'Chance']
    accs = [backbone_probe_acc * 100, slot_probe_acc * 100, 100 / len(TF_NAMES)]
    colors = ['steelblue', 'coral', 'gray']
    bars = ax.bar(layers, accs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Linear Probe Accuracy (%)')
    ax.set_title('Where Does TF Information Emerge?')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "representation_geometry.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {
        'rsa': {
            'correlation': float(rsa_r),
            'p_value': float(rsa_p),
            'model_similarity': model_sim.tolist(),
            'biological_similarity': bio_sim.tolist(),
        },
        'linear_probing': {
            'slot_embeddings': float(slot_probe_acc),
            'backbone_features': float(backbone_probe_acc),
            'chance': 1.0 / len(TF_NAMES),
        },
        'pca': {
            'variance_explained': pca_var.tolist(),
        },
    }

    with open(output_dir / "geometry_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
