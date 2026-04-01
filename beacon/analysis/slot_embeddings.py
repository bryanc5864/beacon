"""
Slot embedding analysis for BEACON.

Analyzes learned slot representations to understand what BEACON's slots
encode, including cross-seed consistency, linear probing, and TF family
clustering.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans


def extract_slot_embeddings(model, sequences, device, batch_size=32):
    """Extract slot embeddings and metadata from model.

    Args:
        model: BEACON model in eval mode.
        sequences: [N, L, 4] numpy one-hot sequences.
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Dict with 'slot_embeddings' [N, K, D], 'occupancy' [N, K],
        'tf_logits' [N, K, T].
    """
    import torch
    from collections import defaultdict

    all_outputs = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[i:i + batch_size], dtype=torch.float32
            ).to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(batch, return_attention=True, return_slots=True)
            for k in ['slot_embeddings', 'occupancy', 'tf_logits']:
                if k in outputs:
                    all_outputs[k].append(outputs[k].cpu().numpy())

    result = {}
    for k, v in all_outputs.items():
        arr = np.concatenate(v, axis=0)
        if k == 'occupancy' and arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], arr.shape[1])
        result[k] = arr
    return result


def compute_rsa(embeddings_a, embeddings_b, metric='cosine'):
    """Representational Similarity Analysis between two sets of embeddings.

    Computes pairwise distance matrices for each set and returns the
    Spearman correlation between them.

    Args:
        embeddings_a: [N, D] embedding matrix A.
        embeddings_b: [N, D] embedding matrix B (same N as A).
        metric: Distance metric for pdist.

    Returns:
        Dict with rsa_correlation and p_value.
    """
    if len(embeddings_a) != len(embeddings_b):
        n = min(len(embeddings_a), len(embeddings_b))
        embeddings_a = embeddings_a[:n]
        embeddings_b = embeddings_b[:n]

    dist_a = pdist(embeddings_a, metric=metric)
    dist_b = pdist(embeddings_b, metric=metric)

    rho, pval = spearmanr(dist_a, dist_b)
    return {
        'rsa_correlation': float(rho) if not np.isnan(rho) else 0.0,
        'p_value': float(pval) if not np.isnan(pval) else 1.0,
        'n_pairs': len(dist_a),
    }


def cross_seed_embedding_consistency(seed_checkpoints, test_sequences, n_tfs, device,
                                     max_samples=500):
    """Compute RSA between slot embeddings from different seeds.

    Args:
        seed_checkpoints: Dict mapping seed label to checkpoint path.
        test_sequences: [N, L, 4] numpy sequences.
        n_tfs: Number of TFs for model loading.
        device: Torch device.
        max_samples: Max samples to use (for speed).

    Returns:
        Dict with pairwise RSA and summary.
    """
    from .evaluation import load_model

    seqs = test_sequences[:max_samples]

    # Extract embeddings per seed
    seed_embeddings = {}
    for seed_name, ckpt_path in seed_checkpoints.items():
        model = load_model(ckpt_path, n_tfs, device)
        emb_dict = extract_slot_embeddings(model, seqs, device)
        # Flatten: [N, K, D] → [N*K, D]
        slot_emb = emb_dict['slot_embeddings']
        seed_embeddings[seed_name] = slot_emb.reshape(-1, slot_emb.shape[-1])
        del model
        import torch
        torch.cuda.empty_cache()

    # Pairwise RSA
    seeds = list(seed_embeddings.keys())
    pairwise = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            key = f"{seeds[i]}_vs_{seeds[j]}"
            pairwise[key] = compute_rsa(seed_embeddings[seeds[i]],
                                        seed_embeddings[seeds[j]])

    rsa_values = [v['rsa_correlation'] for v in pairwise.values()]
    return {
        'pairwise': pairwise,
        'mean_rsa': float(np.mean(rsa_values)),
        'std_rsa': float(np.std(rsa_values)),
        'min_rsa': float(np.min(rsa_values)),
        'n_seeds': len(seeds),
    }


def linear_probe_cv(slot_embeddings, labels, n_folds=5):
    """Cross-validated linear probe predicting TF from slot embeddings.

    Args:
        slot_embeddings: [M, D] embeddings for active slots.
        labels: [M] TF index labels.
        n_folds: Number of CV folds.

    Returns:
        Dict with mean_accuracy, std_accuracy, per_fold_accuracies.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    # Need at least n_folds samples per class
    min_count = min(np.bincount(labels))
    if min_count < n_folds:
        n_folds = max(2, min_count)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in skf.split(slot_embeddings, labels):
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(slot_embeddings[train_idx], labels[train_idx])
        pred = clf.predict(slot_embeddings[test_idx])
        fold_accuracies.append(float(accuracy_score(labels[test_idx], pred)))

    return {
        'mean_accuracy': float(np.mean(fold_accuracies)),
        'std_accuracy': float(np.std(fold_accuracies)),
        'per_fold': fold_accuracies,
        'n_folds': n_folds,
        'n_samples': len(labels),
        'n_classes': len(unique_labels),
    }


def tf_family_clustering_metrics(slot_embeddings, tf_labels, tf_families):
    """Evaluate TF family clustering quality in embedding space.

    Args:
        slot_embeddings: [M, D] embeddings for active slots.
        tf_labels: [M] TF name strings.
        tf_families: Dict mapping family name to list of TF names.

    Returns:
        Dict with silhouette_score, adjusted_rand_index, n_families.
    """
    # Map TF name → family
    tf_to_family = {}
    for family, members in tf_families.items():
        for tf in members:
            tf_to_family[tf] = family

    family_labels = np.array([tf_to_family.get(tf, 'unknown') for tf in tf_labels])
    unique_families = np.unique(family_labels)

    if len(unique_families) < 2:
        return {'silhouette_score': None, 'adjusted_rand_index': None,
                'n_families': len(unique_families)}

    # Integer encode families for metrics
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    family_ints = np.array([family_to_idx[f] for f in family_labels])

    # Silhouette score on true family labels
    sil = float(silhouette_score(slot_embeddings, family_ints, metric='cosine'))

    # K-means clustering and ARI
    n_clusters = len(unique_families)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(slot_embeddings)
    ari = float(adjusted_rand_score(family_ints, cluster_labels))

    return {
        'silhouette_score': sil,
        'adjusted_rand_index': ari,
        'n_families': len(unique_families),
        'n_samples': len(tf_labels),
        'families': {f: int((family_labels == f).sum()) for f in unique_families},
    }
