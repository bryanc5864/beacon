"""
Extended cross-cell-type generalization analysis for BEACON.

Deepens the existing K562->HepG2 transfer results (94.3% efficiency)
by analyzing which features transfer and which don't.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def motif_level_transfer(k562_grad_importance, hepg2_grad_importance,
                         shared_tf_names, k562_tf_map, hepg2_tf_map):
    """Compare gradient importance patterns between cell types for shared TFs.

    Args:
        k562_grad_importance: Dict from compute_per_tf_gradient_importance for K562.
        hepg2_grad_importance: Dict from compute_per_tf_gradient_importance for HepG2.
        shared_tf_names: List of shared TF names.
        k562_tf_map: Dict mapping TF name to K562 TF index.
        hepg2_tf_map: Dict mapping TF name to HepG2 TF index.

    Returns:
        Dict per shared TF with gradient pattern correlation.
    """
    results = {}
    for tf_name in shared_tf_names:
        k562_idx = k562_tf_map.get(tf_name)
        hepg2_idx = hepg2_tf_map.get(tf_name)

        if k562_idx is None or hepg2_idx is None:
            continue
        if k562_idx not in k562_grad_importance or hepg2_idx not in hepg2_grad_importance:
            continue

        # Average importance across samples: [L, 4] → [L] (magnitude)
        k562_imp = np.abs(k562_grad_importance[k562_idx]['importance']).mean(axis=0).sum(axis=-1)
        hepg2_imp = np.abs(hepg2_grad_importance[hepg2_idx]['importance']).mean(axis=0).sum(axis=-1)

        # Correlate the importance profiles
        if np.std(k562_imp) > 0 and np.std(hepg2_imp) > 0:
            r = float(np.corrcoef(k562_imp, hepg2_imp)[0, 1])
        else:
            r = 0.0

        results[tf_name] = {
            'importance_correlation': r,
            'k562_n_samples': len(k562_grad_importance[k562_idx]['sequences']),
            'hepg2_n_samples': len(hepg2_grad_importance[hepg2_idx]['sequences']),
        }

    return results


def attention_pattern_transfer(k562_outputs, hepg2_outputs,
                               k562_tf_idx, hepg2_tf_idx,
                               shared_tf_names, k562_tf_map, hepg2_tf_map):
    """Compare attention pattern statistics between cell types for shared TFs.

    Args:
        k562_outputs: Dict from run_model_batch(return_attention=True) on K562 data.
        hepg2_outputs: Dict from run_model_batch(return_attention=True) on HepG2 data.
        k562_tf_idx: [N_k562] TF index per K562 sample.
        hepg2_tf_idx: [N_hepg2] TF index per HepG2 sample.
        shared_tf_names: List of shared TF names.
        k562_tf_map: Dict mapping TF name to K562 TF index.
        hepg2_tf_map: Dict mapping TF name to HepG2 TF index.

    Returns:
        Dict per shared TF with attention statistics comparison.
    """
    results = {}

    for tf_name in shared_tf_names:
        k_idx = k562_tf_map.get(tf_name)
        h_idx = hepg2_tf_map.get(tf_name)
        if k_idx is None or h_idx is None:
            continue

        k_mask = k562_tf_idx == k_idx
        h_mask = hepg2_tf_idx == h_idx

        if not k_mask.any() or not h_mask.any():
            continue

        k_attn = k562_outputs['attention'][k_mask]  # [N_k, K, L]
        h_attn = hepg2_outputs['attention'][h_mask]  # [N_h, K, L]

        # Max attention across slots: [N, L]
        k_max = k_attn.max(axis=1)
        h_max = h_attn.max(axis=1)

        # Entropy of attention distributions
        def attn_entropy(attn):
            """Mean entropy across samples and slots."""
            # attn: [N, K, L]
            eps = 1e-10
            ent = -np.sum(attn * np.log(attn + eps), axis=-1)  # [N, K]
            return float(ent.mean())

        # Peak width: number of positions above 50th percentile of max attention
        def peak_width(max_attn):
            widths = []
            for i in range(len(max_attn)):
                thr = np.percentile(max_attn[i], 50)
                widths.append(int((max_attn[i] > thr).sum()))
            return float(np.mean(widths))

        # Peak count: number of local maxima in max attention
        def peak_count(max_attn):
            counts = []
            for i in range(len(max_attn)):
                a = max_attn[i]
                peaks = 0
                for j in range(1, len(a) - 1):
                    if a[j] > a[j-1] and a[j] > a[j+1] and a[j] > np.percentile(a, 90):
                        peaks += 1
                counts.append(peaks)
            return float(np.mean(counts))

        results[tf_name] = {
            'k562_entropy': attn_entropy(k_attn),
            'hepg2_entropy': attn_entropy(h_attn),
            'k562_peak_width': peak_width(k_max),
            'hepg2_peak_width': peak_width(h_max),
            'k562_peak_count': peak_count(k_max),
            'hepg2_peak_count': peak_count(h_max),
            'k562_n': int(k_mask.sum()),
            'hepg2_n': int(h_mask.sum()),
        }

    return results


def cell_specific_vs_shared_features(k562_embeddings, hepg2_embeddings,
                                     k562_tf_labels, hepg2_tf_labels):
    """Quantify cell-type vs TF identity variance in slot embeddings.

    Args:
        k562_embeddings: [M1, D] slot embeddings from K562 active slots.
        hepg2_embeddings: [M2, D] slot embeddings from HepG2 active slots.
        k562_tf_labels: [M1] TF name strings for K562.
        hepg2_tf_labels: [M2] TF name strings for HepG2.

    Returns:
        Dict with PCA variance analysis and R^2 decomposition.
    """
    # Combine embeddings
    all_embeddings = np.vstack([k562_embeddings, hepg2_embeddings])
    cell_labels = np.array(['K562'] * len(k562_embeddings) + ['HepG2'] * len(hepg2_embeddings))
    tf_labels = np.concatenate([k562_tf_labels, hepg2_tf_labels])

    # PCA
    n_components = min(10, all_embeddings.shape[1], len(all_embeddings) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(all_embeddings)

    # Variance explained by cell type (R^2)
    cell_int = (cell_labels == 'HepG2').astype(float).reshape(-1, 1)
    reg_cell = LinearRegression().fit(cell_int, pca_coords)
    r2_cell = float(reg_cell.score(cell_int, pca_coords))

    # Variance explained by TF identity (one-hot)
    unique_tfs = np.unique(tf_labels)
    tf_onehot = np.zeros((len(tf_labels), len(unique_tfs)))
    for i, tf in enumerate(tf_labels):
        idx = np.where(unique_tfs == tf)[0][0]
        tf_onehot[i, idx] = 1.0

    reg_tf = LinearRegression().fit(tf_onehot, pca_coords)
    r2_tf = float(reg_tf.score(tf_onehot, pca_coords))

    # Combined
    combined = np.hstack([cell_int, tf_onehot])
    reg_combined = LinearRegression().fit(combined, pca_coords)
    r2_combined = float(reg_combined.score(combined, pca_coords))

    return {
        'r2_cell_type': r2_cell,
        'r2_tf_identity': r2_tf,
        'r2_combined': r2_combined,
        'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
        'n_k562': len(k562_embeddings),
        'n_hepg2': len(hepg2_embeddings),
        'n_shared_tfs': len(set(k562_tf_labels) & set(hepg2_tf_labels)),
    }
