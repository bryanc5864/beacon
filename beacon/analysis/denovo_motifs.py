"""
De novo motif discovery using gradient-based attribution.

Improves on attention-based motif extraction (0.574 mean JASPAR Pearson r)
by using gradient x input to identify sequence-level importance.
"""

import numpy as np
import torch
from .motif_extraction import PerSlotMotifExtractor


def compute_per_tf_gradient_importance(model, sequences, tf_indices, device,
                                       max_per_tf=50, batch_size=4):
    """Compute per-TF gradient x input importance maps.

    For each TF, selects samples of that TF and computes gradient x input
    w.r.t. the sum of the profile output.

    Args:
        model: BEACON model.
        sequences: [N, L, 4] one-hot sequences.
        tf_indices: [N] TF index per sample.
        device: Torch device.
        max_per_tf: Max samples per TF for gradient computation.
        batch_size: Batch size for gradient computation.

    Returns:
        Dict mapping tf_idx to {
            'importance': [M, L, 4] gradient x input,
            'sequences': [M, L, 4] corresponding sequences,
            'sample_indices': list of original indices,
        }
    """
    model.eval()
    unique_tfs = np.unique(tf_indices)
    result = {}

    for tf_idx in unique_tfs:
        tf_mask = tf_indices == tf_idx
        tf_sample_indices = np.where(tf_mask)[0][:max_per_tf]
        tf_seqs = sequences[tf_sample_indices]

        all_grads = []
        for i in range(0, len(tf_seqs), batch_size):
            batch_np = tf_seqs[i:i + batch_size]
            x = torch.tensor(batch_np, dtype=torch.float32, device=device)
            x.requires_grad_(True)

            outputs = model(x, return_attention=False, return_slots=False)
            scalar = outputs['profile'].sum()
            scalar.backward()

            grad = x.grad.detach().cpu().numpy()
            grad_x_input = grad * batch_np
            all_grads.append(grad_x_input)
            model.zero_grad()

        importance = np.concatenate(all_grads, axis=0)
        result[int(tf_idx)] = {
            'importance': importance,
            'sequences': tf_seqs,
            'sample_indices': tf_sample_indices.tolist(),
        }

    return result


def extract_denovo_pwm(importance_maps, sequences, window=20, n_motifs=1,
                       percentile=95.0):
    """Extract de novo PWMs from gradient importance maps.

    Uses peak detection on averaged importance to find motif positions,
    then builds PFM from aligned sequence windows.

    Args:
        importance_maps: [M, L, 4] gradient x input for samples of one TF.
        sequences: [M, L, 4] corresponding one-hot sequences.
        window: Window size for motif extraction.
        n_motifs: Number of motifs to extract (uses top peaks).
        percentile: Threshold percentile for peak detection.

    Returns:
        List of dicts, each with 'pfm', 'cwm', 'info_content', 'n_seqlets'.
    """
    # Average importance across samples
    avg_importance = importance_maps.mean(axis=0)  # [L, 4]

    extractor = PerSlotMotifExtractor(
        window_size=window,
        percentile_threshold=percentile,
        min_seqlets=3,
        trim_ic_threshold=0.2,
        top_k_peaks=max(n_motifs * 3, 10),
    )

    motif = extractor.extract_from_importance(
        importance=avg_importance,
        sequences=sequences,
        slot_id=0,
        tf_name=None,
    )

    if motif.pfm is None:
        return []

    return [{
        'pfm': motif.pfm,
        'cwm': motif.cwm,
        'info_content': motif.info_content,
        'n_seqlets': motif.n_seqlets,
        'motif_length': len(motif.pfm),
    }]


def compare_to_jaspar(discovered_pwm, jaspar_pwm):
    """Compare discovered PWM to JASPAR reference via Pearson correlation.

    Slides the shorter PWM along the longer one to find the best alignment,
    similar to TOMTOM.

    Args:
        discovered_pwm: [M1, 4] numpy PFM.
        jaspar_pwm: [M2, 4] numpy PFM.

    Returns:
        Dict with best_pearson, best_offset, aligned_length.
    """
    if discovered_pwm is None or jaspar_pwm is None:
        return {'best_pearson': 0.0, 'best_offset': 0, 'aligned_length': 0}

    len_d = len(discovered_pwm)
    len_j = len(jaspar_pwm)

    best_r = -1.0
    best_offset = 0

    # Slide discovered along jaspar and vice versa
    for offset in range(-len_d + 4, len_j - 3):
        # Overlapping region
        d_start = max(0, -offset)
        d_end = min(len_d, len_j - offset)
        j_start = max(0, offset)
        j_end = min(len_j, len_d + offset)

        overlap = d_end - d_start
        if overlap < 4:
            continue

        d_slice = discovered_pwm[d_start:d_end].flatten()
        j_slice = jaspar_pwm[j_start:j_end].flatten()

        if np.std(d_slice) > 0 and np.std(j_slice) > 0:
            r = np.corrcoef(d_slice, j_slice)[0, 1]
            if not np.isnan(r) and r > best_r:
                best_r = r
                best_offset = offset

    # Also try reverse complement
    rc_discovered = discovered_pwm[::-1, ::-1].copy()
    for offset in range(-len_d + 4, len_j - 3):
        d_start = max(0, -offset)
        d_end = min(len_d, len_j - offset)
        j_start = max(0, offset)
        j_end = min(len_j, len_d + offset)

        overlap = d_end - d_start
        if overlap < 4:
            continue

        d_slice = rc_discovered[d_start:d_end].flatten()
        j_slice = jaspar_pwm[j_start:j_end].flatten()

        if np.std(d_slice) > 0 and np.std(j_slice) > 0:
            r = np.corrcoef(d_slice, j_slice)[0, 1]
            if not np.isnan(r) and r > best_r:
                best_r = r
                best_offset = offset

    return {
        'best_pearson': float(max(best_r, 0.0)),
        'best_offset': int(best_offset),
        'discovered_length': len_d,
        'jaspar_length': len_j,
    }


def full_motif_discovery_pipeline(model, sequences, tf_indices, tf_names, device,
                                  jaspar_pwms=None, max_per_tf=50):
    """End-to-end gradient-based motif discovery and JASPAR comparison.

    Args:
        model: BEACON model.
        sequences: [N, L, 4] one-hot sequences.
        tf_indices: [N] TF index per sample.
        tf_names: List of TF name strings.
        device: Torch device.
        jaspar_pwms: Dict mapping TF name to [M, 4] JASPAR PFM.
        max_per_tf: Max samples per TF.

    Returns:
        Dict with per-TF motif discovery results and JASPAR comparisons.
    """
    print("  Computing per-TF gradient importance...")
    per_tf_data = compute_per_tf_gradient_importance(
        model, sequences, tf_indices, device, max_per_tf=max_per_tf
    )

    results = {}
    pearson_scores = []

    for tf_idx, tf_name in enumerate(tf_names):
        if tf_idx not in per_tf_data:
            continue

        data = per_tf_data[tf_idx]
        print(f"  {tf_name}: extracting motifs from {len(data['sequences'])} samples...")

        motifs = extract_denovo_pwm(
            data['importance'], data['sequences'],
            window=20, n_motifs=1,
        )

        tf_result = {
            'n_samples': len(data['sequences']),
            'n_motifs_found': len(motifs),
        }

        if motifs:
            best = motifs[0]
            tf_result['info_content'] = best['info_content']
            tf_result['motif_length'] = best['motif_length']
            tf_result['n_seqlets'] = best['n_seqlets']

            if jaspar_pwms and tf_name in jaspar_pwms:
                comparison = compare_to_jaspar(best['pfm'], jaspar_pwms[tf_name])
                tf_result['jaspar_pearson'] = comparison['best_pearson']
                tf_result['jaspar_offset'] = comparison['best_offset']
                pearson_scores.append(comparison['best_pearson'])
                print(f"    JASPAR Pearson r = {comparison['best_pearson']:.3f}")
            else:
                tf_result['jaspar_pearson'] = None
        else:
            tf_result['info_content'] = 0.0
            tf_result['jaspar_pearson'] = None

        results[tf_name] = tf_result

    summary = {
        'per_tf': results,
        'mean_jaspar_pearson': float(np.mean(pearson_scores)) if pearson_scores else None,
        'median_jaspar_pearson': float(np.median(pearson_scores)) if pearson_scores else None,
        'n_tfs_with_motifs': sum(1 for r in results.values() if r['n_motifs_found'] > 0),
        'n_tfs_compared': len(pearson_scores),
    }
    return summary
