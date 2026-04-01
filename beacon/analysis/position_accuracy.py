"""
Position prediction accuracy analysis for BEACON.

Quantifies how accurately BEACON predicts binding site positions —
a unique capability compared to profile-only models like BPNet.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .statistical import bootstrap_ci


def hungarian_position_errors(outputs, binding_sites):
    """Compute position errors (bp) after Hungarian matching.

    For each sample, match predicted slots to ground-truth sites via
    Hungarian algorithm, then compute the distance between predicted
    position mean and ground-truth site center.

    Args:
        outputs: Dict from run_model_batch with 'positions', 'occupancy', 'tf_logits'.
        binding_sites: [N, K, 3] ground truth (normalized_pos, tf_id, occupancy).

    Returns:
        Dict with:
          - errors: list of signed errors (pred - gt, in bp)
          - abs_errors: list of absolute errors (bp)
          - per_sample: list of dicts per matched site with gt_tf, pred_tf, etc.
    """
    pred_positions = outputs['positions']
    pred_tf_logits = outputs['tf_logits']
    pred_occupancy = outputs['occupancy']

    errors = []
    abs_errors = []
    per_sample = []

    for b in range(len(pred_positions)):
        gt_sites = []
        for s in range(binding_sites.shape[1]):
            pos, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_sites.append((float(pos), int(tf_id), float(occ)))
        if not gt_sites:
            continue

        active_mask = pred_occupancy[b] > 0.3
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            continue

        n_gt = len(gt_sites)
        n_pred = len(active_indices)
        cost_matrix = np.zeros((n_gt, n_pred))

        for i, (gt_pos, gt_tf, _) in enumerate(gt_sites):
            for j, pred_idx in enumerate(active_indices):
                pos_val = pred_positions[b, pred_idx]
                pred_pos = pos_val[0] if pos_val.shape[0] > 1 else pos_val.flat[0]
                pred_tf = int(pred_tf_logits[b, pred_idx].argmax())
                pos_dist = abs(gt_pos * 2000 - pred_pos * 2000)
                cost_matrix[i, j] = pos_dist - (1.0 if pred_tf == gt_tf else 0.0) * 500

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            gt_pos, gt_tf, _ = gt_sites[i]
            pred_idx = active_indices[j]
            pos_val = pred_positions[b, pred_idx]
            pred_pos = pos_val[0] if pos_val.shape[0] > 1 else pos_val.flat[0]
            pred_tf = int(pred_tf_logits[b, pred_idx].argmax())

            gt_bp = gt_pos * 2000
            pred_bp = pred_pos * 2000
            signed_err = float(pred_bp - gt_bp)
            abs_err = abs(signed_err)

            if abs_err < 500:  # only count reasonable matches
                errors.append(signed_err)
                abs_errors.append(abs_err)
                per_sample.append({
                    'sample_idx': int(b),
                    'gt_tf': int(gt_tf),
                    'pred_tf': int(pred_tf),
                    'gt_pos_bp': float(gt_bp),
                    'pred_pos_bp': float(pred_bp),
                    'signed_error': signed_err,
                    'abs_error': abs_err,
                    'tf_correct': int(pred_tf == gt_tf),
                })

    return {
        'errors': errors,
        'abs_errors': abs_errors,
        'per_sample': per_sample,
    }


def per_tf_position_stats(errors_dict, tf_names):
    """Group position errors by TF and compute per-TF statistics.

    Args:
        errors_dict: Output of hungarian_position_errors.
        tf_names: List of TF name strings.

    Returns:
        Dict mapping TF name to {mae, medae, mae_ci, signed_mean, n}.
    """
    per_tf = {}
    for tf_idx, tf_name in enumerate(tf_names):
        tf_errors = [s['abs_error'] for s in errors_dict['per_sample']
                     if s['gt_tf'] == tf_idx]
        tf_signed = [s['signed_error'] for s in errors_dict['per_sample']
                     if s['gt_tf'] == tf_idx]
        if len(tf_errors) < 3:
            per_tf[tf_name] = {'n': len(tf_errors), 'mae': None, 'medae': None}
            continue

        tf_errors = np.array(tf_errors)
        tf_signed = np.array(tf_signed)
        per_tf[tf_name] = {
            'n': len(tf_errors),
            'mae': float(np.mean(tf_errors)),
            'medae': float(np.median(tf_errors)),
            'mae_ci': bootstrap_ci(tf_errors),
            'medae_ci': bootstrap_ci(tf_errors, statistic=np.median),
            'signed_mean': float(np.mean(tf_signed)),
            'signed_std': float(np.std(tf_signed)),
        }
    return per_tf


def position_error_by_signal(errors_dict, profiles):
    """Stratify position error by profile signal strength quartiles.

    Args:
        errors_dict: Output of hungarian_position_errors.
        profiles: [N, L] ground-truth profiles.

    Returns:
        Dict mapping quartile label to {n, mae, medae}.
    """
    sample_indices = [s['sample_idx'] for s in errors_dict['per_sample']]
    abs_errors = [s['abs_error'] for s in errors_dict['per_sample']]

    if not sample_indices:
        return {}

    signal_strengths = np.array([profiles[idx].sum() for idx in sample_indices])
    quartiles = np.percentile(signal_strengths, [25, 50, 75])
    labels = ['Q1 (weak)', 'Q2', 'Q3', 'Q4 (strong)']

    result = {}
    for q, label in enumerate(labels):
        if q == 0:
            mask = signal_strengths <= quartiles[0]
        elif q == 3:
            mask = signal_strengths > quartiles[2]
        else:
            mask = (signal_strengths > quartiles[q - 1]) & (signal_strengths <= quartiles[q])

        q_errors = np.array([e for e, m in zip(abs_errors, mask) if m])
        if len(q_errors) > 0:
            result[label] = {
                'n': len(q_errors),
                'mae': float(np.mean(q_errors)),
                'medae': float(np.median(q_errors)),
            }
    return result


def position_error_by_gc(errors_dict, sequences):
    """Stratify position error by GC content quartiles.

    Args:
        errors_dict: Output of hungarian_position_errors.
        sequences: [N, L, 4] one-hot sequences (A, C, G, T order).

    Returns:
        Dict mapping quartile label to {n, mae, medae, gc_range}.
    """
    sample_indices = [s['sample_idx'] for s in errors_dict['per_sample']]
    abs_errors = [s['abs_error'] for s in errors_dict['per_sample']]

    if not sample_indices:
        return {}

    gc_vals = np.array([(sequences[idx, :, 1] + sequences[idx, :, 2]).mean()
                        for idx in sample_indices])
    quartiles = np.percentile(gc_vals, [25, 50, 75])
    labels = ['Q1 (low GC)', 'Q2', 'Q3', 'Q4 (high GC)']

    result = {}
    for q, label in enumerate(labels):
        if q == 0:
            mask = gc_vals <= quartiles[0]
        elif q == 3:
            mask = gc_vals > quartiles[2]
        else:
            mask = (gc_vals > quartiles[q - 1]) & (gc_vals <= quartiles[q])

        q_errors = np.array([e for e, m in zip(abs_errors, mask) if m])
        q_gc = gc_vals[mask]
        if len(q_errors) > 0:
            result[label] = {
                'n': len(q_errors),
                'mae': float(np.mean(q_errors)),
                'medae': float(np.median(q_errors)),
                'gc_range': [float(q_gc.min()), float(q_gc.max())],
            }
    return result


def systematic_offset_analysis(errors_dict, tf_names):
    """Check for systematic position bias per TF.

    Tests whether the mean signed error significantly differs from zero
    using bootstrap CIs.

    Args:
        errors_dict: Output of hungarian_position_errors.
        tf_names: List of TF name strings.

    Returns:
        Dict mapping TF name to {mean_signed_error, ci, is_biased, n}.
    """
    result = {}
    for tf_idx, tf_name in enumerate(tf_names):
        signed = np.array([s['signed_error'] for s in errors_dict['per_sample']
                           if s['gt_tf'] == tf_idx])
        if len(signed) < 5:
            result[tf_name] = {'n': len(signed), 'mean_signed_error': None}
            continue

        ci = bootstrap_ci(signed)
        is_biased = (ci['ci_lower'] > 0) or (ci['ci_upper'] < 0)
        result[tf_name] = {
            'n': len(signed),
            'mean_signed_error': float(np.mean(signed)),
            'std_signed_error': float(np.std(signed)),
            'ci': ci,
            'is_biased': bool(is_biased),
        }
    return result
