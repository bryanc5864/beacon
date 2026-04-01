"""
Profile-level evaluation metrics for BEACON.

Provides Jensen-Shannon Divergence, Multinomial Negative Log-Likelihood,
and AUPRC for site detection — standard metrics used by BPNet/ChromBPNet.
"""

import numpy as np
from scipy.special import rel_entr
from sklearn.metrics import average_precision_score


def _to_prob(profile, pseudocount=1e-6):
    """Normalize a profile to a probability distribution."""
    p = np.maximum(profile, 0) + pseudocount
    return p / p.sum()


def jensen_shannon_divergence(pred, target):
    """Jensen-Shannon Divergence between two profiles.

    Lower is better.  JSD ∈ [0, ln(2)].
    """
    p = _to_prob(pred)
    q = _to_prob(target)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))


def multinomial_nll(pred, target):
    """Multinomial negative log-likelihood.

    Treats the target as counts drawn from a multinomial whose
    probabilities are given by the normalized prediction.
    """
    pred_prob = _to_prob(pred)
    counts = np.maximum(target, 0)
    total = counts.sum()
    if total == 0:
        return 0.0
    frac = counts / (total + 1e-10)
    return float(-np.sum(frac * np.log(pred_prob + 1e-10)))


def per_sample_jsd(pred_profiles, gt_profiles):
    """Compute per-sample JSD.  Returns array of length N."""
    return np.array([jensen_shannon_divergence(pred_profiles[i], gt_profiles[i])
                     for i in range(len(pred_profiles))])


def per_sample_mnll(pred_profiles, gt_profiles):
    """Compute per-sample MNLL.  Returns array of length N."""
    return np.array([multinomial_nll(pred_profiles[i], gt_profiles[i])
                     for i in range(len(pred_profiles))])


def site_auprc(pred_profiles, gt_binding_sites, seq_len=2000, resolutions=(1, 10, 50)):
    """AUPRC for binding-site detection at multiple bin resolutions.

    Args:
        pred_profiles: [N, L] predicted profiles.
        gt_binding_sites: [N, K, 3] (position, tf_id, occupancy).
        seq_len: Sequence length.
        resolutions: Tuple of bin sizes in bp.

    Returns:
        dict mapping resolution label (e.g. '10bp') to
        {'auprc': float, 'n_positive': int, 'n_total': int}.
    """
    results = {}
    for res in resolutions:
        n_bins = seq_len // res
        all_gt, all_pred = [], []
        for i in range(len(pred_profiles)):
            gt_binary = np.zeros(n_bins)
            for s in range(gt_binding_sites.shape[1]):
                pos, _, occ = gt_binding_sites[i, s]
                if occ > 0.5:
                    center = int(pos * seq_len) // res
                    for off in range(-1, 2):
                        idx = center + off
                        if 0 <= idx < n_bins:
                            gt_binary[idx] = 1
            pred = pred_profiles[i].flatten()
            pred_binned = np.add.reduceat(
                pred, np.arange(0, seq_len, res)
            )[:n_bins]
            all_gt.append(gt_binary)
            all_pred.append(pred_binned)

        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)
        auprc = float(average_precision_score(all_gt, all_pred)) if all_gt.sum() > 0 else 0.0
        results[f"{res}bp"] = {
            "auprc": auprc,
            "n_positive": int(all_gt.sum()),
            "n_total": len(all_gt),
        }
    return results
