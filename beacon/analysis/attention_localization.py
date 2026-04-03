"""
Attention localization analysis for BEACON.

Measures whether attention peaks actually localize to binding sites —
the core interpretability claim of slot attention.
"""

import numpy as np
from .statistical import bootstrap_ci


def attention_peak_at_binding_site(outputs, binding_sites, window_bp=50):
    """Measure how well attention peaks align with ground-truth binding sites.

    For each sample, finds the position of max attention (per active slot)
    and computes the distance to the nearest ground-truth binding site.

    Args:
        outputs: Dict from run_model_batch with 'attention' [N,K,L], 'occupancy' [N,K].
        binding_sites: [N, K_gt, 3] ground truth (norm_pos, tf_id, occupancy).
        window_bp: Window (bp) to consider a "hit".

    Returns:
        Dict with localization metrics.
    """
    attention = outputs['attention']  # [N, K, L]
    occupancy = outputs['occupancy']  # [N, K]
    seq_len = attention.shape[2]

    distances = []
    within_window = []
    attention_at_site = []
    attention_at_random = []
    rng = np.random.RandomState(42)

    for b in range(len(attention)):
        # Get ground-truth site positions
        gt_positions_bp = []
        for s in range(binding_sites.shape[1]):
            pos, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_positions_bp.append(float(pos) * seq_len)

        if not gt_positions_bp:
            continue

        # For each active slot, find attention peak
        for k in range(occupancy.shape[1]):
            if occupancy[b, k] <= 0.3:
                continue

            attn = attention[b, k]  # [L]
            peak_pos = int(np.argmax(attn))

            # Distance to nearest GT site
            min_dist = min(abs(peak_pos - gt) for gt in gt_positions_bp)
            distances.append(float(min_dist))
            within_window.append(min_dist <= window_bp)

            # Attention concentration at GT sites vs random
            for gt_bp in gt_positions_bp:
                gt_int = int(gt_bp)
                lo = max(0, gt_int - window_bp)
                hi = min(seq_len, gt_int + window_bp)
                attention_at_site.append(float(attn[lo:hi].sum()))

                # Random control (same window size)
                rand_pos = rng.randint(window_bp, seq_len - window_bp)
                attention_at_random.append(float(attn[rand_pos - window_bp:rand_pos + window_bp].sum()))

    if not distances:
        return {'error': 'no matched sites'}

    distances = np.array(distances)
    site_attn = np.array(attention_at_site)
    rand_attn = np.array(attention_at_random)

    return {
        'mean_distance_bp': float(np.mean(distances)),
        'median_distance_bp': float(np.median(distances)),
        'distance_ci': bootstrap_ci(distances),
        'within_50bp': float(np.mean(within_window)),
        'within_100bp': float(np.mean(distances <= 100)),
        'within_200bp': float(np.mean(distances <= 200)),
        'attention_at_site_mean': float(site_attn.mean()),
        'attention_at_random_mean': float(rand_attn.mean()),
        'attention_enrichment': float(site_attn.mean() / (rand_attn.mean() + 1e-10)),
        'n_sites': len(distances),
    }


def per_tf_attention_localization(outputs, binding_sites, tf_names, window_bp=50):
    """Per-TF attention localization analysis.

    Args:
        outputs: Dict with 'attention', 'occupancy', 'tf_logits'.
        binding_sites: [N, K_gt, 3].
        tf_names: List of TF name strings.
        window_bp: Window for hit detection.

    Returns:
        Dict per TF with localization metrics.
    """
    attention = outputs['attention']
    occupancy = outputs['occupancy']
    tf_logits = outputs['tf_logits']
    seq_len = attention.shape[2]

    per_tf = {tf: {'distances': [], 'within_window': []} for tf in tf_names}

    for b in range(len(attention)):
        gt_sites = []
        for s in range(binding_sites.shape[1]):
            pos, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_sites.append((float(pos) * seq_len, int(tf_id)))

        if not gt_sites:
            continue

        for k in range(occupancy.shape[1]):
            if occupancy[b, k] <= 0.3:
                continue

            attn = attention[b, k]
            peak_pos = int(np.argmax(attn))
            pred_tf = int(tf_logits[b, k].argmax())

            # Match to nearest GT site
            min_dist = float('inf')
            matched_tf = -1
            for gt_bp, gt_tf in gt_sites:
                d = abs(peak_pos - gt_bp)
                if d < min_dist:
                    min_dist = d
                    matched_tf = gt_tf

            if matched_tf >= 0 and matched_tf < len(tf_names):
                tf_name = tf_names[matched_tf]
                per_tf[tf_name]['distances'].append(float(min_dist))
                per_tf[tf_name]['within_window'].append(min_dist <= window_bp)

    result = {}
    for tf_name, data in per_tf.items():
        if len(data['distances']) < 3:
            result[tf_name] = {'n': len(data['distances'])}
            continue
        dists = np.array(data['distances'])
        result[tf_name] = {
            'n': len(dists),
            'mean_distance_bp': float(np.mean(dists)),
            'median_distance_bp': float(np.median(dists)),
            'within_50bp': float(np.mean(data['within_window'])),
            'within_100bp': float(np.mean(dists <= 100)),
        }
    return result


def attention_entropy_analysis(outputs, binding_sites, tf_names):
    """Analyze attention entropy: concentrated (interpretable) vs diffuse.

    Args:
        outputs: Dict with 'attention', 'occupancy'.
        binding_sites: [N, K_gt, 3].
        tf_names: List of TF name strings.

    Returns:
        Dict with entropy statistics.
    """
    attention = outputs['attention']  # [N, K, L]
    occupancy = outputs['occupancy']  # [N, K]
    seq_len = attention.shape[2]
    max_entropy = np.log(seq_len)

    entropies = []
    per_tf_entropy = {tf: [] for tf in tf_names}

    for b in range(len(attention)):
        gt_tf = -1
        for s in range(binding_sites.shape[1]):
            _, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_tf = int(tf_id)
                break

        for k in range(occupancy.shape[1]):
            if occupancy[b, k] <= 0.3:
                continue
            attn = attention[b, k]
            attn_norm = attn / (attn.sum() + 1e-10)
            ent = -np.sum(attn_norm * np.log(attn_norm + 1e-10))
            entropies.append(float(ent))

            if 0 <= gt_tf < len(tf_names):
                per_tf_entropy[tf_names[gt_tf]].append(float(ent))

    if not entropies:
        return {'error': 'no active slots'}

    entropies = np.array(entropies)
    result = {
        'mean_entropy': float(np.mean(entropies)),
        'std_entropy': float(np.std(entropies)),
        'max_possible_entropy': float(max_entropy),
        'normalized_entropy': float(np.mean(entropies) / max_entropy),
        'n_slots': len(entropies),
        'per_tf': {},
    }

    for tf, ents in per_tf_entropy.items():
        if len(ents) > 0:
            result['per_tf'][tf] = {
                'mean_entropy': float(np.mean(ents)),
                'normalized': float(np.mean(ents) / max_entropy),
                'n': len(ents),
            }
    return result


def effective_slot_utilization(outputs, tf_names, binding_sites):
    """Analyze how slots are actually used across the dataset.

    Args:
        outputs: Dict with 'occupancy', 'tf_logits'.
        tf_names: List of TF names.
        binding_sites: [N, K_gt, 3].

    Returns:
        Dict with slot utilization metrics.
    """
    occupancy = outputs['occupancy']  # [N, K]
    tf_logits = outputs['tf_logits']  # [N, K, T]
    n_samples, n_slots = occupancy.shape

    # Active slots per sample
    active_per_sample = (occupancy > 0.3).sum(axis=1)

    # Which slots are active across the dataset
    slot_activation_freq = np.zeros(n_slots)
    slot_tf_matrix = np.zeros((n_slots, len(tf_names)))

    for b in range(n_samples):
        gt_tf = -1
        for s in range(binding_sites.shape[1]):
            _, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_tf = int(tf_id)
                break

        for k in range(n_slots):
            if occupancy[b, k] > 0.3:
                slot_activation_freq[k] += 1
                if 0 <= gt_tf < len(tf_names):
                    slot_tf_matrix[k, gt_tf] += 1

    slot_activation_freq /= n_samples

    # Effective number of slots (exponential of entropy)
    freq = slot_activation_freq[slot_activation_freq > 0]
    freq_norm = freq / freq.sum()
    slot_entropy = -np.sum(freq_norm * np.log(freq_norm + 1e-10))
    effective_slots = float(np.exp(slot_entropy))

    # Slot-TF specificity: for each active slot, how specific is it to one TF?
    slot_specificity = []
    for k in range(n_slots):
        if slot_tf_matrix[k].sum() > 0:
            dist = slot_tf_matrix[k] / slot_tf_matrix[k].sum()
            max_frac = float(dist.max())
            slot_specificity.append(max_frac)

    return {
        'mean_active_slots': float(np.mean(active_per_sample)),
        'std_active_slots': float(np.std(active_per_sample)),
        'max_active_slots': int(np.max(active_per_sample)),
        'effective_n_slots': effective_slots,
        'slot_activation_freq': slot_activation_freq.tolist(),
        'n_slots_ever_active': int((slot_activation_freq > 0).sum()),
        'mean_slot_specificity': float(np.mean(slot_specificity)) if slot_specificity else 0.0,
        'slot_tf_matrix': slot_tf_matrix.tolist(),
    }
