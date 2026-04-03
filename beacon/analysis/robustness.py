"""
Robustness analysis for BEACON.

Tests how prediction quality degrades under sequence perturbation,
and compares BEACON vs BPNet robustness.
"""

import numpy as np


def perturb_sequences(sequences, mutation_rate, seed=42):
    """Randomly mutate a fraction of positions in each sequence.

    Args:
        sequences: [N, L, 4] one-hot sequences.
        mutation_rate: Fraction of positions to mutate (0.0-1.0).
        seed: Random seed.

    Returns:
        [N, L, 4] perturbed sequences.
    """
    rng = np.random.RandomState(seed)
    perturbed = sequences.copy()
    n_samples, seq_len, _ = sequences.shape
    n_mutate = int(seq_len * mutation_rate)

    for i in range(n_samples):
        positions = rng.choice(seq_len, size=n_mutate, replace=False)
        for p in positions:
            perturbed[i, p] = 0.0
            perturbed[i, p, rng.randint(0, 4)] = 1.0

    return perturbed


def robustness_curve(model, sequences, profiles, device, mutation_rates,
                     run_model_batch_fn, per_sample_pearson_fn, seed=42):
    """Compute profile correlation at varying mutation rates.

    Args:
        model: Model in eval mode.
        sequences: [N, L, 4] original sequences.
        profiles: [N, L] ground-truth profiles.
        device: Torch device.
        mutation_rates: List of mutation rates to test.
        run_model_batch_fn: Function to run inference.
        per_sample_pearson_fn: Function to compute per-sample Pearson r.
        seed: Random seed.

    Returns:
        Dict mapping mutation rate to {mean_r, std_r, delta_from_clean}.
    """
    # Clean baseline
    clean_outputs = run_model_batch_fn(model, sequences, device)
    clean_r = per_sample_pearson_fn(clean_outputs['profile'], profiles)
    clean_mean = float(np.mean(clean_r))

    results = {
        '0.0': {
            'mean_r': clean_mean,
            'std_r': float(np.std(clean_r)),
            'delta': 0.0,
            'relative_retained': 1.0,
        }
    }

    for rate in mutation_rates:
        if rate == 0.0:
            continue
        perturbed = perturb_sequences(sequences, rate, seed=seed)
        mut_outputs = run_model_batch_fn(model, perturbed, device)
        mut_r = per_sample_pearson_fn(mut_outputs['profile'], profiles)
        mut_mean = float(np.mean(mut_r))

        results[str(rate)] = {
            'mean_r': mut_mean,
            'std_r': float(np.std(mut_r)),
            'delta': mut_mean - clean_mean,
            'relative_retained': mut_mean / (clean_mean + 1e-10),
        }

    return results


def multi_site_synthetic_test(model, sequences, binding_sites, device,
                              run_model_batch_fn, n_pairs=100, seed=42):
    """Test BEACON's multi-site capability by combining two single-site samples.

    Creates synthetic 2-site inputs by averaging profiles from two samples
    with different TFs, then checks if BEACON activates 2 slots.

    Args:
        model: BEACON model.
        sequences: [N, L, 4] single-site sequences.
        binding_sites: [N, K, 3] binding site info.
        device: Torch device.
        run_model_batch_fn: Inference function.
        n_pairs: Number of pairs to test.
        seed: Random seed.

    Returns:
        Dict with multi-site detection metrics.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(sequences)

    # Group samples by TF
    tf_groups = {}
    for i in range(n_samples):
        for s in range(binding_sites.shape[1]):
            _, tf_id, occ = binding_sites[i, s]
            if occ > 0.5:
                tf_id = int(tf_id)
                if tf_id not in tf_groups:
                    tf_groups[tf_id] = []
                tf_groups[tf_id].append(i)
                break

    tf_ids = list(tf_groups.keys())
    if len(tf_ids) < 2:
        return {'error': 'need at least 2 TFs'}

    # Create synthetic 2-site sequences by combining two single-site sequences
    # Use a simple approach: take sequence from sample A but add binding signal from B
    # In practice, we just feed different sequences and check slot behavior
    combined_seqs = []
    pair_info = []

    for _ in range(n_pairs):
        tf_a, tf_b = rng.choice(tf_ids, size=2, replace=False)
        idx_a = rng.choice(tf_groups[tf_a])
        idx_b = rng.choice(tf_groups[tf_b])

        # Average the one-hot sequences (creates ambiguous positions)
        combined = (sequences[idx_a] + sequences[idx_b]) / 2.0
        combined_seqs.append(combined)
        pair_info.append({'tf_a': int(tf_a), 'tf_b': int(tf_b),
                          'idx_a': int(idx_a), 'idx_b': int(idx_b)})

    combined_seqs = np.array(combined_seqs)

    # Run BEACON on combined sequences
    outputs = run_model_batch_fn(model, combined_seqs, device)
    occupancy = outputs['occupancy']  # [N, K]
    tf_logits = outputs['tf_logits']  # [N, K, T]

    n_active = (occupancy > 0.3).sum(axis=1)  # [N]
    n_two_plus = int((n_active >= 2).sum())

    # Check if both TFs are predicted
    both_detected = 0
    for i in range(len(pair_info)):
        tf_a = pair_info[i]['tf_a']
        tf_b = pair_info[i]['tf_b']
        active_mask = occupancy[i] > 0.3
        pred_tfs = set(tf_logits[i, active_mask].argmax(axis=1).tolist())
        if tf_a in pred_tfs and tf_b in pred_tfs:
            both_detected += 1

    # Compare to single-site baseline
    single_seqs_a = np.array([sequences[p['idx_a']] for p in pair_info])
    single_outputs = run_model_batch_fn(model, single_seqs_a, device)
    single_active = (single_outputs['occupancy'] > 0.3).sum(axis=1)

    return {
        'n_pairs': n_pairs,
        'mean_active_slots_combined': float(np.mean(n_active)),
        'mean_active_slots_single': float(np.mean(single_active)),
        'two_plus_slots_rate': float(n_two_plus / n_pairs),
        'both_tfs_detected_rate': float(both_detected / n_pairs),
        'active_slot_distribution': {
            str(k): int((n_active == k).sum()) for k in range(int(n_active.max()) + 1)
        },
    }
