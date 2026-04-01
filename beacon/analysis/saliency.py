"""
Gradient-based saliency analysis for BEACON.

Validates that the model attends to biologically meaningful sequence
features by computing gradient x input saliency and comparing with
attention weights.
"""

import numpy as np
import torch


def compute_gradient_x_input(model, sequences, device, target='profile', batch_size=8):
    """Compute gradient x input saliency maps.

    Args:
        model: BEACON model (will be set to eval mode).
        sequences: [N, L, 4] numpy one-hot sequences.
        device: Torch device.
        target: 'profile' (sum of profile) or 'occupancy' (sum of occupied slot occupancies).
        batch_size: Batch size for gradient computation.

    Returns:
        [N, L, 4] numpy array of gradient x input values.
    """
    model.eval()
    all_grads = []

    for i in range(0, len(sequences), batch_size):
        batch_np = sequences[i:i + batch_size]
        x = torch.tensor(batch_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)

        outputs = model(x, return_attention=False, return_slots=False)

        if target == 'profile':
            scalar = outputs['profile'].sum()
        elif target == 'occupancy':
            occ = outputs['occupancy']
            if occ.ndim == 3:
                occ = occ.squeeze(-1)
            scalar = occ[occ > 0.3].sum()
        else:
            raise ValueError(f"Unknown target: {target}")

        scalar.backward()
        grad = x.grad.detach().cpu().numpy()
        grad_x_input = grad * batch_np
        all_grads.append(grad_x_input)

        model.zero_grad()

    return np.concatenate(all_grads, axis=0)


def attention_vs_gradient_correlation(attention, grad_importance):
    """Compute per-sample correlation between attention and gradient importance.

    Args:
        attention: [N, K, L] attention weights from run_model_batch.
        grad_importance: [N, L, 4] gradient x input saliency.

    Returns:
        Dict with correlations array and summary statistics.
    """
    n_samples = len(attention)
    # Max attention over slots: [N, L]
    max_attention = attention.max(axis=1)
    # Sum gradient magnitude over channels: [N, L]
    grad_mag = np.abs(grad_importance).sum(axis=-1)

    correlations = np.zeros(n_samples)
    for i in range(n_samples):
        a = max_attention[i]
        g = grad_mag[i]
        if np.std(a) > 0 and np.std(g) > 0:
            r = np.corrcoef(a, g)[0, 1]
            correlations[i] = r if not np.isnan(r) else 0.0

    return {
        'correlations': correlations.tolist(),
        'mean': float(np.mean(correlations)),
        'std': float(np.std(correlations)),
        'median': float(np.median(correlations)),
        'q25': float(np.percentile(correlations, 25)),
        'q75': float(np.percentile(correlations, 75)),
        'n_positive': int((correlations > 0).sum()),
        'n_samples': n_samples,
    }


def gradient_enrichment_at_motifs(grad_importance, sequences, motif_positions, window=50):
    """Compare gradient magnitude at motif positions vs flanking/random.

    Args:
        grad_importance: [N, L, 4] gradient x input saliency.
        sequences: [N, L, 4] one-hot sequences.
        motif_positions: Dict mapping sample_idx to list of (start, end) tuples
            for known motif locations in that sample.
        window: Size of flanking region to compare (bp on each side).

    Returns:
        Dict with motif_mean, flanking_mean, random_mean, enrichment_ratio.
    """
    grad_mag = np.abs(grad_importance).sum(axis=-1)  # [N, L]
    seq_len = grad_mag.shape[1]

    motif_vals = []
    flanking_vals = []
    random_vals = []

    rng = np.random.RandomState(42)

    for sample_idx, positions in motif_positions.items():
        if sample_idx >= len(grad_mag):
            continue
        sample_grad = grad_mag[sample_idx]
        motif_mask = np.zeros(seq_len, dtype=bool)

        for start, end in positions:
            start, end = max(0, start), min(seq_len, end)
            motif_vals.extend(sample_grad[start:end].tolist())
            motif_mask[start:end] = True

            # Flanking regions
            flank_left = max(0, start - window)
            flank_right = min(seq_len, end + window)
            for p in range(flank_left, start):
                if not motif_mask[p]:
                    flanking_vals.append(float(sample_grad[p]))
            for p in range(end, flank_right):
                if p < seq_len and not motif_mask[p]:
                    flanking_vals.append(float(sample_grad[p]))

        # Random positions (same count as motif)
        n_motif = int(motif_mask.sum())
        non_motif = np.where(~motif_mask)[0]
        if len(non_motif) > 0 and n_motif > 0:
            rand_idx = rng.choice(non_motif, size=min(n_motif, len(non_motif)), replace=False)
            random_vals.extend(sample_grad[rand_idx].tolist())

    motif_mean = float(np.mean(motif_vals)) if motif_vals else 0.0
    flanking_mean = float(np.mean(flanking_vals)) if flanking_vals else 0.0
    random_mean = float(np.mean(random_vals)) if random_vals else 0.0

    return {
        'motif_mean': motif_mean,
        'flanking_mean': flanking_mean,
        'random_mean': random_mean,
        'motif_vs_flanking_ratio': motif_mean / (flanking_mean + 1e-8),
        'motif_vs_random_ratio': motif_mean / (random_mean + 1e-8),
        'n_motif_positions': len(motif_vals),
        'n_flanking_positions': len(flanking_vals),
        'n_random_positions': len(random_vals),
    }
