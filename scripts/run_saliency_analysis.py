#!/usr/bin/env python3
"""
Gradient-based saliency analysis for BEACON.

Computes gradient x input saliency maps and compares with attention
weights to validate that attention is a faithful explanation.

Usage:
    python scripts/run_saliency_analysis.py --device cuda:3
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_named_model, load_test_data, get_test_path,
    run_model_batch, json_serialize, bootstrap_ci,
    DATASET_CONFIGS, BASE_DIR,
)
from beacon.analysis.saliency import (
    compute_gradient_x_input, attention_vs_gradient_correlation,
    gradient_enrichment_at_motifs,
)

# JASPAR PWMs for motif scanning (reused from enhanced ISM)
JASPAR_PWMS = {
    "CTCF": np.array([
        [0.15,0.40,0.30,0.15],[0.10,0.60,0.15,0.15],[0.15,0.15,0.55,0.15],
        [0.10,0.50,0.30,0.10],[0.15,0.15,0.55,0.15],[0.20,0.20,0.20,0.40],
        [0.15,0.15,0.55,0.15],[0.15,0.15,0.55,0.15],[0.20,0.20,0.20,0.40],
        [0.15,0.15,0.55,0.15],[0.15,0.15,0.55,0.15],[0.10,0.55,0.20,0.15],
        [0.50,0.15,0.20,0.15],[0.15,0.15,0.55,0.15],
    ]),
    "GATA1": np.array([
        [0.55,0.10,0.20,0.15],[0.15,0.10,0.60,0.15],[0.70,0.05,0.15,0.10],
        [0.10,0.10,0.10,0.70],[0.65,0.10,0.10,0.15],[0.55,0.10,0.20,0.15],
    ]),
    "TAL1": np.array([
        [0.15,0.50,0.20,0.15],[0.55,0.15,0.15,0.15],[0.15,0.15,0.55,0.15],
        [0.15,0.50,0.15,0.20],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "MYC": np.array([
        [0.15,0.55,0.15,0.15],[0.60,0.10,0.15,0.15],[0.15,0.55,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "MAX": np.array([
        [0.15,0.55,0.15,0.15],[0.60,0.10,0.15,0.15],[0.15,0.55,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "SPI1": np.array([
        [0.55,0.10,0.15,0.20],[0.55,0.10,0.15,0.20],[0.60,0.10,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.55,0.10,0.15,0.20],[0.15,0.10,0.60,0.15],
        [0.15,0.10,0.60,0.15],[0.55,0.15,0.15,0.15],[0.55,0.15,0.15,0.15],
        [0.15,0.10,0.55,0.20],[0.10,0.10,0.10,0.70],[0.15,0.10,0.60,0.15],
    ]),
    "CEBPB": np.array([
        [0.10,0.10,0.10,0.70],[0.10,0.10,0.10,0.70],[0.15,0.10,0.60,0.15],
        [0.15,0.55,0.15,0.15],[0.15,0.10,0.55,0.20],[0.15,0.55,0.15,0.15],
        [0.60,0.10,0.15,0.15],[0.55,0.15,0.15,0.15],
    ]),
}


def scan_pwm(seq_onehot, pwm, threshold_pct=80):
    """Scan a one-hot sequence with a PWM log-odds matrix."""
    L, M = seq_onehot.shape[0], pwm.shape[0]
    if L < M:
        return []
    log_pwm = np.log2(pwm / 0.25 + 1e-10)
    scores_fwd = np.array([np.sum(seq_onehot[p:p+M] * log_pwm) for p in range(L - M + 1)])
    rc = seq_onehot[::-1, ::-1].copy()
    scores_rev = np.array([np.sum(rc[p:p+M] * log_pwm) for p in range(L - M + 1)])
    scores = np.maximum(scores_fwd, scores_rev)
    pos_scores = scores[scores > 0]
    if len(pos_scores) == 0:
        return []
    thr = np.percentile(pos_scores, threshold_pct)
    hits = [(p, float(scores[p])) for p in range(len(scores)) if scores[p] >= thr]
    hits.sort(key=lambda x: -x[1])
    kept = []
    for p, s in hits:
        if not any(abs(p - kp) < M for kp, _ in kept):
            kept.append((p, s))
    return kept[:5]


def analyze_dataset(name, device, max_samples=500):
    """Run saliency analysis on one dataset."""
    print(f"\n--- {name} ---")
    model, cfg = load_named_model(name, device)
    test_path = get_test_path(name)
    seqs, profs, tf_idx, bs = load_test_data(test_path, max_samples=max_samples)
    tf_names = cfg["tf_names"]

    print(f"  Samples: {len(seqs)}")

    # Get attention weights
    print("  Running model with attention...")
    outputs = run_model_batch(model, seqs, device, return_attention=True)

    # Compute gradient saliency
    print("  Computing gradient x input saliency...")
    grad_importance = compute_gradient_x_input(model, seqs, device, target='profile')

    # Attention vs gradient correlation
    print("  Attention vs gradient correlation...")
    if 'attention' in outputs:
        attn_grad = attention_vs_gradient_correlation(outputs['attention'], grad_importance)
        print(f"    Mean correlation: {attn_grad['mean']:.3f} ± {attn_grad['std']:.3f}")
        print(f"    Positive correlations: {attn_grad['n_positive']}/{attn_grad['n_samples']}")
        attn_grad_ci = bootstrap_ci(np.array(attn_grad['correlations']))
        # Remove raw correlations from output to save space
        attn_grad_summary = {k: v for k, v in attn_grad.items() if k != 'correlations'}
        attn_grad_summary['ci'] = attn_grad_ci
    else:
        attn_grad_summary = {'error': 'no attention weights available'}

    # Gradient enrichment at motif positions
    print("  Scanning for motif positions...")
    motif_positions = {}
    for i in range(len(seqs)):
        ti = tf_idx[i]
        tn = tf_names[ti] if ti < len(tf_names) else None
        if tn and tn in JASPAR_PWMS:
            hits = scan_pwm(seqs[i], JASPAR_PWMS[tn])
            if hits:
                mlen = JASPAR_PWMS[tn].shape[0]
                motif_positions[i] = [(p, p + mlen) for p, _ in hits]

    print(f"  Samples with motif hits: {len(motif_positions)}/{len(seqs)}")

    enrichment = {}
    if motif_positions:
        print("  Computing gradient enrichment at motifs...")
        enrichment = gradient_enrichment_at_motifs(grad_importance, seqs, motif_positions)
        print(f"    Motif vs random ratio: {enrichment['motif_vs_random_ratio']:.2f}x")
        print(f"    Motif vs flanking ratio: {enrichment['motif_vs_flanking_ratio']:.2f}x")

    # Per-TF gradient magnitude statistics
    print("  Per-TF gradient statistics...")
    per_tf_grad = {}
    grad_mag = np.abs(grad_importance).sum(axis=-1)  # [N, L]
    for ti, tn in enumerate(tf_names):
        mask = tf_idx == ti
        if mask.sum() > 0:
            tf_grad = grad_mag[mask]
            per_tf_grad[tn] = {
                'mean_magnitude': float(tf_grad.mean()),
                'std_magnitude': float(tf_grad.std()),
                'max_magnitude': float(tf_grad.max()),
                'n_samples': int(mask.sum()),
            }

    result = {
        'attention_vs_gradient': attn_grad_summary,
        'motif_enrichment': enrichment,
        'per_tf_gradient': per_tf_grad,
        'n_samples': len(seqs),
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Gradient saliency analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gradient-Based Saliency Analysis")
    print("=" * 70)

    results = {}
    for name in ["K562-7tf", "K562-fulltf"]:
        try:
            results[name] = analyze_dataset(name, device, args.max_samples)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")

    path = out_dir / "saliency_analysis.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
