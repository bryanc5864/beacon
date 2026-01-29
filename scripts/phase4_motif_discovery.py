#!/usr/bin/env python3
"""
Phase 4.1: Motif Discovery from BEACON Slot Attention

Extracts DNA motifs that each slot has learned to recognize and compares
them to JASPAR reference motifs.

This demonstrates BEACON's key advantage: interpretable motif discovery
built into the model, without needing separate tools like TF-MoDISco.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


# JASPAR matrix IDs for our TFs
JASPAR_MATRIX_IDS = {
    "CTCF": "MA0139.1",
    "GATA1": "MA0035.4",
    "TAL1": "MA0140.2",
    "MYC": "MA0147.3",
    "MAX": "MA0058.3",
    "SPI1": "MA0080.5",
    "CEBPB": "MA0466.3",
}


def download_jaspar_pwm(tf_name, matrix_id=None):
    """Download PWM from JASPAR REST API."""
    if matrix_id is None:
        matrix_id = JASPAR_MATRIX_IDS.get(tf_name)

    if matrix_id is None:
        # Search by name
        search_url = f"https://jaspar.elixir.no/api/v1/matrix/?name={tf_name}&collection=CORE"
        response = requests.get(search_url)
        if response.status_code == 200:
            results = response.json()
            if results.get('results'):
                matrix_id = results['results'][0]['matrix_id']

    if matrix_id is None:
        print(f"  Could not find JASPAR matrix for {tf_name}")
        return None

    # Get PFM
    pfm_url = f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}/"
    response = requests.get(pfm_url)

    if response.status_code != 200:
        print(f"  Failed to download {tf_name}: {response.status_code}")
        return None

    data = response.json()

    # Parse PFM from response
    pfm = data.get('pfm', {})
    if not pfm:
        return None

    # Convert to numpy array [length, 4] (A, C, G, T)
    length = len(pfm.get('A', []))
    pwm = np.zeros((length, 4))

    for i in range(length):
        pwm[i, 0] = pfm['A'][i]
        pwm[i, 1] = pfm['C'][i]
        pwm[i, 2] = pfm['G'][i]
        pwm[i, 3] = pfm['T'][i]

    # Normalize to probabilities
    pwm = pwm / (pwm.sum(axis=1, keepdims=True) + 1e-10)

    return {
        'matrix_id': matrix_id,
        'name': data.get('name', tf_name),
        'pwm': pwm,
        'consensus': data.get('sequence_logo', ''),
    }


def one_hot_to_sequence(one_hot):
    """Convert one-hot encoded sequence to string."""
    bases = ['A', 'C', 'G', 'T']
    seq = []
    for pos in one_hot:
        idx = np.argmax(pos)
        seq.append(bases[idx])
    return ''.join(seq)


def extract_attention_peaks(model, data_loader, device, threshold_percentile=90):
    """Extract sequences at high attention positions for each slot."""
    model.eval()

    slot_sequences = {}  # {slot_idx: [(sequence, attention_score), ...]}

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            tf_indices = batch['tf_index'].numpy()

            outputs = model(sequences)

            attention = outputs.get('attention')
            if attention is None:
                continue

            attention = attention.cpu().numpy()  # [B, K, L]
            sequences_np = sequences.cpu().numpy()  # [B, L, 4]

            batch_size, n_slots, seq_len = attention.shape

            for b in range(batch_size):
                tf_idx = tf_indices[b]

                for slot_idx in range(n_slots):
                    slot_attn = attention[b, slot_idx]  # [L]

                    # Find peak attention position
                    peak_pos = np.argmax(slot_attn)
                    peak_score = slot_attn[peak_pos]

                    # Skip if attention is too diffuse
                    if peak_score < np.percentile(slot_attn, threshold_percentile):
                        continue

                    # Extract 20bp window around peak
                    window = 10
                    start = max(0, peak_pos - window)
                    end = min(seq_len, peak_pos + window)

                    seq_window = sequences_np[b, start:end]
                    seq_str = one_hot_to_sequence(seq_window)

                    if slot_idx not in slot_sequences:
                        slot_sequences[slot_idx] = []

                    slot_sequences[slot_idx].append({
                        'sequence': seq_str,
                        'attention': peak_score,
                        'tf_index': tf_idx,
                        'position': peak_pos,
                    })

    return slot_sequences


def sequences_to_pwm(sequences, min_length=None):
    """Convert list of aligned sequences to PWM."""
    if not sequences:
        return None

    # Use most common length or specified min_length
    lengths = [len(s) for s in sequences]
    target_len = min_length or max(set(lengths), key=lengths.count)

    # Filter to sequences of target length
    filtered = [s for s in sequences if len(s) == target_len]
    if len(filtered) < 10:
        return None

    # Build PWM
    pwm = np.zeros((target_len, 4))
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    for seq in filtered:
        for pos, base in enumerate(seq):
            if base in base_to_idx:
                pwm[pos, base_to_idx[base]] += 1

    # Add pseudocount and normalize
    pwm = pwm + 0.5
    pwm = pwm / pwm.sum(axis=1, keepdims=True)

    return pwm


def pwm_correlation(pwm1, pwm2):
    """Compute Pearson correlation between two PWMs with alignment."""
    if pwm1 is None or pwm2 is None:
        return 0.0

    best_corr = -1

    # Try all alignments
    for offset in range(-len(pwm2) + 3, len(pwm1) - 2):
        start1 = max(0, offset)
        end1 = min(len(pwm1), len(pwm2) + offset)
        start2 = max(0, -offset)
        end2 = min(len(pwm2), len(pwm1) - offset)

        if end1 - start1 < 3:
            continue

        region1 = pwm1[start1:end1].flatten()
        region2 = pwm2[start2:end2].flatten()

        if len(region1) != len(region2):
            continue

        corr = np.corrcoef(region1, region2)[0, 1]
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr

    return best_corr


def pwm_to_consensus(pwm):
    """Convert PWM to consensus sequence."""
    bases = ['A', 'C', 'G', 'T']
    consensus = []
    for pos in pwm:
        idx = np.argmax(pos)
        consensus.append(bases[idx])
    return ''.join(consensus)


def plot_pwm_logo(pwm, title, ax):
    """Plot sequence logo from PWM."""
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba

    bases = ['A', 'C', 'G', 'T']
    colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}

    # Calculate information content
    bg = 0.25
    ic = np.zeros(len(pwm))
    for i, pos in enumerate(pwm):
        ic[i] = 2 + np.sum(pos * np.log2(pos + 1e-10))

    for i in range(len(pwm)):
        # Sort bases by frequency
        sorted_idx = np.argsort(pwm[i])

        y = 0
        for idx in sorted_idx:
            height = pwm[i, idx] * ic[i]
            if height > 0.01:
                ax.text(i + 0.5, y + height/2, bases[idx],
                       fontsize=min(12, 100 * height),
                       ha='center', va='center',
                       color=colors[bases[idx]],
                       fontweight='bold',
                       family='monospace')
            y += height

    ax.set_xlim(0, len(pwm))
    ax.set_ylim(0, 2)
    ax.set_xlabel('Position')
    ax.set_ylabel('Information (bits)')
    ax.set_title(title)


def plot_motif_comparison(discovered_pwm, jaspar_pwm, slot_name, tf_name, correlation, output_path):
    """Plot discovered motif vs JASPAR reference."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Discovered motif
    plot_pwm_logo(discovered_pwm, f'Discovered: {slot_name}', axes[0])

    # JASPAR reference
    plot_pwm_logo(jaspar_pwm, f'JASPAR: {tf_name}', axes[1])

    plt.suptitle(f'Pearson Correlation: {correlation:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 4.1: Motif Discovery from Slots")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=500)

    args = parser.parse_args()

    # Setup paths
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = Path(__file__).parent.parent / experiment_dir

    # Find latest run
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    output_dir = args.output_dir or experiment_dir / "motif_discovery"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4.1: Motif Discovery from BEACON Slots")
    print("=" * 60)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load config
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        config_path = experiment_dir / experiment_dir.name / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    tf_names = config.get("tf_names", ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"])
    print(f"TFs: {tf_names}")

    # Download JASPAR PWMs
    print("\n1. Downloading JASPAR reference motifs...")
    jaspar_pwms = {}
    for tf in tf_names:
        result = download_jaspar_pwm(tf)
        if result:
            jaspar_pwms[tf] = result
            print(f"  {tf}: {result['matrix_id']} (length={len(result['pwm'])})")
        else:
            print(f"  {tf}: NOT FOUND")

    # Load model
    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"\n2. Loading model from {checkpoint_path}...")

    model = BEACON(
        seq_len=config.get("seq_len", 2000),
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16),
        slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 7),
        position_mode="gaussian",
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load data
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir

    test_path = data_dir / "test.h5"
    if not test_path.exists():
        print(f"Test data not found: {test_path}")
        return 1

    print(f"\n3. Loading test data...")
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Extract attention peaks
    print(f"\n4. Extracting attention peaks from {min(args.n_samples, len(dataset))} samples...")

    # Limit samples
    limited_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    n_batches = args.n_samples // 32 + 1

    slot_sequences = {}
    batch_count = 0
    for batch in limited_loader:
        if batch_count >= n_batches:
            break

        sequences = batch['sequence'].to(device)
        tf_indices = batch['tf_index'].numpy()

        with torch.no_grad():
            outputs = model(sequences)

        attention = outputs.get('attention')
        if attention is None:
            batch_count += 1
            continue

        attention = attention.cpu().numpy()
        sequences_np = sequences.cpu().numpy()

        batch_size, n_slots, seq_len = attention.shape

        for b in range(batch_size):
            tf_idx = tf_indices[b]

            for slot_idx in range(n_slots):
                slot_attn = attention[b, slot_idx]
                peak_pos = np.argmax(slot_attn)
                peak_score = slot_attn[peak_pos]

                if peak_score < 0.1:
                    continue

                window = 10
                start = max(0, peak_pos - window)
                end = min(seq_len, peak_pos + window)

                seq_window = sequences_np[b, start:end]
                seq_str = one_hot_to_sequence(seq_window)

                if slot_idx not in slot_sequences:
                    slot_sequences[slot_idx] = []

                slot_sequences[slot_idx].append({
                    'sequence': seq_str,
                    'attention': peak_score,
                    'tf_index': tf_idx,
                })

        batch_count += 1

    print(f"  Extracted sequences from {sum(len(v) for v in slot_sequences.values())} attention peaks")

    # Build PWMs from slot attention
    print("\n5. Building PWMs from slot attention peaks...")
    slot_pwms = {}
    slot_tf_mapping = {}  # Which TF does each slot primarily detect?

    for slot_idx, seqs in slot_sequences.items():
        if len(seqs) < 20:
            continue

        # Find dominant TF for this slot
        tf_counts = Counter(s['tf_index'] for s in seqs)
        dominant_tf_idx = tf_counts.most_common(1)[0][0]
        dominant_tf = tf_names[dominant_tf_idx]

        # Build PWM from sequences
        sequences = [s['sequence'] for s in seqs]
        pwm = sequences_to_pwm(sequences, min_length=20)

        if pwm is not None:
            slot_pwms[slot_idx] = pwm
            slot_tf_mapping[slot_idx] = dominant_tf
            print(f"  Slot {slot_idx}: {len(seqs)} seqs, dominant TF = {dominant_tf}")

    # Compare to JASPAR
    print("\n6. Comparing discovered motifs to JASPAR...")
    comparisons = []

    for slot_idx, discovered_pwm in slot_pwms.items():
        dominant_tf = slot_tf_mapping[slot_idx]

        # Compare to all JASPAR motifs
        best_match = None
        best_corr = -1

        for tf_name, jaspar_data in jaspar_pwms.items():
            corr = pwm_correlation(discovered_pwm, jaspar_data['pwm'])
            if corr > best_corr:
                best_corr = corr
                best_match = tf_name

        # Also compute correlation to dominant TF
        dominant_corr = 0
        if dominant_tf in jaspar_pwms:
            dominant_corr = pwm_correlation(discovered_pwm, jaspar_pwms[dominant_tf]['pwm'])

        comparisons.append({
            'slot': slot_idx,
            'dominant_tf': dominant_tf,
            'best_jaspar_match': best_match,
            'best_correlation': best_corr,
            'dominant_tf_correlation': dominant_corr,
            'n_sequences': len(slot_sequences[slot_idx]),
        })

        print(f"  Slot {slot_idx}: dominant={dominant_tf} (r={dominant_corr:.3f}), best_match={best_match} (r={best_corr:.3f})")

        # Save motif comparison plot
        if dominant_tf in jaspar_pwms:
            plot_path = output_dir / f"slot_{slot_idx}_vs_jaspar.png"
            plot_motif_comparison(
                discovered_pwm,
                jaspar_pwms[dominant_tf]['pwm'],
                f"Slot {slot_idx}",
                dominant_tf,
                dominant_corr,
                plot_path
            )

    # Summary statistics
    print("\n" + "=" * 60)
    print("MOTIF DISCOVERY SUMMARY")
    print("=" * 60)

    correlations = [c['dominant_tf_correlation'] for c in comparisons if c['dominant_tf_correlation'] > 0]
    if correlations:
        print(f"\nCorrelation to dominant TF:")
        print(f"  Mean: {np.mean(correlations):.3f}")
        print(f"  Max:  {np.max(correlations):.3f}")
        print(f"  Min:  {np.min(correlations):.3f}")

        n_good = sum(1 for c in correlations if c > 0.5)
        print(f"\n  Slots with r > 0.5: {n_good}/{len(correlations)}")

        n_excellent = sum(1 for c in correlations if c > 0.7)
        print(f"  Slots with r > 0.7: {n_excellent}/{len(correlations)}")

    # Save results
    results = {
        'comparisons': comparisons,
        'n_slots_with_motifs': len(slot_pwms),
        'mean_correlation': float(np.mean(correlations)) if correlations else 0,
        'max_correlation': float(np.max(correlations)) if correlations else 0,
    }

    with open(output_dir / "motif_discovery_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create summary figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation distribution
    ax = axes[0]
    if correlations:
        ax.hist(correlations, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.5, color='orange', linestyle='--', label='r=0.5')
        ax.axvline(x=0.7, color='green', linestyle='--', label='r=0.7 (target)')
        ax.axvline(x=np.mean(correlations), color='red', linestyle='-',
                   label=f'Mean={np.mean(correlations):.3f}')
    ax.set_xlabel('Pearson Correlation with JASPAR')
    ax.set_ylabel('Count')
    ax.set_title('Slot-to-JASPAR Motif Correlation')
    ax.legend()

    # Slot-TF heatmap
    ax = axes[1]
    slot_indices = sorted(slot_tf_mapping.keys())
    tf_data = np.zeros((len(slot_indices), len(tf_names)))

    for i, slot_idx in enumerate(slot_indices):
        seqs = slot_sequences[slot_idx]
        for s in seqs:
            tf_data[i, s['tf_index']] += 1

    tf_data = tf_data / (tf_data.sum(axis=1, keepdims=True) + 1e-10)

    sns.heatmap(tf_data, xticklabels=tf_names,
                yticklabels=[f"Slot {s}" for s in slot_indices],
                cmap='YlOrRd', ax=ax)
    ax.set_title('Slot TF Specialization')
    ax.set_xlabel('TF')
    ax.set_ylabel('Slot')

    plt.tight_layout()
    plt.savefig(output_dir / "motif_discovery_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll outputs saved to: {output_dir}")

    # Success metric
    if correlations and np.mean(correlations) > 0.5:
        print("\n*** SUCCESS: Mean correlation > 0.5 ***")
    elif correlations and np.max(correlations) > 0.7:
        print("\n*** PARTIAL SUCCESS: At least one slot has r > 0.7 ***")
    else:
        print("\n*** NEEDS IMPROVEMENT: Correlations below target ***")

    return 0


if __name__ == "__main__":
    sys.exit(main())
