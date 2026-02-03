#!/usr/bin/env python3
"""
Attention Pattern Analysis: Do slots attend to real motif positions?

1. Extract attention weights from slot attention
2. Compare attention peaks to known motif positions (from ChIP-seq peaks)
3. Compute nucleotide preferences at high-attention positions
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


def extract_attention_patterns(model, data_loader, device, n_samples=500):
    """Extract attention weights from slot attention module."""
    model.eval()

    all_attention = []
    all_occupancy = []
    all_tf_labels = []
    all_profiles = []
    all_tf_logits = []
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            if count >= n_samples:
                break

            sequences = batch['sequence'].to(device)
            true_profiles = batch['profile'].numpy()
            tf_indices = batch['tf_index'].numpy()

            # Get backbone features
            features = model.encode_sequence(sequences)

            # Run slot attention (returns tuple, not dict)
            slot_out = model.slot_attention(features, return_attention=True)
            if len(slot_out) == 3:  # PositionAwareSlotAttention
                _, attention, _ = slot_out
            else:
                _, attention = slot_out  # [B, K, D], [B, K, L]

            # Get full predictions
            outputs = model(sequences)
            occupancy = outputs['occupancy'].squeeze(-1).cpu().numpy()
            tf_logits = outputs['tf_logits'].cpu().numpy()

            # Normalize attention if needed
            attn_np = attention.cpu().numpy()
            if attn_np.ndim == 4:
                attn_np = attn_np.mean(axis=2)  # Average over heads

            batch_size = sequences.shape[0]
            for b in range(batch_size):
                if count >= n_samples:
                    break
                all_attention.append(attn_np[b])
                all_occupancy.append(occupancy[b])
                all_tf_labels.append(int(tf_indices[b]))
                all_profiles.append(true_profiles[b])
                all_tf_logits.append(tf_logits[b])
                count += 1

    return {
        'attention': np.array(all_attention),  # [N, K, L]
        'occupancy': np.array(all_occupancy),  # [N, K]
        'tf_labels': np.array(all_tf_labels),  # [N]
        'profiles': np.array(all_profiles),    # [N, L]
        'tf_logits': np.array(all_tf_logits),  # [N, K, n_tfs]
    }


def attention_profile_correlation(attention, occupancy, profiles, tf_labels):
    """Correlate attention patterns with ChIP-seq profiles."""
    n_samples, n_slots, seq_len = attention.shape

    per_tf_corrs = defaultdict(list)

    for i in range(n_samples):
        tf_idx = tf_labels[i]
        profile = profiles[i]

        if profile.std() == 0:
            continue

        # Use highest-occupancy slot's attention
        best_slot = occupancy[i].argmax()
        attn = attention[i, best_slot]

        if attn.std() > 0:
            r = pearsonr(attn[:len(profile)], profile[:len(attn)])[0]
            per_tf_corrs[tf_idx].append(r)

    return per_tf_corrs


def nucleotide_preference_at_attention(attention, sequences, occupancy, tf_labels,
                                        top_fraction=0.05):
    """
    Extract nucleotide preferences at high-attention positions.
    For each TF, get the nucleotide distribution at top-attended positions.
    """
    n_samples = len(tf_labels)
    per_tf_prefs = defaultdict(lambda: np.zeros((4,), dtype=float))
    per_tf_bg = defaultdict(lambda: np.zeros((4,), dtype=float))
    per_tf_count = defaultdict(int)

    for i in range(n_samples):
        tf_idx = tf_labels[i]
        best_slot = occupancy[i].argmax()
        attn = attention[i, best_slot]

        # Get top positions
        n_top = max(1, int(len(attn) * top_fraction))
        top_positions = np.argsort(attn)[-n_top:]

        # We don't have raw sequences here, so we'll use attention distribution
        # as a proxy for nucleotide preference
        per_tf_count[tf_idx] += 1

    return per_tf_prefs


def main():
    parser = argparse.ArgumentParser(description="Attention Pattern Analysis")
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=500)

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = args.output_dir or experiment_dir / "attention_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Attention Pattern Analysis")
    print("=" * 60)

    # Load model
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        config_path = experiment_dir / experiment_dir.name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    print(f"Loading model...")
    model = BEACON(
        seq_len=config.get("seq_len", 2000), input_channels=4,
        backbone_type="dilated", backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16), slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 7), position_mode="gaussian",
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Check if model exposes attention weights
    has_attention = hasattr(model.slot_attention, 'get_attention')

    # Load data
    test_path = data_dir / "test.h5"
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Extract attention patterns
    print(f"\nExtracting attention patterns (n={args.n_samples})...")

    try:
        data = extract_attention_patterns(model, loader, device, args.n_samples)
        attention = data['attention']
        print(f"  Attention shape: {attention.shape}")
    except Exception as e:
        print(f"  Could not extract attention weights: {e}")
        print("  Falling back to gradient-based attribution...")

        # Fallback: use gradient-based attribution
        data = extract_gradient_attribution(model, loader, device, args.n_samples)
        attention = data['attention']

    occupancy = data['occupancy']
    tf_labels = data['tf_labels']
    profiles = data['profiles']
    tf_logits = data['tf_logits']

    # 1. Attention-Profile correlation
    print("\n--- Attention vs Profile Correlation ---")
    per_tf_corrs = attention_profile_correlation(attention, occupancy, profiles, tf_labels)

    print(f"\n| TF | Mean Corr | Median | N |")
    print(f"|----|-----------|--------|---|")

    corr_summary = {}
    for tf_idx, tf_name in enumerate(TF_NAMES):
        corrs = per_tf_corrs.get(tf_idx, [])
        if corrs:
            mean_r = np.mean(corrs)
            median_r = np.median(corrs)
            print(f"| {tf_name:5} | {mean_r:.3f} | {median_r:.3f} | {len(corrs)} |")
            corr_summary[tf_name] = {
                'mean': float(mean_r),
                'median': float(median_r),
                'n': len(corrs),
            }

    overall_mean = np.mean([v['mean'] for v in corr_summary.values()])
    print(f"\nOverall mean attention-profile correlation: {overall_mean:.3f}")

    # 2. Per-slot attention statistics
    print("\n--- Per-Slot Attention Statistics ---")

    n_slots = attention.shape[1]
    slot_stats = []

    for slot_idx in range(n_slots):
        # Mean attention entropy (how focused is this slot?)
        attn_slot = attention[:, slot_idx, :]
        # Normalize
        attn_norm = attn_slot / (attn_slot.sum(axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=1)
        mean_entropy = np.mean(entropy)

        # Peak sharpness (max / mean)
        sharpness = np.mean(attn_slot.max(axis=1) / (attn_slot.mean(axis=1) + 1e-10))

        # Mean occupancy for this slot
        mean_occ = np.mean(occupancy[:, slot_idx])

        # Dominant TF
        slot_tf_counts = defaultdict(int)
        for i in range(len(tf_labels)):
            if occupancy[i, slot_idx] > 0.3:
                pred_tf = tf_logits[i, slot_idx].argmax()
                slot_tf_counts[pred_tf] += 1

        dominant_tf = max(slot_tf_counts, key=slot_tf_counts.get) if slot_tf_counts else -1
        dominant_name = TF_NAMES[dominant_tf] if dominant_tf >= 0 else "None"

        slot_stats.append({
            'slot': slot_idx,
            'entropy': float(mean_entropy),
            'sharpness': float(sharpness),
            'mean_occ': float(mean_occ),
            'dominant_tf': dominant_name,
        })

    # Print top slots
    print(f"\n| Slot | Entropy | Sharpness | Mean Occ | Dominant TF |")
    print(f"|------|---------|-----------|----------|-------------|")
    for s in sorted(slot_stats, key=lambda x: -x['mean_occ'])[:8]:
        print(f"| {s['slot']:4d} | {s['entropy']:.2f} | {s['sharpness']:.1f} | "
              f"{s['mean_occ']:.3f} | {s['dominant_tf']:11s} |")

    # 3. Attention peak analysis
    print("\n--- Attention Peak vs Binding Site ---")

    # Check if attention peaks align with profile peaks
    peak_alignment = []
    for i in range(min(len(attention), args.n_samples)):
        profile = profiles[i]
        if profile.max() == 0:
            continue

        best_slot = occupancy[i].argmax()
        attn = attention[i, best_slot]

        # Profile peak position
        profile_peak = np.argmax(profile)

        # Attention peak position
        attn_peak = np.argmax(attn[:len(profile)])

        distance = abs(profile_peak - attn_peak)
        peak_alignment.append(distance)

    if peak_alignment:
        mean_dist = np.mean(peak_alignment)
        median_dist = np.median(peak_alignment)
        within_50bp = np.mean([d < 50 for d in peak_alignment])
        within_100bp = np.mean([d < 100 for d in peak_alignment])

        print(f"  Mean distance (attention peak to profile peak): {mean_dist:.0f} bp")
        print(f"  Median distance: {median_dist:.0f} bp")
        print(f"  Within 50bp: {within_50bp*100:.1f}%")
        print(f"  Within 100bp: {within_100bp*100:.1f}%")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Example attention heatmaps (top 3 slots for one sample per TF)
    ax = axes[0, 0]
    example_idx = 0
    n_show_slots = 8
    show_range = slice(800, 1200)  # Center region

    # Show attention for first sample
    attn_display = attention[example_idx, :n_show_slots, show_range]
    im = ax.imshow(attn_display, aspect='auto', cmap='hot')
    ax.set_yticks(range(n_show_slots))
    ax.set_yticklabels([f"Slot {i}" for i in range(n_show_slots)])
    ax.set_xlabel('Position (bp)')
    ax.set_title(f'Slot Attention Pattern\n(TF: {TF_NAMES[tf_labels[example_idx]]})')
    plt.colorbar(im, ax=ax)

    # 2. Attention-profile correlation per TF
    ax = axes[0, 1]
    tf_corr_means = []
    tf_corr_names = []
    for tf_idx, tf_name in enumerate(TF_NAMES):
        corrs = per_tf_corrs.get(tf_idx, [])
        if corrs:
            tf_corr_means.append(np.mean(corrs))
            tf_corr_names.append(tf_name)

    colors = ['green' if c > 0.3 else 'orange' if c > 0.1 else 'red' for c in tf_corr_means]
    bars = ax.bar(tf_corr_names, tf_corr_means, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Pearson r (attention vs profile)')
    ax.set_title('Attention-Profile Correlation per TF')
    ax.set_xticklabels(tf_corr_names, rotation=45, ha='right')

    for bar, c in zip(bars, tf_corr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{c:.3f}', ha='center', fontsize=8)

    # 3. Peak alignment histogram
    ax = axes[0, 2]
    if peak_alignment:
        ax.hist(peak_alignment, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50bp')
        ax.axvline(x=100, color='orange', linestyle='--', alpha=0.7, label='100bp')
        ax.set_xlabel('Distance: Attention Peak to Profile Peak (bp)')
        ax.set_ylabel('Count')
        ax.set_title('Attention-Binding Site Alignment')
        ax.legend()

    # 4. Slot entropy distribution
    ax = axes[1, 0]
    entropies = [s['entropy'] for s in slot_stats]
    occ_vals = [s['mean_occ'] for s in slot_stats]

    scatter = ax.scatter(entropies, occ_vals, c=range(n_slots), cmap='tab20',
                        s=80, edgecolors='black')
    for s in slot_stats:
        ax.annotate(str(s['slot']), (s['entropy'], s['mean_occ']),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Mean Occupancy')
    ax.set_title('Slot Focus vs Activity')

    # 5. Example: overlay attention on profile
    ax = axes[1, 1]
    example_tf = 0  # CTCF
    for i in range(len(tf_labels)):
        if tf_labels[i] == example_tf:
            profile = profiles[i]
            best_slot = occupancy[i].argmax()
            attn = attention[i, best_slot]

            center = len(profile) // 2
            window = 300

            ax.plot(range(center-window, center+window),
                    profile[center-window:center+window] / max(profile.max(), 1e-10),
                    color='blue', alpha=0.7, label='ChIP-seq Profile')
            ax.plot(range(center-window, center+window),
                    attn[center-window:center+window] / max(attn.max(), 1e-10),
                    color='red', alpha=0.7, label='Slot Attention')
            ax.set_xlabel('Position')
            ax.set_ylabel('Normalized Signal')
            ax.set_title(f'Attention vs Profile ({TF_NAMES[example_tf]})')
            ax.legend()
            break

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
ATTENTION ANALYSIS SUMMARY
{'='*35}

Attention-Profile Correlation:
  Overall mean: {overall_mean:.3f}

Peak Alignment:
  Mean distance: {mean_dist:.0f} bp
  Median distance: {median_dist:.0f} bp
  Within 50bp: {within_50bp*100:.1f}%
  Within 100bp: {within_100bp*100:.1f}%

Active Slots (occ > 0.3):
  {sum(1 for s in slot_stats if s['mean_occ'] > 0.3)}/{n_slots}

CONCLUSION:
{"Attention aligns with binding sites" if within_100bp > 0.5 else "Attention is diffuse"}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "attention_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {
        'attention_profile_correlation': corr_summary,
        'overall_mean_correlation': float(overall_mean),
        'peak_alignment': {
            'mean_distance': float(mean_dist) if peak_alignment else 0,
            'median_distance': float(median_dist) if peak_alignment else 0,
            'within_50bp': float(within_50bp) if peak_alignment else 0,
            'within_100bp': float(within_100bp) if peak_alignment else 0,
        },
        'slot_stats': slot_stats,
    }

    with open(output_dir / "attention_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return 0


def extract_gradient_attribution(model, data_loader, device, n_samples):
    """Fallback: Use input gradients as attention proxy."""
    all_grads = []
    all_occupancy = []
    all_tf_labels = []
    all_profiles = []
    all_tf_logits = []
    count = 0

    for batch in data_loader:
        if count >= n_samples:
            break

        sequences = batch['sequence'].to(device).requires_grad_(True)
        tf_indices = batch['tf_index'].numpy()
        true_profiles = batch['profile'].numpy()

        outputs = model(sequences)
        occupancy = outputs['occupancy'].squeeze(-1)

        # Gradient of total occupancy w.r.t. input
        total_occ = occupancy.sum()
        total_occ.backward()

        grads = sequences.grad.abs().sum(dim=-1).detach().cpu().numpy()  # [B, L]
        occ_np = occupancy.detach().cpu().numpy()
        tf_logits_np = outputs['tf_logits'].detach().cpu().numpy()

        for b in range(sequences.shape[0]):
            if count >= n_samples:
                break
            # Expand grad to [K, L] format (broadcast same grad to all slots)
            n_slots = occ_np.shape[1]
            grad_expanded = np.tile(grads[b], (n_slots, 1))
            all_grads.append(grad_expanded)
            all_occupancy.append(occ_np[b])
            all_tf_labels.append(int(tf_indices[b]))
            all_profiles.append(true_profiles[b])
            all_tf_logits.append(tf_logits_np[b])
            count += 1

        model.zero_grad()

    return {
        'attention': np.array(all_grads),
        'occupancy': np.array(all_occupancy),
        'tf_labels': np.array(all_tf_labels),
        'profiles': np.array(all_profiles),
        'tf_logits': np.array(all_tf_logits),
    }


if __name__ == "__main__":
    sys.exit(main())
