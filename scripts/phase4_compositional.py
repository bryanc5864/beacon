#!/usr/bin/env python3
"""
Phase 4.3: Compositional Queries - TF Co-occurrence Analysis

Demonstrates BEACON's unique ability to answer questions about
TF co-binding that BPNet cannot answer directly.

Queries:
1. Which TFs co-bind within 50bp?
2. What's the typical spacing between TF pairs?
3. Are certain TF combinations enriched?
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


def analyze_co_binding(model, data_loader, device, tf_names, occupancy_threshold=0.3):
    """
    Analyze TF co-binding patterns from BEACON predictions.

    For each sequence, identifies active slots and computes:
    - TF co-occurrence (which TFs appear together)
    - Spacing between binding sites
    """
    model.eval()

    # Data structures
    co_occurrence = np.zeros((len(tf_names), len(tf_names)), dtype=int)
    spacing_data = defaultdict(list)  # {(tf1, tf2): [distances]}
    samples_with_multi_tf = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            outputs = model(sequences)

            occupancy = outputs['occupancy'].squeeze(-1).cpu().numpy()  # [B, K]
            positions = outputs['positions'].cpu().numpy()  # [B, K, 2] or [B, K, 1]
            tf_logits = outputs['tf_logits'].cpu().numpy()  # [B, K, n_tfs]

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                total_samples += 1

                # Find active slots
                active_mask = occupancy[b] > occupancy_threshold
                active_slots = np.where(active_mask)[0]

                if len(active_slots) < 2:
                    continue

                samples_with_multi_tf += 1

                # Get TF identities and positions for active slots
                sites = []
                for slot_idx in active_slots:
                    tf_idx = np.argmax(tf_logits[b, slot_idx])
                    pos = positions[b, slot_idx, 0]  # Mean position
                    occ = occupancy[b, slot_idx]
                    sites.append({
                        'slot': slot_idx,
                        'tf_index': tf_idx,
                        'position': pos,
                        'occupancy': occ,
                    })

                # Analyze pairs
                tfs_present = set(s['tf_index'] for s in sites)

                # Update co-occurrence matrix
                for tf_idx in tfs_present:
                    co_occurrence[tf_idx, tf_idx] += 1

                for tf1, tf2 in combinations(sorted(tfs_present), 2):
                    co_occurrence[tf1, tf2] += 1
                    co_occurrence[tf2, tf1] += 1

                # Compute pairwise spacing
                for s1, s2 in combinations(sites, 2):
                    tf1, tf2 = s1['tf_index'], s2['tf_index']
                    distance = abs(s1['position'] - s2['position'])

                    # Store as sorted tuple for consistency
                    key = tuple(sorted([tf1, tf2]))
                    spacing_data[key].append(distance)

    return {
        'co_occurrence': co_occurrence,
        'spacing_data': dict(spacing_data),
        'samples_with_multi_tf': samples_with_multi_tf,
        'total_samples': total_samples,
    }


def compute_enrichment(co_occurrence, marginals=None):
    """
    Compute co-occurrence enrichment (observed/expected).

    Expected = (count_tf1 * count_tf2) / total
    """
    n_tfs = co_occurrence.shape[0]

    if marginals is None:
        marginals = np.diag(co_occurrence)

    total = marginals.sum()
    if total == 0:
        return np.ones_like(co_occurrence, dtype=float)

    expected = np.outer(marginals, marginals) / total
    expected = np.maximum(expected, 1)  # Avoid division by zero

    enrichment = co_occurrence / expected

    return enrichment


def main():
    parser = argparse.ArgumentParser(description="Phase 4.3: Compositional Queries")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--occupancy-threshold", type=float, default=0.3)

    args = parser.parse_args()

    # Setup paths
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = Path(__file__).parent.parent / experiment_dir

    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    output_dir = args.output_dir or experiment_dir / "compositional_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4.3: Compositional Queries - TF Co-occurrence")
    print("=" * 60)
    print(f"Experiment: {experiment_dir}")
    print(f"Occupancy threshold: {args.occupancy_threshold}")
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

    # Load model
    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    print(f"\nLoading model...")

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

    # Load data
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir

    test_path = data_dir / "test.h5"
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Test samples: {len(dataset)}")

    # Analyze co-binding
    print("\nAnalyzing TF co-binding patterns...")
    results = analyze_co_binding(model, loader, device, tf_names,
                                  occupancy_threshold=args.occupancy_threshold)

    co_occurrence = results['co_occurrence']
    spacing_data = results['spacing_data']

    print(f"\nSamples with multiple active TFs: {results['samples_with_multi_tf']}/{results['total_samples']}")
    print(f"  ({100*results['samples_with_multi_tf']/results['total_samples']:.1f}%)")

    # Compute enrichment
    enrichment = compute_enrichment(co_occurrence)

    print("\n" + "=" * 60)
    print("QUERY 1: Which TFs co-bind together?")
    print("=" * 60)

    print("\nCo-occurrence counts:")
    for i, tf1 in enumerate(tf_names):
        for j, tf2 in enumerate(tf_names):
            if i < j and co_occurrence[i, j] > 0:
                print(f"  {tf1}-{tf2}: {co_occurrence[i, j]} sequences")

    print("\nEnrichment (observed/expected):")
    enriched_pairs = []
    for i, tf1 in enumerate(tf_names):
        for j, tf2 in enumerate(tf_names):
            if i < j:
                enr = enrichment[i, j]
                if enr > 1.5:
                    enriched_pairs.append((tf1, tf2, enr))
                    print(f"  {tf1}-{tf2}: {enr:.2f}x enriched")

    print("\n" + "=" * 60)
    print("QUERY 2: What's the spacing between TF pairs?")
    print("=" * 60)

    spacing_summary = {}
    for key, distances in spacing_data.items():
        tf1, tf2 = key
        tf1_name, tf2_name = tf_names[tf1], tf_names[tf2]

        if len(distances) >= 5:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            median_dist = np.median(distances)

            spacing_summary[f"{tf1_name}-{tf2_name}"] = {
                'count': len(distances),
                'mean': mean_dist,
                'std': std_dist,
                'median': median_dist,
            }

            print(f"\n  {tf1_name}-{tf2_name} (n={len(distances)}):")
            print(f"    Mean spacing: {mean_dist*2000:.0f} bp")
            print(f"    Median: {median_dist*2000:.0f} bp")
            print(f"    Std: {std_dist*2000:.0f} bp")

    print("\n" + "=" * 60)
    print("QUERY 3: Known biological relationships")
    print("=" * 60)

    # Check known biology
    known_pairs = [
        ("MYC", "MAX", "Should co-occur (heterodimerize)"),
        ("TAL1", "GATA1", "Known to co-bind in erythroid cells"),
        ("CTCF", "CEBPB", "Different functions, less likely"),
    ]

    print("\nValidating against known biology:")
    for tf1, tf2, explanation in known_pairs:
        if tf1 in tf_names and tf2 in tf_names:
            i, j = tf_names.index(tf1), tf_names.index(tf2)
            count = co_occurrence[i, j]
            enr = enrichment[i, j]
            print(f"\n  {tf1}-{tf2}: {explanation}")
            print(f"    Co-occurrence: {count} sequences")
            print(f"    Enrichment: {enr:.2f}x")
            if enr > 1.5:
                print(f"    --> ENRICHED (matches expectation)" if "Should" in explanation else "    --> ENRICHED")
            elif enr < 0.5:
                print(f"    --> DEPLETED")
            else:
                print(f"    --> NEUTRAL")

    # Create visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Co-occurrence heatmap
    ax = axes[0, 0]
    mask = np.triu(np.ones_like(co_occurrence, dtype=bool), k=1)
    sns.heatmap(co_occurrence, xticklabels=tf_names, yticklabels=tf_names,
                annot=True, fmt='d', cmap='Blues', ax=ax, mask=mask.T)
    ax.set_title('TF Co-occurrence Counts')

    # Enrichment heatmap
    ax = axes[0, 1]
    enrichment_display = enrichment.copy()
    np.fill_diagonal(enrichment_display, 1)
    sns.heatmap(enrichment_display, xticklabels=tf_names, yticklabels=tf_names,
                annot=True, fmt='.2f', cmap='RdBu_r', center=1, ax=ax,
                vmin=0.5, vmax=2.0)
    ax.set_title('Co-occurrence Enrichment (obs/exp)')

    # Spacing distributions for top pairs
    ax = axes[1, 0]
    top_pairs = sorted(spacing_data.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    for (tf1, tf2), distances in top_pairs:
        if len(distances) >= 10:
            label = f"{tf_names[tf1]}-{tf_names[tf2]}"
            distances_bp = np.array(distances) * 2000
            ax.hist(distances_bp, bins=30, alpha=0.5, label=label)

    ax.set_xlabel('Spacing (bp)')
    ax.set_ylabel('Count')
    ax.set_title('Binding Site Spacing Distribution')
    ax.legend()

    # Summary bar chart
    ax = axes[1, 1]
    pair_names = []
    pair_enrichments = []

    for i, tf1 in enumerate(tf_names):
        for j, tf2 in enumerate(tf_names):
            if i < j:
                pair_names.append(f"{tf1[:4]}-{tf2[:4]}")
                pair_enrichments.append(enrichment[i, j])

    # Sort by enrichment
    sorted_idx = np.argsort(pair_enrichments)[::-1]
    pair_names = [pair_names[i] for i in sorted_idx]
    pair_enrichments = [pair_enrichments[i] for i in sorted_idx]

    colors = ['green' if e > 1.5 else 'red' if e < 0.67 else 'gray' for e in pair_enrichments]
    ax.barh(range(len(pair_names)), pair_enrichments, color=colors, alpha=0.7)
    ax.axvline(x=1.0, color='black', linestyle='--', label='Expected')
    ax.axvline(x=1.5, color='green', linestyle=':', alpha=0.5, label='1.5x')
    ax.axvline(x=0.67, color='red', linestyle=':', alpha=0.5, label='0.67x')
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names)
    ax.set_xlabel('Enrichment (obs/exp)')
    ax.set_title('TF Pair Enrichment')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "compositional_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {
        'co_occurrence': co_occurrence.tolist(),
        'enrichment': enrichment.tolist(),
        'tf_names': tf_names,
        'spacing_summary': spacing_summary,
        'samples_with_multi_tf': results['samples_with_multi_tf'],
        'total_samples': results['total_samples'],
        'enriched_pairs': [(p[0], p[1], float(p[2])) for p in enriched_pairs],
    }

    with open(output_dir / "compositional_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: What BEACON Can Answer That BPNet Cannot")
    print("=" * 60)
    print("""
1. "How many binding sites are in this sequence?"
   BEACON: Instant (count active slots)
   BPNet: Requires DeepSHAP + TF-MoDISco (minutes)

2. "Which TF is at each binding site?"
   BEACON: Instant (argmax on TF logits)
   BPNet: Cannot answer directly

3. "Which TFs co-occur in the same region?"
   BEACON: Instant (check slot predictions)
   BPNet: Would require running multiple models + overlap analysis

4. "What's the typical spacing between MYC and MAX sites?"
   BEACON: Instant (compute from slot positions)
   BPNet: Cannot answer directly
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
