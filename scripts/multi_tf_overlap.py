#!/usr/bin/env python3
"""
Multi-TF Overlap Experiment: Demonstrate multi-slot activation on real co-bound regions.

Finds genomic regions where 2+ TFs have overlapping ChIP-seq peaks, runs BEACON,
and checks whether multiple slots activate with different TF predictions.

This addresses the critical reviewer concern: "Why use slot attention if only one slot
is ever active?" By providing sequences with multiple binding sites, we show that
slots can specialize for different TFs when the input demands it.
"""

import sys
import gzip
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from beacon.models import BEACON

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]

PEAK_DIR = Path("data/raw/encode/multi_tf_k562")


def load_peaks(tf_name, peak_dir, min_score=0):
    """Load narrowPeak peaks for a TF."""
    tf_dir = peak_dir / tf_name
    peak_files = list(tf_dir.glob("*peaks.narrowPeak.gz")) + list(tf_dir.glob("*.narrowPeak.gz"))

    if not peak_files:
        print(f"  WARNING: No peak files for {tf_name} in {tf_dir}")
        return []

    peaks = []
    for pf in peak_files:
        with gzip.open(pf, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 10:
                    continue
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                score = float(parts[4]) if parts[4] != '.' else 0
                summit_offset = int(parts[9])
                summit = start + summit_offset

                if score >= min_score:
                    peaks.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'summit': summit,
                        'score': score,
                        'tf': tf_name,
                    })

    return peaks


def find_overlapping_regions(all_peaks, window=500):
    """
    Find regions where peaks from 2+ TFs overlap.

    Uses a simple approach: group peaks by chromosome, sort by position,
    and find clusters where peaks from different TFs are within `window` bp.
    """
    # Group by chromosome
    by_chrom = defaultdict(list)
    for p in all_peaks:
        by_chrom[p['chrom']].append(p)

    multi_tf_regions = []

    for chrom, peaks in by_chrom.items():
        # Sort by summit position
        peaks.sort(key=lambda p: p['summit'])

        # Sliding window: find clusters of peaks from different TFs
        for i, anchor in enumerate(peaks):
            tfs_in_window = {anchor['tf']: anchor}

            # Look forward for nearby peaks from different TFs
            for j in range(i + 1, len(peaks)):
                if peaks[j]['summit'] - anchor['summit'] > window:
                    break
                other_tf = peaks[j]['tf']
                if other_tf not in tfs_in_window:
                    tfs_in_window[other_tf] = peaks[j]

            if len(tfs_in_window) >= 2:
                # Center on midpoint of all summits
                summits = [p['summit'] for p in tfs_in_window.values()]
                center = int(np.mean(summits))
                tf_set = frozenset(tfs_in_window.keys())

                multi_tf_regions.append({
                    'chrom': chrom,
                    'center': center,
                    'tfs': sorted(tfs_in_window.keys()),
                    'n_tfs': len(tfs_in_window),
                    'tf_set': tf_set,
                    'summits': {tf: p['summit'] for tf, p in tfs_in_window.items()},
                    'spacing': max(summits) - min(summits),
                })

    # Deduplicate: keep unique regions (within 200bp)
    multi_tf_regions.sort(key=lambda r: (r['chrom'], r['center']))
    deduped = []
    for region in multi_tf_regions:
        if not deduped or region['chrom'] != deduped[-1]['chrom'] or \
           abs(region['center'] - deduped[-1]['center']) > 200:
            deduped.append(region)
        elif region['n_tfs'] > deduped[-1]['n_tfs']:
            deduped[-1] = region  # Keep the one with more TFs

    return deduped


def one_hot_encode(seq):
    """Convert DNA sequence to one-hot [L, 4]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            one_hot[i, mapping[base]] = 1.0
        else:
            one_hot[i, :] = 0.25
    return one_hot


def analyze_slot_activation(model, sequences, device, tf_names, occ_threshold=0.3):
    """
    Run BEACON on sequences and analyze slot activation patterns.

    Returns per-sequence slot activation info.
    """
    results = []

    for i, seq_str in enumerate(sequences):
        oh = one_hot_encode(seq_str)
        tensor = torch.from_numpy(oh).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = model(tensor)

        occ = outputs['occupancy'].squeeze(-1).cpu().numpy()[0]  # [K]
        tf_logits = outputs['tf_logits'].cpu().numpy()[0]  # [K, N_TFs]
        tf_preds = tf_logits.argmax(axis=1)  # [K]

        # Find active slots
        active_slots = np.where(occ > occ_threshold)[0]
        n_active = len(active_slots)

        # Get TF predictions for active slots
        active_tfs = []
        for s in active_slots:
            tf_idx = tf_preds[s]
            tf_name = tf_names[tf_idx] if tf_idx < len(tf_names) else f"TF{tf_idx}"
            active_tfs.append({
                'slot': int(s),
                'tf': tf_name,
                'occupancy': float(occ[s]),
                'confidence': float(tf_logits[s].max()),
            })

        unique_tfs = set(a['tf'] for a in active_tfs)

        results.append({
            'n_active_slots': n_active,
            'n_unique_tfs': len(unique_tfs),
            'active_tfs': active_tfs,
            'unique_tf_set': sorted(unique_tfs),
            'all_occupancies': occ.tolist(),
        })

        if i < 5 or (n_active > 1 and i < 20):
            # Print examples
            print(f"  Seq {i}: {n_active} active slots, "
                  f"{len(unique_tfs)} unique TFs: {sorted(unique_tfs)}")
            for a in active_tfs:
                print(f"    Slot {a['slot']}: {a['tf']} (occ={a['occupancy']:.3f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-TF Overlap Experiment")
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--genome", type=Path, default=Path("data/raw/genome/hg38.fa"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-regions", type=int, default=500)
    parser.add_argument("--window", type=int, default=500,
                        help="Max distance between peaks to consider co-bound (bp)")
    parser.add_argument("--occ-threshold", type=float, default=0.3,
                        help="Occupancy threshold for considering a slot active")
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="Second experiment dir (e.g. outputs/beacon_multi) for side-by-side")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    genome_path = args.genome
    if not genome_path.is_absolute():
        genome_path = base_dir / genome_path

    peak_dir = base_dir / PEAK_DIR
    output_dir = experiment_dir / "multi_tf_overlap"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-TF Overlap Experiment")
    print("=" * 60)
    print(f"Peak directory: {peak_dir}")
    print(f"Overlap window: {args.window} bp")
    print(f"Occupancy threshold: {args.occ_threshold}")

    # 1. Load all peaks
    print("\n--- Loading Peaks ---")
    all_peaks = []
    for tf in TF_NAMES:
        peaks = load_peaks(tf, peak_dir)
        print(f"  {tf}: {len(peaks)} peaks")
        all_peaks.extend(peaks)

    print(f"  Total: {len(all_peaks)} peaks")

    # 2. Find overlapping regions
    print("\n--- Finding Multi-TF Regions ---")
    regions = find_overlapping_regions(all_peaks, window=args.window)
    print(f"  Found {len(regions)} multi-TF regions")

    # Breakdown by number of TFs
    by_n_tfs = Counter(r['n_tfs'] for r in regions)
    for n, count in sorted(by_n_tfs.items()):
        print(f"    {n} TFs: {count} regions")

    # Most common TF pairs
    pair_counts = Counter()
    for r in regions:
        tfs = r['tfs']
        for i in range(len(tfs)):
            for j in range(i + 1, len(tfs)):
                pair_counts[(tfs[i], tfs[j])] += 1

    print("\n  Top TF co-binding pairs:")
    for (tf1, tf2), count in pair_counts.most_common(10):
        print(f"    {tf1}-{tf2}: {count}")

    # Filter to test chromosomes (chr20-22, chrX) for fair comparison
    test_chroms = {'chr20', 'chr21', 'chr22', 'chrX'}
    test_regions = [r for r in regions if r['chrom'] in test_chroms]
    print(f"\n  Test chromosome regions: {len(test_regions)}")

    if not test_regions:
        print("  No test regions found, using all regions")
        test_regions = regions

    # Cap to max_regions
    test_regions = test_regions[:args.max_regions]

    # 3. Extract sequences
    print("\n--- Extracting Sequences ---")
    try:
        import pysam
    except ImportError:
        print("ERROR: pysam required. pip install pysam")
        return 1

    fasta = pysam.FastaFile(str(genome_path))
    available_chroms = set(fasta.references)

    seq_len = 2000
    sequences = []
    valid_regions = []

    for r in test_regions:
        chrom = r['chrom']
        if chrom not in available_chroms:
            continue

        center = r['center']
        start = max(0, center - seq_len // 2)
        end = start + seq_len

        chrom_len = fasta.get_reference_length(chrom)
        if end > chrom_len:
            continue

        seq = fasta.fetch(chrom, start, end).upper()
        if len(seq) == seq_len and 'N' * 50 not in seq:
            sequences.append(seq)
            valid_regions.append(r)

    fasta.close()
    print(f"  Extracted {len(sequences)} valid sequences")

    if not sequences:
        print("ERROR: No valid sequences extracted!")
        return 1

    # 4. Also get single-TF control sequences from the test set
    print("\n--- Loading Single-TF Controls ---")
    import h5py
    test_h5 = base_dir / "data" / "processed" / "multi_tf_k562" / "test.h5"
    if test_h5.exists():
        with h5py.File(test_h5, 'r') as f:
            ctrl_seqs_oh = f['sequences'][:200]  # [N, 2000, 4]
            ctrl_tf_indices = f['tf_indices'][:200]

        # Convert one-hot back to sequences for consistency
        base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        ctrl_sequences = []
        for oh in ctrl_seqs_oh:
            seq = ''.join(base_map.get(oh[i].argmax(), 'N') for i in range(oh.shape[0]))
            ctrl_sequences.append(seq)
        print(f"  Loaded {len(ctrl_sequences)} single-TF controls")
    else:
        ctrl_sequences = []
        print("  WARNING: test.h5 not found, skipping controls")

    # 5. Load model
    print("\n--- Loading Model ---")
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        # Handle double-nested directory structure
        nested = experiment_dir / experiment_dir.name / "best_model.pt"
        if nested.exists():
            checkpoint_path = nested
    if not config_path.exists():
        nested_config = experiment_dir / experiment_dir.name / "config.json"
        if nested_config.exists():
            with open(nested_config) as f:
                config = json.load(f)
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

    # 6. Run BEACON on multi-TF sequences
    print(f"\n--- Analyzing Multi-TF Regions (n={len(sequences)}) ---")
    multi_results = analyze_slot_activation(
        model, sequences, device, TF_NAMES, args.occ_threshold)

    # 7. Run BEACON on single-TF controls
    if ctrl_sequences:
        print(f"\n--- Analyzing Single-TF Controls (n={len(ctrl_sequences)}) ---")
        ctrl_results = analyze_slot_activation(
            model, ctrl_sequences, device, TF_NAMES, args.occ_threshold)
    else:
        ctrl_results = []

    # 8. Compute summary statistics
    print("\n" + "=" * 60)
    print("MULTI-TF OVERLAP RESULTS")
    print("=" * 60)

    multi_active = [r['n_active_slots'] for r in multi_results]
    multi_unique = [r['n_unique_tfs'] for r in multi_results]

    print(f"\n  Multi-TF Regions (n={len(multi_results)}):")
    print(f"    Mean active slots: {np.mean(multi_active):.2f}")
    print(f"    Mean unique TFs predicted: {np.mean(multi_unique):.2f}")
    print(f"    Regions with >1 active slot: {sum(1 for x in multi_active if x > 1)} "
          f"({sum(1 for x in multi_active if x > 1)/len(multi_active)*100:.1f}%)")
    print(f"    Regions with >1 unique TF: {sum(1 for x in multi_unique if x > 1)} "
          f"({sum(1 for x in multi_unique if x > 1)/len(multi_unique)*100:.1f}%)")

    # Distribution of active slots
    slot_dist = Counter(multi_active)
    print(f"\n    Active slot distribution:")
    for n_slots, count in sorted(slot_dist.items()):
        pct = count / len(multi_active) * 100
        print(f"      {n_slots} slots: {count} ({pct:.1f}%)")

    if ctrl_results:
        ctrl_active = [r['n_active_slots'] for r in ctrl_results]
        ctrl_unique = [r['n_unique_tfs'] for r in ctrl_results]

        print(f"\n  Single-TF Controls (n={len(ctrl_results)}):")
        print(f"    Mean active slots: {np.mean(ctrl_active):.2f}")
        print(f"    Mean unique TFs predicted: {np.mean(ctrl_unique):.2f}")
        print(f"    Controls with >1 active slot: {sum(1 for x in ctrl_active if x > 1)} "
              f"({sum(1 for x in ctrl_active if x > 1)/len(ctrl_active)*100:.1f}%)")

    # 9. Check if multi-TF regions activate more slots
    if ctrl_results:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(multi_active, ctrl_active, alternative='greater')
        print(f"\n  Mann-Whitney U test (multi > single):")
        print(f"    U = {stat:.1f}, p = {pval:.4f}")

    # 10. TF co-detection accuracy
    print(f"\n  TF Co-Detection Analysis:")
    correct_tf_detected = 0
    total_tf_expected = 0
    for region, result in zip(valid_regions, multi_results):
        expected_tfs = set(region['tfs'])
        predicted_tfs = set(result['unique_tf_set'])
        for tf in expected_tfs:
            total_tf_expected += 1
            if tf in predicted_tfs:
                correct_tf_detected += 1

    if total_tf_expected > 0:
        tf_recall = correct_tf_detected / total_tf_expected
        print(f"    TF recall (expected TFs detected): {tf_recall*100:.1f}% "
              f"({correct_tf_detected}/{total_tf_expected})")

    # 11. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Active slots: multi-TF vs single-TF
    ax = axes[0, 0]
    if ctrl_results:
        ax.hist(ctrl_active, bins=range(0, max(max(multi_active), max(ctrl_active)) + 2),
                alpha=0.6, color='gray', label='Single-TF', density=True, align='left')
    ax.hist(multi_active, bins=range(0, max(multi_active) + 2),
            alpha=0.6, color='steelblue', label='Multi-TF', density=True, align='left')
    ax.set_xlabel('Number of Active Slots')
    ax.set_ylabel('Density')
    ax.set_title('Slot Activation: Multi-TF vs Single-TF')
    ax.legend()

    # 2. Unique TFs predicted
    ax = axes[0, 1]
    if ctrl_results:
        ax.hist(ctrl_unique, bins=range(0, max(max(multi_unique), max(ctrl_unique)) + 2),
                alpha=0.6, color='gray', label='Single-TF', density=True, align='left')
    ax.hist(multi_unique, bins=range(0, max(multi_unique) + 2),
            alpha=0.6, color='coral', label='Multi-TF', density=True, align='left')
    ax.set_xlabel('Number of Unique TFs Predicted')
    ax.set_ylabel('Density')
    ax.set_title('TF Diversity: Multi-TF vs Single-TF')
    ax.legend()

    # 3. Co-binding pair frequencies
    ax = axes[0, 2]
    top_pairs = pair_counts.most_common(10)
    if top_pairs:
        labels = [f"{t1}-{t2}" for (t1, t2), _ in top_pairs]
        counts = [c for _, c in top_pairs]
        ax.barh(labels[::-1], counts[::-1], color='teal', alpha=0.7)
        ax.set_xlabel('Number of Co-bound Regions')
        ax.set_title('Top TF Co-binding Pairs (ChIP-seq)')

    # 4. Occupancy distribution across all slots for multi-TF
    ax = axes[1, 0]
    all_occs = np.array([r['all_occupancies'] for r in multi_results])  # [N, K]
    mean_occ_per_slot = all_occs.mean(axis=0)
    ax.bar(range(len(mean_occ_per_slot)), mean_occ_per_slot, color='steelblue', alpha=0.7)
    ax.set_xlabel('Slot Index')
    ax.set_ylabel('Mean Occupancy')
    ax.set_title('Per-Slot Mean Occupancy (Multi-TF Regions)')

    # 5. Number of expected vs detected TFs
    ax = axes[1, 1]
    expected_vs_detected = []
    for region, result in zip(valid_regions, multi_results):
        expected_vs_detected.append((region['n_tfs'], result['n_unique_tfs']))
    if expected_vs_detected:
        exp, det = zip(*expected_vs_detected)
        ax.scatter(exp, det, alpha=0.3, s=20, c='steelblue')
        ax.plot([0, max(exp) + 1], [0, max(exp) + 1], 'r--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Expected TFs (ChIP-seq peaks)')
        ax.set_ylabel('Detected TFs (BEACON)')
        ax.set_title('Expected vs Detected TF Count')
        ax.legend()

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
MULTI-TF OVERLAP SUMMARY
{'='*40}

Overlap Regions Found: {len(regions)}
Test Regions Analyzed: {len(sequences)}

Multi-TF Regions:
  Mean active slots: {np.mean(multi_active):.2f}
  Mean unique TFs:   {np.mean(multi_unique):.2f}
  >1 slot active:    {sum(1 for x in multi_active if x > 1)/len(multi_active)*100:.1f}%
  >1 unique TF:      {sum(1 for x in multi_unique if x > 1)/len(multi_unique)*100:.1f}%
"""
    if ctrl_results:
        summary_text += f"""
Single-TF Controls:
  Mean active slots: {np.mean(ctrl_active):.2f}
  Mean unique TFs:   {np.mean(ctrl_unique):.2f}
  >1 slot active:    {sum(1 for x in ctrl_active if x > 1)/len(ctrl_active)*100:.1f}%
"""
    if ctrl_results:
        summary_text += f"""
Statistical Test:
  Mann-Whitney p = {pval:.4f}
"""
    if total_tf_expected > 0:
        summary_text += f"""
TF Co-Detection:
  Recall: {tf_recall*100:.1f}%
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "multi_tf_overlap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {
        'n_total_overlap_regions': len(regions),
        'n_test_regions': len(sequences),
        'overlap_window_bp': args.window,
        'occ_threshold': args.occ_threshold,
        'multi_tf': {
            'mean_active_slots': float(np.mean(multi_active)),
            'mean_unique_tfs': float(np.mean(multi_unique)),
            'pct_multi_slot': float(sum(1 for x in multi_active if x > 1) / len(multi_active)),
            'pct_multi_tf': float(sum(1 for x in multi_unique if x > 1) / len(multi_unique)),
            'active_slot_distribution': {str(k): v for k, v in sorted(slot_dist.items())},
        },
        'top_co_binding_pairs': {f"{t1}-{t2}": c for (t1, t2), c in pair_counts.most_common(10)},
    }

    if ctrl_results:
        results_json['single_tf_control'] = {
            'mean_active_slots': float(np.mean(ctrl_active)),
            'mean_unique_tfs': float(np.mean(ctrl_unique)),
            'pct_multi_slot': float(sum(1 for x in ctrl_active if x > 1) / len(ctrl_active)),
        }
        results_json['mann_whitney_p'] = float(pval)

    if total_tf_expected > 0:
        results_json['tf_co_detection_recall'] = float(tf_recall)

    with open(output_dir / "multi_tf_overlap_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # 12. Optional: Compare with BEACON-multi
    if args.compare_with:
        compare_dir = args.compare_with
        if not compare_dir.is_absolute():
            compare_dir = base_dir / compare_dir
        if compare_dir.exists():
            subdirs = [d for d in compare_dir.iterdir() if d.is_dir()]
            if subdirs:
                compare_dir = sorted(subdirs)[-1]

        cp2 = compare_dir / "best_model.pt"
        if not cp2.exists():
            cp2 = compare_dir / compare_dir.name / "best_model.pt"
        cfg2 = compare_dir / "config.json"
        if not cfg2.exists():
            cfg2 = compare_dir / compare_dir.name / "config.json"

        if cp2.exists():
            print(f"\n--- Comparing with BEACON-multi: {compare_dir.name} ---")
            if cfg2.exists():
                with open(cfg2) as f:
                    config2 = json.load(f)
            else:
                config2 = config if 'config' in dir() else {
                    "seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                    "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

            model2 = BEACON(
                seq_len=config2.get("seq_len", 2000), input_channels=4,
                backbone_type="dilated", backbone_dim=config2.get("backbone_dim", 128),
                backbone_layers=config2.get("backbone_layers", 4),
                n_slots=config2.get("n_slots", 16), slot_dim=config2.get("slot_dim", 128),
                n_iterations=config2.get("n_iterations", 3),
                n_tfs=config2.get("n_tfs", 7), position_mode="gaussian",
            )
            ckpt2 = torch.load(cp2, map_location='cpu')
            sd2 = ckpt2['model_state_dict']
            sd2 = {k.replace('module.', ''): v for k, v in sd2.items()}
            model2.load_state_dict(sd2)
            model2 = model2.to(device)
            model2.eval()

            print(f"  Running BEACON-multi on {len(sequences)} multi-TF sequences...")
            multi2_results = analyze_slot_activation(
                model2, sequences, device, TF_NAMES, args.occ_threshold)

            multi2_active = [r['n_active_slots'] for r in multi2_results]
            multi2_unique = [r['n_unique_tfs'] for r in multi2_results]

            print(f"\n{'='*60}")
            print("BEACON vs BEACON-multi COMPARISON (Multi-TF Regions)")
            print(f"{'='*60}")
            print(f"\n| Metric                  | BEACON | BEACON-multi |")
            print(f"|-------------------------|--------|--------------|")
            print(f"| Mean active slots       | {np.mean(multi_active):.2f}   | {np.mean(multi2_active):.2f}         |")
            print(f"| Mean unique TFs         | {np.mean(multi_unique):.2f}   | {np.mean(multi2_unique):.2f}         |")
            pct1 = sum(1 for x in multi_active if x > 1) / len(multi_active) * 100
            pct2 = sum(1 for x in multi2_active if x > 1) / len(multi2_active) * 100
            print(f"| >1 slot active          | {pct1:.1f}%  | {pct2:.1f}%        |")
            pct1u = sum(1 for x in multi_unique if x > 1) / len(multi_unique) * 100
            pct2u = sum(1 for x in multi2_unique if x > 1) / len(multi2_unique) * 100
            print(f"| >1 unique TF predicted  | {pct1u:.1f}%  | {pct2u:.1f}%        |")

            # TF co-detection recall for model2
            correct2 = 0
            total2 = 0
            for region, result in zip(valid_regions, multi2_results):
                for tf in region['tfs']:
                    total2 += 1
                    if tf in result['unique_tf_set']:
                        correct2 += 1
            recall2 = correct2 / total2 if total2 > 0 else 0
            print(f"| TF co-detection recall  | {tf_recall*100:.1f}%  | {recall2*100:.1f}%        |")

            # Mann-Whitney
            from scipy.stats import mannwhitneyu
            stat2, pval2 = mannwhitneyu(multi2_active, multi_active, alternative='greater')
            print(f"\n  Mann-Whitney (multi > original): U={stat2:.1f}, p={pval2:.4f}")

            # Save comparison
            results_json['beacon_multi_comparison'] = {
                'mean_active_slots': float(np.mean(multi2_active)),
                'mean_unique_tfs': float(np.mean(multi2_unique)),
                'pct_multi_slot': float(sum(1 for x in multi2_active if x > 1) / len(multi2_active)),
                'pct_multi_tf': float(sum(1 for x in multi2_unique if x > 1) / len(multi2_unique)),
                'tf_recall': float(recall2),
                'mann_whitney_p': float(pval2),
            }
        else:
            print(f"\n  BEACON-multi checkpoint not found at {cp2}")

    with open(output_dir / "multi_tf_overlap_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
