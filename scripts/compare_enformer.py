#!/usr/bin/env python3
"""
Compare BEACON vs Enformer on K562 TF ChIP-seq profile prediction.

Enformer (Avsec et al., Nature Methods 2021) is a 250M-param model that
predicts 5,313 genomic tracks from 196,608bp input. We compare its
per-TF ChIP-seq predictions against BEACON's on the same test regions.

Usage:
    # Step 1: Find available K562 tracks
    python scripts/compare_enformer.py --mode find-tracks

    # Step 2: Run comparison
    python scripts/compare_enformer.py --mode compare --device cuda --max-samples 100
"""

import sys
import gzip
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from beacon.models import BEACON

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
TEST_CHROMS = {"chr20", "chr21", "chr22", "chrX"}
ENFORMER_SEQ_LEN = 196_608
ENFORMER_BIN_SIZE = 128
ENFORMER_N_BINS = 896
BEACON_SEQ_LEN = 2000


def find_k562_tracks():
    """Download Basenji2 targets file and find K562 ChIP-seq tracks for our TFs."""
    import pandas as pd

    url = "https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_human.txt"
    print(f"Downloading targets from {url}...")
    df = pd.read_csv(url, sep='\t')
    print(f"Total tracks: {len(df)}")

    tf_tracks = {}
    for tf in TF_NAMES:
        # Search for ChIP-seq tracks matching TF name and K562
        matches = df[
            df['description'].str.contains('CHIP', case=False, na=False) &
            df['description'].str.contains(tf, case=False, na=False) &
            df['description'].str.contains('K562', case=False, na=False)
        ]
        if len(matches) > 0:
            tf_tracks[tf] = []
            for _, row in matches.iterrows():
                tf_tracks[tf].append({
                    'index': int(row['index']),
                    'description': row['description'],
                })
                print(f"  {tf}: track {row['index']} = {row['description']}")
        else:
            # Try broader search without K562
            broad = df[
                df['description'].str.contains('CHIP', case=False, na=False) &
                df['description'].str.contains(tf, case=False, na=False)
            ]
            if len(broad) > 0:
                print(f"  {tf}: No K562 track, but {len(broad)} tracks in other cell types")
                # Use the first match as fallback
                row = broad.iloc[0]
                tf_tracks[tf] = [{
                    'index': int(row['index']),
                    'description': row['description'],
                    'note': 'Not K562 - using best available',
                }]
                print(f"    Using: track {row['index']} = {row['description']}")
            else:
                print(f"  {tf}: No ChIP-seq track found")

    return tf_tracks


def load_test_peaks(peak_dir):
    """Load test chromosome peaks for all TFs."""
    peaks = []
    for tf in TF_NAMES:
        tf_dir = peak_dir / tf
        peak_files = list(tf_dir.glob("*peaks.narrowPeak.gz")) + \
                     list(tf_dir.glob("*.narrowPeak.gz"))
        for pf in peak_files:
            with gzip.open(pf, 'rt') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 10:
                        continue
                    chrom = parts[0]
                    if chrom not in TEST_CHROMS:
                        continue
                    start = int(parts[1])
                    summit_offset = int(parts[9])
                    summit = start + summit_offset
                    signal = float(parts[6]) if parts[6] != '.' else 1.0
                    peaks.append({
                        'chrom': chrom, 'summit': summit,
                        'signal': signal, 'tf': tf,
                    })
    return peaks


def one_hot_encode(seq):
    """One-hot encode for BEACON [L, 4]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in mapping:
            oh[i, mapping[b]] = 1.0
        else:
            oh[i, :] = 0.25
    return oh


def enformer_one_hot(seq):
    """One-hot encode for Enformer [L, 4] (same format, different N handling)."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in mapping:
            oh[i] = mapping[b]
    return oh


def run_comparison(tf_tracks, beacon_model, device, peak_dir, genome_path,
                   bigwig_dir, max_samples=100):
    """Run BEACON vs Enformer comparison on test peaks."""
    import pysam
    import pyBigWig
    from enformer_pytorch import from_pretrained

    # Load Enformer
    print("\nLoading Enformer...")
    enformer = from_pretrained('EleutherAI/enformer-official-rough')
    enformer.eval()
    enformer_device = device
    enformer = enformer.to(enformer_device)
    print(f"  Enformer loaded on {enformer_device}")

    # Load genome
    fasta = pysam.FastaFile(str(genome_path))

    # Load bigWig files for ground truth
    bigwigs = {}
    for tf in TF_NAMES:
        tf_dir = bigwig_dir / tf
        bw_files = list(tf_dir.glob("*signal.bigWig")) + list(tf_dir.glob("*.bigWig")) + \
                   list(tf_dir.glob("*.bw"))
        if bw_files:
            bigwigs[tf] = pyBigWig.open(str(bw_files[0]))

    # Load test peaks
    peaks = load_test_peaks(peak_dir)
    print(f"\nTest peaks: {len(peaks)}")

    # Sample peaks per TF
    per_tf_peaks = defaultdict(list)
    for p in peaks:
        per_tf_peaks[p['tf']].append(p)

    sampled_peaks = []
    samples_per_tf = max(1, max_samples // len(TF_NAMES))
    for tf in TF_NAMES:
        tf_peaks = per_tf_peaks[tf]
        tf_peaks.sort(key=lambda p: p['signal'], reverse=True)
        sampled_peaks.extend(tf_peaks[:samples_per_tf])

    np.random.seed(42)
    np.random.shuffle(sampled_peaks)
    sampled_peaks = sampled_peaks[:max_samples]
    print(f"Evaluating {len(sampled_peaks)} peaks...")

    # Build track index map (first track per TF)
    track_map = {}
    for tf, tracks in tf_tracks.items():
        if tracks:
            track_map[tf] = tracks[0]['index']

    # Compare
    results = defaultdict(lambda: {'beacon': [], 'enformer': [], 'ground_truth': []})

    for i, peak in enumerate(sampled_peaks):
        chrom = peak['chrom']
        summit = peak['summit']
        tf = peak['tf']
        chrom_len = fasta.get_reference_length(chrom)

        # BEACON: extract 2000bp
        b_start = max(0, summit - BEACON_SEQ_LEN // 2)
        b_end = b_start + BEACON_SEQ_LEN
        if b_end > chrom_len:
            continue

        beacon_seq = fasta.fetch(chrom, b_start, b_end).upper()
        if len(beacon_seq) != BEACON_SEQ_LEN:
            continue

        # Ground truth from bigWig
        if tf in bigwigs:
            try:
                gt_signal = np.array(bigwigs[tf].values(chrom, b_start, b_end),
                                     dtype=np.float32)
                gt_signal = np.nan_to_num(gt_signal, nan=0.0)
            except Exception:
                continue
        else:
            continue

        if gt_signal.max() == 0:
            continue

        # Normalize ground truth
        gt_norm = gt_signal / gt_signal.max()

        # Run BEACON
        beacon_oh = one_hot_encode(beacon_seq)
        beacon_tensor = torch.from_numpy(beacon_oh).unsqueeze(0).float().to(device)
        with torch.no_grad():
            beacon_out = beacon_model(beacon_tensor)
        beacon_profile = beacon_out['profile'].squeeze().cpu().numpy()
        if beacon_profile.ndim > 1:
            beacon_profile = beacon_profile.squeeze(-1)
        if beacon_profile.max() > 0:
            beacon_profile_norm = beacon_profile / beacon_profile.max()
        else:
            beacon_profile_norm = beacon_profile

        # Compute BEACON Pearson
        if gt_norm.std() > 0 and beacon_profile_norm.std() > 0:
            beacon_r = pearsonr(gt_norm, beacon_profile_norm[:len(gt_norm)])[0]
        else:
            beacon_r = 0

        # Enformer: extract 196,608bp
        e_start = max(0, summit - ENFORMER_SEQ_LEN // 2)
        e_end = e_start + ENFORMER_SEQ_LEN
        if e_end > chrom_len:
            e_end = chrom_len
            e_start = max(0, e_end - ENFORMER_SEQ_LEN)

        enformer_seq = fasta.fetch(chrom, e_start, e_end).upper()
        if len(enformer_seq) < ENFORMER_SEQ_LEN:
            # Pad with N
            enformer_seq = enformer_seq + 'N' * (ENFORMER_SEQ_LEN - len(enformer_seq))

        # Run Enformer
        if tf in track_map:
            enformer_oh = enformer_one_hot(enformer_seq)
            enformer_tensor = torch.from_numpy(enformer_oh).unsqueeze(0).float().to(enformer_device)

            with torch.no_grad():
                enformer_out = enformer(enformer_tensor, head='human')  # [1, 896, 5313]

            track_idx = track_map[tf]
            enformer_pred = enformer_out[0, :, track_idx].cpu().numpy()  # [896]

            # Map BEACON's 2000bp region to Enformer bins
            # BEACON region in Enformer coordinates
            beacon_start_in_enf = b_start - e_start
            beacon_end_in_enf = b_end - e_start

            # Enformer output covers central region
            # Bins map to positions: bin_i covers [i*128, (i+1)*128) relative to input
            # But Enformer crops: output covers central 114,688bp
            crop_start = (ENFORMER_SEQ_LEN - ENFORMER_N_BINS * ENFORMER_BIN_SIZE) // 2
            bin_start = max(0, (beacon_start_in_enf - crop_start) // ENFORMER_BIN_SIZE)
            bin_end = min(ENFORMER_N_BINS, (beacon_end_in_enf - crop_start) // ENFORMER_BIN_SIZE + 1)

            if bin_start < bin_end and bin_end <= ENFORMER_N_BINS:
                enformer_region = enformer_pred[bin_start:bin_end]  # ~15-16 bins

                # Upsample Enformer to BEACON resolution for comparison
                enformer_upsampled = np.repeat(enformer_region,
                                               ENFORMER_BIN_SIZE)[:BEACON_SEQ_LEN]
                if len(enformer_upsampled) < BEACON_SEQ_LEN:
                    enformer_upsampled = np.pad(enformer_upsampled,
                                                (0, BEACON_SEQ_LEN - len(enformer_upsampled)))

                if enformer_upsampled.max() > 0:
                    enformer_norm = enformer_upsampled / enformer_upsampled.max()
                else:
                    enformer_norm = enformer_upsampled

                if enformer_norm.std() > 0:
                    enformer_r = pearsonr(gt_norm, enformer_norm[:len(gt_norm)])[0]
                else:
                    enformer_r = 0
            else:
                enformer_r = 0
        else:
            enformer_r = None

        results[tf]['beacon'].append(beacon_r)
        if enformer_r is not None:
            results[tf]['enformer'].append(enformer_r)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(sampled_peaks)}...")

    # Cleanup
    fasta.close()
    for bw in bigwigs.values():
        bw.close()

    return dict(results)


def main():
    parser = argparse.ArgumentParser(description="BEACON vs Enformer comparison")
    parser.add_argument("--mode", choices=["find-tracks", "compare"], default="find-tracks")
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--peak-dir", type=Path,
                        default=Path("data/raw/encode/multi_tf_k562"))
    parser.add_argument("--genome", type=Path, default=Path("data/raw/genome/hg38.fa"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    if args.mode == "find-tracks":
        print("=" * 60)
        print("Finding K562 ChIP-seq Tracks in Enformer")
        print("=" * 60)
        tf_tracks = find_k562_tracks()

        # Save track mapping
        out = base_dir / "data" / "enformer_tracks.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(tf_tracks, f, indent=2)
        print(f"\nTrack mapping saved to {out}")
        print("\nNext: python scripts/compare_enformer.py --mode compare --device cuda")
        return 0

    # Compare mode
    print("=" * 60)
    print("BEACON vs Enformer Comparison")
    print("=" * 60)

    # Resolve paths
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    peak_dir = args.peak_dir
    if not peak_dir.is_absolute():
        peak_dir = base_dir / peak_dir

    genome_path = args.genome
    if not genome_path.is_absolute():
        genome_path = base_dir / genome_path

    output_dir = args.output_dir or experiment_dir / "enformer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load track mapping
    tracks_file = base_dir / "data" / "enformer_tracks.json"
    if not tracks_file.exists():
        print("ERROR: Run --mode find-tracks first")
        return 1
    with open(tracks_file) as f:
        tf_tracks = json.load(f)

    available_tfs = [tf for tf in TF_NAMES if tf in tf_tracks and tf_tracks[tf]]
    print(f"TFs with Enformer tracks: {available_tfs}")

    # Load BEACON model
    print("\nLoading BEACON model...")
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
        checkpoint_path = experiment_dir / experiment_dir.name / "best_model.pt"

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
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Run comparison
    results = run_comparison(tf_tracks, model, device, peak_dir, genome_path,
                             peak_dir, args.max_samples)

    # Print results
    print("\n" + "=" * 60)
    print("BEACON vs ENFORMER RESULTS")
    print("=" * 60)

    print(f"\n| TF     | N   | BEACON r | Enformer r | Winner  | Delta   |")
    print(f"|--------|-----|----------|------------|---------|---------|")

    summary = {}
    all_beacon = []
    all_enformer = []

    for tf in TF_NAMES:
        if tf not in results:
            continue
        b_scores = results[tf]['beacon']
        e_scores = results[tf]['enformer']

        if not b_scores:
            continue

        b_mean = np.mean(b_scores)
        all_beacon.extend(b_scores)

        if e_scores:
            e_mean = np.mean(e_scores)
            all_enformer.extend(e_scores)
            winner = "BEACON" if b_mean > e_mean else "Enformer"
            delta = b_mean - e_mean
            print(f"| {tf:6s} | {len(b_scores):3d} | {b_mean:.4f}   | {e_mean:.4f}     | "
                  f"{winner:7s} | {delta:+.4f}  |")
            summary[tf] = {
                'n': len(b_scores),
                'beacon_mean': float(b_mean),
                'enformer_mean': float(e_mean),
                'winner': winner,
                'delta': float(delta),
            }
        else:
            print(f"| {tf:6s} | {len(b_scores):3d} | {b_mean:.4f}   | N/A        | "
                  f"N/A     | N/A     |")
            summary[tf] = {'n': len(b_scores), 'beacon_mean': float(b_mean)}

    if all_beacon and all_enformer:
        b_overall = np.mean(all_beacon)
        e_overall = np.mean(all_enformer)
        print(f"| {'Mean':6s} | {len(all_beacon):3d} | {b_overall:.4f}   | {e_overall:.4f}     | "
              f"{'BEACON' if b_overall > e_overall else 'Enformer':7s} | "
              f"{b_overall - e_overall:+.4f}  |")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-TF comparison bars
    ax = axes[0]
    tfs_with_both = [tf for tf in TF_NAMES if tf in summary and 'enformer_mean' in summary[tf]]
    if tfs_with_both:
        x = np.arange(len(tfs_with_both))
        b_vals = [summary[tf]['beacon_mean'] for tf in tfs_with_both]
        e_vals = [summary[tf]['enformer_mean'] for tf in tfs_with_both]
        width = 0.35
        ax.bar(x - width/2, b_vals, width, label='BEACON (851K params)', color='steelblue')
        ax.bar(x + width/2, e_vals, width, label='Enformer (250M params)', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(tfs_with_both, rotation=45, ha='right')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Per-TF Profile Prediction')
        ax.legend()

    # Summary text
    ax = axes[1]
    ax.axis('off')
    txt = f"""
BEACON vs Enformer
{'='*35}

Model Sizes:
  BEACON:   851K parameters
  Enformer: 250M parameters (294x)

Input Lengths:
  BEACON:   2,000 bp
  Enformer: 196,608 bp (98x)

Profile Prediction (Pearson r):
  BEACON mean:   {np.mean(all_beacon):.4f}
  Enformer mean: {np.mean(all_enformer):.4f if all_enformer else 'N/A'}
"""
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "enformer_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    with open(output_dir / "enformer_results.json", "w") as f:
        json.dump({
            'per_tf': summary,
            'beacon_overall': float(np.mean(all_beacon)) if all_beacon else None,
            'enformer_overall': float(np.mean(all_enformer)) if all_enformer else None,
            'n_samples': len(all_beacon),
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
