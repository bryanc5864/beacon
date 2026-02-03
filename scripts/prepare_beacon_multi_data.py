#!/usr/bin/env python3
"""
Prepare BEACON-multi training data: sequences with multiple TF binding sites.

Creates a dataset mixing:
- Multi-TF sequences (from overlapping ChIP-seq peaks, 2+ TFs per sequence)
- Single-TF sequences (from original peak data)

The binding_sites tensor has multiple filled rows per sequence, each with a
different TF identity, so the model learns to activate multiple slots.
"""

import sys
import gzip
import json
import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
TF_TO_IDX = {tf: i for i, tf in enumerate(TF_NAMES)}

TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18)]
VAL_CHROMS = ["chr18", "chr19"]
TEST_CHROMS = ["chr20", "chr21", "chr22", "chrX"]

SEQ_LENGTH = 2000
N_SLOTS = 16


def one_hot_encode(sequence):
    """One-hot encode DNA sequence [L, 4]."""
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            encoding[i, base_to_idx[base]] = 1.0
        else:
            encoding[i, :] = 0.25
    return encoding


def load_peaks(tf_name, peak_dir):
    """Load narrowPeak peaks for a TF."""
    tf_dir = peak_dir / tf_name
    peak_files = list(tf_dir.glob("*peaks.narrowPeak.gz")) + \
                 list(tf_dir.glob("*.narrowPeak.gz"))
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
                signal = float(parts[6]) if parts[6] != '.' else 1.0
                summit_offset = int(parts[9])
                summit = start + summit_offset
                peaks.append({
                    'chrom': chrom, 'start': start, 'end': end,
                    'summit': summit, 'signal': signal, 'tf': tf_name,
                })
    return peaks


def find_overlap_regions(all_peaks, window=500):
    """Find regions where 2+ TFs have peaks within window bp."""
    by_chrom = defaultdict(list)
    for p in all_peaks:
        by_chrom[p['chrom']].append(p)

    regions = []
    for chrom, peaks in by_chrom.items():
        peaks.sort(key=lambda p: p['summit'])
        used = set()

        for i, anchor in enumerate(peaks):
            if i in used:
                continue
            cluster = {anchor['tf']: [anchor]}
            cluster_indices = {i}

            for j in range(i + 1, len(peaks)):
                if peaks[j]['summit'] - anchor['summit'] > window:
                    break
                tf = peaks[j]['tf']
                if tf not in cluster:
                    cluster[tf] = []
                cluster[tf].append(peaks[j])
                cluster_indices.add(j)

            if len(cluster) >= 2:
                # Pick best peak per TF (highest signal)
                best_peaks = {}
                for tf, tf_peaks in cluster.items():
                    best_peaks[tf] = max(tf_peaks, key=lambda p: p['signal'])

                summits = [p['summit'] for p in best_peaks.values()]
                center = int(np.mean(summits))

                regions.append({
                    'chrom': chrom,
                    'center': center,
                    'peaks': best_peaks,
                    'n_tfs': len(best_peaks),
                })
                used.update(cluster_indices)

    # Deduplicate nearby regions
    regions.sort(key=lambda r: (r['chrom'], r['center']))
    deduped = []
    for r in regions:
        if not deduped or r['chrom'] != deduped[-1]['chrom'] or \
           abs(r['center'] - deduped[-1]['center']) > 300:
            deduped.append(r)
        elif r['n_tfs'] > deduped[-1]['n_tfs']:
            deduped[-1] = r
    return deduped


def load_blacklist(blacklist_path):
    """Load ENCODE blacklist regions."""
    blacklist = defaultdict(list)
    if not blacklist_path.exists():
        return blacklist
    with open(blacklist_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                blacklist[parts[0]].append((int(parts[1]), int(parts[2])))
    return blacklist


def in_blacklist(chrom, start, end, blacklist):
    """Check if region overlaps blacklist."""
    for bl_start, bl_end in blacklist.get(chrom, []):
        if start < bl_end and end > bl_start:
            return True
    return False


def extract_sample(chrom, center, peaks_dict, fasta, bigwigs, blacklist,
                   seq_length=2000):
    """
    Extract a multi-TF sample: sequence, composite profile, binding sites.

    Args:
        peaks_dict: {tf_name: peak_info} for TFs in this region
    """
    half = seq_length // 2
    start = center - half
    end = center + half

    chrom_len = fasta.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None

    if in_blacklist(chrom, start, end, blacklist):
        return None

    # Extract sequence
    seq = fasta.fetch(chrom, start, end).upper()
    if len(seq) != seq_length or 'N' * 50 in seq:
        return None

    seq_encoded = one_hot_encode(seq)

    # Build composite profile from all TFs
    composite_profile = np.zeros(seq_length, dtype=np.float32)
    for tf_name, peak in peaks_dict.items():
        if tf_name in bigwigs and bigwigs[tf_name] is not None:
            try:
                signal = np.array(bigwigs[tf_name].values(chrom, start, end),
                                  dtype=np.float32)
                signal = np.nan_to_num(signal, nan=0.0)
                composite_profile += signal
            except Exception:
                # Fallback: Gaussian at peak summit
                summit_pos = peak['summit'] - start
                x = np.arange(seq_length, dtype=np.float32)
                composite_profile += np.exp(-0.5 * ((x - summit_pos) / 30.0) ** 2)
        else:
            summit_pos = peak['summit'] - start
            x = np.arange(seq_length, dtype=np.float32)
            composite_profile += np.exp(-0.5 * ((x - summit_pos) / 30.0) ** 2)

    # Normalize profile
    if composite_profile.max() > 0:
        composite_profile /= composite_profile.max()

    # Build binding sites tensor [N_SLOTS, 3]: (position, tf_id, occupancy)
    binding_sites = np.zeros((N_SLOTS, 3), dtype=np.float32)
    for i, (tf_name, peak) in enumerate(peaks_dict.items()):
        if i >= N_SLOTS:
            break
        summit_pos = peak['summit'] - start
        position = summit_pos / seq_length  # Normalize to [0, 1]
        tf_id = TF_TO_IDX[tf_name]
        occupancy = min(1.0, peak['signal'] / 100.0)
        binding_sites[i] = [position, tf_id, occupancy]

    # Primary TF = highest signal
    primary_tf = max(peaks_dict.items(), key=lambda x: x[1]['signal'])[0]
    tf_index = TF_TO_IDX[primary_tf]

    return {
        'sequence': seq_encoded,
        'profile': composite_profile,
        'binding_sites': binding_sites,
        'tf_index': tf_index,
        'n_tfs': len(peaks_dict),
    }


def extract_single_tf_sample(peak, fasta, bigwigs, blacklist, seq_length=2000):
    """Extract a single-TF sample (same as original pipeline)."""
    chrom = peak['chrom']
    center = peak['summit']
    peaks_dict = {peak['tf']: peak}
    return extract_sample(chrom, center, peaks_dict, fasta, bigwigs, blacklist,
                          seq_length)


def write_hdf5(samples, output_path, tf_vocab):
    """Write samples to HDF5 file."""
    n = len(samples)
    sequences = np.zeros((n, SEQ_LENGTH, 4), dtype=np.float32)
    profiles = np.zeros((n, SEQ_LENGTH), dtype=np.float32)
    binding_sites = np.zeros((n, N_SLOTS, 3), dtype=np.float32)
    tf_indices = np.zeros(n, dtype=np.int64)

    for i, s in enumerate(samples):
        sequences[i] = s['sequence']
        profiles[i] = s['profile']
        binding_sites[i] = s['binding_sites']
        tf_indices[i] = s['tf_index']

    with h5py.File(output_path, "w") as f:
        f.create_dataset("sequences", data=sequences, compression="gzip", chunks=True)
        f.create_dataset("profiles", data=profiles, compression="gzip", chunks=True)
        f.create_dataset("binding_sites", data=binding_sites, compression="gzip", chunks=True)
        f.create_dataset("tf_indices", data=tf_indices, compression="gzip", chunks=True)
        f.create_dataset("tf_vocab", data=json.dumps(tf_vocab))
        f.attrs['n_samples'] = n
        f.attrs['seq_length'] = SEQ_LENGTH
        f.attrs['n_tfs'] = len(tf_vocab)
        f.attrs['n_slots'] = N_SLOTS

    print(f"  Wrote {n} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BEACON-multi training data")
    parser.add_argument("--peak-dir", type=Path,
                        default=Path("data/raw/encode/multi_tf_k562"))
    parser.add_argument("--genome", type=Path,
                        default=Path("data/raw/genome/hg38.fa"))
    parser.add_argument("--blacklist", type=Path,
                        default=Path("data/raw/genome/hg38-blacklist.v2.bed"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/beacon_multi"))
    parser.add_argument("--max-multi-per-split", type=int, default=8000,
                        help="Max multi-TF samples per split")
    parser.add_argument("--max-single-per-tf", type=int, default=1500,
                        help="Max single-TF samples per TF per split")
    parser.add_argument("--window", type=int, default=500,
                        help="Max distance between peaks for co-binding (bp)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    peak_dir = args.peak_dir
    if not peak_dir.is_absolute():
        peak_dir = base_dir / peak_dir

    genome_path = args.genome
    if not genome_path.is_absolute():
        genome_path = base_dir / genome_path

    blacklist_path = args.blacklist
    if not blacklist_path.is_absolute():
        blacklist_path = base_dir / blacklist_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEACON-multi Data Preparation")
    print("=" * 60)

    # Load dependencies
    import pysam
    import pyBigWig

    # Load peaks
    print("\n--- Loading Peaks ---")
    all_peaks = []
    for tf in TF_NAMES:
        peaks = load_peaks(tf, peak_dir)
        print(f"  {tf}: {len(peaks)} peaks")
        all_peaks.extend(peaks)

    # Find multi-TF regions
    print(f"\n--- Finding Multi-TF Regions (window={args.window}bp) ---")
    multi_regions = find_overlap_regions(all_peaks, window=args.window)
    print(f"  Total multi-TF regions: {len(multi_regions)}")

    by_n = Counter(r['n_tfs'] for r in multi_regions)
    for n, count in sorted(by_n.items()):
        print(f"    {n} TFs: {count}")

    # Load genome and bigWig files
    print("\n--- Loading Genome & Signal Files ---")
    fasta = pysam.FastaFile(str(genome_path))
    blacklist = load_blacklist(blacklist_path)

    bigwigs = {}
    for tf in TF_NAMES:
        tf_dir = peak_dir / tf
        bw_files = list(tf_dir.glob("*signal.bigWig")) + list(tf_dir.glob("*.bigWig")) + \
                   list(tf_dir.glob("*.bw"))
        if bw_files:
            try:
                bigwigs[tf] = pyBigWig.open(str(bw_files[0]))
                print(f"  {tf}: {bw_files[0].name}")
            except Exception as e:
                print(f"  {tf}: Failed to open bigWig: {e}")
                bigwigs[tf] = None
        else:
            print(f"  {tf}: No bigWig found (will use Gaussian)")
            bigwigs[tf] = None

    # Process by chromosome split
    for split_name, chroms in [("train", TRAIN_CHROMS), ("val", VAL_CHROMS),
                                ("test", TEST_CHROMS)]:
        print(f"\n--- Processing {split_name} (chroms: {chroms[0]}-{chroms[-1]}) ---")
        chrom_set = set(chroms)

        # 1. Multi-TF samples
        split_multi = [r for r in multi_regions if r['chrom'] in chrom_set]
        # Sort by number of TFs (prefer more diverse) then by signal
        split_multi.sort(key=lambda r: r['n_tfs'], reverse=True)
        split_multi = split_multi[:args.max_multi_per_split]

        multi_samples = []
        for r in split_multi:
            sample = extract_sample(r['chrom'], r['center'], r['peaks'],
                                    fasta, bigwigs, blacklist)
            if sample is not None:
                multi_samples.append(sample)

        print(f"  Multi-TF samples: {len(multi_samples)}")
        multi_tf_dist = Counter(s['n_tfs'] for s in multi_samples)
        for n, count in sorted(multi_tf_dist.items()):
            print(f"    {n} TFs: {count}")

        # 2. Single-TF samples
        single_samples = []
        for tf in TF_NAMES:
            tf_peaks = [p for p in all_peaks if p['tf'] == tf and p['chrom'] in chrom_set]
            tf_peaks.sort(key=lambda p: p['signal'], reverse=True)
            tf_peaks = tf_peaks[:args.max_single_per_tf]

            count = 0
            for peak in tf_peaks:
                sample = extract_single_tf_sample(peak, fasta, bigwigs, blacklist)
                if sample is not None:
                    single_samples.append(sample)
                    count += 1
            print(f"  {tf} single-TF: {count}")

        # Combine and shuffle
        all_samples = multi_samples + single_samples
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in indices]

        n_multi = len(multi_samples)
        n_single = len(single_samples)
        print(f"  Total: {len(all_samples)} ({n_multi} multi + {n_single} single)")

        # Write HDF5
        output_path = output_dir / f"{split_name}.h5"
        write_hdf5(all_samples, output_path, TF_NAMES)

    # Close files
    fasta.close()
    for bw in bigwigs.values():
        if bw is not None:
            bw.close()

    # Save summary
    summary = {
        'tf_vocab': TF_NAMES,
        'n_tfs': len(TF_NAMES),
        'seq_length': SEQ_LENGTH,
        'n_slots': N_SLOTS,
        'overlap_window': args.window,
        'total_multi_regions': len(multi_regions),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Done! Data saved to {output_dir} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
