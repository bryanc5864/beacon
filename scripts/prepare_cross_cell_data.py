#!/usr/bin/env python3
"""
Prepare HepG2 data for cross-cell-line transfer experiment.
Uses same processing pipeline as K562 multi-TF data.
"""

import os
import sys
import json
import gzip
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import pysam

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_peaks(bed_path, min_score=0):
    """Load peaks from BED file."""
    peaks = []
    opener = gzip.open if str(bed_path).endswith('.gz') else open

    with opener(bed_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            score = float(parts[4]) if len(parts) > 4 else 100

            if score >= min_score:
                peaks.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'score': score,
                })

    return peaks


def extract_sequence(fasta, chrom, start, end, seq_len=2000):
    """Extract and center sequence around peak."""
    # Center the window
    center = (start + end) // 2
    window_start = center - seq_len // 2
    window_end = window_start + seq_len

    # Handle boundaries
    if window_start < 0:
        window_start = 0
        window_end = seq_len

    try:
        seq = fasta.fetch(chrom, window_start, window_end).upper()
        if len(seq) < seq_len:
            seq = seq + 'N' * (seq_len - len(seq))
        return seq[:seq_len], window_start
    except:
        return None, None


def extract_signal(bigwig_path, chrom, start, length):
    """Extract signal from bigWig."""
    try:
        import pyBigWig
        bw = pyBigWig.open(str(bigwig_path))
        values = bw.values(chrom, start, start + length)
        bw.close()
        values = np.array(values, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0)
        return values
    except Exception as e:
        return np.zeros(length, dtype=np.float32)


def one_hot_encode(seq):
    """One-hot encode DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            one_hot[i, mapping[base]] = 1.0
        else:
            one_hot[i, :] = 0.25
    return one_hot


def main():
    base_dir = Path(__file__).parent.parent
    cell_line = "hepg2"

    # Paths
    input_dir = base_dir / "data" / "raw" / "encode" / f"multi_tf_{cell_line}"
    output_dir = base_dir / "data" / "processed" / f"multi_tf_{cell_line}"
    output_dir.mkdir(parents=True, exist_ok=True)

    genome_path = base_dir / "data" / "raw" / "genome" / "hg38.fa"

    print(f"{'='*60}")
    print(f"Preparing {cell_line.upper()} data for cross-cell transfer")
    print(f"{'='*60}")

    # Load manifest
    with open(input_dir / "manifest.json") as f:
        manifest = json.load(f)

    tf_names = manifest["tfs"]
    print(f"TFs: {', '.join(tf_names)}")

    # Load genome
    print(f"\nLoading genome...")
    fasta = pysam.FastaFile(str(genome_path))

    # Configuration
    seq_len = 2000
    n_slots = 16
    max_peaks_per_tf = 3000  # Limit for balanced dataset
    test_chroms = {'chr20', 'chr21', 'chr22', 'chrX'}

    # Process each TF
    all_samples = []

    for tf_idx, tf in enumerate(tf_names):
        print(f"\n{tf}:")

        tf_dir = input_dir / tf
        exp_info = manifest["experiments"][tf]

        # Find files
        peaks_file = tf_dir / f"{exp_info['peaks']}.bed.gz"
        signal_file = tf_dir / f"{exp_info['signal']}.bigWig"

        if not peaks_file.exists():
            print(f"  Peaks file not found: {peaks_file}")
            continue
        if not signal_file.exists():
            print(f"  Signal file not found: {signal_file}")
            continue

        # Load peaks
        peaks = load_peaks(peaks_file, min_score=0)
        print(f"  Loaded {len(peaks)} peaks")

        # Sort by score and limit
        peaks = sorted(peaks, key=lambda x: -x['score'])[:max_peaks_per_tf]
        print(f"  Using top {len(peaks)} peaks")

        # Process peaks
        tf_samples = []
        for peak in peaks:
            seq, window_start = extract_sequence(fasta, peak['chrom'],
                                                  peak['start'], peak['end'], seq_len)
            if seq is None:
                continue

            signal = extract_signal(signal_file, peak['chrom'], window_start, seq_len)

            # Create binding site target (center of peak)
            center = (peak['start'] + peak['end']) // 2
            rel_center = center - window_start
            rel_center = max(0, min(seq_len - 1, rel_center))

            # Determine split
            is_test = peak['chrom'] in test_chroms

            tf_samples.append({
                'sequence': one_hot_encode(seq),
                'profile': signal,
                'tf_index': tf_idx,
                'position': rel_center / seq_len,  # Normalized
                'chrom': peak['chrom'],
                'start': window_start,
                'is_test': is_test,
            })

        print(f"  Processed {len(tf_samples)} samples")
        all_samples.extend(tf_samples)

    fasta.close()

    # Split into test set only (for cross-cell transfer)
    test_samples = [s for s in all_samples if s['is_test']]
    print(f"\nTest samples (chr20-22, chrX): {len(test_samples)}")

    # Balance test set
    tf_counts = defaultdict(int)
    for s in test_samples:
        tf_counts[s['tf_index']] += 1

    min_count = min(tf_counts.values())
    print(f"Balancing to {min_count} samples per TF")

    balanced_test = []
    tf_used = defaultdict(int)
    for s in test_samples:
        if tf_used[s['tf_index']] < min_count:
            balanced_test.append(s)
            tf_used[s['tf_index']] += 1

    print(f"Balanced test samples: {len(balanced_test)}")

    # Save to HDF5
    print(f"\nSaving to HDF5...")

    def save_h5(samples, path):
        with h5py.File(path, 'w') as f:
            n = len(samples)
            f.create_dataset('sequences', data=np.array([s['sequence'] for s in samples]))
            f.create_dataset('profiles', data=np.array([s['profile'] for s in samples]))
            f.create_dataset('tf_indices', data=np.array([s['tf_index'] for s in samples]))
            f.create_dataset('positions', data=np.array([s['position'] for s in samples]))

            # Metadata
            f.attrs['n_samples'] = n
            f.attrs['seq_len'] = seq_len
            f.attrs['n_tfs'] = len(tf_names)
            f.attrs['tf_names'] = json.dumps(tf_names)

    save_h5(balanced_test, output_dir / "test.h5")

    # Save config
    config = {
        "cell_line": cell_line,
        "tf_names": tf_names,
        "n_tfs": len(tf_names),
        "seq_len": seq_len,
        "n_slots": n_slots,
        "test_samples": len(balanced_test),
        "source": "ENCODE",
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"TFs: {tf_names}")
    print(f"Test samples: {len(balanced_test)} ({min_count} per TF)")
    print(f"\nReady for cross-cell transfer experiment!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
