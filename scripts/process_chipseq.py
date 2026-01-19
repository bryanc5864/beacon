#!/usr/bin/env python3
"""
Process ENCODE ChIP-seq data into training-ready format for BEACON.

This script:
1. Reads narrowPeak files for each TF
2. Extracts sequences centered on peak summits
3. Creates HDF5 dataset with sequences, profiles, and metadata
"""

import sys
import json
import gzip
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_narrowpeak(filepath: Path) -> list:
    """Parse narrowPeak file, return list of peak dicts."""
    peaks = []
    opener = gzip.open if str(filepath).endswith('.gz') else open

    with opener(filepath, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue

            peak = {
                'chrom': fields[0],
                'start': int(fields[1]),
                'end': int(fields[2]),
                'name': fields[3],
                'score': float(fields[4]) if fields[4] != '.' else 0,
                'strand': fields[5],
                'signal': float(fields[6]) if fields[6] != '.' else 0,
                'pvalue': float(fields[7]) if fields[7] != '.' else 0,
                'qvalue': float(fields[8]) if fields[8] != '.' else 0,
                'summit': int(fields[9]) if fields[9] != '.' else (int(fields[2]) - int(fields[1])) // 2,
            }
            peaks.append(peak)

    return peaks


def extract_sequence(fasta, chrom: str, center: int, flank: int = 500) -> str:
    """Extract sequence centered on position."""
    start = max(0, center - flank)
    end = center + flank

    if chrom not in fasta:
        return None

    chrom_len = len(fasta[chrom])
    if end > chrom_len:
        return None

    seq = str(fasta[chrom][start:end])
    return seq.upper()


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)

    for i, base in enumerate(sequence):
        if base in mapping:
            encoding[i, mapping[base]] = 1.0
        else:
            encoding[i, :] = 0.25  # N or other

    return encoding


def create_gaussian_profile(length: int, center: int, sigma: float = 50) -> np.ndarray:
    """Create Gaussian profile centered at position."""
    x = np.arange(length)
    profile = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return profile.astype(np.float32)


def main():
    import pyfaidx

    data_dir = Path(__file__).parent.parent / "data"
    encode_dir = data_dir / "raw" / "encode"
    genome_path = data_dir / "raw" / "genome" / "hg38.fa"
    output_dir = data_dir / "processed" / "chipseq"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check genome exists
    if not genome_path.exists():
        print(f"ERROR: Genome not found at {genome_path}")
        return 1

    print(f"Loading genome from {genome_path}")
    fasta = pyfaidx.Fasta(str(genome_path))
    print(f"  Loaded {len(fasta.keys())} chromosomes")

    # Find all TF directories with peak files
    tf_dirs = [d for d in encode_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(tf_dirs)} TF directories")

    # Process each TF
    all_data = defaultdict(list)
    tf_labels = []

    for tf_dir in tf_dirs:
        tf_name = tf_dir.name
        peak_files = list(tf_dir.glob("*.narrowPeak.gz"))

        if not peak_files:
            continue

        print(f"\nProcessing {tf_name}: {len(peak_files)} peak files")

        tf_sequences = []
        tf_profiles = []
        tf_metadata = []

        for peak_file in peak_files:
            peaks = parse_narrowpeak(peak_file)
            print(f"  {peak_file.name}: {len(peaks)} peaks")

            # Sort by signal and take top peaks
            peaks = sorted(peaks, key=lambda x: -x['signal'])
            peaks = peaks[:2000]  # Limit per file

            for peak in peaks:
                # Calculate summit position in genomic coordinates
                summit_pos = peak['start'] + peak['summit']

                # Extract sequence
                seq = extract_sequence(fasta, peak['chrom'], summit_pos, flank=500)
                if seq is None or len(seq) != 1000:
                    continue

                # Encode sequence
                encoded = one_hot_encode(seq)
                tf_sequences.append(encoded)

                # Create binding profile (Gaussian at center)
                profile = create_gaussian_profile(1000, 500, sigma=50)
                # Scale by signal strength
                profile *= min(peak['signal'] / 100, 1.0)
                tf_profiles.append(profile)

                # Store metadata
                tf_metadata.append({
                    'tf': tf_name,
                    'chrom': peak['chrom'],
                    'start': peak['start'],
                    'end': peak['end'],
                    'summit': summit_pos,
                    'signal': peak['signal'],
                    'qvalue': peak['qvalue'],
                })

        if tf_sequences:
            all_data['sequences'].extend(tf_sequences)
            all_data['profiles'].extend(tf_profiles)
            all_data['metadata'].extend(tf_metadata)
            tf_labels.extend([tf_name] * len(tf_sequences))
            print(f"  Total: {len(tf_sequences)} valid sequences")

    if not all_data['sequences']:
        print("\nNo sequences extracted!")
        return 1

    # Convert to arrays
    print(f"\nTotal sequences: {len(all_data['sequences'])}")

    sequences = np.stack(all_data['sequences'], axis=0)
    profiles = np.stack(all_data['profiles'], axis=0)

    print(f"Sequences shape: {sequences.shape}")
    print(f"Profiles shape: {profiles.shape}")

    # Create TF label encoding
    unique_tfs = sorted(set(tf_labels))
    tf_to_idx = {tf: i for i, tf in enumerate(unique_tfs)}
    tf_indices = np.array([tf_to_idx[tf] for tf in tf_labels])

    print(f"\nTF distribution:")
    for tf in unique_tfs:
        count = tf_labels.count(tf)
        print(f"  {tf}: {count}")

    # Split into train/val/test (80/10/10)
    n_samples = len(sequences)
    indices = np.random.RandomState(42).permutation(n_samples)

    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save datasets
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        output_path = output_dir / f"{split_name}.h5"
        print(f"\nSaving {split_name} to {output_path}")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('sequences', data=sequences[split_idx], compression='gzip')
            f.create_dataset('profiles', data=profiles[split_idx], compression='gzip')
            f.create_dataset('tf_indices', data=tf_indices[split_idx])

            # Store metadata as JSON
            split_metadata = [all_data['metadata'][i] for i in split_idx]
            f.create_dataset('metadata', data=json.dumps(split_metadata))

            # Store TF vocabulary
            f.create_dataset('tf_vocab', data=json.dumps(unique_tfs))

    # Save summary
    summary = {
        'total_sequences': n_samples,
        'sequence_length': 1000,
        'n_tfs': len(unique_tfs),
        'tf_names': unique_tfs,
        'splits': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx),
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {summary_path}")
    print("\nChIP-seq processing complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
