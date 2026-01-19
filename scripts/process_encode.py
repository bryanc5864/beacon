#!/usr/bin/env python3
"""
Process ENCODE ChIP-seq data for BEACON training.

Pipeline:
1. Load peak files (narrowPeak) and signal tracks (bigWig)
2. Extract sequences around peak summits
3. Generate binding profiles from signal tracks
4. Split by chromosome for train/val/test
5. Save to HDF5 format

Chromosome splits (following ENCODE DREAM challenge convention):
- Train: chr1-17
- Validation: chr18-19
- Test: chr20-22, chrX
"""

import os
import sys
import argparse
import json
import gzip
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Try to import optional dependencies
try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False
    print("Warning: pyBigWig not installed. Install with: pip install pyBigWig")

try:
    import pyfaidx
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False
    print("Warning: pyfaidx not installed. Install with: pip install pyfaidx")


# DNA alphabet
DNA_ALPHABET = "ACGT"
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}

# Chromosome split configuration
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18)]
VAL_CHROMS = ["chr18", "chr19"]
TEST_CHROMS = ["chr20", "chr21", "chr22", "chrX"]


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a DNA sequence."""
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in DNA_ALPHABET:
            encoding[i, DNA_ALPHABET.index(base)] = 1.0
        else:
            encoding[i, :] = 0.25
    return encoding


def parse_narrowpeak(filepath: Path) -> List[Dict]:
    """
    Parse a narrowPeak file.

    narrowPeak format (BED6+4):
    chrom, start, end, name, score, strand, signalValue, pValue, qValue, peak

    Returns list of peak dictionaries.
    """
    peaks = []

    opener = gzip.open if str(filepath).endswith('.gz') else open
    mode = 'rt' if str(filepath).endswith('.gz') else 'r'

    with opener(filepath, mode) as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue

            try:
                peak = {
                    'chrom': fields[0],
                    'start': int(fields[1]),
                    'end': int(fields[2]),
                    'name': fields[3],
                    'score': float(fields[4]) if fields[4] != '.' else 0.0,
                    'strand': fields[5] if fields[5] in ['+', '-'] else '.',
                    'signal_value': float(fields[6]) if fields[6] != '.' else 0.0,
                    'p_value': float(fields[7]) if fields[7] != '.' else 0.0,
                    'q_value': float(fields[8]) if fields[8] != '.' else 0.0,
                    'summit': int(fields[9]) if fields[9] != '.' else (int(fields[2]) - int(fields[1])) // 2,
                }
                peaks.append(peak)
            except (ValueError, IndexError) as e:
                continue

    return peaks


def extract_sequence(
    fasta: 'pyfaidx.Fasta',
    chrom: str,
    center: int,
    flank: int,
) -> Optional[str]:
    """Extract sequence from FASTA around a center position."""
    try:
        start = max(0, center - flank)
        end = center + flank

        if chrom not in fasta:
            # Try without 'chr' prefix or with it
            alt_chrom = chrom[3:] if chrom.startswith('chr') else f"chr{chrom}"
            if alt_chrom in fasta:
                chrom = alt_chrom
            else:
                return None

        chrom_len = len(fasta[chrom])
        if end > chrom_len:
            end = chrom_len
            start = max(0, end - 2 * flank)

        seq = str(fasta[chrom][start:end]).upper()

        # Pad if necessary
        if len(seq) < 2 * flank:
            pad_needed = 2 * flank - len(seq)
            seq = 'N' * (pad_needed // 2) + seq + 'N' * (pad_needed - pad_needed // 2)

        return seq
    except Exception as e:
        return None


def extract_profile_from_bigwig(
    bw,
    chrom: str,
    center: int,
    flank: int,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """Extract signal profile from bigWig around a center position."""
    try:
        start = max(0, center - flank)
        end = center + flank

        # Get chromosome length
        chroms = bw.chroms()
        if chrom not in chroms:
            alt_chrom = chrom[3:] if chrom.startswith('chr') else f"chr{chrom}"
            if alt_chrom in chroms:
                chrom = alt_chrom
            else:
                return None

        chrom_len = chroms[chrom]
        if end > chrom_len:
            end = chrom_len
            start = max(0, end - 2 * flank)

        # Extract values
        values = bw.values(chrom, start, end)
        profile = np.array(values, dtype=np.float32)

        # Handle NaN values
        profile = np.nan_to_num(profile, nan=0.0)

        # Pad if necessary
        target_len = 2 * flank
        if len(profile) < target_len:
            pad_needed = target_len - len(profile)
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            profile = np.pad(profile, (pad_left, pad_right), mode='constant', constant_values=0)

        # Normalize
        if normalize and profile.max() > 0:
            profile = profile / profile.max()

        return profile
    except Exception as e:
        return None


def create_synthetic_profile(
    seq_length: int,
    peak_position: int,
    sigma: float = 30.0,
) -> np.ndarray:
    """Create a synthetic Gaussian profile when bigWig not available."""
    x = np.arange(seq_length)
    profile = np.exp(-0.5 * ((x - peak_position) / sigma) ** 2)
    return profile.astype(np.float32)


class ENCODEProcessor:
    """Process ENCODE ChIP-seq data for BEACON training."""

    def __init__(
        self,
        data_dir: Path,
        genome_path: Optional[Path] = None,
        seq_length: int = 2000,
        min_peak_score: float = 0.0,
        max_peaks_per_tf: int = 50000,
    ):
        self.data_dir = Path(data_dir)
        self.genome_path = genome_path
        self.seq_length = seq_length
        self.flank = seq_length // 2
        self.min_peak_score = min_peak_score
        self.max_peaks_per_tf = max_peaks_per_tf

        # Load genome if available
        self.fasta = None
        if genome_path and HAS_PYFAIDX:
            try:
                self.fasta = pyfaidx.Fasta(str(genome_path))
                print(f"Loaded genome: {genome_path}")
            except Exception as e:
                print(f"Warning: Could not load genome: {e}")

    def discover_data_files(self) -> Dict[str, Dict[str, List[Path]]]:
        """Discover peak and signal files organized by TF."""
        data_files = defaultdict(lambda: {'peaks': [], 'signals': []})

        # Look for narrowPeak files
        for peak_file in self.data_dir.rglob("*.narrowPeak*"):
            # Try to extract TF name from path or filename
            tf_name = self._extract_tf_name(peak_file)
            if tf_name:
                data_files[tf_name]['peaks'].append(peak_file)

        # Look for bigWig files
        for bw_file in self.data_dir.rglob("*.bigWig"):
            tf_name = self._extract_tf_name(bw_file)
            if tf_name:
                data_files[tf_name]['signals'].append(bw_file)

        for bw_file in self.data_dir.rglob("*.bw"):
            tf_name = self._extract_tf_name(bw_file)
            if tf_name:
                data_files[tf_name]['signals'].append(bw_file)

        return dict(data_files)

    def _extract_tf_name(self, filepath: Path) -> Optional[str]:
        """Extract TF name from file path."""
        # Try parent directory name first
        parent_name = filepath.parent.name
        if parent_name and parent_name not in ['encode', 'raw', 'processed']:
            return parent_name

        # Try parsing filename
        name = filepath.name
        parts = name.split('_')
        if parts:
            return parts[0]

        return None

    def process_tf(
        self,
        tf_name: str,
        peak_files: List[Path],
        signal_files: List[Path],
    ) -> Dict[str, List]:
        """Process data for a single TF."""
        all_peaks = []

        # Load all peaks
        for peak_file in peak_files:
            peaks = parse_narrowpeak(peak_file)
            all_peaks.extend(peaks)

        print(f"  {tf_name}: {len(all_peaks)} peaks from {len(peak_files)} files")

        if not all_peaks:
            return {'train': [], 'val': [], 'test': []}

        # Filter by score
        if self.min_peak_score > 0:
            all_peaks = [p for p in all_peaks if p['signal_value'] >= self.min_peak_score]
            print(f"    After score filter: {len(all_peaks)} peaks")

        # Sort by signal value and limit
        all_peaks = sorted(all_peaks, key=lambda x: -x['signal_value'])
        if len(all_peaks) > self.max_peaks_per_tf:
            all_peaks = all_peaks[:self.max_peaks_per_tf]
            print(f"    After limit: {len(all_peaks)} peaks")

        # Load bigWig if available
        bw = None
        if signal_files and HAS_PYBIGWIG:
            try:
                bw = pyBigWig.open(str(signal_files[0]))
            except Exception as e:
                print(f"    Warning: Could not open bigWig: {e}")

        # Process peaks and split by chromosome
        results = {'train': [], 'val': [], 'test': []}

        for peak in tqdm(all_peaks, desc=f"    Processing {tf_name}", leave=False):
            chrom = peak['chrom']

            # Determine split
            if chrom in TRAIN_CHROMS:
                split = 'train'
            elif chrom in VAL_CHROMS:
                split = 'val'
            elif chrom in TEST_CHROMS:
                split = 'test'
            else:
                continue  # Skip other chromosomes

            # Get center position (summit)
            center = peak['start'] + peak['summit']

            # Extract sequence
            if self.fasta:
                seq = extract_sequence(self.fasta, chrom, center, self.flank)
            else:
                # Generate random sequence as placeholder
                seq = ''.join(np.random.choice(list('ACGT'), size=self.seq_length))

            if seq is None or len(seq) != self.seq_length:
                continue

            # Extract profile
            if bw:
                profile = extract_profile_from_bigwig(bw, chrom, center, self.flank)
            else:
                profile = create_synthetic_profile(self.seq_length, self.flank)

            if profile is None or len(profile) != self.seq_length:
                continue

            results[split].append({
                'sequence': seq,
                'profile': profile,
                'chrom': chrom,
                'position': center,
                'signal_value': peak['signal_value'],
            })

        if bw:
            bw.close()

        return results

    def process_all(self, output_dir: Path) -> Dict:
        """Process all TFs and create datasets."""
        data_files = self.discover_data_files()

        if not data_files:
            print("No data files found!")
            return {}

        print(f"Found {len(data_files)} TFs:")
        for tf_name, files in data_files.items():
            print(f"  {tf_name}: {len(files['peaks'])} peak files, {len(files['signals'])} signal files")

        # Process each TF
        all_data = {'train': [], 'val': [], 'test': []}
        tf_indices = {}
        tf_vocab = []

        for tf_idx, (tf_name, files) in enumerate(data_files.items()):
            print(f"\nProcessing {tf_name}...")
            tf_data = self.process_tf(tf_name, files['peaks'], files['signals'])

            # Add TF index to each sample
            for split in ['train', 'val', 'test']:
                for sample in tf_data[split]:
                    sample['tf_idx'] = tf_idx
                all_data[split].extend(tf_data[split])

            tf_indices[tf_name] = tf_idx
            tf_vocab.append(tf_name)

        # Print summary
        print("\n" + "=" * 60)
        print("Processing Summary:")
        print(f"  TFs: {len(tf_vocab)}")
        print(f"  Train samples: {len(all_data['train'])}")
        print(f"  Val samples: {len(all_data['val'])}")
        print(f"  Test samples: {len(all_data['test'])}")

        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test']:
            if not all_data[split]:
                continue

            output_path = output_dir / f"{split}.h5"
            self._save_hdf5(all_data[split], output_path, tf_vocab)
            print(f"Saved {split} data to: {output_path}")

        # Save summary
        summary = {
            'n_tfs': len(tf_vocab),
            'tfs': tf_vocab,
            'tf_indices': tf_indices,
            'seq_length': self.seq_length,
            'n_train': len(all_data['train']),
            'n_val': len(all_data['val']),
            'n_test': len(all_data['test']),
            'train_chroms': TRAIN_CHROMS,
            'val_chroms': VAL_CHROMS,
            'test_chroms': TEST_CHROMS,
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def _save_hdf5(
        self,
        data: List[Dict],
        output_path: Path,
        tf_vocab: List[str],
    ):
        """Save data to HDF5 format."""
        n_samples = len(data)

        sequences = np.zeros((n_samples, self.seq_length, 4), dtype=np.float32)
        profiles = np.zeros((n_samples, self.seq_length), dtype=np.float32)
        tf_indices = np.zeros(n_samples, dtype=np.int64)

        for i, sample in enumerate(tqdm(data, desc="Encoding sequences")):
            sequences[i] = one_hot_encode(sample['sequence'])
            profiles[i] = sample['profile']
            tf_indices[i] = sample['tf_idx']

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('sequences', data=sequences, compression='gzip')
            f.create_dataset('profiles', data=profiles, compression='gzip')
            f.create_dataset('tf_indices', data=tf_indices, compression='gzip')
            f.create_dataset('tf_vocab', data=json.dumps(tf_vocab))

            f.attrs['n_samples'] = n_samples
            f.attrs['seq_length'] = self.seq_length
            f.attrs['n_tfs'] = len(tf_vocab)


def main():
    parser = argparse.ArgumentParser(
        description="Process ENCODE ChIP-seq data for BEACON training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_encode.py                           # Process all ENCODE data
  python scripts/process_encode.py --genome path/to/hg38.fa  # Use reference genome
  python scripts/process_encode.py --seq-length 2000         # Longer sequences
        """
    )

    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "raw" / "encode",
                        help="Directory containing ENCODE downloads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "processed" / "encode",
                        help="Output directory for processed data")
    parser.add_argument("--genome", type=Path, default=None,
                        help="Path to reference genome FASTA")
    parser.add_argument("--seq-length", type=int, default=2000,
                        help="Output sequence length")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum peak signal value")
    parser.add_argument("--max-peaks", type=int, default=50000,
                        help="Maximum peaks per TF")

    args = parser.parse_args()

    print("=" * 70)
    print("ENCODE Data Processor for BEACON")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Genome: {args.genome or 'Not provided (will use synthetic profiles)'}")
    print(f"Sequence length: {args.seq_length}")
    print()

    if not args.data_dir.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print("Please run: python scripts/download_encode.py first")
        return 1

    processor = ENCODEProcessor(
        data_dir=args.data_dir,
        genome_path=args.genome,
        seq_length=args.seq_length,
        min_peak_score=args.min_score,
        max_peaks_per_tf=args.max_peaks,
    )

    summary = processor.process_all(args.output_dir)

    if summary:
        print("\n" + "=" * 70)
        print("Processing complete!")
        print("=" * 70)
        print(f"\nTo train BEACON on this data:")
        print(f"  python scripts/train.py --data-dir {args.output_dir} --seq-len {args.seq_length}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
