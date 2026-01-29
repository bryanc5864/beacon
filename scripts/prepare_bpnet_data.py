#!/usr/bin/env python3
"""
Prepare CTCF K562 data in BPNet format.

BPNet expects:
- narrowPeak file for peak regions
- bigWig file for signal profiles
- Reference genome (hg38)

We'll create a JSON configuration file that points to our existing data.
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Our existing ENCODE data
ENCODE_DIR = DATA_DIR / "raw" / "encode"
CTCF_DIR = ENCODE_DIR / "CTCF_K562"

# BPNet output directory
BPNET_DIR = DATA_DIR / "processed" / "bpnet_ctcf"


def find_files():
    """Find our existing CTCF data files."""
    # Find narrowPeak
    peak_files = list(CTCF_DIR.glob("*.narrowPeak*"))
    if not peak_files:
        peak_files = list(CTCF_DIR.glob("*peaks*.bed*"))

    # Find bigWig
    signal_files = list(CTCF_DIR.glob("*.bigWig")) + list(CTCF_DIR.glob("*.bw"))

    return peak_files, signal_files


def create_bpnet_config(peak_file, signal_file, output_dir):
    """Create BPNet JSON configuration."""

    # BPNet config format
    config = {
        "task0_plus": {
            "strand": 0,
            "task_id": 0,
            "signal": str(signal_file.absolute()),
            "peaks": str(peak_file.absolute()),
            "control": None
        }
    }

    # For single-stranded ChIP-seq, we use same file for both strands
    # or just use one strand

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "bpnet_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"BPNet config saved to: {config_path}")
    return config_path


def create_splits(peak_file, output_dir):
    """Create train/val/test chromosome splits matching BEACON."""
    import gzip

    # Read peaks
    opener = gzip.open if str(peak_file).endswith('.gz') else open
    mode = 'rt' if str(peak_file).endswith('.gz') else 'r'

    train_chroms = {f"chr{i}" for i in range(1, 18)}  # chr1-17
    val_chroms = {"chr18", "chr19"}
    test_chroms = {"chr20", "chr21", "chr22", "chrX"}

    train_peaks = []
    val_peaks = []
    test_peaks = []

    with opener(peak_file, mode) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]

            if chrom in train_chroms:
                train_peaks.append(line)
            elif chrom in val_chroms:
                val_peaks.append(line)
            elif chrom in test_chroms:
                test_peaks.append(line)

    # Save split files
    for name, peaks in [("train", train_peaks), ("val", val_peaks), ("test", test_peaks)]:
        out_path = output_dir / f"{name}_peaks.bed"
        with open(out_path, "w") as f:
            f.writelines(peaks)
        print(f"  {name}: {len(peaks)} peaks -> {out_path.name}")

    return len(train_peaks), len(val_peaks), len(test_peaks)


def main():
    print("=" * 60)
    print("Preparing BPNet Data (CTCF K562)")
    print("=" * 60)

    # Find existing files
    peak_files, signal_files = find_files()

    if not peak_files:
        print(f"ERROR: No peak files found in {CTCF_DIR}")
        print("Run: python scripts/download_ctcf_k562.py first")
        return 1

    if not signal_files:
        print(f"ERROR: No signal files found in {CTCF_DIR}")
        print("Run: python scripts/download_ctcf_k562.py first")
        return 1

    peak_file = peak_files[0]
    signal_file = signal_files[0]

    print(f"\nUsing:")
    print(f"  Peaks: {peak_file.name}")
    print(f"  Signal: {signal_file.name}")

    # Create BPNet config
    print(f"\nCreating BPNet configuration...")
    config_path = create_bpnet_config(peak_file, signal_file, BPNET_DIR)

    # Create chromosome splits
    print(f"\nCreating train/val/test splits...")
    n_train, n_val, n_test = create_splits(peak_file, BPNET_DIR)

    # Save summary
    summary = {
        "peak_file": str(peak_file),
        "signal_file": str(signal_file),
        "config_file": str(config_path),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_chroms": "chr1-17",
        "val_chroms": "chr18-19",
        "test_chroms": "chr20-22,chrX",
    }

    with open(BPNET_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 60)
    print("BPNet Data Preparation Complete")
    print("=" * 60)
    print(f"Output: {BPNET_DIR}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"\nNext: python scripts/train_bpnet.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
