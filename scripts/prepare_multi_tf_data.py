#!/usr/bin/env python3
"""
Prepare multi-TF K562 ChIP-seq data for BEACON training.

Phase 2B: Multi-TF discrimination.

Processes 7 TFs from ENCODE K562 ChIP-seq:
- CTCF, GATA1, TAL1, MYC, MAX, SPI1, CEBPB

Each TF gets its own peaks and signal profiles, combined into
a single dataset with TF identity labels for discrimination testing.

Key tests:
- Can BEACON distinguish different TF families? (GATA1 vs MYC)
- Can BEACON distinguish same-family TFs? (MYC vs MAX, both bHLH)
- Can BEACON handle varying peak counts per TF?
"""

import os
import sys
import gzip
import json
import h5py
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    import pyfaidx
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False

try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False

# Constants
DNA_ALPHABET = "ACGT"
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18)]
VAL_CHROMS = ["chr18", "chr19"]
TEST_CHROMS = ["chr20", "chr21", "chr22", "chrX"]
ALL_CHROMS = TRAIN_CHROMS + VAL_CHROMS + TEST_CHROMS

TF_PANEL = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
TF_FAMILIES = {
    "CTCF": "Zinc finger",
    "GATA1": "GATA",
    "TAL1": "bHLH",
    "MYC": "bHLH",
    "MAX": "bHLH",
    "SPI1": "ETS",
    "CEBPB": "bZIP",
}

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GENOME = PROJECT_ROOT / "data" / "raw" / "genome" / "hg38.fa"
DEFAULT_BLACKLIST = PROJECT_ROOT / "data" / "raw" / "genome" / "hg38-blacklist.v2.bed"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "encode" / "multi_tf_k562"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "multi_tf_k562"


def one_hot_encode(sequence):
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            encoding[i, base_to_idx[base]] = 1.0
        else:
            encoding[i, :] = 0.25
    return encoding


def parse_narrowpeak(filepath):
    peaks = []
    opener = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    with opener(filepath, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            try:
                peak = {
                    "chrom": fields[0],
                    "start": int(fields[1]),
                    "end": int(fields[2]),
                    "signal_value": float(fields[6]) if fields[6] != "." else 0.0,
                    "summit_offset": int(fields[9]) if fields[9] != "." else (int(fields[2]) - int(fields[1])) // 2,
                }
                peak["summit"] = peak["start"] + peak["summit_offset"]
                peaks.append(peak)
            except (ValueError, IndexError):
                continue
    return peaks


def load_blacklist(filepath):
    blacklist = defaultdict(list)
    if not filepath or not filepath.exists():
        return blacklist
    opener = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    with opener(filepath, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) >= 3:
                blacklist[fields[0]].append((int(fields[1]), int(fields[2])))
    return blacklist


def is_blacklisted(chrom, start, end, blacklist):
    for bl_start, bl_end in blacklist.get(chrom, []):
        if start < bl_end and end > bl_start:
            return True
    return False


def create_gaussian_profile(seq_length, peak_position, sigma=30.0):
    x = np.arange(seq_length, dtype=np.float32)
    return np.exp(-0.5 * ((x - peak_position) / sigma) ** 2)


class MultiTFDataPreparer:
    def __init__(self, genome_path, data_dir, output_dir, blacklist_path=None,
                 seq_length=2000, max_peaks_per_tf=20000, min_signal=0.0,
                 balance_tfs=True):
        self.genome_path = genome_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.blacklist_path = blacklist_path
        self.seq_length = seq_length
        self.flank = seq_length // 2
        self.max_peaks_per_tf = max_peaks_per_tf
        self.min_signal = min_signal
        self.balance_tfs = balance_tfs

        self.stats = defaultdict(lambda: defaultdict(int))

    def run(self):
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("Multi-TF K562 Data Preparation for BEACON (Phase 2B)")
        logger.info("=" * 70)
        logger.info(f"Genome: {self.genome_path}")
        logger.info(f"Data dir: {self.data_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Sequence length: {self.seq_length}")
        logger.info(f"Max peaks per TF: {self.max_peaks_per_tf}")
        logger.info(f"Balance TFs: {self.balance_tfs}")
        logger.info("")

        # Load genome
        logger.info("Loading genome...")
        fasta = pyfaidx.Fasta(str(self.genome_path))
        logger.info(f"  {len(fasta.keys())} chromosomes")

        # Load blacklist
        blacklist = load_blacklist(self.blacklist_path)
        logger.info(f"  Blacklist: {sum(len(v) for v in blacklist.values())} regions")

        # Discover and process each TF
        tf_vocab = []
        available_tfs = []

        for tf in TF_PANEL:
            tf_dir = self.data_dir / tf
            if not tf_dir.exists():
                logger.warning(f"  {tf}: directory not found, skipping")
                continue

            peak_files = list(tf_dir.glob("*.narrowPeak*"))
            signal_files = list(tf_dir.glob("*.bigWig")) + list(tf_dir.glob("*.bw"))

            if not peak_files:
                logger.warning(f"  {tf}: no peak files found, skipping")
                continue

            available_tfs.append(tf)
            logger.info(f"  {tf}: {len(peak_files)} peak file(s), {len(signal_files)} signal file(s)")

        if not available_tfs:
            logger.error("No TFs with data found!")
            return False

        logger.info(f"\nProcessing {len(available_tfs)} TFs: {', '.join(available_tfs)}")
        logger.info("")

        # Process each TF
        all_data = {"train": [], "val": [], "test": []}
        tf_peak_counts = {}

        for tf_idx, tf in enumerate(available_tfs):
            tf_vocab.append(tf)
            tf_dir = self.data_dir / tf

            logger.info(f"{'='*50}")
            logger.info(f"Processing {tf} (index {tf_idx}, family: {TF_FAMILIES.get(tf, '?')})")
            logger.info(f"{'='*50}")

            # Parse peaks
            peak_files = list(tf_dir.glob("*.narrowPeak*"))
            all_peaks = []
            for pf in peak_files:
                # Resolve symlinks
                actual_path = pf.resolve()
                peaks = parse_narrowpeak(actual_path)
                logger.info(f"  Parsed {len(peaks)} peaks from {pf.name}")
                all_peaks.extend(peaks)

            self.stats[tf]["total_parsed"] = len(all_peaks)

            # Filter
            valid_chroms = set(ALL_CHROMS)
            all_peaks = [p for p in all_peaks if p["chrom"] in valid_chroms]
            self.stats[tf]["after_chrom"] = len(all_peaks)

            all_peaks = [p for p in all_peaks
                         if not is_blacklisted(p["chrom"], p["summit"] - self.flank,
                                               p["summit"] + self.flank, blacklist)]
            self.stats[tf]["after_blacklist"] = len(all_peaks)

            if self.min_signal > 0:
                all_peaks = [p for p in all_peaks if p["signal_value"] >= self.min_signal]

            # Filter by genome boundaries
            all_peaks = [p for p in all_peaks
                         if p["chrom"] in fasta
                         and p["summit"] - self.flank >= 0
                         and p["summit"] + self.flank <= len(fasta[p["chrom"]])]
            self.stats[tf]["after_filter"] = len(all_peaks)

            # Sort by signal, limit
            all_peaks = sorted(all_peaks, key=lambda x: -x["signal_value"])
            if len(all_peaks) > self.max_peaks_per_tf:
                all_peaks = all_peaks[:self.max_peaks_per_tf]

            tf_peak_counts[tf] = len(all_peaks)
            logger.info(f"  Peaks after filtering: {len(all_peaks)}")

            if not all_peaks:
                logger.warning(f"  No peaks remaining for {tf}!")
                continue

            # Load bigWig
            bw = None
            signal_files = list(tf_dir.glob("*.bigWig")) + list(tf_dir.glob("*.bw"))
            if signal_files:
                try:
                    actual_path = signal_files[0].resolve()
                    bw = pyBigWig.open(str(actual_path))
                    logger.info(f"  Loaded bigWig: {signal_files[0].name}")
                except Exception as e:
                    logger.warning(f"  Could not load bigWig: {e}")

            # Extract data
            n_processed = 0
            log_interval = max(1, len(all_peaks) // 10)

            for i, peak in enumerate(all_peaks):
                if i % log_interval == 0:
                    logger.info(f"  Progress: {i}/{len(all_peaks)} ({100*i/len(all_peaks):.0f}%)")

                chrom = peak["chrom"]
                summit = peak["summit"]
                start = summit - self.flank
                end = summit + self.flank

                # Determine split
                if chrom in TRAIN_CHROMS:
                    split = "train"
                elif chrom in VAL_CHROMS:
                    split = "val"
                elif chrom in TEST_CHROMS:
                    split = "test"
                else:
                    continue

                # Extract sequence
                try:
                    seq = str(fasta[chrom][start:end]).upper()
                    if len(seq) != self.seq_length:
                        continue
                except:
                    continue

                # Extract profile
                profile = None
                if bw:
                    try:
                        values = bw.values(chrom, start, end)
                        profile = np.array(values, dtype=np.float32)
                        profile = np.nan_to_num(profile, nan=0.0)
                        if len(profile) != self.seq_length:
                            continue
                        if profile.max() > 0:
                            profile = profile / profile.max()
                    except:
                        profile = None

                if profile is None:
                    profile = create_gaussian_profile(self.seq_length, self.flank)

                seq_encoded = one_hot_encode(seq)
                occ = min(1.0, peak["signal_value"] / 100.0)

                binding_site = np.array([0.5, tf_idx, occ], dtype=np.float32)

                all_data[split].append({
                    "sequence": seq_encoded,
                    "profile": profile,
                    "binding_sites": binding_site,
                    "tf_idx": tf_idx,
                    "tf_name": tf,
                    "chrom": chrom,
                    "position": summit,
                    "signal_value": peak["signal_value"],
                })

                n_processed += 1
                self.stats[tf][f"split_{split}"] += 1

            if bw:
                bw.close()

            self.stats[tf]["processed"] = n_processed
            logger.info(f"  Processed: {n_processed}")
            for split in ["train", "val", "test"]:
                logger.info(f"    {split}: {self.stats[tf][f'split_{split}']}")

        # Balance TFs if requested
        if self.balance_tfs and len(available_tfs) > 1:
            logger.info(f"\nBalancing TFs across splits...")
            for split in ["train", "val", "test"]:
                tf_counts = Counter(s["tf_name"] for s in all_data[split])
                if not tf_counts:
                    continue
                min_count = min(tf_counts.values())
                logger.info(f"  {split}: counts before balance: {dict(tf_counts)}")
                logger.info(f"  {split}: balancing to {min_count} per TF")

                # Downsample to min_count per TF
                balanced = []
                for tf in available_tfs:
                    tf_samples = [s for s in all_data[split] if s["tf_name"] == tf]
                    if len(tf_samples) > min_count:
                        # Keep top by signal
                        tf_samples = sorted(tf_samples, key=lambda x: -x["signal_value"])[:min_count]
                    balanced.extend(tf_samples)

                all_data[split] = balanced
                tf_counts_after = Counter(s["tf_name"] for s in all_data[split])
                logger.info(f"  {split}: counts after balance: {dict(tf_counts_after)}")

        # Save
        logger.info(f"\nSaving datasets...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        n_slots = 16

        for split in ["train", "val", "test"]:
            samples = all_data[split]
            if not samples:
                logger.warning(f"  No samples for {split}")
                continue

            n = len(samples)
            output_path = self.output_dir / f"{split}.h5"
            logger.info(f"  {split}: {n} samples -> {output_path}")

            sequences = np.zeros((n, self.seq_length, 4), dtype=np.float32)
            profiles = np.zeros((n, self.seq_length), dtype=np.float32)
            binding_sites = np.zeros((n, n_slots, 3), dtype=np.float32)
            tf_indices = np.zeros(n, dtype=np.int64)

            for i, sample in enumerate(samples):
                sequences[i] = sample["sequence"]
                profiles[i] = sample["profile"]
                binding_sites[i, 0, :] = sample["binding_sites"]
                tf_indices[i] = sample["tf_idx"]

            with h5py.File(output_path, "w") as f:
                f.create_dataset("sequences", data=sequences, compression="gzip", chunks=True)
                f.create_dataset("profiles", data=profiles, compression="gzip", chunks=True)
                f.create_dataset("binding_sites", data=binding_sites, compression="gzip", chunks=True)
                f.create_dataset("tf_indices", data=tf_indices, compression="gzip", chunks=True)
                f.create_dataset("tf_vocab", data=json.dumps(tf_vocab))
                f.attrs["n_samples"] = n
                f.attrs["seq_length"] = self.seq_length
                f.attrs["n_tfs"] = len(tf_vocab)
                f.attrs["n_slots"] = n_slots

            logger.info(f"    Saved: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")

        # Save summary
        summary = {
            "n_tfs": len(tf_vocab),
            "tfs": tf_vocab,
            "tf_indices": {tf: i for i, tf in enumerate(tf_vocab)},
            "tf_families": {tf: TF_FAMILIES.get(tf, "?") for tf in tf_vocab},
            "seq_length": self.seq_length,
            "n_train": len(all_data["train"]),
            "n_val": len(all_data["val"]),
            "n_test": len(all_data["test"]),
            "balanced": self.balance_tfs,
            "stats": {tf: dict(v) for tf, v in self.stats.items()},
            "created": datetime.now().isoformat(),
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info("=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"TFs: {len(tf_vocab)} ({', '.join(tf_vocab)})")
        logger.info(f"Train: {len(all_data['train'])}")
        logger.info(f"Val: {len(all_data['val'])}")
        logger.info(f"Test: {len(all_data['test'])}")
        logger.info("")
        logger.info("Per-TF stats:")
        for tf in tf_vocab:
            s = self.stats[tf]
            logger.info(f"  {tf} ({TF_FAMILIES.get(tf, '?')}): "
                        f"parsed={s['total_parsed']}, processed={s['processed']}, "
                        f"train={s['split_train']}, val={s['split_val']}, test={s['split_test']}")
        logger.info("")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Next: python scripts/train_multi_tf.py")
        logger.info("=" * 70)

        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-TF K562 data for BEACON (Phase 2B)")
    parser.add_argument("--genome", type=Path, default=DEFAULT_GENOME)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-length", type=int, default=2000)
    parser.add_argument("--max-peaks-per-tf", type=int, default=20000)
    parser.add_argument("--min-signal", type=float, default=0.0)
    parser.add_argument("--no-balance", action="store_true", help="Don't balance TF counts")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(args.output_dir / "preparation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    preparer = MultiTFDataPreparer(
        genome_path=args.genome,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        blacklist_path=args.blacklist,
        seq_length=args.seq_length,
        max_peaks_per_tf=args.max_peaks_per_tf,
        min_signal=args.min_signal,
        balance_tfs=not args.no_balance,
    )

    success = preparer.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
