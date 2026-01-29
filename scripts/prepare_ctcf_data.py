#!/usr/bin/env python3
"""
Prepare CTCF K562 ChIP-seq data for BEACON training.

Phase 2A: Single-TF real data pipeline.

Pipeline:
1. Parse narrowPeak file for CTCF binding sites
2. Extract 2kb sequences around peak summits from hg38
3. Extract binding profiles from bigWig signal tracks
4. Filter by blacklist regions
5. Split by chromosome (train/val/test)
6. Save to HDF5 format compatible with BEACONDataset

Chromosome splits (ENCODE DREAM convention):
- Train: chr1-17
- Validation: chr18-19
- Test: chr20-22, chrX

Usage:
    python scripts/prepare_ctcf_data.py
    python scripts/prepare_ctcf_data.py --no-bigwig  # Skip bigWig, use Gaussian profiles
    python scripts/prepare_ctcf_data.py --seq-length 1000  # Shorter sequences
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import pyfaidx
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False
    logger.warning("pyfaidx not installed. Install with: pip install pyfaidx")

try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False
    logger.warning("pyBigWig not installed. Install with: pip install pyBigWig")

# ============================================================================
# Constants
# ============================================================================

DNA_ALPHABET = "ACGT"

TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18)]
VAL_CHROMS = ["chr18", "chr19"]
TEST_CHROMS = ["chr20", "chr21", "chr22", "chrX"]
ALL_CHROMS = TRAIN_CHROMS + VAL_CHROMS + TEST_CHROMS

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GENOME = PROJECT_ROOT / "data" / "raw" / "genome" / "hg38.fa"
DEFAULT_BLACKLIST = PROJECT_ROOT / "data" / "raw" / "genome" / "hg38-blacklist.v2.bed"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "encode" / "CTCF_K562"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ctcf_k562"


# ============================================================================
# Utility Functions
# ============================================================================

def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a DNA sequence. N -> [0.25, 0.25, 0.25, 0.25]."""
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            encoding[i, base_to_idx[base]] = 1.0
        else:
            encoding[i, :] = 0.25  # N or ambiguous
    return encoding


def parse_narrowpeak(filepath: Path) -> list:
    """
    Parse narrowPeak file (BED6+4 format).

    Columns: chrom, start, end, name, score, strand, signalValue, pValue, qValue, peak(summit offset)
    """
    peaks = []
    opener = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"

    with opener(filepath, mode) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue

            fields = line.split("\t")
            if len(fields) < 10:
                logger.warning(f"Line {line_num}: Expected 10 fields, got {len(fields)}. Skipping.")
                continue

            try:
                peak = {
                    "chrom": fields[0],
                    "start": int(fields[1]),
                    "end": int(fields[2]),
                    "name": fields[3],
                    "score": float(fields[4]) if fields[4] != "." else 0.0,
                    "strand": fields[5] if fields[5] in ["+", "-"] else ".",
                    "signal_value": float(fields[6]) if fields[6] != "." else 0.0,
                    "p_value": float(fields[7]) if fields[7] != "." else 0.0,
                    "q_value": float(fields[8]) if fields[8] != "." else 0.0,
                    "summit_offset": int(fields[9]) if fields[9] != "." else (int(fields[2]) - int(fields[1])) // 2,
                }
                # Summit in absolute coordinates
                peak["summit"] = peak["start"] + peak["summit_offset"]
                peaks.append(peak)
            except (ValueError, IndexError) as e:
                logger.warning(f"Line {line_num}: Parse error: {e}. Skipping.")
                continue

    return peaks


def load_blacklist(filepath: Path) -> dict:
    """Load ENCODE blacklist regions as dict of chrom -> list of (start, end)."""
    blacklist = defaultdict(list)

    if not filepath.exists():
        logger.warning(f"Blacklist file not found: {filepath}")
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
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                blacklist[chrom].append((start, end))

    total = sum(len(v) for v in blacklist.values())
    logger.info(f"Loaded blacklist: {total} regions across {len(blacklist)} chromosomes")
    return blacklist


def is_blacklisted(chrom: str, start: int, end: int, blacklist: dict) -> bool:
    """Check if a region overlaps any blacklist region."""
    for bl_start, bl_end in blacklist.get(chrom, []):
        if start < bl_end and end > bl_start:
            return True
    return False


def create_gaussian_profile(seq_length: int, peak_position: int, sigma: float = 30.0) -> np.ndarray:
    """Create a synthetic Gaussian profile centered on peak summit."""
    x = np.arange(seq_length, dtype=np.float32)
    profile = np.exp(-0.5 * ((x - peak_position) / sigma) ** 2)
    return profile


# ============================================================================
# Main Processing Class
# ============================================================================

class CTCFDataPreparer:
    """Prepare CTCF K562 data for BEACON training."""

    def __init__(
        self,
        genome_path: Path,
        data_dir: Path,
        output_dir: Path,
        blacklist_path: Path = None,
        seq_length: int = 2000,
        min_signal: float = 0.0,
        max_peaks: int = 100000,
        use_bigwig: bool = True,
        profile_sigma: float = 30.0,
    ):
        self.genome_path = genome_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.blacklist_path = blacklist_path
        self.seq_length = seq_length
        self.flank = seq_length // 2
        self.min_signal = min_signal
        self.max_peaks = max_peaks
        self.use_bigwig = use_bigwig
        self.profile_sigma = profile_sigma

        # Stats
        self.stats = {
            "total_peaks_parsed": 0,
            "peaks_after_chrom_filter": 0,
            "peaks_after_blacklist_filter": 0,
            "peaks_after_signal_filter": 0,
            "peaks_after_boundary_filter": 0,
            "peaks_processed": 0,
            "peaks_failed_sequence": 0,
            "peaks_failed_profile": 0,
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "chrom_counts": Counter(),
        }

    def run(self):
        """Execute the full pipeline."""
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("CTCF K562 Data Preparation for BEACON")
        logger.info("=" * 70)
        logger.info(f"Genome: {self.genome_path}")
        logger.info(f"Data dir: {self.data_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Sequence length: {self.seq_length}")
        logger.info(f"Min signal: {self.min_signal}")
        logger.info(f"Max peaks: {self.max_peaks}")
        logger.info(f"Use bigWig: {self.use_bigwig}")
        logger.info(f"Profile sigma (if no bigWig): {self.profile_sigma}")
        logger.info("")

        # Step 1: Load genome
        logger.info("Step 1: Loading reference genome...")
        fasta = self._load_genome()
        if fasta is None:
            logger.error("Failed to load genome. Aborting.")
            return False

        # Step 2: Load blacklist
        logger.info("Step 2: Loading blacklist...")
        blacklist = {}
        if self.blacklist_path:
            blacklist = load_blacklist(self.blacklist_path)

        # Step 3: Find and parse peaks
        logger.info("Step 3: Finding and parsing peak files...")
        peaks = self._find_and_parse_peaks()
        if not peaks:
            logger.error("No peaks found. Aborting.")
            return False

        # Step 4: Filter peaks
        logger.info("Step 4: Filtering peaks...")
        peaks = self._filter_peaks(peaks, fasta, blacklist)
        if not peaks:
            logger.error("No peaks remaining after filtering. Aborting.")
            return False

        # Step 5: Load bigWig (optional)
        bw = None
        if self.use_bigwig:
            logger.info("Step 5: Loading bigWig signal track...")
            bw = self._load_bigwig()
            if bw is None:
                logger.warning("No bigWig available. Using Gaussian profiles.")
        else:
            logger.info("Step 5: Skipping bigWig (using Gaussian profiles)...")

        # Step 6: Extract sequences and profiles
        logger.info("Step 6: Extracting sequences and profiles...")
        data_by_split = self._extract_data(peaks, fasta, bw)

        if bw:
            bw.close()

        # Step 7: Save to HDF5
        logger.info("Step 7: Saving to HDF5...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_datasets(data_by_split)

        # Step 8: Print summary
        elapsed = (datetime.now() - start_time).total_seconds()
        self._print_summary(elapsed)

        return True

    def _load_genome(self):
        """Load hg38 reference genome."""
        if not HAS_PYFAIDX:
            logger.error("pyfaidx not installed!")
            return None

        if not self.genome_path.exists():
            logger.error(f"Genome file not found: {self.genome_path}")
            return None

        try:
            fasta = pyfaidx.Fasta(str(self.genome_path))
            chroms = list(fasta.keys())
            logger.info(f"  Loaded genome with {len(chroms)} chromosomes")
            logger.info(f"  First few: {chroms[:5]}")
            return fasta
        except Exception as e:
            logger.error(f"  Failed to load genome: {e}")
            return None

    def _find_and_parse_peaks(self):
        """Find narrowPeak files and parse all peaks."""
        peak_files = list(self.data_dir.glob("*.narrowPeak*"))

        if not peak_files:
            # Try subdirectories
            peak_files = list(self.data_dir.rglob("*.narrowPeak*"))

        logger.info(f"  Found {len(peak_files)} peak file(s):")
        for pf in peak_files:
            logger.info(f"    {pf.name} ({pf.stat().st_size / 1e6:.2f} MB)")

        all_peaks = []
        for pf in peak_files:
            peaks = parse_narrowpeak(pf)
            logger.info(f"    Parsed {len(peaks)} peaks from {pf.name}")
            all_peaks.extend(peaks)

        self.stats["total_peaks_parsed"] = len(all_peaks)
        logger.info(f"  Total peaks parsed: {len(all_peaks)}")
        return all_peaks

    def _filter_peaks(self, peaks, fasta, blacklist):
        """Filter peaks by chromosome, blacklist, signal, boundaries."""
        # Filter by valid chromosomes
        valid_chroms = set(ALL_CHROMS)
        peaks = [p for p in peaks if p["chrom"] in valid_chroms]
        self.stats["peaks_after_chrom_filter"] = len(peaks)
        logger.info(f"  After chromosome filter: {len(peaks)} (kept chroms: {len(valid_chroms)})")

        # Filter by blacklist
        if blacklist:
            before = len(peaks)
            peaks = [
                p for p in peaks
                if not is_blacklisted(
                    p["chrom"],
                    p["summit"] - self.flank,
                    p["summit"] + self.flank,
                    blacklist
                )
            ]
            removed = before - len(peaks)
            logger.info(f"  After blacklist filter: {len(peaks)} (removed {removed})")
        self.stats["peaks_after_blacklist_filter"] = len(peaks)

        # Filter by signal value
        if self.min_signal > 0:
            peaks = [p for p in peaks if p["signal_value"] >= self.min_signal]
            logger.info(f"  After signal filter (>= {self.min_signal}): {len(peaks)}")
        self.stats["peaks_after_signal_filter"] = len(peaks)

        # Filter by genome boundaries
        before = len(peaks)
        filtered = []
        for p in peaks:
            chrom = p["chrom"]
            if chrom not in fasta:
                continue
            chrom_len = len(fasta[chrom])
            start = p["summit"] - self.flank
            end = p["summit"] + self.flank
            if start >= 0 and end <= chrom_len:
                filtered.append(p)
        peaks = filtered
        removed = before - len(peaks)
        self.stats["peaks_after_boundary_filter"] = len(peaks)
        logger.info(f"  After boundary filter: {len(peaks)} (removed {removed})")

        # Sort by signal and limit
        peaks = sorted(peaks, key=lambda x: -x["signal_value"])
        if len(peaks) > self.max_peaks:
            peaks = peaks[:self.max_peaks]
            logger.info(f"  After limit ({self.max_peaks}): {len(peaks)}")

        # Log signal distribution
        signals = [p["signal_value"] for p in peaks]
        logger.info(f"  Signal value stats:")
        logger.info(f"    Min: {min(signals):.2f}")
        logger.info(f"    Max: {max(signals):.2f}")
        logger.info(f"    Mean: {np.mean(signals):.2f}")
        logger.info(f"    Median: {np.median(signals):.2f}")

        return peaks

    def _load_bigwig(self):
        """Load bigWig signal file."""
        if not HAS_PYBIGWIG:
            logger.warning("pyBigWig not installed!")
            return None

        bw_files = list(self.data_dir.glob("*.bigWig")) + list(self.data_dir.glob("*.bw"))

        if not bw_files:
            logger.warning("No bigWig files found")
            return None

        bw_path = bw_files[0]
        logger.info(f"  Loading: {bw_path.name} ({bw_path.stat().st_size / 1e6:.1f} MB)")

        try:
            bw = pyBigWig.open(str(bw_path))
            chroms = bw.chroms()
            logger.info(f"  BigWig chromosomes: {len(chroms)}")
            return bw
        except Exception as e:
            logger.error(f"  Failed to load bigWig: {e}")
            return None

    def _extract_data(self, peaks, fasta, bw):
        """Extract sequences and profiles for all peaks."""
        data_by_split = {"train": [], "val": [], "test": []}

        n_total = len(peaks)
        log_interval = max(1, n_total // 20)  # Log every 5%

        for i, peak in enumerate(peaks):
            if i % log_interval == 0:
                pct = 100 * i / n_total
                logger.info(f"  Progress: {i}/{n_total} ({pct:.0f}%)")

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
                    self.stats["peaks_failed_sequence"] += 1
                    continue
            except Exception as e:
                self.stats["peaks_failed_sequence"] += 1
                continue

            # Extract profile
            profile = None
            if bw:
                try:
                    values = bw.values(chrom, start, end)
                    profile = np.array(values, dtype=np.float32)
                    profile = np.nan_to_num(profile, nan=0.0)

                    if len(profile) != self.seq_length:
                        self.stats["peaks_failed_profile"] += 1
                        continue

                    # Normalize to [0, 1]
                    if profile.max() > 0:
                        profile = profile / profile.max()
                except Exception:
                    profile = None

            if profile is None:
                # Use Gaussian profile centered on summit
                profile = create_gaussian_profile(
                    self.seq_length,
                    self.flank,  # Summit is centered
                    sigma=self.profile_sigma,
                )

            # One-hot encode
            seq_encoded = one_hot_encode(seq)

            # Create binding site annotation
            # For single-TF: 1 binding site at center, TF index 0, occupancy from signal
            occ = min(1.0, peak["signal_value"] / 100.0)  # Rough normalization
            binding_site = np.array([
                0.5,  # Normalized position (center)
                0,    # TF index (CTCF = 0)
                occ,  # Occupancy
            ], dtype=np.float32)

            data_by_split[split].append({
                "sequence": seq_encoded,      # [L, 4]
                "profile": profile,           # [L]
                "binding_sites": binding_site, # [3]
                "chrom": chrom,
                "position": summit,
                "signal_value": peak["signal_value"],
            })

            self.stats["peaks_processed"] += 1
            self.stats["split_counts"][split] += 1
            self.stats["chrom_counts"][chrom] += 1

        logger.info(f"  Extraction complete: {self.stats['peaks_processed']} peaks processed")
        for split, samples in data_by_split.items():
            logger.info(f"    {split}: {len(samples)} samples")

        return data_by_split

    def _save_datasets(self, data_by_split):
        """Save train/val/test datasets to HDF5."""
        tf_vocab = ["CTCF"]

        for split, samples in data_by_split.items():
            if not samples:
                logger.warning(f"  No samples for {split} split, skipping.")
                continue

            n = len(samples)
            output_path = self.output_dir / f"{split}.h5"

            logger.info(f"  Saving {split}: {n} samples -> {output_path}")

            # Pre-allocate arrays
            sequences = np.zeros((n, self.seq_length, 4), dtype=np.float32)
            profiles = np.zeros((n, self.seq_length), dtype=np.float32)
            # binding_sites: [N, K, 3] where K=max slots
            n_slots = 16  # Match model config
            binding_sites = np.zeros((n, n_slots, 3), dtype=np.float32)
            tf_indices = np.zeros(n, dtype=np.int64)

            chroms = []
            positions = []
            signal_values = []

            for i, sample in enumerate(samples):
                sequences[i] = sample["sequence"]
                profiles[i] = sample["profile"]
                # Put the binding site in slot 0
                binding_sites[i, 0, :] = sample["binding_sites"]
                tf_indices[i] = 0  # CTCF
                chroms.append(sample["chrom"])
                positions.append(sample["position"])
                signal_values.append(sample["signal_value"])

            # Save to HDF5
            with h5py.File(output_path, "w") as f:
                f.create_dataset("sequences", data=sequences, compression="gzip", chunks=True)
                f.create_dataset("profiles", data=profiles, compression="gzip", chunks=True)
                f.create_dataset("binding_sites", data=binding_sites, compression="gzip", chunks=True)
                f.create_dataset("tf_indices", data=tf_indices, compression="gzip", chunks=True)
                f.create_dataset("tf_vocab", data=json.dumps(tf_vocab))

                # Metadata
                f.attrs["n_samples"] = n
                f.attrs["seq_length"] = self.seq_length
                f.attrs["n_tfs"] = 1
                f.attrs["n_slots"] = n_slots
                f.attrs["tf_name"] = "CTCF"
                f.attrs["cell_line"] = "K562"
                f.attrs["split"] = split

                # Store chromosome info for reference
                f.create_dataset("chroms", data=json.dumps(chroms))
                f.create_dataset("positions", data=np.array(positions, dtype=np.int64))
                f.create_dataset("signal_values", data=np.array(signal_values, dtype=np.float32))

            logger.info(f"    Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

            # Log data stats
            logger.info(f"    Sequence shape: {sequences.shape}")
            logger.info(f"    Profile shape: {profiles.shape}")
            logger.info(f"    Profile range: [{profiles.min():.4f}, {profiles.max():.4f}]")
            logger.info(f"    N bases (non-N): {(sequences.max(axis=-1) > 0.5).sum() / n:.0f} per sequence")

        # Save summary JSON
        summary = {
            "tf": "CTCF",
            "cell_line": "K562",
            "n_tfs": 1,
            "tfs": tf_vocab,
            "tf_indices": {"CTCF": 0},
            "seq_length": self.seq_length,
            "n_train": self.stats["split_counts"]["train"],
            "n_val": self.stats["split_counts"]["val"],
            "n_test": self.stats["split_counts"]["test"],
            "train_chroms": TRAIN_CHROMS,
            "val_chroms": VAL_CHROMS,
            "test_chroms": TEST_CHROMS,
            "stats": {k: v if not isinstance(v, Counter) else dict(v) for k, v in self.stats.items()},
            "has_bigwig_profiles": self.use_bigwig,
            "created": datetime.now().isoformat(),
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Summary saved: {summary_path}")

    def _print_summary(self, elapsed):
        """Print processing summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"")
        logger.info(f"Peak filtering pipeline:")
        logger.info(f"  Total parsed:        {self.stats['total_peaks_parsed']:>8}")
        logger.info(f"  After chrom filter:  {self.stats['peaks_after_chrom_filter']:>8}")
        logger.info(f"  After blacklist:     {self.stats['peaks_after_blacklist_filter']:>8}")
        logger.info(f"  After signal filter: {self.stats['peaks_after_signal_filter']:>8}")
        logger.info(f"  After boundary:      {self.stats['peaks_after_boundary_filter']:>8}")
        logger.info(f"  Successfully processed: {self.stats['peaks_processed']:>8}")
        logger.info(f"  Failed (sequence):   {self.stats['peaks_failed_sequence']:>8}")
        logger.info(f"  Failed (profile):    {self.stats['peaks_failed_profile']:>8}")
        logger.info(f"")
        logger.info(f"Split distribution:")
        for split, count in self.stats["split_counts"].items():
            logger.info(f"  {split:>5}: {count:>8}")
        logger.info(f"")
        logger.info(f"Chromosome distribution:")
        for chrom in ALL_CHROMS:
            count = self.stats["chrom_counts"].get(chrom, 0)
            if count > 0:
                logger.info(f"  {chrom:>5}: {count:>6}")
        logger.info(f"")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"")
        logger.info(f"Next step:")
        logger.info(f"  python scripts/train_ctcf.py")
        logger.info("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CTCF K562 ChIP-seq data for BEACON training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--genome", type=Path, default=DEFAULT_GENOME,
                        help="Path to hg38 reference genome FASTA")
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST,
                        help="Path to ENCODE blacklist BED file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing CTCF narrowPeak and bigWig files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for processed HDF5 files")
    parser.add_argument("--seq-length", type=int, default=2000,
                        help="Sequence length around peak summits")
    parser.add_argument("--min-signal", type=float, default=0.0,
                        help="Minimum signal value filter")
    parser.add_argument("--max-peaks", type=int, default=100000,
                        help="Maximum number of peaks to process")
    parser.add_argument("--no-bigwig", action="store_true",
                        help="Skip bigWig loading, use Gaussian profiles")
    parser.add_argument("--profile-sigma", type=float, default=30.0,
                        help="Gaussian sigma for synthetic profiles (if no bigWig)")

    args = parser.parse_args()

    # Add file logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "preparation.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    preparer = CTCFDataPreparer(
        genome_path=args.genome,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        blacklist_path=args.blacklist,
        seq_length=args.seq_length,
        min_signal=args.min_signal,
        max_peaks=args.max_peaks,
        use_bigwig=not args.no_bigwig,
        profile_sigma=args.profile_sigma,
    )

    success = preparer.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
