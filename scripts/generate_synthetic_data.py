#!/usr/bin/env python3
"""
Generate synthetic binding data for BEACON validation.

Creates realistic synthetic data with:
- Configurable number of TF classes (up to 50+)
- Multiple binding sites per sequence (1-8+)
- Overlapping binding sites allowed
- Gaussian noise on profiles
- Realistic PWM-based motifs from JASPAR

Stage A Parameters (Harder Synthetic):
- Sites per sequence: 3-8
- TF classes: 50
- Overlapping: Yes (50% overlap allowed)
- Profile noise: sigma = 0.1-0.2
- Sequence length: 2000bp
- Training samples: 100k
"""

import os
import sys
import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.stats import truncnorm
import requests
from tqdm import tqdm


# DNA alphabet
DNA_ALPHABET = "ACGT"
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a DNA sequence."""
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in DNA_ALPHABET:
            encoding[i, DNA_ALPHABET.index(base)] = 1.0
        else:
            encoding[i, :] = 0.25
    return encoding


def sample_sequence_from_pwm(pwm: np.ndarray) -> str:
    """Sample a DNA sequence from a PWM."""
    sequence = []
    for position in pwm:
        # Convert log-odds to probabilities
        probs = np.exp(position)
        probs = probs / probs.sum()
        base_idx = np.random.choice(4, p=probs)
        sequence.append(DNA_ALPHABET[base_idx])
    return "".join(sequence)


def reverse_complement(sequence: str) -> str:
    """Get reverse complement of DNA sequence."""
    return "".join(COMPLEMENT.get(base, "N") for base in reversed(sequence.upper()))


class JASPARMotifLoader:
    """Load and process motifs from JASPAR database."""

    JASPAR_URL = "https://jaspar.elixir.no/api/v1/matrix/"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "jaspar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.motifs = {}

    def load_from_file(self, filepath: Path) -> Dict[str, np.ndarray]:
        """Load motifs from a JASPAR format file."""
        motifs = {}
        current_id = None
        current_matrix = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_matrix:
                        motifs[current_id] = self._parse_matrix(current_matrix)
                    parts = line[1:].split()
                    current_id = parts[0] if parts else None
                    current_matrix = []
                elif line and current_id:
                    current_matrix.append(line)

            if current_id and current_matrix:
                motifs[current_id] = self._parse_matrix(current_matrix)

        return motifs

    def _parse_matrix(self, lines: List[str]) -> np.ndarray:
        """Parse JASPAR matrix format to numpy array."""
        matrix = []
        for line in lines:
            # Remove letter prefix (A, C, G, T) and brackets
            line = line.strip()
            for char in "ACGT[]":
                line = line.replace(char, "")
            values = [float(x) for x in line.split()]
            if values:
                matrix.append(values)

        if not matrix:
            return np.zeros((10, 4))

        matrix = np.array(matrix)
        if matrix.shape[0] == 4:  # Rows are bases
            matrix = matrix.T
        return matrix

    def pfm_to_pwm(self, pfm: np.ndarray, pseudocount: float = 0.01) -> np.ndarray:
        """Convert Position Frequency Matrix to Position Weight Matrix."""
        # Normalize to probabilities
        row_sums = pfm.sum(axis=1, keepdims=True)
        ppm = (pfm + pseudocount) / (row_sums + 4 * pseudocount)
        # Log-odds against uniform background
        pwm = np.log2(ppm / 0.25 + 1e-10)
        return pwm


class RealisticMotifLibrary:
    """
    Library of realistic TF motifs based on JASPAR patterns.

    Creates motifs that capture known TF binding characteristics:
    - Core motif patterns (e.g., E-box, TATA, GC-box)
    - Variable flanking regions
    - Information content distribution
    """

    # Known core motif patterns
    CORE_MOTIFS = {
        # E-box motifs (bHLH factors: MAX, MYC, etc.)
        "E-box": "CACGTG",
        "E-box_alt": "CANNTG",

        # GATA motifs
        "GATA": "WGATAR",
        "GATA_extended": "NWGATAAGN",

        # ETS family
        "ETS": "GGAA",
        "ETS_extended": "ACGGAAGT",

        # AP-1 (JUN/FOS)
        "AP1": "TGASTCA",

        # CEBP
        "CEBP": "TTGCGYAAY",

        # NF-kB
        "NFKB": "GGGRNWYYCC",

        # CTCF
        "CTCF": "CCGCGNGGNGGCAG",

        # SOX family
        "SOX": "AACAAAG",

        # FOXA (HNF3)
        "FOXA": "TGTTTAC",

        # SP1/GC-box
        "SP1": "GGGCGG",

        # TATA box
        "TATA": "TATAAAA",

        # Oct4/Pou
        "POU": "ATGCAAAT",

        # Nuclear receptors
        "NR_half": "AGGTCA",

        # Homeobox
        "HOX": "TAATNN",

        # ZNF/KRAB
        "ZNF": "GNGNGGNNN",
    }

    # IUPAC ambiguity codes
    IUPAC = {
        "A": "A", "C": "C", "G": "G", "T": "T",
        "R": "AG", "Y": "CT", "S": "GC", "W": "AT",
        "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
        "H": "ACT", "V": "ACG", "N": "ACGT",
    }

    def __init__(self, n_motifs: int = 50, seed: int = 42):
        self.n_motifs = n_motifs
        self.rng = np.random.RandomState(seed)
        self.motifs = []
        self.motif_names = []
        self.motif_lengths = []
        self._generate_motifs()

    def _iupac_to_pwm_row(self, code: str, information: float = 2.0) -> np.ndarray:
        """Convert IUPAC code to PWM probability row."""
        bases = self.IUPAC.get(code.upper(), "ACGT")
        probs = np.zeros(4)
        for base in bases:
            probs[DNA_ALPHABET.index(base)] = 1.0
        probs = probs / probs.sum()

        # Add some noise based on information content
        noise_scale = max(0.01, (2.0 - information) / 4)
        probs = probs * (1 - noise_scale) + noise_scale / 4

        return np.log2(probs / 0.25 + 1e-10)

    def _consensus_to_pwm(self, consensus: str, info_content: float = 1.8) -> np.ndarray:
        """Convert consensus sequence to PWM."""
        pwm = []
        for base in consensus:
            # Vary information content along the motif
            pos_info = info_content + self.rng.uniform(-0.3, 0.3)
            pwm.append(self._iupac_to_pwm_row(base, pos_info))
        return np.array(pwm)

    def _add_flanks(self, pwm: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
        """Add low-information flanking positions."""
        left_flank = np.zeros((left, 4)) + self.rng.uniform(-0.2, 0.2, (left, 4))
        right_flank = np.zeros((right, 4)) + self.rng.uniform(-0.2, 0.2, (right, 4))
        return np.vstack([left_flank, pwm, right_flank])

    def _generate_motifs(self):
        """Generate diverse set of realistic motifs."""
        # Use core patterns as base
        core_patterns = list(self.CORE_MOTIFS.items())

        motif_id = 0
        while len(self.motifs) < self.n_motifs:
            # Select a core pattern (cycle through them)
            pattern_name, consensus = core_patterns[motif_id % len(core_patterns)]

            # Create base PWM
            info_content = 1.5 + self.rng.uniform(0, 0.5)
            pwm = self._consensus_to_pwm(consensus, info_content)

            # Add variation for different "family members"
            variation = self.rng.normal(0, 0.2, pwm.shape)
            pwm = pwm + variation

            # Add flanks
            left_flank = self.rng.randint(1, 4)
            right_flank = self.rng.randint(1, 4)
            pwm = self._add_flanks(pwm, left_flank, right_flank)

            # Store
            self.motifs.append(pwm.astype(np.float32))
            variant = (motif_id // len(core_patterns)) + 1
            self.motif_names.append(f"{pattern_name}_v{variant}")
            self.motif_lengths.append(len(pwm))

            motif_id += 1

    def get_motif(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get motif by index."""
        return self.motifs[idx], self.motif_names[idx]

    def sample_sequence(self, idx: int) -> str:
        """Sample a sequence from a motif PWM."""
        pwm = self.motifs[idx]
        return sample_sequence_from_pwm(pwm)

    def save(self, filepath: Path):
        """Save motif library to HDF5."""
        max_len = max(self.motif_lengths)
        n_motifs = len(self.motifs)

        # Pad motifs to same length
        padded_pwms = np.zeros((n_motifs, max_len, 4), dtype=np.float32)
        for i, pwm in enumerate(self.motifs):
            padded_pwms[i, :len(pwm)] = pwm

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('pwms', data=padded_pwms)
            f.create_dataset('lengths', data=np.array(self.motif_lengths))
            f.create_dataset('names', data=np.array(self.motif_names, dtype='S50'))
            f.attrs['n_motifs'] = n_motifs
            f.attrs['max_length'] = max_len


class SyntheticDataGenerator:
    """
    Generate synthetic ChIP-seq-like binding data.

    Creates sequences with multiple TF binding sites and corresponding
    profiles that simulate ChIP-seq signal.
    """

    def __init__(
        self,
        seq_length: int = 2000,
        n_tfs: int = 50,
        min_sites: int = 1,
        max_sites: int = 8,
        allow_overlap: bool = True,
        min_overlap_gap: int = 5,  # Minimum gap even with overlap
        profile_sigma: float = 30.0,  # Width of Gaussian peaks
        noise_sigma: float = 0.1,  # Noise level
        seed: int = 42,
    ):
        self.seq_length = seq_length
        self.n_tfs = n_tfs
        self.min_sites = min_sites
        self.max_sites = max_sites
        self.allow_overlap = allow_overlap
        self.min_overlap_gap = min_overlap_gap
        self.profile_sigma = profile_sigma
        self.noise_sigma = noise_sigma
        self.rng = np.random.RandomState(seed)

        # Initialize motif library
        self.motif_library = RealisticMotifLibrary(n_motifs=n_tfs, seed=seed)

    def _generate_background_sequence(self) -> str:
        """Generate random background DNA sequence."""
        # Use slightly biased nucleotide frequencies (like real genome)
        # GC content ~40%
        probs = [0.30, 0.20, 0.20, 0.30]  # A, C, G, T
        indices = self.rng.choice(4, size=self.seq_length, p=probs)
        return "".join(DNA_ALPHABET[i] for i in indices)

    def _insert_motif(
        self,
        sequence: str,
        position: int,
        tf_idx: int,
    ) -> Tuple[str, str, int]:
        """Insert a motif into sequence at position."""
        motif_seq = self.motif_library.sample_sequence(tf_idx)
        motif_len = len(motif_seq)

        # Randomly choose strand
        strand = "+" if self.rng.random() > 0.5 else "-"
        if strand == "-":
            motif_seq = reverse_complement(motif_seq)

        # Insert motif
        end_pos = min(position + motif_len, self.seq_length)
        actual_len = end_pos - position

        new_seq = sequence[:position] + motif_seq[:actual_len] + sequence[end_pos:]

        return new_seq, strand, actual_len

    def _sample_binding_positions(
        self,
        n_sites: int,
        motif_lengths: List[int],
    ) -> List[int]:
        """Sample non-colliding (or minimally overlapping) positions."""
        positions = []
        margin = 50  # Keep sites away from edges

        max_attempts = 100 * n_sites
        attempts = 0

        while len(positions) < n_sites and attempts < max_attempts:
            attempts += 1
            pos = self.rng.randint(margin, self.seq_length - margin)

            # Check for collisions
            valid = True
            for i, existing_pos in enumerate(positions):
                existing_len = motif_lengths[i] if i < len(motif_lengths) else 15
                distance = abs(pos - existing_pos)

                min_distance = self.min_overlap_gap if self.allow_overlap else existing_len + 10

                if distance < min_distance:
                    valid = False
                    break

            if valid:
                positions.append(pos)

        return sorted(positions)

    def _generate_profile(
        self,
        positions: List[int],
        strengths: List[float],
    ) -> np.ndarray:
        """Generate binding profile from site positions."""
        profile = np.zeros(self.seq_length, dtype=np.float32)

        for pos, strength in zip(positions, strengths):
            # Create Gaussian peak centered at position
            x = np.arange(self.seq_length)
            peak = strength * np.exp(-0.5 * ((x - pos) / self.profile_sigma) ** 2)
            profile += peak

        # Add noise
        if self.noise_sigma > 0:
            noise = self.rng.normal(0, self.noise_sigma, self.seq_length)
            profile = profile + noise
            profile = np.maximum(profile, 0)  # Ensure non-negative

        # Normalize
        if profile.max() > 0:
            profile = profile / profile.max()

        return profile

    def generate_sample(self) -> Dict:
        """Generate a single training sample."""
        # Sample number of binding sites
        n_sites = self.rng.randint(self.min_sites, self.max_sites + 1)

        # Sample TF indices for each site
        tf_indices = self.rng.randint(0, self.n_tfs, size=n_sites)

        # Get motif lengths
        motif_lengths = [self.motif_library.motif_lengths[idx] for idx in tf_indices]

        # Sample positions
        positions = self._sample_binding_positions(n_sites, motif_lengths)
        n_sites = len(positions)  # May be fewer if couldn't place all
        tf_indices = tf_indices[:n_sites]
        motif_lengths = motif_lengths[:n_sites]

        # Generate background sequence
        sequence = self._generate_background_sequence()

        # Insert motifs and track site info
        sites = []
        for i, (pos, tf_idx) in enumerate(zip(positions, tf_indices)):
            sequence, strand, actual_len = self._insert_motif(sequence, pos, tf_idx)

            # Sample binding strength (0.5 to 1.0)
            strength = 0.5 + 0.5 * self.rng.random()

            sites.append({
                'position': pos,
                'tf_idx': int(tf_idx),
                'strand': strand,
                'strength': strength,
                'motif_length': actual_len,
            })

        # Generate profile
        profile = self._generate_profile(
            [s['position'] for s in sites],
            [s['strength'] for s in sites],
        )

        return {
            'sequence': sequence,
            'profile': profile,
            'sites': sites,
            'n_sites': n_sites,
        }

    def generate_dataset(
        self,
        n_samples: int,
        output_path: Path,
        n_slots: int = 16,
        show_progress: bool = True,
    ):
        """Generate full dataset and save to HDF5."""
        sequences = []
        profiles = []
        binding_sites = []
        tf_indices = []  # Dominant TF for each sample

        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating samples")

        for _ in iterator:
            sample = self.generate_sample()

            # One-hot encode sequence
            seq_encoded = one_hot_encode(sample['sequence'])
            sequences.append(seq_encoded)

            # Profile
            profiles.append(sample['profile'])

            # Binding sites tensor: [n_slots, 3] = (position, tf_id, occupancy)
            sites_tensor = np.zeros((n_slots, 3), dtype=np.float32)
            for i, site in enumerate(sample['sites'][:n_slots]):
                sites_tensor[i, 0] = site['position'] / self.seq_length  # Normalized
                sites_tensor[i, 1] = site['tf_idx']
                sites_tensor[i, 2] = site['strength']
            binding_sites.append(sites_tensor)

            # Dominant TF (most frequent or first)
            if sample['sites']:
                tf_counts = {}
                for site in sample['sites']:
                    tf_counts[site['tf_idx']] = tf_counts.get(site['tf_idx'], 0) + 1
                dominant_tf = max(tf_counts, key=tf_counts.get)
            else:
                dominant_tf = 0
            tf_indices.append(dominant_tf)

        # Convert to arrays
        sequences = np.stack(sequences, axis=0)
        profiles = np.stack(profiles, axis=0)
        binding_sites = np.stack(binding_sites, axis=0)
        tf_indices = np.array(tf_indices, dtype=np.int64)

        # Save to HDF5
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('sequences', data=sequences, compression='gzip')
            f.create_dataset('profiles', data=profiles, compression='gzip')
            f.create_dataset('binding_sites', data=binding_sites, compression='gzip')
            f.create_dataset('tf_indices', data=tf_indices, compression='gzip')

            # Metadata
            f.attrs['n_samples'] = n_samples
            f.attrs['seq_length'] = self.seq_length
            f.attrs['n_tfs'] = self.n_tfs
            f.attrs['n_slots'] = n_slots

            # TF vocabulary
            tf_vocab = self.motif_library.motif_names
            f.create_dataset('tf_vocab', data=json.dumps(tf_vocab))

        print(f"Saved {n_samples} samples to {output_path}")
        return sequences, profiles, binding_sites


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic binding data for BEACON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configurations:
  --config easy      : 1-2 sites, 10 TFs, no overlap, no noise (current)
  --config medium    : 2-4 sites, 25 TFs, some overlap, low noise
  --config hard      : 3-8 sites, 50 TFs, overlap allowed, medium noise (Stage A)
  --config extreme   : 5-12 sites, 100 TFs, high overlap, high noise

Examples:
  python scripts/generate_synthetic_data.py --config hard --n-train 100000
  python scripts/generate_synthetic_data.py --n-tfs 50 --min-sites 3 --max-sites 8
        """
    )

    # Output
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "processed" / "synthetic",
                        help="Output directory")

    # Quick config presets
    parser.add_argument("--config", type=str,
                        choices=["easy", "medium", "hard", "extreme"],
                        help="Use preset configuration")

    # Dataset sizes
    parser.add_argument("--n-train", type=int, default=100000,
                        help="Number of training samples")
    parser.add_argument("--n-val", type=int, default=10000,
                        help="Number of validation samples")
    parser.add_argument("--n-test", type=int, default=10000,
                        help="Number of test samples")

    # Sequence parameters
    parser.add_argument("--seq-length", type=int, default=2000,
                        help="Sequence length")
    parser.add_argument("--n-slots", type=int, default=16,
                        help="Number of binding site slots")

    # Binding site parameters
    parser.add_argument("--n-tfs", type=int, default=50,
                        help="Number of TF classes")
    parser.add_argument("--min-sites", type=int, default=3,
                        help="Minimum binding sites per sequence")
    parser.add_argument("--max-sites", type=int, default=8,
                        help="Maximum binding sites per sequence")
    parser.add_argument("--allow-overlap", action="store_true", default=True,
                        help="Allow overlapping binding sites")
    parser.add_argument("--no-overlap", dest="allow_overlap", action="store_false",
                        help="Disable overlapping binding sites")

    # Profile parameters
    parser.add_argument("--profile-sigma", type=float, default=30.0,
                        help="Width of Gaussian peaks in profile")
    parser.add_argument("--noise-sigma", type=float, default=0.15,
                        help="Noise standard deviation")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Apply config presets
    configs = {
        "easy": {
            "n_tfs": 10, "min_sites": 1, "max_sites": 2,
            "allow_overlap": False, "noise_sigma": 0.0,
            "seq_length": 1000, "n_train": 32000, "n_val": 4000, "n_test": 4000,
        },
        "medium": {
            "n_tfs": 25, "min_sites": 2, "max_sites": 4,
            "allow_overlap": True, "noise_sigma": 0.1,
            "seq_length": 1500, "n_train": 64000, "n_val": 8000, "n_test": 8000,
        },
        "hard": {
            "n_tfs": 50, "min_sites": 3, "max_sites": 8,
            "allow_overlap": True, "noise_sigma": 0.15,
            "seq_length": 2000, "n_train": 100000, "n_val": 10000, "n_test": 10000,
        },
        "extreme": {
            "n_tfs": 100, "min_sites": 5, "max_sites": 12,
            "allow_overlap": True, "noise_sigma": 0.2,
            "seq_length": 2000, "n_train": 150000, "n_val": 15000, "n_test": 15000,
        },
    }

    if args.config:
        cfg = configs[args.config]
        for key, value in cfg.items():
            setattr(args, key, value)

    print("=" * 70)
    print("BEACON Synthetic Data Generator")
    print("=" * 70)
    print(f"Configuration: {args.config or 'custom'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sequence length: {args.seq_length}")
    print(f"TF classes: {args.n_tfs}")
    print(f"Sites per sequence: {args.min_sites}-{args.max_sites}")
    print(f"Allow overlap: {args.allow_overlap}")
    print(f"Profile noise (sigma): {args.noise_sigma}")
    print(f"Training samples: {args.n_train:,}")
    print(f"Validation samples: {args.n_val:,}")
    print(f"Test samples: {args.n_test:,}")
    print()

    # Create generator
    generator = SyntheticDataGenerator(
        seq_length=args.seq_length,
        n_tfs=args.n_tfs,
        min_sites=args.min_sites,
        max_sites=args.max_sites,
        allow_overlap=args.allow_overlap,
        profile_sigma=args.profile_sigma,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )

    # Generate datasets
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating training data...")
    generator.generate_dataset(
        args.n_train,
        args.output_dir / "train.h5",
        n_slots=args.n_slots,
    )

    print("Generating validation data...")
    generator.rng = np.random.RandomState(args.seed + 1)
    generator.generate_dataset(
        args.n_val,
        args.output_dir / "val.h5",
        n_slots=args.n_slots,
    )

    print("Generating test data...")
    generator.rng = np.random.RandomState(args.seed + 2)
    generator.generate_dataset(
        args.n_test,
        args.output_dir / "test.h5",
        n_slots=args.n_slots,
    )

    # Save motif library
    print("Saving motif library...")
    generator.motif_library.save(args.output_dir / "motifs.h5")

    # Save summary
    summary = {
        "config": args.config or "custom",
        "seq_length": args.seq_length,
        "n_tfs": args.n_tfs,
        "n_slots": args.n_slots,
        "min_sites": args.min_sites,
        "max_sites": args.max_sites,
        "allow_overlap": args.allow_overlap,
        "noise_sigma": args.noise_sigma,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "tfs": generator.motif_library.motif_names,
    }

    with open(args.output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("Generation complete!")
    print("=" * 70)
    print(f"Data saved to: {args.output_dir}")
    print()
    print("To train BEACON on this data:")
    print(f"  python scripts/train.py --data-dir {args.output_dir} --seq-len {args.seq_length} --n-slots {args.n_slots}")


if __name__ == "__main__":
    main()
