"""
In-Silico Mutagenesis (ISM) Variant Effect Scoring for BEACON.

Provides systematic variant effect prediction by comparing model outputs
between reference and mutant sequences. Scores combine profile changes
with binding site occupancy changes for comprehensive variant impact.

Supports: SNVs, indels, multi-base substitutions, batch processing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class VariantEffect:
    """Container for a single variant's predicted effect."""
    chrom: str = ""
    pos: int = 0
    ref: str = ""
    alt: str = ""
    # Profile effect
    profile_delta_sum: float = 0.0
    profile_delta_max: float = 0.0
    profile_delta_local: float = 0.0  # Effect within 200bp of variant
    # Occupancy effect
    occupancy_delta_max: float = 0.0
    occupancy_delta_mean: float = 0.0
    # Binding site disruption
    n_disrupted: int = 0
    n_gained: int = 0
    disrupted_tfs: List[int] = field(default_factory=list)
    gained_tfs: List[int] = field(default_factory=list)
    # Combined score
    ism_score: float = 0.0
    # Per-slot details
    slot_effects: List[Dict] = field(default_factory=list)


class VariantScorer:
    """
    Score variant effects using BEACON's compositional predictions.

    Uses ISM: compare model outputs for reference vs alternate allele,
    then quantify changes in binding profiles and slot occupancies.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        local_window: int = 200,
        occupancy_threshold: float = 0.3,
        disruption_threshold: float = 0.2,
    ):
        """
        Args:
            model: Trained BEACON model
            device: Device for computation
            local_window: Window around variant for local profile effect
            occupancy_threshold: Min occupancy to consider a site active
            disruption_threshold: Min occupancy change to flag disruption
        """
        self.model = model
        self.device = device
        self.local_window = local_window
        self.occupancy_threshold = occupancy_threshold
        self.disruption_threshold = disruption_threshold

        self.model.eval()
        self.model.to(device)
        self.seq_len = model.seq_len if hasattr(model, 'seq_len') else 2000

    @staticmethod
    def encode_sequence(seq: str) -> np.ndarray:
        """Convert ACGT string to one-hot [L, 4]."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        L = len(seq)
        onehot = np.zeros((L, 4), dtype=np.float32)
        for i, base in enumerate(seq.upper()):
            if base in mapping:
                onehot[i, mapping[base]] = 1.0
            else:
                onehot[i] = 0.25  # N
        return onehot

    def apply_snv(
        self,
        ref_seq: np.ndarray,
        position: int,
        alt_base: int,
    ) -> np.ndarray:
        """Apply a single nucleotide variant. alt_base: 0=A, 1=C, 2=G, 3=T."""
        alt_seq = ref_seq.copy()
        alt_seq[position] = 0.0
        alt_seq[position, alt_base] = 1.0
        return alt_seq

    def apply_substitution(
        self,
        ref_seq: np.ndarray,
        start: int,
        ref_allele: np.ndarray,
        alt_allele: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Apply multi-base substitution (same length ref and alt)."""
        if len(ref_allele) != len(alt_allele):
            return None  # Use apply_indel for length-changing variants
        alt_seq = ref_seq.copy()
        alt_seq[start:start + len(alt_allele)] = alt_allele
        return alt_seq

    def apply_indel(
        self,
        ref_seq: np.ndarray,
        position: int,
        ref_len: int,
        alt_allele: np.ndarray,
    ) -> np.ndarray:
        """
        Apply insertion or deletion, padding/trimming to maintain seq_len.

        For insertions: insert alt, trim from edges to maintain length.
        For deletions: remove ref bases, pad from edges.
        """
        L = ref_seq.shape[0]
        alt_len = len(alt_allele)

        # Build new sequence
        pre = ref_seq[:position]
        post = ref_seq[position + ref_len:]

        new_seq = np.concatenate([pre, alt_allele, post], axis=0)

        # Pad or trim to original length
        if len(new_seq) > L:
            # Trim symmetrically from edges
            excess = len(new_seq) - L
            trim_left = excess // 2
            trim_right = excess - trim_left
            new_seq = new_seq[trim_left:len(new_seq) - trim_right]
        elif len(new_seq) < L:
            # Pad with N (0.25 each base) at edges
            deficit = L - len(new_seq)
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            n_pad = np.full((1, 4), 0.25, dtype=np.float32)
            left_pad = np.repeat(n_pad, pad_left, axis=0)
            right_pad = np.repeat(n_pad, pad_right, axis=0)
            new_seq = np.concatenate([left_pad, new_seq, right_pad], axis=0)

        return new_seq

    @torch.no_grad()
    def score_variant(
        self,
        ref_seq: np.ndarray,
        alt_seq: np.ndarray,
        variant_pos: int,
        chrom: str = "",
        ref_allele: str = "",
        alt_allele: str = "",
    ) -> VariantEffect:
        """
        Score a single variant by comparing ref vs alt model outputs.

        Args:
            ref_seq: Reference one-hot [L, 4]
            alt_seq: Alternate one-hot [L, 4]
            variant_pos: Position of variant in sequence
            chrom, ref_allele, alt_allele: Metadata for output
        """
        ref_t = torch.from_numpy(ref_seq).unsqueeze(0).to(self.device)
        alt_t = torch.from_numpy(alt_seq).unsqueeze(0).to(self.device)

        ref_out = self.model(ref_t)
        alt_out = self.model(alt_t)

        effect = VariantEffect(
            chrom=chrom,
            pos=variant_pos,
            ref=ref_allele,
            alt=alt_allele,
        )

        # Profile effects
        ref_profile = ref_out["profile"][0].cpu().numpy()
        alt_profile = alt_out["profile"][0].cpu().numpy()
        delta_profile = alt_profile - ref_profile

        effect.profile_delta_sum = float(np.abs(delta_profile).sum())
        effect.profile_delta_max = float(np.abs(delta_profile).max())

        # Local effect around variant
        half_w = self.local_window // 2
        local_start = max(0, variant_pos - half_w)
        local_end = min(len(delta_profile), variant_pos + half_w)
        effect.profile_delta_local = float(
            np.abs(delta_profile[local_start:local_end]).sum()
        )

        # Occupancy effects
        ref_occ = ref_out["occupancy"][0].squeeze(-1).cpu().numpy()
        alt_occ = alt_out["occupancy"][0].squeeze(-1).cpu().numpy()
        delta_occ = alt_occ - ref_occ

        effect.occupancy_delta_max = float(np.abs(delta_occ).max())
        effect.occupancy_delta_mean = float(np.abs(delta_occ).mean())

        # TF predictions
        ref_tfs = ref_out["tf_logits"][0].argmax(dim=-1).cpu().numpy()
        alt_tfs = alt_out["tf_logits"][0].argmax(dim=-1).cpu().numpy()

        # Identify disrupted and gained sites
        n_slots = len(ref_occ)
        for k in range(n_slots):
            slot_effect = {
                "slot_id": k,
                "ref_occupancy": float(ref_occ[k]),
                "alt_occupancy": float(alt_occ[k]),
                "delta_occupancy": float(delta_occ[k]),
                "ref_tf": int(ref_tfs[k]),
                "alt_tf": int(alt_tfs[k]),
            }
            effect.slot_effects.append(slot_effect)

            # Disrupted: was active, now less active
            if ref_occ[k] > self.occupancy_threshold:
                if delta_occ[k] < -self.disruption_threshold:
                    effect.n_disrupted += 1
                    effect.disrupted_tfs.append(int(ref_tfs[k]))

            # Gained: was inactive, now active
            if alt_occ[k] > self.occupancy_threshold:
                if delta_occ[k] > self.disruption_threshold:
                    effect.n_gained += 1
                    effect.gained_tfs.append(int(alt_tfs[k]))

        # Combined ISM score
        effect.ism_score = (
            effect.profile_delta_local
            + 10.0 * effect.occupancy_delta_max
        )

        return effect

    @torch.no_grad()
    def score_variants_batch(
        self,
        ref_seq: np.ndarray,
        variants: List[Dict],
        batch_size: int = 32,
    ) -> List[VariantEffect]:
        """
        Score multiple variants against the same reference sequence.

        Args:
            ref_seq: Reference one-hot [L, 4]
            variants: List of dicts with keys: pos, ref, alt (and optionally chrom)
            batch_size: Processing batch size

        Returns:
            List of VariantEffect objects
        """
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Get reference output once
        ref_t = torch.from_numpy(ref_seq).unsqueeze(0).to(self.device)
        ref_out = self.model(ref_t)

        results = []

        # Create all alternate sequences
        alt_seqs = []
        variant_positions = []
        for var in variants:
            pos = var['pos']
            ref_allele = var.get('ref', '')
            alt_allele = var.get('alt', '')

            if len(ref_allele) == 1 and len(alt_allele) == 1:
                # SNV
                alt_base = base_map.get(alt_allele.upper(), 0)
                alt_seq = self.apply_snv(ref_seq, pos, alt_base)
            elif len(ref_allele) == len(alt_allele):
                # Multi-base substitution
                alt_onehot = self.encode_sequence(alt_allele)
                alt_seq = self.apply_substitution(
                    ref_seq, pos,
                    ref_seq[pos:pos + len(ref_allele)],
                    alt_onehot,
                )
                if alt_seq is None:
                    continue
            else:
                # Indel
                alt_onehot = self.encode_sequence(alt_allele)
                alt_seq = self.apply_indel(ref_seq, pos, len(ref_allele), alt_onehot)

            alt_seqs.append(alt_seq)
            variant_positions.append(var)

        # Batch process alternate sequences
        for batch_start in range(0, len(alt_seqs), batch_size):
            batch_end = min(batch_start + batch_size, len(alt_seqs))
            batch_alts = alt_seqs[batch_start:batch_end]
            batch_vars = variant_positions[batch_start:batch_end]

            alt_batch = torch.from_numpy(np.stack(batch_alts)).to(self.device)
            alt_out = self.model(alt_batch)

            # Expand ref outputs to batch size
            bs = alt_batch.shape[0]

            for i in range(bs):
                var = batch_vars[i]
                effect = VariantEffect(
                    chrom=var.get('chrom', ''),
                    pos=var['pos'],
                    ref=var.get('ref', ''),
                    alt=var.get('alt', ''),
                )

                # Profile effects
                ref_profile = ref_out["profile"][0].cpu().numpy()
                alt_profile = alt_out["profile"][i].cpu().numpy()
                delta = alt_profile - ref_profile

                effect.profile_delta_sum = float(np.abs(delta).sum())
                effect.profile_delta_max = float(np.abs(delta).max())

                vpos = var['pos']
                half_w = self.local_window // 2
                ls = max(0, vpos - half_w)
                le = min(len(delta), vpos + half_w)
                effect.profile_delta_local = float(np.abs(delta[ls:le]).sum())

                # Occupancy effects
                ref_occ = ref_out["occupancy"][0].squeeze(-1).cpu().numpy()
                alt_occ = alt_out["occupancy"][i].squeeze(-1).cpu().numpy()
                delta_occ = alt_occ - ref_occ

                effect.occupancy_delta_max = float(np.abs(delta_occ).max())
                effect.occupancy_delta_mean = float(np.abs(delta_occ).mean())

                # TF changes
                ref_tfs = ref_out["tf_logits"][0].argmax(dim=-1).cpu().numpy()
                alt_tfs = alt_out["tf_logits"][i].argmax(dim=-1).cpu().numpy()

                for k in range(len(ref_occ)):
                    if ref_occ[k] > self.occupancy_threshold:
                        if delta_occ[k] < -self.disruption_threshold:
                            effect.n_disrupted += 1
                            effect.disrupted_tfs.append(int(ref_tfs[k]))
                    if alt_occ[k] > self.occupancy_threshold:
                        if delta_occ[k] > self.disruption_threshold:
                            effect.n_gained += 1
                            effect.gained_tfs.append(int(alt_tfs[k]))

                effect.ism_score = (
                    effect.profile_delta_local
                    + 10.0 * effect.occupancy_delta_max
                )
                results.append(effect)

        return results

    @torch.no_grad()
    def saturation_mutagenesis(
        self,
        ref_seq: np.ndarray,
        start: int = 0,
        end: Optional[int] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Perform saturation mutagenesis over a region.

        Tests all 3 possible SNVs at each position in [start, end).

        Args:
            ref_seq: Reference one-hot [L, 4]
            start: Start position
            end: End position (default: seq_len)
            batch_size: Batch size for processing

        Returns:
            ISM scores [positions, 4] (one score per possible base at each position)
        """
        if end is None:
            end = min(len(ref_seq), self.seq_len)

        n_positions = end - start
        scores = np.zeros((n_positions, 4), dtype=np.float32)

        # Get reference output
        ref_t = torch.from_numpy(ref_seq).unsqueeze(0).to(self.device)
        ref_out = self.model(ref_t)
        ref_profile = ref_out["profile"][0].cpu().numpy()
        ref_occ = ref_out["occupancy"][0].squeeze(-1).cpu().numpy()

        # Generate all variants
        variants = []
        for pos in range(start, end):
            ref_base = ref_seq[pos].argmax()
            for alt_base in range(4):
                alt_seq = self.apply_snv(ref_seq, pos, alt_base)
                variants.append((pos, alt_base, alt_seq))

        # Batch process
        for batch_start in range(0, len(variants), batch_size):
            batch_end = min(batch_start + batch_size, len(variants))
            batch = variants[batch_start:batch_end]

            alt_batch = torch.from_numpy(
                np.stack([v[2] for v in batch])
            ).to(self.device)
            alt_out = self.model(alt_batch)

            for i, (pos, alt_base, _) in enumerate(batch):
                alt_profile = alt_out["profile"][i].cpu().numpy()
                alt_occ = alt_out["occupancy"][i].squeeze(-1).cpu().numpy()

                # Local profile delta
                half_w = self.local_window // 2
                ls = max(0, pos - half_w)
                le = min(len(ref_profile), pos + half_w)
                profile_delta = float(np.abs(alt_profile[ls:le] - ref_profile[ls:le]).sum())

                # Occupancy delta
                occ_delta = float(np.abs(alt_occ - ref_occ).max())

                scores[pos - start, alt_base] = profile_delta + 10.0 * occ_delta

        return scores

    def to_bed(self, effects: List[VariantEffect], output_path: str):
        """Export variant effects as BED file."""
        with open(output_path, 'w') as f:
            f.write("#chrom\tstart\tend\tref\talt\tism_score\tprofile_delta\tocc_delta\tn_disrupted\tn_gained\n")
            for e in sorted(effects, key=lambda x: -x.ism_score):
                f.write(
                    f"{e.chrom}\t{e.pos}\t{e.pos + max(1, len(e.ref))}\t"
                    f"{e.ref}\t{e.alt}\t{e.ism_score:.4f}\t"
                    f"{e.profile_delta_local:.4f}\t{e.occupancy_delta_max:.4f}\t"
                    f"{e.n_disrupted}\t{e.n_gained}\n"
                )
