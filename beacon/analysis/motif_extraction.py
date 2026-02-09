"""
Per-Slot Motif Extraction (replaces TF-MoDISco clustering).

Complexity: O(n_slots) per sequence instead of O(n_seqlets^2) for TF-MoDISco.

Produces TF-MoDISco-compatible output format:
  - Position Frequency Matrices (PFMs)
  - Contribution Weight Matrices (CWMs)
  - Hypothetical CWMs
  - Seqlet coordinates
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ExtractedMotif:
    """Container for a single extracted motif."""
    slot_id: int
    tf_name: Optional[str] = None
    pfm: Optional[np.ndarray] = None           # [M, 4] position frequency matrix
    cwm: Optional[np.ndarray] = None           # [M, 4] contribution weight matrix
    hypothetical_cwm: Optional[np.ndarray] = None  # [M, 4]
    seqlet_coords: List[Tuple[int, int, int]] = field(default_factory=list)  # (sample_idx, start, end)
    n_seqlets: int = 0
    info_content: float = 0.0


class PerSlotMotifExtractor:
    """
    Extract motifs from BEACON per-slot attributions.

    For each slot the extractor:
      1. Identifies high-importance positions (above percentile threshold).
      2. Extracts contributing sequence segments (seqlets).
      3. Weights them by importance to build CWMs.
      4. Trims to the core motif by removing low-information flanks.

    Output is compatible with TF-MoDISco downstream tools.
    """

    def __init__(
        self,
        window_size: int = 24,
        percentile_threshold: float = 95.0,
        min_seqlets: int = 5,
        trim_ic_threshold: float = 0.2,
        top_k_peaks: int = 15,
    ):
        self.window_size = window_size
        self.percentile_threshold = percentile_threshold
        self.min_seqlets = min_seqlets
        self.trim_ic_threshold = trim_ic_threshold
        self.top_k_peaks = top_k_peaks

    def extract_from_importance(
        self,
        importance: np.ndarray,
        sequences: np.ndarray,
        slot_id: int = 0,
        tf_name: Optional[str] = None,
    ) -> ExtractedMotif:
        """
        Extract a motif from averaged importance scores.

        Args:
            importance: [L, 4] averaged importance map for one slot
            sequences:  [N, L, 4] one-hot sequences that contributed
            slot_id:    Slot index
            tf_name:    Optional TF name label

        Returns:
            ExtractedMotif with PFM, CWM, and seqlet coordinates
        """
        total_imp = np.abs(importance).sum(axis=1)  # [L]

        # Smooth for peak detection
        kernel = np.ones(3) / 3.0
        smoothed = np.convolve(total_imp, kernel, mode="same")

        threshold = np.percentile(smoothed, self.percentile_threshold)
        peaks = self._find_peaks(smoothed, threshold)
        peaks = peaks[: self.top_k_peaks]

        if len(peaks) == 0:
            return ExtractedMotif(slot_id=slot_id, tf_name=tf_name)

        # Extract windows and build CWM
        half_w = self.window_size // 2
        cwm, seqlet_coords = self._build_cwm(
            importance, sequences, peaks, half_w
        )

        if cwm is None:
            return ExtractedMotif(slot_id=slot_id, tf_name=tf_name)

        # Build PFM from seqlet sequences
        pfm = self._build_pfm(sequences, seqlet_coords, self.window_size)

        # Hypothetical CWM = importance values at each position
        hyp_cwm = cwm.copy()

        # Trim to core motif
        cwm_trimmed, pfm_trimmed, hyp_trimmed, trim_start, trim_end = (
            self._trim_motif(cwm, pfm, hyp_cwm)
        )

        ic = self._information_content(pfm_trimmed)

        return ExtractedMotif(
            slot_id=slot_id,
            tf_name=tf_name,
            pfm=pfm_trimmed,
            cwm=cwm_trimmed,
            hypothetical_cwm=hyp_trimmed,
            seqlet_coords=seqlet_coords,
            n_seqlets=len(seqlet_coords),
            info_content=float(ic),
        )

    def extract_all_slots(
        self,
        per_slot_importance: np.ndarray,
        sequences: np.ndarray,
        occupancy: np.ndarray,
        tf_names: Optional[List[str]] = None,
        tf_predictions: Optional[np.ndarray] = None,
        occupancy_threshold: float = 0.3,
    ) -> List[ExtractedMotif]:
        """
        Extract motifs from all active slots.

        Args:
            per_slot_importance: [K, L, 4] importance per slot (averaged)
            sequences:           [N, L, 4] contributing sequences
            occupancy:           [K] mean occupancy per slot
            tf_names:            Optional list of TF name labels
            tf_predictions:      [K] predicted TF index per slot
            occupancy_threshold: Minimum occupancy to consider a slot active

        Returns:
            List of ExtractedMotif, one per active slot
        """
        n_slots = per_slot_importance.shape[0]
        motifs = []

        for k in range(n_slots):
            if occupancy[k] < occupancy_threshold:
                continue

            tf_name = None
            if tf_names is not None and tf_predictions is not None:
                tf_idx = int(tf_predictions[k])
                if 0 <= tf_idx < len(tf_names):
                    tf_name = tf_names[tf_idx]

            motif = self.extract_from_importance(
                per_slot_importance[k], sequences, slot_id=k, tf_name=tf_name
            )
            if motif.pfm is not None:
                motifs.append(motif)

        return motifs

    def to_modisco_format(
        self, motifs: List[ExtractedMotif]
    ) -> Dict[str, Dict]:
        """
        Convert extracted motifs to TF-MoDISco-compatible format.

        Returns:
            Dictionary mimicking TF-MoDISco output structure with keys
            "metacluster_idx_to_submetacluster_results".
        """
        results = {}
        for i, motif in enumerate(motifs):
            pattern_key = f"pattern_{i}"
            pattern = {
                "sequence": motif.pfm.tolist() if motif.pfm is not None else [],
                "contrib_scores": motif.cwm.tolist() if motif.cwm is not None else [],
                "hypothetical_contribs": (
                    motif.hypothetical_cwm.tolist()
                    if motif.hypothetical_cwm is not None
                    else []
                ),
                "seqlets_and_alnmts": {
                    "seqlets": [
                        {"example_idx": s[0], "start": s[1], "end": s[2]}
                        for s in motif.seqlet_coords
                    ]
                },
                "num_seqlets": motif.n_seqlets,
                "slot_id": motif.slot_id,
                "tf_name": motif.tf_name,
                "info_content_bits": motif.info_content,
            }
            results[pattern_key] = pattern

        return {
            "metacluster_idx_to_submetacluster_results": {
                "metacluster_0": {
                    "activity_pattern": [1] * len(motifs),
                    "seqlets_to_patterns_result": {
                        "patterns": results,
                    },
                }
            }
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peaks(
        signal: np.ndarray, threshold: float
    ) -> List[Tuple[float, int]]:
        """Find local maxima above threshold, sorted by descending height."""
        peaks = []
        for i in range(2, len(signal) - 2):
            if (
                signal[i] > threshold
                and signal[i] >= signal[i - 1]
                and signal[i] >= signal[i + 1]
                and signal[i] >= signal[i - 2]
                and signal[i] >= signal[i + 2]
            ):
                peaks.append((signal[i], i))
        peaks.sort(reverse=True)
        return peaks

    @staticmethod
    def _build_cwm(
        importance: np.ndarray,
        sequences: np.ndarray,
        peaks: List[Tuple[float, int]],
        half_w: int,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int]]]:
        """Build contribution weight matrix from peak windows."""
        window = 2 * half_w
        accumulator = np.zeros((window, 4))
        weight_sum = 0.0
        coords = []

        for score, pos in peaks:
            start = max(0, pos - half_w)
            end = min(len(importance), pos + half_w)
            if end - start != window:
                continue
            accumulator += importance[start:end] * score
            weight_sum += score
            coords.append((0, start, end))

        if weight_sum < 1e-12:
            return None, coords
        accumulator /= weight_sum
        return accumulator, coords

    @staticmethod
    def _build_pfm(
        sequences: np.ndarray,
        coords: List[Tuple[int, int, int]],
        window: int,
    ) -> np.ndarray:
        """Build PFM from seqlet coordinates."""
        pfm = np.zeros((window, 4))
        count = 0
        for sample_idx, start, end in coords:
            if end - start != window:
                continue
            idx = sample_idx % len(sequences)
            pfm += sequences[idx, start:end]
            count += 1
        if count > 0:
            pfm /= count
        else:
            pfm[:] = 0.25
        return pfm

    def _trim_motif(
        self,
        cwm: np.ndarray,
        pfm: np.ndarray,
        hyp_cwm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Trim low-information flanks."""
        ic = np.log2(4) + np.sum(pfm * np.log2(pfm + 1e-8), axis=1)
        informative = ic > self.trim_ic_threshold

        if informative.sum() < 4:
            return cwm, pfm, hyp_cwm, 0, len(cwm)

        first = int(np.argmax(informative))
        last = len(informative) - 1 - int(np.argmax(informative[::-1]))
        s, e = first, last + 1
        return cwm[s:e], pfm[s:e], hyp_cwm[s:e], s, e

    @staticmethod
    def _information_content(pfm: np.ndarray) -> float:
        """Mean information content in bits."""
        if pfm is None or len(pfm) == 0:
            return 0.0
        ic = np.log2(4) + np.sum(pfm * np.log2(pfm + 1e-8), axis=1)
        return float(ic.mean())
