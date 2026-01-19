"""
Positional encodings for BEACON.

Includes helical phase-aware encoding that captures DNA's 10.5bp periodicity,
enabling the model to understand TF cooperativity on the same face of the helix.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Used as a baseline and combined with helical encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Tensor with positional encoding added [B, L, D]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HelicalPositionalEncoding(nn.Module):
    """
    Helical phase-aware positional encoding for DNA sequences.

    DNA has a helical structure with ~10.5 bp per turn. TFs often cooperate
    when they bind on the same face of the helix. This encoding captures:

    1. Absolute position (standard sinusoidal)
    2. Helical phase (10.5 bp periodicity)
    3. Half-helical phase (5.25 bp for minor groove)
    4. Learned position embeddings

    The helical components help the model understand which positions
    are on the same face of the DNA helix.
    """

    DNA_HELIX_PERIOD = 10.5  # bp per full turn
    MINOR_GROOVE_PERIOD = 5.25  # bp for minor groove accessibility

    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        dropout: float = 0.1,
        n_helical_harmonics: int = 4,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Dimensions for each component
        # We allocate dimensions: helical, standard sinusoidal, learned
        self.n_helical = n_helical_harmonics * 4  # sin/cos for major and minor groove
        self.n_sinusoidal = d_model - self.n_helical

        # Helical encoding (fixed, based on DNA structure)
        helical_pe = self._create_helical_encoding(max_len, n_helical_harmonics)
        self.register_buffer("helical_pe", helical_pe)

        # Standard sinusoidal encoding for remaining dimensions
        sinusoidal_pe = self._create_sinusoidal_encoding(max_len, self.n_sinusoidal)
        self.register_buffer("sinusoidal_pe", sinusoidal_pe)

        # Learnable scaling factors for helical vs sinusoidal
        if learnable_scale:
            self.helical_scale = nn.Parameter(torch.ones(1))
            self.sinusoidal_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("helical_scale", torch.ones(1))
            self.register_buffer("sinusoidal_scale", torch.ones(1))

        # Optional learned residual
        self.learned_bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def _create_helical_encoding(
        self, max_len: int, n_harmonics: int
    ) -> torch.Tensor:
        """
        Create helical positional encoding based on DNA structure.

        Uses multiple harmonics of the helical period to capture
        phase relationships at different scales.
        """
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Major groove periodicity (10.5 bp)
        major_freqs = torch.tensor([
            2 * math.pi / (self.DNA_HELIX_PERIOD * (i + 1))
            for i in range(n_harmonics)
        ])

        # Minor groove periodicity (5.25 bp)
        minor_freqs = torch.tensor([
            2 * math.pi / (self.MINOR_GROOVE_PERIOD * (i + 1))
            for i in range(n_harmonics)
        ])

        # Compute sin/cos for each harmonic
        pe_parts = []

        # Major groove harmonics
        for freq in major_freqs:
            pe_parts.append(torch.sin(position * freq))
            pe_parts.append(torch.cos(position * freq))

        # Minor groove harmonics
        for freq in minor_freqs:
            pe_parts.append(torch.sin(position * freq))
            pe_parts.append(torch.cos(position * freq))

        helical_pe = torch.cat(pe_parts, dim=1)
        return helical_pe.unsqueeze(0)  # [1, max_len, n_helical]

    def _create_sinusoidal_encoding(self, max_len: int, d: int) -> torch.Tensor:
        """Create standard sinusoidal encoding."""
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use different frequency bands than helical
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe.unsqueeze(0)  # [1, max_len, n_sinusoidal]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Tensor with helical positional encoding added [B, L, D]
        """
        seq_len = x.size(1)

        # Combine helical and sinusoidal encodings
        helical = self.helical_pe[:, :seq_len, :] * self.helical_scale
        sinusoidal = self.sinusoidal_pe[:, :seq_len, :] * self.sinusoidal_scale

        # Concatenate along feature dimension
        pe = torch.cat([helical, sinusoidal], dim=-1)

        # Add to input with learned bias
        x = x + pe + self.learned_bias

        return self.dropout(x)

    def get_helical_phase(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get the helical phase for given positions.

        Useful for analyzing which positions are on the same face of the helix.

        Args:
            positions: Position indices [B, N]

        Returns:
            Phase values in [0, 2π] representing position on helix [B, N]
        """
        phase = (2 * math.pi * positions.float() / self.DNA_HELIX_PERIOD) % (2 * math.pi)
        return phase

    def same_face_mask(
        self,
        positions: torch.Tensor,
        tolerance: float = 0.3,
    ) -> torch.Tensor:
        """
        Create a mask indicating which position pairs are on the same face.

        Args:
            positions: Position indices [B, N]
            tolerance: Phase difference tolerance (fraction of π)

        Returns:
            Boolean mask [B, N, N] where True means same face
        """
        phase = self.get_helical_phase(positions)  # [B, N]

        # Compute pairwise phase differences
        phase_diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)  # [B, N, N]

        # Normalize to [0, π]
        phase_diff = torch.abs(phase_diff)
        phase_diff = torch.minimum(phase_diff, 2 * math.pi - phase_diff)

        # Same face if phase difference is small
        same_face = phase_diff < (tolerance * math.pi)

        return same_face


class RelativeHelicalEncoding(nn.Module):
    """
    Relative positional encoding with helical awareness.

    Instead of absolute positions, encodes relative distances with
    awareness of helical phase relationships.
    """

    def __init__(
        self,
        d_model: int,
        max_relative_position: int = 512,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_relative_position = max_relative_position

        # Learned relative position embeddings
        self.relative_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )

        # Helical phase embeddings (discretized into 21 bins for ~10.5 bp period)
        self.n_phase_bins = 21
        self.phase_embeddings = nn.Embedding(self.n_phase_bins, d_model)

    def forward(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relative positional encodings.

        Args:
            query_positions: Query position indices [B, Q]
            key_positions: Key position indices [B, K]

        Returns:
            Relative encodings [B, Q, K, D]
        """
        # Relative distances
        relative_pos = query_positions.unsqueeze(-1) - key_positions.unsqueeze(-2)
        relative_pos = relative_pos.clamp(
            -self.max_relative_position, self.max_relative_position
        )
        relative_pos = relative_pos + self.max_relative_position

        # Look up embeddings
        rel_emb = self.relative_embeddings(relative_pos)

        # Add helical phase information
        phase_diff = (query_positions.unsqueeze(-1) - key_positions.unsqueeze(-2)).float()
        phase_bins = ((phase_diff / 10.5) % 1.0 * self.n_phase_bins).long()
        phase_bins = phase_bins.clamp(0, self.n_phase_bins - 1)
        phase_emb = self.phase_embeddings(phase_bins)

        return rel_emb + phase_emb
