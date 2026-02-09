"""
Output heads for BEACON.

Each slot produces three outputs:
1. Position: Where is the binding event? [B, K, 1]
2. TF Identity: What TF is binding? [B, K, P]
3. Occupancy: How strong is the binding? [B, K, 1]

Additionally:
- Profile Head reconstructs binding profiles for BPNet compatibility.
- AttributionHead provides per-base importance per slot (replaces DeepSHAP).
- MotifEmbeddingHead projects slots to an open-vocabulary motif space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class PositionHead(nn.Module):
    """
    Predicts binding site position from slot representation.

    Can output either:
    - Continuous position (normalized 0-1)
    - Position distribution over sequence length
    - Gaussian parameters (mean, std) for soft position
    """

    def __init__(
        self,
        slot_dim: int = 256,
        hidden_dim: int = 256,
        output_mode: str = "gaussian",  # "continuous", "distribution", "gaussian"
        seq_len: int = 1000,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.seq_len = seq_len

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        if output_mode == "continuous":
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        elif output_mode == "distribution":
            self.output = nn.Linear(hidden_dim, seq_len)
        elif output_mode == "gaussian":
            # Output mean and log-std
            self.output_mu = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.output_log_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: Slot representations [B, K, D]

        Returns:
            Depends on output_mode:
            - "continuous": [B, K, 1] normalized positions
            - "distribution": [B, K, L] position probabilities
            - "gaussian": [B, K, 2] (mean, std) pairs
        """
        h = self.mlp(slots)

        if self.output_mode == "continuous":
            return self.output(h)  # [B, K, 1]

        elif self.output_mode == "distribution":
            logits = self.output(h)  # [B, K, L]
            return F.softmax(logits, dim=-1)

        elif self.output_mode == "gaussian":
            mu = self.output_mu(h)  # [B, K, 1]
            log_sigma = self.output_log_sigma(h)  # [B, K, 1]
            sigma = F.softplus(log_sigma) + 0.01  # Ensure positive, min std
            return torch.cat([mu, sigma], dim=-1)  # [B, K, 2]

    def sample_positions(
        self,
        slots: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample discrete positions from predicted distribution.

        Args:
            slots: Slot representations [B, K, D]
            n_samples: Number of samples per slot

        Returns:
            Sampled positions [B, K, n_samples]
        """
        output = self.forward(slots)

        if self.output_mode == "gaussian":
            mu, sigma = output[..., 0:1], output[..., 1:2]
            # Sample from Gaussian, clamp to valid range
            samples = mu + sigma * torch.randn(
                *mu.shape[:-1], n_samples, device=mu.device
            )
            samples = samples.clamp(0, 1)
            # Convert to discrete positions
            positions = (samples * (self.seq_len - 1)).round().long()

        elif self.output_mode == "distribution":
            # Sample from categorical
            dist = torch.distributions.Categorical(probs=output)
            positions = dist.sample((n_samples,)).permute(1, 2, 0)

        else:
            # Continuous mode - just round
            positions = (output * (self.seq_len - 1)).round().long()
            positions = positions.expand(*positions.shape[:-1], n_samples)

        return positions


class TFIdentityHead(nn.Module):
    """
    Predicts TF identity from slot representation.

    Outputs a distribution over TF types/families, enabling the model
    to explicitly represent which TF is predicted to bind.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        hidden_dim: int = 256,
        n_tfs: int = 100,
        use_pwm_similarity: bool = False,
        pwm_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            slot_dim: Slot representation dimension
            hidden_dim: Hidden layer dimension
            n_tfs: Number of TF classes
            use_pwm_similarity: If True, predict by similarity to PWM embeddings
            pwm_embeddings: Pre-computed PWM embeddings [N_TFs, D_pwm]
        """
        super().__init__()
        self.n_tfs = n_tfs
        self.use_pwm_similarity = use_pwm_similarity

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        if use_pwm_similarity and pwm_embeddings is not None:
            # Project slot to PWM space and compare
            pwm_dim = pwm_embeddings.shape[1]
            self.slot_to_pwm = nn.Linear(hidden_dim, pwm_dim)
            self.register_buffer("pwm_embeddings", pwm_embeddings)
            self.temperature = nn.Parameter(torch.tensor(1.0))
        else:
            # Direct classification
            self.classifier = nn.Linear(hidden_dim, n_tfs)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: Slot representations [B, K, D]

        Returns:
            TF identity logits [B, K, N_TFs]
        """
        h = self.mlp(slots)  # [B, K, H]

        if self.use_pwm_similarity:
            # Project to PWM space
            slot_pwm = self.slot_to_pwm(h)  # [B, K, D_pwm]

            # Normalize
            slot_pwm = F.normalize(slot_pwm, dim=-1)
            pwm_norm = F.normalize(self.pwm_embeddings, dim=-1)

            # Cosine similarity
            similarity = torch.einsum("bkd,nd->bkn", slot_pwm, pwm_norm)
            logits = similarity / self.temperature

        else:
            logits = self.classifier(h)

        return logits

    def predict_tf(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict most likely TF for each slot.

        Returns:
            tf_ids: Predicted TF indices [B, K]
            confidences: Prediction confidences [B, K]
        """
        logits = self.forward(slots)
        probs = F.softmax(logits, dim=-1)
        confidences, tf_ids = probs.max(dim=-1)
        return tf_ids, confidences


class DeepTFIdentityHead(nn.Module):
    """
    Deeper TF classifier with dedicated capacity for multi-TF discrimination.

    Phase 9.1: Expands from 2-layer MLP to 4-layer with dropout,
    giving the TF classifier more representational power.
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 256,
        n_tfs: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_tfs = n_tfs

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_tfs),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.mlp(slots)

    def predict_tf(self, slots: torch.Tensor):
        logits = self.forward(slots)
        probs = F.softmax(logits, dim=-1)
        confidences, tf_ids = probs.max(dim=-1)
        return tf_ids, confidences


class SlotDropout(nn.Module):
    """
    Phase 8.1: During training, randomly zero out the highest-occupancy slot,
    forcing the model to distribute information across multiple slots.
    """

    def __init__(self, drop_prob: float = 0.3):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(
        self,
        slot_embeddings: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slot_embeddings: [B, K, D]
            occupancy: [B, K, 1]
        Returns:
            Masked slot_embeddings, masked occupancy
        """
        if not self.training:
            return slot_embeddings, occupancy

        batch_size = slot_embeddings.shape[0]
        occ_flat = occupancy.squeeze(-1)  # [B, K]

        # For each sample, find the highest-occupancy slot and drop it with probability
        mask = torch.ones_like(occ_flat)
        top_slots = occ_flat.argmax(dim=-1)  # [B]

        for b in range(batch_size):
            if torch.rand(1).item() < self.drop_prob:
                mask[b, top_slots[b]] = 0.0

        mask = mask.unsqueeze(-1)  # [B, K, 1]
        return slot_embeddings * mask, occupancy * mask


class OccupancyHead(nn.Module):
    """
    Predicts binding occupancy/strength from slot representation.

    Outputs a value in [0, 1] representing:
    - 0: Slot is empty (no binding event)
    - 1: Strong binding event

    This enables soft slot filtering - slots with low occupancy
    can be ignored during inference.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        hidden_dim: int = 256,
        output_activation: str = "sigmoid",  # "sigmoid", "softplus"
    ):
        super().__init__()
        self.output_activation = output_activation

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: Slot representations [B, K, D]

        Returns:
            Occupancy values [B, K, 1]
        """
        out = self.mlp(slots)

        if self.output_activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.output_activation == "softplus":
            return F.softplus(out)
        else:
            return out

    def get_active_slots(
        self,
        slots: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mask of active slots (occupancy above threshold).

        Returns:
            occupancy: Occupancy values [B, K, 1]
            active_mask: Boolean mask [B, K]
        """
        occupancy = self.forward(slots)
        active_mask = occupancy.squeeze(-1) > threshold
        return occupancy, active_mask


class ProfileHead(nn.Module):
    """
    Reconstructs binding profile from slots.

    For compatibility with BPNet-style training, this head takes
    slot representations and reconstructs the full binding signal
    profile across the sequence.

    Each slot contributes a Gaussian-shaped peak at its predicted
    position, scaled by its occupancy.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        hidden_dim: int = 256,
        seq_len: int = 1000,
        peak_width: float = 50.0,
        learnable_width: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.peak_width = peak_width

        # Position and width prediction
        self.position_net = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        if learnable_width:
            self.width_net = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            )
        else:
            self.width_net = None

        # Amplitude/occupancy prediction
        self.amplitude_net = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Position grid
        self.register_buffer(
            "position_grid",
            torch.linspace(0, 1, seq_len).view(1, 1, seq_len)
        )

    def forward(
        self,
        slots: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Reconstruct binding profile from slots.

        Args:
            slots: Slot representations [B, K, D]
            return_components: If True, also return per-slot profiles

        Returns:
            profile: Reconstructed binding profile [B, L]
            (optional) slot_profiles: Per-slot profiles [B, K, L]
        """
        batch_size = slots.shape[0]

        # Predict parameters for each slot
        positions = self.position_net(slots)  # [B, K, 1]
        amplitudes = self.amplitude_net(slots)  # [B, K, 1]

        if self.width_net is not None:
            widths = self.width_net(slots) + 10.0  # [B, K, 1], min width
        else:
            widths = torch.full_like(amplitudes, self.peak_width)

        # Normalize width to sequence scale
        widths = widths / self.seq_len

        # Compute Gaussian profiles for each slot
        # positions: [B, K, 1], grid: [1, 1, L]
        distances = (self.position_grid - positions) ** 2  # [B, K, L]
        gaussian = torch.exp(-distances / (2 * widths ** 2))  # [B, K, L]

        # Scale by amplitude
        slot_profiles = gaussian * amplitudes  # [B, K, L]

        # Sum over slots
        profile = slot_profiles.sum(dim=1)  # [B, L]

        if return_components:
            return profile, slot_profiles
        return profile


class DecoderHead(nn.Module):
    """
    Full decoder head that combines all outputs.

    Aggregates position, TF identity, occupancy, and profile predictions
    into a unified output structure.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        hidden_dim: int = 256,
        n_tfs: int = 100,
        seq_len: int = 1000,
        position_mode: str = "gaussian",
    ):
        super().__init__()

        self.position_head = PositionHead(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            output_mode=position_mode,
            seq_len=seq_len,
        )

        self.tf_head = TFIdentityHead(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            n_tfs=n_tfs,
        )

        self.occupancy_head = OccupancyHead(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
        )

        self.profile_head = ProfileHead(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
        )

    def forward(
        self,
        slots: torch.Tensor,
    ) -> dict:
        """
        Decode all outputs from slots.

        Args:
            slots: Slot representations [B, K, D]

        Returns:
            Dictionary with:
            - position: Position predictions [B, K, 2] (mean, std) or [B, K, 1]
            - tf_logits: TF identity logits [B, K, N_TFs]
            - occupancy: Occupancy values [B, K, 1]
            - profile: Reconstructed profile [B, L]
        """
        return {
            "position": self.position_head(slots),
            "tf_logits": self.tf_head(slots),
            "occupancy": self.occupancy_head(slots),
            "profile": self.profile_head(slots),
        }


class AttributionHead(nn.Module):
    """
    Per-Base Attribution Head (replaces DeepSHAP).

    Learns to predict which bases contribute to each slot's binding event
    via cross-attention from slots to sequence features, supervised during
    training by gradient-derived importance from profile reconstruction loss.

    At inference a single forward pass produces per-base attribution for each
    slot -- no reference sequences required.

    Output: [B, K, L] per-slot per-position importance scores.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        feature_dim: int = 256,
        n_heads: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.head_dim = slot_dim // n_heads

        # Cross-attention: slots (queries) attend to sequence features (keys/values)
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, slot_dim, bias=False)
        # NOTE: v_proj and out_proj are defined for checkpoint compatibility but
        # not used in forward(). Only attention weights (from Q/K) are used, not V.
        self.v_proj = nn.Linear(feature_dim, slot_dim)
        self.out_proj = nn.Linear(slot_dim, slot_dim)

        # Per-position importance MLP (applied to sequence features)
        self.importance_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.layer_norm_slots = nn.LayerNorm(slot_dim)
        self.layer_norm_features = nn.LayerNorm(feature_dim)

    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slots: Slot embeddings [B, K, D]
            features: Backbone sequence features [B, L, D_feat]

        Returns:
            importance: Per-slot per-position importance [B, K, L]
        """
        B, K, D = slots.shape
        L = features.shape[1]

        slots_norm = self.layer_norm_slots(slots)
        features_norm = self.layer_norm_features(features)

        # Multi-head cross-attention
        Q = (
            self.q_proj(slots_norm)
            .view(B, K, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, K, D/H]
        Kp = (
            self.k_proj(features_norm)
            .view(B, L, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, L, D/H]

        # Attention weights: [B, H, K, L]
        attn = torch.matmul(Q, Kp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn, dim=-1)

        # Average over heads → [B, K, L]
        slot_position_attn = attn_weights.mean(dim=1)

        # Per-position importance from features → [B, L]
        pos_importance = torch.sigmoid(
            self.importance_head(features_norm).squeeze(-1)
        )

        # Final importance: slot-position attention * position importance
        importance = slot_position_attn * pos_importance.unsqueeze(1)  # [B, K, L]
        return importance


class MotifEmbeddingHead(nn.Module):
    """
    Open-Vocabulary Motif Discovery Head (replaces fixed TF classifier).

    Projects slot embeddings into a continuous motif embedding space anchored by
    known TFs but open to novel motif patterns.  A codebook of learnable
    prototypes covers the embedding volume; known TF anchors provide grounding.

    Outputs both prototype soft-assignments (for novel motif discovery) and
    standard TF classification logits (for backward compatibility with existing
    training / evaluation code).
    """

    def __init__(
        self,
        slot_dim: int = 256,
        motif_dim: int = 64,
        n_prototypes: int = 64,
        n_known_tfs: int = 7,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.motif_dim = motif_dim
        self.n_prototypes = n_prototypes
        self.n_known_tfs = n_known_tfs

        # Learnable temperature for distance scaling
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature))
        )

        # Project slot → motif embedding space
        self.slot_to_motif = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, motif_dim),
        )

        # Learnable prototype codebook (initialized uniformly in unit sphere)
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(n_prototypes, motif_dim), dim=-1) * 0.5
        )

        # Known TF anchor points (initialized with slight separation)
        self.anchors = nn.Parameter(
            F.normalize(torch.randn(n_known_tfs, motif_dim), dim=-1) * 0.5
        )

        # Linear head for backward-compatible TF logits
        self.known_tf_classifier = nn.Linear(motif_dim, n_known_tfs)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            slots: Slot embeddings [B, K, D]

        Returns:
            Dictionary with:
            - motif_embeddings:       [B, K, motif_dim]
            - prototype_assignments:  [B, K, n_prototypes]  (soft)
            - anchor_distances:       [B, K, n_known_tfs]
            - tf_logits:              [B, K, n_known_tfs]  (backward compat)
        """
        motif_emb = self.slot_to_motif(slots)  # [B, K, motif_dim]
        motif_emb_norm = F.normalize(motif_emb, dim=-1)

        # Soft prototype assignments via cosine similarity
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [P, D_m]
        proto_sim = torch.matmul(motif_emb_norm, proto_norm.t())  # [B, K, P]
        proto_assignments = F.softmax(proto_sim / self.temperature, dim=-1)

        # L2 distance to known TF anchors
        anchor_norm = F.normalize(self.anchors, dim=-1)  # [T, D_m]
        # [B, K, D_m] vs [T, D_m] → [B, K, T]
        anchor_distances = torch.cdist(motif_emb_norm, anchor_norm.unsqueeze(0).expand(slots.shape[0], -1, -1))

        # Standard TF logits for backward compatibility
        tf_logits = self.known_tf_classifier(motif_emb)

        return {
            "motif_embeddings": motif_emb,
            "prototype_assignments": proto_assignments,
            "anchor_distances": anchor_distances,
            "tf_logits": tf_logits,
        }

    def get_novel_motifs(
        self,
        motif_embeddings: torch.Tensor,
        distance_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Identify embeddings far from all known TF anchors (potentially novel).

        Args:
            motif_embeddings: [B, K, motif_dim] or [N, motif_dim]
            distance_threshold: Minimum distance to all anchors for novelty

        Returns:
            Boolean mask of the same leading shape (True = novel)
        """
        anchor_norm = F.normalize(self.anchors, dim=-1)
        emb_norm = F.normalize(motif_embeddings, dim=-1)

        if emb_norm.dim() == 3:
            B, K, D = emb_norm.shape
            flat = emb_norm.reshape(B * K, D)
            dists = torch.cdist(flat.unsqueeze(0), anchor_norm.unsqueeze(0)).squeeze(0)  # [B*K, T]
            min_dist = dists.min(dim=-1).values.view(B, K)
        else:
            dists = torch.cdist(emb_norm.unsqueeze(0), anchor_norm.unsqueeze(0)).squeeze(0)
            min_dist = dists.min(dim=-1).values

        return min_dist > distance_threshold
