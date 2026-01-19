"""
Slot Attention module for BEACON.

Core innovation: treats transcription factor binding events as discrete "objects"
that slots compete to represent, enabling explicit enumeration and compositional
reasoning about regulatory grammar.

Based on Locatello et al. (2020) "Object-Centric Learning with Slot Attention"
but adapted for 1D genomic sequences with domain-specific modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SlotAttention(nn.Module):
    """
    Slot Attention mechanism for binding site discovery.

    K slots compete via attention to bind to different binding events in the
    input sequence. Each slot learns to represent a discrete binding event
    with its position, TF identity, and occupancy.

    Key modifications from original slot attention:
    1. PWM-grounded initialization (optional)
    2. Position-aware attention with helical phase
    3. Occupancy-based slot filtering
    """

    def __init__(
        self,
        input_dim: int = 256,
        slot_dim: int = 256,
        n_slots: int = 32,
        n_iterations: int = 3,
        hidden_dim: int = 512,
        epsilon: float = 1e-8,
        learnable_slots: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input sequence features
            slot_dim: Dimension of slot representations
            n_slots: Number of slots (max binding events per sequence)
            n_iterations: Number of iterative refinement steps
            hidden_dim: Hidden dimension for slot update MLP
            epsilon: Small constant for numerical stability
            learnable_slots: Whether to use learnable slot initialization
        """
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.n_iterations = n_iterations
        self.epsilon = epsilon

        # Slot initialization
        if learnable_slots:
            # Learnable slot means and log-variances
            self.slot_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim) * 0.1)
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, slot_dim))
        else:
            self.register_buffer("slot_mu", torch.randn(1, n_slots, slot_dim) * 0.1)
            self.register_buffer("slot_log_sigma", torch.zeros(1, n_slots, slot_dim))

        # Layer norms
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Attention projections
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)

        # Slot update GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Slot update MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        # Scaling factor for attention
        self.scale = slot_dim ** -0.5

    def _initialize_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize slots with learnable Gaussian parameters.

        Returns:
            Initial slot representations [B, K, D]
        """
        # Sample from learned Gaussian
        mu = self.slot_mu.expand(batch_size, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, -1, -1)

        if self.training:
            # Add noise during training for diversity
            slots = mu + sigma * torch.randn_like(mu)
        else:
            # Use mean during inference for determinism
            slots = mu

        return slots

    def forward(
        self,
        inputs: torch.Tensor,
        slot_init: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform slot attention.

        Args:
            inputs: Sequence features [B, L, D_in]
            slot_init: Optional custom slot initialization [B, K, D_slot]
            return_attention: Whether to return attention weights

        Returns:
            slots: Final slot representations [B, K, D_slot]
            attention: (optional) Attention weights [B, K, L]
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Normalize inputs
        inputs = self.norm_input(inputs)

        # Project inputs to keys and values
        k = self.project_k(inputs)  # [B, L, D_slot]
        v = self.project_v(inputs)  # [B, L, D_slot]

        # Initialize slots
        if slot_init is not None:
            slots = slot_init
        else:
            slots = self._initialize_slots(batch_size, device)

        # Store attention for visualization
        all_attention = []

        # Iterative refinement
        for iteration in range(self.n_iterations):
            slots_prev = slots

            # Normalize slots
            slots = self.norm_slots(slots)

            # Compute attention
            q = self.project_q(slots)  # [B, K, D_slot]

            # Attention scores: [B, K, L]
            attn_logits = torch.einsum("bkd,bld->bkl", q, k) * self.scale

            # Softmax over slots (competition)
            # Each position attends to slots, slots compete for positions
            attn = F.softmax(attn_logits, dim=1)  # [B, K, L]

            if return_attention:
                all_attention.append(attn)

            # Weighted mean of values
            # Normalize attention over positions
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum("bkl,bld->bkd", attn_norm, v)  # [B, K, D_slot]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(batch_size, self.n_slots, self.slot_dim)

            # MLP residual update
            slots = slots + self.mlp(self.norm_mlp(slots))

        if return_attention:
            # Return final attention
            attention = all_attention[-1]
            return slots, attention
        else:
            return slots, None


class PWMGroundedSlotAttention(SlotAttention):
    """
    Slot Attention with PWM-grounded initialization.

    Instead of random initialization, slots are initialized from learned
    embeddings of known TF PWMs, organized by TF family. This provides
    biological priors that help slots specialize to different TF types.
    """

    def __init__(
        self,
        input_dim: int = 256,
        slot_dim: int = 256,
        n_slots: int = 32,
        n_iterations: int = 3,
        hidden_dim: int = 512,
        epsilon: float = 1e-8,
        pwm_embeddings: Optional[torch.Tensor] = None,
        n_pwm_clusters: int = 16,
    ):
        """
        Args:
            pwm_embeddings: Pre-computed PWM embeddings [N_motifs, D_pwm]
            n_pwm_clusters: Number of PWM clusters for slot initialization
        """
        # Initialize parent without learnable slots
        super().__init__(
            input_dim=input_dim,
            slot_dim=slot_dim,
            n_slots=n_slots,
            n_iterations=n_iterations,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            learnable_slots=False,
        )

        self.n_pwm_clusters = n_pwm_clusters

        # PWM embedding projection
        if pwm_embeddings is not None:
            pwm_dim = pwm_embeddings.shape[1]
            self.pwm_projection = nn.Linear(pwm_dim, slot_dim)
            self.register_buffer("pwm_embeddings", pwm_embeddings)
        else:
            self.pwm_projection = None
            self.pwm_embeddings = None

        # Learnable cluster centers for slots
        self.cluster_centers = nn.Parameter(torch.randn(n_pwm_clusters, slot_dim) * 0.1)

        # Slot-to-cluster assignment (learnable soft assignment)
        self.slot_cluster_logits = nn.Parameter(torch.randn(n_slots, n_pwm_clusters))

        # Slot-specific refinement
        self.slot_refinement = nn.Parameter(torch.zeros(n_slots, slot_dim))

    def initialize_from_pwms(self, pwm_embeddings: torch.Tensor):
        """
        Initialize cluster centers from PWM embeddings using k-means.

        Args:
            pwm_embeddings: PWM embeddings [N_motifs, D_pwm]
        """
        from sklearn.cluster import KMeans

        # Project PWMs to slot dimension
        with torch.no_grad():
            projected = self.pwm_projection(pwm_embeddings)

            # Cluster PWMs
            kmeans = KMeans(n_clusters=self.n_pwm_clusters, random_state=42)
            kmeans.fit(projected.cpu().numpy())

            # Set cluster centers
            self.cluster_centers.data = torch.from_numpy(kmeans.cluster_centers_).to(
                self.cluster_centers.device
            )

    def _initialize_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize slots from PWM-grounded cluster centers.
        """
        # Soft assignment of slots to clusters
        cluster_weights = F.softmax(self.slot_cluster_logits, dim=-1)  # [K, C]

        # Weighted combination of cluster centers
        slot_base = torch.einsum("kc,cd->kd", cluster_weights, self.cluster_centers)

        # Add slot-specific refinement
        slots = slot_base + self.slot_refinement

        # Expand for batch
        slots = slots.unsqueeze(0).expand(batch_size, -1, -1)

        if self.training:
            # Add small noise for diversity
            slots = slots + torch.randn_like(slots) * 0.01

        return slots


class PositionAwareSlotAttention(SlotAttention):
    """
    Slot Attention with explicit position modeling.

    Each slot additionally learns to predict a position, and attention
    is modulated by distance from that position. This helps slots
    focus on localized binding events.
    """

    def __init__(
        self,
        input_dim: int = 256,
        slot_dim: int = 256,
        n_slots: int = 32,
        n_iterations: int = 3,
        hidden_dim: int = 512,
        epsilon: float = 1e-8,
        position_sigma: float = 50.0,
    ):
        super().__init__(
            input_dim=input_dim,
            slot_dim=slot_dim,
            n_slots=n_slots,
            n_iterations=n_iterations,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
        )

        self.position_sigma = position_sigma

        # Position prediction head (predicts normalized position 0-1)
        self.position_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Learned position-attention interaction
        self.position_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        inputs: torch.Tensor,
        slot_init: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with position-aware attention.

        Returns:
            slots: Final slot representations [B, K, D_slot]
            attention: (optional) Attention weights [B, K, L]
            positions: Predicted slot positions [B, K, 1]
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Normalize inputs
        inputs_norm = self.norm_input(inputs)

        # Project inputs
        k = self.project_k(inputs_norm)
        v = self.project_v(inputs_norm)

        # Initialize slots
        if slot_init is not None:
            slots = slot_init
        else:
            slots = self._initialize_slots(batch_size, device)

        # Position grid for distance computation
        positions_grid = torch.linspace(0, 1, seq_len, device=device)

        all_attention = []

        for iteration in range(self.n_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Predict slot positions
            slot_positions = self.position_head(slots)  # [B, K, 1]

            # Compute position-based attention bias
            # Distance from each slot's predicted position to all sequence positions
            distances = (positions_grid.view(1, 1, -1) - slot_positions) ** 2
            position_bias = -distances / (2 * (self.position_sigma / seq_len) ** 2)

            # Compute content attention
            q = self.project_q(slots)
            content_attn = torch.einsum("bkd,bld->bkl", q, k) * self.scale

            # Combine content and position attention
            gate = torch.sigmoid(self.position_gate)
            attn_logits = gate * content_attn + (1 - gate) * position_bias

            # Softmax over slots
            attn = F.softmax(attn_logits, dim=1)

            if return_attention:
                all_attention.append(attn)

            # Update slots
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum("bkl,bld->bkd", attn_norm, v)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(batch_size, self.n_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        # Final position prediction
        final_positions = self.position_head(self.norm_slots(slots))

        if return_attention:
            return slots, all_attention[-1], final_positions
        else:
            return slots, None, final_positions
