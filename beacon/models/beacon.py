"""
BEACON: Binding Event Attention-based Compositional Object Network

Full model integrating:
1. Convolutional backbone encoder
2. Helical phase-aware positional encoding
3. Slot attention for binding site discovery
4. Output heads for position, TF identity, and occupancy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

from .backbone import ConvBackbone, DilatedConvBackbone
from .positional import HelicalPositionalEncoding
from .slot_attention import SlotAttention, PWMGroundedSlotAttention, PositionAwareSlotAttention
from .heads import PositionHead, TFIdentityHead, OccupancyHead, ProfileHead


class BEACON(nn.Module):
    """
    BEACON: Binding Event Attention-based Compositional Object Network

    A novel framework for inherently interpretable transcription factor
    binding site discovery using slot attention.

    Key features:
    - Explicit binding site discovery (not post-hoc)
    - Direct enumeration of binding events
    - Compositional representation of regulatory grammar
    - Interpretable variant effect prediction
    """

    def __init__(
        self,
        # Sequence parameters
        seq_len: int = 1000,
        input_channels: int = 4,
        # Backbone parameters
        backbone_type: str = "dilated",  # "conv", "dilated", "multiscale"
        backbone_dim: int = 256,
        backbone_layers: int = 8,
        # Positional encoding
        use_helical_encoding: bool = True,
        # Slot attention parameters
        n_slots: int = 32,
        slot_dim: int = 256,
        n_iterations: int = 3,
        slot_type: str = "standard",  # "standard", "pwm_grounded", "position_aware"
        attention_mode: str = "competitive",  # "competitive" or "independent"
        # Output parameters
        n_tfs: int = 100,
        position_mode: str = "gaussian",
        # PWM initialization (optional)
        pwm_embeddings: Optional[torch.Tensor] = None,
        # Regularization
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_tfs = n_tfs

        # Build backbone encoder
        self.backbone = self._build_backbone(
            backbone_type=backbone_type,
            input_channels=input_channels,
            hidden_dim=backbone_dim,
            output_dim=slot_dim,
            n_layers=backbone_layers,
            dropout=dropout,
        )

        # Positional encoding
        if use_helical_encoding:
            self.positional_encoding = HelicalPositionalEncoding(
                d_model=slot_dim,
                max_len=seq_len,
                dropout=dropout,
            )
        else:
            self.positional_encoding = nn.Identity()

        # Slot attention module
        self.slot_attention = self._build_slot_attention(
            slot_type=slot_type,
            input_dim=slot_dim,
            slot_dim=slot_dim,
            n_slots=n_slots,
            n_iterations=n_iterations,
            pwm_embeddings=pwm_embeddings,
            attention_mode=attention_mode,
        )

        # Output heads
        self.position_head = PositionHead(
            slot_dim=slot_dim,
            hidden_dim=slot_dim,
            output_mode=position_mode,
            seq_len=seq_len,
        )

        self.tf_head = TFIdentityHead(
            slot_dim=slot_dim,
            hidden_dim=slot_dim,
            n_tfs=n_tfs,
            use_pwm_similarity=pwm_embeddings is not None,
            pwm_embeddings=pwm_embeddings,
        )

        self.occupancy_head = OccupancyHead(
            slot_dim=slot_dim,
            hidden_dim=slot_dim,
        )

        self.profile_head = ProfileHead(
            slot_dim=slot_dim,
            hidden_dim=slot_dim,
            seq_len=seq_len,
        )

        # Store config
        self.config = {
            "seq_len": seq_len,
            "n_slots": n_slots,
            "slot_dim": slot_dim,
            "n_tfs": n_tfs,
            "backbone_type": backbone_type,
            "slot_type": slot_type,
        }

    def _build_backbone(
        self,
        backbone_type: str,
        input_channels: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Build the backbone encoder."""
        if backbone_type == "conv":
            return ConvBackbone(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                dropout=dropout,
            )
        elif backbone_type == "dilated":
            return DilatedConvBackbone(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def _build_slot_attention(
        self,
        slot_type: str,
        input_dim: int,
        slot_dim: int,
        n_slots: int,
        n_iterations: int,
        pwm_embeddings: Optional[torch.Tensor],
        attention_mode: str = "competitive",
    ) -> nn.Module:
        """Build the slot attention module."""
        if slot_type == "standard":
            return SlotAttention(
                input_dim=input_dim,
                slot_dim=slot_dim,
                n_slots=n_slots,
                n_iterations=n_iterations,
                attention_mode=attention_mode,
            )
        elif slot_type == "pwm_grounded":
            return PWMGroundedSlotAttention(
                input_dim=input_dim,
                slot_dim=slot_dim,
                n_slots=n_slots,
                n_iterations=n_iterations,
                pwm_embeddings=pwm_embeddings,
            )
        elif slot_type == "position_aware":
            return PositionAwareSlotAttention(
                input_dim=input_dim,
                slot_dim=slot_dim,
                n_slots=n_slots,
                n_iterations=n_iterations,
            )
        else:
            raise ValueError(f"Unknown slot type: {slot_type}")

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode one-hot sequence to continuous representation.

        Args:
            x: One-hot encoded sequence [B, L, 4]

        Returns:
            Sequence features [B, L, D]
        """
        # Backbone encoding
        features = self.backbone(x)

        # Add positional encoding
        features = self.positional_encoding(features)

        return features

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True,
        return_slots: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BEACON.

        Args:
            x: One-hot encoded sequence [B, L, 4]
            return_attention: Whether to return slot attention weights
            return_slots: Whether to return raw slot representations

        Returns:
            Dictionary containing:
            - profile: Reconstructed binding profile [B, L]
            - positions: Predicted binding positions [B, K, 2]
            - tf_logits: TF identity logits [B, K, N_TFs]
            - occupancy: Slot occupancy values [B, K, 1]
            - attention: Attention weights [B, K, L] (for diversity loss)
            - slot_embeddings: Slot representations [B, K, D] (for orthogonality loss)
        """
        # Encode sequence
        features = self.encode_sequence(x)  # [B, L, D]

        # Slot attention - always get attention for diversity loss
        if isinstance(self.slot_attention, PositionAwareSlotAttention):
            slots, attention, _ = self.slot_attention(
                features, return_attention=True
            )
        else:
            slots, attention = self.slot_attention(
                features, return_attention=True
            )

        # Decode outputs from slots
        outputs = {
            "profile": self.profile_head(slots),
            "positions": self.position_head(slots),
            "tf_logits": self.tf_head(slots),
            "occupancy": self.occupancy_head(slots),
        }

        # Always include attention for diversity loss during training
        if attention is not None:
            outputs["attention"] = attention

        # Always include slot embeddings for orthogonality loss during training
        outputs["slot_embeddings"] = slots

        return outputs

    def predict_binding_sites(
        self,
        x: torch.Tensor,
        occupancy_threshold: float = 0.5,
        return_all: bool = False,
    ) -> List[List[Dict]]:
        """
        Predict discrete binding sites from sequence.

        This is the key capability of BEACON: directly enumerating
        binding events without post-hoc interpretation.

        Args:
            x: One-hot encoded sequence [B, L, 4]
            occupancy_threshold: Minimum occupancy for a binding site
            return_all: If True, return all slots regardless of occupancy

        Returns:
            List of binding sites per sequence, each site is a dict with:
            - position: Predicted position (bp)
            - position_std: Position uncertainty
            - tf_id: Predicted TF index
            - tf_confidence: TF prediction confidence
            - occupancy: Binding strength
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)

            batch_size = x.shape[0]
            all_sites = []

            for b in range(batch_size):
                sites = []

                occupancy = outputs["occupancy"][b].squeeze(-1)  # [K]
                positions = outputs["positions"][b]  # [K, 2] for gaussian mode

                tf_probs = F.softmax(outputs["tf_logits"][b], dim=-1)  # [K, N_TFs]
                tf_confidence, tf_ids = tf_probs.max(dim=-1)  # [K]

                for k in range(self.n_slots):
                    occ = occupancy[k].item()

                    if occ < occupancy_threshold and not return_all:
                        continue

                    # Get position (handle different modes)
                    if positions.shape[-1] == 2:  # Gaussian mode
                        pos_mean = positions[k, 0].item() * self.seq_len
                        pos_std = positions[k, 1].item() * self.seq_len
                    else:
                        pos_mean = positions[k, 0].item() * self.seq_len
                        pos_std = 0.0

                    sites.append({
                        "slot_id": k,
                        "position": pos_mean,
                        "position_std": pos_std,
                        "tf_id": tf_ids[k].item(),
                        "tf_confidence": tf_confidence[k].item(),
                        "occupancy": occ,
                    })

                # Sort by position
                sites.sort(key=lambda s: s["position"])
                all_sites.append(sites)

            return all_sites

    def count_binding_sites(
        self,
        x: torch.Tensor,
        occupancy_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Count number of binding sites per sequence.

        Direct binding site enumeration - a key capability of BEACON.

        Args:
            x: One-hot encoded sequence [B, L, 4]
            occupancy_threshold: Minimum occupancy threshold

        Returns:
            Binding site counts [B]
        """
        with torch.no_grad():
            outputs = self.forward(x)
            occupancy = outputs["occupancy"].squeeze(-1)  # [B, K]
            counts = (occupancy > occupancy_threshold).sum(dim=-1)  # [B]
            return counts

    def get_binding_composition(
        self,
        x: torch.Tensor,
        occupancy_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Get composition of TFs bound in each sequence.

        Returns a histogram of TF types across binding sites.

        Args:
            x: One-hot encoded sequence [B, L, 4]
            occupancy_threshold: Minimum occupancy threshold

        Returns:
            TF composition [B, N_TFs]
        """
        with torch.no_grad():
            outputs = self.forward(x)

            occupancy = outputs["occupancy"].squeeze(-1)  # [B, K]
            tf_logits = outputs["tf_logits"]  # [B, K, N_TFs]
            tf_probs = F.softmax(tf_logits, dim=-1)

            # Mask by occupancy
            mask = (occupancy > occupancy_threshold).unsqueeze(-1)  # [B, K, 1]
            masked_probs = tf_probs * mask

            # Sum across slots
            composition = masked_probs.sum(dim=1)  # [B, N_TFs]

            return composition

    def analyze_variant_effect(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor,
        occupancy_threshold: float = 0.3,
    ) -> Dict:
        """
        Analyze the effect of a variant on binding sites.

        Key interpretability feature: know which specific binding
        site a variant disrupts, not just effect magnitude.

        Args:
            ref_seq: Reference sequence [1, L, 4]
            alt_seq: Alternate sequence [1, L, 4]
            occupancy_threshold: Threshold for considering a site

        Returns:
            Dictionary with variant effect analysis
        """
        self.eval()
        with torch.no_grad():
            ref_outputs = self.forward(ref_seq, return_slots=True)
            alt_outputs = self.forward(alt_seq, return_slots=True)

            # Get binding sites
            ref_sites = self.predict_binding_sites(
                ref_seq, occupancy_threshold, return_all=True
            )[0]
            alt_sites = self.predict_binding_sites(
                alt_seq, occupancy_threshold, return_all=True
            )[0]

            # Compare profiles
            profile_diff = (
                alt_outputs["profile"] - ref_outputs["profile"]
            ).squeeze(0)

            # Identify affected slots
            ref_occupancy = ref_outputs["occupancy"].squeeze()
            alt_occupancy = alt_outputs["occupancy"].squeeze()
            occupancy_change = alt_occupancy - ref_occupancy

            # Find disrupted sites (occupancy decreased significantly)
            disrupted = []
            for k in range(self.n_slots):
                if ref_occupancy[k] > occupancy_threshold:
                    change = occupancy_change[k].item()
                    if change < -0.2:  # Significant decrease
                        disrupted.append({
                            "slot_id": k,
                            "ref_occupancy": ref_occupancy[k].item(),
                            "alt_occupancy": alt_occupancy[k].item(),
                            "occupancy_change": change,
                            "position": ref_sites[k]["position"],
                            "tf_id": ref_sites[k]["tf_id"],
                        })

            # Find gained sites
            gained = []
            for k in range(self.n_slots):
                if alt_occupancy[k] > occupancy_threshold:
                    change = occupancy_change[k].item()
                    if change > 0.2:  # Significant increase
                        gained.append({
                            "slot_id": k,
                            "ref_occupancy": ref_occupancy[k].item(),
                            "alt_occupancy": alt_occupancy[k].item(),
                            "occupancy_change": change,
                            "position": alt_sites[k]["position"],
                            "tf_id": alt_sites[k]["tf_id"],
                        })

            return {
                "profile_diff": profile_diff,
                "disrupted_sites": disrupted,
                "gained_sites": gained,
                "total_effect": profile_diff.abs().sum().item(),
            }

    @classmethod
    def from_pretrained(cls, path: str) -> "BEACON":
        """Load a pretrained BEACON model."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save_pretrained(self, path: str):
        """Save model with config."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)


class BEACONSmall(BEACON):
    """Small BEACON model for development and testing."""

    def __init__(self, **kwargs):
        defaults = {
            "backbone_dim": 128,
            "backbone_layers": 4,
            "slot_dim": 128,
            "n_slots": 16,
            "n_iterations": 2,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class BEACONBase(BEACON):
    """Base BEACON model - balanced size and performance."""

    def __init__(self, **kwargs):
        defaults = {
            "backbone_dim": 256,
            "backbone_layers": 8,
            "slot_dim": 256,
            "n_slots": 32,
            "n_iterations": 3,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class BEACONLarge(BEACON):
    """Large BEACON model for maximum performance."""

    def __init__(self, **kwargs):
        defaults = {
            "backbone_dim": 512,
            "backbone_layers": 12,
            "slot_dim": 512,
            "n_slots": 64,
            "n_iterations": 5,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
