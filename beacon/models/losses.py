"""
Training objectives for BEACON.

Implements the multi-task loss combining:
1. Binding Profile Reconstruction (BPNet compatibility)
2. Slot-Supervised Losses (when ChIP-seq peaks available)
3. Unsupervised Binding Discovery (ATAC-seq only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class ProfileReconstructionLoss(nn.Module):
    """
    Binding profile reconstruction loss.

    Compatible with BPNet-style training using multinomial NLL
    for profile shape and MSE for total counts.
    """

    def __init__(
        self,
        profile_weight: float = 1.0,
        count_weight: float = 1.0,
        smoothing: float = 1e-6,
    ):
        super().__init__()
        self.profile_weight = profile_weight
        self.count_weight = count_weight
        self.smoothing = smoothing

    def forward(
        self,
        pred_profile: torch.Tensor,
        target_profile: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute profile reconstruction loss.

        Args:
            pred_profile: Predicted profile [B, L]
            target_profile: Target profile [B, L]

        Returns:
            Dictionary with loss components
        """
        # Normalize profiles to probability distributions
        pred_prob = pred_profile / (pred_profile.sum(dim=-1, keepdim=True) + self.smoothing)
        target_prob = target_profile / (target_profile.sum(dim=-1, keepdim=True) + self.smoothing)

        # Profile shape loss (KL divergence)
        # Add smoothing to avoid log(0)
        pred_prob = pred_prob + self.smoothing
        target_prob = target_prob + self.smoothing

        kl_loss = F.kl_div(
            pred_prob.log(),
            target_prob,
            reduction='batchmean',
        )

        # Count loss (total signal)
        pred_counts = pred_profile.sum(dim=-1)
        target_counts = target_profile.sum(dim=-1)
        count_loss = F.mse_loss(pred_counts.log1p(), target_counts.log1p())

        # Combined loss
        total_loss = self.profile_weight * kl_loss + self.count_weight * count_loss

        return {
            "profile_loss": kl_loss,
            "count_loss": count_loss,
            "total": total_loss,
        }


class PositionLoss(nn.Module):
    """
    Loss for binding site position prediction.

    Supports different position output modes:
    - Gaussian: negative log-likelihood of true position under predicted Gaussian
    - Distribution: cross-entropy over position bins
    - Continuous: MSE loss
    """

    def __init__(
        self,
        mode: str = "gaussian",
        seq_len: int = 1000,
    ):
        super().__init__()
        self.mode = mode
        self.seq_len = seq_len

    def forward(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute position prediction loss.

        Args:
            pred_positions: Predicted positions [B, K, 2] for gaussian, [B, K, 1] otherwise
            target_positions: Target positions [B, K] (normalized 0-1)
            mask: Optional mask for valid slots [B, K]

        Returns:
            Position loss scalar
        """
        if self.mode == "gaussian":
            # pred_positions: [B, K, 2] = (mu, sigma)
            mu = pred_positions[..., 0]  # [B, K]
            # Clamp sigma to reasonable range to prevent instability
            sigma = pred_positions[..., 1].clamp(min=0.01, max=1.0) + 1e-6  # [B, K]

            # Negative log likelihood of Gaussian (clamped to prevent extreme values)
            nll = 0.5 * ((target_positions - mu) / sigma) ** 2 + torch.log(sigma)
            nll = nll.clamp(max=10.0)  # Prevent extreme loss values

            if mask is not None:
                nll = nll * mask
                loss = nll.sum() / (mask.sum() + 1e-6)
            else:
                loss = nll.mean()

        elif self.mode == "distribution":
            # pred_positions: [B, K, L] logits
            target_bins = (target_positions * (self.seq_len - 1)).long()
            loss = F.cross_entropy(
                pred_positions.reshape(-1, self.seq_len),
                target_bins.reshape(-1),
                reduction='none',
            )
            if mask is not None:
                loss = (loss.reshape(mask.shape) * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss = loss.mean()

        else:  # continuous
            pred = pred_positions.squeeze(-1)
            loss = F.mse_loss(pred, target_positions, reduction='none')
            if mask is not None:
                loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss = loss.mean()

        return loss


class TFIdentityLoss(nn.Module):
    """
    Loss for TF identity prediction.

    Cross-entropy loss over TF classes.
    """

    def __init__(
        self,
        n_tfs: int,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.n_tfs = n_tfs
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_tfs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute TF identity loss.

        Args:
            pred_logits: Predicted TF logits [B, K, N_TFs]
            target_tfs: Target TF indices [B, K]
            mask: Optional mask for valid slots [B, K]

        Returns:
            TF identity loss scalar
        """
        batch_size, n_slots = target_tfs.shape

        # Reshape for cross entropy
        pred_flat = pred_logits.reshape(-1, self.n_tfs)
        target_flat = target_tfs.reshape(-1)

        # Cross entropy with label smoothing
        loss = F.cross_entropy(
            pred_flat,
            target_flat,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )

        loss = loss.reshape(batch_size, n_slots)

        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss


class OccupancyLoss(nn.Module):
    """
    Loss for slot occupancy prediction.

    Binary cross-entropy between predicted and target occupancy.
    Encourages slots to be either fully occupied or empty.
    """

    def __init__(
        self,
        sparsity_weight: float = 0.1,
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        pred_occupancy: torch.Tensor,
        target_occupancy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute occupancy loss.

        Args:
            pred_occupancy: Predicted occupancy [B, K, 1]
            target_occupancy: Target occupancy [B, K] (binary or continuous)

        Returns:
            Dictionary with loss components
        """
        pred = pred_occupancy.squeeze(-1).float()
        target_occupancy = target_occupancy.float()

        # BCE loss - use mse instead for AMP compatibility
        bce_loss = F.mse_loss(pred, target_occupancy, reduction='mean')

        # Sparsity regularization - encourage binary occupancy
        # Entropy of Bernoulli: -p*log(p) - (1-p)*log(1-p)
        # Maximum at p=0.5, minimum at p=0 or p=1
        entropy = -(pred * torch.log(pred + 1e-6) + (1 - pred) * torch.log(1 - pred + 1e-6))
        sparsity_loss = entropy.mean()

        total = bce_loss + self.sparsity_weight * sparsity_loss

        return {
            "bce_loss": bce_loss,
            "sparsity_loss": sparsity_loss,
            "total": total,
        }


class SlotDiversityLoss(nn.Module):
    """
    Encourages slots to attend to different positions.

    Penalizes overlap in slot attention patterns to ensure
    each binding site is represented by a single slot.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Compute slot diversity loss.

        Args:
            attention: Slot attention weights [B, K, L]

        Returns:
            Diversity loss scalar
        """
        # Normalize attention per slot
        attn_norm = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute pairwise cosine similarity between slot attention patterns
        # [B, K, L] x [B, L, K] -> [B, K, K]
        similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))

        # Mask diagonal (self-similarity)
        batch_size, n_slots = similarity.shape[:2]
        mask = 1 - torch.eye(n_slots, device=similarity.device).unsqueeze(0)

        # Penalize high similarity between different slots
        similarity = similarity * mask
        loss = (similarity ** 2).sum() / (mask.sum() * batch_size)

        return loss


class SlotOrthogonalityLoss(nn.Module):
    """
    Encourages slot embeddings to be orthogonal.

    This ensures slots learn different representations and
    specialize on different binding patterns.
    """

    def forward(self, slot_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute slot orthogonality loss.

        Args:
            slot_embeddings: Slot embeddings [B, K, D]

        Returns:
            Orthogonality loss scalar
        """
        # Normalize embeddings
        slot_norm = F.normalize(slot_embeddings, dim=-1)  # [B, K, D]

        # Compute Gram matrix (pairwise dot products)
        gram = torch.bmm(slot_norm, slot_norm.transpose(1, 2))  # [B, K, K]

        # Target: identity matrix (slots orthogonal to each other)
        batch_size, n_slots = gram.shape[:2]
        target = torch.eye(n_slots, device=gram.device).unsqueeze(0)

        # Frobenius norm of difference from identity
        loss = ((gram - target) ** 2).sum() / (batch_size * n_slots * n_slots)

        return loss


class BindingSiteSupervisionLoss(nn.Module):
    """
    Supervises slot predictions with binding site pseudo-labels.

    Uses extracted peaks from profiles to supervise:
    - Position predictions
    - Occupancy predictions
    - TF identity (when available)
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        occupancy_weight: float = 1.0,
        tf_weight: float = 1.0,
        use_hungarian: bool = True,
    ):
        super().__init__()
        self.position_weight = position_weight
        self.occupancy_weight = occupancy_weight
        self.tf_weight = tf_weight
        self.use_hungarian = use_hungarian

    def forward(
        self,
        pred_positions: torch.Tensor,
        pred_occupancy: torch.Tensor,
        pred_tf_logits: torch.Tensor,
        target_sites: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute binding site supervision loss.

        Args:
            pred_positions: Predicted positions [B, K, 2] (mu, sigma) or [B, K, 1]
            pred_occupancy: Predicted occupancy [B, K, 1]
            pred_tf_logits: Predicted TF logits [B, K, N_TFs]
            target_sites: Target binding sites [B, K, 3] (position, tf_id, occupancy)

        Returns:
            Dictionary with loss components
        """
        batch_size, n_slots = pred_occupancy.shape[:2]
        device = pred_occupancy.device

        # Extract target components
        target_positions = target_sites[..., 0]  # [B, K] normalized 0-1
        target_tf_ids = target_sites[..., 1].long()  # [B, K]
        target_occupancy = target_sites[..., 2]  # [B, K]

        # Create mask for valid sites (occupancy > 0)
        valid_mask = (target_occupancy > 0.1).float()

        # Position loss - only for valid sites
        if pred_positions.shape[-1] == 2:
            # Gaussian mode - use MSE instead of NLL to avoid sigma instability
            pred_mu = pred_positions[..., 0]  # [B, K]
            pos_error = (pred_mu - target_positions) ** 2
        else:
            pred_mu = pred_positions.squeeze(-1)
            pos_error = (pred_mu - target_positions) ** 2

        # Clamp position error to prevent extreme values
        pos_error = pos_error.clamp(max=10.0)
        pos_loss = (pos_error * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        # Occupancy loss - MSE for AMP compatibility
        pred_occ = pred_occupancy.squeeze(-1).float()
        occ_loss = F.mse_loss(
            pred_occ,
            target_occupancy.clamp(0, 1).float(),
            reduction='mean'
        )

        # TF identity loss - only for valid sites
        if pred_tf_logits is not None and pred_tf_logits.shape[-1] > 1:
            # Only supervise slots that should be occupied
            tf_loss = F.cross_entropy(
                pred_tf_logits.view(-1, pred_tf_logits.shape[-1]),
                target_tf_ids.view(-1),
                reduction='none'
            ).view(batch_size, n_slots)
            tf_loss = (tf_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            tf_loss = torch.tensor(0.0, device=device)

        total = (
            self.position_weight * pos_loss +
            self.occupancy_weight * occ_loss +
            self.tf_weight * tf_loss
        )

        return {
            "position_loss": pos_loss,
            "occupancy_loss": occ_loss,
            "tf_loss": tf_loss,
            "total": total,
        }


class HungarianMatchingLoss(nn.Module):
    """
    Hungarian matching loss for set prediction.

    Finds optimal assignment between predicted slots and ground truth
    binding sites, then computes loss on matched pairs.

    Similar to DETR's bipartite matching for object detection.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        tf_weight: float = 1.0,
        occupancy_weight: float = 1.0,
    ):
        super().__init__()
        self.position_weight = position_weight
        self.tf_weight = tf_weight
        self.occupancy_weight = occupancy_weight

    def forward(
        self,
        pred_positions: torch.Tensor,
        pred_tf_logits: torch.Tensor,
        pred_occupancy: torch.Tensor,
        target_positions: torch.Tensor,
        target_tfs: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Hungarian matching loss.

        Args:
            pred_positions: [B, K, 2] predicted positions (mu, sigma)
            pred_tf_logits: [B, K, N_TFs] TF logits
            pred_occupancy: [B, K, 1] occupancy
            target_positions: [B, M] target positions (normalized)
            target_tfs: [B, M] target TF indices
            target_mask: [B, M] mask for valid targets

        Returns:
            Dictionary with matched losses
        """
        from scipy.optimize import linear_sum_assignment

        batch_size = pred_positions.shape[0]
        n_slots = pred_positions.shape[1]
        n_targets = target_positions.shape[1]

        total_pos_loss = 0.0
        total_tf_loss = 0.0
        total_occ_loss = 0.0
        n_matched = 0

        for b in range(batch_size):
            # Get valid targets for this sample
            valid_mask = target_mask[b].bool()
            n_valid = valid_mask.sum().item()

            if n_valid == 0:
                continue

            valid_positions = target_positions[b, valid_mask]  # [n_valid]
            valid_tfs = target_tfs[b, valid_mask]  # [n_valid]

            # Compute cost matrix [K, n_valid]
            pred_mu = pred_positions[b, :, 0]  # [K]

            # Position cost
            pos_cost = (pred_mu.unsqueeze(1) - valid_positions.unsqueeze(0)) ** 2

            # TF cost (negative log prob of correct class)
            tf_probs = F.softmax(pred_tf_logits[b], dim=-1)  # [K, N_TFs]
            tf_cost = -torch.log(tf_probs[:, valid_tfs.long()] + 1e-6)  # [K, n_valid]

            # Occupancy cost (prefer high occupancy slots)
            occ_cost = 1 - pred_occupancy[b].squeeze(-1).unsqueeze(1)  # [K, 1] -> [K, n_valid]
            occ_cost = occ_cost.expand(-1, n_valid)

            # Combined cost
            cost_matrix = (
                self.position_weight * pos_cost +
                self.tf_weight * tf_cost +
                self.occupancy_weight * occ_cost
            )

            # Hungarian matching
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # Compute losses on matched pairs
            for slot_idx, target_idx in zip(row_ind, col_ind):
                # Position loss
                total_pos_loss += pos_cost[slot_idx, target_idx]

                # TF loss
                total_tf_loss += tf_cost[slot_idx, target_idx]

                # Occupancy should be high for matched slots
                total_occ_loss += (1 - pred_occupancy[b, slot_idx, 0]) ** 2

            n_matched += len(row_ind)

            # Unmatched slots should have low occupancy
            unmatched_slots = list(set(range(n_slots)) - set(row_ind))
            for slot_idx in unmatched_slots:
                total_occ_loss += pred_occupancy[b, slot_idx, 0] ** 2

        # Average losses
        if n_matched > 0:
            total_pos_loss = total_pos_loss / n_matched
            total_tf_loss = total_tf_loss / n_matched
        total_occ_loss = total_occ_loss / (batch_size * n_slots)

        total = (
            self.position_weight * total_pos_loss +
            self.tf_weight * total_tf_loss +
            self.occupancy_weight * total_occ_loss
        )

        return {
            "position_loss": total_pos_loss,
            "tf_loss": total_tf_loss,
            "occupancy_loss": total_occ_loss,
            "total": total,
        }


class BEACONLoss(nn.Module):
    """
    Combined loss function for BEACON training.

    Multi-task loss combining:
    1. Profile reconstruction (BPNet compatibility)
    2. Position prediction
    3. TF identity prediction
    4. Occupancy prediction
    5. Slot diversity regularization
    6. Slot orthogonality regularization
    7. Binding site supervision (from pseudo-labels)
    """

    def __init__(
        self,
        n_tfs: int = 100,
        seq_len: int = 1000,
        position_mode: str = "gaussian",
        # Loss weights
        profile_weight: float = 1.0,
        position_weight: float = 1.0,
        tf_weight: float = 1.0,
        occupancy_weight: float = 1.0,
        diversity_weight: float = 0.1,
        orthogonality_weight: float = 0.1,
        site_supervision_weight: float = 0.5,
        # Sub-loss parameters
        label_smoothing: float = 0.1,
        sparsity_weight: float = 0.1,
    ):
        super().__init__()

        self.profile_weight = profile_weight
        self.position_weight = position_weight
        self.tf_weight = tf_weight
        self.occupancy_weight = occupancy_weight
        self.diversity_weight = diversity_weight
        self.orthogonality_weight = orthogonality_weight
        self.site_supervision_weight = site_supervision_weight

        # Individual loss components
        self.profile_loss = ProfileReconstructionLoss()
        self.position_loss = PositionLoss(mode=position_mode, seq_len=seq_len)
        self.tf_loss = TFIdentityLoss(n_tfs=n_tfs, label_smoothing=label_smoothing)
        self.occupancy_loss = OccupancyLoss(sparsity_weight=sparsity_weight)
        self.diversity_loss = SlotDiversityLoss()
        self.orthogonality_loss = SlotOrthogonalityLoss()
        self.site_supervision_loss = BindingSiteSupervisionLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined BEACON loss.

        Args:
            outputs: Model outputs dictionary
                - profile: [B, L]
                - positions: [B, K, 2]
                - tf_logits: [B, K, N_TFs]
                - occupancy: [B, K, 1]
                - attention: [B, K, L] (optional)

            targets: Target dictionary
                - profile: [B, L]
                - positions: [B, K] (optional)
                - tf_ids: [B, K] (optional)
                - occupancy: [B, K] (optional)
                - slot_mask: [B, K] (optional)

        Returns:
            Dictionary with all loss components and total
        """
        losses = {}

        # Profile reconstruction loss (always computed)
        if "profile" in targets:
            profile_losses = self.profile_loss(outputs["profile"], targets["profile"])
            losses["profile"] = profile_losses["total"]
        else:
            losses["profile"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Slot-supervised losses (when ground truth available)
        slot_mask = targets.get("slot_mask", None)

        if "positions" in targets:
            losses["position"] = self.position_loss(
                outputs["positions"],
                targets["positions"],
                mask=slot_mask,
            )
        else:
            losses["position"] = torch.tensor(0.0, device=outputs["profile"].device)

        if "tf_ids" in targets:
            losses["tf_identity"] = self.tf_loss(
                outputs["tf_logits"],
                targets["tf_ids"],
                mask=slot_mask,
            )
        else:
            losses["tf_identity"] = torch.tensor(0.0, device=outputs["profile"].device)

        if "occupancy" in targets:
            occ_losses = self.occupancy_loss(outputs["occupancy"], targets["occupancy"])
            losses["occupancy"] = occ_losses["total"]
        else:
            # Unsupervised sparsity regularization
            pred_occ = outputs["occupancy"].squeeze(-1)
            entropy = -(pred_occ * torch.log(pred_occ + 1e-6) +
                       (1 - pred_occ) * torch.log(1 - pred_occ + 1e-6))
            losses["occupancy"] = entropy.mean() * 0.1

        # Slot diversity loss (if attention available)
        if "attention" in outputs:
            losses["diversity"] = self.diversity_loss(outputs["attention"])
        else:
            losses["diversity"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Slot orthogonality loss (if slot embeddings available)
        if "slot_embeddings" in outputs:
            losses["orthogonality"] = self.orthogonality_loss(outputs["slot_embeddings"])
        else:
            losses["orthogonality"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Binding site supervision loss (using pseudo-labels from profiles)
        if "binding_sites" in targets:
            site_losses = self.site_supervision_loss(
                pred_positions=outputs["positions"],
                pred_occupancy=outputs["occupancy"],
                pred_tf_logits=outputs["tf_logits"],
                target_sites=targets["binding_sites"],
            )
            losses["site_position"] = site_losses["position_loss"]
            losses["site_occupancy"] = site_losses["occupancy_loss"]
            losses["site_tf"] = site_losses["tf_loss"]
            losses["site_supervision"] = site_losses["total"]
        else:
            losses["site_supervision"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Total weighted loss
        total = (
            self.profile_weight * losses["profile"] +
            self.position_weight * losses["position"] +
            self.tf_weight * losses["tf_identity"] +
            self.occupancy_weight * losses["occupancy"] +
            self.diversity_weight * losses["diversity"] +
            self.orthogonality_weight * losses["orthogonality"] +
            self.site_supervision_weight * losses["site_supervision"]
        )

        losses["total"] = total

        return losses
