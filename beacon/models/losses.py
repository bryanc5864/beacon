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
        device = pred_positions.device

        zero = torch.tensor(0.0, device=device)
        total_pos_loss = zero.clone()
        total_tf_loss = zero.clone()
        total_occ_loss = zero.clone()
        n_matched = 0

        for b in range(batch_size):
            # Get valid targets for this sample
            valid_mask = target_mask[b].bool()
            n_valid = valid_mask.sum().item()

            if n_valid == 0:
                # All slots should be empty
                total_occ_loss = total_occ_loss + (pred_occupancy[b, :, 0] ** 2).mean()
                continue

            valid_positions = target_positions[b, valid_mask]  # [n_valid]
            valid_tfs = target_tfs[b, valid_mask]  # [n_valid]

            # Compute cost matrix [K, n_valid]
            pred_mu = pred_positions[b, :, 0]  # [K]

            # Position cost (clamped to prevent explosion early in training)
            pos_cost = ((pred_mu.unsqueeze(1) - valid_positions.unsqueeze(0)) ** 2).clamp(max=10.0)

            # TF cost (negative log prob of correct class, clamped)
            tf_log_probs = F.log_softmax(pred_tf_logits[b], dim=-1)  # [K, N_TFs]
            tf_cost = (-tf_log_probs[:, valid_tfs.long()]).clamp(max=10.0)  # [K, n_valid]

            # Occupancy cost (prefer high occupancy slots)
            occ = pred_occupancy[b, :, 0]  # [K]
            occ_cost = (1 - occ).unsqueeze(1).expand(-1, n_valid)

            # Combined cost
            cost_matrix = (
                self.position_weight * pos_cost +
                self.tf_weight * tf_cost +
                self.occupancy_weight * occ_cost
            )

            # Hungarian matching (no grad — just finds assignment)
            cost_np = cost_matrix.detach().cpu().float().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # Compute losses on matched pairs using tensor indexing
            row_t = torch.tensor(row_ind, device=device, dtype=torch.long)
            col_t = torch.tensor(col_ind, device=device, dtype=torch.long)

            total_pos_loss = total_pos_loss + pos_cost[row_t, col_t].sum()
            total_tf_loss = total_tf_loss + tf_cost[row_t, col_t].sum()

            # Matched slots should have high occupancy
            total_occ_loss = total_occ_loss + ((1 - occ[row_t]) ** 2).sum()

            n_matched += len(row_ind)

            # Unmatched slots should have low occupancy
            matched_mask = torch.zeros(n_slots, dtype=torch.bool, device=device)
            matched_mask[row_t] = True
            unmatched_occ = occ[~matched_mask]
            if unmatched_occ.numel() > 0:
                total_occ_loss = total_occ_loss + (unmatched_occ ** 2).sum()

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


class SlotCountLoss(nn.Module):
    """
    Penalizes mismatch between number of active slots and target binding sites.

    Uses a soft count (sum of occupancies) to remain differentiable.
    """

    def forward(
        self,
        pred_occupancy: torch.Tensor,
        target_sites: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_occupancy: [B, K, 1] predicted occupancy
            target_sites: [B, K, 3] (position, tf_id, occupancy)
        Returns:
            Scalar loss
        """
        pred_occ = pred_occupancy.squeeze(-1)  # [B, K]
        target_occ = target_sites[..., 2]  # [B, K]

        # Soft count of predicted active slots
        pred_count = pred_occ.sum(dim=-1)  # [B]

        # Count target sites (occupancy > 0.1)
        target_count = (target_occ > 0.1).float().sum(dim=-1)  # [B]

        # MSE between counts
        return F.mse_loss(pred_count, target_count)


class AttentionLoadBalancingLoss(nn.Module):
    """
    Encourages attention mass to be distributed across multiple slots.

    Unlike SlotDiversityLoss which penalizes overlap, this penalizes
    concentration — when most attention mass goes to a single slot.
    """

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention: [B, K, L] slot attention weights
        Returns:
            Scalar loss (lower = more balanced)
        """
        # Sum attention per slot across positions: [B, K]
        slot_mass = attention.sum(dim=-1)

        # Normalize to get slot usage distribution: [B, K]
        slot_dist = slot_mass / (slot_mass.sum(dim=-1, keepdim=True) + 1e-6)

        # Negative entropy: high when concentrated on few slots
        log_dist = torch.log(slot_dist + 1e-6)
        neg_entropy = (slot_dist * log_dist).sum(dim=-1)  # [B]

        # Max entropy for K slots = log(K), normalize
        n_slots = attention.shape[1]
        max_entropy = math.log(n_slots)

        # Loss = normalized negative entropy (0 = uniform, 1 = single slot)
        return (neg_entropy / max_entropy + 1.0).mean()


class SlotContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss on slot embeddings.

    Pulls same-TF slot embeddings together, pushes different-TF apart.
    Only operates on active (matched) slots.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        slot_embeddings: torch.Tensor,
        pred_occupancy: torch.Tensor,
        target_sites: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slot_embeddings: [B, K, D]
            pred_occupancy: [B, K, 1]
            target_sites: [B, K, 3] (position, tf_id, occupancy)
        Returns:
            Scalar contrastive loss
        """
        target_occ = target_sites[..., 2]  # [B, K]
        target_tfs = target_sites[..., 1].long()  # [B, K]

        # Collect active slot embeddings and their TF labels
        valid_mask = target_occ > 0.1
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=slot_embeddings.device)

        embeddings = slot_embeddings[valid_mask]  # [N, D]
        labels = target_tfs[valid_mask]  # [N]

        if len(labels.unique()) < 2:
            return torch.tensor(0.0, device=slot_embeddings.device)

        # Normalize embeddings
        normed = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        similarity = torch.mm(normed, normed.t()) / self.temperature  # [N, N]

        # Labels match matrix
        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]

        # Exclude self-similarity
        mask_self = ~torch.eye(len(labels), dtype=torch.bool, device=similarity.device)
        label_match = label_match & mask_self

        # InfoNCE loss
        exp_sim = torch.exp(similarity) * mask_self.float()
        pos_sum = (exp_sim * label_match.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        # Only compute for samples that have at least one positive pair
        has_pos = label_match.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=slot_embeddings.device)

        loss = -torch.log(pos_sum[has_pos] / (all_sum[has_pos] + 1e-8)).mean()
        return loss


class SequenceTFPresenceLoss(nn.Module):
    """
    Multi-label auxiliary loss: predict which TFs are present in each sequence.

    Provides a global signal about TF composition before per-slot assignment.
    """

    def __init__(self, n_tfs: int = 7):
        super().__init__()
        self.n_tfs = n_tfs

    def forward(
        self,
        slot_embeddings: torch.Tensor,
        pred_occupancy: torch.Tensor,
        target_sites: torch.Tensor,
        tf_presence_head: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            slot_embeddings: [B, K, D]
            pred_occupancy: [B, K, 1]
            target_sites: [B, K, 3]
            tf_presence_head: nn.Linear(D, n_tfs)
        Returns:
            Binary cross-entropy loss
        """
        # Pool slot embeddings weighted by occupancy
        occ_weights = pred_occupancy.squeeze(-1).softmax(dim=-1)  # [B, K]
        pooled = torch.einsum('bk,bkd->bd', occ_weights, slot_embeddings)  # [B, D]

        # Predict TF presence
        logits = tf_presence_head(pooled)  # [B, n_tfs]

        # Build multi-label target
        target_occ = target_sites[..., 2]  # [B, K]
        target_tfs = target_sites[..., 1].long()  # [B, K]

        batch_size = target_sites.shape[0]
        target_labels = torch.zeros(batch_size, self.n_tfs, device=logits.device)
        for b in range(batch_size):
            valid = target_occ[b] > 0.1
            if valid.sum() > 0:
                present_tfs = target_tfs[b][valid].unique()
                target_labels[b, present_tfs] = 1.0

        return F.binary_cross_entropy_with_logits(logits, target_labels)


class AnchorLoss(nn.Module):
    """
    Pulls known-TF motif embeddings toward their designated anchor points.

    Ensures the n_known_tfs TFs remain well-separated and identifiable in
    the motif embedding space.  Uses cosine distance so the loss operates
    on direction, not magnitude.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        motif_embeddings: torch.Tensor,
        anchors: torch.Tensor,
        target_tfs: torch.Tensor,
        occupancy: torch.Tensor,
        target_sites: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            motif_embeddings: [B, K, D_m] slot motif embeddings
            anchors:          [T, D_m]     known TF anchor points (from head)
            target_tfs:       [B, K]       target TF indices per slot
            occupancy:        [B, K, 1]    predicted occupancy
            target_sites:     [B, K, 3]    (position, tf_id, occupancy)

        Returns:
            Scalar anchor loss
        """
        target_occ = target_sites[..., 2]  # [B, K]
        valid = target_occ > 0.1

        if valid.sum() == 0:
            return torch.tensor(0.0, device=motif_embeddings.device)

        # Gather valid embeddings and their target anchor index
        emb = F.normalize(motif_embeddings[valid], dim=-1)  # [N, D_m]
        tgt = target_tfs[valid].long()  # [N]

        # Clamp to valid anchor range
        n_anchors = anchors.shape[0]
        tgt = tgt.clamp(0, n_anchors - 1)

        anchor_norm = F.normalize(anchors, dim=-1)  # [T, D_m]
        target_anchor = anchor_norm[tgt]  # [N, D_m]

        # Cosine distance to target anchor (pull)
        cos_sim = (emb * target_anchor).sum(dim=-1)  # [N]
        pull_loss = (1.0 - cos_sim).mean()

        # Push: distance to non-target anchors should exceed margin
        all_sim = torch.matmul(emb, anchor_norm.t())  # [N, T]
        # Mask out target anchor
        mask = torch.ones_like(all_sim, dtype=torch.bool)
        mask.scatter_(1, tgt.unsqueeze(1), False)
        neg_sim = all_sim[mask].view(emb.shape[0], n_anchors - 1)
        push_loss = F.relu(neg_sim - (1.0 - self.margin)).mean()

        return pull_loss + push_loss


class ImportanceSupervisionLoss(nn.Module):
    """
    Supervises the per-base attribution head with gradient-derived importance.

    During training, ground-truth importance is obtained by backpropagating
    the profile reconstruction loss to the input and computing
    ``|grad * input|``.  The attribution head should approximate this signal.

    Uses Pearson-correlation-based loss so the head learns the *shape* of
    importance rather than its absolute scale.
    """

    def forward(
        self,
        pred_importance: torch.Tensor,
        target_importance: torch.Tensor,
        occupancy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_importance:   [B, K, L] predicted per-slot importance
            target_importance: [B, L]    gradient-derived total importance
            occupancy:         [B, K, 1] optional, weight by occupancy

        Returns:
            Scalar loss (lower = better correlation)
        """
        # Aggregate predicted importance across slots (occupancy-weighted)
        if occupancy is not None:
            occ_w = occupancy.squeeze(-1)  # [B, K]
            pred_total = (pred_importance * occ_w.unsqueeze(-1)).sum(dim=1)  # [B, L]
        else:
            pred_total = pred_importance.sum(dim=1)  # [B, L]

        # Negative Pearson correlation loss
        pred_c = pred_total - pred_total.mean(dim=-1, keepdim=True)
        tgt_c = target_importance - target_importance.mean(dim=-1, keepdim=True)

        num = (pred_c * tgt_c).sum(dim=-1)
        den = torch.sqrt(
            (pred_c ** 2).sum(dim=-1) * (tgt_c ** 2).sum(dim=-1) + 1e-8
        )
        pearson = num / den  # [B]

        # Loss = 1 - mean_correlation  (0 when perfectly correlated)
        return (1.0 - pearson).mean()


class PrototypeDiversityLoss(nn.Module):
    """
    Encourages codebook prototypes to spread uniformly through the motif
    embedding space, preventing dead prototypes and ensuring coverage for
    novel motif discovery.

    Combines two terms:
    1. Anti-collapse: penalises high pairwise similarity between prototypes.
    2. Usage balance: penalises when assignments concentrate on few prototypes.
    """

    def forward(
        self,
        prototypes: torch.Tensor,
        prototype_assignments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prototypes:            [P, D_m] learnable prototype embeddings
            prototype_assignments: [B, K, P] soft assignment weights (optional)

        Returns:
            Scalar diversity loss
        """
        proto_norm = F.normalize(prototypes, dim=-1)  # [P, D_m]

        # Pairwise cosine similarity [P, P]
        sim = torch.matmul(proto_norm, proto_norm.t())

        # Mask diagonal
        P = sim.shape[0]
        mask = 1.0 - torch.eye(P, device=sim.device)
        off_diag = sim * mask

        # Anti-collapse: penalise high off-diagonal similarity
        collapse_loss = (off_diag ** 2).sum() / (P * (P - 1))

        # Usage balance: penalise concentration when assignments given
        balance_loss = torch.tensor(0.0, device=prototypes.device)
        if prototype_assignments is not None:
            # Average assignment across batch & slots → [P]
            avg_usage = prototype_assignments.mean(dim=(0, 1))
            # Negative entropy of usage distribution
            log_usage = torch.log(avg_usage + 1e-8)
            neg_entropy = (avg_usage * log_usage).sum()
            max_entropy = math.log(P)
            balance_loss = (neg_entropy / max_entropy) + 1.0

        return collapse_loss + 0.5 * balance_loss


class ContrastiveTFEmbeddingLoss(nn.Module):
    """
    Contrastive loss pulling slot embeddings of the same TF closer
    in representation space and pushing different-TF embeddings apart.

    Unlike SlotContrastiveLoss which operates on raw slot embeddings,
    this operates on TF logit space for more direct TF discrimination.
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        tf_logits: torch.Tensor,
        target_sites: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tf_logits: [B, K, N_TFs] predicted TF logits
            target_sites: [B, K, 3] (position, tf_id, occupancy)
        Returns:
            Scalar contrastive loss
        """
        target_occ = target_sites[..., 2]
        target_tfs = target_sites[..., 1].long()

        valid_mask = target_occ > 0.1
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=tf_logits.device)

        # Get hidden representations (pre-softmax logits as embeddings)
        embeddings = tf_logits[valid_mask]  # [N, N_TFs]
        labels = target_tfs[valid_mask]  # [N]

        if len(labels.unique()) < 2:
            return torch.tensor(0.0, device=tf_logits.device)

        normed = F.normalize(embeddings, dim=-1)
        similarity = torch.mm(normed, normed.t()) / self.temperature

        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_self = ~torch.eye(len(labels), dtype=torch.bool, device=similarity.device)

        # Triplet-style: pull positives, push negatives past margin
        pos_mask = label_match & mask_self
        neg_mask = ~label_match & mask_self

        loss = torch.tensor(0.0, device=tf_logits.device)
        n_valid = 0

        for i in range(len(labels)):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_sim = similarity[i, pos_indices].mean()
            neg_sim = similarity[i, neg_indices]

            # Hinge loss: push negatives below positive - margin
            violations = F.relu(neg_sim - pos_sim + self.margin)
            loss = loss + violations.mean()
            n_valid += 1

        if n_valid > 0:
            loss = loss / n_valid
        return loss


class PerTFDifficultyLoss(nn.Module):
    """
    Adaptive per-TF difficulty weighting for TF classification.

    Maintains running estimates of per-TF accuracy and upweights
    losses for harder TFs, similar to focal loss but at TF level.
    """

    def __init__(self, n_tfs: int = 7, momentum: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.n_tfs = n_tfs
        self.momentum = momentum
        self.gamma = gamma
        self.register_buffer('tf_accuracy', torch.ones(n_tfs) * 0.5)
        self.register_buffer('tf_counts', torch.zeros(n_tfs))

    def forward(
        self,
        tf_logits: torch.Tensor,
        target_sites: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tf_logits: [B, K, N_TFs]
            target_sites: [B, K, 3]
        Returns:
            Difficulty-weighted TF loss
        """
        target_occ = target_sites[..., 2]
        target_tfs = target_sites[..., 1].long()
        valid_mask = target_occ > 0.1

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=tf_logits.device)

        logits = tf_logits[valid_mask]  # [N, N_TFs]
        labels = target_tfs[valid_mask]  # [N]

        # Ensure buffers are on the same device as inputs
        if self.tf_accuracy.device != logits.device:
            self.tf_accuracy = self.tf_accuracy.to(logits.device)
            self.tf_counts = self.tf_counts.to(logits.device)

        # Per-sample cross entropy
        ce = F.cross_entropy(logits, labels, reduction='none')

        # Compute difficulty weights from running accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()

            for tf_id in range(self.n_tfs):
                tf_mask = labels == tf_id
                if tf_mask.sum() > 0:
                    tf_acc = correct[tf_mask].mean()
                    self.tf_accuracy[tf_id] = (
                        self.momentum * self.tf_accuracy[tf_id]
                        + (1 - self.momentum) * tf_acc
                    )
                    self.tf_counts[tf_id] += tf_mask.sum().item()

        # Focal-style weighting: harder TFs get higher weight
        difficulty = (1 - self.tf_accuracy[labels]) ** self.gamma
        weighted_ce = ce * difficulty

        return weighted_ce.mean()


class MultiScaleImportanceLoss(nn.Module):
    """
    Multi-scale importance supervision for the attribution head.

    Computes Pearson correlation at multiple resolutions (1bp, 10bp, 50bp, 100bp)
    to help the attribution head learn both fine-grained and broad importance patterns.
    """

    def __init__(self, scales: Optional[list] = None):
        super().__init__()
        self.scales = scales or [1, 10, 50, 100]

    def forward(
        self,
        pred_importance: torch.Tensor,
        target_importance: torch.Tensor,
        occupancy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_importance: [B, K, L] per-slot importance
            target_importance: [B, L] gradient importance
            occupancy: [B, K, 1] optional occupancy weights
        Returns:
            Scalar loss (average correlation loss across scales)
        """
        if occupancy is not None:
            occ_w = occupancy.squeeze(-1)
            pred_total = (pred_importance * occ_w.unsqueeze(-1)).sum(dim=1)
        else:
            pred_total = pred_importance.sum(dim=1)

        total_loss = torch.tensor(0.0, device=pred_importance.device)
        n_scales = 0

        for scale in self.scales:
            if scale == 1:
                pred_s = pred_total
                tgt_s = target_importance
            else:
                L = pred_total.shape[-1]
                new_L = L // scale
                if new_L < 4:
                    continue
                pred_s = pred_total[:, :new_L * scale].reshape(-1, new_L, scale).mean(dim=-1)
                tgt_s = target_importance[:, :new_L * scale].reshape(-1, new_L, scale).mean(dim=-1)

            # Pearson correlation loss
            pred_c = pred_s - pred_s.mean(dim=-1, keepdim=True)
            tgt_c = tgt_s - tgt_s.mean(dim=-1, keepdim=True)
            num = (pred_c * tgt_c).sum(dim=-1)
            den = torch.sqrt(
                (pred_c ** 2).sum(dim=-1) * (tgt_c ** 2).sum(dim=-1) + 1e-8
            )
            pearson = num / den
            total_loss = total_loss + (1.0 - pearson).mean()
            n_scales += 1

        if n_scales > 0:
            total_loss = total_loss / n_scales
        return total_loss


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
    8. Hungarian matching (optional, for multi-TF training)
    9. Slot count loss (optional, for multi-TF training)
    10. Attention load balancing (optional, for multi-TF training)
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
        # Multi-TF specific
        use_hungarian: bool = False,
        slot_count_weight: float = 0.0,
        load_balancing_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        tf_presence_weight: float = 0.0,
        # Phase 1 losses
        anchor_weight: float = 0.0,
        importance_weight: float = 0.0,
        prototype_diversity_weight: float = 0.0,
        # Phase B additions
        tf_contrastive_weight: float = 0.0,
        tf_difficulty_weight: float = 0.0,
        multiscale_importance_weight: float = 0.0,
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
        self.use_hungarian = use_hungarian
        self.slot_count_weight = slot_count_weight
        self.load_balancing_weight = load_balancing_weight
        self.contrastive_weight = contrastive_weight
        self.tf_presence_weight = tf_presence_weight
        self.anchor_weight = anchor_weight
        self.importance_weight = importance_weight
        self.prototype_diversity_weight = prototype_diversity_weight
        self.tf_contrastive_weight = tf_contrastive_weight
        self.tf_difficulty_weight = tf_difficulty_weight
        self.multiscale_importance_weight = multiscale_importance_weight

        # Individual loss components
        self.profile_loss = ProfileReconstructionLoss()
        self.position_loss = PositionLoss(mode=position_mode, seq_len=seq_len)
        self.tf_loss = TFIdentityLoss(n_tfs=n_tfs, label_smoothing=label_smoothing)
        self.occupancy_loss = OccupancyLoss(sparsity_weight=sparsity_weight)
        self.diversity_loss = SlotDiversityLoss()
        self.orthogonality_loss = SlotOrthogonalityLoss()
        self.site_supervision_loss = BindingSiteSupervisionLoss()

        if use_hungarian:
            self.hungarian_loss = HungarianMatchingLoss()
        if slot_count_weight > 0:
            self.slot_count_loss = SlotCountLoss()
        if load_balancing_weight > 0:
            self.load_balancing_loss = AttentionLoadBalancingLoss()
        if contrastive_weight > 0:
            self.contrastive_loss = SlotContrastiveLoss()
        if tf_presence_weight > 0:
            self.tf_presence_loss = SequenceTFPresenceLoss(n_tfs=n_tfs)
        if anchor_weight > 0:
            self.anchor_loss = AnchorLoss()
        if importance_weight > 0:
            self.importance_loss = ImportanceSupervisionLoss()
        if prototype_diversity_weight > 0:
            self.prototype_loss = PrototypeDiversityLoss()
        if tf_contrastive_weight > 0:
            self.tf_contrastive_loss = ContrastiveTFEmbeddingLoss()
        if tf_difficulty_weight > 0:
            self.tf_difficulty_loss = PerTFDifficultyLoss(n_tfs=n_tfs)
        if multiscale_importance_weight > 0:
            self.multiscale_importance_loss = MultiScaleImportanceLoss()

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
            if self.use_hungarian:
                # Use Hungarian matching for optimal slot-to-site assignment
                target_sites = targets["binding_sites"]
                target_positions = target_sites[..., 0]  # [B, K]
                target_tfs = target_sites[..., 1]  # [B, K]
                target_occ = target_sites[..., 2]  # [B, K]
                target_mask = (target_occ > 0.1).float()

                hungarian_losses = self.hungarian_loss(
                    pred_positions=outputs["positions"],
                    pred_tf_logits=outputs["tf_logits"],
                    pred_occupancy=outputs["occupancy"],
                    target_positions=target_positions,
                    target_tfs=target_tfs,
                    target_mask=target_mask,
                )
                losses["site_position"] = hungarian_losses["position_loss"]
                losses["site_occupancy"] = hungarian_losses["occupancy_loss"]
                losses["site_tf"] = hungarian_losses["tf_loss"]
                losses["site_supervision"] = hungarian_losses["total"]
            else:
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

        # Slot count loss (penalize wrong number of active slots)
        if self.slot_count_weight > 0 and "binding_sites" in targets:
            losses["slot_count"] = self.slot_count_loss(
                outputs["occupancy"], targets["binding_sites"]
            )
        else:
            losses["slot_count"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Attention load balancing (penalize attention concentration)
        if self.load_balancing_weight > 0 and "attention" in outputs:
            losses["load_balancing"] = self.load_balancing_loss(outputs["attention"])
        else:
            losses["load_balancing"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Contrastive slot loss
        if self.contrastive_weight > 0 and "slot_embeddings" in outputs and "binding_sites" in targets:
            losses["contrastive"] = self.contrastive_loss(
                outputs["slot_embeddings"], outputs["occupancy"], targets["binding_sites"]
            )
        else:
            losses["contrastive"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Sequence-level TF presence loss
        if self.tf_presence_weight > 0 and "slot_embeddings" in outputs and "binding_sites" in targets:
            tf_head = outputs.get("tf_presence_head", targets.get("tf_presence_head", None))
            losses["tf_presence"] = self.tf_presence_loss(
                outputs["slot_embeddings"], outputs["occupancy"],
                targets["binding_sites"], tf_head
            )
        else:
            losses["tf_presence"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Phase 1: Anchor loss (motif embedding → known TF anchors)
        if (
            self.anchor_weight > 0
            and "motif_embeddings" in outputs
            and "motif_anchors" in outputs
            and "binding_sites" in targets
        ):
            target_tfs = targets["binding_sites"][..., 1].long()
            losses["anchor"] = self.anchor_loss(
                outputs["motif_embeddings"],
                outputs["motif_anchors"],
                target_tfs,
                outputs["occupancy"],
                targets["binding_sites"],
            )
        else:
            losses["anchor"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Phase 1: Importance supervision (attribution head)
        if (
            self.importance_weight > 0
            and "attribution" in outputs
            and "importance_target" in targets
        ):
            losses["importance"] = self.importance_loss(
                outputs["attribution"],
                targets["importance_target"],
                outputs.get("occupancy", None),
            )
        else:
            losses["importance"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Phase 1: Prototype diversity
        if (
            self.prototype_diversity_weight > 0
            and "motif_prototypes" in outputs
        ):
            losses["prototype_diversity"] = self.prototype_loss(
                outputs["motif_prototypes"],
                outputs.get("prototype_assignments", None),
            )
        else:
            losses["prototype_diversity"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Phase B: Contrastive TF embedding loss
        if self.tf_contrastive_weight > 0 and "tf_logits" in outputs and "binding_sites" in targets:
            losses["tf_contrastive"] = self.tf_contrastive_loss(
                outputs["tf_logits"], targets["binding_sites"]
            )
        else:
            losses["tf_contrastive"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Phase B: Per-TF difficulty-weighted loss
        if self.tf_difficulty_weight > 0 and "tf_logits" in outputs and "binding_sites" in targets:
            losses["tf_difficulty"] = self.tf_difficulty_loss(
                outputs["tf_logits"], targets["binding_sites"]
            )
        else:
            losses["tf_difficulty"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Phase B: Multi-scale importance loss
        if (
            self.multiscale_importance_weight > 0
            and "attribution" in outputs
            and "importance_target" in targets
        ):
            losses["multiscale_importance"] = self.multiscale_importance_loss(
                outputs["attribution"],
                targets["importance_target"],
                outputs.get("occupancy", None),
            )
        else:
            losses["multiscale_importance"] = torch.tensor(0.0, device=outputs["profile"].device)

        # Total weighted loss
        total = (
            self.profile_weight * losses["profile"] +
            self.position_weight * losses["position"] +
            self.tf_weight * losses["tf_identity"] +
            self.occupancy_weight * losses["occupancy"] +
            self.diversity_weight * losses["diversity"] +
            self.orthogonality_weight * losses["orthogonality"] +
            self.site_supervision_weight * losses["site_supervision"] +
            self.slot_count_weight * losses["slot_count"] +
            self.load_balancing_weight * losses["load_balancing"] +
            self.contrastive_weight * losses["contrastive"] +
            self.tf_presence_weight * losses["tf_presence"] +
            self.anchor_weight * losses["anchor"] +
            self.importance_weight * losses["importance"] +
            self.prototype_diversity_weight * losses["prototype_diversity"] +
            self.tf_contrastive_weight * losses["tf_contrastive"] +
            self.tf_difficulty_weight * losses["tf_difficulty"] +
            self.multiscale_importance_weight * losses["multiscale_importance"]
        )

        losses["total"] = total

        return losses
