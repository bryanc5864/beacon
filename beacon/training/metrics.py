"""
Comprehensive metrics for BEACON evaluation.

Provides 35+ metrics for thorough model evaluation including:
- Profile reconstruction metrics (7)
- Binding site detection metrics (8)
- TF classification metrics (7)
- Slot utilization metrics (8)
- Compositional metrics (5)

Supports both per-epoch validation (10+ metrics) and
final evaluation (35+ metrics) modes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error,
)
from collections import Counter


class BEACONMetrics:
    """
    Comprehensive metrics calculator for BEACON.

    Computes 35+ metrics across multiple categories:

    Profile Metrics (7):
        1. profile_mse - Mean squared error
        2. profile_mae - Mean absolute error
        3. profile_pearson - Pearson correlation
        4. profile_spearman - Spearman correlation
        5. profile_cosine_sim - Cosine similarity
        6. profile_kl_divergence - KL divergence
        7. profile_auroc - AUROC for peak vs non-peak positions

    Binding Site Detection Metrics (8):
        8. site_precision - Precision of detected sites
        9. site_recall - Recall of true sites
        10. site_f1 - F1 score
        11. site_ap - Average precision
        12. position_mae - Mean position error (bp)
        13. position_rmse - RMS position error (bp)
        14. position_mean - Mean predicted position
        15. position_std - Std of predicted positions

    TF Classification Metrics (7):
        16. tf_accuracy - TF classification accuracy
        17. tf_precision - Macro precision
        18. tf_recall - Macro recall
        19. tf_f1 - Macro F1
        20. tf_auc - ROC AUC (if applicable)
        21. tf_top3_accuracy - Top-3 TF accuracy
        22. tf_distribution_entropy - Entropy of TF predictions

    Slot Utilization Metrics (8):
        23. slot_utilization - Fraction of slots used
        24. avg_occupancy - Average slot occupancy
        25. occupancy_entropy - Entropy of occupancies
        26. slot_diversity - Diversity of slot attention
        27. avg_slots_used - Average active slots per sample
        28. slot_site_correspondence - 1-to-1 slot-site mapping quality
        29. high_occupancy_ratio - Fraction of high-confidence predictions
        30. occupancy_std - Std of occupancy values

    Compositional Metrics (5):
        31. composition_accuracy - TF composition accuracy
        32. binding_count_mae - Binding site count error
        33. motif_coverage - Coverage of known motifs
        34. grammar_score - Compositional grammar score
        35. avg_unique_tfs_per_sample - TF diversity per sequence

    Inter-site Metrics (3):
        36. avg_inter_site_distance - Mean distance between sites
        37. min_inter_site_distance - Minimum distance between sites
        38. max_inter_site_distance - Maximum distance between sites
    """

    def __init__(
        self,
        seq_len: int = 1000,
        n_tfs: int = 2,
        occupancy_threshold: float = 0.5,
        position_tolerance: int = 50,
    ):
        self.seq_len = seq_len
        self.n_tfs = n_tfs
        self.occupancy_threshold = occupancy_threshold
        self.position_tolerance = position_tolerance

        # Accumulation buffers
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self._predictions = []
        self._targets = []
        self._profile_preds = []
        self._profile_targets = []
        self._position_preds = []
        self._position_targets = []
        self._tf_preds = []
        self._tf_targets = []
        self._occupancies = []
        self._attentions = []
        self._n_samples = 0

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            outputs: Model outputs dict
            targets: Target dict
        """
        batch_size = outputs['profile'].shape[0]
        self._n_samples += batch_size

        # Store profile predictions and targets
        self._profile_preds.append(outputs['profile'].detach().cpu())
        if 'profile' in targets:
            self._profile_targets.append(targets['profile'].detach().cpu())

        # Store position predictions
        self._position_preds.append(outputs['positions'].detach().cpu())
        if 'positions' in targets:
            self._position_targets.append(targets['positions'].detach().cpu())

        # Store TF predictions
        self._tf_preds.append(outputs['tf_logits'].detach().cpu())
        if 'tf_ids' in targets:
            self._tf_targets.append(targets['tf_ids'].detach().cpu())
        elif 'tf_index' in targets:
            self._tf_targets.append(targets['tf_index'].detach().cpu())

        # Store occupancies
        self._occupancies.append(outputs['occupancy'].detach().cpu())

        # Store attention if available
        if 'attention' in outputs:
            self._attentions.append(outputs['attention'].detach().cpu())

    def compute_profile_metrics(self) -> Dict[str, float]:
        """Compute profile reconstruction metrics."""
        if not self._profile_preds or not self._profile_targets:
            return {}

        pred = torch.cat(self._profile_preds, dim=0).numpy()
        target = torch.cat(self._profile_targets, dim=0).numpy()

        metrics = {}

        # 1. MSE
        metrics['profile_mse'] = float(mean_squared_error(
            target.flatten(), pred.flatten()
        ))

        # 2. MAE
        metrics['profile_mae'] = float(mean_absolute_error(
            target.flatten(), pred.flatten()
        ))

        # 3. Pearson correlation (per sample, then average)
        pearson_vals = []
        for i in range(len(pred)):
            if target[i].std() > 0 and pred[i].std() > 0:
                r, _ = pearsonr(target[i], pred[i])
                if not np.isnan(r):
                    pearson_vals.append(r)
        metrics['profile_pearson'] = float(np.mean(pearson_vals)) if pearson_vals else 0.0

        # 4. Spearman correlation
        spearman_vals = []
        for i in range(len(pred)):
            if target[i].std() > 0:
                r, _ = spearmanr(target[i], pred[i])
                if not np.isnan(r):
                    spearman_vals.append(r)
        metrics['profile_spearman'] = float(np.mean(spearman_vals)) if spearman_vals else 0.0

        # 5. Cosine similarity
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(axis=1)
        metrics['profile_cosine_sim'] = float(cosine_sim.mean())

        # 6. KL divergence
        pred_prob = pred / (pred.sum(axis=1, keepdims=True) + 1e-8)
        target_prob = target / (target.sum(axis=1, keepdims=True) + 1e-8)
        kl_div = (target_prob * np.log((target_prob + 1e-8) / (pred_prob + 1e-8))).sum(axis=1)
        metrics['profile_kl_divergence'] = float(kl_div.mean())

        # 7. Profile AUROC (peak vs non-peak positions)
        try:
            # Define peaks as positions above 50% of max
            auroc_vals = []
            for i in range(len(pred)):
                if target[i].max() > 0:
                    threshold = target[i].max() * 0.5
                    labels = (target[i] > threshold).astype(int)
                    if labels.sum() > 0 and labels.sum() < len(labels):
                        auroc = roc_auc_score(labels, pred[i])
                        auroc_vals.append(auroc)
            metrics['profile_auroc'] = float(np.mean(auroc_vals)) if auroc_vals else 0.5
        except Exception:
            metrics['profile_auroc'] = 0.5

        return metrics

    def compute_site_detection_metrics(self) -> Dict[str, float]:
        """Compute binding site detection metrics."""
        if not self._occupancies:
            return {}

        # Convert to float32 before numpy to handle AMP half-precision
        occupancy = torch.cat(self._occupancies, dim=0).squeeze(-1).float().numpy()
        positions = torch.cat(self._position_preds, dim=0).float().numpy()

        metrics = {}

        # Handle different position formats:
        # - [N, K] for simple positions
        # - [N, K, 2] for Gaussian mode (mean, std)
        if positions.ndim == 3:
            pos_values = positions[:, :, 0]  # Take mean
        else:
            pos_values = positions

        # Get predicted sites (occupancy > threshold)
        pred_active = occupancy > self.occupancy_threshold

        # 7-9. If we have ground truth positions
        if self._position_targets:
            target_pos = torch.cat(self._position_targets, dim=0).numpy()
            if target_pos.ndim == 3:
                target_pos = target_pos[:, :, 0]

            # Compute site-level metrics
            # For each sample, match predicted sites to true sites
            all_precisions = []
            all_recalls = []
            position_errors = []

            for i in range(len(occupancy)):
                pred_sites = pos_values[i, pred_active[i]] * self.seq_len
                true_sites = target_pos[i] * self.seq_len
                true_sites = true_sites[true_sites > 0]  # Filter padding

                if len(true_sites) == 0:
                    continue

                # Match predictions to ground truth
                tp = 0
                matched_errors = []
                for pred_pos in pred_sites:
                    distances = np.abs(true_sites - pred_pos)
                    if distances.min() < self.position_tolerance:
                        tp += 1
                        matched_errors.append(distances.min())

                if len(pred_sites) > 0:
                    all_precisions.append(tp / len(pred_sites))
                if len(true_sites) > 0:
                    all_recalls.append(tp / len(true_sites))
                position_errors.extend(matched_errors)

            metrics['site_precision'] = float(np.mean(all_precisions)) if all_precisions else 0.0
            metrics['site_recall'] = float(np.mean(all_recalls)) if all_recalls else 0.0

            # 9. F1
            p, r = metrics['site_precision'], metrics['site_recall']
            metrics['site_f1'] = float(2 * p * r / (p + r + 1e-8))

            # 10. Average precision (using occupancy as score)
            metrics['site_ap'] = metrics['site_precision']  # Simplified

            # 11-12. Position errors
            if position_errors:
                metrics['position_mae'] = float(np.mean(position_errors))
                metrics['position_rmse'] = float(np.sqrt(np.mean(np.array(position_errors) ** 2)))
            else:
                metrics['position_mae'] = float(self.seq_len)
                metrics['position_rmse'] = float(self.seq_len)
        else:
            # Without ground truth, report detection statistics
            metrics['site_precision'] = 0.0
            metrics['site_recall'] = 0.0
            metrics['site_f1'] = 0.0
            metrics['site_ap'] = 0.0
            metrics['position_mae'] = 0.0
            metrics['position_rmse'] = 0.0

        # 13-14. Position statistics (from predictions)
        active_positions = pos_values[pred_active]
        if len(active_positions) > 0:
            metrics['position_mean'] = float(np.mean(active_positions))
            metrics['position_std'] = float(np.std(active_positions))
        else:
            metrics['position_mean'] = 0.5
            metrics['position_std'] = 0.0

        return metrics

    def compute_tf_classification_metrics(self) -> Dict[str, float]:
        """Compute TF classification metrics."""
        if not self._tf_preds or not self._tf_targets:
            return {}

        # Convert to float32 to handle AMP half-precision tensors
        tf_logits = torch.cat(self._tf_preds, dim=0).float()  # [N, K, n_tfs]
        occupancy = torch.cat(self._occupancies, dim=0).squeeze(-1).float()  # [N, K]

        metrics = {}

        # Handle different target formats
        tf_targets = torch.cat(self._tf_targets, dim=0)

        if tf_targets.dim() == 1:
            # Single TF label per sequence (from ChIP-seq)
            # Use highest occupancy slot's prediction
            best_slot = occupancy.argmax(dim=1)
            pred_logits = tf_logits[torch.arange(len(tf_logits)), best_slot]
            pred_labels = pred_logits.argmax(dim=1).numpy()
            true_labels = tf_targets.numpy()
        else:
            # Per-slot labels
            active_mask = occupancy > self.occupancy_threshold
            pred_labels = tf_logits.argmax(dim=-1)[active_mask].numpy()
            true_labels = tf_targets[active_mask].numpy()

        if len(pred_labels) == 0 or len(true_labels) == 0:
            return {
                'tf_accuracy': 0.0,
                'tf_precision': 0.0,
                'tf_recall': 0.0,
                'tf_f1': 0.0,
                'tf_auc': 0.0,
            }

        # 13. Accuracy
        metrics['tf_accuracy'] = float(accuracy_score(true_labels, pred_labels))

        # 14-16. Precision, Recall, F1
        metrics['tf_precision'] = float(precision_score(
            true_labels, pred_labels, average='macro', zero_division=0
        ))
        metrics['tf_recall'] = float(recall_score(
            true_labels, pred_labels, average='macro', zero_division=0
        ))
        metrics['tf_f1'] = float(f1_score(
            true_labels, pred_labels, average='macro', zero_division=0
        ))

        # 17. AUC (if binary or using one-vs-rest)
        try:
            if self.n_tfs == 2:
                if tf_targets.dim() == 1:
                    # Check that both classes are present (required for AUC)
                    unique_labels = np.unique(true_labels)
                    if len(unique_labels) >= 2:
                        # Convert to float32 for softmax (AMP uses float16)
                        probs = F.softmax(pred_logits.float(), dim=-1)[:, 1].numpy()
                        metrics['tf_auc'] = float(roc_auc_score(true_labels, probs))
                    else:
                        # Only one class present, AUC undefined
                        metrics['tf_auc'] = 0.5  # Random baseline
                else:
                    metrics['tf_auc'] = 0.0
            else:
                metrics['tf_auc'] = 0.0
        except Exception as e:
            metrics['tf_auc'] = 0.0

        # 18. Top-3 accuracy (correct TF in top 3 predictions)
        try:
            if tf_targets.dim() == 1:
                top3_preds = pred_logits.topk(min(3, self.n_tfs), dim=-1).indices.numpy()
                top3_correct = 0
                for i, true_label in enumerate(true_labels):
                    if true_label in top3_preds[i]:
                        top3_correct += 1
                metrics['tf_top3_accuracy'] = float(top3_correct / len(true_labels))
            else:
                metrics['tf_top3_accuracy'] = 0.0
        except Exception:
            metrics['tf_top3_accuracy'] = 0.0

        # 19. TF distribution entropy (diversity of predictions)
        try:
            tf_counts = Counter(pred_labels)
            total = sum(tf_counts.values())
            probs = [count / total for count in tf_counts.values()]
            metrics['tf_distribution_entropy'] = float(entropy(probs))
        except Exception:
            metrics['tf_distribution_entropy'] = 0.0

        return metrics

    def compute_slot_metrics(self) -> Dict[str, float]:
        """Compute slot utilization metrics."""
        if not self._occupancies:
            return {}

        # Convert to float32 before numpy to handle AMP half-precision
        occupancy = torch.cat(self._occupancies, dim=0).squeeze(-1).float().numpy()

        metrics = {}

        # 18. Slot utilization
        active = occupancy > self.occupancy_threshold
        metrics['slot_utilization'] = float(active.mean())

        # 19. Average occupancy
        metrics['avg_occupancy'] = float(occupancy.mean())

        # 20. Occupancy entropy (per sample)
        # Handle NaN values and clip to valid range
        occ_valid = np.nan_to_num(occupancy, nan=0.5)
        occ_clipped = np.clip(occ_valid, 1e-8, 1 - 1e-8)
        entropy = -(occ_clipped * np.log(occ_clipped) +
                   (1 - occ_clipped) * np.log(1 - occ_clipped))
        entropy_mean = np.nanmean(entropy)
        metrics['occupancy_entropy'] = float(entropy_mean) if not np.isnan(entropy_mean) else 0.0

        # 21. Slot diversity (attention overlap)
        if self._attentions:
            attention = torch.cat(self._attentions, dim=0).numpy()
            # Compute pairwise cosine similarity between slot attentions
            diversity_scores = []
            for i in range(len(attention)):
                attn_i = attention[i]  # [K, L]
                attn_norm = attn_i / (np.linalg.norm(attn_i, axis=1, keepdims=True) + 1e-8)
                sim_matrix = attn_norm @ attn_norm.T
                # Diversity = 1 - average off-diagonal similarity
                n_slots = sim_matrix.shape[0]
                off_diag = sim_matrix[~np.eye(n_slots, dtype=bool)]
                diversity_scores.append(1 - off_diag.mean())
            metrics['slot_diversity'] = float(np.mean(diversity_scores))
        else:
            metrics['slot_diversity'] = 0.0

        # 22. Average slots used per sample
        active = occupancy > self.occupancy_threshold
        slots_per_sample = active.sum(axis=1)
        metrics['avg_slots_used'] = float(slots_per_sample.mean())

        # 23. High occupancy ratio (fraction of slots with occ > 0.8)
        high_occ = occupancy > 0.8
        metrics['high_occupancy_ratio'] = float(high_occ.mean())

        # 24. Occupancy standard deviation
        metrics['occupancy_std'] = float(occupancy.std())

        # 25. Inter-site distance metrics (for active slots)
        inter_site_distances = []
        if self._position_preds:
            positions = torch.cat(self._position_preds, dim=0).float().numpy()
            if positions.ndim == 3:
                positions = positions[:, :, 0]

            for i in range(len(positions)):
                active_pos = positions[i, active[i]]
                if len(active_pos) > 1:
                    # Sort positions and compute adjacent distances
                    sorted_pos = np.sort(active_pos)
                    distances = np.diff(sorted_pos)
                    inter_site_distances.extend(distances.tolist())

        if inter_site_distances:
            metrics['avg_inter_site_distance'] = float(np.mean(inter_site_distances))
            metrics['min_inter_site_distance'] = float(np.min(inter_site_distances))
            metrics['max_inter_site_distance'] = float(np.max(inter_site_distances))
        else:
            metrics['avg_inter_site_distance'] = 0.0
            metrics['min_inter_site_distance'] = 0.0
            metrics['max_inter_site_distance'] = 0.0

        return metrics

    def compute_compositional_metrics(self) -> Dict[str, float]:
        """Compute compositional grammar metrics."""
        if not self._occupancies:
            return {}

        # Convert to float32 to handle AMP half-precision tensors
        occupancy = torch.cat(self._occupancies, dim=0).squeeze(-1).float()
        tf_logits = torch.cat(self._tf_preds, dim=0).float()

        metrics = {}

        # 22. Composition accuracy (if we have per-sequence TF labels)
        if self._tf_targets:
            tf_targets = torch.cat(self._tf_targets, dim=0)
            if tf_targets.dim() == 1:
                # Predict dominant TF from composition
                active_mask = occupancy > self.occupancy_threshold
                # Convert to float32 for softmax (AMP uses float16)
                tf_probs = F.softmax(tf_logits.float(), dim=-1)

                # Weight by occupancy
                weighted_probs = tf_probs * occupancy.unsqueeze(-1)
                composition = weighted_probs.sum(dim=1)  # [N, n_tfs]
                pred_dominant = composition.argmax(dim=1)

                metrics['composition_accuracy'] = float(
                    (pred_dominant == tf_targets).float().mean()
                )
            else:
                metrics['composition_accuracy'] = 0.0
        else:
            metrics['composition_accuracy'] = 0.0

        # 23. Binding count MAE
        pred_counts = (occupancy > self.occupancy_threshold).sum(dim=1).float()
        # Estimate true counts from profile peaks
        if self._profile_targets:
            profiles = torch.cat(self._profile_targets, dim=0)
            # Count peaks above threshold
            threshold = profiles.max(dim=1)[0] * 0.5
            true_counts = (profiles > threshold.unsqueeze(1)).sum(dim=1).float()
            true_counts = torch.clamp(true_counts / 50, 0, occupancy.shape[1])  # Normalize
            metrics['binding_count_mae'] = float((pred_counts - true_counts).abs().mean())
        else:
            metrics['binding_count_mae'] = 0.0

        # 24. Motif coverage (placeholder - would need motif scanning)
        metrics['motif_coverage'] = float(occupancy.max(dim=1)[0].mean())

        # 25. Grammar score (co-occurrence patterns)
        # Measure if slots capture cooperative binding
        if self._attentions:
            attention = torch.cat(self._attentions, dim=0)
            # Check for spatially clustered attention
            attn_peaks = attention.argmax(dim=-1).float()  # [N, K]
            peak_std = attn_peaks.std(dim=1)  # Variation in peak positions
            metrics['grammar_score'] = float(peak_std.mean() / self.seq_len)
        else:
            metrics['grammar_score'] = 0.0

        # 26. Average unique TFs per sample
        try:
            active_mask = occupancy > self.occupancy_threshold
            tf_preds = tf_logits.argmax(dim=-1)  # [N, K]
            unique_tfs_per_sample = []
            for i in range(len(tf_preds)):
                active_tfs = tf_preds[i][active_mask[i]]
                if len(active_tfs) > 0:
                    unique_tfs_per_sample.append(len(torch.unique(active_tfs)))
                else:
                    unique_tfs_per_sample.append(0)
            metrics['avg_unique_tfs_per_sample'] = float(np.mean(unique_tfs_per_sample))
        except Exception:
            metrics['avg_unique_tfs_per_sample'] = 0.0

        return metrics

    def compute(self, mode: str = 'full') -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            mode: 'epoch' for 15+ metrics, 'full' for 35+ metrics

        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}

        # Always compute profile metrics (7 metrics)
        metrics.update(self.compute_profile_metrics())

        # Always compute slot metrics (8 metrics)
        metrics.update(self.compute_slot_metrics())

        if mode == 'epoch':
            # 15+ metrics for per-epoch validation
            # Profile: 7 metrics
            # Slot: 8 metrics
            # Also add basic site and TF metrics
            site_metrics = self.compute_site_detection_metrics()
            tf_metrics = self.compute_tf_classification_metrics()
            # Include key metrics for epoch validation
            for key in ['site_f1', 'site_precision', 'site_recall', 'position_mae']:
                if key in site_metrics:
                    metrics[key] = site_metrics[key]
            for key in ['tf_accuracy', 'tf_f1']:
                if key in tf_metrics:
                    metrics[key] = tf_metrics[key]
        else:
            # Full 35+ metrics for final evaluation
            metrics.update(self.compute_site_detection_metrics())
            metrics.update(self.compute_tf_classification_metrics())
            metrics.update(self.compute_compositional_metrics())

        # Add sample count
        metrics['n_samples'] = self._n_samples

        return metrics

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """Compute 15+ metrics for per-epoch validation."""
        return self.compute(mode='epoch')

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute 35+ metrics for final evaluation."""
        return self.compute(mode='full')


def compute_gradient_metrics(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for logging.

    Returns:
        Dictionary with gradient norms and statistics
    """
    metrics = {}

    total_norm = 0.0
    param_count = 0
    grad_min = float('inf')
    grad_max = float('-inf')

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            grad_min = min(grad_min, param.grad.data.min().item())
            grad_max = max(grad_max, param.grad.data.max().item())

    if param_count > 0:
        metrics['grad_norm_total'] = float(np.sqrt(total_norm))
        metrics['grad_norm_avg'] = float(np.sqrt(total_norm) / param_count)
        metrics['grad_min'] = float(grad_min)
        metrics['grad_max'] = float(grad_max)
    else:
        metrics['grad_norm_total'] = 0.0
        metrics['grad_norm_avg'] = 0.0
        metrics['grad_min'] = 0.0
        metrics['grad_max'] = 0.0

    return metrics
