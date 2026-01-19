"""
BEACON Trainer with comprehensive logging and validation.

Features:
- Batch-level logging (batch number, loss components, gradients, learning rate)
- Epoch-level logging with 10+ validation metrics
- End-of-training evaluation with 20+ metrics
- Checkpoint saving and early stopping
- Mixed precision training support
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from .metrics import BEACONMetrics
from .logger import BEACONLogger


class BEACONTrainer:
    """
    Comprehensive trainer for BEACON models.

    Provides:
    - Full training loop with batch/epoch logging
    - Gradient tracking and clipping
    - Per-epoch validation with 10+ metrics
    - Final evaluation with 20+ metrics
    - Checkpoint management
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        output_dir: Path = Path("outputs"),
        experiment_name: Optional[str] = None,
        device: str = "cuda",
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        log_interval: int = 10,
        val_interval: int = 1,
        save_interval: int = 5,
        early_stopping_patience: int = 15,
        use_tensorboard: bool = True,
        num_slots: int = 16,
        num_tfs: int = 879,
    ):
        """
        Args:
            model: BEACON model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader for final evaluation
            loss_fn: Loss function (BEACONLoss)
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            output_dir: Directory for outputs
            experiment_name: Name for this experiment
            device: Device to train on
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            grad_clip: Gradient clipping value
            use_amp: Use automatic mixed precision
            log_interval: Log every N batches
            val_interval: Validate every N epochs
            save_interval: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            use_tensorboard: Enable TensorBoard logging
            num_slots: Number of slots for metrics
            num_tfs: Number of TFs for metrics
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and torch.cuda.is_available()
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.early_stopping_patience = early_stopping_patience

        # Move model to device
        self.model = self.model.to(self.device)

        # Track if model is wrapped in DataParallel
        self.is_data_parallel = isinstance(self.model, nn.DataParallel)
        self.base_model = self.model.module if self.is_data_parallel else self.model

        # Setup loss function
        if loss_fn is None:
            from beacon.models.losses import BEACONLoss
            self.loss_fn = BEACONLoss()
        else:
            self.loss_fn = loss_fn

        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Setup scheduler
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=learning_rate * 0.01,
            )
        else:
            self.scheduler = scheduler

        # Setup AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Setup logger
        self.output_dir = Path(output_dir)
        self.logger = BEACONLogger(
            output_dir=self.output_dir,
            experiment_name=experiment_name,
            use_tensorboard=use_tensorboard,
        )

        # Setup metrics calculator
        self.metrics = BEACONMetrics(
            seq_len=self.base_model.seq_len if hasattr(self.base_model, 'seq_len') else 1000,
            n_tfs=num_tfs,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with training history and final metrics
        """
        self.logger.log_message("Starting BEACON training")
        self.logger.log_model_summary(self.model)

        # Log config
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        config = {
            'num_epochs': self.num_epochs,
            'batch_size': self.train_loader.batch_size,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0),
            'grad_clip': self.grad_clip,
            'use_amp': self.use_amp,
            'device': str(self.device),
            'n_gpus': n_gpus,
            'multi_gpu': self.is_data_parallel,
            'model_params': sum(p.numel() for p in self.base_model.parameters()),
        }
        self.logger.log_config(config)

        training_start = time.time()

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch

            # Train one epoch
            self.logger.log_epoch_start(epoch, self.num_epochs)
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = None
            if self.val_loader and epoch % self.val_interval == 0:
                val_metrics = self._validate()
                self.logger.log_validation(val_metrics, mode='val')

                # Check for improvement
                val_loss = val_metrics.get('loss', float('inf'))
                val_f1 = val_metrics.get('site_f1', 0.0)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_metric = val_f1
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.epochs_without_improvement += 1

            # Log epoch summary
            epoch_time = time.time() - training_start
            self.logger.log_epoch_end(train_metrics, val_metrics, epoch_time)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Save periodic checkpoint
            if epoch % self.save_interval == 0:
                self._save_checkpoint()

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.log_message(
                    f"Early stopping after {epoch} epochs "
                    f"({self.early_stopping_patience} epochs without improvement)"
                )
                break

        # Final evaluation on test set
        final_metrics = {}
        if self.test_loader:
            self.logger.log_message("Running final evaluation on test set")
            # Load best model
            best_path = self.output_dir / self.logger.experiment_name / "best_model.pt"
            if best_path.exists():
                self._load_checkpoint(best_path)
            final_metrics = self._final_evaluation()
            self.logger.log_validation(final_metrics, mode='test')

        total_time = time.time() - training_start
        self.logger.log_message(f"Training completed in {total_time/3600:.2f} hours")

        # Save final history
        self.logger.close()

        return {
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'final_test_metrics': final_metrics,
            'total_epochs': self.current_epoch,
            'total_time': total_time,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with comprehensive logging.

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()

        epoch_losses = {}
        epoch_grad_norms = []
        batch_times = []
        samples_processed = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(batch['sequence'])
                    losses = self._compute_losses(outputs, batch)

                # Backward pass with scaling
                self.scaler.scale(losses['total']).backward()

                # Unscale for gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = self._clip_gradients()

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch['sequence'])
                losses = self._compute_losses(outputs, batch)

                losses['total'].backward()
                grad_norm = self._clip_gradients()
                self.optimizer.step()

            # Track metrics
            self.global_step += 1
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            epoch_grad_norms.append(grad_norm)
            samples_processed += batch['sequence'].size(0)

            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)

            # Log batch metrics
            if batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                grad_metrics = self._compute_gradient_metrics()
                grad_metrics['grad_norm_total'] = grad_norm

                samples_per_sec = batch['sequence'].size(0) / batch_time

                loss_dict = {k: v[-1] for k, v in epoch_losses.items()}
                self.logger.log_batch(
                    batch_idx=batch_idx,
                    total_batches=len(self.train_loader),
                    losses=loss_dict,
                    lr=lr,
                    grad_metrics=grad_metrics,
                    batch_time=batch_time,
                    samples_per_sec=samples_per_sec,
                )

        # Compute epoch averages
        epoch_time = time.time() - epoch_start
        avg_metrics = {
            f'avg_{key}': sum(values) / len(values)
            for key, values in epoch_losses.items()
        }
        avg_metrics['avg_grad_norm'] = sum(epoch_grad_norms) / len(epoch_grad_norms)
        avg_metrics['samples_per_sec'] = samples_processed / epoch_time
        avg_metrics['epoch_time'] = epoch_time

        return avg_metrics

    def _validate(self) -> Dict[str, float]:
        """
        Run validation with 10+ metrics.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics.reset()

        all_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch['sequence'])
                        losses = self._compute_losses(outputs, batch)
                else:
                    outputs = self.model(batch['sequence'])
                    losses = self._compute_losses(outputs, batch)

                all_losses.append(losses['total'].item())

                # Update metrics
                self._update_metrics(outputs, batch)

        # Compute all metrics (10+ metrics)
        metrics = self.metrics.compute()
        metrics['loss'] = sum(all_losses) / len(all_losses)
        metrics['n_samples'] = len(self.val_loader.dataset)

        return metrics

    def _final_evaluation(self) -> Dict[str, float]:
        """
        Run comprehensive final evaluation with 20+ metrics.

        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        self.metrics.reset()

        all_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = self._to_device(batch)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch['sequence'])
                        losses = self._compute_losses(outputs, batch)
                else:
                    outputs = self.model(batch['sequence'])
                    losses = self._compute_losses(outputs, batch)

                all_losses.append(losses['total'].item())

                # Update metrics
                self._update_metrics(outputs, batch)

                # Store for additional analysis
                all_predictions.append({
                    k: v.cpu() for k, v in outputs.items()
                })
                all_targets.append({
                    k: v.cpu() for k, v in batch.items()
                    if k != 'sequence'
                })

        # Compute all metrics (20+ metrics for final evaluation)
        metrics = self.metrics.compute()

        # Add additional final evaluation metrics
        metrics['loss'] = sum(all_losses) / len(all_losses)
        metrics['n_samples'] = len(self.test_loader.dataset)

        # Compute additional compositional metrics
        additional_metrics = self._compute_compositional_metrics(
            all_predictions, all_targets
        )
        metrics.update(additional_metrics)

        return metrics

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components using BEACONLoss."""
        # Build targets dict for loss function
        targets = {}

        if 'profile' in batch:
            targets['profile'] = batch['profile']

        # Add binding sites pseudo-labels for slot supervision
        if 'binding_sites' in batch:
            targets['binding_sites'] = batch['binding_sites']
            # Also extract individual components for legacy losses
            binding_sites = batch['binding_sites']
            targets['positions'] = binding_sites[:, :, 0]  # positions (normalized 0-1)
            targets['tf_ids'] = binding_sites[:, :, 1].long()  # TF indices
            targets['occupancy'] = binding_sites[:, :, 2]  # occupancy/strength

        if 'tf_index' in batch:
            targets['tf_index'] = batch['tf_index']

        # Use the full loss function
        losses = self.loss_fn(outputs, targets)
        # Ensure total is a tensor
        if not torch.is_tensor(losses.get('total')):
            losses['total'] = torch.tensor(0.0, device=self.device)

        return losses

    def _update_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ):
        """Update metrics tracker with batch results."""
        # Build targets dict from batch
        targets = {}

        if 'profile' in batch:
            targets['profile'] = batch['profile']

        if 'binding_sites' in batch:
            # Extract components from binding sites [B, K, 3]
            binding_sites = batch['binding_sites']
            targets['positions'] = binding_sites[:, :, 0]  # positions
            targets['tf_ids'] = binding_sites[:, :, 1].long()  # TF indices
            targets['occupancy'] = binding_sites[:, :, 2]  # occupancy

        if 'tf_index' in batch:
            targets['tf_index'] = batch['tf_index']

        # Update metrics with outputs and targets dicts
        try:
            self.metrics.update(outputs, targets)
        except Exception:
            # If update fails (e.g., missing keys), skip this batch
            pass

    def _compute_compositional_metrics(
        self,
        predictions: list,
        targets: list,
    ) -> Dict[str, float]:
        """
        Compute additional compositional metrics for final evaluation.

        These are computed over the full test set and include:
        - Binding grammar metrics
        - Motif co-occurrence analysis
        - Spatial relationship accuracy
        """
        metrics = {}

        # Aggregate predictions
        all_pred_positions = []
        all_pred_tf = []
        all_pred_occupancy = []

        for pred in predictions:
            if 'positions' in pred:
                all_pred_positions.append(pred['positions'])
            if 'tf_logits' in pred:
                all_pred_tf.append(pred['tf_logits'].argmax(dim=-1))
            if 'occupancy' in pred:
                all_pred_occupancy.append(pred['occupancy'])

        if all_pred_positions:
            all_positions = torch.cat(all_pred_positions, dim=0)

            # Compute position statistics
            metrics['position_mean'] = all_positions.mean().item()
            metrics['position_std'] = all_positions.std().item()

            # Compute inter-site distance statistics
            if all_positions.dim() >= 2 and all_positions.size(1) > 1:
                # Sort positions for each sample
                sorted_pos, _ = all_positions.sort(dim=1)
                # Compute distances between consecutive sites
                distances = sorted_pos[:, 1:] - sorted_pos[:, :-1]
                distances = distances[distances > 0]  # Filter valid distances

                if len(distances) > 0:
                    metrics['avg_inter_site_distance'] = distances.mean().item()
                    metrics['min_inter_site_distance'] = distances.min().item()
                    metrics['max_inter_site_distance'] = distances.max().item()

        if all_pred_tf:
            all_tf = torch.cat(all_pred_tf, dim=0)

            # TF diversity per sample
            unique_counts = []
            for sample_tf in all_tf:
                unique_counts.append(len(sample_tf.unique()))

            metrics['avg_unique_tfs_per_sample'] = sum(unique_counts) / len(unique_counts)

            # Global TF distribution entropy
            tf_counts = torch.bincount(all_tf.flatten())
            tf_probs = tf_counts.float() / tf_counts.sum()
            tf_probs = tf_probs[tf_probs > 0]
            metrics['tf_distribution_entropy'] = -(tf_probs * tf_probs.log()).sum().item()

        if all_pred_occupancy:
            all_occ = torch.cat(all_pred_occupancy, dim=0)

            metrics['avg_occupancy'] = all_occ.mean().item()
            metrics['occupancy_std'] = all_occ.std().item()
            metrics['high_occupancy_ratio'] = (all_occ > 0.5).float().mean().item()

        return metrics

    def _clip_gradients(self) -> float:
        """Clip gradients and return the total norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip,
        )
        return total_norm.item() if torch.is_tensor(total_norm) else total_norm

    def _compute_gradient_metrics(self) -> Dict[str, float]:
        """Compute detailed gradient statistics."""
        metrics = {}

        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        num_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                grad_abs = param.grad.data.abs()
                max_grad = max(max_grad, grad_abs.max().item())
                min_grad = min(min_grad, grad_abs.min().item())
                num_params += 1

        if num_params > 0:
            metrics['grad_norm_l2'] = total_norm ** 0.5
            metrics['grad_max'] = max_grad
            metrics['grad_min'] = min_grad if min_grad != float('inf') else 0.0

        return metrics

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / self.logger.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save base model state dict (unwrap DataParallel if needed)
        model_state = self.base_model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
        }

        # Save periodic checkpoint
        path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.log_checkpoint(path)

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.log_checkpoint(best_path, is_best=True)

    def _load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load into base model (handles DataParallel wrapping)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint['best_val_metric']

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.log_message(f"Loaded checkpoint from {path}")
