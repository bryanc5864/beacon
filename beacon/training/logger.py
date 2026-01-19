"""
Comprehensive logging for BEACON training.

Logs:
- Training progress (epoch, batch, iteration)
- Loss components
- Gradient statistics
- Learning rate
- Validation metrics
- Model statistics
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class BEACONLogger:
    """
    Comprehensive logger for BEACON training.

    Features:
    - Console logging with progress
    - File logging (JSON lines format)
    - TensorBoard integration (optional)
    - Metric tracking and history
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str = None,
        use_tensorboard: bool = True,
        log_level: int = logging.INFO,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("beacon_%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # Setup logging directory
        self.log_dir = self.output_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup console and file logging
        self._setup_logging(log_level)

        # Setup TensorBoard
        self.writer = None
        if use_tensorboard and HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        # Metric history
        self.history = {
            'train': [],
            'val': [],
            'test': [],
        }

        # Current state
        self.current_epoch = 0
        self.global_step = 0

        self.logger.info(f"Logging to: {self.log_dir}")

    def _setup_logging(self, log_level: int):
        """Setup Python logging."""
        self.logger = logging.getLogger(f"BEACON.{self.experiment_name}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_fmt)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setLevel(log_level)
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_fmt)
        self.logger.addHandler(file_handler)

        # JSON log file
        self.json_log_path = self.log_dir / "metrics.jsonl"

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Config saved to {config_path}")

    def log_model_summary(self, model: torch.nn.Module):
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("=" * 60)
        self.logger.info("Model Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log per-module parameters
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            self.logger.info(f"  {name}: {params:,}")

        self.logger.info("=" * 60)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.current_epoch = epoch
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info("=" * 60)

    def log_batch(
        self,
        batch_idx: int,
        total_batches: int,
        losses: Dict[str, float],
        lr: float,
        grad_metrics: Optional[Dict[str, float]] = None,
        batch_time: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
    ):
        """
        Log batch-level metrics.

        Args:
            batch_idx: Current batch index
            total_batches: Total batches in epoch
            losses: Dictionary of loss values
            lr: Current learning rate
            grad_metrics: Gradient statistics
            batch_time: Time for this batch
            samples_per_sec: Throughput
        """
        self.global_step += 1

        # Build log message
        msg_parts = [
            f"Batch {batch_idx + 1}/{total_batches}",
            f"loss={losses.get('total', 0):.4f}",
        ]

        # Add individual losses
        for key, value in losses.items():
            if key != 'total':
                msg_parts.append(f"{key}={value:.4f}")

        msg_parts.append(f"lr={lr:.2e}")

        if grad_metrics:
            msg_parts.append(f"grad_norm={grad_metrics.get('grad_norm_total', 0):.4f}")

        if samples_per_sec:
            msg_parts.append(f"samples/s={samples_per_sec:.1f}")

        self.logger.info(" | ".join(msg_parts))

        # TensorBoard logging
        if self.writer:
            for key, value in losses.items():
                self.writer.add_scalar(f"train/loss_{key}", value, self.global_step)
            self.writer.add_scalar("train/learning_rate", lr, self.global_step)
            if grad_metrics:
                for key, value in grad_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)

        # JSON logging
        log_entry = {
            'type': 'batch',
            'epoch': self.current_epoch,
            'batch': batch_idx,
            'global_step': self.global_step,
            'losses': losses,
            'lr': lr,
            'grad_metrics': grad_metrics,
            'batch_time': batch_time,
            'samples_per_sec': samples_per_sec,
            'timestamp': datetime.now().isoformat(),
        }
        self._write_json(log_entry)

    def log_epoch_end(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_time: Optional[float] = None,
    ):
        """
        Log epoch-end metrics.

        Args:
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics (10+ metrics)
            epoch_time: Total epoch time
        """
        self.logger.info("-" * 60)
        self.logger.info("Epoch Summary")
        self.logger.info("-" * 60)

        # Training metrics
        self.logger.info("Training:")
        for key, value in sorted(train_metrics.items()):
            self.logger.info(f"  {key}: {value:.6f}")

        # Validation metrics (10+ metrics)
        if val_metrics:
            self.logger.info("Validation:")
            for key, value in sorted(val_metrics.items()):
                self.logger.info(f"  {key}: {value:.6f}")

        if epoch_time:
            self.logger.info(f"Epoch time: {epoch_time:.1f}s")

        # Store in history
        self.history['train'].append({
            'epoch': self.current_epoch,
            **train_metrics
        })
        if val_metrics:
            self.history['val'].append({
                'epoch': self.current_epoch,
                **val_metrics
            })

        # TensorBoard logging
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"epoch/train_{key}", value, self.current_epoch)
            if val_metrics:
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"epoch/val_{key}", value, self.current_epoch)

        # JSON logging
        log_entry = {
            'type': 'epoch',
            'epoch': self.current_epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat(),
        }
        self._write_json(log_entry)

    def log_validation(
        self,
        metrics: Dict[str, float],
        mode: str = 'val',
    ):
        """
        Log validation metrics.

        Args:
            metrics: Validation metrics dictionary
            mode: 'val' for validation, 'test' for final test
        """
        n_metrics = len([k for k in metrics.keys() if not k.startswith('n_')])
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"{'Validation' if mode == 'val' else 'Final Test'} Results ({n_metrics} metrics)")
        self.logger.info("=" * 60)

        # Group metrics by category
        categories = {
            'Profile': ['profile_'],
            'Site Detection': ['site_', 'position_'],
            'TF Classification': ['tf_'],
            'Slot Utilization': ['slot_', 'avg_', 'occupancy_'],
            'Compositional': ['composition_', 'binding_', 'motif_', 'grammar_'],
        }

        for category, prefixes in categories.items():
            category_metrics = {
                k: v for k, v in metrics.items()
                if any(k.startswith(p) for p in prefixes)
            }
            if category_metrics:
                self.logger.info(f"\n{category} Metrics:")
                for key, value in sorted(category_metrics.items()):
                    self.logger.info(f"  {key}: {value:.6f}")

        # Other metrics
        other_metrics = {
            k: v for k, v in metrics.items()
            if not any(k.startswith(p) for cat_prefixes in categories.values() for p in cat_prefixes)
            and not k.startswith('n_')
        }
        if other_metrics:
            self.logger.info("\nOther Metrics:")
            for key, value in sorted(other_metrics.items()):
                self.logger.info(f"  {key}: {value:.6f}")

        self.logger.info("")
        self.logger.info(f"Total samples: {metrics.get('n_samples', 'N/A')}")
        self.logger.info("=" * 60)

        # Store in history
        self.history[mode].append({
            'epoch': self.current_epoch,
            **metrics
        })

        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if not key.startswith('n_'):
                    self.writer.add_scalar(f"{mode}/{key}", value, self.current_epoch)

        # JSON logging
        log_entry = {
            'type': f'{mode}_metrics',
            'epoch': self.current_epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        self._write_json(log_entry)

    def log_checkpoint(self, path: Path, is_best: bool = False):
        """Log checkpoint save."""
        msg = f"Checkpoint saved: {path}"
        if is_best:
            msg += " (best model)"
        self.logger.info(msg)

    def log_message(self, msg: str, level: str = 'info'):
        """Log a general message."""
        getattr(self.logger, level)(msg)

    def _write_json(self, entry: Dict):
        """Write entry to JSON log file."""
        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def save_history(self):
        """Save full training history."""
        history_path = self.log_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"History saved to {history_path}")

    def close(self):
        """Close logger and save history."""
        self.save_history()
        if self.writer:
            self.writer.close()
        self.logger.info("Training complete. Logs saved.")
