#!/usr/bin/env python3
"""
Training script for BEACON.

Uses comprehensive BEACONTrainer with:
- Batch-level logging (batch number, epoch, gradients, loss components, learning rate)
- Per-epoch validation with 10+ metrics
- End-of-training evaluation with 20+ metrics
- TensorBoard integration
- Checkpoint management with early stopping
- Multi-GPU support with DataParallel

Trains on ENCODE ChIP-seq data with profile reconstruction
and slot-based binding site prediction.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Set default GPUs to 7, 8, 9
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7,8,9")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON, BEACONSmall, BEACONBase, BEACONLarge, BEACONLoss
from beacon.data.dataset import BEACONDataset
from beacon.training import BEACONTrainer, BEACONLogger, BEACONMetrics


def main():
    parser = argparse.ArgumentParser(
        description="Train BEACON model with comprehensive logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                           # Train with defaults
  python scripts/train.py --model-size base         # Train larger model
  python scripts/train.py --epochs 100 --batch-size 64  # Custom training
        """
    )

    # Data arguments
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "processed" / "chipseq",
                        help="Directory containing processed ChIP-seq data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "outputs",
                        help="Directory for outputs (checkpoints, logs, etc.)")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for this experiment (default: auto-generated)")

    # Model arguments
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Model size variant")
    parser.add_argument("--seq-len", type=int, default=1000,
                        help="Input sequence length")
    parser.add_argument("--n-slots", type=int, default=16,
                        help="Number of binding site slots")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping threshold")

    # Logging arguments
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches")
    parser.add_argument("--val-interval", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--early-stopping", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")

    # Hardware arguments
    parser.add_argument("--gpus", type=str, default="7,8,9",
                        help="GPU IDs to use (comma-separated, e.g., '7,8,9')")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    args = parser.parse_args()

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    use_multi_gpu = n_gpus > 1 and torch.cuda.is_available()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # After setting CUDA_VISIBLE_DEVICES, GPUs are renumbered 0, 1, 2, ...
        available_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        available_gpus = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BEACON Training with Comprehensive Logging")
    print("=" * 70)
    print(f"GPUs requested: {args.gpus} ({n_gpus} GPUs)")
    print(f"GPUs available: {available_gpus}")
    print(f"Multi-GPU training: {use_multi_gpu}")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model size: {args.model_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of slots: {args.n_slots}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping patience: {args.early_stopping} epochs")
    print()

    # Load data
    print("Loading data...")
    train_path = args.data_dir / "train.h5"
    val_path = args.data_dir / "val.h5"
    test_path = args.data_dir / "test.h5"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Please run: python scripts/prepare_data.py --all")
        return 1

    # Create datasets with peak extraction for binding site supervision
    train_dataset = BEACONDataset(
        train_path,
        seq_length=args.seq_len,
        augment=True,
        n_slots=args.n_slots,
        extract_peaks=True,  # Generate binding site pseudo-labels from profiles
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_path.exists():
        val_dataset = BEACONDataset(
            val_path,
            seq_length=args.seq_len,
            augment=False,
            n_slots=args.n_slots,
            extract_peaks=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    test_loader = None
    if test_path.exists():
        test_dataset = BEACONDataset(
            test_path,
            seq_length=args.seq_len,
            augment=False,
            n_slots=args.n_slots,
            extract_peaks=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print(f"  Train samples: {len(train_dataset)}")
    if val_loader:
        print(f"  Val samples: {len(val_dataset)}")
    if test_loader:
        print(f"  Test samples: {len(test_dataset)}")

    # Load TF vocabulary from summary
    summary_path = args.data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        n_tfs = summary.get("n_tfs", 2)
        tf_names = summary.get("tfs", [])
    else:
        n_tfs = 2
        tf_names = []

    print(f"  Number of TFs: {n_tfs}")
    if tf_names:
        print(f"  TFs: {', '.join(tf_names)}")
    print()

    # Create model
    print("Creating model...")
    model_classes = {
        "small": BEACONSmall,
        "base": BEACONBase,
        "large": BEACONLarge,
    }

    ModelClass = model_classes[args.model_size]
    model = ModelClass(
        seq_len=args.seq_len,
        n_tfs=n_tfs,
        n_slots=args.n_slots,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # Wrap model in DataParallel for multi-GPU training
    if use_multi_gpu:
        print(f"  Using DataParallel on {available_gpus} GPUs")
        model = nn.DataParallel(model)
    print()

    # Create loss function with all components for improved training
    loss_fn = BEACONLoss(
        n_tfs=n_tfs,
        seq_len=args.seq_len,
        profile_weight=1.0,           # Profile reconstruction (main objective)
        position_weight=0.5,          # Binding position prediction
        tf_weight=0.5,                # TF identity prediction
        occupancy_weight=0.3,         # Slot occupancy prediction
        diversity_weight=0.2,         # Slot attention diversity (prevent collapse)
        orthogonality_weight=0.1,     # Slot embedding orthogonality
        site_supervision_weight=0.5,  # Binding site pseudo-label supervision
    )

    # Create trainer with comprehensive logging
    print("Initializing trainer with comprehensive logging...")
    print("  - Batch-level logging: loss components, gradients, learning rate")
    print("  - Per-epoch validation: 10+ metrics")
    print("  - Final evaluation: 20+ metrics")
    print("  - TensorBoard: " + ("enabled" if not args.no_tensorboard else "disabled"))
    print()

    trainer = BEACONTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        device=str(device),
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        early_stopping_patience=args.early_stopping,
        use_tensorboard=not args.no_tensorboard,
        num_slots=args.n_slots,
        num_tfs=n_tfs,
    )

    # Run training
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    results = trainer.train()

    # Print final results
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Total time: {results['total_time']/3600:.2f} hours")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Best validation metric: {results['best_val_metric']:.6f}")

    if results['final_test_metrics']:
        print()
        print("Final Test Results (20+ metrics):")
        print("-" * 50)
        for key, value in sorted(results['final_test_metrics'].items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    print()
    print(f"Outputs saved to: {args.output_dir}")
    print("  - checkpoints/: Model checkpoints")
    print("  - tensorboard/: TensorBoard logs (run: tensorboard --logdir outputs)")
    print("  - metrics.jsonl: Detailed metrics log")
    print("  - training.log: Full training log")
    print("  - history.json: Training history")

    return 0


if __name__ == "__main__":
    sys.exit(main())
