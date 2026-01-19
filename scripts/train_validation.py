#!/usr/bin/env python3
"""
Training script for BEACON validation pipeline.

Supports multiple data configurations:
- Synthetic data (easy, medium, hard, extreme)
- ENCODE real data

Includes comprehensive logging and all 35+ metrics.
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


# Configuration presets for different data complexities
CONFIGS = {
    "synthetic-easy": {
        "description": "Easy synthetic: 1-2 sites, 10 TFs, no overlap",
        "data_dir": "data/processed/synthetic",
        "seq_len": 1000,
        "n_slots": 16,
        "batch_size": 32,
        "epochs": 100,
        "lr": 1e-4,
    },
    "synthetic-hard": {
        "description": "Hard synthetic (Stage A): 3-8 sites, 50 TFs, overlap, noise",
        "data_dir": "data/processed/synthetic",
        "seq_len": 2000,
        "n_slots": 32,  # More slots for more sites
        "batch_size": 16,  # Smaller batch for longer sequences
        "epochs": 150,
        "lr": 5e-5,
    },
    "synthetic-extreme": {
        "description": "Extreme synthetic: 5-12 sites, 100 TFs, high overlap",
        "data_dir": "data/processed/synthetic",
        "seq_len": 2000,
        "n_slots": 64,  # Even more slots
        "batch_size": 8,
        "epochs": 200,
        "lr": 3e-5,
    },
    "encode": {
        "description": "ENCODE real data (Stage B)",
        "data_dir": "data/processed/encode",
        "seq_len": 2000,
        "n_slots": 32,
        "batch_size": 16,
        "epochs": 200,
        "lr": 5e-5,
    },
    "encode-multi": {
        "description": "ENCODE multi-TF (Stage C)",
        "data_dir": "data/processed/encode",
        "seq_len": 2000,
        "n_slots": 64,
        "batch_size": 8,
        "epochs": 250,
        "lr": 3e-5,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Train BEACON for validation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configurations:
  synthetic-easy    : Easy synthetic data (current baseline)
  synthetic-hard    : Hard synthetic data (Stage A validation)
  synthetic-extreme : Extreme synthetic data (stress test)
  encode            : ENCODE real data (Stage B validation)
  encode-multi      : ENCODE multi-TF data (Stage C validation)

Examples:
  python scripts/train_validation.py --config synthetic-hard
  python scripts/train_validation.py --config encode --gpus 7,8,9
  python scripts/train_validation.py --data-dir custom/path --seq-len 2000
        """
    )

    # Configuration
    parser.add_argument("--config", type=str,
                        choices=list(CONFIGS.keys()),
                        help="Use preset configuration")

    # Data arguments
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing processed data (overrides config)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "outputs",
                        help="Directory for outputs")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for this experiment")

    # Model arguments
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Model size variant")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Input sequence length (overrides config)")
    parser.add_argument("--n-slots", type=int, default=None,
                        help="Number of binding site slots (overrides config)")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping threshold")

    # Loss weights
    parser.add_argument("--profile-weight", type=float, default=1.0,
                        help="Weight for profile reconstruction loss")
    parser.add_argument("--position-weight", type=float, default=0.5,
                        help="Weight for position prediction loss")
    parser.add_argument("--tf-weight", type=float, default=0.5,
                        help="Weight for TF classification loss")
    parser.add_argument("--diversity-weight", type=float, default=0.2,
                        help="Weight for slot diversity loss")
    parser.add_argument("--site-supervision-weight", type=float, default=0.5,
                        help="Weight for binding site supervision loss")

    # Logging arguments
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches")
    parser.add_argument("--val-interval", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--early-stopping", type=int, default=20,
                        help="Early stopping patience (epochs)")

    # Hardware arguments
    parser.add_argument("--gpus", type=str, default="7,8,9",
                        help="GPU IDs to use")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    # Resume training
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Apply config preset
    if args.config:
        cfg = CONFIGS[args.config]
        if args.data_dir is None:
            args.data_dir = Path(__file__).parent.parent / cfg["data_dir"]
        if args.seq_len is None:
            args.seq_len = cfg["seq_len"]
        if args.n_slots is None:
            args.n_slots = cfg["n_slots"]
        if args.batch_size is None:
            args.batch_size = cfg["batch_size"]
        if args.epochs is None:
            args.epochs = cfg["epochs"]
        if args.lr is None:
            args.lr = cfg["lr"]

    # Apply defaults if not set
    if args.data_dir is None:
        args.data_dir = Path(__file__).parent.parent / "data" / "processed" / "synthetic"
    if args.seq_len is None:
        args.seq_len = 1000
    if args.n_slots is None:
        args.n_slots = 16
    if args.batch_size is None:
        args.batch_size = 32
    if args.epochs is None:
        args.epochs = 100
    if args.lr is None:
        args.lr = 1e-4

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        available_gpus = torch.cuda.device_count()
        use_multi_gpu = available_gpus > 1
    else:
        device = torch.device("cpu")
        available_gpus = 0
        use_multi_gpu = False

    # Print configuration
    print("=" * 70)
    print("BEACON Validation Training")
    print("=" * 70)
    if args.config:
        print(f"Configuration: {args.config}")
        print(f"  {CONFIGS[args.config]['description']}")
    print()
    print("Settings:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Number of slots: {args.n_slots}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model size: {args.model_size}")
    print()
    print("Hardware:")
    print(f"  GPUs requested: {args.gpus}")
    print(f"  GPUs available: {available_gpus}")
    print(f"  Multi-GPU: {use_multi_gpu}")
    print(f"  Device: {device}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data paths
    train_path = args.data_dir / "train.h5"
    val_path = args.data_dir / "val.h5"
    test_path = args.data_dir / "test.h5"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("\nTo generate synthetic data:")
        print(f"  python scripts/generate_synthetic_data.py --config hard")
        print("\nTo process ENCODE data:")
        print(f"  python scripts/download_encode.py")
        print(f"  python scripts/process_encode.py")
        return 1

    # Load summary FIRST to get data configuration
    summary_path = args.data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        n_tfs = summary.get("n_tfs", 2)
        tf_names = summary.get("tfs", [])

        # IMPORTANT: Override n_slots and seq_len from data if available
        # This ensures model matches data
        data_n_slots = summary.get("n_slots")
        data_seq_len = summary.get("seq_length")

        if data_n_slots is not None:
            if args.n_slots != data_n_slots:
                print(f"  NOTE: Overriding n_slots from {args.n_slots} to {data_n_slots} (from data)")
            args.n_slots = data_n_slots

        if data_seq_len is not None:
            if args.seq_len != data_seq_len:
                print(f"  NOTE: Overriding seq_len from {args.seq_len} to {data_seq_len} (from data)")
            args.seq_len = data_seq_len
    else:
        n_tfs = 2
        tf_names = []

    print(f"  Number of TFs: {n_tfs}")
    print(f"  Number of slots: {args.n_slots}")
    print(f"  Sequence length: {args.seq_len}")
    if tf_names and len(tf_names) <= 10:
        print(f"  TFs: {', '.join(tf_names)}")
    elif tf_names:
        print(f"  TFs: {', '.join(tf_names[:5])}... ({len(tf_names)} total)")
    print()

    # NOW create datasets with correct n_slots
    print("Loading datasets...")
    train_dataset = BEACONDataset(
        train_path,
        seq_length=args.seq_len,
        augment=True,
        n_slots=args.n_slots,
        extract_peaks=True,
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

    # Multi-GPU
    if use_multi_gpu:
        print(f"  Using DataParallel on {available_gpus} GPUs")
        model = nn.DataParallel(model)
    print()

    # Create loss function
    loss_fn = BEACONLoss(
        n_tfs=n_tfs,
        seq_len=args.seq_len,
        profile_weight=args.profile_weight,
        position_weight=args.position_weight,
        tf_weight=args.tf_weight,
        occupancy_weight=0.3,
        diversity_weight=args.diversity_weight,
        orthogonality_weight=0.1,
        site_supervision_weight=args.site_supervision_weight,
    )

    # Create trainer
    print("Initializing trainer...")
    print("  - Per-epoch validation: 15+ metrics")
    print("  - Final evaluation: 35+ metrics")
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
        use_tensorboard=True,
        num_slots=args.n_slots,
        num_tfs=n_tfs,
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    results = trainer.train()

    # Print results
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Total time: {results['total_time']/3600:.2f} hours")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")

    if results['final_test_metrics']:
        print()
        print("Final Test Results (35+ metrics):")
        print("-" * 50)

        # Group metrics by category
        categories = {
            'Profile Metrics': ['profile_'],
            'Site Detection': ['site_', 'position_'],
            'TF Classification': ['tf_'],
            'Slot Utilization': ['slot_', 'occupancy_', 'avg_', 'high_', 'inter_'],
            'Compositional': ['composition_', 'binding_', 'motif_', 'grammar_', 'unique_'],
            'Other': [],
        }

        for category, prefixes in categories.items():
            matching = {}
            for key, value in results['final_test_metrics'].items():
                if key in ['n_samples', 'loss']:
                    continue
                if prefixes:
                    if any(key.startswith(p) for p in prefixes):
                        matching[key] = value
                else:
                    # 'Other' category - metrics that don't match any prefix
                    matched = False
                    for _, ps in categories.items():
                        if any(key.startswith(p) for p in ps):
                            matched = True
                            break
                    if not matched:
                        matching[key] = value

            if matching:
                print(f"\n{category}:")
                for key, value in sorted(matching.items()):
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")

    print()
    print(f"Outputs saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
