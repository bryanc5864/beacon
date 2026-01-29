#!/usr/bin/env python3
"""
Train BEACON on real CTCF K562 ChIP-seq data.

Phase 2A: Single-TF real data validation.

This script:
1. Loads processed CTCF data (2kb windows, hg38, ChIP-seq profiles)
2. Trains BEACON with comprehensive logging
3. Evaluates binding site detection, profile reconstruction, and position accuracy

Target metrics:
- Site F1 > 0.70 (can BEACON find real binding sites?)
- Position MAE < 25 bp (precise enough for motif analysis?)
- Profile Pearson > 0.50 (reasonable reconstruction?)

Usage:
    python scripts/train_ctcf.py
    python scripts/train_ctcf.py --gpus 2,4,6 --epochs 150
    python scripts/train_ctcf.py --seq-len 2000 --batch-size 32
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Default GPUs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7,8,9")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON, BEACONSmall, BEACONLoss
from beacon.data.dataset import BEACONDataset
from beacon.training import BEACONTrainer, BEACONLogger, BEACONMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ctcf_k562"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ctcf_k562"


def main():
    parser = argparse.ArgumentParser(
        description="Train BEACON on CTCF K562 ChIP-seq data (Phase 2A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target metrics (Phase 2A):
  Site F1        > 0.70  (binding site detection)
  Position MAE   < 25 bp (localization accuracy)
  Profile Pearson > 0.50 (profile reconstruction)
        """
    )

    # Data
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name (default: ctcf_k562_YYYYMMDD_HHMMSS)")

    # Model
    parser.add_argument("--seq-len", type=int, default=2000,
                        help="Input sequence length (should match data prep)")
    parser.add_argument("--n-slots", type=int, default=16,
                        help="Number of binding site slots")
    parser.add_argument("--backbone-dim", type=int, default=128,
                        help="Backbone hidden dimension")
    parser.add_argument("--backbone-layers", type=int, default=4,
                        help="Number of backbone layers")
    parser.add_argument("--slot-dim", type=int, default=128,
                        help="Slot attention dimension")
    parser.add_argument("--n-iterations", type=int, default=3,
                        help="Slot attention iterations")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # Loss weights
    parser.add_argument("--profile-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument("--tf-weight", type=float, default=0.1,
                        help="Low weight since only 1 TF")
    parser.add_argument("--occupancy-weight", type=float, default=0.3)
    parser.add_argument("--diversity-weight", type=float, default=0.2)
    parser.add_argument("--orthogonality-weight", type=float, default=0.1)
    parser.add_argument("--site-supervision-weight", type=float, default=0.5)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--no-tensorboard", action="store_true")

    # Hardware
    parser.add_argument("--gpus", type=str, default="7,8,9")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        available_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        available_gpus = 0

    use_multi_gpu = n_gpus > 1 and available_gpus > 1

    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"ctcf_k562_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dir = args.output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    file_handler = logging.FileHandler(experiment_dir / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Print config
    logger.info("=" * 70)
    logger.info("BEACON Phase 2A: CTCF K562 Real Data Training")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Device: {device} ({available_gpus} GPUs available, using {n_gpus})")
    logger.info(f"Multi-GPU: {use_multi_gpu}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {experiment_dir}")
    logger.info("")
    logger.info("Model config:")
    logger.info(f"  Sequence length: {args.seq_len}")
    logger.info(f"  Backbone: {args.backbone_dim}d x {args.backbone_layers} layers")
    logger.info(f"  Slots: {args.n_slots} x {args.slot_dim}d x {args.n_iterations} iters")
    logger.info("")
    logger.info("Training config:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Grad clip: {args.grad_clip}")
    logger.info(f"  Early stopping: {args.early_stopping} epochs")
    logger.info(f"  AMP: {not args.no_amp}")
    logger.info("")
    logger.info("Loss weights:")
    logger.info(f"  Profile: {args.profile_weight}")
    logger.info(f"  Position: {args.position_weight}")
    logger.info(f"  TF identity: {args.tf_weight}")
    logger.info(f"  Occupancy: {args.occupancy_weight}")
    logger.info(f"  Diversity: {args.diversity_weight}")
    logger.info(f"  Orthogonality: {args.orthogonality_weight}")
    logger.info(f"  Site supervision: {args.site_supervision_weight}")
    logger.info("")

    # Save config
    config = vars(args)
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, Path):
            config_serializable[k] = str(v)
        else:
            config_serializable[k] = v

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config_serializable, f, indent=2)

    # ================================================================
    # Load data
    # ================================================================
    logger.info("Loading data...")

    train_path = args.data_dir / "train.h5"
    val_path = args.data_dir / "val.h5"
    test_path = args.data_dir / "test.h5"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run: python scripts/prepare_ctcf_data.py first")
        return 1

    # Load summary
    summary_path = args.data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data_summary = json.load(f)
        n_tfs = data_summary.get("n_tfs", 1)
        tf_names = data_summary.get("tfs", ["CTCF"])
        logger.info(f"  Data summary: {n_tfs} TF(s): {tf_names}")
        logger.info(f"  Train: {data_summary.get('n_train', '?')} samples")
        logger.info(f"  Val: {data_summary.get('n_val', '?')} samples")
        logger.info(f"  Test: {data_summary.get('n_test', '?')} samples")
    else:
        n_tfs = 1
        tf_names = ["CTCF"]

    # Create datasets
    train_dataset = BEACONDataset(
        train_path,
        seq_length=args.seq_len,
        augment=True,
        n_slots=args.n_slots,
        extract_peaks=True,  # Extract peaks from profiles for supervision
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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

    logger.info(f"  Train dataset: {len(train_dataset)} samples (with augmentation)")
    if val_loader:
        logger.info(f"  Val dataset: {len(val_dataset)} samples")
    if test_loader:
        logger.info(f"  Test dataset: {len(test_dataset)} samples")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info("")

    # ================================================================
    # Create model
    # ================================================================
    logger.info("Creating BEACON model...")
    model = BEACON(
        seq_len=args.seq_len,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=args.backbone_dim,
        backbone_layers=args.backbone_layers,
        n_slots=args.n_slots,
        slot_dim=args.slot_dim,
        n_iterations=args.n_iterations,
        n_tfs=n_tfs,
        position_mode="gaussian",
        dropout=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {n_params:,}")
    logger.info(f"  Trainable: {n_trainable:,}")
    logger.info(f"  Model size: {n_params * 4 / 1e6:.1f} MB (fp32)")

    if use_multi_gpu:
        logger.info(f"  Wrapping in DataParallel ({available_gpus} GPUs)")
        model = nn.DataParallel(model)

    logger.info("")

    # ================================================================
    # Create loss
    # ================================================================
    loss_fn = BEACONLoss(
        n_tfs=n_tfs,
        seq_len=args.seq_len,
        profile_weight=args.profile_weight,
        position_weight=args.position_weight,
        tf_weight=args.tf_weight,
        occupancy_weight=args.occupancy_weight,
        diversity_weight=args.diversity_weight,
        orthogonality_weight=args.orthogonality_weight,
        site_supervision_weight=args.site_supervision_weight,
    )

    logger.info("Loss function: BEACONLoss")
    logger.info(f"  Components: profile, position, TF identity, occupancy, diversity, orthogonality, site supervision")
    logger.info("")

    # ================================================================
    # Train
    # ================================================================
    logger.info("Initializing trainer...")
    trainer = BEACONTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        output_dir=experiment_dir,
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

    logger.info("")
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    logger.info("")

    results = trainer.train()

    # ================================================================
    # Report results
    # ================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total epochs: {results['total_epochs']}")
    logger.info(f"Total time: {results['total_time']/3600:.2f} hours")
    logger.info(f"Best val loss: {results['best_val_loss']:.6f}")
    logger.info(f"Best val metric: {results['best_val_metric']:.6f}")

    if results.get("final_test_metrics"):
        logger.info("")
        logger.info("=" * 70)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 70)

        test_metrics = results["final_test_metrics"]
        for key in sorted(test_metrics.keys()):
            val = test_metrics[key]
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.6f}")
            else:
                logger.info(f"  {key}: {val}")

        # Check against Phase 2A targets
        logger.info("")
        logger.info("Phase 2A Target Check:")
        logger.info("-" * 50)

        site_f1 = test_metrics.get("site_f1", test_metrics.get("site_detection/f1", 0))
        profile_r = test_metrics.get("profile_pearson", test_metrics.get("profile/pearson_r", 0))

        logger.info(f"  Site F1:        {site_f1:.4f}  (target: > 0.70)")
        logger.info(f"  Profile Pearson: {profile_r:.4f}  (target: > 0.50)")

        if site_f1 > 0.70:
            logger.info("  ✓ Site F1 target MET")
        else:
            logger.info("  ✗ Site F1 target NOT MET")

        if profile_r > 0.50:
            logger.info("  ✓ Profile Pearson target MET")
        else:
            logger.info("  ✗ Profile Pearson target NOT MET")

    # Save results
    results_path = experiment_dir / "final_results.json"
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)):
            results_serializable[k] = v
        elif isinstance(v, dict):
            results_serializable[k] = {
                sk: sv for sk, sv in v.items()
                if isinstance(sv, (int, float, str, bool))
            }
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"\nAll outputs saved to: {experiment_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
