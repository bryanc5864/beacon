#!/usr/bin/env python3
"""
Train BEACON on multi-TF K562 ChIP-seq data.

Phase 2B: Multi-TF discrimination on real data.

Tests whether BEACON can:
1. Reconstruct profiles for multiple TFs
2. Distinguish TF identity from sequence alone
3. Handle same-family TFs (MYC vs MAX, TAL1 vs MYC)

Usage:
    python scripts/train_multi_tf.py
    python scripts/train_multi_tf.py --gpus 2,4,6 --epochs 150
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7,8,9")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON, BEACONLoss
from beacon.data.dataset import BEACONDataset
from beacon.training import BEACONTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "multi_tf_k562"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multi_tf_k562"


def main():
    parser = argparse.ArgumentParser(
        description="Train BEACON on multi-TF K562 data (Phase 2B)",
    )

    # Data
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", type=str, default=None)

    # Model
    parser.add_argument("--seq-len", type=int, default=2000)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--backbone-dim", type=int, default=128)
    parser.add_argument("--backbone-layers", type=int, default=4)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--n-iterations", type=int, default=3)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Loss weights — TF weight higher for multi-TF
    parser.add_argument("--profile-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument("--tf-weight", type=float, default=1.0,
                        help="Higher than Phase 2A since TF discrimination is key")
    parser.add_argument("--occupancy-weight", type=float, default=0.3)
    parser.add_argument("--diversity-weight", type=float, default=0.2)
    parser.add_argument("--orthogonality-weight", type=float, default=0.1)
    parser.add_argument("--site-supervision-weight", type=float, default=0.5)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--no-tensorboard", action="store_true")

    # Hardware
    parser.add_argument("--gpus", type=str, default="7,8,9")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

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

    if args.experiment_name is None:
        args.experiment_name = f"multi_tf_k562_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dir = args.output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(experiment_dir / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Load data summary
    summary_path = args.data_dir / "summary.json"
    if not summary_path.exists():
        logger.error(f"Summary not found: {summary_path}")
        logger.error("Run: python scripts/prepare_multi_tf_data.py first")
        return 1

    with open(summary_path) as f:
        data_summary = json.load(f)

    n_tfs = data_summary["n_tfs"]
    tf_names = data_summary["tfs"]
    tf_families = data_summary.get("tf_families", {})

    logger.info("=" * 70)
    logger.info("BEACON Phase 2B: Multi-TF K562 Training")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Device: {device} ({available_gpus} GPUs)")
    logger.info(f"")
    logger.info(f"TFs ({n_tfs}):")
    for i, tf in enumerate(tf_names):
        family = tf_families.get(tf, "?")
        logger.info(f"  {i}: {tf} ({family})")
    logger.info(f"")
    logger.info(f"Data: {data_summary['n_train']} train, {data_summary['n_val']} val, {data_summary['n_test']} test")
    logger.info(f"Balanced: {data_summary.get('balanced', False)}")
    logger.info(f"")
    logger.info(f"Model: {args.backbone_dim}d backbone, {args.n_slots} slots, {args.slot_dim}d slots")
    logger.info(f"Training: bs={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    logger.info(f"TF weight: {args.tf_weight} (key metric for Phase 2B)")
    logger.info("")

    # Save config
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["n_tfs"] = n_tfs
    config["tf_names"] = tf_names
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    train_path = args.data_dir / "train.h5"
    val_path = args.data_dir / "val.h5"
    test_path = args.data_dir / "test.h5"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return 1

    train_dataset = BEACONDataset(train_path, seq_length=args.seq_len, augment=True,
                                   n_slots=args.n_slots, extract_peaks=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if val_path.exists():
        val_dataset = BEACONDataset(val_path, seq_length=args.seq_len, augment=False,
                                     n_slots=args.n_slots, extract_peaks=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    test_loader = None
    if test_path.exists():
        test_dataset = BEACONDataset(test_path, seq_length=args.seq_len, augment=False,
                                      n_slots=args.n_slots, extract_peaks=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    if val_loader:
        logger.info(f"Val: {len(val_dataset)} samples")
    if test_loader:
        logger.info(f"Test: {len(test_dataset)} samples")
    logger.info("")

    # Create model
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
    logger.info(f"Model: {n_params:,} parameters")

    if use_multi_gpu:
        model = nn.DataParallel(model)
        logger.info(f"DataParallel on {available_gpus} GPUs")

    # Loss
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

    # Train
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

    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    results = trainer.train()

    # Results
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Epochs: {results['total_epochs']}")
    logger.info(f"Time: {results['total_time']/3600:.2f} hours")
    logger.info(f"Best val loss: {results['best_val_loss']:.6f}")

    if results.get("final_test_metrics"):
        test_metrics = results["final_test_metrics"]
        logger.info("")
        logger.info("FINAL TEST RESULTS")
        logger.info("-" * 50)
        for key in sorted(test_metrics.keys()):
            val = test_metrics[key]
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.6f}")
            else:
                logger.info(f"  {key}: {val}")

        # Phase 2B targets
        logger.info("")
        logger.info("Phase 2B Key Questions:")
        profile_r = test_metrics.get("profile_pearson", test_metrics.get("profile/pearson_r", 0))
        tf_acc = test_metrics.get("tf_accuracy", test_metrics.get("tf/accuracy", 0))
        site_f1 = test_metrics.get("site_f1", test_metrics.get("site_detection/f1", 0))

        logger.info(f"  Profile Pearson: {profile_r:.4f}")
        logger.info(f"  TF Accuracy:     {tf_acc:.4f} (chance = {1/n_tfs:.4f})")
        logger.info(f"  Site F1:         {site_f1:.4f}")

        if tf_acc > 1/n_tfs:
            logger.info(f"  TF discrimination: ABOVE CHANCE")
        else:
            logger.info(f"  TF discrimination: AT OR BELOW CHANCE")

    # Save results
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)):
            results_serializable[k] = v
        elif isinstance(v, dict):
            results_serializable[k] = {sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str, bool))}
    with open(experiment_dir / "final_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"\nOutputs: {experiment_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
