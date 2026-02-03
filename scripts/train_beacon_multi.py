#!/usr/bin/env python3
"""
Train BEACON-multi: slot attention model on multi-TF binding data.

Uses data prepared by prepare_beacon_multi_data.py which contains sequences
with 2+ TF binding sites per sequence, forcing the model to activate
multiple slots with different TF predictions.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.models.losses import BEACONLoss
from beacon.training.trainer import BEACONTrainer
from beacon.data.dataset import BEACONDataset


def main():
    parser = argparse.ArgumentParser(description="Train BEACON-multi")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/beacon_multi"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/beacon_multi"))

    # Architecture (same as original BEACON)
    parser.add_argument("--seq-len", type=int, default=2000)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--backbone-dim", type=int, default=128)
    parser.add_argument("--backbone-layers", type=int, default=4)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--n-iterations", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)

    # Loss weights — key difference: Hungarian matching + load balancing
    parser.add_argument("--profile-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument("--tf-weight", type=float, default=5.0,
                        help="High weight to force per-slot TF discrimination")
    parser.add_argument("--occupancy-weight", type=float, default=0.5,
                        help="Higher than original (0.3) to encourage multi-slot activation")
    parser.add_argument("--diversity-weight", type=float, default=0.5,
                        help="Higher than original (0.2) to prevent slot collapse")
    parser.add_argument("--orthogonality-weight", type=float, default=0.5,
                        help="Higher than original (0.1) for slot specialization")
    parser.add_argument("--site-supervision-weight", type=float, default=1.0,
                        help="Higher than original (0.5) to match slots to binding sites")
    parser.add_argument("--slot-count-weight", type=float, default=0.5,
                        help="Penalize mismatch between active slots and target sites")
    parser.add_argument("--load-balancing-weight", type=float, default=0.3,
                        help="Penalize attention concentration on single slot")

    # Hardware
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"beacon_multi_{timestamp}"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEACON-multi Training")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Output: {experiment_dir}")

    # Load data summary
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data_summary = json.load(f)
        n_tfs = data_summary['n_tfs']
    else:
        n_tfs = 7

    # Datasets
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"
    test_path = data_dir / "test.h5"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run prepare_beacon_multi_data.py first.")
            return 1

    train_dataset = BEACONDataset(str(train_path), seq_length=args.seq_len,
                                   augment=True, n_slots=args.n_slots,
                                   extract_peaks=True)
    val_dataset = BEACONDataset(str(val_path), seq_length=args.seq_len,
                                 augment=False, n_slots=args.n_slots,
                                 extract_peaks=True)
    test_dataset = BEACONDataset(str(test_path), seq_length=args.seq_len,
                                  augment=False, n_slots=args.n_slots,
                                  extract_peaks=True)

    print(f"\nDatasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Model — independent attention mode for multi-slot activation
    model = BEACON(
        seq_len=args.seq_len, input_channels=4,
        backbone_type="dilated", backbone_dim=args.backbone_dim,
        backbone_layers=args.backbone_layers,
        n_slots=args.n_slots, slot_dim=args.slot_dim,
        n_iterations=args.n_iterations,
        n_tfs=n_tfs, position_mode="gaussian", dropout=0.1,
        attention_mode="independent",
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # Multi-GPU
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model = model.cuda()
        print(f"  Using GPU(s): {gpu_ids}")
    else:
        print("  Using CPU")

    # Loss — tuned for multi-TF data with Hungarian matching
    loss_fn = BEACONLoss(
        n_tfs=n_tfs, seq_len=args.seq_len,
        profile_weight=args.profile_weight,
        position_weight=args.position_weight,
        tf_weight=args.tf_weight,
        occupancy_weight=args.occupancy_weight,
        diversity_weight=args.diversity_weight,
        orthogonality_weight=args.orthogonality_weight,
        site_supervision_weight=args.site_supervision_weight,
        use_hungarian=True,
        slot_count_weight=args.slot_count_weight,
        load_balancing_weight=args.load_balancing_weight,
    )

    print(f"\nLoss weights:")
    print(f"  profile: {args.profile_weight}")
    print(f"  position: {args.position_weight}")
    print(f"  tf_identity: {args.tf_weight}")
    print(f"  occupancy: {args.occupancy_weight}")
    print(f"  diversity: {args.diversity_weight}")
    print(f"  orthogonality: {args.orthogonality_weight}")
    print(f"  site_supervision: {args.site_supervision_weight} (Hungarian matching)")
    print(f"  slot_count: {args.slot_count_weight}")
    print(f"  load_balancing: {args.load_balancing_weight}")

    # Save config
    config = {
        'experiment_name': experiment_name,
        'model_type': 'BEACON-multi',
        'seq_len': args.seq_len,
        'n_slots': args.n_slots,
        'backbone_dim': args.backbone_dim,
        'backbone_layers': args.backbone_layers,
        'slot_dim': args.slot_dim,
        'n_iterations': args.n_iterations,
        'n_tfs': n_tfs,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'patience': args.patience,
        'loss_weights': {
            'profile': args.profile_weight,
            'position': args.position_weight,
            'tf_identity': args.tf_weight,
            'occupancy': args.occupancy_weight,
            'diversity': args.diversity_weight,
            'orthogonality': args.orthogonality_weight,
            'site_supervision': args.site_supervision_weight,
            'slot_count': args.slot_count_weight,
            'load_balancing': args.load_balancing_weight,
        },
        'use_hungarian': True,
        'attention_mode': 'independent',
        'gpu_ids': gpu_ids,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Trainer
    trainer = BEACONTrainer(
        model, train_loader, val_loader, test_loader,
        loss_fn, output_dir=str(experiment_dir),
        num_epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        use_amp=True, log_interval=10, val_interval=1,
        save_interval=5, early_stopping_patience=args.patience,
        num_slots=args.n_slots, num_tfs=n_tfs,
    )

    print(f"\n--- Starting Training ---")
    results = trainer.train()

    print(f"\n--- Training Complete ---")
    print(f"Results saved to: {experiment_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
