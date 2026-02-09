#!/usr/bin/env python3
"""
Train BEACON-multi v2: Enhanced multi-slot training with:
  - Phase 8: Slot dropout (force multi-slot usage)
  - Phase 9: Deeper TF classifier, contrastive loss, TF presence auxiliary loss
  - Mixed single-TF + multi-TF training data

Usage:
  python scripts/train_beacon_multi_v2.py --gpus 1,3,4 --experiment-name phase8_slot_dropout
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.models.losses import BEACONLoss
from beacon.models.heads import DeepTFIdentityHead, SlotDropout
from beacon.training.trainer import BEACONTrainer
from beacon.data.dataset import BEACONDataset


class BEACONMultiV2(BEACON):
    """
    Enhanced BEACON with slot dropout and deeper TF classifier.

    Inherits from BEACON and adds:
    - SlotDropout after slot attention (Phase 8.1)
    - DeepTFIdentityHead replacing standard TF head (Phase 9.1)
    - TF presence head for sequence-level auxiliary loss (Phase 9.4)
    """

    def __init__(
        self,
        slot_dropout_prob: float = 0.3,
        deep_tf_hidden: int = 256,
        deep_tf_dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Replace TF head with deeper version
        n_tfs = kwargs.get('n_tfs', 7)
        slot_dim = kwargs.get('slot_dim', 128)

        self.tf_head = DeepTFIdentityHead(
            slot_dim=slot_dim,
            hidden_dim=deep_tf_hidden,
            n_tfs=n_tfs,
            dropout=deep_tf_dropout,
        )

        # Add slot dropout
        self.slot_dropout = SlotDropout(drop_prob=slot_dropout_prob)

        # TF presence head for auxiliary loss
        self.tf_presence_head = nn.Linear(slot_dim, n_tfs)

    def forward(self, x, return_attention=True, return_slots=True):
        # Encode sequence
        features = self.encode_sequence(x)

        # Slot attention
        if hasattr(self.slot_attention, 'position_head'):
            slots, attention, _ = self.slot_attention(features, return_attention=True)
        else:
            slots, attention = self.slot_attention(features, return_attention=True)

        # Decode occupancy first (needed for slot dropout)
        occupancy = self.occupancy_head(slots)

        # Apply slot dropout during training
        slots_dropped, occupancy_dropped = self.slot_dropout(slots, occupancy)

        # Decode other outputs from (potentially dropped) slots
        outputs = {
            "profile": self.profile_head(slots_dropped),
            "positions": self.position_head(slots_dropped),
            "tf_logits": self.tf_head(slots_dropped),
            "occupancy": occupancy_dropped,
        }

        if attention is not None:
            outputs["attention"] = attention
        outputs["slot_embeddings"] = slots_dropped
        outputs["tf_presence_head"] = self.tf_presence_head

        return outputs


def main():
    parser = argparse.ArgumentParser(description="Train BEACON-multi v2")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/beacon_multi"))
    parser.add_argument("--single-tf-data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/beacon_multi_v2"))
    parser.add_argument("--experiment-name", type=str, default=None)

    # Architecture
    parser.add_argument("--seq-len", type=int, default=2000)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--backbone-dim", type=int, default=128)
    parser.add_argument("--backbone-layers", type=int, default=4)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--n-iterations", type=int, default=3)

    # Phase 8: Slot dropout
    parser.add_argument("--slot-dropout", type=float, default=0.3,
                        help="Probability of dropping dominant slot during training")

    # Phase 9: Deeper TF classifier
    parser.add_argument("--deep-tf-hidden", type=int, default=256,
                        help="Hidden dim for deeper TF classifier")
    parser.add_argument("--deep-tf-dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)

    # Loss weights
    parser.add_argument("--profile-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument("--tf-weight", type=float, default=5.0)
    parser.add_argument("--occupancy-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--orthogonality-weight", type=float, default=0.5)
    parser.add_argument("--site-supervision-weight", type=float, default=1.0)
    parser.add_argument("--slot-count-weight", type=float, default=0.5)
    parser.add_argument("--load-balancing-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.5,
                        help="Weight for contrastive slot loss (Phase 9.2)")
    parser.add_argument("--tf-presence-weight", type=float, default=0.3,
                        help="Weight for sequence-level TF presence loss (Phase 9.4)")

    # Data mixing (Phase 9.5)
    parser.add_argument("--mix-single-tf", action="store_true", default=True,
                        help="Mix in single-TF data for better TF classification signal")
    parser.add_argument("--single-tf-ratio", type=float, default=0.3,
                        help="Fraction of single-TF data in mix")

    # Hardware
    parser.add_argument("--gpus", type=str, default="1,3,4")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    single_tf_dir = args.single_tf_data_dir
    if not single_tf_dir.is_absolute():
        single_tf_dir = base_dir / single_tf_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"beacon_multi_v2_{timestamp}"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEACON-multi v2 Training (Phase 8 + Phase 9)")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Output: {experiment_dir}")

    # Data summary
    summary_path = data_dir / "summary.json"
    n_tfs = 7
    if summary_path.exists():
        with open(summary_path) as f:
            data_summary = json.load(f)
        n_tfs = data_summary.get('n_tfs', 7)

    # Datasets
    train_dataset = BEACONDataset(str(data_dir / "train.h5"),
                                   seq_length=args.seq_len, augment=True,
                                   n_slots=args.n_slots, extract_peaks=True)
    val_dataset = BEACONDataset(str(data_dir / "val.h5"),
                                 seq_length=args.seq_len, augment=False,
                                 n_slots=args.n_slots, extract_peaks=True)
    test_dataset = BEACONDataset(str(data_dir / "test.h5"),
                                  seq_length=args.seq_len, augment=False,
                                  n_slots=args.n_slots, extract_peaks=True)

    print(f"\nMulti-TF data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    # Mix in single-TF data if requested
    if args.mix_single_tf and (single_tf_dir / "train.h5").exists():
        single_train = BEACONDataset(str(single_tf_dir / "train.h5"),
                                      seq_length=args.seq_len, augment=True,
                                      n_slots=args.n_slots, extract_peaks=True)
        print(f"Single-TF data: {len(single_train)} train samples")

        # Calculate subset size to achieve desired ratio
        # We want single_tf_ratio of total to be single-TF
        desired_single = int(len(train_dataset) * args.single_tf_ratio / (1 - args.single_tf_ratio))
        if desired_single < len(single_train):
            indices = torch.randperm(len(single_train))[:desired_single].tolist()
            single_train = torch.utils.data.Subset(single_train, indices)
            print(f"  Subsampled to {len(single_train)} for {args.single_tf_ratio:.0%} mix ratio")

        train_dataset = ConcatDataset([train_dataset, single_train])
        print(f"Mixed training set: {len(train_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Model
    model = BEACONMultiV2(
        seq_len=args.seq_len, input_channels=4,
        backbone_type="dilated", backbone_dim=args.backbone_dim,
        backbone_layers=args.backbone_layers,
        n_slots=args.n_slots, slot_dim=args.slot_dim,
        n_iterations=args.n_iterations,
        n_tfs=n_tfs, position_mode="gaussian", dropout=0.1,
        attention_mode="independent",
        slot_dropout_prob=args.slot_dropout,
        deep_tf_hidden=args.deep_tf_hidden,
        deep_tf_dropout=args.deep_tf_dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"  Slot dropout: {args.slot_dropout}")
    print(f"  Deep TF hidden: {args.deep_tf_hidden}")
    print(f"  Contrastive weight: {args.contrastive_weight}")
    print(f"  TF presence weight: {args.tf_presence_weight}")

    # Multi-GPU
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model = model.cuda()
        print(f"  Using GPU(s): {gpu_ids}")

    # Loss
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
        contrastive_weight=args.contrastive_weight,
        tf_presence_weight=args.tf_presence_weight,
    )

    print(f"\nLoss weights:")
    print(f"  profile: {args.profile_weight}")
    print(f"  tf_identity: {args.tf_weight}")
    print(f"  site_supervision: {args.site_supervision_weight} (Hungarian)")
    print(f"  contrastive: {args.contrastive_weight}")
    print(f"  tf_presence: {args.tf_presence_weight}")
    print(f"  slot_count: {args.slot_count_weight}")
    print(f"  load_balancing: {args.load_balancing_weight}")

    # Save config
    config = {
        'experiment_name': experiment_name,
        'model_type': 'BEACON-multi-v2',
        'seq_len': args.seq_len, 'n_slots': args.n_slots,
        'backbone_dim': args.backbone_dim, 'backbone_layers': args.backbone_layers,
        'slot_dim': args.slot_dim, 'n_iterations': args.n_iterations,
        'n_tfs': n_tfs, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip, 'patience': args.patience,
        'slot_dropout': args.slot_dropout,
        'deep_tf_hidden': args.deep_tf_hidden,
        'deep_tf_dropout': args.deep_tf_dropout,
        'mix_single_tf': args.mix_single_tf,
        'single_tf_ratio': args.single_tf_ratio,
        'loss_weights': {
            'profile': args.profile_weight, 'position': args.position_weight,
            'tf_identity': args.tf_weight, 'occupancy': args.occupancy_weight,
            'diversity': args.diversity_weight, 'orthogonality': args.orthogonality_weight,
            'site_supervision': args.site_supervision_weight,
            'slot_count': args.slot_count_weight,
            'load_balancing': args.load_balancing_weight,
            'contrastive': args.contrastive_weight,
            'tf_presence': args.tf_presence_weight,
        },
        'use_hungarian': True, 'attention_mode': 'independent',
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
