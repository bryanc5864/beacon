#!/usr/bin/env python3
"""
Phase 10: Slot Count Ablation Study

Trains BEACON-multi with different slot counts (4, 8, 16, 24) to demonstrate
that 16 slots is optimal for 7-TF binding prediction. Also evaluates each
model using proper Hungarian-matched metrics.

Usage:
    python scripts/phase10_slot_ablation.py --n-slots 4 --gpus 0
    python scripts/phase10_slot_ablation.py --n-slots 8 --gpus 6
    python scripts/phase10_slot_ablation.py --n-slots 24 --gpus 7
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.models.losses import BEACONLoss
from beacon.training.trainer import BEACONTrainer
from beacon.data.dataset import BEACONDataset


class SlotAdjustedDataset(Dataset):
    """Wraps BEACONDataset to adjust binding_sites to match model's n_slots."""

    def __init__(self, dataset, target_n_slots):
        self.dataset = dataset
        self.target_n_slots = target_n_slots

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'binding_sites' in item:
            bs = item['binding_sites']  # [data_n_slots, 3]
            data_n_slots = bs.shape[0]

            if self.target_n_slots < data_n_slots:
                # Sort by occupancy (col 2) descending, keep top target_n_slots
                occupancies = bs[:, 2]
                _, sorted_idx = occupancies.sort(descending=True)
                item['binding_sites'] = bs[sorted_idx[:self.target_n_slots]]
            elif self.target_n_slots > data_n_slots:
                # Pad with zeros
                padding = torch.zeros(self.target_n_slots - data_n_slots, 3)
                item['binding_sites'] = torch.cat([bs, padding], dim=0)
        return item


def hungarian_matched_evaluation(model, test_loader, device, n_tfs=7):
    """Evaluate model using proper Hungarian matching (same as analyze_beacon_multi.py)."""
    model.eval()

    total_matched = 0
    total_correct_tf = 0
    total_sites = 0
    total_detected = 0
    profile_correlations = []

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            binding_sites = batch['binding_sites']  # [B, n_slots, 3]

            outputs = model(sequences, return_attention=True, return_slots=True)
            pred_positions = outputs['positions'].cpu()
            pred_tf_logits = outputs['tf_logits'].cpu()
            pred_occupancy = outputs['occupancy'].cpu()
            pred_profiles = outputs['profile'].cpu()
            target_profiles = batch['profile']

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                # Get ground truth sites
                gt_sites = []
                for s in range(binding_sites.shape[1]):
                    pos, tf_id, occ = binding_sites[b, s]
                    if occ > 0.5:
                        gt_sites.append((pos.item(), int(tf_id.item()), occ.item()))

                if not gt_sites:
                    continue

                total_sites += len(gt_sites)

                # Get predicted slots with occupancy > 0.3
                pred_occ = pred_occupancy[b].squeeze(-1) if pred_occupancy[b].dim() > 1 else pred_occupancy[b]
                active_mask = pred_occ > 0.3
                active_indices = torch.where(active_mask)[0]

                if len(active_indices) == 0:
                    continue

                # Build cost matrix for Hungarian matching
                n_gt = len(gt_sites)
                n_pred = len(active_indices)
                cost_matrix = np.zeros((n_gt, n_pred))

                for i, (gt_pos, gt_tf, gt_occ) in enumerate(gt_sites):
                    for j, pred_idx in enumerate(active_indices):
                        pos_val = pred_positions[b, pred_idx]
                        pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                        pred_tf = pred_tf_logits[b, pred_idx].argmax().item()

                        # Both gt_pos and pred_pos are normalized [0, 1]; denormalize both
                        pos_dist = abs(gt_pos * 2000 - pred_pos * 2000)
                        tf_match = 1.0 if pred_tf == gt_tf else 0.0

                        cost_matrix[i, j] = pos_dist - tf_match * 500  # favor TF matches

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    gt_pos, gt_tf, _ = gt_sites[i]
                    pred_idx = active_indices[j]
                    pos_val = pred_positions[b, pred_idx]
                    pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                    pred_tf = pred_tf_logits[b, pred_idx].argmax().item()
                    pos_dist_bp = abs(gt_pos * 2000 - pred_pos * 2000)

                    if pos_dist_bp < 200:  # within 200bp
                        total_matched += 1
                        total_detected += 1
                        if pred_tf == gt_tf:
                            total_correct_tf += 1

                # Profile correlation
                pred_p = pred_profiles[b].numpy()
                tgt_p = target_profiles[b].numpy()
                if pred_p.std() > 0 and tgt_p.std() > 0:
                    r = np.corrcoef(pred_p.flatten(), tgt_p.flatten())[0, 1]
                    if not np.isnan(r):
                        profile_correlations.append(r)

    results = {
        'total_sites': total_sites,
        'total_matched': total_matched,
        'total_correct_tf': total_correct_tf,
        'match_rate': total_matched / max(total_sites, 1),
        'tf_accuracy': total_correct_tf / max(total_matched, 1),
        'mean_profile_r': float(np.mean(profile_correlations)) if profile_correlations else 0.0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 10: Slot Count Ablation")
    parser.add_argument("--n-slots", type=int, required=True, help="Number of slots (4, 8, 16, 24)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase10_slot_ablation"))
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = args.data_dir if args.data_dir.is_absolute() else base_dir / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir

    experiment_name = f"slots_{args.n_slots}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase 10: Slot Count Ablation — {args.n_slots} slots")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Output: {experiment_dir}")

    # Load data
    summary_path = data_dir / "summary.json"
    with open(summary_path) as f:
        data_summary = json.load(f)
    n_tfs = data_summary['n_tfs']

    # Load with 16 slots (matching h5 data), then wrap to adjust to target n_slots
    raw_train = BEACONDataset(str(data_dir / "train.h5"), seq_length=2000,
                               augment=True, n_slots=16, extract_peaks=True)
    raw_val = BEACONDataset(str(data_dir / "val.h5"), seq_length=2000,
                             augment=False, n_slots=16, extract_peaks=True)
    raw_test = BEACONDataset(str(data_dir / "test.h5"), seq_length=2000,
                              augment=False, n_slots=16, extract_peaks=True)

    train_dataset = SlotAdjustedDataset(raw_train, args.n_slots)
    val_dataset = SlotAdjustedDataset(raw_val, args.n_slots)
    test_dataset = SlotAdjustedDataset(raw_test, args.n_slots)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model — same architecture as best model, only changing n_slots
    model = BEACON(
        seq_len=2000, input_channels=4,
        backbone_type="dilated", backbone_dim=128,
        backbone_layers=4,
        n_slots=args.n_slots, slot_dim=128,
        n_iterations=3,
        n_tfs=n_tfs, position_mode="gaussian", dropout=0.1,
        attention_mode="independent",
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters ({args.n_slots} slots)")

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    torch.cuda.set_device(gpu_ids[0])
    model = model.cuda()
    print(f"  GPU: {gpu_ids[0]}")

    # Loss — same as best model
    loss_fn = BEACONLoss(
        n_tfs=n_tfs, seq_len=2000,
        profile_weight=1.0, position_weight=0.5,
        tf_weight=5.0, occupancy_weight=0.5,
        diversity_weight=0.5, orthogonality_weight=0.5,
        site_supervision_weight=1.0,
        use_hungarian=True,
        slot_count_weight=0.5,
        load_balancing_weight=0.3,
    )

    # Save config
    config = {
        'experiment': 'phase10_slot_ablation',
        'n_slots': args.n_slots,
        'n_params': n_params,
        'n_tfs': n_tfs,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'attention_mode': 'independent',
        'tf_weight': 5.0,
        'use_hungarian': True,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Train
    trainer = BEACONTrainer(
        model, train_loader, val_loader, test_loader,
        loss_fn, output_dir=str(experiment_dir),
        num_epochs=args.epochs, learning_rate=args.lr,
        weight_decay=0.01, grad_clip=1.0,
        use_amp=True, log_interval=10, val_interval=1,
        save_interval=5, early_stopping_patience=args.patience,
        num_slots=args.n_slots, num_tfs=n_tfs,
    )

    print(f"\n--- Training ({args.n_slots} slots) ---")
    results = trainer.train()

    # Hungarian-matched evaluation
    print(f"\n--- Hungarian-Matched Evaluation ---")
    device = next(model.parameters()).device
    eval_results = hungarian_matched_evaluation(model, test_loader, device, n_tfs)

    print(f"  Match rate: {eval_results['match_rate']:.1%}")
    print(f"  TF accuracy (matched): {eval_results['tf_accuracy']:.1%}")
    print(f"  Profile correlation: {eval_results['mean_profile_r']:.3f}")

    # Save final results
    final_results = {
        'n_slots': args.n_slots,
        'n_params': n_params,
        'training_results': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in results.items()} if results else {},
        'hungarian_eval': eval_results,
    }
    with open(experiment_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {experiment_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
