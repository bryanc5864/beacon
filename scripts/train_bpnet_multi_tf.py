#!/usr/bin/env python3
"""
Train 7 separate BPNet models (one per TF) for fair comparison with BEACON.

BEACON: 1 model → 7 TFs (profile + TF identity + binding sites)
BPNet:  7 models → 7 TFs (profile only, no TF identity)

This demonstrates BEACON's efficiency advantage.
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


class SimpleBPNet(nn.Module):
    """BPNet-style dilated CNN for profile prediction."""

    def __init__(self, seq_len=2000, n_filters=64, n_layers=8):
        super().__init__()

        self.conv1 = nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=1),
            ))

        self.profile_head = nn.Conv1d(n_filters, 1, kernel_size=75, padding=37)

        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, 1),
        )

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        for block in self.blocks:
            h = h + block(h)
        profile = self.profile_head(h).squeeze(1)
        counts = self.count_head(h).squeeze(1)
        return profile, counts


def load_tf_data(data_dir, tf_idx, split='train'):
    """Load data for a single TF from multi-TF HDF5."""
    h5_path = data_dir / f"{split}.h5"

    with h5py.File(h5_path, 'r') as f:
        tf_indices = f['tf_indices'][:]
        mask = tf_indices == tf_idx

        sequences = f['sequences'][mask]  # [N, L, 4]
        profiles = f['profiles'][mask]    # [N, L]

    # Transpose to channels-first: [N, L, 4] → [N, 4, L]
    sequences = np.transpose(sequences, (0, 2, 1)).astype(np.float32)
    profiles = profiles.astype(np.float32)

    return sequences, profiles


def multinomial_nll(pred, target):
    """Multinomial NLL for profiles."""
    pred_log_softmax = torch.log_softmax(pred, dim=-1)
    target_norm = target / (target.sum(dim=-1, keepdim=True) + 1e-10)
    return -torch.sum(target_norm * pred_log_softmax, dim=-1).mean()


def count_mse(pred_counts, target):
    """MSE on log-counts."""
    true_counts = torch.log1p(target.sum(dim=-1))
    return torch.nn.functional.mse_loss(pred_counts, true_counts)


def train_single_bpnet(tf_idx, tf_name, data_dir, output_dir, args):
    """Train a single BPNet model for one TF."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training BPNet for {tf_name} (TF index {tf_idx})")
    logger.info(f"{'='*60}")

    tf_output = output_dir / tf_name
    tf_output.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train = load_tf_data(data_dir, tf_idx, 'train')
    X_val, y_val = load_tf_data(data_dir, tf_idx, 'val')

    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Val: {X_val.shape[0]} samples")

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    seq_len = X_train.shape[2]
    model = SimpleBPNet(seq_len, n_filters=64, n_layers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    epoch_log = open(tf_output / "epoch_logs.csv", "w", newline='')
    epoch_writer = csv.writer(epoch_log)
    epoch_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_pearson', 'val_spearman', 'lr'])

    import time
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            profile_pred, count_pred = model(batch_x)

            loss_profile = multinomial_nll(profile_pred, batch_y)
            loss_count = count_mse(count_pred, batch_y)
            loss = loss_profile + 0.1 * loss_count

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        all_pred = []
        all_true = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                profile_pred, count_pred = model(batch_x)
                loss_profile = multinomial_nll(profile_pred, batch_y)
                loss_count = count_mse(count_pred, batch_y)
                loss = loss_profile + 0.1 * loss_count

                val_losses.append(loss.item())

                profile_pred_prob = torch.softmax(profile_pred, dim=-1)
                all_pred.append(profile_pred_prob.cpu().numpy())
                all_true.append(batch_y.cpu().numpy())

        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)

        pearson_vals = []
        spearman_vals = []
        for i in range(len(all_pred)):
            if all_true[i].std() > 0 and all_pred[i].std() > 0:
                pearson_vals.append(pearsonr(all_pred[i], all_true[i])[0])
                spearman_vals.append(spearmanr(all_pred[i], all_true[i])[0])

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_pearson = np.mean(pearson_vals) if pearson_vals else 0
        avg_spearman = np.mean(spearman_vals) if spearman_vals else 0
        lr = optimizer.param_groups[0]['lr']

        epoch_writer.writerow([epoch+1, avg_train, avg_val, avg_pearson, avg_spearman, lr])
        epoch_log.flush()

        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{args.epochs}: "
                       f"train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, "
                       f"pearson={avg_pearson:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), tf_output / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    epoch_log.close()

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(tf_output / "best_model.pt", map_location=device))

    X_test, y_test = load_tf_data(data_dir, tf_idx, 'test')
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    test_preds = []
    test_trues = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            profile_pred, _ = model(batch_x)
            profile_pred_prob = torch.softmax(profile_pred, dim=-1)
            test_preds.append(profile_pred_prob.cpu().numpy())
            test_trues.append(batch_y.numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_trues = np.concatenate(test_trues, axis=0)

    test_pearson = []
    test_spearman = []
    for i in range(len(test_preds)):
        if test_trues[i].std() > 0 and test_preds[i].std() > 0:
            test_pearson.append(pearsonr(test_preds[i], test_trues[i])[0])
            test_spearman.append(spearmanr(test_preds[i], test_trues[i])[0])

    results = {
        'tf_name': tf_name,
        'tf_idx': tf_idx,
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'best_epoch': best_epoch,
        'training_time_sec': elapsed,
        'n_params': n_params,
        'test_pearson_mean': float(np.mean(test_pearson)),
        'test_pearson_median': float(np.median(test_pearson)),
        'test_spearman_mean': float(np.mean(test_spearman)),
        'test_spearman_median': float(np.median(test_spearman)),
    }

    logger.info(f"\n  {tf_name} Results:")
    logger.info(f"    Test Pearson: {results['test_pearson_mean']:.4f}")
    logger.info(f"    Test Spearman: {results['test_spearman_mean']:.4f}")
    logger.info(f"    Best epoch: {best_epoch}")
    logger.info(f"    Training time: {elapsed:.1f}s")

    with open(tf_output / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train 7 BPNet models for multi-TF comparison")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bpnet_multi_tf"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tfs", type=str, nargs='+', default=None,
                        help="Specific TFs to train (default: all)")

    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir / f"bpnet_multi_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-TF BPNet Training: 7 Separate Models")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Determine which TFs to train
    tfs_to_train = args.tfs or TF_NAMES
    tf_indices = {tf: TF_NAMES.index(tf) for tf in tfs_to_train}

    print(f"Training models for: {', '.join(tfs_to_train)}")
    print()

    # Train each TF
    all_results = {}
    total_time = 0

    for tf_name in tfs_to_train:
        tf_idx = tf_indices[tf_name]
        results = train_single_bpnet(tf_idx, tf_name, data_dir, output_dir, args)
        all_results[tf_name] = results
        total_time += results['training_time_sec']

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: 7-Model BPNet Results")
    print("=" * 60)

    print(f"\n| TF | Test Pearson | Test Spearman | Train Samples | Time (s) |")
    print(f"|-----|-------------|---------------|---------------|----------|")

    pearson_vals = []
    for tf_name in tfs_to_train:
        r = all_results[tf_name]
        print(f"| {tf_name:6} | {r['test_pearson_mean']:.4f} | "
              f"{r['test_spearman_mean']:.4f} | {r['train_samples']:>13} | "
              f"{r['training_time_sec']:.0f} |")
        pearson_vals.append(r['test_pearson_mean'])

    mean_pearson = np.mean(pearson_vals)
    print(f"\nMean Pearson: {mean_pearson:.4f}")
    print(f"Total training time: {total_time:.1f}s ({total_time/3600:.2f} hrs)")
    print(f"Total parameters: {all_results[tfs_to_train[0]]['n_params'] * len(tfs_to_train):,} "
          f"({len(tfs_to_train)} × {all_results[tfs_to_train[0]]['n_params']:,})")

    # Save combined results
    combined = {
        'all_results': {tf: r for tf, r in all_results.items()},
        'summary': {
            'mean_pearson': float(mean_pearson),
            'total_training_time_sec': total_time,
            'n_models': len(tfs_to_train),
            'total_parameters': all_results[tfs_to_train[0]]['n_params'] * len(tfs_to_train),
        }
    }

    with open(output_dir / "combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
