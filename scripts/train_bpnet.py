#!/usr/bin/env python3
"""
Train BPNet baseline on CTCF K562 data.

This provides a direct comparison with BEACON on the same data.
BPNet is the current SOTA for base-resolution ChIP-seq profile prediction.

Requirements:
    pip install bpnet-lite tensorflow

Usage:
    python scripts/train_bpnet.py
    python scripts/train_bpnet.py --epochs 50 --batch-size 64
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Set default GPU
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn

# We use a PyTorch reimplementation of BPNet architecture
# This avoids TensorFlow dependency conflicts
BPNET_VERSION = "pytorch"
import h5py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bpnet_ctcf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bpnet_baseline"


def load_beacon_data(data_dir):
    """Load data from BEACON HDF5 format."""
    # Use BEACON's processed data
    beacon_data_dir = PROJECT_ROOT / "data" / "processed" / "ctcf_k562_bigwig"

    train_path = beacon_data_dir / "train.h5"
    val_path = beacon_data_dir / "val.h5"
    test_path = beacon_data_dir / "test.h5"

    def load_h5(path):
        with h5py.File(path, 'r') as f:
            sequences = f['sequences'][:]  # [N, L, 4]
            profiles = f['profiles'][:]    # [N, L]
        return sequences, profiles

    logger.info(f"Loading train data from {train_path}...")
    X_train, y_train = load_h5(train_path)

    logger.info(f"Loading val data from {val_path}...")
    X_val, y_val = load_h5(val_path)

    logger.info(f"Loading test data from {test_path}...")
    X_test, y_test = load_h5(test_path)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_bpnet_lite(train_data, val_data, args, output_dir):
    """Train using bpnet-lite."""
    from bpnetlite import BPNet
    from bpnetlite.losses import MNLLLoss, log1pMSELoss

    X_train, y_train = train_data
    X_val, y_val = val_data

    # BPNet expects [N, 4, L] for sequences (channels first)
    X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
    X_val = np.transpose(X_val, (0, 2, 1)).astype(np.float32)

    # Profiles: [N, L] -> [N, 1, L] (single task)
    y_train = y_train[:, np.newaxis, :].astype(np.float32)
    y_val = y_val[:, np.newaxis, :].astype(np.float32)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Create BPNet model
    seq_len = X_train.shape[2]

    # BPNet architecture
    model = BPNet(
        n_filters=64,
        n_layers=8,
        n_outputs=1,  # Single TF (CTCF)
        n_control_tracks=0,
        alpha=1.0,
        profile_output_bias=True,
        count_output_bias=True,
        name="CTCF_BPNet",
        trimming=(seq_len - 1000) // 2 if seq_len > 1000 else 0,
        verbose=True,
    )

    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # Training
    logger.info(f"\nStarting BPNet training...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fit model
    history = model.fit(
        X_train_t, y_train_t,
        X_valid=X_val_t, y_valid=y_val_t,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        verbose=True,
    )

    # Save model
    model_path = output_dir / "bpnet_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    return model, history


def train_bpnet_manual(train_data, val_data, args, output_dir):
    """Manual BPNet-style training for comparison."""
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import spearmanr

    X_train, y_train = train_data
    X_val, y_val = val_data

    # BPNet expects [N, 4, L] for sequences (channels first)
    X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
    X_val = np.transpose(X_val, (0, 2, 1)).astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Simple BPNet-style model
    class SimpleBPNet(nn.Module):
        def __init__(self, seq_len, n_filters=64, n_layers=8):
            super().__init__()

            # First conv layer
            self.conv1 = nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

            # Dilated residual blocks
            self.blocks = nn.ModuleList()
            for i in range(n_layers):
                dilation = 2 ** i
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(n_filters, n_filters, kernel_size=3,
                              padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    nn.Conv1d(n_filters, n_filters, kernel_size=1),
                ))

            # Profile head
            self.profile_head = nn.Conv1d(n_filters, 1, kernel_size=75, padding=37)

            # Count head
            self.count_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(n_filters, 1),
            )

        def forward(self, x):
            # x: [B, 4, L]
            h = torch.relu(self.conv1(x))

            for block in self.blocks:
                h = h + block(h)

            profile = self.profile_head(h).squeeze(1)  # [B, L]
            counts = self.count_head(h).squeeze(1)     # [B]

            return profile, counts

    seq_len = X_train.shape[2]
    model = SimpleBPNet(seq_len, n_filters=64, n_layers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"SimpleBPNet: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Loss functions
    def multinomial_nll(pred, target):
        """Multinomial negative log-likelihood for profiles."""
        pred_log_softmax = torch.log_softmax(pred, dim=-1)
        target_norm = target / (target.sum(dim=-1, keepdim=True) + 1e-10)
        return -torch.sum(target_norm * pred_log_softmax, dim=-1).mean()

    def count_mse(pred_counts, target):
        """MSE for total counts."""
        true_counts = torch.log1p(target.sum(dim=-1))
        return torch.nn.functional.mse_loss(pred_counts, true_counts)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_pearson': []}
    best_val_loss = float('inf')

    # CSV logging
    import csv
    batch_csv = open(output_dir / "batch_logs.csv", "w", newline='')
    batch_writer = csv.writer(batch_csv)
    batch_writer.writerow(['epoch', 'batch', 'loss', 'profile_loss', 'count_loss'])

    epoch_csv = open(output_dir / "epoch_logs.csv", "w", newline='')
    epoch_writer = csv.writer(epoch_csv)
    epoch_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_pearson', 'val_spearman', 'lr'])

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []

        n_batches = len(train_loader)
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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

            # Save to CSV every batch
            batch_writer.writerow([epoch+1, batch_idx+1, loss.item(),
                                   loss_profile.item(), loss_count.item()])

            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = np.mean(train_losses[-50:])
                logger.info(f"  Batch {batch_idx+1}/{n_batches}: loss={avg_loss:.4f}")
                batch_csv.flush()  # Ensure writes are saved

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

                # Softmax for correlation
                profile_pred_prob = torch.softmax(profile_pred, dim=-1)
                all_pred.append(profile_pred_prob.cpu().numpy())
                all_true.append(batch_y.cpu().numpy())

        # Compute Pearson and Spearman
        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)

        pearson_vals = []
        spearman_vals = []
        for i in range(len(all_pred)):
            corr = np.corrcoef(all_pred[i], all_true[i])[0, 1]
            if not np.isnan(corr):
                pearson_vals.append(corr)
            scorr, _ = spearmanr(all_pred[i], all_true[i])
            if not np.isnan(scorr):
                spearman_vals.append(scorr)

        val_pearson = np.mean(pearson_vals) if pearson_vals else 0
        val_spearman = np.mean(spearman_vals) if spearman_vals else 0

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pearson'].append(val_pearson)

        # Save epoch to CSV
        current_lr = optimizer.param_groups[0]['lr']
        epoch_writer.writerow([epoch+1, train_loss, val_loss, val_pearson, val_spearman, current_lr])
        epoch_csv.flush()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        # Log every epoch
        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_pearson={val_pearson:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Close CSV files
    batch_csv.close()
    epoch_csv.close()
    logger.info(f"Saved batch_logs.csv and epoch_logs.csv")

    return model, history


def evaluate_model(model, test_data, args, output_dir):
    """Evaluate BPNet on test set."""
    X_test, y_test = test_data

    # Prepare data
    X_test = np.transpose(X_test, (0, 2, 1)).astype(np.float32)
    y_test = y_test.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Run predictions
    all_pred = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.from_numpy(X_test[i:i+batch_size]).to(device)
            profile_pred, _ = model(batch_x)
            profile_pred_prob = torch.softmax(profile_pred, dim=-1)
            all_pred.append(profile_pred_prob.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)

    # Compute metrics
    pearson_vals = []
    spearman_vals = []
    from scipy.stats import spearmanr

    for i in range(len(all_pred)):
        # Pearson
        corr = np.corrcoef(all_pred[i], y_test[i])[0, 1]
        if not np.isnan(corr):
            pearson_vals.append(corr)

        # Spearman
        scorr, _ = spearmanr(all_pred[i], y_test[i])
        if not np.isnan(scorr):
            spearman_vals.append(scorr)

    metrics = {
        'profile_pearson': float(np.mean(pearson_vals)),
        'profile_pearson_std': float(np.std(pearson_vals)),
        'profile_spearman': float(np.mean(spearman_vals)),
        'profile_spearman_std': float(np.std(spearman_vals)),
        'n_test_samples': len(y_test),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train BPNet baseline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.experiment_name is None:
        args.experiment_name = f"bpnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("BPNet Baseline Training")
    logger.info("=" * 60)
    logger.info(f"BPNet version: {BPNET_VERSION}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("")

    # Load data
    train_data, val_data, test_data = load_beacon_data(args.data_dir)

    logger.info(f"Train: {train_data[0].shape[0]} samples")
    logger.info(f"Val: {val_data[0].shape[0]} samples")
    logger.info(f"Test: {test_data[0].shape[0]} samples")
    logger.info("")

    # Save config
    config = vars(args).copy()
    config['data_dir'] = str(config['data_dir'])
    config['output_dir'] = str(config['output_dir'])
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Train using PyTorch BPNet implementation
    model, history = train_bpnet_manual(train_data, val_data, args, output_dir)

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_data, args, output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    for key, val in test_metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.6f}")
        else:
            logger.info(f"  {key}: {val}")

    # Save results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("")
    logger.info(f"All outputs saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
