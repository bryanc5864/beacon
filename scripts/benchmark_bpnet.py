#!/usr/bin/env python3
"""
Benchmark BPNet (via bpnet-lite) against BEACON on the same datasets.

Trains a BPNet model on BEACON's data and evaluates profile Pearson correlation,
providing a direct comparison with BEACON's slot-attention architecture.

BPNet expects: X = (N, 4, seq_len), y = (N, n_outputs, output_len)
BEACON data:   sequences = (N, 2000, 4), profiles = (N, 2000)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_bpnet.py
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_bpnet.py --dataset K562-fulltf
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent

DATASETS = {
    'K562-7tf': {
        'data_dir': 'data/processed/multi_tf_k562',
        'n_tfs': 7,
    },
    'K562-fulltf': {
        'data_dir': 'data/processed/multi_tf_k562_fulltf',
        'n_tfs': 14,
    },
    'HepG2-7tf': {
        'data_dir': 'data/processed/multi_tf_hepg2_7tf',
        'n_tfs': 7,
    },
    'HepG2-fulltf': {
        'data_dir': 'data/processed/multi_tf_hepg2_fulltf',
        'n_tfs': 12,
    },
}


def load_data(data_dir, split='train', max_samples=None):
    """Load BEACON HDF5 data and convert to BPNet format."""
    path = Path(data_dir) / f'{split}.h5'
    with h5py.File(path, 'r') as f:
        sequences = f['sequences'][:]  # (N, 2000, 4)
        profiles = f['profiles'][:]    # (N, 2000)

    if max_samples is not None:
        sequences = sequences[:max_samples]
        profiles = profiles[:max_samples]

    # Convert to BPNet format: channels-first
    X = sequences.transpose(0, 2, 1)  # (N, 4, 2000)
    # BPNet expects multi-output profiles: (N, n_outputs, output_len)
    # We have single profile, so n_outputs=1
    y = profiles[:, np.newaxis, :]  # (N, 1, 2000)

    return X, y


def evaluate_profile(model, X_test, y_test, device, batch_size=32, trimming=500):
    """Evaluate BPNet profile prediction via Pearson correlation."""
    model.eval()
    correlations = []
    seq_len = X_test.shape[2]

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.from_numpy(X_test[i:i+batch_size]).float().to(device)
            pred = model(batch_X)

            # BPNet returns (y_profile_logits, y_counts)
            if isinstance(pred, tuple):
                pred_profile = pred[0].cpu().numpy()
            else:
                pred_profile = pred.cpu().numpy()

            # Trim target to match BPNet output length
            tgt = y_test[i:i+batch_size, :, trimming:seq_len-trimming]

            for j in range(len(batch_X)):
                p = pred_profile[j].flatten()
                t = tgt[j].flatten()
                if p.std() > 0 and t.std() > 0:
                    r = np.corrcoef(p, t)[0, 1]
                    if not np.isnan(r):
                        correlations.append(r)

    return correlations


def train_bpnet(dataset_name, device='cuda:0', max_epochs=100, early_stopping=10,
                batch_size=64, lr=1e-3, output_dir=None):
    """Train BPNet on a dataset and evaluate."""
    from bpnetlite import BPNet

    cfg = DATASETS[dataset_name]
    data_dir = BASE_DIR / cfg['data_dir']

    print(f"\n{'='*70}")
    print(f"  BPNet Benchmark: {dataset_name}")
    print(f"{'='*70}")

    # Load data
    print("  Loading training data...")
    X_train, y_train = load_data(data_dir, 'train')
    print(f"  Train: {X_train.shape}")

    print("  Loading validation data...")
    X_val, y_val = load_data(data_dir, 'val')
    print(f"  Val: {X_val.shape}")

    print("  Loading test data...")
    X_test, y_test = load_data(data_dir, 'test')
    print(f"  Test: {X_test.shape}")

    seq_len = X_train.shape[2]  # 2000

    # Output length for BPNet's trimmed output
    output_len = 1000
    trimming = (seq_len - output_len) // 2

    # Create DataLoader
    # BPNet's fit expects (X, y, labels) where labels is a 1D mask (0/1)
    # All samples are valid so labels = ones
    # y must be trimmed to match BPNet output length
    y_train_trimmed = y_train[:, :, trimming:seq_len-trimming]  # (N, 1, 1000)
    labels_train = torch.ones(len(X_train), dtype=torch.long)  # all active
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train_trimmed).float(),
        labels_train,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    # Create BPNet model
    # n_outputs=1 (single aggregate profile), n_control_tracks=0
    model = BPNet(
        n_filters=64,
        n_layers=8,
        n_outputs=1,
        n_control_tracks=0,
        profile_output_bias=True,
        count_output_bias=True,
        trimming=trimming,
        verbose=True,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BPNet params: {n_params:,}")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare validation tensors — trim y to output length to match BPNet output
    X_val_t = torch.from_numpy(X_val).float()
    y_val_trimmed = y_val[:, :, trimming:seq_len-trimming]  # (N, 1, 1000)
    y_val_t = torch.from_numpy(y_val_trimmed).float()

    # Train
    print(f"\n  Training BPNet (max {max_epochs} epochs, ES patience={early_stopping})...")
    start_time = time.time()

    model.fit(
        training_data=train_loader,
        optimizer=optimizer,
        X_valid=X_val_t,
        y_valid=y_val_t,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        early_stopping=early_stopping,
    )

    train_time = time.time() - start_time
    print(f"  Training time: {train_time/3600:.1f}h")

    # Evaluate on test set
    print(f"\n  Evaluating on test set...")
    model = model.to(device)
    correlations = evaluate_profile(model, X_test, y_test, device, batch_size, trimming=trimming)

    results = {
        'dataset': dataset_name,
        'n_params': n_params,
        'train_time_h': train_time / 3600,
        'mean_profile_pearson': float(np.mean(correlations)),
        'median_profile_pearson': float(np.median(correlations)),
        'std_profile_pearson': float(np.std(correlations)),
        'n_test_samples': len(correlations),
    }

    print(f"\n  Results:")
    print(f"    Profile Pearson (mean):   {results['mean_profile_pearson']:.4f}")
    print(f"    Profile Pearson (median): {results['median_profile_pearson']:.4f}")
    print(f"    Test samples evaluated:   {results['n_test_samples']}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / f'bpnet_{dataset_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark BPNet against BEACON')
    parser.add_argument('--dataset', type=str, default='K562-7tf',
                       choices=list(DATASETS.keys()),
                       help='Dataset to benchmark on')
    parser.add_argument('--all', action='store_true',
                       help='Run on all datasets')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='outputs/bpnet_benchmark')
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.all else [args.dataset]
    all_results = {}

    for ds in datasets:
        results = train_bpnet(
            ds,
            device=args.device,
            max_epochs=args.max_epochs,
            early_stopping=args.early_stopping,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir,
        )
        all_results[ds] = results

    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"  BPNet Benchmark Summary")
    print(f"{'='*70}")
    print(f"  {'Dataset':<16} {'Profile r':<12} {'Params':<10} {'Time':<8}")
    for ds, r in all_results.items():
        print(f"  {ds:<16} {r['mean_profile_pearson']:.4f}       {r['n_params']:,}  {r['train_time_h']:.1f}h")

    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'bpnet_all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return 0


if __name__ == '__main__':
    sys.exit(main())
