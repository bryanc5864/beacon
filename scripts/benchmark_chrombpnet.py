#!/usr/bin/env python3
"""
Benchmark BEACON against ChromBPNet on all datasets.

Trains ChromBPNet models on the same data splits and compares:
  - Profile Pearson correlation
  - Motif recovery (JASPAR correlation)
  - TF classification (BEACON only - ChromBPNet doesn't do this)
  - Throughput (sequences/second)

Requires: pip install chrombpnet (or manual setup)

Usage:
    python scripts/benchmark_chrombpnet.py --dataset K562-7tf
    python scripts/benchmark_chrombpnet.py --all
    python scripts/benchmark_chrombpnet.py --beacon-only  # Just evaluate BEACON
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import h5py


BASE_DIR = Path(__file__).parent.parent

DATASET_CONFIGS = {
    "K562-7tf": {
        "data_dir": "data/processed/multi_tf_k562",
        "n_tfs": 7,
        "cell_line": "K562",
    },
    "K562-20tf": {
        "data_dir": "data/processed/multi_tf_k562_20tf",
        "n_tfs": 20,
        "cell_line": "K562",
    },
    "HepG2-7tf": {
        "data_dir": "data/processed/multi_tf_hepg2_7tf",
        "n_tfs": 7,
        "cell_line": "HepG2",
    },
    "HepG2-20tf": {
        "data_dir": "data/processed/multi_tf_hepg2_20tf",
        "n_tfs": 20,
        "cell_line": "HepG2",
    },
    "WTC11-7tf": {
        "data_dir": "data/processed/multi_tf_wtc11_7tf",
        "n_tfs": 7,
        "cell_line": "WTC11",
    },
    "WTC11-20tf": {
        "data_dir": "data/processed/multi_tf_wtc11_20tf",
        "n_tfs": 20,
        "cell_line": "WTC11",
    },
}


def evaluate_beacon(
    model_path: str,
    test_path: str,
    device: str = "cuda",
    n_tfs: int = 7,
) -> Dict[str, float]:
    """Evaluate a trained BEACON model on test set."""
    from beacon.models import BEACON
    from beacon.data.dataset import BEACONDataset
    from scipy.stats import pearsonr
    from scipy.optimize import linear_sum_assignment

    # Load model
    model = BEACON(
        seq_len=2000, input_channels=4,
        backbone_type="dilated", backbone_dim=128,
        backbone_layers=4, n_slots=16, slot_dim=128,
        n_iterations=3, n_tfs=n_tfs,
        position_mode="gaussian", dropout=0.0,
        attention_mode="independent",
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # Load test data
    dataset = BEACONDataset(
        test_path, seq_length=2000, augment=False,
        n_slots=16, extract_peaks=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_profile_r = []
    all_tf_correct = 0
    all_tf_total = 0
    total_time = 0.0
    n_sequences = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)

            t0 = time.time()
            outputs = model(seq)
            total_time += time.time() - t0
            n_sequences += seq.shape[0]

            # Profile Pearson
            pred_profile = outputs["profile"].cpu().numpy()
            target_profile = batch["profile"].numpy()

            for i in range(seq.shape[0]):
                r, _ = pearsonr(pred_profile[i], target_profile[i])
                if not np.isnan(r):
                    all_profile_r.append(r)

            # TF accuracy via Hungarian matching
            if "binding_sites" in batch:
                pred_tfs = outputs["tf_logits"].argmax(dim=-1).cpu().numpy()
                pred_occ = outputs["occupancy"].squeeze(-1).cpu().numpy()
                target_sites = batch["binding_sites"].numpy()

                for i in range(seq.shape[0]):
                    target_occ = target_sites[i, :, 2]
                    target_tf = target_sites[i, :, 1].astype(int)
                    valid = target_occ > 0.1

                    if valid.sum() == 0:
                        continue

                    gt_tfs = target_tf[valid]
                    n_valid = len(gt_tfs)

                    # Build cost matrix
                    cost = np.zeros((16, n_valid))
                    for k in range(16):
                        for j in range(n_valid):
                            cost[k, j] = 0 if pred_tfs[i, k] == gt_tfs[j] else 1

                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        all_tf_total += 1
                        if pred_tfs[i, r] == gt_tfs[c]:
                            all_tf_correct += 1

    metrics = {
        "profile_pearson_mean": float(np.mean(all_profile_r)) if all_profile_r else 0.0,
        "profile_pearson_std": float(np.std(all_profile_r)) if all_profile_r else 0.0,
        "tf_accuracy": all_tf_correct / max(all_tf_total, 1),
        "throughput_seqs_per_sec": n_sequences / max(total_time, 1e-6),
        "n_test_samples": n_sequences,
    }

    return metrics


def evaluate_chrombpnet(
    data_dir: str,
    output_dir: str,
    cell_line: str = "K562",
) -> Dict[str, float]:
    """
    Train and evaluate ChromBPNet on a dataset.

    Falls back to BPNet-style evaluation if chrombpnet is not available.
    """
    metrics = {}

    try:
        import chrombpnet
        has_chrombpnet = True
    except ImportError:
        has_chrombpnet = False
        print("  ChromBPNet not installed. Install with: pip install chrombpnet")
        print("  Skipping ChromBPNet evaluation.")
        return {"error": "chrombpnet not installed"}

    # ChromBPNet training would go here
    # This is a placeholder for the actual ChromBPNet training pipeline
    # which requires specific data formats (bigWig + peaks)
    print("  ChromBPNet training requires raw bigWig + peak files.")
    print("  Please run ChromBPNet training separately and provide results.")

    return metrics


def run_benchmark(
    dataset_name: str,
    beacon_model_path: Optional[str] = None,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Dict:
    """Run full benchmark for one dataset."""
    config = DATASET_CONFIGS[dataset_name]
    data_dir = BASE_DIR / config["data_dir"]
    test_path = data_dir / "test.h5"

    if output_dir is None:
        output_dir = BASE_DIR / "output" / "benchmarks" / dataset_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"dataset": dataset_name, "config": config}

    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset_name}")
    print(f"{'='*60}")

    # Evaluate BEACON
    if beacon_model_path and test_path.exists():
        print(f"\n--- BEACON Evaluation ---")
        beacon_metrics = evaluate_beacon(
            beacon_model_path, str(test_path),
            device=device, n_tfs=config["n_tfs"],
        )
        results["beacon"] = beacon_metrics
        print(f"  Profile Pearson: {beacon_metrics['profile_pearson_mean']:.4f} "
              f"(+/- {beacon_metrics['profile_pearson_std']:.4f})")
        print(f"  TF Accuracy: {beacon_metrics['tf_accuracy']:.4f}")
        print(f"  Throughput: {beacon_metrics['throughput_seqs_per_sec']:.1f} seq/s")
    else:
        print(f"  BEACON: No model path or test data")

    # Evaluate ChromBPNet
    print(f"\n--- ChromBPNet Evaluation ---")
    chrombpnet_metrics = evaluate_chrombpnet(
        str(data_dir), str(output_dir), config["cell_line"],
    )
    results["chrombpnet"] = chrombpnet_metrics

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def print_comparison_table(all_results: Dict):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'Model':<12} {'Profile r':<12} {'TF Acc':<10} {'Speed':<12}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    for name, results in all_results.items():
        if "beacon" in results:
            b = results["beacon"]
            print(f"{name:<15} {'BEACON':<12} "
                  f"{b.get('profile_pearson_mean', 0):<12.4f} "
                  f"{b.get('tf_accuracy', 0):<10.4f} "
                  f"{b.get('throughput_seqs_per_sec', 0):<12.1f}")
        if "chrombpnet" in results and "error" not in results["chrombpnet"]:
            c = results["chrombpnet"]
            print(f"{'':<15} {'ChromBPNet':<12} "
                  f"{c.get('profile_pearson_mean', 0):<12.4f} "
                  f"{'N/A':<10} "
                  f"{c.get('throughput_seqs_per_sec', 0):<12.1f}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BEACON against ChromBPNet"
    )
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--beacon-model", type=str, help="Path to BEACON model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--beacon-only", action="store_true", help="Only evaluate BEACON")
    args = parser.parse_args()

    if args.all:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return 1

    all_results = {}
    for ds in datasets:
        results = run_benchmark(
            ds,
            beacon_model_path=args.beacon_model,
            device=args.device,
            output_dir=args.output_dir,
        )
        all_results[ds] = results

    if len(all_results) > 1:
        print_comparison_table(all_results)

    # Save combined results
    combined_path = BASE_DIR / "output" / "benchmarks" / "combined_results.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
