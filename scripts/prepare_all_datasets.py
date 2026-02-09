#!/usr/bin/env python3
"""
Prepare all 6 datasets for BEACON multi-cell-line experiments.

Runs the data preparation pipeline for each combination of:
  - Cell lines: K562, HepG2, WTC11
  - TF panels: 7TF, 20TF

Uses the same chromosome splits across all datasets:
  - Train: chr1-17
  - Val: chr18-19
  - Test: chr20-22, chrX

Usage:
    python scripts/prepare_all_datasets.py --all
    python scripts/prepare_all_datasets.py --cell-line K562 --panel 20tf
    python scripts/prepare_all_datasets.py --list  # Show all datasets
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def load_config():
    """Load TF panel configuration."""
    config_path = BASE_DIR / "configs" / "tf_panels.json"
    with open(config_path) as f:
        return json.load(f)


def get_datasets(config, cell_line=None, panel=None):
    """Get list of datasets to prepare."""
    datasets = []
    for name, ds_config in config["datasets"].items():
        if cell_line and ds_config["cell_line"] != cell_line:
            continue
        if panel and ds_config["panel"] != panel:
            continue
        datasets.append((name, ds_config))
    return datasets


def check_raw_data(cell_line, panel, config):
    """Check if raw ENCODE data exists for a dataset."""
    raw_dir = BASE_DIR / "data" / "raw" / "encode" / f"multi_tf_{cell_line.lower()}_{panel}"
    manifest_path = raw_dir / "manifest.json"
    
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        n_tfs = len(manifest.get("tfs_downloaded", []))
        return True, n_tfs
    return False, 0


def prepare_dataset(name, ds_config, config, force=False):
    """Prepare a single dataset."""
    cell_line = ds_config["cell_line"]
    panel = ds_config["panel"]
    output_dir = BASE_DIR / ds_config["output_dir"]
    
    # Get TF list for this dataset
    cell_config = config["cell_lines"][cell_line]
    tfs = cell_config.get(panel, [])
    
    print(f"\n{'='*60}")
    print(f"Preparing: {name}")
    print(f"  Cell line: {cell_line}")
    print(f"  Panel: {panel} ({len(tfs)} TFs)")
    print(f"  TFs: {', '.join(tfs)}")
    print(f"  Output: {output_dir}")
    
    # Check if already exists
    train_path = output_dir / "train.h5"
    if train_path.exists() and not force:
        print(f"  SKIP: {train_path} already exists (use --force to overwrite)")
        return True
    
    # Check raw data availability
    raw_exists, n_raw_tfs = check_raw_data(cell_line, panel, config)
    if not raw_exists:
        print(f"  WARNING: Raw data not found for {cell_line}/{panel}")
        print(f"  Run: python scripts/download_encode_20tf.py --cell-line {cell_line} --panel {panel}")
        return False
    
    print(f"  Raw data: {n_raw_tfs} TFs available")
    
    # Determine raw data directory
    raw_dir = BASE_DIR / "data" / "raw" / "encode" / f"multi_tf_{cell_line.lower()}_{panel}"
    if not raw_dir.exists():
        # Try without panel suffix for original K562
        raw_dir = BASE_DIR / "data" / "raw" / "encode" / f"multi_tf_{cell_line.lower()}"

    # Run prepare_multi_tf_data.py with appropriate arguments
    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "prepare_multi_tf_data.py"),
        "--data-dir", str(raw_dir),
        "--output-dir", str(output_dir),
        "--seq-length", "2000",
        "--tfs", ",".join(tfs),
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            return False
        print(f"  OK: Dataset prepared successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Data preparation exceeded 1 hour")
        return False
    except FileNotFoundError:
        print(f"  ERROR: prepare_multi_tf_data.py not found")
        print(f"  You may need to run the data preparation manually")
        return False


def prepare_gradient_importance(name, ds_config, model_path=None):
    """Precompute gradient importance for a dataset."""
    output_dir = BASE_DIR / ds_config["output_dir"]
    importance_path = output_dir / "gradient_importance.h5"
    
    if importance_path.exists():
        print(f"  Gradient importance already exists: {importance_path}")
        return True
    
    if model_path is None:
        print(f"  SKIP: No model path provided for gradient importance computation")
        return False
    
    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "precompute_gradient_importance.py"),
        "--model-path", str(model_path),
        "--data-dir", str(output_dir),
        "--output-path", str(importance_path),
    ]
    
    print(f"  Computing gradient importance...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Gradient importance exceeded 2 hours")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all datasets for BEACON multi-cell-line experiments"
    )
    parser.add_argument("--all", action="store_true", help="Prepare all 6 datasets")
    parser.add_argument("--cell-line", type=str, choices=["K562", "HepG2", "WTC11"])
    parser.add_argument("--panel", type=str, choices=["7tf", "20tf", "original_7tf"])
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing datasets")
    parser.add_argument("--with-importance", action="store_true", help="Also compute gradient importance")
    parser.add_argument("--model-path", type=Path, help="Model path for gradient importance")
    args = parser.parse_args()
    
    config = load_config()
    
    if args.list:
        print("Available datasets:")
        for name, ds_config in config["datasets"].items():
            cell_line = ds_config["cell_line"]
            panel = ds_config["panel"]
            tfs = config["cell_lines"][cell_line].get(panel, [])
            output_dir = BASE_DIR / ds_config["output_dir"]
            exists = (output_dir / "train.h5").exists()
            status = "EXISTS" if exists else "MISSING"
            print(f"  {name}: {cell_line}/{panel} ({len(tfs)} TFs) [{status}]")
        return 0
    
    if args.all:
        datasets = get_datasets(config)
    elif args.cell_line or args.panel:
        datasets = get_datasets(config, args.cell_line, args.panel)
    else:
        parser.print_help()
        return 1
    
    print(f"Preparing {len(datasets)} dataset(s)")
    
    results = {}
    for name, ds_config in datasets:
        ok = prepare_dataset(name, ds_config, config, force=args.force)
        results[name] = ok
        
        if ok and args.with_importance and args.model_path:
            prepare_gradient_importance(name, ds_config, args.model_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} datasets prepared successfully")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
