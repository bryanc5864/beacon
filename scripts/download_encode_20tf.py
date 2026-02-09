#!/usr/bin/env python3
"""
Download ENCODE ChIP-seq data for multi-TF, multi-cell-line BEACON experiments.

Queries ENCODE API for IDR-thresholded peaks and fold-change-over-control
bigWig signal tracks for specified TF panels across K562, HepG2, and WTC11.

Usage:
    python scripts/download_encode_20tf.py --cell-line K562 --panel 20tf
    python scripts/download_encode_20tf.py --cell-line HepG2 --panel 7tf --dry-run
    python scripts/download_encode_20tf.py --all  # Download all 6 datasets
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ENCODE_BASE = "https://www.encodeproject.org"
BASE_DIR = Path(__file__).parent.parent


def load_tf_panels() -> dict:
    """Load TF panel configuration."""
    config_path = BASE_DIR / "configs" / "tf_panels.json"
    with open(config_path) as f:
        return json.load(f)


def search_encode(tf: str, cell_line: str, assay: str = "TF ChIP-seq") -> list:
    """Search ENCODE for experiments matching TF + cell line."""
    config = load_tf_panels()
    biosample = config["cell_lines"].get(cell_line, {}).get("biosample_ontology", cell_line)

    params = {
        "type": "Experiment",
        "assay_title": assay,
        "target.label": tf,
        "biosample_ontology.term_name": biosample,
        "status": "released",
        "format": "json",
        "limit": "10",
    }

    url = f"{ENCODE_BASE}/search/?{'&'.join(f'{k}={urllib.parse.quote_plus(str(v))}' for k, v in params.items())}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("@graph", [])
    except Exception as e:
        print(f"  Search failed for {tf}/{cell_line}: {e}")
        return []


def get_experiment_files(accession: str) -> list:
    """Get file list for an ENCODE experiment."""
    url = f"{ENCODE_BASE}/experiments/{accession}/?format=json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("files", [])
    except Exception as e:
        print(f"  Failed to get experiment {accession}: {e}")
        return []


def find_best_files(files: list) -> Tuple[Optional[str], Optional[str]]:
    """Find best IDR peaks and fold-change signal files."""
    peaks = None
    signal = None
    peaks_priority = 0
    signal_priority = 0

    for f in files:
        if isinstance(f, str):
            try:
                url = f"{ENCODE_BASE}{f}?format=json"
                with urllib.request.urlopen(url, timeout=30) as response:
                    f = json.loads(response.read().decode())
            except Exception:
                continue

        file_type = f.get("file_type", "")
        output_type = f.get("output_type", "")
        status = f.get("status", "")
        assembly = f.get("assembly", "")
        accession = f.get("accession", "")

        if status != "released":
            continue
        # Prefer GRCh38/hg38
        if assembly not in ("GRCh38", "hg38", ""):
            continue

        # Peaks: prefer IDR > optimal > conservative > any
        if file_type == "bed" or "Peak" in file_type or "bigBed" in file_type:
            ot = output_type.lower()
            priority = 0
            if "idr thresholded" in ot:
                priority = 4
            elif "optimal" in ot:
                priority = 3
            elif "conservative" in ot:
                priority = 2
            elif "peak" in ot:
                priority = 1
            if priority > peaks_priority:
                href = f.get("href", "")
                peaks = (accession, href)
                peaks_priority = priority

        # Signal: prefer fold change > signal p-value > any signal
        if file_type == "bigWig":
            ot = output_type.lower()
            priority = 0
            if "fold change" in ot:
                priority = 3
            elif "signal p-value" in ot:
                priority = 2
            elif "signal" in ot:
                priority = 1
            if priority > signal_priority:
                href = f.get("href", "")
                signal = (accession, href)
                signal_priority = priority

    return peaks, signal


def download_file(accession: str, output_dir: Path, file_format: str, href: str = "") -> bool:
    """Download a file from ENCODE."""
    if href:
        # Use the actual href from ENCODE metadata (has correct extension)
        filename = href.split("/")[-1]
        url = f"{ENCODE_BASE}{href}"
        output_path = output_dir / filename
    elif file_format == "bed":
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bed.gz"
        output_path = output_dir / f"{accession}.bed.gz"
    else:
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bigWig"
        output_path = output_dir / f"{accession}.bigWig"

    if output_path.exists():
        print(f"    Already exists: {output_path.name}")
        return True

    print(f"    Downloading: {output_path.name}")
    cmd = ["wget", "-q", "--show-progress", "-O", str(output_path), url]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1000:
            if output_path.exists():
                output_path.unlink()
            print(f"    FAILED: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        if output_path.exists():
            output_path.unlink()
        print(f"    TIMEOUT")
        return False

    size_mb = output_path.stat().st_size / 1e6
    print(f"    OK ({size_mb:.1f} MB)")
    return True


def download_tf_data(
    tf: str,
    cell_line: str,
    output_dir: Path,
    dry_run: bool = False,
) -> Optional[dict]:
    """Download peaks + signal for one TF in one cell line."""
    print(f"\n  {tf}:")

    experiments = search_encode(tf, cell_line)
    if not experiments:
        print(f"    No experiments found")
        return None

    print(f"    Found {len(experiments)} experiment(s)")

    for exp in experiments:
        exp_acc = exp.get("accession", "")
        print(f"    Trying {exp_acc}...")

        files = get_experiment_files(exp_acc)
        peaks_info, signal_info = find_best_files(files)

        if peaks_info and signal_info:
            peaks_acc, peaks_href = peaks_info
            signal_acc, signal_href = signal_info
            print(f"      Peaks: {peaks_acc}")
            print(f"      Signal: {signal_acc}")

            if dry_run:
                return {
                    "experiment": exp_acc,
                    "peaks": peaks_acc,
                    "signal": signal_acc,
                    "status": "dry_run",
                }

            tf_dir = output_dir / tf
            tf_dir.mkdir(parents=True, exist_ok=True)

            peaks_ok = download_file(peaks_acc, tf_dir, "bed", href=peaks_href)
            signal_ok = download_file(signal_acc, tf_dir, "bigWig", href=signal_href)

            if peaks_ok and signal_ok:
                return {
                    "experiment": exp_acc,
                    "peaks": peaks_acc,
                    "signal": signal_acc,
                    "status": "downloaded",
                }
        else:
            missing = []
            if not peaks_info:
                missing.append("peaks")
            if not signal_info:
                missing.append("signal")
            print(f"      Missing: {', '.join(missing)}")

    return None


def download_panel(
    cell_line: str,
    panel: str,
    dry_run: bool = False,
) -> dict:
    """Download all TFs for a cell line + panel combination."""
    config = load_tf_panels()
    cell_config = config["cell_lines"].get(cell_line)
    if cell_config is None:
        print(f"Unknown cell line: {cell_line}")
        return {}

    tfs = cell_config.get(panel, cell_config.get("7tf", []))
    output_dir = BASE_DIR / "data" / "raw" / "encode" / f"multi_tf_{cell_line.lower()}_{panel}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Downloading {cell_line} {panel} ChIP-seq data")
    print(f"TFs: {', '.join(tfs)}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")

    results = {}
    for tf in tfs:
        result = download_tf_data(tf, cell_line, output_dir, dry_run)
        if result:
            results[tf] = result
            print(f"    -> {tf} OK")
        else:
            print(f"    -> {tf} FAILED")

    # Save manifest
    manifest = {
        "cell_line": cell_line,
        "panel": panel,
        "tfs_requested": tfs,
        "tfs_downloaded": list(results.keys()),
        "experiments": results,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Summary: {len(results)}/{len(tfs)} TFs {'found' if dry_run else 'downloaded'}")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download ENCODE ChIP-seq data for BEACON multi-TF experiments"
    )
    parser.add_argument(
        "--cell-line", type=str, default="K562",
        choices=["K562", "HepG2", "WTC11"],
        help="Cell line to download data for",
    )
    parser.add_argument(
        "--panel", type=str, default="20tf",
        choices=["7tf", "20tf", "original_7tf"],
        help="TF panel size",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all 6 datasets (3 cell lines x 2 panels)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only search ENCODE, don't download files",
    )
    args = parser.parse_args()

    if args.all:
        all_results = {}
        for cell_line in ["K562", "HepG2", "WTC11"]:
            for panel in ["7tf", "20tf"]:
                key = f"{cell_line}_{panel}"
                all_results[key] = download_panel(cell_line, panel, args.dry_run)
                n = len(all_results[key])
                print(f"\n{key}: {n} TFs")

        # Summary
        print(f"\n{'=' * 60}")
        print("Overall Summary:")
        for key, results in all_results.items():
            print(f"  {key}: {len(results)} TFs")
    else:
        results = download_panel(args.cell_line, args.panel, args.dry_run)
        return 0 if len(results) >= 2 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
