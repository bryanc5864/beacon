#!/usr/bin/env python3
"""
Search ENCODE portal for ChIP-seq experiments and download data.
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path

ENCODE_BASE = "https://www.encodeproject.org"

# Target TFs
TARGET_TFS = ["CTCF", "MYC", "MAX", "CEBPB"]  # Available in HepG2
CELL_LINE = "HepG2"


def search_encode(tf, cell_line, assay="ChIP-seq"):
    """Search ENCODE for experiments."""
    # Build search URL
    params = {
        "type": "Experiment",
        "assay_title": assay,
        "target.label": tf,
        "biosample_ontology.term_name": cell_line,
        "status": "released",
        "format": "json",
        "limit": "5",
    }

    url = f"{ENCODE_BASE}/search/?{'&'.join(f'{k}={urllib.parse.quote(str(v))}' for k, v in params.items())}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("@graph", [])
    except Exception as e:
        print(f"  Search failed: {e}")
        return []


def get_experiment_files(experiment_accession):
    """Get files for an experiment."""
    url = f"{ENCODE_BASE}/experiments/{experiment_accession}/?format=json"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("files", [])
    except Exception as e:
        print(f"  Failed to get experiment: {e}")
        return []


def find_best_files(files):
    """Find best peak and signal files."""
    peaks = None
    signal = None

    for f in files:
        if isinstance(f, str):
            # Need to fetch file details
            try:
                url = f"{ENCODE_BASE}{f}?format=json"
                with urllib.request.urlopen(url, timeout=30) as response:
                    f = json.loads(response.read().decode())
            except:
                continue

        file_type = f.get("file_type", "")
        output_type = f.get("output_type", "")
        status = f.get("status", "")
        accession = f.get("accession", "")

        if status != "released":
            continue

        # Look for IDR peaks or conservative peaks
        if file_type == "bed" and "idr" in output_type.lower():
            peaks = accession
        elif file_type == "bed" and "peak" in output_type.lower() and not peaks:
            peaks = accession

        # Look for fold change bigWig
        if file_type == "bigWig" and "fold change" in output_type.lower():
            signal = accession
        elif file_type == "bigWig" and "signal" in output_type.lower() and not signal:
            signal = accession

    return peaks, signal


def download_file(accession, output_dir, file_format):
    """Download file from ENCODE."""
    if file_format == "bed":
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
    result = subprocess.run(cmd, capture_output=True, timeout=300)

    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1000:
        if output_path.exists():
            output_path.unlink()
        print(f"    FAILED")
        return False

    return True


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "raw" / "encode" / f"multi_tf_{CELL_LINE.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Searching ENCODE for {CELL_LINE} ChIP-seq experiments")
    print(f"{'='*60}")

    results = {}

    for tf in TARGET_TFS:
        print(f"\n{tf}:")

        # Search for experiments
        experiments = search_encode(tf, CELL_LINE)

        if not experiments:
            print(f"  No experiments found")
            continue

        print(f"  Found {len(experiments)} experiment(s)")

        # Try each experiment until we find good files
        for exp in experiments:
            exp_acc = exp.get("accession", "")
            print(f"  Trying {exp_acc}...")

            files = get_experiment_files(exp_acc)
            peaks, signal = find_best_files(files)

            if peaks and signal:
                print(f"    Found peaks: {peaks}")
                print(f"    Found signal: {signal}")

                # Download
                tf_dir = output_dir / tf
                tf_dir.mkdir(exist_ok=True)

                peaks_ok = download_file(peaks, tf_dir, "bed")
                signal_ok = download_file(signal, tf_dir, "bigWig")

                if peaks_ok and signal_ok:
                    results[tf] = {
                        "experiment": exp_acc,
                        "peaks": peaks,
                        "signal": signal,
                    }
                    print(f"  ✓ {tf} complete")
                    break
            else:
                print(f"    No suitable files found")

        if tf not in results:
            print(f"  ✗ {tf} failed")

    # Save manifest
    manifest = {
        "cell_line": CELL_LINE,
        "tfs": list(results.keys()),
        "experiments": results,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Downloaded: {len(results)}/{len(TARGET_TFS)} TFs")
    print(f"TFs: {', '.join(results.keys())}")
    print(f"Output: {output_dir}")

    return 0 if len(results) >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
