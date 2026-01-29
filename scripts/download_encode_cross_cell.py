#!/usr/bin/env python3
"""
Download ENCODE ChIP-seq data for cross-cell-line transfer experiment.
Uses proper ENCODE API to find and download files.
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
import time

ENCODE_BASE = "https://www.encodeproject.org"

# Target TFs (ones available in HepG2)
TARGET_TFS = ["CTCF", "MYC", "MAX", "CEBPB"]
CELL_LINE = "HepG2"


def encode_search(tf, cell_line):
    """Search ENCODE for TF ChIP-seq experiments."""
    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "target.label": tf,
        "biosample_ontology.term_name": cell_line,
        "status": "released",
        "format": "json",
        "limit": "3",
    }

    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{ENCODE_BASE}/search/?{query}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
            return data.get("@graph", [])
    except Exception as e:
        print(f"    Search error: {e}")
        return []


def get_experiment_details(accession):
    """Get full experiment details including files."""
    url = f"{ENCODE_BASE}/experiments/{accession}/?format=json"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"    Error getting experiment: {e}")
        return None


def find_files(experiment_data):
    """Find best peak and signal files from experiment."""
    peaks_file = None
    signal_file = None

    files = experiment_data.get("files", [])

    for file_ref in files:
        # Fetch file details
        if isinstance(file_ref, str):
            file_url = f"{ENCODE_BASE}{file_ref}?format=json"
        else:
            file_url = f"{ENCODE_BASE}{file_ref.get('@id', '')}?format=json"

        try:
            req = urllib.request.Request(file_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                f = json.loads(response.read().decode())
        except:
            continue

        status = f.get("status", "")
        file_type = f.get("file_type", "")
        output_type = f.get("output_type", "")
        assembly = f.get("assembly", "")
        accession = f.get("accession", "")

        # Only released files with GRCh38
        if status != "released" or assembly not in ["GRCh38", "hg38", ""]:
            continue

        # Prefer IDR peaks, then optimal peaks, then replicated peaks
        if file_type == "bed narrowPeak":
            if "IDR" in output_type:
                peaks_file = accession
            elif "optimal" in output_type.lower() and not peaks_file:
                peaks_file = accession
            elif "peak" in output_type.lower() and not peaks_file:
                peaks_file = accession

        # Prefer fold change signal
        if file_type == "bigWig":
            if "fold change" in output_type.lower():
                signal_file = accession
            elif "signal" in output_type.lower() and not signal_file:
                signal_file = accession

    return peaks_file, signal_file


def download_file(accession, output_path, file_type):
    """Download file from ENCODE."""
    if file_type == "peaks":
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bed.gz"
    else:
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bigWig"

    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"      Already exists: {output_path.name}")
        return True

    print(f"      Downloading: {output_path.name}...")
    cmd = ["wget", "-q", "-O", str(output_path), url]

    try:
        result = subprocess.run(cmd, timeout=600)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            print(f"      Done ({output_path.stat().st_size / 1024:.1f} KB)")
            return True
    except subprocess.TimeoutExpired:
        print(f"      Timeout")

    if output_path.exists():
        output_path.unlink()
    return False


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "raw" / "encode" / f"multi_tf_{CELL_LINE.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Downloading {CELL_LINE} ChIP-seq data from ENCODE")
    print(f"{'='*60}")
    print(f"Target TFs: {', '.join(TARGET_TFS)}")
    print()

    results = {}

    for tf in TARGET_TFS:
        print(f"\n{tf}:")

        # Search for experiments
        print(f"  Searching ENCODE...")
        experiments = encode_search(tf, CELL_LINE)

        if not experiments:
            print(f"  No experiments found")
            continue

        print(f"  Found {len(experiments)} experiment(s)")

        # Try each experiment
        for exp in experiments:
            exp_acc = exp.get("accession", "")
            print(f"  Checking {exp_acc}...")

            # Get full experiment details
            exp_data = get_experiment_details(exp_acc)
            if not exp_data:
                continue

            # Find best files
            peaks_acc, signal_acc = find_files(exp_data)

            if not peaks_acc:
                print(f"    No peak file found")
                continue
            if not signal_acc:
                print(f"    No signal file found")
                continue

            print(f"    Peaks: {peaks_acc}")
            print(f"    Signal: {signal_acc}")

            # Download files
            tf_dir = output_dir / tf
            tf_dir.mkdir(exist_ok=True)

            peaks_path = tf_dir / f"{peaks_acc}.bed.gz"
            signal_path = tf_dir / f"{signal_acc}.bigWig"

            peaks_ok = download_file(peaks_acc, peaks_path, "peaks")
            time.sleep(1)  # Rate limiting
            signal_ok = download_file(signal_acc, signal_path, "signal")

            if peaks_ok and signal_ok:
                results[tf] = {
                    "experiment": exp_acc,
                    "peaks": peaks_acc,
                    "peaks_file": str(peaks_path),
                    "signal": signal_acc,
                    "signal_file": str(signal_path),
                }
                print(f"  ✓ {tf} complete")
                break
            else:
                print(f"    Download failed, trying next experiment...")

        if tf not in results:
            print(f"  ✗ {tf} failed")

        time.sleep(2)  # Rate limiting between TFs

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
    print(f"Successfully downloaded: {len(results)}/{len(TARGET_TFS)} TFs")
    print(f"TFs: {', '.join(results.keys()) if results else 'none'}")
    print(f"Output directory: {output_dir}")

    if len(results) >= 2:
        print(f"\n✓ Sufficient data for cross-cell transfer experiment")
        return 0
    else:
        print(f"\n✗ Need at least 2 TFs for meaningful comparison")
        return 1


if __name__ == "__main__":
    sys.exit(main())
