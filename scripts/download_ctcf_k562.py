#!/usr/bin/env python3
"""
Download CTCF ChIP-seq data from ENCODE for K562.

Phase 2A: Single-TF real data pipeline.

Downloads:
1. Optimal IDR narrowPeak (binding sites)
2. Signal p-value bigWig (binding profiles)

Uses direct ENCODE accession IDs for reproducibility.
"""

import os
import sys
import requests
import json
from pathlib import Path
from tqdm import tqdm

ENCODE_BASE_URL = "https://www.encodeproject.org"
HEADERS = {"accept": "application/json"}

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "encode" / "CTCF_K562"


def query_encode_api(tf="CTCF", cell_line="K562"):
    """Query ENCODE API for CTCF ChIP-seq in K562."""
    print(f"Querying ENCODE for {tf} ChIP-seq in {cell_line}...")

    # Search for experiments
    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "target.label": tf,
        "biosample_ontology.term_name": cell_line,
        "assembly": "GRCh38",
        "status": "released",
        "format": "json",
        "limit": "all",
    }

    resp = requests.get(f"{ENCODE_BASE_URL}/search/", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json()

    experiments = results.get("@graph", [])
    print(f"  Found {len(experiments)} experiments")

    for exp in experiments:
        acc = exp.get("accession", "")
        biosample = exp.get("biosample_summary", "")
        print(f"    {acc}: {biosample}")

    return experiments


def get_best_files(experiment_accession):
    """Get the best narrowPeak and bigWig files for an experiment."""
    print(f"\n  Fetching files for {experiment_accession}...")

    resp = requests.get(
        f"{ENCODE_BASE_URL}/experiments/{experiment_accession}/?format=json",
        headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    exp_data = resp.json()

    peak_files = []
    signal_files = []

    for f in exp_data.get("files", []):
        if isinstance(f, str):
            # Fetch file details
            try:
                f_resp = requests.get(f"{ENCODE_BASE_URL}{f}?format=json", headers=HEADERS, timeout=30)
                f_resp.raise_for_status()
                f = f_resp.json()
            except Exception as e:
                print(f"    Warning: Could not fetch file details for {f}: {e}")
                continue

        if not isinstance(f, dict):
            continue

        status = f.get("status", "")
        if status != "released":
            continue

        file_type = f.get("file_type", "")
        output_type = f.get("output_type", "")
        assembly = f.get("assembly", "")

        # Only GRCh38
        if assembly != "GRCh38":
            continue

        if file_type == "bed narrowPeak":
            peak_files.append({
                "accession": f.get("accession", ""),
                "output_type": output_type,
                "href": f.get("href", ""),
                "file_size": f.get("file_size", 0),
                "file_type": file_type,
            })
            print(f"    Peak: {f.get('accession')} - {output_type} ({f.get('file_size', 0)/1e6:.1f} MB)")

        elif file_type == "bigWig":
            signal_files.append({
                "accession": f.get("accession", ""),
                "output_type": output_type,
                "href": f.get("href", ""),
                "file_size": f.get("file_size", 0),
                "file_type": file_type,
            })
            print(f"    Signal: {f.get('accession')} - {output_type} ({f.get('file_size', 0)/1e6:.1f} MB)")

    # Prefer optimal IDR peaks
    best_peak = None
    for p in peak_files:
        ot = p["output_type"].lower()
        if "optimal" in ot or "idr" in ot:
            best_peak = p
            break
    if best_peak is None and peak_files:
        best_peak = peak_files[0]

    # Prefer signal p-value bigWig
    best_signal = None
    for s in signal_files:
        ot = s["output_type"].lower()
        if "p-value" in ot:
            best_signal = s
            break
    # Fall back to fold change
    if best_signal is None:
        for s in signal_files:
            ot = s["output_type"].lower()
            if "fold" in ot:
                best_signal = s
                break
    if best_signal is None and signal_files:
        best_signal = signal_files[0]

    return best_peak, best_signal


def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    full_url = f"{ENCODE_BASE_URL}{url}" if url.startswith("/") else url

    print(f"  Downloading: {desc or dest_path.name}")
    print(f"  URL: {full_url}")

    resp = requests.get(full_url, stream=True, timeout=300)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc or dest_path.name) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"  Saved: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENCODE CTCF ChIP-seq Download (K562)")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Step 1: Query ENCODE API
    experiments = query_encode_api("CTCF", "K562")

    if not experiments:
        print("ERROR: No experiments found!")
        return 1

    # Step 2: Get best files from first experiment
    # Use the first (usually highest quality) experiment
    exp_accession = experiments[0].get("accession")
    best_peak, best_signal = get_best_files(exp_accession)

    # If first experiment didn't have what we need, try others
    for exp in experiments[1:]:
        if best_peak and best_signal:
            break
        acc = exp.get("accession")
        peak, signal = get_best_files(acc)
        if not best_peak and peak:
            best_peak = peak
        if not best_signal and signal:
            best_signal = signal

    print("\n" + "=" * 70)
    print("Selected files:")
    if best_peak:
        print(f"  Peak: {best_peak['accession']} ({best_peak['output_type']})")
    else:
        print("  Peak: NONE FOUND")
    if best_signal:
        print(f"  Signal: {best_signal['accession']} ({best_signal['output_type']})")
    else:
        print("  Signal: NONE FOUND")
    print()

    # Step 3: Download files
    downloaded = {}

    if best_peak:
        peak_path = OUTPUT_DIR / f"{best_peak['accession']}_peaks.narrowPeak.gz"
        if not peak_path.exists():
            download_file(best_peak["href"], peak_path, "CTCF peaks")
        else:
            print(f"  Peak file exists: {peak_path}")
        downloaded["peaks"] = str(peak_path)

    if best_signal:
        signal_path = OUTPUT_DIR / f"{best_signal['accession']}_signal.bigWig"
        if not signal_path.exists():
            download_file(best_signal["href"], signal_path, "CTCF signal")
        else:
            print(f"  Signal file exists: {signal_path}")
        downloaded["signal"] = str(signal_path)

    # Save manifest
    manifest = {
        "tf": "CTCF",
        "cell_line": "K562",
        "experiment": exp_accession,
        "peak_file": best_peak,
        "signal_file": best_signal,
        "downloaded": downloaded,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)
    print(f"\nNext step: Process data with:")
    print(f"  python scripts/prepare_ctcf_data.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
