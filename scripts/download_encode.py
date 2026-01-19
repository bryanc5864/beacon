#!/usr/bin/env python3
"""
Download ENCODE ChIP-seq data for BEACON training.

We focus on a curated set of well-characterized transcription factors
with high-quality ChIP-seq data across multiple cell types.

For development, we start with:
- Key pluripotency TFs: NANOG, OCT4 (POU5F1), SOX2
- Pioneer factors: FOXA1, FOXA2
- General TFs: CTCF, SP1
- Signal-responsive TFs: JUN, FOS

Data types:
- Optimal IDR-thresholded peaks (narrowPeak)
- Signal p-value tracks (bigWig)
- Raw reads (fastq) - optional, for reprocessing
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time

# ENCODE API configuration
ENCODE_BASE_URL = "https://www.encodeproject.org"
HEADERS = {"accept": "application/json"}

# Priority transcription factors for BEACON development
# These cover key TF families and have abundant ChIP-seq data
PRIORITY_TFS = [
    # Pluripotency network
    "NANOG",
    "POU5F1",  # OCT4
    "SOX2",
    # Pioneer factors
    "FOXA1",
    "FOXA2",
    # Architectural protein
    "CTCF",
    # General transcription
    "SP1",
    # AP-1 complex
    "JUN",
    "FOS",
    # Additional important TFs
    "MYC",
    "MAX",
    "REST",
    "YY1",
    "ELF1",
    "GABPA",
]

# Priority cell lines with most data
PRIORITY_CELL_LINES = [
    "K562",      # Chronic myelogenous leukemia
    "HepG2",     # Hepatocellular carcinoma
    "GM12878",   # Lymphoblastoid
    "H1",        # Human ESC
    "HeLa-S3",   # Cervical carcinoma
    "A549",      # Lung carcinoma
    "MCF-7",     # Breast cancer
]


def query_encode(url: str, params: dict = None) -> dict:
    """Query ENCODE REST API with rate limiting."""
    full_url = f"{ENCODE_BASE_URL}{url}"
    try:
        response = requests.get(full_url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(0.1)  # Rate limiting
        return response.json()
    except Exception as e:
        print(f"Error querying {full_url}: {e}")
        return {}


def search_chipseq_experiments(
    tf_name: str,
    cell_lines: List[str] = None,
    assembly: str = "GRCh38"
) -> List[dict]:
    """
    Search for ChIP-seq experiments for a given TF.

    Returns list of experiment metadata.
    """
    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "target.label": tf_name,
        "assembly": assembly,
        "status": "released",
        "files.file_type": "bed narrowPeak",
        "format": "json",
        "limit": "100",
    }

    if cell_lines:
        params["biosample_ontology.term_name"] = cell_lines

    result = query_encode("/search/", params)
    return result.get("@graph", [])


def get_experiment_files(
    experiment_accession: str,
    file_types: List[str] = None
) -> List[dict]:
    """Get downloadable files for an experiment."""
    if file_types is None:
        file_types = ["bed narrowPeak", "bigWig"]

    result = query_encode(f"/experiments/{experiment_accession}/")
    if not result:
        return []

    files = []
    for f in result.get("files", []):
        if isinstance(f, str):
            # Need to fetch file details
            file_data = query_encode(f)
            f = file_data

        if not isinstance(f, dict):
            continue

        file_type = f.get("file_type", "")
        output_type = f.get("output_type", "")
        status = f.get("status", "")

        # Filter for desired file types and released status
        if status == "released" and file_type in file_types:
            # Prefer optimal IDR peaks
            if file_type == "bed narrowPeak":
                if "optimal" in output_type.lower() or "idr" in output_type.lower():
                    files.append({
                        "accession": f.get("accession", ""),
                        "file_type": file_type,
                        "output_type": output_type,
                        "href": f.get("href", ""),
                        "assembly": f.get("assembly", ""),
                        "file_size": f.get("file_size", 0),
                    })
            elif file_type == "bigWig":
                if "signal" in output_type.lower():
                    files.append({
                        "accession": f.get("accession", ""),
                        "file_type": file_type,
                        "output_type": output_type,
                        "href": f.get("href", ""),
                        "assembly": f.get("assembly", ""),
                        "file_size": f.get("file_size", 0),
                    })

    return files


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        full_url = f"{ENCODE_BASE_URL}{url}" if url.startswith("/") else url
        response = requests.get(full_url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def create_encode_manifest(
    output_dir: Path,
    tfs: List[str] = None,
    cell_lines: List[str] = None,
    max_experiments_per_tf: int = 5
) -> dict:
    """
    Query ENCODE and create a manifest of files to download.

    Returns manifest dictionary with download information.
    """
    if tfs is None:
        tfs = PRIORITY_TFS
    if cell_lines is None:
        cell_lines = PRIORITY_CELL_LINES

    manifest = {
        "tfs": {},
        "total_files": 0,
        "total_size_bytes": 0,
    }

    print(f"Querying ENCODE for {len(tfs)} transcription factors...")
    print("=" * 60)

    for tf in tqdm(tfs, desc="Scanning TFs"):
        experiments = search_chipseq_experiments(tf, cell_lines)

        if not experiments:
            print(f"  {tf}: No experiments found")
            continue

        tf_data = {
            "experiments": [],
            "files": [],
        }

        # Limit experiments per TF for development
        for exp in experiments[:max_experiments_per_tf]:
            exp_accession = exp.get("accession", "")
            biosample = exp.get("biosample_summary", "Unknown")

            # Get files for this experiment
            files = get_experiment_files(exp_accession)

            if files:
                tf_data["experiments"].append({
                    "accession": exp_accession,
                    "biosample": biosample,
                    "file_count": len(files),
                })

                for f in files:
                    f["experiment"] = exp_accession
                    f["tf"] = tf
                    f["biosample"] = biosample
                    tf_data["files"].append(f)
                    manifest["total_size_bytes"] += f.get("file_size", 0)

        manifest["tfs"][tf] = tf_data
        manifest["total_files"] += len(tf_data["files"])

    return manifest


def download_from_manifest(
    manifest: dict,
    output_dir: Path,
    file_types: List[str] = None,
    max_files: int = None
) -> dict:
    """Download files from manifest."""
    if file_types is None:
        file_types = ["bed narrowPeak"]  # Start with just peaks

    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = []
    skipped = []

    file_count = 0

    for tf, tf_data in manifest["tfs"].items():
        tf_dir = output_dir / tf
        tf_dir.mkdir(exist_ok=True)

        for f in tf_data["files"]:
            if max_files and file_count >= max_files:
                break

            if f["file_type"] not in file_types:
                continue

            # Construct filename
            filename = f"{f['accession']}_{f['experiment']}.{f['file_type'].replace(' ', '_')}"
            if f["file_type"] == "bed narrowPeak":
                filename = f"{f['accession']}_{f['experiment']}.narrowPeak.gz"
            elif f["file_type"] == "bigWig":
                filename = f"{f['accession']}_{f['experiment']}.bigWig"

            dest_path = tf_dir / filename

            if dest_path.exists():
                skipped.append(str(dest_path))
                continue

            desc = f"{tf}/{f['accession']}"
            success = download_file(f["href"], dest_path, desc)

            if success:
                downloaded.append(str(dest_path))
            else:
                failed.append(f["href"])

            file_count += 1

        if max_files and file_count >= max_files:
            break

    return {
        "downloaded": downloaded,
        "failed": failed,
        "skipped": skipped,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download ENCODE ChIP-seq data")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path(__file__).parent.parent / "data" / "raw" / "encode")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only create manifest, don't download")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum files to download (for testing)")
    parser.add_argument("--max-tfs", type=int, default=None,
                        help="Maximum TFs to query (for testing)")
    parser.add_argument("--tfs", nargs="+", default=None,
                        help="Specific TFs to download")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    tfs = args.tfs if args.tfs else PRIORITY_TFS[:args.max_tfs] if args.max_tfs else PRIORITY_TFS

    print(f"Output directory: {output_dir}")
    print(f"Target TFs: {tfs}")
    print()

    # Create manifest
    manifest_path = output_dir / "manifest.json"

    print("Creating download manifest...")
    manifest = create_encode_manifest(output_dir, tfs=tfs)

    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    # Summary
    print()
    print("=" * 60)
    print("Manifest Summary:")
    print(f"  TFs found: {len(manifest['tfs'])}")
    print(f"  Total files: {manifest['total_files']}")
    print(f"  Total size: {manifest['total_size_bytes'] / 1e9:.2f} GB")
    print()

    for tf, data in manifest["tfs"].items():
        print(f"  {tf}: {len(data['experiments'])} experiments, {len(data['files'])} files")

    if args.manifest_only:
        print("\nManifest-only mode, skipping download.")
        return 0

    # Download files
    print()
    print("Downloading files...")
    results = download_from_manifest(
        manifest,
        output_dir,
        max_files=args.max_files
    )

    print()
    print("=" * 60)
    print("Download Summary:")
    print(f"  Downloaded: {len(results['downloaded'])}")
    print(f"  Skipped (existing): {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")

    if results['failed']:
        print("\nFailed downloads:")
        for url in results['failed']:
            print(f"  {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
