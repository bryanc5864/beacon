#!/usr/bin/env python3
"""
Download JASPAR 2024 motif database.

JASPAR provides the PWM vocabulary for slot initialization in BEACON.
We download all vertebrate transcription factor motifs in multiple formats.
"""

import os
import sys
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

# JASPAR 2024 URLs
JASPAR_BASE = "https://jaspar.elixir.no/download/data/2024"
JASPAR_URLS = {
    # Position Frequency Matrices (PFM) - raw counts
    "pfm_vertebrates": f"{JASPAR_BASE}/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    # Position Weight Matrices (PWM) - log-odds scores
    "pwm_vertebrates": f"{JASPAR_BASE}/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    # MEME format for compatibility with other tools
    "meme_vertebrates": f"{JASPAR_BASE}/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
    # Metadata with TF family annotations
    "metadata": "https://jaspar.elixir.no/api/v1/matrix/?tax_group=vertebrates&collection=CORE&version=latest&page_size=2000&format=json",
}


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
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


def download_jaspar_motifs(output_dir: Path) -> dict:
    """Download all JASPAR motif files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("Downloading JASPAR 2024 motif database...")
    print("=" * 60)

    for name, url in JASPAR_URLS.items():
        ext = ".json" if "json" in url else ".txt"
        dest_path = output_dir / f"{name}{ext}"

        if dest_path.exists():
            print(f"  {name}: already exists, skipping")
            results[name] = str(dest_path)
            continue

        print(f"  Downloading {name}...")
        success = download_file(url, dest_path, desc=name)

        if success:
            results[name] = str(dest_path)
            print(f"    Saved to {dest_path}")
        else:
            results[name] = None
            print(f"    FAILED")

    return results


def parse_jaspar_pfm(filepath: Path) -> dict:
    """
    Parse JASPAR format PFM file into dictionary.

    Returns dict of {motif_id: {"name": str, "matrix": np.array}}
    """
    import numpy as np

    motifs = {}
    current_id = None
    current_name = None
    current_matrix = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous motif
                if current_id and current_matrix:
                    matrix = np.array(current_matrix).T  # Transpose to [length, 4]
                    motifs[current_id] = {
                        "name": current_name,
                        "matrix": matrix
                    }

                # Parse header: >MA0001.1 AGL3
                parts = line[1:].split()
                current_id = parts[0]
                current_name = parts[1] if len(parts) > 1 else current_id
                current_matrix = []
            else:
                # Parse matrix row: A [ 0 3 79 40 ...]
                if '[' in line:
                    # Extract numbers between brackets
                    start = line.index('[') + 1
                    end = line.index(']')
                    values = [float(x) for x in line[start:end].split()]
                    current_matrix.append(values)

    # Don't forget last motif
    if current_id and current_matrix:
        matrix = np.array(current_matrix).T
        motifs[current_id] = {
            "name": current_name,
            "matrix": matrix
        }

    return motifs


def main():
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "raw" / "jaspar"

    # Allow override from command line
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    print(f"Output directory: {output_dir}")
    print()

    # Download files
    results = download_jaspar_motifs(output_dir)

    # Summary
    print()
    print("=" * 60)
    print("Download Summary:")
    successful = sum(1 for v in results.values() if v is not None)
    print(f"  Successfully downloaded: {successful}/{len(results)}")

    # Parse and validate
    pfm_path = output_dir / "pfm_vertebrates.txt"
    if pfm_path.exists():
        print()
        print("Validating PFM file...")
        try:
            motifs = parse_jaspar_pfm(pfm_path)
            print(f"  Parsed {len(motifs)} motifs")

            # Show some examples
            print("  Example motifs:")
            for i, (motif_id, data) in enumerate(list(motifs.items())[:5]):
                shape = data['matrix'].shape
                print(f"    {motif_id}: {data['name']} (length={shape[0]})")
        except Exception as e:
            print(f"  Validation failed: {e}")

    print()
    print("JASPAR download complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
