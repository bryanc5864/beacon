#!/usr/bin/env python3
"""
Download HepG2 ChIP-seq data for cross-cell-line transfer experiment.

Downloads the same 7 TFs as K562 from ENCODE:
- CTCF, GATA1, TAL1, MYC, MAX, SPI1, CEBPB

Note: Some TFs may not be available in HepG2, will use what's available.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# ENCODE portal base URL
ENCODE_BASE = "https://www.encodeproject.org"

# Target TFs - same as K562 experiment
TARGET_TFS = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]

# Alternative cell lines to try (in order of preference)
CELL_LINES = ["HepG2", "GM12878", "H1-hESC", "HeLa-S3"]

# ENCODE experiment accessions for HepG2 (manually curated for reliability)
# Format: {TF: {"peaks": accession, "signal": accession}}
HEPG2_EXPERIMENTS = {
    "CTCF": {
        "experiment": "ENCSR000BMY",
        "peaks": "ENCFF660GHM",  # IDR thresholded peaks
        "signal": "ENCFF473IZV",  # fold change over control
    },
    "MYC": {
        "experiment": "ENCSR000EFJ",
        "peaks": "ENCFF632NVQ",
        "signal": "ENCFF795SLI",
    },
    "MAX": {
        "experiment": "ENCSR000BQT",
        "peaks": "ENCFF148VLC",
        "signal": "ENCFF653VNQ",
    },
    "CEBPB": {
        "experiment": "ENCSR000BQY",
        "peaks": "ENCFF782GBP",
        "signal": "ENCFF143JGT",
    },
    # Note: GATA1, TAL1, SPI1 are not typically expressed in HepG2 (liver)
    # They are hematopoietic TFs, so we'll use what's available
}

# GM12878 experiments (B-lymphoblastoid, has more TFs)
GM12878_EXPERIMENTS = {
    "CTCF": {
        "experiment": "ENCSR000AKB",
        "peaks": "ENCFF710VEH",
        "signal": "ENCFF831YFZ",
    },
    "MYC": {
        "experiment": "ENCSR000DKU",
        "peaks": "ENCFF133HPN",
        "signal": "ENCFF641LXN",
    },
    "MAX": {
        "experiment": "ENCSR000BMN",
        "peaks": "ENCFF812TAI",
        "signal": "ENCFF870OLV",
    },
    "SPI1": {
        "experiment": "ENCSR000BGW",
        "peaks": "ENCFF519CWX",
        "signal": "ENCFF529MXQ",
    },
    # GATA1, TAL1, CEBPB not available in GM12878
}


def download_file(url, output_path, description=""):
    """Download a file using wget."""
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return True

    print(f"  Downloading {description}: {output_path.name}")
    cmd = ["wget", "-q", "-O", str(output_path), url]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.decode()[:100]}")
        return False
    return True


def download_encode_file(accession, output_dir, file_type="bed"):
    """Download a file from ENCODE by accession."""
    # Construct download URL
    if file_type == "bed":
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bed.gz"
        output_path = output_dir / f"{accession}.bed.gz"
    elif file_type == "bigWig":
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bigWig"
        output_path = output_dir / f"{accession}.bigWig"
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    return download_file(url, output_path, file_type)


def main():
    # Setup directories
    base_dir = Path(__file__).parent.parent

    # Try HepG2 first, then GM12878
    for cell_line, experiments in [("HepG2", HEPG2_EXPERIMENTS),
                                    ("GM12878", GM12878_EXPERIMENTS)]:
        print(f"\n{'='*60}")
        print(f"Downloading {cell_line} ChIP-seq data")
        print(f"{'='*60}")

        output_dir = base_dir / "data" / "raw" / "encode" / f"multi_tf_{cell_line.lower()}"
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_tfs = []

        for tf, files in experiments.items():
            print(f"\n{tf}:")

            tf_dir = output_dir / tf
            tf_dir.mkdir(exist_ok=True)

            # Download peaks
            peaks_ok = download_encode_file(files["peaks"], tf_dir, "bed")

            # Download signal
            signal_ok = download_encode_file(files["signal"], tf_dir, "bigWig")

            if peaks_ok and signal_ok:
                downloaded_tfs.append(tf)
                print(f"  ✓ {tf} complete")
            else:
                print(f"  ✗ {tf} failed")

        # Save manifest
        manifest = {
            "cell_line": cell_line,
            "tfs": downloaded_tfs,
            "experiments": {tf: experiments[tf] for tf in downloaded_tfs},
        }

        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{cell_line} Summary:")
        print(f"  Downloaded: {len(downloaded_tfs)}/{len(experiments)} TFs")
        print(f"  TFs: {', '.join(downloaded_tfs)}")

        if len(downloaded_tfs) >= 3:
            print(f"\n  ✓ Sufficient data for cross-cell transfer experiment")
            break

    print(f"\nData saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
