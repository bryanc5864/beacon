#!/usr/bin/env python3
"""
Master script for BEACON data preparation.

This script orchestrates:
1. Downloading JASPAR motifs
2. Downloading ENCODE ChIP-seq data
3. Downloading reference genome
4. Processing all data into training-ready format

Usage:
    python scripts/prepare_data.py --all           # Download and process everything
    python scripts/prepare_data.py --download-only # Just download
    python scripts/prepare_data.py --process-only  # Just process (assumes downloads exist)
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_jaspar(data_dir: Path) -> bool:
    """Download JASPAR motif database."""
    print("\n" + "=" * 60)
    print("STEP 1: Downloading JASPAR 2024 motifs")
    print("=" * 60)

    from scripts.download_jaspar import download_jaspar_motifs
    output_dir = data_dir / "raw" / "jaspar"

    try:
        results = download_jaspar_motifs(output_dir)
        success = sum(1 for v in results.values() if v is not None)
        print(f"\nDownloaded {success}/{len(results)} files")
        return success > 0
    except Exception as e:
        print(f"Error downloading JASPAR: {e}")
        return False


def download_encode(data_dir: Path, max_tfs: int = 3, max_files: int = 10) -> bool:
    """Download ENCODE ChIP-seq data."""
    print("\n" + "=" * 60)
    print("STEP 2: Downloading ENCODE ChIP-seq data")
    print("=" * 60)

    from scripts.download_encode import create_encode_manifest, download_from_manifest, PRIORITY_TFS

    output_dir = data_dir / "raw" / "encode"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create manifest for subset of TFs
        tfs = PRIORITY_TFS[:max_tfs]
        print(f"Querying ENCODE for: {tfs}")

        manifest = create_encode_manifest(output_dir, tfs=tfs)

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\nFound {manifest['total_files']} files across {len(manifest['tfs'])} TFs")

        # Download a subset of files
        print(f"\nDownloading up to {max_files} files...")
        results = download_from_manifest(manifest, output_dir, max_files=max_files)

        print(f"Downloaded: {len(results['downloaded'])}")
        print(f"Skipped: {len(results['skipped'])}")
        print(f"Failed: {len(results['failed'])}")

        return len(results['downloaded']) > 0 or len(results['skipped']) > 0

    except Exception as e:
        print(f"Error downloading ENCODE: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_genome(data_dir: Path) -> bool:
    """Download reference genome."""
    print("\n" + "=" * 60)
    print("STEP 3: Downloading reference genome (hg38)")
    print("=" * 60)
    print("NOTE: This downloads ~3GB and may take a while")

    from scripts.download_genome import download_genome as _download_genome

    output_dir = data_dir / "raw" / "genome"

    try:
        results = _download_genome(output_dir)
        return "genome" in results
    except Exception as e:
        print(f"Error downloading genome: {e}")
        return False


def process_jaspar_motifs(data_dir: Path) -> bool:
    """Process JASPAR motifs into training-ready format."""
    print("\n" + "=" * 60)
    print("STEP 4: Processing JASPAR motifs")
    print("=" * 60)

    try:
        from scripts.process_jaspar import main as process_jaspar_main
        return process_jaspar_main() == 0
    except Exception as e:
        print(f"Error processing JASPAR: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_chipseq_data(data_dir: Path) -> bool:
    """Process ChIP-seq peaks into training regions."""
    print("\n" + "=" * 60)
    print("STEP 5: Processing ChIP-seq data")
    print("=" * 60)

    try:
        from scripts.process_chipseq import main as process_chipseq_main
        return process_chipseq_main() == 0
    except Exception as e:
        print(f"Error processing ChIP-seq: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for BEACON training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py --all           # Full download and processing
  python scripts/prepare_data.py --download-only # Just download raw data
  python scripts/prepare_data.py --process-only  # Just process existing data
        """
    )

    parser.add_argument("--all", action="store_true",
                        help="Download and process all data")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download data, don't process")
    parser.add_argument("--process-only", action="store_true",
                        help="Only process data, don't download")
    parser.add_argument("--skip-genome", action="store_true",
                        help="Skip genome download (large file)")
    parser.add_argument("--max-tfs", type=int, default=5,
                        help="Maximum TFs to download from ENCODE")
    parser.add_argument("--max-files", type=int, default=20,
                        help="Maximum files to download per TF")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent / "data",
                        help="Base data directory")

    args = parser.parse_args()

    # Default to --all if no options specified
    if not any([args.all, args.download_only, args.process_only]):
        args.all = True

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEACON Data Preparation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    results = {}

    # Download steps
    if args.all or args.download_only:
        results['jaspar_download'] = download_jaspar(data_dir)
        results['encode_download'] = download_encode(data_dir, args.max_tfs, args.max_files)

        if not args.skip_genome:
            results['genome_download'] = download_genome(data_dir)
        else:
            print("\nSkipping genome download (--skip-genome)")

    # Processing steps
    if args.all or args.process_only:
        results['jaspar_process'] = process_jaspar_motifs(data_dir)
        results['chipseq_process'] = process_chipseq_data(data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for step, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {step}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
