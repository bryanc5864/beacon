#!/usr/bin/env python3
"""
Download reference genome (hg38/GRCh38) for BEACON.

We download the primary assembly from UCSC or NCBI and create
the necessary index files for fast sequence extraction.
"""

import os
import sys
import subprocess
import gzip
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

# Genome URLs
GENOME_URLS = {
    # UCSC hg38 - smaller, no alt contigs
    "hg38_ucsc": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    # Chromosome sizes
    "hg38_chrom_sizes": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
    # Blacklist regions (ENCODE)
    "blacklist": "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz",
}


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=300)
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


def decompress_gzip(gz_path: Path, output_path: Path) -> bool:
    """Decompress a gzip file."""
    try:
        print(f"Decompressing {gz_path.name}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"Error decompressing: {e}")
        return False


def create_fasta_index(fasta_path: Path) -> bool:
    """Create .fai index using samtools or pyfaidx."""
    fai_path = Path(str(fasta_path) + ".fai")

    if fai_path.exists():
        print(f"Index already exists: {fai_path}")
        return True

    # Try samtools first
    try:
        result = subprocess.run(
            ["samtools", "faidx", str(fasta_path)],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            print(f"Created index with samtools: {fai_path}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fall back to pyfaidx
    try:
        import pyfaidx
        print("Creating index with pyfaidx...")
        pyfaidx.Faidx(str(fasta_path))
        print(f"Created index: {fai_path}")
        return True
    except ImportError:
        print("pyfaidx not installed, skipping index creation")
        return False
    except Exception as e:
        print(f"Error creating index: {e}")
        return False


def download_genome(output_dir: Path) -> dict:
    """Download genome and create index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Download genome
    genome_gz = output_dir / "hg38.fa.gz"
    genome_fa = output_dir / "hg38.fa"

    if genome_fa.exists():
        print(f"Genome already exists: {genome_fa}")
        results["genome"] = str(genome_fa)
    else:
        print("Downloading hg38 reference genome (~1GB compressed)...")
        print("This may take several minutes...")

        if not genome_gz.exists():
            success = download_file(GENOME_URLS["hg38_ucsc"], genome_gz, desc="hg38.fa.gz")
            if not success:
                return results

        # Decompress
        if decompress_gzip(genome_gz, genome_fa):
            results["genome"] = str(genome_fa)
            # Optionally remove compressed file to save space
            # genome_gz.unlink()

    # Create index
    if "genome" in results:
        create_fasta_index(Path(results["genome"]))

    # Download chromosome sizes
    chrom_sizes = output_dir / "hg38.chrom.sizes"
    if chrom_sizes.exists():
        print(f"Chrom sizes already exists: {chrom_sizes}")
    else:
        print("Downloading chromosome sizes...")
        download_file(GENOME_URLS["hg38_chrom_sizes"], chrom_sizes, desc="chrom.sizes")
    results["chrom_sizes"] = str(chrom_sizes)

    # Download blacklist
    blacklist_gz = output_dir / "hg38-blacklist.v2.bed.gz"
    blacklist_bed = output_dir / "hg38-blacklist.v2.bed"

    if blacklist_bed.exists():
        print(f"Blacklist already exists: {blacklist_bed}")
    else:
        print("Downloading ENCODE blacklist regions...")
        if download_file(GENOME_URLS["blacklist"], blacklist_gz, desc="blacklist"):
            decompress_gzip(blacklist_gz, blacklist_bed)
    results["blacklist"] = str(blacklist_bed)

    return results


def main():
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "raw" / "genome"

    # Allow override from command line
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    print(f"Output directory: {output_dir}")
    print()

    results = download_genome(output_dir)

    print()
    print("=" * 60)
    print("Download Summary:")
    for name, path in results.items():
        if path and Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            print(f"  {name}: {path} ({size_mb:.1f} MB)")

    print()
    print("Genome download complete!")

    # Verify we can load the genome
    genome_path = results.get("genome")
    if genome_path and Path(genome_path).exists():
        try:
            import pyfaidx
            print("\nValidating genome file...")
            fasta = pyfaidx.Fasta(genome_path)
            print(f"  Chromosomes: {len(fasta.keys())}")
            print(f"  Example: chr1 length = {len(fasta['chr1'])}")
            print("  Validation successful!")
        except ImportError:
            print("\npyfaidx not installed, skipping validation")
        except Exception as e:
            print(f"\nValidation failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
