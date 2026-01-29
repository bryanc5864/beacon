#!/usr/bin/env python3
"""
Download multi-TF ChIP-seq data from ENCODE for K562.

Phase 2B: Multi-TF discrimination.

Downloads narrowPeak + bigWig for 7 TFs:
- CTCF (zinc finger) - already downloaded, will skip
- GATA1 (GATA family)
- TAL1 (bHLH)
- MYC (bHLH - same family as TAL1)
- MAX (bHLH - dimerizes with MYC)
- SPI1 (ETS family)
- CEBPB (bZIP family)

Key discrimination tests:
- GATA1 vs MYC (different families - should be easy)
- MYC vs MAX (same family - should be hard)
- TAL1 vs MYC (both bHLH - intermediate)
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

ENCODE_BASE_URL = "https://www.encodeproject.org"
HEADERS = {"accept": "application/json"}

# Phase 2B TF panel
TF_PANEL = [
    "CTCF",   # Already downloaded
    "GATA1",
    "TAL1",
    "MYC",
    "MAX",
    "SPI1",
    "CEBPB",
]

CELL_LINE = "K562"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "encode" / "multi_tf_k562"

# Logging
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def query_encode(url, params=None):
    """Query ENCODE REST API."""
    full_url = f"{ENCODE_BASE_URL}{url}"
    try:
        resp = requests.get(full_url, headers=HEADERS, params=params, timeout=60)
        resp.raise_for_status()
        time.sleep(0.2)
        return resp.json()
    except Exception as e:
        log(f"  ERROR querying {full_url}: {e}")
        return {}


def find_best_files(tf_name, cell_line="K562"):
    """Find the best narrowPeak and bigWig for a TF in a cell line."""
    log(f"  Searching ENCODE for {tf_name} in {cell_line}...")

    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "target.label": tf_name,
        "biosample_ontology.term_name": cell_line,
        "assembly": "GRCh38",
        "status": "released",
        "format": "json",
        "limit": "all",
    }

    result = query_encode("/search/", params)
    experiments = result.get("@graph", [])
    log(f"  Found {len(experiments)} experiments")

    if not experiments:
        return None, None, None

    best_peak = None
    best_signal = None
    best_exp = None

    for exp in experiments:
        acc = exp.get("accession", "")
        biosample = exp.get("biosample_summary", "")
        log(f"    Checking {acc}: {biosample}")

        # Fetch experiment details
        exp_data = query_encode(f"/experiments/{acc}/?format=json")
        if not exp_data:
            continue

        peak_candidates = []
        signal_candidates = []

        for f in exp_data.get("files", []):
            if isinstance(f, str):
                try:
                    f = query_encode(f"{f}?format=json")
                except:
                    continue
            if not isinstance(f, dict):
                continue

            if f.get("status") != "released":
                continue
            if f.get("assembly") != "GRCh38":
                continue

            file_type = f.get("file_type", "")
            output_type = f.get("output_type", "")

            if file_type == "bed narrowPeak":
                peak_candidates.append({
                    "accession": f.get("accession", ""),
                    "output_type": output_type,
                    "href": f.get("href", ""),
                    "file_size": f.get("file_size", 0),
                })

            elif file_type == "bigWig":
                signal_candidates.append({
                    "accession": f.get("accession", ""),
                    "output_type": output_type,
                    "href": f.get("href", ""),
                    "file_size": f.get("file_size", 0),
                })

        # Select best peak: prefer optimal IDR > IDR > conservative IDR > any
        for p in peak_candidates:
            ot = p["output_type"].lower()
            if "optimal" in ot and "idr" in ot:
                best_peak = p
                best_exp = acc
                break
        if not best_peak:
            for p in peak_candidates:
                ot = p["output_type"].lower()
                if "idr" in ot and "conservative" not in ot:
                    best_peak = p
                    best_exp = acc
                    break
        if not best_peak and peak_candidates:
            best_peak = peak_candidates[0]
            best_exp = acc

        # Select best signal: prefer fold change (smaller) > signal p-value
        for s in signal_candidates:
            ot = s["output_type"].lower()
            if "fold" in ot and "change" in ot:
                best_signal = s
                break
        if not best_signal:
            for s in signal_candidates:
                ot = s["output_type"].lower()
                if "signal" in ot:
                    best_signal = s
                    break
        if not best_signal and signal_candidates:
            best_signal = signal_candidates[0]

        if best_peak and best_signal:
            break

    if best_peak:
        log(f"  Peak: {best_peak['accession']} ({best_peak['output_type']}, {best_peak['file_size']/1e6:.1f} MB)")
    else:
        log(f"  Peak: NOT FOUND")

    if best_signal:
        log(f"  Signal: {best_signal['accession']} ({best_signal['output_type']}, {best_signal['file_size']/1e6:.1f} MB)")
    else:
        log(f"  Signal: NOT FOUND")

    return best_peak, best_signal, best_exp


def download_file(href, dest_path, desc=""):
    """Download file from ENCODE."""
    url = f"{ENCODE_BASE_URL}{href}" if href.startswith("/") else href
    log(f"  Downloading {desc}: {dest_path.name}")

    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 65536:
                    pct = 100 * downloaded / total
                    log(f"    {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)")

        log(f"  Done: {dest_path.name} ({dest_path.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:
        log(f"  FAILED: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ENCODE Multi-TF ChIP-seq Download (K562) - Phase 2B")
    log("=" * 70)
    log(f"TFs: {', '.join(TF_PANEL)}")
    log(f"Cell line: {CELL_LINE}")
    log(f"Output: {OUTPUT_DIR}")
    log("")

    manifest = {}
    summary = []

    for tf in TF_PANEL:
        log(f"\n{'='*50}")
        log(f"TF: {tf}")
        log(f"{'='*50}")

        tf_dir = OUTPUT_DIR / tf
        tf_dir.mkdir(exist_ok=True)

        # Check if already downloaded (e.g., CTCF)
        existing_peaks = list(tf_dir.glob("*.narrowPeak*"))
        existing_signals = list(tf_dir.glob("*.bigWig")) + list(tf_dir.glob("*.bw"))

        if existing_peaks and existing_signals:
            log(f"  Already have {len(existing_peaks)} peak file(s) and {len(existing_signals)} signal file(s). Skipping.")
            manifest[tf] = {
                "status": "existing",
                "peak_file": str(existing_peaks[0]),
                "signal_file": str(existing_signals[0]),
            }
            summary.append(f"  {tf}: EXISTING ({existing_peaks[0].name})")
            continue

        # Also check the CTCF_K562 directory
        if tf == "CTCF":
            ctcf_dir = OUTPUT_DIR.parent / "CTCF_K562"
            ctcf_peaks = list(ctcf_dir.glob("*.narrowPeak*"))
            ctcf_signals = list(ctcf_dir.glob("*.bigWig"))
            if ctcf_peaks and ctcf_signals:
                # Symlink to multi_tf directory
                import shutil
                for p in ctcf_peaks:
                    dest = tf_dir / p.name
                    if not dest.exists():
                        os.symlink(p, dest)
                for s in ctcf_signals:
                    dest = tf_dir / s.name
                    if not dest.exists():
                        os.symlink(s, dest)
                log(f"  Linked CTCF data from {ctcf_dir}")
                manifest[tf] = {
                    "status": "linked",
                    "peak_file": str(ctcf_peaks[0]),
                    "signal_file": str(ctcf_signals[0]),
                }
                summary.append(f"  {tf}: LINKED from CTCF_K562")
                continue

        # Find best files from ENCODE
        best_peak, best_signal, exp_acc = find_best_files(tf, CELL_LINE)

        tf_manifest = {"experiment": exp_acc, "status": "downloaded"}

        # Download peak file
        if best_peak:
            peak_path = tf_dir / f"{best_peak['accession']}_peaks.narrowPeak.gz"
            if not peak_path.exists():
                success = download_file(best_peak["href"], peak_path, f"{tf} peaks")
                if not success:
                    tf_manifest["peak_status"] = "failed"
            else:
                log(f"  Peak file exists: {peak_path.name}")
            tf_manifest["peak_file"] = str(peak_path)
            tf_manifest["peak_accession"] = best_peak["accession"]
        else:
            tf_manifest["peak_status"] = "not_found"

        # Download signal file
        if best_signal:
            signal_path = tf_dir / f"{best_signal['accession']}_signal.bigWig"
            if not signal_path.exists():
                success = download_file(best_signal["href"], signal_path, f"{tf} signal")
                if not success:
                    tf_manifest["signal_status"] = "failed"
            else:
                log(f"  Signal file exists: {signal_path.name}")
            tf_manifest["signal_file"] = str(signal_path)
            tf_manifest["signal_accession"] = best_signal["accession"]
        else:
            tf_manifest["signal_status"] = "not_found"

        manifest[tf] = tf_manifest

        has_peak = "peak_file" in tf_manifest and "peak_status" not in tf_manifest
        has_signal = "signal_file" in tf_manifest and "signal_status" not in tf_manifest
        status = "OK" if has_peak and has_signal else "PARTIAL" if has_peak or has_signal else "FAILED"
        summary.append(f"  {tf}: {status} (peak={best_peak['accession'] if best_peak else 'NONE'}, signal={best_signal['accession'] if best_signal else 'NONE'})")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    log("")
    log("=" * 70)
    log("DOWNLOAD SUMMARY")
    log("=" * 70)
    for line in summary:
        log(line)
    log("")
    log(f"Manifest: {manifest_path}")
    log("")
    log("Next step:")
    log("  python scripts/prepare_multi_tf_data.py")
    log("=" * 70)


if __name__ == "__main__":
    main()
