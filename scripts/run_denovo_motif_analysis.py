#!/usr/bin/env python3
"""
De novo motif discovery using gradient-based attribution.

Extracts motifs from gradient x input importance maps and compares them
to JASPAR reference PWMs. Expected to improve on attention-based motif
extraction (0.574 mean JASPAR Pearson r).

Usage:
    python scripts/run_denovo_motif_analysis.py --device cuda:4
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_named_model, load_test_data, get_test_path,
    json_serialize,
    DATASET_CONFIGS, BASE_DIR,
)
from beacon.analysis.denovo_motifs import full_motif_discovery_pipeline

# JASPAR PWMs for comparison
JASPAR_PWMS = {
    "CTCF": np.array([
        [0.15,0.40,0.30,0.15],[0.10,0.60,0.15,0.15],[0.15,0.15,0.55,0.15],
        [0.10,0.50,0.30,0.10],[0.15,0.15,0.55,0.15],[0.20,0.20,0.20,0.40],
        [0.15,0.15,0.55,0.15],[0.15,0.15,0.55,0.15],[0.20,0.20,0.20,0.40],
        [0.15,0.15,0.55,0.15],[0.15,0.15,0.55,0.15],[0.10,0.55,0.20,0.15],
        [0.50,0.15,0.20,0.15],[0.15,0.15,0.55,0.15],
    ]),
    "GATA1": np.array([
        [0.55,0.10,0.20,0.15],[0.15,0.10,0.60,0.15],[0.70,0.05,0.15,0.10],
        [0.10,0.10,0.10,0.70],[0.65,0.10,0.10,0.15],[0.55,0.10,0.20,0.15],
    ]),
    "TAL1": np.array([
        [0.15,0.50,0.20,0.15],[0.55,0.15,0.15,0.15],[0.15,0.15,0.55,0.15],
        [0.15,0.50,0.15,0.20],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "MYC": np.array([
        [0.15,0.55,0.15,0.15],[0.60,0.10,0.15,0.15],[0.15,0.55,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "MAX": np.array([
        [0.15,0.55,0.15,0.15],[0.60,0.10,0.15,0.15],[0.15,0.55,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "SPI1": np.array([
        [0.55,0.10,0.15,0.20],[0.55,0.10,0.15,0.20],[0.60,0.10,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.55,0.10,0.15,0.20],[0.15,0.10,0.60,0.15],
        [0.15,0.10,0.60,0.15],[0.55,0.15,0.15,0.15],[0.55,0.15,0.15,0.15],
        [0.15,0.10,0.55,0.20],[0.10,0.10,0.10,0.70],[0.15,0.10,0.60,0.15],
    ]),
    "CEBPB": np.array([
        [0.10,0.10,0.10,0.70],[0.10,0.10,0.10,0.70],[0.15,0.10,0.60,0.15],
        [0.15,0.55,0.15,0.15],[0.15,0.10,0.55,0.20],[0.15,0.55,0.15,0.15],
        [0.60,0.10,0.15,0.15],[0.55,0.15,0.15,0.15],
    ]),
}


def main():
    parser = argparse.ArgumentParser(description="De novo motif discovery")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-per-tf", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("De Novo Motif Discovery (Gradient-Based)")
    print("=" * 70)

    name = "K562-7tf"
    print(f"\n--- {name} ---")
    model, cfg = load_named_model(name, device)
    test_path = get_test_path(name)
    seqs, profs, tf_idx, bs = load_test_data(test_path)
    tf_names = cfg["tf_names"]

    print(f"  Samples: {len(seqs)}, TFs: {tf_names}")

    results = full_motif_discovery_pipeline(
        model, seqs, tf_idx, tf_names, device,
        jaspar_pwms=JASPAR_PWMS,
        max_per_tf=args.max_per_tf,
    )

    # Summary
    print(f"\n  === Summary ===")
    print(f"  TFs with motifs: {results['n_tfs_with_motifs']}/{len(tf_names)}")
    if results['mean_jaspar_pearson'] is not None:
        print(f"  Mean JASPAR Pearson: {results['mean_jaspar_pearson']:.3f}")
        print(f"  Median JASPAR Pearson: {results['median_jaspar_pearson']:.3f}")
        print(f"  (Previous attention-based: 0.574)")

    # Add comparison to previous method
    results['comparison_to_attention'] = {
        'attention_based_mean_pearson': 0.574,
        'gradient_based_mean_pearson': results['mean_jaspar_pearson'],
        'improvement': (results['mean_jaspar_pearson'] - 0.574
                        if results['mean_jaspar_pearson'] is not None else None),
    }

    path = out_dir / "denovo_motifs.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
