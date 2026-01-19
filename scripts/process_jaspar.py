#!/usr/bin/env python3
"""
Process JASPAR motifs into training-ready format for BEACON.

Creates:
- PWM embeddings for slot initialization
- TF family clustering
- Motif metadata index
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_jaspar import parse_jaspar_pfm


def pfm_to_pwm(pfm: np.ndarray, pseudocount: float = 0.01) -> np.ndarray:
    """Convert Position Frequency Matrix to Position Weight Matrix."""
    background = np.array([0.25, 0.25, 0.25, 0.25])
    row_sums = pfm.sum(axis=1, keepdims=True)
    ppm = (pfm + pseudocount) / (row_sums + 4 * pseudocount)
    pwm = np.log2(ppm / background)
    return pwm


def pwm_to_embedding(pwm: np.ndarray, max_length: int = 30, embedding_dim: int = 128) -> np.ndarray:
    """Convert PWM to fixed-size embedding vector."""
    length = len(pwm)

    # Pad or truncate
    if length > max_length:
        pwm = pwm[:max_length]
    elif length < max_length:
        padding = np.zeros((max_length - length, 4))
        pwm = np.vstack([pwm, padding])

    # Flatten and project
    flat = pwm.flatten()  # [max_length * 4]

    # Learned projection would be here; for now use deterministic random projection
    np.random.seed(42)
    projection = np.random.randn(len(flat), embedding_dim).astype(np.float32)
    projection /= np.sqrt(len(flat))

    embedding = (flat @ projection).astype(np.float32)
    return embedding


def get_information_content(pwm: np.ndarray) -> float:
    """Calculate total information content of PWM."""
    # Convert to probabilities
    probs = 2 ** pwm * 0.25
    probs = np.clip(probs, 1e-10, 1)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Information content
    ic = 0
    for i in range(len(probs)):
        for j in range(4):
            if probs[i, j] > 0:
                ic += probs[i, j] * np.log2(probs[i, j] / 0.25)
    return ic


def main():
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw" / "jaspar"
    processed_dir = data_dir / "processed" / "jaspar"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Parse PFMs
    pfm_path = raw_dir / "pfm_vertebrates.txt"
    print(f"Loading PFMs from {pfm_path}")
    motifs = parse_jaspar_pfm(pfm_path)
    print(f"Parsed {len(motifs)} motifs")

    # Load metadata for TF family info
    metadata_path = raw_dir / "metadata.json"
    tf_families = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            for entry in meta.get('results', []):
                matrix_id = entry.get('matrix_id', '')
                family = entry.get('family', ['Unknown'])
                if isinstance(family, list):
                    family = family[0] if family else 'Unknown'
                tf_families[matrix_id] = family

    # Process each motif
    motif_ids = []
    motif_names = []
    pwm_list = []
    embeddings = []
    lengths = []
    ic_scores = []
    families = []
    consensuses = []

    print("Processing motifs...")
    for motif_id, data in motifs.items():
        pfm = data['matrix']
        pwm = pfm_to_pwm(pfm)
        embedding = pwm_to_embedding(pwm)
        ic = get_information_content(pwm)

        # Consensus sequence
        consensus = "".join("ACGT"[np.argmax(pwm[i])] for i in range(len(pwm)))

        motif_ids.append(motif_id)
        motif_names.append(data['name'])
        pwm_list.append(pwm)
        embeddings.append(embedding)
        lengths.append(len(pwm))
        ic_scores.append(ic)
        families.append(tf_families.get(motif_id.split('.')[0], 'Unknown'))
        consensuses.append(consensus)

    # Stack embeddings
    embeddings = np.stack(embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # Pad PWMs to same length for storage
    max_len = max(lengths)
    padded_pwms = np.zeros((len(pwm_list), max_len, 4), dtype=np.float32)
    for i, pwm in enumerate(pwm_list):
        padded_pwms[i, :len(pwm)] = pwm

    # Save to HDF5
    output_path = processed_dir / "jaspar_processed.h5"
    print(f"\nSaving to {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Core data
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        f.create_dataset('pwms', data=padded_pwms, compression='gzip')
        f.create_dataset('lengths', data=np.array(lengths))
        f.create_dataset('ic_scores', data=np.array(ic_scores))

        # String data
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('motif_ids', data=np.array(motif_ids, dtype=object), dtype=dt)
        f.create_dataset('motif_names', data=np.array(motif_names, dtype=object), dtype=dt)
        f.create_dataset('families', data=np.array(families, dtype=object), dtype=dt)
        f.create_dataset('consensuses', data=np.array(consensuses, dtype=object), dtype=dt)

    # Save summary JSON
    summary = {
        'n_motifs': len(motif_ids),
        'embedding_dim': embeddings.shape[1],
        'max_length': max_len,
        'family_counts': {},
    }
    for fam in families:
        summary['family_counts'][fam] = summary['family_counts'].get(fam, 0) + 1

    summary_path = processed_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessed {len(motif_ids)} motifs")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Max motif length: {max_len}")
    print(f"\nTop TF families:")
    for fam, count in sorted(summary['family_counts'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {fam}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
