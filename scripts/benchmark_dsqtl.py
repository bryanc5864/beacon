#!/usr/bin/env python3
"""
Benchmark BEACON variant effect predictions on the Lee et al. 2015 dsQTL dataset.

Uses 574 positive dsQTLs + 27,735 negative controls from the deltaSVM benchmark
(GM12878 LCL DNase I sensitivity QTLs). Scores each variant by computing ISM
(ref vs alt profile change) in a 2000bp genomic context window.

Computes AUROC and AUPRC as primary metrics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dsqtl.py
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dsqtl.py --model outputs/improved/K562-7tf_slot_dropout/beacon_*/best_model.pt
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON


def parse_dsqtl_fasta(major_path, minor_path):
    """Parse paired major/minor allele FASTA files from deltaSVM benchmark.

    Returns list of dicts with: chrom, start, end, major_seq, minor_seq, variant_pos
    """
    variants = []

    with open(major_path) as f_maj, open(minor_path) as f_min:
        while True:
            header_maj = f_maj.readline().strip()
            seq_maj = f_maj.readline().strip()
            header_min = f_min.readline().strip()
            seq_min = f_min.readline().strip()

            if not header_maj:
                break

            # Parse header: >chr1:846437-846455
            coord = header_maj[1:]  # remove >
            chrom, positions = coord.split(':')
            start, end = positions.split('-')
            start, end = int(start), int(end)

            # Find the variant position (where major and minor differ)
            variant_pos = None
            for i in range(len(seq_maj)):
                if seq_maj[i] != seq_min[i]:
                    variant_pos = i
                    break

            if variant_pos is None:
                continue

            variants.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'major_seq': seq_maj,
                'minor_seq': seq_min,
                'variant_pos_in_seq': variant_pos,
                'variant_pos_genomic': start + variant_pos,
                'ref': seq_maj[variant_pos],
                'alt': seq_min[variant_pos],
            })

    return variants


def get_genomic_context(chrom, center_pos, genome_fasta, window=2000):
    """Extract 2000bp genomic context around a variant position using pyfaidx."""
    half = window // 2
    start = max(0, center_pos - half)
    end = start + window

    try:
        seq = str(genome_fasta[chrom][start:end])
        if len(seq) < window:
            seq = seq + 'N' * (window - len(seq))
        return seq, center_pos - start
    except (KeyError, ValueError):
        return None, None


def encode_sequence(seq):
    """One-hot encode a DNA sequence [L, 4]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
               'a': 0, 'c': 1, 'g': 2, 't': 3}
    L = len(seq)
    onehot = np.zeros((L, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            onehot[i, mapping[base]] = 1.0
        else:
            onehot[i] = 0.25  # N
    return onehot


def apply_snv(seq_onehot, position, alt_base_idx):
    """Apply SNV to one-hot encoded sequence."""
    alt = seq_onehot.copy()
    alt[position] = 0.0
    alt[position, alt_base_idx] = 1.0
    return alt


@torch.no_grad()
def score_variants_batch(model, ref_seqs, alt_seqs, device, batch_size=32):
    """Score a batch of ref/alt sequence pairs. Returns ISM scores."""
    scores = []

    for i in range(0, len(ref_seqs), batch_size):
        batch_ref = torch.from_numpy(np.stack(ref_seqs[i:i+batch_size])).to(device)
        batch_alt = torch.from_numpy(np.stack(alt_seqs[i:i+batch_size])).to(device)

        ref_out = model(batch_ref)
        alt_out = model(batch_alt)

        ref_profile = ref_out['profile'].cpu().numpy()
        alt_profile = alt_out['profile'].cpu().numpy()
        ref_occ = ref_out['occupancy'].squeeze(-1).cpu().numpy()
        alt_occ = alt_out['occupancy'].squeeze(-1).cpu().numpy()

        for j in range(len(batch_ref)):
            delta_profile = np.abs(alt_profile[j] - ref_profile[j])
            delta_occ = np.abs(alt_occ[j] - ref_occ[j])

            profile_score = float(delta_profile.sum())
            occ_score = float(delta_occ.max())
            local_score = float(delta_profile[900:1100].sum())  # central 200bp

            ism_score = local_score + 10.0 * occ_score
            scores.append({
                'ism_score': ism_score,
                'profile_delta_sum': profile_score,
                'profile_delta_local': local_score,
                'occ_delta_max': occ_score,
            })

    return scores


def compute_metrics(scores, labels):
    """Compute AUROC, AUPRC from scores and binary labels."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    scores = np.array(scores)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    pos_mask = labels == 1
    neg_mask = labels == 0

    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'mean_score_pos': float(scores[pos_mask].mean()),
        'mean_score_neg': float(scores[neg_mask].mean()),
        'median_score_pos': float(np.median(scores[pos_mask])),
        'median_score_neg': float(np.median(scores[neg_mask])),
        'score_ratio': float(scores[pos_mask].mean() / max(scores[neg_mask].mean(), 1e-8)),
        'n_pos': int(pos_mask.sum()),
        'n_neg': int(neg_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark BEACON on dsQTL dataset')
    parser.add_argument('--model', type=str, default=None,
                       help='Model checkpoint path (glob patterns ok)')
    parser.add_argument('--n-tfs', type=int, default=7)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--genome', type=str, default='data/raw/genome/hg38.fa')
    parser.add_argument('--dsqtl-dir', type=str,
                       default='data/benchmarks/dsqtl/gm12878_sequence_sets')
    parser.add_argument('--use-liftover', action='store_true', default=True,
                       help='Liftover hg19→hg38 coordinates')
    parser.add_argument('--no-liftover', dest='use_liftover', action='store_false',
                       help='Use padded 19-mers instead of genomic context')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='outputs/dsqtl_benchmark')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Find model checkpoint
    if args.model is None:
        # Default to best K562-7tf model
        pattern = str(base_dir / 'outputs/improved/K562-7tf_slot_dropout/beacon_*/best_model.pt')
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"No model found at {pattern}")
            return 1
        model_path = matches[-1]
    else:
        matches = sorted(glob.glob(args.model))
        model_path = matches[-1] if matches else args.model

    print(f"{'='*70}")
    print(f"  BEACON dsQTL Benchmark")
    print(f"{'='*70}")
    print(f"  Model: {model_path}")

    # Load model
    model = BEACON(
        seq_len=2000, input_channels=4,
        backbone_type="dilated", backbone_dim=128,
        backbone_layers=4, n_slots=16, slot_dim=128,
        n_iterations=3, n_tfs=args.n_tfs,
        position_mode="gaussian", dropout=0.0,
        attention_mode="independent",
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(args.device).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Parse dsQTL data
    dsqtl_dir = base_dir / args.dsqtl_dir
    pos_variants = parse_dsqtl_fasta(
        dsqtl_dir / 'dsqtl_test_pos.major.fa',
        dsqtl_dir / 'dsqtl_test_pos.minor.fa',
    )
    neg_variants = parse_dsqtl_fasta(
        dsqtl_dir / 'dsqtl_test_neg.major.fa',
        dsqtl_dir / 'dsqtl_test_neg.minor.fa',
    )
    print(f"  Positive dsQTLs: {len(pos_variants)}")
    print(f"  Negative controls: {len(neg_variants)}")

    all_variants = [(v, 1) for v in pos_variants] + [(v, 0) for v in neg_variants]

    # Prepare sequences
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

    if args.use_liftover:
        print(f"\n  Using hg38 genomic context (liftover from hg19)...")
        try:
            from pyliftover import LiftOver
            import pyfaidx
        except ImportError as e:
            print(f"  ERROR: {e}. Install with: pip install pyliftover pyfaidx")
            return 1

        lo = LiftOver('hg19', 'hg38')
        genome = pyfaidx.Fasta(str(base_dir / args.genome))

        ref_seqs = []
        alt_seqs = []
        labels = []
        skipped = 0

        for i, (var, label) in enumerate(all_variants):
            if i % 5000 == 0:
                print(f"    Processing {i}/{len(all_variants)}...")

            # Liftover coordinate
            result = lo.convert_coordinate(var['chrom'], var['variant_pos_genomic'])
            if not result:
                skipped += 1
                continue

            chrom38, pos38 = result[0][0], int(result[0][1])

            # Get 2000bp context
            seq_str, var_pos_in_window = get_genomic_context(
                chrom38, pos38, genome, window=2000
            )
            if seq_str is None:
                skipped += 1
                continue

            ref_onehot = encode_sequence(seq_str)
            alt_base = base_map.get(var['alt'], 0)
            alt_onehot = apply_snv(ref_onehot, var_pos_in_window, alt_base)

            ref_seqs.append(ref_onehot)
            alt_seqs.append(alt_onehot)
            labels.append(label)

        print(f"    Processed: {len(labels)}, Skipped (liftover fail): {skipped}")
    else:
        print(f"\n  Using padded 19-mers (no genomic context)...")
        ref_seqs = []
        alt_seqs = []
        labels = []

        for var, label in all_variants:
            # Pad 19-mer to 2000bp centered
            pad_left = (2000 - len(var['major_seq'])) // 2
            pad_right = 2000 - len(var['major_seq']) - pad_left

            ref_str = 'N' * pad_left + var['major_seq'] + 'N' * pad_right
            alt_str = 'N' * pad_left + var['minor_seq'] + 'N' * pad_right

            ref_seqs.append(encode_sequence(ref_str))
            alt_seqs.append(encode_sequence(alt_str))
            labels.append(label)

    # Score all variants
    print(f"\n  Scoring {len(labels)} variants...")
    all_scores = score_variants_batch(
        model, ref_seqs, alt_seqs, args.device, args.batch_size
    )

    ism_scores = [s['ism_score'] for s in all_scores]
    profile_scores = [s['profile_delta_sum'] for s in all_scores]
    local_scores = [s['profile_delta_local'] for s in all_scores]
    occ_scores = [s['occ_delta_max'] for s in all_scores]

    # Compute metrics for different scoring strategies
    print(f"\n{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")

    results = {}
    for score_name, scores in [
        ('ism_combined', ism_scores),
        ('profile_delta_sum', profile_scores),
        ('profile_delta_local', local_scores),
        ('occ_delta_max', occ_scores),
    ]:
        metrics = compute_metrics(scores, labels)
        results[score_name] = metrics
        print(f"\n  {score_name}:")
        print(f"    AUROC:  {metrics['auroc']:.4f}")
        print(f"    AUPRC:  {metrics['auprc']:.4f}")
        print(f"    Mean score (pos/neg): {metrics['mean_score_pos']:.4f} / {metrics['mean_score_neg']:.4f}")
        print(f"    Score ratio: {metrics['score_ratio']:.2f}x")

    # Save results
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'dsqtl_benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Save per-variant scores
    scores_path = output_dir / 'dsqtl_variant_scores.tsv'
    with open(scores_path, 'w') as f:
        f.write("chrom\tpos\tref\talt\tlabel\tism_score\tprofile_delta\tlocal_delta\tocc_delta\n")
        for (var, label), score in zip(all_variants[:len(all_scores)], all_scores):
            f.write(f"{var['chrom']}\t{var['variant_pos_genomic']}\t"
                   f"{var['ref']}\t{var['alt']}\t{label}\t"
                   f"{score['ism_score']:.6f}\t{score['profile_delta_sum']:.6f}\t"
                   f"{score['profile_delta_local']:.6f}\t{score['occ_delta_max']:.6f}\n")

    print(f"  Scores saved to: {scores_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
