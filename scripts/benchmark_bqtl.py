#!/usr/bin/env python3
"""
bQTL Benchmark: Tehranchi et al. 2016 Binding QTLs

Tests BEACON's ability to predict allele-specific TF binding using
binding quantitative trait loci (bQTLs) from Tehranchi et al. 2016.

Available TFs in bQTL data:
  - PU.1 (= SPI1, K562-7tf index 5): 94K variants
  - JunD (= JUND, K562-fulltf index 9): 157K variants

Approach:
  1. Load bQTL data, liftOver hg19 → hg38
  2. Extract 2000bp sequences centered on each variant
  3. Run BEACON on ref and alt alleles
  4. Score = delta profile / delta occupancy at variant position
  5. Evaluate: AUROC for predicting higher-binding allele direction
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import pysam
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score
import openpyxl

sys.path.insert(0, str(Path(__file__).parent.parent))
from beacon.models import BEACON

# Nucleotide encoding
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def liftover_hg19_to_hg38(chrom, pos):
    """Convert hg19 coordinates to hg38 using pyliftover."""
    from pyliftover import LiftOver
    if not hasattr(liftover_hg19_to_hg38, '_lo'):
        print("  Loading hg19→hg38 chain file...")
        liftover_hg19_to_hg38._lo = LiftOver('hg19', 'hg38')
    result = liftover_hg19_to_hg38._lo.convert_coordinate(chrom, pos)
    if result and len(result) > 0:
        return result[0][0], result[0][1]
    return None, None


def load_bqtl_data(xlsx_path, sheet_name, p_threshold=0.01, max_variants=5000):
    """Load bQTL variants from Tehranchi 2016 Excel file."""
    print(f"\nLoading bQTL data: {sheet_name} (p < {p_threshold})...")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb[sheet_name]

    variants = []
    header = None
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            header = list(row)
            continue
        chrom, pos, alt, ref, higher, pval = row[:6]
        if pval is None or pval > p_threshold:
            continue
        variants.append({
            'chrom': str(chrom),
            'pos': int(pos),
            'ref': str(ref).upper(),
            'alt': str(alt).upper(),
            'higher_binding': str(higher).upper(),
            'pvalue': float(pval),
        })

    wb.close()

    # Sort by p-value (most significant first) and take top N
    variants.sort(key=lambda v: v['pvalue'])
    if len(variants) > max_variants:
        variants = variants[:max_variants]

    print(f"  Loaded {len(variants)} significant variants (sorted by p-value)")
    return variants


def extract_sequences(variants, genome_fa, seq_len=2000):
    """Extract ref/alt sequences centered on variant positions."""
    fasta = pysam.FastaFile(genome_fa)
    half = seq_len // 2
    valid = []

    for v in variants:
        # LiftOver hg19 → hg38
        chrom38, pos38 = liftover_hg19_to_hg38(v['chrom'], v['pos'])
        if chrom38 is None:
            continue

        pos38 = int(pos38)
        start = pos38 - half
        end = pos38 + half

        if start < 0:
            continue

        try:
            seq = fasta.fetch(chrom38, start, end).upper()
        except (ValueError, KeyError):
            continue

        if len(seq) != seq_len:
            continue

        # Verify ref allele matches at center position
        center = half
        ref_base = seq[center]
        if ref_base != v['ref']:
            # Try alt as ref (strand flip possible)
            if ref_base == v['alt']:
                v['ref'], v['alt'] = v['alt'], v['ref']
                if v['higher_binding'] == v['ref']:
                    v['higher_binding'] = v['alt']
                elif v['higher_binding'] == v['alt']:
                    v['higher_binding'] = v['ref']
            else:
                continue

        # One-hot encode ref sequence
        ref_onehot = np.zeros((seq_len, 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in NUC_TO_IDX:
                ref_onehot[i, NUC_TO_IDX[base]] = 1.0

        # Create alt sequence by substituting at center
        alt_onehot = ref_onehot.copy()
        alt_onehot[center] = 0.0
        if v['alt'] in NUC_TO_IDX:
            alt_onehot[center, NUC_TO_IDX[v['alt']]] = 1.0
        else:
            continue

        v['chrom38'] = chrom38
        v['pos38'] = pos38
        v['center'] = center
        v['ref_seq'] = ref_onehot
        v['alt_seq'] = alt_onehot
        valid.append(v)

    fasta.close()
    print(f"  Valid variants after liftOver + sequence extraction: {len(valid)}")
    return valid


def predict_batch(model, sequences, device, batch_size=64):
    """Run BEACON on a batch of sequences, return profiles and occupancies."""
    model.eval()
    all_profiles = []
    all_occupancies = []
    all_tf_probs = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            x = torch.tensor(batch, dtype=torch.float32).to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(x)

            profile = outputs['profile'].squeeze(1).cpu().numpy()  # [B, L]
            occupancy = outputs['occupancy'].squeeze(-1).cpu().numpy()  # [B, K]
            tf_logits = outputs['tf_logits'].cpu().numpy()  # [B, K, n_tfs]

            all_profiles.append(profile)
            all_occupancies.append(occupancy)
            all_tf_probs.append(tf_logits)

    profiles = np.concatenate(all_profiles, axis=0)
    occupancies = np.concatenate(all_occupancies, axis=0)
    tf_probs = np.concatenate(all_tf_probs, axis=0)

    return profiles, occupancies, tf_probs


def score_variants(variants, model, device, tf_idx, batch_size=64):
    """Score each variant by comparing ref vs alt predictions."""
    print(f"\nScoring {len(variants)} variants...")

    # Collect all ref and alt sequences
    ref_seqs = np.stack([v['ref_seq'] for v in variants])
    alt_seqs = np.stack([v['alt_seq'] for v in variants])

    # Predict for ref and alt
    print("  Predicting ref alleles...")
    ref_profiles, ref_occ, ref_tf = predict_batch(model, ref_seqs, device, batch_size)
    print("  Predicting alt alleles...")
    alt_profiles, alt_occ, alt_tf = predict_batch(model, alt_seqs, device, batch_size)

    scores = []
    for i, v in enumerate(variants):
        center = v['center']

        # Profile delta at variant position (window around variant)
        window = 25  # +/- 25bp around variant
        s = max(0, center - window)
        e = min(ref_profiles.shape[1], center + window)
        ref_signal = ref_profiles[i, s:e].sum()
        alt_signal = alt_profiles[i, s:e].sum()
        delta_profile = alt_signal - ref_signal

        # Find slots predicted as this TF
        ref_tf_pred = ref_tf[i].argmax(axis=-1)  # [K]
        alt_tf_pred = alt_tf[i].argmax(axis=-1)

        # Mean occupancy for slots predicted as target TF
        ref_tf_mask = (ref_tf_pred == tf_idx)
        alt_tf_mask = (alt_tf_pred == tf_idx)

        ref_tf_occ = ref_occ[i][ref_tf_mask].sum() if ref_tf_mask.any() else 0.0
        alt_tf_occ = alt_occ[i][alt_tf_mask].sum() if alt_tf_mask.any() else 0.0
        delta_occ = alt_tf_occ - ref_tf_occ

        # Combined score
        delta_combined = delta_profile + delta_occ * 10  # occupancy scaled

        # Ground truth: does alt have higher binding?
        alt_is_higher = 1 if v['higher_binding'] == v['alt'] else 0

        scores.append({
            'delta_profile': float(delta_profile),
            'delta_occ': float(delta_occ),
            'delta_combined': float(delta_combined),
            'alt_is_higher': alt_is_higher,
            'pvalue': v['pvalue'],
        })

    return scores


def evaluate_scores(scores, score_key='delta_combined'):
    """Compute AUROC and direction accuracy."""
    deltas = np.array([s[score_key] for s in scores])
    labels = np.array([s['alt_is_higher'] for s in scores])

    # AUROC: positive delta should predict alt=higher binding
    try:
        auroc = roc_auc_score(labels, deltas)
    except ValueError:
        auroc = 0.5

    # Direction accuracy: delta > 0 predicts alt higher, delta < 0 predicts ref higher
    preds = (deltas > 0).astype(int)
    accuracy = accuracy_score(labels, preds)

    # For top-N most significant variants
    results = {'auroc': auroc, 'accuracy': accuracy, 'n_variants': len(scores)}

    for top_n in [100, 500, 1000, 2000]:
        if len(scores) >= top_n:
            top_deltas = deltas[:top_n]
            top_labels = labels[:top_n]
            try:
                top_auroc = roc_auc_score(top_labels, top_deltas)
            except ValueError:
                top_auroc = 0.5
            top_preds = (top_deltas > 0).astype(int)
            top_acc = accuracy_score(top_labels, top_preds)
            results[f'auroc_top{top_n}'] = top_auroc
            results[f'accuracy_top{top_n}'] = top_acc

    return results


def run_bqtl_benchmark(
    model_path, n_tfs, tf_name, tf_idx, bqtl_sheet,
    data_xlsx, genome_fa, device, output_dir,
    p_threshold=0.01, max_variants=5000, batch_size=64
):
    """Run full bQTL benchmark for one TF."""
    print("=" * 70)
    print(f"bQTL Benchmark: {tf_name} (sheet={bqtl_sheet})")
    print(f"  Model: {model_path}")
    print(f"  TF index: {tf_idx}")
    print(f"  Device: {device}")
    print("=" * 70)

    # Load model
    model = BEACON(
        seq_len=2000,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=16,
        slot_dim=128,
        n_iterations=3,
        n_tfs=n_tfs,
        position_mode="gaussian",
        dropout=0.0,
        attention_mode="independent",
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load bQTL data
    variants = load_bqtl_data(data_xlsx, bqtl_sheet, p_threshold, max_variants)
    if len(variants) == 0:
        print("  No significant variants found!")
        return None

    # Extract sequences
    variants = extract_sequences(variants, genome_fa)
    if len(variants) == 0:
        print("  No valid sequences extracted!")
        return None

    # Score variants
    scores = score_variants(variants, model, device, tf_idx, batch_size)

    # Evaluate
    results = {}
    for score_key in ['delta_profile', 'delta_occ', 'delta_combined']:
        eval_result = evaluate_scores(scores, score_key)
        results[score_key] = eval_result
        print(f"\n  {score_key}:")
        print(f"    AUROC: {eval_result['auroc']:.3f}")
        print(f"    Accuracy: {eval_result['accuracy']:.3f}")
        for k, v in eval_result.items():
            if 'top' in k:
                print(f"    {k}: {v:.3f}")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json = {
        'experiment': 'bqtl_benchmark',
        'tf_name': tf_name,
        'bqtl_sheet': bqtl_sheet,
        'model_path': str(model_path),
        'n_variants': len(variants),
        'p_threshold': p_threshold,
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    result_path = output_dir / f"bqtl_{tf_name}_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\n  Results saved to: {result_path}")

    return result_json


def main():
    parser = argparse.ArgumentParser(description="bQTL Benchmark (Tehranchi 2016)")
    parser.add_argument("--bqtl-xlsx", type=str,
                        default="data/benchmarks/bqtl/mmc2.xlsx")
    parser.add_argument("--genome", type=str,
                        default="data/raw/genome/hg38.fa")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/bqtl_benchmark")
    parser.add_argument("--p-threshold", type=float, default=0.01)
    parser.add_argument("--max-variants", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    bqtl_path = Path(args.bqtl_xlsx)
    if not bqtl_path.is_absolute():
        bqtl_path = base_dir / bqtl_path
    genome_path = Path(args.genome)
    if not genome_path.is_absolute():
        genome_path = base_dir / genome_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_results = {}

    # PU.1 (SPI1) - K562-7tf model, TF index 5
    k562_7tf_model = base_dir / "outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt"
    if k562_7tf_model.exists():
        result = run_bqtl_benchmark(
            model_path=k562_7tf_model,
            n_tfs=7, tf_name="SPI1_PU1", tf_idx=5,
            bqtl_sheet="PU.1",
            data_xlsx=str(bqtl_path),
            genome_fa=str(genome_path),
            device=device,
            output_dir=output_dir,
            p_threshold=args.p_threshold,
            max_variants=args.max_variants,
            batch_size=args.batch_size,
        )
        if result:
            all_results['SPI1_PU1'] = result

    # JunD (JUND) - K562-fulltf model, TF index 9
    k562_fulltf_model = base_dir / "outputs/k562_20tf/K562-20tf_baseline/beacon_20260205_190727/best_model.pt"
    if k562_fulltf_model.exists():
        result = run_bqtl_benchmark(
            model_path=k562_fulltf_model,
            n_tfs=14, tf_name="JUND", tf_idx=9,
            bqtl_sheet="JunD",
            data_xlsx=str(bqtl_path),
            genome_fa=str(genome_path),
            device=device,
            output_dir=output_dir,
            p_threshold=args.p_threshold,
            max_variants=args.max_variants,
            batch_size=args.batch_size,
        )
        if result:
            all_results['JUND'] = result

    # Summary
    print("\n" + "=" * 70)
    print("bQTL Benchmark Summary")
    print("=" * 70)
    for tf_name, result in all_results.items():
        best = result['results']['delta_combined']
        print(f"  {tf_name}: AUROC={best['auroc']:.3f}, Acc={best['accuracy']:.3f} "
              f"(n={result['n_variants']})")

    # Save combined results
    combined_path = output_dir / "bqtl_all_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
