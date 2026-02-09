#!/usr/bin/env python3
"""
Phase 11 Experiments:
  11.1: Gradient-based variant scoring (rescue variant prediction, target AUROC>0.60)
  11.2: TF grammar discovery from co-binding patterns
  11.3: Computational efficiency benchmarks
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


def load_model(config, checkpoint_path, device):
    model = BEACON(
        seq_len=config.get("seq_len", 2000), input_channels=4,
        backbone_type="dilated", backbone_dim=config.get("backbone_dim", 128),
        backbone_layers=config.get("backbone_layers", 4),
        n_slots=config.get("n_slots", 16), slot_dim=config.get("slot_dim", 128),
        n_iterations=config.get("n_iterations", 3),
        n_tfs=config.get("n_tfs", 7), position_mode="gaussian",
        attention_mode=config.get("attention_mode", "competitive"),
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Phase 11.1: Gradient-Based Variant Scoring
# ============================================================

def variant_scoring_ism(model, dataset, device, n_samples=500):
    """
    In-silico mutagenesis (ISM) variant scoring.
    For each sample, mutate the center position and measure profile change.
    Compare to random positions as negative control.
    """
    print("\n" + "=" * 60)
    print("Phase 11.1: Gradient-Based Variant Scoring (ISM)")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = {'binding_site_scores': [], 'random_scores': [], 'gradient_scores': []}

    count = 0
    for batch in loader:
        if count >= n_samples:
            break

        seq = batch['sequence'].to(device)  # [1, L, 4]
        profile = batch['profile'].numpy()[0]  # [L]
        binding_sites = batch['binding_sites'].numpy()[0]  # [K, 3]

        # Find binding site positions
        valid_sites = binding_sites[binding_sites[:, 2] > 0.1]
        if len(valid_sites) == 0:
            continue

        seq_len = seq.shape[1]

        # --- ISM at binding site ---
        site_pos = int(valid_sites[0][0] * seq_len)
        site_pos = max(0, min(site_pos, seq_len - 1))

        with torch.no_grad():
            ref_out = model(seq)
            ref_profile = ref_out['profile'][0].cpu().numpy()
            ref_occ = ref_out['occupancy'][0].squeeze(-1).cpu().numpy()

        # Mutate: cycle through all 4 bases, take max effect
        max_delta = 0.0
        orig_base = seq[0, site_pos].argmax().item()
        for alt_base in range(4):
            if alt_base == orig_base:
                continue
            alt_seq = seq.clone()
            alt_seq[0, site_pos, :] = 0
            alt_seq[0, site_pos, alt_base] = 1

            with torch.no_grad():
                alt_out = model(alt_seq)
                alt_profile = alt_out['profile'][0].cpu().numpy()
                alt_occ = alt_out['occupancy'][0].squeeze(-1).cpu().numpy()

            delta_profile = np.abs(ref_profile - alt_profile).sum()
            delta_occ = np.abs(ref_occ - alt_occ).max()
            max_delta = max(max_delta, delta_profile + delta_occ * 100)

        results['binding_site_scores'].append(max_delta)

        # --- ISM at random position (control) ---
        rand_pos = np.random.randint(0, seq_len)
        # Avoid being near binding site
        while abs(rand_pos - site_pos) < 100:
            rand_pos = np.random.randint(0, seq_len)

        max_delta_rand = 0.0
        orig_base_rand = seq[0, rand_pos].argmax().item()
        for alt_base in range(4):
            if alt_base == orig_base_rand:
                continue
            alt_seq = seq.clone()
            alt_seq[0, rand_pos, :] = 0
            alt_seq[0, rand_pos, alt_base] = 1

            with torch.no_grad():
                alt_out = model(alt_seq)
                alt_profile = alt_out['profile'][0].cpu().numpy()
                alt_occ = alt_out['occupancy'][0].squeeze(-1).cpu().numpy()

            delta_profile = np.abs(ref_profile - alt_profile).sum()
            delta_occ = np.abs(ref_occ - alt_occ).max()
            max_delta_rand = max(max_delta_rand, delta_profile + delta_occ * 100)

        results['random_scores'].append(max_delta_rand)

        # --- Gradient attribution score at binding site ---
        seq_grad = seq.clone().detach().requires_grad_(True)
        out_grad = model(seq_grad)
        # Backprop from max occupancy slot
        best_slot = out_grad['occupancy'][0].squeeze(-1).argmax()
        target = out_grad['occupancy'][0, best_slot, 0]
        target.backward()

        grad_mag = seq_grad.grad[0].abs().sum(dim=-1).cpu().numpy()  # [L]
        results['gradient_scores'].append(float(grad_mag[site_pos]))

        count += 1
        if count % 100 == 0:
            print(f"  Processed {count}/{n_samples} samples")

    # Compute AUROC: can we distinguish binding sites from random positions?
    from sklearn.metrics import roc_auc_score

    bs_scores = np.array(results['binding_site_scores'])
    rand_scores = np.array(results['random_scores'])

    labels = np.concatenate([np.ones(len(bs_scores)), np.zeros(len(rand_scores))])
    scores = np.concatenate([bs_scores, rand_scores])

    auroc = roc_auc_score(labels, scores)

    print(f"\n  ISM Variant Scoring Results:")
    print(f"    N samples: {count}")
    print(f"    Binding site ISM score (mean): {bs_scores.mean():.3f} ± {bs_scores.std():.3f}")
    print(f"    Random position ISM score (mean): {rand_scores.mean():.3f} ± {rand_scores.std():.3f}")
    print(f"    AUROC (binding vs random): {auroc:.3f}")
    print(f"    Fold enrichment: {bs_scores.mean() / max(rand_scores.mean(), 1e-6):.2f}x")

    # Gradient scores
    grad_scores = np.array(results['gradient_scores'])
    print(f"\n  Gradient Attribution at Binding Sites:")
    print(f"    Mean gradient magnitude: {grad_scores.mean():.4f}")

    results['auroc'] = float(auroc)
    results['n_samples'] = count
    results['binding_site_mean'] = float(bs_scores.mean())
    results['random_mean'] = float(rand_scores.mean())
    results['fold_enrichment'] = float(bs_scores.mean() / max(rand_scores.mean(), 1e-6))

    return results


# ============================================================
# Phase 11.2: TF Grammar Discovery
# ============================================================

def tf_grammar_discovery(model, dataset, device, n_samples=2000):
    """
    Discover TF binding grammar rules from multi-slot predictions.
    Analyze spacing, orientation, and co-occurrence patterns.
    """
    print("\n" + "=" * 60)
    print("Phase 11.2: TF Grammar Discovery")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    pair_spacings = defaultdict(list)
    pair_counts = Counter()
    tf_position_distributions = defaultdict(list)
    n_processed = 0

    with torch.no_grad():
        for batch in loader:
            if n_processed >= n_samples:
                break

            sequences = batch['sequence'].to(device)
            outputs = model(sequences)

            occ = outputs['occupancy'].squeeze(-1).cpu().numpy()
            tf_logits = outputs['tf_logits'].cpu().numpy()
            positions = outputs['positions'].cpu().numpy()

            pred_tf = tf_logits.argmax(axis=-1)

            for b in range(sequences.shape[0]):
                if n_processed >= n_samples:
                    break

                active = np.where(occ[b] > 0.3)[0]
                if len(active) < 2:
                    n_processed += 1
                    continue

                # Record active sites
                sites = []
                for s in active:
                    tf_id = int(pred_tf[b][s])
                    pos = float(positions[b][s][0])
                    sites.append((s, tf_id, pos))
                    tf_position_distributions[tf_id].append(pos)

                # Record all pairs
                for i in range(len(sites)):
                    for j in range(i + 1, len(sites)):
                        tf_a = sites[i][1]
                        tf_b = sites[j][1]
                        spacing = abs(sites[i][2] - sites[j][2]) * 2000  # bp

                        pair = tuple(sorted([tf_a, tf_b]))
                        pair_spacings[pair].append(spacing)
                        pair_counts[pair] += 1

                n_processed += 1

    print(f"  Processed {n_processed} samples")

    # Analyze spacing distributions
    print(f"\n--- TF Pair Spacing Grammar ---")
    print(f"{'Pair':<20} {'N':<8} {'Mean bp':<10} {'Median bp':<10} {'Std bp':<10} {'Mode Range'}")
    print("-" * 75)

    grammar_rules = {}
    for pair, count in pair_counts.most_common(15):
        spacings = np.array(pair_spacings[pair])
        if len(spacings) < 10:
            continue

        tf_a, tf_b = TF_NAMES[pair[0]], TF_NAMES[pair[1]]
        pair_name = f"{tf_a}-{tf_b}"

        mean_sp = spacings.mean()
        median_sp = np.median(spacings)
        std_sp = spacings.std()

        # Find mode (peak of histogram)
        hist, edges = np.histogram(spacings, bins=20, range=(0, 1000))
        mode_idx = hist.argmax()
        mode_range = f"{edges[mode_idx]:.0f}-{edges[mode_idx+1]:.0f}bp"

        print(f"  {pair_name:<18} {count:<8} {mean_sp:<10.0f} {median_sp:<10.0f} {std_sp:<10.0f} {mode_range}")

        grammar_rules[pair_name] = {
            'count': int(count),
            'mean_spacing': float(mean_sp),
            'median_spacing': float(median_sp),
            'std_spacing': float(std_sp),
            'mode_range': mode_range,
        }

    # Biological grammar validation
    print(f"\n--- Biological Grammar Validation ---")
    key_rules = [
        ("MYC-MAX", "E-box co-occupancy (~0bp for heterodimer)"),
        ("GATA1-TAL1", "Erythroid enhancer (~30-150bp apart)"),
        ("CTCF-CTCF", "Insulator boundary (helical phasing ~10.5bp multiples)"),
    ]

    for pair_name, description in key_rules:
        if pair_name in grammar_rules:
            rule = grammar_rules[pair_name]
            print(f"\n  {pair_name}: {description}")
            print(f"    Predicted: {rule['mean_spacing']:.0f}bp mean, {rule['median_spacing']:.0f}bp median")
            print(f"    Mode: {rule['mode_range']}")

    return grammar_rules


# ============================================================
# Phase 11.3: Computational Efficiency Benchmarks
# ============================================================

def efficiency_benchmarks(model, dataset, device):
    """
    Benchmark BEACON's throughput for genome-wide annotation.
    """
    print("\n" + "=" * 60)
    print("Phase 11.3: Computational Efficiency Benchmarks")
    print("=" * 60)

    # Warm up
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    seq = batch['sequence'].to(device)
    with torch.no_grad():
        _ = model(seq)

    # Benchmark different batch sizes
    results = {}
    for bs in [1, 8, 16, 32, 64]:
        try:
            loader_bs = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
            n_batches = min(50, len(loader_bs))

            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()

            count = 0
            with torch.no_grad():
                for batch in loader_bs:
                    if count >= n_batches:
                        break
                    seq = batch['sequence'].to(device)
                    outputs = model(seq)
                    # Extract all outputs (simulating real usage)
                    _ = outputs['occupancy'].cpu()
                    _ = outputs['tf_logits'].cpu()
                    _ = outputs['positions'].cpu()
                    _ = outputs['profile'].cpu()
                    count += 1

            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed = time.time() - start
            total_samples = count * bs
            throughput = total_samples / elapsed

            results[f'batch_{bs}'] = {
                'batch_size': bs,
                'total_samples': total_samples,
                'total_time_s': float(elapsed),
                'throughput_seq_per_s': float(throughput),
                'ms_per_seq': float(1000 / throughput),
            }

            print(f"  Batch size {bs:>3}: {throughput:.1f} seq/s ({1000/throughput:.1f} ms/seq)")
        except RuntimeError as e:
            print(f"  Batch size {bs:>3}: OOM or error - {e}")

    # Genome-wide extrapolation
    best_throughput = max(r['throughput_seq_per_s'] for r in results.values())
    genome_windows = 1_500_000  # ~3Gb genome / 2000bp windows
    genome_time_hours = genome_windows / best_throughput / 3600

    print(f"\n  --- Genome-Wide Extrapolation ---")
    print(f"  Best throughput: {best_throughput:.1f} seq/s")
    print(f"  Genome windows (2kb, 3Gb): {genome_windows:,}")
    print(f"  Time to annotate full genome: {genome_time_hours:.1f} hours")
    print(f"  Time for 100Mb region: {genome_time_hours * 100/3000:.1f} hours")

    # Compare to BPNet+TF-MoDISco pipeline (literature estimates)
    bpnet_per_seq_s = 8.438  # DeepSHAP + TF-MoDISco
    bpnet_genome_hours = genome_windows * bpnet_per_seq_s / 3600

    speedup = bpnet_per_seq_s / (1.0 / best_throughput)
    print(f"\n  --- vs BPNet+TF-MoDISco Pipeline ---")
    print(f"  BEACON: {1000/best_throughput:.1f} ms/seq")
    print(f"  BPNet+DeepSHAP+TF-MoDISco: {bpnet_per_seq_s*1000:.0f} ms/seq (literature)")
    print(f"  Speedup: {speedup:.0f}x")
    print(f"  BEACON genome annotation: {genome_time_hours:.1f} hours")
    print(f"  BPNet pipeline genome annotation: {bpnet_genome_hours:.0f} hours")

    results['genome_annotation'] = {
        'best_throughput': float(best_throughput),
        'genome_windows': genome_windows,
        'beacon_hours': float(genome_time_hours),
        'bpnet_hours': float(bpnet_genome_hours),
        'speedup': float(speedup),
    }

    return results


def create_phase11_figures(variant_results, grammar_results, efficiency_results, output_dir):
    """Create summary figures for Phase 11."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Variant scoring: binding vs random
    ax = axes[0]
    if variant_results:
        bs = np.array(variant_results['binding_site_scores'])
        rand = np.array(variant_results['random_scores'])
        ax.hist(bs, bins=30, alpha=0.7, label=f'Binding sites', color='coral', density=True)
        ax.hist(rand, bins=30, alpha=0.7, label=f'Random positions', color='steelblue', density=True)
        ax.set_xlabel('ISM Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Variant Effect Scoring\nAUROC={variant_results["auroc"]:.3f}')
        ax.legend()

    # 2. TF pair spacing distributions
    ax = axes[1]
    if grammar_results:
        pairs = list(grammar_results.keys())[:8]
        means = [grammar_results[p]['mean_spacing'] for p in pairs]
        stds = [grammar_results[p]['std_spacing'] for p in pairs]
        y_pos = np.arange(len(pairs))
        ax.barh(y_pos, means, xerr=stds, alpha=0.8, color='mediumseagreen', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=8)
        ax.set_xlabel('Spacing (bp)')
        ax.set_title('TF Pair Spacing Grammar')

    # 3. Throughput by batch size
    ax = axes[2]
    if efficiency_results:
        batch_sizes = []
        throughputs = []
        for k, v in efficiency_results.items():
            if k.startswith('batch_'):
                batch_sizes.append(v['batch_size'])
                throughputs.append(v['throughput_seq_per_s'])
        if batch_sizes:
            ax.bar(range(len(batch_sizes)), throughputs, tick_label=[str(b) for b in batch_sizes],
                   color='steelblue', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (seq/s)')
            ax.set_title(f'BEACON Throughput\n(Best: {max(throughputs):.0f} seq/s)')

    plt.tight_layout()
    plt.savefig(output_dir / "phase11_experiments.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {output_dir / 'phase11_experiments.png'}")


def main():
    parser = argparse.ArgumentParser(description="Phase 11 Experiments")
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("outputs/multi_tf_k562"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--n-variant-samples", type=int, default=500)
    parser.add_argument("--n-grammar-samples", type=int, default=1197)

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = base_dir / experiment_dir
    if experiment_dir.exists():
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if subdirs:
            experiment_dir = sorted(subdirs)[-1]

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = args.output_dir or experiment_dir / "phase11"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        config_path = experiment_dir / experiment_dir.name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"seq_len": 2000, "n_slots": 16, "backbone_dim": 128,
                  "backbone_layers": 4, "slot_dim": 128, "n_iterations": 3, "n_tfs": 7}

    checkpoint_path = experiment_dir / "best_model.pt"
    if not checkpoint_path.exists():
        nested = experiment_dir / experiment_dir.name
        if nested.exists():
            checkpoint_path = nested / "best_model.pt"

    print("=" * 60)
    print("Phase 11: Additional Experiments")
    print("=" * 60)
    print(f"Model: {checkpoint_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")

    device = torch.device(args.device)
    model = load_model(config, checkpoint_path, device)

    test_path = data_dir / "test.h5"
    dataset = BEACONDataset(test_path, seq_length=config.get("seq_len", 2000),
                            augment=False, n_slots=config.get("n_slots", 16))
    print(f"Test samples: {len(dataset)}")

    # Run experiments
    variant_results = variant_scoring_ism(model, dataset, device, n_samples=args.n_variant_samples)
    grammar_results = tf_grammar_discovery(model, dataset, device, n_samples=args.n_grammar_samples)
    efficiency_results = efficiency_benchmarks(model, dataset, device)

    # Save figures
    create_phase11_figures(variant_results, grammar_results, efficiency_results, output_dir)

    # Save results
    all_results = {
        'variant_scoring': {k: v for k, v in variant_results.items()
                           if k not in ('binding_site_scores', 'random_scores', 'gradient_scores')},
        'grammar': grammar_results,
        'efficiency': efficiency_results,
    }
    with open(output_dir / "phase11_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
