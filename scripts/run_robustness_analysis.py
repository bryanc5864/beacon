#!/usr/bin/env python3
"""
Robustness analysis: mutation perturbation + multi-site synthetic test.

Tests how BEACON handles noise and whether slots can discover multiple
sites when presented with synthetic multi-site inputs.

Usage:
    python scripts/run_robustness_analysis.py --device cuda:2
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_named_model, load_test_data, get_test_path,
    run_model_batch, per_sample_pearson, json_serialize,
    DATASET_CONFIGS, BASE_DIR,
)
from beacon.analysis.robustness import (
    robustness_curve, multi_site_synthetic_test,
)


def main():
    parser = argparse.ArgumentParser(description="Robustness analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Robustness & Multi-Site Analysis")
    print("=" * 70)

    results = {}

    # ── BEACON robustness curve ─────────────────────────────────────────
    print("\n--- BEACON Robustness (K562-7tf) ---")
    model, cfg = load_named_model("K562-7tf", device)
    test_path = get_test_path("K562-7tf")
    seqs, profs, tf_idx, bs = load_test_data(test_path, max_samples=args.max_samples)

    mutation_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"  Testing {len(mutation_rates)} mutation rates...")

    beacon_robustness = robustness_curve(
        model, seqs, profs, device, mutation_rates,
        run_model_batch, per_sample_pearson,
    )

    for rate, data in beacon_robustness.items():
        print(f"    {float(rate)*100:5.1f}% mutation: r={data['mean_r']:.4f} "
              f"(retained {data['relative_retained']:.1%})")

    results['beacon_robustness'] = beacon_robustness

    # ── BPNet robustness curve ──────────────────────────────────────────
    print("\n--- BPNet Robustness ---")
    try:
        from bpnetlite import BPNet
        import numpy as np

        bpnet_path = BASE_DIR / "bpnet.64.8.final.torch"
        if not bpnet_path.exists():
            bpnet_path = BASE_DIR / "bpnet.64.8.torch"

        if bpnet_path.exists():
            bpnet = BPNet(n_filters=64, n_layers=8, n_outputs=7,
                          n_control_tracks=0, trimming=500)
            state = torch.load(bpnet_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and 'model_state_dict' in state:
                bpnet.load_state_dict(state['model_state_dict'])
            elif isinstance(state, dict):
                bpnet.load_state_dict(state)
            else:
                bpnet = state
            bpnet = bpnet.to(device)
            bpnet.eval()

            def bpnet_run(model, sequences, device, **kwargs):
                """Adapter for BPNet inference matching BEACON API."""
                import torch as th
                from collections import defaultdict
                all_profs = []
                model.eval()
                # BPNet: channels-first (N, 4, L)
                with th.no_grad():
                    for i in range(0, len(sequences), 32):
                        batch_np = sequences[i:i + 32]
                        x = th.tensor(batch_np.transpose(0, 2, 1),
                                      dtype=th.float32).to(device)
                        out = model(x)
                        # BPNet output may be tuple (profile, counts) or tensor
                        if isinstance(out, tuple):
                            out = out[0]
                        prof = out.cpu().numpy()
                        # Sum across TF outputs and pad to original length
                        prof_sum = prof.sum(axis=1)  # [N, 1000]
                        padded = np.zeros((len(prof_sum), 2000))
                        padded[:, 500:1500] = prof_sum
                        all_profs.append(padded)
                return {'profile': np.concatenate(all_profs, axis=0)}

            bpnet_robustness = robustness_curve(
                bpnet, seqs, profs, device, mutation_rates,
                bpnet_run, per_sample_pearson,
            )
            for rate, data in bpnet_robustness.items():
                print(f"    {float(rate)*100:5.1f}% mutation: r={data['mean_r']:.4f} "
                      f"(retained {data['relative_retained']:.1%})")
            results['bpnet_robustness'] = bpnet_robustness
            del bpnet; torch.cuda.empty_cache()
        else:
            print("  No BPNet checkpoint found, skipping")
    except ImportError:
        print("  bpnetlite not installed, skipping")

    # ── Multi-site synthetic test ───────────────────────────────────────
    print("\n--- Multi-Site Synthetic Test ---")
    seqs_full, profs_full, tf_idx_full, bs_full = load_test_data(
        test_path)  # need all for TF diversity

    multi_site = multi_site_synthetic_test(
        model, seqs_full, bs_full, device,
        run_model_batch, n_pairs=200,
    )
    print(f"  Mean active slots (combined): {multi_site['mean_active_slots_combined']:.2f}")
    print(f"  Mean active slots (single): {multi_site['mean_active_slots_single']:.2f}")
    print(f"  2+ slots rate: {multi_site['two_plus_slots_rate']:.1%}")
    print(f"  Both TFs detected: {multi_site['both_tfs_detected_rate']:.1%}")
    print(f"  Slot distribution: {multi_site['active_slot_distribution']}")

    results['multi_site'] = multi_site

    # ── Fair BPNet comparison (per-TF) ──────────────────────────────────
    print("\n--- Fair BPNet Per-TF Comparison ---")
    bpnet_multi_path = BASE_DIR / "outputs" / "bpnet_multi_tf"
    combined_path = None
    if bpnet_multi_path.exists():
        # Find combined_results.json
        for p in bpnet_multi_path.rglob("combined_results.json"):
            combined_path = p
            break

    if combined_path and combined_path.exists():
        with open(combined_path) as f:
            bpnet_raw = json.load(f)
        # Results may be nested under 'all_results' key
        bpnet_per_tf = bpnet_raw.get("all_results", bpnet_raw)
        print(f"  Loaded per-TF BPNet results from {combined_path}")

        # Build fair comparison table
        tf_names = cfg["tf_names"]
        fair_comparison = {}
        for tf in tf_names:
            bp = bpnet_per_tf.get(tf, {})
            bp_r = bp.get("test_pearson_mean", 0)
            fair_comparison[tf] = {
                'bpnet_r': bp_r,
                'bpnet_epochs': bp.get('best_epoch', 0),
            }

        # Get BEACON per-TF results
        outputs = run_model_batch(model, seqs_full, device)
        for ti, tf in enumerate(tf_names):
            mask = tf_idx_full == ti
            if mask.sum() > 0:
                tf_r = per_sample_pearson(outputs['profile'][mask], profs_full[mask])
                fair_comparison[tf]['beacon_r'] = float(tf_r.mean())
                fair_comparison[tf]['beacon_n'] = int(mask.sum())
                delta = fair_comparison[tf]['beacon_r'] - fair_comparison[tf]['bpnet_r']
                fair_comparison[tf]['delta'] = delta
                print(f"    {tf}: BEACON {fair_comparison[tf]['beacon_r']:.3f} vs "
                      f"BPNet {fair_comparison[tf]['bpnet_r']:.3f} (Δ={delta:+.3f})")

        # Summary
        beacon_vals = [v['beacon_r'] for v in fair_comparison.values() if 'beacon_r' in v]
        bpnet_vals = [v['bpnet_r'] for v in fair_comparison.values() if 'bpnet_r' in v]
        import numpy as np
        print(f"\n    Mean: BEACON {np.mean(beacon_vals):.3f} vs "
              f"7×BPNet {np.mean(bpnet_vals):.3f} "
              f"(Δ={np.mean(beacon_vals) - np.mean(bpnet_vals):+.3f})")
        print(f"    BEACON: 1 model (851K params) vs BPNet: 7 models (997K params total)")

        results['fair_bpnet_comparison'] = fair_comparison
    else:
        print("  No per-TF BPNet results found")

    path = out_dir / "robustness_analysis.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
