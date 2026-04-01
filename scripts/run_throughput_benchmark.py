#!/usr/bin/env python3
"""
Inference throughput and memory benchmarking for BEACON vs BPNet.

Demonstrates practical applicability: BEACON provides richer outputs
in comparable time to BPNet.

Usage:
    python scripts/run_throughput_benchmark.py --device cuda:2
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_named_model, json_serialize, BASE_DIR,
)
from beacon.analysis.throughput import (
    benchmark_inference, benchmark_memory, count_parameters,
)


def benchmark_beacon(device, batch_sizes):
    """Benchmark BEACON model."""
    print("\n--- BEACON ---")
    model, cfg = load_named_model("K562-7tf", device)
    params = count_parameters(model)
    print(f"  Parameters: {params['total_params']:,}")

    results = {'parameters': params, 'batch_results': {}}

    for bs in batch_sizes:
        print(f"  Batch size {bs}...")
        input_shape = (bs, 2000, 4)  # BEACON: [B, L, 4]
        try:
            timing = benchmark_inference(model, input_shape, device)
            memory = benchmark_memory(model, input_shape, device)
            results['batch_results'][str(bs)] = {**timing, **memory}
            print(f"    {timing['samples_per_sec']:.0f} samples/sec, "
                  f"{timing['mean_latency_ms']:.1f} ms, "
                  f"{memory['peak_memory_mb']:.0f} MB peak")
        except RuntimeError as e:
            print(f"    OOM at batch size {bs}: {e}")
            results['batch_results'][str(bs)] = {'error': str(e)}

    del model
    torch.cuda.empty_cache()
    return results


def benchmark_bpnet(device, batch_sizes):
    """Benchmark BPNet model if available."""
    print("\n--- BPNet ---")

    # Try to load BPNet
    bpnet_path = BASE_DIR / "bpnet.64.8.final.torch"
    if not bpnet_path.exists():
        bpnet_path = BASE_DIR / "bpnet.64.8.torch"
    if not bpnet_path.exists():
        print("  No BPNet checkpoint found, skipping")
        return None

    try:
        from bpnetlite import BPNet
    except ImportError:
        print("  bpnetlite not installed, skipping")
        return None

    try:
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
    except Exception as e:
        print(f"  Failed to load BPNet: {e}")
        return None

    params = count_parameters(bpnet)
    print(f"  Parameters: {params['total_params']:,}")

    results = {'parameters': params, 'batch_results': {}}

    for bs in batch_sizes:
        print(f"  Batch size {bs}...")
        input_shape = (bs, 4, 2000)  # BPNet: [B, 4, L] channels-first
        try:
            timing = benchmark_inference(bpnet, input_shape, device)
            memory = benchmark_memory(bpnet, input_shape, device)
            results['batch_results'][str(bs)] = {**timing, **memory}
            print(f"    {timing['samples_per_sec']:.0f} samples/sec, "
                  f"{timing['mean_latency_ms']:.1f} ms, "
                  f"{memory['peak_memory_mb']:.0f} MB peak")
        except RuntimeError as e:
            print(f"    OOM at batch size {bs}: {e}")
            results['batch_results'][str(bs)] = {'error': str(e)}

    del bpnet
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Inference Throughput Benchmark")
    print("=" * 70)

    batch_sizes = [1, 8, 32, 64]

    results = {
        'beacon': benchmark_beacon(device, batch_sizes),
        'bpnet': benchmark_bpnet(device, batch_sizes),
    }

    # Summary comparison
    print("\n=== Summary ===")
    if results['beacon'] and results.get('bpnet'):
        for bs in ['32']:
            b_res = results['beacon']['batch_results'].get(bs, {})
            p_res = results['bpnet']['batch_results'].get(bs, {})
            if 'samples_per_sec' in b_res and 'samples_per_sec' in p_res:
                ratio = b_res['samples_per_sec'] / p_res['samples_per_sec']
                print(f"  BS={bs}: BEACON {b_res['samples_per_sec']:.0f} vs "
                      f"BPNet {p_res['samples_per_sec']:.0f} samples/sec "
                      f"(ratio: {ratio:.2f}x)")

    path = out_dir / "throughput_benchmark.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
