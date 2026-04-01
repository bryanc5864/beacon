"""
Inference throughput and memory benchmarking for BEACON.

Demonstrates practical applicability: BEACON provides richer outputs
(profiles + positions + TF identity + occupancy) in comparable time.
"""

import time
import numpy as np
import torch


def benchmark_inference(model, input_shape, device, n_warmup=10, n_runs=100):
    """Benchmark inference throughput with accurate GPU timing.

    Args:
        model: Model in eval mode.
        input_shape: Tuple (batch_size, seq_len, channels) for BEACON
                     or (batch_size, channels, seq_len) for BPNet.
        device: Torch device.
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.

    Returns:
        Dict with samples_per_sec, mean_latency_ms, std_latency_ms.
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    batch_size = input_shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    torch.cuda.synchronize()

    # Timed runs using CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]

    with torch.no_grad():
        for i in range(n_runs):
            start_events[i].record()
            _ = model(dummy)
            end_events[i].record()

    torch.cuda.synchronize()

    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    samples_per_sec = batch_size / (mean_ms / 1000.0)

    return {
        'batch_size': batch_size,
        'mean_latency_ms': mean_ms,
        'std_latency_ms': std_ms,
        'samples_per_sec': samples_per_sec,
        'n_runs': n_runs,
    }


def benchmark_memory(model, input_shape, device):
    """Measure peak GPU memory during inference.

    Args:
        model: Model in eval mode.
        input_shape: Input tensor shape.
        device: Torch device.

    Returns:
        Dict with peak_memory_mb, model_memory_mb.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated(device) / 1024 / 1024

    dummy = torch.randn(*input_shape, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    return {
        'peak_memory_mb': float(peak),
        'model_memory_mb': float(baseline),
        'inference_memory_mb': float(peak - baseline),
        'batch_size': input_shape[0],
    }


def count_parameters(model):
    """Count total and trainable parameters.

    Returns:
        Dict with total_params, trainable_params.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
    }
