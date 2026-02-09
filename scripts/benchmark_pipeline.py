#!/usr/bin/env python3
"""
B5: Benchmarking BEACON Attribution + Motif Extraction vs Traditional Pipeline

Benchmarks BEACON's Phase 1 extensions (attribution head + PerSlotMotifExtractor)
against the traditional BPNet + DeepSHAP + TF-MoDISco pipeline.

Three pipelines compared:
1. BEACON Attribution Pipeline: single forward pass -> attribution [K,L,4] -> motifs
2. BEACON Gradient Pipeline: forward + backward per TF -> input x gradient -> motifs
3. Traditional Pipeline (BPNet + DeepSHAP + TF-MoDISco): simulated timing from literature

Metrics:
- Speed: ms/sample for each pipeline
- Quality: JASPAR Pearson r for discovered motifs, per TF
- Novel motifs: clusters far from known TF anchors (if motif embedding head available)

Usage:
    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --device cuda --max-samples 500
    python scripts/benchmark_pipeline.py --model-path outputs/phase1_extensions/stage3/best_model.pt
"""

import sys
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset
from beacon.analysis.motif_extraction import PerSlotMotifExtractor

base_dir = Path(__file__).parent.parent

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]

# Literature timing constants for traditional pipeline (BPNet + DeepSHAP + TF-MoDISco)
LITERATURE_BPNET_FORWARD_MS = 5.0          # BPNet forward pass per sample
LITERATURE_DEEPSHAP_MS = 500.0             # DeepSHAP per sample (Lundberg et al.)
LITERATURE_TFMODISCO_HOURS_1K = 2.0        # TF-MoDISco for 1000 samples (Shrikumar et al.)
LITERATURE_TFMODISCO_HOURS_10K = 8.0       # TF-MoDISco for 10000 samples


def load_jaspar_meme(meme_path):
    """Load JASPAR motifs from MEME format file.

    Parses the standard MEME motif format, extracting the motif name
    from the MOTIF line and the position frequency matrix from the
    letter-probability section.

    Args:
        meme_path: Path to MEME format file.

    Returns:
        Dictionary mapping motif name -> PFM numpy array [L, 4].
    """
    motifs = {}
    current_name = None
    current_matrix = []
    in_matrix = False

    with open(meme_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if current_name and current_matrix:
                    motifs[current_name] = np.array(current_matrix)
                parts = line.split()
                current_name = parts[2] if len(parts) > 2 else parts[1]
                current_matrix = []
                in_matrix = False
            elif line.startswith("letter-probability"):
                in_matrix = True
                current_matrix = []
            elif in_matrix and line:
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 4:
                        current_matrix.append(values)
                except ValueError:
                    in_matrix = False
            elif not line:
                in_matrix = False

    if current_name and current_matrix:
        motifs[current_name] = np.array(current_matrix)

    return motifs


def match_jaspar_to_tfs(all_jaspar_motifs, tf_names):
    """Match loaded JASPAR motifs to our TF names (case-insensitive).

    Args:
        all_jaspar_motifs: Dictionary from load_jaspar_meme.
        tf_names: List of TF names to match against.

    Returns:
        Dictionary mapping TF name -> JASPAR PFM array.
    """
    matched = {}
    for tf in tf_names:
        if tf in all_jaspar_motifs:
            matched[tf] = all_jaspar_motifs[tf]
        else:
            for name, motif in all_jaspar_motifs.items():
                if tf.upper() in name.upper():
                    matched[tf] = motif
                    break
    return matched


def compute_input_x_gradient(model, sequence, tf_idx, device):
    """Compute input x gradient importance for a specific TF class.

    Uses the occupancy-weighted TF logit across all slots as the target
    scalar, then backpropagates to get per-base importance via gradient
    times input.

    Args:
        model: BEACON model instance.
        sequence: Input sequence tensor [L, 4].
        tf_idx: Target TF class index.
        device: Torch device.

    Returns:
        Importance map as numpy array [L, 4].
    """
    seq = sequence.unsqueeze(0).to(device).requires_grad_(True)
    outputs = model(seq)
    tf_logits = outputs["tf_logits"][0]      # [K, N_TFs]
    occupancy = outputs["occupancy"][0, :, 0]  # [K]
    tf_score = (tf_logits[:, tf_idx] * occupancy).sum()
    tf_score.backward()
    importance = (seq.grad[0] * seq.data[0]).detach().cpu().numpy()
    return importance


def extract_motif_denovo(importance, window=24, top_k=10):
    """Extract a PWM from an importance map by finding top peaks de novo.

    Finds the highest importance peaks, extracts windows around them,
    computes a weighted-average window, converts to a PFM, and trims
    low-information-content edges.

    Args:
        importance: Importance map [L, 4].
        window: Window size around each peak.
        top_k: Number of top peaks to consider.

    Returns:
        PFM numpy array [motif_len, 4] or None if extraction fails.
    """
    total_imp = np.abs(importance).sum(axis=1)

    # Smooth for peak detection
    kernel = np.ones(3) / 3
    smoothed = np.convolve(total_imp, kernel, mode="same")

    # Find peaks above 90th percentile
    threshold = np.percentile(smoothed, 90)
    peaks = []
    for i in range(2, len(smoothed) - 2):
        if (smoothed[i] > threshold
                and smoothed[i] >= smoothed[i - 1]
                and smoothed[i] >= smoothed[i + 1]
                and smoothed[i] >= smoothed[i - 2]
                and smoothed[i] >= smoothed[i + 2]):
            peaks.append((smoothed[i], i))

    peaks.sort(reverse=True)
    peaks = peaks[:top_k]

    if not peaks:
        return None

    half_w = window // 2
    all_windows = []
    weights = []

    for score, pos in peaks:
        start = max(0, pos - half_w)
        end = min(len(importance), pos + half_w)
        w = importance[start:end]
        if len(w) == window:
            all_windows.append(w)
            weights.append(score)

    if not all_windows:
        return None

    weights = np.array(weights)
    weights /= weights.sum()

    avg_window = np.zeros((window, 4))
    for w, weight in zip(all_windows, weights):
        avg_window += w * weight

    # Convert to PFM
    pos_imp = np.maximum(avg_window, 0)
    row_sums = pos_imp.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pwm = pos_imp / row_sums

    # Trim by information content (IC = log2(4) + sum_b p(b)*log2(p(b)))
    info_content = np.log2(4) + np.sum(pwm * np.log2(pwm + 1e-8), axis=1)
    informative = info_content > 0.3
    if informative.sum() < 4:
        return pwm

    first = np.argmax(informative)
    last = len(informative) - 1 - np.argmax(informative[::-1])
    return pwm[first:last + 1]


def compare_motifs_pearson(discovered, reference):
    """Best-alignment Pearson correlation between discovered and reference PFM.

    Slides the shorter motif along the longer one at all valid offsets,
    computes Pearson r for each overlap (including reverse complement),
    and returns the maximum.

    Args:
        discovered: Discovered PFM [L1, 4].
        reference: Reference PFM [L2, 4].

    Returns:
        Best Pearson correlation (float).
    """
    if discovered is None or reference is None:
        return 0.0

    d_len, r_len = len(discovered), len(reference)
    best_r = 0.0

    for offset in range(-(d_len - 4), r_len - 3):
        d_start = max(0, -offset)
        d_end = min(d_len, r_len - offset)
        r_start = max(0, offset)
        r_end = min(r_len, offset + d_len)

        overlap = d_end - d_start
        if overlap < 4:
            continue

        d_flat = discovered[d_start:d_end].flatten()
        r_flat = reference[r_start:r_end].flatten()

        min_len = min(len(d_flat), len(r_flat))
        d_flat, r_flat = d_flat[:min_len], r_flat[:min_len]

        d_c = d_flat - d_flat.mean()
        r_c = r_flat - r_flat.mean()
        denom = np.sqrt((d_c ** 2).sum() * (r_c ** 2).sum())
        if denom < 1e-8:
            continue

        r_val = (d_c * r_c).sum() / denom

        # Also try reverse complement
        d_rc = discovered[d_start:d_end][::-1, ::-1].flatten()[:min_len]
        d_rc_c = d_rc - d_rc.mean()
        denom_rc = np.sqrt((d_rc_c ** 2).sum() * (r_c ** 2).sum())
        if denom_rc > 1e-8:
            r_val = max(r_val, (d_rc_c * r_c).sum() / denom_rc)

        best_r = max(best_r, r_val)

    return float(best_r)


def find_model_path(base_dir):
    """Search for a model checkpoint using the priority order:
    1. Phase 1 extensions stage 3
    2. Phase 1 extensions stage 2
    3. Phase 10 slot ablation (slots_16)

    Args:
        base_dir: Project root directory.

    Returns:
        Tuple of (model_path, has_attribution_head, has_motif_head).
    """
    # Priority 1: Phase 1 extensions stage 3
    p1_s3 = base_dir / "outputs" / "phase1_extensions" / "stage3" / "best_model.pt"
    if p1_s3.exists():
        return p1_s3, True, True

    # Priority 2: Phase 1 extensions stage 2
    p1_s2 = base_dir / "outputs" / "phase1_extensions" / "stage2" / "best_model.pt"
    if p1_s2.exists():
        return p1_s2, True, False

    # Priority 3: Phase 10 slot ablation slots_16
    p10 = (base_dir / "outputs" / "phase10_slot_ablation" / "slots_16"
           / "beacon_20260203_231900" / "best_model.pt")
    if p10.exists():
        return p10, False, False

    return None, False, False


def load_model_config(checkpoint_path):
    """Load model configuration from config.json near the checkpoint.

    Searches for config.json in the checkpoint directory and its parent.

    Args:
        checkpoint_path: Path to the model checkpoint.

    Returns:
        Configuration dictionary with defaults filled in.
    """
    config = {}
    for p in [checkpoint_path.parent / "config.json",
              checkpoint_path.parent.parent / "config.json"]:
        if p.exists():
            with open(p) as f:
                config = json.load(f)
            break

    # Apply defaults
    defaults = {
        "n_tfs": 7,
        "seq_len": 2000,
        "n_slots": 16,
        "backbone_dim": 128,
        "backbone_layers": 4,
        "slot_dim": 128,
        "n_iterations": 3,
    }
    for k, v in defaults.items():
        if k not in config:
            # Check alternate key names
            alt_keys = {"n_tfs": "num_tfs", "seq_len": "seq_length", "n_slots": "num_slots"}
            alt = alt_keys.get(k)
            if alt and alt in config:
                config[k] = config[alt]
            else:
                config[k] = v

    return config


def build_model(config, has_attribution_head, has_motif_head):
    """Construct a BEACON model from the given configuration.

    Args:
        config: Model configuration dictionary.
        has_attribution_head: Whether to include the attribution head.
        has_motif_head: Whether to include the motif embedding head.

    Returns:
        BEACON model instance (on CPU).
    """
    model = BEACON(
        seq_len=config["seq_len"],
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=config["backbone_dim"],
        backbone_layers=config["backbone_layers"],
        n_slots=config["n_slots"],
        slot_dim=config["slot_dim"],
        n_iterations=config["n_iterations"],
        n_tfs=config["n_tfs"],
        position_mode="gaussian",
        dropout=0.0,
        use_attribution_head=has_attribution_head,
        use_motif_embedding=has_motif_head,
        motif_embed_dim=64,
        attention_mode="independent",
    )
    return model


def load_checkpoint(model, checkpoint_path, has_attribution_head, has_motif_head):
    """Load model weights from checkpoint, handling mismatches gracefully.

    If the model expects attribution/motif heads but the checkpoint does
    not contain them (or vice versa), loads with strict=False and retries
    without those heads if needed.

    Args:
        model: BEACON model instance.
        checkpoint_path: Path to checkpoint file.
        has_attribution_head: Whether attribution head was requested.
        has_motif_head: Whether motif head was requested.

    Returns:
        Tuple of (model, actual_has_attribution, actual_has_motif).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean up DataParallel prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Check what heads the checkpoint actually has
    ckpt_has_attr = any(k.startswith("attribution_head.") for k in state_dict)
    ckpt_has_motif = any(k.startswith("motif_head.") for k in state_dict)

    # If mismatch, rebuild model with correct head configuration
    actual_attr = has_attribution_head and ckpt_has_attr
    actual_motif = has_motif_head and ckpt_has_motif

    if actual_attr != has_attribution_head or actual_motif != has_motif_head:
        config_dict = {
            "seq_len": model.seq_len,
            "n_slots": model.n_slots,
            "slot_dim": model.slot_dim,
            "n_tfs": model.n_tfs,
            "backbone_dim": model.backbone.output_dim if hasattr(model.backbone, "output_dim") else 128,
            "backbone_layers": 4,
            "n_iterations": 3,
        }
        # Re-read config from model
        if hasattr(model, "config"):
            for k, v in model.config.items():
                if k in config_dict:
                    config_dict[k] = v

        model = build_model(
            {**config_dict,
             "backbone_dim": config_dict.get("backbone_dim", 128),
             "backbone_layers": config_dict.get("backbone_layers", 4),
             "n_iterations": config_dict.get("n_iterations", 3)},
            actual_attr, actual_motif,
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Try strict=False for partial matches
        model.load_state_dict(state_dict, strict=False)

    return model, actual_attr, actual_motif


def benchmark_attribution_pipeline(model, dataset, device, max_samples, config,
                                   jaspar_motifs, extractor):
    """Run BEACON attribution pipeline benchmark.

    For models with an attribution head, performs a single forward pass
    per sample to obtain per-slot attribution maps, then uses
    PerSlotMotifExtractor to discover motifs. Measures timing and
    compares discovered motifs to JASPAR references.

    Args:
        model: BEACON model with attribution head.
        dataset: BEACONDataset instance.
        device: Torch device.
        max_samples: Maximum number of samples to process.
        config: Model configuration dict.
        jaspar_motifs: Dict mapping TF name -> JASPAR PFM.
        extractor: PerSlotMotifExtractor instance.

    Returns:
        Dictionary with timing, quality, and per-TF results.
    """
    print("\n" + "=" * 70)
    print("Pipeline 1: BEACON Attribution (Single Forward Pass)")
    print("=" * 70)

    n_samples = min(len(dataset), max_samples)
    n_tfs = config["n_tfs"]

    # Collect per-TF motifs for quality assessment
    tf_motifs = defaultdict(list)  # tf_idx -> list of (pfm, ic)
    total_motifs_found = 0
    timing_forward = []
    timing_extraction = []

    model.eval()

    # Warmup
    with torch.no_grad():
        warmup_seq = dataset[0]["sequence"].unsqueeze(0).to(device)
        _ = model(warmup_seq)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for i in range(n_samples):
        sample = dataset[i]
        seq = sample["sequence"].unsqueeze(0).to(device)
        tf_idx = sample["tf_index"].item() if "tf_index" in sample else -1

        # Time forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(seq)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timing_forward.append(t1 - t0)

        # Extract attribution, occupancy, tf_logits
        attribution = outputs["attribution"][0].cpu().numpy()  # [K, L, 4]
        occupancy = outputs["occupancy"][0, :, 0].cpu().numpy()  # [K]
        tf_logits = outputs["tf_logits"][0].cpu().numpy()  # [K, N_TFs]
        seq_np = sample["sequence"].numpy()  # [L, 4]
        tf_preds = tf_logits.argmax(axis=-1)  # [K]

        # Time motif extraction
        t2 = time.perf_counter()
        motifs = extractor.extract_from_sample(
            attribution=attribution,
            occupancy=occupancy,
            sequence=seq_np,
            tf_predictions=tf_preds,
        )
        t3 = time.perf_counter()
        timing_extraction.append(t3 - t2)

        total_motifs_found += len(motifs)

        for motif in motifs:
            pred_tf = motif.get("tf_id", -1)
            if pred_tf is not None and 0 <= pred_tf < n_tfs:
                tf_motifs[pred_tf].append(motif)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples, "
                  f"{total_motifs_found} motifs found so far")

    # Compute quality: best motif per TF vs JASPAR
    print(f"\n  Total motifs found: {total_motifs_found}")
    per_tf_quality = {}

    for tf_idx in range(n_tfs):
        tf_name = TF_NAMES[tf_idx] if tf_idx < len(TF_NAMES) else f"TF_{tf_idx}"
        motif_list = tf_motifs.get(tf_idx, [])
        if not motif_list:
            per_tf_quality[tf_name] = {
                "n_motifs": 0,
                "best_jaspar_r": 0.0,
                "mean_ic": 0.0,
            }
            continue

        # Find best motif by IC
        best_motif = max(motif_list, key=lambda m: m.get("mean_ic", 0.0))
        pfm = best_motif["pfm"]
        mean_ic = best_motif.get("mean_ic", 0.0)

        # Compare to JASPAR
        jaspar_r = 0.0
        if tf_name in jaspar_motifs:
            jaspar_r = PerSlotMotifExtractor.compare_to_jaspar(pfm, jaspar_motifs[tf_name])

        per_tf_quality[tf_name] = {
            "n_motifs": len(motif_list),
            "best_jaspar_r": float(jaspar_r),
            "mean_ic": float(mean_ic),
            "motif_length": int(best_motif.get("motif_length", len(pfm))),
        }

    # Timing summary
    avg_forward_ms = np.mean(timing_forward) * 1000
    avg_extraction_ms = np.mean(timing_extraction) * 1000
    avg_total_ms = avg_forward_ms + avg_extraction_ms

    results = {
        "pipeline": "beacon_attribution",
        "n_samples": n_samples,
        "total_motifs_found": total_motifs_found,
        "timing": {
            "avg_forward_ms": float(avg_forward_ms),
            "avg_extraction_ms": float(avg_extraction_ms),
            "avg_total_ms": float(avg_total_ms),
            "median_forward_ms": float(np.median(timing_forward) * 1000),
            "median_total_ms": float((np.median(timing_forward)
                                      + np.median(timing_extraction)) * 1000),
        },
        "per_tf_quality": per_tf_quality,
    }

    # Print summary
    print(f"\n  Timing:")
    print(f"    Forward pass:      {avg_forward_ms:.2f} ms/sample")
    print(f"    Motif extraction:  {avg_extraction_ms:.2f} ms/sample")
    print(f"    Total:             {avg_total_ms:.2f} ms/sample")

    valid_rs = [v["best_jaspar_r"] for v in per_tf_quality.values()
                if v["best_jaspar_r"] > 0]
    if valid_rs:
        print(f"\n  JASPAR correlation (attribution pipeline):")
        for tf_name, info in per_tf_quality.items():
            if info["n_motifs"] > 0:
                print(f"    {tf_name:<8}: r = {info['best_jaspar_r']:.3f}  "
                      f"(IC={info['mean_ic']:.3f}, n={info['n_motifs']}, "
                      f"len={info.get('motif_length', '?')})")
        print(f"    Mean JASPAR r: {np.mean(valid_rs):.3f}")

    return results


def benchmark_gradient_pipeline(model, dataset, device, max_samples_per_tf,
                                config, jaspar_motifs):
    """Run BEACON gradient pipeline benchmark.

    For each TF, selects samples of that TF class and computes
    input x gradient importance maps, averages them, and extracts
    a de novo motif. Measures timing and compares to JASPAR.

    Args:
        model: BEACON model instance.
        dataset: BEACONDataset instance.
        device: Torch device.
        max_samples_per_tf: Maximum samples per TF for gradient computation.
        config: Model configuration dict.
        jaspar_motifs: Dict mapping TF name -> JASPAR PFM.

    Returns:
        Dictionary with timing, quality, and per-TF results.
    """
    print("\n" + "=" * 70)
    print("Pipeline 2: BEACON Gradient (Forward + Backward per TF)")
    print("=" * 70)

    n_tfs = config["n_tfs"]

    # Organize samples by TF
    tf_samples = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        tf_idx = sample["tf_index"].item() if "tf_index" in sample else -1
        if 0 <= tf_idx < n_tfs and len(tf_samples[tf_idx]) < max_samples_per_tf:
            tf_samples[tf_idx].append(sample["sequence"])

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx] if tf_idx < len(TF_NAMES) else f"TF_{tf_idx}"
        print(f"  {tf_name}: {len(tf_samples[tf_idx])} samples")

    # Warmup
    if tf_samples:
        first_tf = list(tf_samples.keys())[0]
        seq_w = tf_samples[first_tf][0].unsqueeze(0).to(device).requires_grad_(True)
        out_w = model(seq_w)
        dummy_score = out_w["tf_logits"][0, :, first_tf].sum()
        dummy_score.backward()
        model.zero_grad()

    per_tf_quality = {}
    timing_per_sample = []
    total_samples_computed = 0

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx] if tf_idx < len(TF_NAMES) else f"TF_{tf_idx}"
        samples = tf_samples[tf_idx]

        all_importance = []
        tf_times = []

        for seq_tensor in samples:
            try:
                model.zero_grad()
                t0 = time.perf_counter()
                imp = compute_input_x_gradient(model, seq_tensor, tf_idx, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                all_importance.append(imp)
                tf_times.append(t1 - t0)
                total_samples_computed += 1
            except Exception:
                continue

        if not all_importance:
            print(f"  {tf_name}: no valid gradients")
            per_tf_quality[tf_name] = {
                "n_samples": 0,
                "jaspar_r": 0.0,
                "mean_ic": 0.0,
            }
            continue

        timing_per_sample.extend(tf_times)

        # Average importance across samples
        avg_importance = np.mean(all_importance, axis=0)

        # Extract motif de novo
        motif = extract_motif_denovo(avg_importance, window=24, top_k=15)

        jaspar_r = 0.0
        mean_ic = 0.0
        motif_length = 0

        if motif is not None:
            # Information content
            ic = np.log2(4) + np.sum(motif * np.log2(motif + 1e-8), axis=1)
            mean_ic = float(ic.mean())
            motif_length = len(motif)

            # Compare to JASPAR
            if tf_name in jaspar_motifs:
                jaspar_r = compare_motifs_pearson(motif, jaspar_motifs[tf_name])

        per_tf_quality[tf_name] = {
            "n_samples": len(all_importance),
            "jaspar_r": float(jaspar_r),
            "mean_ic": float(mean_ic),
            "motif_length": motif_length,
            "avg_time_ms": float(np.mean(tf_times) * 1000),
        }

        print(f"  {tf_name}: r = {jaspar_r:.3f}, IC = {mean_ic:.3f}, "
              f"len = {motif_length}, n = {len(all_importance)}, "
              f"avg = {np.mean(tf_times)*1000:.1f} ms/sample")

    # Timing summary
    avg_per_sample_ms = np.mean(timing_per_sample) * 1000 if timing_per_sample else 0.0
    # Full pipeline per sample: n_tfs backward passes
    avg_full_pipeline_ms = avg_per_sample_ms * n_tfs

    results = {
        "pipeline": "beacon_gradient",
        "n_tfs": n_tfs,
        "total_samples_computed": total_samples_computed,
        "timing": {
            "avg_per_backward_ms": float(avg_per_sample_ms),
            "avg_full_pipeline_ms": float(avg_full_pipeline_ms),
            "median_per_backward_ms": float(np.median(timing_per_sample) * 1000)
            if timing_per_sample else 0.0,
            "note": f"Full pipeline = {n_tfs} backward passes per sample",
        },
        "per_tf_quality": per_tf_quality,
    }

    valid_rs = [v["jaspar_r"] for v in per_tf_quality.values() if v["jaspar_r"] > 0]
    if valid_rs:
        print(f"\n  Mean JASPAR r (gradient): {np.mean(valid_rs):.3f}")

    print(f"\n  Timing:")
    print(f"    Per backward pass: {avg_per_sample_ms:.2f} ms")
    print(f"    Full pipeline ({n_tfs} TFs): {avg_full_pipeline_ms:.2f} ms/sample")

    return results


def compute_traditional_pipeline_timing(n_samples, n_tfs):
    """Compute simulated timing for BPNet + DeepSHAP + TF-MoDISco.

    Uses literature-reported timing values since we cannot run the
    actual traditional pipeline.

    Args:
        n_samples: Number of samples that would be processed.
        n_tfs: Number of TFs.

    Returns:
        Dictionary with per-sample and total timing estimates.
    """
    bpnet_forward_total = n_samples * LITERATURE_BPNET_FORWARD_MS
    deepshap_total = n_samples * LITERATURE_DEEPSHAP_MS
    # TF-MoDISco scales roughly linearly with samples
    tfmodisco_total_ms = (LITERATURE_TFMODISCO_HOURS_1K * 3600 * 1000
                          * (n_samples / 1000))

    per_sample_ms = (LITERATURE_BPNET_FORWARD_MS + LITERATURE_DEEPSHAP_MS
                     + tfmodisco_total_ms / n_samples)

    return {
        "pipeline": "traditional_bpnet_deepshap_modisco",
        "source": "literature_estimates",
        "n_samples": n_samples,
        "timing": {
            "bpnet_forward_ms_per_sample": LITERATURE_BPNET_FORWARD_MS,
            "deepshap_ms_per_sample": LITERATURE_DEEPSHAP_MS,
            "tfmodisco_hours_for_1k_samples": LITERATURE_TFMODISCO_HOURS_1K,
            "tfmodisco_hours_for_10k_samples": LITERATURE_TFMODISCO_HOURS_10K,
            "total_per_sample_ms": float(per_sample_ms),
            "bpnet_forward_total_ms": float(bpnet_forward_total),
            "deepshap_total_ms": float(deepshap_total),
            "tfmodisco_total_ms": float(tfmodisco_total_ms),
            "grand_total_ms": float(bpnet_forward_total + deepshap_total
                                    + tfmodisco_total_ms),
        },
        "literature_notes": {
            "bpnet_forward": "BPNet forward pass ~5ms on GPU (Avsec et al. 2021)",
            "deepshap": "DeepSHAP attribution ~500ms per sample (Lundberg & Lee 2017)",
            "tfmodisco": ("TF-MoDISco clustering ~2h for 1K samples, "
                          "~8h for 10K (Shrikumar et al. 2018)"),
            "note": "These are simulated times based on published benchmarks",
        },
    }


def check_novel_motifs(model, dataset, device, config, max_samples=200):
    """Detect potentially novel motifs using the motif embedding head.

    For models with a motif embedding head, computes novelty scores
    (1 - max cosine similarity to any TF anchor prototype) for each
    active slot and reports slots that may represent unknown motifs.

    Args:
        model: BEACON model with motif embedding head.
        dataset: BEACONDataset instance.
        device: Torch device.
        config: Model configuration dict.
        max_samples: Maximum samples to scan.

    Returns:
        Dictionary with novel motif detection results.
    """
    print("\n" + "=" * 70)
    print("Novel Motif Detection (Motif Embedding Head)")
    print("=" * 70)

    n_samples = min(len(dataset), max_samples)
    novelty_scores_all = []
    novel_count = 0
    total_active = 0

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            sample = dataset[i]
            seq = sample["sequence"].unsqueeze(0).to(device)
            outputs = model(seq)

            slots = outputs["slot_embeddings"]  # [1, K, D]
            occupancy = outputs["occupancy"]    # [1, K, 1]

            novel_info = model.motif_head.get_novel_motif_scores(
                slots, occupancy, novelty_threshold=0.5
            )

            novelty = novel_info["novelty_scores"][0].cpu().numpy()  # [K]
            novel_mask = novel_info["novel_mask"][0].cpu().numpy()    # [K]
            active_mask = occupancy[0, :, 0].cpu().numpy() > 0.3

            n_active = int(active_mask.sum())
            n_novel = int(novel_mask.sum())
            total_active += n_active
            novel_count += n_novel

            for k in range(len(novelty)):
                if active_mask[k]:
                    novelty_scores_all.append(novelty[k])

    if novelty_scores_all:
        novelty_arr = np.array(novelty_scores_all)
        print(f"  Total active slots scanned: {total_active}")
        print(f"  Novel slots (similarity < 0.5 to all anchors): {novel_count}")
        print(f"  Novel fraction: {novel_count / max(total_active, 1):.3f}")
        print(f"  Mean novelty score: {novelty_arr.mean():.3f}")
        print(f"  Median novelty score: {np.median(novelty_arr):.3f}")

        return {
            "total_active_slots": total_active,
            "novel_slots": novel_count,
            "novel_fraction": float(novel_count / max(total_active, 1)),
            "mean_novelty_score": float(novelty_arr.mean()),
            "median_novelty_score": float(np.median(novelty_arr)),
            "novelty_histogram": {
                "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "counts": [int(x) for x in np.histogram(
                    novelty_arr,
                    bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                )[0]],
            },
        }
    else:
        print("  No active slots found.")
        return {"total_active_slots": 0, "novel_slots": 0}


def create_benchmark_figure(attr_results, grad_results, trad_results,
                            output_dir, has_attribution):
    """Create matplotlib comparison figure of speed and quality.

    Generates a multi-panel figure:
    - Left: Speed comparison (ms/sample, log scale) across pipelines
    - Right: JASPAR Pearson r per TF for available pipelines

    Args:
        attr_results: Attribution pipeline results (or None).
        grad_results: Gradient pipeline results.
        trad_results: Traditional pipeline results.
        output_dir: Directory to save the figure.
        has_attribution: Whether attribution pipeline results exist.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Speed comparison ---
    ax = axes[0]
    pipelines = []
    times = []
    colors = []

    if has_attribution and attr_results is not None:
        pipelines.append("BEACON\nAttribution")
        times.append(attr_results["timing"]["avg_total_ms"])
        colors.append("#2ecc71")

    pipelines.append("BEACON\nGradient")
    times.append(grad_results["timing"]["avg_full_pipeline_ms"])
    colors.append("#3498db")

    pipelines.append("BPNet+DeepSHAP\n+TF-MoDISco\n(literature)")
    times.append(trad_results["timing"]["total_per_sample_ms"])
    colors.append("#e74c3c")

    bars = ax.bar(pipelines, times, color=colors, alpha=0.8, edgecolor="black",
                  linewidth=0.8)
    ax.set_ylabel("Time per sample (ms)", fontsize=12)
    ax.set_title("Speed Comparison", fontsize=13, fontweight="bold")
    ax.set_yscale("log")

    for bar, t in zip(bars, times):
        if t >= 1000:
            label = f"{t/1000:.1f}s"
        else:
            label = f"{t:.1f}ms"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                label, ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(bottom=0.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Panel 2: Quality comparison (JASPAR r per TF) ---
    ax = axes[1]

    # Collect per-TF JASPAR r values
    tf_labels = []
    attr_rs = []
    grad_rs = []

    for tf_name in TF_NAMES:
        tf_labels.append(tf_name)

        if (has_attribution and attr_results is not None
                and tf_name in attr_results.get("per_tf_quality", {})):
            attr_rs.append(attr_results["per_tf_quality"][tf_name].get("best_jaspar_r", 0.0))
        else:
            attr_rs.append(0.0)

        if tf_name in grad_results.get("per_tf_quality", {}):
            grad_rs.append(grad_results["per_tf_quality"][tf_name].get("jaspar_r", 0.0))
        else:
            grad_rs.append(0.0)

    x = np.arange(len(tf_labels))
    width = 0.35

    if has_attribution and attr_results is not None:
        offset = width / 2
        ax.bar(x - offset, attr_rs, width, label="Attribution", color="#2ecc71",
               alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(x + offset, grad_rs, width, label="Gradient", color="#3498db",
               alpha=0.8, edgecolor="black", linewidth=0.5)
    else:
        ax.bar(x, grad_rs, width, label="Gradient", color="#3498db",
               alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("JASPAR Pearson r", fontsize=12)
    ax.set_title("Motif Quality (JASPAR Correlation)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tf_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="r = 0.5 baseline")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("B5: BEACON Pipeline Benchmarking", fontsize=15, fontweight="bold",
                 y=1.02)
    plt.tight_layout()
    fig_path = output_dir / "benchmark_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved to {fig_path}")


def print_comparison_tables(attr_results, grad_results, trad_results,
                            has_attribution, novel_results):
    """Print formatted comparison tables to stdout.

    Generates three tables:
    1. Speed comparison across all three pipelines
    2. Per-TF motif quality for BEACON methods
    3. Overall summary with speedup factors

    Args:
        attr_results: Attribution pipeline results (or None).
        grad_results: Gradient pipeline results.
        trad_results: Traditional pipeline results.
        has_attribution: Whether attribution pipeline results exist.
        novel_results: Novel motif detection results (or None).
    """
    print("\n" + "=" * 70)
    print("COMPARISON TABLES")
    print("=" * 70)

    # Table 1: Speed
    print("\n--- Table 1: Speed (ms per sample) ---")
    print(f"{'Pipeline':<35} {'Forward':>10} {'Extract':>10} {'Total':>10}")
    print("-" * 68)

    if has_attribution and attr_results is not None:
        a = attr_results["timing"]
        print(f"{'BEACON Attribution':<35} {a['avg_forward_ms']:>10.2f} "
              f"{a['avg_extraction_ms']:>10.2f} {a['avg_total_ms']:>10.2f}")

    g = grad_results["timing"]
    print(f"{'BEACON Gradient':<35} {g['avg_per_backward_ms']:>10.2f} "
          f"{'--':>10} {g['avg_full_pipeline_ms']:>10.2f}")

    t = trad_results["timing"]
    print(f"{'BPNet+DeepSHAP+MoDISco (lit.)':<35} "
          f"{t['bpnet_forward_ms_per_sample']:>10.1f} "
          f"{t['deepshap_ms_per_sample']:>10.1f} "
          f"{t['total_per_sample_ms']:>10.1f}")

    # Table 2: Quality per TF
    print("\n--- Table 2: JASPAR Pearson r per TF ---")
    header = f"{'TF':<8}"
    if has_attribution and attr_results is not None:
        header += f" {'Attribution r':>15} {'Attr n':>8}"
    header += f" {'Gradient r':>15} {'Grad n':>8}"
    print(header)
    print("-" * len(header))

    for tf_name in TF_NAMES:
        row = f"{tf_name:<8}"
        if has_attribution and attr_results is not None:
            aq = attr_results["per_tf_quality"].get(tf_name, {})
            ar = aq.get("best_jaspar_r", 0.0)
            an = aq.get("n_motifs", 0)
            row += f" {ar:>15.3f} {an:>8}"

        gq = grad_results["per_tf_quality"].get(tf_name, {})
        gr = gq.get("jaspar_r", 0.0)
        gn = gq.get("n_samples", 0)
        row += f" {gr:>15.3f} {gn:>8}"
        print(row)

    # Means
    mean_row = f"{'Mean':<8}"
    if has_attribution and attr_results is not None:
        attr_vals = [v.get("best_jaspar_r", 0.0)
                     for v in attr_results["per_tf_quality"].values()
                     if v.get("best_jaspar_r", 0.0) > 0]
        mean_attr = np.mean(attr_vals) if attr_vals else 0.0
        mean_row += f" {mean_attr:>15.3f} {'':>8}"

    grad_vals = [v.get("jaspar_r", 0.0)
                 for v in grad_results["per_tf_quality"].values()
                 if v.get("jaspar_r", 0.0) > 0]
    mean_grad = np.mean(grad_vals) if grad_vals else 0.0
    mean_row += f" {mean_grad:>15.3f} {'':>8}"
    print("-" * len(header))
    print(mean_row)

    # Table 3: Summary
    print("\n--- Table 3: Overall Summary ---")
    trad_ms = trad_results["timing"]["total_per_sample_ms"]

    if has_attribution and attr_results is not None:
        attr_ms = attr_results["timing"]["avg_total_ms"]
        attr_speedup = trad_ms / attr_ms if attr_ms > 0 else float("inf")
        print(f"  BEACON Attribution vs Traditional: {attr_speedup:.0f}x faster")
        print(f"    ({attr_ms:.2f} ms vs {trad_ms:.1f} ms per sample)")

    grad_ms = grad_results["timing"]["avg_full_pipeline_ms"]
    grad_speedup = trad_ms / grad_ms if grad_ms > 0 else float("inf")
    print(f"  BEACON Gradient vs Traditional:    {grad_speedup:.0f}x faster")
    print(f"    ({grad_ms:.2f} ms vs {trad_ms:.1f} ms per sample)")

    if has_attribution and attr_results is not None and grad_ms > 0:
        attr_vs_grad = grad_ms / attr_ms if attr_ms > 0 else float("inf")
        print(f"  Attribution vs Gradient:            {attr_vs_grad:.1f}x faster")

    # Novel motifs summary
    if novel_results is not None and novel_results.get("total_active_slots", 0) > 0:
        print(f"\n  Novel Motifs (motif embedding head):")
        print(f"    Active slots scanned: {novel_results['total_active_slots']}")
        print(f"    Novel slots detected: {novel_results['novel_slots']}")
        print(f"    Novel fraction: {novel_results['novel_fraction']:.3f}")
        print(f"    Mean novelty score: {novel_results['mean_novelty_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="B5: Benchmark BEACON attribution + motif extraction "
                    "vs traditional BPNet+DeepSHAP+TF-MoDISco pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to model checkpoint. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/processed/multi_tf_k562"),
        help="Directory containing test.h5",
    )
    parser.add_argument(
        "--jaspar", type=Path,
        default=Path("data/raw/jaspar/meme_vertebrates.txt"),
        help="Path to JASPAR MEME format file",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/benchmark_pipeline"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Max samples for attribution pipeline benchmark",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    jaspar_path = args.jaspar
    if not jaspar_path.is_absolute():
        jaspar_path = base_dir / jaspar_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("B5: BEACON Pipeline Benchmarking")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Data:   {data_dir}")
    print(f"  JASPAR: {jaspar_path}")
    print(f"  Output: {output_dir}")

    # --- Step 1: Find and load model ---
    print("\n--- Step 1: Loading Model ---")

    if args.model_path is not None:
        model_path = args.model_path
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        # Detect heads from checkpoint contents
        checkpoint_peek = torch.load(model_path, map_location="cpu", weights_only=False)
        state_keys = set()
        if "model_state_dict" in checkpoint_peek:
            state_keys = set(checkpoint_peek["model_state_dict"].keys())
        elif "state_dict" in checkpoint_peek:
            state_keys = set(checkpoint_peek["state_dict"].keys())
        else:
            state_keys = set(checkpoint_peek.keys())
        has_attribution_head = any(k.startswith("attribution_head.") for k in state_keys)
        has_motif_head = any(k.startswith("motif_head.") for k in state_keys)
        del checkpoint_peek
    else:
        model_path, has_attribution_head, has_motif_head = find_model_path(base_dir)

    if model_path is None or not model_path.exists():
        print(f"ERROR: No model checkpoint found. Searched:")
        print(f"  - outputs/phase1_extensions/stage3/best_model.pt")
        print(f"  - outputs/phase1_extensions/stage2/best_model.pt")
        print(f"  - outputs/phase10_slot_ablation/slots_16/beacon_20260203_231900/best_model.pt")
        print(f"Specify --model-path to provide a checkpoint.")
        return 1

    print(f"  Checkpoint: {model_path}")
    print(f"  Attribution head: {has_attribution_head}")
    print(f"  Motif embedding head: {has_motif_head}")

    config = load_model_config(model_path)
    print(f"  Config: seq_len={config['seq_len']}, n_slots={config['n_slots']}, "
          f"n_tfs={config['n_tfs']}, slot_dim={config['slot_dim']}")

    model = build_model(config, has_attribution_head, has_motif_head)
    model, has_attribution_head, has_motif_head = load_checkpoint(
        model, model_path, has_attribution_head, has_motif_head
    )
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")

    # --- Step 2: Load test data ---
    print("\n--- Step 2: Loading Test Data ---")
    test_path = data_dir / "test.h5"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found")
        return 1

    dataset = BEACONDataset(
        str(test_path),
        seq_length=config["seq_len"],
        augment=False,
        n_slots=config["n_slots"],
        extract_peaks=True,
    )
    print(f"  {len(dataset)} test samples loaded")

    # --- Step 3: Load JASPAR references ---
    print("\n--- Step 3: Loading JASPAR References ---")
    jaspar_motifs = {}
    if jaspar_path.exists():
        all_jaspar = load_jaspar_meme(str(jaspar_path))
        jaspar_motifs = match_jaspar_to_tfs(all_jaspar, TF_NAMES)
        for tf in TF_NAMES:
            if tf in jaspar_motifs:
                print(f"  {tf}: {jaspar_motifs[tf].shape[0]}bp motif loaded")
            else:
                print(f"  {tf}: not found in JASPAR")
        print(f"  Total JASPAR motifs loaded: {len(all_jaspar)}, matched: {len(jaspar_motifs)}")
    else:
        print(f"  WARNING: {jaspar_path} not found, JASPAR comparison disabled")

    # --- Step 4: Run BEACON Attribution Pipeline ---
    attr_results = None
    if has_attribution_head:
        extractor = PerSlotMotifExtractor(
            occupancy_threshold=0.3,
            window_size=30,
            ic_threshold=0.3,
        )
        attr_results = benchmark_attribution_pipeline(
            model, dataset, device, args.max_samples, config,
            jaspar_motifs, extractor,
        )
    else:
        print("\n" + "=" * 70)
        print("Pipeline 1: BEACON Attribution -- SKIPPED (no attribution head)")
        print("=" * 70)
        print("  Model does not have attribution head. Using gradient fallback only.")

    # --- Step 5: Run BEACON Gradient Pipeline ---
    max_grad_samples = min(200, args.max_samples)
    grad_results = benchmark_gradient_pipeline(
        model, dataset, device, max_grad_samples, config, jaspar_motifs,
    )

    # --- Step 6: Compute Traditional Pipeline Timing ---
    print("\n" + "=" * 70)
    print("Pipeline 3: Traditional BPNet + DeepSHAP + TF-MoDISco (Simulated)")
    print("=" * 70)

    trad_results = compute_traditional_pipeline_timing(
        n_samples=args.max_samples,
        n_tfs=config["n_tfs"],
    )

    t = trad_results["timing"]
    print(f"  BPNet forward:   {t['bpnet_forward_ms_per_sample']:.1f} ms/sample")
    print(f"  DeepSHAP:        {t['deepshap_ms_per_sample']:.1f} ms/sample")
    print(f"  TF-MoDISco:      {t['tfmodisco_hours_for_1k_samples']:.1f}h for 1K samples, "
          f"{t['tfmodisco_hours_for_10k_samples']:.1f}h for 10K samples")
    print(f"  Total:           {t['total_per_sample_ms']:.1f} ms/sample")
    print(f"\n  Note: These are literature-reported estimates, not measured on this hardware.")
    for key, note in trad_results["literature_notes"].items():
        print(f"    - {note}")

    # --- Step 7: Novel Motif Detection ---
    novel_results = None
    if has_motif_head:
        novel_results = check_novel_motifs(model, dataset, device, config, max_samples=200)

    # --- Step 8: Print Comparison Tables ---
    print_comparison_tables(attr_results, grad_results, trad_results,
                            has_attribution_head, novel_results)

    # --- Step 9: Create Figure ---
    print("\n--- Creating Benchmark Figure ---")
    try:
        create_benchmark_figure(
            attr_results, grad_results, trad_results,
            output_dir, has_attribution_head,
        )
    except Exception as e:
        print(f"  WARNING: Could not create figure: {e}")

    # --- Step 10: Save Results JSON ---
    print("\n--- Saving Results ---")

    all_results = {
        "benchmark": "B5_pipeline_comparison",
        "device": str(device),
        "model_path": str(model_path),
        "n_model_params": n_params,
        "has_attribution_head": has_attribution_head,
        "has_motif_head": has_motif_head,
        "config": {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool))},
        "max_samples": args.max_samples,
        "pipelines": {},
    }

    if attr_results is not None:
        all_results["pipelines"]["beacon_attribution"] = attr_results
    all_results["pipelines"]["beacon_gradient"] = grad_results
    all_results["pipelines"]["traditional_simulated"] = trad_results
    if novel_results is not None:
        all_results["novel_motifs"] = novel_results

    # Compute speedup summary
    trad_ms = trad_results["timing"]["total_per_sample_ms"]
    speedups = {}
    if attr_results is not None:
        attr_ms = attr_results["timing"]["avg_total_ms"]
        speedups["attribution_vs_traditional"] = float(trad_ms / attr_ms) if attr_ms > 0 else 0.0
    grad_ms = grad_results["timing"]["avg_full_pipeline_ms"]
    speedups["gradient_vs_traditional"] = float(trad_ms / grad_ms) if grad_ms > 0 else 0.0
    if attr_results is not None and grad_ms > 0:
        attr_ms = attr_results["timing"]["avg_total_ms"]
        speedups["attribution_vs_gradient"] = float(grad_ms / attr_ms) if attr_ms > 0 else 0.0
    all_results["speedups"] = speedups

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    for name, factor in speedups.items():
        print(f"  {name}: {factor:.0f}x")
    print(f"\n  All outputs in: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
