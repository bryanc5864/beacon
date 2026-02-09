#!/usr/bin/env python3
"""
B4.0 - Precompute Gradient Importance Maps

Precomputes input x gradient importance maps for all training samples.
These maps serve as supervision targets for the Attribution Head, enabling
the model to learn gradient-based feature importance as a direct output
rather than requiring expensive backward passes at inference time.

For each training sample:
  1. Forward pass -> get profile prediction
  2. Compute profile reconstruction loss (KL divergence of predicted vs target profile)
  3. Backward pass -> compute input x gradient
  4. Store as float16 to save space

Output: HDF5 file with 'importance' dataset of shape [N, L, 4] in float16.
Expected size: ~230 MB for N=14406, L=2000.

Supports resuming from a partially completed run.
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import h5py

torch.backends.cudnn.enabled = False

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


def compute_profile_gradient(model, sequence, target_profile, device):
    """Compute input x gradient of profile loss.

    Uses KL divergence between predicted and target profile distributions
    as the scalar loss to differentiate through.

    Args:
        model: BEACON model in eval mode.
        sequence: One-hot encoded sequence tensor [L, 4].
        target_profile: Target binding profile tensor [L].
        device: Torch device.

    Returns:
        importance: numpy array [L, 4] in float16.
    """
    seq = sequence.unsqueeze(0).to(device).requires_grad_(True)
    target = target_profile.unsqueeze(0).to(device)

    outputs = model(seq)
    pred_profile = outputs["profile"]

    # Profile loss (simplified KL divergence)
    pred_prob = pred_profile / (pred_profile.sum(dim=-1, keepdim=True) + 1e-6)
    target_prob = target / (target.sum(dim=-1, keepdim=True) + 1e-6)
    pred_prob = pred_prob + 1e-6
    target_prob = target_prob + 1e-6
    loss = F.kl_div(pred_prob.log(), target_prob, reduction='batchmean')

    loss.backward()

    importance = (seq.grad[0] * seq.data[0]).detach().cpu().numpy()
    return importance.astype(np.float16)


def main():
    parser = argparse.ArgumentParser(
        description="B4.0 - Precompute gradient importance maps for Attribution Head supervision"
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("outputs/phase10_slot_ablation/slots_16/beacon_20260203_231900/best_model.pt"),
        help="Path to trained BEACON model checkpoint",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/processed/multi_tf_k562"),
        help="Path to directory containing train.h5",
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=Path("data/processed/multi_tf_k562/gradient_importance.h5"),
        help="Path for output HDF5 file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (gradient computation requires individual backward passes, so this controls checkpoint frequency logging only)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Resolve paths relative to project root if not absolute
    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_path = args.output_path
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    train_path = data_dir / "train.h5"

    # Model hyperparameters
    seq_len = 2000
    n_slots = 16
    backbone_dim = 128
    slot_dim = 128
    n_iterations = 3
    n_tfs = 7

    print("=" * 70)
    print("B4.0 - Precompute Gradient Importance Maps")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Train data: {train_path}")
    print(f"Output:     {output_path}")
    print(f"Device:     {args.device}")
    print()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("--- Loading Model ---")
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found: {model_path}")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = BEACON(
        seq_len=seq_len, input_channels=4,
        backbone_type="dilated",
        backbone_dim=backbone_dim,
        backbone_layers=4,
        n_slots=n_slots,
        slot_dim=slot_dim,
        n_iterations=n_iterations,
        n_tfs=n_tfs,
        position_mode="gaussian",
        dropout=0.0,
        attention_mode="independent",
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model with {n_params:,} parameters")
    print(f"  seq_len={seq_len}, n_slots={n_slots}, backbone_dim={backbone_dim}")
    print(f"  slot_dim={slot_dim}, n_iterations={n_iterations}, n_tfs={n_tfs}")
    print()

    # ------------------------------------------------------------------
    # 2. Load training dataset
    # ------------------------------------------------------------------
    print("--- Loading Training Data ---")
    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        return 1

    dataset = BEACONDataset(
        str(train_path),
        seq_length=seq_len,
        augment=False,
        n_slots=n_slots,
        extract_peaks=False,
    )
    n_samples = len(dataset)
    print(f"  {n_samples} training samples")
    print(f"  Output shape will be: [{n_samples}, {seq_len}, 4] float16")
    estimated_mb = n_samples * seq_len * 4 * 2 / (1024 * 1024)
    print(f"  Estimated output size: {estimated_mb:.0f} MB")
    print()

    # ------------------------------------------------------------------
    # 3. Handle resume: check existing output file
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_idx = 0

    if output_path.exists():
        print("--- Checking Existing Output (Resume Support) ---")
        try:
            with h5py.File(output_path, "r") as f:
                if "importance" in f and "n_completed" in f.attrs:
                    existing_n = f.attrs["n_completed"]
                    existing_shape = f["importance"].shape
                    if existing_shape == (n_samples, seq_len, 4):
                        start_idx = int(existing_n)
                        print(f"  Found existing file with {start_idx}/{n_samples} samples completed")
                        if start_idx >= n_samples:
                            print("  All samples already computed. Nothing to do.")
                            return 0
                        print(f"  Resuming from sample {start_idx}")
                    else:
                        print(f"  Existing file has wrong shape {existing_shape}, expected ({n_samples}, {seq_len}, 4)")
                        print("  Starting from scratch")
                        start_idx = 0
                else:
                    print("  Existing file missing metadata, starting from scratch")
                    start_idx = 0
        except Exception as e:
            print(f"  Could not read existing file ({e}), starting from scratch")
            start_idx = 0
        print()

    # ------------------------------------------------------------------
    # 4. Create or open HDF5 output file
    # ------------------------------------------------------------------
    if start_idx == 0:
        print("--- Creating Output File ---")
        with h5py.File(output_path, "w") as f:
            f.create_dataset(
                "importance",
                shape=(n_samples, seq_len, 4),
                dtype=np.float16,
                chunks=(1, seq_len, 4),
                compression="gzip",
                compression_opts=4,
            )
            f.attrs["n_completed"] = 0
            f.attrs["n_samples"] = n_samples
            f.attrs["seq_len"] = seq_len
            f.attrs["model_path"] = str(model_path)
            f.attrs["description"] = (
                "Input x gradient importance maps computed from profile KL divergence loss. "
                "Shape: [N, L, 4] where N=number of samples, L=sequence length, 4=nucleotide channels."
            )
        print(f"  Created {output_path}")
        print()

    # ------------------------------------------------------------------
    # 5. Compute gradients sample by sample
    # ------------------------------------------------------------------
    print("--- Computing Gradient Importance Maps ---")
    save_interval = 1000
    log_interval = 100

    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("  (tqdm not available, using basic progress logging)")

    # Track statistics
    importance_abs_sum = 0.0
    importance_max_val = 0.0
    importance_nonzero_frac = 0.0
    n_processed = 0
    n_errors = 0

    t_start = time.time()

    # Open HDF5 file for writing
    h5file = h5py.File(output_path, "a")
    importance_ds = h5file["importance"]

    if use_tqdm:
        iterator = tqdm(range(start_idx, n_samples), desc="Computing gradients",
                        initial=start_idx, total=n_samples, unit="sample")
    else:
        iterator = range(start_idx, n_samples)

    try:
        for i in iterator:
            sample = dataset[i]
            sequence = sample["sequence"]  # [L, 4]

            # Check that we have a target profile
            if "profile" not in sample:
                # If no profile available, use zero gradient
                importance_ds[i] = np.zeros((seq_len, 4), dtype=np.float16)
                n_processed += 1
                continue

            target_profile = sample["profile"]  # [L]

            try:
                model.zero_grad()
                importance = compute_profile_gradient(
                    model, sequence, target_profile, device
                )
                importance_ds[i] = importance

                # Accumulate statistics
                abs_imp = np.abs(importance.astype(np.float32))
                importance_abs_sum += abs_imp.sum()
                importance_max_val = max(importance_max_val, abs_imp.max())
                importance_nonzero_frac += (abs_imp > 1e-7).mean()
                n_processed += 1

            except Exception as e:
                # On error, store zeros and continue
                importance_ds[i] = np.zeros((seq_len, 4), dtype=np.float16)
                n_errors += 1
                n_processed += 1
                if n_errors <= 5:
                    print(f"\n  WARNING: Error at sample {i}: {e}")
                elif n_errors == 6:
                    print(f"\n  WARNING: Suppressing further error messages...")

            # Periodic save
            total_done = i - start_idx + 1
            if total_done % save_interval == 0:
                h5file.attrs["n_completed"] = i + 1
                h5file.flush()
                if not use_tqdm:
                    elapsed = time.time() - t_start
                    rate = total_done / elapsed
                    eta = (n_samples - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{n_samples}] "
                          f"{rate:.1f} samples/sec, "
                          f"ETA: {eta/60:.1f} min, "
                          f"errors: {n_errors}")

            # Basic progress logging when tqdm is not available
            elif not use_tqdm and total_done % log_interval == 0:
                elapsed = time.time() - t_start
                rate = total_done / elapsed
                eta = (n_samples - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{n_samples}] "
                      f"{rate:.1f} samples/sec, "
                      f"ETA: {eta/60:.1f} min")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving progress at sample {i}...")
        h5file.attrs["n_completed"] = i
        h5file.flush()
        h5file.close()
        print(f"Progress saved. Resume by running the script again.")
        return 1

    # Final save
    h5file.attrs["n_completed"] = n_samples
    h5file.flush()
    h5file.close()

    elapsed = time.time() - t_start
    total_computed = n_samples - start_idx

    # ------------------------------------------------------------------
    # 6. Print summary statistics
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total samples:      {n_samples}")
    print(f"  Newly computed:     {total_computed}")
    print(f"  Errors:             {n_errors}")
    print(f"  Time elapsed:       {elapsed:.1f} sec ({elapsed/60:.1f} min)")
    if total_computed > 0:
        print(f"  Rate:               {total_computed/elapsed:.1f} samples/sec")
    print()

    if n_processed > 0:
        mean_abs = importance_abs_sum / (n_processed * seq_len * 4)
        mean_nonzero = importance_nonzero_frac / n_processed
        print(f"  Importance statistics:")
        print(f"    Mean |importance|:       {mean_abs:.6f}")
        print(f"    Max |importance|:        {importance_max_val:.6f}")
        print(f"    Mean non-zero fraction:  {mean_nonzero:.4f}")
    print()

    # Verify output
    print("--- Verifying Output ---")
    with h5py.File(output_path, "r") as f:
        ds = f["importance"]
        print(f"  File:       {output_path}")
        print(f"  Shape:      {ds.shape}")
        print(f"  Dtype:      {ds.dtype}")
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size:  {file_size_mb:.1f} MB")
        print(f"  Completed:  {f.attrs['n_completed']}/{f.attrs['n_samples']}")

        # Quick sanity check: read a few samples
        sample_indices = [0, n_samples // 2, n_samples - 1]
        for idx in sample_indices:
            imp = ds[idx]
            abs_imp = np.abs(imp.astype(np.float32))
            print(f"  Sample {idx}: mean |imp|={abs_imp.mean():.6f}, "
                  f"max |imp|={abs_imp.max():.6f}, "
                  f"non-zero={( abs_imp > 1e-7).mean():.4f}")

    print()
    print(f"Gradient importance maps saved to: {output_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
