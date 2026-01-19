#!/usr/bin/env python3
"""
Quick test to verify BEACON model architecture works correctly.
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model():
    print("Testing BEACON model architecture...")
    print("=" * 60)

    from beacon.models import BEACON, BEACONSmall, BEACONLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create small model for testing
    print("\n1. Creating BEACONSmall model...")
    model = BEACONSmall(
        seq_len=1000,
        n_tfs=10,
        n_slots=16,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    seq_len = 1000

    # Random one-hot encoded sequence
    x = torch.zeros(batch_size, seq_len, 4, device=device)
    indices = torch.randint(0, 4, (batch_size, seq_len), device=device)
    x.scatter_(2, indices.unsqueeze(-1), 1.0)

    print(f"   Input shape: {x.shape}")

    outputs = model(x, return_attention=True, return_slots=True)

    print(f"   Profile shape: {outputs['profile'].shape}")
    print(f"   Positions shape: {outputs['positions'].shape}")
    print(f"   TF logits shape: {outputs['tf_logits'].shape}")
    print(f"   Occupancy shape: {outputs['occupancy'].shape}")
    print(f"   Attention shape: {outputs['attention'].shape}")
    print(f"   Slots shape: {outputs['slots'].shape}")

    # Test loss computation
    print("\n3. Testing loss computation...")
    criterion = BEACONLoss(n_tfs=10, seq_len=1000)

    # Create fake target profile
    target_profile = torch.rand(batch_size, seq_len, device=device)

    targets = {"profile": target_profile}
    losses = criterion(outputs, targets)

    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Profile loss: {losses['profile'].item():.4f}")
    print(f"   Diversity loss: {losses['diversity'].item():.4f}")

    # Test backward pass
    print("\n4. Testing backward pass...")
    losses["total"].backward()
    print("   Backward pass successful!")

    # Test binding site prediction
    print("\n5. Testing binding site prediction...")
    model.eval()
    sites = model.predict_binding_sites(x[:1], occupancy_threshold=0.3)
    print(f"   Predicted {len(sites[0])} binding sites in first sequence")

    if sites[0]:
        site = sites[0][0]
        print(f"   Example site: position={site['position']:.1f}, "
              f"TF={site['tf_id']}, occupancy={site['occupancy']:.3f}")

    # Test binding site counting
    print("\n6. Testing binding site counting...")
    counts = model.count_binding_sites(x, occupancy_threshold=0.3)
    print(f"   Counts per sequence: {counts.tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    return True


def test_data_loading():
    print("\nTesting data loading...")
    print("=" * 60)

    from beacon.data.dataset import BEACONDataset

    data_dir = Path(__file__).parent.parent / "data" / "processed" / "chipseq"
    train_path = data_dir / "train.h5"

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        return False

    print(f"Loading dataset from {train_path}")
    dataset = BEACONDataset(train_path, seq_length=1000)

    print(f"   Total samples: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"   Sequence shape: {sample['sequence'].shape}")
    print(f"   Profile shape: {sample['profile'].shape}")

    # Check data looks reasonable
    seq = sample['sequence'].numpy()
    print(f"   Sequence sum per position (should be ~1): {seq.sum(axis=1).mean():.3f}")

    profile = sample['profile'].numpy()
    print(f"   Profile range: [{profile.min():.3f}, {profile.max():.3f}]")

    print("\nData loading tests passed!")
    return True


if __name__ == "__main__":
    success = test_model()
    success = success and test_data_loading()
    sys.exit(0 if success else 1)
