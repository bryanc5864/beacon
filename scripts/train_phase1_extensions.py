#!/usr/bin/env python3
"""
B4.3 Training Script for Phase 1 Extensions.

Trains the Phase 1 extension heads (AttributionHead and MotifEmbeddingHead) in
two stages on top of the Phase 10 checkpoint.

Stage 2: Attribution Head (~20 epochs)
  - Load Phase 10 checkpoint, add AttributionHead
  - Freeze backbone + slot_attention for first 10 epochs
  - Custom training loop with BEACONLoss + AttributionSupervisionLoss
  - LR: 3e-5, unfreeze backbone at epoch 10 with 0.1x LR multiplier

Stage 3: Motif Embedding Head (~20 epochs)
  - Load Stage 2 checkpoint, add MotifEmbeddingHead
  - Initialize anchors from JASPAR via simple PWM encoding
  - Loss: BEACONLoss + AnchorLoss + PrototypeDiversityLoss
  - LR: 1e-5
"""

import sys
import json
import argparse
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.models.losses import (
    BEACONLoss,
    AttributionSupervisionLoss,
    AnchorLoss,
    PrototypeDiversityLoss,
)
from beacon.data.dataset import BEACONDataset

base_dir = Path(__file__).parent.parent

# TF names used throughout the project
TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]

# JASPAR matrix IDs for our TFs
JASPAR_IDS = {
    "CTCF": "MA0139",
    "GATA1": "MA0035",
    "TAL1": "MA0140",
    "MYC": "MA0147",
    "MAX": "MA0058",
    "SPI1": "MA0080",
    "CEBPB": "MA0466",
}


# ---------------------------------------------------------------------------
# Dataset wrapper for gradient importance supervision
# ---------------------------------------------------------------------------

class GradientImportanceDataset(Dataset):
    """
    Wraps a BEACONDataset and adds precomputed gradient importance maps
    loaded from an HDF5 file.
    """

    def __init__(self, base_dataset, gradient_h5_path):
        self.base = base_dataset
        self.gradient_h5 = h5py.File(gradient_h5_path, "r")
        self.importance = self.gradient_h5["importance"]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # Handle augmentation indexing: if base dataset doubles via augment,
        # real_idx is the underlying sample index
        real_idx = (
            idx % (len(self.base) // 2)
            if hasattr(self.base, "augment") and self.base.augment
            else idx
        )
        item["gradient_importance"] = torch.from_numpy(
            self.importance[real_idx].astype(np.float32)
        )
        return item

    def __del__(self):
        try:
            self.gradient_h5.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# JASPAR MEME format loader
# ---------------------------------------------------------------------------

def load_jaspar_meme(meme_path):
    """
    Load motif PWMs from a JASPAR MEME-format file.

    Returns:
        dict mapping motif name -> numpy array [L, 4] (probability matrix)
    """
    motifs = {}
    current_name = None
    current_matrix = []
    in_matrix = False

    with open(meme_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                # Save previous motif
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

    # Save last motif
    if current_name and current_matrix:
        motifs[current_name] = np.array(current_matrix)

    return motifs


def get_tf_pwm_matrices(meme_path, n_tfs=7, embed_dim=64):
    """
    Load JASPAR PWMs for our TFs and encode them as fixed-length vectors.

    Each PWM is flattened (padded/cropped to a fixed length) then projected
    to embed_dim via a simple linear transform.

    Returns:
        torch.Tensor [n_tfs, embed_dim] -- initial anchor embeddings
    """
    motifs = load_jaspar_meme(meme_path)
    max_pwm_len = 30  # pad/crop all PWMs to this length for uniformity
    flat_dim = max_pwm_len * 4

    pwm_vectors = []
    for tf_name in TF_NAMES[:n_tfs]:
        jaspar_id = JASPAR_IDS.get(tf_name, "")
        # Find the best matching motif
        pwm = None
        for motif_name, matrix in motifs.items():
            if jaspar_id and jaspar_id in motif_name:
                pwm = matrix
                break
            if tf_name.upper() in motif_name.upper():
                pwm = matrix
                break

        if pwm is None:
            # Fallback: use a random unit vector
            vec = np.random.randn(flat_dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            pwm_vectors.append(vec)
            continue

        # Pad or crop to max_pwm_len
        if len(pwm) > max_pwm_len:
            pwm = pwm[:max_pwm_len]
        elif len(pwm) < max_pwm_len:
            padding = np.zeros((max_pwm_len - len(pwm), 4), dtype=np.float32)
            pwm = np.vstack([pwm, padding])

        flat = pwm.flatten().astype(np.float32)
        flat = flat / (np.linalg.norm(flat) + 1e-8)
        pwm_vectors.append(flat)

    pwm_matrix = np.stack(pwm_vectors)  # [n_tfs, flat_dim]
    pwm_tensor = torch.from_numpy(pwm_matrix)

    # Project to embed_dim via a random orthogonal projection (fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    proj = rng.randn(flat_dim, embed_dim).astype(np.float32)
    # Orthogonalize via QR
    q, _ = np.linalg.qr(proj)
    proj_tensor = torch.from_numpy(q[:, :embed_dim].astype(np.float32))

    anchor_embeddings = pwm_tensor @ proj_tensor  # [n_tfs, embed_dim]
    # Normalize
    anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
    return anchor_embeddings


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def to_device(batch, device):
    """Move all tensors in a batch dict to the specified device."""
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def build_targets(batch):
    """
    Extract a targets dict from a batch dict for BEACONLoss.

    BEACONDataset returns: sequence [L,4], profile [L], binding_sites [K,3],
    tf_index. BEACONLoss expects targets with keys: profile, binding_sites,
    positions, tf_ids, occupancy.
    """
    targets = {}
    if "profile" in batch:
        targets["profile"] = batch["profile"]
    if "binding_sites" in batch:
        bs = batch["binding_sites"]
        targets["binding_sites"] = bs
        targets["positions"] = bs[:, :, 0]
        targets["tf_ids"] = bs[:, :, 1].long()
        targets["occupancy"] = bs[:, :, 2]
    if "tf_index" in batch:
        targets["tf_index"] = batch["tf_index"]
    return targets


def count_parameters(model, trainable_only=True):
    """Count total (or trainable) parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_backbone_and_slots(model):
    """Freeze backbone and slot_attention parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name or "slot_attention" in name or "positional_encoding" in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Stage 2: Attribution Head Training
# ---------------------------------------------------------------------------

def train_stage2(args):
    """
    Train the AttributionHead on top of the Phase 10 checkpoint.

    - First 10 epochs: backbone + slot_attention frozen, only new head trains
    - Epoch 10+: unfreeze backbone with 0.1x LR multiplier
    - Loss: BEACONLoss + attribution_supervision (weight=0.5)
    """
    print("=" * 70)
    print("STAGE 2: Attribution Head Training")
    print("=" * 70)

    device = args.device
    n_tfs = 7
    seq_len = 2000

    # Resolve paths
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = base_dir / checkpoint_path

    data_dir = args.data_dir
    if not Path(data_dir).is_absolute():
        data_dir = base_dir / data_dir

    gradient_h5_path = args.gradient_h5
    if not Path(gradient_h5_path).is_absolute():
        gradient_h5_path = base_dir / gradient_h5_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    stage2_dir = output_dir / "stage2_attribution"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Verify files exist
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None
    if not (Path(data_dir) / "train.h5").exists():
        print(f"ERROR: Training data not found: {data_dir}/train.h5")
        return None
    if not Path(gradient_h5_path).exists():
        print(f"ERROR: Gradient importance file not found: {gradient_h5_path}")
        return None

    # ---- Model ----
    print("\nBuilding model with AttributionHead...")
    model = BEACON(
        seq_len=seq_len,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=16,
        slot_dim=128,
        n_iterations=3,
        n_tfs=n_tfs,
        position_mode="gaussian",
        dropout=0.1,
        attention_mode="independent",
        use_attribution_head=True,
        use_motif_embedding=False,
    )

    # Load Phase 10 checkpoint (missing attribution head params is OK)
    print(f"Loading Phase 10 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys (expected for new head): {len(missing)}")
    if missing:
        for k in missing:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected:
            print(f"    {k}")

    model = model.to(device)

    total_params = count_parameters(model, trainable_only=False)
    print(f"Total parameters: {total_params:,}")

    # Freeze backbone and slot attention initially
    freeze_backbone_and_slots(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Trainable parameters (frozen backbone): {trainable_params:,}")

    # ---- Data ----
    print("\nLoading datasets...")
    train_base = BEACONDataset(
        str(Path(data_dir) / "train.h5"),
        seq_length=seq_len,
        augment=True,
        n_slots=16,
        extract_peaks=True,
    )
    train_dataset = GradientImportanceDataset(train_base, str(gradient_h5_path))

    val_base = BEACONDataset(
        str(Path(data_dir) / "val.h5"),
        seq_length=seq_len,
        augment=False,
        n_slots=16,
        extract_peaks=True,
    )
    # Validation also needs gradient importance for loss computation
    val_dataset = GradientImportanceDataset(val_base, str(gradient_h5_path))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    # ---- Loss functions ----
    beacon_loss_fn = BEACONLoss(
        n_tfs=n_tfs,
        seq_len=seq_len,
        profile_weight=1.0,
        position_weight=0.5,
        tf_weight=5.0,
        occupancy_weight=0.5,
        diversity_weight=0.5,
        orthogonality_weight=0.5,
        site_supervision_weight=1.0,
        use_hungarian=True,
        slot_count_weight=0.5,
        load_balancing_weight=0.3,
    )
    attribution_loss_fn = AttributionSupervisionLoss()
    attribution_weight = 0.5

    # ---- Optimizer with parameter groups ----
    lr = args.lr if args.lr is not None else 3e-5
    num_epochs = args.epochs if args.epochs is not None else 20
    unfreeze_epoch = num_epochs // 2  # epoch 10 for 20-epoch run

    # Initially only new head parameters are trainable
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Mixed precision
    use_amp = device != "cpu" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Training loop ----
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 10
    best_model_path = stage2_dir / "best_model.pt"

    print(f"\nTraining config:")
    print(f"  LR: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Unfreeze backbone at epoch: {unfreeze_epoch}")
    print(f"  Attribution loss weight: {attribution_weight}")
    print(f"  AMP: {use_amp}")
    print(f"  Patience: {patience}")
    print()

    training_log = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ---- Unfreeze backbone at the midpoint ----
        if epoch == unfreeze_epoch + 1:
            print(f"\n>>> Epoch {epoch}: Unfreezing backbone with 0.1x LR multiplier")
            unfreeze_all(model)

            # Rebuild optimizer with parameter groups
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "backbone" in name or "slot_attention" in name or "positional_encoding" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": lr * 0.1},
                    {"params": head_params, "lr": lr},
                ],
                weight_decay=0.01,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - epoch + 1,
                eta_min=lr * 0.001,
            )
            trainable_params = count_parameters(model, trainable_only=True)
            print(f"  Trainable parameters (unfrozen): {trainable_params:,}")

        # ---- Train ----
        model.train()
        epoch_losses = defaultdict(list)

        for batch_idx, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch["sequence"])
                    targets = build_targets(batch)
                    base_losses = beacon_loss_fn(outputs, targets)

                    # Attribution supervision
                    attr_loss = attribution_loss_fn(
                        outputs["attribution"],
                        batch["gradient_importance"],
                        outputs["occupancy"],
                    )

                    total_loss = base_losses["total"] + attribution_weight * attr_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch["sequence"])
                targets = build_targets(batch)
                base_losses = beacon_loss_fn(outputs, targets)

                attr_loss = attribution_loss_fn(
                    outputs["attribution"],
                    batch["gradient_importance"],
                    outputs["occupancy"],
                )

                total_loss = base_losses["total"] + attribution_weight * attr_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Log losses
            epoch_losses["total"].append(total_loss.item())
            epoch_losses["base"].append(base_losses["total"].item())
            epoch_losses["attribution"].append(attr_loss.item())
            epoch_losses["profile"].append(base_losses["profile"].item())
            epoch_losses["tf_identity"].append(base_losses["tf_identity"].item())
            epoch_losses["occupancy"].append(base_losses["occupancy"].item())

            if batch_idx % 20 == 0:
                print(
                    f"  Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Base: {base_losses['total'].item():.4f} | "
                    f"Attr: {attr_loss.item():.4f}"
                )

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_losses = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch["sequence"])
                        targets = build_targets(batch)
                        base_losses = beacon_loss_fn(outputs, targets)

                        attr_loss = attribution_loss_fn(
                            outputs["attribution"],
                            batch["gradient_importance"],
                            outputs["occupancy"],
                        )

                        total_loss = base_losses["total"] + attribution_weight * attr_loss
                else:
                    outputs = model(batch["sequence"])
                    targets = build_targets(batch)
                    base_losses = beacon_loss_fn(outputs, targets)

                    attr_loss = attribution_loss_fn(
                        outputs["attribution"],
                        batch["gradient_importance"],
                        outputs["occupancy"],
                    )

                    total_loss = base_losses["total"] + attribution_weight * attr_loss

                val_losses["total"].append(total_loss.item())
                val_losses["base"].append(base_losses["total"].item())
                val_losses["attribution"].append(attr_loss.item())

        # Compute averages
        avg_train = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        avg_val = {k: sum(v) / len(v) for k, v in val_losses.items()}

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {avg_train['total']:.4f} (base={avg_train['base']:.4f}, attr={avg_train['attribution']:.4f}) | "
            f"Val Loss: {avg_val['total']:.4f} (base={avg_val['base']:.4f}, attr={avg_val['attribution']:.4f}) | "
            f"LR: {current_lr:.2e}"
        )

        training_log.append(
            {
                "epoch": epoch,
                "train_loss": avg_train["total"],
                "train_base": avg_train["base"],
                "train_attr": avg_train["attribution"],
                "val_loss": avg_val["total"],
                "val_base": avg_val["base"],
                "val_attr": avg_val["attribution"],
                "lr": current_lr,
                "time": epoch_time,
            }
        )

        # ---- Checkpointing ----
        if avg_val["total"] < best_val_loss:
            best_val_loss = avg_val["total"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        "stage": 2,
                        "use_attribution_head": True,
                        "use_motif_embedding": False,
                    },
                },
                best_model_path,
            )
            print(f"  >> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch} epochs ({patience} without improvement)")
            break

    # Save training log
    with open(stage2_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nStage 2 complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return str(best_model_path)


# ---------------------------------------------------------------------------
# Stage 3: Motif Embedding Head Training
# ---------------------------------------------------------------------------

def train_stage3(args, stage2_checkpoint=None):
    """
    Train the MotifEmbeddingHead on top of the Stage 2 checkpoint.

    - Initialize anchor prototypes from JASPAR PWMs
    - Loss: BEACONLoss + anchor_loss (weight=1.0) + diversity_loss (weight=0.5)
    - TF identity weight reduced to 0.5 (motif head takes over TF classification)
    - LR: 1e-5
    """
    print("\n" + "=" * 70)
    print("STAGE 3: Motif Embedding Head Training")
    print("=" * 70)

    device = args.device
    n_tfs = 7
    seq_len = 2000
    motif_embed_dim = 64

    # Resolve paths
    if stage2_checkpoint is not None:
        checkpoint_path = stage2_checkpoint
    else:
        checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = base_dir / checkpoint_path

    data_dir = args.data_dir
    if not Path(data_dir).is_absolute():
        data_dir = base_dir / data_dir

    jaspar_path = args.jaspar
    if not Path(jaspar_path).is_absolute():
        jaspar_path = base_dir / jaspar_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    stage3_dir = output_dir / "stage3_motif_embedding"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # Verify files
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None
    if not (Path(data_dir) / "train.h5").exists():
        print(f"ERROR: Training data not found: {data_dir}/train.h5")
        return None
    if not Path(jaspar_path).exists():
        print(f"WARNING: JASPAR file not found: {jaspar_path}")
        print("  Anchors will be initialized randomly.")
        jaspar_available = False
    else:
        jaspar_available = True

    # ---- Model ----
    print("\nBuilding model with AttributionHead + MotifEmbeddingHead...")
    model = BEACON(
        seq_len=seq_len,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=16,
        slot_dim=128,
        n_iterations=3,
        n_tfs=n_tfs,
        position_mode="gaussian",
        dropout=0.1,
        attention_mode="independent",
        use_attribution_head=True,
        use_motif_embedding=True,
        motif_embed_dim=motif_embed_dim,
    )

    # Load Stage 2 checkpoint (missing motif head params is OK)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys (expected for new head): {len(missing)}")
    if missing:
        for k in missing:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected:
            print(f"    {k}")

    # Initialize anchor prototypes from JASPAR
    if jaspar_available:
        print(f"\nInitializing anchor prototypes from JASPAR: {jaspar_path}")
        anchor_embeddings = get_tf_pwm_matrices(
            str(jaspar_path), n_tfs=n_tfs, embed_dim=motif_embed_dim
        )
        with torch.no_grad():
            model.motif_head.anchor_prototypes.copy_(anchor_embeddings)
        print(f"  Initialized {n_tfs} anchor prototypes (dim={motif_embed_dim})")
    else:
        print("  Using random anchor initialization.")

    model = model.to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---- Data ----
    print("\nLoading datasets...")
    train_dataset = BEACONDataset(
        str(Path(data_dir) / "train.h5"),
        seq_length=seq_len,
        augment=True,
        n_slots=16,
        extract_peaks=True,
    )
    val_dataset = BEACONDataset(
        str(Path(data_dir) / "val.h5"),
        seq_length=seq_len,
        augment=False,
        n_slots=16,
        extract_peaks=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    # ---- Loss functions ----
    # Reduce tf_identity weight to 0.5 since motif head takes over TF classification
    beacon_loss_fn = BEACONLoss(
        n_tfs=n_tfs,
        seq_len=seq_len,
        profile_weight=1.0,
        position_weight=0.5,
        tf_weight=0.5,  # Reduced: motif head takes over
        occupancy_weight=0.5,
        diversity_weight=0.5,
        orthogonality_weight=0.5,
        site_supervision_weight=1.0,
        use_hungarian=True,
        slot_count_weight=0.5,
        load_balancing_weight=0.3,
    )
    anchor_loss_fn = AnchorLoss(margin=0.5)
    diversity_loss_fn = PrototypeDiversityLoss()
    anchor_weight = 1.0
    diversity_weight = 0.5

    # ---- Optimizer ----
    lr = args.lr if args.lr is not None else 1e-5
    num_epochs = args.epochs if args.epochs is not None else 20

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Mixed precision
    use_amp = device != "cpu" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Training loop ----
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 10
    best_model_path = stage3_dir / "best_model.pt"

    print(f"\nTraining config:")
    print(f"  LR: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Anchor loss weight: {anchor_weight}")
    print(f"  Diversity loss weight: {diversity_weight}")
    print(f"  TF identity weight (reduced): 0.5")
    print(f"  AMP: {use_amp}")
    print(f"  Patience: {patience}")
    print()

    training_log = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        epoch_losses = defaultdict(list)

        for batch_idx, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch["sequence"])
                    targets = build_targets(batch)
                    base_losses = beacon_loss_fn(outputs, targets)

                    # Build target TFs and occupancy mask for anchor loss
                    if "binding_sites" in batch:
                        bs = batch["binding_sites"]
                        target_tfs = bs[:, :, 1].long()
                        occ_mask = (bs[:, :, 2] > 0.1).float()
                    else:
                        target_tfs = torch.zeros(
                            outputs["motif_embeddings"].shape[:2],
                            dtype=torch.long,
                            device=device,
                        )
                        occ_mask = None

                    a_loss = anchor_loss_fn(
                        outputs["motif_embeddings"],
                        model.motif_head.anchor_prototypes,
                        target_tfs,
                        mask=occ_mask,
                    )
                    d_loss = diversity_loss_fn(model.motif_head.anchor_prototypes)

                    total_loss = (
                        base_losses["total"]
                        + anchor_weight * a_loss
                        + diversity_weight * d_loss
                    )

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch["sequence"])
                targets = build_targets(batch)
                base_losses = beacon_loss_fn(outputs, targets)

                if "binding_sites" in batch:
                    bs = batch["binding_sites"]
                    target_tfs = bs[:, :, 1].long()
                    occ_mask = (bs[:, :, 2] > 0.1).float()
                else:
                    target_tfs = torch.zeros(
                        outputs["motif_embeddings"].shape[:2],
                        dtype=torch.long,
                        device=device,
                    )
                    occ_mask = None

                a_loss = anchor_loss_fn(
                    outputs["motif_embeddings"],
                    model.motif_head.anchor_prototypes,
                    target_tfs,
                    mask=occ_mask,
                )
                d_loss = diversity_loss_fn(model.motif_head.anchor_prototypes)

                total_loss = (
                    base_losses["total"]
                    + anchor_weight * a_loss
                    + diversity_weight * d_loss
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Log losses
            epoch_losses["total"].append(total_loss.item())
            epoch_losses["base"].append(base_losses["total"].item())
            epoch_losses["anchor"].append(a_loss.item())
            epoch_losses["diversity"].append(d_loss.item())
            epoch_losses["profile"].append(base_losses["profile"].item())
            epoch_losses["tf_identity"].append(base_losses["tf_identity"].item())
            epoch_losses["occupancy"].append(base_losses["occupancy"].item())

            if batch_idx % 20 == 0:
                print(
                    f"  Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Base: {base_losses['total'].item():.4f} | "
                    f"Anchor: {a_loss.item():.4f} | "
                    f"Div: {d_loss.item():.4f}"
                )

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_losses = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch["sequence"])
                        targets = build_targets(batch)
                        base_losses = beacon_loss_fn(outputs, targets)

                        if "binding_sites" in batch:
                            bs = batch["binding_sites"]
                            target_tfs = bs[:, :, 1].long()
                            occ_mask = (bs[:, :, 2] > 0.1).float()
                        else:
                            target_tfs = torch.zeros(
                                outputs["motif_embeddings"].shape[:2],
                                dtype=torch.long,
                                device=device,
                            )
                            occ_mask = None

                        a_loss = anchor_loss_fn(
                            outputs["motif_embeddings"],
                            model.motif_head.anchor_prototypes,
                            target_tfs,
                            mask=occ_mask,
                        )
                        d_loss = diversity_loss_fn(model.motif_head.anchor_prototypes)

                        total_loss = (
                            base_losses["total"]
                            + anchor_weight * a_loss
                            + diversity_weight * d_loss
                        )
                else:
                    outputs = model(batch["sequence"])
                    targets = build_targets(batch)
                    base_losses = beacon_loss_fn(outputs, targets)

                    if "binding_sites" in batch:
                        bs = batch["binding_sites"]
                        target_tfs = bs[:, :, 1].long()
                        occ_mask = (bs[:, :, 2] > 0.1).float()
                    else:
                        target_tfs = torch.zeros(
                            outputs["motif_embeddings"].shape[:2],
                            dtype=torch.long,
                            device=device,
                        )
                        occ_mask = None

                    a_loss = anchor_loss_fn(
                        outputs["motif_embeddings"],
                        model.motif_head.anchor_prototypes,
                        target_tfs,
                        mask=occ_mask,
                    )
                    d_loss = diversity_loss_fn(model.motif_head.anchor_prototypes)

                    total_loss = (
                        base_losses["total"]
                        + anchor_weight * a_loss
                        + diversity_weight * d_loss
                    )

                val_losses["total"].append(total_loss.item())
                val_losses["base"].append(base_losses["total"].item())
                val_losses["anchor"].append(a_loss.item())
                val_losses["diversity"].append(d_loss.item())

        # Compute averages
        avg_train = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        avg_val = {k: sum(v) / len(v) for k, v in val_losses.items()}

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {avg_train['total']:.4f} "
            f"(base={avg_train['base']:.4f}, anchor={avg_train['anchor']:.4f}, div={avg_train['diversity']:.4f}) | "
            f"Val Loss: {avg_val['total']:.4f} "
            f"(base={avg_val['base']:.4f}, anchor={avg_val['anchor']:.4f}, div={avg_val['diversity']:.4f}) | "
            f"LR: {current_lr:.2e}"
        )

        training_log.append(
            {
                "epoch": epoch,
                "train_loss": avg_train["total"],
                "train_base": avg_train["base"],
                "train_anchor": avg_train["anchor"],
                "train_diversity": avg_train["diversity"],
                "val_loss": avg_val["total"],
                "val_base": avg_val["base"],
                "val_anchor": avg_val["anchor"],
                "val_diversity": avg_val["diversity"],
                "lr": current_lr,
                "time": epoch_time,
            }
        )

        # ---- Checkpointing ----
        if avg_val["total"] < best_val_loss:
            best_val_loss = avg_val["total"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "anchor_prototypes": model.motif_head.anchor_prototypes.data.cpu(),
                    "config": {
                        "stage": 3,
                        "use_attribution_head": True,
                        "use_motif_embedding": True,
                        "motif_embed_dim": motif_embed_dim,
                    },
                },
                best_model_path,
            )
            print(f"  >> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch} epochs ({patience} without improvement)")
            break

    # Save training log
    with open(stage3_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nStage 3 complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return str(best_model_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 1 Extension Heads (AttributionHead + MotifEmbeddingHead)"
    )

    parser.add_argument(
        "--stage",
        choices=["2", "3", "both"],
        default="both",
        help="Which stage to train: 2 (attribution), 3 (motif), or both (default: both)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/phase10_slot_ablation/slots_16/beacon_20260203_231900/best_model.pt",
        help="Path to Phase 10 checkpoint (or Stage 2 checkpoint for stage 3)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/multi_tf_k562",
        help="Directory with train.h5, val.h5, test.h5",
    )
    parser.add_argument(
        "--gradient-h5",
        type=str,
        default="data/processed/multi_tf_k562/gradient_importance.h5",
        help="Path to precomputed gradient importance HDF5",
    )
    parser.add_argument(
        "--jaspar",
        type=str,
        default="data/raw/jaspar/meme_vertebrates.txt",
        help="Path to JASPAR MEME-format file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase1_extensions",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs per stage (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 3e-5 for stage 2, 1e-5 for stage 3)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1 Extensions Training")
    print(f"  Stage: {args.stage}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 70)

    start_time = time.time()

    if args.stage == "2":
        # Override LR default for stage 2
        if args.lr is None:
            args.lr = 3e-5
        if args.epochs is None:
            args.epochs = 20
        train_stage2(args)

    elif args.stage == "3":
        # Override LR default for stage 3
        if args.lr is None:
            args.lr = 1e-5
        if args.epochs is None:
            args.epochs = 20
        train_stage3(args)

    elif args.stage == "both":
        # Stage 2 with its defaults
        stage2_args = argparse.Namespace(**vars(args))
        stage2_args.lr = args.lr if args.lr is not None else 3e-5
        stage2_args.epochs = args.epochs if args.epochs is not None else 20

        stage2_checkpoint = train_stage2(stage2_args)

        if stage2_checkpoint is None:
            print("\nStage 2 failed. Aborting.")
            return 1

        # Stage 3 uses Stage 2 checkpoint, with its own defaults
        stage3_args = argparse.Namespace(**vars(args))
        stage3_args.lr = args.lr if args.lr is not None else 1e-5
        stage3_args.epochs = args.epochs if args.epochs is not None else 20

        train_stage3(stage3_args, stage2_checkpoint=stage2_checkpoint)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 3600:.2f} hours")

    return 0


if __name__ == "__main__":
    sys.exit(main())
