#!/usr/bin/env python3
"""
Consolidated BEACON training with all improvements.

Supports:
  - PCGrad gradient surgery
  - GradNorm dynamic loss balancing
  - Enhanced slot utilization losses (contrastive, count, load balancing)
  - SlotDropout regularization
  - Per-TF difficulty weighting
  - Multi-dataset training (K562/HepG2/WTC11 x 7TF/20TF)
  - Ablation study configurations
  - Attribution head staged training

Usage:
    # Full training with all improvements on K562-7tf
    python scripts/train_improved.py --dataset K562-7tf --config full

    # Ablation: baseline without improvements
    python scripts/train_improved.py --dataset K562-7tf --config baseline

    # Ablation: only slot losses
    python scripts/train_improved.py --dataset K562-7tf --config slot_losses

    # Train on all 6 datasets
    python scripts/train_improved.py --all --config full

    # Phase 1 extension training (attribution + motif heads)
    python scripts/train_improved.py --dataset K562-7tf --config phase1 --base-model path/to/model.pt
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.backends.cudnn.enabled = False  # Workaround for cuDNN init errors on this system
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from beacon.models import BEACON, BEACONLoss
from beacon.models.heads import SlotDropout
from beacon.training import BEACONTrainer, PCGradOptimizer, GradNorm
from beacon.data.dataset import BEACONDataset


BASE_DIR = Path(__file__).parent.parent

DATASET_CONFIGS = {
    "K562-7tf": {
        "data_dir": "data/processed/multi_tf_k562",
        "n_tfs": 7,
        "importance_path": "data/processed/multi_tf_k562/gradient_importance.h5",
    },
    "K562-fulltf": {
        "data_dir": "data/processed/multi_tf_k562_fulltf",
        "n_tfs": 14,  # CTCF,GATA1,TAL1,MYC,MAX,CEBPB,REST,YY1,NRF1,JUND,FOS,ATF3,ELF1,GABPA
        "importance_path": "data/processed/multi_tf_k562_fulltf/gradient_importance.h5",
    },
    "HepG2-7tf": {
        "data_dir": "data/processed/multi_tf_hepg2_7tf",
        "n_tfs": 7,
        "importance_path": "data/processed/multi_tf_hepg2_7tf/gradient_importance.h5",
    },
    "HepG2-fulltf": {
        "data_dir": "data/processed/multi_tf_hepg2_fulltf",
        "n_tfs": 12,  # CTCF,MYC,MAX,CEBPB,REST,YY1,NRF1,ELF1,FOXA2,HNF4A,MAFK,NFE2L2
        "importance_path": "data/processed/multi_tf_hepg2_fulltf/gradient_importance.h5",
    },
}

# Ablation study configurations
TRAINING_CONFIGS = {
    "baseline": {
        "description": "Baseline: Hungarian matching + tf_weight=5.0, no extras",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.0,
        "load_balancing_weight": 0.0,
        "contrastive_weight": 0.0,
        "tf_contrastive_weight": 0.0,
        "tf_difficulty_weight": 0.0,
        "use_pcgrad": False,
        "use_gradnorm": False,
        "use_slot_dropout": False,
    },
    "slot_losses": {
        "description": "+SlotLosses: contrastive + count + load balancing",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.5,
        "load_balancing_weight": 0.3,
        "contrastive_weight": 0.5,
        "tf_contrastive_weight": 0.0,
        "tf_difficulty_weight": 0.0,
        "use_pcgrad": False,
        "use_gradnorm": False,
        "use_slot_dropout": False,
    },
    "slot_dropout": {
        "description": "+SlotDropout: adds slot dropout regularization",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.5,
        "load_balancing_weight": 0.3,
        "contrastive_weight": 0.5,
        "tf_contrastive_weight": 0.0,
        "tf_difficulty_weight": 0.0,
        "use_pcgrad": False,
        "use_gradnorm": False,
        "use_slot_dropout": True,
    },
    "pcgrad": {
        "description": "+PCGrad: gradient surgery for multi-task",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.5,
        "load_balancing_weight": 0.3,
        "contrastive_weight": 0.5,
        "tf_contrastive_weight": 0.0,
        "tf_difficulty_weight": 0.0,
        "use_pcgrad": True,
        "use_gradnorm": False,
        "use_slot_dropout": True,
    },
    "gradnorm": {
        "description": "+GradNorm: dynamic loss balancing",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.5,
        "load_balancing_weight": 0.3,
        "contrastive_weight": 0.5,
        "tf_contrastive_weight": 0.0,
        "tf_difficulty_weight": 0.0,
        "use_pcgrad": False,
        "use_gradnorm": True,
        "use_slot_dropout": True,
    },
    "full": {
        "description": "Full: all improvements enabled",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.5,
        "load_balancing_weight": 0.3,
        "contrastive_weight": 0.5,
        "tf_contrastive_weight": 0.3,
        "tf_difficulty_weight": 0.5,
        "use_pcgrad": True,
        "use_gradnorm": False,  # PCGrad and GradNorm are alternatives
        "use_slot_dropout": True,
    },
    "phase1": {
        "description": "Phase 1: attribution + motif embedding heads",
        "use_hungarian": True,
        "tf_weight": 5.0,
        "slot_count_weight": 0.5,
        "load_balancing_weight": 0.3,
        "contrastive_weight": 0.5,
        "tf_contrastive_weight": 0.3,
        "tf_difficulty_weight": 0.5,
        "use_pcgrad": True,
        "use_gradnorm": False,
        "use_slot_dropout": True,
        "use_attribution_head": True,
        "use_motif_embedding_head": True,
        "importance_weight": 1.0,
        "anchor_weight": 0.5,
        "multiscale_importance_weight": 0.5,
        "prototype_diversity_weight": 0.3,
        "staged_unfreezing": True,
        "freeze_epochs": 30,
    },
}


def create_model(n_tfs, config):
    """Create BEACON model with config-appropriate heads."""
    model_kwargs = dict(
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
        dropout=0.1,
        attention_mode="independent",
    )

    if config.get("use_attribution_head"):
        model_kwargs["use_attribution_head"] = True
    if config.get("use_motif_embedding_head"):
        model_kwargs["use_motif_embedding_head"] = True
        model_kwargs["motif_dim"] = 64
        model_kwargs["n_prototypes"] = 64

    return BEACON(**model_kwargs)


def create_loss_fn(n_tfs, config):
    """Create BEACONLoss with config weights."""
    return BEACONLoss(
        n_tfs=n_tfs,
        seq_len=2000,
        position_mode="gaussian",
        profile_weight=1.0,
        position_weight=1.0,
        tf_weight=config.get("tf_weight", 5.0),
        occupancy_weight=1.0,
        diversity_weight=0.1,
        orthogonality_weight=0.1,
        site_supervision_weight=0.5,
        use_hungarian=config.get("use_hungarian", True),
        slot_count_weight=config.get("slot_count_weight", 0.0),
        load_balancing_weight=config.get("load_balancing_weight", 0.0),
        contrastive_weight=config.get("contrastive_weight", 0.0),
        tf_contrastive_weight=config.get("tf_contrastive_weight", 0.0),
        tf_difficulty_weight=config.get("tf_difficulty_weight", 0.0),
        importance_weight=config.get("importance_weight", 0.0),
        anchor_weight=config.get("anchor_weight", 0.0),
        multiscale_importance_weight=config.get("multiscale_importance_weight", 0.0),
        prototype_diversity_weight=config.get("prototype_diversity_weight", 0.0),
    )


def train_dataset(
    dataset_name: str,
    train_config_name: str,
    device: str = "cuda",
    num_epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    patience: int = 25,
    base_model_path: str = None,
    output_root: str = None,
):
    """Train BEACON on a single dataset with specified config."""
    ds_config = DATASET_CONFIGS[dataset_name]
    train_config = TRAINING_CONFIGS[train_config_name]

    data_dir = BASE_DIR / ds_config["data_dir"]
    n_tfs = ds_config["n_tfs"]

    if output_root is None:
        output_root = BASE_DIR / "outputs" / "improved"
    else:
        output_root = Path(output_root)

    experiment_name = f"{dataset_name}_{train_config_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / experiment_name

    print(f"\n{'='*70}")
    print(f"Training: {experiment_name}")
    print(f"Config: {train_config['description']}")
    print(f"Dataset: {dataset_name} (n_tfs={n_tfs})")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Check data exists
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"
    test_path = data_dir / "test.h5"

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        return None

    # Load importance data if needed
    use_importance = train_config.get("importance_weight", 0) > 0
    importance_path = None
    if use_importance:
        imp_path = BASE_DIR / ds_config.get("importance_path", "")
        if imp_path.exists():
            importance_path = str(imp_path)
            print(f"Using gradient importance: {importance_path}")

    # Create datasets
    train_dataset = BEACONDataset(
        str(train_path), seq_length=2000, augment=True,
        n_slots=16, extract_peaks=False,
        importance_path=importance_path,
    )
    val_dataset = BEACONDataset(
        str(val_path), seq_length=2000, augment=False,
        n_slots=16, extract_peaks=False,
    ) if val_path.exists() else None

    test_dataset = BEACONDataset(
        str(test_path), seq_length=2000, augment=False,
        n_slots=16, extract_peaks=False,
    ) if test_path.exists() else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    ) if test_dataset else None

    print(f"Train: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test: {len(test_dataset)} samples")

    # Create model
    model = create_model(n_tfs, train_config)

    # Load base model if provided (for phase1 or fine-tuning)
    if base_model_path:
        print(f"Loading base model: {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Staged unfreezing for phase1
    if train_config.get("staged_unfreezing") and base_model_path:
        freeze_epochs = train_config.get("freeze_epochs", 30)
        print(f"Staged unfreezing: backbone frozen for first {freeze_epochs} epochs")
        for name, param in model.named_parameters():
            if "backbone" in name or "slot_attention" in name:
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (frozen backbone): {n_trainable:,}")

    # Create loss function
    loss_fn = create_loss_fn(n_tfs, train_config)

    # Create trainer
    trainer = BEACONTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        output_dir=output_dir,
        experiment_name=f"beacon_{timestamp}",
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        grad_clip=1.0,
        use_amp=True,
        log_interval=10,
        val_interval=1,
        save_interval=5,
        early_stopping_patience=patience,
        num_slots=16,
        num_tfs=n_tfs,
        use_pcgrad=train_config.get("use_pcgrad", False),
        use_gradnorm=train_config.get("use_gradnorm", False),
        track_slot_utilization=True,
    )

    # Save config
    config_path = output_dir / f"beacon_{timestamp}" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "train_config": train_config_name,
            "train_config_details": train_config,
            "n_tfs": n_tfs,
            "n_params": n_params,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }, f, indent=2)

    # Warmup cuDNN with a small forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 2000, 4, device=device)
        _ = model(dummy)
    model.train()

    # Train
    results = trainer.train()

    # Handle staged unfreezing
    if train_config.get("staged_unfreezing") and base_model_path:
        print(f"\n--- Unfreezing backbone for continued training ---")
        for param in model.parameters():
            param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (unfrozen): {n_trainable:,}")

        # Continue training with lower LR
        trainer2 = BEACONTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            output_dir=output_dir,
            experiment_name=f"beacon_{timestamp}_unfrozen",
            device=device,
            num_epochs=50,
            learning_rate=learning_rate * 0.1,
            weight_decay=0.01,
            grad_clip=1.0,
            use_amp=True,
            early_stopping_patience=patience,
            num_slots=16,
            num_tfs=n_tfs,
            use_pcgrad=train_config.get("use_pcgrad", False),
            use_gradnorm=train_config.get("use_gradnorm", False),
        )
        results2 = trainer2.train()
        results["unfrozen_results"] = results2

    print(f"\nTraining complete: {experiment_name}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Total epochs: {results['total_epochs']}")

    return results


def run_ablation(device="cuda", output_root=None, skip_configs=None, base_model_path=None):
    """Run full ablation study on K562-7tf."""
    ablation_configs = [
        "baseline", "slot_losses", "slot_dropout",
        "pcgrad", "gradnorm", "full",
    ]

    results = {}
    for config_name in ablation_configs:
        if skip_configs and config_name in skip_configs:
            print(f"\nSkipping already-completed config: {config_name}")
            continue
        print(f"\n{'#'*70}")
        print(f"# ABLATION: {config_name}")
        print(f"{'#'*70}")

        result = train_dataset(
            "K562-7tf", config_name,
            device=device, output_root=output_root,
        )
        results[config_name] = result

    # Print ablation summary
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Best Val Loss':<15} {'Epochs':<10}")
    print(f"{'-'*20} {'-'*15} {'-'*10}")

    for name, result in results.items():
        if result:
            print(f"{name:<20} {result['best_val_loss']:<15.4f} {result['total_epochs']:<10}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train BEACON with improvements"
    )
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--config", type=str, default="full",
                       choices=list(TRAINING_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true", help="Train on all 6 datasets")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study on K562-7tf")
    parser.add_argument("--base-model", type=str, help="Base model for fine-tuning")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, help="Specific GPU ID")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--skip-configs", type=str, help="Comma-separated configs to skip in ablation")
    args = parser.parse_args()

    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    if args.ablation:
        skip = args.skip_configs.split(",") if args.skip_configs else None
        run_ablation(device=args.device, output_root=args.output_dir, skip_configs=skip)
    elif args.all:
        for ds_name in DATASET_CONFIGS:
            train_dataset(
                ds_name, args.config,
                device=args.device,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                patience=args.patience,
                base_model_path=args.base_model,
                output_root=args.output_dir,
            )
    elif args.dataset:
        train_dataset(
            args.dataset, args.config,
            device=args.device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            base_model_path=args.base_model,
            output_root=args.output_dir,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
