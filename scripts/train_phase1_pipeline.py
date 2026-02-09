#!/usr/bin/env python3
"""
Phase 1: Replace DeepSHAP + TF-MoDISco Pipeline

Trains BEACON-v3 with:
  - Per-base Attribution Head (replaces DeepSHAP)
  - Open-Vocabulary Motif Embedding Head (replaces fixed TF classifier)
  - Per-slot motif extraction (replaces TF-MoDISco clustering)

Loss weights (from plan):
  - Profile reconstruction:   1.0
  - TF classification:        1.0  (via Hungarian matching)
  - Anchor loss:              0.5  (ground known TFs)
  - Importance supervision:   0.3  (attribution head quality)
  - Prototype diversity:      0.2  (spread prototypes for novel motifs)
  - Contrastive loss:         0.1  (cluster same-TF slots)

Fine-tunes from best Phase 10 checkpoint (16 slots) to retain
existing profile/TF performance while adding new heads.

Usage:
  python scripts/train_phase1_pipeline.py --gpus 0,9
  python scripts/train_phase1_pipeline.py --gpus 0,9 --from-scratch
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.models.losses import BEACONLoss
from beacon.training.trainer import BEACONTrainer
from beacon.data.dataset import BEACONDataset


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train BEACON-v3 with Attribution + Motif Embedding heads"
    )

    # Data
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/multi_tf_k562"))
    parser.add_argument("--importance-path", type=Path,
                        default=Path("data/processed/multi_tf_k562/gradient_importance.h5"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/phase1_pipeline"))
    parser.add_argument("--experiment-name", type=str, default=None)

    # Pretrained checkpoint to fine-tune from
    parser.add_argument("--pretrained-path", type=Path,
                        default=Path("outputs/phase10_slot_ablation/slots_16/beacon_20260203_231900/best_model.pt"),
                        help="Pretrained BEACON checkpoint to initialize from")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from scratch instead of fine-tuning")

    # Architecture (must match pretrained model)
    parser.add_argument("--seq-len", type=int, default=2000)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--backbone-dim", type=int, default=128)
    parser.add_argument("--backbone-layers", type=int, default=4)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--n-iterations", type=int, default=3)

    # Phase 1 specific
    parser.add_argument("--motif-dim", type=int, default=64,
                        help="Motif embedding space dimension")
    parser.add_argument("--n-prototypes", type=int, default=64,
                        help="Number of learnable prototype vectors")

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than Phase 10 since fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=25)

    # Loss weights (from plan Section 4.4)
    parser.add_argument("--profile-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument("--tf-weight", type=float, default=5.0,
                        help="TF identity weight (Hungarian matching)")
    parser.add_argument("--occupancy-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--orthogonality-weight", type=float, default=0.5)
    parser.add_argument("--site-supervision-weight", type=float, default=1.0)
    parser.add_argument("--slot-count-weight", type=float, default=0.5)
    parser.add_argument("--load-balancing-weight", type=float, default=0.3)
    # Phase 1 losses
    parser.add_argument("--anchor-weight", type=float, default=0.5,
                        help="Ground known TFs in embedding space")
    parser.add_argument("--importance-weight", type=float, default=0.3,
                        help="Per-base attribution quality")
    parser.add_argument("--prototype-diversity-weight", type=float, default=0.2,
                        help="Spread prototypes for novel motifs")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Cluster same-TF slots together")

    # Hardware
    parser.add_argument("--gpus", type=str, default="0,9")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    # Resolve paths
    data_dir = args.data_dir if args.data_dir.is_absolute() else base_dir / args.data_dir
    importance_path = args.importance_path if args.importance_path.is_absolute() else base_dir / args.importance_path
    pretrained_path = args.pretrained_path if args.pretrained_path.is_absolute() else base_dir / args.pretrained_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"phase1_{timestamp}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_tfs = 7  # CTCF, GATA1, TAL1, MYC, MAX, SPI1, CEBPB

    print("=" * 70)
    print("Phase 1: BEACON-v3 Pipeline Training")
    print("  Replace DeepSHAP + TF-MoDISco with native attribution & motifs")
    print("=" * 70)
    print(f"Data:         {data_dir}")
    print(f"Importance:   {importance_path}")
    print(f"Pretrained:   {pretrained_path}")
    print(f"Output:       {experiment_dir}")
    print()

    # ------------------------------------------------------------------
    # 1. Datasets
    # ------------------------------------------------------------------
    has_importance = importance_path.exists()
    if has_importance:
        print(f"  Gradient importance found: {importance_path}")
    else:
        print(f"  WARNING: Gradient importance not found at {importance_path}")
        print(f"  Attribution head will train WITHOUT importance supervision")
        print(f"  Run precompute_gradient_importance.py first for best results")

    imp_str = str(importance_path) if has_importance else None

    train_dataset = BEACONDataset(
        str(data_dir / "train.h5"),
        seq_length=args.seq_len, augment=True,
        n_slots=args.n_slots, extract_peaks=True,
        importance_path=imp_str,
    )
    val_dataset = BEACONDataset(
        str(data_dir / "val.h5"),
        seq_length=args.seq_len, augment=False,
        n_slots=args.n_slots, extract_peaks=True,
    )
    test_dataset = BEACONDataset(
        str(data_dir / "test.h5"),
        seq_length=args.seq_len, augment=False,
        n_slots=args.n_slots, extract_peaks=True,
    )

    print(f"\nDatasets:")
    print(f"  Train: {len(train_dataset)} samples"
          f" (importance: {'yes' if train_dataset.importance is not None else 'no'})")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 2. Model: BEACON with Phase 1 heads
    # ------------------------------------------------------------------
    model = BEACON(
        seq_len=args.seq_len, input_channels=4,
        backbone_type="dilated",
        backbone_dim=args.backbone_dim,
        backbone_layers=args.backbone_layers,
        n_slots=args.n_slots,
        slot_dim=args.slot_dim,
        n_iterations=args.n_iterations,
        n_tfs=n_tfs,
        position_mode="gaussian",
        dropout=0.1,
        attention_mode="independent",
        # Phase 1 heads
        use_attribution_head=True,
        use_motif_embedding_head=True,
        motif_dim=args.motif_dim,
        n_prototypes=args.n_prototypes,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_new_params = 0
    if model.attribution_head is not None:
        n_new_params += sum(p.numel() for p in model.attribution_head.parameters())
    if model.motif_embedding_head is not None:
        n_new_params += sum(p.numel() for p in model.motif_embedding_head.parameters())

    print(f"\nModel: BEACON-v3")
    print(f"  Total parameters:     {n_params:,}")
    print(f"  New Phase 1 params:   {n_new_params:,}")
    print(f"  Attribution head:     enabled")
    print(f"  Motif embedding head: enabled (dim={args.motif_dim}, prototypes={args.n_prototypes})")

    # ------------------------------------------------------------------
    # 3. Load pretrained weights (partial — skip new heads)
    # ------------------------------------------------------------------
    if not args.from_scratch and pretrained_path.exists():
        print(f"\n--- Loading Pretrained Weights ---")
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            pretrained_state = checkpoint["model_state_dict"]
        else:
            pretrained_state = checkpoint

        # Filter out keys that don't exist in new model or have size mismatch
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

        model.load_state_dict(model_state)
        print(f"  Loaded {len(loaded_keys)} parameters from pretrained checkpoint")
        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} keys (new or shape-mismatched):")
            for k in skipped_keys[:10]:
                print(f"    - {k}")
            if len(skipped_keys) > 10:
                print(f"    ... and {len(skipped_keys) - 10} more")

        # New heads that will be randomly initialized
        new_modules = []
        for k in model_state:
            if k not in pretrained_state:
                module_name = k.rsplit('.', 1)[0] if '.' in k else k
                if module_name not in new_modules:
                    new_modules.append(module_name)
        if new_modules:
            print(f"\n  New modules (randomly initialized):")
            # Deduplicate to top-level
            seen = set()
            for m in new_modules:
                top = m.split('.')[0]
                if top not in seen:
                    print(f"    - {top}")
                    seen.add(top)
    elif args.from_scratch:
        print(f"\n  Training from scratch (--from-scratch)")
    else:
        print(f"\n  WARNING: Pretrained checkpoint not found at {pretrained_path}")
        print(f"  Training from scratch")

    # ------------------------------------------------------------------
    # 4. Multi-GPU setup
    # ------------------------------------------------------------------
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        model = model.cuda()
        print(f"\n  Using GPU(s): {gpu_ids}")

    # ------------------------------------------------------------------
    # 5. Loss function with Phase 1 losses
    # ------------------------------------------------------------------
    # Determine effective importance weight — set to 0 if no importance data
    effective_importance_weight = args.importance_weight if has_importance else 0.0
    if not has_importance and args.importance_weight > 0:
        print(f"\n  NOTE: importance_weight set to 0.0 (no importance data available)")

    loss_fn = BEACONLoss(
        n_tfs=n_tfs, seq_len=args.seq_len,
        position_mode="gaussian",
        # Standard losses
        profile_weight=args.profile_weight,
        position_weight=args.position_weight,
        tf_weight=args.tf_weight,
        occupancy_weight=args.occupancy_weight,
        diversity_weight=args.diversity_weight,
        orthogonality_weight=args.orthogonality_weight,
        site_supervision_weight=args.site_supervision_weight,
        use_hungarian=True,
        slot_count_weight=args.slot_count_weight,
        load_balancing_weight=args.load_balancing_weight,
        contrastive_weight=args.contrastive_weight,
        # Phase 1 losses
        anchor_weight=args.anchor_weight,
        importance_weight=effective_importance_weight,
        prototype_diversity_weight=args.prototype_diversity_weight,
    )

    print(f"\nLoss weights:")
    print(f"  profile:              {args.profile_weight}")
    print(f"  tf_identity:          {args.tf_weight} (Hungarian)")
    print(f"  site_supervision:     {args.site_supervision_weight}")
    print(f"  --- Phase 1 ---")
    print(f"  anchor:               {args.anchor_weight}")
    print(f"  importance:           {effective_importance_weight}")
    print(f"  prototype_diversity:  {args.prototype_diversity_weight}")
    print(f"  contrastive:          {args.contrastive_weight}")

    # ------------------------------------------------------------------
    # 6. Save config
    # ------------------------------------------------------------------
    config = {
        'experiment_name': experiment_name,
        'model_type': 'BEACON-v3-phase1',
        'phase': 'Phase 1: Replace DeepSHAP + TF-MoDISco',
        'seq_len': args.seq_len,
        'n_slots': args.n_slots,
        'backbone_dim': args.backbone_dim,
        'backbone_layers': args.backbone_layers,
        'slot_dim': args.slot_dim,
        'n_iterations': args.n_iterations,
        'n_tfs': n_tfs,
        'motif_dim': args.motif_dim,
        'n_prototypes': args.n_prototypes,
        'use_attribution_head': True,
        'use_motif_embedding_head': True,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'patience': args.patience,
        'pretrained_path': str(pretrained_path) if not args.from_scratch else None,
        'has_importance_data': has_importance,
        'attention_mode': 'independent',
        'use_hungarian': True,
        'loss_weights': {
            'profile': args.profile_weight,
            'position': args.position_weight,
            'tf_identity': args.tf_weight,
            'occupancy': args.occupancy_weight,
            'diversity': args.diversity_weight,
            'orthogonality': args.orthogonality_weight,
            'site_supervision': args.site_supervision_weight,
            'slot_count': args.slot_count_weight,
            'load_balancing': args.load_balancing_weight,
            'contrastive': args.contrastive_weight,
            'anchor': args.anchor_weight,
            'importance': effective_importance_weight,
            'prototype_diversity': args.prototype_diversity_weight,
        },
        'gpu_ids': gpu_ids,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    trainer = BEACONTrainer(
        model, train_loader, val_loader, test_loader,
        loss_fn, output_dir=str(experiment_dir),
        num_epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        use_amp=True, log_interval=10, val_interval=1,
        save_interval=5, early_stopping_patience=args.patience,
        num_slots=args.n_slots, num_tfs=n_tfs,
    )

    print(f"\n{'=' * 70}")
    print(f"Starting Phase 1 Training")
    print(f"{'=' * 70}")
    results = trainer.train()

    print(f"\n{'=' * 70}")
    print(f"Phase 1 Training Complete")
    print(f"{'=' * 70}")
    print(f"Results saved to: {experiment_dir}")

    # ------------------------------------------------------------------
    # 8. Quick attribution benchmark
    # ------------------------------------------------------------------
    print(f"\n--- Quick Attribution Head Evaluation ---")
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    device = next(base_model.parameters()).device

    try:
        import time
        base_model.eval()
        # Time a single forward pass including attribution
        sample = next(iter(test_loader))
        seq = sample['sequence'].to(device)

        # Warm up
        with torch.no_grad():
            _ = base_model(seq[:1])

        # Time attribution (forward pass only, no backward)
        times = []
        n_timing = min(20, len(test_dataset))
        with torch.no_grad():
            for i in range(n_timing):
                t0 = time.time()
                outputs = base_model(seq[:1])
                torch.cuda.synchronize()
                times.append(time.time() - t0)

        avg_time = sum(times) / len(times) * 1000  # ms
        print(f"  Attribution forward pass: {avg_time:.1f} ms/sequence")
        print(f"  Target: <100 ms (DeepSHAP: ~8400 ms)")
        if avg_time < 100:
            print(f"  PASS: {8400/avg_time:.0f}x faster than DeepSHAP")
        else:
            print(f"  Speedup vs DeepSHAP: {8400/avg_time:.0f}x")

        # Check attribution output exists and has correct shape
        if 'attribution' in outputs:
            attr_shape = outputs['attribution'].shape
            print(f"  Attribution shape: {attr_shape}")
        if 'motif_embeddings' in outputs:
            print(f"  Motif embeddings: {outputs['motif_embeddings'].shape}")
            print(f"  Prototype assignments: {outputs['prototype_assignments'].shape}")
            if 'anchor_distances' in outputs:
                print(f"  Anchor distances: {outputs['anchor_distances'].shape}")

    except Exception as e:
        print(f"  Attribution evaluation skipped: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
