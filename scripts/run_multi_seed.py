#!/usr/bin/env python3
"""
Multi-seed training and evaluation for K562-7tf slot_dropout.

Trains with 3 seeds and reports cross-seed consistency.

Usage:
    # Train all seeds sequentially
    python scripts/run_multi_seed.py --device cuda:2

    # Train one seed (for parallel runs in separate terminals)
    python scripts/run_multi_seed.py --device cuda:2 --seeds 42

    # Evaluate only (after training)
    python scripts/run_multi_seed.py --evaluate-only --device cuda:2
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_model, load_test_data, run_model_batch,
    per_sample_pearson, json_serialize,
    bootstrap_ci, per_sample_jsd, per_sample_mnll,
    DATASET_CONFIGS, BASE_DIR,
)

SEEDS = [42, 123, 456]
OUTPUT_ROOT = BASE_DIR / "outputs" / "multi_seed"


def train_seed(seed, device):
    """Train K562-7tf slot_dropout with one seed."""
    output_dir = OUTPUT_ROOT / f"seed_{seed}"
    if (output_dir / "training_complete").exists():
        print(f"  Seed {seed}: already done (marker found)")
        return

    print(f"\n{'='*70}\nTraining seed {seed} on {device}\n{'='*70}")

    from beacon.models import BEACON, BEACONLoss
    from beacon.training import BEACONTrainer
    from beacon.data.dataset import BEACONDataset
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = BASE_DIR / "data/processed/multi_tf_k562"
    n_tfs = 7

    train_ds = BEACONDataset(str(data_dir / "train.h5"), seq_length=2000, augment=True,
                              n_slots=16, extract_peaks=False)
    val_ds = BEACONDataset(str(data_dir / "val.h5"), seq_length=2000, augment=False,
                            n_slots=16, extract_peaks=False)
    test_ds = BEACONDataset(str(data_dir / "test.h5"), seq_length=2000, augment=False,
                             n_slots=16, extract_peaks=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = BEACON(
        seq_len=2000, input_channels=4, backbone_type="dilated",
        backbone_dim=128, backbone_layers=4, n_slots=16, slot_dim=128,
        n_iterations=3, n_tfs=n_tfs, position_mode="gaussian",
        dropout=0.1, attention_mode="independent",
    )
    loss_fn = BEACONLoss(
        n_tfs=n_tfs, seq_len=2000, position_mode="gaussian",
        profile_weight=1.0, position_weight=1.0, tf_weight=5.0,
        occupancy_weight=1.0, diversity_weight=0.1, orthogonality_weight=0.1,
        site_supervision_weight=0.5, use_hungarian=True,
        slot_count_weight=0.5, load_balancing_weight=0.3, contrastive_weight=0.5,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = BEACONTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, loss_fn=loss_fn,
        output_dir=output_dir, experiment_name=f"beacon_seed{seed}_{ts}",
        device=device, num_epochs=80, learning_rate=3e-4, weight_decay=0.01,
        grad_clip=1.0, use_amp=True, log_interval=10, val_interval=1,
        save_interval=5, early_stopping_patience=25,
        num_slots=16, num_tfs=n_tfs, track_slot_utilization=True,
        use_tensorboard=False,
    )

    # Warmup
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 2000, 4, device=torch.device(device)))
    model.train()

    res = trainer.train()
    with open(output_dir / "training_complete", 'w') as f:
        json.dump({"seed": seed, "best_val_loss": res['best_val_loss'],
                    "epochs": res['total_epochs']}, f)
    print(f"  Seed {seed} done: val_loss={res['best_val_loss']:.4f}")


def evaluate_seeds(device, n_boot=10000):
    """Evaluate all trained seed checkpoints."""
    print("\n" + "=" * 70)
    print("Multi-seed evaluation")
    print("=" * 70)

    cfg = DATASET_CONFIGS["K562-7tf"]
    test_path = BASE_DIR / cfg["data_dir"] / "test.h5"
    seqs, profs, tf_idx, _ = load_test_data(test_path)
    tf_names = cfg["tf_names"]

    seed_data = {}
    all_rs = {}

    for seed in SEEDS:
        candidates = list((OUTPUT_ROOT / f"seed_{seed}").glob("**/best_model.pt"))
        if not candidates:
            print(f"  Seed {seed}: no checkpoint")
            continue
        ckpt = candidates[0]
        print(f"\n  Seed {seed}: {ckpt}")
        model = load_model(ckpt, cfg["n_tfs"], device)
        out = run_model_batch(model, seqs, device)
        pred = out['profile']

        rs = per_sample_pearson(pred, profs)
        jsds = per_sample_jsd(pred, profs)
        mnlls = per_sample_mnll(pred, profs)
        all_rs[seed] = rs

        per_tf = {}
        for ti, tn in enumerate(tf_names):
            mask = tf_idx == ti
            if mask.sum() < 5:
                continue
            per_tf[tn] = {"mean_r": float(np.mean(rs[mask])), "n": int(mask.sum())}

        r_ci = bootstrap_ci(rs, n_boot)
        print(f"    r: {r_ci['mean']:.4f} [{r_ci['ci_lower']:.4f}, {r_ci['ci_upper']:.4f}]")

        seed_data[str(seed)] = {
            "checkpoint": str(ckpt),
            "pearson": r_ci,
            "jsd": bootstrap_ci(jsds, n_boot),
            "mnll": bootstrap_ci(mnlls, n_boot),
            "per_tf": per_tf,
        }
        del model; torch.cuda.empty_cache()

    cross_seed = {}
    if len(seed_data) >= 2:
        means = [seed_data[s]["pearson"]["mean"] for s in seed_data]
        cross_seed["mean_r"] = float(np.mean(means))
        cross_seed["std_r"] = float(np.std(means))

        seeds_list = list(all_rs.keys())
        corrs = {}
        for i in range(len(seeds_list)):
            for j in range(i + 1, len(seeds_list)):
                s1, s2 = seeds_list[i], seeds_list[j]
                r, _ = pearsonr(all_rs[s1], all_rs[s2])
                corrs[f"seed_{s1}_vs_{s2}"] = float(r)
                print(f"    Per-sample r (seed {s1} vs {s2}): {r:.4f}")
        cross_seed["per_sample_correlations"] = corrs

    return {"per_seed": seed_data, "cross_seed": cross_seed}


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training & evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    if not args.evaluate_only:
        for s in seeds:
            train_seed(s, device)

    results = evaluate_seeds(device, args.n_bootstrap)
    path = out_dir / "multi_seed.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
