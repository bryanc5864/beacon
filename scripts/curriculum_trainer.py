#!/usr/bin/env python3
"""
Curriculum Trainer for BEACON - Systematic Complexity Analysis

Runs through multiple configurations to find where the model breaks:
- Experiment A: 10 TFs, 3-5 sites, no overlap, no noise (site count test)
- Experiment B: 25 TFs, 1-2 sites, no overlap, no noise (TF count test)
- Experiment C: 10 TFs, 1-2 sites, overlap, no noise (overlap test)
- Experiment D: 10 TFs, 1-2 sites, no overlap, noise (noise test)
- Experiment E: 25 TFs, 3-5 sites, no overlap, no noise (combined medium)

Each experiment: generates data, trains, validates per epoch, final evaluation.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models.beacon import BEACON
from beacon.training.metrics import BEACONMetrics


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    n_tfs: int
    min_sites: int
    max_sites: int
    allow_overlap: bool
    noise_sigma: float
    seq_length: int = 1000
    n_train: int = 10000
    n_val: int = 1000
    n_test: int = 1000
    n_slots: int = 16
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    description: str = ""


# Define all experiments
EXPERIMENTS = {
    # Ablation experiments to find breaking point
    "A_site_count": ExperimentConfig(
        name="A_site_count",
        n_tfs=10,
        min_sites=3,
        max_sites=5,
        allow_overlap=False,
        noise_sigma=0.0,
        description="Test if site count (3-5) breaks the model",
    ),
    "B_tf_count": ExperimentConfig(
        name="B_tf_count",
        n_tfs=25,
        min_sites=1,
        max_sites=2,
        allow_overlap=False,
        noise_sigma=0.0,
        description="Test if TF count (25) breaks the model",
    ),
    "C_overlap": ExperimentConfig(
        name="C_overlap",
        n_tfs=10,
        min_sites=1,
        max_sites=2,
        allow_overlap=True,
        noise_sigma=0.0,
        description="Test if overlap breaks the model",
    ),
    "D_noise": ExperimentConfig(
        name="D_noise",
        n_tfs=10,
        min_sites=1,
        max_sites=2,
        allow_overlap=False,
        noise_sigma=0.15,
        description="Test if noise (σ=0.15) breaks the model",
    ),
    "E_combined_medium": ExperimentConfig(
        name="E_combined_medium",
        n_tfs=25,
        min_sites=3,
        max_sites=5,
        allow_overlap=False,
        noise_sigma=0.0,
        description="Combined medium difficulty",
    ),
    # Additional diagnostic experiments
    "F_mild_overlap_noise": ExperimentConfig(
        name="F_mild_overlap_noise",
        n_tfs=15,
        min_sites=2,
        max_sites=4,
        allow_overlap=True,
        noise_sigma=0.05,
        description="Mild overlap + mild noise",
    ),
    "G_high_tf_simple": ExperimentConfig(
        name="G_high_tf_simple",
        n_tfs=40,
        min_sites=1,
        max_sites=2,
        allow_overlap=False,
        noise_sigma=0.0,
        description="High TF count (40) with simple sites",
    ),
    "H_many_sites_simple": ExperimentConfig(
        name="H_many_sites_simple",
        n_tfs=10,
        min_sites=5,
        max_sites=8,
        allow_overlap=False,
        noise_sigma=0.0,
        description="Many sites (5-8) with few TFs",
    ),
}

# Curriculum stages (for progressive training)
CURRICULUM_STAGES = [
    ExperimentConfig(
        name="curriculum_stage1",
        n_tfs=10,
        min_sites=1,
        max_sites=2,
        allow_overlap=False,
        noise_sigma=0.0,
        epochs=10,
        description="Stage 1: Easy baseline",
    ),
    ExperimentConfig(
        name="curriculum_stage2",
        n_tfs=20,
        min_sites=2,
        max_sites=4,
        allow_overlap=False,
        noise_sigma=0.0,
        epochs=20,
        description="Stage 2: Medium TFs and sites",
    ),
    ExperimentConfig(
        name="curriculum_stage3",
        n_tfs=35,
        min_sites=3,
        max_sites=6,
        allow_overlap=True,
        noise_sigma=0.05,
        epochs=30,
        description="Stage 3: More TFs, light overlap",
    ),
    ExperimentConfig(
        name="curriculum_stage4",
        n_tfs=50,
        min_sites=3,
        max_sites=8,
        allow_overlap=True,
        noise_sigma=0.15,
        epochs=50,
        description="Stage 4: Full difficulty",
    ),
]


class SyntheticDataGenerator:
    """Generate synthetic binding data with configurable complexity."""

    # Base motifs for TF families
    BASE_MOTIFS = {
        "E-box": "CACGTG",
        "GATA": "GATA",
        "ETS": "GGAA",
        "AP1": "TGACTCA",
        "CEBP": "TTGCGCAA",
        "NFKB": "GGGACTTTCC",
        "CTCF": "CCGCGNGGNGGCAG",
        "SOX": "AACAAT",
        "FOXA": "TGTTTAC",
        "SP1": "GGGCGG",
        "TATA": "TATAAAA",
        "POU": "ATGCAAAT",
        "NR": "AGGTCA",
        "HOX": "TAAT",
        "ZNF": "GCGCGC",
    }

    def __init__(self, config: ExperimentConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.motifs = self._create_motif_library()

    def _create_motif_library(self) -> Dict[int, Tuple[str, np.ndarray]]:
        """Create motif library with variants for n_tfs transcription factors."""
        motifs = {}
        base_motif_list = list(self.BASE_MOTIFS.items())

        for tf_idx in range(self.config.n_tfs):
            # Cycle through base motifs and create variants
            base_name, base_seq = base_motif_list[tf_idx % len(base_motif_list)]
            variant_num = tf_idx // len(base_motif_list)

            # Create variant by adding mutations
            seq = list(base_seq)
            for _ in range(variant_num % 3):
                if seq:
                    pos = self.rng.randint(0, len(seq))
                    seq[pos] = self.rng.choice(list("ACGT"))
            motif_seq = "".join(seq).replace("N", self.rng.choice(list("ACGT")))

            # Create PWM from sequence
            pwm = self._seq_to_pwm(motif_seq)
            motifs[tf_idx] = (f"{base_name}_v{variant_num}", pwm)

        return motifs

    def _seq_to_pwm(self, seq: str, noise: float = 0.1) -> np.ndarray:
        """Convert sequence to PWM with some noise."""
        alphabet = "ACGT"
        pwm = np.zeros((len(seq), 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in alphabet:
                pwm[i, alphabet.index(base)] = 1.0 - noise
                for j in range(4):
                    if j != alphabet.index(base):
                        pwm[i, j] = noise / 3
            else:
                pwm[i, :] = 0.25
        return pwm

    def generate_sample(self) -> Dict:
        """Generate a single sample with binding sites."""
        seq_len = self.config.seq_length

        # Generate random background sequence
        seq_onehot = np.zeros((seq_len, 4), dtype=np.float32)
        bg_seq = self.rng.choice(4, size=seq_len)
        seq_onehot[np.arange(seq_len), bg_seq] = 1.0

        # Initialize profile and site labels
        profile = np.zeros(seq_len, dtype=np.float32)
        site_positions = []
        site_tf_indices = []
        site_occupancies = []

        # Determine number of sites
        n_sites = self.rng.randint(self.config.min_sites, self.config.max_sites + 1)

        # Place binding sites
        occupied_ranges = []
        attempts = 0
        max_attempts = n_sites * 20

        while len(site_positions) < n_sites and attempts < max_attempts:
            attempts += 1

            # Choose TF and get motif
            tf_idx = self.rng.randint(0, self.config.n_tfs)
            motif_name, pwm = self.motifs[tf_idx]
            motif_len = len(pwm)

            # Choose position
            max_pos = seq_len - motif_len - 10
            if max_pos < 10:
                continue
            pos = self.rng.randint(10, max_pos)

            # Check overlap
            if not self.config.allow_overlap:
                overlaps = False
                for start, end in occupied_ranges:
                    if not (pos + motif_len < start or pos > end):
                        overlaps = True
                        break
                if overlaps:
                    continue

            # Place motif in sequence
            for i, pwm_row in enumerate(pwm):
                if pos + i < seq_len:
                    seq_onehot[pos + i] = pwm_row

            # Add to profile (Gaussian peak)
            center = pos + motif_len // 2
            sigma = motif_len * 1.5
            x = np.arange(seq_len)
            peak = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            occupancy = 0.5 + 0.5 * self.rng.random()
            profile += peak * occupancy

            # Record site
            site_positions.append(center / seq_len)  # Normalized position
            site_tf_indices.append(tf_idx)
            site_occupancies.append(occupancy)
            occupied_ranges.append((pos - 5, pos + motif_len + 5))

        # Normalize profile
        if profile.max() > 0:
            profile = profile / profile.max()

        # Add noise
        if self.config.noise_sigma > 0:
            noise = self.rng.normal(0, self.config.noise_sigma, seq_len)
            profile = np.clip(profile + noise, 0, 1).astype(np.float32)

        # Pad to n_slots
        n_slots = self.config.n_slots
        positions = np.zeros(n_slots, dtype=np.float32)
        tf_indices = np.zeros(n_slots, dtype=np.int64)
        occupancies = np.zeros(n_slots, dtype=np.float32)

        n_actual = min(len(site_positions), n_slots)
        positions[:n_actual] = site_positions[:n_actual]
        tf_indices[:n_actual] = site_tf_indices[:n_actual]
        occupancies[:n_actual] = site_occupancies[:n_actual]

        return {
            "sequence": seq_onehot,
            "profile": profile,
            "positions": positions,
            "tf_indices": tf_indices,
            "occupancies": occupancies,
            "n_sites": n_actual,
        }

    def generate_dataset(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate full dataset."""
        sequences = []
        profiles = []
        positions = []
        tf_indices = []
        occupancies = []

        for _ in range(n_samples):
            sample = self.generate_sample()
            sequences.append(sample["sequence"])
            profiles.append(sample["profile"])
            positions.append(sample["positions"])
            tf_indices.append(sample["tf_indices"])
            occupancies.append(sample["occupancies"])

        return {
            "sequences": np.stack(sequences),
            "profiles": np.stack(profiles),
            "positions": np.stack(positions),
            "tf_indices": np.stack(tf_indices),
            "occupancies": np.stack(occupancies),
        }


class CurriculumTrainer:
    """Trainer that runs through multiple experiments systematically."""

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda",
        gpus: List[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.gpus = gpus or [0]

        # Setup logging
        self.setup_logging()

        # Results storage
        self.all_results = {}

    def setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "curriculum_training.log"

        self.logger = logging.getLogger("CurriculumTrainer")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(ch)

    def log(self, msg: str):
        """Log message."""
        self.logger.info(msg)

    def create_model(self, config: ExperimentConfig) -> nn.Module:
        """Create BEACON model for experiment."""
        model = BEACON(
            seq_len=config.seq_length,
            n_tfs=config.n_tfs,
            n_slots=config.n_slots,
            backbone_dim=128,
            backbone_layers=4,
            slot_dim=128,
            n_iterations=3,
        )

        if len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        return model.to(self.device)

    def create_dataloaders(
        self,
        config: ExperimentConfig,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generate data and create dataloaders."""
        self.log(f"Generating data: {config.n_train} train, {config.n_val} val, {config.n_test} test")

        generator = SyntheticDataGenerator(config, seed=seed)

        train_data = generator.generate_dataset(config.n_train)
        val_data = generator.generate_dataset(config.n_val)
        test_data = generator.generate_dataset(config.n_test)

        def make_loader(data: Dict, shuffle: bool) -> DataLoader:
            dataset = TensorDataset(
                torch.tensor(data["sequences"]),
                torch.tensor(data["profiles"]),
                torch.tensor(data["positions"]),
                torch.tensor(data["tf_indices"]),
                torch.tensor(data["occupancies"]),
            )
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True,
            )

        return (
            make_loader(train_data, shuffle=True),
            make_loader(val_data, shuffle=False),
            make_loader(test_data, shuffle=False),
        )

    def compute_loss(
        self,
        model: nn.Module,
        batch: Tuple,
        config: ExperimentConfig,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss."""
        sequences, profiles, positions, tf_indices, occupancies = batch
        sequences = sequences.to(self.device)
        profiles = profiles.to(self.device)
        positions = positions.to(self.device)
        tf_indices = tf_indices.to(self.device)
        occupancies = occupancies.to(self.device)

        # Forward pass
        outputs = model(sequences)

        # Profile loss (MSE)
        profile_loss = nn.functional.mse_loss(
            outputs["profile"].squeeze(-1),
            profiles
        )

        # Position loss (MSE on occupied slots)
        # positions output is [B, K, 2] where dim 2 is (mean, std)
        mask = occupancies > 0.1
        if mask.sum() > 0:
            pred_pos = outputs["positions"][:, :, 0]  # Take mean position [B, K]
            position_loss = nn.functional.mse_loss(
                pred_pos[mask],
                positions[mask]
            )
        else:
            position_loss = torch.tensor(0.0, device=self.device)

        # TF classification loss
        if mask.sum() > 0:
            pred_tf = outputs["tf_logits"]  # [B, n_slots, n_tfs]
            tf_loss = nn.functional.cross_entropy(
                pred_tf[mask],
                tf_indices[mask]
            )
        else:
            tf_loss = torch.tensor(0.0, device=self.device)

        # Occupancy loss
        pred_occ = outputs["occupancy"].squeeze(-1)  # [B, K]
        occupancy_loss = nn.functional.mse_loss(pred_occ, occupancies)

        # Total loss
        total_loss = profile_loss + 5.0 * position_loss + 2.0 * tf_loss + occupancy_loss

        return total_loss, {
            "total": total_loss.item(),
            "profile": profile_loss.item(),
            "position": position_loss.item(),
            "tf": tf_loss.item(),
            "occupancy": occupancy_loss.item(),
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: ExperimentConfig,
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()

        # Initialize metrics calculator
        metrics_calc = BEACONMetrics(
            seq_len=config.seq_length,
            n_tfs=config.n_tfs,
            occupancy_threshold=0.5,
        )

        for batch in dataloader:
            sequences, profiles, positions, tf_indices, occupancies = batch
            sequences = sequences.to(self.device)

            outputs = model(sequences)

            # Prepare targets dict
            targets = {
                "profile": profiles,
                "positions": positions,
                "tf_ids": tf_indices,
                "occupancy": occupancies,
            }

            # Update metrics
            metrics_calc.update(outputs, targets)

        # Compute all metrics
        metrics = metrics_calc.compute()

        return metrics

    def train_experiment(
        self,
        config: ExperimentConfig,
        seed: int = 42,
    ) -> Dict:
        """Train a single experiment configuration."""
        exp_dir = self.output_dir / config.name
        exp_dir.mkdir(exist_ok=True)

        self.log("=" * 70)
        self.log(f"EXPERIMENT: {config.name}")
        self.log(f"Description: {config.description}")
        self.log(f"Config: {config.n_tfs} TFs, {config.min_sites}-{config.max_sites} sites, "
                f"overlap={config.allow_overlap}, noise={config.noise_sigma}")
        self.log("=" * 70)

        # Save config
        with open(exp_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Create model and data
        model = self.create_model(config)
        train_loader, val_loader, test_loader = self.create_dataloaders(config, seed)

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        scaler = GradScaler()

        # Training history
        history = {
            "train_loss": [],
            "val_metrics": [],
            "batch_logs": [],
        }

        best_val_f1 = 0.0
        best_epoch = 0
        start_time = time.time()

        for epoch in range(config.epochs):
            model.train()
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                with autocast():
                    loss, loss_dict = self.compute_loss(model, batch, config)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss_dict)

                # Log every 50 batches
                if batch_idx % 50 == 0:
                    batch_log = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": loss_dict["total"],
                        "profile": loss_dict["profile"],
                        "position": loss_dict["position"],
                        "tf": loss_dict["tf"],
                    }
                    history["batch_logs"].append(batch_log)
                    self.log(f"  Epoch {epoch+1}/{config.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                            f"Loss: {loss_dict['total']:.4f}")

            scheduler.step()

            # Epoch average loss
            avg_loss = np.mean([l["total"] for l in epoch_losses])
            history["train_loss"].append(avg_loss)

            # Validation
            val_metrics = self.evaluate(model, val_loader, config)
            history["val_metrics"].append(val_metrics)

            self.log(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_loss:.4f} | "
                    f"Val Site F1: {val_metrics.get('site_f1', 0):.4f} | "
                    f"Val TF Acc: {val_metrics.get('tf_accuracy', 0):.4f} | "
                    f"Val Profile r: {val_metrics.get('profile_pearson', 0):.4f}")

            # Save best model
            if val_metrics.get("site_f1", 0) > best_val_f1:
                best_val_f1 = val_metrics.get("site_f1", 0)
                best_epoch = epoch
                torch.save(model.state_dict(), exp_dir / "best_model.pt")

        # Load best model for final test
        model.load_state_dict(torch.load(exp_dir / "best_model.pt"))
        test_metrics = self.evaluate(model, test_loader, config)

        elapsed = time.time() - start_time

        # Final results
        results = {
            "config": asdict(config),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "training_time_seconds": elapsed,
            "test_metrics": test_metrics,
            "history": history,
        }

        # Save results
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

        # Log final results
        self.log("-" * 70)
        self.log(f"FINAL RESULTS: {config.name}")
        self.log(f"  Training time: {elapsed/60:.1f} minutes")
        self.log(f"  Best epoch: {best_epoch + 1}")
        self.log(f"  Test Site F1: {test_metrics.get('site_f1', 0):.4f}")
        self.log(f"  Test TF Accuracy: {test_metrics.get('tf_accuracy', 0):.4f}")
        self.log(f"  Test Profile Pearson: {test_metrics.get('profile_pearson', 0):.4f}")
        self.log("-" * 70)

        return results

    def run_all_experiments(self, experiments: Dict[str, ExperimentConfig] = None):
        """Run all experiments."""
        if experiments is None:
            experiments = EXPERIMENTS

        self.log("=" * 70)
        self.log("CURRICULUM TRAINING - SYSTEMATIC COMPLEXITY ANALYSIS")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Experiments: {list(experiments.keys())}")
        self.log(f"Device: {self.device}, GPUs: {self.gpus}")
        self.log("=" * 70)

        for exp_name, config in experiments.items():
            try:
                results = self.train_experiment(config)
                self.all_results[exp_name] = results
            except Exception as e:
                self.log(f"ERROR in experiment {exp_name}: {e}")
                import traceback
                self.log(traceback.format_exc())
                self.all_results[exp_name] = {"error": str(e)}

        # Save summary
        self.save_summary()

    def run_curriculum(self):
        """Run curriculum learning (progressive difficulty)."""
        self.log("=" * 70)
        self.log("CURRICULUM LEARNING - PROGRESSIVE TRAINING")
        self.log("=" * 70)

        # Start with fresh model
        model = None

        for stage_idx, config in enumerate(CURRICULUM_STAGES):
            self.log(f"\n{'='*70}")
            self.log(f"CURRICULUM STAGE {stage_idx + 1}/{len(CURRICULUM_STAGES)}")
            self.log(f"{'='*70}")

            stage_dir = self.output_dir / f"curriculum_stage{stage_idx + 1}"
            stage_dir.mkdir(exist_ok=True)

            # Create or continue model
            if model is None:
                model = self.create_model(config)
            else:
                # Adapt model for new n_tfs if needed
                old_n_tfs = model.module.n_tfs if hasattr(model, 'module') else model.n_tfs
                if config.n_tfs != old_n_tfs:
                    self.log(f"Adapting model from {old_n_tfs} to {config.n_tfs} TFs")
                    model = self.create_model(config)

            # Train this stage
            results = self.train_experiment(config)
            self.all_results[f"curriculum_stage{stage_idx + 1}"] = results

        self.save_summary()

    def save_summary(self):
        """Save summary of all experiments."""
        summary_path = self.output_dir / "summary.json"

        # Extract key metrics for comparison
        summary = {
            "timestamp": datetime.now().isoformat(),
            "experiments": {}
        }

        for exp_name, results in self.all_results.items():
            if "error" in results:
                summary["experiments"][exp_name] = {"error": results["error"]}
            else:
                test_metrics = results.get("test_metrics", {})
                summary["experiments"][exp_name] = {
                    "config": {
                        "n_tfs": results["config"]["n_tfs"],
                        "sites": f"{results['config']['min_sites']}-{results['config']['max_sites']}",
                        "overlap": results["config"]["allow_overlap"],
                        "noise": results["config"]["noise_sigma"],
                    },
                    "site_f1": test_metrics.get("site_f1", 0),
                    "tf_accuracy": test_metrics.get("tf_accuracy", 0),
                    "profile_pearson": test_metrics.get("profile_pearson", 0),
                    "training_time_min": results.get("training_time_seconds", 0) / 60,
                }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary table
        self.log("\n" + "=" * 90)
        self.log("SUMMARY TABLE")
        self.log("=" * 90)
        self.log(f"{'Experiment':<25} {'TFs':>5} {'Sites':>7} {'Overlap':>8} {'Noise':>6} "
                f"{'Site F1':>8} {'TF Acc':>8} {'Profile r':>10}")
        self.log("-" * 90)

        for exp_name, data in summary["experiments"].items():
            if "error" in data:
                self.log(f"{exp_name:<25} ERROR: {data['error'][:50]}")
            else:
                cfg = data["config"]
                self.log(f"{exp_name:<25} {cfg['n_tfs']:>5} {cfg['sites']:>7} "
                        f"{str(cfg['overlap']):>8} {cfg['noise']:>6.2f} "
                        f"{data['site_f1']:>8.4f} {data['tf_accuracy']:>8.4f} "
                        f"{data['profile_pearson']:>10.4f}")

        self.log("=" * 90)
        self.log(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Trainer - Systematic BEACON Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/curriculum_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
                        help="Output directory")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1,2')")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=None,
                        help="Specific experiments to run (default: all)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Run curriculum learning instead of ablation experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer samples and epochs for testing")

    args = parser.parse_args()

    # Parse GPUs
    gpus = [int(g) for g in args.gpus.split(",")]
    device = f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu"

    # Quick mode modifications
    if args.quick:
        for exp in EXPERIMENTS.values():
            exp.n_train = 1000
            exp.n_val = 200
            exp.n_test = 200
            exp.epochs = 5
        for stage in CURRICULUM_STAGES:
            stage.n_train = 1000
            stage.n_val = 200
            stage.n_test = 200
            stage.epochs = 3

    # Create trainer
    trainer = CurriculumTrainer(
        output_dir=args.output_dir,
        device=device,
        gpus=gpus,
    )

    if args.curriculum:
        trainer.run_curriculum()
    else:
        # Select experiments
        if args.experiments:
            experiments = {k: EXPERIMENTS[k] for k in args.experiments if k in EXPERIMENTS}
        else:
            experiments = EXPERIMENTS

        trainer.run_all_experiments(experiments)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
