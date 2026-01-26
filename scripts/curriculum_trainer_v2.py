#!/usr/bin/env python3
"""
Curriculum Trainer V2 - With Overlap Separation Loss

Key changes from V1:
1. Added overlap_separation_loss to encourage slots to maintain minimum distance
2. Uses BEACONLoss from the main codebase instead of simple MSE
3. Added Experiment I (many sites + overlap) to verify the fix
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
from beacon.models.losses import BEACONLoss
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
    # V2: Overlap separation parameters
    use_overlap_loss: bool = True
    overlap_loss_weight: float = 0.5
    min_slot_distance: float = 0.05  # Normalized distance (5% of sequence)


# V2: Added experiment I for overlap verification
EXPERIMENTS = {
    # PRIORITY: Medium TF test (25 TFs - realistic)
    "MEDIUM_TF": ExperimentConfig(
        name="MEDIUM_TF",
        n_tfs=25,
        min_sites=3,
        max_sites=8,
        allow_overlap=True,
        noise_sigma=0.15,
        seq_length=2000,
        n_train=100000,
        n_val=10000,
        n_test=10000,
        n_slots=16,
        epochs=100,
        batch_size=32,
        lr=1e-4,
        description="25 TFs (realistic) with overlap + noise - DECISION POINT",
        use_overlap_loss=True,
        overlap_loss_weight=0.5,
        min_slot_distance=0.03,
    ),
    # Hard synthetic V2 test (50 TFs - stretch goal)
    "HARD_V2": ExperimentConfig(
        name="HARD_V2",
        n_tfs=50,
        min_sites=3,
        max_sites=8,
        allow_overlap=True,
        noise_sigma=0.15,
        seq_length=2000,
        n_train=100000,
        n_val=10000,
        n_test=10000,
        n_slots=16,
        epochs=100,
        batch_size=32,
        lr=1e-4,
        description="Full hard synthetic with V2 overlap fix",
        use_overlap_loss=True,
        overlap_loss_weight=0.5,
        min_slot_distance=0.03,
    ),
    # Long training test for profile reconstruction
    "LONG_OVERLAP": ExperimentConfig(
        name="LONG_OVERLAP",
        n_tfs=10,
        min_sites=5,
        max_sites=8,
        allow_overlap=True,
        noise_sigma=0.0,
        seq_length=1000,
        n_train=50000,
        n_val=5000,
        n_test=5000,
        n_slots=16,
        epochs=100,
        batch_size=32,
        lr=1e-4,
        description="Long training to test profile reconstruction",
        use_overlap_loss=True,
        overlap_loss_weight=0.5,
        min_slot_distance=0.05,
    ),
    # Original experiments
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
    "H_many_sites_simple": ExperimentConfig(
        name="H_many_sites_simple",
        n_tfs=10,
        min_sites=5,
        max_sites=8,
        allow_overlap=False,
        noise_sigma=0.0,
        description="Many sites (5-8) with few TFs",
    ),
    # V2: New experiment to verify overlap fix
    "I_many_sites_overlap": ExperimentConfig(
        name="I_many_sites_overlap",
        n_tfs=10,
        min_sites=5,
        max_sites=8,
        allow_overlap=True,
        noise_sigma=0.0,
        description="Many sites WITH overlap - tests if V2 fixes overlap",
    ),
    # V2: Harder overlap test
    "J_overlap_stress": ExperimentConfig(
        name="J_overlap_stress",
        n_tfs=15,
        min_sites=4,
        max_sites=6,
        allow_overlap=True,
        noise_sigma=0.1,
        description="Overlap + noise + more TFs - stress test",
    ),
}


class SyntheticDataGenerator:
    """Generate synthetic binding data with configurable complexity."""

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
            base_name, base_seq = base_motif_list[tf_idx % len(base_motif_list)]
            variant_num = tf_idx // len(base_motif_list)

            seq = list(base_seq)
            for _ in range(variant_num % 3):
                if seq:
                    pos = self.rng.randint(0, len(seq))
                    seq[pos] = self.rng.choice(list("ACGT"))
            motif_seq = "".join(seq).replace("N", self.rng.choice(list("ACGT")))

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

        seq_onehot = np.zeros((seq_len, 4), dtype=np.float32)
        bg_seq = self.rng.choice(4, size=seq_len)
        seq_onehot[np.arange(seq_len), bg_seq] = 1.0

        profile = np.zeros(seq_len, dtype=np.float32)
        site_positions = []
        site_tf_indices = []
        site_occupancies = []

        n_sites = self.rng.randint(self.config.min_sites, self.config.max_sites + 1)

        occupied_ranges = []
        attempts = 0
        max_attempts = n_sites * 20

        while len(site_positions) < n_sites and attempts < max_attempts:
            attempts += 1

            tf_idx = self.rng.randint(0, self.config.n_tfs)
            motif_name, pwm = self.motifs[tf_idx]
            motif_len = len(pwm)

            max_pos = seq_len - motif_len - 10
            if max_pos < 10:
                continue
            pos = self.rng.randint(10, max_pos)

            if not self.config.allow_overlap:
                overlaps = False
                for start, end in occupied_ranges:
                    if not (pos + motif_len < start or pos > end):
                        overlaps = True
                        break
                if overlaps:
                    continue

            for i, pwm_row in enumerate(pwm):
                if pos + i < seq_len:
                    seq_onehot[pos + i] = pwm_row

            center = pos + motif_len // 2
            sigma = motif_len * 1.5
            x = np.arange(seq_len)
            peak = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            occupancy = 0.5 + 0.5 * self.rng.random()
            profile += peak * occupancy

            site_positions.append(center / seq_len)
            site_tf_indices.append(tf_idx)
            site_occupancies.append(occupancy)
            occupied_ranges.append((pos - 5, pos + motif_len + 5))

        if profile.max() > 0:
            profile = profile / profile.max()

        if self.config.noise_sigma > 0:
            noise = self.rng.normal(0, self.config.noise_sigma, seq_len)
            profile = np.clip(profile + noise, 0, 1).astype(np.float32)

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


def overlap_separation_loss(
    pred_positions: torch.Tensor,
    pred_occupancy: torch.Tensor,
    min_distance: float = 0.05,
    occupancy_threshold: float = 0.3,
) -> torch.Tensor:
    """
    V2 FIX: Encourage slots to maintain minimum separation.

    This loss penalizes active slots that are too close together,
    forcing the model to separate overlapping binding sites into
    distinct slots.

    Args:
        pred_positions: Predicted positions [B, K, 2] (mean, std)
        pred_occupancy: Predicted occupancy [B, K, 1]
        min_distance: Minimum allowed distance between slots (normalized)
        occupancy_threshold: Only consider slots above this occupancy

    Returns:
        Scalar loss encouraging slot separation
    """
    batch_size, n_slots, _ = pred_positions.shape

    # Get position means [B, K]
    pos_mean = pred_positions[:, :, 0]

    # Get occupancy [B, K]
    occ = pred_occupancy.squeeze(-1)

    # Create mask for active slots [B, K]
    active_mask = occ > occupancy_threshold

    total_loss = torch.tensor(0.0, device=pred_positions.device)
    count = 0

    for b in range(batch_size):
        active_idx = torch.where(active_mask[b])[0]
        n_active = len(active_idx)

        if n_active < 2:
            continue

        # Get positions of active slots
        active_pos = pos_mean[b, active_idx]  # [n_active]

        # Compute pairwise distances
        # [n_active, 1] - [1, n_active] = [n_active, n_active]
        distances = torch.abs(active_pos.unsqueeze(1) - active_pos.unsqueeze(0))

        # Mask diagonal (self-distance)
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask.fill_diagonal_(False)

        # Penalize distances below threshold
        # penalty = max(0, min_distance - distance)
        violations = torch.relu(min_distance - distances[mask])

        if violations.numel() > 0:
            total_loss = total_loss + violations.mean()
            count += 1

    if count > 0:
        return total_loss / count
    return total_loss


class CurriculumTrainerV2:
    """V2 Trainer with overlap separation loss."""

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
        self.setup_logging()
        self.all_results = {}

    def setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "curriculum_training_v2.log"

        self.logger = logging.getLogger("CurriculumTrainerV2")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(ch)

    def log(self, msg: str):
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
        loss_fn: BEACONLoss,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss with V2 overlap separation."""
        sequences, profiles, positions, tf_indices, occupancies = batch
        sequences = sequences.to(self.device)
        profiles = profiles.to(self.device)
        positions = positions.to(self.device)
        tf_indices = tf_indices.to(self.device)
        occupancies = occupancies.to(self.device)

        # Forward pass
        outputs = model(sequences)

        # Prepare targets for BEACONLoss
        targets = {
            "profile": profiles,
            "positions": positions,
            "tf_ids": tf_indices,
            "occupancy": occupancies,
        }

        # Compute main loss using BEACONLoss
        loss_dict = loss_fn(outputs, targets)
        total_loss = loss_dict["total"]

        # V2: Add overlap separation loss
        if config.use_overlap_loss:
            sep_loss = overlap_separation_loss(
                outputs["positions"],
                outputs["occupancy"],
                min_distance=config.min_slot_distance,
            )
            total_loss = total_loss + config.overlap_loss_weight * sep_loss
            loss_dict["overlap_separation"] = sep_loss.item()

        return total_loss, {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: ExperimentConfig,
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()

        metrics_calc = BEACONMetrics(
            seq_len=config.seq_length,
            n_tfs=config.n_tfs,
            occupancy_threshold=0.5,
        )

        for batch in dataloader:
            sequences, profiles, positions, tf_indices, occupancies = batch
            sequences = sequences.to(self.device)

            outputs = model(sequences)

            targets = {
                "profile": profiles,
                "positions": positions,
                "tf_ids": tf_indices,
                "occupancy": occupancies,
            }

            metrics_calc.update(outputs, targets)

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
        self.log(f"EXPERIMENT: {config.name} (V2 - with overlap separation)")
        self.log(f"Description: {config.description}")
        self.log(f"Config: {config.n_tfs} TFs, {config.min_sites}-{config.max_sites} sites, "
                f"overlap={config.allow_overlap}, noise={config.noise_sigma}")
        self.log(f"V2 Settings: overlap_loss={config.use_overlap_loss}, "
                f"weight={config.overlap_loss_weight}, min_dist={config.min_slot_distance}")
        self.log("=" * 70)

        with open(exp_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        model = self.create_model(config)
        train_loader, val_loader, test_loader = self.create_dataloaders(config, seed)

        # V2: Use BEACONLoss instead of simple MSE
        loss_fn = BEACONLoss(
            n_tfs=config.n_tfs,
            seq_len=config.seq_length,
            profile_weight=1.0,
            position_weight=5.0,
            tf_weight=2.0,
            occupancy_weight=1.0,
            diversity_weight=0.1,
            orthogonality_weight=0.1,
        )

        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        scaler = GradScaler()

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
                    loss, loss_dict = self.compute_loss(model, batch, config, loss_fn)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss_dict)

                if batch_idx % 50 == 0:
                    sep_loss_str = f", sep={loss_dict.get('overlap_separation', 0):.4f}" if config.use_overlap_loss else ""
                    self.log(f"  Epoch {epoch+1}/{config.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                            f"Loss: {loss_dict['total']:.4f}{sep_loss_str}")

            scheduler.step()

            avg_loss = np.mean([l["total"] for l in epoch_losses])
            history["train_loss"].append(avg_loss)

            val_metrics = self.evaluate(model, val_loader, config)
            history["val_metrics"].append(val_metrics)

            self.log(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_loss:.4f} | "
                    f"Val Site F1: {val_metrics.get('site_f1', 0):.4f} | "
                    f"Val TF Acc: {val_metrics.get('tf_accuracy', 0):.4f} | "
                    f"Val Profile r: {val_metrics.get('profile_pearson', 0):.4f}")

            if val_metrics.get("site_f1", 0) > best_val_f1:
                best_val_f1 = val_metrics.get("site_f1", 0)
                best_epoch = epoch
                torch.save(model.state_dict(), exp_dir / "best_model.pt")

        model.load_state_dict(torch.load(exp_dir / "best_model.pt"))
        test_metrics = self.evaluate(model, test_loader, config)

        elapsed = time.time() - start_time

        results = {
            "config": asdict(config),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "training_time_seconds": elapsed,
            "test_metrics": test_metrics,
            "history": history,
        }

        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

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
        self.log("CURRICULUM TRAINING V2 - WITH OVERLAP SEPARATION LOSS")
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

        self.save_summary()

    def save_summary(self):
        """Save summary of all experiments."""
        summary_path = self.output_dir / "summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "version": "V2 - with overlap separation loss",
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

        self.log("\n" + "=" * 90)
        self.log("SUMMARY TABLE (V2)")
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
        description="Curriculum Trainer V2 - With Overlap Separation Loss",
    )

    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/curriculum_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
                        help="Output directory")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=None,
                        help="Specific experiments to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for testing")

    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    device = f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu"

    if args.quick:
        for exp in EXPERIMENTS.values():
            exp.n_train = 1000
            exp.n_val = 200
            exp.n_test = 200
            exp.epochs = 5

    trainer = CurriculumTrainerV2(
        output_dir=args.output_dir,
        device=device,
        gpus=gpus,
    )

    if args.experiments:
        experiments = {k: EXPERIMENTS[k] for k in args.experiments if k in EXPERIMENTS}
    else:
        experiments = EXPERIMENTS

    trainer.run_all_experiments(experiments)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
