#!/usr/bin/env python3
"""
Validate Multi-TF Co-binding Detection (A1) and Spatial Position Prediction (A2).

Tests zero-shot multi-TF decomposition using the Phase 10 model (trained on
single-TF data from multi_tf_k562) evaluated on beacon_multi test data
(which contains multiple TFs per sequence at non-centered positions).

A1 -- Multi-TF Co-binding Detection:
  - Load Phase 10 best model, run on beacon_multi test set
  - Filter active slots by occupancy > 0.3
  - Hungarian-match active slots to ground truth binding_sites
  - Stratify by number of ground truth sites (1, 2, 3, 4+)
  - Metrics: multi-site recall, per-site TF accuracy, position MAE, profile Pearson r

A2 -- Spatial Position Prediction:
  - Extract position MAE on non-centered sites
  - Create synthetic shifted test data by rolling sequences to place peaks
    at fractional positions 0.2, 0.3, 0.7, 0.8
  - Position MAE at various true positions and position bias analysis

Usage:
    python scripts/validate_multi_tf_cobinding.py
    python scripts/validate_multi_tf_cobinding.py --device cuda:0
    python scripts/validate_multi_tf_cobinding.py --n-synthetic 500 --batch-size 16
"""

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]
SEQ_LEN = 2000
N_SLOTS = 16


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path, device):
    """Instantiate and load Phase 10 best model."""
    model = BEACON(
        seq_len=SEQ_LEN,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=N_SLOTS,
        slot_dim=128,
        n_iterations=3,
        n_tfs=7,
        position_mode="gaussian",
        dropout=0.0,
        attention_mode="independent",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Strip DataParallel prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Hungarian matching utilities
# ============================================================

def extract_gt_sites(binding_sites_row, occ_threshold=0.1):
    """Return list of (norm_pos, tf_id, occupancy) from a [K, 3] ground truth row."""
    sites = []
    for s in range(binding_sites_row.shape[0]):
        pos, tf_id, occ = binding_sites_row[s]
        if occ > occ_threshold:
            sites.append((float(pos), int(tf_id), float(occ)))
    return sites


def hungarian_match_sample(pred_positions, pred_tf, pred_occ, gt_sites,
                           occ_threshold=0.3, seq_len=2000):
    """Match predicted slots to ground truth sites via Hungarian algorithm.

    Parameters
    ----------
    pred_positions : ndarray [K, 2]  (mean, std) normalised to [0,1]
    pred_tf : ndarray [K]  argmax TF id per slot
    pred_occ : ndarray [K]  occupancy per slot
    gt_sites : list of (norm_pos, tf_id, occupancy)

    Returns
    -------
    matches : list of dicts with per-match details
    n_gt : number of ground truth sites
    """
    n_gt = len(gt_sites)
    if n_gt == 0:
        return [], 0

    active_indices = np.where(pred_occ > occ_threshold)[0]
    n_pred = len(active_indices)
    if n_pred == 0:
        return [], n_gt

    # Cost matrix [n_gt, n_pred]
    cost = np.zeros((n_gt, n_pred))
    for i, (gt_pos, gt_tf, _) in enumerate(gt_sites):
        for j, pi in enumerate(active_indices):
            pred_pos = float(pred_positions[pi, 0])
            pos_dist_bp = abs(gt_pos * seq_len - pred_pos * seq_len)
            tf_match_bonus = 500.0 if int(pred_tf[pi]) == gt_tf else 0.0
            cost[i, j] = pos_dist_bp - tf_match_bonus

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(row_ind, col_ind):
        gt_pos, gt_tf, gt_occ = gt_sites[i]
        pi = active_indices[j]
        pred_pos = float(pred_positions[pi, 0])
        pos_dist_bp = abs(gt_pos * seq_len - pred_pos * seq_len)
        matches.append({
            "gt_pos": gt_pos,
            "gt_tf": gt_tf,
            "gt_occ": gt_occ,
            "pred_pos": pred_pos,
            "pred_tf": int(pred_tf[pi]),
            "pred_occ": float(pred_occ[pi]),
            "pos_error_bp": pos_dist_bp,
            "tf_correct": int(pred_tf[pi]) == gt_tf,
            "matched_within_200bp": pos_dist_bp < 200,
        })
    return matches, n_gt


# ============================================================
# A1  --  Multi-TF Co-binding Detection
# ============================================================

def run_a1_cobinding(model, loader, device):
    """Evaluate multi-TF co-binding detection on beacon_multi test set."""
    print("\n" + "=" * 70)
    print("A1: Multi-TF Co-binding Detection")
    print("=" * 70)

    # Accumulators stratified by n_gt_sites
    strata = defaultdict(lambda: {
        "n_samples": 0,
        "n_gt_sites": 0,
        "n_detected": 0,        # matched within 200 bp
        "n_tf_correct": 0,      # correct TF among detected
        "pos_errors_bp": [],    # position errors of detected matches
        "profile_r": [],
    })

    all_matches = []  # flat list for global stats

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequence"].to(device)
            gt_binding = batch["binding_sites"].numpy()   # [B, K, 3]
            gt_profiles = batch["profile"].numpy()        # [B, L]

            outputs = model(sequences)
            pred_occ = outputs["occupancy"].squeeze(-1).cpu().numpy()
            pred_tf_logits = outputs["tf_logits"].cpu().numpy()
            pred_positions = outputs["positions"].cpu().numpy()
            pred_profiles = outputs["profile"].cpu().numpy()

            pred_tf = pred_tf_logits.argmax(axis=-1)
            batch_size = sequences.shape[0]

            for b in range(batch_size):
                gt_sites = extract_gt_sites(gt_binding[b])
                n_gt = len(gt_sites)
                if n_gt == 0:
                    continue

                # Stratum label
                if n_gt == 1:
                    stratum = "1"
                elif n_gt == 2:
                    stratum = "2"
                elif n_gt == 3:
                    stratum = "3"
                else:
                    stratum = "4+"

                matches, _ = hungarian_match_sample(
                    pred_positions[b], pred_tf[b], pred_occ[b], gt_sites
                )

                st = strata[stratum]
                st["n_samples"] += 1
                st["n_gt_sites"] += n_gt

                for m in matches:
                    all_matches.append(m)
                    if m["matched_within_200bp"]:
                        st["n_detected"] += 1
                        st["pos_errors_bp"].append(m["pos_error_bp"])
                        if m["tf_correct"]:
                            st["n_tf_correct"] += 1

                # Profile correlation
                pp = pred_profiles[b]
                gp = gt_profiles[b]
                if np.std(pp) > 0 and np.std(gp) > 0:
                    r = float(np.corrcoef(pp, gp)[0, 1])
                    if not np.isnan(r):
                        st["profile_r"].append(r)

    # Print summary table
    print(f"\n{'Stratum':<10} {'N Samp':<8} {'GT Sites':<10} {'Recall%':<10} "
          f"{'TF Acc%':<10} {'Pos MAE':<10} {'Profile r'}")
    print("-" * 75)

    results = {}
    for stratum in ["1", "2", "3", "4+"]:
        st = strata[stratum]
        if st["n_samples"] == 0:
            continue
        recall = st["n_detected"] / max(st["n_gt_sites"], 1) * 100
        tf_acc = st["n_tf_correct"] / max(st["n_detected"], 1) * 100
        pos_mae = float(np.mean(st["pos_errors_bp"])) if st["pos_errors_bp"] else float("nan")
        prof_r = float(np.mean(st["profile_r"])) if st["profile_r"] else float("nan")

        label = f"{stratum} site{'s' if stratum != '1' else ''}"
        print(f"  {label:<8} {st['n_samples']:<8} {st['n_gt_sites']:<10} "
              f"{recall:<10.1f} {tf_acc:<10.1f} {pos_mae:<10.1f} {prof_r:.4f}")

        results[f"stratum_{stratum}"] = {
            "n_samples": st["n_samples"],
            "n_gt_sites": st["n_gt_sites"],
            "n_detected": st["n_detected"],
            "recall_pct": float(recall),
            "tf_accuracy_pct": float(tf_acc),
            "position_mae_bp": float(pos_mae) if not np.isnan(pos_mae) else None,
            "profile_pearson_r": float(prof_r) if not np.isnan(prof_r) else None,
        }

    # Overall
    total_gt = sum(st["n_gt_sites"] for st in strata.values())
    total_det = sum(st["n_detected"] for st in strata.values())
    total_tf_correct = sum(st["n_tf_correct"] for st in strata.values())
    all_pos_errors = [e for st in strata.values() for e in st["pos_errors_bp"]]
    all_prof_r = [r for st in strata.values() for r in st["profile_r"]]

    overall_recall = total_det / max(total_gt, 1) * 100
    overall_tf_acc = total_tf_correct / max(total_det, 1) * 100
    overall_mae = float(np.mean(all_pos_errors)) if all_pos_errors else float("nan")
    overall_prof_r = float(np.mean(all_prof_r)) if all_prof_r else float("nan")

    print(f"  {'Overall':<8} {sum(st['n_samples'] for st in strata.values()):<8} "
          f"{total_gt:<10} {overall_recall:<10.1f} {overall_tf_acc:<10.1f} "
          f"{overall_mae:<10.1f} {overall_prof_r:.4f}")

    results["overall"] = {
        "n_gt_sites": total_gt,
        "n_detected": total_det,
        "recall_pct": float(overall_recall),
        "tf_accuracy_pct": float(overall_tf_acc),
        "position_mae_bp": float(overall_mae) if not np.isnan(overall_mae) else None,
        "profile_pearson_r": float(overall_prof_r) if not np.isnan(overall_prof_r) else None,
    }

    return results, all_matches


# ============================================================
# A2  --  Spatial Position Prediction  (natural non-centred)
# ============================================================

def run_a2_position_analysis(all_matches):
    """Analyse position prediction accuracy on non-centred sites."""
    print("\n" + "=" * 70)
    print("A2: Spatial Position Prediction (natural non-centred sites)")
    print("=" * 70)

    if not all_matches:
        print("  No matches available.")
        return {}

    detected = [m for m in all_matches if m["matched_within_200bp"]]
    if not detected:
        print("  No detected matches.")
        return {}

    gt_positions = np.array([m["gt_pos"] for m in detected])
    pred_positions = np.array([m["pred_pos"] for m in detected])
    pos_errors = np.array([m["pos_error_bp"] for m in detected])

    # Identify centred vs non-centred  (centred = within 10% of 0.5)
    centred_mask = np.abs(gt_positions - 0.5) < 0.10
    non_centred_mask = ~centred_mask

    print(f"\n  Total detected matches: {len(detected)}")
    print(f"  Centred (|pos - 0.5| < 0.1): {centred_mask.sum()}")
    print(f"  Non-centred:                 {non_centred_mask.sum()}")

    results = {}

    for label, mask in [("centred", centred_mask), ("non_centred", non_centred_mask),
                         ("all", np.ones(len(detected), dtype=bool))]:
        if mask.sum() == 0:
            continue
        mae = float(np.mean(pos_errors[mask]))
        median_ae = float(np.median(pos_errors[mask]))
        std = float(np.std(pos_errors[mask]))
        print(f"\n  {label} (n={mask.sum()}):")
        print(f"    Position MAE  : {mae:.1f} bp")
        print(f"    Position MedAE: {median_ae:.1f} bp")
        print(f"    Position Std  : {std:.1f} bp")

        results[label] = {
            "n": int(mask.sum()),
            "position_mae_bp": mae,
            "position_median_ae_bp": median_ae,
            "position_std_bp": std,
        }

    # Binned accuracy by ground truth position
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print(f"\n  Position MAE by ground truth region:")
    print(f"  {'Region':<12} {'N':<8} {'MAE (bp)':<12} {'MedAE (bp)'}")
    print("  " + "-" * 44)

    bin_results = {}
    for lo, hi in bins:
        mask = (gt_positions >= lo) & (gt_positions < hi)
        if mask.sum() == 0:
            continue
        mae = float(np.mean(pos_errors[mask]))
        median_ae = float(np.median(pos_errors[mask]))
        label = f"[{lo:.1f},{hi:.1f})"
        print(f"  {label:<12} {mask.sum():<8} {mae:<12.1f} {median_ae:.1f}")
        bin_results[label] = {
            "n": int(mask.sum()),
            "mae_bp": mae,
            "median_ae_bp": median_ae,
        }
    results["by_region"] = bin_results

    # Position bias: mean signed error  (pred - gt)
    signed_errors = (pred_positions - gt_positions) * SEQ_LEN
    overall_bias = float(np.mean(signed_errors))
    print(f"\n  Position bias (mean signed error): {overall_bias:+.1f} bp")
    results["position_bias_bp"] = overall_bias

    return results


# ============================================================
# A2 (synthetic) -- Shifted synthetic data
# ============================================================

class ShiftedSyntheticDataset(Dataset):
    """Create shifted copies of multi_tf_k562 test data.

    For each source sample we roll the sequence+profile so that the dominant
    peak lands at a specified fractional position (0.2, 0.3, 0.7, 0.8).
    Binding sites are adjusted accordingly.
    """

    def __init__(self, source_dataset, target_positions, max_per_position=200):
        self.samples = []
        n_src = min(len(source_dataset), max_per_position)

        for target_frac in target_positions:
            count = 0
            for idx in range(len(source_dataset)):
                if count >= max_per_position:
                    break
                item = source_dataset[idx]
                bs = item["binding_sites"].numpy()  # [K, 3]
                valid = bs[:, 2] > 0.1
                if valid.sum() == 0:
                    continue

                # Find dominant site (highest occupancy)
                valid_idx = np.where(valid)[0]
                dom_idx = valid_idx[np.argmax(bs[valid_idx, 2])]
                dom_pos = bs[dom_idx, 0]  # normalised [0,1]

                shift_frac = target_frac - dom_pos
                shift_bp = int(round(shift_frac * SEQ_LEN))

                if abs(shift_bp) >= SEQ_LEN:
                    continue

                seq = item["sequence"].numpy()       # [L, 4]
                profile = item["profile"].numpy()     # [L]
                new_bs = bs.copy()

                # Roll sequence and profile
                rolled_seq = np.roll(seq, shift_bp, axis=0)
                rolled_profile = np.roll(profile, shift_bp, axis=0)

                # Zero out the wrapped-around region to avoid artefacts
                if shift_bp > 0:
                    rolled_seq[:shift_bp] = 0.25  # uniform
                    rolled_profile[:shift_bp] = 0.0
                elif shift_bp < 0:
                    rolled_seq[shift_bp:] = 0.25
                    rolled_profile[shift_bp:] = 0.0

                # Shift binding site positions
                for s in range(new_bs.shape[0]):
                    if new_bs[s, 2] > 0.1:
                        new_pos = new_bs[s, 0] + shift_frac
                        if new_pos < 0 or new_pos > 1:
                            new_bs[s, 2] = 0.0  # invalidate out-of-bounds
                        else:
                            new_bs[s, 0] = new_pos

                self.samples.append({
                    "sequence": torch.from_numpy(rolled_seq).float(),
                    "profile": torch.from_numpy(rolled_profile).float(),
                    "binding_sites": torch.from_numpy(new_bs).float(),
                    "tf_index": item["tf_index"],
                    "target_frac": target_frac,
                })
                count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_a2_synthetic(model, source_dataset, device, target_positions,
                     max_per_position, batch_size):
    """Evaluate position prediction on synthetically shifted data."""
    print("\n" + "=" * 70)
    print("A2 (synthetic): Position Prediction on Shifted Data")
    print("=" * 70)

    synth = ShiftedSyntheticDataset(source_dataset, target_positions,
                                    max_per_position=max_per_position)
    if len(synth) == 0:
        print("  No synthetic samples generated.")
        return {}

    loader = DataLoader(synth, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Synthetic samples: {len(synth)}")

    # Run inference and match
    pos_results = defaultdict(lambda: {
        "n": 0, "pos_errors": [], "tf_correct": 0, "n_detected": 0,
        "signed_errors": [],
    })

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequence"].to(device)
            gt_binding = batch["binding_sites"].numpy()
            target_fracs = batch["target_frac"].numpy()

            outputs = model(sequences)
            pred_occ = outputs["occupancy"].squeeze(-1).cpu().numpy()
            pred_tf_logits = outputs["tf_logits"].cpu().numpy()
            pred_positions = outputs["positions"].cpu().numpy()
            pred_tf = pred_tf_logits.argmax(axis=-1)

            for b in range(sequences.shape[0]):
                gt_sites = extract_gt_sites(gt_binding[b])
                if not gt_sites:
                    continue

                matches, _ = hungarian_match_sample(
                    pred_positions[b], pred_tf[b], pred_occ[b], gt_sites
                )

                tfrac = float(target_fracs[b])
                key = f"{tfrac:.1f}"
                pr = pos_results[key]
                pr["n"] += len(gt_sites)

                for m in matches:
                    if m["matched_within_200bp"]:
                        pr["n_detected"] += 1
                        pr["pos_errors"].append(m["pos_error_bp"])
                        pr["signed_errors"].append(
                            (m["pred_pos"] - m["gt_pos"]) * SEQ_LEN
                        )
                        if m["tf_correct"]:
                            pr["tf_correct"] += 1

    # Print table
    print(f"\n  {'Target Pos':<12} {'N GT':<8} {'Recall%':<10} {'MAE (bp)':<10} "
          f"{'Bias (bp)':<10} {'TF Acc%'}")
    print("  " + "-" * 60)

    results = {}
    for key in sorted(pos_results.keys()):
        pr = pos_results[key]
        recall = pr["n_detected"] / max(pr["n"], 1) * 100
        mae = float(np.mean(pr["pos_errors"])) if pr["pos_errors"] else float("nan")
        bias = float(np.mean(pr["signed_errors"])) if pr["signed_errors"] else float("nan")
        tf_acc = pr["tf_correct"] / max(pr["n_detected"], 1) * 100

        print(f"  {key:<12} {pr['n']:<8} {recall:<10.1f} "
              f"{mae:<10.1f} {bias:<+10.1f} {tf_acc:.1f}")

        results[key] = {
            "n_gt": pr["n"],
            "n_detected": pr["n_detected"],
            "recall_pct": float(recall),
            "position_mae_bp": float(mae) if not np.isnan(mae) else None,
            "position_bias_bp": float(bias) if not np.isnan(bias) else None,
            "tf_accuracy_pct": float(tf_acc),
        }

    return results


# ============================================================
# Visualization
# ============================================================

def create_figures(a1_results, a2_natural, a2_synthetic, all_matches, output_dir):
    """Generate summary plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # -----------------------------------------------------------
    # Panel 1: Recall by stratum
    # -----------------------------------------------------------
    ax = axes[0, 0]
    strata_labels = []
    recalls = []
    for s in ["1", "2", "3", "4+"]:
        key = f"stratum_{s}"
        if key in a1_results:
            strata_labels.append(f"{s} site{'s' if s != '1' else ''}")
            recalls.append(a1_results[key]["recall_pct"])
    if strata_labels:
        bars = ax.bar(strata_labels, recalls, color="steelblue", alpha=0.85,
                      edgecolor="black")
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Multi-site Recall (%)")
    ax.set_title("A1: Recall by GT Complexity")
    ax.set_ylim(0, 105)

    # -----------------------------------------------------------
    # Panel 2: TF accuracy by stratum
    # -----------------------------------------------------------
    ax = axes[0, 1]
    strata_labels2 = []
    tf_accs = []
    for s in ["1", "2", "3", "4+"]:
        key = f"stratum_{s}"
        if key in a1_results:
            strata_labels2.append(f"{s}")
            tf_accs.append(a1_results[key]["tf_accuracy_pct"])
    if strata_labels2:
        bars = ax.bar(strata_labels2, tf_accs, color="coral", alpha=0.85,
                      edgecolor="black")
        for bar, val in zip(bars, tf_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=9)
    ax.axhline(y=100 / 7, color="gray", linestyle="--", label="Chance (14.3%)")
    ax.set_ylabel("Per-site TF Accuracy (%)")
    ax.set_title("A1: TF ID Accuracy by Complexity")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # -----------------------------------------------------------
    # Panel 3: Position error histogram
    # -----------------------------------------------------------
    ax = axes[0, 2]
    detected = [m for m in all_matches if m["matched_within_200bp"]]
    if detected:
        errors = np.array([m["pos_error_bp"] for m in detected])
        ax.hist(errors, bins=40, color="mediumseagreen", alpha=0.85,
                edgecolor="black", range=(0, 200))
        ax.axvline(np.mean(errors), color="red", linestyle="--",
                   label=f"Mean={np.mean(errors):.1f} bp")
        ax.set_xlabel("Position Error (bp)")
        ax.set_ylabel("Count")
        ax.set_title("A1: Position Error Distribution")
        ax.legend(fontsize=8)

    # -----------------------------------------------------------
    # Panel 4: Position MAE by GT region
    # -----------------------------------------------------------
    ax = axes[1, 0]
    if "by_region" in a2_natural:
        regions = []
        maes = []
        for label in ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]:
            if label in a2_natural["by_region"]:
                regions.append(label)
                maes.append(a2_natural["by_region"][label]["mae_bp"])
        if regions:
            bars = ax.bar(regions, maes, color="mediumpurple", alpha=0.85,
                          edgecolor="black")
            for bar, val in zip(bars, maes):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", fontsize=8)
            ax.set_ylabel("Position MAE (bp)")
            ax.set_title("A2: Position MAE by GT Region")
            ax.tick_params(axis="x", rotation=30)

    # -----------------------------------------------------------
    # Panel 5: Synthetic shift - MAE vs target position
    # -----------------------------------------------------------
    ax = axes[1, 1]
    if a2_synthetic:
        syn_labels = sorted(a2_synthetic.keys())
        syn_maes = []
        syn_biases = []
        for k in syn_labels:
            mae_val = a2_synthetic[k].get("position_mae_bp")
            bias_val = a2_synthetic[k].get("position_bias_bp")
            syn_maes.append(mae_val if mae_val is not None else 0)
            syn_biases.append(bias_val if bias_val is not None else 0)
        x = np.arange(len(syn_labels))
        width = 0.35
        ax.bar(x - width / 2, syn_maes, width, label="MAE (bp)", color="steelblue",
               alpha=0.85, edgecolor="black")
        ax.bar(x + width / 2, [abs(b) for b in syn_biases], width, label="|Bias| (bp)",
               color="salmon", alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(syn_labels)
        ax.set_xlabel("Target Fractional Position")
        ax.set_ylabel("bp")
        ax.set_title("A2 Synthetic: MAE & Bias vs Target Position")
        ax.legend(fontsize=8)

    # -----------------------------------------------------------
    # Panel 6: Predicted vs GT position scatter
    # -----------------------------------------------------------
    ax = axes[1, 2]
    if detected:
        gt_pos = np.array([m["gt_pos"] for m in detected]) * SEQ_LEN
        pred_pos = np.array([m["pred_pos"] for m in detected]) * SEQ_LEN
        ax.scatter(gt_pos, pred_pos, alpha=0.15, s=6, c="steelblue")
        ax.plot([0, SEQ_LEN], [0, SEQ_LEN], "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("Ground Truth Position (bp)")
        ax.set_ylabel("Predicted Position (bp)")
        ax.set_title("A2: Predicted vs GT Position")
        ax.legend(fontsize=8)
        ax.set_xlim(0, SEQ_LEN)
        ax.set_ylim(0, SEQ_LEN)

    plt.tight_layout()
    fig_path = output_dir / "validate_multi_tf_cobinding.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate Multi-TF Co-binding (A1) and Spatial Position (A2)"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/phase10_slot_ablation/slots_16/"
                     "beacon_20260203_231900/best_model.pt"),
        help="Path to Phase 10 best model checkpoint",
    )
    parser.add_argument(
        "--multi-data", type=Path,
        default=Path("data/processed/beacon_multi/test.h5"),
        help="Path to beacon_multi test HDF5",
    )
    parser.add_argument(
        "--single-data", type=Path,
        default=Path("data/processed/multi_tf_k562/test.h5"),
        help="Path to multi_tf_k562 test HDF5 (used for synthetic shifts)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/validate_cobinding"),
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--n-synthetic", type=int, default=300,
        help="Max synthetic samples per target position",
    )
    parser.add_argument(
        "--synthetic-positions", type=float, nargs="+",
        default=[0.2, 0.3, 0.7, 0.8],
        help="Target fractional positions for synthetic shift test",
    )

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    # Resolve relative paths
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else base_dir / args.checkpoint
    multi_data_path = args.multi_data if args.multi_data.is_absolute() else base_dir / args.multi_data
    single_data_path = args.single_data if args.single_data.is_absolute() else base_dir / args.single_data
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multi-TF Co-binding & Spatial Position Validation")
    print("=" * 70)
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Multi data  : {multi_data_path}")
    print(f"  Single data : {single_data_path}")
    print(f"  Output      : {output_dir}")
    print(f"  Device      : {args.device}")

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading model...")
    model = load_model(checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Load multi-TF test data
    # ------------------------------------------------------------------
    print("\nLoading beacon_multi test data...")
    multi_dataset = BEACONDataset(
        str(multi_data_path), seq_length=SEQ_LEN, augment=False,
        n_slots=N_SLOTS, extract_peaks=False,
    )
    multi_loader = DataLoader(
        multi_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(f"  Samples: {len(multi_dataset)}")

    # ------------------------------------------------------------------
    # A1: Co-binding detection
    # ------------------------------------------------------------------
    a1_results, all_matches = run_a1_cobinding(model, multi_loader, device)

    # ------------------------------------------------------------------
    # A2 (natural): Position analysis on natural non-centred sites
    # ------------------------------------------------------------------
    a2_natural = run_a2_position_analysis(all_matches)

    # ------------------------------------------------------------------
    # A2 (synthetic): Shifted data from multi_tf_k562
    # ------------------------------------------------------------------
    a2_synthetic = {}
    if single_data_path.exists():
        print("\nLoading multi_tf_k562 test data for synthetic shifts...")
        single_dataset = BEACONDataset(
            str(single_data_path), seq_length=SEQ_LEN, augment=False,
            n_slots=N_SLOTS, extract_peaks=True,
        )
        print(f"  Source samples: {len(single_dataset)}")
        a2_synthetic = run_a2_synthetic(
            model, single_dataset, device,
            target_positions=args.synthetic_positions,
            max_per_position=args.n_synthetic,
            batch_size=args.batch_size,
        )
    else:
        print(f"\n  Skipping synthetic A2: {single_data_path} not found")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    create_figures(a1_results, a2_natural, a2_synthetic, all_matches, output_dir)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    combined = {
        "a1_cobinding": a1_results,
        "a2_position_natural": a2_natural,
        "a2_position_synthetic": a2_synthetic,
        "config": {
            "checkpoint": str(checkpoint_path),
            "multi_data": str(multi_data_path),
            "single_data": str(single_data_path),
            "device": args.device,
            "batch_size": args.batch_size,
            "n_synthetic": args.n_synthetic,
            "synthetic_positions": args.synthetic_positions,
        },
    }

    results_path = output_dir / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "overall" in a1_results:
        ov = a1_results["overall"]
        print(f"\n  A1 Multi-TF Co-binding Detection:")
        print(f"    Overall recall         : {ov['recall_pct']:.1f}%")
        print(f"    Overall TF accuracy    : {ov['tf_accuracy_pct']:.1f}%")
        mae_str = f"{ov['position_mae_bp']:.1f}" if ov["position_mae_bp"] is not None else "N/A"
        print(f"    Overall position MAE   : {mae_str} bp")
        prof_str = f"{ov['profile_pearson_r']:.4f}" if ov["profile_pearson_r"] is not None else "N/A"
        print(f"    Overall profile r      : {prof_str}")

    if a2_natural:
        print(f"\n  A2 Spatial Position (natural):")
        for key in ["centred", "non_centred", "all"]:
            if key in a2_natural:
                d = a2_natural[key]
                print(f"    {key:<14} MAE={d['position_mae_bp']:.1f} bp  "
                      f"(n={d['n']})")
        if "position_bias_bp" in a2_natural:
            print(f"    Global bias  : {a2_natural['position_bias_bp']:+.1f} bp")

    if a2_synthetic:
        print(f"\n  A2 Spatial Position (synthetic shifts):")
        for key in sorted(a2_synthetic.keys()):
            d = a2_synthetic[key]
            mae_s = f"{d['position_mae_bp']:.1f}" if d["position_mae_bp"] is not None else "N/A"
            bias_s = f"{d['position_bias_bp']:+.1f}" if d["position_bias_bp"] is not None else "N/A"
            print(f"    pos={key}  MAE={mae_s} bp  bias={bias_s} bp  "
                  f"recall={d['recall_pct']:.1f}%")

    print(f"\n  Results saved to: {results_path}")
    print(f"  Figure  saved to: {output_dir / 'validate_multi_tf_cobinding.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
