#!/usr/bin/env python3
"""
A3: Cross-Cell-Type Transfer Validation

Validates that BEACON trained on K562 generalizes to HepG2 cell line.
Uses the best Phase 10 model (16 slots) and evaluates on HepG2 test data
with proper TF index remapping and Hungarian-matched binding site evaluation.

Metrics:
- TF classification accuracy (with HepG2->K562 index remapping)
- Profile Pearson correlation per shared TF
- Binding site detection rate
- Transfer efficiency relative to K562 baseline
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset
from beacon.data.peaks import OnlinePeakExtractor, create_binding_sites_tensor


# ---------------------------------------------------------------------------
# TF vocabulary (K562 model was trained with this ordering)
# ---------------------------------------------------------------------------
K562_TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


def build_tf_mapping(hepg2_tf_names):
    """Build HepG2-index -> K562-index mapping for shared TFs.

    HepG2 TFs (from config.json): CTCF=0, MYC=1, MAX=2, CEBPB=3
    K562 TFs:                      CTCF=0, MYC=3, MAX=4, CEBPB=6

    Returns:
        mapping: dict mapping HepG2 tf index -> K562 tf index
        shared_names: list of shared TF names in HepG2 index order
    """
    mapping = {}
    shared_names = []
    for hepg2_idx, tf_name in enumerate(hepg2_tf_names):
        if tf_name in K562_TF_NAMES:
            k562_idx = K562_TF_NAMES.index(tf_name)
            mapping[hepg2_idx] = k562_idx
            shared_names.append(tf_name)
    return mapping, shared_names


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_hepg2_with_binding_sites(data_path, seq_length=2000, n_slots=16):
    """Load HepG2 dataset and create binding_sites from profiles + tf_indices.

    The HepG2 h5 may not have a 'binding_sites' key, so we use
    OnlinePeakExtractor to generate them from profile data.
    """
    dataset = BEACONDataset(
        data_path,
        seq_length=seq_length,
        augment=False,
        n_slots=n_slots,
        extract_peaks=True,
    )
    return dataset


def load_k562_test(data_path, seq_length=2000, n_slots=16):
    """Load K562 test dataset."""
    dataset = BEACONDataset(
        data_path,
        seq_length=seq_length,
        augment=False,
        n_slots=n_slots,
        extract_peaks=True,
    )
    return dataset


# ---------------------------------------------------------------------------
# Evaluation with Hungarian matching and TF remapping
# ---------------------------------------------------------------------------

def evaluate_with_remapping(model, data_loader, device, tf_mapping,
                            hepg2_tf_names, seq_len=2000, n_slots=16):
    """Evaluate K562-trained model on HepG2 data with TF index remapping.

    For each sample:
    1. Run model forward pass
    2. Extract ground truth binding sites from the data
    3. Remap ground truth TF indices from HepG2 space to K562 space
    4. Hungarian-match predicted slots to ground truth sites
    5. Score TF accuracy and profile correlation per shared TF

    Returns dict with overall and per-TF metrics.
    """
    model.eval()

    # Accumulators
    total_sites = 0
    total_matched = 0
    total_correct_tf = 0

    per_tf_total = defaultdict(int)
    per_tf_matched = defaultdict(int)
    per_tf_correct = defaultdict(int)
    per_tf_profile_r = defaultdict(list)

    all_true_k562_tfs = []
    all_pred_tfs = []
    all_profile_r = []
    all_occupancies = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch["sequence"].to(device)
            binding_sites = batch["binding_sites"]  # [B, K, 3]
            profiles = batch["profile"]              # [B, L]
            tf_indices = batch["tf_index"]            # [B]

            outputs = model(sequences, return_attention=True, return_slots=True)
            pred_positions = outputs["positions"].cpu()    # [B, K, 2]
            pred_tf_logits = outputs["tf_logits"].cpu()    # [B, K, N_TFs]
            pred_occupancy = outputs["occupancy"].cpu()    # [B, K, 1]
            pred_profiles = outputs["profile"].cpu()       # [B, L]

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                hepg2_tf_idx = int(tf_indices[b].item())

                # Skip samples whose TF is not shared with K562
                if hepg2_tf_idx not in tf_mapping:
                    continue

                k562_tf_idx = tf_mapping[hepg2_tf_idx]
                tf_name = hepg2_tf_names[hepg2_tf_idx]

                # ---- Profile Pearson r ----
                pred_p = pred_profiles[b].numpy()
                tgt_p = profiles[b].numpy()
                if pred_p.std() > 0 and tgt_p.std() > 0:
                    r = np.corrcoef(pred_p.flatten(), tgt_p.flatten())[0, 1]
                    if not np.isnan(r):
                        all_profile_r.append(r)
                        per_tf_profile_r[tf_name].append(r)

                # ---- TF classification from highest-occupancy slot ----
                pred_occ = pred_occupancy[b].squeeze(-1) if pred_occupancy[b].dim() > 1 else pred_occupancy[b]
                best_slot = pred_occ.argmax().item()
                pred_tf = pred_tf_logits[b, best_slot].argmax().item()

                all_true_k562_tfs.append(k562_tf_idx)
                all_pred_tfs.append(pred_tf)
                all_occupancies.append(pred_occ.max().item())

                # ---- Hungarian-matched binding site evaluation ----
                gt_sites = []
                for s in range(binding_sites.shape[1]):
                    pos, tf_id, occ = binding_sites[b, s]
                    if occ > 0.5:
                        # Remap tf_id from HepG2 space to K562 space
                        gt_hepg2_tf = int(tf_id.item())
                        if gt_hepg2_tf in tf_mapping:
                            gt_k562_tf = tf_mapping[gt_hepg2_tf]
                            gt_sites.append((pos.item(), gt_k562_tf, occ.item()))

                if not gt_sites:
                    continue

                n_gt = len(gt_sites)
                total_sites += n_gt
                for _, gt_tf, _ in gt_sites:
                    per_tf_total[gt_tf] += 1

                # Active predicted slots (occupancy > 0.3)
                active_mask = pred_occ > 0.3
                active_indices = torch.where(active_mask)[0]

                if len(active_indices) == 0:
                    continue

                # Build cost matrix
                n_pred = len(active_indices)
                cost_matrix = np.zeros((n_gt, n_pred))

                for i, (gt_pos, gt_tf, _) in enumerate(gt_sites):
                    for j, pred_idx in enumerate(active_indices):
                        pos_val = pred_positions[b, pred_idx]
                        pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                        p_tf = pred_tf_logits[b, pred_idx].argmax().item()

                        pos_dist = abs(gt_pos * seq_len - pred_pos * seq_len)
                        tf_match = 1.0 if p_tf == gt_tf else 0.0
                        cost_matrix[i, j] = pos_dist - tf_match * 500

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    gt_pos, gt_tf, _ = gt_sites[i]
                    pred_idx = active_indices[j]
                    pos_val = pred_positions[b, pred_idx]
                    pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                    p_tf = pred_tf_logits[b, pred_idx].argmax().item()
                    pos_dist_bp = abs(gt_pos * seq_len - pred_pos * seq_len)

                    if pos_dist_bp < 200:
                        total_matched += 1
                        per_tf_matched[gt_tf] += 1
                        if p_tf == gt_tf:
                            total_correct_tf += 1
                            per_tf_correct[gt_tf] += 1

    # ---- TF accuracy from highest-occupancy slot ----
    slot_tf_accuracy = 0.0
    if len(all_true_k562_tfs) > 0:
        slot_tf_accuracy = sum(
            1 for t, p in zip(all_true_k562_tfs, all_pred_tfs) if t == p
        ) / len(all_true_k562_tfs)

    # ---- Compile per-TF results ----
    per_tf_results = {}
    for hepg2_idx, k562_idx in tf_mapping.items():
        tf_name = hepg2_tf_names[hepg2_idx]
        total = per_tf_total.get(k562_idx, 0)
        matched = per_tf_matched.get(k562_idx, 0)
        correct = per_tf_correct.get(k562_idx, 0)

        # Per-TF slot-level accuracy
        tf_true_count = sum(1 for t in all_true_k562_tfs if t == k562_idx)
        tf_slot_correct = sum(
            1 for t, p in zip(all_true_k562_tfs, all_pred_tfs)
            if t == k562_idx and t == p
        )
        tf_slot_acc = tf_slot_correct / tf_true_count if tf_true_count > 0 else 0.0

        profile_rs = per_tf_profile_r.get(tf_name, [])

        per_tf_results[tf_name] = {
            "k562_index": k562_idx,
            "hepg2_index": hepg2_idx,
            "total_sites": total,
            "matched_sites": matched,
            "correct_tf_sites": correct,
            "match_rate": matched / max(total, 1),
            "hungarian_tf_accuracy": correct / max(matched, 1),
            "slot_tf_accuracy": tf_slot_acc,
            "n_samples": tf_true_count,
            "mean_profile_r": float(np.mean(profile_rs)) if profile_rs else 0.0,
            "median_profile_r": float(np.median(profile_rs)) if profile_rs else 0.0,
        }

    results = {
        "overall": {
            "n_samples": len(all_true_k562_tfs),
            "slot_tf_accuracy": slot_tf_accuracy,
            "total_sites": total_sites,
            "total_matched": total_matched,
            "total_correct_tf": total_correct_tf,
            "match_rate": total_matched / max(total_sites, 1),
            "hungarian_tf_accuracy": total_correct_tf / max(total_matched, 1),
            "mean_profile_r": float(np.mean(all_profile_r)) if all_profile_r else 0.0,
            "median_profile_r": float(np.median(all_profile_r)) if all_profile_r else 0.0,
            "mean_occupancy": float(np.mean(all_occupancies)) if all_occupancies else 0.0,
            "site_detection_rate": float(np.mean([o > 0.5 for o in all_occupancies])) if all_occupancies else 0.0,
        },
        "per_tf": per_tf_results,
    }

    return results


def evaluate_k562_baseline(model, data_loader, device, seq_len=2000, n_slots=16):
    """Evaluate model on K562 test set (same cell line baseline).

    Uses the same Hungarian-matched evaluation and slot-level TF accuracy
    so metrics are directly comparable to HepG2 evaluation.
    """
    model.eval()

    total_sites = 0
    total_matched = 0
    total_correct_tf = 0

    per_tf_total = defaultdict(int)
    per_tf_matched = defaultdict(int)
    per_tf_correct = defaultdict(int)
    per_tf_profile_r = defaultdict(list)

    all_true_tfs = []
    all_pred_tfs = []
    all_profile_r = []
    all_occupancies = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch["sequence"].to(device)
            binding_sites = batch["binding_sites"]
            profiles = batch["profile"]
            tf_indices = batch["tf_index"]

            outputs = model(sequences, return_attention=True, return_slots=True)
            pred_positions = outputs["positions"].cpu()
            pred_tf_logits = outputs["tf_logits"].cpu()
            pred_occupancy = outputs["occupancy"].cpu()
            pred_profiles = outputs["profile"].cpu()

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                tf_idx = int(tf_indices[b].item())
                tf_name = K562_TF_NAMES[tf_idx]

                # Profile Pearson r
                pred_p = pred_profiles[b].numpy()
                tgt_p = profiles[b].numpy()
                if pred_p.std() > 0 and tgt_p.std() > 0:
                    r = np.corrcoef(pred_p.flatten(), tgt_p.flatten())[0, 1]
                    if not np.isnan(r):
                        all_profile_r.append(r)
                        per_tf_profile_r[tf_name].append(r)

                # TF classification from highest-occupancy slot
                pred_occ = pred_occupancy[b].squeeze(-1) if pred_occupancy[b].dim() > 1 else pred_occupancy[b]
                best_slot = pred_occ.argmax().item()
                pred_tf = pred_tf_logits[b, best_slot].argmax().item()

                all_true_tfs.append(tf_idx)
                all_pred_tfs.append(pred_tf)
                all_occupancies.append(pred_occ.max().item())

                # Hungarian-matched binding site evaluation
                gt_sites = []
                for s in range(binding_sites.shape[1]):
                    pos, gt_tf_id, occ = binding_sites[b, s]
                    if occ > 0.5:
                        gt_sites.append((pos.item(), int(gt_tf_id.item()), occ.item()))

                if not gt_sites:
                    continue

                n_gt = len(gt_sites)
                total_sites += n_gt
                for _, gt_tf, _ in gt_sites:
                    per_tf_total[gt_tf] += 1

                active_mask = pred_occ > 0.3
                active_indices = torch.where(active_mask)[0]

                if len(active_indices) == 0:
                    continue

                n_pred = len(active_indices)
                cost_matrix = np.zeros((n_gt, n_pred))

                for i, (gt_pos, gt_tf, _) in enumerate(gt_sites):
                    for j, pred_idx in enumerate(active_indices):
                        pos_val = pred_positions[b, pred_idx]
                        pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                        p_tf = pred_tf_logits[b, pred_idx].argmax().item()

                        pos_dist = abs(gt_pos * seq_len - pred_pos * seq_len)
                        tf_match = 1.0 if p_tf == gt_tf else 0.0
                        cost_matrix[i, j] = pos_dist - tf_match * 500

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    gt_pos, gt_tf, _ = gt_sites[i]
                    pred_idx = active_indices[j]
                    pos_val = pred_positions[b, pred_idx]
                    pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                    p_tf = pred_tf_logits[b, pred_idx].argmax().item()
                    pos_dist_bp = abs(gt_pos * seq_len - pred_pos * seq_len)

                    if pos_dist_bp < 200:
                        total_matched += 1
                        per_tf_matched[gt_tf] += 1
                        if p_tf == gt_tf:
                            total_correct_tf += 1
                            per_tf_correct[gt_tf] += 1

    # Slot-level TF accuracy
    slot_tf_accuracy = 0.0
    if len(all_true_tfs) > 0:
        slot_tf_accuracy = sum(
            1 for t, p in zip(all_true_tfs, all_pred_tfs) if t == p
        ) / len(all_true_tfs)

    per_tf_results = {}
    for tf_idx, tf_name in enumerate(K562_TF_NAMES):
        total = per_tf_total.get(tf_idx, 0)
        matched = per_tf_matched.get(tf_idx, 0)
        correct = per_tf_correct.get(tf_idx, 0)

        tf_true_count = sum(1 for t in all_true_tfs if t == tf_idx)
        tf_slot_correct = sum(
            1 for t, p in zip(all_true_tfs, all_pred_tfs)
            if t == tf_idx and t == p
        )
        tf_slot_acc = tf_slot_correct / tf_true_count if tf_true_count > 0 else 0.0

        profile_rs = per_tf_profile_r.get(tf_name, [])

        per_tf_results[tf_name] = {
            "k562_index": tf_idx,
            "total_sites": total,
            "matched_sites": matched,
            "correct_tf_sites": correct,
            "match_rate": matched / max(total, 1),
            "hungarian_tf_accuracy": correct / max(matched, 1),
            "slot_tf_accuracy": tf_slot_acc,
            "n_samples": tf_true_count,
            "mean_profile_r": float(np.mean(profile_rs)) if profile_rs else 0.0,
            "median_profile_r": float(np.median(profile_rs)) if profile_rs else 0.0,
        }

    results = {
        "overall": {
            "n_samples": len(all_true_tfs),
            "slot_tf_accuracy": slot_tf_accuracy,
            "total_sites": total_sites,
            "total_matched": total_matched,
            "total_correct_tf": total_correct_tf,
            "match_rate": total_matched / max(total_sites, 1),
            "hungarian_tf_accuracy": total_correct_tf / max(total_matched, 1),
            "mean_profile_r": float(np.mean(all_profile_r)) if all_profile_r else 0.0,
            "median_profile_r": float(np.median(all_profile_r)) if all_profile_r else 0.0,
            "mean_occupancy": float(np.mean(all_occupancies)) if all_occupancies else 0.0,
            "site_detection_rate": float(np.mean([o > 0.5 for o in all_occupancies])) if all_occupancies else 0.0,
        },
        "per_tf": per_tf_results,
    }

    return results


# ---------------------------------------------------------------------------
# Transfer efficiency computation
# ---------------------------------------------------------------------------

def compute_transfer_efficiency(k562_results, hepg2_results, shared_tf_names):
    """Compute transfer efficiency percentages: HepG2 / K562 * 100.

    Returns dict with overall and per-TF transfer percentages.
    """
    k562_o = k562_results["overall"]
    hepg2_o = hepg2_results["overall"]

    def safe_pct(hepg2_val, k562_val):
        if k562_val == 0:
            return 0.0
        return hepg2_val / k562_val * 100.0

    # For K562 baseline, restrict to shared TFs only for fair comparison
    k562_shared_profile_rs = []
    for tf_name in shared_tf_names:
        if tf_name in k562_results["per_tf"]:
            tf_r = k562_results["per_tf"][tf_name]["mean_profile_r"]
            n = k562_results["per_tf"][tf_name]["n_samples"]
            k562_shared_profile_rs.extend([tf_r] * n)
    k562_shared_mean_r = float(np.mean(k562_shared_profile_rs)) if k562_shared_profile_rs else k562_o["mean_profile_r"]

    overall = {
        "tf_accuracy": safe_pct(hepg2_o["slot_tf_accuracy"], k562_o["slot_tf_accuracy"]),
        "hungarian_tf_accuracy": safe_pct(hepg2_o["hungarian_tf_accuracy"], k562_o["hungarian_tf_accuracy"]),
        "match_rate": safe_pct(hepg2_o["match_rate"], k562_o["match_rate"]),
        "profile_r": safe_pct(hepg2_o["mean_profile_r"], k562_shared_mean_r),
        "site_detection": safe_pct(hepg2_o["site_detection_rate"], k562_o["site_detection_rate"]),
    }

    per_tf = {}
    for tf_name in shared_tf_names:
        k562_tf = k562_results["per_tf"].get(tf_name, {})
        hepg2_tf = hepg2_results["per_tf"].get(tf_name, {})

        per_tf[tf_name] = {
            "slot_tf_accuracy": safe_pct(
                hepg2_tf.get("slot_tf_accuracy", 0),
                k562_tf.get("slot_tf_accuracy", 0),
            ),
            "hungarian_tf_accuracy": safe_pct(
                hepg2_tf.get("hungarian_tf_accuracy", 0),
                k562_tf.get("hungarian_tf_accuracy", 0),
            ),
            "profile_r": safe_pct(
                hepg2_tf.get("mean_profile_r", 0),
                k562_tf.get("mean_profile_r", 0),
            ),
        }

    return {"overall": overall, "per_tf": per_tf}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_visualization(k562_results, hepg2_results, transfer, shared_tf_names,
                         output_path):
    """Create a 2x2 figure summarizing cross-cell-type transfer results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("A3: Cross-Cell-Type Transfer Validation (K562 -> HepG2)",
                 fontsize=14, fontweight="bold", y=0.98)

    k562_o = k562_results["overall"]
    hepg2_o = hepg2_results["overall"]

    # ---- Panel 1: Overall comparison bar chart ----
    ax = axes[0, 0]
    metric_labels = ["TF Accuracy\n(slot)", "TF Accuracy\n(Hungarian)", "Match Rate",
                     "Profile r", "Site Detection"]
    k562_vals = [
        k562_o["slot_tf_accuracy"] * 100,
        k562_o["hungarian_tf_accuracy"] * 100,
        k562_o["match_rate"] * 100,
        k562_o["mean_profile_r"] * 100,
        k562_o["site_detection_rate"] * 100,
    ]
    hepg2_vals = [
        hepg2_o["slot_tf_accuracy"] * 100,
        hepg2_o["hungarian_tf_accuracy"] * 100,
        hepg2_o["match_rate"] * 100,
        hepg2_o["mean_profile_r"] * 100,
        hepg2_o["site_detection_rate"] * 100,
    ]

    x = np.arange(len(metric_labels))
    width = 0.35
    bars_k = ax.bar(x - width / 2, k562_vals, width, label="K562 (baseline)",
                    color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_h = ax.bar(x + width / 2, hepg2_vals, width, label="HepG2 (transfer)",
                    color="coral", alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar_set in [bars_k, bars_h]:
        for bar in bar_set:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                    f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Performance (%)")
    ax.set_title("Overall Performance Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(k562_vals), max(hepg2_vals)) * 1.15)

    # ---- Panel 2: Per-TF profile Pearson r comparison ----
    ax = axes[0, 1]
    k562_rs = [k562_results["per_tf"].get(tf, {}).get("mean_profile_r", 0) for tf in shared_tf_names]
    hepg2_rs = [hepg2_results["per_tf"].get(tf, {}).get("mean_profile_r", 0) for tf in shared_tf_names]

    x = np.arange(len(shared_tf_names))
    bars_k = ax.bar(x - width / 2, k562_rs, width, label="K562", color="steelblue",
                    alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_h = ax.bar(x + width / 2, hepg2_rs, width, label="HepG2", color="coral",
                    alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar_set in [bars_k, bars_h]:
        for bar in bar_set:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(shared_tf_names, fontsize=10)
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-TF Profile Correlation")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(k562_rs + [0.01]), max(hepg2_rs + [0.01])) * 1.2)

    # ---- Panel 3: Transfer efficiency per TF ----
    ax = axes[1, 0]
    transfer_vals = [transfer["per_tf"].get(tf, {}).get("profile_r", 0)
                     for tf in shared_tf_names]
    tf_acc_transfer = [transfer["per_tf"].get(tf, {}).get("slot_tf_accuracy", 0)
                       for tf in shared_tf_names]

    x = np.arange(len(shared_tf_names))
    width2 = 0.35
    colors_profile = ["#2ca02c" if v > 80 else "#ff7f0e" if v > 50 else "#d62728"
                      for v in transfer_vals]
    colors_tf = ["#2ca02c" if v > 80 else "#ff7f0e" if v > 50 else "#d62728"
                 for v in tf_acc_transfer]

    bars1 = ax.bar(x - width2 / 2, transfer_vals, width2, label="Profile r Transfer",
                   color=colors_profile, alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width2 / 2, tf_acc_transfer, width2, label="TF Acc Transfer",
                   color=colors_tf, alpha=0.5, edgecolor="black", linewidth=0.5,
                   hatch="//")

    ax.axhline(y=100, color="blue", linestyle="--", alpha=0.4, label="100% (perfect)")
    ax.axhline(y=60, color="green", linestyle=":", alpha=0.4, label="60% threshold")

    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                    f"{height:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(shared_tf_names, fontsize=10)
    ax.set_ylabel("Transfer Efficiency (%)")
    ax.set_title("Per-TF Transfer Efficiency")
    ax.legend(loc="upper right", fontsize=8)

    # ---- Panel 4: Summary text ----
    ax = axes[1, 1]
    ax.axis("off")

    t_overall = transfer["overall"]
    tf_acc_eff = t_overall["tf_accuracy"]
    prof_eff = t_overall["profile_r"]

    if tf_acc_eff > 80:
        verdict = "EXCELLENT transfer - model learns universal TF motifs"
    elif tf_acc_eff > 60:
        verdict = "GOOD transfer - model generalizes well across cell types"
    elif tf_acc_eff > 40:
        verdict = "MODERATE transfer - some cell-type-specific learning"
    else:
        verdict = "POOR transfer - model may overfit to cell-specific patterns"

    summary_lines = [
        "CROSS-CELL-TYPE TRANSFER SUMMARY",
        "=" * 40,
        "",
        f"Training cell line:   K562 ({k562_o['n_samples']} test samples)",
        f"Transfer cell line:   HepG2 ({hepg2_o['n_samples']} test samples)",
        f"Shared TFs:           {', '.join(shared_tf_names)}",
        "",
        "TRANSFER EFFICIENCY:",
        f"  TF Accuracy:        {t_overall['tf_accuracy']:.1f}%",
        f"  Hungarian TF Acc:   {t_overall['hungarian_tf_accuracy']:.1f}%",
        f"  Match Rate:         {t_overall['match_rate']:.1f}%",
        f"  Profile Pearson r:  {t_overall['profile_r']:.1f}%",
        f"  Site Detection:     {t_overall['site_detection']:.1f}%",
        "",
        f"VERDICT: {verdict}",
        "",
        "KEY INSIGHT:",
        "BEACON learns TF binding motifs from sequence,",
        "which transfer across cell types because motifs",
        "are conserved while chromatin state differs.",
    ]
    summary_text = "\n".join(summary_lines)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved to: {output_path}")


# ---------------------------------------------------------------------------
# Formatted printing
# ---------------------------------------------------------------------------

def print_results_table(k562_results, hepg2_results, transfer, shared_tf_names):
    """Print formatted results tables to stdout."""
    k562_o = k562_results["overall"]
    hepg2_o = hepg2_results["overall"]
    t_o = transfer["overall"]

    print("\n" + "=" * 80)
    print("A3: CROSS-CELL-TYPE TRANSFER RESULTS")
    print("=" * 80)

    # Overall metrics table
    print("\n--- Overall Metrics ---")
    header = f"{'Metric':<28s} {'K562':>10s} {'HepG2':>10s} {'Transfer %':>12s}"
    print(header)
    print("-" * 62)
    rows = [
        ("TF Accuracy (slot)",
         f"{k562_o['slot_tf_accuracy']*100:.1f}%",
         f"{hepg2_o['slot_tf_accuracy']*100:.1f}%",
         f"{t_o['tf_accuracy']:.1f}%"),
        ("TF Accuracy (Hungarian)",
         f"{k562_o['hungarian_tf_accuracy']*100:.1f}%",
         f"{hepg2_o['hungarian_tf_accuracy']*100:.1f}%",
         f"{t_o['hungarian_tf_accuracy']:.1f}%"),
        ("Site Match Rate",
         f"{k562_o['match_rate']*100:.1f}%",
         f"{hepg2_o['match_rate']*100:.1f}%",
         f"{t_o['match_rate']:.1f}%"),
        ("Profile Pearson r",
         f"{k562_o['mean_profile_r']:.3f}",
         f"{hepg2_o['mean_profile_r']:.3f}",
         f"{t_o['profile_r']:.1f}%"),
        ("Site Detection Rate",
         f"{k562_o['site_detection_rate']*100:.1f}%",
         f"{hepg2_o['site_detection_rate']*100:.1f}%",
         f"{t_o['site_detection']:.1f}%"),
    ]
    for label, k, h, t in rows:
        print(f"{label:<28s} {k:>10s} {h:>10s} {t:>12s}")

    # Per-TF table
    print("\n--- Per-TF Profile Pearson r ---")
    header = f"{'TF':<10s} {'K562 r':>10s} {'HepG2 r':>10s} {'Transfer %':>12s} {'K562 n':>8s} {'HepG2 n':>8s}"
    print(header)
    print("-" * 60)
    for tf_name in shared_tf_names:
        k_tf = k562_results["per_tf"].get(tf_name, {})
        h_tf = hepg2_results["per_tf"].get(tf_name, {})
        t_tf = transfer["per_tf"].get(tf_name, {})
        print(f"{tf_name:<10s} "
              f"{k_tf.get('mean_profile_r', 0):>10.3f} "
              f"{h_tf.get('mean_profile_r', 0):>10.3f} "
              f"{t_tf.get('profile_r', 0):>11.1f}% "
              f"{k_tf.get('n_samples', 0):>8d} "
              f"{h_tf.get('n_samples', 0):>8d}")

    # Per-TF TF accuracy table
    print("\n--- Per-TF Slot TF Accuracy ---")
    header = f"{'TF':<10s} {'K562 Acc':>10s} {'HepG2 Acc':>10s} {'Transfer %':>12s}"
    print(header)
    print("-" * 44)
    for tf_name in shared_tf_names:
        k_tf = k562_results["per_tf"].get(tf_name, {})
        h_tf = hepg2_results["per_tf"].get(tf_name, {})
        t_tf = transfer["per_tf"].get(tf_name, {})
        print(f"{tf_name:<10s} "
              f"{k_tf.get('slot_tf_accuracy', 0)*100:>9.1f}% "
              f"{h_tf.get('slot_tf_accuracy', 0)*100:>9.1f}% "
              f"{t_tf.get('slot_tf_accuracy', 0):>11.1f}%")

    # Interpretation
    tf_acc_eff = t_o["tf_accuracy"]
    print("\n--- Interpretation ---")
    if tf_acc_eff > 80:
        print("EXCELLENT: Model learns universal TF motifs that transfer across cell types.")
    elif tf_acc_eff > 60:
        print("GOOD: Model generalizes well, capturing sequence-intrinsic binding determinants.")
    elif tf_acc_eff > 40:
        print("MODERATE: Partial transfer; some cell-type-specific features learned.")
    else:
        print("POOR: Model may be overfitting to K562-specific chromatin patterns.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A3: Cross-Cell-Type Transfer Validation (K562 -> HepG2)"
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("outputs/phase10_slot_ablation/slots_16/beacon_20260203_231900"),
        help="Directory containing best_model.pt (Phase 10, 16 slots)"
    )
    parser.add_argument(
        "--k562-data", type=Path,
        default=Path("data/processed/multi_tf_k562"),
        help="K562 processed data directory"
    )
    parser.add_argument(
        "--hepg2-data", type=Path,
        default=Path("data/processed/multi_tf_hepg2"),
        help="HepG2 processed data directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (cpu, cuda:0, etc.)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/a3_cross_cell_transfer"),
        help="Output directory for results and figures"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Resolve relative paths
    model_dir = args.model_dir if args.model_dir.is_absolute() else base_dir / args.model_dir
    k562_data = args.k562_data if args.k562_data.is_absolute() else base_dir / args.k562_data
    hepg2_data = args.hepg2_data if args.hepg2_data.is_absolute() else base_dir / args.hepg2_data
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A3: Cross-Cell-Type Transfer Validation")
    print("    K562-trained BEACON -> HepG2 evaluation")
    print("=" * 70)
    print(f"Model dir:   {model_dir}")
    print(f"K562 data:   {k562_data}")
    print(f"HepG2 data:  {hepg2_data}")
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {args.device}")
    print()

    # ---- Load HepG2 config and build TF mapping ----
    hepg2_config_path = hepg2_data / "config.json"
    with open(hepg2_config_path) as f:
        hepg2_config = json.load(f)
    hepg2_tf_names = hepg2_config["tf_names"]

    tf_mapping, shared_tf_names = build_tf_mapping(hepg2_tf_names)

    print(f"K562 TFs:    {K562_TF_NAMES}")
    print(f"HepG2 TFs:   {hepg2_tf_names}")
    print(f"Shared TFs:  {shared_tf_names}")
    print(f"TF mapping (HepG2 idx -> K562 idx): {tf_mapping}")
    print()

    # ---- Load model ----
    checkpoint_path = model_dir / "best_model.pt"
    print(f"Loading model from: {checkpoint_path}")

    model = BEACON(
        seq_len=2000,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=16,
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
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present
    cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print()

    # ---- Load datasets ----
    print("Loading K562 test data...")
    k562_test = load_k562_test(k562_data / "test.h5", seq_length=2000, n_slots=16)
    k562_loader = DataLoader(k562_test, batch_size=32, shuffle=False, num_workers=0)
    print(f"  K562 test samples: {len(k562_test)}")

    print("Loading HepG2 test data...")
    hepg2_test = load_hepg2_with_binding_sites(hepg2_data / "test.h5",
                                                seq_length=2000, n_slots=16)
    hepg2_loader = DataLoader(hepg2_test, batch_size=32, shuffle=False, num_workers=0)
    print(f"  HepG2 test samples: {len(hepg2_test)}")
    print()

    # ---- Evaluate on K562 baseline ----
    print("Evaluating on K562 (same cell line - baseline)...")
    k562_results = evaluate_k562_baseline(model, k562_loader, device,
                                          seq_len=2000, n_slots=16)
    print(f"  K562 TF accuracy (slot): {k562_results['overall']['slot_tf_accuracy']*100:.1f}%")
    print(f"  K562 profile r: {k562_results['overall']['mean_profile_r']:.3f}")
    print(f"  K562 match rate: {k562_results['overall']['match_rate']*100:.1f}%")
    print()

    # ---- Evaluate on HepG2 (transfer) ----
    print("Evaluating on HepG2 (cross-cell transfer)...")
    hepg2_results = evaluate_with_remapping(
        model, hepg2_loader, device, tf_mapping, hepg2_tf_names,
        seq_len=2000, n_slots=16,
    )
    print(f"  HepG2 TF accuracy (slot): {hepg2_results['overall']['slot_tf_accuracy']*100:.1f}%")
    print(f"  HepG2 profile r: {hepg2_results['overall']['mean_profile_r']:.3f}")
    print(f"  HepG2 match rate: {hepg2_results['overall']['match_rate']*100:.1f}%")
    print()

    # ---- Compute transfer efficiency ----
    transfer = compute_transfer_efficiency(k562_results, hepg2_results, shared_tf_names)

    # ---- Print formatted tables ----
    print_results_table(k562_results, hepg2_results, transfer, shared_tf_names)

    # ---- Create visualization ----
    print("Creating visualization...")
    fig_path = output_dir / "cross_cell_type_transfer.png"
    create_visualization(k562_results, hepg2_results, transfer, shared_tf_names,
                         fig_path)

    # ---- Save results JSON ----
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results_json = {
        "experiment": "A3_cross_cell_type_transfer",
        "timestamp": datetime.now().isoformat(),
        "model_path": str(checkpoint_path),
        "k562_data": str(k562_data),
        "hepg2_data": str(hepg2_data),
        "n_params": n_params,
        "k562_tf_names": K562_TF_NAMES,
        "hepg2_tf_names": hepg2_tf_names,
        "shared_tf_names": shared_tf_names,
        "tf_mapping_hepg2_to_k562": {str(k): v for k, v in tf_mapping.items()},
        "k562_results": k562_results,
        "hepg2_results": hepg2_results,
        "transfer_efficiency": transfer,
    }

    results_path = output_dir / "cross_cell_type_results.json"
    serializable = json.loads(json.dumps(results_json, default=convert_numpy))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to: {results_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
