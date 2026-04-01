"""
Shared evaluation utilities for BEACON.

Centralizes model loading, batch inference, per-sample metrics, and
dataset/model configurations used across all evaluation scripts.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path(__file__).parent.parent.parent  # beacon/

# ── Dataset configurations ────────────────────────────────────────────
DATASET_CONFIGS = {
    "K562-7tf": {
        "data_dir": "data/processed/multi_tf_k562",
        "n_tfs": 7,
        "tf_names": ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"],
    },
    "K562-fulltf": {
        "data_dir": "data/processed/multi_tf_k562_fulltf",
        "n_tfs": 14,
        "tf_names": ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "CEBPB",
                      "REST", "YY1", "NRF1", "JUND", "FOS", "ATF3", "ELF1", "GABPA"],
    },
    "HepG2-7tf": {
        "data_dir": "data/processed/multi_tf_hepg2_7tf",
        "n_tfs": 7,
        "tf_names": ["CTCF", "MYC", "MAX", "CEBPB", "REST", "YY1", "NRF1"],
    },
    "HepG2-fulltf": {
        "data_dir": "data/processed/multi_tf_hepg2_fulltf",
        "n_tfs": 12,
        "tf_names": ["CTCF", "MYC", "MAX", "CEBPB", "REST", "YY1", "NRF1",
                      "ELF1", "FOXA2", "HNF4A", "MAFK", "NFE2L2"],
    },
}

# Best checkpoints per model
MODEL_CHECKPOINTS = {
    "K562-7tf": "outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt",
    "K562-fulltf": "outputs/k562_20tf/K562-20tf_baseline/beacon_20260205_190727/best_model.pt",
    "HepG2-7tf": "outputs/hepg2_7tf/HepG2-7tf_baseline/beacon_20260205_190727/best_model.pt",
    "HepG2-fulltf": "outputs/hepg2_20tf/HepG2-20tf_baseline/beacon_20260205_190728/best_model.pt",
}

# Ablation study checkpoints (all K562-7tf)
ABLATION_CHECKPOINTS = {
    "baseline": "outputs/improved/K562-7tf_baseline/beacon_20260206_121701/best_model.pt",
    "slot_losses": "outputs/improved/K562-7tf_slot_losses/beacon_20260206_184050/best_model.pt",
    "slot_dropout": "outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt",
    "pcgrad": "outputs/improved/K562-7tf_pcgrad/beacon_20260206_193802/best_model.pt",
    "gradnorm": "outputs/improved/K562-7tf_gradnorm/beacon_20260206_193916/best_model.pt",
    "full": "outputs/improved/K562-7tf_full/beacon_20260206_194034/best_model.pt",
}

# TF family groupings
TF_FAMILIES = {
    "Zinc finger": ["CTCF"],
    "GATA": ["GATA1"],
    "bHLH": ["TAL1", "MYC", "MAX"],
    "ETS": ["SPI1", "ELF1", "GABPA"],
    "bZIP": ["CEBPB", "JUND", "FOS", "ATF3", "MAFK", "NFE2L2"],
    "C2H2 ZF": ["REST", "YY1"],
    "NRF": ["NRF1"],
    "Forkhead": ["FOXA2"],
    "Nuclear receptor": ["HNF4A"],
}


# ── Model loading ─────────────────────────────────────────────────────
def load_model(checkpoint_path, n_tfs, device):
    """Load a BEACON model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        n_tfs: Number of TFs the model was trained on.
        device: Torch device string or device object.

    Returns:
        BEACON model in eval mode on the given device.
    """
    from beacon.models import BEACON

    model = BEACON(
        seq_len=2000, input_channels=4, backbone_type="dilated",
        backbone_dim=128, backbone_layers=4, n_slots=16, slot_dim=128,
        n_iterations=3, n_tfs=n_tfs, position_mode="gaussian",
        dropout=0.0, attention_mode="independent",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_named_model(name, device):
    """Load a model by dataset name (e.g. 'K562-7tf').

    Returns:
        (model, dataset_config) tuple or raises FileNotFoundError.
    """
    cfg = DATASET_CONFIGS[name]
    ckpt = BASE_DIR / MODEL_CHECKPOINTS[name]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = load_model(ckpt, cfg["n_tfs"], device)
    return model, cfg


# ── Data loading ──────────────────────────────────────────────────────
def load_test_data(data_path, max_samples=None):
    """Load test data from HDF5 file.

    Returns:
        (sequences, profiles, tf_indices, binding_sites) numpy arrays.
    """
    with h5py.File(data_path, 'r') as f:
        n = f['sequences'].shape[0]
        if max_samples and max_samples < n:
            n = max_samples
        sequences = f['sequences'][:n]
        profiles = f['profiles'][:n]
        tf_indices = f['tf_indices'][:n]
        binding_sites = f['binding_sites'][:n]
    return sequences, profiles, tf_indices, binding_sites


def get_test_path(dataset_name):
    """Return absolute path to test.h5 for a dataset."""
    return BASE_DIR / DATASET_CONFIGS[dataset_name]["data_dir"] / "test.h5"


# ── Batch inference ───────────────────────────────────────────────────
def run_model_batch(model, sequences, device, batch_size=32, return_attention=False):
    """Run BEACON inference on a batch of sequences.

    Args:
        model: BEACON model in eval mode.
        sequences: numpy array [N, L, 4] one-hot encoded.
        device: Torch device.
        batch_size: Batch size for inference.
        return_attention: If True, also returns 'attention' and 'slot_embeddings'.

    Returns:
        Dict with keys 'profile' [N,L], 'occupancy' [N,K],
        'tf_logits' [N,K,T], 'positions' [N,K,2], and optionally
        'attention' [N,K,L], 'slot_embeddings' [N,K,D].
    """
    output_keys = ['profile', 'occupancy', 'tf_logits', 'positions']
    if return_attention:
        output_keys += ['attention', 'slot_embeddings']

    all_outputs = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[i:i+batch_size], dtype=torch.float32
            ).to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(batch, return_attention=return_attention,
                                return_slots=return_attention)
            for k in output_keys:
                if k in outputs:
                    all_outputs[k].append(outputs[k].cpu().numpy())

    result = {}
    for k, v in all_outputs.items():
        arr = np.concatenate(v, axis=0)
        if k == 'profile' and arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif k == 'occupancy' and arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], arr.shape[1])
        result[k] = arr
    return result


# ── Per-sample metrics ────────────────────────────────────────────────
def per_sample_pearson(pred_profiles, gt_profiles):
    """Compute per-sample Pearson r between predicted and ground truth profiles.

    Returns:
        numpy array of length N with per-sample correlations.
    """
    n = len(pred_profiles)
    rs = np.zeros(n)
    for i in range(n):
        p = pred_profiles[i].flatten()
        g = gt_profiles[i].flatten()
        if np.std(p) > 0 and np.std(g) > 0:
            r = np.corrcoef(p, g)[0, 1]
            rs[i] = r if not np.isnan(r) else 0.0
    return rs


def hungarian_accuracy_per_site(outputs, binding_sites, n_tfs):
    """Compute per-matched-site correct/incorrect via Hungarian matching.

    Returns:
        numpy array of 1s and 0s (correct TF / incorrect) for each matched site.
    """
    pred_positions = outputs['positions']
    pred_tf_logits = outputs['tf_logits']
    pred_occupancy = outputs['occupancy']
    correct_list = []

    for b in range(len(pred_positions)):
        gt_sites = []
        for s in range(binding_sites.shape[1]):
            pos, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_sites.append((float(pos), int(tf_id), float(occ)))
        if not gt_sites:
            continue

        active_mask = pred_occupancy[b] > 0.3
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            correct_list.extend([0] * len(gt_sites))
            continue

        n_gt = len(gt_sites)
        n_pred = len(active_indices)
        cost_matrix = np.zeros((n_gt, n_pred))

        for i, (gt_pos, gt_tf, _) in enumerate(gt_sites):
            for j, pred_idx in enumerate(active_indices):
                pos_val = pred_positions[b, pred_idx]
                pred_pos = pos_val[0] if pos_val.shape[0] > 1 else pos_val.flat[0]
                pred_tf = int(pred_tf_logits[b, pred_idx].argmax())
                pos_dist = abs(gt_pos * 2000 - pred_pos * 2000)
                cost_matrix[i, j] = pos_dist - (1.0 if pred_tf == gt_tf else 0.0) * 500

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            gt_pos, gt_tf, _ = gt_sites[i]
            pred_idx = active_indices[j]
            pos_val = pred_positions[b, pred_idx]
            pred_pos = pos_val[0] if pos_val.shape[0] > 1 else pos_val.flat[0]
            pred_tf = int(pred_tf_logits[b, pred_idx].argmax())
            if abs(gt_pos * 2000 - pred_pos * 2000) < 200:
                correct_list.append(1 if pred_tf == gt_tf else 0)

    return np.array(correct_list)


def hungarian_confusion_matrix(outputs, binding_sites, n_tfs):
    """Build TF confusion matrix via Hungarian matching.

    Returns:
        numpy array [n_tfs, n_tfs] where rows=GT, cols=predicted.
    """
    pred_positions = outputs['positions']
    pred_tf_logits = outputs['tf_logits']
    pred_occupancy = outputs['occupancy']
    confusion = np.zeros((n_tfs, n_tfs), dtype=int)

    for b in range(len(pred_positions)):
        gt_sites = []
        for s in range(binding_sites.shape[1]):
            pos, tf_id, occ = binding_sites[b, s]
            if occ > 0.5:
                gt_sites.append((float(pos), int(tf_id), float(occ)))
        if not gt_sites:
            continue

        active_mask = pred_occupancy[b] > 0.3
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            continue

        n_gt = len(gt_sites)
        n_pred = len(active_indices)
        cost_matrix = np.zeros((n_gt, n_pred))

        for i, (gt_pos, gt_tf, _) in enumerate(gt_sites):
            for j, pred_idx in enumerate(active_indices):
                pos_val = pred_positions[b, pred_idx]
                pred_pos = pos_val[0] if pos_val.shape[0] > 1 else pos_val.flat[0]
                pred_tf = int(pred_tf_logits[b, pred_idx].argmax())
                pos_dist = abs(gt_pos * 2000 - pred_pos * 2000)
                cost_matrix[i, j] = pos_dist - (1.0 if pred_tf == gt_tf else 0.0) * 500

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            gt_pos, gt_tf, _ = gt_sites[i]
            pred_idx = active_indices[j]
            pos_val = pred_positions[b, pred_idx]
            pred_pos = pos_val[0] if pos_val.shape[0] > 1 else pos_val.flat[0]
            pred_tf = int(pred_tf_logits[b, pred_idx].argmax())
            if abs(gt_pos * 2000 - pred_pos * 2000) < 200:
                if gt_tf < n_tfs and pred_tf < n_tfs:
                    confusion[gt_tf, pred_tf] += 1

    return confusion


# ── Utility ───────────────────────────────────────────────────────────
def gc_content(sequence_onehot):
    """GC content from one-hot [L, 4] with order A, C, G, T."""
    return float((sequence_onehot[:, 1] + sequence_onehot[:, 2]).mean())


def json_serialize(obj):
    """Default serializer for numpy types in json.dump."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
