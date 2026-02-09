#!/usr/bin/env python3
"""
Comprehensive Hungarian-matched evaluation of all BEACON models.
Evaluates Phase 10 slot ablation models (4, 8, 24 slots) and the original
16-slot model on the same test set using proper per-slot matching.
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


class SlotAdjustedDataset(Dataset):
    """Wraps BEACONDataset to adjust binding_sites to match model's n_slots."""
    def __init__(self, dataset, target_n_slots):
        self.dataset = dataset
        self.target_n_slots = target_n_slots

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'binding_sites' in item:
            bs = item['binding_sites']
            data_n_slots = bs.shape[0]
            if self.target_n_slots < data_n_slots:
                occupancies = bs[:, 2]
                _, sorted_idx = occupancies.sort(descending=True)
                item['binding_sites'] = bs[sorted_idx[:self.target_n_slots]]
            elif self.target_n_slots > data_n_slots:
                padding = torch.zeros(self.target_n_slots - data_n_slots, 3)
                item['binding_sites'] = torch.cat([bs, padding], dim=0)
        return item


def hungarian_matched_evaluation(model, test_loader, device, n_tfs=7, n_slots=16):
    """Full Hungarian-matched evaluation with per-TF breakdown."""
    model.eval()

    total_matched = 0
    total_correct_tf = 0
    total_sites = 0
    profile_correlations = []

    # Per-TF tracking
    per_tf_total = defaultdict(int)
    per_tf_matched = defaultdict(int)
    per_tf_correct = defaultdict(int)

    # Complexity tracking
    complexity_results = defaultdict(lambda: {'total': 0, 'matched': 0, 'correct_tf': 0, 'profile_r': []})

    tf_names = ['CTCF', 'GATA1', 'TAL1', 'MYC', 'MAX', 'SPI1', 'CEBPB']

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            binding_sites = batch['binding_sites']

            outputs = model(sequences, return_attention=True, return_slots=True)
            pred_positions = outputs['positions'].cpu()
            pred_tf_logits = outputs['tf_logits'].cpu()
            pred_occupancy = outputs['occupancy'].cpu()
            pred_profiles = outputs['profile'].cpu()
            target_profiles = batch['profile']

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                # Get ground truth sites
                gt_sites = []
                for s in range(binding_sites.shape[1]):
                    pos, tf_id, occ = binding_sites[b, s]
                    if occ > 0.5:
                        gt_sites.append((pos.item(), int(tf_id.item()), occ.item()))

                if not gt_sites:
                    continue

                n_gt = len(gt_sites)
                total_sites += n_gt

                # Track complexity
                complexity_key = min(n_gt, 4)  # 1, 2, 3, 4+
                complexity_results[complexity_key]['total'] += n_gt

                # For per-TF tracking
                for _, tf_id, _ in gt_sites:
                    per_tf_total[tf_id] += 1

                # Get predicted active slots
                pred_occ = pred_occupancy[b].squeeze(-1) if pred_occupancy[b].dim() > 1 else pred_occupancy[b]
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
                        pred_tf = pred_tf_logits[b, pred_idx].argmax().item()
                        # Both gt_pos and pred_pos are normalized [0, 1]; denormalize both
                        pos_dist = abs(gt_pos * 2000 - pred_pos * 2000)
                        tf_match = 1.0 if pred_tf == gt_tf else 0.0
                        cost_matrix[i, j] = pos_dist - tf_match * 500

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    gt_pos, gt_tf, _ = gt_sites[i]
                    pred_idx = active_indices[j]
                    pos_val = pred_positions[b, pred_idx]
                    pred_pos = pos_val[0].item() if pos_val.dim() > 0 and pos_val.numel() > 1 else pos_val.item()
                    pred_tf = pred_tf_logits[b, pred_idx].argmax().item()
                    # Both normalized [0, 1]; denormalize to bp
                    pos_dist_bp = abs(gt_pos * 2000 - pred_pos * 2000)

                    if pos_dist_bp < 200:
                        total_matched += 1
                        per_tf_matched[gt_tf] += 1
                        complexity_results[complexity_key]['matched'] += 1

                        if pred_tf == gt_tf:
                            total_correct_tf += 1
                            per_tf_correct[gt_tf] += 1
                            complexity_results[complexity_key]['correct_tf'] += 1

                # Profile correlation
                pred_p = pred_profiles[b].numpy()
                tgt_p = target_profiles[b].numpy()
                if pred_p.std() > 0 and tgt_p.std() > 0:
                    r = np.corrcoef(pred_p.flatten(), tgt_p.flatten())[0, 1]
                    if not np.isnan(r):
                        profile_correlations.append(r)
                        complexity_results[complexity_key]['profile_r'].append(r)

    # Compile results
    results = {
        'overall': {
            'total_sites': total_sites,
            'total_matched': total_matched,
            'total_correct_tf': total_correct_tf,
            'match_rate': total_matched / max(total_sites, 1),
            'tf_accuracy': total_correct_tf / max(total_matched, 1),
            'mean_profile_r': float(np.mean(profile_correlations)) if profile_correlations else 0.0,
        },
        'per_tf': {},
        'complexity': {},
    }

    for tf_id in range(n_tfs):
        name = tf_names[tf_id] if tf_id < len(tf_names) else f"TF{tf_id}"
        total = per_tf_total.get(tf_id, 0)
        matched = per_tf_matched.get(tf_id, 0)
        correct = per_tf_correct.get(tf_id, 0)
        results['per_tf'][name] = {
            'total': total,
            'matched': matched,
            'correct': correct,
            'match_rate': matched / max(total, 1),
            'tf_accuracy': correct / max(matched, 1),
        }

    for k in sorted(complexity_results.keys()):
        cr = complexity_results[k]
        label = f"{k}_sites" if k < 4 else "4+_sites"
        results['complexity'][label] = {
            'total_sites': cr['total'],
            'matched': cr['matched'],
            'correct_tf': cr['correct_tf'],
            'match_rate': cr['matched'] / max(cr['total'], 1),
            'tf_accuracy': cr['correct_tf'] / max(cr['matched'], 1),
            'mean_profile_r': float(np.mean(cr['profile_r'])) if cr['profile_r'] else 0.0,
            'n_sequences': len(cr['profile_r']),
        }

    return results


def load_and_evaluate(checkpoint_path, n_slots, data_dir, device, n_tfs=7):
    """Load a model checkpoint and run evaluation."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"  Slots: {n_slots}")
    print(f"{'='*60}")

    # Build model
    model = BEACON(
        seq_len=2000, input_channels=4,
        backbone_type="dilated", backbone_dim=128,
        backbone_layers=4,
        n_slots=n_slots, slot_dim=128,
        n_iterations=3,
        n_tfs=n_tfs, position_mode="gaussian", dropout=0.1,
        attention_mode="independent",
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('module.', '')] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Load test data
    raw_test = BEACONDataset(str(data_dir / "test.h5"), seq_length=2000,
                              augment=False, n_slots=16, extract_peaks=True)
    test_dataset = SlotAdjustedDataset(raw_test, n_slots)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"  Test samples: {len(test_dataset)}")

    # Run evaluation
    results = hungarian_matched_evaluation(model, test_loader, device, n_tfs, n_slots)

    # Print results
    o = results['overall']
    print(f"\n  Overall Results:")
    print(f"    Match rate: {o['match_rate']:.1%}")
    print(f"    TF accuracy (Hungarian-matched): {o['tf_accuracy']:.1%}")
    print(f"    Profile correlation: {o['mean_profile_r']:.3f}")
    print(f"    Total sites: {o['total_sites']}, Matched: {o['total_matched']}")

    print(f"\n  Per-TF Accuracy:")
    for name, tf_r in results['per_tf'].items():
        if tf_r['total'] > 0:
            print(f"    {name:8s}: {tf_r['tf_accuracy']:.1%} ({tf_r['correct']}/{tf_r['matched']} matched of {tf_r['total']})")

    print(f"\n  By Complexity:")
    for label, cr in results['complexity'].items():
        if cr['total_sites'] > 0:
            print(f"    {label:10s}: TF acc {cr['tf_accuracy']:.1%}, Match rate {cr['match_rate']:.1%}, Profile r {cr['mean_profile_r']:.3f} (n={cr['n_sequences']})")

    return results, n_params


def main():
    parser = argparse.ArgumentParser(description="Evaluate all BEACON models")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--output", type=str, default="outputs/evaluation_comparison.json")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed" / "multi_tf_k562"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    models = {
        '4_slots': {
            'path': base_dir / "outputs/phase10_slot_ablation/slots_4/beacon_20260203_202150/best_model.pt",
            'n_slots': 4,
        },
        '8_slots': {
            'path': base_dir / "outputs/phase10_slot_ablation/slots_8/beacon_20260203_202152/best_model.pt",
            'n_slots': 8,
        },
        '16_slots_phase10': {
            'path': base_dir / "outputs/phase10_slot_ablation/slots_16/beacon_20260203_231900/best_model.pt",
            'n_slots': 16,
        },
        '16_slots_original': {
            'path': base_dir / "outputs/multi_tf_k562/multi_tf_k562_20260128_172512/multi_tf_k562_20260128_172512/best_model.pt",
            'n_slots': 16,
        },
        '24_slots': {
            'path': base_dir / "outputs/phase10_slot_ablation/slots_24/beacon_20260203_202152/best_model.pt",
            'n_slots': 24,
        },
    }

    all_results = {}
    for name, info in models.items():
        if not info['path'].exists():
            print(f"\nSkipping {name}: {info['path']} not found")
            continue
        results, n_params = load_and_evaluate(
            info['path'], info['n_slots'], data_dir, device
        )
        all_results[name] = {
            'n_slots': info['n_slots'],
            'n_params': n_params,
            'results': results,
        }

    # Summary comparison table
    print(f"\n{'='*80}")
    print(f"SUMMARY: Hungarian-Matched Evaluation Comparison")
    print(f"{'='*80}")
    print(f"{'Model':20s} {'Slots':>6s} {'Params':>10s} {'Match Rate':>12s} {'TF Accuracy':>12s} {'Profile r':>10s}")
    print("-" * 80)
    for name, data in all_results.items():
        o = data['results']['overall']
        print(f"{name:20s} {data['n_slots']:>6d} {data['n_params']:>10,d} {o['match_rate']:>11.1%} {o['tf_accuracy']:>11.1%} {o['mean_profile_r']:>10.3f}")

    # Save all results
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
