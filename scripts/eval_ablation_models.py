#!/usr/bin/env python3
"""Hungarian-matched evaluation for ablation study models (K562-7tf)."""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset


def hungarian_eval(model, test_loader, device, n_tfs, tf_names):
    """Hungarian-matched evaluation with per-TF breakdown."""
    model.eval()

    total_matched = 0
    total_correct_tf = 0
    total_sites = 0
    profile_correlations = []
    position_errors = []

    per_tf_total = defaultdict(int)
    per_tf_matched = defaultdict(int)
    per_tf_correct = defaultdict(int)

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            binding_sites = batch['binding_sites']

            outputs = model(sequences)
            pred_positions = outputs['positions'].cpu()
            pred_tf_logits = outputs['tf_logits'].cpu()
            pred_occupancy = outputs['occupancy'].cpu()
            pred_profiles = outputs['profile'].cpu()
            target_profiles = batch['profile']

            batch_size = sequences.shape[0]

            for b in range(batch_size):
                gt_sites = []
                for s in range(binding_sites.shape[1]):
                    pos, tf_id, occ = binding_sites[b, s]
                    if occ > 0.5:
                        gt_sites.append((pos.item(), int(tf_id.item()), occ.item()))

                if not gt_sites:
                    continue

                n_gt = len(gt_sites)
                total_sites += n_gt

                for _, tf_id, _ in gt_sites:
                    per_tf_total[tf_id] += 1

                pred_occ = pred_occupancy[b].squeeze(-1) if pred_occupancy[b].dim() > 1 else pred_occupancy[b]
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
                        pred_tf = pred_tf_logits[b, pred_idx].argmax().item()
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
                    pos_dist_bp = abs(gt_pos * 2000 - pred_pos * 2000)

                    if pos_dist_bp < 200:
                        total_matched += 1
                        per_tf_matched[gt_tf] += 1
                        position_errors.append(pos_dist_bp)

                        if pred_tf == gt_tf:
                            total_correct_tf += 1
                            per_tf_correct[gt_tf] += 1

                pred_p = pred_profiles[b].numpy()
                tgt_p = target_profiles[b].numpy()
                if pred_p.std() > 0 and tgt_p.std() > 0:
                    r = np.corrcoef(pred_p.flatten(), tgt_p.flatten())[0, 1]
                    if not np.isnan(r):
                        profile_correlations.append(r)

    results = {
        'total_sites': total_sites,
        'total_matched': total_matched,
        'total_correct_tf': total_correct_tf,
        'match_rate': total_matched / max(total_sites, 1),
        'tf_accuracy': total_correct_tf / max(total_matched, 1),
        'mean_profile_r': float(np.mean(profile_correlations)) if profile_correlations else 0.0,
        'median_position_error_bp': float(np.median(position_errors)) if position_errors else 0.0,
    }

    per_tf = {}
    for tf_id in range(n_tfs):
        name = tf_names[tf_id] if tf_id < len(tf_names) else f"TF{tf_id}"
        total = per_tf_total.get(tf_id, 0)
        matched = per_tf_matched.get(tf_id, 0)
        correct = per_tf_correct.get(tf_id, 0)
        per_tf[name] = {
            'total': total, 'matched': matched, 'correct': correct,
            'tf_accuracy': correct / max(matched, 1),
        }
    results['per_tf'] = per_tf
    return results


ABLATION_MODELS = {
    'baseline': 'outputs/improved/K562-7tf_baseline/beacon_20260206_121701/best_model.pt',
    'slot_losses': 'outputs/improved/K562-7tf_slot_losses/beacon_20260206_184050/best_model.pt',
    'slot_dropout': 'outputs/improved/K562-7tf_slot_dropout/beacon_20260207_003853/best_model.pt',
    'pcgrad': 'outputs/improved/K562-7tf_pcgrad/beacon_20260206_193802/best_model.pt',
    'gradnorm': 'outputs/improved/K562-7tf_gradnorm/beacon_20260206_193916/best_model.pt',
    'full': 'outputs/improved/K562-7tf_full/beacon_20260206_194034/best_model.pt',
}

N_TFS = 7
TF_NAMES = ['CTCF', 'GATA1', 'TAL1', 'MYC', 'MAX', 'SPI1', 'CEBPB']
DATA_DIR = 'data/processed/multi_tf_k562'


def main():
    device = 'cuda:0'

    # Load test data once
    test_path = Path(DATA_DIR) / 'test.h5'
    test_ds = BEACONDataset(str(test_path), seq_length=2000, augment=False,
                            n_slots=16, extract_peaks=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_ds)}")

    all_results = {}
    for name, ckpt_path in ABLATION_MODELS.items():
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP {name}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"{'='*60}")

        model = BEACON(
            seq_len=2000, input_channels=4,
            backbone_type="dilated", backbone_dim=128,
            backbone_layers=4,
            n_slots=16, slot_dim=128, n_iterations=3,
            n_tfs=N_TFS, position_mode="gaussian",
            dropout=0.1, attention_mode="independent",
        )
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        cleaned = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        model = model.to(device).eval()

        results = hungarian_eval(model, test_loader, device, N_TFS, TF_NAMES)
        all_results[name] = results

        print(f"  Profile r:   {results['mean_profile_r']:.4f}")
        print(f"  TF accuracy: {results['tf_accuracy']:.1%}")
        print(f"  Match rate:  {results['match_rate']:.1%}")
        print(f"  Sites: {results['total_matched']}/{results['total_sites']}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Ablation Study: Test Set Results (Hungarian-matched)")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Profile r':<12} {'TF Acc':<10} {'Match Rate':<12} {'Sites':<10}")
    for name, r in all_results.items():
        print(f"  {name:<15} {r['mean_profile_r']:.4f}       {r['tf_accuracy']:.1%}      {r['match_rate']:.1%}        {r['total_matched']}/{r['total_sites']}")

    # Save
    out_path = Path('outputs/ablation_eval_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
