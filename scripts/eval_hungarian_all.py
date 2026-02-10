#!/usr/bin/env python3
"""Hungarian-matched evaluation for all multi-cell-line models."""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
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
        'overall': {
            'total_sites': total_sites,
            'total_matched': total_matched,
            'total_correct_tf': total_correct_tf,
            'match_rate': total_matched / max(total_sites, 1),
            'tf_accuracy': total_correct_tf / max(total_matched, 1),
            'mean_profile_r': float(np.mean(profile_correlations)) if profile_correlations else 0.0,
            'median_position_error_bp': float(np.median(position_errors)) if position_errors else 0.0,
        },
        'per_tf': {},
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

    return results


MODELS = {
    'K562-7tf': {
        'checkpoint': 'outputs/improved/K562-7tf_slot_dropout/beacon_*/best_model.pt',
        'data_dir': 'data/processed/multi_tf_k562',
        'n_tfs': 7,
        'tf_names': ['CTCF', 'GATA1', 'TAL1', 'MYC', 'MAX', 'SPI1', 'CEBPB'],
    },
    'K562-fulltf': {
        'checkpoint': 'outputs/k562_20tf/K562-20tf_baseline/beacon_*/best_model.pt',
        'data_dir': 'data/processed/multi_tf_k562_fulltf',
        'n_tfs': 14,
        'tf_names': ['CTCF', 'GATA1', 'TAL1', 'MYC', 'MAX', 'SPI1', 'CEBPB',
                      'REST', 'YY1', 'NRF1', 'JUND', 'FOS', 'ATF3', 'ELF1'],
    },
    'HepG2-7tf': {
        'checkpoint': 'outputs/hepg2_7tf/HepG2-7tf_baseline/beacon_*/best_model.pt',
        'data_dir': 'data/processed/multi_tf_hepg2_7tf',
        'n_tfs': 7,
        'tf_names': ['CTCF', 'MYC', 'MAX', 'CEBPB', 'REST', 'YY1', 'NRF1'],
    },
    'HepG2-fulltf': {
        'checkpoint': 'outputs/hepg2_20tf/HepG2-20tf_baseline/beacon_*/best_model.pt',
        'data_dir': 'data/processed/multi_tf_hepg2_fulltf',
        'n_tfs': 12,
        'tf_names': ['CTCF', 'MYC', 'MAX', 'CEBPB', 'REST', 'YY1', 'NRF1',
                      'ELF1', 'FOXA2', 'HNF4A', 'MAFK', 'NFE2L2'],
    },
}


def main():
    import glob
    device = 'cuda:0'

    all_results = {}
    for name, cfg in MODELS.items():
        print(f"\n{'='*70}")
        print(f"  {name} ({cfg['n_tfs']} TFs)")
        print(f"{'='*70}")

        # Find checkpoint
        matches = sorted(glob.glob(str(Path(cfg['checkpoint']))))
        if not matches:
            print(f"  SKIP: No checkpoint found at {cfg['checkpoint']}")
            continue
        ckpt_path = matches[-1]
        print(f"  Checkpoint: {ckpt_path}")

        # Check data
        data_dir = Path(cfg['data_dir'])
        test_path = data_dir / 'test.h5'
        if not test_path.exists():
            print(f"  SKIP: No test data at {test_path}")
            continue

        # Load model
        model = BEACON(
            seq_len=2000, input_channels=4,
            backbone_type="dilated", backbone_dim=128,
            backbone_layers=4,
            n_slots=16, slot_dim=128, n_iterations=3,
            n_tfs=cfg['n_tfs'], position_mode="gaussian",
            dropout=0.1, attention_mode="independent",
        )
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        cleaned = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        model = model.to(device).eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        # Load test data
        test_ds = BEACONDataset(str(test_path), seq_length=2000, augment=False,
                                n_slots=16, extract_peaks=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                 num_workers=4, pin_memory=True)
        print(f"  Test samples: {len(test_ds)}")

        # Evaluate
        results = hungarian_eval(model, test_loader, device, cfg['n_tfs'], cfg['tf_names'])

        o = results['overall']
        print(f"\n  Overall:")
        print(f"    Match rate:     {o['match_rate']:.1%}")
        print(f"    TF accuracy:    {o['tf_accuracy']:.1%}")
        print(f"    Profile r:      {o['mean_profile_r']:.4f}")
        print(f"    Position MAE:   {o['median_position_error_bp']:.1f} bp")
        print(f"    Sites: {o['total_matched']}/{o['total_sites']} matched")

        print(f"\n  Per-TF:")
        for tf_name, tf_r in results['per_tf'].items():
            if tf_r['total'] > 0:
                print(f"    {tf_name:8s}: {tf_r['tf_accuracy']:5.1%}  ({tf_r['correct']}/{tf_r['matched']} of {tf_r['total']})")

        all_results[name] = results

    # Save
    out_path = Path('outputs/hungarian_eval_all.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
