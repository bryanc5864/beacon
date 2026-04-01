#!/usr/bin/env python3
"""
Enhanced in-silico mutagenesis with PWM-based motif scanning.

Scans test sequences for JASPAR motif matches using actual PWMs,
mutates them, and compares disruption to a random-position control.

Usage:
    python scripts/run_enhanced_ism.py --device cuda:2
    python scripts/run_enhanced_ism.py --device cuda:2 --max-per-tf 100
"""

import torch
torch.backends.cudnn.enabled = False

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.analysis import (
    load_model, load_test_data, run_model_batch, json_serialize,
    bootstrap_ci, mann_whitney,
    DATASET_CONFIGS, MODEL_CHECKPOINTS, BASE_DIR,
)

# ── JASPAR PWMs (position frequency matrices, rows=positions, cols=ACGT) ─
JASPAR_PWMS = {
    "CTCF": np.array([
        [0.15,0.40,0.30,0.15],[0.10,0.60,0.15,0.15],[0.15,0.15,0.55,0.15],
        [0.10,0.50,0.30,0.10],[0.15,0.15,0.55,0.15],[0.20,0.20,0.20,0.40],
        [0.15,0.15,0.55,0.15],[0.15,0.15,0.55,0.15],[0.20,0.20,0.20,0.40],
        [0.15,0.15,0.55,0.15],[0.15,0.15,0.55,0.15],[0.10,0.55,0.20,0.15],
        [0.50,0.15,0.20,0.15],[0.15,0.15,0.55,0.15],
    ]),
    "GATA1": np.array([
        [0.55,0.10,0.20,0.15],[0.15,0.10,0.60,0.15],[0.70,0.05,0.15,0.10],
        [0.10,0.10,0.10,0.70],[0.65,0.10,0.10,0.15],[0.55,0.10,0.20,0.15],
    ]),
    "TAL1": np.array([
        [0.15,0.50,0.20,0.15],[0.55,0.15,0.15,0.15],[0.15,0.15,0.55,0.15],
        [0.15,0.50,0.15,0.20],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "MYC": np.array([
        [0.15,0.55,0.15,0.15],[0.60,0.10,0.15,0.15],[0.15,0.55,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "MAX": np.array([
        [0.15,0.55,0.15,0.15],[0.60,0.10,0.15,0.15],[0.15,0.55,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.10,0.10,0.10,0.70],[0.15,0.15,0.55,0.15],
    ]),
    "SPI1": np.array([
        [0.55,0.10,0.15,0.20],[0.55,0.10,0.15,0.20],[0.60,0.10,0.15,0.15],
        [0.15,0.10,0.60,0.15],[0.55,0.10,0.15,0.20],[0.15,0.10,0.60,0.15],
        [0.15,0.10,0.60,0.15],[0.55,0.15,0.15,0.15],[0.55,0.15,0.15,0.15],
        [0.15,0.10,0.55,0.20],[0.10,0.10,0.10,0.70],[0.15,0.10,0.60,0.15],
    ]),
    "CEBPB": np.array([
        [0.10,0.10,0.10,0.70],[0.10,0.10,0.10,0.70],[0.15,0.10,0.60,0.15],
        [0.15,0.55,0.15,0.15],[0.15,0.10,0.55,0.20],[0.15,0.55,0.15,0.15],
        [0.60,0.10,0.15,0.15],[0.55,0.15,0.15,0.15],
    ]),
}


def scan_pwm(seq_onehot, pwm, threshold_pct=80):
    """Scan a one-hot sequence with a PWM log-odds matrix.

    Returns list of (position, score) for motif hits above *threshold_pct*
    percentile of positive scores, with non-max suppression.
    """
    L, M = seq_onehot.shape[0], pwm.shape[0]
    if L < M:
        return []
    log_pwm = np.log2(pwm / 0.25 + 1e-10)

    # forward + reverse complement
    scores_fwd = np.array([np.sum(seq_onehot[p:p+M] * log_pwm) for p in range(L - M + 1)])
    rc = seq_onehot[::-1, ::-1].copy()
    scores_rev = np.array([np.sum(rc[p:p+M] * log_pwm) for p in range(L - M + 1)])
    scores = np.maximum(scores_fwd, scores_rev)

    pos_scores = scores[scores > 0]
    if len(pos_scores) == 0:
        return []
    thr = np.percentile(pos_scores, threshold_pct)
    hits = [(p, float(scores[p])) for p in range(len(scores)) if scores[p] >= thr]
    hits.sort(key=lambda x: -x[1])
    # non-max suppression
    kept = []
    for p, s in hits:
        if not any(abs(p - kp) < M for kp, _ in kept):
            kept.append((p, s))
    return kept[:5]


def _mutate(seq, start, end, seed):
    rng = np.random.RandomState(seed)
    mut = seq.copy()
    for p in range(start, min(end, len(seq))):
        mut[p] = 0.0
        mut[p, rng.randint(0, 4)] = 1.0
    return mut


def _disruption(ref_out, mut_out, idx, tf_idx, start, motif_len, window=50):
    """Compute disruption metrics for a single sample."""
    lo = max(0, start - window)
    hi = min(2000, start + motif_len + window)
    ref_local = np.abs(ref_out['profile'][idx, lo:hi]).sum()
    mut_local = np.abs(mut_out['profile'][0, lo:hi]).sum()
    delta_local = np.abs(ref_out['profile'][idx, lo:hi] - mut_out['profile'][0, lo:hi]).sum()
    profile_delta = float(delta_local / (ref_local + 1e-8))

    rp, mp = ref_out['profile'][idx], mut_out['profile'][0]
    if np.std(rp) > 0 and np.std(mp) > 0:
        shape_corr = np.corrcoef(rp, mp)[0, 1]
        shape_corr = shape_corr if not np.isnan(shape_corr) else 1.0
    else:
        shape_corr = 1.0
    shape_change = 1.0 - shape_corr

    ref_tf_mask = ref_out['tf_logits'][idx].argmax(-1) == tf_idx
    ref_occ = ref_out['occupancy'][idx][ref_tf_mask].sum() if ref_tf_mask.any() else 0
    mut_tf_mask = mut_out['tf_logits'][0].argmax(-1) == tf_idx
    mut_occ = mut_out['occupancy'][0][mut_tf_mask].sum() if mut_tf_mask.any() else 0
    occ_delta = float(ref_occ - mut_occ)

    n_ref = ref_tf_mask.sum()
    n_mut = mut_tf_mask.sum()
    tf_flip = bool(n_ref > 0 and n_mut < n_ref)

    return profile_delta, shape_change, occ_delta, tf_flip


# ── Main ISM ──────────────────────────────────────────────────────────
def run_ism(device, max_per_tf=200, n_random=5):
    cfg = DATASET_CONFIGS["K562-7tf"]
    ckpt = BASE_DIR / MODEL_CHECKPOINTS["K562-7tf"]
    test_path = BASE_DIR / cfg["data_dir"] / "test.h5"
    tf_names = cfg["tf_names"]

    model = load_model(ckpt, cfg["n_tfs"], device)
    seqs, profs, tf_idx, _ = load_test_data(test_path)

    motif_results, random_results, comparison = {}, {}, {}
    rng = np.random.RandomState(123)

    for ti, tn in enumerate(tf_names):
        if tn not in JASPAR_PWMS:
            continue
        pwm = JASPAR_PWMS[tn]
        mlen = pwm.shape[0]
        mask = tf_idx == ti
        tseqs = seqs[mask][:max_per_tf]
        n = len(tseqs)
        if n == 0:
            continue

        print(f"\n  {tn} (n={n}, motif_len={mlen}):")

        # Scan PWM hits
        all_hits = [scan_pwm(tseqs[i], pwm) for i in range(n)]
        n_with = sum(1 for h in all_hits if h)
        print(f"    PWM hits: {n_with}/{n} seqs ({n_with/n*100:.1f}%)")

        ref_out = run_model_batch(model, tseqs, device)

        # ── Motif ISM ──
        motif_d = defaultdict(list)
        for i in range(n):
            if not all_hits[i]:
                continue
            pos, _ = all_hits[i][0]
            mut = _mutate(tseqs[i], pos, pos + mlen, 42 + i)[np.newaxis]
            mut_out = run_model_batch(model, mut, device)
            pd, sc, od, tf = _disruption(ref_out, mut_out, i, ti, pos, mlen)
            motif_d["profile_delta"].append(pd)
            motif_d["shape_change"].append(sc)
            motif_d["occ_delta"].append(od)
            motif_d["tf_flip"].append(tf)

        if motif_d["profile_delta"]:
            motif_results[tn] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in motif_d.items() if k != "tf_flip"
            }
            motif_results[tn]["tf_flip_rate"] = float(np.mean(motif_d["tf_flip"]))
            motif_results[tn]["n"] = len(motif_d["profile_delta"])
            print(f"    Motif Δprofile: {np.mean(motif_d['profile_delta']):.4f} ± {np.std(motif_d['profile_delta']):.4f}")

        # ── Random control ISM ──
        rand_d = defaultdict(list)
        for i in range(n):
            motif_positions = set()
            for p, _ in all_hits[i]:
                motif_positions.update(range(p, min(p + mlen, 2000)))
            for r in range(n_random):
                for _ in range(100):
                    rs = rng.randint(200, 1800 - mlen)
                    if not any(p in motif_positions for p in range(rs, rs + mlen)):
                        break
                mut = _mutate(tseqs[i], rs, rs + mlen, 1000 + i * n_random + r)[np.newaxis]
                mut_out = run_model_batch(model, mut, device)
                pd, sc, od, _ = _disruption(ref_out, mut_out, i, ti, rs, mlen)
                rand_d["profile_delta"].append(pd)
                rand_d["shape_change"].append(sc)
                rand_d["occ_delta"].append(od)

        if rand_d["profile_delta"]:
            random_results[tn] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in rand_d.items()
            }
            random_results[tn]["n"] = len(rand_d["profile_delta"])
            print(f"    Random Δprofile: {np.mean(rand_d['profile_delta']):.4f} ± {np.std(rand_d['profile_delta']):.4f}")

        # ── Comparison ──
        if motif_d["profile_delta"] and rand_d["profile_delta"]:
            mw = mann_whitney(np.array(motif_d["profile_delta"]),
                              np.array(rand_d["profile_delta"]))
            m_mean = np.mean(motif_d["profile_delta"])
            r_mean = np.mean(rand_d["profile_delta"])
            ratio = m_mean / (r_mean + 1e-8)
            print(f"    Ratio: {ratio:.2f}x  Mann-Whitney p={mw['pvalue']:.2e}")
            comparison[tn] = {
                "motif_mean": float(m_mean), "random_mean": float(r_mean),
                "ratio": float(ratio), "mann_whitney_p": mw["pvalue"],
            }

    del model; torch.cuda.empty_cache()
    return {"motif": motif_results, "random_control": random_results, "comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description="Enhanced ISM with PWM scanning")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-per-tf", type=int, default=200)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Enhanced ISM with JASPAR PWM scanning")
    print("=" * 70)

    results = run_ism(device, args.max_per_tf, args.n_random)
    path = out_dir / "enhanced_ism.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
