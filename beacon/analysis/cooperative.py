"""
Cooperative binding and motif syntax analysis for BEACON.

Since each test sample contains a single binding site, cooperative binding
is analysed across samples rather than within samples:

1. Cross-sample attention similarity — do known partners (e.g. MYC+MAX)
   share attention patterns more than non-partners?
2. Slot representation similarity — do cooperative TFs use overlapping slots?
3. Position distribution analysis — do cooperative TFs show overlapping
   binding positions?
"""

import math
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, ks_2samp


DNA_HELIX_PERIOD = 10.5  # bp per full turn

# Known cooperative TF pairs
COOPERATIVE_PAIRS = [
    {"name": "MYC+MAX", "tf_a": "MYC", "tf_b": "MAX",
     "biology": "Heterodimer at E-box (CACGTG), obligate partners"},
    {"name": "GATA1+TAL1", "tf_a": "GATA1", "tf_b": "TAL1",
     "biology": "Composite element in erythroid enhancers"},
]


def _helical_phase(bp):
    return (2 * math.pi * bp / DNA_HELIX_PERIOD) % (2 * math.pi)


def _phase_diff(a, b):
    d = abs(a - b)
    return min(d, 2 * math.pi - d)


def _summarize(vals):
    if not vals:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
    }


def _get_active_slot_idx(occupancy_row, occ_threshold=0.3):
    """Return index of highest-occupancy active slot, or None."""
    active = occupancy_row > occ_threshold
    if not active.any():
        return None
    return int(np.argmax(occupancy_row))


def attention_pair_analysis(outputs, tf_names, tf_idx=None, occ_threshold=0.3):
    """Analyse attention similarity between known TF pairs across samples.

    For each cooperative pair (e.g. MYC+MAX), computes:
    - Mean attention profile for each TF
    - Cross-TF cosine similarity (are attention patterns similar?)
    - Comparison with non-partner TFs as control

    Args:
        outputs: dict from run_model_batch (must include 'attention').
        tf_names: list of TF name strings.
        tf_idx: ground-truth TF index per sample (np.array of ints).
                If None, uses tf_logits predictions.
        occ_threshold: occupancy threshold for active slots.

    Returns:
        dict keyed by pair name.
    """
    occupancy = outputs['occupancy']
    tf_preds = outputs['tf_logits'].argmax(axis=-1)
    attention = outputs['attention']
    n_samples = len(occupancy)
    results = {}

    # Collect per-TF attention profiles
    tf_attentions = defaultdict(list)  # tf_name -> list of attention vectors
    for i in range(n_samples):
        slot = _get_active_slot_idx(occupancy[i], occ_threshold)
        if slot is None:
            continue
        # Use ground-truth TF if available, otherwise predicted
        if tf_idx is not None:
            ti = int(tf_idx[i])
        else:
            ti = int(tf_preds[i, slot])
        if ti < len(tf_names):
            tf_attentions[tf_names[ti]].append(attention[i, slot])

    # Compute mean attention per TF
    tf_mean_attn = {}
    for tn, attns in tf_attentions.items():
        tf_mean_attn[tn] = np.mean(attns, axis=0)

    # Pair analysis
    for pair in COOPERATIVE_PAIRS:
        tf_a, tf_b = pair["tf_a"], pair["tf_b"]
        if tf_a not in tf_attentions or tf_b not in tf_attentions:
            continue

        attns_a = tf_attentions[tf_a]
        attns_b = tf_attentions[tf_b]

        # Cross-sample cosine: mean(cos(a_i, b_j)) for random pairs
        n_pairs = min(500, len(attns_a) * len(attns_b))
        rng = np.random.RandomState(42)
        cross_cosines = []
        for _ in range(n_pairs):
            ai = rng.randint(len(attns_a))
            bi = rng.randint(len(attns_b))
            c = 1 - cosine_dist(attns_a[ai], attns_b[bi])
            cross_cosines.append(float(c))

        # Mean-vs-mean cosine
        mean_cos = 1 - cosine_dist(tf_mean_attn[tf_a], tf_mean_attn[tf_b])

        # Control: cosines between tf_a and non-partner TFs
        control_cosines = []
        for other_tf in tf_names:
            if other_tf in (tf_a, tf_b) or other_tf not in tf_mean_attn:
                continue
            c = 1 - cosine_dist(tf_mean_attn[tf_a], tf_mean_attn[other_tf])
            control_cosines.append(float(c))

        results[pair["name"]] = {
            "biology": pair["biology"],
            "n_samples_a": len(attns_a),
            "n_samples_b": len(attns_b),
            "mean_attention_cosine": float(mean_cos),
            "cross_sample_cosine": _summarize(cross_cosines),
            "control_cosines": _summarize(control_cosines),
            "partner_vs_control_diff": (
                float(mean_cos - np.mean(control_cosines))
                if control_cosines else None
            ),
        }
    return results


def slot_utilization_analysis(outputs, tf_names, tf_idx=None, occ_threshold=0.3):
    """Which slots does each TF preferentially use?

    Shows whether cooperative TFs share slot preferences (indicating
    the model learned similar representations for them).
    """
    occupancy = outputs['occupancy']
    tf_preds = outputs['tf_logits'].argmax(axis=-1)
    n_slots = occupancy.shape[1]

    # Build slot usage histograms per TF
    slot_usage = defaultdict(lambda: np.zeros(n_slots))
    for i in range(len(occupancy)):
        slot = _get_active_slot_idx(occupancy[i], occ_threshold)
        if slot is None:
            continue
        if tf_idx is not None:
            ti = int(tf_idx[i])
        else:
            ti = int(tf_preds[i, slot])
        if ti < len(tf_names):
            slot_usage[tf_names[ti]][slot] += 1

    # Normalize to distributions
    slot_dist = {}
    for tn, counts in slot_usage.items():
        total = counts.sum()
        slot_dist[tn] = counts / total if total > 0 else counts

    results = {}

    # Pair slot overlap (Jensen-Shannon divergence between slot distributions)
    for pair in COOPERATIVE_PAIRS:
        tf_a, tf_b = pair["tf_a"], pair["tf_b"]
        if tf_a not in slot_dist or tf_b not in slot_dist:
            continue

        da, db = slot_dist[tf_a], slot_dist[tf_b]
        m = (da + db) / 2
        eps = 1e-10
        jsd = 0.5 * np.sum(da * np.log((da + eps) / (m + eps))) + \
              0.5 * np.sum(db * np.log((db + eps) / (m + eps)))

        # Top-3 slot overlap
        top3_a = set(np.argsort(-da)[:3])
        top3_b = set(np.argsort(-db)[:3])
        overlap = len(top3_a & top3_b)

        # Control: average JSD with non-partner TFs
        control_jsds = []
        for other_tf in tf_names:
            if other_tf in (tf_a, tf_b) or other_tf not in slot_dist:
                continue
            do = slot_dist[other_tf]
            mo = (da + do) / 2
            j = 0.5 * np.sum(da * np.log((da + eps) / (mo + eps))) + \
                0.5 * np.sum(do * np.log((do + eps) / (mo + eps)))
            control_jsds.append(float(j))

        results[pair["name"]] = {
            "biology": pair["biology"],
            "jsd": float(jsd),
            "top3_slot_overlap": overlap,
            "slot_dist_a": {tf_a: da.tolist()},
            "slot_dist_b": {tf_b: db.tolist()},
            "control_jsd": _summarize(control_jsds),
            "partner_more_similar": bool(jsd < np.mean(control_jsds)) if control_jsds else None,
        }

    # Full TF-TF slot similarity matrix
    all_tfs = sorted(slot_dist.keys())
    sim_matrix = {}
    for i, ta in enumerate(all_tfs):
        for j, tb in enumerate(all_tfs):
            if i >= j:
                continue
            c = 1 - cosine_dist(slot_dist[ta], slot_dist[tb])
            sim_matrix[f"{ta}+{tb}"] = float(c)

    results["slot_similarity_matrix"] = sim_matrix
    return results


def position_distribution_analysis(outputs, tf_names, tf_idx=None, occ_threshold=0.3):
    """Compare binding position distributions between cooperative TFs.

    Cooperative TFs that bind nearby motifs should show similar or
    systematically offset position distributions.
    """
    occupancy = outputs['occupancy']
    positions = outputs['positions']
    tf_preds = outputs['tf_logits'].argmax(axis=-1)

    tf_positions = defaultdict(list)
    for i in range(len(occupancy)):
        slot = _get_active_slot_idx(occupancy[i], occ_threshold)
        if slot is None:
            continue
        if tf_idx is not None:
            ti = int(tf_idx[i])
        else:
            ti = int(tf_preds[i, slot])
        if ti < len(tf_names):
            pos_bp = float(positions[i, slot, 0]) * 2000
            tf_positions[tf_names[ti]].append(pos_bp)

    results = {}
    for pair in COOPERATIVE_PAIRS:
        tf_a, tf_b = pair["tf_a"], pair["tf_b"]
        if tf_a not in tf_positions or tf_b not in tf_positions:
            continue

        pos_a = np.array(tf_positions[tf_a])
        pos_b = np.array(tf_positions[tf_b])

        # KS test: are position distributions similar?
        ks_stat, ks_p = ks_2samp(pos_a, pos_b)

        results[pair["name"]] = {
            "biology": pair["biology"],
            "n_a": len(pos_a), "n_b": len(pos_b),
            "mean_pos_a": float(np.mean(pos_a)),
            "mean_pos_b": float(np.mean(pos_b)),
            "std_pos_a": float(np.std(pos_a)),
            "std_pos_b": float(np.std(pos_b)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "distributions_similar": bool(ks_p > 0.05),
        }
    return results


def profile_shape_similarity(outputs, gt_profiles, tf_names, tf_idx,
                             occ_threshold=0.3):
    """Compare predicted profile shapes between cooperative TF pairs.

    If MYC and MAX produce similar ChIP profiles (both at E-boxes),
    BEACON should predict similar profile shapes for them.
    """
    occupancy = outputs['occupancy']
    pred_profiles = outputs['profile']

    tf_profiles = defaultdict(list)
    for i in range(len(occupancy)):
        slot = _get_active_slot_idx(occupancy[i], occ_threshold)
        if slot is None:
            continue
        ti = int(tf_idx[i])
        if ti < len(tf_names):
            tf_profiles[tf_names[ti]].append(pred_profiles[i])

    results = {}
    for pair in COOPERATIVE_PAIRS:
        tf_a, tf_b = pair["tf_a"], pair["tf_b"]
        if tf_a not in tf_profiles or tf_b not in tf_profiles:
            continue

        profs_a = np.array(tf_profiles[tf_a])
        profs_b = np.array(tf_profiles[tf_b])

        # Mean profile shapes
        mean_a = np.mean(profs_a, axis=0)
        mean_b = np.mean(profs_b, axis=0)

        # Profile cosine similarity
        if np.std(mean_a) > 0 and np.std(mean_b) > 0:
            cos_sim = 1 - cosine_dist(mean_a, mean_b)
            pearson = float(np.corrcoef(mean_a, mean_b)[0, 1])
        else:
            cos_sim = 0.0
            pearson = 0.0

        # Control: similarity with non-partner TFs
        control_pearson = []
        for other_tf in tf_names:
            if other_tf in (tf_a, tf_b) or other_tf not in tf_profiles:
                continue
            mean_o = np.mean(np.array(tf_profiles[other_tf]), axis=0)
            if np.std(mean_o) > 0 and np.std(mean_a) > 0:
                r = float(np.corrcoef(mean_a, mean_o)[0, 1])
                control_pearson.append(r)

        results[pair["name"]] = {
            "biology": pair["biology"],
            "n_a": len(profs_a), "n_b": len(profs_b),
            "mean_profile_cosine": float(cos_sim),
            "mean_profile_pearson": pearson,
            "control_pearson": _summarize(control_pearson),
            "partner_more_similar": (
                bool(pearson > np.mean(control_pearson))
                if control_pearson else None
            ),
        }
    return results
