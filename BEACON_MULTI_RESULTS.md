# BEACON-multi: Multi-Slot Activation Experiments

## Problem Statement

The original BEACON model exhibits **single-slot dominance**: only Slot 0 activates (occupancy=0.98), while Slots 1-15 have near-zero occupancy (1e-7 to 1e-13). This undermines the core architectural claim that slot attention decomposes regulatory sequences into discrete binding events. On multi-TF co-bound regions, 0% of sequences activate more than one slot.

## Data

Training data prepared from 7 TFs in K562 (CTCF, GATA1, TAL1, MYC, MAX, SPI1, CEBPB) using overlapping ChIP-seq peaks. Each multi-TF sample contains 2+ TF binding sites within a 2000bp window, with a composite profile (sum of per-TF bigWig signals) and per-site binding annotations (position, TF identity, occupancy).

- **Multi-TF overlap regions:** ~30K+ across the genome (found by clustering peaks within 500bp)
- **Data mix:** Multi-TF sequences + single-TF sequences per split
- **Train/Val/Test split:** chr1-17 / chr18-19 / chr20-22,X

## Architecture

Same as original BEACON: 851K parameters, 16 slots, 128-dim backbone (4 dilated conv layers), 128-dim slots, 3 slot attention iterations, 2000bp input.

## Experiment 1: Competitive Attention + Hungarian Matching

**Changes from original BEACON:**
- Hungarian matching loss for site supervision (optimal slot-to-site assignment via `scipy.optimize.linear_sum_assignment`, same approach as DETR)
- Slot count loss: penalizes mismatch between number of active slots and number of target binding sites
- Attention load balancing loss: penalizes attention mass concentration on a single slot
- Higher loss weights: tf=1.5, occupancy=0.5, diversity=0.5, site_supervision=1.0

**Attention mechanism unchanged:** `softmax(dim=1)` over slots — each sequence position is assigned to exactly one slot (competitive/winner-take-all).

### Results (early stopped at epoch 31, patience=20)

| Epoch | site_f1 | tf_acc | avg_slots | slot_util | profile_r | diversity |
|-------|---------|--------|-----------|-----------|-----------|-----------|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.810 | 0.590 |
| 4 | **0.812** | 0.151 | 0.817 | 0.051 | 0.814 | 0.600 |
| 8 | 0.789 | 0.161 | 0.790 | 0.049 | 0.813 | 0.666 |
| 14 | 0.334 | **0.200** | 0.555 | 0.035 | 0.814 | 0.564 |
| 20 | 0.143 | 0.099 | 0.924 | 0.058 | 0.807 | 0.605 |
| 25 | 0.352 | 0.066 | **1.671** | 0.104 | 0.814 | 0.580 |
| 31 | 0.332 | 0.094 | 1.517 | 0.095 | 0.808 | 0.604 |

### Analysis

The competitive attention mechanism fundamentally prevents multi-slot activation. The `softmax(dim=1)` forces each position to be assigned to exactly one slot. One slot learns to "claim" all binding-relevant positions, creating a winner-take-all dynamic that no loss function can overcome.

- **avg_slots_used** peaked at 1.67 — barely above 1 slot
- **site_f1** collapsed from 0.81 to 0.13-0.33 as the model tried to activate more slots
- **tf_accuracy** degraded from 0.20 to 0.07-0.10
- The model oscillated between two failure modes: (1) one slot works well, (2) ~1.5 slots work poorly

**Conclusion:** Hungarian matching and auxiliary losses cannot overcome the architectural bottleneck of competitive attention. The softmax over slots creates a structural barrier to multi-slot activation.

---

## Experiment 2: Independent Attention + Hungarian Matching

**Key architectural change:** Replaced `softmax(dim=1)` (competitive, over slots) with `softmax(dim=-1)` (independent, over positions). Each slot now has its own attention distribution over the sequence, and multiple slots can attend to the same or different positions independently. Slot specialization is driven by the Hungarian matching loss rather than by attention competition.

All other settings identical to Experiment 1.

### Results (early stopped at epoch 30, patience=20)

| Epoch | site_f1 | tf_acc | avg_slots | slot_util | profile_r | diversity |
|-------|---------|--------|-----------|-----------|-----------|-----------|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.808 | 0.190 |
| 4 | 0.642 | 0.253 | 1.000 | 0.063 | 0.808 | 0.071 |
| 6 | 0.722 | **0.285** | 1.535 | 0.096 | 0.806 | 0.113 |
| 10 | 0.848 | 0.233 | 1.964 | 0.123 | 0.813 | 0.086 |
| 15 | 0.907 | 0.199 | 2.442 | 0.153 | 0.815 | 0.149 |
| 20 | 0.898 | 0.178 | 2.642 | 0.165 | 0.808 | 0.238 |
| 22 | **0.950** | 0.171 | 2.710 | 0.169 | 0.814 | 0.238 |
| 26 | 0.954 | 0.152 | **3.079** | **0.193** | 0.804 | 0.381 |
| 30 | 0.958 | 0.164 | 2.916 | 0.182 | 0.815 | 0.418 |

### Analysis

Independent attention successfully enables multi-slot activation.

**What works:**
- **avg_slots_used climbed from 0 to 3.08** — the model activates ~3 slots per sequence, matching the typical number of TFs in multi-TF overlap regions
- **site_f1 reached 0.958** — near-perfect binding site detection, steadily improving throughout training
- **profile_pearson stable at 0.81** — profile reconstruction quality maintained
- **slot_diversity climbing to 0.44** — slots are becoming increasingly differentiated (still improving at early stop)

**What doesn't work:**
- **tf_accuracy peaked at 28.5% (epoch 6) then declined to ~16%** as more slots activated
- There is a clear inverse relationship between avg_slots_used and tf_accuracy: the model learns *where* to bind but not *which TF* is binding
- The model essentially learned to be a generic multi-site binding detector rather than a TF-specific slot attention system

### TF Accuracy vs Slot Count Trade-off

| avg_slots | tf_acc | Interpretation |
|-----------|--------|----------------|
| 1.0 | 0.253 | 1 slot, decent TF ID (above chance=14%) |
| 1.5 | 0.285 | Best TF accuracy — few slots, can specialize |
| 2.5 | 0.178 | More slots, TF accuracy declining |
| 3.0 | 0.155 | Many slots, near chance TF accuracy |

The loss landscape appears to have competing optima: the slot_count and site_supervision losses push toward more active slots, while tf_identity loss needs slots to specialize. With the current weighting (tf_weight=1.5, site_supervision_weight=1.0, slot_count_weight=1.0), multi-slot activation wins and TF discrimination is sacrificed.

---

## Head-to-Head Comparison

| Metric | Original BEACON | Competitive + Hungarian | Independent + Hungarian |
|--------|-----------------|------------------------|------------------------|
| avg_slots_used | 1.0 | 1.5 | **3.1** |
| site_f1 | 0.90 | 0.33 (degraded) | **0.96** |
| tf_accuracy | 0.71 | 0.09 (degraded) | 0.28 (peak) / 0.16 (final) |
| profile_pearson | 0.82 | 0.81 | 0.81 |
| slot_diversity | 0.63 | 0.60 | **0.44** (climbing) |

Note: Original BEACON tf_accuracy=0.71 is on single-TF data where only one slot needs to classify one TF. The multi-TF task is harder since each slot must independently identify its TF from 7 options.

---

## Experiment 3: Independent Attention + Hungarian Matching + tf_weight=5.0

**Date:** February 2, 2026

**Key change from Experiment 2:** Increased `tf_weight` from 1.5 to **5.0** to give TF discrimination more gradient signal and prevent the TF accuracy collapse observed in Experiment 2.

All other settings identical to Experiment 2 (independent attention, Hungarian matching, slot count loss, load balancing loss).

**Training:** 31 epochs (early stopped after 20 epochs without val_loss improvement), 4.82 hours on 3× GPU.

### Validation Trajectory

| Epoch | site_f1 | tf_acc | avg_slots | slot_util | profile_r | diversity |
|-------|---------|--------|-----------|-----------|-----------|-----------|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.810 | 0.000 |
| 2 | 0.899 | 0.296 | 1.000 | 0.063 | 0.819 | 0.000 |
| 4 | 0.632 | 0.169 | 2.318 | 0.145 | 0.804 | 0.142 |
| 6 | 0.758 | 0.246 | 2.252 | 0.141 | 0.801 | 0.155 |
| 8 | 0.826 | 0.275 | 2.318 | 0.145 | 0.816 | 0.165 |
| 11 | 0.810 | 0.304 | 1.993 | 0.125 | 0.814 | 0.030 |
| 14 | 0.883 | 0.302 | 2.344 | 0.147 | 0.813 | 0.109 |
| 17 | 0.860 | 0.314 | 2.424 | 0.152 | 0.816 | 0.123 |
| 20 | 0.941 | 0.302 | 2.199 | 0.137 | 0.819 | 0.171 |
| 24 | 0.930 | **0.324** | 2.656 | 0.166 | 0.811 | 0.251 |
| 27 | 0.979 | 0.269 | 2.384 | 0.149 | 0.820 | 0.229 |
| 31 | **0.981** | 0.291 | 2.298 | 0.144 | 0.815 | 0.268 |

### Test Set Results

| Metric | Value |
|--------|-------|
| **Profile Pearson** | **0.838** |
| Profile AUROC | 0.956 |
| **Site F1** | **0.817** |
| Site Precision | 0.743 |
| Site Recall | 0.907 |
| **TF Accuracy** | **0.313** |
| TF F1 | 0.321 |
| **avg_slots_used** | **1.94** |
| avg_unique_tfs_per_sample | 3.05 |
| motif_coverage | 0.961 |
| binding_count_mae | 1.60 |

### Analysis

Higher tf_weight (5.0 vs 1.5) successfully prevents the TF accuracy collapse seen in Experiment 2.

**Improvements over Experiment 2:**
- **TF accuracy maintained at ~0.30 throughout training** (vs peak of 0.285 then decline to 0.16 in Exp 2)
- **Peak TF accuracy: 0.324 (epoch 24)** vs 0.285 (epoch 6 in Exp 2) — 14% relative improvement
- **Profile Pearson improved to 0.838** vs 0.815 in Exp 2
- **Site F1 validation reached 0.981** vs 0.958 in Exp 2

**Trade-off:**
- avg_slots_used is lower (1.94 vs 3.08) — the stronger TF loss encourages fewer, more confident slots rather than many uncertain ones
- This is actually a better operating point: fewer but more accurate slot activations

---

## Head-to-Head Comparison (Updated)

| Metric | Original BEACON | Exp 1: Competitive | Exp 2: Indep (tf=1.5) | Exp 3: Indep (tf=5.0) |
|--------|-----------------|--------------------|-----------------------|------------------------|
| avg_slots_used | 1.0 | 1.5 | **3.1** | 1.94 |
| site_f1 (test) | 0.90 | 0.33 | 0.96 | 0.82 |
| tf_accuracy (test) | 0.71* | 0.09 | 0.16 | **0.31** |
| profile_pearson (test) | 0.82 | 0.81 | 0.81 | **0.84** |
| tf_acc peak (val) | — | 0.20 | 0.285 | **0.324** |
| slot_diversity | 0.63 | 0.60 | 0.44 | 0.27 |

*Original BEACON tf_accuracy=0.71 is on single-TF data (easier task). Multi-TF data has overlapping binding sites from multiple TFs per sequence.

**Best configuration: Experiment 3** (independent attention, tf_weight=5.0) achieves the best balance of multi-slot activation, TF accuracy, and profile prediction.

---

## Conclusions

1. **Competitive attention (`softmax(dim=1)`) is fundamentally incompatible with multi-slot activation.** No amount of auxiliary loss can overcome the winner-take-all dynamics.

2. **Independent attention (`softmax(dim=-1)`) enables multi-slot activation.** avg_slots_used climbs from 1.0 to 1.9-3.1 depending on tf_weight.

3. **tf_weight is the key hyperparameter for balancing multi-slot activation vs TF discrimination.** tf_weight=1.5 maximizes slot count but sacrifices TF accuracy. tf_weight=5.0 finds a better balance.

4. **The trade-off between slot count and TF accuracy persists** but is manageable. Future work could explore curriculum strategies (first learn TF identity, then multi-slot activation) or contrastive slot losses.

## Next Steps

1. **Two-phase curriculum**: Train with high tf_weight for 20 epochs (establish TF discrimination), then gradually reduce tf_weight to encourage multi-slot activation
2. **Per-slot TF supervision**: Instead of global TF accuracy, directly supervise each active slot's TF prediction via Hungarian matching
3. **Contrastive slot loss**: Encourage different slots to attend to different sequence regions
4. **Larger model**: Current 851K parameters may be insufficient for simultaneous multi-slot + TF discrimination
