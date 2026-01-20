# BEACON Training Results

**Binding Event Attention-based Compositional Object Network**

Training Date: January 16-17, 2026

---

## Model Architecture

| Component | Parameters |
|-----------|------------|
| **Total** | **850,442** |
| Backbone | 424,192 |
| Slot Attention | 284,800 |
| Profile Head | 49,923 |
| Position Head | 33,282 |
| TF Head | 33,282 |
| Occupancy Head | 24,833 |
| Positional Encoding | 130 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (max) | 100 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |
| Mixed Precision (AMP) | Enabled |
| GPUs | 3x NVIDIA GPU |
| Early Stopping Patience | 15 epochs |

---

## Best Run Results (Run 1)

**Run ID:** `beacon_20260116_051830`
**Training Time:** 8.21 hours
**Epochs:** 80 (best model), 95 (early stop)
**Test Samples:** 3,932

### Profile Reconstruction

| Metric | Value |
|--------|-------|
| Pearson Correlation | **0.9994** |
| Spearman Correlation | 0.9999 |
| Cosine Similarity | 0.9995 |
| MSE | 0.0021 |
| MAE | 0.0140 |
| KL Divergence | 0.0013 |

### Binding Site Detection

| Metric | Value |
|--------|-------|
| F1 Score | **1.0000** |
| Precision | 1.0000 |
| Recall | 1.0000 |
| Average Precision | 1.0000 |

### Position Prediction

| Metric | Value |
|--------|-------|
| MAE | **0.0006** |
| RMSE | 0.0178 |

### Transcription Factor Classification

| Metric | Value |
|--------|-------|
| Accuracy | **94.81%** |
| F1 Score | 0.9347 |
| Precision | 0.9413 |
| Recall | 0.9287 |

### Slot Utilization

| Metric | Value |
|--------|-------|
| Slot Diversity | **0.5483** |
| Slot Utilization | 0.0625 |
| Avg Inter-site Distance | 0.3101 |
| Avg Unique TFs/Sample | 1.03 |

---

## Run Comparison

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| **Profile Pearson** | 0.9994 | 0.9979 |
| **Site F1** | 1.0000 | 1.0000 |
| **TF Accuracy** | 94.81% | 93.46% |
| **Position MAE** | 0.0006 | 0.0540 |
| **Slot Diversity** | 0.5483 | 0.3021 |
| Training Time | 8.21 hrs | 3.74 hrs |
| Best Epoch | 80 | 41 |
| Early Stop Epoch | 95 | 56 |

Run 1 achieved better results due to:
1. Longer training (80 vs 41 epochs to best model)
2. Higher slot diversity (better specialization)
3. More precise position predictions

---

## Key Findings

### Strengths
- **Perfect site detection** (F1 = 1.0) - model reliably identifies binding sites
- **Excellent profile reconstruction** (r = 0.999) - accurate signal prediction
- **Strong TF classification** (~95%) - correctly identifies transcription factors
- **Sub-base position accuracy** (MAE < 1bp) - precise localization

### Model Capabilities
- Discovers discrete binding sites from continuous profiles
- Assigns TF identity to each detected site
- Learns compositional structure via slot attention
- Generalizes well to held-out test data

---

## Output Files

Best model checkpoint:
```
/home/bcheng/beacon/outputs/beacon_20260116_051830/best_model.pt
```

Training logs:
```
/home/bcheng/beacon/outputs/beacon_20260116_051830/training.log
```

Full metrics history:
```
/home/bcheng/beacon/outputs/beacon_20260116_051830/history.json
```

TensorBoard logs:
```
/home/bcheng/beacon/outputs/beacon_20260116_051830/tensorboard/
```

---

## Bug Fixes Applied During Training

1. **BCE/AMP Incompatibility** - Replaced `binary_cross_entropy` with MSE in `OccupancyLoss` and `BindingSiteSupervisionLoss` for AMP compatibility

2. **Silent Loss Fallback** - Removed try/except block in trainer that masked loss computation errors

3. **TF AUC Metric** - Added check for both classes before computing ROC-AUC (returns 0.5 baseline for single-class batches)

4. **Occupancy Entropy NaN** - Added `nan_to_num` handling for edge cases in entropy calculation

---

---

## Phase 1 Validation: Hard Synthetic Data

**Run ID:** `beacon_20260118_061201`
**Training Date:** January 18-19, 2026
**Training Time:** 37.39 hours
**Epochs:** 150 (full training)
**GPUs:** 3x NVIDIA RTX 2080 Ti

### Hard Synthetic Configuration

| Parameter | Easy (Original) | Hard (Phase 1) |
|-----------|-----------------|----------------|
| TF Classes | 10 | **50** |
| Sites per Sequence | 1-2 | **3-8** |
| Overlapping Sites | No | **Yes** |
| Noise (σ) | 0.0 | **0.15** |
| Sequence Length | 1000 bp | **2000 bp** |
| Training Samples | 10,000 | **100,000** |

### Final Test Results (38 Metrics)

#### Profile Reconstruction

| Metric | Easy | Hard | Delta |
|--------|------|------|-------|
| Pearson Correlation | 0.9994 | 0.5232 | **-48%** |
| Spearman Correlation | 0.9999 | 0.3759 | -62% |
| Cosine Similarity | 0.9995 | 0.7170 | -28% |
| AUROC | - | 0.8210 | - |
| MSE | 0.0021 | 0.0281 | +13x |
| MAE | 0.0140 | 0.1184 | +8x |
| KL Divergence | 0.0013 | 0.6475 | +498x |

#### Binding Site Detection

| Metric | Easy | Hard | Delta |
|--------|------|------|-------|
| F1 Score | **1.0000** | 0.3295 | **-67%** |
| Precision | 1.0000 | 0.3264 | -67% |
| Recall | 1.0000 | 0.3327 | -67% |
| Average Precision | 1.0000 | 0.3264 | -67% |

#### Position Prediction

| Metric | Easy | Hard | Delta |
|--------|------|------|-------|
| MAE | 0.0006 | 22.59 bp | +37,650x |
| RMSE | 0.0178 | 26.79 bp | +1,505x |

#### Transcription Factor Classification

| Metric | Easy | Hard | Delta |
|--------|------|------|-------|
| Accuracy | **94.81%** | 12.28% | **-82%** |
| F1 Score | 0.9347 | 0.0673 | -93% |
| Precision | 0.9413 | 0.0727 | -92% |
| Recall | 0.9287 | 0.0907 | -90% |
| Top-3 Accuracy | - | 0.00% | - |

*Note: Random baseline for 50 classes = 2%, so 12.28% is 6x better than random*

#### Slot Utilization

| Metric | Easy | Hard |
|--------|------|------|
| Avg Slots Used | ~1 | 5.50 |
| Slot Diversity | 0.5483 | 0.7027 |
| Slot Utilization | 0.0625 | 0.3441 |
| Avg Unique TFs/Sample | 1.03 | 3.88 |
| Motif Coverage | - | 77.5% |

#### Compositional Metrics

| Metric | Value |
|--------|-------|
| Binding Count MAE | 2.42 |
| Grammar Score | 0.299 |
| Composition Accuracy | 0.00% |

---

## Analysis: Why Hard Synthetic Failed

### Key Observations

1. **Massive complexity increase**: 50 TFs (5x) × 3-8 sites (4x) × overlap × noise = ~20-40x harder task

2. **Site detection collapsed**: F1 dropped from 1.0 to 0.33, meaning model misses 2/3 of binding sites

3. **TF classification degraded**: 95% → 12% accuracy, though still 6x better than random (2%)

4. **Position prediction failed**: Sub-base precision → 23bp error (>2x typical motif width)

5. **Profile reconstruction moderate**: AUROC 0.82 shows some signal learned, but correlation poor

### Likely Failure Modes

1. **Overlapping sites**: Model may not handle site overlap well - slots may "merge" nearby sites

2. **TF confusion**: 50 similar motif families may be too fine-grained for current architecture

3. **Capacity limitation**: 850K parameters may be insufficient for 50-class, 8-site task

4. **Slot attention saturation**: 16 slots for up to 8 sites may not provide enough redundancy

### Recommendations Before Real Data

1. **Intermediate difficulty test**: Try 20 TFs, 2-4 sites, no overlap to find failure threshold

2. **Increase model capacity**: Double backbone channels, more slot attention iterations

3. **Architecture modifications**:
   - Add explicit overlap handling
   - Hierarchical slot attention
   - Stronger position encoding

4. **Curriculum learning**: Train on easy → medium → hard progressively

---

## Conclusions

### What Works
- BEACON achieves **perfect performance on simple synthetic data** (10 TFs, 1-2 non-overlapping sites)
- Slot attention mechanism successfully discovers discrete binding events
- Multi-task learning (profile + sites + TF identity) is viable

### What Doesn't Work (Yet)
- **Does not scale** to realistic complexity (50 TFs, overlapping sites, noise)
- Current architecture/capacity insufficient for hard tasks
- Needs significant improvements before real ENCODE data

### Status: **Not Ready for Phase 2**

The model requires architectural improvements or hyperparameter tuning before proceeding to real ChIP-seq data or benchmarking against BPNet/ChromBPNet.

---

## Output Files

### Easy Synthetic (Best)
```
/home/bcheng/beacon/outputs/beacon_20260116_051830/best_model.pt
```

### Hard Synthetic
```
/home/bcheng/beacon/outputs/beacon_20260118_061201/best_model.pt
/home/bcheng/beacon/outputs/beacon_20260118_061201/training.log
/home/bcheng/beacon/outputs/beacon_20260118_061201/history.json
```

---

## Next Steps

1. **Diagnose failure modes** - Analyze predictions on hard synthetic to understand what's breaking
2. **Intermediate difficulty** - Find the complexity threshold where model starts failing
3. **Architecture search** - Test increased capacity and modified attention mechanisms
4. **Re-evaluate** - Only proceed to real data after achieving >0.7 Site F1 on hard synthetic
