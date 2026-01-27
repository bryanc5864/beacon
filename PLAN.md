# BEACON Research Plan

**Last Updated:** January 25, 2026

---

## Where We Are

### Completed Work (Weeks 1-2)

| Phase | Status | Key Result |
|-------|--------|------------|
| Architecture design | ✅ Done | 850K param slot attention model |
| Easy synthetic (10 TFs) | ✅ Done | Perfect: F1=1.0, TF Acc=95%, Profile r=0.999 |
| Hard synthetic (50 TFs) | ✅ Done | Failed: F1=0.33, TF Acc=12% |
| Ablation study (8 experiments) | ✅ Done | Overlap = primary bottleneck |
| Overlap separation loss (V2) | ✅ Done | Fixed: overlap F1 0.10 → 0.52 |
| Profile reconstruction fix | ✅ Done | Fixed: r=0.01 → 0.62 with more epochs |
| Multi-objective trade-off analysis | ✅ Done | Feature drift prevents simultaneous optimization |
| Two-stage training attempt | ✅ Done | Failed: feature drift, not weight drift |

### Key Insights

1. **BEACON works on simple data** - Proof of concept validated
2. **Overlap separation loss** - Novel contribution, works well
3. **25+ TF classification** - Not solvable with current architecture in multi-task setup
4. **Real data will be different** - 1 TF per experiment, not 25-class classification

---

## Current Phase: Real ENCODE Data (Phase 2)

### Phase 2A: Single-TF CTCF (This Week)

**Goal:** Demonstrate BEACON works on real ChIP-seq data

#### Data
| Item | Source |
|------|--------|
| TF | CTCF |
| Cell Type | K562 |
| Data Type | ChIP-seq peaks + signal (bigWig) |
| Genome | hg38 (already downloaded) |
| Sequence Length | 2000 bp |
| Split | chr1-17 train / chr18-19 val / chr20-22,chrX test |

#### Pipeline Steps
1. Download CTCF ChIP-seq data from ENCODE
2. Call peaks with MACS2 (or use ENCODE-provided peaks)
3. Extract 2kb windows around peak summits
4. Generate profiles from bigWig signal tracks
5. Create HDF5 datasets with chromosome splits
6. Train BEACON (single TF, binary detection)

#### Target Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| Site F1 | > 0.70 | Can BEACON find real binding sites? |
| Position MAE | < 25 bp | Precise enough for motif analysis |
| Profile Pearson | > 0.50 | Reasonable reconstruction |

---

### Phase 2B: Multi-TF K562 (Next Week)

**Goal:** 5-10 TFs in same cell type, test TF discrimination

#### TF Panel
| TF | Motif Family | Why |
|----|-------------|-----|
| CTCF | Zinc finger | Best characterized, clear motif |
| GATA1 | GATA | Distinct family, strong signal |
| TAL1 | bHLH | Hematopoietic master regulator |
| MYC | bHLH | Same family as TAL1 (test confusion) |
| MAX | bHLH | Dimerizes with MYC |
| SPI1 | ETS | Different family |
| CEBPB | bZIP | Different family |

#### Key Test
Can BEACON distinguish:
- GATA1 vs MYC? (different families - should be easy)
- MYC vs MAX? (same family - should be hard)
- TAL1 vs MYC? (both bHLH - intermediate)

---

### Phase 3: Benchmarking vs BPNet (Week 3-4)

**Goal:** Head-to-head comparison on same data

#### Comparison Framework
| Capability | BPNet | BEACON |
|------------|-------|--------|
| Profile prediction | Compare Pearson r | Compare Pearson r |
| Site detection | TF-MoDISco (post-hoc) | Direct (native) |
| TF identity | Not native | Native |
| Speed | Slow (attribution needed) | Fast (single forward pass) |
| Interpretability | Post-hoc | Inherent |

#### Benchmarks
1. **Profile correlation** - Same test set, compare Pearson/Spearman
2. **Motif recall** - Known CTCF motif, compare detection rates
3. **Site-level F1** - Against ChIP-seq peaks as ground truth
4. **Speed** - Inference time comparison

---

### Phase 4: Novel Capabilities (Week 4-5)

**Goal:** Demonstrate what BEACON can do that BPNet cannot

1. **Direct binding site enumeration** - Count sites per sequence
2. **Compositional reasoning** - Which TFs co-bind within 100bp?
3. **Variant effect attribution** - Which specific binding site does a SNP affect?
4. **Unsupervised motif discovery** - Cluster slot attention patterns

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| Week 3 (Jan 27-31) | 2A: CTCF single-TF | Working real data pipeline, initial results |
| Week 4 (Feb 3-7) | 2B: Multi-TF K562 | 5-10 TF results, TF discrimination |
| Week 5 (Feb 10-14) | 3: BPNet benchmark | Head-to-head comparison |
| Week 6 (Feb 17-21) | 4: Novel capabilities | Unique BEACON demonstrations |
| Week 7-8 (Feb 24-Mar 7) | Paper writing | Methods, results, figures |

---

## Architecture Decisions (Locked In)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backbone | Dilated CNN (128 dim, 4 layers) | Lightweight, sufficient for 2kb |
| Slot attention | 16 slots, 128 dim, 3 iterations | Handles up to 8 sites comfortably |
| TF head | Linear classifier from slot embeddings | Simple, proven on synthetic |
| Position head | Gaussian mean/std prediction | Continuous positions |
| Profile head | Slot-based Gaussian reconstruction | Each slot contributes a peak |
| Loss | BEACONLoss + overlap separation | Multi-task with overlap fix |

---

## Deferred Work (Post-Paper)

| Item | Why Deferred |
|------|-------------|
| 50 TF classification | Feature drift problem, not needed for paper |
| Hierarchical TF classification | Future work section |
| Curriculum learning | Synthetic-specific, not relevant for real data |
| Larger model (3-5M params) | Current model sufficient for proof of concept |
| HyenaDNA backbone | Requires more compute, incremental benefit |

---

## Files and Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/generate_synthetic_data.py` | Generate synthetic binding data | ✅ Done |
| `scripts/train_validation.py` | Train on synthetic data | ✅ Done |
| `scripts/curriculum_trainer.py` | V1 ablation experiments | ✅ Done |
| `scripts/curriculum_trainer_v2.py` | V2 with overlap fix + two-stage | ✅ Done |
| `scripts/process_encode.py` | Process ENCODE ChIP-seq data | ✅ Written, needs testing |
| `scripts/download_encode.py` | Download ENCODE files | ✅ Written |
| `scripts/train.py` | Main training script | ✅ Done |
| `scripts/test_model.py` | Model evaluation | ✅ Done |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CTCF data too easy (single TF) | Low | Low | Expand to multi-TF quickly |
| Real data much harder than synthetic | Medium | High | Start with strong peaks only |
| BPNet significantly outperforms | Medium | Medium | Focus on interpretability angle |
| Slot attention doesn't transfer to real data | Low | High | Architecture already validated on synthetic |
| Compute insufficient | Low | Low | Single GPU sufficient for current model |
