# BEACON: Binding Event Attention-based Compositional Object Network

**Interpretable Transcription Factor Binding Site Discovery via Slot Attention**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

BEACON is a deep learning framework for discovering transcription factor (TF) binding sites from ChIP-seq data using object-centric slot attention. Unlike traditional predict-then-interpret approaches (e.g., BPNet + DeepSHAP), BEACON directly decomposes regulatory sequences into discrete binding events, providing inherent interpretability while improving predictive accuracy.

### Key Features

- **Direct binding event enumeration**: Outputs structured events (position, TF identity, occupancy) without post-hoc attribution
- **419× faster** than BPNet + TF-MoDISco for genome-wide binding catalogues (20 ms vs 8.4 s per sequence)
- **State-of-the-art accuracy**: Mean profile Pearson r=0.901 vs BPNet r=0.813 on 7 TFs (K562)
- **Perfect site detection**: F1=1.000 (100% precision, 100% recall)
- **Cross-cell generalization**: 93% profile accuracy retained on unseen cell types
- **Interpretable architecture**: Slots self-organize by TF identity, recover canonical JASPAR motifs

## Architecture

BEACON combines four key components:

1. **Dilated CNN backbone** (2⁰–2⁸ dilation rates, 1029 bp receptive field)
2. **Helical positional encoding** (captures DNA's 10.5 bp periodicity for cooperative binding)
3. **Slot attention mechanism** (K=16 competing slots with independent attention)
4. **Per-slot prediction heads** (position, TF identity, occupancy)

Training uses Hungarian matching to solve permutation invariance, enabling end-to-end differentiable binding event decomposition.

## Installation

```bash
# Clone repository
git clone https://github.com/bryanc5864/beacon.git
cd beacon

# Create conda environment
conda create -n beacon python=3.10
conda activate beacon

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from beacon.model import BEACON

# Load pre-trained model
model = BEACON.from_pretrained("K562-7tf")
model.eval()

# Run inference on DNA sequence (2000 bp)
sequence = torch.randn(1, 2000, 4)  # [batch, length, channels]
outputs = model(sequence)

# Extract binding events
positions = outputs['positions']      # [batch, n_slots, 1]
tf_identity = outputs['tf_logits']    # [batch, n_slots, n_tfs]
occupancy = outputs['occupancy']      # [batch, n_slots, 1]
profile = outputs['profile']          # [batch, 1, length]
```

## Benchmarks

### Performance (K562, 7 TFs)

| Model | Profile r | TF Accuracy | Site Detection F1 | Speed (per seq) | Parameters |
|-------|-----------|-------------|-------------------|-----------------|------------|
| **BEACON** | **0.901** | **75.5%** | **1.000** | 20 ms | 851K |
| 7×BPNet | 0.813 | N/A | N/A | 8.4 s* | 997K |

\*Includes TF-MoDISco post-hoc attribution

### Cross-Cell Transfer (K562 → HepG2)

| Metric | K562 | HepG2 | Retention |
|--------|------|-------|-----------|
| Profile r | 0.904 | 0.840 | 93% |
| Site F1 | 1.000 | 0.870 | 87% |

## Citation

If you use BEACON in your research, please cite:

```bibtex
@inproceedings{beacon2026,
  title={Interpretable Transcription Factor Binding Site Discovery via Compositional Slot Attention},
  author={Cheng, Bryan},
  booktitle={ISMB 2026},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- GitHub Issues: [https://github.com/bryanc5864/beacon/issues](https://github.com/bryanc5864/beacon/issues)
- Email: bcheng@example.edu

## Acknowledgments

BEACON builds on slot attention from computer vision ([Locatello et al., NeurIPS 2020](https://arxiv.org/abs/2006.15055)) and Hungarian matching for permutation invariance ([Carion et al., ECCV 2020](https://arxiv.org/abs/2005.12872)).
