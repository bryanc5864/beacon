#!/bin/bash
# Setup BPNet baseline for comparison with BEACON
# Phase 3: BPNet Benchmark

set -e

echo "========================================"
echo "BPNet Setup for BEACON Comparison"
echo "========================================"

# Create conda environment
echo ""
echo "Creating conda environment..."
conda create -n bpnet python=3.10 -y || true
eval "$(conda shell.bash hook)"
conda activate bpnet

# Install bioinformatics tools
echo ""
echo "Installing bioinformatics tools..."
conda install -y -c bioconda -c conda-forge samtools bedtools ucsc-bedgraphtobigwig

# Install BPNet and dependencies
echo ""
echo "Installing BPNet..."
pip install tensorflow
pip install bpnet-lite  # Lighter weight alternative that's easier to install

# Alternative: full bpnet (may have dependency issues)
# pip install bpnet

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import bpnetlite; print('bpnet-lite: OK')" || echo "bpnet-lite not available, trying bpnet..."
python -c "import bpnet; print('bpnet: OK')" || echo "Will need to use bpnet-lite"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate bpnet"
echo "  2. Prepare data: python scripts/prepare_bpnet_data.py"
echo "  3. Train: python scripts/train_bpnet.py"
