#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment.yml || true
conda activate CGR-GS

pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization/
