#!/usr/bin/env bash
set -e
ENV_NAME="gtrends-ocr"
echo "🔍 Removing python=3.13 pin..."
conda config --remove-key pinned_packages 2>/dev/null || true
conda config --remove pinned_packages python=3.13 2>/dev/null || true
for condarc in ~/.condarc ~/miniconda3/.condarc ~/anaconda3/.condarc; do
  [[ -f "$condarc" ]] && sed -i '' '/pinned_packages/d; /python=3.13/d' "$condarc" 2>/dev/null || true
done
rm -f ~/miniconda3/envs/gtrends-ocr/conda-meta/pinned ~/anaconda3/envs/gtrends-ocr/conda-meta/pinned 2>/dev/null || true

echo "🧹 Clearing cache & removing old env..."
conda clean --all -y 2>/dev/null || true
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

echo "📦 Creating environment with Python 3.11..."
conda create -n "$ENV_NAME" python=3.11 -y
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "🚀 Installing dependencies for Qwen3-VL-2B-Instruct..."
pip install --upgrade pip
# Transformers >= 4.50 required for full Qwen3-VL support
pip install "transformers>=4.50" accelerate
pip install torch torchvision
pip install pillow opencv-python numpy qwen-vl-utils
pip install pydantic tiktoken regex packaging huggingface-hub requests

echo ""
echo "✅ Environment ready!"
echo ""
echo "📝 Usage:"
echo "   conda activate $ENV_NAME"
echo "   ./gtrends-ocr.sh test --workers 4"
echo ""
echo "💡 Note:"
echo "   - Model (~4.5GB) downloads ONCE to ./models/ directory"
echo "   - First run: downloads model (~2-5 min depending on connection)"
echo "   - Subsequent runs: load instantly from disk (offline)"
echo "   - Requires ~4-6GB RAM during inference"