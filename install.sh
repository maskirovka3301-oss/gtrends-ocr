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

echo "🚀 Installing HuggingFace & vision model dependencies..."
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install openai requests pillow numpy opencv-python

echo "🔑 Setting up HuggingFace API access..."
echo ""
echo "⚠️  IMPORTANT: You need a HuggingFace API token to use Kimi-K2.6"
echo "Get one at: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace API token (or press Enter to set later): " HF_TOKEN
if [[ -n "$HF_TOKEN" ]]; then
    echo "export HUGGINGFACE_API_KEY='$HF_TOKEN'" >> ~/.bashrc
    export HUGGINGFACE_API_KEY="$HF_TOKEN"
    echo "✅ Token saved to ~/.bashrc and current session"
else
    echo "⚠️  You'll need to set HUGGINGFACE_API_KEY environment variable before running"
    echo "   export HUGGINGFACE_API_KEY='your_token_here'"
fi

echo "✅ Environment ready! Run: conda activate $ENV_NAME"
echo ""
echo "📝 To use the script, set your API key:"
echo "   export HUGGINGFACE_API_KEY='hf_xxxxx'"
echo "   ./run.sh test"