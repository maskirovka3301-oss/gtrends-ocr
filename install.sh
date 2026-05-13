#!/usr/bin/env bash
set -e

ENV_NAME="gtrends-ocr"

echo "🔍 Forcefully removing python=3.13 pin..."
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

echo "🚀 Installing packages via pip & conda..."
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install torch torchvision easyocr opencv-python Pillow numpy
conda install -c conda-forge tesseract -y  # System OCR binary
conda run -n "$ENV_NAME" pip install pytesseract  # Python binding

echo "✅ Verifying installation..."
conda run -n "$ENV_NAME" python -c "
import torch, easyocr, cv2, pytesseract
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ EasyOCR ready')
print(f'✅ Tesseract version: {pytesseract.get_tesseract_version()}')
"
echo "🎉 Environment ready! Run: conda activate $ENV_NAME"