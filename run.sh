#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1
export HUGGINGFACE_API_KEY='YOUR_HUGGING_FACE_API_KEY'
python gtrends-ocr.py "$1" --workers 8

