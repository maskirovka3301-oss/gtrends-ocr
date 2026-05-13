#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1
python gtrends-ocr.py "$1" --workers 8

