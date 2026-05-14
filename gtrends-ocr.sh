#!/bin/bash
# =============================================
# Pure macOS memory-limited launcher
# Uses taskpolicy -m (native, no ulimit)
# =============================================

set -eo pipefail

echo "=== Resource-limited launcher for gtrends-ocr.py ==="

# Calculate 75% of total RAM in MiB
TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
LIMIT_MB=$(( TOTAL_MB * 75 / 100 ))

echo "Total RAM detected : ${TOTAL_MB} MB"
echo "Setting memory limit: ${LIMIT_MB} MB (75%)"

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Launching with background priority + nice + memory limit..."

# taskpolicy -m <MiB> is the native way on macOS
taskpolicy -b -m "$LIMIT_MB" \
    nice -n 10 \
    python gtrends-ocr.py "$@" --workers 1
