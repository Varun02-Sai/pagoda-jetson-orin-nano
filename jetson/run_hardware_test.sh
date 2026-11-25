#!/bin/bash
# -----------------------------------------------
# Pagoda: Jetson Orin Nano Super Hardware Benchmark
# -----------------------------------------------
# Just run this one script on the Jetson board.
#
# What it does:
#   1. Installs dependencies (onnxruntime, numpy, matplotlib)
#   2. Checks that nvpmodel works
#   3. Runs all 5 models x 4 batch sizes x 3 power modes (MAXN_SUPER, 25W, 15W)
#   4. Saves results to results/jetson_hardware_results.csv
#
# How to run:
#   cd ~/pagoda/jetson
#   chmod +x run_hardware_test.sh
#   sudo bash run_hardware_test.sh
# -----------------------------------------------

set -e

echo "============================================================"
echo "  PAGODA: Jetson Orin Nano Super Hardware Benchmark"
echo "============================================================"

# Need root for nvpmodel
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Run as root (sudo bash run_hardware_test.sh)"
    exit 1
fi

# Make sure we're on a Jetson
if ! command -v nvpmodel &> /dev/null; then
    echo "ERROR: nvpmodel not found. Are you on a Jetson board?"
    exit 1
fi

echo ""
echo "[1/4] Installing dependencies..."

# Try pip3 first, fall back to apt
if command -v pip3 &> /dev/null; then
    pip3 install --quiet --break-system-packages numpy 2>/dev/null || true
    pip3 install --quiet --break-system-packages onnxruntime-gpu 2>/dev/null || \
        pip3 install --quiet --break-system-packages onnxruntime 2>/dev/null || true
    pip3 install --quiet --break-system-packages matplotlib 2>/dev/null || true
else
    echo "  pip3 not found. Trying apt..."
    apt-get install -y -qq python3-numpy 2>/dev/null || true
    # Try installing from local wheels if available
    WHEEL_DIR="$(dirname "$0")/../wheels"
    if [ -d "$WHEEL_DIR" ]; then
        echo "  Installing from local wheels..."
        python3 -m pip install --break-system-packages "$WHEEL_DIR"/*.whl 2>/dev/null || true
    fi
fi

echo ""
echo "[2/4] Checking power mode..."
nvpmodel -q
echo ""

echo "[3/4] Checking ONNX models..."
for model in ../onnx_models/*.onnx; do
    if [ -f "$model" ]; then
        size=$(du -h "$model" | cut -f1)
        echo "  Found: $(basename $model) ($size)"
    fi
done

echo ""
echo "[4/4] Starting benchmark..."
echo "  This takes about 30-60 minutes."
echo "  Running: 5 models x 4 batch sizes x 3 power modes (MAXN_SUPER, 25W, 15W) = 60 configs"
echo ""

python3 sweep_all.py

echo ""
echo "============================================================"
echo "  DONE! Results: results/jetson_hardware_results.csv"
echo "============================================================"
