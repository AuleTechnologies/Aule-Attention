#!/bin/bash
# Test aule-attention with LLaMA on MI300X (without ROCm)
#
# Usage: ./test_llama_mi300x.sh <droplet-ip>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    exit 1
fi

DROPLET_IP="$1"
REMOTE="root@$DROPLET_IP"

echo "=== Testing aule-attention + LLaMA on MI300X ==="
echo "Target: $REMOTE"
echo ""

# Copy files
echo "[1/3] Copying test files..."
cd /home/yab/Sndr
scp python/aule_opencl.py python/aule_unified.py python/test_llama_aule.py "$REMOTE:~/aule-attention/python/"

# Install dependencies and run test
echo "[2/3] Installing dependencies..."
ssh "$REMOTE" << 'SETUP_EOF'
cd ~/aule-attention

# Install required packages
pip3 install -q numpy pyopencl transformers accelerate sentencepiece torch --break-system-packages 2>/dev/null || \
pip3 install -q numpy pyopencl transformers accelerate sentencepiece torch

echo "Dependencies installed"
SETUP_EOF

# Run the LLaMA test
echo "[3/3] Running LLaMA test..."
ssh "$REMOTE" << 'TEST_EOF'
cd ~/aule-attention/python

# Set environment to avoid ROCm/CUDA
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# Run test
python3 test_llama_aule.py
TEST_EOF

echo ""
echo "=== Test Complete ==="
