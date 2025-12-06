#!/bin/bash
# Test llama.cpp with Vulkan backend on MI300X (No ROCm!)
#
# This tests a different approach: using llama.cpp's native Vulkan backend
# instead of patching HuggingFace transformers.
#
# Usage: ./test_llamacpp_vulkan.sh <droplet-ip>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    exit 1
fi

DROPLET_IP="$1"
REMOTE="root@$DROPLET_IP"

echo "============================================================"
echo "LLAMA.CPP VULKAN BACKEND TEST ON MI300X"
echo "No ROCm required - uses Vulkan compute shaders"
echo "============================================================"
echo ""

# Setup and run test
ssh "$REMOTE" << 'EOF'
set -e

echo "=== Step 1: Check Vulkan Support ==="
if command -v vulkaninfo &> /dev/null; then
    echo "Vulkan info:"
    vulkaninfo --summary 2>/dev/null | head -20 || echo "vulkaninfo available but no GPU detected"
else
    echo "Installing Vulkan tools..."
    apt-get update -qq && apt-get install -y -qq vulkan-tools mesa-vulkan-drivers libvulkan1 2>/dev/null || true
fi

# Check for AMD GPU
echo ""
echo "=== GPU Detection ==="
lspci | grep -i "VGA\|3D\|Display" || echo "No GPU found via lspci"

echo ""
echo "=== Step 2: Download llama.cpp with Vulkan ==="
cd ~

# Check if we already have llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
else
    echo "llama.cpp already exists, updating..."
    cd llama.cpp && git pull --depth 1 || true
    cd ~
fi

cd llama.cpp

echo ""
echo "=== Step 3: Build with Vulkan Backend ==="
# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    apt-get install -y -qq cmake build-essential 2>/dev/null || true
fi

# Clean and build with Vulkan
rm -rf build 2>/dev/null || true
mkdir -p build && cd build

echo "Configuring with Vulkan..."
cmake .. \
    -DGGML_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tail -20

echo ""
echo "Building (this may take a few minutes)..."
cmake --build . --config Release -j$(nproc) 2>&1 | tail -30

# Check if build succeeded
if [ ! -f "bin/llama-cli" ]; then
    echo "ERROR: Build failed - llama-cli not found"
    ls -la bin/ 2>/dev/null || echo "bin/ directory doesn't exist"
    exit 1
fi

echo ""
echo "=== Step 4: Download a Small Test Model ==="
cd ~/llama.cpp

# Download TinyLlama GGUF (small, fast)
MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    echo "Downloading TinyLlama 1.1B (Q4_K_M quantized, ~700MB)..."
    cd "$MODEL_DIR"
    wget -q --show-progress \
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
        -O tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf || {
            echo "Download failed, trying alternative..."
            # Try smaller model
            wget -q --show-progress \
                "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf" \
                -O tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf || echo "Model download failed"
        }
    cd ..
else
    echo "Model already downloaded"
fi

echo ""
echo "=== Step 5: Test Inference with Vulkan ==="
cd ~/llama.cpp

MODEL="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found at $MODEL"
    echo "Skipping inference test..."
    exit 1
fi

echo "Testing Vulkan backend..."
echo ""

# Test 1: Check if Vulkan GPU is detected
echo "--- Vulkan Device Detection ---"
./build/bin/llama-cli --version 2>&1 | head -5 || true

# Test 2: Simple inference
echo ""
echo "--- Running Inference Test ---"
echo "Prompt: 'The capital of France is'"
echo ""

# Run with Vulkan (-ngl 99 = offload all layers to GPU)
./build/bin/llama-cli \
    -m "$MODEL" \
    -p "The capital of France is" \
    -n 50 \
    -ngl 99 \
    --no-display-prompt \
    2>&1 | head -30

echo ""
echo "--- Benchmark Test ---"
./build/bin/llama-bench \
    -m "$MODEL" \
    -p 512 \
    -n 128 \
    -ngl 99 \
    2>&1 | tail -20

echo ""
echo "=== Test Complete ==="
echo ""
echo "Summary:"
echo "- llama.cpp built with Vulkan backend"
echo "- TinyLlama 1.1B model loaded"
echo "- Inference test completed"
EOF

echo ""
echo "============================================================"
echo "LLAMA.CPP VULKAN TEST COMPLETE"
echo "============================================================"
