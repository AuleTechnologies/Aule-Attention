#!/bin/bash
# Build HIP kernel for AMD GPUs
#
# Requires ROCm to be installed: https://rocm.docs.amd.com/
# Tested with ROCm 6.x
#
# Output: attention_hip.hsaco (HIP code object)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please install ROCm."
    echo "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

# Detect GPU architecture
GPU_ARCH=""
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo | grep -oP 'gfx\d+[a-z]?' | head -1)
fi

if [ -z "$GPU_ARCH" ]; then
    # Default architectures for common datacenter GPUs
    # MI300X: gfx942, MI250: gfx90a, MI100: gfx908
    GPU_ARCH="gfx942"
    echo "Warning: Could not detect GPU architecture, using default: $GPU_ARCH"
fi

echo "Building HIP kernel for architecture: $GPU_ARCH"

# Compile to code object (.hsaco)
hipcc -O3 \
    --genco \
    --offload-arch=$GPU_ARCH \
    -o attention_hip.hsaco \
    attention_hip.cpp

echo "Successfully built: attention_hip.hsaco"

# Also build a shared library version for direct linking
hipcc -O3 \
    -shared \
    -fPIC \
    --offload-arch=$GPU_ARCH \
    -o libattention_hip.so \
    attention_hip.cpp

echo "Successfully built: libattention_hip.so"
