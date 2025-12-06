#!/bin/bash
# Deploy aule-attention Vulkan backend to MI300X for testing
# Usage: ./deploy_vulkan_mi300x.sh <server_ip>

set -e

SERVER="${1:-134.199.200.252}"
REMOTE_DIR="~/aule-vulkan"

echo "=== Deploying aule-attention Vulkan backend to $SERVER ==="

# Build locally first
echo "Building library..."
zig build

# Create tarball of necessary files
echo "Creating deployment package..."
tar czf /tmp/aule-vulkan.tar.gz \
    zig-out/lib/libaule.so \
    python/aule.py \
    tests/test_vulkan_attention.py \
    build.zig \
    build.zig.zon \
    src/ \
    shaders/

# Copy to server
echo "Copying to server..."
scp /tmp/aule-vulkan.tar.gz root@$SERVER:/tmp/

# Setup on remote
echo "Setting up on remote..."
ssh root@$SERVER bash << 'REMOTE_SCRIPT'
set -e

# Create directory
mkdir -p ~/aule-vulkan
cd ~/aule-vulkan

# Extract
tar xzf /tmp/aule-vulkan.tar.gz

# Install Vulkan if not present
if ! command -v vulkaninfo &> /dev/null; then
    echo "Installing Vulkan SDK..."
    apt-get update
    apt-get install -y vulkan-tools libvulkan1 libvulkan-dev mesa-vulkan-drivers
fi

# Check Vulkan
echo ""
echo "=== Vulkan Info ==="
vulkaninfo --summary 2>/dev/null || echo "Vulkan summary not available"

# Try to run test
echo ""
echo "=== Running Tests ==="
cd ~/aule-vulkan
export LD_LIBRARY_PATH=$PWD/zig-out/lib:$LD_LIBRARY_PATH
python3 tests/test_vulkan_attention.py 2>&1 || echo "Test failed or Vulkan not available"

REMOTE_SCRIPT

echo ""
echo "=== Deployment Complete ==="
echo "To run tests manually:"
echo "  ssh root@$SERVER 'cd ~/aule-vulkan && LD_LIBRARY_PATH=zig-out/lib python3 tests/test_vulkan_attention.py'"
