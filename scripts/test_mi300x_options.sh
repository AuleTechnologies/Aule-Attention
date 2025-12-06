#!/bin/bash
# Test what GPU compute options are available on MI300X
# without installing full ROCm
#
# Usage: ./test_mi300x_options.sh <droplet-ip>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    exit 1
fi

DROPLET_IP="$1"
DROPLET_USER="${2:-root}"
REMOTE="$DROPLET_USER@$DROPLET_IP"

echo "=== Testing GPU compute options on $REMOTE ==="
echo ""

ssh "$REMOTE" << 'EOF'
set -e

echo "=== 1. System Info ==="
uname -a
echo ""

echo "=== 2. GPU Detection (lspci) ==="
lspci | grep -i "vga\|display\|3d\|amd" || echo "No GPU found"
echo ""

echo "=== 3. Kernel Driver Status ==="
echo "Loaded modules:"
lsmod | grep -i "amdgpu\|radeon\|kfd" || echo "No AMD GPU modules loaded"
echo ""
echo "Device nodes:"
ls -la /dev/dri/ 2>/dev/null || echo "/dev/dri not found"
ls -la /dev/kfd 2>/dev/null || echo "/dev/kfd not found (ROCm kernel driver)"
echo ""

echo "=== 4. Install Mesa and test tools ==="
apt-get update -qq
apt-get install -y -qq mesa-utils mesa-opencl-icd clinfo vulkan-tools 2>/dev/null || \
    apt-get install -y mesa-utils clinfo vulkan-tools

echo ""
echo "=== 5. Test Vulkan ==="
echo "vulkaninfo summary:"
vulkaninfo --summary 2>&1 | head -30 || echo "vulkaninfo failed"
echo ""

echo "=== 6. Test OpenCL (standard) ==="
echo "clinfo platforms:"
clinfo -l 2>&1 || echo "No OpenCL platforms found"
echo ""

echo "=== 7. Test Rusticl (Mesa OpenCL) ==="
# Install Rusticl if available
apt-get install -y -qq mesa-opencl-icd 2>/dev/null || true

echo "Testing with RUSTICL_ENABLE=radeonsi:"
RUSTICL_ENABLE=radeonsi clinfo -l 2>&1 || echo "Rusticl not available"
echo ""

echo "=== 8. Check for any ROCm remnants ==="
ls -la /opt/rocm* 2>/dev/null || echo "No ROCm installation found"
which rocm-smi 2>/dev/null || echo "rocm-smi not installed"
which hipcc 2>/dev/null || echo "hipcc not installed"
echo ""

echo "=== 9. GPU driver info ==="
if [ -f /sys/class/drm/card0/device/vendor ]; then
    echo "Card vendor: $(cat /sys/class/drm/card0/device/vendor)"
    echo "Card device: $(cat /sys/class/drm/card0/device/device)"
fi

# Check dmesg for GPU info
echo ""
echo "Recent GPU-related kernel messages:"
dmesg | grep -i "amdgpu\|drm\|gpu" | tail -20 || echo "No GPU messages in dmesg"

echo ""
echo "=== Summary ==="
echo "If Vulkan shows a real GPU (not llvmpipe): Vulkan backend will work"
echo "If OpenCL/Rusticl shows a device: OpenCL backend could work"
echo "If neither works: ROCm is required for MI300X"
EOF

echo ""
echo "=== Test Complete ==="
