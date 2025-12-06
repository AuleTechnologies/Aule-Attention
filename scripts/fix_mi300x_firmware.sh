#!/bin/bash
# Fix MI300X firmware on Ubuntu
# This installs the missing amdgpu firmware files
#
# Usage: ./fix_mi300x_firmware.sh <droplet-ip>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    exit 1
fi

DROPLET_IP="$1"
DROPLET_USER="${2:-root}"
REMOTE="$DROPLET_USER@$DROPLET_IP"

echo "=== Installing MI300X firmware on $REMOTE ==="
echo ""

ssh "$REMOTE" << 'EOF'
set -e

echo "=== 1. Check current firmware ==="
ls /lib/firmware/amdgpu/ 2>/dev/null | head -10 || echo "No amdgpu firmware directory"
echo ""

echo "=== 2. Install linux-firmware package ==="
apt-get update -qq
apt-get install -y linux-firmware

echo ""
echo "=== 3. Check for MI300X firmware files ==="
echo "Looking for gfx942/psp_13/gc_9_4_3 firmware..."
ls /lib/firmware/amdgpu/ | grep -E "psp_13_0_6|gc_9_4_3|sdma_4_4_2|vcn_4_0_3" || echo "MI300X firmware not found in linux-firmware"

echo ""
echo "=== 4. If firmware still missing, try linux-firmware-git or manual download ==="

# Check if the specific firmware files exist
MISSING=0
for fw in psp_13_0_6_ta.bin gc_9_4_3_rlc.bin sdma_4_4_2.bin vcn_4_0_3.bin; do
    if [ ! -f "/lib/firmware/amdgpu/$fw" ]; then
        echo "MISSING: $fw"
        MISSING=1
    else
        echo "FOUND: $fw"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "=== Attempting to download firmware from linux-firmware git ==="
    cd /tmp

    # Try to get firmware from linux-firmware.git
    if ! command -v git &> /dev/null; then
        apt-get install -y git
    fi

    # Clone just the amdgpu directory (sparse checkout)
    rm -rf linux-firmware-temp
    git clone --depth 1 --filter=blob:none --sparse https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git linux-firmware-temp 2>/dev/null || \
        git clone --depth 1 https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git linux-firmware-temp

    cd linux-firmware-temp
    git sparse-checkout set amdgpu 2>/dev/null || true

    # Copy firmware files
    if [ -d "amdgpu" ]; then
        echo "Copying firmware files..."
        cp -v amdgpu/*.bin /lib/firmware/amdgpu/ 2>/dev/null || true
    fi

    cd /
    rm -rf /tmp/linux-firmware-temp
fi

echo ""
echo "=== 5. Reload amdgpu driver ==="
echo "Removing amdgpu module..."
modprobe -r amdgpu 2>/dev/null || echo "Could not remove amdgpu (may be in use or not loaded)"

echo "Loading amdgpu module..."
modprobe amdgpu 2>/dev/null || echo "Could not load amdgpu"

echo ""
echo "=== 6. Check dmesg for GPU status ==="
dmesg | grep -i amdgpu | tail -20

echo ""
echo "=== 7. Final verification ==="
echo "Checking /dev/dri..."
ls -la /dev/dri/

echo ""
echo "Testing Vulkan..."
vulkaninfo --summary 2>&1 | grep -E "GPU|deviceName|apiVersion" | head -5 || echo "No Vulkan device"

echo ""
echo "Testing OpenCL..."
clinfo -l 2>&1

echo ""
echo "=== Done ==="
echo "If GPU still not working, a REBOOT may be required:"
echo "  reboot"
EOF

echo ""
echo "=== Script Complete ==="
echo "If the GPU is still not detected, reboot the droplet and re-test:"
echo "  ssh root@$DROPLET_IP 'reboot'"
echo "  # Wait 1 minute"
echo "  ./test_mi300x_options.sh $DROPLET_IP"
