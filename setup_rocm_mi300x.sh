#!/bin/bash
# Install ROCm on Ubuntu for MI300X support
#
# Usage: ./setup_rocm_mi300x.sh <droplet-ip>
# This script SSHs into the droplet and installs ROCm

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    exit 1
fi

DROPLET_IP="$1"
DROPLET_USER="${2:-root}"
REMOTE="$DROPLET_USER@$DROPLET_IP"

echo "=== Installing ROCm on $REMOTE ==="
echo "This may take 10-15 minutes..."
echo ""

ssh "$REMOTE" << 'EOF'
set -e

echo "=== Step 1: Check current GPU status ==="
lspci | grep -i amd || echo "No AMD GPU found in lspci"
ls -la /dev/kfd 2>/dev/null && echo "KFD already available" || echo "/dev/kfd not found"
ls -la /dev/dri/ 2>/dev/null || echo "/dev/dri not found"

echo ""
echo "=== Step 2: Check if ROCm is already installed ==="
if command -v rocm-smi &> /dev/null; then
    echo "ROCm already installed!"
    rocm-smi
    exit 0
fi

echo ""
echo "=== Step 3: Install ROCm ==="

# Detect Ubuntu version
. /etc/os-release
echo "Detected: $NAME $VERSION_ID"

# For Ubuntu 24.04+ (noble) or 25.x
if [[ "$VERSION_ID" == "24.04" ]] || [[ "$VERSION_ID" == "25."* ]]; then
    UBUNTU_CODENAME="noble"
else
    UBUNTU_CODENAME="jammy"  # 22.04
fi

echo "Using ROCm repository for: $UBUNTU_CODENAME"

# Add ROCm GPG key
echo "Adding ROCm repository key..."
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - 2>/dev/null || \
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

# Add ROCm repository
echo "Adding ROCm repository..."
if [ -f /etc/apt/keyrings/rocm.gpg ]; then
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.3 $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/rocm.list
else
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.3.3 $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/rocm.list
fi

# Pin ROCm packages
echo 'Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600' > /etc/apt/preferences.d/rocm-pin-600

# Update and install
echo "Updating package lists..."
apt-get update

echo "Installing ROCm HIP runtime..."
DEBIAN_FRONTEND=noninteractive apt-get install -y rocm-hip-runtime hip-dev

echo ""
echo "=== Step 4: Set up environment ==="
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' >> ~/.bashrc
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

echo ""
echo "=== Step 5: Verify installation ==="
echo "ROCm version:"
cat /opt/rocm/.info/version 2>/dev/null || echo "Version file not found"

echo ""
echo "Checking rocm-smi..."
rocm-smi 2>/dev/null || echo "rocm-smi not working yet"

echo ""
echo "Checking rocminfo..."
rocminfo 2>/dev/null | head -30 || echo "rocminfo not working"

echo ""
echo "=== Step 6: Install HIP Python bindings ==="
pip3 install --break-system-packages hip-python 2>/dev/null || pip3 install hip-python

echo ""
echo "=== ROCm Installation Complete ==="
echo "You may need to reboot for the kernel driver to load."
echo "Run: reboot"
EOF

echo ""
echo "=== Done ==="
echo "If rocm-smi didn't show the GPU, you may need to reboot the droplet:"
echo "  ssh $REMOTE 'reboot'"
echo ""
echo "Then wait ~1 minute and run the deploy script:"
echo "  ./deploy_mi300x.sh $DROPLET_IP"
