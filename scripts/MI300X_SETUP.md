# aule-attention on AMD MI300X

This guide explains how to run aule-attention on AMD Instinct MI300X datacenter GPUs.

## Background

The MI300X is a compute-only accelerator that doesn't have traditional Vulkan drivers. Instead, it uses AMD's ROCm/HIP stack for GPU compute.

aule-attention supports MI300X through a dedicated HIP backend that provides the same FlashAttention functionality.

## Prerequisites

1. **DigitalOcean GPU Droplet** (or other MI300X system)
2. **ROCm 6.x** installed
3. **Python 3.8+**

## Installation

### Step 1: Verify GPU Detection

```bash
# Check ROCm sees the GPU
rocm-smi

# Should show something like:
# ======================= ROCm System Management Interface =======================
# GPU  Temp   Perf  Power  Memory    GPU%
# 0    45C    auto  150W   0%        0%
```

### Step 2: Install hip-python

```bash
# Install HIP Python bindings
pip install hip-python

# Verify installation
python -c "from hip import hip; print(f'HIP devices: {hip.hipGetDeviceCount()}')"
```

### Step 3: Install aule-attention

```bash
# Clone the repository
git clone https://github.com/yourusername/aule-attention.git
cd aule-attention

# Install Python package
pip install -e python/
```

## Usage

### Automatic Backend Selection

```python
import aule_unified as aule
import numpy as np

# Create test data
Q = np.random.randn(1, 8, 64, 64).astype(np.float32)
K = np.random.randn(1, 8, 64, 64).astype(np.float32)
V = np.random.randn(1, 8, 64, 64).astype(np.float32)

# aule automatically selects HIP on MI300X
with aule.Attention() as attn:
    print(f"Using backend: {attn.backend_name}")
    output = attn.forward(Q, K, V)
```

### Force HIP Backend

```python
# Via code
with aule.Attention(backend='hip') as attn:
    output = attn.forward(Q, K, V)

# Via environment variable
# export AULE_BACKEND=hip
# python your_script.py
```

### Direct HIP API

```python
from aule_hip import HipAttention

with HipAttention() as attn:
    output = attn.forward(Q, K, V)
```

## Performance Tips

### Batch Processing
For best performance on MI300X, use larger batch sizes:

```python
# Good: Large batches
Q = np.random.randn(32, 8, 512, 64).astype(np.float32)

# Less efficient: Small batches
Q = np.random.randn(1, 8, 64, 64).astype(np.float32)
```

### Persistent Tensors (Coming Soon)
The HIP backend will support persistent GPU tensors to eliminate copy overhead:

```python
# Future API
q_gpu = attn.tensor(Q.shape)
q_gpu.upload(Q)

for step in range(1000):
    attn.forward_gpu(q_gpu, k_gpu, v_gpu, out_gpu)

result = out_gpu.download()
```

## Troubleshooting

### "No HIP devices found"

```bash
# Check if amdgpu driver is loaded
lsmod | grep amdgpu

# Check ROCm installation
rocm-smi

# Verify /dev/kfd exists (KFD = Kernel Fusion Driver)
ls -la /dev/kfd
```

### "hip-python import error"

```bash
# Ensure ROCm is in PATH
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

# Reinstall hip-python
pip uninstall hip-python
pip install hip-python
```

### "Kernel compilation failed"

The HIP kernel is compiled at runtime using hiprtc. If this fails:

```bash
# Check hiprtc is available
ls /opt/rocm/lib/libhiprtc*

# Verify GPU architecture detection
rocminfo | grep gfx
```

## Comparison with Vulkan Backend

| Feature | Vulkan | HIP |
|---------|--------|-----|
| Consumer AMD GPUs (RX 7900, etc.) | ✅ | ❌ |
| Datacenter GPUs (MI300X) | ❌ | ✅ |
| NVIDIA GPUs | ✅ | ❌ |
| Intel GPUs | ✅ | ❌ |
| Apple Silicon | ✅ (MoltenVK) | ❌ |
| Cross-vendor | ✅ | AMD only |

## Architecture

```
┌─────────────────────────────────────────────┐
│             aule_unified.py                 │
│         (Automatic backend selection)       │
└─────────────────┬───────────────────────────┘
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Vulkan  │ │   HIP    │ │   CPU    │
│ (aule.py)│ │(aule_hip)│ │(fallback)│
└────┬─────┘ └────┬─────┘ └──────────┘
     │            │
     ▼            ▼
┌──────────┐ ┌──────────┐
│ libaule  │ │  ROCm    │
│  (Zig)   │ │  hiprtc  │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            ▼
┌──────────┐ ┌──────────┐
│  SPIR-V  │ │   HIP    │
│ Compute  │ │  Kernel  │
│ Shader   │ │  (.cpp)  │
└──────────┘ └──────────┘
```

## Benchmarking on MI300X

```python
import aule_unified as aule
import numpy as np
import time

# Test configurations
configs = [
    (1, 8, 128, 64),    # Small
    (4, 16, 512, 64),   # Medium
    (8, 32, 1024, 64),  # Large
    (16, 32, 2048, 64), # Very large
]

with aule.Attention(backend='hip') as attn:
    print(f"Backend: {attn.backend_name}\n")

    for batch, heads, seq, dim in configs:
        Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        # Warmup
        _ = attn.forward(Q, K, V)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = attn.forward(Q, K, V)
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        print(f"{batch}x{heads}x{seq}x{dim}: {avg_ms:.2f} ms")
```
