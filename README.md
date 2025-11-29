# aule-attention

Hardware-agnostic FlashAttention implementation via Zig + Vulkan Compute.

**Works on AMD, NVIDIA, Intel, and Apple GPUs — no ROCm, no CUDA, no compilation required.**

## Installation

```bash
pip install aule-attention
```

## Quick Start

```python
from aule import flash_attention
import numpy as np

# Create Q, K, V tensors [batch, heads, seq_len, head_dim]
batch, heads, seq_len, head_dim = 1, 12, 512, 64
Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

# Run FlashAttention (causal by default for LLMs)
output = flash_attention(Q, K, V, causal=True)
```

## Features

- **Universal GPU Support**: Works on AMD, NVIDIA, Intel, and Apple GPUs via Vulkan
- **Zero Dependencies**: No ROCm, CUDA, or compilation needed at install time
- **AMD Optimized**: 64-wide wavefront optimizations for AMD GPUs
- **PyTorch Integration**: Drop-in replacement for `torch.nn.functional.scaled_dot_product_attention`
- **Numerically Accurate**: Matches PyTorch SDPA output exactly

## PyTorch Integration

```python
import torch
from aule import aule_attention

Q = torch.randn(1, 12, 512, 64)
K = torch.randn(1, 12, 512, 64)
V = torch.randn(1, 12, 512, 64)

# Drop-in replacement for F.scaled_dot_product_attention
output = aule_attention(Q, K, V, causal=True)
output.sum().backward()  # Gradients work!
```

## Backends

aule-attention tries backends in this order:

1. **Vulkan** (fastest, AMD-optimized) — via `libaule.so`
2. **OpenCL** (fallback) — via Mesa Rusticl
3. **CPU** (always available) — pure NumPy

Check available backends:

```python
from aule import print_backend_info
print_backend_info()
```

## GPU Tensor API (Zero-Copy)

For repeated operations, use the GPU tensor API to avoid copying:

```python
from aule import Aule

with Aule() as aule:
    # Check device info
    info = aule.get_device_info()
    print(f"GPU: {info['device_name']}")
    print(f"AMD optimized: {info['amd_optimized']}")

    # Create GPU tensors
    q_gpu = aule.tensor((batch, heads, seq_len, head_dim))
    k_gpu = aule.tensor((batch, heads, seq_len, head_dim))
    v_gpu = aule.tensor((batch, heads, seq_len, head_dim))
    out_gpu = aule.tensor((batch, heads, seq_len, head_dim))

    # Upload data
    q_gpu.upload(Q)
    k_gpu.upload(K)
    v_gpu.upload(V)

    # Run attention (zero-copy)
    aule.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=True)

    # Download result
    result = out_gpu.download()
```

## Building from Source

Requires Zig 0.14+ and Vulkan SDK:

```bash
# Build the native library
zig build

# Run tests
zig build test

# Build Python wheel
cd python && python -m build --wheel
```

## Supported Hardware

| Vendor | Minimum | Notes |
|--------|---------|-------|
| AMD | RX 5000+ / MI100+ | Primary target, 64-wide wavefront |
| NVIDIA | GTX 1000+ | Full support |
| Intel | Arc A-series | Full support |
| Apple | M1+ | Via MoltenVK |

## License

MIT License - Aule Technologies
