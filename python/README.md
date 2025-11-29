# aule-attention

Hardware-agnostic FlashAttention implementation that works on AMD, NVIDIA, Intel, and Apple GPUs.

**No ROCm, CUDA, or compilation required at install time.**

## Performance

On AMD Instinct MI300X:

| Backend | TFLOPS | Requirements |
|---------|--------|--------------|
| ROCm FlashAttention (CK) | 150-300 | ROCm + flash-attn |
| PyTorch SDPA on ROCm | 50-150 | ROCm + PyTorch |
| OpenCL via Rusticl | 0.5-2 | **None!** |
| CPU | 0.01 | numpy |

The key feature: **OpenCL works without ROCm** via Mesa Rusticl, making it truly portable.

## Installation

```bash
# Basic (CPU + OpenCL if available)
pip install aule-attention

# With OpenCL support
pip install aule-attention[opencl]

# With PyTorch integration (autograd support)
pip install aule-attention[torch]

# Full installation
pip install aule-attention[full]
```

## Quick Start

```python
from aule import attention, Attention
import numpy as np

# Create tensors [batch, heads, seq_len, head_dim]
Q = np.random.randn(1, 8, 512, 64).astype(np.float32)
K = np.random.randn(1, 8, 512, 64).astype(np.float32)
V = np.random.randn(1, 8, 512, 64).astype(np.float32)

# One-shot usage (auto-selects best backend)
output = attention(Q, K, V, causal=True)

# Reusable (lower overhead for repeated calls)
with Attention() as attn:
    output = attn.forward(Q, K, V, causal=True)
    print(f"Using: {attn.backend_name}")
```

## PyTorch Integration

```python
import torch
from aule import aule_attention

# Tensors with gradients
Q = torch.randn(1, 8, 512, 64, requires_grad=True)
K = torch.randn(1, 8, 512, 64, requires_grad=True)
V = torch.randn(1, 8, 512, 64, requires_grad=True)

# Forward pass (uses GPU if available)
output = aule_attention(Q, K, V, causal=True)

# Backward pass (computes gradients)
loss = output.sum()
loss.backward()

print(Q.grad.shape)  # [1, 8, 512, 64]
```

## Check Available Backends

```python
from aule import get_available_backends, print_backend_info

# List available backends
print(get_available_backends())  # ['opencl', 'cpu']

# Detailed info
print_backend_info()
```

Output:
```
======================================================================
AULE-ATTENTION BACKEND STATUS
======================================================================
Available backends (in priority order):
  [1] opencl - ✓ AVAILABLE
      Performance: ~0.5-2 TFLOPS
  [2] cpu - ✓ AVAILABLE
      Performance: ~0.01 TFLOPS
======================================================================
```

## Features

- **FlashAttention-2 Algorithm**: O(N) memory usage via online softmax
- **Causal Masking**: Built-in support for autoregressive models
- **FP16 Support**: 2x memory bandwidth on supported devices
- **Backward Pass**: Full gradient computation for training
- **Long Sequences**: Tested up to 32K tokens
- **Multi-GPU Ready**: Explicit device selection

## Backend Priority

aule-attention automatically selects the best available backend:

1. **ROCm FlashAttention (CK)** - Peak performance on AMD MI-series
2. **PyTorch SDPA on ROCm** - High performance, easier setup
3. **HIP MFMA** - Our custom matrix-core kernel
4. **HIP Scalar** - Basic ROCm support
5. **Vulkan** - Consumer GPUs (NVIDIA, AMD, Intel, Apple)
6. **OpenCL** - Works on MI300X via Mesa Rusticl (no ROCm!)
7. **CPU** - NumPy fallback (always available)

## Force Specific Backend

```python
from aule import Attention

# Use environment variable
import os
os.environ['AULE_BACKEND'] = 'opencl'

# Or specify directly
with Attention(backend='cpu') as attn:
    output = attn.forward(Q, K, V)
```

## Why aule-attention?

1. **Truly Portable**: Works on AMD GPUs without ROCm via Mesa Rusticl
2. **Zero Compile**: Pre-built wheels, no compilation at install time
3. **Familiar API**: Drop-in replacement for PyTorch attention
4. **Training Ready**: Full backward pass support
5. **Production Ready**: Tested on real hardware (MI300X)

## Requirements

- Python 3.8+
- numpy
- pyopencl (optional, for GPU acceleration)
- torch (optional, for autograd integration)

## License

MIT License - Aule Technologies
