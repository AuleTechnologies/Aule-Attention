<p align="center">
  <img src="Aule-Attention.png" alt="aule-attention" width="400">
</p>

<h1 align="center">aule-attention</h1>

<p align="center">
  <strong>Hardware-agnostic FlashAttention implementation</strong><br>
  No compilation required. Works on any GPU.
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#api-reference">API Reference</a> |
  <a href="#supported-hardware">Hardware</a>
</p>

---

**Version: 0.2.0**

## Overview

aule-attention provides a drop-in FlashAttention implementation that works across all major GPU vendors without requiring compilation at install time. It automatically selects the optimal backend for your hardware:

- **Triton**: For AMD ROCm and NVIDIA CUDA (training and inference)
- **Vulkan**: For Intel, Apple, AMD consumer GPUs, and any Vulkan-capable device (inference)
- **CPU**: NumPy fallback for systems without GPU support

## Installation

```bash
pip install aule-attention
```

For AMD datacenter GPUs (MI300X, MI250, MI100):

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install aule-attention
```

For NVIDIA GPUs:

```bash
pip install torch
pip install aule-attention
```

## Quick Start

### Basic Usage

```python
from aule import flash_attention
import torch

# Create input tensors [batch, heads, seq_len, head_dim]
q = torch.randn(1, 8, 512, 64, device='cuda')
k = torch.randn(1, 8, 512, 64, device='cuda')
v = torch.randn(1, 8, 512, 64, device='cuda')

# Compute attention with causal masking
output = flash_attention(q, k, v, causal=True)
```

### Training with Gradient Computation

```python
from aule import flash_attention
import torch

# Enable gradient tracking
q = torch.randn(2, 8, 256, 64, device='cuda', requires_grad=True)
k = torch.randn(2, 8, 256, 64, device='cuda', requires_grad=True)
v = torch.randn(2, 8, 256, 64, device='cuda', requires_grad=True)

# Forward pass
output = flash_attention(q, k, v, causal=True)

# Backward pass
loss = output.sum()
loss.backward()

# Gradients are now available
print(f"Query gradient shape: {q.grad.shape}")
print(f"Key gradient shape: {k.grad.shape}")
print(f"Value gradient shape: {v.grad.shape}")
```

### Grouped Query Attention (GQA)

Modern large language models like Llama 2, Mistral, and Qwen use grouped query attention where the number of query heads exceeds the number of key/value heads:

```python
from aule import flash_attention
import torch

# Configuration: 32 query heads, 8 key/value heads (4:1 ratio)
batch_size = 1
num_q_heads = 32
num_kv_heads = 8
seq_len = 512
head_dim = 128

q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda')

output = flash_attention(q, k, v, causal=True)
```

### Multi-Query Attention (MQA)

For single key/value head shared across all query heads:

```python
from aule import flash_attention
import torch

q = torch.randn(1, 32, 512, 64, device='cuda')
k = torch.randn(1, 1, 512, 64, device='cuda')   # Single KV head
v = torch.randn(1, 1, 512, 64, device='cuda')

output = flash_attention(q, k, v, causal=True)
```

### NumPy Arrays (CPU or Vulkan Backend)

```python
from aule import flash_attention
import numpy as np

q = np.random.randn(1, 8, 256, 64).astype(np.float32)
k = np.random.randn(1, 8, 256, 64).astype(np.float32)
v = np.random.randn(1, 8, 256, 64).astype(np.float32)

output = flash_attention(q, k, v, causal=True)
```

### Checking Available Backends

```python
from aule import get_available_backends, print_backend_info

# List available backends
backends = get_available_backends()
print(f"Available backends: {backends}")

# Display detailed backend information
print_backend_info()
```

Example output:

```
============================================================
AULE-ATTENTION v0.2.0
============================================================

Available backends: ['triton', 'vulkan', 'cpu']

[1] TRITON (primary)
    GPU: AMD Instinct MI300X
    Status: FlashAttention-2 kernel

[2] VULKAN
    GPU: AMD Radeon RX 7900 XTX
    Status: Vulkan compute shader

[3] CPU
    Status: NumPy fallback

============================================================
```

## API Reference

### flash_attention

```python
flash_attention(query, key, value, causal=True, scale=None)
```

Computes scaled dot-product attention using the FlashAttention-2 algorithm.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | torch.Tensor or np.ndarray | Query tensor of shape `[batch, heads_q, seq_len_q, head_dim]` |
| `key` | torch.Tensor or np.ndarray | Key tensor of shape `[batch, heads_kv, seq_len_k, head_dim]` |
| `value` | torch.Tensor or np.ndarray | Value tensor of shape `[batch, heads_kv, seq_len_k, head_dim]` |
| `causal` | bool | If True, applies causal masking to prevent attending to future tokens. Default: True |
| `scale` | float or None | Scaling factor for attention scores. Default: 1/sqrt(head_dim) |

**Returns:**

Output tensor with the same shape and type as the query tensor.

**Constraints:**

- `heads_q` must be divisible by `heads_kv` for grouped query attention
- `head_dim` must be <= 128 for Triton backend, <= 64 for Vulkan backend
- Supported dtypes: float16, bfloat16, float32

### get_available_backends

```python
get_available_backends()
```

Returns a list of available backend names in priority order.

**Returns:** List[str] - Backend names (e.g., `['triton', 'vulkan', 'cpu']`)

### print_backend_info

```python
print_backend_info()
```

Prints detailed information about available backends and detected GPU hardware.

## Supported Hardware

### Triton Backend

Full training and inference support with backward pass gradients.

| GPU Family | Models | Status |
|------------|--------|--------|
| AMD Instinct | MI300X, MI300A, MI250X, MI250, MI210, MI100 | Tested |
| NVIDIA Datacenter | H100, A100, A10, L40S | Supported |
| NVIDIA Consumer | RTX 4090, 4080, 3090, 3080 | Supported |

### Vulkan Backend

Inference support via pre-compiled SPIR-V compute shaders.

| GPU Family | Models | Status |
|------------|--------|--------|
| AMD RDNA3 | RX 7900 XTX, 7900 XT, 7800 XT, 7700 XT | Supported |
| AMD RDNA2 | RX 6900 XT, 6800 XT, 6700 XT | Supported |
| Intel Arc | A770, A750, A580, A380 | Tested |
| Intel Integrated | 12th/13th/14th Gen (Iris Xe) | Tested |
| Apple Silicon | M1, M2, M3 (via MoltenVK) | Supported |
| NVIDIA | Any GPU with Vulkan 1.2+ | Supported |

## Architecture

```
                    flash_attention()
                          |
          +---------------+---------------+
          |               |               |
     [PyTorch]       [NumPy]         [NumPy]
     CUDA/ROCm
          |               |               |
    Triton Backend   Vulkan Backend   CPU Backend
          |               |               |
    AMD ROCm        Any Vulkan GPU      NumPy
    NVIDIA CUDA     (Intel, AMD,       (fallback)
                     Apple, etc.)
```

## Performance Characteristics

| Metric | Traditional Attention | aule-attention |
|--------|----------------------|----------------|
| Memory Complexity | O(N^2) | O(N) |
| Attention Matrix | Fully materialized | Never materialized |
| Sequence Length | Limited by VRAM | Extended sequences possible |

Numerical accuracy compared to PyTorch scaled_dot_product_attention:

| Precision | Maximum Absolute Error | Maximum Relative Error |
|-----------|----------------------|----------------------|
| float32 | < 1e-6 | < 1e-5 |
| float16 | < 1e-3 | < 1e-2 |

## Building from Source

### Requirements

- Zig 0.13.0 or later
- Vulkan SDK (for glslc shader compiler)
- Python 3.8 or later

### Build Commands

```bash
# Clone the repository
git clone https://github.com/aule-dev/aule-attention.git
cd aule-attention

# Build the native library
zig build -Doptimize=ReleaseFast

# Run tests
zig build test

# Build Python wheel
cd python
python -m build --wheel
```

### Project Structure

```
aule-attention/
├── src/                          # Zig source code
│   ├── lib.zig                   # C API exports
│   ├── vulkan_context.zig        # Vulkan device setup
│   ├── attention_gpu.zig         # Attention computation engine
│   ├── attention_pipeline.zig    # Forward pass pipeline
│   ├── attention_backward_pipeline.zig  # Backward pass pipeline
│   └── backends/
│       ├── backend.zig           # Unified backend interface
│       └── hip.zig               # HIP/ROCm backend
├── shaders/                      # GLSL compute shaders
│   ├── attention_f32.comp        # Generic FP32 attention
│   ├── attention_f32_amd.comp    # AMD-optimized (64-wide wavefront)
│   ├── attention_forward_f32.comp    # Forward with LSE output
│   └── attention_backward_f32.comp   # Backward pass gradients
├── python/
│   └── aule/
│       ├── __init__.py           # Public API
│       ├── vulkan.py             # Vulkan backend bindings
│       ├── triton_flash.py       # Triton FlashAttention-2 kernel
│       └── lib/                  # Pre-built native libraries
├── tests/                        # Test suite
└── build.zig                     # Build configuration
```

## Troubleshooting

### Triton backend not available

Ensure PyTorch is installed with CUDA or ROCm support:

```bash
# For AMD ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# For NVIDIA CUDA
pip install torch
```

### Vulkan backend not available

Verify Vulkan drivers are installed:

```bash
vulkaninfo --summary
```

Install Vulkan drivers if needed:

```bash
# Ubuntu/Debian
sudo apt install vulkan-tools mesa-vulkan-drivers

# Fedora
sudo dnf install vulkan-tools mesa-vulkan-drivers

# Arch Linux
sudo pacman -S vulkan-tools vulkan-icd-loader
```

### Out of memory errors

Reduce batch size or sequence length. While aule-attention uses O(N) memory instead of O(N^2), the input tensors (Q, K, V) still require GPU memory.

## License

MIT License

Copyright (c) 2025 Aule Technologies

## Acknowledgments

This project builds upon the work of the open source community:

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) by Tri Dao and the Dao-AILab team for the memory-efficient attention algorithm
- [Triton](https://github.com/openai/triton) by OpenAI for the GPU programming language and compiler
- [Vulkan](https://www.vulkan.org/) by the Khronos Group for the cross-platform GPU compute API
- [vulkan-zig](https://github.com/Snektron/vulkan-zig) by Snektron for Zig Vulkan bindings
- [Zig](https://ziglang.org/) by Andrew Kelley and the Zig Software Foundation

We are grateful to these projects and their maintainers for making hardware-agnostic GPU computing possible.

---

<p align="center">
  Made with dedication by <a href="https://auletechnologies.com">Aule Technologies</a>
</p>
