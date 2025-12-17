# aule-attention

**FlashAttention that just works. No compilation. Any GPU.**

Version: 0.4.0

## What's New in 0.4.0

- **AMD MI300X Optimized Kernel**: New Triton FlashAttention-2 kernel tuned for CDNA3
  - Auto-detects AMD GPUs and routes to optimized kernel
  - Uses `exp2` optimization (faster than `exp` on AMD hardware)
  - Extended autotune configs for 7B/13B/70B/405B models

### MI300X Benchmark Results (vs PyTorch SDPA)

| Config | Speedup |
|--------|---------|
| Llama-70B 4K context | **+20.4%** |
| Llama-70B 8K context | **+19.9%** |
| Llama-70B batch=4 | **+21.0%** |
| Long context (32K-128K) | **+17-22%** |
| Average across all configs | **+15%** |

## What's New in 0.3.7

- **Windows DLL included** - Cross-compiled Windows support with PagedAttention

## What's New in 0.3.6

- **PagedAttention (vLLM-style)**: Block-based KV cache for 90% memory savings
- **7-13x Faster Vulkan**: New fast shader with 32x32 blocks and vec4 loads
- **Sliding Window Attention**: Efficient local attention with `window_size` parameter
- **Native FP16/BF16 Compute**: Triton kernels now use native precision (no FP32 conversion overhead)
- **Multiple Shader Variants**: Choose baseline, fast, fp16, or fp16_amd for your hardware
- **Long Context Support**: Validated up to 4K+ tokens

### Performance Highlights

| Feature | Improvement |
|---------|-------------|
| PagedAttention | 90% memory savings, 7.9K tokens/sec batched |
| Fast Shader | 7-13x speedup over baseline |
| Sliding Window (S=1024, W=128) | 7.6x faster (544ms → 71ms) |
| Triton FP16/BF16 | ~20-30% faster (native compute) |

### New Shader Variants API

```python
from aule.vulkan import Aule

with Aule() as aule:
    # Check available variants
    print(aule.available_shader_variants)  # [0, 1, 2, 3]
    print(aule.shader_variant_name)  # "fast" (default)

    # Switch variants
    aule.set_shader_variant(Aule.SHADER_BASELINE)  # 0: Original 16x16
    aule.set_shader_variant(Aule.SHADER_FAST)      # 1: Optimized 32x32 (default)
    aule.set_shader_variant(Aule.SHADER_FP16)      # 2: FP16 with FP32 accumulation
    aule.set_shader_variant(Aule.SHADER_FP16_AMD)  # 3: FP16 for AMD 64-wide wavefronts
```

### Sliding Window Attention

```python
# Mistral-style sliding window (only attend to nearby tokens)
output = aule.attention_gravity(q, k, v, indices, window_size=256)

# Massive speedup for long sequences
# S=4096, window=512 → ~8x faster than full attention
```

## What's New in 0.3.3

- **Windows Vulkan Support**: Added Windows DLL - Vulkan backend now works on Windows!
- **Windows AMD Fix**: Automatic fallback to Vulkan on Windows + AMD (Triton AMD doesn't support Windows)

## What's New in 0.3.0

- **GQA/MQA Support**: Full Grouped Query Attention and Multi-Query Attention for Vulkan backend
- **Cross-Attention**: Different sequence lengths for Q and K/V tensors
- **ComfyUI Compatible**: Works with Stable Diffusion, SDXL, Flux, SD3 (use `causal=False`)
- **20% Faster**: Phase 1 performance optimizations
- **Bug Fixes**: Tensor cache fix for GQA, blocky noise fix for diffusion models

## Installation

```bash
pip install aule-attention
```

## Quick Start

```python
from aule import flash_attention
import torch

q = torch.randn(1, 8, 512, 64, device='cuda')
k = torch.randn(1, 8, 512, 64, device='cuda')
v = torch.randn(1, 8, 512, 64, device='cuda')

output = flash_attention(q, k, v, causal=True)
```

### ComfyUI / Diffusion Models

```python
import aule
aule.install()  # Patches PyTorch SDPA globally

# Now all models using F.scaled_dot_product_attention use aule
# For diffusion models, causal=False is used automatically
```

### GQA (Grouped Query Attention)

```python
# 32 query heads, 8 key/value heads (4:1 ratio)
q = torch.randn(1, 32, 512, 64)
k = torch.randn(1, 8, 512, 64)
v = torch.randn(1, 8, 512, 64)

output = flash_attention(q, k, v, causal=True)
```

## Features

- No compilation at install time
- Works on AMD, NVIDIA, Intel, and Apple GPUs
- Training support with backward pass (Triton backend)
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA) support
- Cross-attention with different Q/KV sequence lengths
- Sliding window attention for efficient long sequences
- O(N) memory complexity via FlashAttention-2 algorithm

## Backends

| Backend | Hardware | Features |
|---------|----------|----------|
| Triton | AMD ROCm (Linux), NVIDIA CUDA | Training + Inference, head_dim up to 128 |
| Vulkan | Any Vulkan 1.2+ GPU (including Windows AMD) | Inference, head_dim up to 64, GQA/MQA, Sliding Window |
| CPU | NumPy | Fallback, any head_dim |

> **Windows + AMD**: Automatically uses Vulkan backend (Triton AMD only supports Linux).

## API

```python
from aule import flash_attention, get_available_backends, install

# Compute attention
output = flash_attention(query, key, value, causal=True, scale=None)

# Check available backends
backends = get_available_backends()  # ['vulkan', 'cpu'] or ['triton', 'vulkan', 'cpu']

# Install as PyTorch SDPA replacement (for ComfyUI, Transformers, etc.)
install()  # Auto-select best backend
install(backend='vulkan', verbose=True)  # Force backend + logging
```

## Supported Hardware

### Triton Backend (Training + Inference)

- AMD Instinct: MI300X, MI300A, MI250X, MI250, MI210, MI100
- NVIDIA Datacenter: H100, A100, A10, L40S
- NVIDIA Consumer: RTX 4090, 4080, 3090, 3080

### Vulkan Backend (Inference)

- AMD RDNA3: RX 7900 XTX, 7900 XT, 7800 XT
- AMD RDNA2: RX 6900 XT, 6800 XT, 6700 XT
- Intel Arc: A770, A750, A580
- Intel Integrated: 12th/13th/14th Gen
- Apple Silicon: M1, M2, M3 (via MoltenVK)

## License

MIT License - Aule Technologies

## Links

- [GitHub Repository](https://github.com/xenn0010/Aule-Attention)
- [PyPI](https://pypi.org/project/aule-attention/)
