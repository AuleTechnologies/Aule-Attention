# aule-attention

**FlashAttention that just works. No compilation. Any GPU.**

Version: 0.5.0

## What's New in 0.5.0

- **Sliding Window Attention**: Mistral-style local attention with `window_size` parameter
- **PagedAttention**: vLLM-compatible block-based KV cache for efficient serving
- **Native GQA**: No K/V tensor expansion needed - faster than PyTorch on GQA models
- **NaN Fix**: Stable numerics for sliding window with fully-masked blocks

### MI300X Benchmark Results (vs PyTorch SDPA)

**GQA Models** (native GQA vs PyTorch expand):

| Config | Speedup |
|--------|---------|
| LLaMA-70B 4K context | **+6.1%** |
| LLaMA-70B batch=4 | **+6.0%** |
| LLaMA-405B 4K context | **+8.5%** |
| Mistral batch=8 | **+9.6%** |

**PagedAttention Decode** (batch=8):

| Context Length | Throughput |
|----------------|------------|
| 1K tokens | 34,397 tok/s |
| 2K tokens | 20,083 tok/s |
| 4K tokens | 10,915 tok/s |
| 8K tokens | 5,744 tok/s |

**Sliding Window** (window=256 vs full attention):

| Sequence Length | Speedup |
|-----------------|---------|
| 2K tokens | +6.0% |
| 4K tokens | +8.9% |
| 8K tokens | +11.0% |

### Sliding Window Attention

```python
from aule import flash_attention

# Mistral-style sliding window (only attend to last N tokens)
output = flash_attention(q, k, v, causal=True, window_size=256)

# For long sequences, reduces O(N^2) to O(N*W) memory
# seq=8K, window=256 → 97% memory savings
```

### PagedAttention (vLLM-compatible)

```python
from aule import flash_attention_paged_amd

# Block-based KV cache for efficient serving
output = flash_attention_paged_amd(
    q,              # [batch, heads_q, 1, head_dim]
    k_cache,        # [num_blocks, block_size, heads_kv, head_dim]
    v_cache,        # [num_blocks, block_size, heads_kv, head_dim]
    block_tables,   # [batch, max_blocks] int32
    context_lens,   # [batch] int32
    window_size=-1  # optional sliding window
)
```

## What's New in 0.4.0

- **AMD MI300X Optimized Kernel**: New Triton FlashAttention-2 kernel tuned for CDNA3
  - Auto-detects AMD GPUs and routes to optimized kernel
  - Uses `exp2` optimization (faster than `exp` on AMD hardware)
  - Extended autotune configs for 7B/13B/70B/405B models

## What's New in 0.3.7

- **Windows DLL included** - Cross-compiled Windows support with PagedAttention

## What's New in 0.3.6

- **PagedAttention (vLLM-style)**: Block-based KV cache for 90% memory savings
- **7-13x Faster Vulkan**: New fast shader with 32x32 blocks and vec4 loads
- **Native FP16/BF16 Compute**: Triton kernels now use native precision
- **Multiple Shader Variants**: Choose baseline, fast, fp16, or fp16_amd for your hardware

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
# Native GQA - no K/V expansion needed!
q = torch.randn(1, 32, 512, 128, device='cuda')
k = torch.randn(1, 8, 512, 128, device='cuda')
v = torch.randn(1, 8, 512, 128, device='cuda')

output = flash_attention(q, k, v, causal=True)
```

## Features

- No compilation at install time
- Works on AMD, NVIDIA, Intel, and Apple GPUs
- Training support with backward pass (Triton backend)
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA) support
- Cross-attention with different Q/KV sequence lengths
- Sliding window attention for efficient long sequences
- PagedAttention for vLLM-style serving
- O(N) memory complexity via FlashAttention-2 algorithm

## Backends

| Backend | Hardware | Features |
|---------|----------|----------|
| Triton-AMD | AMD ROCm (Linux) | Training + Inference, GQA, Sliding Window, PagedAttention |
| Triton | NVIDIA CUDA | Training + Inference, head_dim up to 128 |
| Vulkan | Any Vulkan 1.2+ GPU | Inference, head_dim up to 64, GQA/MQA |
| CPU | NumPy | Fallback, any head_dim |

> **Windows + AMD**: Automatically uses Vulkan backend (Triton AMD only supports Linux).

## API

```python
from aule import flash_attention, get_available_backends, install

# Compute attention
output = flash_attention(query, key, value, causal=True, scale=None, window_size=-1)

# Check available backends
backends = get_available_backends()  # ['triton-amd', 'triton', 'vulkan', 'cpu']

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

- [GitHub Repository](https://github.com/AuleTechnologies/Aule-Attention)
- [PyPI](https://pypi.org/project/aule-attention/)
