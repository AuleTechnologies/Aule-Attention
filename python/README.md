# aule-attention

**Hardware-agnostic FlashAttention implementation. No compilation required. Works on any GPU.**

Version: 0.2.0

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

## Features

- No compilation at install time
- Works on AMD, NVIDIA, Intel, and Apple GPUs
- Training support with backward pass (Triton backend)
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA) support
- O(N) memory complexity

## Backends

| Backend | Hardware | Features |
|---------|----------|----------|
| Triton | AMD ROCm, NVIDIA CUDA | Training and Inference |
| Vulkan | Any Vulkan 1.2+ GPU | Inference |
| CPU | NumPy | Fallback |

## API

```python
from aule import flash_attention, get_available_backends, print_backend_info

# Compute attention
output = flash_attention(query, key, value, causal=True, scale=None)

# Check available backends
backends = get_available_backends()

# Display backend information
print_backend_info()
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

- [GitHub Repository](https://github.com/aule-dev/aule-attention)
- [Documentation](https://github.com/aule-dev/aule-attention#readme)
