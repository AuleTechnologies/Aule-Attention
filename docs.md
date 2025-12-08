# aule-attention

**FlashAttention for Everyone. No CUDA Required.**

Run production-grade LLMs on AMD, Intel, and Apple Silicon with the speed they deserve.

Tired of `RuntimeError: CUDA not found`?
Aule Attention is a hardware-agnostic implementation of FlashAttention-2 optimized for the rest of the hardware world.
*   🚀 **Hyper-Fast:** Optimized 64-wide wavefront kernels for AMD RDNA3 & CDNA.
*   🌐 **Universal:** Runs on Vulkan Compute (Consumer GPUs) and HIP (Datacenter MI300X).
*   🧠 **Memory Efficient:** Drop-in replacement for PyTorch SDPA to double your context window.

**Installation:**
```bash
pip install aule-attention
```

**Quick Start Code:**
```python
import torch
import aule

# Make your AMD GPU go fast
torch.nn.functional.scaled_dot_product_attention = aule.flash_attention

# Load your model as usual...
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
```

**View Source on GitHub:** [https://github.com/aule-tech/aule-attention](https://github.com/aule-tech/aule-attention)

---

# Documentation

## Introduction to aule-attention

**aule-attention** is a high-performance computing library that brings the **FlashAttention-2** algorithm to non-NVIDIA hardware. It is built in **Zig** for low-level control and exposes a zero-overhead **Python** API.

### Why does this exist?
Standard attention implementations scale quadratically with sequence length ($O(N^2)$). FlashAttention reduces memory access overhead, making attention linear in memory and significantly faster. Until now, high-quality implementations were locked behind CUDA.

**Aule unlocks this performance for:**
*   **AMD Radeon** (RX 7900 XTX, 7800 XT, etc.)
*   **AMD Instinct** (MI300X, MI250)
*   **Intel Arc**
*   **Apple Silicon** (M1/M2/M3) via MoltenVK

---

## Getting Started

### Installation

**From PyPI (Recommended):**
```bash
pip install aule-attention
```

**From Source (for developers or specific needs):**
Requirements: Zig 0.14+, Vulkan SDK.
```bash
git clone https://github.com/aule-tech/aule-attention
cd aule-attention
zig build -Doptimize=ReleaseFast
cd python && python -m build --wheel
pip install dist/*.whl
```

### Usage with PyTorch

The easiest way to use Aule is as a drop-in replacement for PyTorch's Scaled Dot-Product Attention (SDPA).

```python
import torch
import aule

# Option 1: Monkey-patch globally (Simplest)
torch.nn.functional.scaled_dot_product_attention = aule.flash_attention

# Option 2: Use explicitly in your model code
from aule import flash_attention

def forward(self, q, k, v):
    return flash_attention(q, k, v, causal=True)
```

### Performance Tuning

**Batch Size:**
Aule is optimized for throughput. Increase your batch size until you saturate your VRAM. Single-threaded execution is by design; the GPU handles the parallelism.

**AMD Optimization:**
The library automatically detects AMD hardware (RDNA or CDNA architecture) and switches to a specialized "Wave64" shader kernel that uses subgroup intrinsics for maximum bandwidth utilization.

---

## Architecture

### The Unified Backend

Aule uses a hybrid backend system to target the best API for your hardware:

1.  **Vulkan Backend (Consumer):**
    *   Targets: AMD Radeon, Intel Arc, NVIDIA (fallback), Apple.
    *   Tech: SPIR-V Compute Shaders.
    *   Memory: `HOST_VISIBLE` storage buffers for zero-copy IO on integrated graphics; Staging buffers for discrete GPUs.

2.  **HIP Backend (Datacenter):**
    *   Targets: AMD MI300X / MI250.
    *   Tech: Native HIP C++ Kernels.
    *   Optimization: Direct access to Matrix Cores (CDNA 3).

### Zero-Copy Tensors

For advanced users building custom inference engines, Aule provides a `GpuTensor` API. This allows you to keep data on the GPU permanently, avoiding the overhead of copying Q/K/V matrices from the CPU every forward pass.

```python
# Advanced Zero-Copy Usage
import aule
import numpy as np

# Assuming you've initialized aule
ctx = aule.AttentionContext()

batch, heads, seq, dim = 1, 12, 512, 64
shape = (batch, heads, seq, dim)
q_cpu_data = np.random.randn(*shape).astype(np.float32)
k_cpu_data = np.random.randn(*shape).astype(np.float32)
v_cpu_data = np.random.randn(*shape).astype(np.float32)

q_gpu = ctx.tensor(shape)
q_gpu.upload(q_cpu_data)
k_gpu = ctx.tensor(shape)
k_gpu.upload(k_cpu_data)
v_gpu = ctx.tensor(shape)
v_gpu.upload(v_cpu_data)

out_gpu = ctx.tensor(shape)

# This call has practically zero CPU overhead
ctx.attention(q_gpu, k_gpu, v_gpu, out_gpu)

result_cpu = out_gpu.download()

q_gpu.free()
k_gpu.free()
v_gpu.free()
out_gpu.free()
ctx.shutdown()
```

---

## Launch Announcement

**Title:** Unlocking the MI300X and Radeon: Introducing aule-attention

We are excited to launch **aule-attention**, the missing piece of the puzzle for open-source AI on AMD hardware.

For too long, "FlashAttention" has been synonymous with "NVIDIA". Today, we are breaking that dependency.

**What is it?**
A highly optimized implementation of FlashAttention-2 that runs natively on Vulkan and HIP.

**What does it mean for you?**
*   **Radeon Users:** Your RX 7900 XTX is now a first-class citizen for LLM inference. Expect 3-4x speedups in token generation compared to standard eager implementation.
*   **MI300X Users:** Finally, a lightweight, portable kernel that pushes your hardware without the complexity of the full ROCm rigid stack.

Go fast. Run everywhere.
👉 `pip install aule-attention`
⭐ Star us on GitHub: [github.com/aule-tech/aule-attention](https://github.com/aule-tech/aule-attention)
