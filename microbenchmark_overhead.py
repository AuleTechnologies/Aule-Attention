#!/usr/bin/env python3
"""
Micro-benchmark to identify exact overhead bottlenecks.
"""

import time
import torch
import numpy as np
import aule

print("=" * 80)
print("MICRO-BENCHMARK: Overhead Analysis")
print("=" * 80)

# Test configuration
batch, heads, seq, dim = 1, 8, 512, 64
q = torch.randn(batch, heads, seq, dim)
k = torch.randn(batch, heads, seq, dim)
v = torch.randn(batch, heads, seq, dim)

# Warmup
for _ in range(3):
    _ = aule.flash_attention(q, k, v, causal=True)

print("\n" + "=" * 80)
print("BREAKDOWN: Where does the time go?")
print("=" * 80)

iterations = 100

# 1. Full pipeline (baseline)
print("\n[1/5] Full Aule pipeline...")
start = time.perf_counter()
for _ in range(iterations):
    _ = aule.flash_attention(q, k, v, causal=True)
end = time.perf_counter()
full_time_ms = ((end - start) / iterations) * 1000
print(f"  Time: {full_time_ms:.3f} ms/iter")

# 2. PyTorch SDPA (comparison)
print("\n[2/5] PyTorch SDPA...")
start = time.perf_counter()
for _ in range(iterations):
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
end = time.perf_counter()
pytorch_time_ms = ((end - start) / iterations) * 1000
print(f"  Time: {pytorch_time_ms:.3f} ms/iter")
print(f"  Gap: {full_time_ms / pytorch_time_ms:.2f}x slower")

# 3. Just tensor conversion (torch -> numpy)
print("\n[3/5] Torch->NumPy conversion overhead...")
start = time.perf_counter()
for _ in range(iterations):
    q_np = q.cpu().numpy()
    k_np = k.cpu().numpy()
    v_np = v.cpu().numpy()
end = time.perf_counter()
conversion_time_ms = ((end - start) / iterations) * 1000
print(f"  Time: {full_time_ms:.3f} ms/iter")
print(f"  Percentage of total: {(conversion_time_ms / full_time_ms) * 100:.1f}%")

# 4. NumPy attention (pure CPU)
print("\n[4/5] NumPy CPU attention...")
q_np = q.cpu().numpy()
k_np = k.cpu().numpy()
v_np = v.cpu().numpy()

def numpy_attention(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.einsum('bhqd,bhkd->bhqk', q, k) * scale
    # Causal mask
    mask = np.triu(np.ones((seq, seq)), k=1).astype(bool)
    scores = np.where(mask, -1e9, scores)
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    # Output
    return np.einsum('bhqk,bhkd->bhqd', attn, v)

# Warmup
for _ in range(3):
    _ = numpy_attention(q_np, k_np, v_np)

start = time.perf_counter()
for _ in range(iterations):
    _ = numpy_attention(q_np, k_np, v_np)
end = time.perf_counter()
numpy_time_ms = ((end - start) / iterations) * 1000
print(f"  Time: {numpy_time_ms:.3f} ms/iter")

# 5. Vulkan.attention() directly (bypass torch wrapper)
print("\n[5/5] Direct Vulkan call (no torch wrapper)...")
from aule.vulkan import attention as vulkan_attention

start = time.perf_counter()
for _ in range(iterations):
    _ = vulkan_attention(q_np, k_np, v_np, causal=True)
end = time.perf_counter()
vulkan_direct_time_ms = ((end - start) / iterations) * 1000
print(f"  Time: {vulkan_direct_time_ms:.3f} ms/iter")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nPyTorch SDPA:           {pytorch_time_ms:7.3f} ms")
print(f"Aule (full):            {full_time_ms:7.3f} ms ({full_time_ms / pytorch_time_ms:.1f}x slower)")
print(f"Aule (vulkan direct):   {vulkan_direct_time_ms:7.3f} ms ({vulkan_direct_time_ms / pytorch_time_ms:.1f}x slower)")
print(f"NumPy CPU:              {numpy_time_ms:7.3f} ms ({numpy_time_ms / pytorch_time_ms:.1f}x slower)")

print(f"\nOverhead breakdown:")
print(f"  Torch wrapper overhead: {full_time_ms - vulkan_direct_time_ms:.3f} ms ({((full_time_ms - vulkan_direct_time_ms) / full_time_ms) * 100:.1f}%)")
print(f"  Vulkan GPU compute:     {vulkan_direct_time_ms:.3f} ms ({(vulkan_direct_time_ms / full_time_ms) * 100:.1f}%)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if vulkan_direct_time_ms < pytorch_time_ms:
    print(f"\n✓ Vulkan backend is {pytorch_time_ms / vulkan_direct_time_ms:.2f}x FASTER than PyTorch!")
    print(f"  Problem: Torch wrapper adds {full_time_ms - vulkan_direct_time_ms:.3f} ms overhead")
elif vulkan_direct_time_ms < numpy_time_ms:
    print(f"\n✓ Vulkan backend is {numpy_time_ms / vulkan_direct_time_ms:.2f}x faster than NumPy")
    print(f"  ✗ But still {vulkan_direct_time_ms / pytorch_time_ms:.2f}x slower than PyTorch")
    print(f"\n  Likely cause: PyTorch using optimized oneDNN/MKL on CPU")
    print(f"  Solution: Intel iGPU compute cores may not be faster than CPU for this size")
else:
    print(f"\n✗ Vulkan backend slower than everything")
    print(f"  This is unexpected - GPU should beat pure NumPy")

print("\n" + "=" * 80)
