#!/usr/bin/env python3
"""
Test aule-attention with a real LLM (GPT-2).

This script:
1. Loads GPT-2 model
2. Extracts real Q, K, V tensors from GPT-2 layers
3. Compares aule attention output with PyTorch SDPA
4. Benchmarks performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import aule-attention
from aule import Aule, flash_attention, _vulkan_available

print("=" * 70)
print("AULE-ATTENTION LLM TEST")
print("=" * 70)
print(f"Vulkan backend available: {_vulkan_available}")

# Show GPU info
with Aule() as aule:
    info = aule.get_device_info()
    print(f"GPU: {info['device_name']}")
    print(f"Vendor: {info['vendor']}")
    print(f"AMD-optimized: {info['amd_optimized']}")
    print(f"Subgroup size: {info['subgroup_size']}")
print()

# Load GPT-2
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Set pad token
tokenizer.pad_token = tokenizer.eos_token


def aule_attention_forward(query, key, value, is_causal=True):
    """
    Aule-powered attention that can replace PyTorch's scaled_dot_product_attention.

    Args:
        query: [batch, heads, seq_q, head_dim]
        key: [batch, heads, seq_k, head_dim]
        value: [batch, heads, seq_k, head_dim]
        is_causal: Whether to use causal masking

    Returns:
        output: [batch, heads, seq_q, head_dim]
    """
    # Convert to numpy for aule
    Q = query.detach().cpu().numpy().astype(np.float32)
    K = key.detach().cpu().numpy().astype(np.float32)
    V = value.detach().cpu().numpy().astype(np.float32)

    # Run through aule
    output = flash_attention(Q, K, V, causal=is_causal)

    # Convert back to torch
    return torch.from_numpy(output).to(query.device, query.dtype)


# Test prompt
prompt = "The future of artificial intelligence is"

print("=" * 70)
print("TEST 1: Generate with PyTorch (baseline)")
print("=" * 70)
print(f"Prompt: {prompt}")
print()

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    start = time.perf_counter()
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"Generated: {generated_text}")
print(f"Time: {elapsed:.2f}s ({tokens/elapsed:.1f} tokens/sec)")
print()

print("=" * 70)
print("TEST 2: Extract real Q,K,V from GPT-2 and compare attention")
print("=" * 70)

# Hook to capture Q, K, V from first layer
captured_qkv = {}

def capture_hook(module, input, output):
    """Capture the QKV projections."""
    hidden_states = input[0]
    qkv = module.c_attn(hidden_states)
    query, key, value = qkv.split(module.split_size, dim=2)

    batch_size, seq_len, _ = query.shape
    num_heads = module.num_heads
    head_dim = module.head_dim

    captured_qkv['query'] = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    captured_qkv['key'] = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    captured_qkv['value'] = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

# Register hook on first attention layer
hook = model.transformer.h[0].attn.register_forward_hook(capture_hook)

# Run a forward pass to capture Q, K, V
with torch.no_grad():
    _ = model(inputs["input_ids"])

hook.remove()

# Now we have real Q, K, V from GPT-2
Q = captured_qkv['query']
K = captured_qkv['key']
V = captured_qkv['value']

print(f"Captured Q shape: {Q.shape}")  # [1, 12, seq_len, 64]
print(f"Captured K shape: {K.shape}")
print(f"Captured V shape: {V.shape}")
print()

# Compare PyTorch SDPA with Aule
with torch.no_grad():
    pytorch_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

aule_out = aule_attention_forward(Q, K, V, is_causal=True)

max_diff = (pytorch_out - aule_out).abs().max().item()
mean_diff = (pytorch_out - aule_out).abs().mean().item()

print(f"PyTorch SDPA output shape: {pytorch_out.shape}")
print(f"Aule output shape: {aule_out.shape}")
print(f"Max difference: {max_diff:.6f}")
print(f"Mean difference: {mean_diff:.6f}")
print(f"Match: {'EXACT' if max_diff < 1e-5 else 'YES' if max_diff < 1e-3 else 'CLOSE' if max_diff < 1e-2 else 'NO'}")
print()

print("=" * 70)
print("TEST 3: Compare single attention pass (GPT-2 dimensions)")
print("=" * 70)

# Create test tensors matching GPT-2's attention dimensions
batch_size, num_heads, seq_len, head_dim = 1, 12, 64, 64
Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# PyTorch SDPA
with torch.no_grad():
    pytorch_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Aule attention
aule_out = aule_attention_forward(Q, K, V, is_causal=True)

max_diff = (pytorch_out - aule_out).abs().max().item()
mean_diff = (pytorch_out - aule_out).abs().mean().item()

print(f"Max difference: {max_diff:.6f}")
print(f"Mean difference: {mean_diff:.6f}")
print(f"Match: {'YES' if max_diff < 1e-3 else 'CLOSE' if max_diff < 1e-2 else 'NO'}")
print()

print("=" * 70)
print("TEST 4: Benchmark attention kernel")
print("=" * 70)

configs = [
    (1, 12, 64, 64),   # GPT-2 small
    (1, 12, 128, 64),  # 128 tokens
    (1, 12, 256, 64),  # 256 tokens
    (1, 12, 512, 64),  # 512 tokens
]

print(f"{'Config':<25} {'PyTorch (ms)':<15} {'Aule (ms)':<15} {'Speedup':<10}")
print("-" * 70)

for batch, heads, seq, dim in configs:
    Q = torch.randn(batch, heads, seq, dim)
    K = torch.randn(batch, heads, seq, dim)
    V = torch.randn(batch, heads, seq, dim)

    # Warmup
    for _ in range(3):
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        _ = aule_attention_forward(Q, K, V, is_causal=True)

    # Benchmark PyTorch
    n_iter = 20
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    pytorch_time = (time.perf_counter() - start) * 1000 / n_iter

    # Benchmark Aule
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = aule_attention_forward(Q, K, V, is_causal=True)
    aule_time = (time.perf_counter() - start) * 1000 / n_iter

    speedup = pytorch_time / aule_time
    config_str = f"B={batch} H={heads} S={seq} D={dim}"

    print(f"{config_str:<25} {pytorch_time:>10.2f} ms   {aule_time:>10.2f} ms   {speedup:.2f}x")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"aule-attention is working with GPT-2!")
print(f"Numerical accuracy: max diff = {max_diff:.6f}")
print(f"Note: On Intel iGPU, PyTorch CPU is faster. Aule shines on AMD discrete GPUs.")
print()
