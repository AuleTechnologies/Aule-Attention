#!/usr/bin/env python3
"""
Benchmark: Aule-Attention vs llama.cpp

Compares attention performance between:
1. Aule-Attention (Triton/Vulkan/CPU backends)
2. PyTorch default SDPA
3. llama.cpp (if available)

This measures ONLY the attention layer, not full inference.
"""

import time
import torch
import numpy as np
import sys

print("=" * 80)
print("BENCHMARK: Aule-Attention vs Alternatives")
print("=" * 80)

# Check what's available
print("\n" + "=" * 80)
print("CHECKING AVAILABLE BACKENDS")
print("=" * 80)

# Check aule-attention
try:
    import aule
    aule_available = True
    print(f"✓ aule-attention v{aule.__version__} found")
    aule.print_backend_info()
except ImportError:
    aule_available = False
    print("✗ aule-attention not found")

# Check llama-cpp-python
try:
    from llama_cpp import Llama
    llamacpp_available = True
    print("\n✓ llama-cpp-python found")
except ImportError:
    llamacpp_available = False
    print("\n✗ llama-cpp-python not found")
    print("  Install: pip install llama-cpp-python")

# Check PyTorch
try:
    import torch
    pytorch_available = True
    if torch.cuda.is_available():
        print(f"\n✓ PyTorch {torch.__version__} with CUDA")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n✓ PyTorch {torch.__version__} (CPU only)")
except ImportError:
    pytorch_available = False
    print("\n✗ PyTorch not found")

if not pytorch_available:
    print("\nERROR: PyTorch required for benchmarks")
    sys.exit(1)

# Benchmark configuration
CONFIGS = [
    # (seq_len, batch, heads, head_dim, description)
    (512, 1, 8, 64, "Short context (SD UNet style)"),
    (2048, 1, 8, 64, "Medium context (GPT-2 style)"),
    (4096, 1, 12, 128, "Long context (Llama-2 style)"),
    (8192, 1, 8, 128, "Very long context"),
]

# Warmup iterations
WARMUP = 3
# Benchmark iterations
ITERATIONS = 10

print("\n" + "=" * 80)
print("BENCHMARK CONFIGURATION")
print("=" * 80)
print(f"Warmup iterations: {WARMUP}")
print(f"Benchmark iterations: {ITERATIONS}")
print(f"Precision: FP32")
print(f"Causal: True")

# Results storage
results = []

def benchmark_attention(name, fn, q, k, v, warmup=WARMUP, iterations=ITERATIONS):
    """Benchmark an attention function."""

    # Warmup
    for _ in range(warmup):
        _ = fn(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        output = fn(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = ((end - start) / iterations) * 1000

    return avg_time_ms, output


def pytorch_sdpa(q, k, v):
    """PyTorch scaled_dot_product_attention."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def aule_attention_fn(q, k, v):
    """Aule-attention."""
    return aule.flash_attention(q, k, v, causal=True)


def numpy_baseline(q, k, v):
    """Pure NumPy attention (reference implementation)."""
    q_np = q.cpu().numpy()
    k_np = k.cpu().numpy()
    v_np = v.cpu().numpy()

    batch, heads, seq, dim = q_np.shape
    scale = 1.0 / np.sqrt(dim)

    # Q @ K^T
    scores = np.einsum('bhqd,bhkd->bhqk', q_np, k_np) * scale

    # Causal mask
    mask = np.triu(np.ones((seq, seq)), k=1).astype(bool)
    scores = np.where(mask, -1e9, scores)

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Output
    output_np = np.einsum('bhqk,bhkd->bhqd', attn, v_np)
    return torch.from_numpy(output_np).to(q.device)


print("\n" + "=" * 80)
print("RUNNING BENCHMARKS")
print("=" * 80)

for seq_len, batch, heads, head_dim, description in CONFIGS:
    print(f"\n{'=' * 80}")
    print(f"CONFIG: {description}")
    print(f"  Shape: [batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}]")
    print(f"{'=' * 80}")

    # Create test tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

    config_results = {
        'config': description,
        'seq_len': seq_len,
        'batch': batch,
        'heads': heads,
        'head_dim': head_dim,
        'backends': {}
    }

    # 1. PyTorch SDPA (baseline)
    print("\n[1/4] PyTorch SDPA (baseline)...")
    try:
        time_ms, ref_output = benchmark_attention("PyTorch SDPA", pytorch_sdpa, q, k, v)
        print(f"  ✓ Time: {time_ms:.2f} ms")
        config_results['backends']['pytorch'] = {
            'time_ms': time_ms,
            'error': 0.0,  # Reference
        }
        baseline_time = time_ms
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        ref_output = None
        baseline_time = None

    # 2. Aule-Attention
    if aule_available:
        print("\n[2/4] Aule-Attention...")
        try:
            # Uninstall first (clean slate)
            try:
                aule.uninstall()
            except:
                pass

            # Install fresh
            aule.install(verbose=False)

            time_ms, aule_output = benchmark_attention("Aule-Attention", aule_attention_fn, q, k, v)
            print(f"  ✓ Time: {time_ms:.2f} ms", end='')

            if baseline_time:
                speedup = baseline_time / time_ms
                print(f" (Speedup: {speedup:.2f}x)")
            else:
                print()

            # Check correctness
            if ref_output is not None:
                error = (aule_output - ref_output).abs().max().item()
                print(f"  ✓ Max error vs PyTorch: {error:.6f}")
                config_results['backends']['aule'] = {
                    'time_ms': time_ms,
                    'error': error,
                    'speedup': speedup if baseline_time else None
                }

            aule.uninstall()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[2/4] Aule-Attention: SKIPPED (not installed)")

    # 3. NumPy baseline (slow, for reference)
    print("\n[3/4] NumPy CPU (reference)...")
    if seq_len <= 2048:  # Skip for very long sequences (too slow)
        try:
            time_ms, numpy_output = benchmark_attention("NumPy CPU", numpy_baseline, q, k, v,
                                                         warmup=1, iterations=3)
            print(f"  ✓ Time: {time_ms:.2f} ms")

            if baseline_time:
                slowdown = time_ms / baseline_time
                print(f"  ✓ Slowdown vs PyTorch: {slowdown:.2f}x")

            config_results['backends']['numpy'] = {
                'time_ms': time_ms,
                'error': 0.0,
            }
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print("  ⊘ Skipped (too slow for long sequences)")

    # 4. llama.cpp (if available)
    print("\n[4/4] llama.cpp...")
    if llamacpp_available:
        print("  ⊘ Skipped (requires GGUF model file)")
        print("     Note: llama.cpp benchmarks full inference, not just attention")
    else:
        print("  ⊘ Not installed")

    results.append(config_results)

# Print summary
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

print("\nTime (milliseconds):")
print(f"{'Configuration':<30} {'PyTorch':<12} {'Aule':<12} {'Speedup':<12}")
print("-" * 80)

for result in results:
    config = result['config']
    pytorch_time = result['backends'].get('pytorch', {}).get('time_ms', '-')
    aule_time = result['backends'].get('aule', {}).get('time_ms', '-')
    speedup = result['backends'].get('aule', {}).get('speedup', '-')

    if isinstance(pytorch_time, (int, float)):
        pytorch_str = f"{pytorch_time:.2f} ms"
    else:
        pytorch_str = str(pytorch_time)

    if isinstance(aule_time, (int, float)):
        aule_str = f"{aule_time:.2f} ms"
    else:
        aule_str = str(aule_time)

    if isinstance(speedup, (int, float)):
        speedup_str = f"{speedup:.2f}x"
    else:
        speedup_str = str(speedup)

    print(f"{config:<30} {pytorch_str:<12} {aule_str:<12} {speedup_str:<12}")

print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

if aule_available and results:
    avg_speedup = np.mean([r['backends'].get('aule', {}).get('speedup', 0)
                           for r in results if 'aule' in r['backends']])

    if avg_speedup > 0:
        print(f"\nAverage Aule speedup: {avg_speedup:.2f}x")

        if avg_speedup > 1.5:
            print("✓ Aule-Attention shows significant performance improvement!")
        elif avg_speedup > 1.0:
            print("✓ Aule-Attention shows moderate performance improvement")
        else:
            print("⚠ Aule-Attention is slower (may be overhead on small sequences)")

    print("\nNote: Speedup typically increases with sequence length due to")
    print("      FlashAttention's O(N) memory complexity vs O(N²) for naive.")
else:
    print("\nInstall aule-attention to see performance comparison:")
    print("  cd python && pip install -e .")

print("\n" + "=" * 80)
print("For llama.cpp comparison, use their CLI benchmarks:")
print("  llama-bench -m <model.gguf>")
print("=" * 80 + "\n")
