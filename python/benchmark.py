#!/usr/bin/env python3
"""
Comprehensive benchmark suite for aule-attention backends.

Compares performance across:
- HIP MFMA (matrix cores for MI200/MI300)
- HIP scalar (ROCm baseline)
- OpenCL (portable via Mesa Rusticl)
- CPU (reference)
- PyTorch FlashAttention (if available)

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --backend hip_mfma # Test specific backend
    python benchmark.py --seq-len 4096     # Custom sequence length
"""

import argparse
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import sys

# Import backends
try:
    import aule_unified as aule
except ImportError:
    print("Error: aule_unified not found. Run from python/ directory.")
    sys.exit(1)

# Try PyTorch for comparison
PYTORCH_AVAILABLE = False
FLASH_ATTN_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass


def benchmark_backend(
    backend: str,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
    dtype: str = 'float32',
    n_warmup: int = 3,
    n_iters: int = 10
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Benchmark a specific backend.

    Returns:
        (output, time_ms, tflops)
    """
    batch, heads, seq_len, head_dim = Q.shape

    try:
        with aule.Attention(backend=backend) as attn:
            # Warmup
            for _ in range(n_warmup):
                _ = attn.forward(Q, K, V, causal=causal, dtype=dtype)

            # Benchmark
            start = time.perf_counter()
            for _ in range(n_iters):
                output = attn.forward(Q, K, V, causal=causal, dtype=dtype)
            elapsed = time.perf_counter() - start

            time_ms = elapsed / n_iters * 1000

            # Calculate TFLOPS
            # FlashAttention: 4 * batch * heads * seq^2 * dim (2 matmuls + online softmax)
            flops = 4 * batch * heads * seq_len * seq_len * head_dim
            tflops = flops / (elapsed / n_iters) / 1e12

            return output, time_ms, tflops

    except Exception as e:
        print(f"  {backend}: FAILED - {e}")
        return None, 0, 0


def benchmark_pytorch(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
    dtype: str = 'float32',
    n_warmup: int = 3,
    n_iters: int = 10
) -> Tuple[Optional[np.ndarray], float, float]:
    """Benchmark PyTorch scaled_dot_product_attention."""
    if not FLASH_ATTN_AVAILABLE:
        return None, 0, 0

    batch, heads, seq_len, head_dim = Q.shape

    # Convert to PyTorch tensors
    torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Q_t = torch.from_numpy(Q).to(device=device, dtype=torch_dtype)
    K_t = torch.from_numpy(K).to(device=device, dtype=torch_dtype)
    V_t = torch.from_numpy(V).to(device=device, dtype=torch_dtype)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = torch.nn.functional.scaled_dot_product_attention(
                Q_t, K_t, V_t, is_causal=causal
            )
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            output = torch.nn.functional.scaled_dot_product_attention(
                Q_t, K_t, V_t, is_causal=causal
            )
        if device == 'cuda':
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_ms = elapsed / n_iters * 1000

    # Calculate TFLOPS
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed / n_iters) / 1e12

    output_np = output.cpu().numpy()
    return output_np, time_ms, tflops


def run_benchmark_suite(
    batch_size: int = 1,
    num_heads: int = 8,
    seq_len: int = 2048,
    head_dim: int = 64,
    backends: Optional[List[str]] = None,
    dtype: str = 'float32'
):
    """Run comprehensive benchmark suite."""

    print("=" * 70)
    print("AULE-ATTENTION BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Batch size:    {batch_size}")
    print(f"  Num heads:     {num_heads}")
    print(f"  Sequence len:  {seq_len}")
    print(f"  Head dim:      {head_dim}")
    print(f"  Dtype:         {dtype}")
    print(f"  Total params:  {batch_size * num_heads * seq_len * head_dim * 3 / 1e6:.1f}M")

    # Generate random inputs
    np.random.seed(42)
    np_dtype = np.float16 if dtype == 'float16' else np.float32
    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)

    print(f"\nAvailable backends: {aule.get_available_backends()}")

    if backends is None:
        backends = aule.get_available_backends()

    results = {}

    # Run benchmarks
    print("\n" + "-" * 70)
    print("RESULTS (Non-Causal Attention)")
    print("-" * 70)
    print(f"{'Backend':<25} {'Time (ms)':<15} {'TFLOPS':<15} {'Status':<15}")
    print("-" * 70)

    for backend in backends:
        output, time_ms, tflops = benchmark_backend(
            backend, Q, K, V, causal=False, dtype=dtype
        )
        if output is not None:
            results[backend] = {'output': output, 'time_ms': time_ms, 'tflops': tflops}
            print(f"{backend:<25} {time_ms:<15.2f} {tflops:<15.3f} OK")
        else:
            print(f"{backend:<25} {'N/A':<15} {'N/A':<15} FAILED")

    # PyTorch comparison
    if FLASH_ATTN_AVAILABLE:
        output, time_ms, tflops = benchmark_pytorch(Q, K, V, causal=False, dtype=dtype)
        if output is not None:
            results['pytorch_sdpa'] = {'output': output, 'time_ms': time_ms, 'tflops': tflops}
            device = 'CUDA' if torch.cuda.is_available() else 'CPU'
            print(f"{'pytorch_sdpa (' + device + ')':<25} {time_ms:<15.2f} {tflops:<15.3f} OK")

    # Causal attention benchmark
    print("\n" + "-" * 70)
    print("RESULTS (Causal Attention)")
    print("-" * 70)
    print(f"{'Backend':<25} {'Time (ms)':<15} {'TFLOPS':<15} {'Status':<15}")
    print("-" * 70)

    for backend in backends:
        output, time_ms, tflops = benchmark_backend(
            backend, Q, K, V, causal=True, dtype=dtype
        )
        if output is not None:
            print(f"{backend:<25} {time_ms:<15.2f} {tflops:<15.3f} OK")
        else:
            print(f"{backend:<25} {'N/A':<15} {'N/A':<15} FAILED")

    if FLASH_ATTN_AVAILABLE:
        output, time_ms, tflops = benchmark_pytorch(Q, K, V, causal=True, dtype=dtype)
        if output is not None:
            device = 'CUDA' if torch.cuda.is_available() else 'CPU'
            print(f"{'pytorch_sdpa (' + device + ')':<25} {time_ms:<15.2f} {tflops:<15.3f} OK")

    # Correctness verification
    print("\n" + "-" * 70)
    print("CORRECTNESS VERIFICATION")
    print("-" * 70)

    # Use CPU as reference
    if 'cpu' in results:
        ref_output = results['cpu']['output']
        print(f"Reference: CPU (numpy)")

        for backend, data in results.items():
            if backend == 'cpu':
                continue
            max_diff = np.abs(data['output'] - ref_output).max()
            mean_diff = np.abs(data['output'] - ref_output).mean()
            status = "PASS" if max_diff < 1e-3 else "WARN" if max_diff < 1e-2 else "FAIL"
            print(f"  {backend:<20} max_diff={max_diff:.6f}  mean_diff={mean_diff:.8f}  [{status}]")

    # Performance comparison
    print("\n" + "-" * 70)
    print("PERFORMANCE COMPARISON")
    print("-" * 70)

    if 'cpu' in results:
        cpu_time = results['cpu']['time_ms']
        for backend, data in results.items():
            if backend == 'cpu':
                print(f"  {backend:<20} 1.00x (baseline)")
            else:
                speedup = cpu_time / data['time_ms']
                print(f"  {backend:<20} {speedup:.2f}x vs CPU")

    # Find best GPU backend
    gpu_backends = [b for b in results if b not in ('cpu', 'pytorch_sdpa')]
    if gpu_backends:
        best_gpu = max(gpu_backends, key=lambda b: results[b]['tflops'])
        print(f"\nBest GPU backend: {best_gpu} ({results[best_gpu]['tflops']:.3f} TFLOPS)")

    print("\n" + "=" * 70)

    return results


def run_scaling_benchmark(backends: Optional[List[str]] = None):
    """Benchmark performance scaling with sequence length."""

    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH SCALING")
    print("=" * 70)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    if backends is None:
        backends = aule.get_available_backends()
        # Exclude CPU for scaling benchmark (too slow)
        backends = [b for b in backends if b != 'cpu']

    results = {b: [] for b in backends}

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        np.random.seed(42)
        Q = np.random.randn(1, 8, seq_len, 64).astype(np.float32)
        K = np.random.randn(1, 8, seq_len, 64).astype(np.float32)
        V = np.random.randn(1, 8, seq_len, 64).astype(np.float32)

        for backend in backends:
            output, time_ms, tflops = benchmark_backend(
                backend, Q, K, V, causal=True, dtype='float32', n_iters=5
            )
            if output is not None:
                results[backend].append((seq_len, time_ms, tflops))
                print(f"  {backend:<20} {time_ms:>8.2f} ms  {tflops:>8.3f} TFLOPS")

    # Print summary table
    print("\n" + "-" * 70)
    print("SCALING SUMMARY (Time in ms)")
    print("-" * 70)

    header = f"{'Backend':<20}" + "".join(f"{s:>10}" for s in seq_lengths)
    print(header)
    print("-" * 70)

    for backend in backends:
        if results[backend]:
            times = [f"{t[1]:>10.2f}" for t in results[backend]]
            print(f"{backend:<20}" + "".join(times))

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark aule-attention backends')
    parser.add_argument('--backend', type=str, help='Specific backend to test')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'])
    parser.add_argument('--scaling', action='store_true', help='Run sequence length scaling benchmark')

    args = parser.parse_args()

    backends = [args.backend] if args.backend else None

    # Main benchmark
    run_benchmark_suite(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        backends=backends,
        dtype=args.dtype
    )

    # Scaling benchmark
    if args.scaling:
        run_scaling_benchmark(backends=backends)


if __name__ == '__main__':
    main()
