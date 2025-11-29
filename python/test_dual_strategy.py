#!/usr/bin/env python3
"""
Test script for dual-strategy aule-attention

This validates that the backend selection works correctly:
1. WITH ROCm: Uses CK FlashAttention for peak performance (~150-300 TFLOPS)
2. WITHOUT ROCm: Uses optimized OpenCL for portability (~0.5 TFLOPS)

Run this script to:
- See which backends are available
- Benchmark each backend
- Verify numerical correctness
"""

import numpy as np
import time
import sys


def test_backend_detection():
    """Test that backend detection works correctly."""
    print("=" * 70)
    print("BACKEND DETECTION TEST")
    print("=" * 70)

    import aule_unified as aule

    available = aule.get_available_backends()
    info = aule.get_backend_info()

    print(f"\nAvailable backends (in priority order):")
    for i, backend in enumerate(available, 1):
        perf = info.get(backend, {}).get('performance', 'Unknown')
        print(f"  {i}. {backend} ({perf})")

    best = available[0] if available else 'None'
    print(f"\nBest available: {best}")

    # Check expected behavior
    if 'ck' in available or 'rocm_flash' in available:
        print("\n[OK] ROCm path available - will use CK FlashAttention")
    elif 'opencl_optimized' in available:
        print("\n[OK] OpenCL path available - will use optimized vec4 kernel")
    elif 'opencl' in available:
        print("\n[OK] OpenCL path available - will use basic kernel")
    else:
        print("\n[WARN] Only CPU fallback available")

    return available


def test_numerical_correctness():
    """Test that all backends produce correct results."""
    print("\n" + "=" * 70)
    print("NUMERICAL CORRECTNESS TEST")
    print("=" * 70)

    import aule_unified as aule

    # Small test case
    np.random.seed(42)
    batch, heads, seq, dim = 1, 4, 32, 64
    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

    # Reference: CPU implementation
    print("\nComputing CPU reference...")
    with aule.Attention(backend='cpu') as attn:
        reference = attn.forward(Q, K, V, causal=True)

    # Test each available backend (that actually initializes)
    available = aule.get_available_backends()
    results = {}
    tested_any = False

    for backend in available:
        if backend == 'cpu':
            continue

        try:
            with aule.Attention(backend=backend) as attn:
                output = attn.forward(Q, K, V, causal=True)
                max_diff = np.abs(output - reference).max()
                results[backend] = max_diff
                tested_any = True

                status = "PASS" if max_diff < 1e-3 else "FAIL"
                print(f"  {backend}: max_diff={max_diff:.6f} [{status}]")
        except Exception as e:
            # Skip backends that fail to initialize (missing dependencies)
            err_msg = str(e)
            if 'not available' in err_msg.lower():
                print(f"  {backend}: SKIPPED (not installed)")
            else:
                print(f"  {backend}: ERROR - {e}")
            results[backend] = float('inf')

    if not tested_any:
        print("\n  [INFO] No GPU backends available for testing")
        print("         Install pyopencl or ROCm for GPU acceleration")

    return results


def benchmark_all_backends():
    """Benchmark all available backends."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    import aule_unified as aule

    configs = [
        {"name": "Small", "batch": 1, "heads": 8, "seq": 512, "dim": 64},
        {"name": "LLaMA-7B", "batch": 1, "heads": 32, "seq": 2048, "dim": 128},
    ]

    available = aule.get_available_backends()
    results = {}

    for cfg in configs:
        print(f"\n--- {cfg['name']} [{cfg['batch']}, {cfg['heads']}, {cfg['seq']}, {cfg['dim']}] ---")

        # Use FP16 for ROCm backends (much faster)
        Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float16)
        K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float16)
        V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float16)

        for backend in available:
            try:
                # Use FP16 dtype for ROCm, FP32 for CPU
                dtype = 'float16' if backend in ('pytorch_rocm', 'rocm_flash', 'ck') else 'float32'

                with aule.Attention(backend=backend) as attn:
                    # Warmup
                    _ = attn.forward(Q, K, V, causal=True, dtype=dtype)

                    # Benchmark
                    n_iters = 5 if backend == 'cpu' else 20
                    times = []
                    for _ in range(n_iters):
                        start = time.perf_counter()
                        _ = attn.forward(Q, K, V, causal=True, dtype=dtype)
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)

                    avg_ms = np.mean(times) * 1000
                    flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
                    tflops = flops / np.mean(times) / 1e12

                    print(f"  {backend:20s}: {avg_ms:8.2f}ms, {tflops:.3f} TFLOPS ({dtype})")

                    results[(cfg['name'], backend)] = tflops

            except Exception as e:
                print(f"  {backend:20s}: ERROR - {e}")

    return results


def test_auto_backend():
    """Test that auto backend selection works correctly."""
    print("\n" + "=" * 70)
    print("AUTO BACKEND SELECTION TEST")
    print("=" * 70)

    import aule_unified as aule

    # Test auto selection
    with aule.Attention(backend='auto') as attn:
        print(f"\nAuto-selected backend: {attn.backend_name}")

        # Quick test
        Q = np.random.randn(1, 4, 64, 64).astype(np.float32)
        K = np.random.randn(1, 4, 64, 64).astype(np.float32)
        V = np.random.randn(1, 4, 64, 64).astype(np.float32)

        output = attn.forward(Q, K, V, causal=True)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")


def test_dual_strategy():
    """Test the dual-strategy approach specifically."""
    print("\n" + "=" * 70)
    print("DUAL-STRATEGY TEST")
    print("=" * 70)

    import aule_unified as aule

    available = aule.get_available_backends()

    # Strategy 1: ROCm path (CK or HIP)
    rocm_backends = ['ck', 'rocm_flash', 'pytorch_rocm', 'hip_mfma', 'hip']
    rocm_available = [b for b in rocm_backends if b in available]

    # Strategy 2: Portable path (OpenCL/Vulkan)
    portable_backends = ['opencl_optimized', 'opencl', 'vulkan']
    portable_available = [b for b in portable_backends if b in available]

    print("\nStrategy Analysis:")
    print("-" * 40)

    if rocm_available:
        print(f"ROCm path: AVAILABLE")
        print(f"  Best ROCm backend: {rocm_available[0]}")
        print(f"  All ROCm backends: {', '.join(rocm_available)}")
    else:
        print(f"ROCm path: NOT AVAILABLE")

    if portable_available:
        print(f"\nPortable path: AVAILABLE")
        print(f"  Best portable backend: {portable_available[0]}")
        print(f"  All portable backends: {', '.join(portable_available)}")
    else:
        print(f"\nPortable path: NOT AVAILABLE")

    # Recommendation
    print("\n" + "-" * 40)
    print("Recommendation:")

    if rocm_available:
        best = rocm_available[0]
        if best == 'ck' or best == 'rocm_flash':
            print(f"  Use '{best}' for peak performance (~150-300 TFLOPS)")
        elif best == 'hip_mfma':
            print(f"  Use '{best}' for high performance (~50-100 TFLOPS)")
        else:
            print(f"  Use '{best}' for moderate performance")
    elif portable_available:
        best = portable_available[0]
        print(f"  Use '{best}' for portable GPU acceleration (~0.5 TFLOPS)")
        print("  Note: Install ROCm + flash-attn for ~100x better performance")
    else:
        print("  CPU fallback only. Install GPU drivers for acceleration.")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AULE-ATTENTION DUAL-STRATEGY TEST SUITE")
    print("=" * 70)
    print(f"\nThis script validates the dual-strategy approach:")
    print("  - WITH ROCm: CK FlashAttention (~150-300 TFLOPS)")
    print("  - WITHOUT ROCm: Optimized OpenCL (~0.5 TFLOPS)")

    # Run tests
    test_backend_detection()
    test_auto_backend()
    test_dual_strategy()
    test_numerical_correctness()

    # Only run benchmark if explicitly requested (takes time)
    if '--benchmark' in sys.argv:
        benchmark_all_backends()
    else:
        print("\n[INFO] Run with --benchmark to include performance tests")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
