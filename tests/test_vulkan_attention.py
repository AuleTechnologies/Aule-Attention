#!/usr/bin/env python3
"""
Comprehensive test suite for aule-attention Vulkan backend.
Run this on any GPU to verify the library works correctly.

Usage:
    python3 tests/test_vulkan_attention.py

Expected output shows GPU info and test results.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import importlib.util
spec = importlib.util.spec_from_file_location(
    'aule_vulkan',
    str(Path(__file__).parent.parent / "python" / "aule.py")
)
aule_vulkan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aule_vulkan)


def reference_attention(Q, K, V, causal=False):
    """NumPy reference implementation for validation."""
    batch, heads, seq, dim = Q.shape
    scale = 1.0 / np.sqrt(dim)

    # Q @ K^T
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale

    # Causal mask
    if causal:
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # @ V
    output = np.einsum('bhqk,bhkd->bhqd', attention, V)
    return output.astype(np.float32)


def test_device_info():
    """Test GPU device detection."""
    print("\n" + "="*60)
    print("TEST: Device Info")
    print("="*60)

    with aule_vulkan.Aule() as aule:
        info = aule.get_device_info()

        print(f"  Device:        {info['device_name']}")
        print(f"  Vendor:        {info['vendor']}")
        print(f"  AMD Optimized: {info['amd_optimized']}")
        print(f"  FP16 Support:  {info['fp16_supported']}")
        print(f"  Subgroup Size: {info['subgroup_size']}")

        # Validate expected values based on vendor
        if info['vendor'] == 'amd':
            assert info['subgroup_size'] == 64, "AMD should have 64-wide wavefront"
            assert info['amd_optimized'] == True, "AMD should use optimized shader"
        elif info['vendor'] == 'nvidia':
            assert info['subgroup_size'] == 32, "NVIDIA should have 32-wide warp"
            assert info['amd_optimized'] == False
        elif info['vendor'] == 'intel':
            assert info['subgroup_size'] in [8, 16, 32], "Intel EU width varies"
            assert info['amd_optimized'] == False

        print("  ✓ Device info test PASSED")
        return info


def test_basic_attention():
    """Test basic attention computation."""
    print("\n" + "="*60)
    print("TEST: Basic Attention")
    print("="*60)

    batch, heads, seq, dim = 1, 2, 16, 32

    np.random.seed(42)
    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

    with aule_vulkan.Aule() as aule:
        output = aule.attention(Q, K, V, causal=False)

    # Compare with reference
    ref_output = reference_attention(Q, K, V, causal=False)

    max_diff = np.abs(output - ref_output).max()
    mean_diff = np.abs(output - ref_output).mean()

    print(f"  Shape: {output.shape}")
    print(f"  Max diff from reference: {max_diff:.6f}")
    print(f"  Mean diff from reference: {mean_diff:.6f}")

    assert max_diff < 1e-2, f"Max diff too high: {max_diff}"
    print("  ✓ Basic attention test PASSED")


def test_causal_attention():
    """Test causal masking."""
    print("\n" + "="*60)
    print("TEST: Causal Attention (LLM Mode)")
    print("="*60)

    batch, heads, seq, dim = 1, 2, 32, 32

    np.random.seed(42)
    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

    with aule_vulkan.Aule() as aule:
        out_causal = aule.attention(Q, K, V, causal=True)
        out_bidirectional = aule.attention(Q, K, V, causal=False)

    # Compare with reference
    ref_causal = reference_attention(Q, K, V, causal=True)
    ref_bidirectional = reference_attention(Q, K, V, causal=False)

    diff_causal = np.abs(out_causal - ref_causal).max()
    diff_bidirectional = np.abs(out_bidirectional - ref_bidirectional).max()

    # Causal and bidirectional should be different
    diff_modes = np.abs(out_causal - out_bidirectional).max()

    print(f"  Causal max diff from reference: {diff_causal:.6f}")
    print(f"  Bidirectional max diff from reference: {diff_bidirectional:.6f}")
    print(f"  Diff between modes: {diff_modes:.4f}")

    assert diff_causal < 1e-2, f"Causal diff too high: {diff_causal}"
    assert diff_bidirectional < 1e-2, f"Bidirectional diff too high: {diff_bidirectional}"
    assert diff_modes > 0.1, "Causal and bidirectional should differ significantly"

    print("  ✓ Causal attention test PASSED")


def test_gpu_tensor_api():
    """Test zero-copy GPU tensor path."""
    print("\n" + "="*60)
    print("TEST: GPU Tensor API (Zero-Copy)")
    print("="*60)

    batch, heads, seq, dim = 1, 4, 64, 64

    np.random.seed(42)
    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

    with aule_vulkan.Aule() as aule:
        # Create GPU tensors
        q_gpu = aule.tensor((batch, heads, seq, dim))
        k_gpu = aule.tensor((batch, heads, seq, dim))
        v_gpu = aule.tensor((batch, heads, seq, dim))
        out_gpu = aule.tensor((batch, heads, seq, dim))

        # Upload
        q_gpu.upload(Q)
        k_gpu.upload(K)
        v_gpu.upload(V)

        # Compute on GPU
        aule.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=True)

        # Download
        output = out_gpu.download()

    # Compare with reference
    ref_output = reference_attention(Q, K, V, causal=True)
    max_diff = np.abs(output - ref_output).max()

    print(f"  Shape: {output.shape}")
    print(f"  Max diff from reference: {max_diff:.6f}")

    assert max_diff < 1e-2, f"Max diff too high: {max_diff}"
    print("  ✓ GPU tensor API test PASSED")


def test_performance():
    """Benchmark attention performance."""
    print("\n" + "="*60)
    print("TEST: Performance Benchmark")
    print("="*60)

    configs = [
        (1, 4, 64, 64),
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
    ]

    with aule_vulkan.Aule() as aule:
        print(f"  Device: {aule.device_name}")
        print(f"  Shader: {'AMD-optimized' if aule.is_amd_optimized else 'Generic'}")
        print()

        for batch, heads, seq, dim in configs:
            Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
            K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
            V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

            # GPU tensor path
            q_gpu = aule.tensor((batch, heads, seq, dim))
            k_gpu = aule.tensor((batch, heads, seq, dim))
            v_gpu = aule.tensor((batch, heads, seq, dim))
            out_gpu = aule.tensor((batch, heads, seq, dim))

            q_gpu.upload(Q)
            k_gpu.upload(K)
            v_gpu.upload(V)

            # Warmup
            for _ in range(5):
                aule.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=True)

            # Benchmark
            num_iters = 50
            start = time.perf_counter()
            for _ in range(num_iters):
                aule.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=True)
            elapsed = time.perf_counter() - start

            # Calculate TFLOPS
            flops = 4 * batch * heads * seq * seq * dim * num_iters
            tflops = flops / elapsed / 1e12
            ms_per_iter = elapsed * 1000 / num_iters

            print(f"  B={batch} H={heads} S={seq:4d} D={dim}: "
                  f"{ms_per_iter:6.2f} ms/iter, ~{tflops:.4f} TFLOPS")

    print("  ✓ Performance benchmark completed")


def test_flash_attention_convenience():
    """Test convenience function."""
    print("\n" + "="*60)
    print("TEST: flash_attention() Convenience Function")
    print("="*60)

    batch, heads, seq, dim = 1, 2, 16, 32

    np.random.seed(42)
    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

    # flash_attention defaults to causal=True
    output = aule_vulkan.flash_attention(Q, K, V)
    ref_output = reference_attention(Q, K, V, causal=True)

    max_diff = np.abs(output - ref_output).max()
    print(f"  Max diff from reference: {max_diff:.6f}")

    assert max_diff < 1e-2, f"Max diff too high: {max_diff}"
    print("  ✓ flash_attention() test PASSED")


def main():
    print("="*60)
    print("AULE-ATTENTION VULKAN TEST SUITE")
    print("="*60)

    try:
        info = test_device_info()
        test_basic_attention()
        test_causal_attention()
        test_gpu_tensor_api()
        test_flash_attention_convenience()
        test_performance()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  GPU: {info['device_name']}")
        print(f"  Vendor: {info['vendor']}")
        print(f"  Optimized Path: {'AMD' if info['amd_optimized'] else 'Generic'}")
        print(f"  FP16: {'Supported' if info['fp16_supported'] else 'Not supported'}")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
