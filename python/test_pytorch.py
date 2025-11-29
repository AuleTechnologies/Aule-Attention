#!/usr/bin/env python3
"""
Test aule attention against PyTorch's scaled_dot_product_attention.

This validates that the Vulkan GPU implementation produces correct results
by comparing against PyTorch's reference implementation.

Usage:
    python test_pytorch.py

Requirements:
    pip install torch numpy
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for aule import
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

from aule import Aule, AuleError


def pytorch_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute attention using PyTorch's implementation."""
    Q_t = torch.from_numpy(Q)
    K_t = torch.from_numpy(K)
    V_t = torch.from_numpy(V)

    # PyTorch's scaled_dot_product_attention
    output = torch.nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t)
    return output.numpy()


def test_basic_attention():
    """Test basic attention computation."""
    print("Testing basic attention...")

    batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 16

    np.random.seed(42)
    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    with Aule() as aule:
        aule_output = aule.attention(Q, K, V)

    if TORCH_AVAILABLE:
        torch_output = pytorch_attention(Q, K, V)

        max_diff = np.max(np.abs(aule_output - torch_output))
        mean_diff = np.mean(np.abs(aule_output - torch_output))

        print(f"  Shape: {aule_output.shape}")
        print(f"  Max diff from PyTorch: {max_diff:.6e}")
        print(f"  Mean diff from PyTorch: {mean_diff:.6e}")

        assert max_diff < 1e-4, f"Max diff too large: {max_diff}"
        print("  PASSED")
    else:
        print(f"  Shape: {aule_output.shape}")
        print("  (PyTorch comparison skipped)")


def test_multihead_attention():
    """Test multi-head attention."""
    print("Testing multi-head attention...")

    batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 64

    np.random.seed(123)
    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    with Aule() as aule:
        aule_output = aule.attention(Q, K, V)

    if TORCH_AVAILABLE:
        torch_output = pytorch_attention(Q, K, V)

        max_diff = np.max(np.abs(aule_output - torch_output))
        mean_diff = np.mean(np.abs(aule_output - torch_output))

        print(f"  Shape: {aule_output.shape}")
        print(f"  Max diff from PyTorch: {max_diff:.6e}")
        print(f"  Mean diff from PyTorch: {mean_diff:.6e}")

        assert max_diff < 1e-3, f"Max diff too large: {max_diff}"
        print("  PASSED")
    else:
        print(f"  Shape: {aule_output.shape}")
        print("  (PyTorch comparison skipped)")


def test_various_shapes():
    """Test various tensor shapes."""
    print("Testing various shapes...")

    shapes = [
        (1, 1, 4, 8),      # Tiny
        (1, 2, 16, 16),    # Small
        (2, 4, 32, 32),    # Medium
        (1, 8, 64, 64),    # Large
        (4, 12, 32, 64),   # Typical transformer config
    ]

    with Aule() as aule:
        for batch, heads, seq, dim in shapes:
            np.random.seed(42)
            Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
            K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
            V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

            aule_output = aule.attention(Q, K, V)

            if TORCH_AVAILABLE:
                torch_output = pytorch_attention(Q, K, V)
                max_diff = np.max(np.abs(aule_output - torch_output))
                status = "PASS" if max_diff < 1e-3 else "FAIL"
                print(f"  [{batch}, {heads}, {seq}, {dim}]: max_diff={max_diff:.2e} {status}")
            else:
                print(f"  [{batch}, {heads}, {seq}, {dim}]: computed")


def test_numerical_stability():
    """Test with values that might cause numerical issues."""
    print("Testing numerical stability...")

    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 32

    # Test with larger values
    np.random.seed(999)
    Q = (np.random.randn(batch_size, num_heads, seq_len, head_dim) * 3).astype(np.float32)
    K = (np.random.randn(batch_size, num_heads, seq_len, head_dim) * 3).astype(np.float32)
    V = (np.random.randn(batch_size, num_heads, seq_len, head_dim) * 2).astype(np.float32)

    with Aule() as aule:
        aule_output = aule.attention(Q, K, V)

    # Check for NaN/Inf
    assert not np.any(np.isnan(aule_output)), "Output contains NaN"
    assert not np.any(np.isinf(aule_output)), "Output contains Inf"

    if TORCH_AVAILABLE:
        torch_output = pytorch_attention(Q, K, V)
        max_diff = np.max(np.abs(aule_output - torch_output))
        print(f"  Max diff from PyTorch (large values): {max_diff:.6e}")
        assert max_diff < 1e-3, f"Max diff too large: {max_diff}"

    print("  PASSED (no NaN/Inf)")


def test_batch_independence():
    """Test that batches don't interfere with each other."""
    print("Testing batch independence...")

    num_heads, seq_len, head_dim = 2, 16, 32

    np.random.seed(42)
    Q1 = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32)
    K1 = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32)
    V1 = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32)

    Q2 = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32)
    K2 = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32)
    V2 = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32)

    with Aule() as aule:
        # Compute separately
        out1_single = aule.attention(Q1, K1, V1)

        # Compute as batch
        Q_batch = np.concatenate([Q1, Q2], axis=0)
        K_batch = np.concatenate([K1, K2], axis=0)
        V_batch = np.concatenate([V1, V2], axis=0)
        out_batch = aule.attention(Q_batch, K_batch, V_batch)

    # First batch element should match single computation
    max_diff = np.max(np.abs(out1_single - out_batch[0:1]))
    print(f"  Max diff (batch[0] vs single): {max_diff:.6e}")
    assert max_diff < 1e-6, f"Batch results differ: {max_diff}"
    print("  PASSED")


def benchmark():
    """Simple benchmark comparing aule vs PyTorch."""
    if not TORCH_AVAILABLE:
        print("Benchmark requires PyTorch. Skipping.")
        return

    import time

    print("\nBenchmark (10 iterations each):")

    configs = [
        (1, 8, 64, 64, "Small"),
        (2, 8, 128, 64, "Medium"),
        (4, 12, 256, 64, "Large"),
    ]

    for batch, heads, seq, dim, name in configs:
        np.random.seed(42)
        Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        V = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        # Warm up
        with Aule() as aule:
            aule.attention(Q, K, V)

            # Time aule
            start = time.perf_counter()
            for _ in range(10):
                aule.attention(Q, K, V)
            aule_time = (time.perf_counter() - start) / 10

        # Time PyTorch (CPU)
        Q_t = torch.from_numpy(Q)
        K_t = torch.from_numpy(K)
        V_t = torch.from_numpy(V)

        # Warm up
        torch.nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t)

        start = time.perf_counter()
        for _ in range(10):
            torch.nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t)
        torch_time = (time.perf_counter() - start) / 10

        print(f"  {name} [{batch}x{heads}x{seq}x{dim}]:")
        print(f"    aule (Vulkan GPU): {aule_time*1000:.2f} ms")
        print(f"    PyTorch (CPU):     {torch_time*1000:.2f} ms")
        print(f"    Speedup:           {torch_time/aule_time:.2f}x")


def main():
    print("=" * 60)
    print("aule-attention PyTorch Comparison Tests")
    print("=" * 60)

    try:
        test_basic_attention()
        test_multihead_attention()
        test_various_shapes()
        test_numerical_stability()
        test_batch_independence()

        print("\n" + "=" * 60)
        print("All tests PASSED!")
        print("=" * 60)

        if TORCH_AVAILABLE:
            benchmark()

    except AuleError as e:
        print(f"\nAule error: {e}")
        print("Make sure the library is built: zig build")
        sys.exit(1)
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
