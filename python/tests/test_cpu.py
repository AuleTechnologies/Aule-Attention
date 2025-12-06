"""Tests for CPU fallback backend."""

import pytest
import numpy as np


class TestCPUBackend:
    """Test CPU (NumPy) fallback attention."""

    def test_import(self):
        """Test basic import works."""
        from aule import flash_attention, get_available_backends
        assert 'cpu' in get_available_backends()

    def test_forward_basic(self, random_qkv_numpy, reference_attention):
        """Test basic forward pass."""
        from aule import flash_attention

        q, k, v = random_qkv_numpy(batch=1, heads=4, seq_len=32, head_dim=64)
        out = flash_attention(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)

        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_forward_non_causal(self, random_qkv_numpy, reference_attention):
        """Test non-causal attention."""
        from aule import flash_attention

        q, k, v = random_qkv_numpy(batch=1, heads=4, seq_len=32, head_dim=64)
        out = flash_attention(q, k, v, causal=False)
        ref = reference_attention(q, k, v, causal=False)

        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_batch_size(self, random_qkv_numpy, reference_attention):
        """Test with larger batch size."""
        from aule import flash_attention

        q, k, v = random_qkv_numpy(batch=4, heads=8, seq_len=64, head_dim=64)
        out = flash_attention(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)

        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_different_head_dims(self, reference_attention):
        """Test various head dimensions."""
        from aule import flash_attention

        for head_dim in [32, 64, 128]:
            np.random.seed(42)
            q = np.random.randn(1, 4, 32, head_dim).astype(np.float32)
            k = np.random.randn(1, 4, 32, head_dim).astype(np.float32)
            v = np.random.randn(1, 4, 32, head_dim).astype(np.float32)

            out = flash_attention(q, k, v, causal=True)
            ref = reference_attention(q, k, v, causal=True)

            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
