"""Tests for Vulkan backend."""

import pytest
import numpy as np


def vulkan_available():
    """Check if Vulkan backend is available."""
    try:
        from aule import get_available_backends
        return 'vulkan' in get_available_backends()
    except:
        return False


@pytest.mark.vulkan
@pytest.mark.skipif(not vulkan_available(), reason="Vulkan not available")
class TestVulkanBackend:
    """Test Vulkan compute shader attention."""

    def test_import(self):
        """Test Vulkan module imports."""
        from aule.vulkan import Aule, attention

    def test_forward_first(self, random_qkv_numpy, reference_attention):
        """Initialize the global singleton with a forward pass first.

        This test must run first to avoid a pytest-specific segfault issue
        that occurs when test_device_info runs before forward tests.
        """
        from aule.vulkan import attention

        q, k, v = random_qkv_numpy(batch=1, heads=4, seq_len=32, head_dim=64)
        out = attention(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)

        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_device_info(self):
        """Test device info retrieval.

        Note: This test must run after test_forward_first to avoid a segfault.
        """
        from aule.vulkan import _AULE_INSTANCE_SINGLETON

        # Singleton should already be initialized from test_forward_first
        assert _AULE_INSTANCE_SINGLETON is not None
        info = _AULE_INSTANCE_SINGLETON.get_device_info()
        assert 'device_name' in info
        print(f"Vulkan device: {info.get('device_name', 'Unknown')}")

    def test_forward_basic(self, random_qkv_numpy, reference_attention):
        """Test basic forward pass."""
        from aule.vulkan import attention

        q, k, v = random_qkv_numpy(batch=1, heads=4, seq_len=32, head_dim=64)
        out = attention(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)

        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_forward_non_causal(self, random_qkv_numpy, reference_attention):
        """Test non-causal attention."""
        from aule.vulkan import attention

        q, k, v = random_qkv_numpy(batch=1, heads=4, seq_len=32, head_dim=64)
        out = attention(q, k, v, causal=False)
        ref = reference_attention(q, k, v, causal=False)

        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_batch_size(self, random_qkv_numpy, reference_attention):
        """Test with larger batch size."""
        from aule.vulkan import attention

        q, k, v = random_qkv_numpy(batch=2, heads=8, seq_len=64, head_dim=64)
        out = attention(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)

        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_different_head_dims(self, reference_attention):
        """Test various head dimensions."""
        from aule.vulkan import attention

        for head_dim in [32, 64]:
            np.random.seed(42)
            q = np.random.randn(1, 4, 32, head_dim).astype(np.float32)
            k = np.random.randn(1, 4, 32, head_dim).astype(np.float32)
            v = np.random.randn(1, 4, 32, head_dim).astype(np.float32)

            out = attention(q, k, v, causal=True)
            ref = reference_attention(q, k, v, causal=True)

            np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_unified_api(self, random_qkv_numpy, reference_attention):
        """Test through unified flash_attention API."""
        from aule import flash_attention

        q, k, v = random_qkv_numpy(batch=1, heads=4, seq_len=32, head_dim=64)
        out = flash_attention(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)

        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
