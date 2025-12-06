"""Pytest configuration and fixtures for aule-attention tests."""

import pytest
import numpy as np


@pytest.fixture
def random_qkv_numpy():
    """Generate random Q, K, V tensors as NumPy arrays."""
    def _make(batch=1, heads=8, seq_len=64, head_dim=64, dtype=np.float32):
        np.random.seed(42)
        q = np.random.randn(batch, heads, seq_len, head_dim).astype(dtype)
        k = np.random.randn(batch, heads, seq_len, head_dim).astype(dtype)
        v = np.random.randn(batch, heads, seq_len, head_dim).astype(dtype)
        return q, k, v
    return _make


@pytest.fixture
def random_qkv_torch():
    """Generate random Q, K, V tensors as PyTorch tensors."""
    def _make(batch=1, heads=8, seq_len=64, head_dim=64, device='cpu', dtype=None, requires_grad=False):
        import torch
        if dtype is None:
            dtype = torch.float32
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=requires_grad)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=requires_grad)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=requires_grad)
        return q, k, v
    return _make


@pytest.fixture
def reference_attention():
    """Reference attention implementation for verification."""
    def _compute(q, k, v, causal=True):
        import math
        # Works with both numpy and torch
        if hasattr(q, 'numpy'):
            # PyTorch tensor
            import torch
            import torch.nn.functional as F
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            # NumPy array
            batch, heads, seq_q, head_dim = q.shape
            _, _, seq_k, _ = k.shape
            scale = 1.0 / math.sqrt(head_dim)
            scores = np.einsum('bhqd,bhkd->bhqk', q, k) * scale
            if causal:
                mask = np.triu(np.ones((seq_q, seq_k)), k=1).astype(bool)
                scores = np.where(mask, -1e9, scores)
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            return np.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    return _compute


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: tests requiring CUDA/ROCm GPU")
    config.addinivalue_line("markers", "vulkan: tests requiring Vulkan GPU")
    config.addinivalue_line("markers", "slow: slow tests")
