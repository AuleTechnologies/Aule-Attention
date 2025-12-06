"""Tests for Triton backend."""

import pytest


def triton_available():
    """Check if Triton backend is available."""
    try:
        from aule import get_available_backends
        return 'triton' in get_available_backends()
    except:
        return False


def cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="CUDA/ROCm not available")
@pytest.mark.skipif(not triton_available(), reason="Triton not available")
class TestTritonBackend:
    """Test Triton FlashAttention-2 kernel."""

    def test_import(self):
        """Test Triton module imports."""
        from aule.triton_flash import flash_attention_triton, is_triton_available
        assert is_triton_available()

    def test_forward_basic(self, random_qkv_torch):
        """Test basic forward pass."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=64, device='cuda')
        out = flash_attention_triton(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_forward_non_causal(self, random_qkv_torch):
        """Test non-causal attention."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=64, device='cuda')
        out = flash_attention_triton(q, k, v, causal=False)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_backward_basic(self, random_qkv_torch):
        """Test backward pass computes gradients."""
        import torch
        from aule.triton_flash import flash_attention_triton

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=64, device='cuda', requires_grad=True)
        out = flash_attention_triton(q, k, v, causal=True)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_backward_accuracy(self, random_qkv_torch):
        """Test backward pass matches PyTorch reference."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=64, device='cuda', requires_grad=True)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        out = flash_attention_triton(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)

        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        ref.backward(grad_out)

        torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-2, atol=1e-2)

    def test_gqa(self, random_qkv_torch):
        """Test Grouped Query Attention (GQA)."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        torch.manual_seed(42)
        q = torch.randn(1, 12, 64, 64, device='cuda')  # 12 query heads
        k = torch.randn(1, 2, 64, 64, device='cuda')   # 2 KV heads
        v = torch.randn(1, 2, 64, 64, device='cuda')

        out = flash_attention_triton(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        assert out.shape == (1, 12, 64, 64)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_mqa(self, random_qkv_torch):
        """Test Multi-Query Attention (MQA)."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        torch.manual_seed(42)
        q = torch.randn(1, 8, 64, 64, device='cuda')  # 8 query heads
        k = torch.randn(1, 1, 64, 64, device='cuda')  # 1 KV head
        v = torch.randn(1, 1, 64, 64, device='cuda')

        out = flash_attention_triton(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        assert out.shape == (1, 8, 64, 64)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_head_dim_128(self, random_qkv_torch):
        """Test with head_dim=128 (common in modern LLMs)."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=128, device='cuda')
        out = flash_attention_triton(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_fp16(self, random_qkv_torch):
        """Test fp16 precision."""
        import torch
        import torch.nn.functional as F
        from aule.triton_flash import flash_attention_triton

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=64, device='cuda', dtype=torch.float16)
        out = flash_attention_triton(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    def test_unified_api(self, random_qkv_torch):
        """Test through unified flash_attention API."""
        import torch
        import torch.nn.functional as F
        from aule import flash_attention

        q, k, v = random_qkv_torch(batch=1, heads=8, seq_len=64, head_dim=64, device='cuda')
        out = flash_attention(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
