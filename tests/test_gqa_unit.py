
import unittest
import numpy as np
import torch
import aule.vulkan as vk

class TestGQA(unittest.TestCase):
    def test_gqa(self):
        """Test Grouped Query Attention (GQA/MQA) implementation."""
        batch_size = 1
        num_heads_q = 4
        num_heads_kv = 1 # MQA case
        seq_len = 16
        head_dim = 64
        
        # Init Aule
        ctx = vk.Aule()
        
        # Create tensors
        np.random.seed(42)
        q_np = np.random.randn(batch_size, num_heads_q, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch_size, num_heads_kv, seq_len, head_dim).astype(np.float32)
        v_np = np.random.randn(batch_size, num_heads_kv, seq_len, head_dim).astype(np.float32)
        
        q_gpu = ctx.tensor(q_np.shape)
        k_gpu = ctx.tensor(k_np.shape)
        v_gpu = ctx.tensor(v_np.shape)
        out_gpu = ctx.tensor(q_np.shape)
        
        q_gpu.upload(q_np)
        k_gpu.upload(k_np)
        v_gpu.upload(v_np)
        
        # Run Kernel with GQA
        # Backend should detect mismatched heads and pass num_kv_heads=1
        ctx.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=False)
        out_vk = out_gpu.download()
        
        # Reference PyTorch implementation using repeat_interleave
        q_pt = torch.tensor(q_np)
        k_pt = torch.tensor(k_np)
        v_pt = torch.tensor(v_np)
        
        # Manually expand K/V to match Q heads for reference calculation
        # MQA: [B, 1, S, D] -> [B, 4, S, D]
        k_pt_Expanded = k_pt.repeat_interleave(num_heads_q // num_heads_kv, dim=1)
        v_pt_Expanded = v_pt.repeat_interleave(num_heads_q // num_heads_kv, dim=1)
        
        scale = 1.0 / np.sqrt(head_dim)
        attn = (q_pt @ k_pt_Expanded.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out_ref = attn @ v_pt_Expanded
        
        # Compare
        np.testing.assert_allclose(out_vk, out_ref.numpy(), atol=1e-3, rtol=1e-3)
        print("GQA verification passed!")
        
        ctx.close()

if __name__ == '__main__':
    unittest.main()
