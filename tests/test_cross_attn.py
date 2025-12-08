import unittest
import numpy as np
import torch
from aule.vulkan import Aule, AuleError

class TestCrossAttention(unittest.TestCase):
    def test_cross_attention(self):
        """Test Cross-Attention (Query SeqLen != Key/Value SeqLen)"""
        print("\nTest Cross-Attention implementation.")
        
        # Dimensions
        B = 1
        H = 4      # Num heads
        Sq = 16    # Query sequence length
        Skv = 32   # Key/Value sequence length (longer context)
        D = 32     # Head dim
        
        # Initialize
        ctx = Aule()
        
        # Create Inputs
        np.random.seed(42)
        q_np = np.random.randn(B, H, Sq, D).astype(np.float32)
        k_np = np.random.randn(B, H, Skv, D).astype(np.float32)
        v_np = np.random.randn(B, H, Skv, D).astype(np.float32)
        
        # Create GPU Tensors
        q_gpu = ctx.tensor((B, H, Sq, D))
        k_gpu = ctx.tensor((B, H, Skv, D))
        v_gpu = ctx.tensor((B, H, Skv, D))
        out_gpu = ctx.tensor((B, H, Sq, D)) # Output matches Query shape
        
        # Upload
        q_gpu.upload(q_np)
        k_gpu.upload(k_np)
        v_gpu.upload(v_np)
        
        # Dispatch
        try:
            ctx.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=False)
        except AuleError as e:
            self.fail(f"GPU attention failed: {e}")
            
        # Download
        out_vk = out_gpu.download()
        
        # Reference PyTorch implementation
        q_pt = torch.tensor(q_np)
        k_pt = torch.tensor(k_np)
        v_pt = torch.tensor(v_np)
        
        # Scaled Dot Product Attention
        # PyTorch handles cross-attention naturally
        scale = 1.0 / np.sqrt(D)
        attn = (q_pt @ k_pt.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out_ref = attn @ v_pt
        
        # Compare
        np.testing.assert_allclose(out_vk, out_ref.numpy(), atol=1e-3, rtol=1e-3)
        print("Cross-Attention verification passed!")

if __name__ == '__main__':
    unittest.main()
