
import unittest
import numpy as np
import torch
import aule.vulkan as vk

class TestRoPE(unittest.TestCase):
    def test_rope_gpu(self):
        """Test Fused RoPE implementation in Vulkan backend against PyTorch reference."""
        batch_size = 1
        num_heads = 1
        seq_len = 8
        head_dim = 64
        
        # Init Aule
        ctx = vk.Aule()
        
        # Create tensors
        np.random.seed(42)
        q_np = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        v_np = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        
        dim_half = head_dim // 2
        freqs = np.arange(0, dim_half, dtype=np.float32)
        t = np.arange(seq_len, dtype=np.float32)
        freqs = 1.0 / (10000 ** (freqs / dim_half))
        emb = np.outer(t, freqs) # [seq, dim/2]
        
        cos_np = np.cos(emb).astype(np.float32)
        sin_np = np.sin(emb).astype(np.float32)
        
        # DEBUG: Force Identity Rotation
        # cos_np = np.ones((1, 1, seq_len, dim_half), dtype=np.float32)
        # sin_np = np.zeros((1, 1, seq_len, dim_half), dtype=np.float32)
        
        # Debug prints
        print(f"Q[0,0,0,0]: {q_np[0,0,0,0]}")
        
        q_gpu = ctx.tensor(q_np.shape)
        k_gpu = ctx.tensor(k_np.shape)
        v_gpu = ctx.tensor(v_np.shape)
        out_gpu = ctx.tensor(q_np.shape)
        
        cos_gpu = ctx.tensor((1, 1, seq_len, dim_half))
        sin_gpu = ctx.tensor((1, 1, seq_len, dim_half))
        
        q_gpu.upload(q_np)
        k_gpu.upload(k_np)
        v_gpu.upload(v_np)
        cos_gpu.upload(cos_np)
        cos_back = cos_gpu.download()
        print(f"Cos GPU readback[0,0,0,0]: {cos_back[0,0,0,0]}")
        
        sin_gpu.upload(sin_np)
        
        # Run Kernel WITH RoPE
        ctx.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, rot_cos=cos_gpu, rot_sin=sin_gpu, causal=False)
        out_rope = out_gpu.download()
        
        # Run Kernel WITHOUT RoPE (to verify toggling)
        out_no_rope_gpu = ctx.tensor(q_np.shape)
        ctx.attention_gpu(q_gpu, k_gpu, v_gpu, out_no_rope_gpu, rot_cos=None, rot_sin=None, causal=False)
        out_no_rope = out_no_rope_gpu.download()
        
        print(f"Out RoPE[0,0,0,0]: {out_rope[0,0,0,0]}")
        print(f"Out NoRoPE[0,0,0,0]: {out_no_rope[0,0,0,0]}")
        
        # Reference PyTorch implementation
        q_pt = torch.tensor(q_np)
        k_pt = torch.tensor(k_np)
        v_pt = torch.tensor(v_np)
        cos_pt = torch.tensor(cos_np)
        sin_pt = torch.tensor(sin_np)
        
        def apply_rope(x, c, s):
            x_reshaped = x.view(batch_size, num_heads, seq_len, dim_half, 2)
            x1 = x_reshaped[..., 0]
            x2 = x_reshaped[..., 1]
            c = c.view(1, 1, seq_len, dim_half)
            s = s.view(1, 1, seq_len, dim_half)
            x1_rot = x1 * c - x2 * s
            x2_rot = x1 * s + x2 * c
            x_rot = torch.stack([x1_rot, x2_rot], dim=-1).view_as(x)
            return x_rot

        q_rot = apply_rope(q_pt, cos_pt, sin_pt)
        k_rot = apply_rope(k_pt, cos_pt, sin_pt)
        
        # Manual output check
        print(f"Ref Q_rot[0,0,0,0]: {q_rot[0,0,0,0]}")
        
        scale = 1.0 / np.sqrt(head_dim)
        attn = (q_rot @ k_rot.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out_ref = attn @ v_pt
        
        print(f"Ref Out[0,0,0,0]: {out_ref[0,0,0,0]}")
        
        # Compare No Rope
        attn_base = (q_pt @ k_pt.transpose(-2, -1)) * scale
        attn_base = torch.softmax(attn_base, dim=-1)
        out_ref_base = attn_base @ v_pt
        print(f"Ref Out NoRoPE[0,0,0,0]: {out_ref_base[0,0,0,0]}")
        
        # Assertions
        np.testing.assert_allclose(out_no_rope, out_ref_base.numpy(), atol=1e-3, rtol=1e-3, err_msg="Base Attention Failed")
        print("Base attention passed.")
        
        np.testing.assert_allclose(out_rope, out_ref.numpy(), atol=1e-3, rtol=1e-3, err_msg="RoPE Attention Failed")
        print("RoPE verification passed!")
        
        ctx.close()

if __name__ == '__main__':
    unittest.main()
