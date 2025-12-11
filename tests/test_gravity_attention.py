import pytest
import numpy as np
import torch
from aule.vulkan import Aule

def ref_attention(q, k, v, rot_cos=None, rot_sin=None, causal=False):
    """PyTorch reference attention"""
    # Convert to torch
    q_t = torch.from_numpy(q)
    k_t = torch.from_numpy(k)
    v_t = torch.from_numpy(v)
    
    # RoPE
    if rot_cos is not None and rot_sin is not None:
        cos_t = torch.from_numpy(rot_cos)
        sin_t = torch.from_numpy(rot_sin)
        
        # Apply RoPE (Adjacent Pairs)
        q_embed = apply_rotary_adjacent(q, rot_cos, rot_sin)
        k_embed = apply_rotary_adjacent(k, rot_cos, rot_sin)
        
        q_embed = torch.from_numpy(q_embed)
        k_embed = torch.from_numpy(k_embed)
    else:
        q_embed = q_t
        k_embed = k_t

    # Scaled Dot Product
    d_head = q.shape[-1]
    scale = 1.0 / np.sqrt(d_head)
    
    # Attention Scores: (B, H, S, S)
    attn = torch.matmul(q_embed, k_embed.transpose(-2, -1)) * scale
    
    # Causal Mask
    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))

    # Softmax
    attn = torch.softmax(attn, dim=-1)
    
    # Output
    output = torch.matmul(attn, v_t)
    return output.numpy()

def apply_rotary_adjacent(x, cos, sin):
    # x: (..., D)
    # cos, sin: (..., D/2) or (..., D) depending on implementation
    # Shader uses adjacent pairs (0,1), (2,3)...
    # even: x[2i] * c - x[2i+1] * s
    # odd:  x[2i] * s + x[2i+1] * c
    
    # Reshape x to (..., D/2, 2)
    x_reshaped = x.reshape(x.shape[:-1] + (-1, 2))
    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]
    
    # cos/sin are (..., D) but we formed them by repeating pairs?
    # In test, we did: rot_cos = np.concatenate([rot_cos, rot_cos], axis=-1).
    # This repeats the FULL BUFFER.
    # But shader logic: rot_idx = global_row * (D/2) + (d/2).
    # d iterates 0..D. d/2 iterates 0..D/2.
    # So for pair (d, d+1), we use SAME cos/sin index.
    # So we need cos/sin of shape (..., D/2).
    # Let's use the original un-expanded cos/sin for this function.
    
    # If passed expanded cos/sin (..., D), we slice.
    if cos.shape[-1] == x.shape[-1]:
        c = cos[..., ::2]
        s = sin[..., ::2]
    else:
        c = cos
        s = sin
        
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    
    return np.stack([out0, out1], axis=-1).reshape(x.shape)

@pytest.fixture(scope="module")
def aule():
    instance = Aule()
    yield instance
    instance.close()

def test_gravity_identity(aule):
    """Test gravity attention with identity indices (0, 1, 2...) matches standard attention."""
    B, H, S, D = 1, 4, 128, 64
    
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)
    
    # Identity indices: (B, H, S)
    # Each head has indices 0..S-1
    indices = np.tile(np.arange(S, dtype=np.uint32), (B, H, 1))
    
    # 1. Standard Attention
    ref_out = ref_attention(q, k, v)
    
    # 2. Gravity Attention (Identity)
    grav_out = aule.attention_gravity(q, k, v, indices, causal=False)
    
    np.testing.assert_allclose(grav_out, ref_out, atol=1e-3, rtol=1e-3)

def test_gravity_shuffled(aule):
    """Test gravity attention with shuffled indices matches standard attention.
    Attention is permutation invariant wrt summation order, PROVIDED RoPE/Masking uses original positions.
    Our gravity kernel MUST use original positions (derived from indices) to match.
    """
    B, H, S, D = 1, 4, 128, 64
    
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)
    
    # Create shuffled indices
    indices = np.empty((B, H, S), dtype=np.uint32)
    for b in range(B):
        for h in range(H):
            indices[b, h] = np.random.permutation(S).astype(np.uint32)
            
    # 1. Standard Attention
    ref_out = ref_attention(q, k, v)
    
    # 2. Gravity Attention (Shuffled)
    grav_out = aule.attention_gravity(q, k, v, indices, causal=False)
    
    np.testing.assert_allclose(grav_out, ref_out, atol=1e-3, rtol=1e-3)

def test_gravity_rope_causal(aule):
    """Test gravity attention with RoPE and Causal Masking."""
    B, H, S, D = 1, 2, 64, 64
    
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)
    
    # RoPE
    theta = 10000.0
    freqs = 1.0 / (theta ** (np.arange(0, D, 2)[: (D // 2)].astype(np.float32) / D))
    t = np.arange(S).astype(np.float32)
    freqs = np.outer(t, freqs) # (S, D/2)
    rot_cos = np.cos(freqs) # (S, D/2)
    rot_sin = np.sin(freqs) # (S, D/2)
    
    # Expand to (1, 1, S, D/2) for broadcasting
    rot_cos = rot_cos.reshape(1, 1, S, D // 2)
    rot_sin = rot_sin.reshape(1, 1, S, D // 2)
    
    # Identity indices first
    indices = np.tile(np.arange(S, dtype=np.uint32), (B, H, 1))
    
    # 1. Ref
    ref_out = ref_attention(q, k, v, rot_cos, rot_sin, causal=True)
    
    # 2. Gravity (uses packed buffers (S, D/2))
    grav_out = aule.attention_gravity(q, k, v, indices, rot_cos=rot_cos, rot_sin=rot_sin, causal=True)
    
    np.testing.assert_allclose(grav_out, ref_out, atol=1e-3, rtol=1e-3)
    
    # Shuffled indices + RoPE + Causal
    # ...
    indices_shuff = np.empty((B, H, S), dtype=np.uint32)
    for b in range(B):
        for h in range(H):
            indices_shuff[b, h] = np.random.permutation(S).astype(np.uint32)
            
    grav_out_shuff = aule.attention_gravity(q, k, v, indices_shuff, rot_cos=rot_cos, rot_sin=rot_sin, causal=True)
    
    # Note: With "Force Identity", shuffled test will fail because we ignore indices_shuff inside shader
    # So we might expect failure here if we keep DEBUG code.
    # But let's see identity pass first.
    np.testing.assert_allclose(grav_out_shuff, ref_out, atol=1e-3, rtol=1e-3)

def test_gravity_truncated(aule):
    """Test gravity attention with truncation (max_attend < S)."""
    B, H, S, D = 1, 1, 128, 64
    max_attend = 32  # Only attend to top 32
    
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)
    
    # Sort indices by proximity to Q (heuristic)
    # For now, just identity indices
    indices = np.tile(np.arange(S, dtype=np.uint32), (B, H, 1))

    # 1. Full Attention
    ref_out = ref_attention(q, k, v)
    
    # 2. Truncated Gravity Attention (max_attend=32)
    # Since we use identity indices, this is equivalent to Local Window Attention (window size 32)
    # but only looking at 0..31 for EVERY token?
    # Wait, the shader loop is: for j in 0..limit.
    # So if indices are 0..S, it attends to indices 0..31 for ALL queries.
    
    grav_out = aule.attention_gravity(q, k, v, indices, causal=False, max_attend=max_attend)
    
    # Verify it runs (no crash)
    assert grav_out.shape == ref_out.shape
    
    # Verify it is DIFFERENT from full attention (since we dropped tokens)
    # Unless S <= max_attend
    if S > max_attend:
        assert not np.allclose(grav_out, ref_out, atol=1e-5), "Truncated should differ from full attention"

    # Verify max_attend=S matches full attention
    grav_out_full = aule.attention_gravity(q, k, v, indices, causal=False, max_attend=S)
    np.testing.assert_allclose(grav_out_full, ref_out, atol=1e-3, rtol=1e-3)

