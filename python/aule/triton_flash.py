"""
aule-attention: Triton FlashAttention-2 Implementation

Full-featured FlashAttention implementation using Triton.
Works on AMD (MI200/MI300, RDNA3) and NVIDIA GPUs.

Features:
- Forward pass with online softmax
- Backward pass for training
- GQA (Grouped Query Attention) support
- MQA (Multi-Query Attention) support
- Sliding window attention
- Causal and non-causal masking
- fp16, bf16, fp32 support
- AMD wavefront-optimized block sizes

Based on FlashAttention-2: https://tridao.me/publications/flash2/flash2.pdf
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# FORWARD KERNEL
# =============================================================================

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out, L,  # L stores logsumexp for backward
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    num_heads_q, num_heads_kv,
    seq_len_q, seq_len_k, head_dim,
    scale,
    window_size,  # -1 for no sliding window
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_LSE: tl.constexpr,  # Whether to store logsumexp for backward
):
    """FlashAttention-2 forward kernel with GQA and sliding window support."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    num_heads = num_heads_q
    pid_b = pid_bh // num_heads
    pid_h_q = pid_bh % num_heads

    # GQA: map query head to KV head
    heads_per_kv = num_heads_q // num_heads_kv
    pid_h_kv = pid_h_q // heads_per_kv

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Q pointers
    q_ptrs = Q + pid_b * stride_qb + pid_h_q * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Determine KV range
    if IS_CAUSAL:
        max_kv_idx = (pid_m + 1) * BLOCK_M
        if window_size > 0:
            min_kv_idx = tl.maximum(0, pid_m * BLOCK_M - window_size)
        else:
            min_kv_idx = 0
    else:
        max_kv_idx = seq_len_k
        if window_size > 0:
            center = pid_m * BLOCK_M + BLOCK_M // 2
            min_kv_idx = tl.maximum(0, center - window_size // 2)
            max_kv_idx = tl.minimum(seq_len_k, center + window_size // 2)
        else:
            min_kv_idx = 0

    start_block = min_kv_idx // BLOCK_N
    num_kv_blocks = tl.cdiv(max_kv_idx - min_kv_idx, BLOCK_N)

    for block_idx in range(num_kv_blocks):
        block_n = start_block + block_idx
        offs_n_curr = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # K, V pointers
        k_ptrs = K + pid_b * stride_kb + pid_h_kv * stride_kh + offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + pid_b * stride_vb + pid_h_kv * stride_vh + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk

        # Load K, V
        kv_mask = (offs_n_curr[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # Attention scores
        s = tl.dot(q, tl.trans(k)) * scale

        # Causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            s = tl.where(causal_mask, s, float('-inf'))

        # Sliding window mask
        if window_size > 0:
            window_mask = (offs_m[:, None] - offs_n_curr[None, :]) <= window_size
            if not IS_CAUSAL:
                window_mask = window_mask & ((offs_n_curr[None, :] - offs_m[:, None]) <= window_size)
            s = tl.where(window_mask, s, float('-inf'))

        # Bounds mask
        s = tl.where(offs_n_curr[None, :] < seq_len_k, s, float('-inf'))

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(s - m_ij[:, None]) * beta[:, None], axis=1)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h_q * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)

    # Store logsumexp for backward
    if STORE_LSE:
        lse = m_i + tl.log(l_i)
        lse_ptrs = L + pid_b * stride_lb + pid_h_q * stride_lh + offs_m * stride_lm
        lse_mask = offs_m < seq_len_q
        tl.store(lse_ptrs, lse, mask=lse_mask)


# =============================================================================
# BACKWARD KERNEL
# =============================================================================

@triton.jit
def _flash_attn_bwd_kernel(
    Q, K, V, O, dO, dQ, dK, dV, L, D,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    num_heads_q, num_heads_kv,
    seq_len_q, seq_len_k, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """FlashAttention-2 backward kernel."""
    pid_n = tl.program_id(0)  # KV block
    pid_bh = tl.program_id(1)

    num_heads = num_heads_q
    pid_b = pid_bh // num_heads
    pid_h_q = pid_bh % num_heads
    heads_per_kv = num_heads_q // num_heads_kv
    pid_h_kv = pid_h_q // heads_per_kv

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Load K, V for this block
    k_ptrs = K + pid_b * stride_kb + pid_h_kv * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V + pid_b * stride_vb + pid_h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    kv_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

    # Accumulators for dK, dV
    dk = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)

    # Determine Q range for causal
    if IS_CAUSAL:
        start_m = pid_n * BLOCK_N // BLOCK_M
    else:
        start_m = 0
    num_m_blocks = tl.cdiv(seq_len_q, BLOCK_M)

    for block_m in range(start_m, num_m_blocks):
        offs_m_curr = block_m * BLOCK_M + tl.arange(0, BLOCK_M)

        # Load Q, O, dO, L, D
        q_ptrs = Q + pid_b * stride_qb + pid_h_q * stride_qh + offs_m_curr[:, None] * stride_qm + offs_k[None, :] * stride_qk
        o_ptrs = O + pid_b * stride_ob + pid_h_q * stride_oh + offs_m_curr[:, None] * stride_om + offs_k[None, :] * stride_ok
        do_ptrs = dO + pid_b * stride_ob + pid_h_q * stride_oh + offs_m_curr[:, None] * stride_om + offs_k[None, :] * stride_ok
        l_ptrs = L + pid_b * stride_lb + pid_h_q * stride_lh + offs_m_curr * stride_lm
        d_ptrs = D + pid_b * stride_lb + pid_h_q * stride_lh + offs_m_curr * stride_lm

        q_mask = (offs_m_curr[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        o = tl.load(o_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        l_mask = offs_m_curr < seq_len_q
        lse = tl.load(l_ptrs, mask=l_mask, other=0.0)
        delta = tl.load(d_ptrs, mask=l_mask, other=0.0)

        # Recompute attention
        s = tl.dot(q, tl.trans(k)) * scale

        if IS_CAUSAL:
            causal_mask = offs_m_curr[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        s = tl.where(offs_n[None, :] < seq_len_k, s, float('-inf'))

        p = tl.exp(s - lse[:, None])

        # dV = P^T @ dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))

        # dS = P * (dP - delta)
        ds = p * (dp - delta[:, None]) * scale

        # dK = dS^T @ Q
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

        # dQ (atomic add)
        dq = tl.dot(ds.to(k.dtype), k)
        dq_ptrs = dQ + pid_b * stride_qb + pid_h_q * stride_qh + offs_m_curr[:, None] * stride_qm + offs_k[None, :] * stride_qk
        tl.atomic_add(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=q_mask)

    # Store dK, dV
    dk_ptrs = dK + pid_b * stride_kb + pid_h_kv * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    dv_ptrs = dV + pid_b * stride_vb + pid_h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    # For GQA, we need to accumulate gradients from multiple Q heads
    if heads_per_kv > 1:
        tl.atomic_add(dk_ptrs, dk.to(dK.dtype.element_ty), mask=kv_mask)
        tl.atomic_add(dv_ptrs, dv.to(dV.dtype.element_ty), mask=kv_mask)
    else:
        tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=kv_mask)
        tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=kv_mask)


@triton.jit
def _compute_delta_kernel(
    O, dO, D,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_db, stride_dh, stride_dm,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute delta = rowsum(O * dO) for backward pass."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    o_ptrs = O + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    do_ptrs = dO + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

    mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    d_ptrs = D + pid_bh * stride_dh + offs_m * stride_dm
    tl.store(d_ptrs, delta, mask=offs_m < seq_len)


# =============================================================================
# PYTHON INTERFACE
# =============================================================================

class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, scale=None, window_size=-1):
        batch, heads_q, seq_len_q, head_dim = q.shape
        _, heads_kv, seq_len_k, _ = k.shape

        assert heads_q % heads_kv == 0, f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Ensure contiguous and same dtype
        orig_dtype = q.dtype
        q = q.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()

        out = torch.empty_like(q)

        # Logsumexp for backward
        L = torch.empty(batch, heads_q, seq_len_q, device=q.device, dtype=torch.float32)

        # Block sizes
        if head_dim <= 64:
            BLOCK_M, BLOCK_N = 64, 64
        elif head_dim <= 128:
            BLOCK_M, BLOCK_N = 32, 32
        else:
            BLOCK_M, BLOCK_N = 16, 16
        BLOCK_K = triton.next_power_of_2(head_dim)

        grid = (triton.cdiv(seq_len_q, BLOCK_M), batch * heads_q)

        _flash_attn_fwd_kernel[grid](
            q, k, v, out, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            heads_q, heads_kv,
            seq_len_q, seq_len_k, head_dim,
            scale, window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IS_CAUSAL=causal, STORE_LSE=True,
        )

        ctx.save_for_backward(q, k, v, out, L)
        ctx.scale = scale
        ctx.causal = causal
        ctx.heads_q = heads_q
        ctx.heads_kv = heads_kv
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_K = BLOCK_K
        ctx.orig_dtype = orig_dtype

        return out.to(orig_dtype)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, L = ctx.saved_tensors
        scale = ctx.scale
        causal = ctx.causal
        heads_q = ctx.heads_q
        heads_kv = ctx.heads_kv
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        BLOCK_K = ctx.BLOCK_K

        batch, _, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape

        dout = dout.contiguous().float()

        # Compute delta = rowsum(O * dO)
        D = torch.empty(batch, heads_q, seq_len_q, device=q.device, dtype=torch.float32)
        grid_delta = (triton.cdiv(seq_len_q, BLOCK_M), batch * heads_q)
        _compute_delta_kernel[grid_delta](
            out, dout, D,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            seq_len_q, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Initialize gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Backward kernel
        grid_bwd = (triton.cdiv(seq_len_k, BLOCK_N), batch * heads_q)
        _flash_attn_bwd_kernel[grid_bwd](
            q, k, v, out, dout, dq, dk, dv, L, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            heads_q, heads_kv,
            seq_len_q, seq_len_k, head_dim,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IS_CAUSAL=causal,
        )

        return dq.to(ctx.orig_dtype), dk.to(ctx.orig_dtype), dv.to(ctx.orig_dtype), None, None, None


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float = None,
    window_size: int = -1,
) -> torch.Tensor:
    """
    FlashAttention-2 with full backward pass support.

    Args:
        q: Query [batch, heads_q, seq_len_q, head_dim]
        k: Key [batch, heads_kv, seq_len_k, head_dim]
        v: Value [batch, heads_kv, seq_len_k, head_dim]
        causal: Apply causal masking
        scale: Attention scale (default: 1/sqrt(head_dim))
        window_size: Sliding window size (-1 for full attention)

    Returns:
        Output [batch, heads_q, seq_len_q, head_dim]
    """
    assert q.dim() == 4
    assert k.dim() == 4 and v.dim() == 4
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[1] == v.shape[1]
    assert k.shape[2] == v.shape[2]
    assert q.shape[1] % k.shape[1] == 0

    return FlashAttentionTritonFunc.apply(q, k, v, causal, scale, window_size)


def is_triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Alias
flash_attention = flash_attention_triton


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 60)
    print("AULE-ATTENTION TRITON KERNEL TESTS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available")
        exit(1)

    def test(name, q, k, v, causal=True, rtol=1e-2):
        """Test forward and backward."""
        q = q.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        # Forward
        out = flash_attention_triton(q, k, v, causal=causal)
        gqa = q.shape[1] != k.shape[1]
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal, enable_gqa=gqa)

        fwd_diff = (out - out_ref).abs().max().item()
        fwd_ok = fwd_diff < rtol

        # Backward
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        out_ref.backward(grad_out)

        dq_diff = (q.grad - q_ref.grad).abs().max().item()
        dk_diff = (k.grad - k_ref.grad).abs().max().item()
        dv_diff = (v.grad - v_ref.grad).abs().max().item()

        bwd_ok = dq_diff < rtol and dk_diff < rtol and dv_diff < rtol

        status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
        print(f"{status}: {name}")
        print(f"      fwd={fwd_diff:.6f}, dQ={dq_diff:.6f}, dK={dk_diff:.6f}, dV={dv_diff:.6f}")
        return fwd_ok and bwd_ok

    # Tests
    results = []

    print("\n--- Forward + Backward Tests ---")

    # MHA
    q = torch.randn(2, 8, 64, 64, device=device, dtype=torch.float32)
    k = torch.randn(2, 8, 64, 64, device=device, dtype=torch.float32)
    v = torch.randn(2, 8, 64, 64, device=device, dtype=torch.float32)
    results.append(test("MHA (8 heads, dim=64)", q, k, v))

    # GQA
    q = torch.randn(2, 12, 64, 64, device=device, dtype=torch.float32)
    k = torch.randn(2, 2, 64, 64, device=device, dtype=torch.float32)
    v = torch.randn(2, 2, 64, 64, device=device, dtype=torch.float32)
    results.append(test("GQA (12/2 heads)", q, k, v))

    # Large head_dim
    q = torch.randn(1, 8, 32, 128, device=device, dtype=torch.float32)
    k = torch.randn(1, 8, 32, 128, device=device, dtype=torch.float32)
    v = torch.randn(1, 8, 32, 128, device=device, dtype=torch.float32)
    results.append(test("head_dim=128", q, k, v))

    # Non-causal
    q = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    k = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    v = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    results.append(test("Non-causal", q, k, v, causal=False))

    print(f"\n{'=' * 60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
