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
- Fused RoPE (Rotary Position Embedding)
- fp16, bf16, fp32 support
- AMD wavefront-optimized block sizes

Based on FlashAttention-2: https://tridao.me/publications/flash2/flash2.pdf
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# ROPE HELPERS
# =============================================================================

@triton.jit
def _apply_rope(x, cos, sin, BLOCK_K: tl.constexpr):
    """Apply rotary position embedding to a tensor.

    x: [BLOCK_M/N, BLOCK_K] - query or key block
    cos, sin: [BLOCK_M/N, BLOCK_K] - precomputed cos/sin for positions

    RoPE formula for each pair (x[i], x[i+d/2]):
        x_rot[i]     = x[i] * cos[i] - x[i+d/2] * sin[i]
        x_rot[i+d/2] = x[i] * sin[i] + x[i+d/2] * cos[i]
    """
    # Split into first half and second half
    half_k = BLOCK_K // 2
    x1 = x[:, :half_k]
    x2 = x[:, half_k:]
    cos1 = cos[:, :half_k]
    sin1 = sin[:, :half_k]

    # Rotate
    o1 = x1 * cos1 - x2 * sin1
    o2 = x1 * sin1 + x2 * cos1

    # Concatenate back - use explicit indexing
    # Note: tl.join not available in all Triton versions, so we write back directly
    return o1, o2


# =============================================================================
# FORWARD KERNEL (WITH OPTIONAL ROPE)
# =============================================================================

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out, L,  # L stores logsumexp for backward
    Cos, Sin,  # RoPE cos/sin buffers (can be None via USE_ROPE=False)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    stride_cos_s, stride_cos_d,  # RoPE strides
    num_heads_q, num_heads_kv,
    seq_len_q, seq_len_k, head_dim,
    scale,
    window_size,  # -1 for no sliding window
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_LSE: tl.constexpr,  # Whether to store logsumexp for backward
    USE_ROPE: tl.constexpr,  # Whether to apply RoPE
):
    """FlashAttention-2 forward kernel with GQA, sliding window, and fused RoPE support."""
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

    # Apply RoPE to Q if enabled
    if USE_ROPE:
        # Load cos/sin for Q positions
        half_k = BLOCK_K // 2
        offs_k_half = tl.arange(0, half_k)
        cos_q_ptrs = Cos + offs_m[:, None] * stride_cos_s + offs_k_half[None, :] * stride_cos_d
        sin_q_ptrs = Sin + offs_m[:, None] * stride_cos_s + offs_k_half[None, :] * stride_cos_d
        cos_mask = (offs_m[:, None] < seq_len_q) & (offs_k_half[None, :] < head_dim // 2)
        cos_q = tl.load(cos_q_ptrs, mask=cos_mask, other=1.0).to(tl.float32)
        sin_q = tl.load(sin_q_ptrs, mask=cos_mask, other=0.0).to(tl.float32)

        # Apply rotation: q1' = q1*cos - q2*sin, q2' = q1*sin + q2*cos
        q1 = q[:, :half_k]
        q2 = q[:, half_k:BLOCK_K]
        q1_rot = q1 * cos_q - q2 * sin_q
        q2_rot = q1 * sin_q + q2 * cos_q
        # Reassemble q (write back to registers)
        q = tl.join(q1_rot, q2_rot)

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

        # Apply RoPE to K if enabled
        if USE_ROPE:
            half_k = BLOCK_K // 2
            offs_k_half = tl.arange(0, half_k)
            cos_k_ptrs = Cos + offs_n_curr[:, None] * stride_cos_s + offs_k_half[None, :] * stride_cos_d
            sin_k_ptrs = Sin + offs_n_curr[:, None] * stride_cos_s + offs_k_half[None, :] * stride_cos_d
            cos_k_mask = (offs_n_curr[:, None] < seq_len_k) & (offs_k_half[None, :] < head_dim // 2)
            cos_k = tl.load(cos_k_ptrs, mask=cos_k_mask, other=1.0).to(tl.float32)
            sin_k = tl.load(sin_k_ptrs, mask=cos_k_mask, other=0.0).to(tl.float32)

            # Apply rotation: k1' = k1*cos - k2*sin, k2' = k1*sin + k2*cos
            k1 = k[:, :half_k]
            k2 = k[:, half_k:BLOCK_K]
            k1_rot = k1 * cos_k - k2 * sin_k
            k2_rot = k1 * sin_k + k2 * cos_k
            k = tl.join(k1_rot, k2_rot)

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
    def forward(ctx, q, k, v, causal=True, scale=None, window_size=-1, cos=None, sin=None):
        batch, heads_q, seq_len_q, head_dim = q.shape
        _, heads_kv, seq_len_k, _ = k.shape

        assert heads_q % heads_kv == 0, f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Ensure contiguous - keep native dtype for FP16/BF16 compute
        orig_dtype = q.dtype
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Use native FP16/BF16 for compute if input is FP16/BF16 (faster on modern GPUs)
        # Fall back to FP32 for FP32 input or other dtypes
        if orig_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = orig_dtype
        else:
            compute_dtype = torch.float32
            q = q.float()
            k = k.float()
            v = v.float()

        # RoPE handling
        use_rope = cos is not None and sin is not None
        if use_rope:
            assert cos.shape[-1] == head_dim // 2, f"cos must have shape [..., {head_dim // 2}], got {cos.shape}"
            assert sin.shape[-1] == head_dim // 2, f"sin must have shape [..., {head_dim // 2}], got {sin.shape}"
            cos = cos.contiguous().float()
            sin = sin.contiguous().float()
            # Ensure 2D: [max_seq_len, head_dim // 2]
            if cos.dim() == 3:
                cos = cos.squeeze(0)
            if sin.dim() == 3:
                sin = sin.squeeze(0)
            stride_cos_s = cos.stride(0)
            stride_cos_d = cos.stride(1)
        else:
            # Dummy tensors for kernel (won't be accessed when USE_ROPE=False)
            cos = torch.empty(1, 1, device=q.device, dtype=torch.float32)
            sin = torch.empty(1, 1, device=q.device, dtype=torch.float32)
            stride_cos_s = 0
            stride_cos_d = 0

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
            cos, sin,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            stride_cos_s, stride_cos_d,
            heads_q, heads_kv,
            seq_len_q, seq_len_k, head_dim,
            scale, window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IS_CAUSAL=causal, STORE_LSE=True, USE_ROPE=use_rope,
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

        return dq.to(ctx.orig_dtype), dk.to(ctx.orig_dtype), dv.to(ctx.orig_dtype), None, None, None, None, None


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


def flash_attention_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    causal: bool = True,
    scale: float = None,
    window_size: int = -1,
) -> torch.Tensor:
    """
    Fused RoPE + FlashAttention-2 in a single kernel.

    This applies Rotary Position Embedding to Q and K inside the attention
    kernel, avoiding separate memory round-trips for RoPE. This is more
    efficient than applying RoPE separately before attention.

    Args:
        q: Query [batch, heads_q, seq_len_q, head_dim]
        k: Key [batch, heads_kv, seq_len_k, head_dim]
        v: Value [batch, heads_kv, seq_len_k, head_dim]
        cos: Cosine frequencies [seq_len, head_dim // 2] or [1, seq_len, head_dim // 2]
        sin: Sine frequencies [seq_len, head_dim // 2] or [1, seq_len, head_dim // 2]
        causal: Apply causal masking (default True)
        scale: Attention scale (default: 1/sqrt(head_dim))
        window_size: Sliding window size (-1 for full attention)

    Returns:
        Output [batch, heads_q, seq_len_q, head_dim]

    Note:
        RoPE is applied as: x_rot = x * cos + rotate_half(x) * sin
        Where rotate_half swaps the two halves of the head dimension with negation.
    """
    assert q.dim() == 4
    assert k.dim() == 4 and v.dim() == 4
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[1] == v.shape[1]
    assert k.shape[2] == v.shape[2]
    assert q.shape[1] % k.shape[1] == 0
    assert cos is not None and sin is not None, "cos and sin are required for RoPE"

    return FlashAttentionTritonFunc.apply(q, k, v, causal, scale, window_size, cos, sin)


def is_triton_available() -> bool:
    """Check if Triton is available and functional.

    Note: Triton AMD backend doesn't work on Windows (uses Linux-specific ldconfig).
    On Windows with AMD GPU, we return False to fall back to Vulkan.
    """
    try:
        import triton
        import torch
        import sys

        if not torch.cuda.is_available():
            return False

        # Check for Windows + AMD combination (Triton AMD doesn't work on Windows)
        if sys.platform == 'win32':
            # Check if this is an AMD GPU (ROCm presents as CUDA on Windows)
            device_name = torch.cuda.get_device_name(0).lower()
            if 'amd' in device_name or 'radeon' in device_name or 'gfx' in device_name:
                # Triton AMD backend uses Linux-specific commands, doesn't work on Windows
                return False

        return True
    except ImportError:
        return False
    except Exception:
        # Any error during detection means Triton isn't usable
        return False


# Alias
flash_attention = flash_attention_triton


# =============================================================================
# ROPE HELPERS
# =============================================================================

def precompute_rope_frequencies(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """
    Precompute RoPE cos/sin frequencies.

    Args:
        seq_len: Maximum sequence length
        head_dim: Attention head dimension
        base: RoPE base frequency (default 10000.0)
        device: Device to put tensors on
        dtype: Data type for the frequencies

    Returns:
        cos, sin: Tensors of shape [seq_len, head_dim // 2]
    """
    # Compute frequencies: theta_i = base^(-2i/d) for i in 0..d/2
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))

    # Compute positions
    positions = torch.arange(seq_len, device=device, dtype=dtype)

    # Outer product: [seq_len, half_dim]
    angles = positions[:, None] * freqs[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope_separate(q, k, cos, sin):
    """
    Apply RoPE to Q and K separately (for comparison/reference).

    This is the standard non-fused implementation.
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    seq_len = q.shape[2]
    # Expand cos/sin to match q/k shape: [1, 1, seq_len, head_dim // 2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Expand to full head_dim by concatenating
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)

    q_rotated = q * cos_full + rotate_half(q) * sin_full
    k_rotated = k * cos_full + rotate_half(k) * sin_full

    return q_rotated, k_rotated


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

    # --- Fused RoPE Tests ---
    print("\n--- Fused RoPE + Attention Tests ---")

    def test_rope(name, batch, heads, seq_len, head_dim, causal=True, rtol=1e-2):
        """Test fused RoPE against separate RoPE + attention."""
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

        # Precompute frequencies
        cos, sin = precompute_rope_frequencies(seq_len, head_dim, device=device)

        # Reference: separate RoPE + attention
        q_rot, k_rot = apply_rope_separate(q, k, cos, sin)
        out_ref = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=causal)

        # Fused: RoPE + attention in one kernel
        out_fused = flash_attention_rope(q, k, v, cos, sin, causal=causal)

        diff = (out_fused - out_ref).abs().max().item()
        ok = diff < rtol

        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name} | diff={diff:.6f}")
        return ok

    results.append(test_rope("Fused RoPE (64 dim)", 2, 8, 64, 64))
    results.append(test_rope("Fused RoPE (128 dim)", 1, 8, 32, 128))
    results.append(test_rope("Fused RoPE (non-causal)", 1, 4, 64, 64, causal=False))

    print(f"\n{'=' * 60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
