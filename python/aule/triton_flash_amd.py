"""
aule-attention: AMD-Optimized Triton FlashAttention-2

Optimized for AMD MI300X (CDNA3) and MI200 series GPUs.
Uses Triton autotuning to find optimal configurations for each workload.

Key optimizations:
- @triton.autotune: Automatically finds best config for BLOCK_M/N, waves_per_eu
- num_stages=1: Correct pipeline depth for fused attention (FlashAttention)
- Optimized for AMD wavefront (64 threads) and HBM3 bandwidth

References:
- https://github.com/ROCm/triton/blob/triton-mlir/python/perf-kernels/flash-attention.py
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
"""

import torch
import triton
import triton.language as tl
import math


def is_amd_gpu() -> bool:
    """Check if running on AMD GPU with ROCm."""
    try:
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except:
        return False


def get_amd_gpu_arch() -> str:
    """Get AMD GPU architecture (e.g., 'gfx942' for MI300X)."""
    if not is_amd_gpu():
        return ""
    try:
        props = torch.cuda.get_device_properties(0)
        name = props.name.lower()
        if 'mi300' in name:
            return 'cdna3'
        elif 'mi200' in name or 'mi250' in name or 'mi210' in name:
            return 'cdna2'
        elif 'mi100' in name:
            return 'cdna'
        elif 'rx 7' in name or 'radeon 7' in name or 'gfx11' in name:
            return 'rdna3'
        elif 'rx 6' in name or 'radeon 6' in name or 'gfx10' in name:
            return 'rdna2'
        return 'unknown'
    except:
        return 'unknown'


# =============================================================================
# AMD AUTOTUNE CONFIGURATIONS
# From ROCm's official flash-attention.py performance kernel
# =============================================================================

def get_autotune_configs():
    """
    Get AMD-optimized autotune configurations.

    Extended configs to cover all workloads including:
    - 13B (40 heads), 70B (64 heads) - need more parallelism
    - 8K context - medium sequence optimization
    """
    return [
        # Large block configs - good for single inference, long sequences
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2}, num_stages=1, num_warps=4),
        # Medium block configs - good for 8K context, batched inference
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1}, num_stages=1, num_warps=4),
        # Higher warp configs for more heads (13B/40 heads, 70B/64 heads)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'waves_per_eu': 2}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 4}, num_stages=1, num_warps=8),
        # Smaller blocks for edge cases
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'waves_per_eu': 4}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4}, num_stages=1, num_warps=8),
        # Fallback without AMD-specific params
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
    ]


# =============================================================================
# AMD-OPTIMIZED FORWARD KERNEL WITH AUTOTUNING
# =============================================================================

@triton.autotune(
    configs=get_autotune_configs(),
    key=['head_dim'],  # Only key on head_dim - same config for all seq_lens (critical for generation)
)
@triton.jit
def _flash_attn_fwd_amd(
    Q, K, V, Out, L,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    num_heads_q, num_heads_kv,
    seq_len_q, seq_len_k, head_dim,
    scale, window_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    """
    AMD-optimized FlashAttention-2 forward kernel with autotuning.

    Automatically selects best BLOCK_M, BLOCK_N, waves_per_eu for each workload.
    """
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

    # Q pointers - coalesced access pattern for AMD HBM
    q_ptrs = Q + pid_b * stride_qb + pid_h_q * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

    # Initialize accumulators in fp32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block with masking
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Determine KV range for causal masking
    if IS_CAUSAL:
        max_kv_idx = tl.minimum((pid_m + 1) * BLOCK_M, seq_len_k)
    else:
        max_kv_idx = seq_len_k

    # Calculate number of KV blocks to process
    num_kv_blocks = tl.cdiv(max_kv_idx, BLOCK_N)

    # Main attention loop - process KV blocks
    for block_idx in range(num_kv_blocks):
        offs_n_curr = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        # K, V pointers with coalesced access
        k_ptrs = K + pid_b * stride_kb + pid_h_kv * stride_kh + offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + pid_b * stride_vb + pid_h_kv * stride_vh + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk

        # Load K, V with masking
        kv_mask = (offs_n_curr[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # Compute attention scores: S = Q @ K^T * scale
        # Maps to AMD MFMA instructions
        s = tl.dot(q, tl.trans(k)) * scale

        # Use a large negative number instead of -inf to avoid numerical issues
        NEG_INF = -1e20

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            s = tl.where(causal_mask, s, NEG_INF)

        # Sliding window mask: keep positions where q_pos - k_pos < window_size
        # For causal: attend to positions in range [q_pos - window_size + 1, q_pos]
        if window_size > 0:
            window_mask = (offs_m[:, None] - offs_n_curr[None, :]) < window_size
            s = tl.where(window_mask, s, NEG_INF)

        # Bounds mask for sequence length
        s = tl.where(offs_n_curr[None, :] < seq_len_k, s, NEG_INF)

        # Online softmax computation (FlashAttention-2 algorithm)
        # Use exp2 for faster computation on AMD (maps to v_exp_f32)
        m_ij = tl.max(s, axis=1)

        # Handle case where entire row in block is masked (m_ij ~ NEG_INF)
        # In this case, we should skip the update for that row to avoid NaN
        row_has_valid = m_ij > (NEG_INF + 1.0)

        # For rows with no valid entries, keep old max
        # Use 0.0 as a safe value for m_ij when computing m_new
        # This avoids -inf - (-inf) = nan in alpha calculation
        m_ij_safe = tl.where(row_has_valid, m_ij, 0.0)
        m_new = tl.where(row_has_valid, tl.maximum(m_i, m_ij_safe), m_i)

        # Correction factor using exp2 (faster than exp on AMD)
        # exp2(x) = 2^x, so we scale by log2(e) ≈ 1.4427
        # When m_i = -inf and row_has_valid, m_new will be a valid value
        # so alpha = exp2(-inf - valid) = 0, which is correct
        # When !row_has_valid, m_new = m_i, so m_i - m_new = 0, alpha = 1
        LOG2E: tl.constexpr = 1.4426950408889634
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        # Clamp alpha to avoid inf * 0 = nan issues
        alpha = tl.where(m_i > NEG_INF, alpha, 0.0)

        # Compute softmax weights using exp2
        p = tl.math.exp2((s - m_new[:, None]) * LOG2E)

        # Zero out rows with no valid entries to prevent garbage accumulation
        p = tl.where(row_has_valid[:, None], p, 0.0)

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Accumulate output: acc = acc * alpha + P @ V
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        # Only update m_i for rows that had valid entries
        m_i = tl.where(row_has_valid, m_new, m_i)

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h_q * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)

    # Store logsumexp for backward pass
    if STORE_LSE:
        lse = m_i + tl.log(l_i)
        lse_ptrs = L + pid_b * stride_lb + pid_h_q * stride_lh + offs_m * stride_lm
        lse_mask = offs_m < seq_len_q
        tl.store(lse_ptrs, lse, mask=lse_mask)


# =============================================================================
# NON-AUTOTUNED KERNEL (for backward pass)
# =============================================================================

@triton.jit
def _flash_attn_bwd_amd(
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
    """AMD-optimized FlashAttention-2 backward kernel."""
    pid_n = tl.program_id(0)
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
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

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
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        o = tl.load(o_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)

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

    if heads_per_kv > 1:
        tl.atomic_add(dk_ptrs, dk.to(dK.dtype.element_ty), mask=kv_mask)
        tl.atomic_add(dv_ptrs, dv.to(dV.dtype.element_ty), mask=kv_mask)
    else:
        tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=kv_mask)
        tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=kv_mask)


@triton.jit
def _compute_delta_amd(
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

# Cache for autotuned configs
_AUTOTUNE_CACHE = {}


class FlashAttentionAMDFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, scale=None, window_size=-1):
        batch, heads_q, seq_len_q, head_dim = q.shape
        _, heads_kv, seq_len_k, _ = k.shape

        assert heads_q % heads_kv == 0, \
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Keep native dtype for compute
        orig_dtype = q.dtype
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = torch.empty_like(q)
        L = torch.empty(batch, heads_q, seq_len_q,
                        device=q.device, dtype=torch.float32)

        BLOCK_K = triton.next_power_of_2(head_dim)

        # Grid must be computed dynamically based on BLOCK_M from autotune
        # Use lambda to let triton compute correct grid for each config
        def grid(META):
            return (triton.cdiv(seq_len_q, META['BLOCK_M']), batch * heads_q)

        # Launch kernel - autotune will find optimal config
        _flash_attn_fwd_amd[grid](
            q, k, v, out, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            heads_q, heads_kv,
            seq_len_q, seq_len_k, head_dim,
            scale, window_size,
            BLOCK_K=BLOCK_K,
            IS_CAUSAL=causal, STORE_LSE=True,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, L)
        ctx.scale = scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.heads_q = heads_q
        ctx.heads_kv = heads_kv
        ctx.BLOCK_K = BLOCK_K
        ctx.orig_dtype = orig_dtype

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, L = ctx.saved_tensors
        scale = ctx.scale
        causal = ctx.causal
        heads_q = ctx.heads_q
        heads_kv = ctx.heads_kv
        BLOCK_K = ctx.BLOCK_K

        batch, _, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape

        dout = dout.contiguous()

        # Use fixed block sizes for backward (simpler, still fast)
        BLOCK_M = 64
        BLOCK_N = 64

        # Compute delta = rowsum(O * dO)
        D = torch.empty(batch, heads_q, seq_len_q, device=q.device, dtype=torch.float32)
        grid_delta = (triton.cdiv(seq_len_q, BLOCK_M), batch * heads_q)
        _compute_delta_amd[grid_delta](
            out, dout, D,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            seq_len_q, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_stages=1, num_warps=4,
        )

        # Initialize gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Backward kernel
        grid_bwd = (triton.cdiv(seq_len_k, BLOCK_N), batch * heads_q)
        _flash_attn_bwd_amd[grid_bwd](
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
            num_stages=1, num_warps=4,
        )

        return (dq.to(ctx.orig_dtype), dk.to(ctx.orig_dtype),
                dv.to(ctx.orig_dtype), None, None, None)


def flash_attention_amd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float = None,
    window_size: int = -1,
) -> torch.Tensor:
    """
    AMD-optimized FlashAttention-2 with autotuning.

    Automatically finds optimal kernel configuration for MI300X and MI200 GPUs.
    First call for each head_dim runs autotuning, subsequent calls use cached
    optimal config.

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

    return FlashAttentionAMDFunc.apply(q, k, v, causal, scale, window_size)


# =============================================================================
# PAGED ATTENTION KERNEL (vLLM-compatible)
# =============================================================================

@triton.jit
def _paged_attention_fwd_amd(
    Q, K_cache, V_cache, Out,
    Block_tables,
    Context_lens,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kblk, stride_kh, stride_kd,
    stride_vb, stride_vblk, stride_vh, stride_vd,
    stride_btb, stride_btm,
    stride_ob, stride_oh, stride_od,
    num_heads_q, num_heads_kv,
    head_dim, block_size, max_context_len,
    scale, window_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
):
    """
    PagedAttention kernel for decode phase (single query token per batch).

    Compatible with vLLM-style block tables.
    K_cache, V_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    Block_tables: [batch, max_blocks_per_seq]
    """
    pid_bh = tl.program_id(0)

    pid_b = pid_bh // num_heads_q
    pid_h_q = pid_bh % num_heads_q

    # GQA mapping
    heads_per_kv = num_heads_q // num_heads_kv
    pid_h_kv = pid_h_q // heads_per_kv

    # Get context length for this batch
    context_len = tl.load(Context_lens + pid_b)
    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)

    # Load query [head_dim]
    offs_d = tl.arange(0, BLOCK_K)
    q_ptrs = Q + pid_b * stride_qb + pid_h_q * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < head_dim, other=0.0).to(tl.float32)

    # Use a large negative number instead of -inf to avoid numerical issues
    NEG_INF = -1e20

    # Initialize accumulators
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    m_i = NEG_INF
    l_i = 0.0

    # Iterate over all blocks up to MAX_NUM_BLOCKS (compile-time constant)
    offs_blk = tl.arange(0, BLOCK_SIZE)

    for block_idx in range(MAX_NUM_BLOCKS):
        # Get physical block number from block table
        # Use block 0 for out-of-range to avoid invalid memory access
        bt_idx = tl.minimum(block_idx, num_blocks - 1)
        physical_block = tl.load(Block_tables + pid_b * stride_btb + bt_idx)

        # Load K block [BLOCK_SIZE, head_dim]
        k_ptrs = (K_cache + physical_block * stride_kb +
                  offs_blk[:, None] * stride_kblk +
                  pid_h_kv * stride_kh + offs_d[None, :] * stride_kd)
        k_mask = (offs_blk[:, None] < BLOCK_SIZE) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Compute attention scores: q @ k^T * scale
        s = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_SIZE]

        # Mask invalid positions (including blocks beyond num_blocks)
        token_positions = block_idx * BLOCK_SIZE + offs_blk
        valid_mask = token_positions < context_len

        # Sliding window mask
        if window_size > 0:
            window_mask = (context_len - 1 - token_positions) < window_size
            valid_mask = valid_mask & window_mask

        s = tl.where(valid_mask, s, NEG_INF)

        # Online softmax with NaN protection
        m_ij = tl.max(s)

        # Check if block has any valid positions
        block_has_valid = m_ij > (NEG_INF + 1.0)

        # For blocks with no valid entries, keep old max to avoid -inf - (-inf) = nan
        m_ij_safe = tl.where(block_has_valid, m_ij, 0.0)
        m_new = tl.where(block_has_valid, tl.maximum(m_i, m_ij_safe), m_i)

        LOG2E: tl.constexpr = 1.4426950408889634
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        # Clamp alpha when m_i is still initial NEG_INF
        alpha = tl.where(m_i > NEG_INF, alpha, 0.0)

        p = tl.math.exp2((s - m_new) * LOG2E)
        # Zero out invalid positions
        p = tl.where(valid_mask, p, 0.0)

        l_i = l_i * alpha + tl.sum(p)

        # Load V block and accumulate
        v_ptrs = (V_cache + physical_block * stride_vb +
                  offs_blk[:, None] * stride_vblk +
                  pid_h_kv * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        # Only update m_i for blocks with valid entries
        m_i = tl.where(block_has_valid, m_new, m_i)

    # Normalize
    acc = acc / l_i

    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h_q * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_d < head_dim)


def flash_attention_paged_amd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float = None,
    window_size: int = -1,
) -> torch.Tensor:
    """
    AMD-optimized PagedAttention for decode phase.

    Compatible with vLLM-style block tables for efficient KV cache management.

    Args:
        q: Query [batch, heads_q, head_dim] - single token per sequence
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: Block mapping [batch, max_blocks_per_seq] int32
        context_lens: Actual seq length per batch [batch] int32
        scale: Attention scale (default: 1/sqrt(head_dim))
        window_size: Sliding window size (-1 for full attention)

    Returns:
        Output [batch, heads_q, head_dim]
    """
    # Handle 4D query input (squeeze seq_len dim if 1)
    if q.dim() == 4:
        assert q.shape[2] == 1, "PagedAttention only supports single query token"
        q = q.squeeze(2)

    batch, heads_q, head_dim = q.shape
    num_blocks_total, block_size, heads_kv, _ = k_cache.shape

    assert heads_q % heads_kv == 0, \
        f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous().to(torch.int32)
    context_lens = context_lens.contiguous().to(torch.int32)

    out = torch.empty(batch, heads_q, head_dim, device=q.device, dtype=q.dtype)

    BLOCK_K = triton.next_power_of_2(head_dim)
    max_context = int(context_lens.max().item())

    grid = (batch * heads_q,)

    # Calculate max blocks for compile-time loop bound
    max_num_blocks = (max_context + block_size - 1) // block_size

    _paged_attention_fwd_amd[grid](
        q, k_cache, v_cache, out,
        block_tables, context_lens,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1),
        k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1),
        v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        heads_q, heads_kv,
        head_dim, block_size, max_context,
        scale, window_size,
        BLOCK_SIZE=block_size,
        BLOCK_K=BLOCK_K,
        MAX_NUM_BLOCKS=max_num_blocks,
        num_warps=4, num_stages=1,
    )

    return out


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("AMD-OPTIMIZED FLASHATTENTION TEST (WITH AUTOTUNING)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Is AMD: {is_amd_gpu()}")
    print(f"Architecture: {get_amd_gpu_arch()}")

    # Test correctness
    print("\n--- Correctness Test ---")
    q = torch.randn(2, 32, 512, 128, device=device, dtype=torch.float16)
    k = torch.randn(2, 32, 512, 128, device=device, dtype=torch.float16)
    v = torch.randn(2, 32, 512, 128, device=device, dtype=torch.float16)

    out_amd = flash_attention_amd(q, k, v, causal=True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    diff = (out_amd - out_ref).abs().max().item()
    print(f"Max diff vs PyTorch SDPA: {diff:.6f}")
    print(f"Correctness: {'PASS' if diff < 0.01 else 'FAIL'}")

    # Benchmark
    print("\n--- Performance Benchmark (with autotuning warmup) ---")
    configs = [
        (1, 32, 2048, 128, "7B-single"),
        (8, 32, 2048, 128, "7B-batch8"),
        (1, 32, 8192, 128, "7B-8k-ctx"),
    ]

    for batch, heads, seq_len, head_dim, name in configs:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

        # Warmup (includes autotuning on first call)
        for _ in range(10):
            _ = flash_attention_amd(q, k, v, causal=True)
        torch.cuda.synchronize()

        # Benchmark AMD kernel
        start = time.perf_counter()
        for _ in range(20):
            _ = flash_attention_amd(q, k, v, causal=True)
        torch.cuda.synchronize()
        amd_time = time.perf_counter() - start
        amd_tps = (batch * seq_len * 20) / amd_time

        # Warmup PyTorch
        for _ in range(5):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(20):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        pt_time = time.perf_counter() - start
        pt_tps = (batch * seq_len * 20) / pt_time

        speedup = (amd_tps / pt_tps - 1) * 100
        print(f"{name}: PyTorch {pt_tps/1e6:.2f}M | AMD-Opt {amd_tps/1e6:.2f}M | {speedup:+.1f}%")

    print("\n" + "=" * 60)
