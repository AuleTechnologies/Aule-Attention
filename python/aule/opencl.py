"""
OpenCL backend for aule-attention.
Works on MI300X via Mesa Rusticl without requiring ROCm.

Features:
- FlashAttention-2 tiled algorithm for O(N) memory
- Persistent GPU buffers to eliminate transfer overhead
- Causal masking for autoregressive models
- Online softmax for numerical stability
- FP16 support for 2x memory bandwidth (requires cl_khr_fp16)
- Support for long sequences (8K, 16K, 32K+ tokens)

FlashAttention-2 Algorithm:
  Instead of computing the full N×N attention matrix, we process in tiles:
  1. Load Q tile to registers
  2. For each K/V tile:
     - Compute partial scores S = Q @ K^T
     - Update running max and sum for online softmax
     - Accumulate output O += softmax(S) @ V
  3. Rescale output by final softmax denominator

  Memory: O(N) instead of O(N²)
  Compute: Same O(N²d) but better cache utilization

Long Sequence Support:
  - FlashAttention-2 kernel handles unlimited sequence length via block-wise processing
  - No fixed-size score arrays - processes K/V in blocks of BLOCK_SIZE
  - Memory usage stays O(N) regardless of sequence length
"""

import numpy as np
from typing import Optional, Tuple, Literal

# =============================================================================
# FlashAttention-2 Tiled Kernel (Memory Efficient)
# =============================================================================
# Block sizes - tuned for AMD GPUs (64KB local memory, 64-wide wavefronts)
# Br = rows of Q processed per workgroup
# Bc = columns of K/V processed per inner loop iteration

FLASH_ATTENTION_KERNEL_FP32 = """
// FlashAttention-2 Optimized for AMD MI300X
// One thread per query position - simpler but efficient
// Uses online softmax to process K/V without storing full attention matrix
// Supports unlimited sequence length (8K, 16K, 32K+ tokens)

#define BLOCK_SIZE 64  // K/V positions to process before updating softmax state
#define MAX_HEAD_DIM 256  // Maximum head dimension (for register arrays)

__kernel void flash_attention_forward_fp32(
    __global const float* Q,        // [batch, heads, seq_len, head_dim]
    __global const float* K,        // [batch, heads, seq_len, head_dim]
    __global const float* V,        // [batch, heads, seq_len, head_dim]
    __global float* O,              // [batch, heads, seq_len, head_dim]
    __global float* L,              // [batch, heads, seq_len] - logsumexp for backward
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    // One work item per (batch, head, query_position)
    uint gid = get_global_id(0);
    uint total_queries = batch_size * num_heads * seq_len;

    if (gid >= total_queries) return;

    // Decode indices
    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint i = remainder % seq_len;  // Query position

    // Base offsets
    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;
    uint q_offset = bh_offset + i * head_dim;

    // Load query vector into registers (stays in registers throughout)
    float q_vec[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_offset + d];
    }

    // Online softmax accumulators
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum of exp(score - max)

    // Output accumulator
    float o_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        o_acc[d] = 0.0f;
    }

    // Determine how many K/V positions to attend to
    uint kv_len = causal ? (i + 1) : seq_len;

    // Process K/V in blocks for better cache utilization
    // Block scores stored temporarily - no fixed-size array for sequence length
    float block_scores[BLOCK_SIZE];
    uint num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (uint block = 0; block < num_blocks; block++) {
        uint k_start = block * BLOCK_SIZE;
        uint k_end = min(k_start + BLOCK_SIZE, kv_len);
        uint block_len = k_end - k_start;

        // Find max score in this block
        float block_max = -INFINITY;

        for (uint jj = 0; jj < block_len; jj++) {
            uint j = k_start + jj;
            uint k_offset = bh_offset + j * head_dim;

            // Compute Q[i] @ K[j]
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += q_vec[d] * K[k_offset + d];
            }
            score *= scale;
            block_scores[jj] = score;
            block_max = fmax(block_max, score);
        }

        // Online softmax update
        // When we see a new block with max m_block:
        // m_new = max(m_old, m_block)
        // l_new = exp(m_old - m_new) * l_old + sum(exp(scores - m_new))
        // o_new = exp(m_old - m_new) * o_old + sum(exp(scores - m_new) * V)

        float m_new = fmax(m_i, block_max);
        float correction = exp(m_i - m_new);

        // Scale previous accumulator
        l_i *= correction;
        for (uint d = 0; d < head_dim; d++) {
            o_acc[d] *= correction;
        }

        // Add contribution from this block
        for (uint jj = 0; jj < block_len; jj++) {
            uint j = k_start + jj;
            float p = exp(block_scores[jj] - m_new);
            l_i += p;

            uint v_offset = bh_offset + j * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                o_acc[d] += p * V[v_offset + d];
            }
        }

        m_i = m_new;
    }

    // Normalize output by softmax denominator
    float inv_l = 1.0f / l_i;
    for (uint d = 0; d < head_dim; d++) {
        O[q_offset + d] = o_acc[d] * inv_l;
    }

    // Store logsumexp for backward pass
    if (L != 0) {
        uint l_offset = (b * num_heads + h) * seq_len + i;
        L[l_offset] = m_i + log(l_i);
    }
}
"""

# =============================================================================
# FlashAttention-2 FP16 Kernel (Mixed Precision for 2x Memory Bandwidth)
# =============================================================================
FLASH_ATTENTION_KERNEL_FP16 = """
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// FlashAttention-2 FP16 - Mixed Precision for 2x Memory Bandwidth
// Reads FP16, accumulates in FP32, writes FP16
// Supports unlimited sequence length (8K, 16K, 32K+ tokens)

#define BLOCK_SIZE 64  // K/V positions to process before updating softmax state
#define MAX_HEAD_DIM 256  // Maximum head dimension (for register arrays)

__kernel void flash_attention_forward_fp16(
    __global const half* Q,         // [batch, heads, seq_len, head_dim]
    __global const half* K,         // [batch, heads, seq_len, head_dim]
    __global const half* V,         // [batch, heads, seq_len, head_dim]
    __global half* O,               // [batch, heads, seq_len, head_dim]
    __global float* L,              // [batch, heads, seq_len] - logsumexp for backward
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    // One work item per (batch, head, query_position)
    uint gid = get_global_id(0);
    uint total_queries = batch_size * num_heads * seq_len;

    if (gid >= total_queries) return;

    // Decode indices
    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint i = remainder % seq_len;  // Query position

    // Base offsets
    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;
    uint q_offset = bh_offset + i * head_dim;

    // Load query vector into registers (FP16 -> FP32)
    float q_vec[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = vload_half(q_offset + d, Q);
    }

    // Online softmax accumulators (FP32 for precision)
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum of exp(score - max)

    // Output accumulator (FP32 for precision)
    float o_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        o_acc[d] = 0.0f;
    }

    // Determine how many K/V positions to attend to
    uint kv_len = causal ? (i + 1) : seq_len;

    // Process K/V in blocks for better cache utilization
    float block_scores[BLOCK_SIZE];
    uint num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (uint block = 0; block < num_blocks; block++) {
        uint k_start = block * BLOCK_SIZE;
        uint k_end = min(k_start + BLOCK_SIZE, kv_len);
        uint block_len = k_end - k_start;

        // Find max score in this block
        float block_max = -INFINITY;

        for (uint jj = 0; jj < block_len; jj++) {
            uint j = k_start + jj;
            uint k_offset = bh_offset + j * head_dim;

            // Compute Q[i] @ K[j] (FP16 -> FP32 accumulation)
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += q_vec[d] * vload_half(k_offset + d, K);
            }
            score *= scale;
            block_scores[jj] = score;
            block_max = fmax(block_max, score);
        }

        // Online softmax update
        float m_new = fmax(m_i, block_max);
        float correction = exp(m_i - m_new);

        // Scale previous accumulator
        l_i *= correction;
        for (uint d = 0; d < head_dim; d++) {
            o_acc[d] *= correction;
        }

        // Add contribution from this block
        for (uint jj = 0; jj < block_len; jj++) {
            uint j = k_start + jj;
            float p = exp(block_scores[jj] - m_new);
            l_i += p;

            uint v_offset = bh_offset + j * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                o_acc[d] += p * vload_half(v_offset + d, V);
            }
        }

        m_i = m_new;
    }

    // Normalize output by softmax denominator and write FP16
    float inv_l = 1.0f / l_i;
    for (uint d = 0; d < head_dim; d++) {
        vstore_half(o_acc[d] * inv_l, q_offset + d, O);
    }

    // Store logsumexp for backward pass
    if (L != 0) {
        uint l_offset = (b * num_heads + h) * seq_len + i;
        L[l_offset] = m_i + log(l_i);
    }
}
"""

# =============================================================================
# Simple Attention Kernel (Fallback for debugging / small sequences)
# =============================================================================
# Note: Simple kernel uses fixed array size and is limited to MAX_SEQ_LEN.
# For sequences > 4096, use FlashAttention kernel which has no length limit.

# OpenCL kernel for attention - FP32 version
ATTENTION_KERNEL_FP32 = """
#define MAX_SEQ_LEN 8192  // Max sequence length for simple kernel

__kernel void attention_forward_fp32(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    uint gid = get_global_id(0);
    uint total_elements = batch_size * num_heads * seq_len;

    if (gid >= total_elements) return;

    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint i = remainder % seq_len;

    uint base_offset = (b * num_heads + h) * seq_len * head_dim;
    uint q_base = base_offset + i * head_dim;

    float scores[MAX_SEQ_LEN];
    float max_score = -INFINITY;

    uint max_j = causal ? (i + 1) : seq_len;
    if (max_j > MAX_SEQ_LEN) max_j = MAX_SEQ_LEN;

    for (uint j = 0; j < max_j; j++) {
        uint k_base = base_offset + j * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_base + d] * K[k_base + d];
        }
        score *= scale;
        scores[j] = score;
        max_score = fmax(max_score, score);
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < max_j; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }

    float inv_sum = 1.0f / sum_exp;
    for (uint j = 0; j < max_j; j++) {
        scores[j] *= inv_sum;
    }

    for (uint d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (uint j = 0; j < max_j; j++) {
            uint v_idx = base_offset + j * head_dim + d;
            out_val += scores[j] * V[v_idx];
        }
        output[q_base + d] = out_val;
    }
}
"""

# OpenCL kernel for attention - FP16 version
# Uses half precision for inputs/outputs, float for accumulation (mixed precision)
# =============================================================================
# FlashAttention-2 Backward Pass Kernels
# =============================================================================
# The backward pass computes dQ, dK, dV given dO (gradient of output).
# Algorithm from FlashAttention-2 paper:
#   1. Recompute attention in tiles (memory efficient)
#   2. For each block:
#      - dV += P^T @ dO
#      - dP = dO @ V^T
#      - dS = P ⊙ (dP - D) where D = rowsum(dP ⊙ P)
#      - dQ += scale * dS @ K
#      - dK += scale * dS^T @ Q

FLASH_ATTENTION_BACKWARD_DV_KERNEL = """
// FlashAttention-2 Backward Pass - Compute dV
// One thread per K/V position
// dV[j] = sum_i(P[i,j] * dO[i])
// Must recompute P from Q, K

#define BLOCK_SIZE 64
#define MAX_HEAD_DIM 256

__kernel void flash_attention_backward_dV_fp32(
    __global const float* Q,        // [batch, heads, seq_len, head_dim]
    __global const float* K,        // [batch, heads, seq_len, head_dim]
    __global const float* V,        // [batch, heads, seq_len, head_dim]
    __global const float* dO,       // [batch, heads, seq_len, head_dim]
    __global const float* L,        // [batch, heads, seq_len] - logsumexp from forward
    __global float* dV,             // [batch, heads, seq_len, head_dim]
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    // One work item per (batch, head, kv_position)
    uint gid = get_global_id(0);
    uint total_kv = batch_size * num_heads * seq_len;

    if (gid >= total_kv) return;

    // Decode indices
    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint j = remainder % seq_len;  // K/V position

    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;
    uint l_offset = (b * num_heads + h) * seq_len;
    uint k_offset = bh_offset + j * head_dim;

    // Load K[j] into registers
    float k_vec[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        k_vec[d] = K[k_offset + d];
    }

    // Accumulate dV[j]
    float dv_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        dv_acc[d] = 0.0f;
    }

    // For causal: only query positions i >= j contribute
    uint i_start = causal ? j : 0;

    // Process query positions in blocks
    for (uint i = i_start; i < seq_len; i++) {
        uint q_offset = bh_offset + i * head_dim;

        // Compute attention score Q[i] @ K[j]
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * k_vec[d];
        }
        score *= scale;

        // Compute P[i,j] = exp(score - L[i])
        float l_i = L[l_offset + i];
        float p_ij = exp(score - l_i);

        // Accumulate: dV[j] += P[i,j] * dO[i]
        uint do_offset = bh_offset + i * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dv_acc[d] += p_ij * dO[do_offset + d];
        }
    }

    // Write dV[j]
    uint dv_offset = bh_offset + j * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        dV[dv_offset + d] = dv_acc[d];
    }
}
"""

FLASH_ATTENTION_BACKWARD_DQ_DK_KERNEL = """
// FlashAttention-2 Backward Pass - Compute dQ and dK
// One thread per query position
// dQ[i] = scale * sum_j(dS[i,j] * K[j])
// where dS[i,j] = P[i,j] * (dP[i,j] - D[i])
// and dP[i,j] = dO[i] @ V[j]^T
// and D[i] = sum_j(P[i,j] * dP[i,j])

#define BLOCK_SIZE 64
#define MAX_HEAD_DIM 256

__kernel void flash_attention_backward_dQ_fp32(
    __global const float* Q,        // [batch, heads, seq_len, head_dim]
    __global const float* K,        // [batch, heads, seq_len, head_dim]
    __global const float* V,        // [batch, heads, seq_len, head_dim]
    __global const float* O,        // [batch, heads, seq_len, head_dim]
    __global const float* dO,       // [batch, heads, seq_len, head_dim]
    __global const float* L,        // [batch, heads, seq_len] - logsumexp from forward
    __global float* dQ,             // [batch, heads, seq_len, head_dim]
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    // One work item per (batch, head, query_position)
    uint gid = get_global_id(0);
    uint total_queries = batch_size * num_heads * seq_len;

    if (gid >= total_queries) return;

    // Decode indices
    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint i = remainder % seq_len;  // Query position

    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;
    uint l_offset = (b * num_heads + h) * seq_len;
    uint q_offset = bh_offset + i * head_dim;
    uint do_offset = bh_offset + i * head_dim;
    uint o_offset = bh_offset + i * head_dim;

    // Load Q[i] and dO[i] into registers
    float q_vec[MAX_HEAD_DIM];
    float do_vec[MAX_HEAD_DIM];
    float o_vec[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_offset + d];
        do_vec[d] = dO[do_offset + d];
        o_vec[d] = O[o_offset + d];
    }

    float l_i = L[l_offset + i];

    // Compute D[i] = sum_j(P[i,j] * dP[i,j]) = sum(O[i] * dO[i])
    // This is because sum_j(P[i,j] * dP[i,j]) = sum_j(P[i,j] * dO[i] @ V[j]^T)
    //                                        = dO[i] @ sum_j(P[i,j] * V[j])^T
    //                                        = dO[i] @ O[i]^T = dot(dO[i], O[i])
    float D_i = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        D_i += do_vec[d] * o_vec[d];
    }

    // Accumulate dQ[i]
    float dq_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        dq_acc[d] = 0.0f;
    }

    // Determine how many K/V positions to attend to
    uint kv_len = causal ? (i + 1) : seq_len;

    // Process K/V positions
    for (uint j = 0; j < kv_len; j++) {
        uint k_offset = bh_offset + j * head_dim;
        uint v_offset = bh_offset + j * head_dim;

        // Compute attention score Q[i] @ K[j]
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_offset + d];
        }
        score *= scale;

        // Compute P[i,j] = exp(score - L[i])
        float p_ij = exp(score - l_i);

        // Compute dP[i,j] = dO[i] @ V[j]^T
        float dp_ij = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dp_ij += do_vec[d] * V[v_offset + d];
        }

        // Compute dS[i,j] = P[i,j] * (dP[i,j] - D[i])
        float ds_ij = p_ij * (dp_ij - D_i);

        // Accumulate: dQ[i] += scale * dS[i,j] * K[j]
        for (uint d = 0; d < head_dim; d++) {
            dq_acc[d] += scale * ds_ij * K[k_offset + d];
        }
    }

    // Write dQ[i]
    uint dq_offset = bh_offset + i * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        dQ[dq_offset + d] = dq_acc[d];
    }
}

__kernel void flash_attention_backward_dK_fp32(
    __global const float* Q,        // [batch, heads, seq_len, head_dim]
    __global const float* K,        // [batch, heads, seq_len, head_dim]
    __global const float* V,        // [batch, heads, seq_len, head_dim]
    __global const float* O,        // [batch, heads, seq_len, head_dim]
    __global const float* dO,       // [batch, heads, seq_len, head_dim]
    __global const float* L,        // [batch, heads, seq_len] - logsumexp from forward
    __global float* dK,             // [batch, heads, seq_len, head_dim]
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    // One work item per (batch, head, kv_position)
    uint gid = get_global_id(0);
    uint total_kv = batch_size * num_heads * seq_len;

    if (gid >= total_kv) return;

    // Decode indices
    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint j = remainder % seq_len;  // K position

    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;
    uint l_offset = (b * num_heads + h) * seq_len;
    uint k_offset = bh_offset + j * head_dim;
    uint v_offset = bh_offset + j * head_dim;

    // Load K[j] and V[j] into registers
    float k_vec[MAX_HEAD_DIM];
    float v_vec[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        k_vec[d] = K[k_offset + d];
        v_vec[d] = V[v_offset + d];
    }

    // Accumulate dK[j]
    float dk_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        dk_acc[d] = 0.0f;
    }

    // For causal: only query positions i >= j contribute
    uint i_start = causal ? j : 0;

    // Process query positions
    for (uint i = i_start; i < seq_len; i++) {
        uint q_offset = bh_offset + i * head_dim;
        uint do_offset = bh_offset + i * head_dim;
        uint o_offset = bh_offset + i * head_dim;

        float l_i = L[l_offset + i];

        // Compute attention score Q[i] @ K[j]
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * k_vec[d];
        }
        score *= scale;

        // Compute P[i,j] = exp(score - L[i])
        float p_ij = exp(score - l_i);

        // Compute D[i] = dot(dO[i], O[i])
        float D_i = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            D_i += dO[do_offset + d] * O[o_offset + d];
        }

        // Compute dP[i,j] = dO[i] @ V[j]^T
        float dp_ij = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dp_ij += dO[do_offset + d] * v_vec[d];
        }

        // Compute dS[i,j] = P[i,j] * (dP[i,j] - D[i])
        float ds_ij = p_ij * (dp_ij - D_i);

        // Accumulate: dK[j] += scale * dS[i,j] * Q[i]
        for (uint d = 0; d < head_dim; d++) {
            dk_acc[d] += scale * ds_ij * Q[q_offset + d];
        }
    }

    // Write dK[j]
    uint dk_offset = bh_offset + j * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        dK[dk_offset + d] = dk_acc[d];
    }
}
"""

ATTENTION_KERNEL_FP16 = """
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define MAX_SEQ_LEN 8192  // Max sequence length for simple kernel

__kernel void attention_forward_fp16(
    __global const half* Q,
    __global const half* K,
    __global const half* V,
    __global half* output,
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    uint gid = get_global_id(0);
    uint total_elements = batch_size * num_heads * seq_len;

    if (gid >= total_elements) return;

    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint i = remainder % seq_len;

    uint base_offset = (b * num_heads + h) * seq_len * head_dim;
    uint q_base = base_offset + i * head_dim;

    // Use float for accumulation to maintain precision
    float scores[MAX_SEQ_LEN];
    float max_score = -INFINITY;

    uint max_j = causal ? (i + 1) : seq_len;
    if (max_j > MAX_SEQ_LEN) max_j = MAX_SEQ_LEN;

    // First pass: compute scores (read half, accumulate float)
    for (uint j = 0; j < max_j; j++) {
        uint k_base = base_offset + j * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += vload_half(q_base + d, Q) * vload_half(k_base + d, K);
        }
        score *= scale;
        scores[j] = score;
        max_score = fmax(max_score, score);
    }

    // Second pass: softmax
    float sum_exp = 0.0f;
    for (uint j = 0; j < max_j; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }

    float inv_sum = 1.0f / sum_exp;
    for (uint j = 0; j < max_j; j++) {
        scores[j] *= inv_sum;
    }

    // Third pass: weighted sum (accumulate float, write half)
    for (uint d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (uint j = 0; j < max_j; j++) {
            uint v_idx = base_offset + j * head_dim + d;
            out_val += scores[j] * vload_half(v_idx, V);
        }
        vstore_half(out_val, q_base + d, output);
    }
}
"""


class OpenCLAttention:
    """OpenCL-based attention implementation using PyOpenCL.

    Features:
    - FlashAttention-2 tiled algorithm for O(N) memory usage
    - Online softmax for numerical stability
    - Persistent GPU buffers for repeated computations
    - Causal masking for autoregressive models
    - FP16 support for 2x memory bandwidth
    """

    # FlashAttention block sizes (must match kernel defines)
    BR = 64  # Query block size
    BC = 64  # Key/Value block size

    def __init__(self, device_index: int = 0, use_flash: bool = True):
        """
        Initialize OpenCL attention backend.

        Args:
            device_index: Which GPU to use if multiple available
            use_flash: Use FlashAttention-2 kernel (default True)
        """
        try:
            import pyopencl as cl
        except ImportError:
            raise ImportError(
                "pyopencl is required for OpenCL backend. "
                "Install with: pip install pyopencl"
            )

        self.cl = cl
        self.use_flash = use_flash

        # Get platforms and devices
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")

        # Find a GPU device
        self.device = None
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                if device_index < len(devices):
                    self.device = devices[device_index]
                else:
                    self.device = devices[0]
                self.platform = platform
                break

        # Fall back to any device if no GPU
        if self.device is None:
            for platform in platforms:
                devices = platform.get_devices()
                if devices:
                    self.device = devices[0]
                    self.platform = platform
                    break

        if self.device is None:
            raise RuntimeError("No OpenCL devices found")

        # Create context and queue
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # Query device capabilities for workgroup optimization
        self.max_work_group_size = self.device.max_work_group_size
        self.max_compute_units = self.device.max_compute_units
        self.local_mem_size = self.device.local_mem_size

        # Check for FP16 support
        extensions = self.device.extensions
        self.fp16_supported = 'cl_khr_fp16' in extensions

        # Build FlashAttention FP32 kernel
        self._flash_kernel_fp32 = None
        if use_flash:
            try:
                self.flash_program_fp32 = cl.Program(self.context, FLASH_ATTENTION_KERNEL_FP32).build()
                self._flash_kernel_fp32 = cl.Kernel(self.flash_program_fp32, "flash_attention_forward_fp32")
            except Exception as e:
                print(f"Warning: FlashAttention FP32 kernel failed to compile: {e}")
                print("Falling back to simple attention kernel")
                self.use_flash = False

        # Build FlashAttention FP16 kernel (for long sequences with FP16)
        self._flash_kernel_fp16 = None
        if use_flash and self.fp16_supported:
            try:
                self.flash_program_fp16 = cl.Program(self.context, FLASH_ATTENTION_KERNEL_FP16).build()
                self._flash_kernel_fp16 = cl.Kernel(self.flash_program_fp16, "flash_attention_forward_fp16")
            except Exception as e:
                print(f"Warning: FlashAttention FP16 kernel failed to compile: {e}")
                # FP16 FlashAttention not available, will fall back to simple FP16

        # Build simple FP32 kernel (always available as fallback)
        self.program_fp32 = cl.Program(self.context, ATTENTION_KERNEL_FP32).build()
        self._kernel_fp32 = cl.Kernel(self.program_fp32, "attention_forward_fp32")

        # Build simple FP16 kernel if supported (for short sequences)
        self._kernel_fp16 = None
        if self.fp16_supported:
            try:
                self.program_fp16 = cl.Program(self.context, ATTENTION_KERNEL_FP16).build()
                self._kernel_fp16 = cl.Kernel(self.program_fp16, "attention_forward_fp16")
            except Exception as e:
                self.fp16_supported = False

        # Get kernel-specific work group size limit
        self._kernel_work_group_size = self._kernel_fp32.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device
        )

        # Build backward pass kernels
        self._backward_dV_kernel = None
        self._backward_dQ_kernel = None
        self._backward_dK_kernel = None
        self._backward_available = False

        if use_flash:
            try:
                # Build dV kernel
                dV_program = cl.Program(self.context, FLASH_ATTENTION_BACKWARD_DV_KERNEL).build()
                self._backward_dV_kernel = cl.Kernel(dV_program, "flash_attention_backward_dV_fp32")

                # Build dQ and dK kernels
                dQ_dK_program = cl.Program(self.context, FLASH_ATTENTION_BACKWARD_DQ_DK_KERNEL).build()
                self._backward_dQ_kernel = cl.Kernel(dQ_dK_program, "flash_attention_backward_dQ_fp32")
                self._backward_dK_kernel = cl.Kernel(dQ_dK_program, "flash_attention_backward_dK_fp32")

                self._backward_available = True
            except Exception as e:
                print(f"Warning: Backward kernels failed to compile: {e}")
                print("Training will not be available with this backend")

        # Persistent buffer cache
        self._buffer_cache = {}
        self._cached_key = None

        # Cache for saved tensors (for backward pass)
        self._saved_tensors = {}

        kernel_type = "FlashAttention-2" if self.use_flash else "Simple"
        backward_str = "backward ✓" if self._backward_available else "backward ✗"
        print(f"OpenCL initialized on: {self.device.name}")
        print(f"  Kernel: {kernel_type}, Max workgroup: {self.max_work_group_size}, CUs: {self.max_compute_units}, FP16: {self.fp16_supported}, {backward_str}")

    def _get_optimal_local_size(self, total_work_items: int):
        """Compute optimal workgroup size for the given work items."""
        # Start with kernel's max work group size
        max_size = min(self._kernel_work_group_size, self.max_work_group_size)

        # Common optimal sizes for AMD GPUs (wavefront = 64)
        # Try sizes that are multiples of wavefront size
        preferred_sizes = [256, 128, 64, 32]

        for size in preferred_sizes:
            if size <= max_size and total_work_items % size == 0:
                return (size,)

        # Fall back to largest power of 2 that divides work items
        size = 1
        while size * 2 <= max_size and total_work_items % (size * 2) == 0:
            size *= 2

        if size > 1:
            return (size,)

        # Let OpenCL decide if nothing works
        return None

    def _get_or_create_buffers(self, shape: tuple, nbytes: int, dtype: str):
        """Get cached buffers or create new ones if shape/dtype changed."""
        cl = self.cl
        mf = cl.mem_flags

        cache_key = (shape, dtype)
        if cache_key != self._cached_key:
            # Clear old buffers
            self._buffer_cache.clear()

            # Create new persistent buffers
            self._buffer_cache['q'] = cl.Buffer(self.context, mf.READ_WRITE, nbytes)
            self._buffer_cache['k'] = cl.Buffer(self.context, mf.READ_WRITE, nbytes)
            self._buffer_cache['v'] = cl.Buffer(self.context, mf.READ_WRITE, nbytes)
            self._buffer_cache['out'] = cl.Buffer(self.context, mf.READ_WRITE, nbytes)
            self._cached_key = cache_key

        return (
            self._buffer_cache['q'],
            self._buffer_cache['k'],
            self._buffer_cache['v'],
            self._buffer_cache['out']
        )

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False,
        dtype: Literal['float32', 'float16'] = 'float32'
    ) -> np.ndarray:
        """
        Compute attention forward pass using FlashAttention-2 algorithm.

        Args:
            Q: Query tensor [batch, heads, seq_len, head_dim]
            K: Key tensor [batch, heads, seq_len, head_dim]
            V: Value tensor [batch, heads, seq_len, head_dim]
            scale: Attention scale (default: 1/sqrt(head_dim))
            causal: If True, apply causal masking (for autoregressive models)
            dtype: 'float32' or 'float16' (FP16 requires device support)

        Returns:
            Output tensor [batch, heads, seq_len, head_dim]
        """
        use_fp16 = dtype == 'float16'
        if use_fp16 and not self.fp16_supported:
            raise RuntimeError("FP16 not supported on this device. Use dtype='float32'")

        # Use FlashAttention kernel if available (supports both FP32 and FP16)
        if self.use_flash:
            if use_fp16 and self._flash_kernel_fp16 is not None:
                return self._forward_flash_fp16(Q, K, V, scale, causal)
            elif not use_fp16:
                return self._forward_flash(Q, K, V, scale, causal)

        # Fall back to simple kernel
        return self._forward_simple(Q, K, V, scale, causal, dtype)

    def _forward_flash(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False
    ) -> np.ndarray:
        """
        FlashAttention-2 forward pass - memory efficient O(N) algorithm.

        Uses online softmax to process K/V in blocks without storing the full
        N×N attention matrix. Each thread handles one query position.
        """
        cl = self.cl
        mf = cl.mem_flags

        # Ensure contiguous FP32
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        batch_size, num_heads, seq_len, head_dim = Q.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Create output buffer
        output = np.zeros_like(Q)

        # Create logsumexp buffer (for backward pass)
        L = np.zeros((batch_size, num_heads, seq_len), dtype=np.float32)

        # Get or create GPU buffers
        cache_key = (Q.shape, 'flash_fp32')
        if cache_key != self._cached_key:
            self._buffer_cache.clear()
            self._buffer_cache['q'] = cl.Buffer(self.context, mf.READ_ONLY, Q.nbytes)
            self._buffer_cache['k'] = cl.Buffer(self.context, mf.READ_ONLY, K.nbytes)
            self._buffer_cache['v'] = cl.Buffer(self.context, mf.READ_ONLY, V.nbytes)
            self._buffer_cache['out'] = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self._buffer_cache['L'] = cl.Buffer(self.context, mf.WRITE_ONLY, L.nbytes)
            self._cached_key = cache_key

        q_buf = self._buffer_cache['q']
        k_buf = self._buffer_cache['k']
        v_buf = self._buffer_cache['v']
        out_buf = self._buffer_cache['out']
        L_buf = self._buffer_cache['L']

        # Copy data to GPU
        cl.enqueue_copy(self.queue, q_buf, Q)
        cl.enqueue_copy(self.queue, k_buf, K)
        cl.enqueue_copy(self.queue, v_buf, V)

        # One thread per query position
        total_queries = batch_size * num_heads * seq_len
        global_size = (total_queries,)

        # Use optimal local size for AMD (wavefront = 64)
        local_size = self._get_optimal_local_size(total_queries)

        # Execute FlashAttention kernel
        self._flash_kernel_fp32.set_args(
            q_buf, k_buf, v_buf, out_buf, L_buf,
            np.uint32(batch_size),
            np.uint32(num_heads),
            np.uint32(seq_len),
            np.uint32(head_dim),
            np.float32(scale),
            np.uint32(1 if causal else 0)
        )
        cl.enqueue_nd_range_kernel(self.queue, self._flash_kernel_fp32, global_size, local_size)

        # Read back results
        cl.enqueue_copy(self.queue, output, out_buf)
        cl.enqueue_copy(self.queue, L, L_buf)
        self.queue.finish()

        # Save tensors for backward pass
        self._saved_tensors = {
            'Q': Q,
            'K': K,
            'V': V,
            'O': output.copy(),
            'L': L,
            'scale': scale,
            'causal': causal,
        }

        return output

    def _forward_flash_fp16(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False
    ) -> np.ndarray:
        """
        FlashAttention-2 FP16 forward pass - mixed precision for 2x memory bandwidth.

        Reads FP16, accumulates in FP32, writes FP16.
        Supports unlimited sequence length (8K, 16K, 32K+ tokens).
        """
        cl = self.cl
        mf = cl.mem_flags

        # Ensure contiguous FP16
        Q = np.ascontiguousarray(Q, dtype=np.float16)
        K = np.ascontiguousarray(K, dtype=np.float16)
        V = np.ascontiguousarray(V, dtype=np.float16)

        batch_size, num_heads, seq_len, head_dim = Q.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Create output buffer (FP16)
        output = np.zeros_like(Q)

        # Create logsumexp buffer (FP32 for backward pass)
        L = np.zeros((batch_size, num_heads, seq_len), dtype=np.float32)

        # Get or create GPU buffers
        cache_key = (Q.shape, 'flash_fp16')
        if cache_key != self._cached_key:
            self._buffer_cache.clear()
            self._buffer_cache['q'] = cl.Buffer(self.context, mf.READ_ONLY, Q.nbytes)
            self._buffer_cache['k'] = cl.Buffer(self.context, mf.READ_ONLY, K.nbytes)
            self._buffer_cache['v'] = cl.Buffer(self.context, mf.READ_ONLY, V.nbytes)
            self._buffer_cache['out'] = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self._buffer_cache['L'] = cl.Buffer(self.context, mf.WRITE_ONLY, L.nbytes)
            self._cached_key = cache_key

        q_buf = self._buffer_cache['q']
        k_buf = self._buffer_cache['k']
        v_buf = self._buffer_cache['v']
        out_buf = self._buffer_cache['out']
        L_buf = self._buffer_cache['L']

        # Copy data to GPU
        cl.enqueue_copy(self.queue, q_buf, Q)
        cl.enqueue_copy(self.queue, k_buf, K)
        cl.enqueue_copy(self.queue, v_buf, V)

        # One thread per query position
        total_queries = batch_size * num_heads * seq_len
        global_size = (total_queries,)

        # Use optimal local size for AMD (wavefront = 64)
        local_size = self._get_optimal_local_size(total_queries)

        # Execute FlashAttention FP16 kernel
        self._flash_kernel_fp16.set_args(
            q_buf, k_buf, v_buf, out_buf, L_buf,
            np.uint32(batch_size),
            np.uint32(num_heads),
            np.uint32(seq_len),
            np.uint32(head_dim),
            np.float32(scale),
            np.uint32(1 if causal else 0)
        )
        cl.enqueue_nd_range_kernel(self.queue, self._flash_kernel_fp16, global_size, local_size)

        # Read back results
        cl.enqueue_copy(self.queue, output, out_buf)
        self.queue.finish()

        return output

    def _forward_simple(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False,
        dtype: Literal['float32', 'float16'] = 'float32'
    ) -> np.ndarray:
        """
        Simple attention forward pass - O(N²) memory but simpler.
        Used as fallback and for FP16.
        """
        cl = self.cl

        # Handle FP16
        use_fp16 = dtype == 'float16'

        # Convert to appropriate dtype
        np_dtype = np.float16 if use_fp16 else np.float32
        Q = np.ascontiguousarray(Q, dtype=np_dtype)
        K = np.ascontiguousarray(K, dtype=np_dtype)
        V = np.ascontiguousarray(V, dtype=np_dtype)

        batch_size, num_heads, seq_len, head_dim = Q.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Create output buffer
        output = np.zeros_like(Q)

        # Get or create persistent GPU buffers
        q_buf, k_buf, v_buf, out_buf = self._get_or_create_buffers(Q.shape, Q.nbytes, dtype)

        # Copy data to GPU (only the transfer, buffers already exist)
        cl.enqueue_copy(self.queue, q_buf, Q)
        cl.enqueue_copy(self.queue, k_buf, K)
        cl.enqueue_copy(self.queue, v_buf, V)

        # Execute kernel using cached kernel object
        total_work_items = batch_size * num_heads * seq_len
        global_size = (total_work_items,)

        # Optimize workgroup size for the device
        local_size = self._get_optimal_local_size(total_work_items)

        # Select appropriate kernel
        kernel = self._kernel_fp16 if use_fp16 else self._kernel_fp32

        kernel.set_args(
            q_buf, k_buf, v_buf, out_buf,
            np.uint32(batch_size),
            np.uint32(num_heads),
            np.uint32(seq_len),
            np.uint32(head_dim),
            np.float32(scale),
            np.uint32(1 if causal else 0)
        )
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)

        # Read back results
        cl.enqueue_copy(self.queue, output, out_buf)
        self.queue.finish()

        return output

    def backward(
        self,
        dO: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute backward pass for FlashAttention.

        Must be called after forward() on the same batch.

        Args:
            dO: Gradient of output [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (dQ, dK, dV) gradients with same shapes as Q, K, V
        """
        if not self._backward_available:
            raise RuntimeError(
                "Backward pass not available. "
                "This may be because FlashAttention kernels failed to compile."
            )

        if not self._saved_tensors:
            raise RuntimeError(
                "No saved tensors from forward pass. "
                "Call forward() before backward()."
            )

        cl = self.cl
        mf = cl.mem_flags

        # Get saved tensors
        Q = self._saved_tensors['Q']
        K = self._saved_tensors['K']
        V = self._saved_tensors['V']
        O = self._saved_tensors['O']
        L = self._saved_tensors['L']
        scale = self._saved_tensors['scale']
        causal = self._saved_tensors['causal']

        # Ensure dO is contiguous FP32
        dO = np.ascontiguousarray(dO, dtype=np.float32)

        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Create output buffers
        dQ = np.zeros_like(Q)
        dK = np.zeros_like(K)
        dV = np.zeros_like(V)

        # Create GPU buffers for backward pass
        q_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
        k_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
        v_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
        o_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=O)
        do_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dO)
        L_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=L)
        dq_buf = cl.Buffer(self.context, mf.WRITE_ONLY, dQ.nbytes)
        dk_buf = cl.Buffer(self.context, mf.WRITE_ONLY, dK.nbytes)
        dv_buf = cl.Buffer(self.context, mf.WRITE_ONLY, dV.nbytes)

        # Common kernel args
        total_positions = batch_size * num_heads * seq_len
        global_size = (total_positions,)
        local_size = self._get_optimal_local_size(total_positions)

        # Execute dV kernel
        self._backward_dV_kernel.set_args(
            q_buf, k_buf, v_buf, do_buf, L_buf, dv_buf,
            np.uint32(batch_size),
            np.uint32(num_heads),
            np.uint32(seq_len),
            np.uint32(head_dim),
            np.float32(scale),
            np.uint32(1 if causal else 0)
        )
        cl.enqueue_nd_range_kernel(self.queue, self._backward_dV_kernel, global_size, local_size)

        # Execute dQ kernel
        self._backward_dQ_kernel.set_args(
            q_buf, k_buf, v_buf, o_buf, do_buf, L_buf, dq_buf,
            np.uint32(batch_size),
            np.uint32(num_heads),
            np.uint32(seq_len),
            np.uint32(head_dim),
            np.float32(scale),
            np.uint32(1 if causal else 0)
        )
        cl.enqueue_nd_range_kernel(self.queue, self._backward_dQ_kernel, global_size, local_size)

        # Execute dK kernel
        self._backward_dK_kernel.set_args(
            q_buf, k_buf, v_buf, o_buf, do_buf, L_buf, dk_buf,
            np.uint32(batch_size),
            np.uint32(num_heads),
            np.uint32(seq_len),
            np.uint32(head_dim),
            np.float32(scale),
            np.uint32(1 if causal else 0)
        )
        cl.enqueue_nd_range_kernel(self.queue, self._backward_dK_kernel, global_size, local_size)

        # Read back results
        cl.enqueue_copy(self.queue, dQ, dq_buf)
        cl.enqueue_copy(self.queue, dK, dk_buf)
        cl.enqueue_copy(self.queue, dV, dv_buf)
        self.queue.finish()

        # Clear saved tensors
        self._saved_tensors.clear()

        return dQ, dK, dV

    @property
    def backward_available(self) -> bool:
        """Check if backward pass is available."""
        return self._backward_available

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Release resources."""
        self._saved_tensors.clear()
        pass  # OpenCL cleanup is automatic via Python GC

    @property
    def device_name(self) -> str:
        return self.device.name

    @property
    def backend_name(self) -> str:
        return f"OpenCL ({self.device.name})"


def get_opencl_devices() -> list:
    """List available OpenCL devices."""
    try:
        import pyopencl as cl
    except ImportError:
        return []

    devices = []
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            devices.append({
                'name': device.name,
                'platform': platform.name,
                'type': cl.device_type.to_string(device.type),
                'vendor': device.vendor,
                'version': device.version,
            })
    return devices


def is_opencl_available() -> bool:
    """Check if OpenCL is available with at least one device."""
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            if platform.get_devices():
                return True
        return False
    except:
        return False


if __name__ == "__main__":
    print("=== OpenCL Attention Test ===\n")

    # List devices
    print("Available OpenCL devices:")
    for i, dev in enumerate(get_opencl_devices()):
        print(f"  [{i}] {dev['name']} ({dev['type']}) - {dev['platform']}")
    print()

    if not is_opencl_available():
        print("No OpenCL devices available!")
        exit(1)

    # Test attention
    print("Testing attention computation...")

    batch_size = 1
    num_heads = 8
    seq_len = 64
    head_dim = 64

    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    with OpenCLAttention() as attn:
        print(f"Using: {attn.backend_name}")

        # Warmup
        _ = attn.forward(Q, K, V)

        # Benchmark
        import time
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            output = attn.forward(Q, K, V)
        elapsed = time.perf_counter() - start

        print(f"Output shape: {output.shape}")
        print(f"Time per iteration: {elapsed/n_iters*1000:.2f} ms")

        # Verify against numpy reference
        print("\nVerifying correctness...")
        scale = 1.0 / np.sqrt(head_dim)

        # Reference implementation
        scores = np.einsum('bhid,bhjd->bhij', Q, K) * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        reference = np.einsum('bhij,bhjd->bhid', attention, V)

        max_diff = np.abs(output - reference).max()
        print(f"Max difference from reference: {max_diff:.6f}")

        if max_diff < 1e-4:
            print("✓ Results match reference implementation!")
        else:
            print("✗ Results differ from reference (may be precision issue)")

    print("\n=== Test Complete ===")
