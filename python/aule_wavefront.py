#!/usr/bin/env python3
"""
aule-attention: Wavefront-Optimized Kernel for MI300X

This kernel is designed specifically for AMD CDNA3 (MI300X) architecture:
- 64-wide wavefronts (all 64 threads execute in lockstep)
- Subgroup operations for cross-lane communication (no barriers needed)
- Register tiling to maximize FMA/memory ratio
- Aggressive loop unrolling

Target: 2-5 TFLOPS on MI300X via Rusticl (up from 0.47 TFLOPS)

The key insight is that within a wavefront, we can use subgroup shuffle
operations which are essentially free (no sync needed). This lets us
distribute work across 64 lanes and reduce without barriers.
"""

import numpy as np
from typing import Optional, Dict, Any
import time

# =============================================================================
# Wavefront-Optimized Kernel
# =============================================================================
#
# Strategy: Each wavefront (64 threads) processes one query together
# - Distribute head_dim across lanes (64 threads, ~2 elements per thread for dim=128)
# - Use subgroup shuffle to reduce dot products (no barrier needed!)
# - Each thread computes partial scores, then shuffle-reduce
# - This avoids the expensive barrier-based reduction that killed the tiled kernel
#

WAVEFRONT_ATTENTION_KERNEL = """
// Optimized FlashAttention for AMD MI300X via Rusticl
//
// Key insight: The baseline vec4 kernel is already good because it:
// - Has simple control flow (one thread per query)
// - Uses vec4 for coalesced memory access
// - Avoids register spilling by processing one K/V at a time
//
// New strategy: Specialize for head_dim=128 (LLaMA) with:
// - Fully unrolled dot product (no inner loop)
// - Precomputed Q in registers
// - Minimal branching

#define NEG_INF -1e30f

// Specialized kernel for head_dim=128 (LLaMA-7B, LLaMA-13B, etc)
// Uses vec8 to reduce register pressure while maintaining vectorization
// Q is loaded once, K/V loaded per iteration
__kernel void flash_attention_hd128_fp32(
    __global const float8* restrict Q8,   // [batch, heads, seq_len, 16] (128/8=16 vec8)
    __global const float8* restrict K8,
    __global const float8* restrict V8,
    __global float8* restrict O8,
    __global float* restrict L,
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const float scale,
    const uint causal
) {
    uint gid = get_global_id(0);
    uint total = batch_size * num_heads * seq_len;
    if (gid >= total) return;

    uint b = gid / (num_heads * seq_len);
    uint rem = gid % (num_heads * seq_len);
    uint h = rem / seq_len;
    uint i = rem % seq_len;

    // head_dim=128 means 16 float8 vectors
    uint bh = (b * num_heads + h) * seq_len * 16;
    uint q_base = bh + i * 16;

    // Load Q as 16 float8s (128 floats total, uses 128 registers)
    float8 q0 = Q8[q_base + 0];
    float8 q1 = Q8[q_base + 1];
    float8 q2 = Q8[q_base + 2];
    float8 q3 = Q8[q_base + 3];
    float8 q4 = Q8[q_base + 4];
    float8 q5 = Q8[q_base + 5];
    float8 q6 = Q8[q_base + 6];
    float8 q7 = Q8[q_base + 7];
    float8 q8 = Q8[q_base + 8];
    float8 q9 = Q8[q_base + 9];
    float8 q10 = Q8[q_base + 10];
    float8 q11 = Q8[q_base + 11];
    float8 q12 = Q8[q_base + 12];
    float8 q13 = Q8[q_base + 13];
    float8 q14 = Q8[q_base + 14];
    float8 q15 = Q8[q_base + 15];

    float m_i = NEG_INF;
    float l_i = 0.0f;

    // Output accumulators (16 float8s = 128 registers)
    float8 o0 = (float8)(0);
    float8 o1 = (float8)(0);
    float8 o2 = (float8)(0);
    float8 o3 = (float8)(0);
    float8 o4 = (float8)(0);
    float8 o5 = (float8)(0);
    float8 o6 = (float8)(0);
    float8 o7 = (float8)(0);
    float8 o8 = (float8)(0);
    float8 o9 = (float8)(0);
    float8 o10 = (float8)(0);
    float8 o11 = (float8)(0);
    float8 o12 = (float8)(0);
    float8 o13 = (float8)(0);
    float8 o14 = (float8)(0);
    float8 o15 = (float8)(0);

    uint kv_len = causal ? (i + 1) : seq_len;

    for (uint j = 0; j < kv_len; j++) {
        uint k_base = bh + j * 16;

        // Load K and compute dot product using dot() for pairs
        float8 k0 = K8[k_base + 0];
        float8 k1 = K8[k_base + 1];
        float8 k2 = K8[k_base + 2];
        float8 k3 = K8[k_base + 3];
        float8 k4 = K8[k_base + 4];
        float8 k5 = K8[k_base + 5];
        float8 k6 = K8[k_base + 6];
        float8 k7 = K8[k_base + 7];
        float8 k8 = K8[k_base + 8];
        float8 k9 = K8[k_base + 9];
        float8 k10 = K8[k_base + 10];
        float8 k11 = K8[k_base + 11];
        float8 k12 = K8[k_base + 12];
        float8 k13 = K8[k_base + 13];
        float8 k14 = K8[k_base + 14];
        float8 k15 = K8[k_base + 15];

        // Compute dot product
        float8 p0 = q0 * k0;
        float8 p1 = q1 * k1;
        float8 p2 = q2 * k2;
        float8 p3 = q3 * k3;
        float8 p4 = q4 * k4;
        float8 p5 = q5 * k5;
        float8 p6 = q6 * k6;
        float8 p7 = q7 * k7;
        float8 p8 = q8 * k8;
        float8 p9 = q9 * k9;
        float8 p10 = q10 * k10;
        float8 p11 = q11 * k11;
        float8 p12 = q12 * k12;
        float8 p13 = q13 * k13;
        float8 p14 = q14 * k14;
        float8 p15 = q15 * k15;

        // Reduce
        float8 s01 = p0 + p1;
        float8 s23 = p2 + p3;
        float8 s45 = p4 + p5;
        float8 s67 = p6 + p7;
        float8 s89 = p8 + p9;
        float8 s1011 = p10 + p11;
        float8 s1213 = p12 + p13;
        float8 s1415 = p14 + p15;

        float8 t0 = s01 + s23;
        float8 t1 = s45 + s67;
        float8 t2 = s89 + s1011;
        float8 t3 = s1213 + s1415;

        float8 u0 = t0 + t1;
        float8 u1 = t2 + t3;

        float8 final = u0 + u1;
        float score = (final.s0 + final.s1 + final.s2 + final.s3 +
                       final.s4 + final.s5 + final.s6 + final.s7) * scale;

        // Online softmax
        float m_new = fmax(m_i, score);
        float corr = exp(m_i - m_new);
        float p_attn = exp(score - m_new);

        l_i = l_i * corr + p_attn;

        // Load V and accumulate
        o0 = o0 * corr + p_attn * V8[k_base + 0];
        o1 = o1 * corr + p_attn * V8[k_base + 1];
        o2 = o2 * corr + p_attn * V8[k_base + 2];
        o3 = o3 * corr + p_attn * V8[k_base + 3];
        o4 = o4 * corr + p_attn * V8[k_base + 4];
        o5 = o5 * corr + p_attn * V8[k_base + 5];
        o6 = o6 * corr + p_attn * V8[k_base + 6];
        o7 = o7 * corr + p_attn * V8[k_base + 7];
        o8 = o8 * corr + p_attn * V8[k_base + 8];
        o9 = o9 * corr + p_attn * V8[k_base + 9];
        o10 = o10 * corr + p_attn * V8[k_base + 10];
        o11 = o11 * corr + p_attn * V8[k_base + 11];
        o12 = o12 * corr + p_attn * V8[k_base + 12];
        o13 = o13 * corr + p_attn * V8[k_base + 13];
        o14 = o14 * corr + p_attn * V8[k_base + 14];
        o15 = o15 * corr + p_attn * V8[k_base + 15];

        m_i = m_new;
    }

    // Normalize and store
    float inv_l = 1.0f / l_i;
    O8[q_base + 0] = o0 * inv_l;
    O8[q_base + 1] = o1 * inv_l;
    O8[q_base + 2] = o2 * inv_l;
    O8[q_base + 3] = o3 * inv_l;
    O8[q_base + 4] = o4 * inv_l;
    O8[q_base + 5] = o5 * inv_l;
    O8[q_base + 6] = o6 * inv_l;
    O8[q_base + 7] = o7 * inv_l;
    O8[q_base + 8] = o8 * inv_l;
    O8[q_base + 9] = o9 * inv_l;
    O8[q_base + 10] = o10 * inv_l;
    O8[q_base + 11] = o11 * inv_l;
    O8[q_base + 12] = o12 * inv_l;
    O8[q_base + 13] = o13 * inv_l;
    O8[q_base + 14] = o14 * inv_l;
    O8[q_base + 15] = o15 * inv_l;

    if (L != 0) {
        L[(b * num_heads + h) * seq_len + i] = m_i + log(l_i);
    }
}

// Generic tiled kernel (fallback)
#define TILE_KV 4

__kernel void flash_attention_tiled_fp32(
    __global const float* restrict Q,
    __global const float* restrict K,
    __global const float* restrict V,
    __global float* restrict O,
    __global float* restrict L,
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    uint gid = get_global_id(0);
    uint total = batch_size * num_heads * seq_len;
    if (gid >= total) return;

    uint b = gid / (num_heads * seq_len);
    uint rem = gid % (num_heads * seq_len);
    uint h = rem / seq_len;
    uint i = rem % seq_len;

    uint bh = (b * num_heads + h) * seq_len * head_dim;
    uint q_base = bh + i * head_dim;

    // Load Q into registers - unroll for common head_dims
    float q[128];  // Support up to 128 head_dim
    for (uint d = 0; d < head_dim; d++) {
        q[d] = Q[q_base + d];
    }

    float m_i = NEG_INF;
    float l_i = 0.0f;
    float o[128];
    for (uint d = 0; d < head_dim; d++) {
        o[d] = 0.0f;
    }

    uint kv_len = causal ? (i + 1) : seq_len;

    // Process 4 KV positions at a time
    uint j = 0;
    for (; j + TILE_KV <= kv_len; j += TILE_KV) {
        float scores[TILE_KV];
        float tile_max = NEG_INF;

        // Compute 4 scores
        #pragma unroll
        for (uint t = 0; t < TILE_KV; t++) {
            uint k_base = bh + (j + t) * head_dim;
            float s = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                s += q[d] * K[k_base + d];
            }
            scores[t] = s * scale;
            tile_max = fmax(tile_max, scores[t]);
        }

        // Online softmax update
        float m_new = fmax(m_i, tile_max);
        float corr = exp(m_i - m_new);
        l_i *= corr;
        for (uint d = 0; d < head_dim; d++) {
            o[d] *= corr;
        }

        // Accumulate V
        #pragma unroll
        for (uint t = 0; t < TILE_KV; t++) {
            float p = exp(scores[t] - m_new);
            l_i += p;
            uint v_base = bh + (j + t) * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                o[d] += p * V[v_base + d];
            }
        }
        m_i = m_new;
    }

    // Handle remainder
    for (; j < kv_len; j++) {
        uint k_base = bh + j * head_dim;
        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            s += q[d] * K[k_base + d];
        }
        s *= scale;

        float m_new = fmax(m_i, s);
        float corr = exp(m_i - m_new);
        float p = exp(s - m_new);
        l_i = l_i * corr + p;

        for (uint d = 0; d < head_dim; d++) {
            o[d] = o[d] * corr + p * V[bh + j * head_dim + d];
        }
        m_i = m_new;
    }

    // Normalize and store
    float inv_l = 1.0f / l_i;
    for (uint d = 0; d < head_dim; d++) {
        O[q_base + d] = o[d] * inv_l;
    }
    if (L != 0) {
        L[(b * num_heads + h) * seq_len + i] = m_i + log(l_i);
    }
}

// Vec4 kernel with tiling - combines vectorization + register tiling
__kernel void flash_attention_vec4_tiled_fp32(
    __global const float4* restrict Q4,
    __global const float4* restrict K4,
    __global const float4* restrict V4,
    __global float4* restrict O4,
    __global float* restrict L,
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    uint gid = get_global_id(0);
    uint total = batch_size * num_heads * seq_len;
    if (gid >= total) return;

    uint b = gid / (num_heads * seq_len);
    uint rem = gid % (num_heads * seq_len);
    uint h = rem / seq_len;
    uint i = rem % seq_len;

    uint head_dim4 = head_dim / 4;
    uint bh4 = (b * num_heads + h) * seq_len * head_dim4;
    uint q_base4 = bh4 + i * head_dim4;

    // Load Q as vec4
    float4 q[32];  // Support head_dim up to 128
    for (uint d = 0; d < head_dim4; d++) {
        q[d] = Q4[q_base4 + d];
    }

    float m_i = NEG_INF;
    float l_i = 0.0f;
    float4 o[32];
    for (uint d = 0; d < head_dim4; d++) {
        o[d] = (float4)(0.0f);
    }

    uint kv_len = causal ? (i + 1) : seq_len;

    // Process 4 KV at a time
    uint j = 0;
    for (; j + 4 <= kv_len; j += 4) {
        float scores[4];
        float tile_max = NEG_INF;

        // Compute 4 scores with vec4 dot products
        #pragma unroll
        for (uint t = 0; t < 4; t++) {
            uint k_base4 = bh4 + (j + t) * head_dim4;
            float s = 0.0f;
            for (uint d = 0; d < head_dim4; d++) {
                float4 kv = K4[k_base4 + d];
                float4 prod = q[d] * kv;
                s += prod.x + prod.y + prod.z + prod.w;
            }
            scores[t] = s * scale;
            tile_max = fmax(tile_max, scores[t]);
        }

        float m_new = fmax(m_i, tile_max);
        float corr = exp(m_i - m_new);
        l_i *= corr;
        for (uint d = 0; d < head_dim4; d++) {
            o[d] *= corr;
        }

        #pragma unroll
        for (uint t = 0; t < 4; t++) {
            float p = exp(scores[t] - m_new);
            l_i += p;
            uint v_base4 = bh4 + (j + t) * head_dim4;
            for (uint d = 0; d < head_dim4; d++) {
                o[d] += p * V4[v_base4 + d];
            }
        }
        m_i = m_new;
    }

    // Remainder
    for (; j < kv_len; j++) {
        uint k_base4 = bh4 + j * head_dim4;
        float s = 0.0f;
        for (uint d = 0; d < head_dim4; d++) {
            float4 kv = K4[k_base4 + d];
            float4 prod = q[d] * kv;
            s += prod.x + prod.y + prod.z + prod.w;
        }
        s *= scale;

        float m_new = fmax(m_i, s);
        float corr = exp(m_i - m_new);
        float p = exp(s - m_new);
        l_i = l_i * corr + p;

        for (uint d = 0; d < head_dim4; d++) {
            o[d] = o[d] * corr + p * V4[bh4 + j * head_dim4 + d];
        }
        m_i = m_new;
    }

    float inv_l = 1.0f / l_i;
    for (uint d = 0; d < head_dim4; d++) {
        O4[q_base4 + d] = o[d] * inv_l;
    }
    if (L != 0) {
        L[(b * num_heads + h) * seq_len + i] = m_i + log(l_i);
    }
}

// Vec8 kernel with tiling
__kernel void flash_attention_vec8_tiled_fp32(
    __global const float8* restrict Q8,
    __global const float8* restrict K8,
    __global const float8* restrict V8,
    __global float8* restrict O8,
    __global float* restrict L,
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    uint gid = get_global_id(0);
    uint total = batch_size * num_heads * seq_len;
    if (gid >= total) return;

    uint b = gid / (num_heads * seq_len);
    uint rem = gid % (num_heads * seq_len);
    uint h = rem / seq_len;
    uint i = rem % seq_len;

    uint head_dim8 = head_dim / 8;
    uint bh8 = (b * num_heads + h) * seq_len * head_dim8;
    uint q_base8 = bh8 + i * head_dim8;

    // Load Q as vec8
    float8 q[16];  // Support head_dim up to 128
    for (uint d = 0; d < head_dim8; d++) {
        q[d] = Q8[q_base8 + d];
    }

    float m_i = NEG_INF;
    float l_i = 0.0f;
    float8 o[16];
    for (uint d = 0; d < head_dim8; d++) {
        o[d] = (float8)(0.0f);
    }

    uint kv_len = causal ? (i + 1) : seq_len;

    // Process 4 KV at a time
    uint j = 0;
    for (; j + 4 <= kv_len; j += 4) {
        float scores[4];
        float tile_max = NEG_INF;

        #pragma unroll
        for (uint t = 0; t < 4; t++) {
            uint k_base8 = bh8 + (j + t) * head_dim8;
            float s = 0.0f;
            for (uint d = 0; d < head_dim8; d++) {
                float8 kv = K8[k_base8 + d];
                float8 prod = q[d] * kv;
                s += prod.s0 + prod.s1 + prod.s2 + prod.s3 +
                     prod.s4 + prod.s5 + prod.s6 + prod.s7;
            }
            scores[t] = s * scale;
            tile_max = fmax(tile_max, scores[t]);
        }

        float m_new = fmax(m_i, tile_max);
        float corr = exp(m_i - m_new);
        l_i *= corr;
        for (uint d = 0; d < head_dim8; d++) {
            o[d] *= corr;
        }

        #pragma unroll
        for (uint t = 0; t < 4; t++) {
            float p = exp(scores[t] - m_new);
            l_i += p;
            uint v_base8 = bh8 + (j + t) * head_dim8;
            for (uint d = 0; d < head_dim8; d++) {
                o[d] += p * V8[v_base8 + d];
            }
        }
        m_i = m_new;
    }

    // Remainder
    for (; j < kv_len; j++) {
        uint k_base8 = bh8 + j * head_dim8;
        float s = 0.0f;
        for (uint d = 0; d < head_dim8; d++) {
            float8 kv = K8[k_base8 + d];
            float8 prod = q[d] * kv;
            s += prod.s0 + prod.s1 + prod.s2 + prod.s3 +
                 prod.s4 + prod.s5 + prod.s6 + prod.s7;
        }
        s *= scale;

        float m_new = fmax(m_i, s);
        float corr = exp(m_i - m_new);
        float p = exp(s - m_new);
        l_i = l_i * corr + p;

        for (uint d = 0; d < head_dim8; d++) {
            o[d] = o[d] * corr + p * V8[bh8 + j * head_dim8 + d];
        }
        m_i = m_new;
    }

    float inv_l = 1.0f / l_i;
    for (uint d = 0; d < head_dim8; d++) {
        O8[q_base8 + d] = o[d] * inv_l;
    }
    if (L != 0) {
        L[(b * num_heads + h) * seq_len + i] = m_i + log(l_i);
    }
}
"""


class WavefrontAttention:
    """
    Optimized attention for AMD MI300X via Rusticl.

    Uses register tiling and vectorization to maximize throughput.
    """

    def __init__(self, device_index: int = 0, verbose: bool = True):
        self.verbose = verbose
        self.device_index = device_index
        self._init_opencl()

    def _init_opencl(self):
        import pyopencl as cl

        self._cl = cl

        # Find GPU
        platforms = cl.get_platforms()
        device = None

        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                device = devices[min(self.device_index, len(devices) - 1)]
                break

        if device is None:
            raise RuntimeError("No GPU found")

        self._device = device
        self._context = cl.Context([device])
        self._queue = cl.CommandQueue(
            self._context,
            properties=cl.command_queue_properties.PROFILING_ENABLE
        )

        if self.verbose:
            print(f"Device: {device.name}")
            print(f"  Compute units: {device.max_compute_units}")
            print(f"  Max workgroup: {device.max_work_group_size}")

        # Build kernels
        self._build_kernels()

        # Buffer cache
        self._buffers = {}
        self._cache_key = None

    def _build_kernels(self):
        cl = self._cl

        self._kernel_hd128 = None
        self._kernel_tiled = None
        self._kernel_vec4 = None
        self._kernel_vec8 = None

        build_opts = ["-cl-mad-enable"]

        # Try specialized hd128 kernel (for LLaMA)
        try:
            program = cl.Program(self._context, WAVEFRONT_ATTENTION_KERNEL).build(options=build_opts)
            self._kernel_hd128 = cl.Kernel(program, "flash_attention_hd128_fp32")
            if self.verbose:
                print("  Built hd128 kernel (specialized for head_dim=128)")
        except Exception as e:
            if self.verbose:
                print(f"  HD128 kernel failed: {e}")

        # Try scalar tiled kernel
        try:
            program = cl.Program(self._context, WAVEFRONT_ATTENTION_KERNEL).build(options=build_opts)
            self._kernel_tiled = cl.Kernel(program, "flash_attention_tiled_fp32")
            if self.verbose:
                print("  Built tiled kernel (scalar)")
        except Exception as e:
            if self.verbose:
                print(f"  Tiled kernel failed: {e}")

        # Try vec4 tiled kernel
        try:
            program = cl.Program(self._context, WAVEFRONT_ATTENTION_KERNEL).build(options=build_opts)
            self._kernel_vec4 = cl.Kernel(program, "flash_attention_vec4_tiled_fp32")
            if self.verbose:
                print("  Built vec4 tiled kernel")
        except Exception as e:
            if self.verbose:
                print(f"  Vec4 kernel failed: {e}")

        # Try vec8 tiled kernel
        try:
            program = cl.Program(self._context, WAVEFRONT_ATTENTION_KERNEL).build(options=build_opts)
            self._kernel_vec8 = cl.Kernel(program, "flash_attention_vec8_tiled_fp32")
            if self.verbose:
                print("  Built vec8 tiled kernel")
        except Exception as e:
            if self.verbose:
                print(f"  Vec8 kernel failed: {e}")

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False,
        kernel: str = 'auto'
    ) -> np.ndarray:
        """
        Compute attention forward pass.

        Args:
            kernel: 'auto', 'tiled', 'vec4', or 'vec8'
        """
        cl = self._cl
        mf = cl.mem_flags

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        batch_size, num_heads, seq_len, head_dim = Q.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        output = np.zeros_like(Q)
        L = np.zeros((batch_size, num_heads, seq_len), dtype=np.float32)

        # Select kernel
        if kernel == 'auto':
            if head_dim == 128 and self._kernel_hd128 is not None:
                kernel = 'hd128'
            elif head_dim % 8 == 0 and self._kernel_vec8 is not None:
                kernel = 'vec8'
            elif head_dim % 4 == 0 and self._kernel_vec4 is not None:
                kernel = 'vec4'
            elif self._kernel_tiled is not None:
                kernel = 'tiled'
            else:
                raise RuntimeError("No kernel available")

        # Create buffers
        cache_key = (Q.shape, kernel)
        if cache_key != self._cache_key:
            self._buffers.clear()
            self._buffers['q'] = cl.Buffer(self._context, mf.READ_ONLY, Q.nbytes)
            self._buffers['k'] = cl.Buffer(self._context, mf.READ_ONLY, K.nbytes)
            self._buffers['v'] = cl.Buffer(self._context, mf.READ_ONLY, V.nbytes)
            self._buffers['out'] = cl.Buffer(self._context, mf.WRITE_ONLY, output.nbytes)
            self._buffers['L'] = cl.Buffer(self._context, mf.WRITE_ONLY, L.nbytes)
            self._cache_key = cache_key

        # Upload
        cl.enqueue_copy(self._queue, self._buffers['q'], Q)
        cl.enqueue_copy(self._queue, self._buffers['k'], K)
        cl.enqueue_copy(self._queue, self._buffers['v'], V)

        total_queries = batch_size * num_heads * seq_len

        if kernel == 'hd128' and self._kernel_hd128 is not None:
            k = self._kernel_hd128
            k.set_args(
                self._buffers['q'], self._buffers['k'], self._buffers['v'],
                self._buffers['out'], self._buffers['L'],
                np.uint32(batch_size), np.uint32(num_heads),
                np.uint32(seq_len),
                np.float32(scale), np.uint32(1 if causal else 0)
            )
            global_size = (total_queries,)
            local_size = None
            cl.enqueue_nd_range_kernel(self._queue, k, global_size, local_size)

        elif kernel == 'tiled' and self._kernel_tiled is not None:
            k = self._kernel_tiled
            k.set_args(
                self._buffers['q'], self._buffers['k'], self._buffers['v'],
                self._buffers['out'], self._buffers['L'],
                np.uint32(batch_size), np.uint32(num_heads),
                np.uint32(seq_len), np.uint32(head_dim),
                np.float32(scale), np.uint32(1 if causal else 0)
            )
            global_size = (total_queries,)
            local_size = None
            cl.enqueue_nd_range_kernel(self._queue, k, global_size, local_size)

        elif kernel == 'vec4' and self._kernel_vec4 is not None:
            k = self._kernel_vec4
            k.set_args(
                self._buffers['q'], self._buffers['k'], self._buffers['v'],
                self._buffers['out'], self._buffers['L'],
                np.uint32(batch_size), np.uint32(num_heads),
                np.uint32(seq_len), np.uint32(head_dim),
                np.float32(scale), np.uint32(1 if causal else 0)
            )
            global_size = (total_queries,)
            local_size = None
            cl.enqueue_nd_range_kernel(self._queue, k, global_size, local_size)

        elif kernel == 'vec8' and self._kernel_vec8 is not None:
            k = self._kernel_vec8
            k.set_args(
                self._buffers['q'], self._buffers['k'], self._buffers['v'],
                self._buffers['out'], self._buffers['L'],
                np.uint32(batch_size), np.uint32(num_heads),
                np.uint32(seq_len), np.uint32(head_dim),
                np.float32(scale), np.uint32(1 if causal else 0)
            )
            global_size = (total_queries,)
            local_size = None
            cl.enqueue_nd_range_kernel(self._queue, k, global_size, local_size)
        else:
            raise RuntimeError(f"Kernel '{kernel}' not available")

        # Download
        cl.enqueue_copy(self._queue, output, self._buffers['out'])
        self._queue.finish()

        return output

    def close(self):
        self._buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def benchmark_wavefront():
    """Benchmark the optimized kernels."""
    print("=" * 70)
    print("TILED KERNEL BENCHMARK")
    print("=" * 70)

    configs = [
        {"name": "Small", "batch": 1, "heads": 8, "seq": 512, "dim": 64},
        {"name": "TinyLlama", "batch": 1, "heads": 32, "seq": 512, "dim": 64},
        {"name": "LLaMA-7B", "batch": 1, "heads": 32, "seq": 2048, "dim": 128},
    ]

    kernels_to_test = ['tiled', 'vec4', 'vec8']  # Skip hd128 - too many registers

    with WavefrontAttention(verbose=True) as attn:
        for cfg in configs:
            print(f"\n--- {cfg['name']} [{cfg['batch']}, {cfg['heads']}, {cfg['seq']}, {cfg['dim']}] ---")

            Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

            for kernel in kernels_to_test:
                try:
                    # Warmup
                    _ = attn.forward(Q, K, V, causal=True, kernel=kernel)

                    # Benchmark
                    n_iters = 5
                    times = []
                    for _ in range(n_iters):
                        start = time.perf_counter()
                        output = attn.forward(Q, K, V, causal=True, kernel=kernel)
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)

                    avg_ms = np.mean(times) * 1000
                    flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
                    tflops = flops / (np.mean(times) * 1e12)

                    print(f"  {kernel:12s}: {avg_ms:8.2f}ms, {tflops:.4f} TFLOPS")

                except Exception as e:
                    print(f"  {kernel:12s}: FAILED - {e}")

    # Compare with baseline (vec4 from hybrid)
    print("\n--- Baseline Comparison (vec4 from HybridAttention) ---")
    try:
        from aule_hybrid import HybridAttention

        cfg = {"batch": 1, "heads": 32, "seq": 2048, "dim": 128}
        Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
        K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
        V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

        with HybridAttention(backend='opencl', verbose=False) as baseline:
            # Warmup
            _ = baseline.forward(Q, K, V, causal=True)

            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = baseline.forward(Q, K, V, causal=True)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_ms = np.mean(times) * 1000
            flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
            tflops = flops / (np.mean(times) * 1e12)

            print(f"  vec4 baseline: {avg_ms:8.2f}ms, {tflops:.4f} TFLOPS")

    except Exception as e:
        print(f"  Baseline comparison failed: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_wavefront()
