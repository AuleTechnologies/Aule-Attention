#!/usr/bin/env python3
"""
aule-attention: LDS-Tiled OpenCL Backend

This module implements FlashAttention-2 with proper LDS (Local Data Share) tiling
optimized for AMD GPUs via Mesa Rusticl.

Key optimizations:
1. K/V tiles loaded into LDS (10x faster than global memory)
2. Each workgroup processes one Q tile (Br queries)
3. Threads within workgroup share K/V data via LDS
4. Reduced global memory traffic via tiling
5. No excessive barriers - minimal synchronization

Target: 5-15 TFLOPS on MI300X (up from 0.47 TFLOPS with naive kernel)

Based on FlashAttention-2 algorithm and AMD optimization guides.
"""

import numpy as np
from typing import Optional
import time

# =============================================================================
# LDS-Tiled FlashAttention Kernel
# =============================================================================
#
# Tiling strategy:
# - Q: tiles of Br × d (loaded to registers)
# - K/V: tiles of Bc × d (loaded to LDS, shared across threads)
# - Each workgroup processes one Q tile
# - Within workgroup: threads cooperate on K/V blocks
#
# Memory hierarchy:
# - Q: registers (per-thread)
# - K/V: LDS (shared within workgroup)
# - Output: registers -> global memory
#
# MI300X specs:
# - 64 KB LDS per CU (32 KB usable per workgroup)
# - 64-wide wavefronts
# - 128B L1 cache lines

LDS_TILED_FLASH_ATTENTION_KERNEL = """
// FlashAttention-2 with LDS Tiling for AMD MI300X
// Each workgroup handles Br queries, loads K/V tiles to LDS
//
// Tiling parameters tuned for MI300X:
// - Br = 4 queries per workgroup (limited by register pressure)
// - Bc = 64 K/V positions per tile
// - Workgroup size = 64 (one AMD wavefront)
// - LDS usage: Bc * head_dim * 4 bytes * 2 (K+V) = 64 * 128 * 4 * 2 = 64KB

#define BR 4           // Queries per workgroup
#define BC 64          // K/V positions per LDS tile
#define WG_SIZE 64     // Threads per workgroup (AMD wavefront)
#define MAX_DIM 128    // Maximum head dimension

__kernel void flash_attention_tiled_fp32(
    __global const float* restrict Q,     // [batch, heads, seq_len, head_dim]
    __global const float* restrict K,     // [batch, heads, seq_len, head_dim]
    __global const float* restrict V,     // [batch, heads, seq_len, head_dim]
    __global float* restrict O,           // [batch, heads, seq_len, head_dim]
    __global float* restrict L,           // [batch, heads, seq_len] logsumexp
    const uint batch_size,
    const uint num_heads,
    const uint seq_len,
    const uint head_dim,
    const float scale,
    const uint causal
) {
    // LDS arrays - ALL must be declared at kernel scope for OpenCL/Rusticl compliance
    // K tile: BC * head_dim floats
    // V tile: BC * head_dim floats
    __local float lds_k[BC * MAX_DIM];
    __local float lds_v[BC * MAX_DIM];
    __local float lds_reduction[WG_SIZE];  // For parallel reduction
    __local float lds_shared;               // For broadcasting values

    // Workgroup and thread indices
    uint wg_id = get_group_id(0);           // Which Q tile (batch*heads*ceil(seq/BR))
    uint lid = get_local_id(0);             // Thread within workgroup [0, WG_SIZE)

    // Decode which batch/head/Q-tile this workgroup handles
    uint q_tiles_per_head = (seq_len + BR - 1) / BR;
    uint total_q_tiles = batch_size * num_heads * q_tiles_per_head;

    if (wg_id >= total_q_tiles) return;

    uint b = wg_id / (num_heads * q_tiles_per_head);
    uint remainder = wg_id % (num_heads * q_tiles_per_head);
    uint h = remainder / q_tiles_per_head;
    uint q_tile = remainder % q_tiles_per_head;

    uint q_start = q_tile * BR;
    uint q_end = min(q_start + BR, seq_len);

    // Base offset for this batch/head
    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;

    // Each thread handles a subset of the output dimensions
    // With WG_SIZE=64 and head_dim=128, each thread handles 2 dims
    uint dims_per_thread = (head_dim + WG_SIZE - 1) / WG_SIZE;

    // Per-query accumulators (each thread handles part of each query)
    // Online softmax state: max, sum, output accumulator
    float m[BR];           // Running max per query
    float l[BR];           // Running sum per query
    float o[BR * 4];       // Output accumulator (dims_per_thread values per query)

    // Initialize accumulators
    for (uint qi = 0; qi < BR; qi++) {
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
        for (uint e = 0; e < dims_per_thread; e++) {
            o[qi * dims_per_thread + e] = 0.0f;
        }
    }

    // Load Q values for this thread's dimensions into registers
    float q_reg[BR * 4];  // BR queries × dims_per_thread dimensions
    for (uint qi = 0; qi < BR && (q_start + qi) < seq_len; qi++) {
        uint q_pos = q_start + qi;
        for (uint e = 0; e < dims_per_thread && (lid * dims_per_thread + e) < head_dim; e++) {
            uint d = lid * dims_per_thread + e;
            q_reg[qi * dims_per_thread + e] = Q[bh_offset + q_pos * head_dim + d];
        }
    }

    // Determine K/V range (causal: each query can only attend to positions <= itself)
    uint kv_max = causal ? (q_end) : seq_len;  // Max position any query in tile can attend
    uint num_kv_tiles = (kv_max + BC - 1) / BC;

    // Process K/V in tiles
    for (uint kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint k_start = kv_tile * BC;
        uint k_end = min(k_start + BC, kv_max);
        uint tile_len = k_end - k_start;

        // Cooperatively load K tile to LDS
        // Each thread loads head_dim/WG_SIZE elements per K position
        for (uint kp = 0; kp < tile_len; kp++) {
            for (uint e = lid; e < head_dim; e += WG_SIZE) {
                lds_k[kp * head_dim + e] = K[bh_offset + (k_start + kp) * head_dim + e];
            }
        }

        // Cooperatively load V tile to LDS
        for (uint kp = 0; kp < tile_len; kp++) {
            for (uint e = lid; e < head_dim; e += WG_SIZE) {
                lds_v[kp * head_dim + e] = V[bh_offset + (k_start + kp) * head_dim + e];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Now compute attention for each query in the Q tile
        for (uint qi = 0; qi < BR && (q_start + qi) < seq_len; qi++) {
            uint q_pos = q_start + qi;

            // Determine how many K positions this query can attend to
            uint qi_kv_max = causal ? min(q_pos + 1, k_end) : k_end;
            uint qi_k_start = k_start;

            if (qi_k_start >= qi_kv_max) continue;  // This query can't attend to this tile

            uint qi_tile_len = qi_kv_max - qi_k_start;

            // Compute scores for this query against K tile
            // Each thread computes partial dot products, then we need to reduce
            float tile_max = -INFINITY;
            float tile_scores[BC];

            for (uint kp = 0; kp < qi_tile_len; kp++) {
                // Compute Q[qi] @ K[k_start + kp]
                // Each thread computes partial sum for its dimensions
                float partial = 0.0f;
                for (uint e = 0; e < dims_per_thread && (lid * dims_per_thread + e) < head_dim; e++) {
                    uint d = lid * dims_per_thread + e;
                    partial += q_reg[qi * dims_per_thread + e] * lds_k[kp * head_dim + d];
                }

                // Reduce across threads using LDS (reduction array at kernel scope)
                lds_reduction[lid] = partial;
                barrier(CLK_LOCAL_MEM_FENCE);

                // Tree reduction
                for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
                    if (lid < s) {
                        lds_reduction[lid] += lds_reduction[lid + s];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                float score = lds_reduction[0] * scale;

                // Broadcast score to all threads
                if (lid == 0) lds_shared = score;
                barrier(CLK_LOCAL_MEM_FENCE);
                score = lds_shared;

                tile_scores[kp] = score;
                tile_max = fmax(tile_max, score);
            }

            // Broadcast tile_max
            if (lid == 0) lds_shared = tile_max;
            barrier(CLK_LOCAL_MEM_FENCE);
            tile_max = lds_shared;

            // Online softmax update
            float m_new = fmax(m[qi], tile_max);
            float correction = exp(m[qi] - m_new);

            l[qi] *= correction;
            for (uint e = 0; e < dims_per_thread; e++) {
                o[qi * dims_per_thread + e] *= correction;
            }

            // Accumulate contributions from this tile
            for (uint kp = 0; kp < qi_tile_len; kp++) {
                float p = exp(tile_scores[kp] - m_new);
                l[qi] += p;

                // O += p * V[kp]
                for (uint e = 0; e < dims_per_thread && (lid * dims_per_thread + e) < head_dim; e++) {
                    uint d = lid * dims_per_thread + e;
                    o[qi * dims_per_thread + e] += p * lds_v[kp * head_dim + d];
                }
            }

            m[qi] = m_new;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output (each thread writes its dimensions for each query)
    for (uint qi = 0; qi < BR && (q_start + qi) < seq_len; qi++) {
        uint q_pos = q_start + qi;
        float inv_l = 1.0f / l[qi];

        for (uint e = 0; e < dims_per_thread && (lid * dims_per_thread + e) < head_dim; e++) {
            uint d = lid * dims_per_thread + e;
            O[bh_offset + q_pos * head_dim + d] = o[qi * dims_per_thread + e] * inv_l;
        }

        // Store logsumexp (only thread 0)
        if (lid == 0 && L != 0) {
            L[(b * num_heads + h) * seq_len + q_pos] = m[qi] + log(l[qi]);
        }
    }
}

// Simpler version: one thread per query, LDS for K/V tiles only
// Less sync overhead, may be faster for small workgroups
// Each thread handles one query independently, but shares K/V via LDS
__kernel void flash_attention_simple_lds_fp32(
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
    // LDS for K/V tiles - declared at kernel scope
    __local float lds_k[BC * MAX_DIM];
    __local float lds_v[BC * MAX_DIM];

    // Each thread handles one query
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint wg_size = get_local_size(0);

    uint total_queries = batch_size * num_heads * seq_len;
    if (gid >= total_queries) return;

    // Decode indices
    uint b = gid / (num_heads * seq_len);
    uint remainder = gid % (num_heads * seq_len);
    uint h = remainder / seq_len;
    uint i = remainder % seq_len;

    uint bh_offset = (b * num_heads + h) * seq_len * head_dim;
    uint q_offset = bh_offset + i * head_dim;

    // Load Q into registers
    float q_vec[MAX_DIM];
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_offset + d];
    }

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_vec[MAX_DIM];
    for (uint d = 0; d < head_dim; d++) {
        o_vec[d] = 0.0f;
    }

    uint kv_len = causal ? (i + 1) : seq_len;
    uint num_tiles = (kv_len + BC - 1) / BC;

    // Figure out which workgroup this thread belongs to for LDS sharing
    uint wg_id = get_group_id(0);

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_start = tile * BC;
        uint k_end = min(k_start + BC, kv_len);
        uint tile_len = k_end - k_start;

        // Cooperatively load K/V to LDS
        // All threads in workgroup participate
        for (uint j = lid; j < tile_len * head_dim; j += wg_size) {
            uint kp = j / head_dim;
            uint d = j % head_dim;
            lds_k[kp * head_dim + d] = K[bh_offset + (k_start + kp) * head_dim + d];
            lds_v[kp * head_dim + d] = V[bh_offset + (k_start + kp) * head_dim + d];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Now each thread processes its own query using LDS data
        float tile_max = -INFINITY;
        float tile_scores[BC];

        for (uint kp = 0; kp < tile_len; kp++) {
            // Score = Q @ K from LDS
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += q_vec[d] * lds_k[kp * head_dim + d];
            }
            score *= scale;
            tile_scores[kp] = score;
            tile_max = fmax(tile_max, score);
        }

        // Online softmax update
        float m_new = fmax(m_i, tile_max);
        float correction = exp(m_i - m_new);

        l_i *= correction;
        for (uint d = 0; d < head_dim; d++) {
            o_vec[d] *= correction;
        }

        for (uint kp = 0; kp < tile_len; kp++) {
            float p = exp(tile_scores[kp] - m_new);
            l_i += p;

            for (uint d = 0; d < head_dim; d++) {
                o_vec[d] += p * lds_v[kp * head_dim + d];
            }
        }

        m_i = m_new;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output
    float inv_l = 1.0f / l_i;
    for (uint d = 0; d < head_dim; d++) {
        O[q_offset + d] = o_vec[d] * inv_l;
    }

    if (L != 0) {
        L[(b * num_heads + h) * seq_len + i] = m_i + log(l_i);
    }
}
"""


class LDSTiledAttention:
    """
    OpenCL attention with proper LDS tiling for high performance.

    Uses Local Data Share (10x faster than global memory) to cache
    K/V tiles and share them across threads in a workgroup.
    """

    def __init__(self, device_index: int = 0, verbose: bool = True):
        try:
            import pyopencl as cl
        except ImportError:
            raise ImportError("pyopencl required: pip install pyopencl")

        self.cl = cl
        self.verbose = verbose

        # Find GPU
        platforms = cl.get_platforms()
        self.device = None
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                self.device = devices[min(device_index, len(devices) - 1)]
                break

        if self.device is None:
            raise RuntimeError("No GPU found")

        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # Device info
        self.local_mem_size = self.device.local_mem_size
        self.max_work_group_size = self.device.max_work_group_size

        if verbose:
            print(f"LDSTiledAttention on: {self.device.name}")
            print(f"  Local memory: {self.local_mem_size / 1024:.0f} KB")
            print(f"  Max workgroup: {self.max_work_group_size}")

        # Build kernels
        self._build_kernels()

        # Buffer cache
        self._buffers = {}
        self._cache_key = None

    def _build_kernels(self):
        """Build LDS-tiled kernels."""
        cl = self.cl

        try:
            program = cl.Program(self.context, LDS_TILED_FLASH_ATTENTION_KERNEL).build()
            self._tiled_kernel = cl.Kernel(program, "flash_attention_tiled_fp32")
            self._simple_lds_kernel = cl.Kernel(program, "flash_attention_simple_lds_fp32")
            self._kernels_ready = True
            if self.verbose:
                print("  Built LDS-tiled kernels")
        except Exception as e:
            if self.verbose:
                print(f"  Kernel build failed: {e}")
            self._kernels_ready = False

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False,
        use_simple: bool = True,  # Use simpler kernel by default
    ) -> np.ndarray:
        """Compute attention with LDS tiling."""
        if not self._kernels_ready:
            raise RuntimeError("Kernels not available")

        cl = self.cl
        mf = cl.mem_flags

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        batch_size, num_heads, seq_len, head_dim = Q.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        output = np.zeros_like(Q)
        L = np.zeros((batch_size, num_heads, seq_len), dtype=np.float32)

        # Get/create buffers
        cache_key = (Q.shape, 'lds_tiled')
        if cache_key != self._cache_key:
            self._buffers.clear()
            self._buffers['q'] = cl.Buffer(self.context, mf.READ_ONLY, Q.nbytes)
            self._buffers['k'] = cl.Buffer(self.context, mf.READ_ONLY, K.nbytes)
            self._buffers['v'] = cl.Buffer(self.context, mf.READ_ONLY, V.nbytes)
            self._buffers['out'] = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self._buffers['L'] = cl.Buffer(self.context, mf.WRITE_ONLY, L.nbytes)
            self._cache_key = cache_key

        # Upload
        cl.enqueue_copy(self.queue, self._buffers['q'], Q)
        cl.enqueue_copy(self.queue, self._buffers['k'], K)
        cl.enqueue_copy(self.queue, self._buffers['v'], V)

        if use_simple:
            # Simple LDS kernel: one thread per query, LDS for K/V
            kernel = self._simple_lds_kernel
            total_queries = batch_size * num_heads * seq_len

            # Use workgroup size of 64 (AMD wavefront)
            local_size = 64
            global_size = ((total_queries + local_size - 1) // local_size) * local_size

            kernel.set_args(
                self._buffers['q'], self._buffers['k'], self._buffers['v'],
                self._buffers['out'], self._buffers['L'],
                np.uint32(batch_size),
                np.uint32(num_heads),
                np.uint32(seq_len),
                np.uint32(head_dim),
                np.float32(scale),
                np.uint32(1 if causal else 0)
            )

            cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (local_size,))
        else:
            # Full tiled kernel (more complex, may not be faster)
            kernel = self._tiled_kernel
            BR = 4
            q_tiles = (seq_len + BR - 1) // BR
            total_tiles = batch_size * num_heads * q_tiles

            local_size = 64
            global_size = total_tiles * local_size

            kernel.set_args(
                self._buffers['q'], self._buffers['k'], self._buffers['v'],
                self._buffers['out'], self._buffers['L'],
                np.uint32(batch_size),
                np.uint32(num_heads),
                np.uint32(seq_len),
                np.uint32(head_dim),
                np.float32(scale),
                np.uint32(1 if causal else 0)
            )

            cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (local_size,))

        # Download
        cl.enqueue_copy(self.queue, output, self._buffers['out'])
        self.queue.finish()

        return output

    @property
    def backend_name(self) -> str:
        return f"OpenCL LDS-Tiled ({self.device.name})"

    def close(self):
        self._buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def benchmark():
    """Benchmark LDS-tiled vs naive kernels."""
    print("=" * 70)
    print("LDS-TILED ATTENTION BENCHMARK")
    print("=" * 70)

    configs = [
        {"name": "Small", "batch": 1, "heads": 8, "seq": 512, "dim": 64},
        {"name": "LLaMA-7B", "batch": 1, "heads": 32, "seq": 2048, "dim": 128},
    ]

    # Test LDS-tiled
    print("\n--- LDS-Tiled Kernel ---")
    try:
        with LDSTiledAttention(verbose=True) as attn:
            for cfg in configs:
                print(f"\n{cfg['name']} [{cfg['batch']}, {cfg['heads']}, {cfg['seq']}, {cfg['dim']}]:")

                Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
                K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
                V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

                # Warmup
                _ = attn.forward(Q, K, V, causal=True)

                # Benchmark
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    output = attn.forward(Q, K, V, causal=True)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                avg_ms = np.mean(times) * 1000
                flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
                tflops = flops / (np.mean(times) * 1e12)

                print(f"  Time: {avg_ms:.2f}ms, TFLOPS: {tflops:.4f}")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    # Compare with baseline
    print("\n--- Baseline (vec4 kernel) ---")
    try:
        from aule_hybrid import HybridAttention

        with HybridAttention(backend='opencl', verbose=False) as attn:
            for cfg in configs:
                Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
                K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
                V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

                _ = attn.forward(Q, K, V, causal=True)

                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    output = attn.forward(Q, K, V, causal=True)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                avg_ms = np.mean(times) * 1000
                flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
                tflops = flops / (np.mean(times) * 1e12)

                print(f"{cfg['name']}: {avg_ms:.2f}ms, {tflops:.4f} TFLOPS")

    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    benchmark()
