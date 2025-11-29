#!/usr/bin/env python3
"""
aule-attention: Fast OpenCL Kernel for MI300X

This kernel focuses on what actually matters for Rusticl performance:
- Simple control flow (one thread per query, one K/V per iteration)
- Vectorized memory access (float4)
- Fast math intrinsics (native_exp, native_rsqrt)
- Minimal register pressure

The goal is to match or beat the baseline 0.47 TFLOPS.
"""

import numpy as np
from typing import Optional
import time

# The baseline vec4 kernel achieves 0.47 TFLOPS on MI300X
# Let's try to beat it with fast math intrinsics

FAST_ATTENTION_KERNEL = """
// Fast FlashAttention for AMD MI300X via Rusticl
// Uses native_exp for 2-3x faster exp() at cost of precision
// Uses float4 vectorization for memory bandwidth

#define NEG_INF (-1.0f/0.0f)  // -INFINITY

// Fast vec4 kernel using native_exp
__kernel void flash_attention_fast_fp32(
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

    // Load Q as vec4 array
    float4 q[64];
    for (uint d = 0; d < head_dim4; d++) {
        q[d] = Q4[q_base4 + d];
    }

    float m_i = NEG_INF;
    float l_i = 0.0f;
    float4 o[64];
    for (uint d = 0; d < head_dim4; d++) {
        o[d] = (float4)(0.0f);
    }

    uint kv_len = causal ? (i + 1) : seq_len;

    for (uint j = 0; j < kv_len; j++) {
        uint k_base4 = bh4 + j * head_dim4;

        // Compute dot product
        float score = 0.0f;
        for (uint d = 0; d < head_dim4; d++) {
            float4 kv = K4[k_base4 + d];
            float4 prod = q[d] * kv;
            score += prod.x + prod.y + prod.z + prod.w;
        }
        score *= scale;

        // Online softmax using native_exp
        float m_new = fmax(m_i, score);
        float corr = native_exp(m_i - m_new);
        float p = native_exp(score - m_new);

        l_i = l_i * corr + p;

        // Accumulate V
        for (uint d = 0; d < head_dim4; d++) {
            o[d] = o[d] * corr + p * V4[k_base4 + d];
        }

        m_i = m_new;
    }

    // Normalize using native_recip
    float inv_l = native_recip(l_i);
    for (uint d = 0; d < head_dim4; d++) {
        O4[q_base4 + d] = o[d] * inv_l;
    }

    if (L != 0) {
        L[(b * num_heads + h) * seq_len + i] = m_i + native_log(l_i);
    }
}

// Same kernel with standard exp for comparison
__kernel void flash_attention_std_fp32(
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

    float4 q[64];
    for (uint d = 0; d < head_dim4; d++) {
        q[d] = Q4[q_base4 + d];
    }

    float m_i = NEG_INF;
    float l_i = 0.0f;
    float4 o[64];
    for (uint d = 0; d < head_dim4; d++) {
        o[d] = (float4)(0.0f);
    }

    uint kv_len = causal ? (i + 1) : seq_len;

    for (uint j = 0; j < kv_len; j++) {
        uint k_base4 = bh4 + j * head_dim4;

        float score = 0.0f;
        for (uint d = 0; d < head_dim4; d++) {
            float4 kv = K4[k_base4 + d];
            float4 prod = q[d] * kv;
            score += prod.x + prod.y + prod.z + prod.w;
        }
        score *= scale;

        float m_new = fmax(m_i, score);
        float corr = exp(m_i - m_new);
        float p = exp(score - m_new);

        l_i = l_i * corr + p;

        for (uint d = 0; d < head_dim4; d++) {
            o[d] = o[d] * corr + p * V4[k_base4 + d];
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
"""


class FastAttention:
    """Fast attention using native math intrinsics."""

    def __init__(self, device_index: int = 0, verbose: bool = True):
        self.verbose = verbose
        self.device_index = device_index
        self._init_opencl()

    def _init_opencl(self):
        import pyopencl as cl

        self._cl = cl

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

        self._build_kernels()
        self._buffers = {}
        self._cache_key = None

    def _build_kernels(self):
        cl = self._cl

        self._kernel_fast = None
        self._kernel_std = None
        self._kernel_noopts = None

        # Build with fast math
        try:
            program = cl.Program(self._context, FAST_ATTENTION_KERNEL).build(
                options=["-cl-mad-enable", "-cl-fast-relaxed-math", "-cl-unsafe-math-optimizations"]
            )
            self._kernel_fast = cl.Kernel(program, "flash_attention_fast_fp32")
            if self.verbose:
                print("  Built fast kernel (native_exp)")
        except Exception as e:
            if self.verbose:
                print(f"  Fast kernel failed: {e}")

        # Build with standard math
        try:
            program = cl.Program(self._context, FAST_ATTENTION_KERNEL).build(
                options=["-cl-mad-enable"]
            )
            self._kernel_std = cl.Kernel(program, "flash_attention_std_fp32")
            if self.verbose:
                print("  Built std kernel (exp)")
        except Exception as e:
            if self.verbose:
                print(f"  Std kernel failed: {e}")

        # Build with NO options (like baseline)
        try:
            program = cl.Program(self._context, FAST_ATTENTION_KERNEL).build()
            self._kernel_noopts = cl.Kernel(program, "flash_attention_std_fp32")
            if self.verbose:
                print("  Built noopts kernel (no build options)")
        except Exception as e:
            if self.verbose:
                print(f"  Noopts kernel failed: {e}")

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False,
        kernel: str = 'fast'
    ) -> np.ndarray:
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

        cache_key = (Q.shape, kernel)
        if cache_key != self._cache_key:
            self._buffers.clear()
            self._buffers['q'] = cl.Buffer(self._context, mf.READ_ONLY, Q.nbytes)
            self._buffers['k'] = cl.Buffer(self._context, mf.READ_ONLY, K.nbytes)
            self._buffers['v'] = cl.Buffer(self._context, mf.READ_ONLY, V.nbytes)
            self._buffers['out'] = cl.Buffer(self._context, mf.WRITE_ONLY, output.nbytes)
            self._buffers['L'] = cl.Buffer(self._context, mf.WRITE_ONLY, L.nbytes)
            self._cache_key = cache_key

        cl.enqueue_copy(self._queue, self._buffers['q'], Q)
        cl.enqueue_copy(self._queue, self._buffers['k'], K)
        cl.enqueue_copy(self._queue, self._buffers['v'], V)

        total_queries = batch_size * num_heads * seq_len

        if kernel == 'fast':
            k = self._kernel_fast
        elif kernel == 'noopts':
            k = self._kernel_noopts
        else:
            k = self._kernel_std
        k.set_args(
            self._buffers['q'], self._buffers['k'], self._buffers['v'],
            self._buffers['out'], self._buffers['L'],
            np.uint32(batch_size), np.uint32(num_heads),
            np.uint32(seq_len), np.uint32(head_dim),
            np.float32(scale), np.uint32(1 if causal else 0)
        )
        global_size = (total_queries,)
        cl.enqueue_nd_range_kernel(self._queue, k, global_size, None)

        cl.enqueue_copy(self._queue, output, self._buffers['out'])
        self._queue.finish()

        return output

    def close(self):
        self._buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def benchmark_fast():
    """Benchmark fast math kernels."""
    print("=" * 70)
    print("FAST MATH KERNEL BENCHMARK")
    print("=" * 70)

    configs = [
        {"name": "Small", "batch": 1, "heads": 8, "seq": 512, "dim": 64},
        {"name": "TinyLlama", "batch": 1, "heads": 32, "seq": 512, "dim": 64},
        {"name": "LLaMA-7B", "batch": 1, "heads": 32, "seq": 2048, "dim": 128},
    ]

    kernels_to_test = ['fast', 'std', 'noopts']

    with FastAttention(verbose=True) as attn:
        for cfg in configs:
            print(f"\n--- {cfg['name']} [{cfg['batch']}, {cfg['heads']}, {cfg['seq']}, {cfg['dim']}] ---")

            Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

            for kernel in kernels_to_test:
                try:
                    _ = attn.forward(Q, K, V, causal=True, kernel=kernel)

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

    # Compare with baseline
    print("\n--- Baseline Comparison (HybridAttention) ---")
    try:
        from aule_hybrid import HybridAttention

        cfg = {"batch": 1, "heads": 32, "seq": 2048, "dim": 128}
        Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
        K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
        V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

        with HybridAttention(backend='opencl', verbose=False) as baseline:
            _ = baseline.forward(Q, K, V, causal=True)

            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = baseline.forward(Q, K, V, causal=True)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_ms = np.mean(times) * 1000
            flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
            tflops = flops / (np.mean(times) * 1e12)

            print(f"  baseline   : {avg_ms:8.2f}ms, {tflops:.4f} TFLOPS")

    except Exception as e:
        print(f"  Baseline comparison failed: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_fast()
