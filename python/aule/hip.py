"""
HIP/ROCm backend for aule-attention

This module provides direct HIP support for AMD datacenter GPUs (MI300X, MI250, etc.)
with ROCm installed. Uses FlashAttention-2 algorithm for O(N) memory efficiency.

Features:
- FlashAttention-2 tiled algorithm for O(N) memory
- Causal masking for autoregressive models
- Online softmax for numerical stability
- FP16 support for 2x memory bandwidth
- Support for long sequences (8K, 16K, 32K+ tokens)

Requires:
- ROCm installed (provides hiprtc, hip runtime)
- hip-python package: pip install hip-python

Usage:
    from aule_hip import HipAttention

    with HipAttention() as attn:
        output = attn.forward(Q, K, V, causal=True)
"""

import numpy as np
from typing import Optional, Literal

# Try to import HIP bindings
HIP_AVAILABLE = False
try:
    from hip import hip, hiprtc
    HIP_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# FlashAttention-2 Kernel (HIP C++)
# =============================================================================
# Uses online softmax to process K/V in blocks without storing full attention matrix
# Supports unlimited sequence length (8K, 16K, 32K+ tokens)

FLASH_ATTENTION_KERNEL_FP32 = '''
// FlashAttention-2 Kernel for AMD GPUs
// One thread per query position - memory efficient O(N) algorithm
// Processes K/V in blocks using online softmax

#define BLOCK_SIZE 64    // K/V positions per block
#define MAX_HEAD_DIM 256 // Maximum head dimension

extern "C" __global__ void flash_attention_forward_fp32(
    const float* __restrict__ Q,     // [batch, heads, seq_len, head_dim]
    const float* __restrict__ K,     // [batch, heads, seq_len, head_dim]
    const float* __restrict__ V,     // [batch, heads, seq_len, head_dim]
    float* __restrict__ O,           // [batch, heads, seq_len, head_dim]
    float* __restrict__ L,           // [batch, heads, seq_len] - logsumexp for backward
    unsigned int batch_size,
    unsigned int num_heads,
    unsigned int seq_len,
    unsigned int head_dim,
    float scale,
    unsigned int causal
) {
    // One thread per (batch, head, query_position)
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_queries = batch_size * num_heads * seq_len;

    if (gid >= total_queries) return;

    // Decode indices
    const unsigned int b = gid / (num_heads * seq_len);
    const unsigned int remainder = gid % (num_heads * seq_len);
    const unsigned int h = remainder / seq_len;
    const unsigned int i = remainder % seq_len;  // Query position

    // Base offsets
    const unsigned int bh_offset = (b * num_heads + h) * seq_len * head_dim;
    const unsigned int q_offset = bh_offset + i * head_dim;

    // Load query vector into registers
    float q_vec[MAX_HEAD_DIM];
    for (unsigned int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_offset + d];
    }

    // Online softmax accumulators
    float m_i = -1e30f;  // Running max
    float l_i = 0.0f;    // Running sum of exp(score - max)

    // Output accumulator
    float o_acc[MAX_HEAD_DIM];
    for (unsigned int d = 0; d < head_dim; d++) {
        o_acc[d] = 0.0f;
    }

    // Determine how many K/V positions to attend to
    const unsigned int kv_len = causal ? (i + 1) : seq_len;

    // Process K/V in blocks
    float block_scores[BLOCK_SIZE];
    const unsigned int num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (unsigned int block = 0; block < num_blocks; block++) {
        const unsigned int k_start = block * BLOCK_SIZE;
        const unsigned int k_end = min(k_start + BLOCK_SIZE, kv_len);
        const unsigned int block_len = k_end - k_start;

        // Find max score in this block
        float block_max = -1e30f;

        for (unsigned int jj = 0; jj < block_len; jj++) {
            const unsigned int j = k_start + jj;
            const unsigned int k_offset = bh_offset + j * head_dim;

            // Compute Q[i] @ K[j]
            float score = 0.0f;
            for (unsigned int d = 0; d < head_dim; d++) {
                score += q_vec[d] * K[k_offset + d];
            }
            score *= scale;
            block_scores[jj] = score;
            block_max = fmaxf(block_max, score);
        }

        // Online softmax update
        const float m_new = fmaxf(m_i, block_max);
        const float correction = expf(m_i - m_new);

        // Scale previous accumulator
        l_i *= correction;
        for (unsigned int d = 0; d < head_dim; d++) {
            o_acc[d] *= correction;
        }

        // Add contribution from this block
        for (unsigned int jj = 0; jj < block_len; jj++) {
            const unsigned int j = k_start + jj;
            const float p = expf(block_scores[jj] - m_new);
            l_i += p;

            const unsigned int v_offset = bh_offset + j * head_dim;
            for (unsigned int d = 0; d < head_dim; d++) {
                o_acc[d] += p * V[v_offset + d];
            }
        }

        m_i = m_new;
    }

    // Normalize output by softmax denominator
    const float inv_l = 1.0f / l_i;
    for (unsigned int d = 0; d < head_dim; d++) {
        O[q_offset + d] = o_acc[d] * inv_l;
    }

    // Store logsumexp for backward pass
    if (L != nullptr) {
        const unsigned int l_offset = (b * num_heads + h) * seq_len + i;
        L[l_offset] = m_i + logf(l_i);
    }
}
'''

# =============================================================================
# FlashAttention-2 FP16 Kernel (Half precision for 2x memory bandwidth)
# =============================================================================

FLASH_ATTENTION_KERNEL_FP16 = '''
#include <hip/hip_fp16.h>

// FlashAttention-2 FP16 - Mixed Precision for 2x Memory Bandwidth
// Reads FP16, accumulates in FP32, writes FP16

#define BLOCK_SIZE 64
#define MAX_HEAD_DIM 256

extern "C" __global__ void flash_attention_forward_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,
    unsigned int batch_size,
    unsigned int num_heads,
    unsigned int seq_len,
    unsigned int head_dim,
    float scale,
    unsigned int causal
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_queries = batch_size * num_heads * seq_len;

    if (gid >= total_queries) return;

    // Decode indices
    const unsigned int b = gid / (num_heads * seq_len);
    const unsigned int remainder = gid % (num_heads * seq_len);
    const unsigned int h = remainder / seq_len;
    const unsigned int i = remainder % seq_len;

    // Base offsets
    const unsigned int bh_offset = (b * num_heads + h) * seq_len * head_dim;
    const unsigned int q_offset = bh_offset + i * head_dim;

    // Load query vector into registers (FP16 -> FP32)
    float q_vec[MAX_HEAD_DIM];
    for (unsigned int d = 0; d < head_dim; d++) {
        q_vec[d] = __half2float(Q[q_offset + d]);
    }

    // Online softmax accumulators (FP32 for precision)
    float m_i = -1e30f;
    float l_i = 0.0f;

    // Output accumulator (FP32)
    float o_acc[MAX_HEAD_DIM];
    for (unsigned int d = 0; d < head_dim; d++) {
        o_acc[d] = 0.0f;
    }

    const unsigned int kv_len = causal ? (i + 1) : seq_len;
    float block_scores[BLOCK_SIZE];
    const unsigned int num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (unsigned int block = 0; block < num_blocks; block++) {
        const unsigned int k_start = block * BLOCK_SIZE;
        const unsigned int k_end = min(k_start + BLOCK_SIZE, kv_len);
        const unsigned int block_len = k_end - k_start;

        float block_max = -1e30f;

        for (unsigned int jj = 0; jj < block_len; jj++) {
            const unsigned int j = k_start + jj;
            const unsigned int k_offset = bh_offset + j * head_dim;

            // Compute Q[i] @ K[j] (FP16 -> FP32 accumulation)
            float score = 0.0f;
            for (unsigned int d = 0; d < head_dim; d++) {
                score += q_vec[d] * __half2float(K[k_offset + d]);
            }
            score *= scale;
            block_scores[jj] = score;
            block_max = fmaxf(block_max, score);
        }

        const float m_new = fmaxf(m_i, block_max);
        const float correction = expf(m_i - m_new);

        l_i *= correction;
        for (unsigned int d = 0; d < head_dim; d++) {
            o_acc[d] *= correction;
        }

        for (unsigned int jj = 0; jj < block_len; jj++) {
            const unsigned int j = k_start + jj;
            const float p = expf(block_scores[jj] - m_new);
            l_i += p;

            const unsigned int v_offset = bh_offset + j * head_dim;
            for (unsigned int d = 0; d < head_dim; d++) {
                o_acc[d] += p * __half2float(V[v_offset + d]);
            }
        }

        m_i = m_new;
    }

    // Normalize and write FP16
    const float inv_l = 1.0f / l_i;
    for (unsigned int d = 0; d < head_dim; d++) {
        O[q_offset + d] = __float2half(o_acc[d] * inv_l);
    }

    if (L != nullptr) {
        const unsigned int l_offset = (b * num_heads + h) * seq_len + i;
        L[l_offset] = m_i + logf(l_i);
    }
}
'''


class HipAttention:
    """
    HIP-based attention for AMD GPUs with ROCm.

    Features:
    - FlashAttention-2 algorithm for O(N) memory usage
    - Online softmax for numerical stability
    - Causal masking for autoregressive models
    - FP16 support for 2x memory bandwidth
    - Support for long sequences (8K, 16K, 32K+ tokens)

    Example:
        >>> attn = HipAttention()
        >>> Q = np.random.randn(1, 8, 1024, 64).astype(np.float32)
        >>> K = np.random.randn(1, 8, 1024, 64).astype(np.float32)
        >>> V = np.random.randn(1, 8, 1024, 64).astype(np.float32)
        >>> output = attn.forward(Q, K, V, causal=True)
        >>> attn.close()
    """

    def __init__(self, device_index: int = 0):
        if not HIP_AVAILABLE:
            raise RuntimeError(
                "HIP not available. Install with: pip install hip-python\n"
                "Or ensure ROCm is installed: https://rocm.docs.amd.com/"
            )

        # Initialize HIP
        hip.hipInit(0)

        # Get device
        self.device_count = hip.hipGetDeviceCount()
        if self.device_count == 0:
            raise RuntimeError("No HIP devices found")

        hip.hipSetDevice(device_index)

        # Get device properties
        props = hip.hipGetDeviceProperties(device_index)
        self.device_name = props.name.decode() if isinstance(props.name, bytes) else props.name
        self.max_threads_per_block = props.maxThreadsPerBlock
        self.warp_size = props.warpSize  # 64 on AMD

        # Check FP16 support (all modern AMD GPUs support it)
        self.fp16_supported = True

        # Compile kernels
        self._kernel_fp32 = None
        self._kernel_fp16 = None
        self._module_fp32 = None
        self._module_fp16 = None
        self._compile_kernels()

        # Buffer cache for repeated computations
        self._buffer_cache = {}
        self._cached_key = None

        print(f"HIP initialized on: {self.device_name}")
        print(f"  Max threads/block: {self.max_threads_per_block}, Warp size: {self.warp_size}")

        self._initialized = True

    def _compile_kernels(self):
        """Compile FlashAttention kernels using hiprtc."""
        # Compile FP32 kernel
        prog_fp32 = hiprtc.hiprtcCreateProgram(
            FLASH_ATTENTION_KERNEL_FP32.encode(),
            b"flash_attention_fp32",
            0, [], []
        )

        try:
            hiprtc.hiprtcCompileProgram(prog_fp32, 0, [])
        except Exception as e:
            log_size = hiprtc.hiprtcGetProgramLogSize(prog_fp32)
            log = bytearray(log_size)
            hiprtc.hiprtcGetProgramLog(prog_fp32, log)
            raise RuntimeError(f"FP32 kernel compilation failed: {log.decode()}")

        code_size = hiprtc.hiprtcGetCodeSize(prog_fp32)
        code = bytearray(code_size)
        hiprtc.hiprtcGetCode(prog_fp32, code)

        self._module_fp32 = hip.hipModuleLoadData(bytes(code))
        self._kernel_fp32 = hip.hipModuleGetFunction(self._module_fp32, b"flash_attention_forward_fp32")
        hiprtc.hiprtcDestroyProgram(prog_fp32)

        # Compile FP16 kernel
        try:
            prog_fp16 = hiprtc.hiprtcCreateProgram(
                FLASH_ATTENTION_KERNEL_FP16.encode(),
                b"flash_attention_fp16",
                0, [], []
            )
            hiprtc.hiprtcCompileProgram(prog_fp16, 0, [])

            code_size = hiprtc.hiprtcGetCodeSize(prog_fp16)
            code = bytearray(code_size)
            hiprtc.hiprtcGetCode(prog_fp16, code)

            self._module_fp16 = hip.hipModuleLoadData(bytes(code))
            self._kernel_fp16 = hip.hipModuleGetFunction(self._module_fp16, b"flash_attention_forward_fp16")
            hiprtc.hiprtcDestroyProgram(prog_fp16)
        except Exception as e:
            print(f"Warning: FP16 kernel compilation failed: {e}")
            self.fp16_supported = False

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
        Compute scaled dot-product attention using FlashAttention-2.

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
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError("Q, K, V must have same shape")

        if len(Q.shape) != 4:
            raise ValueError("Expected 4D tensors [batch, heads, seq, dim]")

        use_fp16 = dtype == 'float16'
        if use_fp16 and not self.fp16_supported:
            raise RuntimeError("FP16 not supported on this device")

        batch_size, num_heads, seq_len, head_dim = Q.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Convert to appropriate dtype
        np_dtype = np.float16 if use_fp16 else np.float32
        Q = np.ascontiguousarray(Q, dtype=np_dtype)
        K = np.ascontiguousarray(K, dtype=np_dtype)
        V = np.ascontiguousarray(V, dtype=np_dtype)

        output = np.zeros_like(Q)
        L = np.zeros((batch_size, num_heads, seq_len), dtype=np.float32)

        # Allocate device memory
        total_size = Q.nbytes
        L_size = L.nbytes

        d_Q = hip.hipMalloc(total_size)
        d_K = hip.hipMalloc(total_size)
        d_V = hip.hipMalloc(total_size)
        d_O = hip.hipMalloc(total_size)
        d_L = hip.hipMalloc(L_size)

        try:
            # Copy to device
            hip.hipMemcpy(d_Q, Q.ctypes.data, total_size, hip.hipMemcpyHostToDevice)
            hip.hipMemcpy(d_K, K.ctypes.data, total_size, hip.hipMemcpyHostToDevice)
            hip.hipMemcpy(d_V, V.ctypes.data, total_size, hip.hipMemcpyHostToDevice)

            # Launch kernel
            total_queries = batch_size * num_heads * seq_len
            block_size = 256
            grid_size = (total_queries + block_size - 1) // block_size

            kernel = self._kernel_fp16 if use_fp16 else self._kernel_fp32

            hip.hipModuleLaunchKernel(
                kernel,
                grid_size, 1, 1,  # grid
                block_size, 1, 1,  # block
                0,  # shared mem
                None,  # stream
                [d_Q, d_K, d_V, d_O, d_L,
                 np.uint32(batch_size), np.uint32(num_heads),
                 np.uint32(seq_len), np.uint32(head_dim),
                 np.float32(scale), np.uint32(1 if causal else 0)]
            )

            hip.hipDeviceSynchronize()

            # Copy result back
            hip.hipMemcpy(output.ctypes.data, d_O, total_size, hip.hipMemcpyDeviceToHost)

            return output

        finally:
            # Free device memory
            hip.hipFree(d_Q)
            hip.hipFree(d_K)
            hip.hipFree(d_V)
            hip.hipFree(d_O)
            hip.hipFree(d_L)

    def close(self):
        """Release resources."""
        if hasattr(self, '_module_fp32') and self._module_fp32:
            try:
                hip.hipModuleUnload(self._module_fp32)
            except:
                pass
        if hasattr(self, '_module_fp16') and self._module_fp16:
            try:
                hip.hipModuleUnload(self._module_fp16)
            except:
                pass
        self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    @property
    def backend_name(self) -> str:
        return f"HIP/ROCm ({self.device_name})"


def is_hip_available() -> bool:
    """Check if HIP is available."""
    if not HIP_AVAILABLE:
        return False
    try:
        hip.hipInit(0)
        count = hip.hipGetDeviceCount()
        return count > 0
    except:
        return False


def get_hip_device_name() -> Optional[str]:
    """Get the name of the first HIP device."""
    if not is_hip_available():
        return None
    try:
        props = hip.hipGetDeviceProperties(0)
        return props.name.decode() if isinstance(props.name, bytes) else props.name
    except:
        return None


if __name__ == "__main__":
    print("=== HIP Attention Test ===\n")

    if not is_hip_available():
        print("HIP not available. Install hip-python or ensure ROCm is installed.")
        exit(1)

    print(f"HIP Device: {get_hip_device_name()}\n")

    # Test attention
    print("Testing attention computation...")

    batch_size = 1
    num_heads = 8
    seq_len = 512
    head_dim = 64

    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    with HipAttention() as attn:
        print(f"Using: {attn.backend_name}")

        # Warmup
        _ = attn.forward(Q, K, V)

        # Benchmark
        import time
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            output = attn.forward(Q, K, V, causal=True)
        elapsed = time.perf_counter() - start

        print(f"Output shape: {output.shape}")
        print(f"Time per iteration (causal): {elapsed/n_iters*1000:.2f} ms")

        # Verify against numpy reference
        print("\nVerifying correctness...")
        scale = 1.0 / np.sqrt(head_dim)

        # Reference implementation (non-causal for simplicity)
        scores = np.einsum('bhid,bhjd->bhij', Q, K) * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        reference = np.einsum('bhij,bhjd->bhid', attention, V)

        # Non-causal output for comparison
        output_non_causal = attn.forward(Q, K, V, causal=False)
        max_diff = np.abs(output_non_causal - reference).max()
        print(f"Max difference from reference: {max_diff:.6f}")

        if max_diff < 1e-4:
            print("✓ Results match reference implementation!")
        else:
            print("✗ Results differ from reference (may be precision issue)")

    print("\n=== Test Complete ===")
