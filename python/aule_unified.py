"""
aule-attention: Unified Python interface

DUAL-STRATEGY APPROACH:
=======================
1. WITH PyTorch ROCm: ~40-240 TFLOPS on MI300X (FP16)
   - NO BUILDING FROM SOURCE - just pip install!
   - pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

2. WITHOUT ROCm: ~0.5 TFLOPS via OpenCL (Mesa Rusticl)
   - Works on ANY GPU with OpenCL support
   - pip install pyopencl + mesa-opencl-icd

Backend priority order:
1. CK FlashAttention - Peak performance (~150-300 TFLOPS, requires build)
2. ROCm FlashAttention - Peak performance (~150-300 TFLOPS, requires build)
3. PyTorch SDPA on ROCm - High performance (~40-240 TFLOPS, NO BUILD!)
4. HIP MFMA (our kernel) - Good performance (~50-100 TFLOPS)
5. Vulkan - Consumer GPUs
6. OpenCL Optimized - ~0.5 TFLOPS on MI300X via Rusticl
7. OpenCL Basic - ~0.35 TFLOPS
8. CPU fallback - Always available

Usage:
    import aule_unified as aule

    # Automatic backend selection (picks best available)
    with aule.Attention() as attn:
        output = attn.forward(Q, K, V, causal=True, dtype='float16')

    # Or force a specific backend
    with aule.Attention(backend='pytorch_rocm') as attn:
        output = attn.forward(Q, K, V, causal=True, dtype='float16')
"""

import numpy as np
from typing import Optional, Literal
import os

# Import backends - ordered by priority
_ck_available = False
_rocm_flash_available = False
_pytorch_rocm_available = False
_vulkan_available = False
_opencl_available = False
_hip_available = False
_hip_mfma_available = False

# Check for CK FlashAttention (new dedicated wrapper)
try:
    from aule_ck import CKAttention, is_ck_available
    _ck_available = is_ck_available()
except (ImportError, RuntimeError):
    pass

# Check for ROCm FlashAttention (highest performance)
try:
    import torch
    if torch.cuda.is_available() and hasattr(torch.version, 'hip'):
        _pytorch_rocm_available = True
        try:
            from flash_attn import flash_attn_func
            _rocm_flash_available = True
        except ImportError:
            pass
except ImportError:
    pass

try:
    from aule import Aule, AuleError
    _vulkan_available = True
except (ImportError, RuntimeError, OSError):
    pass

try:
    from aule_opencl import OpenCLAttention, is_opencl_available
    _opencl_available = is_opencl_available()
except (ImportError, RuntimeError):
    pass

# Import hybrid backend (provides optimized OpenCL with ROCm fallback)
_hybrid_available = False
try:
    from aule_hybrid import HybridAttention, get_best_backend
    # Only mark as available if OpenCL actually works
    best = get_best_backend()
    if best in ('opencl', 'rocm_flash', 'rocm_sdpa'):
        _hybrid_available = True
except (ImportError, RuntimeError):
    pass

try:
    from aule_hip import HipAttention, is_hip_available
    _hip_available = is_hip_available()
except (ImportError, RuntimeError):
    pass

try:
    from aule_hip_mfma import MFMAAttention, is_mfma_available
    _hip_mfma_available = is_mfma_available()
except (ImportError, RuntimeError):
    pass


Backend = Literal['auto', 'ck', 'vulkan', 'opencl', 'opencl_optimized', 'hip', 'hip_mfma', 'rocm_flash', 'pytorch_rocm', 'cpu']


class Attention:
    """
    Unified attention interface with automatic backend selection.

    Example:
        >>> import aule_unified as aule
        >>> import numpy as np
        >>>
        >>> Q = np.random.randn(1, 8, 64, 64).astype(np.float32)
        >>> K = np.random.randn(1, 8, 64, 64).astype(np.float32)
        >>> V = np.random.randn(1, 8, 64, 64).astype(np.float32)
        >>>
        >>> with aule.Attention() as attn:
        ...     output = attn.forward(Q, K, V)
        ...     print(f"Using backend: {attn.backend_name}")
    """

    def __init__(self, backend: Backend = 'auto'):
        """
        Initialize attention with specified backend.

        Args:
            backend: 'auto', 'rocm_flash', 'pytorch_rocm', 'hip_mfma', 'hip',
                     'vulkan', 'opencl', or 'cpu'
                     Can also be set via AULE_BACKEND environment variable
        """
        # Check environment variable
        env_backend = os.environ.get('AULE_BACKEND', '').lower()
        if env_backend in ('ck', 'vulkan', 'opencl', 'opencl_optimized', 'hip', 'hip_mfma', 'rocm_flash', 'pytorch_rocm', 'cpu'):
            backend = env_backend

        self._backend = None
        self._backend_name = None
        self._impl = None
        self._torch_device = None

        if backend == 'auto':
            self._init_auto()
        elif backend == 'ck':
            self._init_ck()
        elif backend == 'rocm_flash':
            self._init_rocm_flash()
        elif backend == 'pytorch_rocm':
            self._init_pytorch_rocm()
        elif backend == 'vulkan':
            self._init_vulkan()
        elif backend == 'opencl':
            self._init_opencl()
        elif backend == 'opencl_optimized':
            self._init_opencl_optimized()
        elif backend == 'hip':
            self._init_hip()
        elif backend == 'hip_mfma':
            self._init_hip_mfma()
        elif backend == 'cpu':
            self._init_cpu()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_auto(self):
        """Auto-detect best available backend."""
        # Priority order (highest performance first):
        # 1. CK FlashAttention (our wrapper) - ~150-300 TFLOPS on MI300X
        # 2. ROCm FlashAttention (direct) - ~150-300 TFLOPS on MI300X
        # 3. PyTorch SDPA on ROCm - ~50-150 TFLOPS
        # 4. HIP MFMA (our kernel) - ~50-100 TFLOPS
        # 5. HIP scalar - Moderate performance
        # 6. Vulkan - Consumer GPUs
        # 7. OpenCL Optimized (vec4) - ~0.5 TFLOPS on MI300X via Rusticl
        # 8. OpenCL (fallback) - ~0.35 TFLOPS
        # 9. CPU fallback

        # Try CK FlashAttention first (our dedicated wrapper)
        if _ck_available:
            try:
                self._init_ck()
                return
            except Exception:
                pass

        # Try ROCm FlashAttention (peak performance)
        if _rocm_flash_available:
            try:
                self._init_rocm_flash()
                return
            except Exception:
                pass

        # Try PyTorch SDPA on ROCm
        if _pytorch_rocm_available:
            try:
                self._init_pytorch_rocm()
                return
            except Exception:
                pass

        # Try HIP MFMA (our high-performance kernel)
        if _hip_mfma_available:
            try:
                self._init_hip_mfma()
                return
            except Exception:
                pass

        # Try HIP scalar (ROCm but not MFMA-capable)
        if _hip_available:
            try:
                self._init_hip()
                return
            except Exception:
                pass

        # Try Vulkan (most compatible for consumer GPUs)
        if _vulkan_available:
            try:
                self._init_vulkan()
                return
            except Exception:
                pass

        # Try OpenCL Optimized (vec4 kernel - ~40% faster)
        if _hybrid_available:
            try:
                self._init_opencl_optimized()
                return
            except Exception:
                pass

        # Try OpenCL (works on MI300X via Mesa Rusticl without ROCm)
        if _opencl_available:
            try:
                self._init_opencl()
                return
            except Exception:
                pass

        # Fall back to CPU
        self._init_cpu()

    def _init_ck(self):
        """Initialize CK FlashAttention backend (our wrapper)."""
        if not _ck_available:
            raise RuntimeError(
                "CK FlashAttention not available. Install with:\n"
                "  git clone https://github.com/ROCm/flash-attention.git\n"
                "  cd flash-attention && GPU_ARCHS=gfx942 python setup.py install"
            )
        self._impl = CKAttention(verbose=False)
        self._backend = 'ck'
        self._backend_name = self._impl.backend_name

    def _init_rocm_flash(self):
        """Initialize ROCm FlashAttention (Composable Kernel) backend."""
        if not _rocm_flash_available:
            raise RuntimeError(
                "ROCm FlashAttention not available. Install with:\n"
                "  git clone https://github.com/ROCm/flash-attention.git\n"
                "  cd flash-attention && GPU_ARCHS=gfx942 python setup.py install"
            )
        import torch
        self._torch_device = torch.device('cuda:0')
        torch.cuda.set_device(self._torch_device)
        device_name = torch.cuda.get_device_name(0)
        self._backend = 'rocm_flash'
        self._backend_name = f'ROCm FlashAttention CK ({device_name})'

    def _init_pytorch_rocm(self):
        """Initialize PyTorch SDPA on ROCm backend."""
        if not _pytorch_rocm_available:
            raise RuntimeError(
                "PyTorch ROCm not available. Install with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
            )
        import torch
        self._torch_device = torch.device('cuda:0')
        torch.cuda.set_device(self._torch_device)
        device_name = torch.cuda.get_device_name(0)
        self._backend = 'pytorch_rocm'
        self._backend_name = f'PyTorch SDPA ROCm ({device_name})'

    def _init_vulkan(self):
        """Initialize Vulkan backend."""
        if not _vulkan_available:
            raise RuntimeError("Vulkan backend not available")
        self._impl = Aule()
        self._backend = 'vulkan'
        self._backend_name = 'Vulkan'

    def _init_opencl(self):
        """Initialize OpenCL backend."""
        if not _opencl_available:
            raise RuntimeError(
                "OpenCL backend not available. Install with:\n"
                "  pip install pyopencl\n"
                "And ensure mesa-opencl-icd is installed."
            )
        self._impl = OpenCLAttention()
        self._backend = 'opencl'
        self._backend_name = f'OpenCL ({self._impl.device_name})'

    def _init_opencl_optimized(self):
        """Initialize optimized OpenCL backend with vec4 kernel."""
        if not _hybrid_available:
            raise RuntimeError(
                "Optimized OpenCL backend not available. "
                "Ensure aule_hybrid.py is present."
            )
        self._impl = HybridAttention(backend='opencl', verbose=False)
        self._backend = 'opencl_optimized'
        self._backend_name = self._impl.backend_name + ' (optimized)'

    def _init_hip(self):
        """Initialize HIP backend (scalar kernel)."""
        if not _hip_available:
            raise RuntimeError(
                "HIP backend not available. Install with:\n"
                "  pip install hip-python\n"
                "Or ensure ROCm is installed."
            )
        self._impl = HipAttention()
        self._backend = 'hip'
        self._backend_name = f'HIP/ROCm ({self._impl.device_name})'

    def _init_hip_mfma(self):
        """Initialize HIP MFMA backend (matrix core optimized)."""
        if not _hip_mfma_available:
            raise RuntimeError(
                "HIP MFMA backend not available. Requires:\n"
                "  - ROCm 5.0+ with hip-python\n"
                "  - MI200/MI300 series GPU (CDNA2/CDNA3)"
            )
        self._impl = MFMAAttention()
        self._backend = 'hip_mfma'
        self._backend_name = f'HIP/MFMA ({self._impl.device_name})'

    def _init_cpu(self):
        """Initialize CPU fallback."""
        self._impl = None
        self._backend = 'cpu'
        self._backend_name = 'CPU (fallback)'

    @property
    def backend(self) -> str:
        """Current backend name."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        return self._backend_name

    @property
    def fp16_supported(self) -> bool:
        """Check if FP16 is supported on current backend."""
        if self._backend in ('rocm_flash', 'pytorch_rocm'):
            return True  # ROCm backends always support FP16/BF16
        if self._backend in ('opencl', 'hip', 'hip_mfma') and self._impl:
            return self._impl.fp16_supported
        return False

    @property
    def backward_available(self) -> bool:
        """Check if backward pass is available on current backend."""
        if self._backend in ('rocm_flash', 'pytorch_rocm'):
            return True  # PyTorch handles backward
        if self._backend == 'opencl' and self._impl:
            return getattr(self._impl, 'backward_available', False)
        if self._backend == 'cpu':
            return True  # CPU backward is always available
        return False

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
        dtype: Literal['float32', 'float16'] = 'float32',
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor [batch, heads, seq, dim]
            K: Key tensor [batch, heads, seq, dim]
            V: Value tensor [batch, heads, seq, dim]
            causal: If True, apply causal masking (for autoregressive models)
            dtype: 'float32' or 'float16'

        Returns:
            Output tensor [batch, heads, seq, dim]
        """
        if self._backend == 'ck':
            return self._impl.forward(Q, K, V, causal=causal, dtype=dtype)
        elif self._backend == 'rocm_flash':
            return self._forward_rocm_flash(Q, K, V, causal=causal, dtype=dtype)
        elif self._backend == 'pytorch_rocm':
            return self._forward_pytorch_rocm(Q, K, V, causal=causal, dtype=dtype)
        elif self._backend == 'vulkan':
            return self._impl.attention(Q, K, V)
        elif self._backend == 'opencl':
            return self._impl.forward(Q, K, V, causal=causal, dtype=dtype)
        elif self._backend == 'opencl_optimized':
            return self._impl.forward(Q, K, V, causal=causal)
        elif self._backend == 'hip':
            return self._impl.forward(Q, K, V, causal=causal, dtype=dtype)
        elif self._backend == 'hip_mfma':
            return self._impl.forward(Q, K, V, causal=causal, dtype=dtype)
        else:
            return self._cpu_attention(Q, K, V, causal=causal)

    def _forward_rocm_flash(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
        dtype: str = 'float16',
    ) -> np.ndarray:
        """Forward pass using ROCm FlashAttention (CK)."""
        import torch
        from flash_attn import flash_attn_func

        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        scale = 1.0 / np.sqrt(Q.shape[-1])

        # Convert to torch tensors
        Q_t = torch.from_numpy(Q).to(device=self._torch_device, dtype=torch_dtype)
        K_t = torch.from_numpy(K).to(device=self._torch_device, dtype=torch_dtype)
        V_t = torch.from_numpy(V).to(device=self._torch_device, dtype=torch_dtype)

        # flash_attn expects [batch, seq, heads, head_dim] format
        Q_t = Q_t.transpose(1, 2).contiguous()
        K_t = K_t.transpose(1, 2).contiguous()
        V_t = V_t.transpose(1, 2).contiguous()

        # Call FlashAttention
        output = flash_attn_func(Q_t, K_t, V_t, softmax_scale=scale, causal=causal)

        # Transpose back to [batch, heads, seq, dim]
        output = output.transpose(1, 2).contiguous()

        return output.cpu().numpy()

    def _forward_pytorch_rocm(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
        dtype: str = 'float16',
    ) -> np.ndarray:
        """Forward pass using PyTorch SDPA on ROCm."""
        import torch
        import torch.nn.functional as F

        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        scale = 1.0 / np.sqrt(Q.shape[-1])

        # Convert numpy to torch tensors on GPU
        Q_t = torch.from_numpy(np.ascontiguousarray(Q)).to(device=self._torch_device, dtype=torch_dtype)
        K_t = torch.from_numpy(np.ascontiguousarray(K)).to(device=self._torch_device, dtype=torch_dtype)
        V_t = torch.from_numpy(np.ascontiguousarray(V)).to(device=self._torch_device, dtype=torch_dtype)

        with torch.no_grad():
            output = F.scaled_dot_product_attention(
                Q_t, K_t, V_t, scale=scale, is_causal=causal
            )

        # Synchronize before returning to ensure computation is complete
        torch.cuda.synchronize()

        return output.cpu().numpy()

    def _cpu_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        causal: bool = False,
    ) -> np.ndarray:
        """CPU fallback implementation."""
        # Ensure float32
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        batch_size, num_heads, seq_len, head_dim = Q.shape
        scale = 1.0 / np.sqrt(head_dim)

        output = np.zeros_like(Q)

        for b in range(batch_size):
            for h in range(num_heads):
                q = Q[b, h]  # [seq, dim]
                k = K[b, h]  # [seq, dim]
                v = V[b, h]  # [seq, dim]

                # Attention scores: Q @ K^T * scale
                scores = np.matmul(q, k.T) * scale  # [seq, seq]

                # Apply causal mask if requested
                if causal:
                    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
                    scores = scores - mask * 1e9  # Large negative for masked positions

                # Softmax (numerically stable)
                scores_max = np.max(scores, axis=-1, keepdims=True)
                scores_exp = np.exp(scores - scores_max)
                scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
                attn = scores_exp / scores_sum  # [seq, seq]

                # Output: attn @ V
                output[b, h] = np.matmul(attn, v)  # [seq, dim]

        # Save for backward
        self._saved_for_backward = (Q, K, V, scale, causal)

        return output

    def backward(
        self,
        dO: np.ndarray,
    ) -> tuple:
        """
        Compute backward pass gradients.

        Must be called after forward() on the same batch.

        Args:
            dO: Gradient of output [batch, heads, seq, dim]

        Returns:
            Tuple of (dQ, dK, dV) gradients
        """
        if not self.backward_available:
            raise RuntimeError(
                f"Backward pass not available on backend: {self._backend}"
            )

        if self._backend == 'opencl' and self._impl:
            return self._impl.backward(dO)
        elif self._backend == 'cpu':
            return self._cpu_backward(dO)
        else:
            raise RuntimeError(
                f"Backward not implemented for backend: {self._backend}"
            )

    def _cpu_backward(
        self,
        dO: np.ndarray,
    ) -> tuple:
        """CPU backward pass implementation."""
        if not hasattr(self, '_saved_for_backward'):
            raise RuntimeError("No saved tensors from forward pass")

        Q, K, V, scale, causal = self._saved_for_backward

        batch_size, num_heads, seq_len, head_dim = Q.shape

        dQ = np.zeros_like(Q)
        dK = np.zeros_like(K)
        dV = np.zeros_like(V)

        for b in range(batch_size):
            for h in range(num_heads):
                q = Q[b, h]
                k = K[b, h]
                v = V[b, h]
                do = dO[b, h]

                # Recompute attention
                scores = np.matmul(q, k.T) * scale

                if causal:
                    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
                    scores = scores - mask * 1e9

                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                P = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

                # dV = P^T @ dO
                dV[b, h] = np.matmul(P.T, do)

                # dP = dO @ V^T
                dP = np.matmul(do, v.T)

                # D = rowsum(P * dP)
                D = np.sum(P * dP, axis=-1, keepdims=True)

                # dS = P * (dP - D)
                dS = P * (dP - D)

                if causal:
                    dS = dS * (1 - mask)

                # dQ = scale * dS @ K
                dQ[b, h] = scale * np.matmul(dS, k)

                # dK = scale * dS^T @ Q
                dK[b, h] = scale * np.matmul(dS.T, q)

        del self._saved_for_backward
        return dQ, dK, dV

    def close(self):
        """Release resources."""
        if self._impl is not None:
            self._impl.close()
            self._impl = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def get_available_backends() -> list:
    """Return list of available backends in priority order."""
    backends = []

    # CK FlashAttention (highest priority)
    if _ck_available:
        backends.append('ck')

    # ROCm backends
    if _rocm_flash_available:
        backends.append('rocm_flash')

    if _pytorch_rocm_available:
        backends.append('pytorch_rocm')

    # HIP backends
    if _hip_mfma_available:
        backends.append('hip_mfma')

    if _hip_available:
        backends.append('hip')

    # Portable backends
    if _vulkan_available:
        backends.append('vulkan')

    # OpenCL backends
    if _hybrid_available:
        backends.append('opencl_optimized')

    if _opencl_available:
        backends.append('opencl')

    backends.append('cpu')  # Always available

    return backends


def get_backend_info() -> dict:
    """Get detailed information about available backends."""
    return {
        'ck': {
            'available': _ck_available,
            'description': 'CK FlashAttention (our wrapper for AMD CK)',
            'performance': '~150-300 TFLOPS on MI300X (FP16)',
            'requires': 'ROCm + flash-attn package',
        },
        'rocm_flash': {
            'available': _rocm_flash_available,
            'description': 'ROCm FlashAttention (Composable Kernel)',
            'performance': '~150-300 TFLOPS on MI300X',
            'requires': 'ROCm + flash-attn package',
        },
        'pytorch_rocm': {
            'available': _pytorch_rocm_available,
            'description': 'PyTorch SDPA on ROCm (NO BUILD REQUIRED)',
            'performance': '~40-240 TFLOPS on MI300X (FP16)',
            'requires': 'pip install torch --index-url https://download.pytorch.org/whl/rocm6.1',
        },
        'hip_mfma': {
            'available': _hip_mfma_available,
            'description': 'HIP MFMA (our matrix core kernel)',
            'performance': '~50-100 TFLOPS on MI300X',
            'requires': 'ROCm + hip-python',
        },
        'hip': {
            'available': _hip_available,
            'description': 'HIP scalar (our basic kernel)',
            'performance': '~5-10 TFLOPS',
            'requires': 'ROCm + hip-python',
        },
        'vulkan': {
            'available': _vulkan_available,
            'description': 'Vulkan compute',
            'performance': 'Varies by GPU',
            'requires': 'Vulkan driver + our Zig library',
        },
        'opencl_optimized': {
            'available': _hybrid_available,
            'description': 'OpenCL with vec4 vectorization (Hybrid backend)',
            'performance': '~0.5 TFLOPS on MI300X via Rusticl',
            'requires': 'pyopencl + mesa-opencl-icd (NO ROCm needed!)',
        },
        'opencl': {
            'available': _opencl_available,
            'description': 'OpenCL FlashAttention-2 (Mesa Rusticl)',
            'performance': '~0.35 TFLOPS on MI300X via Rusticl',
            'requires': 'pyopencl + mesa-opencl-icd (NO ROCm needed!)',
        },
        'cpu': {
            'available': True,
            'description': 'NumPy CPU fallback',
            'performance': '~0.01 TFLOPS',
            'requires': 'numpy',
        },
    }


def attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    backend: Backend = 'auto',
    causal: bool = False,
    dtype: Literal['float32', 'float16'] = 'float32',
) -> np.ndarray:
    """
    One-shot attention computation.

    For repeated computations, use the Attention class directly
    to avoid initialization overhead.

    Args:
        Q: Query tensor [batch, heads, seq, dim]
        K: Key tensor [batch, heads, seq, dim]
        V: Value tensor [batch, heads, seq, dim]
        backend: 'auto', 'rocm_flash', 'pytorch_rocm', 'hip_mfma', 'hip',
                 'vulkan', 'opencl', or 'cpu'
        causal: If True, apply causal masking (for autoregressive models)
        dtype: 'float32' or 'float16'

    Returns:
        Output tensor [batch, heads, seq, dim]
    """
    with Attention(backend=backend) as attn:
        return attn.forward(Q, K, V, causal=causal, dtype=dtype)


def print_backend_info():
    """Print detailed information about all backends."""
    print("=" * 70)
    print("AULE-ATTENTION BACKEND STATUS")
    print("=" * 70)
    print("\nAvailable backends (in priority order):")

    info = get_backend_info()
    available = get_available_backends()

    for backend in ['ck', 'rocm_flash', 'pytorch_rocm', 'hip_mfma', 'hip', 'vulkan', 'opencl_optimized', 'opencl', 'cpu']:
        data = info[backend]
        status = "✓ AVAILABLE" if data['available'] else "✗ Not available"
        priority = available.index(backend) + 1 if backend in available else "-"

        print(f"\n  [{priority}] {backend}")
        print(f"      Status: {status}")
        print(f"      Description: {data['description']}")
        print(f"      Performance: {data['performance']}")
        print(f"      Requires: {data['requires']}")

    print("\n" + "=" * 70)
    print(f"Best available backend: {available[0] if available else 'None'}")
    print("=" * 70)


if __name__ == '__main__':
    print_backend_info()

    # Quick test
    import time

    print("\nRunning quick test...")
    Q = np.random.randn(1, 4, 32, 64).astype(np.float32)
    K = np.random.randn(1, 4, 32, 64).astype(np.float32)
    V = np.random.randn(1, 4, 32, 64).astype(np.float32)

    for backend in get_available_backends():
        try:
            with Attention(backend=backend) as attn:
                start = time.perf_counter()
                output = attn.forward(Q, K, V)
                elapsed = time.perf_counter() - start
                print(f"  {attn.backend_name}: {elapsed*1000:.2f}ms, shape={output.shape}")
        except Exception as e:
            print(f"  {backend}: FAILED - {e}")
