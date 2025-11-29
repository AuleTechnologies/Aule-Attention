"""
ROCm FlashAttention Backend for aule-attention

This module provides a wrapper around AMD's official ROCm FlashAttention
implementation (Composable Kernel backend) for peak performance on MI200/MI300.

Performance Tiers:
1. ROCm FlashAttention (CK): ~150-300 TFLOPS on MI300X
2. Our HIP MFMA kernel: ~50-100 TFLOPS (fallback if CK unavailable)
3. Our OpenCL kernel: ~0.5-2 TFLOPS (works without ROCm)

This backend automatically detects and uses the best available option.

Requirements for peak performance:
- ROCm 5.6+ installed
- flash-attn package with ROCm support:
    git clone https://github.com/ROCm/flash-attention.git
    cd flash-attention && GPU_ARCHS=gfx942 python setup.py install

Usage:
    from aule_rocm import ROCmAttention, is_rocm_available

    if is_rocm_available():
        with ROCmAttention() as attn:
            output = attn.forward(Q, K, V, causal=True)
"""

import numpy as np
from typing import Optional, Literal, Tuple

# Detection flags
ROCM_FLASH_AVAILABLE = False
PYTORCH_ROCM_AVAILABLE = False
HIP_AVAILABLE = False

# Try to import ROCm FlashAttention (Composable Kernel)
try:
    import torch
    if torch.cuda.is_available() and hasattr(torch.version, 'hip'):
        PYTORCH_ROCM_AVAILABLE = True
        try:
            from flash_attn import flash_attn_func
            ROCM_FLASH_AVAILABLE = True
        except ImportError:
            pass
except ImportError:
    pass

# Try HIP Python bindings as fallback
try:
    from hip import hip, hiprtc
    HIP_AVAILABLE = True
except ImportError:
    pass


def get_rocm_info() -> dict:
    """Get information about ROCm environment."""
    info = {
        'rocm_available': False,
        'pytorch_rocm': PYTORCH_ROCM_AVAILABLE,
        'flash_attn_ck': ROCM_FLASH_AVAILABLE,
        'hip_python': HIP_AVAILABLE,
        'device_name': None,
        'device_arch': None,
        'rocm_version': None,
    }

    if PYTORCH_ROCM_AVAILABLE:
        import torch
        info['rocm_available'] = True
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_arch'] = torch.cuda.get_device_capability(0)
        if hasattr(torch.version, 'hip'):
            info['rocm_version'] = torch.version.hip

    elif HIP_AVAILABLE:
        try:
            hip.hipInit(0)
            count = hip.hipGetDeviceCount()
            if count > 0:
                info['rocm_available'] = True
                props = hip.hipGetDeviceProperties(0)
                info['device_name'] = props.name.decode() if isinstance(props.name, bytes) else props.name
                info['device_arch'] = props.gcnArchName.decode() if isinstance(props.gcnArchName, bytes) else props.gcnArchName
        except Exception:
            pass

    return info


def is_rocm_available() -> bool:
    """Check if any ROCm backend is available."""
    return ROCM_FLASH_AVAILABLE or PYTORCH_ROCM_AVAILABLE or HIP_AVAILABLE


def is_rocm_flash_available() -> bool:
    """Check if ROCm FlashAttention (CK) is available for peak performance."""
    return ROCM_FLASH_AVAILABLE


class ROCmAttention:
    """
    ROCm-optimized attention with automatic backend selection.

    Priority order:
    1. ROCm FlashAttention (CK) - ~150-300 TFLOPS
    2. PyTorch SDPA on ROCm - ~50-150 TFLOPS
    3. HIP MFMA kernel (our implementation) - ~50-100 TFLOPS

    Example:
        >>> with ROCmAttention() as attn:
        ...     output = attn.forward(Q, K, V, causal=True)
        ...     print(f"Using: {attn.backend_name}")
    """

    def __init__(self, device_index: int = 0, prefer_flash: bool = True):
        """
        Initialize ROCm attention.

        Args:
            device_index: GPU device index
            prefer_flash: If True, prefer ROCm FlashAttention over PyTorch SDPA
        """
        self._backend = None
        self._device = None
        self._device_name = None

        if ROCM_FLASH_AVAILABLE and prefer_flash:
            self._init_flash_attn(device_index)
        elif PYTORCH_ROCM_AVAILABLE:
            self._init_pytorch_rocm(device_index)
        elif HIP_AVAILABLE:
            self._init_hip_mfma(device_index)
        else:
            raise RuntimeError(
                "No ROCm backend available. Install one of:\n"
                "  1. ROCm FlashAttention: https://github.com/ROCm/flash-attention\n"
                "  2. PyTorch with ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0\n"
                "  3. HIP Python: pip install hip-python (requires ROCm)"
            )

    def _init_flash_attn(self, device_index: int):
        """Initialize ROCm FlashAttention (Composable Kernel) backend."""
        import torch
        self._device = torch.device(f'cuda:{device_index}')
        torch.cuda.set_device(self._device)
        self._device_name = torch.cuda.get_device_name(device_index)
        self._backend = 'flash_attn_ck'
        print(f"ROCm FlashAttention (CK) initialized on: {self._device_name}")

    def _init_pytorch_rocm(self, device_index: int):
        """Initialize PyTorch SDPA on ROCm backend."""
        import torch
        self._device = torch.device(f'cuda:{device_index}')
        torch.cuda.set_device(self._device)
        self._device_name = torch.cuda.get_device_name(device_index)
        self._backend = 'pytorch_sdpa'
        print(f"PyTorch SDPA (ROCm) initialized on: {self._device_name}")

    def _init_hip_mfma(self, device_index: int):
        """Initialize our HIP MFMA kernel backend."""
        from aule_hip_mfma import MFMAAttention
        self._impl = MFMAAttention(device_index)
        self._device_name = self._impl.device_name
        self._backend = 'hip_mfma'
        print(f"HIP MFMA kernel initialized on: {self._device_name}")

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        names = {
            'flash_attn_ck': f'ROCm FlashAttention CK ({self._device_name})',
            'pytorch_sdpa': f'PyTorch SDPA ROCm ({self._device_name})',
            'hip_mfma': f'HIP MFMA ({self._device_name})',
        }
        return names.get(self._backend, self._backend)

    @property
    def expected_tflops(self) -> str:
        """Expected performance range."""
        ranges = {
            'flash_attn_ck': '150-300 TFLOPS',
            'pytorch_sdpa': '50-150 TFLOPS',
            'hip_mfma': '50-100 TFLOPS',
        }
        return ranges.get(self._backend, 'Unknown')

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: Optional[float] = None,
        causal: bool = False,
        dtype: Literal['float32', 'float16', 'bfloat16'] = 'float16'
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor [batch, heads, seq_len, head_dim]
            K: Key tensor [batch, heads, seq_len, head_dim]
            V: Value tensor [batch, heads, seq_len, head_dim]
            scale: Attention scale (default: 1/sqrt(head_dim))
            causal: Apply causal masking
            dtype: 'float16', 'bfloat16', or 'float32'

        Returns:
            Output tensor [batch, heads, seq_len, head_dim]
        """
        if self._backend == 'flash_attn_ck':
            return self._forward_flash_attn(Q, K, V, scale, causal, dtype)
        elif self._backend == 'pytorch_sdpa':
            return self._forward_pytorch_sdpa(Q, K, V, scale, causal, dtype)
        elif self._backend == 'hip_mfma':
            return self._forward_hip_mfma(Q, K, V, scale, causal, dtype)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def _forward_flash_attn(
        self, Q, K, V, scale, causal, dtype
    ) -> np.ndarray:
        """Forward pass using ROCm FlashAttention (CK)."""
        import torch
        from flash_attn import flash_attn_func

        # Determine torch dtype
        torch_dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }.get(dtype, torch.float16)

        # Convert to torch tensors
        # flash_attn expects [batch, seq, heads, head_dim] format
        Q_t = torch.from_numpy(Q).to(device=self._device, dtype=torch_dtype)
        K_t = torch.from_numpy(K).to(device=self._device, dtype=torch_dtype)
        V_t = torch.from_numpy(V).to(device=self._device, dtype=torch_dtype)

        # Transpose from [batch, heads, seq, dim] to [batch, seq, heads, dim]
        Q_t = Q_t.transpose(1, 2).contiguous()
        K_t = K_t.transpose(1, 2).contiguous()
        V_t = V_t.transpose(1, 2).contiguous()

        # Compute scale
        if scale is None:
            scale = 1.0 / (Q.shape[-1] ** 0.5)

        # Call FlashAttention
        output = flash_attn_func(
            Q_t, K_t, V_t,
            softmax_scale=scale,
            causal=causal
        )

        # Transpose back to [batch, heads, seq, dim]
        output = output.transpose(1, 2).contiguous()

        return output.cpu().numpy()

    def _forward_pytorch_sdpa(
        self, Q, K, V, scale, causal, dtype
    ) -> np.ndarray:
        """Forward pass using PyTorch SDPA on ROCm."""
        import torch
        import torch.nn.functional as F

        torch_dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }.get(dtype, torch.float16)

        Q_t = torch.from_numpy(Q).to(device=self._device, dtype=torch_dtype)
        K_t = torch.from_numpy(K).to(device=self._device, dtype=torch_dtype)
        V_t = torch.from_numpy(V).to(device=self._device, dtype=torch_dtype)

        if scale is None:
            scale = 1.0 / (Q.shape[-1] ** 0.5)

        with torch.no_grad():
            output = F.scaled_dot_product_attention(
                Q_t, K_t, V_t,
                scale=scale,
                is_causal=causal
            )

        return output.cpu().numpy()

    def _forward_hip_mfma(
        self, Q, K, V, scale, causal, dtype
    ) -> np.ndarray:
        """Forward pass using our HIP MFMA kernel."""
        # Map dtype
        hip_dtype = 'float16' if dtype in ('float16', 'bfloat16') else 'float32'
        return self._impl.forward(Q, K, V, scale=scale, causal=causal, dtype=hip_dtype)

    def close(self):
        """Release resources."""
        if hasattr(self, '_impl') and self._impl:
            self._impl.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def benchmark_rocm_backends(
    seq_len: int = 2048,
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype: str = 'float16',
    n_iters: int = 100
) -> dict:
    """
    Benchmark all available ROCm backends.

    Returns dict with timing and TFLOPS for each backend.
    """
    import time

    results = {}

    # Generate test data
    np.random.seed(42)
    np_dtype = np.float16 if dtype == 'float16' else np.float32
    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)

    # FLOPS calculation
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim

    backends_to_test = []

    if ROCM_FLASH_AVAILABLE:
        backends_to_test.append(('flash_attn_ck', True))

    if PYTORCH_ROCM_AVAILABLE:
        backends_to_test.append(('pytorch_sdpa', False))

    if HIP_AVAILABLE:
        backends_to_test.append(('hip_mfma', None))

    for backend_name, prefer_flash in backends_to_test:
        try:
            if backend_name == 'hip_mfma':
                from aule_hip_mfma import MFMAAttention
                attn = MFMAAttention()
                # Warmup
                _ = attn.forward(Q, K, V, causal=True, dtype=dtype)
                # Benchmark
                start = time.perf_counter()
                for _ in range(n_iters):
                    _ = attn.forward(Q, K, V, causal=True, dtype=dtype)
                elapsed = time.perf_counter() - start
                attn.close()
            else:
                attn = ROCmAttention(prefer_flash=prefer_flash)
                # Warmup
                _ = attn.forward(Q, K, V, causal=True, dtype=dtype)
                # Benchmark
                start = time.perf_counter()
                for _ in range(n_iters):
                    _ = attn.forward(Q, K, V, causal=True, dtype=dtype)
                elapsed = time.perf_counter() - start
                attn.close()

            time_ms = elapsed / n_iters * 1000
            tflops = flops / (elapsed / n_iters) / 1e12

            results[backend_name] = {
                'time_ms': time_ms,
                'tflops': tflops,
                'status': 'OK'
            }
        except Exception as e:
            results[backend_name] = {
                'time_ms': 0,
                'tflops': 0,
                'status': f'FAILED: {e}'
            }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("AULE-ATTENTION ROCm Backend Status")
    print("=" * 70)

    info = get_rocm_info()
    print(f"\nROCm Available: {info['rocm_available']}")
    print(f"  PyTorch ROCm: {info['pytorch_rocm']}")
    print(f"  FlashAttention CK: {info['flash_attn_ck']}")
    print(f"  HIP Python: {info['hip_python']}")
    if info['device_name']:
        print(f"  Device: {info['device_name']}")
    if info['device_arch']:
        print(f"  Architecture: {info['device_arch']}")
    if info['rocm_version']:
        print(f"  ROCm Version: {info['rocm_version']}")

    if is_rocm_available():
        print("\n" + "-" * 70)
        print("Running benchmark...")
        print("-" * 70)

        results = benchmark_rocm_backends(seq_len=2048, n_iters=10)

        print(f"\n{'Backend':<25} {'Time (ms)':<15} {'TFLOPS':<15} {'Status'}")
        print("-" * 70)
        for backend, data in results.items():
            print(f"{backend:<25} {data['time_ms']:<15.2f} {data['tflops']:<15.2f} {data['status']}")
    else:
        print("\nNo ROCm backend available.")
        print("To enable ROCm support, install one of:")
        print("  1. ROCm + PyTorch: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
        print("  2. ROCm FlashAttention: https://github.com/ROCm/flash-attention")
        print("  3. HIP Python: pip install hip-python")

    print("\n" + "=" * 70)
