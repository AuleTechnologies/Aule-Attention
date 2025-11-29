"""
aule-attention: Hardware-agnostic FlashAttention implementation

Works on AMD, NVIDIA, Intel, and Apple GPUs via Vulkan/OpenCL.
No ROCm, CUDA, or compilation required at install time.

Backends (in order of preference):
1. Vulkan (via libaule.so) - AMD-optimized 64-wide wavefront, works on all GPUs
2. OpenCL (via Mesa Rusticl) - Fallback for systems without Vulkan
3. CPU (NumPy) - Always available fallback

Usage:
    from aule import attention, flash_attention

    # Simple API (auto-selects best backend)
    output = attention(Q, K, V, causal=True)

    # LLM-optimized (causal=True by default)
    output = flash_attention(Q, K, V)

    # Vulkan backend directly (fastest, AMD-optimized)
    from aule import Aule
    with Aule() as aule:
        output = aule.attention(Q, K, V, causal=True)

    # GPU tensor API (zero-copy, for repeated ops)
    with Aule() as aule:
        q_gpu = aule.tensor((batch, heads, seq, dim))
        q_gpu.upload(Q)
        aule.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=True)
        result = out_gpu.download()

    # Check device info
    from aule import Aule
    with Aule() as aule:
        print(aule.get_device_info())

    # PyTorch integration (requires torch)
    from aule import aule_attention
    out = aule_attention(Q_tensor, K_tensor, V_tensor, causal=True)
    out.sum().backward()  # Gradients computed!
"""

__version__ = "0.1.0"
__author__ = "Aule Technologies"

# Try Vulkan backend first (fastest, AMD-optimized)
_vulkan_available = False
try:
    from .vulkan import Aule, GpuTensor, attention, flash_attention, AuleError
    _vulkan_available = True
except Exception:
    pass

# Fall back to unified backend (OpenCL/CPU)
if not _vulkan_available:
    from .unified import (
        Attention as Aule,
        attention,
    )
    # Add flash_attention alias
    def flash_attention(query, key, value, causal=True):
        """FlashAttention with causal masking (default for LLMs)."""
        return attention(query, key, value, causal=causal)

# Import backend info functions
from .unified import (
    get_available_backends,
    get_backend_info,
    print_backend_info,
)

# Try to import PyTorch integration
try:
    from .autograd import aule_attention, AuleAttention, AuleAttentionFunction
except ImportError:
    pass

# Public API
__all__ = [
    # Core API
    "Aule",
    "attention",
    "flash_attention",
    # Vulkan extras (if available)
    "GpuTensor",
    "AuleError",
    # Backend info
    "get_available_backends",
    "get_backend_info",
    "print_backend_info",
    # PyTorch integration (if available)
    "aule_attention",
    "AuleAttention",
    "AuleAttentionFunction",
    # Version info
    "__version__",
    "_vulkan_available",
]
