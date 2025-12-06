"""
aule-attention: Hardware-agnostic FlashAttention implementation

pip install aule-attention - works everywhere, no compilation.

Backends (in order of preference):
1. Triton (AMD MI200/MI300, RDNA3, NVIDIA) - Our FlashAttention-2 kernel
2. Vulkan (via libaule.so) - Consumer GPUs without ROCm/CUDA
3. CPU (NumPy) - Always available fallback

Usage:
    from aule import flash_attention
    import torch

    q = torch.randn(1, 8, 512, 64, device='cuda')
    k = torch.randn(1, 8, 512, 64, device='cuda')
    v = torch.randn(1, 8, 512, 64, device='cuda')

    # Just works - uses best available backend
    out = flash_attention(q, k, v, causal=True)
"""

__version__ = "0.2.0"
__author__ = "Aule Technologies"

# Backend availability flags
_triton_available = False
_vulkan_available = False
_cpu_available = True

# Try Triton backend first (best for ROCm and CUDA)
try:
    from .triton_flash import flash_attention_triton, is_triton_available
    _triton_available = is_triton_available()
except ImportError:
    pass

# Try Vulkan backend (for consumer GPUs without ROCm/CUDA)
try:
    from .vulkan import Aule, GpuTensor, attention as vulkan_attention, AuleError
    _vulkan_available = True
except Exception:
    pass


def flash_attention(query, key, value, causal=True, scale=None):
    """
    FlashAttention-2 implementation.

    Automatically selects the best backend:
    - Triton: AMD ROCm (MI200/MI300/RDNA3), NVIDIA CUDA
    - Vulkan: Consumer AMD/NVIDIA/Intel/Apple GPUs
    - CPU: Fallback

    Args:
        query: [batch, heads, seq_len_q, head_dim] - torch.Tensor or numpy.ndarray
        key: [batch, heads, seq_len_k, head_dim]
        value: [batch, heads, seq_len_k, head_dim]
        causal: Apply causal masking (default True for LLMs)
        scale: Attention scale (default 1/sqrt(head_dim))

    Returns:
        Output tensor with same shape as query
    """
    import numpy as np

    # Check if input is PyTorch tensor
    is_torch = False
    try:
        import torch
        is_torch = isinstance(query, torch.Tensor)
    except ImportError:
        pass

    if is_torch:
        # Use Triton if available and on CUDA
        if _triton_available and query.is_cuda:
            from .triton_flash import flash_attention_triton
            return flash_attention_triton(query, key, value, causal=causal, scale=scale)

        # Fall back to Vulkan (will use CPU if no GPU Vulkan)
        if _vulkan_available:
            q_np = query.cpu().numpy()
            k_np = key.cpu().numpy()
            v_np = value.cpu().numpy()
            out_np = vulkan_attention(q_np, k_np, v_np, causal=causal)
            return torch.from_numpy(out_np).to(query.device)

        # CPU fallback
        return _cpu_attention(query.cpu().numpy(), key.cpu().numpy(), value.cpu().numpy(), causal)

    else:
        # NumPy input
        if _vulkan_available:
            return vulkan_attention(query, key, value, causal=causal)
        return _cpu_attention(query, key, value, causal)


def _cpu_attention(q, k, v, causal=True):
    """Pure NumPy attention fallback."""
    import numpy as np
    import math

    batch, heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores
    scores = np.einsum('bhqd,bhkd->bhqk', q, k) * scale

    # Causal mask
    if causal:
        mask = np.triu(np.ones((seq_q, seq_k)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Output
    return np.einsum('bhqk,bhkd->bhqd', attn_weights, v)


# Alias for compatibility
attention = flash_attention


def get_available_backends():
    """Return list of available backends."""
    backends = []
    if _triton_available:
        backends.append('triton')
    if _vulkan_available:
        backends.append('vulkan')
    if _cpu_available:
        backends.append('cpu')
    return backends


def get_backend_info():
    """Return detailed backend information."""
    info = {}

    if _triton_available:
        import torch
        info['triton'] = {
            'available': True,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'description': 'Triton FlashAttention-2 (our kernel)',
        }

    if _vulkan_available:
        try:
            with Aule() as aule:
                dev_info = aule.get_device_info()
                info['vulkan'] = {
                    'available': True,
                    'device': dev_info.get('device_name', 'Unknown'),
                    'description': 'Vulkan compute (our kernel)',
                }
        except:
            info['vulkan'] = {'available': True, 'device': 'Unknown'}

    info['cpu'] = {
        'available': True,
        'description': 'NumPy fallback',
    }

    return info


def print_backend_info():
    """Print backend status."""
    print("=" * 60)
    print("AULE-ATTENTION v" + __version__)
    print("=" * 60)
    print()

    backends = get_available_backends()
    print(f"Available backends: {backends}")
    print()

    if _triton_available:
        import torch
        print("[1] TRITON (primary)")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print("    Status: OUR FlashAttention-2 kernel")
        print()

    if _vulkan_available:
        try:
            with Aule() as aule:
                info = aule.get_device_info()
                print(f"[{'2' if _triton_available else '1'}] VULKAN")
                print(f"    GPU: {info.get('device_name', 'Unknown')}")
                print("    Status: OUR Vulkan compute shader")
                print()
        except:
            pass

    print(f"[{'3' if _triton_available else '2'}] CPU")
    print("    Status: NumPy fallback")
    print()
    print("=" * 60)


# Public API
__all__ = [
    # Core API
    "flash_attention",
    "attention",
    # Backend info
    "get_available_backends",
    "get_backend_info",
    "print_backend_info",
    # Vulkan extras (if available)
    "Aule",
    "GpuTensor",
    "AuleError",
    # Version
    "__version__",
]
