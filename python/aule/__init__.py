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

import logging
import warnings

__version__ = "0.5.0"
__author__ = "Aule Technologies"

# Configure logger for the module
logger = logging.getLogger(__name__)

# Backend availability flags
_triton_available = False
_triton_amd_available = False
_vulkan_available = False
_cpu_available = True
_is_amd_gpu = False

# Track backend initialization errors for debugging
_backend_errors = {}

# Detect AMD GPU
def _detect_amd_gpu():
    """Detect if running on AMD GPU with ROCm."""
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return True
    except Exception as e:
        logger.debug(f"AMD GPU detection failed: {e}")
    return False

_is_amd_gpu = _detect_amd_gpu()

# Try AMD-optimized Triton backend first (for MI300X, MI200, RDNA3)
if _is_amd_gpu:
    try:
        from .triton_flash_amd import flash_attention_amd as _flash_attention_amd
        from .triton_flash_amd import flash_attention_paged_amd
        from .triton_flash_amd import get_amd_gpu_arch as _get_amd_gpu_arch
        _triton_amd_available = True
        logger.debug("AMD Triton backend loaded successfully")
    except ImportError as e:
        _backend_errors['triton-amd'] = str(e)
        logger.debug(f"AMD Triton backend failed to load: {e}")
        flash_attention_paged_amd = None

# Try generic Triton backend (for NVIDIA and fallback)
try:
    from .triton_flash import (
        flash_attention_triton,
        flash_attention_rope,
        is_triton_available,
        precompute_rope_frequencies,
        apply_rope_separate,
    )
    _triton_available = is_triton_available()
    if _triton_available:
        logger.debug("Generic Triton backend loaded successfully")
    else:
        _backend_errors['triton'] = "Triton not available on this system"
except ImportError as e:
    _backend_errors['triton'] = str(e)
    logger.debug(f"Generic Triton backend failed to load: {e}")
    flash_attention_rope = None
    precompute_rope_frequencies = None
    apply_rope_separate = None

# Try Vulkan backend (for consumer GPUs without ROCm/CUDA)
try:
    from .vulkan import Aule, GpuTensor, attention as vulkan_attention, AuleError
    _vulkan_available = True
    logger.debug("Vulkan backend loaded successfully")
except Exception as e:
    _backend_errors['vulkan'] = str(e)
    logger.debug(f"Vulkan backend failed to load: {e}")

try:
    from .patching import patch_model
except ImportError as e:
    logger.debug(f"Patching module failed to load: {e}")


def flash_attention(query, key, value, rot_cos=None, rot_sin=None, causal=True, scale=None, window_size=-1):
    """
    FlashAttention-2 implementation.

    Automatically selects the best backend:
    - Triton: AMD ROCm (MI200/MI300/RDNA3), NVIDIA CUDA
    - Vulkan: Consumer AMD/NVIDIA/Intel/Apple GPUs
    - CPU: Fallback

    Args:
        query: [batch, heads_q, seq_len_q, head_dim] - torch.Tensor or numpy.ndarray
        key: [batch, heads_kv, seq_len_k, head_dim]
        value: [batch, heads_kv, seq_len_k, head_dim]
        rot_cos: Optional RoPE cos values [seq_len, head_dim//2]
        rot_sin: Optional RoPE sin values [seq_len, head_dim//2]
        causal: Apply causal masking (default True for LLMs)
        scale: Attention scale (default 1/sqrt(head_dim))
        window_size: Sliding window size (-1 for full attention)

    Returns:
        Output tensor with same shape as query

    Raises:
        ValueError: If input shapes are invalid
    """
    import numpy as np

    # Check if input is PyTorch tensor
    is_torch = False
    try:
        import torch
        is_torch = isinstance(query, torch.Tensor)
    except ImportError:
        pass

    # Input validation
    if query.ndim != 4:
        raise ValueError(f"query must be 4D [batch, heads, seq_len, head_dim], got shape {query.shape}")
    if key.ndim != 4:
        raise ValueError(f"key must be 4D [batch, heads, seq_len, head_dim], got shape {key.shape}")
    if value.ndim != 4:
        raise ValueError(f"value must be 4D [batch, heads, seq_len, head_dim], got shape {value.shape}")

    batch_q, heads_q, seq_q, head_dim_q = query.shape
    batch_k, heads_kv, seq_k, head_dim_k = key.shape
    batch_v, heads_v, seq_v, head_dim_v = value.shape

    if batch_q != batch_k or batch_q != batch_v:
        raise ValueError(f"Batch size mismatch: query={batch_q}, key={batch_k}, value={batch_v}")
    if head_dim_q != head_dim_k or head_dim_q != head_dim_v:
        raise ValueError(f"head_dim mismatch: query={head_dim_q}, key={head_dim_k}, value={head_dim_v}")
    if seq_k != seq_v:
        raise ValueError(f"Key/value seq_len mismatch: key={seq_k}, value={seq_v}")
    if heads_kv != heads_v:
        raise ValueError(f"Key/value heads mismatch: key={heads_kv}, value={heads_v}")
    if heads_q % heads_kv != 0:
        raise ValueError(f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv}) for GQA")

    # Determine which backend to use
    use_triton = False
    use_vulkan = False
    use_cpu = False
    backend_name = None

    if _forced_backend == 'triton':
        if _triton_available:
            use_triton = True
            backend_name = 'triton (forced)'
        else:
            raise RuntimeError("Triton backend forced but not available")
    elif _forced_backend == 'vulkan':
        if _vulkan_available:
            use_vulkan = True
            backend_name = 'vulkan (forced)'
        else:
            raise RuntimeError("Vulkan backend forced but not available")
    elif _forced_backend == 'cpu':
        use_cpu = True
        backend_name = 'cpu (forced)'
    else:
        # Auto-select
        if is_torch and _triton_available and query.is_cuda:
            use_triton = True
            backend_name = 'triton'
        elif _vulkan_available:
            use_vulkan = True
            backend_name = 'vulkan'
        else:
            use_cpu = True
            backend_name = 'cpu'

    if _verbose:
        shape = tuple(query.shape)
        print(f"aule-attention: {backend_name} | shape={shape} | causal={causal}")

    if is_torch:
        if use_triton:
            # Use AMD-optimized kernel on AMD GPUs for maximum performance
            if _triton_amd_available and _is_amd_gpu:
                from .triton_flash_amd import flash_attention_amd
                return flash_attention_amd(query, key, value, causal=causal, scale=scale, window_size=window_size)
            else:
                from .triton_flash import flash_attention_triton
                return flash_attention_triton(query, key, value, causal=causal, scale=scale, window_size=window_size)

        if use_vulkan:
            q_np = query.cpu().numpy()
            k_np = key.cpu().numpy()
            v_np = value.cpu().numpy()

            # RoPE not yet supported in Vulkan backend
            if rot_cos is not None or rot_sin is not None:
                warnings.warn("RoPE not yet supported in Vulkan backend, ignoring rot_cos/rot_sin", stacklevel=2)
            if window_size > 0:
                warnings.warn("Sliding window not yet supported in Vulkan backend, using full attention", stacklevel=2)

            out_np = vulkan_attention(q_np, k_np, v_np, causal=causal)
            return torch.from_numpy(out_np).to(query.device)

        # CPU fallback
        if rot_cos is not None:
            warnings.warn("RoPE not supported on CPU fallback yet", stacklevel=2)
        if window_size > 0:
            warnings.warn("Sliding window not supported on CPU fallback, using full attention", stacklevel=2)
        out_np = _cpu_attention(query.cpu().numpy(), key.cpu().numpy(), value.cpu().numpy(), causal)
        return torch.from_numpy(out_np).to(query.device)

    else:
        # NumPy input
        if use_vulkan:
            # RoPE not yet supported in Vulkan backend
            if rot_cos is not None or rot_sin is not None:
                warnings.warn("RoPE not yet supported in Vulkan backend, ignoring rot_cos/rot_sin", stacklevel=2)
            if window_size > 0:
                warnings.warn("Sliding window not yet supported in Vulkan backend, using full attention", stacklevel=2)
            return vulkan_attention(query, key, value, causal=causal)
        if rot_cos is not None:
            warnings.warn("RoPE not supported on CPU fallback yet", stacklevel=2)
        if window_size > 0:
            warnings.warn("Sliding window not supported on CPU fallback, using full attention", stacklevel=2)
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


# =============================================================================
# PyTorch SDPA Compatibility Layer
# =============================================================================

_original_sdpa = None
_installed = False
_forced_backend = None  # None = auto, 'triton', 'vulkan', 'cpu'
_verbose = False


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """
    Drop-in replacement for torch.nn.functional.scaled_dot_product_attention.

    Uses aule-attention backends (Triton/Vulkan/CPU) while maintaining
    full API compatibility with PyTorch's SDPA.

    Args:
        query: [batch, heads, seq_len_q, head_dim]
        key: [batch, heads_kv, seq_len_k, head_dim]
        value: [batch, heads_kv, seq_len_k, head_dim]
        attn_mask: Optional attention mask (falls back to PyTorch if provided)
        dropout_p: Dropout probability (falls back to PyTorch if > 0)
        is_causal: Apply causal masking
        scale: Attention scale (default 1/sqrt(head_dim))
        enable_gqa: Enable grouped query attention (handled automatically)

    Returns:
        Output tensor [batch, heads, seq_len_q, head_dim]
    """
    import torch

    # Check head_dim - Vulkan backend limited to 64, fall back if larger
    head_dim = query.shape[-1]
    needs_fallback = (
        attn_mask is not None or
        dropout_p > 0.0 or
        (head_dim > 64 and not _triton_available)  # Triton handles any head_dim
    )

    # Fall back to PyTorch for unsupported features
    if needs_fallback:
        if _original_sdpa is not None:
            return _original_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        else:
            # Manual fallback if original not saved
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

    # Use aule-attention
    return flash_attention(query, key, value, causal=is_causal, scale=scale)


def install(backend=None, verbose=False):
    """
    Install aule-attention as the default PyTorch attention backend.

    After calling this, all models using torch.nn.functional.scaled_dot_product_attention
    will automatically use aule-attention (Triton on ROCm/CUDA, Vulkan on consumer GPUs).

    Args:
        backend: Force a specific backend. Options:
                 - None (default): Auto-select best available
                 - 'triton': Force Triton backend (requires CUDA)
                 - 'vulkan': Force Vulkan backend
                 - 'cpu': Force CPU/NumPy backend
        verbose: If True, print which backend is used for each attention call

    Usage:
        import aule
        aule.install()  # Auto-select

        # Or force a specific backend:
        aule.install(backend='vulkan')
        aule.install(backend='triton', verbose=True)

    Works with:
        - ComfyUI (Stable Diffusion, SDXL, Flux, SD3)
        - Hugging Face Transformers
        - Any PyTorch model using F.scaled_dot_product_attention
    """
    global _original_sdpa, _installed, _forced_backend, _verbose

    import torch
    import torch.nn.functional as F

    # Validate backend option
    if backend is not None and backend not in ('triton', 'vulkan', 'cpu'):
        raise ValueError(f"Invalid backend '{backend}'. Choose from: 'triton', 'vulkan', 'cpu', or None (auto)")

    if _installed:
        # Allow changing backend/verbose on reinstall
        _forced_backend = backend
        _verbose = verbose
        print(f"aule-attention: Updated (backend={backend or 'auto'}, verbose={verbose})")
        return

    # Save original for fallback
    _original_sdpa = F.scaled_dot_product_attention

    # Install our version
    F.scaled_dot_product_attention = scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention

    _installed = True
    _forced_backend = backend
    _verbose = verbose

    # Report what backend will be used
    if backend:
        backend_name = backend.capitalize()
    else:
        backends = get_available_backends()
        if 'triton' in backends:
            backend_name = "auto: Triton"
        elif 'vulkan' in backends:
            backend_name = "auto: Vulkan"
        else:
            backend_name = "auto: CPU"

    verbose_str = ", verbose" if verbose else ""
    print(f"aule-attention: Installed ({backend_name}{verbose_str})")


def uninstall():
    """
    Restore the original PyTorch SDPA implementation.
    """
    global _original_sdpa, _installed

    if not _installed:
        print("aule-attention: Not installed")
        return

    import torch
    import torch.nn.functional as F

    if _original_sdpa is not None:
        F.scaled_dot_product_attention = _original_sdpa
        torch.nn.functional.scaled_dot_product_attention = _original_sdpa

    _installed = False
    print("aule-attention: Uninstalled, restored PyTorch SDPA")


def get_available_backends():
    """Return list of available backends."""
    backends = []
    if _triton_amd_available:
        backends.append('triton-amd')
    if _triton_available:
        backends.append('triton')
    if _vulkan_available:
        backends.append('vulkan')
    if _cpu_available:
        backends.append('cpu')
    return backends


def get_backend_errors():
    """
    Return dict of backend initialization errors for debugging.

    Useful when a backend you expect to work isn't available.

    Example:
        >>> import aule
        >>> errors = aule.get_backend_errors()
        >>> print(errors)
        {'vulkan': "Could not find aule library..."}
    """
    return dict(_backend_errors)


def get_backend_info():
    """Return detailed backend information."""
    info = {}

    if _triton_amd_available:
        import torch
        arch = _get_amd_gpu_arch() if '_get_amd_gpu_arch' in dir() else 'unknown'
        info['triton-amd'] = {
            'available': True,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'architecture': arch,
            'description': 'AMD-optimized Triton FlashAttention-2 (MI300X tuned)',
        }

    if _triton_available:
        import torch
        info['triton'] = {
            'available': True,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'description': 'Triton FlashAttention-2 (generic kernel)',
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

    idx = 1
    if _triton_amd_available:
        import torch
        arch = _get_amd_gpu_arch()
        print(f"[{idx}] TRITON-AMD (primary)")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Architecture: {arch}")
        print("    Status: AMD-optimized FlashAttention-2 (MI300X tuned)")
        print()
        idx += 1

    if _triton_available:
        import torch
        print(f"[{idx}] TRITON")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print("    Status: Generic FlashAttention-2 kernel")
        print()
        idx += 1

    if _vulkan_available:
        try:
            with Aule() as aule:
                info = aule.get_device_info()
                print(f"[{idx}] VULKAN")
                print(f"    GPU: {info.get('device_name', 'Unknown')}")
                print("    Status: Vulkan compute shader")
                print()
                idx += 1
        except:
            pass

    print(f"[{idx}] CPU")
    print("    Status: NumPy fallback")
    print()
    print("=" * 60)


# Public API
__all__ = [
    # Core API
    "flash_attention",
    "attention",
    "scaled_dot_product_attention",
    # Fused RoPE + Attention
    "flash_attention_rope",
    "precompute_rope_frequencies",
    "apply_rope_separate",
    # PagedAttention (vLLM-compatible)
    "flash_attention_paged_amd",
    # Installation (for ComfyUI, etc.)
    "install",
    "uninstall",
    # Backend info
    "get_available_backends",
    "get_backend_errors",
    "get_backend_info",
    "print_backend_info",
    # Vulkan extras (if available)
    "Aule",
    "GpuTensor",
    "AuleError",
    # Patching (legacy)
    "patch_model",
    # Version
    "__version__",
]
