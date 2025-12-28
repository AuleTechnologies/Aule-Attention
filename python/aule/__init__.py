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
_is_datacenter_gpu = False
_detected_gpu_type = None  # 'datacenter', 'consumer', or None
_hardware_optimized = False

# Track backend initialization errors for debugging
_backend_errors = {}


# =============================================================================
# GPU DETECTION FUNCTIONS
# =============================================================================

def _detect_amd_gpu():
    """Detect if running on AMD GPU with ROCm."""
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return True
    except Exception as e:
        logger.debug(f"AMD GPU detection failed: {e}")
    return False


def _detect_gpu_type():
    """
    Detect GPU type and return classification.

    Returns:
        tuple: (is_datacenter, gpu_type_str, gpu_name)
            - is_datacenter: True for datacenter GPUs (MI300X, MI250, A100, H100, etc.)
            - gpu_type_str: 'datacenter', 'consumer', or 'unknown'
            - gpu_name: Human-readable GPU name
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 'unknown', 'No CUDA device'

        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name.lower()

        # AMD Datacenter GPUs (MI series - Instinct)
        amd_datacenter_patterns = [
            'mi300', 'mi250', 'mi210', 'mi200', 'mi100', 'mi60', 'mi50',
            'instinct',  # AMD Instinct series
        ]

        # AMD Consumer GPUs (Radeon RX series)
        amd_consumer_patterns = [
            'rx 7', 'rx 6', 'rx 5', 'radeon',
            'gfx11', 'gfx10',  # RDNA architecture IDs
            'navi',  # RDNA codenames
        ]

        # NVIDIA Datacenter GPUs
        nvidia_datacenter_patterns = [
            'a100', 'a800', 'h100', 'h200', 'h800',
            'a30', 'a40', 'a10', 'a16',
            'v100', 'p100', 't4',
            'l40', 'l4',
            'b100', 'b200',  # Blackwell
            'dgx', 'hgx',  # NVIDIA HPC systems
        ]

        # NVIDIA Consumer GPUs
        nvidia_consumer_patterns = [
            'geforce', 'rtx 20', 'rtx 30', 'rtx 40', 'rtx 50',
            'gtx', 'titan',
            'quadro',  # Some Quadro might be prosumer
        ]

        # Check AMD datacenter first
        for pattern in amd_datacenter_patterns:
            if pattern in gpu_name:
                return True, 'datacenter', props.name

        # Check NVIDIA datacenter
        for pattern in nvidia_datacenter_patterns:
            if pattern in gpu_name:
                return True, 'datacenter', props.name

        # Check AMD consumer
        for pattern in amd_consumer_patterns:
            if pattern in gpu_name:
                return False, 'consumer', props.name

        # Check NVIDIA consumer
        for pattern in nvidia_consumer_patterns:
            if pattern in gpu_name:
                return False, 'consumer', props.name

        # Heuristic: high memory bandwidth often indicates datacenter GPU
        # MI300X: ~5.3 TB/s, A100: ~2 TB/s, Consumer: <1 TB/s typically
        # Use compute capability as another hint
        major, minor = props.major, props.minor
        total_memory_gb = props.total_memory / (1024**3)

        # High memory (>40GB) usually means datacenter
        if total_memory_gb > 40:
            return True, 'datacenter', props.name

        return False, 'unknown', props.name

    except Exception as e:
        logger.debug(f"GPU type detection failed: {e}")
        return False, 'unknown', 'Detection failed'


def _get_optimal_backend():
    """
    Determine the optimal backend for the detected hardware.

    Returns:
        str: 'triton', 'triton-amd', 'vulkan', or 'cpu'
    """
    global _is_datacenter_gpu, _detected_gpu_type

    is_datacenter, gpu_type, gpu_name = _detect_gpu_type()
    _is_datacenter_gpu = is_datacenter
    _detected_gpu_type = gpu_type

    if is_datacenter:
        # Datacenter GPUs should always prefer Triton for best performance
        if _is_amd_gpu and _triton_amd_available:
            logger.info(f"Datacenter GPU detected ({gpu_name}): Using AMD-optimized Triton backend")
            return 'triton-amd'
        elif _triton_available:
            logger.info(f"Datacenter GPU detected ({gpu_name}): Using Triton backend")
            return 'triton'
        elif _vulkan_available:
            # Fallback - shouldn't happen on properly configured datacenter
            logger.warning(
                f"Datacenter GPU detected ({gpu_name}) but Triton not available. "
                "Falling back to Vulkan. For optimal performance, install Triton: pip install triton"
            )
            return 'vulkan'
    else:
        # Consumer GPUs - auto-select based on availability
        if _is_amd_gpu and _triton_amd_available:
            return 'triton-amd'
        elif _triton_available:
            return 'triton'
        elif _vulkan_available:
            return 'vulkan'

    return 'cpu'


def is_datacenter_gpu():
    """
    Check if running on a datacenter GPU.

    Datacenter GPUs include:
    - AMD: MI300X, MI250, MI200, MI100, Instinct series
    - NVIDIA: A100, H100, H200, V100, T4, L40, etc.

    Returns:
        bool: True if running on datacenter GPU
    """
    global _is_datacenter_gpu, _detected_gpu_type
    if _detected_gpu_type is None:
        _is_datacenter_gpu, _detected_gpu_type, _ = _detect_gpu_type()
    return _is_datacenter_gpu


def get_gpu_info():
    """
    Get detailed information about the detected GPU.

    Returns:
        dict: GPU information including type, name, and recommended backend
    """
    is_datacenter, gpu_type, gpu_name = _detect_gpu_type()
    optimal_backend = _get_optimal_backend()

    return {
        'name': gpu_name,
        'type': gpu_type,
        'is_datacenter': is_datacenter,
        'is_amd': _is_amd_gpu,
        'recommended_backend': optimal_backend,
        'triton_available': _triton_available,
        'triton_amd_available': _triton_amd_available,
        'vulkan_available': _vulkan_available,
    }


def optimize_for_hardware(verbose=True):
    """
    Optimize aule-attention for the detected hardware.

    This function should be called during initialization to ensure
    the best backend is selected for datacenter GPUs. It will:

    1. Detect the GPU type (datacenter vs consumer)
    2. Select the optimal backend (Triton for datacenter, auto for consumer)
    3. Warm up the selected backend for faster first inference

    For datacenter GPUs (MI300X, A100, H100, etc.), this ensures
    Triton is used instead of Vulkan for maximum performance.

    Args:
        verbose: If True, print hardware detection and backend info

    Returns:
        dict: Hardware info and selected backend

    Example:
        import aule
        info = aule.optimize_for_hardware()
        # {'name': 'AMD Instinct MI300X', 'type': 'datacenter',
        #  'recommended_backend': 'triton-amd', ...}
    """
    global _hardware_optimized, _forced_backend

    info = get_gpu_info()

    if verbose:
        print(f"aule-attention: Hardware optimization")
        print(f"  GPU: {info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Backend: {info['recommended_backend']}")

    # For datacenter GPUs, force the optimal backend
    if info['is_datacenter']:
        if info['recommended_backend'] in ('triton', 'triton-amd'):
            # Don't override if already forced
            if _forced_backend is None:
                _forced_backend = 'triton'
                if verbose:
                    print(f"  Status: Datacenter GPU detected, forcing Triton backend")
        elif info['recommended_backend'] == 'vulkan' and verbose:
            print(f"  Warning: Triton not available, using Vulkan (suboptimal for datacenter)")

    _hardware_optimized = True

    return info


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
        if _triton_available or _triton_amd_available:
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
        # Auto-select with datacenter GPU optimization
        # For datacenter GPUs, strongly prefer Triton over Vulkan
        is_dc = is_datacenter_gpu()

        if is_torch and query.is_cuda:
            # GPU tensor - prefer Triton, especially for datacenter
            if _triton_amd_available and _is_amd_gpu:
                use_triton = True
                backend_name = 'triton-amd' + (' (datacenter)' if is_dc else '')
            elif _triton_available:
                use_triton = True
                backend_name = 'triton' + (' (datacenter)' if is_dc else '')
            elif _vulkan_available:
                use_vulkan = True
                backend_name = 'vulkan'
                # Warn for datacenter GPUs using suboptimal Vulkan
                if is_dc:
                    logger.warning(
                        "Datacenter GPU detected but using Vulkan backend (suboptimal). "
                        "Install Triton for best performance: pip install triton"
                    )
            else:
                use_cpu = True
                backend_name = 'cpu'
        elif is_torch and not query.is_cuda:
            # CPU tensor in PyTorch - check if we should move to GPU
            if is_dc and (_triton_available or _triton_amd_available):
                # Datacenter: suggest moving to GPU
                logger.debug("CPU tensor on datacenter GPU - consider using .cuda() for best performance")
            if _vulkan_available:
                use_vulkan = True
                backend_name = 'vulkan'
            else:
                use_cpu = True
                backend_name = 'cpu'
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


def install(backend=None, verbose=False, auto_optimize=True):
    """
    Install aule-attention as the default PyTorch attention backend.

    After calling this, all models using torch.nn.functional.scaled_dot_product_attention
    will automatically use aule-attention (Triton on ROCm/CUDA, Vulkan on consumer GPUs).

    For datacenter GPUs (MI300X, A100, H100, etc.), this automatically optimizes
    the backend selection to use Triton for best performance.

    Args:
        backend: Force a specific backend. Options:
                 - None (default): Auto-select best available
                 - 'triton': Force Triton backend (requires CUDA)
                 - 'vulkan': Force Vulkan backend
                 - 'cpu': Force CPU/NumPy backend
        verbose: If True, print which backend is used for each attention call
        auto_optimize: If True (default), automatically detect datacenter GPUs
                      and optimize backend selection for best performance

    Usage:
        import aule
        aule.install()  # Auto-select with datacenter optimization

        # Or force a specific backend:
        aule.install(backend='vulkan')
        aule.install(backend='triton', verbose=True)

        # Disable auto-optimization:
        aule.install(auto_optimize=False)

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

    # Auto-optimize for datacenter GPUs if enabled and no explicit backend set
    if auto_optimize and backend is None:
        gpu_info = get_gpu_info()
        if gpu_info['is_datacenter']:
            # Datacenter GPU detected - ensure Triton is used
            if gpu_info['triton_available'] or gpu_info['triton_amd_available']:
                backend = 'triton'
                logger.info(f"Datacenter GPU ({gpu_info['name']}) detected, using Triton backend")
            else:
                logger.warning(
                    f"Datacenter GPU ({gpu_info['name']}) detected but Triton not available. "
                    "Install Triton for best performance: pip install triton"
                )

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
        if 'triton-amd' in backends:
            backend_name = "auto: Triton-AMD"
        elif 'triton' in backends:
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

    # Show GPU type detection
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"Type: {gpu_info['type'].upper()}" + (" (optimized for Triton)" if gpu_info['is_datacenter'] else ""))
    print(f"Recommended backend: {gpu_info['recommended_backend']}")
    print()

    backends = get_available_backends()
    print(f"Available backends: {backends}")
    print()

    idx = 1
    if _triton_amd_available:
        import torch
        arch = _get_amd_gpu_arch()
        is_recommended = gpu_info['recommended_backend'] == 'triton-amd'
        marker = " (recommended)" if is_recommended else ""
        print(f"[{idx}] TRITON-AMD{marker}")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Architecture: {arch}")
        print("    Status: AMD-optimized FlashAttention-2 (MI300X tuned)")
        print()
        idx += 1

    if _triton_available:
        import torch
        is_recommended = gpu_info['recommended_backend'] == 'triton'
        marker = " (recommended)" if is_recommended else ""
        print(f"[{idx}] TRITON{marker}")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print("    Status: Generic FlashAttention-2 kernel")
        print()
        idx += 1

    if _vulkan_available:
        try:
            with Aule() as aule:
                dev_info = aule.get_device_info()
                is_suboptimal = gpu_info['is_datacenter'] and gpu_info['recommended_backend'] == 'vulkan'
                marker = " (suboptimal for datacenter)" if is_suboptimal else ""
                print(f"[{idx}] VULKAN{marker}")
                print(f"    GPU: {dev_info.get('device_name', 'Unknown')}")
                print("    Status: Vulkan compute shader")
                if is_suboptimal:
                    print("    Warning: Install Triton for better datacenter performance")
                print()
                idx += 1
        except:
            pass

    print(f"[{idx}] CPU")
    print("    Status: NumPy fallback")
    print()

    # Summary recommendation for datacenter
    if gpu_info['is_datacenter'] and gpu_info['recommended_backend'] == 'vulkan':
        print("RECOMMENDATION: Datacenter GPU detected but Triton not available.")
        print("For optimal performance, install Triton: pip install triton")
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
    # Hardware detection & optimization
    "optimize_for_hardware",
    "is_datacenter_gpu",
    "get_gpu_info",
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
