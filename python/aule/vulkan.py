"""
aule-attention Python bindings

Provides Python interface to the Vulkan-based FlashAttention implementation.
Works on AMD, NVIDIA, Intel, and any GPU with Vulkan compute support.
"""

import os
import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Library path (resolved at import time)
_LIBRARY_PATH: Optional[Path] = None

# Global singleton to avoid reloading library on every Aule() call
_AULE_LIB_SINGLETON = None

# Track if aule_init() has been called
_AULE_INITIALIZED = False

# Global singleton Aule instance for standalone functions (shares tensor cache)
_AULE_INSTANCE_SINGLETON = None


def _find_library() -> Path:
    """Find the aule shared library."""
    import platform

    # Determine library name based on platform
    system = platform.system()
    if system == "Linux":
        lib_name = "libaule.so"
    elif system == "Darwin":
        lib_name = "libaule.dylib"
    elif system == "Windows":
        lib_name = "aule.dll"
    else:
        lib_name = "libaule.so"

    # Check common locations (package lib first, then dev paths)
    candidates = [
        # Installed package location
        Path(__file__).parent / "lib" / lib_name,
        # Development paths (lib for Linux/macOS, bin for Windows DLLs)
        Path(__file__).parent.parent.parent / "zig-out" / "lib" / lib_name,
        Path(__file__).parent.parent.parent / "zig-out" / "bin" / lib_name,
        Path(__file__).parent.parent / "zig-out" / "lib" / lib_name,
        Path(__file__).parent.parent / "zig-out" / "bin" / lib_name,
        # System paths
        Path("/usr/local/lib") / lib_name,
        Path("/usr/lib") / lib_name,
    ]

    for path in candidates:
            if os.path.exists(path):
                print(f"DEBUG: Loading library from {path}")
                return path

    raise RuntimeError(
        f"Could not find aule library ({lib_name}). "
        "Install with 'pip install aule-attention' or build with 'zig build'.\n"
        f"Searched: {[str(p) for p in candidates]}"
    )


class AuleError(Exception):
    """Exception raised for aule library errors."""
    pass


class GpuTensor:
    """
    A tensor that lives on the GPU.

    Data stays on GPU between operations - use this for repeated computations
    to avoid CPU<->GPU copy overhead.

    Example:
        >>> aule = Aule()
        >>> q = aule.tensor(shape=(1, 8, 64, 64))
        >>> q.upload(numpy_data)
        >>> # ... operations on GPU ...
        >>> result = output.download()
    """

    def __init__(self, aule: 'Aule', handle: int, shape: Tuple[int, ...], dtype: np.dtype = np.float32):
        self._aule = aule
        self._handle = handle
        self._shape = shape
        self._dtype = dtype
        self._size = int(np.prod(shape))

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    @property
    def handle(self) -> int:
        return self._handle

    def upload(self, data: np.ndarray) -> None:
        """Upload data from CPU to GPU."""
        if data.size != self._size:
            raise ValueError(f"Size mismatch: tensor has {self._size} elements, got {data.size}")

        # Preserve bits!
        data = np.ascontiguousarray(data, dtype=self._dtype).ravel()
        
        # Cast to float pointer because C API expects float*, but simply copies bits
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = self._aule._lib.aule_tensor_upload(
            ctypes.c_uint64(self._handle),
            ptr,
            ctypes.c_uint32(self._size)
        )
        if result != 0:
            error = self._aule._lib.aule_get_error()
            raise AuleError(f"Upload failed: {error.decode()}")

    def download(self) -> np.ndarray:
        """Download data from GPU to CPU."""
        # Use underlying dtype of tensor
        # C API treats storage as f32, so we download into a buffer and view it
        
        # Allocate buffer with matching size
        output = np.empty(self._size, dtype=self._dtype)
        
        # Cast pointer to float* for API
        ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = self._aule._lib.aule_tensor_download(
            ctypes.c_uint64(self._handle),
            ptr,
            ctypes.c_uint32(self._size)
        )
        if result != 0:
            error = self._aule._lib.aule_get_error()
            raise AuleError(f"Download failed: {error.decode()}")

        return output.reshape(self._shape)

    def destroy(self) -> None:
        """Free GPU memory. Called automatically when Aule closes."""
        if self._handle != 0:
            self._aule._lib.aule_tensor_destroy(ctypes.c_uint64(self._handle))
            self._handle = 0


class Aule:
    """
    Vulkan-based FlashAttention implementation.

    Example (simple):
        >>> aule = Aule()
        >>> output = aule.attention(Q, K, V)
        >>> aule.close()

    Example (fast, for repeated ops):
        >>> aule = Aule()
        >>> q = aule.tensor(Q.shape)
        >>> k = aule.tensor(K.shape)
        >>> v = aule.tensor(V.shape)
        >>> out = aule.tensor(Q.shape)
        >>>
        >>> q.upload(Q)
        >>> k.upload(K)
        >>> v.upload(V)
        >>>
        >>> for _ in range(1000):  # No CPU<->GPU copy!
        ...     aule.attention_gpu(q, k, v, out)
        >>>
        >>> result = out.download()
    """

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the aule library.

        Args:
            library_path: Optional path to the aule shared library.
                         If None, uses the library found at import time.
        """
        global _AULE_LIB_SINGLETON, _AULE_INITIALIZED

        if library_path:
            lib_path = Path(library_path)
        else:
            lib_path = _LIBRARY_PATH

        # Use singleton to avoid reloading library (MAJOR performance fix)
        if _AULE_LIB_SINGLETON is None:
            _AULE_LIB_SINGLETON = ctypes.CDLL(str(lib_path))

        self._lib = _AULE_LIB_SINGLETON
        self._setup_functions()
        self._tensors = []  # Track tensors for cleanup
        self._tensor_cache = {}  # Cache buffers by shape: (batch, heads, seq, dim) -> (Q, K, V, O)

        # Only call aule_init() once globally (avoid segfault on double-init)
        if not _AULE_INITIALIZED:
            result = self._lib.aule_init()
            if result != 0:
                error = self._lib.aule_get_error()
                raise AuleError(f"Failed to initialize aule: {error.decode()}")
            _AULE_INITIALIZED = True

        self._initialized = True

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        # aule_init
        self._lib.aule_init.argtypes = []
        self._lib.aule_init.restype = ctypes.c_int32

        # aule_shutdown
        self._lib.aule_shutdown.argtypes = []
        self._lib.aule_shutdown.restype = None

        # aule_get_error
        self._lib.aule_get_error.argtypes = []
        self._lib.aule_get_error.restype = ctypes.c_char_p

        # aule_attention_forward (copies data each call)
        self._lib.aule_attention_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # query
            ctypes.POINTER(ctypes.c_float),  # key
            ctypes.POINTER(ctypes.c_float),  # value
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_uint32,  # batch_size
            ctypes.c_uint32,  # num_heads
            ctypes.c_uint32,  # seq_len
            ctypes.c_uint32,  # head_dim
            ctypes.c_int32,   # causal
        ]
        self._lib.aule_attention_forward.restype = ctypes.c_int32

        # Tensor API
        self._lib.aule_tensor_create.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
        ]
        self._lib.aule_tensor_create.restype = ctypes.c_uint64

        self._lib.aule_tensor_destroy.argtypes = [ctypes.c_uint64]
        self._lib.aule_tensor_destroy.restype = None

        # Tensor pool management
        try:
            self._lib.aule_tensor_count.argtypes = []
            self._lib.aule_tensor_count.restype = ctypes.c_uint32
            self._lib.aule_tensor_max.argtypes = []
            self._lib.aule_tensor_max.restype = ctypes.c_uint32
            self._lib.aule_tensor_clear_all.argtypes = []
            self._lib.aule_tensor_clear_all.restype = None
        except AttributeError:
            pass

        self._lib.aule_tensor_upload.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
        ]
        self._lib.aule_tensor_upload.restype = ctypes.c_int32

        self._lib.aule_tensor_download.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
        ]
        self._lib.aule_tensor_download.restype = ctypes.c_int32

        self._lib.aule_tensor_download_u32.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32
        ]
        self._lib.aule_tensor_download_u32.restype = ctypes.c_int32

        self._lib.aule_tensor_size.argtypes = [ctypes.c_uint64]
        self._lib.aule_tensor_size.restype = ctypes.c_uint32

        # GPU tensor attention (no copy)
        self._lib.aule_attention_forward_gpu.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64,  # rot_cos, rot_sin
            ctypes.c_int32,  # causal
            ctypes.c_int32,  # window_size (-1 for full attention)
        ]
        self._lib.aule_attention_forward_gpu.restype = ctypes.c_int32

        # Spatial Sort
        self._lib.aule_spatial_sort.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]
        self._lib.aule_spatial_sort.restype = ctypes.c_int32

        # Gravity Attention
        self._lib.aule_attention_forward_gravity.argtypes = [
            ctypes.c_uint64, # Q
            ctypes.c_uint64, # K
            ctypes.c_uint64, # V
            ctypes.c_uint64, # O
            ctypes.c_uint64, # Cos
            ctypes.c_uint64, # Sin
            ctypes.c_uint64, # Indices
            ctypes.c_int32,  # Causal
            ctypes.c_uint32, # Max Attend (Top-k)
            ctypes.c_int32,  # window_size (-1 for full attention)
        ]
        self._lib.aule_attention_forward_gravity.restype = ctypes.c_int32

        # GPU info functions (optional - may not be present in all versions)
        try:
            self._lib.aule_get_device_name.argtypes = []
            self._lib.aule_get_device_name.restype = ctypes.c_char_p
        except AttributeError:
            pass

        self._lib.aule_get_vendor.argtypes = []
        self._lib.aule_get_vendor.restype = ctypes.c_int32

        # Optional functions
        try:
            self._lib.aule_is_amd_optimized.argtypes = []
            self._lib.aule_is_amd_optimized.restype = ctypes.c_int32
        except AttributeError:
            pass

        try:
            self._lib.aule_has_fp16.argtypes = []
            self._lib.aule_has_fp16.restype = ctypes.c_int32
        except AttributeError:
            pass

        try:
            self._lib.aule_get_subgroup_size.argtypes = []
            self._lib.aule_get_subgroup_size.restype = ctypes.c_uint32
        except AttributeError:
            pass

        try:
            self._lib.aule_get_device_name.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
            self._lib.aule_get_device_name.restype = ctypes.c_int32
        except AttributeError:
            pass

        try:
            self._lib.aule_get_gpu_vendor.argtypes = []
            self._lib.aule_get_gpu_vendor.restype = ctypes.c_int32
        except AttributeError:
            pass

        # Shader variant selection API
        try:
            self._lib.aule_set_shader_variant.argtypes = [ctypes.c_uint8]
            self._lib.aule_set_shader_variant.restype = ctypes.c_int32
            self._lib.aule_get_shader_variant.argtypes = []
            self._lib.aule_get_shader_variant.restype = ctypes.c_int32
            self._lib.aule_has_shader_variant.argtypes = [ctypes.c_uint8]
            self._lib.aule_has_shader_variant.restype = ctypes.c_int32
        except AttributeError:
            pass

        # Backward pass support
        self._lib.aule_supports_backward.argtypes = []
        self._lib.aule_supports_backward.restype = ctypes.c_int32

        # Forward with LSE (for training)
        self._lib.aule_attention_forward_with_lse.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # query
            ctypes.POINTER(ctypes.c_float),  # key
            ctypes.POINTER(ctypes.c_float),  # value
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # lse
            ctypes.c_uint32,  # batch_size
            ctypes.c_uint32,  # num_heads
            ctypes.c_uint32,  # seq_len
            ctypes.c_uint32,  # head_dim
            ctypes.c_int32,   # causal
        ]
        self._lib.aule_attention_forward_with_lse.restype = ctypes.c_int32

        # Backward pass
        self._lib.aule_attention_backward.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # query
            ctypes.POINTER(ctypes.c_float),  # key
            ctypes.POINTER(ctypes.c_float),  # value
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # grad_output
            ctypes.POINTER(ctypes.c_float),  # lse
            ctypes.POINTER(ctypes.c_float),  # grad_query
            ctypes.POINTER(ctypes.c_float),  # grad_key
            ctypes.POINTER(ctypes.c_float),  # grad_value
            ctypes.c_uint32,  # batch_size
            ctypes.c_uint32,  # num_heads
            ctypes.c_uint32,  # seq_len
            ctypes.c_uint32,  # head_dim
            ctypes.c_int32,   # causal
        ]
        self._lib.aule_attention_backward.restype = ctypes.c_int32

    @property
    def device_name(self) -> str:
        """Get the GPU device name."""
        if not self._initialized:
            return "Not initialized"
        try:
            buffer = ctypes.create_string_buffer(256)
            result = self._lib.aule_get_device_name(buffer, 256)
            if result >= 0:
                return buffer.value.decode('utf-8', errors='replace')
            return "Unknown"
        except AttributeError:
            return "Unknown"

    @property
    def vendor(self) -> str:
        """Get the GPU vendor (amd, nvidia, intel, apple, other)."""
        if not self._initialized:
            return "unknown"
        try:
            vendor_id = self._lib.aule_get_gpu_vendor()
            vendors = {0: "other", 1: "amd", 2: "nvidia", 3: "intel", 4: "apple"}
            return vendors.get(vendor_id, "unknown")
        except AttributeError:
            return "unknown"

    @property
    def is_amd_optimized(self) -> bool:
        """Check if using AMD-optimized shader path."""
        if not self._initialized:
            return False
        try:
            return self._lib.aule_is_amd_optimized() == 1
        except AttributeError:
            return False

    @property
    def fp16_supported(self) -> bool:
        """Check if FP16 is supported on this GPU."""
        if not self._initialized:
            return False
        try:
            return self._lib.aule_has_fp16() == 1
        except AttributeError:
            return False

    @property
    def subgroup_size(self) -> int:
        """Get the GPU subgroup/wavefront size (32 for NVIDIA, 64 for AMD)."""
        if not self._initialized:
            return 0
        try:
            return self._lib.aule_get_subgroup_size()
        except AttributeError:
            return 0

    def get_device_info(self) -> dict:
        """Get comprehensive GPU device info."""
        return {
            "device_name": self.device_name,
            "vendor": self.vendor,
            "amd_optimized": self.is_amd_optimized,
            "fp16_supported": self.fp16_supported,
            "subgroup_size": self.subgroup_size,
            "shader_variant": self.shader_variant,
            "available_variants": self.available_shader_variants,
        }

    # Shader variant names
    SHADER_BASELINE = 0  # Original 16x16 block, scalar loads
    SHADER_FAST = 1      # Optimized 32x32 block, vec4 loads, block skipping
    SHADER_FP16 = 2      # FP16 with FP32 accumulation
    SHADER_FP16_AMD = 3  # FP16 optimized for AMD 64-wide wavefronts

    @property
    def shader_variant(self) -> int:
        """Get the current shader variant. Returns -1 if not supported."""
        if not self._initialized:
            return -1
        try:
            return self._lib.aule_get_shader_variant()
        except AttributeError:
            return -1

    @property
    def shader_variant_name(self) -> str:
        """Get the name of the current shader variant."""
        variant = self.shader_variant
        names = {0: "baseline", 1: "fast", 2: "fp16", 3: "fp16_amd"}
        return names.get(variant, "unknown")

    @property
    def available_shader_variants(self) -> list:
        """Get list of available shader variants."""
        if not self._initialized:
            return []
        variants = []
        try:
            for v in [0, 1, 2, 3]:
                if self._lib.aule_has_shader_variant(v) == 1:
                    variants.append(v)
        except AttributeError:
            variants = [0]  # Baseline always available
        return variants

    def set_shader_variant(self, variant: int) -> None:
        """
        Set the shader variant for attention computation.

        Args:
            variant: Shader variant ID (0=baseline, 1=fast, 2=fp16, 3=fp16_amd)
                     Use Aule.SHADER_BASELINE, Aule.SHADER_FAST, etc.

        Raises:
            AuleError: If variant is not available
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")
        try:
            result = self._lib.aule_set_shader_variant(ctypes.c_uint8(variant))
            if result == -1:
                raise AuleError("Aule not initialized")
            elif result == -2:
                names = {0: "baseline", 1: "fast", 2: "fp16", 3: "fp16_amd"}
                raise AuleError(f"Shader variant '{names.get(variant, variant)}' not available")
        except AttributeError:
            raise AuleError("Shader variant selection not supported in this version")

    def has_shader_variant(self, variant: int) -> bool:
        """Check if a shader variant is available."""
        if not self._initialized:
            return False
        try:
            return self._lib.aule_has_shader_variant(ctypes.c_uint8(variant)) == 1
        except AttributeError:
            return variant == 0  # Baseline always available

    @property
    def tensor_count(self) -> int:
        """Get number of active GPU tensors."""
        try:
            return self._lib.aule_tensor_count()
        except AttributeError:
            return len(self._tensors)

    @property
    def tensor_max(self) -> int:
        """Get maximum number of tensors allowed."""
        try:
            return self._lib.aule_tensor_max()
        except AttributeError:
            return 1024  # Default

    def clear_tensors(self) -> None:
        """Clear all GPU tensors to free memory. Use when hitting tensor limits."""
        try:
            self._lib.aule_tensor_clear_all()
        except AttributeError:
            pass
        # Also clear Python-side tracking
        self._tensors.clear()
        self._tensor_cache.clear()

    def tensor(self, shape: Tuple[int, int, int, int], dtype: np.dtype = np.float32) -> GpuTensor:
        """
        Create a GPU tensor.

        Args:
            shape: (batch_size, num_heads, seq_len, head_dim)
            dtype: Data type of the tensor (e.g., np.float32, np.uint32)

        Returns:
            GpuTensor that lives on GPU memory
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        if len(shape) != 4:
            raise ValueError("Shape must be (batch, heads, seq, dim)")

        batch, heads, seq, dim = shape
        if dim > 64:
            raise ValueError(f"head_dim must be <= 64, got {dim}")

        if dtype == np.uint32:
            handle = self._lib.aule_tensor_create_u32(
                ctypes.c_uint32(batch),
                ctypes.c_uint32(heads),
                ctypes.c_uint32(seq),
                ctypes.c_uint32(dim),
            )
        else:
            handle = self._lib.aule_tensor_create(
                ctypes.c_uint32(batch),
                ctypes.c_uint32(heads),
                ctypes.c_uint32(seq),
                ctypes.c_uint32(dim),
            )

        if handle == 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Failed to create tensor: {error.decode()}")

        return GpuTensor(self, handle, shape, dtype)

    def attention_gpu(
        self,
        Q: GpuTensor,
        K: GpuTensor,
        V: GpuTensor,
        output: GpuTensor,
        rot_cos: Optional[GpuTensor] = None,
        rot_sin: Optional[GpuTensor] = None,
        causal: bool = False,
        window_size: int = -1,
    ) -> None:
        """
        Compute attention on GPU tensors - NO CPU<->GPU COPY.

        This is the fast path. Data must already be on GPU via upload().
        Result stays on GPU until you call output.download().

        Args:
            Q: Query tensor on GPU
            K: Key tensor on GPU
            V: Value tensor on GPU
            output: Output tensor on GPU (will be overwritten)
            rot_cos: Optional Rotary Embedding Cosine tensor [1, 1, seq_len, head_dim/2] or similar broadcastable
            rot_sin: Optional Rotary Embedding Sine tensor
            causal: If True, apply causal masking (for autoregressive models like LLMs)
            window_size: Sliding window size (-1 for full attention)
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        rot_cos_handle = rot_cos.handle if rot_cos is not None else 0
        rot_sin_handle = rot_sin.handle if rot_sin is not None else 0

        result = self._lib.aule_attention_forward_gpu(
            ctypes.c_uint64(Q.handle),
            ctypes.c_uint64(K.handle),
            ctypes.c_uint64(V.handle),
            ctypes.c_uint64(output.handle),
            ctypes.c_uint64(rot_cos_handle),
            ctypes.c_uint64(rot_sin_handle),
            ctypes.c_int32(1 if causal else 0),
            ctypes.c_int32(window_size),
        )

        if result != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"GPU attention failed: {error.decode()}")

    def attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        rot_cos: Optional[np.ndarray] = None,
        rot_sin: Optional[np.ndarray] = None,
        causal: bool = False,
        window_size: int = -1,
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention (simple API, copies data).

        For repeated operations, use tensor() + attention_gpu() instead.

        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            causal: If True, apply causal masking (for autoregressive models like LLMs)
            window_size: Sliding window size (-1 for full attention)

        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        # Validate shapes
        if query.shape[0] != key.shape[0]:
             raise ValueError(f"Batch size must match. Q={query.shape}, K={key.shape}")

        if value.shape[0] != key.shape[0] or value.shape[2] != key.shape[2]:
             raise ValueError(f"Value shape must match Key shape [B, H, S, D]. K={key.shape}, V={value.shape}")
        
        # Check for GQA (Q heads multiple of K heads)
        if query.shape[1] % key.shape[1] != 0:
             raise ValueError(f"Num Q heads must be multiple of K heads. Q={query.shape}, K={key.shape}")

        if len(query.shape) != 4:
            raise ValueError(
                f"Expected 4D tensors [batch, heads, seq, dim]. Got shape {query.shape}"
            )

        batch_size, num_heads, seq_len, head_dim = query.shape

        if head_dim > 64:
            raise ValueError(
                f"head_dim must be <= 64. Got {head_dim}. "
                "This limitation will be lifted in future versions."
            )

        # Ensure contiguous float32 arrays
        query = np.ascontiguousarray(query, dtype=np.float32)
        key = np.ascontiguousarray(key, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)

        # Allocate output
        output = np.empty_like(query)

        # Allocate output
        output = np.empty_like(query)

        # Get pointers
        # Old simple path: result = self._lib.aule_attention_forward(...)
        # BUT aule_attention_forward (C API) does not support RoPE yet! only _gpu one does.
        # So if we have RoPE, we MUST use the Tensor API path.
        
        if rot_cos is not None and rot_sin is not None:
            # Use Tensor API for RoPE support
            # This is slower due to malloc/upload overhead, but required until C API is updated.
            # TODO: Update aule_attention_forward C API to support RoPE.
            
            # Check shapes
            if rot_cos.ndim != 4 or rot_sin.ndim != 4:
                 # Auto-reshape if 3D [seq, dim] -> [1, 1, seq, dim]? 
                 # For now assume user provides correct shape or we rely on tensor() validation.
                 pass

            q_gpu = self.tensor(query.shape)
            k_gpu = self.tensor(key.shape)
            v_gpu = self.tensor(value.shape)
            out_gpu = self.tensor(query.shape)
            
            # Check RoPE shape. 
            # rot_cos might be [1,1,S,D/2] or broadcasted. 
            # GpuTensor expects 4 args for shape.
            # We need to manually construct GpuTensor for RoPE.
            # Use raw handle create? No, use self.tensor with manual shape.
            
            # Use provided shape
            rc_shape = rot_cos.shape
            rs_shape = rot_sin.shape
            
            # Ensure 4D
            if len(rc_shape) < 4:
                # Pad with 1s
                pad = (1,) * (4 - len(rc_shape))
                rc_shape = pad + rc_shape
                rot_cos = rot_cos.reshape(rc_shape)
                
            if len(rs_shape) < 4:
                pad = (1,) * (4 - len(rs_shape))
                rs_shape = pad + rs_shape
                rot_sin = rot_sin.reshape(rs_shape)

            cos_gpu = self.tensor(rc_shape)
            sin_gpu = self.tensor(rs_shape)
            
            q_gpu.upload(query)
            k_gpu.upload(key)
            v_gpu.upload(value)
            cos_gpu.upload(rot_cos)
            sin_gpu.upload(rot_sin)
            
            # This calls the fast path
            self.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, cos_gpu, sin_gpu, causal, window_size)
            
            # Download result directly into output buffer if possible?
            # download() returns new array.
            result_tmp = out_gpu.download()
            np.copyto(output, result_tmp) # Copy to user buffer
            
            # Cleanup happen automatically via __exit__ or explicit destroy?
            # Tensors are added to self._tensors and destroyed on close().
            # Since convienence function uses `with Aule()`, they will be cleaned up.
            return output

        # For GQA and cross-attention, K/V may have different shapes than Q
        q_shape = query.shape
        k_shape = key.shape
        v_shape = value.shape
        out_shape = q_shape  # Output always matches Q shape

        # Cache key must include ALL shapes (Q, K, V can differ)
        cache_key = (q_shape, k_shape, v_shape)

        if cache_key not in self._tensor_cache:
            # First call with this shape combination - create buffers
            q_gpu = self.tensor(q_shape)
            k_gpu = self.tensor(k_shape)
            v_gpu = self.tensor(v_shape)
            out_gpu = self.tensor(out_shape)
            self._tensor_cache[cache_key] = (q_gpu, k_gpu, v_gpu, out_gpu)
        else:
            # Reuse cached buffers
            q_gpu, k_gpu, v_gpu, out_gpu = self._tensor_cache[cache_key]

        # Upload data to GPU
        q_gpu.upload(query)
        k_gpu.upload(key)
        v_gpu.upload(value)

        # Compute on GPU
        self.attention_gpu(q_gpu, k_gpu, v_gpu, out_gpu, causal=causal, window_size=window_size)

        # Download result
        return out_gpu.download()

    @property
    def supports_backward(self) -> bool:
        """Check if backward pass (training) is supported."""
        if not self._initialized:
            return False
        return self._lib.aule_supports_backward() == 1

    def attention_forward_with_lse(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention forward pass and return LSE for backward.

        Args:
            query: Query tensor [batch, heads, seq, dim]
            key: Key tensor [batch, heads, seq, dim]
            value: Value tensor [batch, heads, seq, dim]
            causal: Whether to apply causal masking

        Returns:
            Tuple of (output, lse) where lse is needed for backward pass
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        if not self.supports_backward:
            raise AuleError("Backward pass not supported on this GPU/backend")

        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Q, K, V must have same shape")

        if len(query.shape) != 4:
            raise ValueError("Expected 4D tensors [batch, heads, seq, dim]")

        batch_size, num_heads, seq_len, head_dim = query.shape

        if head_dim > 64:
            raise ValueError(f"head_dim must be <= 64. Got {head_dim}")

        # Ensure contiguous float32 arrays
        query = np.ascontiguousarray(query, dtype=np.float32)
        key = np.ascontiguousarray(key, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)

        # Allocate output and LSE
        output = np.empty_like(query)
        lse = np.empty((batch_size, num_heads, seq_len), dtype=np.float32)

        # Get pointers
        q_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = key.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        o_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lse_ptr = lse.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = self._lib.aule_attention_forward_with_lse(
            q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
            ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(head_dim),
            ctypes.c_int32(1 if causal else 0),
        )

        if result != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Forward with LSE failed: {error.decode()}")

        return output, lse

    def attention_backward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        output: np.ndarray,
        grad_output: np.ndarray,
        lse: np.ndarray,
        causal: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for attention backward pass.

        Args:
            query: Query tensor [batch, heads, seq, dim]
            key: Key tensor [batch, heads, seq, dim]
            value: Value tensor [batch, heads, seq, dim]
            output: Output from forward pass [batch, heads, seq, dim]
            grad_output: Gradient of loss w.r.t. output [batch, heads, seq, dim]
            lse: Log-sum-exp from forward pass [batch, heads, seq]
            causal: Whether causal masking was used

        Returns:
            Tuple of (grad_query, grad_key, grad_value)
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        if not self.supports_backward:
            raise AuleError("Backward pass not supported on this GPU/backend")

        batch_size, num_heads, seq_len, head_dim = query.shape

        # Ensure contiguous float32 arrays
        query = np.ascontiguousarray(query, dtype=np.float32)
        key = np.ascontiguousarray(key, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)
        output = np.ascontiguousarray(output, dtype=np.float32)
        grad_output = np.ascontiguousarray(grad_output, dtype=np.float32)
        lse = np.ascontiguousarray(lse, dtype=np.float32)

        # Allocate gradient outputs
        grad_query = np.empty_like(query)
        grad_key = np.empty_like(key)
        grad_value = np.empty_like(value)

        # Get pointers
        q_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = key.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        o_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        do_ptr = grad_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lse_ptr = lse.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dq_ptr = grad_query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dk_ptr = grad_key.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dv_ptr = grad_value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = self._lib.aule_attention_backward(
            q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, lse_ptr,
            dq_ptr, dk_ptr, dv_ptr,
            ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(head_dim),
            ctypes.c_int32(1 if causal else 0),
        )

        if result != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Backward pass failed: {error.decode()}")

        return grad_query, grad_key, grad_value

    def spatial_sort(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        sort_dim: int = 0
    ) -> np.ndarray:
        """
        Sort keys and values spatially based on projection onto sort_dim.
        Returns the sorted indices.
        
        Args:
            keys: [batch, heads, seq, dim]
            values: [batch, heads, seq, dim]
            sort_dim: Dimension index to project onto (0..dim-1)
            
        Returns:
            indices: [batch, heads, seq] (uint32)
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        if keys.shape != values.shape:
             raise ValueError("Keys and Values must have same shape")
             
        batch, heads, seq, dim = keys.shape
        
        # Ensure contiguous
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        
        # Create tensors
        keys_gpu = self.tensor(keys.shape)
        vals_gpu = self.tensor(values.shape)
        # Indices are uint32, but GpuTensor stores them as float32 internally for now.
        # The C API expects a u32 tensor.
        indices_shape = (batch, heads, seq, 1) # Indices are scalar per vector
        inds_gpu = self.tensor(indices_shape, dtype=np.uint32)
        
        keys_gpu.upload(keys)
        vals_gpu.upload(values)
        
        # Dispatch
        ret = self._lib.aule_spatial_sort(
            ctypes.c_uint64(keys_gpu.handle),
            ctypes.c_uint64(vals_gpu.handle),
            ctypes.c_uint64(inds_gpu.handle),
            ctypes.c_uint32(sort_dim)
        )
        
        if ret != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Spatial sort failed: {error.decode()}")
            
        # Download indices
        indices_np = inds_gpu.download()
        indices_np = indices_np.view(np.uint32)
        
        return indices_np.reshape(batch, heads, seq)

    def attention_gravity(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        indices: np.ndarray,
        rot_cos: Optional[np.ndarray] = None,
        rot_sin: Optional[np.ndarray] = None,
        causal: bool = False,
        max_attend: Optional[int] = None,
        window_size: int = -1,
    ) -> np.ndarray:
        """
        Compute attention using indirect lookup via spatial sort indices.
        PROTOTYPE PHASE 3: Validates that we can correctly read K/V via indices.

        Args:
            query: Query tensor [batch, heads, seq, dim]
            key: Key tensor [batch, heads, seq, dim]
            value: Value tensor [batch, heads, seq, dim]
            indices: Indices tensor [batch, heads, seq] (uint32)
            rot_cos: Optional Rotary Embedding Cosine tensor
            rot_sin: Optional Rotary Embedding Sine tensor
            causal: If True, apply causal masking
            max_attend: Top-K keys to attend to. If None, attends to all.
            window_size: Sliding window size (-1 for full attention)

        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        # Validate shapes
        if query.shape[0] != key.shape[0]:
             raise ValueError(f"Batch size must match. Q={query.shape}, K={key.shape}")
        if value.shape[0] != key.shape[0] or value.shape[2] != key.shape[2]:
             raise ValueError(f"Value shape must match Key shape [B, H, S, D]. K={key.shape}, V={value.shape}")
        if query.shape[1] % key.shape[1] != 0:
             raise ValueError(f"Num Q heads must be multiple of K heads. Q={query.shape}, K={key.shape}")
        if len(query.shape) != 4:
            raise ValueError(f"Expected 4D tensors [batch, heads, seq, dim]. Got shape {query.shape}")
        if indices.ndim != 3:
            raise ValueError(f"Expected 3D indices tensor [batch, heads, seq]. Got shape {indices.shape}")
        if indices.shape != key.shape[:3]:
            # Indices represent the sorted order of KEYS, so must match Key shape [B, H, S_K]
            # (assuming shared indices for all queries in a head)
            raise ValueError(f"Indices shape {indices.shape} must match K's batch, heads, seq {key.shape[:3]}")

        # Default max_attend to key sequence length
        if max_attend is None:
            max_attend = key.shape[2]

        # Ensure contiguous float32 arrays for Q, K, V
        query = np.ascontiguousarray(query, dtype=np.float32)
        key = np.ascontiguousarray(key, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)
        indices = np.ascontiguousarray(indices, dtype=np.uint32)

        # Create GPU tensors
        q_gpu = self.tensor(query.shape)
        k_gpu = self.tensor(key.shape)
        v_gpu = self.tensor(value.shape)
        out_gpu = self.tensor(query.shape)
        
        # Indices tensor needs to be 4D for GpuTensor, with last dim 1
        indices_4d_shape = indices.shape + (1,)
        indices_gpu = self.tensor(indices_4d_shape, dtype=np.uint32)
        indices_gpu.upload(indices.reshape(indices_4d_shape))
        
        cos_gpu = None
        sin_gpu = None
        rot_cos_handle = 0
        rot_sin_handle = 0

        if rot_cos is not None:
            rc_shape = rot_cos.shape
            if len(rc_shape) < 4:
                pad = (1,) * (4 - len(rc_shape))
                rc_shape = pad + rc_shape
                rot_cos = rot_cos.reshape(rc_shape)
            cos_gpu = self.tensor(rc_shape)
            cos_gpu.upload(rot_cos)
            rot_cos_handle = cos_gpu.handle

        if rot_sin is not None:
            rs_shape = rot_sin.shape
            if len(rs_shape) < 4:
                pad = (1,) * (4 - len(rs_shape))
                rs_shape = pad + rs_shape
                rot_sin = rot_sin.reshape(rs_shape)
            sin_gpu = self.tensor(rs_shape)
            sin_gpu.upload(rot_sin)
            rot_sin_handle = sin_gpu.handle
            
        # Upload data
        q_gpu.upload(query)
        k_gpu.upload(key)
        v_gpu.upload(value)
            
        ret = self._lib.aule_attention_forward_gravity(
            ctypes.c_uint64(q_gpu.handle),
            ctypes.c_uint64(k_gpu.handle),
            ctypes.c_uint64(v_gpu.handle),
            ctypes.c_uint64(out_gpu.handle),
            ctypes.c_uint64(rot_cos_handle),
            ctypes.c_uint64(rot_sin_handle),
            ctypes.c_uint64(indices_gpu.handle),
            ctypes.c_int32(1 if causal else 0),
            ctypes.c_uint32(max_attend),
            ctypes.c_int32(window_size),
        )
        
        if ret != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Gravity Attention failed: {error.decode()}")
            
        return out_gpu.download()

    def close(self):
        """Shut down the aule library and release GPU resources."""
        global _AULE_INITIALIZED
        if self._initialized:
            # Clean up tensors
            for tensor in self._tensors:
                tensor.destroy()
            self._tensors.clear()
            self._tensor_cache.clear()

            # Only shutdown if we're the singleton, otherwise just mark as not initialized
            # Note: In practice with singletons, we don't want to actually shutdown
            # because other code might still be using it. Only shutdown on program exit.
            # self._lib.aule_shutdown()
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        if hasattr(self, '_initialized') and self._initialized:
            self.close()


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    causal: bool = False,
    window_size: int = -1,
) -> np.ndarray:
    """
    Convenience function for one-off attention computations.

    Uses a global singleton Aule instance to share tensor cache across calls.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        causal: If True, apply causal masking (for autoregressive models like LLMs)
        window_size: Sliding window size (-1 for full attention)

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    global _AULE_INSTANCE_SINGLETON

    if _AULE_INSTANCE_SINGLETON is None:
        _AULE_INSTANCE_SINGLETON = Aule()

    return _AULE_INSTANCE_SINGLETON.attention(query, key, value, causal=causal, window_size=window_size)


def flash_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    rot_cos: Optional[np.ndarray] = None,
    rot_sin: Optional[np.ndarray] = None,
    causal: bool = True,
    window_size: int = -1,
) -> np.ndarray:
    """
    FlashAttention-style scaled dot-product attention.

    Uses a global singleton Aule instance to share tensor cache across calls.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        causal: If True (default), apply causal masking for autoregressive models
        window_size: Sliding window size (-1 for full attention)

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    global _AULE_INSTANCE_SINGLETON

    if _AULE_INSTANCE_SINGLETON is None:
        _AULE_INSTANCE_SINGLETON = Aule()

    return _AULE_INSTANCE_SINGLETON.attention(query, key, value, rot_cos=rot_cos, rot_sin=rot_sin, causal=causal, window_size=window_size)


def supports_backward() -> bool:
    """
    Check if backward pass (training) is supported.

    Returns:
        True if backward pass is available on this GPU/backend
    """
    try:
        with Aule() as aule:
            return aule.supports_backward
    except Exception:
        return False


def attention_forward_with_lse(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention forward pass and return LSE for backward.

    Args:
        query: Query tensor [batch, heads, seq, dim]
        key: Key tensor [batch, heads, seq, dim]
        value: Value tensor [batch, heads, seq, dim]
        causal: Whether to apply causal masking

    Returns:
        Tuple of (output, lse) where lse is needed for backward pass
    """
    with Aule() as aule:
        return aule.attention_forward_with_lse(query, key, value, causal=causal)


def attention_backward(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    output: np.ndarray,
    grad_output: np.ndarray,
    lse: np.ndarray,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradients for attention backward pass.

    Args:
        query: Query tensor [batch, heads, seq, dim]
        key: Key tensor [batch, heads, seq, dim]
        value: Value tensor [batch, heads, seq, dim]
        output: Output from forward pass [batch, heads, seq, dim]
        grad_output: Gradient of loss w.r.t. output [batch, heads, seq, dim]
        lse: Log-sum-exp from forward pass [batch, heads, seq]
        causal: Whether causal masking was used

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    with Aule() as aule:
        return aule.attention_backward(
            query, key, value, output, grad_output, lse, causal=causal
        )


# Verify library exists at import time - this allows __init__.py to
# properly detect if Vulkan backend is available
_LIBRARY_PATH = _find_library()
