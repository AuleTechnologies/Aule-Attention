"""
aule-attention Python bindings

Provides Python interface to the Vulkan-based FlashAttention implementation.
Works on AMD, NVIDIA, Intel, and any GPU with Vulkan compute support.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Find the shared library
def _find_library() -> Path:
    """Find the aule shared library."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent / "zig-out" / "lib" / "libaule.so",
        Path(__file__).parent.parent / "zig-out" / "lib" / "libaule.dylib",
        Path(__file__).parent.parent / "zig-out" / "lib" / "aule.dll",
        Path("/usr/local/lib/libaule.so"),
        Path("/usr/lib/libaule.so"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise RuntimeError(
        "Could not find aule library. Build it with 'zig build' first.\n"
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

    def __init__(self, aule: 'Aule', handle: int, shape: Tuple[int, ...]):
        self._aule = aule
        self._handle = handle
        self._shape = shape
        self._size = int(np.prod(shape))

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

        data = np.ascontiguousarray(data, dtype=np.float32).ravel()
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
        output = np.empty(self._size, dtype=np.float32)
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
                         If None, will search standard locations.
        """
        if library_path:
            lib_path = Path(library_path)
        else:
            lib_path = _find_library()

        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()
        self._tensors = []  # Track tensors for cleanup

        # Initialize the library
        result = self._lib.aule_init()
        if result != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Failed to initialize aule: {error.decode()}")

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

        self._lib.aule_tensor_upload.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
        ]
        self._lib.aule_tensor_upload.restype = ctypes.c_int32

        self._lib.aule_tensor_download.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
        ]
        self._lib.aule_tensor_download.restype = ctypes.c_int32

        self._lib.aule_tensor_size.argtypes = [ctypes.c_uint64]
        self._lib.aule_tensor_size.restype = ctypes.c_uint32

        # GPU tensor attention (no copy)
        self._lib.aule_attention_forward_gpu.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int32,  # causal
        ]
        self._lib.aule_attention_forward_gpu.restype = ctypes.c_int32

        # GPU info functions
        self._lib.aule_get_device_name.argtypes = []
        self._lib.aule_get_device_name.restype = ctypes.c_char_p

        self._lib.aule_get_vendor.argtypes = []
        self._lib.aule_get_vendor.restype = ctypes.c_int32

        self._lib.aule_is_amd_optimized.argtypes = []
        self._lib.aule_is_amd_optimized.restype = ctypes.c_int32

        self._lib.aule_has_fp16.argtypes = []
        self._lib.aule_has_fp16.restype = ctypes.c_int32

        self._lib.aule_get_subgroup_size.argtypes = []
        self._lib.aule_get_subgroup_size.restype = ctypes.c_uint32

    @property
    def device_name(self) -> str:
        """Get the GPU device name."""
        if not self._initialized:
            return "Not initialized"
        name = self._lib.aule_get_device_name()
        return name.decode() if name else "Unknown"

    @property
    def vendor(self) -> str:
        """Get the GPU vendor (amd, nvidia, intel, apple, other)."""
        if not self._initialized:
            return "unknown"
        vendor_id = self._lib.aule_get_vendor()
        vendors = {0: "other", 1: "amd", 2: "nvidia", 3: "intel", 4: "apple"}
        return vendors.get(vendor_id, "unknown")

    @property
    def is_amd_optimized(self) -> bool:
        """Check if using AMD-optimized shader path."""
        if not self._initialized:
            return False
        return self._lib.aule_is_amd_optimized() == 1

    @property
    def fp16_supported(self) -> bool:
        """Check if FP16 is supported on this GPU."""
        if not self._initialized:
            return False
        return self._lib.aule_has_fp16() == 1

    @property
    def subgroup_size(self) -> int:
        """Get the GPU subgroup/wavefront size (32 for NVIDIA, 64 for AMD)."""
        if not self._initialized:
            return 0
        return self._lib.aule_get_subgroup_size()

    def get_device_info(self) -> dict:
        """Get comprehensive GPU device info."""
        return {
            "device_name": self.device_name,
            "vendor": self.vendor,
            "amd_optimized": self.is_amd_optimized,
            "fp16_supported": self.fp16_supported,
            "subgroup_size": self.subgroup_size,
        }

    def tensor(self, shape: Tuple[int, int, int, int]) -> GpuTensor:
        """
        Create a GPU tensor.

        Args:
            shape: (batch_size, num_heads, seq_len, head_dim)

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

        handle = self._lib.aule_tensor_create(
            ctypes.c_uint32(batch),
            ctypes.c_uint32(heads),
            ctypes.c_uint32(seq),
            ctypes.c_uint32(dim),
        )

        if handle == 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Failed to create tensor: {error.decode()}")

        tensor = GpuTensor(self, handle, shape)
        self._tensors.append(tensor)
        return tensor

    def attention_gpu(
        self,
        Q: GpuTensor,
        K: GpuTensor,
        V: GpuTensor,
        output: GpuTensor,
        causal: bool = False,
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
            causal: If True, apply causal masking (for autoregressive models like LLMs)
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        result = self._lib.aule_attention_forward_gpu(
            ctypes.c_uint64(Q.handle),
            ctypes.c_uint64(K.handle),
            ctypes.c_uint64(V.handle),
            ctypes.c_uint64(output.handle),
            ctypes.c_int32(1 if causal else 0),
        )

        if result != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"GPU attention failed: {error.decode()}")

    def attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal: bool = False,
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention (simple API, copies data).

        For repeated operations, use tensor() + attention_gpu() instead.

        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            causal: If True, apply causal masking (for autoregressive models like LLMs)

        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        if not self._initialized:
            raise AuleError("Aule not initialized")

        # Validate shapes
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                f"Q, K, V must have same shape. Got Q={query.shape}, K={key.shape}, V={value.shape}"
            )

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

        # Get pointers
        q_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = key.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        o_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call the library
        result = self._lib.aule_attention_forward(
            q_ptr, k_ptr, v_ptr, o_ptr,
            ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(head_dim),
            ctypes.c_int32(1 if causal else 0),
        )

        if result != 0:
            error = self._lib.aule_get_error()
            raise AuleError(f"Attention forward failed: {error.decode()}")

        return output

    def close(self):
        """Shut down the aule library and release GPU resources."""
        if self._initialized:
            # Clean up tensors
            for tensor in self._tensors:
                tensor.destroy()
            self._tensors.clear()

            self._lib.aule_shutdown()
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
) -> np.ndarray:
    """
    Convenience function for one-off attention computations.

    For repeated computations, use the Aule class with GPU tensors.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        causal: If True, apply causal masking (for autoregressive models like LLMs)

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    with Aule() as aule:
        return aule.attention(query, key, value, causal=causal)


def flash_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """
    FlashAttention-style scaled dot-product attention.

    Optimized for LLM inference with causal masking enabled by default.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        causal: If True (default), apply causal masking for autoregressive models

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    with Aule() as aule:
        return aule.attention(query, key, value, causal=causal)
