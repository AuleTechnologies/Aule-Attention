# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**Name:** aule-attention  
**Purpose:** Hardware-agnostic FlashAttention implementation via Zig + Vulkan Compute  
**Goal:** `pip install aule-attention` that works on AMD, NVIDIA, Intel, Apple — no ROCm, no CUDA, no compilation on install  
**Organization:** Aule Technologies

---

## Engineering Directives

### Absolute Rules

1. **No placeholder code.** Every function must be complete and functional. Stubs like `// TODO: implement` are forbidden. If a feature cannot be implemented, remove it entirely rather than leaving a skeleton.

2. **No "you're absolutely right" patterns.** Do not acknowledge corrections with filler. Implement the fix directly.

3. **Production-ready only.** Code must compile, run, and handle errors. No "example" or "demo" quality — this ships to users.

4. **Test everything.** Every module requires a corresponding test. Tests must be runnable via `zig build test` and `pytest`.

5. **Specialize aggressively.** You are a systems engineer with expertise in GPU compute, Vulkan, and high-performance numerical computing. Apply that lens to every decision.

6. **Index before implementing.** Before writing any code, read the referenced documentation in this file. Do not guess at APIs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Layer                             │
│  from aule import flash_attention                           │
│  out = flash_attention(q, k, v, causal=True)               │
└─────────────────────────┬───────────────────────────────────┘
                          │ ctypes FFI
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Zig Shared Library                       │
│  libaule.so / aule.dll / libaule.dylib                     │
│  - C ABI exports                                            │
│  - Vulkan instance/device management                        │
│  - Buffer marshaling (numpy → Vulkan staging → GPU)        │
│  - Command buffer recording & dispatch                      │
│  - Synchronization (fences)                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 SPIR-V Compute Shaders                      │
│  - attention.comp (FlashAttention-2 forward)               │
│  - attention_backward.comp (gradient computation)          │
│  - softmax.comp, rope.comp (fused ops)                     │
│  Embedded via @embedFile at compile time                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vulkan Compute Runtime                     │
│  AMD / NVIDIA / Intel / Apple (MoltenVK)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
aule-attention/
├── CLAUDE.md                 # This file
├── build.zig                 # Zig build configuration
├── build.zig.zon             # Zig package manifest
├── src/
│   ├── lib.zig               # Library root, C ABI exports
│   ├── vulkan_context.zig    # Instance, device, queue setup
│   ├── compute_pipeline.zig  # Pipeline creation, descriptor sets
│   ├── buffer_manager.zig    # GPU memory allocation, staging
│   ├── attention.zig         # FlashAttention dispatch logic
│   └── c_api.zig             # Explicit C interface definitions
├── shaders/
│   ├── attention.comp        # GLSL compute shader (FlashAttention-2)
│   ├── attention_backward.comp
│   └── compile.sh            # glslc compilation script
├── python/
│   ├── aule/
│   │   ├── __init__.py       # Public API
│   │   ├── _bindings.py      # ctypes wrapper
│   │   └── lib/              # Pre-built .so/.dll/.dylib
│   ├── setup.py
│   ├── pyproject.toml
│   └── tests/
│       ├── test_attention.py
│       └── test_backward.py
└── tests/
    └── test_attention.zig    # Zig-native tests
```

---

## Reference Documentation Index

Before implementing any component, read the corresponding documentation:

### Vulkan Compute Fundamentals
- Vulkan Compute Shader spec: https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html
- VK_KHR_cooperative_matrix (for tensor core access): https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_cooperative_matrix.html
- Existing Vulkan FlashAttention-2 reference: https://github.com/etasnadi/VulkanCooperativeMatrixAttention

### Zig + Vulkan Bindings
- vulkan-zig generator: https://github.com/Snektron/vulkan-zig
- Zig Vulkan tutorial implementation: https://github.com/Vulfox/vulkan-tutorial-zig
- VMA (Vulkan Memory Allocator) Zig bindings: https://github.com/SpexGuy/Zig-VMA

### FlashAttention Algorithm
- FlashAttention-2 paper: https://tridao.me/publications/flash2/flash2.pdf
- Original FlashAttention paper: https://arxiv.org/abs/2205.14135

### GLSL Compute Shaders
- GLSL compute shader basics: https://www.khronos.org/opengl/wiki/Compute_Shader
- Workgroup/invocation model: https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction

---

## Implementation Phases

### Phase 1: Vulkan Compute Skeleton

**Objective:** Dispatch a trivial compute shader, read back results.

**Files to create:**
1. `build.zig` — Zig build with vulkan-zig dependency, shader compilation
2. `src/vulkan_context.zig` — Instance creation, physical device selection, logical device, compute queue
3. `src/compute_pipeline.zig` — Load SPIR-V, create compute pipeline, descriptor set layout
4. `src/buffer_manager.zig` — Create storage buffers, staging buffers, memory barriers
5. `shaders/test.comp` — Trivial shader: `out[i] = in[i] * 2.0`
6. `src/lib.zig` — Export single C function: `aule_test_multiply`
7. `tests/test_multiply.zig` — Verify round-trip

**Acceptance criteria:**
- `zig build` produces `libaule.so`
- `zig build test` passes
- Can call from C: `float in[64], out[64]; aule_test_multiply(in, out, 64);`

### Phase 2: FlashAttention-2 Forward Pass

**Objective:** Implement memory-efficient scaled dot-product attention.

**Algorithm (from FlashAttention-2 paper):**
```
Input: Q, K, V ∈ R^{N×d}, block sizes Br, Bc
Output: O ∈ R^{N×d}

1. Divide Q into Tr blocks, K/V into Tc blocks
2. For each Q block i:
   a. Load Qi to shared memory
   b. Initialize Oi = 0, li = 0, mi = -∞
   c. For each K/V block j:
      - Load Kj, Vj to shared memory
      - Compute Sij = Qi @ Kj^T / sqrt(d)
      - Compute m̃ij = rowmax(Sij), P̃ij = exp(Sij - m̃ij)
      - Update running statistics:
        mi_new = max(mi, m̃ij)
        li_new = exp(mi - mi_new) * li + rowsum(P̃ij) * exp(m̃ij - mi_new)
        Oi = exp(mi - mi_new) * Oi + P̃ij @ Vj * exp(m̃ij - mi_new)
      - mi = mi_new, li = li_new
   d. Oi = Oi / li
3. Return O
```

**Files to create/modify:**
1. `shaders/attention.comp` — GLSL implementation of above
2. `src/attention.zig` — Host-side dispatch logic, parameter validation
3. `src/c_api.zig` — Add `aule_flash_attention_forward`

**Key considerations:**
- Block sizes (Br, Bc) depend on shared memory size — query via `vkGetPhysicalDeviceProperties`
- Use `VK_KHR_cooperative_matrix` if available for matmul acceleration
- Fall back to manual tiled matmul if cooperative_matrix unavailable
- Support fp16 and fp32 inputs

### Phase 3: Python Bindings

**Objective:** `pip install aule-attention` with zero compilation.

**Files to create:**
1. `python/aule/__init__.py` — Public API
2. `python/aule/_bindings.py` — ctypes wrapper
3. `python/setup.py` / `pyproject.toml` — Wheel configuration
4. `python/tests/test_attention.py` — pytest suite

**C API contract:**
```c
// All functions return 0 on success, negative on error

// Initialize library (call once)
int32_t aule_init(void);

// Cleanup (call on exit)
void aule_shutdown(void);

// Flash attention forward pass
// q, k, v: device pointers or host pointers (library handles staging)
// Shapes: q[batch, heads, seq_q, head_dim], k/v[batch, heads, seq_kv, head_dim]
int32_t aule_flash_attention_forward(
    const float* q, const float* k, const float* v,
    float* out,
    int64_t batch, int64_t heads, int64_t seq_q, int64_t seq_kv, int64_t head_dim,
    int32_t causal,  // 0 or 1
    float scale      // typically 1/sqrt(head_dim)
);

// Get last error message
const char* aule_get_error(void);
```

**Python API:**
```python
import torch
from aule import flash_attention

# Basic usage
q = torch.randn(2, 8, 512, 64)  # [batch, heads, seq, head_dim]
k = torch.randn(2, 8, 512, 64)
v = torch.randn(2, 8, 512, 64)

out = flash_attention(q, k, v, causal=True)

# With custom scale
out = flash_attention(q, k, v, causal=False, scale=0.125)
```

### Phase 4: Backward Pass

**Objective:** Enable training via gradient computation.

**Files to create:**
1. `shaders/attention_backward.comp` — dQ, dK, dV computation
2. `src/attention_backward.zig` — Host dispatch
3. `python/aule/_autograd.py` — `torch.autograd.Function` wrapper

**Algorithm (FlashAttention-2 backward):**
```
Input: Q, K, V, O, dO, L (logsumexp from forward)
Output: dQ, dK, dV

1. Recompute attention in tiles (memory efficient)
2. For each block:
   - dV += P^T @ dO
   - dP = dO @ V^T
   - dS = P ⊙ (dP - rowsum(dP ⊙ P))
   - dQ += dS @ K
   - dK += dS^T @ Q
```

### Phase 5: Cross-Platform Wheels

**Objective:** Pre-built wheels for Linux, Windows, macOS (Intel + ARM).

**GitHub Actions matrix:**
```yaml
strategy:
  matrix:
    include:
      - os: ubuntu-22.04
        target: x86_64-linux-gnu
        wheel: manylinux_2_28_x86_64
      - os: windows-2022
        target: x86_64-windows-gnu
        wheel: win_amd64
      - os: macos-13
        target: x86_64-macos
        wheel: macosx_13_0_x86_64
      - os: macos-14
        target: aarch64-macos
        wheel: macosx_14_0_arm64
```

**Build steps per platform:**
1. Install Vulkan SDK
2. `zig build -Dtarget=$TARGET -Doptimize=ReleaseFast`
3. Copy `libaule.*` to `python/aule/lib/`
4. `python -m build --wheel`
5. Upload to PyPI

---

## Technical Specifications

### Supported Hardware

| Vendor | Minimum | Tested | Notes |
|--------|---------|--------|-------|
| AMD | RX 5000+ / MI100+ | MI300X, 7900 XTX | Primary target |
| NVIDIA | GTX 1000+ | RTX 4090, H100 | Cooperative matrix on Ampere+ |
| Intel | Arc A-series | Arc A770 | Limited testing |
| Apple | M1+ | M2 Pro | Via MoltenVK |

### Memory Requirements

- Minimum VRAM: 4 GB
- Recommended: 8+ GB for seq_len > 4096

### Performance Targets

| Operation | Sequence Length | Target Throughput |
|-----------|-----------------|-------------------|
| Forward | 2048 | > 50 TFLOPS (fp16) |
| Forward | 8192 | > 40 TFLOPS (fp16) |
| Backward | 2048 | > 30 TFLOPS (fp16) |

---

## Code Style

### Zig
- Use `std.log` for all logging, not print
- Explicit error handling — no `catch unreachable` except in truly impossible cases
- All allocations via passed allocator, never global
- Use `defer` for cleanup immediately after resource acquisition

### GLSL
- Document block sizes and shared memory usage at top of shader
- Use `#define` for configurable parameters, not magic numbers
- Barrier usage must be commented explaining what it synchronizes

### Python
- Type hints on all public functions
- Docstrings in NumPy format
- No runtime dependencies beyond numpy and torch (optional)

---

## Testing Requirements

### Zig Tests (`zig build test`)
1. Vulkan context initialization on available GPU
2. Buffer allocation/deallocation without leaks
3. Trivial compute shader round-trip
4. Attention forward numerical correctness vs reference
5. Attention backward numerical correctness vs reference

### Python Tests (`pytest python/tests/`)
1. Library loads without error
2. Forward pass matches PyTorch SDPA within 1e-3 (fp32) / 1e-2 (fp16)
3. Backward pass matches PyTorch autograd within 1e-3 (fp32) / 1e-2 (fp16)
4. Causal masking correctness
5. Variable sequence lengths
6. Batch size > 1
7. Multi-head attention

### Benchmarks
- Compare against: PyTorch SDPA, FlashAttention-2 (CUDA), xformers
- Report: throughput (TFLOPS), memory usage, latency (ms)

---

## Error Handling

All errors must be:
1. Recoverable where possible (return error code, not crash)
2. Descriptive (store message in thread-local buffer, retrieve via `aule_get_error()`)
3. Logged at appropriate level

Error categories:
- `AULE_ERROR_VULKAN_INIT` — Failed to create instance/device
- `AULE_ERROR_NO_COMPUTE_QUEUE` — No compute-capable queue family
- `AULE_ERROR_OUT_OF_MEMORY` — GPU allocation failed
- `AULE_ERROR_INVALID_SHAPE` — Tensor dimensions don't match requirements
- `AULE_ERROR_SHADER_COMPILE` — SPIR-V failed to create pipeline

---

## Build Commands

```bash
# Development build
zig build

# Run all tests
zig build test

# Run a single Zig test file
zig build test --test-filter "test_attention"

# Run specific Python test
cd python && pytest tests/test_attention.py -v
pytest tests/test_attention.py::test_forward_causal -v  # single test function

# Release build for current platform
zig build -Doptimize=ReleaseFast

# Cross-compile for Linux
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast

# Compile shaders (requires glslc from Vulkan SDK)
cd shaders && ./compile.sh

# Build Python wheel
cd python && python -m build --wheel

# Run Python tests
cd python && pytest tests/ -v
```

---

## Dependencies

### Build-time
- Zig 0.13.0+
- Vulkan SDK (for glslc shader compiler)
- vulkan-zig (fetched via build.zig.zon)

### Runtime
- Vulkan 1.2+ capable driver
- No other dependencies (statically linked)

### Python
- numpy (required)
- torch (optional, for autograd integration)
- cffi or ctypes (stdlib)

---

## Checklist Before Commit

- [ ] `zig build` succeeds with no warnings
- [ ] `zig build test` passes all tests
- [ ] No `// TODO` or placeholder code
- [ ] All public functions have documentation
- [ ] Shaders compile with glslc without warnings
- [ ] Python tests pass
- [ ] Memory leak check (run with sanitizers if available)
- [ ] Tested on at least one real GPU

---

## Contact

Project Lead: Yeabsira (Aule Technologies)  
Purpose: Production-ready, portable FlashAttention for the AMD/Intel/Apple ecosystem