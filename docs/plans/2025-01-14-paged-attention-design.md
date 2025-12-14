# PagedAttention Design for Aule

**Date:** 2025-01-14
**Version:** 0.4.0 (target)
**Status:** Design Complete, Ready for Implementation

## Overview

Implement PagedAttention to enable efficient memory management for long-context attention and batched inference. This unlocks vLLM-style serving capabilities on Vulkan hardware.

### Goals

- **Primary**: Support sequences >2048 tokens without OOM
- **Secondary**: Enable efficient batched inference with variable-length sequences
- **Tertiary**: Foundation for continuous batching (future)

### Non-Goals (Future Work)

- Continuous batching / dynamic request scheduling (v0.5.0)
- Multi-GPU PagedAttention (v0.6.0)
- Prefix caching / KV cache sharing (v0.5.0)

---

## Architecture

### Core Concept

PagedAttention breaks the KV cache into fixed-size **blocks** (32 tokens each) stored in a shared **physical block pool**. Each request maintains a **block table** that maps logical token positions to physical blocks.

**Memory Model Comparison:**

```
Current (Contiguous):
  Request 1: [----2048 tokens----] = 512KB
  Request 2: [----1024 tokens----] = 256KB
  Total allocated: 768KB (even if requests end early)

Paged:
  Physical Pool: [Block0][Block1][Block2]...[BlockN]
  Request 1 table: [0, 5, 12, ...] → uses 64 blocks
  Request 2 table: [3, 7, 9, ...]  → uses 32 blocks
  Total allocated: 96 blocks × 8KB = 768KB (shared, reusable)
```

### Automatic Path Selection

```zig
if (seq_len > paged_threshold) {
    return forwardPaged(q, k, v);  // Long context
} else {
    return forwardContiguous(q, k, v);  // Fast path
}
```

- **Default threshold**: 2048 tokens
- **Rationale**: Short sequences benefit from contiguous (no indirection), long sequences benefit from paged (memory efficiency)
- **Configurable**: `Aule(paged_threshold=1024)`

---

## Data Structures

### 1. Block Pool Manager (`src/block_pool.zig`)

```zig
pub const BlockPool = struct {
    // Physical storage: all KV blocks live here
    kv_pool: Buffer,  // [num_blocks, 2, num_kv_heads, block_size, head_dim]
                      // 2 = K and V

    // Free block tracking
    free_blocks: std.ArrayList(u32),  // Stack of available block IDs
    total_blocks: u32,
    block_size: u32,  // 32 tokens

    // Chunked allocation
    blocks_per_chunk: u32,  // 512 blocks
    max_blocks: u32,  // Upper limit (e.g., 8192)

    pub fn init(allocator, buffer_manager, config) !BlockPool;
    pub fn allocateBlock() !u32;  // Pop from free_blocks
    pub fn freeBlock(block_id: u32) void;  // Push to free_blocks
    pub fn growPool() !void;  // Allocate next 512-block chunk
    pub fn deinit() void;
};
```

**Chunked Allocation:**
- Start with 512 blocks (~16K tokens, ~4MB for fp16)
- Grow in 512-block chunks when free_blocks exhausted
- Max 8192 blocks (~256K tokens, ~64MB)

**Why 512-block chunks?**
- Allocation frequency: ~every 10K tokens generated
- Latency spike: ~1-2ms (acceptable)
- Memory overhead: 4MB chunks are reasonable

### 2. Block Table (`src/block_table.zig`)

```zig
pub const BlockTable = struct {
    // Maps logical blocks to physical blocks
    // Shape: [batch_size, max_blocks_per_request]
    table_buffer: Buffer,  // GPU-side SSBO
    table_cpu: []i32,  // CPU-side mirror for updates

    batch_size: u32,
    max_blocks: u32,

    pub fn init(allocator, buffer_manager, batch_size, max_blocks) !BlockTable;
    pub fn set(request_id: u32, logical_block: u32, physical_block: u32) void;
    pub fn get(request_id: u32, logical_block: u32) i32;
    pub fn sync() !void;  // Upload CPU table to GPU
    pub fn deinit() void;
};
```

**Padding Strategy:**
- Table is `[batch_size, max_blocks_in_batch]`
- Shorter requests padded with `-1` (sentinel)
- Memory overhead: <1% (4 bytes per entry, even 1024 blocks = 4KB per request)

### 3. Request Metadata

```zig
pub const RequestMetadata = struct {
    request_id: u32,
    seq_len: u32,
    num_blocks: u32,  // ceil(seq_len / 32)
    allocated_blocks: std.ArrayList(u32),  // Physical block IDs
};
```

---

## Shader Implementation

### New Shader: `shaders/attention_paged.comp`

Based on `attention_f32_fast.comp` with paged KV lookup.

**Additional Bindings:**

```glsl
// Binding 6: Block table
layout(std430, set = 0, binding = 6) readonly buffer BlockTableBuffer {
    int data[];  // [batch_size * max_blocks], -1 = sentinel
} BlockTable;

// Binding 7: KV pool
layout(std430, set = 0, binding = 7) readonly buffer KVPoolBuffer {
    vec4 data[];  // [num_blocks * 2 * num_kv_heads * 32 * (head_dim/4)]
} KVPool;

layout(push_constant) uniform PushConstants {
    // ... existing params ...
    uint max_blocks_per_request;  // For indexing block table
    uint block_size;  // 32
} params;
```

**Block Lookup Functions:**

```glsl
// Get physical block ID for a given request and logical position
int getPhysicalBlock(uint request_id, uint token_pos) {
    uint logical_block = token_pos / params.block_size;
    uint table_idx = request_id * params.max_blocks_per_request + logical_block;
    return BlockTable.data[table_idx];
}

// Load K from paged pool
vec4 loadPagedK(uint request_id, uint kv_head, uint token_pos, uint vec_idx) {
    int phys_block = getPhysicalBlock(request_id, token_pos);
    if (phys_block < 0) return vec4(0.0);  // Sentinel, return zero

    uint offset_in_block = token_pos % params.block_size;

    // KV pool layout: [num_blocks, 2, num_kv_heads, block_size, head_dim]
    uint pool_idx = phys_block * (2 * params.num_kv_heads * params.block_size * HEAD_DIM_VEC4)
                  + 0 * (params.num_kv_heads * params.block_size * HEAD_DIM_VEC4)  // K=0
                  + kv_head * (params.block_size * HEAD_DIM_VEC4)
                  + offset_in_block * HEAD_DIM_VEC4
                  + vec_idx;
    return KVPool.data[pool_idx];
}

// Load V from paged pool (similar, with V=1 instead of K=0)
vec4 loadPagedV(uint request_id, uint kv_head, uint token_pos, uint vec_idx) {
    int phys_block = getPhysicalBlock(request_id, token_pos);
    if (phys_block < 0) return vec4(0.0);

    uint offset_in_block = token_pos % params.block_size;
    uint pool_idx = phys_block * (2 * params.num_kv_heads * params.block_size * HEAD_DIM_VEC4)
                  + 1 * (params.num_kv_heads * params.block_size * HEAD_DIM_VEC4)  // V=1
                  + kv_head * (params.block_size * HEAD_DIM_VEC4)
                  + offset_in_block * HEAD_DIM_VEC4
                  + vec_idx;
    return KVPool.data[pool_idx];
}
```

**Main Loop Modification:**

```glsl
// Instead of loading K from contiguous buffer:
// vec4 k_val = K.data[k_idx];

// Use paged lookup:
for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
    uint global_kv_pos = kv_block * BLOCK_SIZE + local_col;

    // Load K tile from paged pool
    for (uint v = 0; v < HEAD_DIM_VEC4; v++) {
        s_K[local_col][v] = loadPagedK(batch_idx, kv_head_idx, global_kv_pos, v);
    }

    barrier();

    // ... rest of attention logic unchanged ...

    // Load V tile from paged pool
    for (uint v = 0; v < HEAD_DIM_VEC4; v++) {
        s_V[local_col][v] = loadPagedV(batch_idx, kv_head_idx, global_kv_pos, v);
    }

    // ... continue as before ...
}
```

---

## Host-Side Workflow

### Attention Dispatch (`src/attention_gpu.zig`)

```zig
pub fn forward(
    self: *AttentionGPU,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    causal: bool,
    scale: ?f32,
) !GpuTensor {
    const seq_len = q.shape[2];

    // Auto-select paged vs contiguous
    if (seq_len > self.paged_threshold) {
        return self.forwardPaged(q, k, v, causal, scale);
    } else {
        return self.forwardContiguous(q, k, v, causal, scale);
    }
}
```

### Paged Forward Pass

```zig
fn forwardPaged(
    self: *AttentionGPU,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    causal: bool,
    scale: ?f32,
) !GpuTensor {
    const batch = q.shape[0];
    const seq_len = q.shape[2];
    const blocks_needed = (seq_len + 31) / 32;  // ceil division

    // 1. Ensure block pool has capacity
    const total_blocks_needed = blocks_needed * batch;
    if (self.block_pool.free_blocks.items.len < total_blocks_needed) {
        try self.block_pool.growPool();
    }

    // 2. Allocate blocks for this batch
    var block_table = try BlockTable.init(
        self.allocator,
        &self.buffer_manager,
        batch,
        blocks_needed,
    );
    defer block_table.deinit();

    // Allocate physical blocks
    for (0..batch) |req_id| {
        for (0..blocks_needed) |block_idx| {
            const phys_block = try self.block_pool.allocateBlock();
            block_table.set(@intCast(req_id), @intCast(block_idx), @intCast(phys_block));
        }
    }
    try block_table.sync();  // Upload to GPU

    // 3. Copy K/V into paged pool
    try self.copyKVToPaged(k, v, &block_table, batch, blocks_needed);

    // 4. Run paged attention shader
    const output = try self.runPagedShader(
        q,
        &block_table,
        causal,
        scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(q.shape[3])))),
    );

    // 5. Free blocks
    for (0..batch) |req_id| {
        for (0..blocks_needed) |block_idx| {
            const phys_block = block_table.get(@intCast(req_id), @intCast(block_idx));
            if (phys_block >= 0) {
                self.block_pool.freeBlock(@intCast(phys_block));
            }
        }
    }

    return output;
}
```

### KV Copy to Paged Pool

```zig
fn copyKVToPaged(
    self: *AttentionGPU,
    k: GpuTensor,
    v: GpuTensor,
    block_table: *BlockTable,
    batch: u32,
    blocks_per_request: u32,
) !void {
    // For each request in batch
    for (0..batch) |req_id| {
        // For each block
        for (0..blocks_per_request) |block_idx| {
            const phys_block = block_table.get(@intCast(req_id), @intCast(block_idx));
            if (phys_block < 0) continue;  // Sentinel

            // Copy 32 tokens of K and V to physical block
            const src_offset = (req_id * k.shape[2] + block_idx * 32) * k.shape[3];
            const dst_offset = phys_block * (2 * k.shape[1] * 32 * k.shape[3]);

            // Use vkCmdCopyBuffer for GPU-side copy
            // (Implementation detail: record copy commands)
        }
    }
}
```

---

## Python API

### Transparent Usage (No API Changes)

```python
import torch
from aule import flash_attention

# Short sequence - uses contiguous automatically
q = torch.randn(2, 8, 1024, 64)
out = flash_attention(q, k, v)  # Contiguous backend

# Long sequence - automatically uses paged
q = torch.randn(2, 8, 4096, 64)
out = flash_attention(q, k, v)  # Paged backend
```

### Advanced Configuration

```python
from aule.vulkan import Aule

# Configure threshold
with Aule(paged_threshold=1024) as aule:
    # Sequences >1024 use paged
    out = aule.attention(q, k, v)

# Force specific backend (for testing)
out = aule.attention(q, k, v, force_paged=True)
out = aule.attention(q, k, v, force_contiguous=True)
```

---

## Testing Strategy

### 1. Unit Tests (`tests/test_paged_attention.py`)

```python
def test_block_pool_allocation():
    """Test block allocation and deallocation"""
    pool = BlockPool(blocks=512)
    blocks = [pool.allocate() for _ in range(100)]
    assert len(pool.free_blocks) == 412
    for b in blocks:
        pool.free(b)
    assert len(pool.free_blocks) == 512

def test_block_table_indexing():
    """Verify correct physical block lookup"""
    table = BlockTable(batch=4, max_blocks=64)
    table.set(request_id=0, logical_block=5, physical_block=123)
    assert table.get(request_id=0, logical_block=5) == 123
    assert table.get(request_id=0, logical_block=6) == -1  # Sentinel

def test_paged_vs_contiguous_correctness():
    """Same input → same output (numerical correctness)"""
    torch.manual_seed(42)
    q = torch.randn(2, 8, 4096, 64)
    k = torch.randn(2, 8, 4096, 64)
    v = torch.randn(2, 8, 4096, 64)

    with Aule() as aule:
        out_contig = aule.attention(q, k, v, force_contiguous=True)
        out_paged = aule.attention(q, k, v, force_paged=True)

        assert torch.allclose(out_contig, out_paged, atol=1e-3)

def test_variable_length_batch():
    """Batched inference with different sequence lengths"""
    # Batch with S=1024, 2048, 4096
    q = torch.randn(3, 8, 4096, 64)
    k = torch.cat([
        torch.randn(1, 8, 1024, 64),
        torch.randn(1, 8, 2048, 64),
        torch.randn(1, 8, 4096, 64),
    ])
    v = k.clone()

    # Should handle via padding in block table
    out = flash_attention(q, k, v)
    assert out.shape == (3, 8, 4096, 64)
```

### 2. Benchmark Tests

```python
def test_paged_memory_efficiency():
    """Verify paged uses less memory for batched long sequences"""
    # Measure peak memory usage
    pass

def test_paged_throughput():
    """Measure tokens/sec for batch of variable-length sequences"""
    # Compare paged vs contiguous for batch=8, mixed lengths
    pass
```

### 3. Stress Tests

```python
def test_block_pool_growth():
    """Allocate beyond initial pool, verify chunked growth"""
    pool = BlockPool(blocks=512, max_blocks=2048)
    blocks = [pool.allocate() for _ in range(1500)]
    assert pool.total_blocks == 1536  # 512 + 512 + 512

def test_fragmentation_handling():
    """Many alloc/free cycles, ensure no leaks"""
    pool = BlockPool(blocks=512)
    for _ in range(1000):
        blocks = [pool.allocate() for _ in range(100)]
        for b in blocks[::2]:  # Free half
            pool.free(b)
        for b in blocks[1::2]:  # Free other half
            pool.free(b)
    assert len(pool.free_blocks) == 512  # No leaks
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Files to create:**
- `src/block_pool.zig` - Block allocator with chunked growth
- `src/block_table.zig` - Block table management
- `tests/test_block_pool.zig` - Zig unit tests

**Deliverables:**
- Block allocation/deallocation works
- Chunked growth at 512-block intervals
- No memory leaks

### Phase 2: Shader Implementation (Week 1-2)

**Files to create:**
- `shaders/attention_paged.comp` - Based on `attention_f32_fast.comp`
- Add block table and KV pool bindings
- Implement `loadPagedK()` and `loadPagedV()`

**Testing:**
- Test with fixed small block tables (manually constructed)
- Verify output matches contiguous for S=128, S=512

### Phase 3: Integration (Week 2)

**Files to modify:**
- `src/attention_gpu.zig` - Add `forwardPaged()`, threshold dispatch
- `src/attention_pipeline.zig` - Support paged shader variant

**Deliverables:**
- End-to-end paged attention works
- Automatic threshold-based selection
- Python API remains unchanged

### Phase 4: Testing & Optimization (Week 3)

**Tasks:**
- Implement full test suite from "Testing Strategy"
- Benchmark paged vs contiguous (latency, memory, throughput)
- Tune threshold (maybe adaptive based on VRAM)
- Stress test block pool growth
- Memory leak checks

**Deliverables:**
- All tests pass
- Documentation updated
- Benchmark results published

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Correctness** | Max error < 1e-3 vs contiguous | `torch.allclose(out_paged, out_contig, atol=1e-3)` |
| **Memory Efficiency** | 2x longer sequences in same VRAM | Compare max S for 8GB GPU: contiguous vs paged |
| **Performance** | <10% overhead vs contiguous at S=4096 | Benchmark forward pass latency |
| **Batching** | Support batch=8 with mixed lengths | Test variable S in batch |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Paged slower than contiguous** | Users avoid paged | Threshold-based selection keeps short sequences fast |
| **Block table lookup overhead** | High latency | Use vec4 loads, cache in L2, minimal indirection per block |
| **Memory fragmentation** | Inefficient memory use | Chunked allocation, free list management |
| **Debugging difficulty** | Slow development | Extensive CPU-side validation, small test cases first |
| **Backward pass complexity** | Delayed training support | Start with forward only, add backward in Phase 5 (v0.4.1) |

---

## Future Work (Post-MVP)

### v0.4.1: Backward Pass
- Paged backward attention shader
- Gradient computation with block table lookup
- Training support for long contexts

### v0.5.0: Continuous Batching
- Persistent block tables across requests
- Dynamic request scheduling
- Prefix caching / KV cache sharing

### v0.5.0: Prefix Caching
- Share common prefixes (e.g., system prompt) across requests
- Reference-counted blocks

### v0.6.0: Multi-GPU
- Distribute block pool across GPUs
- Cross-GPU block table coordination

---

## References

- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- FlashAttention-2: https://tridao.me/publications/flash2/flash2.pdf
- Current codebase: `src/attention_gpu.zig`, `shaders/attention_f32_fast.comp`

---

**End of Design Document**
