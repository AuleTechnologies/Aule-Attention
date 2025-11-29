const std = @import("std");
const vk = @import("vulkan");

pub const VulkanContext = @import("vulkan_context.zig").VulkanContext;
pub const BufferManager = @import("buffer_manager.zig").BufferManager;
pub const Buffer = @import("buffer_manager.zig").Buffer;
pub const ComputePipeline = @import("compute_pipeline.zig").ComputePipeline;
pub const AttentionPipeline = @import("attention_pipeline.zig").AttentionPipeline;
pub const AttentionRef = @import("attention_ref.zig").AttentionRef;
pub const GpuTensor = @import("gpu_tensor.zig").GpuTensor;
pub const AttentionEngine = @import("attention_gpu.zig").AttentionEngine;

const log = std.log.scoped(.aule);

// Global state for C API
var global_context: ?VulkanContext = null;
var global_pipeline: ?ComputePipeline = null;
var global_attention_pipeline: ?AttentionPipeline = null;
var global_buffer_manager: BufferManager = undefined;
var global_buffer_manager_initialized: bool = false;
var global_error_message: [512]u8 = undefined;
var global_error_len: usize = 0;

fn setError(comptime fmt: []const u8, args: anytype) void {
    global_error_len = (std.fmt.bufPrint(&global_error_message, fmt, args) catch &global_error_message).len;
}

// Embedded shader SPIR-V (compiled at build time)
const test_shader_spv = @embedFile("test_shader_spv");
const attention_f32_spv = @embedFile("attention_f32_spv");
const attention_amd_spv = @embedFile("attention_amd_spv");

// ============================================================================
// C API
// ============================================================================

export fn aule_init() callconv(.C) i32 {
    if (global_context != null) {
        return 0; // Already initialized
    }

    global_context = VulkanContext.init(std.heap.c_allocator) catch |err| {
        setError("Failed to initialize Vulkan: {}", .{err});
        log.err("Vulkan init failed: {}", .{err});
        return -1;
    };

    global_buffer_manager = BufferManager.init(&global_context.?);
    global_buffer_manager_initialized = true;

    global_pipeline = ComputePipeline.init(&global_context.?, test_shader_spv) catch |err| {
        setError("Failed to create compute pipeline: {}", .{err});
        log.err("Pipeline init failed: {}", .{err});
        global_context.?.deinit();
        global_context = null;
        global_buffer_manager_initialized = false;
        return -2;
    };

    // Select shader based on GPU vendor for optimal performance
    const gpu_caps = global_context.?.gpu_caps;
    const shader_to_use = if (gpu_caps.isAmd())
        attention_amd_spv // AMD-optimized: 64-wide wavefront, subgroup ops
    else
        attention_f32_spv; // Generic: 16x16 workgroup

    log.info("Using {s} shader for {s} GPU", .{
        if (gpu_caps.isAmd()) "AMD-optimized" else "generic",
        @tagName(gpu_caps.vendor),
    });

    global_attention_pipeline = AttentionPipeline.init(&global_context.?, shader_to_use) catch |err| {
        setError("Failed to create attention pipeline: {}", .{err});
        log.err("Attention pipeline init failed: {}", .{err});
        global_pipeline.?.deinit();
        global_pipeline = null;
        global_context.?.deinit();
        global_context = null;
        global_buffer_manager_initialized = false;
        return -3;
    };

    log.info("aule initialized successfully", .{});
    return 0;
}

export fn aule_shutdown() callconv(.C) void {
    // First destroy all tensors
    for (&tensor_storage) |*slot| {
        if (slot.*) |*tensor| {
            tensor.deinit();
            slot.* = null;
        }
    }

    if (global_attention_pipeline) |*pipeline| {
        pipeline.deinit();
        global_attention_pipeline = null;
    }
    if (global_pipeline) |*pipeline| {
        pipeline.deinit();
        global_pipeline = null;
    }
    global_buffer_manager_initialized = false;
    if (global_context) |*ctx| {
        ctx.waitIdle() catch {};
        ctx.deinit();
        global_context = null;
    }
    log.info("aule shutdown complete", .{});
}

export fn aule_get_error() callconv(.C) [*:0]const u8 {
    if (global_error_len == 0) {
        return "No error";
    }
    global_error_message[global_error_len] = 0;
    return @ptrCast(&global_error_message);
}

/// Get GPU device name
export fn aule_get_device_name() callconv(.C) [*:0]const u8 {
    if (global_context) |ctx| {
        return @ptrCast(&ctx.gpu_caps.device_name);
    }
    return "Not initialized";
}

/// Get GPU vendor ID: 0=other, 1=amd, 2=nvidia, 3=intel, 4=apple
export fn aule_get_vendor() callconv(.C) i32 {
    if (global_context) |ctx| {
        return switch (ctx.gpu_caps.vendor) {
            .other => 0,
            .amd => 1,
            .nvidia => 2,
            .intel => 3,
            .apple => 4,
        };
    }
    return -1; // Not initialized
}

/// Returns 1 if using AMD-optimized shader, 0 for generic
export fn aule_is_amd_optimized() callconv(.C) i32 {
    if (global_context) |ctx| {
        return if (ctx.gpu_caps.isAmd()) 1 else 0;
    }
    return -1;
}

/// Returns 1 if FP16 is supported
export fn aule_has_fp16() callconv(.C) i32 {
    if (global_context) |ctx| {
        return if (ctx.gpu_caps.fp16_supported) 1 else 0;
    }
    return -1;
}

/// Get subgroup/wavefront size
export fn aule_get_subgroup_size() callconv(.C) u32 {
    if (global_context) |ctx| {
        return ctx.gpu_caps.subgroup_size;
    }
    return 0;
}

export fn aule_test_multiply(
    input: [*]const f32,
    output: [*]f32,
    count: u32,
) callconv(.C) i32 {
    const ctx = global_context orelse {
        setError("Library not initialized. Call aule_init() first.", .{});
        return -1;
    };
    if (!global_buffer_manager_initialized) {
        setError("Buffer manager not available.", .{});
        return -1;
    }
    const buffer_manager = &global_buffer_manager;
    const pipeline = global_pipeline orelse {
        setError("Pipeline not available.", .{});
        return -1;
    };

    const size: vk.DeviceSize = @as(vk.DeviceSize, count) * @sizeOf(f32);

    // Create buffers
    var staging_in = buffer_manager.createStagingBuffer(size) catch |err| {
        setError("Failed to create input staging buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&staging_in);

    var staging_out = buffer_manager.createStagingBuffer(size) catch |err| {
        setError("Failed to create output staging buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&staging_out);

    var device_in = buffer_manager.createDeviceLocalBuffer(size) catch |err| {
        setError("Failed to create device input buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&device_in);

    var device_out = buffer_manager.createDeviceLocalBuffer(size) catch |err| {
        setError("Failed to create device output buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&device_out);

    // Copy input data to staging buffer
    const staging_slice = staging_in.getMappedSlice(f32);
    @memcpy(staging_slice, input[0..count]);

    // Update descriptors to point to device buffers
    pipeline.updateDescriptors(device_in.buffer, device_out.buffer, size);

    // Record and execute: staging_in -> device_in -> compute -> device_out -> staging_out
    pipeline.recordCopyAndDispatch(
        staging_in.buffer,
        device_in.buffer,
        device_out.buffer,
        staging_out.buffer,
        size,
        count,
    ) catch |err| {
        setError("Failed to dispatch compute: {}", .{err});
        return -3;
    };

    // Ensure device writes are visible to host
    ctx.waitIdle() catch |err| {
        setError("Failed to wait for device: {}", .{err});
        return -3;
    };

    // Copy output data from staging buffer
    const output_slice = staging_out.getMappedSlice(f32);
    @memcpy(output[0..count], output_slice);

    return 0;
}

/// Compute FlashAttention forward pass on GPU
/// Q, K, V, output: pointers to [batch_size * num_heads * seq_len * head_dim] f32 arrays
/// causal: 1 for causal masking (LLMs/autoregressive), 0 for bidirectional
/// Returns 0 on success, negative error code on failure
export fn aule_attention_forward(
    query: [*]const f32,
    key: [*]const f32,
    value: [*]const f32,
    output: [*]f32,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: i32,
) callconv(.C) i32 {
    const ctx = global_context orelse {
        setError("Library not initialized. Call aule_init() first.", .{});
        return -1;
    };
    if (!global_buffer_manager_initialized) {
        setError("Buffer manager not available.", .{});
        return -1;
    }
    const buffer_manager = &global_buffer_manager;
    const pipeline = global_attention_pipeline orelse {
        setError("Attention pipeline not available.", .{});
        return -1;
    };

    // Validate head_dim (current shader limit)
    if (head_dim > 64) {
        setError("head_dim must be <= 64 (got {})", .{head_dim});
        return -4;
    }

    const total_elements = @as(usize, batch_size) * num_heads * seq_len * head_dim;
    const size: vk.DeviceSize = @as(vk.DeviceSize, total_elements) * @sizeOf(f32);

    // Create GPU buffers
    var q_buf = buffer_manager.createHostVisibleStorageBuffer(size) catch |err| {
        setError("Failed to create Q buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&q_buf);

    var k_buf = buffer_manager.createHostVisibleStorageBuffer(size) catch |err| {
        setError("Failed to create K buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&k_buf);

    var v_buf = buffer_manager.createHostVisibleStorageBuffer(size) catch |err| {
        setError("Failed to create V buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&v_buf);

    var o_buf = buffer_manager.createHostVisibleStorageBuffer(size) catch |err| {
        setError("Failed to create output buffer: {}", .{err});
        return -2;
    };
    defer buffer_manager.destroyBuffer(&o_buf);

    // Copy input data to GPU
    @memcpy(q_buf.getMappedSlice(f32), query[0..total_elements]);
    @memcpy(k_buf.getMappedSlice(f32), key[0..total_elements]);
    @memcpy(v_buf.getMappedSlice(f32), value[0..total_elements]);

    // Update descriptors
    pipeline.updateDescriptors(q_buf.buffer, k_buf.buffer, v_buf.buffer, o_buf.buffer, size);

    // Dispatch compute shader with causal masking option
    pipeline.dispatch(batch_size, num_heads, seq_len, head_dim, causal != 0) catch |err| {
        setError("Failed to dispatch attention: {}", .{err});
        return -3;
    };

    ctx.waitIdle() catch |err| {
        setError("Failed to wait for device: {}", .{err});
        return -3;
    };

    // Copy output back
    @memcpy(output[0..total_elements], o_buf.getMappedSlice(f32));

    return 0;
}

// ============================================================================
// C API - Persistent GPU Tensors (Zero-Copy Operations)
// ============================================================================

/// Opaque handle to a GPU tensor
const TensorHandle = u64;

/// Storage for GPU tensors (simple array for now)
var tensor_storage: [256]?GpuTensor = [_]?GpuTensor{null} ** 256;
var next_tensor_id: u64 = 1;

fn allocTensorSlot() ?usize {
    for (tensor_storage, 0..) |slot, i| {
        if (slot == null) return i;
    }
    return null;
}

/// Create a GPU tensor. Returns handle or 0 on error.
/// Shape: [batch_size, num_heads, seq_len, head_dim]
export fn aule_tensor_create(
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
) callconv(.C) TensorHandle {
    if (!global_buffer_manager_initialized) {
        setError("Library not initialized. Call aule_init() first.", .{});
        return 0;
    }

    const slot = allocTensorSlot() orelse {
        setError("Too many tensors allocated (max 256)", .{});
        return 0;
    };

    const shape = [_]u32{ batch_size, num_heads, seq_len, head_dim };
    tensor_storage[slot] = GpuTensor.init(&global_buffer_manager, &shape, .f32) catch |err| {
        setError("Failed to create tensor: {}", .{err});
        return 0;
    };

    return @as(TensorHandle, slot + 1); // 1-indexed handles (0 = invalid)
}

/// Destroy a GPU tensor and free its memory
export fn aule_tensor_destroy(handle: TensorHandle) callconv(.C) void {
    if (handle == 0 or handle > 256) return;
    const slot = @as(usize, @intCast(handle - 1));

    if (tensor_storage[slot]) |*tensor| {
        tensor.deinit();
        tensor_storage[slot] = null;
    }
}

/// Upload data from CPU to GPU tensor
export fn aule_tensor_upload(
    handle: TensorHandle,
    data: [*]const f32,
    count: u32,
) callconv(.C) i32 {
    if (handle == 0 or handle > 256) {
        setError("Invalid tensor handle", .{});
        return -1;
    }
    const slot = @as(usize, @intCast(handle - 1));

    const tensor = &(tensor_storage[slot] orelse {
        setError("Tensor not found", .{});
        return -1;
    });

    if (count != tensor.element_count) {
        setError("Size mismatch: expected {}, got {}", .{ tensor.element_count, count });
        return -2;
    }

    tensor.upload(data[0..count]) catch |err| {
        setError("Upload failed: {}", .{err});
        return -3;
    };

    return 0;
}

/// Download data from GPU tensor to CPU
export fn aule_tensor_download(
    handle: TensorHandle,
    output: [*]f32,
    count: u32,
) callconv(.C) i32 {
    if (handle == 0 or handle > 256) {
        setError("Invalid tensor handle", .{});
        return -1;
    }
    const slot = @as(usize, @intCast(handle - 1));

    const tensor = &(tensor_storage[slot] orelse {
        setError("Tensor not found", .{});
        return -1;
    });

    if (count != tensor.element_count) {
        setError("Size mismatch: expected {}, got {}", .{ tensor.element_count, count });
        return -2;
    }

    tensor.download(output[0..count]) catch |err| {
        setError("Download failed: {}", .{err});
        return -3;
    };

    return 0;
}

/// Compute attention on GPU tensors - NO CPU<->GPU COPY
/// This is the fast path for repeated operations
/// causal: 1 for causal masking (LLMs/autoregressive), 0 for bidirectional
export fn aule_attention_forward_gpu(
    q_handle: TensorHandle,
    k_handle: TensorHandle,
    v_handle: TensorHandle,
    output_handle: TensorHandle,
    causal: i32,
) callconv(.C) i32 {
    const ctx = global_context orelse {
        setError("Library not initialized", .{});
        return -1;
    };
    const pipeline = global_attention_pipeline orelse {
        setError("Attention pipeline not available", .{});
        return -1;
    };

    // Get tensors
    if (q_handle == 0 or q_handle > 256 or
        k_handle == 0 or k_handle > 256 or
        v_handle == 0 or v_handle > 256 or
        output_handle == 0 or output_handle > 256)
    {
        setError("Invalid tensor handle", .{});
        return -1;
    }

    const q_tensor = &(tensor_storage[@intCast(q_handle - 1)] orelse {
        setError("Q tensor not found", .{});
        return -1;
    });
    const k_tensor = &(tensor_storage[@intCast(k_handle - 1)] orelse {
        setError("K tensor not found", .{});
        return -1;
    });
    const v_tensor = &(tensor_storage[@intCast(v_handle - 1)] orelse {
        setError("V tensor not found", .{});
        return -1;
    });
    const o_tensor = &(tensor_storage[@intCast(output_handle - 1)] orelse {
        setError("Output tensor not found", .{});
        return -1;
    });

    // Verify shapes
    const batch_size = q_tensor.shape[0];
    const num_heads = q_tensor.shape[1];
    const seq_len = q_tensor.shape[2];
    const head_dim = q_tensor.shape[3];

    if (head_dim > 64) {
        setError("head_dim must be <= 64", .{});
        return -4;
    }

    // Update descriptors to point directly to GPU tensors
    pipeline.updateDescriptors(
        q_tensor.getBuffer(),
        k_tensor.getBuffer(),
        v_tensor.getBuffer(),
        o_tensor.getBuffer(),
        q_tensor.byteSize(),
    );

    // Dispatch compute - data stays on GPU!
    pipeline.dispatch(batch_size, num_heads, seq_len, head_dim, causal != 0) catch |err| {
        setError("Dispatch failed: {}", .{err});
        return -3;
    };

    ctx.waitIdle() catch |err| {
        setError("Sync failed: {}", .{err});
        return -3;
    };

    return 0;
}

/// Get element count of a tensor
export fn aule_tensor_size(handle: TensorHandle) callconv(.C) u32 {
    if (handle == 0 or handle > 256) return 0;
    const slot = @as(usize, @intCast(handle - 1));

    if (tensor_storage[slot]) |tensor| {
        return @intCast(tensor.element_count);
    }
    return 0;
}

// ============================================================================
// Zig API (for tests)
// ============================================================================

pub const Aule = struct {
    context: VulkanContext,
    buffer_manager: BufferManager,
    pipeline: ComputePipeline,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        var context = try VulkanContext.init(allocator);
        errdefer context.deinit();

        const buffer_manager = BufferManager.init(&context);
        var pipeline = try ComputePipeline.init(&context, test_shader_spv);
        errdefer pipeline.deinit();

        return Self{
            .context = context,
            .buffer_manager = buffer_manager,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *Self) void {
        self.pipeline.deinit();
        self.context.deinit();
        self.* = undefined;
    }

    pub fn testMultiply(self: *Self, input: []const f32, output: []f32) !void {
        std.debug.assert(input.len == output.len);
        const count: u32 = @intCast(input.len);
        const size: vk.DeviceSize = @as(vk.DeviceSize, count) * @sizeOf(f32);

        // Use host-visible storage buffers (simpler, works on integrated GPUs)
        var input_buf = try self.buffer_manager.createHostVisibleStorageBuffer(size);
        defer self.buffer_manager.destroyBuffer(&input_buf);

        var output_buf = try self.buffer_manager.createHostVisibleStorageBuffer(size);
        defer self.buffer_manager.destroyBuffer(&output_buf);

        // Copy input to buffer
        const input_slice = input_buf.getMappedSlice(f32);
        @memcpy(input_slice, input);

        // Update descriptors
        self.pipeline.updateDescriptors(input_buf.buffer, output_buf.buffer, size);

        // Execute (simple dispatch, no staging/barriers needed for host-visible)
        try self.pipeline.dispatch(count);

        try self.context.waitIdle();

        // Copy output from buffer
        const output_slice = output_buf.getMappedSlice(f32);
        @memcpy(output, output_slice);
    }
};

/// FlashAttention GPU implementation
pub const Attention = struct {
    context: VulkanContext,
    buffer_manager: BufferManager,
    pipeline: AttentionPipeline,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        var context = try VulkanContext.init(allocator);
        errdefer context.deinit();

        const buffer_manager = BufferManager.init(&context);

        // Select shader based on GPU vendor
        const shader_to_use = if (context.gpu_caps.isAmd())
            attention_amd_spv
        else
            attention_f32_spv;

        var pipeline = try AttentionPipeline.init(&context, shader_to_use);
        errdefer pipeline.deinit();

        return Self{
            .context = context,
            .buffer_manager = buffer_manager,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *Self) void {
        self.pipeline.deinit();
        self.context.deinit();
        self.* = undefined;
    }

    /// Compute attention: output = softmax(Q @ K^T / sqrt(d)) @ V
    /// All tensors are [batch_size, num_heads, seq_len, head_dim] in row-major order
    /// causal: true for causal masking (LLMs/autoregressive), false for bidirectional
    pub fn forward(
        self: *Self,
        Q: []const f32,
        K: []const f32,
        V: []const f32,
        output: []f32,
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        head_dim: u32,
        causal: bool,
    ) !void {
        const total_elements = @as(usize, batch_size) * num_heads * seq_len * head_dim;
        std.debug.assert(Q.len == total_elements);
        std.debug.assert(K.len == total_elements);
        std.debug.assert(V.len == total_elements);
        std.debug.assert(output.len == total_elements);

        const size: vk.DeviceSize = @as(vk.DeviceSize, total_elements) * @sizeOf(f32);

        // Create GPU buffers
        var q_buf = try self.buffer_manager.createHostVisibleStorageBuffer(size);
        defer self.buffer_manager.destroyBuffer(&q_buf);

        var k_buf = try self.buffer_manager.createHostVisibleStorageBuffer(size);
        defer self.buffer_manager.destroyBuffer(&k_buf);

        var v_buf = try self.buffer_manager.createHostVisibleStorageBuffer(size);
        defer self.buffer_manager.destroyBuffer(&v_buf);

        var o_buf = try self.buffer_manager.createHostVisibleStorageBuffer(size);
        defer self.buffer_manager.destroyBuffer(&o_buf);

        // Copy input data to GPU
        @memcpy(q_buf.getMappedSlice(f32), Q);
        @memcpy(k_buf.getMappedSlice(f32), K);
        @memcpy(v_buf.getMappedSlice(f32), V);

        // Update descriptors
        self.pipeline.updateDescriptors(q_buf.buffer, k_buf.buffer, v_buf.buffer, o_buf.buffer, size);

        // Dispatch compute shader
        try self.pipeline.dispatch(batch_size, num_heads, seq_len, head_dim, causal);

        try self.context.waitIdle();

        // Copy output back
        @memcpy(output, o_buf.getMappedSlice(f32));
    }
};
