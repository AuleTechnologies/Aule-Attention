const std = @import("std");
const builtin = @import("builtin");
const Backend = @import("backends/backend.zig");
const AttentionContext = Backend.AttentionContext;
const Tensor = Backend.Tensor;

pub const AttentionRef = @import("attention_ref.zig").AttentionRef;

const log = std.log.scoped(.aule);

// Global state for C API
var global_ctx: ?AttentionContext = null;
var global_allocator: std.mem.Allocator = std.heap.c_allocator;

// Tensor storage (simple array for handles)
const MAX_TENSORS = 256;
var tensor_storage: [MAX_TENSORS]?*Tensor = [_]?*Tensor{null} ** MAX_TENSORS;

// Error message buffer
var global_error_message: [512]u8 = undefined;
var global_error_len: usize = 0;

fn setError(comptime fmt: []const u8, args: anytype) void {
    global_error_len = (std.fmt.bufPrint(&global_error_message, fmt, args) catch &global_error_message).len;
}

// Embedded shader SPIR-V (compiled at build time)
const attention_f32_spv = @embedFile("attention_f32_spv");
const attention_amd_spv = @embedFile("attention_amd_spv");
const attention_fwd_lse_spv = @embedFile("attention_fwd_lse_spv");
const attention_bwd_spv = @embedFile("attention_bwd_spv");

// ============================================================================
// C API
// ============================================================================

export fn aule_init() callconv(.C) i32 {
    if (global_ctx != null) {
        return 0; // Already initialized
    }

    global_ctx = AttentionContext.initWithBackward(
        global_allocator,
        attention_f32_spv,
        attention_amd_spv,
        attention_fwd_lse_spv,
        attention_bwd_spv,
    ) catch |err| {
        setError("Failed to initialize backend: {}", .{err});
        log.err("Backend init failed: {}", .{err});
        return -1;
    };

    log.info("aule initialized successfully using {s} backend", .{global_ctx.?.getBackendName()});
    return 0;
}

/// Check if backward pass is supported (training mode)
export fn aule_supports_backward() callconv(.C) i32 {
    if (global_ctx) |*ctx| {
        if (ctx.vulkan_ctx) |engine| {
            return if (engine.supportsBackward()) 1 else 0;
        }
    }
    return 0;
}

export fn aule_shutdown() callconv(.C) void {
    // Destroy all tensors
    for (&tensor_storage) |*slot| {
        if (slot.*) |tensor| {
            if (global_ctx) |*ctx| {
                ctx.destroyTensor(tensor);
                global_allocator.destroy(tensor);
            }
            slot.* = null;
        }
    }

    if (global_ctx) |*ctx| {
        ctx.deinit();
        global_ctx = null;
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

/// Get backend name
export fn aule_get_backend_name() callconv(.C) [*:0]const u8 {
    if (global_ctx) |*ctx| {
        const name = ctx.getBackendName();
        @memcpy(global_error_message[0..name.len], name);
        global_error_message[name.len] = 0;
        return @ptrCast(&global_error_message);
    }
    return "Not initialized";
}

/// Get GPU vendor ID: 0=other, 1=amd, 2=nvidia, 3=intel, 4=apple
export fn aule_get_vendor() callconv(.C) i32 {
    if (global_ctx) |*ctx| {
        switch (ctx.backend) {
            .hip => return 1, // AMD
            .vulkan => {
                if (ctx.vulkan_ctx) |vctx| {
                    return switch (vctx.ctx.gpu_caps.vendor) {
                        .other => 0,
                        .amd => 1,
                        .nvidia => 2,
                        .intel => 3,
                        .apple => 4,
                    };
                }
            },
            .cpu => return 0,
        }
    }
    return -1; // Not initialized
}

/// Get GPU device name
export fn aule_get_device_name() callconv(.C) [*:0]const u8 {
    if (global_ctx) |*ctx| {
        switch (ctx.backend) {
            .vulkan => {
                if (ctx.vulkan_ctx) |vctx| {
                    const name = vctx.ctx.gpu_caps.getDeviceName();
                    @memcpy(global_error_message[0..name.len], name);
                    global_error_message[name.len] = 0;
                    return @ptrCast(&global_error_message);
                }
            },
            .hip => return "AMD HIP Device",
            .cpu => return "CPU",
        }
    }
    return "Not initialized";
}

/// Check if AMD-optimized shader is being used
export fn aule_is_amd_optimized() callconv(.C) i32 {
    if (global_ctx) |*ctx| {
        switch (ctx.backend) {
            .vulkan => {
                if (ctx.vulkan_ctx) |vctx| {
                    return if (vctx.ctx.gpu_caps.isAmd()) 1 else 0;
                }
            },
            .hip => return 1, // HIP is always AMD
            .cpu => return 0,
        }
    }
    return -1;
}

/// Check if FP16 is supported
export fn aule_has_fp16() callconv(.C) i32 {
    if (global_ctx) |*ctx| {
        switch (ctx.backend) {
            .vulkan => {
                if (ctx.vulkan_ctx) |vctx| {
                    return if (vctx.ctx.gpu_caps.fp16_supported) 1 else 0;
                }
            },
            .hip => return 1, // HIP GPUs typically support FP16
            .cpu => return 0,
        }
    }
    return -1;
}

/// Get subgroup (wavefront/warp) size
export fn aule_get_subgroup_size() callconv(.C) i32 {
    if (global_ctx) |*ctx| {
        switch (ctx.backend) {
            .vulkan => {
                if (ctx.vulkan_ctx) |vctx| {
                    return @intCast(vctx.ctx.gpu_caps.subgroup_size);
                }
            },
            .hip => return 64, // AMD wavefront
            .cpu => return 1,
        }
    }
    return -1;
}

/// Compute FlashAttention forward pass
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
    var ctx = global_ctx orelse {
        setError("Library not initialized. Call aule_init() first.", .{});
        return -1;
    };

    const shape = [4]u32{ batch_size, num_heads, seq_len, head_dim };
    const count = batch_size * num_heads * seq_len * head_dim;

    // 1. Create tensors
    const q_tensor = ctx.createTensor(shape) catch |err| { setError("Create Q failed: {}", .{err}); return -2; };
    const q_ptr = global_allocator.create(Tensor) catch return -2;
    q_ptr.* = q_tensor;
    defer { ctx.destroyTensor(q_ptr); global_allocator.destroy(q_ptr); }

    const k_tensor = ctx.createTensor(shape) catch |err| { setError("Create K failed: {}", .{err}); return -2; };
    const k_ptr = global_allocator.create(Tensor) catch return -2;
    k_ptr.* = k_tensor;
    defer { ctx.destroyTensor(k_ptr); global_allocator.destroy(k_ptr); }

    const v_tensor = ctx.createTensor(shape) catch |err| { setError("Create V failed: {}", .{err}); return -2; };
    const v_ptr = global_allocator.create(Tensor) catch return -2;
    v_ptr.* = v_tensor;
    defer { ctx.destroyTensor(v_ptr); global_allocator.destroy(v_ptr); }

    const o_tensor = ctx.createTensor(shape) catch |err| { setError("Create Output failed: {}", .{err}); return -2; };
    const o_ptr = global_allocator.create(Tensor) catch return -2;
    o_ptr.* = o_tensor;
    defer { ctx.destroyTensor(o_ptr); global_allocator.destroy(o_ptr); }

    // 2. Upload data
    ctx.upload(q_ptr, query[0..count]) catch |err| { setError("Upload Q failed: {}", .{err}); return -3; };
    ctx.upload(k_ptr, key[0..count]) catch |err| { setError("Upload K failed: {}", .{err}); return -3; };
    ctx.upload(v_ptr, value[0..count]) catch |err| { setError("Upload V failed: {}", .{err}); return -3; };

    // 3. Compute
    ctx.attention(q_ptr, k_ptr, v_ptr, o_ptr, null, null, causal != 0) catch |err| {
        setError("Attention failed: {}", .{err});
        return -4;
    };

    // 4. Download
    ctx.download(o_ptr, output[0..count]) catch |err| { setError("Download failed: {}", .{err}); return -5; };

    return 0;
}

// ============================================================================
// C API - Persistent GPU Tensors
// ============================================================================

const TensorHandle = u64;

fn allocTensorSlot() ?usize {
    for (tensor_storage, 0..) |slot, i| {
        if (slot == null) return i;
    }
    return null;
}

export fn aule_tensor_create(
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
) callconv(.C) TensorHandle {
    var ctx = global_ctx orelse { setError("Not initialized", .{}); return 0; };
    const slot_idx = allocTensorSlot() orelse { setError("Max tensors reached", .{}); return 0; };

    const shape = [4]u32{ batch_size, num_heads, seq_len, head_dim };
    const tensor = ctx.createTensor(shape) catch |err| {
        setError("Create tensor failed: {}", .{err});
        return 0;
    };

    const ptr = global_allocator.create(Tensor) catch return 0;
    ptr.* = tensor;
    tensor_storage[slot_idx] = ptr;

    return @as(TensorHandle, slot_idx + 1);
}

export fn aule_tensor_destroy(handle: TensorHandle) callconv(.C) void {
    if (handle == 0 or handle > MAX_TENSORS) return;
    const slot_idx = @as(usize, @intCast(handle - 1));

    if (tensor_storage[slot_idx]) |tensor| {
        if (global_ctx) |*ctx| {
            ctx.destroyTensor(tensor);
            global_allocator.destroy(tensor);
        }
        tensor_storage[slot_idx] = null;
    }
}

export fn aule_tensor_upload(handle: TensorHandle, data: [*]const f32, count: u32) callconv(.C) i32 {
    var ctx = global_ctx orelse return -1;
    if (handle == 0 or handle > MAX_TENSORS) return -1;
    const tensor = tensor_storage[@intCast(handle - 1)] orelse return -1;

    ctx.upload(tensor, data[0..count]) catch |err| {
        setError("Upload failed: {}", .{err});
        return -3;
    };
    return 0;
}

export fn aule_tensor_download(handle: TensorHandle, output: [*]f32, count: u32) callconv(.C) i32 {
    var ctx = global_ctx orelse return -1;
    if (handle == 0 or handle > MAX_TENSORS) return -1;
    const tensor = tensor_storage[@intCast(handle - 1)] orelse return -1;

    ctx.download(tensor, output[0..count]) catch |err| {
        setError("Download failed: {}", .{err});
        return -3;
    };
    return 0;
}

export fn aule_attention_forward_gpu(
    q_handle: TensorHandle,
    k_handle: TensorHandle,
    v_handle: TensorHandle,
    output_handle: TensorHandle,
    rot_cos_handle: TensorHandle,
    rot_sin_handle: TensorHandle,
    causal: i32,
) callconv(.C) i32 {
    var ctx = global_ctx orelse return -1;
    
    const q = tensor_storage[@intCast(q_handle - 1)] orelse return -1;
    const k = tensor_storage[@intCast(k_handle - 1)] orelse return -1;
    const v = tensor_storage[@intCast(v_handle - 1)] orelse return -1;
    const o = tensor_storage[@intCast(output_handle - 1)] orelse return -1;
    
    // Optional handles (0 means null)
    var rot_cos: ?*Tensor = null;
    if (rot_cos_handle != 0) {
        rot_cos = tensor_storage[@intCast(rot_cos_handle - 1)];
    }
    
    var rot_sin: ?*Tensor = null;
    if (rot_sin_handle != 0) {
        rot_sin = tensor_storage[@intCast(rot_sin_handle - 1)];
    }

    ctx.attention(q, k, v, o, rot_cos, rot_sin, causal != 0) catch |err| {
        setError("Attention failed: {}", .{err});
        return -3;
    };
    return 0;
}

export fn aule_tensor_size(handle: TensorHandle) callconv(.C) u32 {
    if (handle == 0 or handle > MAX_TENSORS) return 0;
    const slot = @as(usize, @intCast(handle - 1));

    if (tensor_storage[slot]) |tensor| {
        return @intCast(tensor.element_count);
    }
    return 0;
}

/// Compute FlashAttention backward pass (gradients)
/// Requires: Q, K, V, O (output from forward), dO (gradient of output), LSE (log-sum-exp from forward)
/// Outputs: dQ, dK, dV (gradients)
export fn aule_attention_backward(
    query: [*]const f32,
    key: [*]const f32,
    value: [*]const f32,
    output: [*]const f32,
    grad_output: [*]const f32,
    lse: [*]const f32,
    grad_query: [*]f32,
    grad_key: [*]f32,
    grad_value: [*]f32,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: i32,
) callconv(.C) i32 {
    var ctx = global_ctx orelse {
        setError("Library not initialized. Call aule_init() first.", .{});
        return -1;
    };

    // Check if Vulkan backend supports backward
    const engine = ctx.vulkan_ctx orelse {
        setError("Backward pass requires Vulkan backend.", .{});
        return -10;
    };

    if (!engine.supportsBackward()) {
        setError("Backward pass not supported on this backend. Use Triton backend for training.", .{});
        return -10;
    }

    const shape = [4]u32{ batch_size, num_heads, seq_len, head_dim };
    const lse_shape = [4]u32{ batch_size, num_heads, seq_len, 1 };
    const count = batch_size * num_heads * seq_len * head_dim;
    const lse_count = batch_size * num_heads * seq_len;

    // Create tensors for inputs
    const q_tensor = ctx.createTensor(shape) catch |err| { setError("Create Q failed: {}", .{err}); return -2; };
    const q_ptr = global_allocator.create(Tensor) catch return -2;
    q_ptr.* = q_tensor;
    defer { ctx.destroyTensor(q_ptr); global_allocator.destroy(q_ptr); }

    const k_tensor = ctx.createTensor(shape) catch |err| { setError("Create K failed: {}", .{err}); return -2; };
    const k_ptr = global_allocator.create(Tensor) catch return -2;
    k_ptr.* = k_tensor;
    defer { ctx.destroyTensor(k_ptr); global_allocator.destroy(k_ptr); }

    const v_tensor = ctx.createTensor(shape) catch |err| { setError("Create V failed: {}", .{err}); return -2; };
    const v_ptr = global_allocator.create(Tensor) catch return -2;
    v_ptr.* = v_tensor;
    defer { ctx.destroyTensor(v_ptr); global_allocator.destroy(v_ptr); }

    const o_tensor = ctx.createTensor(shape) catch |err| { setError("Create O failed: {}", .{err}); return -2; };
    const o_ptr = global_allocator.create(Tensor) catch return -2;
    o_ptr.* = o_tensor;
    defer { ctx.destroyTensor(o_ptr); global_allocator.destroy(o_ptr); }

    const do_tensor = ctx.createTensor(shape) catch |err| { setError("Create dO failed: {}", .{err}); return -2; };
    const do_ptr = global_allocator.create(Tensor) catch return -2;
    do_ptr.* = do_tensor;
    defer { ctx.destroyTensor(do_ptr); global_allocator.destroy(do_ptr); }

    const lse_tensor = ctx.createTensor(lse_shape) catch |err| { setError("Create LSE failed: {}", .{err}); return -2; };
    const lse_ptr = global_allocator.create(Tensor) catch return -2;
    lse_ptr.* = lse_tensor;
    defer { ctx.destroyTensor(lse_ptr); global_allocator.destroy(lse_ptr); }

    // Create tensors for outputs (gradients)
    const dq_tensor = ctx.createTensor(shape) catch |err| { setError("Create dQ failed: {}", .{err}); return -2; };
    const dq_ptr = global_allocator.create(Tensor) catch return -2;
    dq_ptr.* = dq_tensor;
    defer { ctx.destroyTensor(dq_ptr); global_allocator.destroy(dq_ptr); }

    const dk_tensor = ctx.createTensor(shape) catch |err| { setError("Create dK failed: {}", .{err}); return -2; };
    const dk_ptr = global_allocator.create(Tensor) catch return -2;
    dk_ptr.* = dk_tensor;
    defer { ctx.destroyTensor(dk_ptr); global_allocator.destroy(dk_ptr); }

    const dv_tensor = ctx.createTensor(shape) catch |err| { setError("Create dV failed: {}", .{err}); return -2; };
    const dv_ptr = global_allocator.create(Tensor) catch return -2;
    dv_ptr.* = dv_tensor;
    defer { ctx.destroyTensor(dv_ptr); global_allocator.destroy(dv_ptr); }

    // Upload input data
    ctx.upload(q_ptr, query[0..count]) catch |err| { setError("Upload Q failed: {}", .{err}); return -3; };
    ctx.upload(k_ptr, key[0..count]) catch |err| { setError("Upload K failed: {}", .{err}); return -3; };
    ctx.upload(v_ptr, value[0..count]) catch |err| { setError("Upload V failed: {}", .{err}); return -3; };
    ctx.upload(o_ptr, output[0..count]) catch |err| { setError("Upload O failed: {}", .{err}); return -3; };
    ctx.upload(do_ptr, grad_output[0..count]) catch |err| { setError("Upload dO failed: {}", .{err}); return -3; };
    ctx.upload(lse_ptr, lse[0..lse_count]) catch |err| { setError("Upload LSE failed: {}", .{err}); return -3; };

    // Initialize output gradients to zero
    const zeros = global_allocator.alloc(f32, count) catch return -2;
    defer global_allocator.free(zeros);
    @memset(zeros, 0);
    ctx.upload(dq_ptr, zeros) catch |err| { setError("Upload dQ zeros failed: {}", .{err}); return -3; };
    ctx.upload(dk_ptr, zeros) catch |err| { setError("Upload dK zeros failed: {}", .{err}); return -3; };
    ctx.upload(dv_ptr, zeros) catch |err| { setError("Upload dV zeros failed: {}", .{err}); return -3; };

    // Compute backward pass
    engine.backwardSync(
        q_ptr.handle.vulkan,
        k_ptr.handle.vulkan,
        v_ptr.handle.vulkan,
        o_ptr.handle.vulkan,
        do_ptr.handle.vulkan,
        lse_ptr.handle.vulkan,
        dq_ptr.handle.vulkan,
        dk_ptr.handle.vulkan,
        dv_ptr.handle.vulkan,
        causal != 0,
    ) catch |err| {
        setError("Backward pass failed: {}", .{err});
        return -4;
    };

    // Download gradients
    ctx.download(dq_ptr, grad_query[0..count]) catch |err| { setError("Download dQ failed: {}", .{err}); return -5; };
    ctx.download(dk_ptr, grad_key[0..count]) catch |err| { setError("Download dK failed: {}", .{err}); return -5; };
    ctx.download(dv_ptr, grad_value[0..count]) catch |err| { setError("Download dV failed: {}", .{err}); return -5; };

    return 0;
}

/// Forward pass that also returns LSE (log-sum-exp) for backward
export fn aule_attention_forward_with_lse(
    query: [*]const f32,
    key: [*]const f32,
    value: [*]const f32,
    output: [*]f32,
    lse_out: [*]f32,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: i32,
) callconv(.C) i32 {
    var ctx = global_ctx orelse {
        setError("Library not initialized. Call aule_init() first.", .{});
        return -1;
    };

    // Check if Vulkan backend supports backward
    const engine = ctx.vulkan_ctx orelse {
        setError("Forward with LSE requires Vulkan backend.", .{});
        return -10;
    };

    if (!engine.supportsBackward()) {
        setError("Forward with LSE not supported on this backend.", .{});
        return -10;
    }

    const shape = [4]u32{ batch_size, num_heads, seq_len, head_dim };
    const lse_shape = [4]u32{ batch_size, num_heads, seq_len, 1 };
    const count = batch_size * num_heads * seq_len * head_dim;
    const lse_count = batch_size * num_heads * seq_len;

    // Create tensors
    const q_tensor = ctx.createTensor(shape) catch |err| { setError("Create Q failed: {}", .{err}); return -2; };
    const q_ptr = global_allocator.create(Tensor) catch return -2;
    q_ptr.* = q_tensor;
    defer { ctx.destroyTensor(q_ptr); global_allocator.destroy(q_ptr); }

    const k_tensor = ctx.createTensor(shape) catch |err| { setError("Create K failed: {}", .{err}); return -2; };
    const k_ptr = global_allocator.create(Tensor) catch return -2;
    k_ptr.* = k_tensor;
    defer { ctx.destroyTensor(k_ptr); global_allocator.destroy(k_ptr); }

    const v_tensor = ctx.createTensor(shape) catch |err| { setError("Create V failed: {}", .{err}); return -2; };
    const v_ptr = global_allocator.create(Tensor) catch return -2;
    v_ptr.* = v_tensor;
    defer { ctx.destroyTensor(v_ptr); global_allocator.destroy(v_ptr); }

    const o_tensor = ctx.createTensor(shape) catch |err| { setError("Create Output failed: {}", .{err}); return -2; };
    const o_ptr = global_allocator.create(Tensor) catch return -2;
    o_ptr.* = o_tensor;
    defer { ctx.destroyTensor(o_ptr); global_allocator.destroy(o_ptr); }

    const lse_tensor = ctx.createTensor(lse_shape) catch |err| { setError("Create LSE failed: {}", .{err}); return -2; };
    const lse_ptr = global_allocator.create(Tensor) catch return -2;
    lse_ptr.* = lse_tensor;
    defer { ctx.destroyTensor(lse_ptr); global_allocator.destroy(lse_ptr); }

    // Upload data
    ctx.upload(q_ptr, query[0..count]) catch |err| { setError("Upload Q failed: {}", .{err}); return -3; };
    ctx.upload(k_ptr, key[0..count]) catch |err| { setError("Upload K failed: {}", .{err}); return -3; };
    ctx.upload(v_ptr, value[0..count]) catch |err| { setError("Upload V failed: {}", .{err}); return -3; };

    // Compute forward with LSE
    engine.forwardWithLse(
        q_ptr.handle.vulkan,
        k_ptr.handle.vulkan,
        v_ptr.handle.vulkan,
        o_ptr.handle.vulkan,
        lse_ptr.handle.vulkan,
        causal != 0,
    ) catch |err| {
        setError("Forward with LSE failed: {}", .{err});
        return -4;
    };

    engine.synchronize() catch |err| {
        setError("Synchronize failed: {}", .{err});
        return -4;
    };

    // Download output and LSE
    ctx.download(o_ptr, output[0..count]) catch |err| { setError("Download output failed: {}", .{err}); return -5; };
    ctx.download(lse_ptr, lse_out[0..lse_count]) catch |err| { setError("Download LSE failed: {}", .{err}); return -5; };

    return 0;
}

// ============================================================================
// Zig API (for tests)
// ============================================================================

/// FlashAttention GPU implementation via Unified Backend
pub const Attention = struct {
    context: AttentionContext,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        // We use the embedded shaders from file scope
        const context = try AttentionContext.init(allocator, attention_f32_spv, attention_amd_spv);
        return Self{
            .context = context,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.context.deinit();
        self.* = undefined;
    }

    /// Compute attention: output = softmax(Q @ K^T / sqrt(d)) @ V
    /// All tensors are [batch_size, num_heads, seq_len, head_dim] in row-major order
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
    ) !void {
        // Default to non-causal for basic tests
        return self.forwardCausal(Q, K, V, output, batch_size, num_heads, seq_len, head_dim, false);
    }

    pub fn forwardCausal(
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
        const shape = [4]u32{ batch_size, num_heads, seq_len, head_dim };
        const count = batch_size * num_heads * seq_len * head_dim;

        // 1. Create tensors
        // Create actual Tensor structs on heap as expected by destroyTensor
        
        var q_t = try self.context.createTensor(shape);
        errdefer self.context.destroyTensor(&q_t);
        const q_ptr = try self.allocator.create(Tensor);
        q_ptr.* = q_t;
        defer { self.context.destroyTensor(q_ptr); self.allocator.destroy(q_ptr); }

        var k_t = try self.context.createTensor(shape);
        errdefer self.context.destroyTensor(&k_t);
        const k_ptr = try self.allocator.create(Tensor);
        k_ptr.* = k_t;
        defer { self.context.destroyTensor(k_ptr); self.allocator.destroy(k_ptr); }

        var v_t = try self.context.createTensor(shape);
        errdefer self.context.destroyTensor(&v_t);
        const v_ptr = try self.allocator.create(Tensor);
        v_ptr.* = v_t;
        defer { self.context.destroyTensor(v_ptr); self.allocator.destroy(v_ptr); }

        var o_t = try self.context.createTensor(shape);
        errdefer self.context.destroyTensor(&o_t);
        const o_ptr = try self.allocator.create(Tensor);
        o_ptr.* = o_t;
        defer { self.context.destroyTensor(o_ptr); self.allocator.destroy(o_ptr); }

        // 2. Upload
        try self.context.upload(q_ptr, Q[0..count]);
        try self.context.upload(k_ptr, K[0..count]);
        try self.context.upload(v_ptr, V[0..count]);

        // 3. Compute
        try self.context.attention(q_ptr, k_ptr, v_ptr, o_ptr, causal);

        // 4. Download
        try self.context.download(o_ptr, output[0..count]);
    }
};
