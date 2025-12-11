//! Unified backend interface for aule-attention
//!
//! Supports multiple GPU backends:
//! - Vulkan: Works on most consumer GPUs (AMD, NVIDIA, Intel, Apple)
//! - HIP: Works on AMD datacenter GPUs (MI300X, MI250, etc.)
//!
//! The backend is selected automatically based on available hardware,
//! or can be forced via environment variable AULE_BACKEND=vulkan|hip

const std = @import("std");
const builtin = @import("builtin");

// Import backends
const vulkan_backend = @import("../vulkan_context.zig");
const hip_backend = @import("hip.zig");
const AttentionEngine = @import("../attention_gpu.zig").AttentionEngine;
const GpuTensor = @import("../gpu_tensor.zig").GpuTensor;

pub const Backend = enum {
    vulkan,
    hip,
    cpu, // Fallback CPU implementation
};

pub const BackendError = error{
    NoBackendAvailable,
    BackendInitFailed,
    InvalidTensor,
    ComputeFailed,
    OutOfMemory,
    ShapeMismatch,
    SizeMismatch,
    DTypeMismatch,
    HeadDimTooLarge,
} || std.mem.Allocator.Error || std.process.GetEnvVarOwnedError;

/// Unified tensor handle
pub const Tensor = struct {
    backend: Backend,
    handle: union {
        vulkan: *GpuTensor,
        hip: *hip_backend.HipTensor,
        cpu: []f32,
    },
    shape: [4]u32,
    element_count: usize,
};

/// Unified attention context
pub const AttentionContext = struct {
    backend: Backend,
    allocator: std.mem.Allocator,

    // Backend-specific state
    vulkan_ctx: ?*AttentionEngine = null,
    hip_ctx: ?*hip_backend.HipAttention = null,

    const Self = @This();

    /// Initialize with automatic backend selection
    pub fn init(allocator: std.mem.Allocator, generic_shader: []const u8, amd_shader: []const u8) BackendError!Self {
        return initWithBackward(allocator, generic_shader, amd_shader, null, null, null, null);
    }

    /// Initialize with backward pass support
    pub fn initWithBackward(
        allocator: std.mem.Allocator,
        generic_shader: []const u8,
        amd_shader: []const u8,
        forward_lse_shader: ?[]const u8,
        backward_shader: ?[]const u8,
        sort_shader: ?[]const u8,
        gravity_shader: ?[]const u8,
        radix_count_shader: ?[]const u8,
        radix_scan_shader: ?[]const u8,
        radix_scatter_shader: ?[]const u8,
        iota_shader: ?[]const u8,
    ) BackendError!Self {
        // Check for forced backend via environment
        const forced_backend = std.process.getEnvVarOwned(allocator, "AULE_BACKEND") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => return err,
        };
        defer if (forced_backend) |b| allocator.free(b);

        if (forced_backend) |backend_name| {
            if (std.mem.eql(u8, backend_name, "hip")) {
                return initHip(allocator);
            } else if (std.mem.eql(u8, backend_name, "vulkan")) {
                return initVulkan(allocator, generic_shader, amd_shader, forward_lse_shader, backward_shader, sort_shader, gravity_shader, radix_count_shader, radix_scan_shader, radix_scatter_shader, iota_shader);
            } else if (std.mem.eql(u8, backend_name, "cpu")) {
                return initCpu(allocator);
            }
        }

        // Auto-detect: try Vulkan first (most compatible), then HIP, then CPU
        // Note: Logic suggests trying HIP first if on MI300X, but Vulkan is generally safer fallback
        // unless we know we are on a headless compute node.

        // Try HIP first if available (performance preference for AMD Datacenter)
        if (initHip(allocator)) |ctx| {
            return ctx;
        } else |_| {}

        if (initVulkan(allocator, generic_shader, amd_shader, forward_lse_shader, backward_shader, sort_shader, gravity_shader, radix_count_shader, radix_scan_shader, radix_scatter_shader, iota_shader)) |ctx| {
            return ctx;
        } else |_| {}

        return initCpu(allocator);
    }

    fn initVulkan(
        allocator: std.mem.Allocator,
        generic_shader: []const u8,
        amd_shader: []const u8,
        forward_lse_shader: ?[]const u8,
        backward_shader: ?[]const u8,
        sort_shader: ?[]const u8,
        gravity_shader: ?[]const u8,
        radix_count_shader: ?[]const u8,
        radix_scan_shader: ?[]const u8,
        radix_scatter_shader: ?[]const u8,
        iota_shader: ?[]const u8,
    ) BackendError!Self {
        const engine = allocator.create(AttentionEngine) catch return BackendError.OutOfMemory;
        engine.* = AttentionEngine.initWithBackward(allocator, generic_shader, amd_shader, forward_lse_shader, backward_shader, sort_shader, gravity_shader, radix_count_shader, radix_scan_shader, radix_scatter_shader, iota_shader) catch |err| {
            allocator.destroy(engine);
            return switch (err) {
                error.OutOfMemory => BackendError.OutOfMemory,
                else => BackendError.BackendInitFailed,
            };
        };

        return Self{
            .backend = .vulkan,
            .allocator = allocator,
            .vulkan_ctx = engine,
        };
    }

    fn initHip(allocator: std.mem.Allocator) BackendError!Self {
        if (!hip_backend.isAvailable()) {
            return BackendError.NoBackendAvailable;
        }

        const hip_ctx = allocator.create(hip_backend.HipAttention) catch {
            return BackendError.OutOfMemory;
        };
        hip_ctx.* = hip_backend.HipAttention.init() catch {
            allocator.destroy(hip_ctx);
            return BackendError.BackendInitFailed;
        };

        return Self{
            .backend = .hip,
            .allocator = allocator,
            .hip_ctx = hip_ctx,
        };
    }

    fn initCpu(allocator: std.mem.Allocator) BackendError!Self {
        return Self{
            .backend = .cpu,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        switch (self.backend) {
            .vulkan => {
                if (self.vulkan_ctx) |ctx| {
                    ctx.deinit();
                    self.allocator.destroy(ctx);
                }
            },
            .hip => {
                if (self.hip_ctx) |ctx| {
                    ctx.deinit();
                    self.allocator.destroy(ctx);
                }
            },
            .cpu => {},
        }
        self.* = undefined;
    }

    /// Create a new tensor with shape
    pub fn createTensor(self: *Self, shape: [4]u32) BackendError!Tensor {
        // std.debug.print("Backend.createTensor: shape {any}\n", .{shape});
        var count: usize = 1;
        for (shape) |dim| count *= dim;

        switch (self.backend) {
            .vulkan => {
                const ctx = self.vulkan_ctx orelse return BackendError.BackendInitFailed;
                const gpu_tensor = self.allocator.create(GpuTensor) catch return BackendError.OutOfMemory;
                
                // GpuTensor.init returns !GpuTensor
                gpu_tensor.* = ctx.createTensor(&shape) catch |err| {
                    self.allocator.destroy(gpu_tensor);
                    return switch (err) {
                        error.InvalidShape => BackendError.InvalidTensor,
                        // Map Vulkan memory errors to OutOfMemory
                        else => BackendError.OutOfMemory, 
                    };
                };

                return Tensor{
                    .backend = .vulkan,
                    .handle = .{ .vulkan = gpu_tensor },
                    .shape = shape,
                    .element_count = count,
                };
            },
            .hip => {
                const hip_tensor = self.allocator.create(hip_backend.HipTensor) catch {
                    return BackendError.OutOfMemory;
                };
                hip_tensor.* = hip_backend.HipTensor.init(shape) catch {
                    self.allocator.destroy(hip_tensor);
                    return BackendError.OutOfMemory;
                };
                return Tensor{
                    .backend = .hip,
                    .handle = .{ .hip = hip_tensor },
                    .shape = shape,
                    .element_count = count,
                };
            },
            .cpu => {
                const data = self.allocator.alloc(f32, count) catch {
                    return BackendError.OutOfMemory;
                };
                return Tensor{
                    .backend = .cpu,
                    .handle = .{ .cpu = data },
                    .shape = shape,
                    .element_count = count,
                };
            },
        }
    }

    /// Destroy a tensor
    pub fn destroyTensor(self: *Self, tensor: *Tensor) void {
        switch (tensor.backend) {
            .vulkan => {
                tensor.handle.vulkan.deinit();
                self.allocator.destroy(tensor.handle.vulkan);
            },
            .hip => {
                tensor.handle.hip.deinit();
                self.allocator.destroy(tensor.handle.hip);
            },
            .cpu => {
                self.allocator.free(tensor.handle.cpu);
            },
        }
        tensor.* = undefined;
    }

    /// Upload data to tensor
    pub fn upload(self: *Self, tensor: *Tensor, data: []const f32) BackendError!void {
        _ = self;
        if (data.len != tensor.element_count) return BackendError.SizeMismatch;

        switch (tensor.backend) {
            .vulkan => {
                tensor.handle.vulkan.upload(data) catch |err| switch (err) {
                    error.SizeMismatch => return BackendError.SizeMismatch,
                    error.DTypeMismatch => return BackendError.DTypeMismatch,
                };
            },
            .hip => {
                tensor.handle.hip.upload(data) catch return BackendError.ComputeFailed;
            },
            .cpu => {
                @memcpy(tensor.handle.cpu, data);
            },
        }
    }

    /// Download data from tensor
    pub fn download(self: *Self, tensor: *const Tensor, output: []f32) BackendError!void {
        _ = self;
        if (output.len != tensor.element_count) return BackendError.SizeMismatch;

        switch (tensor.backend) {
            .vulkan => {
                tensor.handle.vulkan.download(output) catch |err| switch (err) {
                    error.SizeMismatch => return BackendError.SizeMismatch,
                    error.DTypeMismatch => return BackendError.DTypeMismatch,
                };
            },
            .hip => {
                tensor.handle.hip.download(output) catch return BackendError.ComputeFailed;
            },
            .cpu => {
                @memcpy(output, tensor.handle.cpu);
            },
        }
    }

    /// Compute attention: output = softmax(Q @ K^T / sqrt(d)) @ V
    pub fn attention(
        self: *Self,
        Q: *Tensor,
        K: *Tensor,
        V: *Tensor,
        output: *Tensor,
        rot_cos: ?*Tensor,
        rot_sin: ?*Tensor,
        causal: bool,
    ) BackendError!void {
        switch (self.backend) {
            .vulkan => {
                const ctx = self.vulkan_ctx orelse return BackendError.BackendInitFailed;
                
                // Extract Vulkan handles for RoPE if present
                var rot_cos_vk: ?*const GpuTensor = null;
                if (rot_cos) |t| {
                    if (t.backend == .vulkan) rot_cos_vk = t.handle.vulkan;
                }
                
                var rot_sin_vk: ?*const GpuTensor = null;
                if (rot_sin) |t| {
                    if (t.backend == .vulkan) rot_sin_vk = t.handle.vulkan;
                }

                ctx.forwardSync(
                    Q.handle.vulkan,
                    K.handle.vulkan,
                    V.handle.vulkan,
                    output.handle.vulkan,
                    rot_cos_vk,
                    rot_sin_vk,
                    causal
                ) catch |err| switch (err) {
                    error.InvalidShape, error.ShapeMismatch => return BackendError.ShapeMismatch,
                    else => return BackendError.ComputeFailed,
                };
            },
            .hip => {
                const ctx = self.hip_ctx orelse return BackendError.BackendInitFailed;
                // HIP backend should support causal if implemented, check signature
                ctx.forward(Q.handle.hip, K.handle.hip, V.handle.hip, output.handle.hip) catch {
                    return BackendError.ComputeFailed;
                };
            },
            .cpu => {
                // Use CPU reference implementation
                try cpuAttention(Q, K, V, output);
            },
        }
    }

    /// Spatial Sort: Reorder Keys and Values based on projection
    pub fn spatialSort(
        self: *Self,
        keys: *Tensor,
        values: *Tensor,
        indices: *Tensor,
        sort_dim: u32,
    ) BackendError!void {
        switch (self.backend) {
            .vulkan => {
                const ctx = self.vulkan_ctx orelse return BackendError.BackendInitFailed;
                ctx.spatialSort(
                    keys.handle.vulkan,
                    values.handle.vulkan,
                    indices.handle.vulkan,
                    sort_dim,
                ) catch |err| switch (err) {
                    error.InvalidShape => return BackendError.ShapeMismatch,
                    else => return BackendError.ComputeFailed,
                };
            },
            .hip => return BackendError.ComputeFailed, // Not implemented
            .cpu => return BackendError.ComputeFailed, // Not implemented
        }
    }

    /// Gravity Attention: Indirect attention using sorted indices
    pub fn forwardGravity(
        self: *Self,
        Q: *Tensor,
        K: *Tensor,
        V: *Tensor,
        output: *Tensor,
        rot_cos: ?*Tensor,
        rot_sin: ?*Tensor,
        indices: *Tensor,
        causal: bool,
        max_attend: u32,
    ) BackendError!void {
        switch (self.backend) {
            .vulkan => {
                const ctx = self.vulkan_ctx orelse return BackendError.BackendInitFailed;
                
                // Extract Vulkan handles for RoPE if present
                var rot_cos_vk: ?*const GpuTensor = null;
                if (rot_cos) |t| {
                    if (t.backend == .vulkan) rot_cos_vk = t.handle.vulkan;
                }
                
                var rot_sin_vk: ?*const GpuTensor = null;
                if (rot_sin) |t| {
                    if (t.backend == .vulkan) rot_sin_vk = t.handle.vulkan;
                }

                ctx.forwardGravity(
                    Q.handle.vulkan,
                    K.handle.vulkan,
                    V.handle.vulkan,
                    output.handle.vulkan,
                    rot_cos_vk,
                    rot_sin_vk,
                    indices.handle.vulkan,
                    causal,
                    max_attend
                ) catch |err| switch (err) {
                    error.InvalidShape, error.ShapeMismatch => return BackendError.ShapeMismatch,
                    else => return BackendError.ComputeFailed,
                };
            },
            .hip => return BackendError.ComputeFailed, // Not implemented
            .cpu => return BackendError.ComputeFailed, // Not implemented
        }
    }

    /// Get backend name for display
    pub fn getBackendName(self: *const Self) []const u8 {
        return switch (self.backend) {
            .vulkan => "Vulkan",
            .hip => "HIP/ROCm",
            .cpu => "CPU (fallback)",
        };
    }
};

/// CPU fallback implementation
fn cpuAttention(Q: *Tensor, K: *Tensor, V: *Tensor, output: *Tensor) BackendError!void {
    const batch_size = Q.shape[0];
    const num_heads = Q.shape[1];
    const seq_len = Q.shape[2];
    const head_dim = Q.shape[3];

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const q_data = Q.handle.cpu;
    const k_data = K.handle.cpu;
    const v_data = V.handle.cpu;
    const o_data = output.handle.cpu;

    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            const base = (b * num_heads + h) * seq_len * head_dim;

            for (0..seq_len) |i| {
                // Compute attention scores
                var max_score: f32 = -std.math.inf(f32);
                
                // Using a dynamic allocation for scores to avoid stack overflow with large seq_len
                // But for fallback/simplicity we'll just handle up to a limit or risk stack issues
                // Ideally this should use an allocator.
                // Given the constraints, let's just loop twice to compute max then exp sum.
                
                // First pass: find max_score
                for (0..seq_len) |j| {
                    var dot: f32 = 0;
                    for (0..head_dim) |d| {
                        dot += q_data[base + i * head_dim + d] * k_data[base + j * head_dim + d];
                    }
                    max_score = @max(max_score, dot * scale);
                }

                // Second pass: compute sum of exponentials and weighted sum of V
                var sum_exp: f32 = 0;
                // Initialize output row to 0
                for (0..head_dim) |d| {
                    o_data[base + i * head_dim + d] = 0;
                }

                for (0..seq_len) |j| {
                    var dot: f32 = 0;
                    for (0..head_dim) |d| {
                        dot += q_data[base + i * head_dim + d] * k_data[base + j * head_dim + d];
                    }
                    const score = @exp((dot * scale) - max_score);
                    sum_exp += score;

                    for (0..head_dim) |d| {
                        o_data[base + i * head_dim + d] += score * v_data[base + j * head_dim + d];
                    }
                }

                // Final pass: normalize
                const inv_sum = 1.0 / sum_exp;
                for (0..head_dim) |d| {
                    o_data[base + i * head_dim + d] *= inv_sum;
                }
            }
        }
    }
}

/// Detect available backends
pub fn detectBackends(allocator: std.mem.Allocator) ![]Backend {
    var backends = std.ArrayList(Backend).init(allocator);

    // Always have CPU fallback
    try backends.append(.cpu);

    // Check HIP
    if (hip_backend.isAvailable()) {
        try backends.append(.hip);
    }
    
    // Check Vulkan - assumes available if library loaded
    // Full availability check happens at runtime when creating instance
    try backends.append(.vulkan);

    return backends.toOwnedSlice();
}
