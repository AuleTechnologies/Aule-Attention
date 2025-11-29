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
};

/// Unified tensor handle
pub const Tensor = struct {
    backend: Backend,
    handle: union {
        vulkan: *anyopaque, // VulkanTensor pointer
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
    vulkan_ctx: ?*anyopaque = null,
    hip_ctx: ?*hip_backend.HipAttention = null,

    const Self = @This();

    /// Initialize with automatic backend selection
    pub fn init(allocator: std.mem.Allocator) BackendError!Self {
        // Check for forced backend via environment
        const forced_backend = std.process.getEnvVarOwned(allocator, "AULE_BACKEND") catch null;
        defer if (forced_backend) |b| allocator.free(b);

        if (forced_backend) |backend_name| {
            if (std.mem.eql(u8, backend_name, "hip")) {
                return initHip(allocator);
            } else if (std.mem.eql(u8, backend_name, "vulkan")) {
                return initVulkan(allocator);
            } else if (std.mem.eql(u8, backend_name, "cpu")) {
                return initCpu(allocator);
            }
        }

        // Auto-detect: try Vulkan first (most compatible), then HIP, then CPU
        if (initVulkan(allocator)) |ctx| {
            return ctx;
        } else |_| {}

        if (initHip(allocator)) |ctx| {
            return ctx;
        } else |_| {}

        return initCpu(allocator);
    }

    fn initVulkan(allocator: std.mem.Allocator) BackendError!Self {
        // TODO: Integrate with existing VulkanContext
        _ = allocator;
        return BackendError.BackendInitFailed;
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
                // TODO: cleanup vulkan
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

    /// Create a tensor on the GPU
    pub fn createTensor(self: *Self, shape: [4]u32) BackendError!Tensor {
        var count: usize = 1;
        for (shape) |dim| count *= dim;

        switch (self.backend) {
            .vulkan => {
                // TODO: create vulkan tensor
                return BackendError.BackendInitFailed;
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
                // TODO
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
        if (data.len != tensor.element_count) return BackendError.InvalidTensor;

        switch (tensor.backend) {
            .vulkan => {
                // TODO
                return BackendError.BackendInitFailed;
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
        if (output.len != tensor.element_count) return BackendError.InvalidTensor;

        switch (tensor.backend) {
            .vulkan => {
                // TODO
                return BackendError.BackendInitFailed;
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
    ) BackendError!void {
        switch (self.backend) {
            .vulkan => {
                // TODO: use existing VulkanAttention
                return BackendError.BackendInitFailed;
            },
            .hip => {
                const ctx = self.hip_ctx orelse return BackendError.BackendInitFailed;
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
                var scores: [1024]f32 = undefined; // Max seq_len

                for (0..seq_len) |j| {
                    var dot: f32 = 0;
                    for (0..head_dim) |d| {
                        dot += q_data[base + i * head_dim + d] * k_data[base + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                    max_score = @max(max_score, scores[j]);
                }

                // Softmax
                var sum: f32 = 0;
                for (0..seq_len) |j| {
                    scores[j] = @exp(scores[j] - max_score);
                    sum += scores[j];
                }
                const inv_sum = 1.0 / sum;

                // Output
                for (0..head_dim) |d| {
                    var acc: f32 = 0;
                    for (0..seq_len) |j| {
                        acc += scores[j] * inv_sum * v_data[base + j * head_dim + d];
                    }
                    o_data[base + i * head_dim + d] = acc;
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

    // TODO: Check Vulkan

    return backends.toOwnedSlice();
}
