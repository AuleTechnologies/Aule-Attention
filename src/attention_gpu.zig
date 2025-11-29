const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const BufferManager = @import("buffer_manager.zig").BufferManager;
const AttentionPipeline = @import("attention_pipeline.zig").AttentionPipeline;
const GpuTensor = @import("gpu_tensor.zig").GpuTensor;

const log = std.log.scoped(.attention_gpu);

/// High-performance attention engine that operates on persistent GPU tensors
/// Eliminates CPU<->GPU copy overhead for repeated operations
pub const AttentionEngine = struct {
    ctx: VulkanContext,
    buffer_manager: BufferManager,
    pipeline: AttentionPipeline,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, shader_code: []const u8) !Self {
        var ctx = try VulkanContext.init(allocator);
        errdefer ctx.deinit();

        const buffer_manager = BufferManager.init(&ctx);

        var pipeline = try AttentionPipeline.init(&ctx, shader_code);
        errdefer pipeline.deinit();

        return Self{
            .ctx = ctx,
            .buffer_manager = buffer_manager,
            .pipeline = pipeline,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.pipeline.deinit();
        self.ctx.deinit();
        self.* = undefined;
    }

    /// Create a GPU tensor for use with this engine
    pub fn createTensor(self: *Self, shape: []const u32) !GpuTensor {
        return GpuTensor.init(&self.buffer_manager, shape, .f32);
    }

    /// Compute attention directly on GPU tensors - NO CPU<->GPU COPY
    /// Q, K, V must already be on GPU, output will be written to GPU tensor
    pub fn forward(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
    ) !void {
        // Validate shapes match
        if (Q.ndim != 4 or K.ndim != 4 or V.ndim != 4 or output.ndim != 4) {
            return error.InvalidShape;
        }

        const batch_size = Q.shape[0];
        const num_heads = Q.shape[1];
        const seq_len = Q.shape[2];
        const head_dim = Q.shape[3];

        // Verify all shapes match
        if (K.shape[0] != batch_size or K.shape[1] != num_heads or
            K.shape[2] != seq_len or K.shape[3] != head_dim)
        {
            return error.ShapeMismatch;
        }
        if (V.shape[0] != batch_size or V.shape[1] != num_heads or
            V.shape[2] != seq_len or V.shape[3] != head_dim)
        {
            return error.ShapeMismatch;
        }
        if (output.shape[0] != batch_size or output.shape[1] != num_heads or
            output.shape[2] != seq_len or output.shape[3] != head_dim)
        {
            return error.ShapeMismatch;
        }

        if (head_dim > 64) {
            return error.HeadDimTooLarge;
        }

        // Update descriptors to point to the GPU tensors
        self.pipeline.updateDescriptors(
            Q.getBuffer(),
            K.getBuffer(),
            V.getBuffer(),
            output.getBuffer(),
            Q.byteSize(),
        );

        // Dispatch - data stays on GPU!
        try self.pipeline.dispatch(batch_size, num_heads, seq_len, head_dim);
    }

    /// Convenience: forward with automatic sync
    pub fn forwardSync(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
    ) !void {
        try self.forward(Q, K, V, output);
        try self.ctx.waitIdle();
    }

    /// Wait for GPU operations to complete
    pub fn synchronize(self: *Self) !void {
        try self.ctx.waitIdle();
    }

    /// Get the buffer manager for creating additional tensors
    pub fn getBufferManager(self: *Self) *const BufferManager {
        return &self.buffer_manager;
    }
};
