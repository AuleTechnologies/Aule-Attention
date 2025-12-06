const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const BufferManager = @import("buffer_manager.zig").BufferManager;
const AttentionPipeline = @import("attention_pipeline.zig").AttentionPipeline;
const BackwardPipelines = @import("attention_backward_pipeline.zig");
const ForwardWithLsePipeline = BackwardPipelines.ForwardWithLsePipeline;
const BackwardPipeline = BackwardPipelines.BackwardPipeline;
const GpuTensor = @import("gpu_tensor.zig").GpuTensor;

const log = std.log.scoped(.attention_gpu);

/// High-performance attention engine that operates on persistent GPU tensors
/// Eliminates CPU<->GPU copy overhead for repeated operations
pub const AttentionEngine = struct {
    ctx: VulkanContext,
    buffer_manager: BufferManager,
    pipeline: AttentionPipeline,
    forward_lse_pipeline: ?ForwardWithLsePipeline,
    backward_pipeline: ?BackwardPipeline,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, generic_shader: []const u8, amd_shader: []const u8) !Self {
        return initWithBackward(allocator, generic_shader, amd_shader, null, null);
    }

    pub fn initWithBackward(
        allocator: std.mem.Allocator,
        generic_shader: []const u8,
        amd_shader: []const u8,
        forward_lse_shader: ?[]const u8,
        backward_shader: ?[]const u8,
    ) !Self {
        var ctx = try VulkanContext.init(allocator);
        errdefer ctx.deinit();

        const buffer_manager = BufferManager.init(&ctx);

        const shader_code = if (ctx.gpu_caps.isAmd()) amd_shader else generic_shader;
        log.info("Selected shader: {s}", .{if (ctx.gpu_caps.isAmd()) "AMD Optimized" else "Generic"});

        var pipeline = try AttentionPipeline.init(&ctx, shader_code);
        errdefer pipeline.deinit();

        // Initialize backward pipelines if shaders provided
        var forward_lse_pipeline: ?ForwardWithLsePipeline = null;
        var backward_pipeline: ?BackwardPipeline = null;

        if (forward_lse_shader) |fwd_shader| {
            forward_lse_pipeline = try ForwardWithLsePipeline.init(&ctx, fwd_shader);
            log.info("Forward-with-LSE pipeline initialized", .{});
        }

        if (backward_shader) |bwd_shader| {
            backward_pipeline = try BackwardPipeline.init(&ctx, bwd_shader);
            log.info("Backward pipeline initialized", .{});
        }

        return Self{
            .ctx = ctx,
            .buffer_manager = buffer_manager,
            .pipeline = pipeline,
            .forward_lse_pipeline = forward_lse_pipeline,
            .backward_pipeline = backward_pipeline,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.backward_pipeline) |*bp| bp.deinit();
        if (self.forward_lse_pipeline) |*fp| fp.deinit();
        self.pipeline.deinit();
        self.ctx.deinit();
        self.* = undefined;
    }

    /// Check if backward pass is supported
    pub fn supportsBackward(self: *const Self) bool {
        return self.backward_pipeline != null and self.forward_lse_pipeline != null;
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
        causal: bool,
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
        try self.pipeline.dispatch(batch_size, num_heads, seq_len, head_dim, causal);
    }

    /// Convenience: forward with automatic sync
    pub fn forwardSync(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
        causal: bool,
    ) !void {
        try self.forward(Q, K, V, output, causal);
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

    /// Forward pass with LSE output (for training)
    /// Returns output and log-sum-exp values needed for backward pass
    pub fn forwardWithLse(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
        lse: *GpuTensor,
        causal: bool,
    ) !void {
        const fwd_pipeline = self.forward_lse_pipeline orelse return error.BackwardNotSupported;

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

        // LSE shape: [batch, heads, seq, 1] or just batch*heads*seq elements
        const lse_elements = batch_size * num_heads * seq_len;
        if (lse.element_count != lse_elements) {
            return error.ShapeMismatch;
        }

        if (head_dim > 64) {
            return error.HeadDimTooLarge;
        }

        fwd_pipeline.updateDescriptors(
            Q.getBuffer(),
            K.getBuffer(),
            V.getBuffer(),
            output.getBuffer(),
            lse.getBuffer(),
            Q.byteSize(),
            lse.byteSize(),
        );

        try fwd_pipeline.dispatch(batch_size, num_heads, seq_len, head_dim, causal);
    }

    /// Backward pass: compute gradients dQ, dK, dV
    /// Requires saved tensors from forward: Q, K, V, O, LSE
    /// Input: dO (gradient of output)
    /// Output: dQ, dK, dV (gradients of Q, K, V)
    pub fn backward(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        O: *const GpuTensor,
        dO: *const GpuTensor,
        lse: *const GpuTensor,
        dQ: *GpuTensor,
        dK: *GpuTensor,
        dV: *GpuTensor,
        causal: bool,
    ) !void {
        const bwd_pipeline = self.backward_pipeline orelse return error.BackwardNotSupported;

        if (Q.ndim != 4) return error.InvalidShape;

        const batch_size = Q.shape[0];
        const num_heads = Q.shape[1];
        const seq_len = Q.shape[2];
        const head_dim = Q.shape[3];

        if (head_dim > 64) {
            return error.HeadDimTooLarge;
        }

        bwd_pipeline.updateDescriptors(
            Q.getBuffer(),
            K.getBuffer(),
            V.getBuffer(),
            O.getBuffer(),
            dO.getBuffer(),
            lse.getBuffer(),
            dQ.getBuffer(),
            dK.getBuffer(),
            dV.getBuffer(),
            Q.byteSize(),
            lse.byteSize(),
        );

        try bwd_pipeline.dispatch(batch_size, num_heads, seq_len, head_dim, causal);
    }

    /// Backward pass with sync
    pub fn backwardSync(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        O: *const GpuTensor,
        dO: *const GpuTensor,
        lse: *const GpuTensor,
        dQ: *GpuTensor,
        dK: *GpuTensor,
        dV: *GpuTensor,
        causal: bool,
    ) !void {
        try self.backward(Q, K, V, O, dO, lse, dQ, dK, dV, causal);
        try self.ctx.waitIdle();
    }
};
