const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const buffer_manager_pkg = @import("buffer_manager.zig");
const BufferManager = buffer_manager_pkg.BufferManager;
const Buffer = buffer_manager_pkg.Buffer;
const AttentionPipeline = @import("attention_pipeline.zig").AttentionPipeline;
const BackwardPipelines = @import("attention_backward_pipeline.zig");
const ForwardWithLsePipeline = BackwardPipelines.ForwardWithLsePipeline;
const BackwardPipeline = BackwardPipelines.BackwardPipeline;
const SortPipeline = @import("sort_pipeline.zig").SortPipeline;
const GravityPipeline = @import("gravity_pipeline.zig").GravityPipeline;
const GpuTensor = @import("gpu_tensor.zig").GpuTensor;

const log = std.log.scoped(.attention_gpu);

/// High-performance attention engine that operates on persistent GPU tensors
/// Eliminates CPU<->GPU copy overhead for repeated operations
pub const AttentionEngine = struct {
    ctx: *VulkanContext,
    buffer_manager: BufferManager,
    pipeline: AttentionPipeline,
    forward_lse_pipeline: ?ForwardWithLsePipeline,
    backward_pipeline: ?BackwardPipeline,
    sort_pipeline: ?SortPipeline,
    gravity_pipeline: ?GravityPipeline,
    allocator: std.mem.Allocator,

    // Persistent Sort Buffers (to avoid async use-after-free)
    radix_hist_buffer: ?Buffer = null,
    radix_inds_temp: ?Buffer = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, generic_shader: []const u8, amd_shader: []const u8) !Self {
        // Warning: This legacy init will fail if new shaders are required by pipeline
        // We should pass null for optional shaders
        return initWithBackward(allocator, generic_shader, amd_shader, null, null, null, null, null, null, null, null);
    }

    pub fn initWithBackward(
        allocator: std.mem.Allocator,
        generic_shader: []const u8,
        amd_shader: []const u8,
        forward_lse_shader: ?[]const u8,
        backward_shader: ?[]const u8,
        spatial_sort_shader: ?[]const u8,
        gravity_shader: ?[]const u8,
        radix_count_shader: ?[]const u8,
        radix_scan_shader: ?[]const u8,
        radix_scatter_shader: ?[]const u8,
        iota_shader: ?[]const u8,
    ) !Self {
        var ctx = try allocator.create(VulkanContext);
        errdefer allocator.destroy(ctx);
        ctx.* = try VulkanContext.init(allocator);
        errdefer ctx.deinit();

        const buffer_manager = BufferManager.init(ctx);

        const shader_code = if (ctx.gpu_caps.isAmd()) amd_shader else generic_shader;
        log.info("Selected shader: {s}", .{if (ctx.gpu_caps.isAmd()) "AMD Optimized" else "Generic"});

        var pipeline = try AttentionPipeline.init(ctx, shader_code);
        errdefer pipeline.deinit();

        var forward_lse_pipeline: ?ForwardWithLsePipeline = null;
        if (forward_lse_shader) |s| forward_lse_pipeline = try ForwardWithLsePipeline.init(ctx, s);

        var backward_pipeline: ?BackwardPipeline = null;
        if (backward_shader) |s| backward_pipeline = try BackwardPipeline.init(ctx, s);

        var sort_pipeline: ?SortPipeline = null;
        // Only init sort pipeline if ALL radix shaders are present
        if (spatial_sort_shader != null and radix_count_shader != null and radix_scan_shader != null and radix_scatter_shader != null and iota_shader != null) {
            sort_pipeline = try SortPipeline.init(ctx, 
                spatial_sort_shader.?,
                radix_count_shader.?,
                radix_scan_shader.?,
                radix_scatter_shader.?,
                iota_shader.?
            );
            log.info("Sort pipeline initialized (Radix enabled)", .{});
        } else if (spatial_sort_shader) |s| {
             _ = s; // Unused
             // Fallback for passing just one shader (this will crash the new init? No, we need separate logic or dummy shaders)
             // The new SortPipeline.init requires 5 args.
             // We cannot support legacy init easily unless we overload or change SortPipeline.
             // Let's assume for now we provide all or nothing for Radix support.
             // If missing radix shaders, we skip sort pipeline.
             log.warn("Missing Radix Sort shaders, SortPipeline skipped", .{});
        }

        var gravity_pipeline: ?GravityPipeline = null;
        if (gravity_shader) |s| gravity_pipeline = try GravityPipeline.init(ctx, s);
        
        return Self{
            .ctx = ctx,
            .buffer_manager = buffer_manager,
            .pipeline = pipeline,
            .forward_lse_pipeline = forward_lse_pipeline,
            .backward_pipeline = backward_pipeline,
            .sort_pipeline = sort_pipeline,
            .gravity_pipeline = gravity_pipeline,
            .allocator = allocator,
        };
    }

    // ... (deinit, createTensor, forward, etc. - keep unchanged)
    
    /// Radix Sort Implementation
    /// Uses 4 passes of 8-bit Radix Sort
    pub fn spatialSort(
        self: *Self,
        keys: *const GpuTensor, // [B, H, S, D]
        values: *const GpuTensor,
        indices: *GpuTensor, // [B, H, S] (Output - Final Indices)
        sort_dim: u32,
    ) !void {
        const sort_pipe = self.sort_pipeline orelse return error.SortPipelineNotInitialized;

        // Validation
        if (keys.ndim != 4) return error.InvalidShape;
        const batch = keys.shape[0];
        const heads = keys.shape[1];
        const seq_len = keys.shape[2];
        const d_model = keys.shape[3];
        const num_elements = batch * heads * seq_len; // Total items to sort?
        
        // PROBLEM: We need to sort each (Batch, Head) independently (Segmented Sort).
        // Our Radix Sort is currently Global.
        // If B=1, H=1, Global Sort is fine.
        // If B>1, we need to handle segments.
        // Current implementation will sort ALL keys across batches together.
        // Check B/H.
        if (batch != 1 or heads != 1) {
            log.warn("Radix Sort currently only supports B=1, H=1! Running global sort anyway.", .{});
            // This will mix batches, but for benchmarking single sequence it's fine.
            // For production, we need segmented sort.
        }

        const WORKGROUP = 256;
        const num_workgroups = (num_elements + WORKGROUP - 1) / WORKGROUP;

        // 1. Allocate/Reuse Persistent Buffers
        // Histograms: [NumGroups * 256] (u32)
        const hist_size = num_workgroups * 256 * 4;
        if (self.radix_hist_buffer == null or self.radix_hist_buffer.?.size < hist_size) {
             if (self.radix_hist_buffer) |*b| self.buffer_manager.destroyBuffer(b);
             self.radix_hist_buffer = try self.buffer_manager.createBuffer(hist_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
        }
        const hist_buf = self.radix_hist_buffer.?;

        // Indices Ping-Pong: We need a secondary indices buffer
        const inds_size = num_elements * 4;
        if (self.radix_inds_temp == null or self.radix_inds_temp.?.size < inds_size) {
             if (self.radix_inds_temp) |*b| self.buffer_manager.destroyBuffer(b);
             self.radix_inds_temp = try self.buffer_manager.createBuffer(inds_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
        }
        const inds_temp = self.radix_inds_temp.?;

        // 2. Initialize Indices to 0..S-1 per segment
        const S = keys.shape[2];
        const num_segments = keys.shape[0] * keys.shape[1];
        try sort_pipe.dispatchIota(indices.getBuffer(), num_elements, @intCast(S));

        // 3. Perform Radix Sort (4 passes)
        std.debug.print("DEBUG: dispatchRadix PRE-CALL. d_model={}, num_elements={}, S={}\n", .{d_model, num_elements, S});
        if (d_model == 0) return error.InvalidDModel;

        // Note: dispatchRadix handles descriptor updates internally now.
        try sort_pipe.dispatchRadix(
            keys.getBuffer(),
            values.getBuffer(),
            indices.getBuffer(),
            inds_temp.buffer,
            hist_buf.buffer,
            num_elements,
            d_model,
            sort_dim,
            @intCast(num_segments),
            @intCast(S),
        );
    }

    pub fn deinit(self: *Self) void {
        if (self.backward_pipeline) |*bp| bp.deinit();
        if (self.forward_lse_pipeline) |*fp| fp.deinit();
        if (self.sort_pipeline) |*sp| sp.deinit();
        if (self.gravity_pipeline) |*gp| gp.deinit();
        if (self.radix_hist_buffer) |*b| self.buffer_manager.destroyBuffer(b);
        if (self.radix_inds_temp) |*b| self.buffer_manager.destroyBuffer(b);
        self.pipeline.deinit();
        self.ctx.deinit();
        self.allocator.destroy(self.ctx);
        self.* = undefined;
    }

    /// Check if backward pass is supported
    pub fn supportsBackward(self: *const Self) bool {
        return self.backward_pipeline != null and self.forward_lse_pipeline != null;
    }

    /// Create a GPU tensor for use with this engine
    pub fn createTensor(self: *Self, shape: []const u32) !GpuTensor {
        std.debug.print("AttentionEngine.createTensor: shape {any}\n", .{shape});
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
        rot_cos: ?*const GpuTensor,
        rot_sin: ?*const GpuTensor,
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

        // Verify all shapes match (GQA: K/V heads can be divisor of Q heads)
        // K.shape[1] (num_kv_heads) must divide num_heads
        if (K.shape[0] != batch_size or 
            (num_heads % K.shape[1] != 0) or
            K.shape[3] != head_dim)
        {
            return error.ShapeMismatch;
        }
        if (V.shape[0] != batch_size or V.shape[1] != K.shape[1] or
            V.shape[2] != K.shape[2] or 
            V.shape[3] != head_dim)
        {
            return error.ShapeMismatch;
        }
        // Output must match Query shape
        if (output.shape[0] != batch_size or output.shape[1] != num_heads or
            output.shape[2] != seq_len or output.shape[3] != head_dim)
        {
            return error.ShapeMismatch;
        }

        if (head_dim > 64) {
            return error.HeadDimTooLarge;
        }
        
        // RoPE validation
        var rope_size: u64 = 0;
        var has_rope = false;
        var cos_buf: ?vk.Buffer = null;
        var sin_buf: ?vk.Buffer = null;
        
        if (rot_cos) |c| {
            if (rot_sin) |s| {
                has_rope = true;
                rope_size = c.byteSize();
                cos_buf = c.getBuffer();
                sin_buf = s.getBuffer();
                // Minimal validation: assume user provides correct shape for now
            } else {
                return error.MissingRotarySin;
            }
        } else if (rot_sin != null) {
            return error.MissingRotaryCos;
        }

        // Update descriptors to point to the GPU tensors
        self.pipeline.updateDescriptors(
            Q.getBuffer(),
            K.getBuffer(),
            V.getBuffer(),
            output.getBuffer(),
            cos_buf,
            sin_buf,
            Q.byteSize(),
            K.byteSize(),
            V.byteSize(),
            output.byteSize(),
            rope_size,
        );

        const num_kv_heads = K.shape[1];
        const key_seq_len = K.shape[2];

        log.info("Dispatching: B={d}, H={d}, KVH={d}, QLen={d}, KLen={d}", .{
            batch_size, num_heads, num_kv_heads, seq_len, key_seq_len
        });

        // Dispatch - data stays on GPU!
        try self.pipeline.dispatch(batch_size, num_heads, num_kv_heads, seq_len, key_seq_len, head_dim, causal, has_rope);
    }

    /// Convenience: forward with automatic sync
    pub fn forwardSync(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
        rot_cos: ?*const GpuTensor,
        rot_sin: ?*const GpuTensor,
        causal: bool,
    ) !void {
        try self.forward(Q, K, V, output, rot_cos, rot_sin, causal);
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



    /// Gravity Attention: Indirect attention using sorted indices
    pub fn forwardGravity(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
        rot_cos: ?*const GpuTensor,
        rot_sin: ?*const GpuTensor,
        indices: *GpuTensor,
        causal: bool,
        max_attend: u32,
    ) !void {
        const gravity_pipe = self.gravity_pipeline orelse return error.GravityPipelineNotInitialized;
        const sort_pipe = self.sort_pipeline orelse return error.SortPipelineNotInitialized;

        // Validations
        if (Q.ndim != 4 or K.ndim != 4 or V.ndim != 4 or output.ndim != 4) return error.InvalidShape;

        const batch_size = Q.shape[0];
        const num_heads = Q.shape[1];
        const seq_len = Q.shape[2];
        const head_dim = Q.shape[3];
        
        // GQA support
        if (K.shape[0] != batch_size or (num_heads % K.shape[1] != 0) or K.shape[3] != head_dim) return error.ShapeMismatch;
        
        // RoPE
        var rope_size: u64 = 0;
        var has_rope = false;
        var cos_buf: ?vk.Buffer = null;
        var sin_buf: ?vk.Buffer = null;
        if (rot_cos) |c| {
            if (rot_sin) |s| {
                 has_rope = true;
                 rope_size = c.byteSize();
                 cos_buf = c.getBuffer();
                 sin_buf = s.getBuffer();
            } else return error.MissingRotarySin;
        } else if (rot_sin != null) return error.MissingRotaryCos;

        const num_kv_heads = K.shape[1];
        const key_seq_len = K.shape[2];

        // --- SORTING PHASE ---
        {
             // 1. Allocate Temp Buffers for Sort
             const num_elements = batch_size * num_kv_heads * key_seq_len;
             const WORKGROUP = 256;
             const num_workgroups = (num_elements + WORKGROUP - 1) / WORKGROUP;
             
             // Histograms: [NumGroups * 256] (u32)
             const hist_size = num_workgroups * 256 * 4;
             var hist_buf = try self.buffer_manager.createBuffer(hist_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
             defer self.buffer_manager.destroyBuffer(&hist_buf);
             
             // Temp Indices for Ping-Pong
             const inds_size = num_elements * 4;
             var inds_temp = try self.buffer_manager.createBuffer(inds_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
             defer self.buffer_manager.destroyBuffer(&inds_temp);
             
             // 2. Dispatch Iota (Initialize Indices)
             const S = key_seq_len;
             const num_segments = batch_size * num_kv_heads;
             try sort_pipe.dispatchIota(indices.getBuffer(), num_elements, @intCast(S));
             
             // 3. Dispatch Radix Sort
             // Use sort_dim = 0 for now (default)
             try sort_pipe.dispatchRadix(
                K.getBuffer(),
                V.getBuffer(), // Passed but assumed unused for value movement
                indices.getBuffer(), // Final
                inds_temp.buffer,    // Temp
                hist_buf.buffer,     // Hist
                num_elements,
                head_dim, // d_model
                0,        // Sort Dim 0
                @intCast(num_segments),
                @intCast(S)
             );
        }
        // --- END SORTING PHASE ---

        gravity_pipe.updateDescriptors(
             Q.getBuffer(),
             K.getBuffer(),
             V.getBuffer(),
             output.getBuffer(),
             cos_buf,
             sin_buf,
             indices.getBuffer(),
             Q.byteSize(),
             K.byteSize(),
             V.byteSize(),
             output.byteSize(),
             rope_size,
             indices.byteSize()
        );
        
        try gravity_pipe.dispatch(
             batch_size, num_heads, num_kv_heads, seq_len, key_seq_len, head_dim, causal, has_rope, max_attend
        );
    }
};
