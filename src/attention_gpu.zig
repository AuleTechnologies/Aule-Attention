const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const buffer_manager_pkg = @import("buffer_manager.zig");
const BufferManager = buffer_manager_pkg.BufferManager;
const Buffer = buffer_manager_pkg.Buffer;
const AttentionPipeline = @import("attention_pipeline.zig").AttentionPipeline;
const PagedAttentionPipeline = @import("paged_attention_pipeline.zig").PagedAttentionPipeline;
const CopyKVPipeline = @import("copy_kv_pipeline.zig").CopyKVPipeline;
const BackwardPipelines = @import("attention_backward_pipeline.zig");
const ForwardWithLsePipeline = BackwardPipelines.ForwardWithLsePipeline;
const BackwardPipeline = BackwardPipelines.BackwardPipeline;
const SortPipeline = @import("sort_pipeline.zig").SortPipeline;
const GravityPipeline = @import("gravity_pipeline.zig").GravityPipeline;
const GpuTensor = @import("gpu_tensor.zig").GpuTensor;
const BlockPool = @import("block_pool.zig").BlockPool;
const BlockTable = @import("block_table.zig").BlockTable;

const log = std.log.scoped(.attention_gpu);

/// Shader variant selection for different performance profiles
pub const ShaderVariant = enum(u8) {
    baseline = 0, // Original 16x16 block, scalar loads
    fast = 1, // Optimized 32x32 block, vec4 loads, block skipping
    fp16 = 2, // FP16 with FP32 accumulation (requires hardware support)
    fp16_amd = 3, // FP16 optimized for AMD 64-wide wavefronts
};

/// High-performance attention engine that operates on persistent GPU tensors
/// Eliminates CPU<->GPU copy overhead for repeated operations
pub const AttentionEngine = struct {
    ctx: *VulkanContext,
    buffer_manager: BufferManager,
    pipeline: AttentionPipeline, // Baseline shader
    fast_pipeline: ?AttentionPipeline, // Optimized FP32 shader
    fp16_pipeline: ?AttentionPipeline, // FP16 shader
    fp16_amd_pipeline: ?AttentionPipeline, // FP16 AMD-optimized
    paged_pipeline: ?PagedAttentionPipeline, // PagedAttention with block pool
    copy_kv_pipeline: ?CopyKVPipeline, // K/V scatter to paged format
    active_variant: ShaderVariant,
    forward_lse_pipeline: ?ForwardWithLsePipeline,
    backward_pipeline: ?BackwardPipeline,
    sort_pipeline: ?SortPipeline,
    gravity_pipeline: ?GravityPipeline,
    allocator: std.mem.Allocator,

    // Persistent Sort Buffers (to avoid async use-after-free)
    radix_hist_buffer: ?Buffer = null,
    radix_inds_temp: ?Buffer = null,

    // PagedAttention block management (optional, initialized on first paged forward)
    block_pool: ?BlockPool = null,
    block_table: ?BlockTable = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, generic_shader: []const u8, amd_shader: []const u8) !Self {
        // Warning: This legacy init will fail if new shaders are required by pipeline
        // We should pass null for optional shaders
        return initWithBackward(allocator, generic_shader, amd_shader, null, null, null, null, null, null, null, null, null, null, null, null, null, null);
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
        magnitude_shader: ?[]const u8,
        fast_shader: ?[]const u8,
        fp16_shader: ?[]const u8,
        fp16_amd_shader: ?[]const u8,
        paged_shader: ?[]const u8,
        copy_kv_shader: ?[]const u8,
    ) !Self {
        var ctx = try allocator.create(VulkanContext);
        errdefer allocator.destroy(ctx);
        ctx.* = try VulkanContext.init(allocator);
        errdefer ctx.deinit();

        const buffer_manager = BufferManager.init(ctx);

        const shader_code = if (ctx.gpu_caps.isAmd()) amd_shader else generic_shader;
        log.info("Selected baseline shader: {s}", .{if (ctx.gpu_caps.isAmd()) "AMD Optimized" else "Generic"});

        var pipeline = try AttentionPipeline.init(ctx, shader_code);
        errdefer pipeline.deinit();

        // Initialize optimized shader pipelines
        var fast_pipeline: ?AttentionPipeline = null;
        if (fast_shader) |s| {
            fast_pipeline = try AttentionPipeline.init(ctx, s);
            log.info("Fast FP32 shader loaded (32x32 blocks, vec4 loads, block skipping)", .{});
        }

        var fp16_pipeline: ?AttentionPipeline = null;
        if (fp16_shader) |s| {
            if (ctx.gpu_caps.fp16_supported) {
                fp16_pipeline = try AttentionPipeline.init(ctx, s);
                log.info("FP16 shader loaded", .{});
            } else {
                log.warn("FP16 shader requested but GPU does not support FP16", .{});
            }
        }

        var fp16_amd_pipeline: ?AttentionPipeline = null;
        if (fp16_amd_shader) |s| {
            if (ctx.gpu_caps.fp16_supported and ctx.gpu_caps.isAmd()) {
                fp16_amd_pipeline = try AttentionPipeline.init(ctx, s);
                log.info("FP16 AMD-optimized shader loaded (64-wide wavefront)", .{});
            }
        }

        var paged_pipeline: ?PagedAttentionPipeline = null;
        if (paged_shader) |s| {
            log.info("Initializing PagedAttention pipeline...", .{});
            paged_pipeline = try PagedAttentionPipeline.init(ctx, s);
            log.info("PagedAttention shader loaded (block-based KV cache)", .{});
        }

        var copy_kv_pipeline: ?CopyKVPipeline = null;
        if (copy_kv_shader) |s| {
            log.info("Initializing CopyKV pipeline...", .{});
            copy_kv_pipeline = try CopyKVPipeline.init(ctx, s);
            log.info("CopyKV shader loaded (K/V scatter to paged format)", .{});
        }

        // Select default active variant based on GPU capabilities
        var active_variant: ShaderVariant = .baseline;
        if (fast_pipeline != null) {
            active_variant = .fast;
            log.info("Default shader variant: fast (optimized)", .{});
        }

        var forward_lse_pipeline: ?ForwardWithLsePipeline = null;
        if (forward_lse_shader) |s| forward_lse_pipeline = try ForwardWithLsePipeline.init(ctx, s);

        var backward_pipeline: ?BackwardPipeline = null;
        if (backward_shader) |s| backward_pipeline = try BackwardPipeline.init(ctx, s);

        var sort_pipeline: ?SortPipeline = null;
        // Only init sort pipeline if ALL radix shaders are present
        if (spatial_sort_shader != null and radix_count_shader != null and radix_scan_shader != null and radix_scatter_shader != null and iota_shader != null) {
            sort_pipeline = try SortPipeline.initWithMagnitude(ctx,
                spatial_sort_shader.?,
                radix_count_shader.?,
                radix_scan_shader.?,
                radix_scatter_shader.?,
                iota_shader.?,
                magnitude_shader // Optional magnitude shader for improved sorting
            );
            log.info("Sort pipeline initialized (Radix enabled, magnitude={s})", .{if (magnitude_shader != null) "yes" else "no"});
        } else if (spatial_sort_shader) |s| {
             _ = s; // Unused
             log.warn("Missing Radix Sort shaders, SortPipeline skipped", .{});
        }

        var gravity_pipeline: ?GravityPipeline = null;
        if (gravity_shader) |s| gravity_pipeline = try GravityPipeline.init(ctx, s);

        return Self{
            .ctx = ctx,
            .buffer_manager = buffer_manager,
            .pipeline = pipeline,
            .fast_pipeline = fast_pipeline,
            .fp16_pipeline = fp16_pipeline,
            .fp16_amd_pipeline = fp16_amd_pipeline,
            .paged_pipeline = paged_pipeline,
            .copy_kv_pipeline = copy_kv_pipeline,
            .active_variant = active_variant,
            .forward_lse_pipeline = forward_lse_pipeline,
            .backward_pipeline = backward_pipeline,
            .sort_pipeline = sort_pipeline,
            .gravity_pipeline = gravity_pipeline,
            .allocator = allocator,
        };
    }

    /// Set the active shader variant for attention computation
    /// Returns error if the requested variant is not available
    pub fn setShaderVariant(self: *Self, variant: ShaderVariant) !void {
        switch (variant) {
            .baseline => {
                self.active_variant = .baseline;
                log.info("Switched to baseline shader", .{});
            },
            .fast => {
                if (self.fast_pipeline == null) return error.ShaderVariantNotAvailable;
                self.active_variant = .fast;
                log.info("Switched to fast FP32 shader", .{});
            },
            .fp16 => {
                if (self.fp16_pipeline == null) return error.ShaderVariantNotAvailable;
                self.active_variant = .fp16;
                log.info("Switched to FP16 shader", .{});
            },
            .fp16_amd => {
                if (self.fp16_amd_pipeline == null) return error.ShaderVariantNotAvailable;
                self.active_variant = .fp16_amd;
                log.info("Switched to FP16 AMD-optimized shader", .{});
            },
        }
    }

    /// Get the currently active shader variant
    pub fn getShaderVariant(self: *const Self) ShaderVariant {
        return self.active_variant;
    }

    /// Get the active pipeline based on current variant
    fn getActivePipeline(self: *const Self) *const AttentionPipeline {
        return switch (self.active_variant) {
            .baseline => &self.pipeline,
            .fast => if (self.fast_pipeline) |*p| p else &self.pipeline,
            .fp16 => if (self.fp16_pipeline) |*p| p else &self.pipeline,
            .fp16_amd => if (self.fp16_amd_pipeline) |*p| p else &self.pipeline,
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

        // Sort keys buffers (for magnitude-based or projection-based sorting)
        const sort_keys_size = num_elements * 4;
        var sort_keys_final = try self.buffer_manager.createBuffer(sort_keys_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
        defer self.buffer_manager.destroyBuffer(&sort_keys_final);
        var sort_keys_temp = try self.buffer_manager.createBuffer(sort_keys_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
        defer self.buffer_manager.destroyBuffer(&sort_keys_temp);

        // 2. Initialize Indices to 0..S-1 per segment
        const S = keys.shape[2];
        const num_segments = keys.shape[0] * keys.shape[1];
        try sort_pipe.dispatchIota(indices.getBuffer(), num_elements, @intCast(S));

        // 3. Compute sort keys (magnitude-based if available)
        if (sort_pipe.hasMagnitudeSort()) {
            try sort_pipe.dispatchMagnitude(
                keys.getBuffer(),
                indices.getBuffer(),
                sort_keys_final.buffer,
                num_elements,
                d_model,
                @intCast(num_segments),
                @intCast(S)
            );
        }

        // 4. Perform Radix Sort (4 passes)
        std.debug.print("DEBUG: dispatchRadix PRE-CALL. d_model={}, num_elements={}, S={}\n", .{d_model, num_elements, S});
        if (d_model == 0) return error.InvalidDModel;

        try sort_pipe.dispatchRadix(
            keys.getBuffer(),
            values.getBuffer(),
            indices.getBuffer(),
            inds_temp.buffer,
            sort_keys_final.buffer,
            sort_keys_temp.buffer,
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
        // Deinit PagedAttention block management
        if (self.block_table) |*bt| bt.deinit();
        if (self.block_pool) |*bp| bp.deinit();
        // Deinit all shader pipelines
        if (self.fast_pipeline) |*p| p.deinit();
        if (self.fp16_pipeline) |*p| p.deinit();
        if (self.fp16_amd_pipeline) |*p| p.deinit();
        if (self.paged_pipeline) |*p| p.deinit();
        if (self.copy_kv_pipeline) |*p| p.deinit();
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
    /// window_size: sliding window size (-1 for full attention)
    pub fn forward(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
        rot_cos: ?*const GpuTensor,
        rot_sin: ?*const GpuTensor,
        causal: bool,
        window_size: i32,
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

        // Get the active pipeline based on shader variant
        const active_pipe = self.getActivePipeline();

        // Update descriptors to point to the GPU tensors
        active_pipe.updateDescriptors(
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

        log.info("Dispatching ({s}): B={d}, H={d}, KVH={d}, QLen={d}, KLen={d}", .{
            @tagName(self.active_variant), batch_size, num_heads, num_kv_heads, seq_len, key_seq_len
        });

        // Dispatch - data stays on GPU!
        try active_pipe.dispatch(batch_size, num_heads, num_kv_heads, seq_len, key_seq_len, head_dim, causal, has_rope, window_size);
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
        window_size: i32,
    ) !void {
        try self.forward(Q, K, V, output, rot_cos, rot_sin, causal, window_size);
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

    /// PagedAttention forward pass with block-based KV cache
    /// Lazily initializes BlockPool and BlockTable on first call
    /// Automatically allocates blocks for sequences and copies K/V into paged format
    pub fn forwardPaged(
        self: *Self,
        Q: *const GpuTensor,
        K: *const GpuTensor,
        V: *const GpuTensor,
        output: *GpuTensor,
        rot_cos: ?*const GpuTensor,
        rot_sin: ?*const GpuTensor,
        causal: bool,
        window_size: i32,
    ) !void {
        const paged_pipe = self.paged_pipeline orelse return error.PagedPipelineNotInitialized;

        // Validate shapes
        if (Q.ndim != 4 or K.ndim != 4 or V.ndim != 4 or output.ndim != 4) {
            return error.InvalidShape;
        }

        const batch_size = Q.shape[0];
        const num_heads = Q.shape[1];
        const seq_len = Q.shape[2];
        const head_dim = Q.shape[3];
        const num_kv_heads = K.shape[1];
        const key_seq_len = K.shape[2];

        // Validate shapes
        if (K.shape[0] != batch_size or (num_heads % num_kv_heads != 0) or K.shape[3] != head_dim) {
            return error.ShapeMismatch;
        }
        if (V.shape[0] != batch_size or V.shape[1] != num_kv_heads or
            V.shape[2] != key_seq_len or V.shape[3] != head_dim) {
            return error.ShapeMismatch;
        }
        if (output.shape[0] != batch_size or output.shape[1] != num_heads or
            output.shape[2] != seq_len or output.shape[3] != head_dim) {
            return error.ShapeMismatch;
        }
        if (head_dim > 64) {
            return error.HeadDimTooLarge;
        }

        // Lazy initialization of block pool and table
        if (self.block_pool == null) {
            const BlockPoolConfig = @import("block_pool.zig").BlockPoolConfig;
            const config = BlockPoolConfig{
                .initial_blocks = 512,
                .blocks_per_chunk = 512,
                .max_blocks = 8192,
                .block_size = 32, // Matches BLOCK_SIZE in shader
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
            };
            self.block_pool = try BlockPool.init(self.allocator, &self.buffer_manager, config);
            log.info("Initialized BlockPool: {} blocks, block_size=32", .{config.initial_blocks});
        }

        if (self.block_table == null) {
            // max_blocks per request = max_seq_len / block_size
            const max_blocks_per_request = 256; // 256 * 32 = 8192 tokens max
            self.block_table = try BlockTable.init(
                self.allocator,
                self.ctx,
                &self.buffer_manager,
                batch_size,
                max_blocks_per_request,
            );
            log.info("Initialized BlockTable: batch={}, max_blocks={}", .{batch_size, max_blocks_per_request});
        }

        var block_pool = &self.block_pool.?;
        var block_table = &self.block_table.?;

        // Calculate blocks needed per sequence
        const tokens_per_block = 32;
        const blocks_needed = (key_seq_len + tokens_per_block - 1) / tokens_per_block;

        log.debug("PagedAttention: seq_len={}, blocks_needed={}", .{key_seq_len, blocks_needed});

        // Allocate blocks for each sequence in batch
        var allocated_blocks = try self.allocator.alloc([]u32, batch_size);
        defer {
            for (allocated_blocks) |blocks| {
                self.allocator.free(blocks);
            }
            self.allocator.free(allocated_blocks);
        }

        for (0..batch_size) |batch_idx| {
            allocated_blocks[batch_idx] = try self.allocator.alloc(u32, blocks_needed);
            for (0..blocks_needed) |block_idx| {
                const physical_block = try block_pool.allocateBlock();
                allocated_blocks[batch_idx][block_idx] = physical_block;
                block_table.set(
                    @intCast(batch_idx),
                    @intCast(block_idx),
                    @intCast(physical_block),
                );
            }
        }

        // Sync block table to GPU
        try block_table.sync();

        // Copy K/V data into paged format
        try self.copyKVToPaged(K, V, block_table, batch_size, num_kv_heads, key_seq_len, head_dim);

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
            } else {
                return error.MissingRotarySin;
            }
        } else if (rot_sin != null) {
            return error.MissingRotaryCos;
        }

        // Update descriptors with paged buffers
        const num_physical_blocks = block_pool.total_blocks;
        const block_table_size = block_table.batch_size * block_table.max_blocks * @sizeOf(i32);
        const kv_pool_size = num_physical_blocks * 2 * num_kv_heads * tokens_per_block * head_dim * @sizeOf(f32);

        paged_pipe.updateDescriptors(
            Q.getBuffer(),
            output.getBuffer(),
            cos_buf,
            sin_buf,
            block_table.getStagingBuffer(), // Use staging buffer directly (MVP)
            block_pool.getBuffer(),
            Q.byteSize(),
            output.byteSize(),
            rope_size,
            block_table_size,
            kv_pool_size,
        );

        log.info("Dispatching Paged: B={d}, H={d}, KVH={d}, QLen={d}, KLen={d}, blocks={d}", .{
            batch_size, num_heads, num_kv_heads, seq_len, key_seq_len, num_physical_blocks,
        });

        // Dispatch paged attention shader
        try paged_pipe.dispatch(
            batch_size,
            num_heads,
            num_kv_heads,
            seq_len,
            key_seq_len,
            head_dim,
            causal,
            has_rope,
            window_size,
            block_table.max_blocks,
            num_physical_blocks,
        );

        // Free allocated blocks (for MVP - in production we'd keep them cached)
        for (0..batch_size) |batch_idx| {
            for (allocated_blocks[batch_idx]) |physical_block| {
                block_pool.freeBlock(physical_block);
            }
        }
    }

    /// Copy contiguous K/V tensors into paged block pool format using compute shader
    fn copyKVToPaged(
        self: *Self,
        K: *const GpuTensor,
        V: *const GpuTensor,
        block_table: *const BlockTable,
        batch_size: u32,
        num_kv_heads: u32,
        seq_len: u32,
        head_dim: u32,
    ) !void {
        const copy_pipe = self.copy_kv_pipeline orelse return error.CopyKVPipelineNotInitialized;
        const block_pool = &(self.block_pool orelse return error.BlockPoolNotInitialized);

        // Get buffer sizes
        const k_size = K.buffer.size;
        const v_size = V.buffer.size;
        const block_table_size = block_table.table_buffer.size;
        const kv_pool_size = block_pool.kv_pool_buffer.size;

        // Update descriptor bindings
        // Use staging buffer (MVP - same as PagedAttention pipeline)
        copy_pipe.updateDescriptors(
            K.buffer.buffer,
            V.buffer.buffer,
            block_table.staging_buffer.buffer,  // Use staging buffer, not table_buffer
            block_pool.kv_pool_buffer.buffer,
            k_size,
            v_size,
            block_table_size,
            kv_pool_size,
        );

        // Dispatch copy shader
        try copy_pipe.dispatch(
            batch_size,
            num_kv_heads,
            seq_len,
            head_dim,
            block_table.max_blocks,
            block_pool.total_blocks,
        );

        log.info("K/V tensors copied to paged format: {} tokens across {} blocks", .{
            seq_len,
            (seq_len + 31) / 32,
        });
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
    /// window_size: sliding window size (-1 for full attention)
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
        window_size: i32,
    ) !void {
        const gravity_pipe = self.gravity_pipeline orelse return error.GravityPipelineNotInitialized;

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

        // Check if indices are per-Q-head (shape B*H*S) or per-KV-head (B*KVH*S)
        // If indices have the right element count for Q-heads and were uploaded from Python,
        // skip the internal sorting phase - the caller provided pre-sorted indices.
        const expected_indices_for_q_heads = batch_size * num_heads * key_seq_len;
        const provided_indices_count = indices.element_count;
        const skip_sorting = (provided_indices_count == expected_indices_for_q_heads);

        if (!skip_sorting) {
            // --- SORTING PHASE (when indices not pre-provided) ---
            const sort_pipe = self.sort_pipeline orelse return error.SortPipelineNotInitialized;

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

            // Sort Keys buffers for ping-pong (magnitude-based sorting)
            const sort_keys_size = num_elements * 4; // uint per element
            var sort_keys_final = try self.buffer_manager.createBuffer(sort_keys_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
            defer self.buffer_manager.destroyBuffer(&sort_keys_final);
            var sort_keys_temp = try self.buffer_manager.createBuffer(sort_keys_size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
            defer self.buffer_manager.destroyBuffer(&sort_keys_temp);

            // 2. Dispatch Iota (Initialize Indices to 0..S-1 per segment)
            const S = key_seq_len;
            const num_segments = batch_size * num_kv_heads;
            try sort_pipe.dispatchIota(indices.getBuffer(), num_elements, @intCast(S));

            // 3. Compute magnitude-based sort keys (if available)
            if (sort_pipe.hasMagnitudeSort()) {
                try sort_pipe.dispatchMagnitude(
                    K.getBuffer(),
                    indices.getBuffer(),
                    sort_keys_final.buffer,
                    num_elements,
                    head_dim, // d_model
                    @intCast(num_segments),
                    @intCast(S)
                );
            }

            // 4. Dispatch Radix Sort using pre-computed sort keys
            try sort_pipe.dispatchRadix(
               K.getBuffer(),
               V.getBuffer(),
               indices.getBuffer(),      // Final indices
               inds_temp.buffer,         // Temp indices
               sort_keys_final.buffer,   // Final sort keys
               sort_keys_temp.buffer,    // Temp sort keys
               hist_buf.buffer,          // Histograms
               num_elements,
               head_dim, // d_model
               0,        // Sort Dim (unused with magnitude sort)
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
             batch_size, num_heads, num_kv_heads, seq_len, key_seq_len, head_dim, causal, has_rope, max_attend, window_size
        );
    }
};
