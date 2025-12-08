const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const BufferManager = @import("buffer_manager.zig").BufferManager;
const Buffer = @import("buffer_manager.zig").Buffer;

const log = std.log.scoped(.attention_pipeline);

/// Push constants for attention shader
pub const AttentionPushConstants = extern struct {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    causal: u32, // 1 for causal masking (LLMs), 0 for bidirectional
    has_rope: u32, // 1 to apply RoPE, 0 to skip
    num_kv_heads: u32, // Number of K/V heads
    key_seq_len: u32, // Sequence length of K/V
};

pub const AttentionPipeline = struct {
    ctx: *const VulkanContext,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptor_pool: vk.DescriptorPool,
    descriptor_set: vk.DescriptorSet,
    command_pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
    fence: vk.Fence,

    const Self = @This();
    const BLOCK_SIZE: u32 = 16;

    pub fn init(ctx: *const VulkanContext, shader_code: []const u8) !Self {
        // Create shader module - handle alignment for SPIR-V (requires 4-byte alignment)
        // Use pointer alignment check since embedFile may not be aligned
        const code_ptr = @intFromPtr(shader_code.ptr);
        const is_aligned = (code_ptr & 3) == 0;

        const shader_module = if (is_aligned) blk: {
            const aligned_code: []align(4) const u8 = @alignCast(shader_code);
            break :blk try ctx.vkd.createShaderModule(ctx.device, &.{
                .code_size = shader_code.len,
                .p_code = std.mem.bytesAsSlice(u32, aligned_code).ptr,
            }, null);
        } else blk: {
            // Create aligned copy
            const allocator = std.heap.c_allocator;
            const aligned_buf = try allocator.alignedAlloc(u8, 4, shader_code.len);
            defer allocator.free(aligned_buf);
            @memcpy(aligned_buf, shader_code);
            break :blk try ctx.vkd.createShaderModule(ctx.device, &.{
                .code_size = shader_code.len,
                .p_code = std.mem.bytesAsSlice(u32, aligned_buf).ptr,
            }, null);
        };
        defer ctx.vkd.destroyShaderModule(ctx.device, shader_module, null);

        // Descriptor set layout: 4 storage buffers (Q, K, V, O)
        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{
                .binding = 0,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 1,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 2,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 3,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 4,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 5,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
        };

        const descriptor_set_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, &.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, descriptor_set_layout, null);

        // Push constant range for attention parameters
        const push_constant_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(AttentionPushConstants),
        };

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_constant_range),
        }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, null);

        // Create compute pipeline
        const pipeline_info = vk.ComputePipelineCreateInfo{
            .stage = .{
                .stage = .{ .compute_bit = true },
                .module = shader_module,
                .p_name = "main",
                .p_specialization_info = null,
            },
            .layout = pipeline_layout,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        var pipeline: vk.Pipeline = undefined;
        _ = try ctx.vkd.createComputePipelines(
            ctx.device,
            .null_handle,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&pipeline),
        );
        errdefer ctx.vkd.destroyPipeline(ctx.device, pipeline, null);

        // Descriptor pool
        const pool_size = vk.DescriptorPoolSize{
            .type = .storage_buffer,
            .descriptor_count = 6,
        };

        const descriptor_pool = try ctx.vkd.createDescriptorPool(ctx.device, &.{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&pool_size),
        }, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, descriptor_pool, null);

        // Allocate descriptor set
        var descriptor_set: vk.DescriptorSet = undefined;
        try ctx.vkd.allocateDescriptorSets(ctx.device, &.{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
        }, @ptrCast(&descriptor_set));

        // Command pool
        const command_pool = try ctx.vkd.createCommandPool(ctx.device, &.{
            .queue_family_index = ctx.queue_family_index,
            .flags = .{ .reset_command_buffer_bit = true },
        }, null);
        errdefer ctx.vkd.destroyCommandPool(ctx.device, command_pool, null);

        // Command buffer
        var command_buffer: vk.CommandBuffer = undefined;
        try ctx.vkd.allocateCommandBuffers(ctx.device, &.{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&command_buffer));

        // Fence
        const fence = try ctx.vkd.createFence(ctx.device, &.{
            .flags = .{},
        }, null);

        return Self{
            .ctx = ctx,
            .pipeline = pipeline,
            .pipeline_layout = pipeline_layout,
            .descriptor_set_layout = descriptor_set_layout,
            .descriptor_pool = descriptor_pool,
            .descriptor_set = descriptor_set,
            .command_pool = command_pool,
            .command_buffer = command_buffer,
            .fence = fence,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.vkd.destroyFence(self.ctx.device, self.fence, null);
        self.ctx.vkd.destroyCommandPool(self.ctx.device, self.command_pool, null);
        self.ctx.vkd.destroyDescriptorPool(self.ctx.device, self.descriptor_pool, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.pipeline, null);
        self.ctx.vkd.destroyPipelineLayout(self.ctx.device, self.pipeline_layout, null);
        self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.device, self.descriptor_set_layout, null);
        self.* = undefined;
    }

    pub fn updateDescriptors(
        self: *const Self,
        q_buffer: vk.Buffer,
        k_buffer: vk.Buffer,
        v_buffer: vk.Buffer,
        o_buffer: vk.Buffer,
        rot_cos_buffer: ?vk.Buffer,
        rot_sin_buffer: ?vk.Buffer,
        size: vk.DeviceSize,
        rope_size: vk.DeviceSize,
    ) void {
        const valid_rope_size = if (rope_size > 0) rope_size else 64; // Fallback size for valid validation if null
        // We must provide valid handles even if unused, due to descriptor set layout.
        // If null, we reuse q_buffer (safe because we won't read it if has_rope=0).
        const cos_buf = if (rot_cos_buffer) |b| b else q_buffer;
        const sin_buf = if (rot_sin_buffer) |b| b else q_buffer;
        
        const buffer_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = q_buffer, .offset = 0, .range = size },
            .{ .buffer = k_buffer, .offset = 0, .range = size },
            .{ .buffer = v_buffer, .offset = 0, .range = size },
            .{ .buffer = o_buffer, .offset = 0, .range = size },
            .{ .buffer = cos_buf, .offset = 0, .range = valid_rope_size },
            .{ .buffer = sin_buf, .offset = 0, .range = valid_rope_size },
        };

        const writes = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = self.descriptor_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[0]),
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = self.descriptor_set,
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[1]),
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = self.descriptor_set,
                .dst_binding = 2,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[2]),
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = self.descriptor_set,
                .dst_binding = 3,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[3]),
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = self.descriptor_set,
                .dst_binding = 4,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[4]),
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = self.descriptor_set,
                .dst_binding = 5,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[5]),
                .p_texel_buffer_view = undefined,
            },
        };

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, writes.len, &writes, 0, null);
    }

    pub fn dispatch(
        self: *const Self,
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        seq_len: u32,
        key_seq_len: u32,
        head_dim: u32,
        causal: bool,
        has_rope: bool,
    ) !void {
        const push_constants = AttentionPushConstants{
            .batch_size = batch_size,
            .num_heads = num_heads,
            .seq_len = seq_len,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .causal = if (causal) 1 else 0,
            .has_rope = if (has_rope) 1 else 0,
            .num_kv_heads = num_kv_heads,
            .key_seq_len = key_seq_len,
        };

        // Workgroup dimensions
        // x: 1 (local_size_x = 16 handles the block width)
        // y: ceil(seq_len / BLOCK_SIZE) blocks for Q rows
        // z: batch_size * num_heads
        const num_row_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const num_batch_head = batch_size * num_heads;

        try self.ctx.vkd.resetCommandBuffer(self.command_buffer, .{});

        try self.ctx.vkd.beginCommandBuffer(self.command_buffer, &.{
            .flags = .{ .one_time_submit_bit = true },
            .p_inheritance_info = null,
        });

        self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, self.pipeline);
        self.ctx.vkd.cmdBindDescriptorSets(
            self.command_buffer,
            .compute,
            self.pipeline_layout,
            0,
            1,
            @ptrCast(&self.descriptor_set),
            0,
            null,
        );
        self.ctx.vkd.cmdPushConstants(
            self.command_buffer,
            self.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(AttentionPushConstants),
            std.mem.asBytes(&push_constants),
        );

        // Dispatch: 1 workgroup in x (handled by local_size), num_row_blocks in y, batch*heads in z
        self.ctx.vkd.cmdDispatch(self.command_buffer, 1, num_row_blocks, num_batch_head);

        try self.ctx.vkd.endCommandBuffer(self.command_buffer);

        // Submit and wait
        try self.ctx.vkd.resetFences(self.ctx.device, 1, @ptrCast(&self.fence));

        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = null,
            .p_wait_dst_stage_mask = null,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.command_buffer),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = null,
        };

        try self.ctx.vkd.queueSubmit(self.ctx.compute_queue, 1, @ptrCast(&submit_info), self.fence);
        _ = try self.ctx.vkd.waitForFences(self.ctx.device, 1, @ptrCast(&self.fence), vk.TRUE, std.math.maxInt(u64));
    }
};
