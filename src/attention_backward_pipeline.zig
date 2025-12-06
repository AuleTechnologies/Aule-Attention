const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;

const log = std.log.scoped(.attention_backward_pipeline);

/// Push constants for forward pass with LSE output
pub const ForwardWithLsePushConstants = extern struct {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
    store_lse: u32, // 1 to store LSE, 0 to skip
};

/// Push constants for backward pass shader
pub const BackwardPushConstants = extern struct {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
};

/// Pipeline for forward pass with LSE output (for training)
pub const ForwardWithLsePipeline = struct {
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
        // Create shader module with alignment handling
        const code_ptr = @intFromPtr(shader_code.ptr);
        const is_aligned = (code_ptr & 3) == 0;

        const shader_module = if (is_aligned) blk: {
            const aligned_code: []align(4) const u8 = @alignCast(shader_code);
            break :blk try ctx.vkd.createShaderModule(ctx.device, &.{
                .code_size = shader_code.len,
                .p_code = std.mem.bytesAsSlice(u32, aligned_code).ptr,
            }, null);
        } else blk: {
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

        // 5 storage buffers: Q, K, V, O, LSE
        const bindings = [_]vk.DescriptorSetLayoutBinding{
            makeBinding(0), // Q
            makeBinding(1), // K
            makeBinding(2), // V
            makeBinding(3), // O
            makeBinding(4), // LSE
        };

        const descriptor_set_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, &.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, descriptor_set_layout, null);

        const push_constant_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(ForwardWithLsePushConstants),
        };

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_constant_range),
        }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, null);

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

        const pool_size = vk.DescriptorPoolSize{
            .type = .storage_buffer,
            .descriptor_count = 5,
        };

        const descriptor_pool = try ctx.vkd.createDescriptorPool(ctx.device, &.{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&pool_size),
        }, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, descriptor_pool, null);

        var descriptor_set: vk.DescriptorSet = undefined;
        try ctx.vkd.allocateDescriptorSets(ctx.device, &.{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
        }, @ptrCast(&descriptor_set));

        const command_pool = try ctx.vkd.createCommandPool(ctx.device, &.{
            .queue_family_index = ctx.queue_family_index,
            .flags = .{ .reset_command_buffer_bit = true },
        }, null);
        errdefer ctx.vkd.destroyCommandPool(ctx.device, command_pool, null);

        var command_buffer: vk.CommandBuffer = undefined;
        try ctx.vkd.allocateCommandBuffers(ctx.device, &.{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&command_buffer));

        const fence = try ctx.vkd.createFence(ctx.device, &.{ .flags = .{} }, null);

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
        lse_buffer: vk.Buffer,
        tensor_size: vk.DeviceSize,
        lse_size: vk.DeviceSize,
    ) void {
        const buffer_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = q_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = k_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = v_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = o_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = lse_buffer, .offset = 0, .range = lse_size },
        };

        var writes: [5]vk.WriteDescriptorSet = undefined;
        for (0..5) |i| {
            writes[i] = .{
                .dst_set = self.descriptor_set,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[i]),
                .p_texel_buffer_view = undefined,
            };
        }

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, 5, &writes, 0, null);
    }

    pub fn dispatch(
        self: *const Self,
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        head_dim: u32,
        causal: bool,
    ) !void {
        const push_constants = ForwardWithLsePushConstants{
            .batch_size = batch_size,
            .num_heads = num_heads,
            .seq_len = seq_len,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .causal = if (causal) 1 else 0,
            .store_lse = 1,
        };

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
            @sizeOf(ForwardWithLsePushConstants),
            std.mem.asBytes(&push_constants),
        );

        self.ctx.vkd.cmdDispatch(self.command_buffer, 1, num_row_blocks, num_batch_head);

        try self.ctx.vkd.endCommandBuffer(self.command_buffer);

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

/// Pipeline for backward pass (gradient computation)
pub const BackwardPipeline = struct {
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
        // Create shader module with alignment handling
        const code_ptr = @intFromPtr(shader_code.ptr);
        const is_aligned = (code_ptr & 3) == 0;

        const shader_module = if (is_aligned) blk: {
            const aligned_code: []align(4) const u8 = @alignCast(shader_code);
            break :blk try ctx.vkd.createShaderModule(ctx.device, &.{
                .code_size = shader_code.len,
                .p_code = std.mem.bytesAsSlice(u32, aligned_code).ptr,
            }, null);
        } else blk: {
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

        // 9 storage buffers: Q, K, V, O, dO, LSE, dQ, dK, dV
        const bindings = [_]vk.DescriptorSetLayoutBinding{
            makeBinding(0), // Q
            makeBinding(1), // K
            makeBinding(2), // V
            makeBinding(3), // O
            makeBinding(4), // dO
            makeBinding(5), // LSE
            makeBinding(6), // dQ
            makeBinding(7), // dK
            makeBinding(8), // dV
        };

        const descriptor_set_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, &.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, descriptor_set_layout, null);

        const push_constant_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(BackwardPushConstants),
        };

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_constant_range),
        }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, null);

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

        const pool_size = vk.DescriptorPoolSize{
            .type = .storage_buffer,
            .descriptor_count = 9,
        };

        const descriptor_pool = try ctx.vkd.createDescriptorPool(ctx.device, &.{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&pool_size),
        }, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, descriptor_pool, null);

        var descriptor_set: vk.DescriptorSet = undefined;
        try ctx.vkd.allocateDescriptorSets(ctx.device, &.{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
        }, @ptrCast(&descriptor_set));

        const command_pool = try ctx.vkd.createCommandPool(ctx.device, &.{
            .queue_family_index = ctx.queue_family_index,
            .flags = .{ .reset_command_buffer_bit = true },
        }, null);
        errdefer ctx.vkd.destroyCommandPool(ctx.device, command_pool, null);

        var command_buffer: vk.CommandBuffer = undefined;
        try ctx.vkd.allocateCommandBuffers(ctx.device, &.{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&command_buffer));

        const fence = try ctx.vkd.createFence(ctx.device, &.{ .flags = .{} }, null);

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
        do_buffer: vk.Buffer,
        lse_buffer: vk.Buffer,
        dq_buffer: vk.Buffer,
        dk_buffer: vk.Buffer,
        dv_buffer: vk.Buffer,
        tensor_size: vk.DeviceSize,
        lse_size: vk.DeviceSize,
    ) void {
        const buffer_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = q_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = k_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = v_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = o_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = do_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = lse_buffer, .offset = 0, .range = lse_size },
            .{ .buffer = dq_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = dk_buffer, .offset = 0, .range = tensor_size },
            .{ .buffer = dv_buffer, .offset = 0, .range = tensor_size },
        };

        var writes: [9]vk.WriteDescriptorSet = undefined;
        for (0..9) |i| {
            writes[i] = .{
                .dst_set = self.descriptor_set,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&buffer_infos[i]),
                .p_texel_buffer_view = undefined,
            };
        }

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, 9, &writes, 0, null);
    }

    pub fn dispatch(
        self: *const Self,
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        head_dim: u32,
        causal: bool,
    ) !void {
        const push_constants = BackwardPushConstants{
            .batch_size = batch_size,
            .num_heads = num_heads,
            .seq_len = seq_len,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .causal = if (causal) 1 else 0,
        };

        // Backward pass iterates over K/V blocks in Y dimension
        const num_kv_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
            @sizeOf(BackwardPushConstants),
            std.mem.asBytes(&push_constants),
        );

        self.ctx.vkd.cmdDispatch(self.command_buffer, 1, num_kv_blocks, num_batch_head);

        try self.ctx.vkd.endCommandBuffer(self.command_buffer);

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

fn makeBinding(binding: u32) vk.DescriptorSetLayoutBinding {
    return .{
        .binding = binding,
        .descriptor_type = .storage_buffer,
        .descriptor_count = 1,
        .stage_flags = .{ .compute_bit = true },
        .p_immutable_samplers = null,
    };
}
