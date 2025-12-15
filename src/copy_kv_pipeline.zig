const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;

const log = std.log.scoped(.copy_kv_pipeline);

/// Push constants for K/V copy shader
pub const CopyKVPushConstants = extern struct {
    batch_size: u32,
    num_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    max_blocks_per_request: u32,
    num_physical_blocks: u32,
};

/// Pipeline for copying contiguous K/V tensors into paged block pool format
/// Requires 4 descriptor bindings: 0=K, 1=V, 2=BlockTable, 3=KVPool
pub const CopyKVPipeline = struct {
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
    const BLOCK_SIZE: u32 = 32; // Must match shader

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

        // Descriptor set layout: 4 storage buffers
        // 0=K, 1=V, 2=BlockTable, 3=KVPool
        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{ // Binding 0: Key
                .binding = 0,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{ // Binding 1: Value
                .binding = 1,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{ // Binding 2: BlockTable
                .binding = 2,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{ // Binding 3: KVPool
                .binding = 3,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
        };

        const descriptor_set_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, &.{
            .flags = .{},
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, descriptor_set_layout, null);

        // Pipeline layout with push constants
        const push_constant_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(CopyKVPushConstants),
        };

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, &.{
            .flags = .{},
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_constant_range),
        }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, null);

        // Compute pipeline
        var pipeline: vk.Pipeline = undefined;
        _ = try ctx.vkd.createComputePipelines(
            ctx.device,
            .null_handle,
            1,
            @ptrCast(&vk.ComputePipelineCreateInfo{
                .flags = .{},
                .stage = .{
                    .flags = .{},
                    .stage = .{ .compute_bit = true },
                    .module = shader_module,
                    .p_name = "main",
                    .p_specialization_info = null,
                },
                .layout = pipeline_layout,
                .base_pipeline_handle = .null_handle,
                .base_pipeline_index = -1,
            }),
            null,
            @ptrCast(&pipeline),
        );
        errdefer ctx.vkd.destroyPipeline(ctx.device, pipeline, null);

        // Descriptor pool
        const pool_size = vk.DescriptorPoolSize{
            .type = .storage_buffer,
            .descriptor_count = 4, // 4 storage buffers
        };

        const descriptor_pool = try ctx.vkd.createDescriptorPool(ctx.device, &.{
            .flags = .{},
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

        // Command pool and buffer
        const command_pool = try ctx.vkd.createCommandPool(ctx.device, &.{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = ctx.queue_family_index,
        }, null);
        errdefer ctx.vkd.destroyCommandPool(ctx.device, command_pool, null);

        var command_buffer: vk.CommandBuffer = undefined;
        try ctx.vkd.allocateCommandBuffers(ctx.device, &.{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&command_buffer));

        // Fence for synchronization
        const fence = try ctx.vkd.createFence(ctx.device, &.{
            .flags = .{},
        }, null);
        errdefer ctx.vkd.destroyFence(ctx.device, fence, null);

        log.info("CopyKVPipeline initialized with 4 bindings", .{});

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

    /// Update descriptor bindings for K/V copy
    pub fn updateDescriptors(
        self: *const Self,
        k_buffer: vk.Buffer,
        v_buffer: vk.Buffer,
        block_table_buffer: vk.Buffer,
        kv_pool_buffer: vk.Buffer,
        k_size: vk.DeviceSize,
        v_size: vk.DeviceSize,
        block_table_size: vk.DeviceSize,
        kv_pool_size: vk.DeviceSize,
    ) void {
        const buffer_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = k_buffer, .offset = 0, .range = k_size },
            .{ .buffer = v_buffer, .offset = 0, .range = v_size },
            .{ .buffer = block_table_buffer, .offset = 0, .range = block_table_size },
            .{ .buffer = kv_pool_buffer, .offset = 0, .range = kv_pool_size },
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
        };

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, @intCast(writes.len), &writes, 0, null);
    }

    /// Dispatch K/V copy shader
    pub fn dispatch(
        self: *const Self,
        batch_size: u32,
        num_kv_heads: u32,
        seq_len: u32,
        head_dim: u32,
        max_blocks_per_request: u32,
        num_physical_blocks: u32,
    ) !void {
        const push_constants = CopyKVPushConstants{
            .batch_size = batch_size,
            .num_kv_heads = num_kv_heads,
            .seq_len = seq_len,
            .head_dim = head_dim,
            .max_blocks_per_request = max_blocks_per_request,
            .num_physical_blocks = num_physical_blocks,
        };

        // Workgroup dimensions: (batch, heads, tokens/32)
        const num_token_groups = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
            @sizeOf(CopyKVPushConstants),
            std.mem.asBytes(&push_constants),
        );

        // Dispatch: (batch, heads, token_groups)
        self.ctx.vkd.cmdDispatch(self.command_buffer, batch_size, num_kv_heads, num_token_groups);

        try self.ctx.vkd.endCommandBuffer(self.command_buffer);

        // Submit command buffer
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
        try self.ctx.vkd.resetFences(self.ctx.device, 1, @ptrCast(&self.fence));
    }
};
