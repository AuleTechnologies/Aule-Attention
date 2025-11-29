const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;

const log = std.log.scoped(.compute_pipeline);

pub const ComputePipeline = struct {
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

    pub fn init(ctx: *const VulkanContext, shader_code: []const u8) !Self {
        // Create shader module - handle alignment for SPIR-V (requires 4-byte alignment)
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

        // Descriptor set layout: 2 storage buffers (input, output)
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
        };

        const descriptor_set_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, &.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, descriptor_set_layout, null);

        // Push constant range for element count
        const push_constant_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(u32),
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
            .descriptor_count = 2,
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

        // Fence for synchronization
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

    pub fn updateDescriptors(self: *const Self, input_buffer: vk.Buffer, output_buffer: vk.Buffer, size: vk.DeviceSize) void {
        const buffer_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = input_buffer, .offset = 0, .range = size },
            .{ .buffer = output_buffer, .offset = 0, .range = size },
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
        };

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, writes.len, &writes, 0, null);
    }

    pub fn dispatch(self: *const Self, count: u32) !void {
        const workgroup_size: u32 = 256;
        const num_workgroups = (count + workgroup_size - 1) / workgroup_size;

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
            @sizeOf(u32),
            std.mem.asBytes(&count),
        );
        self.ctx.vkd.cmdDispatch(self.command_buffer, num_workgroups, 1, 1);

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

    pub fn recordCopyAndDispatch(
        self: *const Self,
        staging_in: vk.Buffer,
        device_in: vk.Buffer,
        device_out: vk.Buffer,
        staging_out: vk.Buffer,
        size: vk.DeviceSize,
        count: u32,
    ) !void {
        const workgroup_size: u32 = 256;
        const num_workgroups = (count + workgroup_size - 1) / workgroup_size;

        try self.ctx.vkd.resetCommandBuffer(self.command_buffer, .{});

        try self.ctx.vkd.beginCommandBuffer(self.command_buffer, &.{
            .flags = .{ .one_time_submit_bit = true },
            .p_inheritance_info = null,
        });

        // Copy from staging to device
        const copy_region = vk.BufferCopy{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        };
        self.ctx.vkd.cmdCopyBuffer(self.command_buffer, staging_in, device_in, 1, @ptrCast(&copy_region));

        // Memory barrier: transfer write -> shader read
        const barrier_to_compute = vk.BufferMemoryBarrier{
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = device_in,
            .offset = 0,
            .size = size,
        };
        self.ctx.vkd.cmdPipelineBarrier(
            self.command_buffer,
            .{ .transfer_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            0,
            null,
            1,
            @ptrCast(&barrier_to_compute),
            0,
            null,
        );

        // Bind and dispatch
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
            @sizeOf(u32),
            std.mem.asBytes(&count),
        );
        self.ctx.vkd.cmdDispatch(self.command_buffer, num_workgroups, 1, 1);

        // Memory barrier: shader write -> transfer read
        const barrier_to_transfer = vk.BufferMemoryBarrier{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .transfer_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = device_out,
            .offset = 0,
            .size = size,
        };
        self.ctx.vkd.cmdPipelineBarrier(
            self.command_buffer,
            .{ .compute_shader_bit = true },
            .{ .transfer_bit = true },
            .{},
            0,
            null,
            1,
            @ptrCast(&barrier_to_transfer),
            0,
            null,
        );

        // Copy from device to staging
        self.ctx.vkd.cmdCopyBuffer(self.command_buffer, device_out, staging_out, 1, @ptrCast(&copy_region));

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
