const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const BufferManager = @import("buffer_manager.zig").BufferManager;
const Buffer = @import("buffer_manager.zig").Buffer;

const log = std.log.scoped(.sort_pipeline);

/// Push constants for spatial sort shader
pub const SortPushConstants = extern struct {
    num_elements: u32,
    shift: u32,   // Added for Radix
    sort_dim: u32,
    d_model: u32,
    // Segmented Sort Support
    num_segments: u32,
    segment_size: u32,
};


pub const SortPipeline = struct {
    const Self = @This();
    const WORKGROUP_SIZE = 256;
    ctx: *const VulkanContext,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    descriptor_set_layout: vk.DescriptorSetLayout,
    // Additional pipelines for Radix Sort
    count_pipeline: vk.Pipeline,
    scan_pipeline: vk.Pipeline,
    scatter_pipeline: vk.Pipeline,
    iota_pipeline: vk.Pipeline,
    magnitude_pipeline: ?vk.Pipeline, // Optional magnitude sort key computation
    descriptor_pool: vk.DescriptorPool,
    descriptor_sets: [2]vk.DescriptorSet, // 0: Final->Temp, 1: Temp->Final
    command_pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
    fence: vk.Fence,

    pub fn init(ctx: *const VulkanContext,
        spatial_code: []const u8,
        count_code: []const u8,
        scan_code: []const u8,
        scatter_code: []const u8,
        iota_code: []const u8,
    ) !Self {
        return initWithMagnitude(ctx, spatial_code, count_code, scan_code, scatter_code, iota_code, null);
    }

    pub fn initWithMagnitude(ctx: *const VulkanContext,
        spatial_code: []const u8,
        count_code: []const u8,
        scan_code: []const u8,
        scatter_code: []const u8,
        iota_code: []const u8,
        magnitude_code: ?[]const u8,
    ) !Self {

        // ... (create pipelines/layout same as before) ...
        const createPipeline = struct {
            fn call(c: *const VulkanContext, code: []const u8, layout: vk.PipelineLayout) !vk.Pipeline {
                const code_ptr = @intFromPtr(code.ptr);
                const is_aligned = (code_ptr & 3) == 0;
                const module = if (is_aligned) blk: {
                    const aligned_code: []align(4) const u8 = @alignCast(code);
                    break :blk try c.vkd.createShaderModule(c.device, &.{
                        .code_size = code.len,
                        .p_code = std.mem.bytesAsSlice(u32, aligned_code).ptr,
                    }, null);
                } else blk: {
                    const allocator = std.heap.c_allocator;
                    const aligned_buf = try allocator.alignedAlloc(u8, 4, code.len);
                    defer allocator.free(aligned_buf);
                    @memcpy(aligned_buf, code);
                    break :blk try c.vkd.createShaderModule(c.device, &.{
                        .code_size = code.len,
                        .p_code = std.mem.bytesAsSlice(u32, aligned_buf).ptr,
                    }, null);
                };
                defer c.vkd.destroyShaderModule(c.device, module, null);
                const info = vk.ComputePipelineCreateInfo{
                    .stage = .{ .stage = .{ .compute_bit = true }, .module = module, .p_name = "main", .p_specialization_info = null },
                    .layout = layout, .base_pipeline_handle = .null_handle, .base_pipeline_index = -1,
                };
                var p: vk.Pipeline = undefined;
                _ = try c.vkd.createComputePipelines(c.device, .null_handle, 1, @ptrCast(&info), null, @ptrCast(&p));
                return p;
            }
        }.call;

        // Descriptor Set Layout (Superset)
        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, // Keys In / Iota Out
            .{ .binding = 1, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, // Vals In
            .{ .binding = 2, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, // Inds In
            .{ .binding = 3, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, 
            .{ .binding = 4, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, 
            .{ .binding = 5, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, // Inds Out
            .{ .binding = 6, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true }, .p_immutable_samplers = null }, // Histograms
        };
        const dsl = try ctx.vkd.createDescriptorSetLayout(ctx.device, &.{ .binding_count = bindings.len, .p_bindings = &bindings }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, dsl, null);
        const pcr = vk.PushConstantRange{ .stage_flags = .{ .compute_bit = true }, .offset = 0, .size = 32 }; // Increased size for safety
        const pl = try ctx.vkd.createPipelineLayout(ctx.device, &.{ .set_layout_count = 1, .p_set_layouts = @ptrCast(&dsl), .push_constant_range_count = 1, .p_push_constant_ranges = @ptrCast(&pcr) }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pl, null);

        const spatial_pipeline = try createPipeline(ctx, spatial_code, pl);
        const count_pipeline = try createPipeline(ctx, count_code, pl);
        const scan_pipeline = try createPipeline(ctx, scan_code, pl);
        const scatter_pipeline = try createPipeline(ctx, scatter_code, pl);
        const iota_pipeline = try createPipeline(ctx, iota_code, pl);

        // Optional magnitude pipeline for improved sorting
        var magnitude_pipeline: ?vk.Pipeline = null;
        if (magnitude_code) |code| {
            magnitude_pipeline = try createPipeline(ctx, code, pl);
        }

        // Pool: Need 2 sets * 7 bindings = 14 descriptors
        const pool_size = vk.DescriptorPoolSize{ .type = .storage_buffer, .descriptor_count = 16 };
        const pool = try ctx.vkd.createDescriptorPool(ctx.device, &.{ .max_sets = 2, .pool_size_count = 1, .p_pool_sizes = @ptrCast(&pool_size) }, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, pool, null);

        var sets: [2]vk.DescriptorSet = undefined;
        // Allocate 2 sets with same layout
        var layouts = [_]vk.DescriptorSetLayout{dsl, dsl};
        try ctx.vkd.allocateDescriptorSets(ctx.device, &.{ .descriptor_pool = pool, .descriptor_set_count = 2, .p_set_layouts = &layouts }, @ptrCast(&sets));

        const cp = try ctx.vkd.createCommandPool(ctx.device, &.{ .queue_family_index = ctx.queue_family_index, .flags = .{ .reset_command_buffer_bit = true } }, null);
        errdefer ctx.vkd.destroyCommandPool(ctx.device, cp, null);
        var cb: vk.CommandBuffer = undefined;
        try ctx.vkd.allocateCommandBuffers(ctx.device, &.{ .command_pool = cp, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&cb));
        const fence = try ctx.vkd.createFence(ctx.device, &.{ .flags = .{} }, null);

        return Self{
            .ctx = ctx,
            .pipeline = spatial_pipeline,
            .count_pipeline = count_pipeline,
            .scan_pipeline = scan_pipeline,
            .scatter_pipeline = scatter_pipeline,
            .iota_pipeline = iota_pipeline,
            .magnitude_pipeline = magnitude_pipeline,
            .pipeline_layout = pl,
            .descriptor_set_layout = dsl,
            .descriptor_pool = pool,
            .descriptor_sets = sets,
            .command_pool = cp,
            .command_buffer = cb,
            .fence = fence,
        };
    }
    
    // Dispatch Iota: Initialize Indices to 0..N (or 0..S-1 per segment)
    pub fn dispatchIota(self: *const Self, indices_buffer: vk.Buffer, num_elements: u32, segment_size: u32) !void {
        try self.ctx.vkd.resetCommandBuffer(self.command_buffer, .{});
        try self.ctx.vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
        
        // Update Set 0 Binding 0 to point to indices_buffer
        // We reuse Set 0. It will be overwritten by dispatchRadix later, which is fine.
        const set = self.descriptor_sets[0];
        const b_info = vk.DescriptorBufferInfo{ .buffer = indices_buffer, .offset = 0, .range = vk.WHOLE_SIZE };
        const write = vk.WriteDescriptorSet{
            .dst_set = set,
            .dst_binding = 0, // Binding 0 in shader
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = @ptrCast(&b_info),
            .dst_array_element = 0,
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };
        self.ctx.vkd.updateDescriptorSets(self.ctx.device, 1, @ptrCast(&write), 0, null);
        
        self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, self.iota_pipeline);
        self.ctx.vkd.cmdBindDescriptorSets(self.command_buffer, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
        
        const PC = extern struct { num_elements: u32, segment_size: u32 };
        const pc = PC{ .num_elements = num_elements, .segment_size = segment_size };
        self.ctx.vkd.cmdPushConstants(self.command_buffer, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(PC), std.mem.asBytes(&pc));
        
        const workgroups = (num_elements + 255) / 256;
        self.ctx.vkd.cmdDispatch(self.command_buffer, workgroups, 1, 1);
        
        try self.ctx.vkd.endCommandBuffer(self.command_buffer);

        try self.ctx.vkd.resetFences(self.ctx.device, 1, @ptrCast(&self.fence));
        const s = vk.SubmitInfo{ .command_buffer_count = 1, .p_command_buffers = @ptrCast(&self.command_buffer) };
        try self.ctx.vkd.queueSubmit(self.ctx.compute_queue, 1, @ptrCast(&s), self.fence);
        try self.ctx.vkd.deviceWaitIdle(self.ctx.device);
    }

    // Dispatch Magnitude: Compute sort keys from key magnitudes (L2 norm)
    // This produces sort keys that can be used by the radix sort
    pub fn dispatchMagnitude(
        self: *const Self,
        keys_buffer: vk.Buffer,      // Input keys [N, D]
        indices_buffer: vk.Buffer,   // Current index permutation
        sort_keys_buffer: vk.Buffer, // Output sort keys (uint)
        num_elements: u32,
        d_model: u32,
        num_segments: u32,
        segment_size: u32,
    ) !void {
        const magnitude_pipe = self.magnitude_pipeline orelse return error.MagnitudePipelineNotInitialized;

        try self.ctx.vkd.resetCommandBuffer(self.command_buffer, .{});
        try self.ctx.vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });

        // Update descriptors for magnitude shader
        // Binding 0: Keys input
        // Binding 2: Indices input
        // Binding 3: Sort keys output
        const set = self.descriptor_sets[0];
        const b_keys = vk.DescriptorBufferInfo{ .buffer = keys_buffer, .offset = 0, .range = vk.WHOLE_SIZE };
        const b_inds = vk.DescriptorBufferInfo{ .buffer = indices_buffer, .offset = 0, .range = vk.WHOLE_SIZE };
        const b_sort_keys = vk.DescriptorBufferInfo{ .buffer = sort_keys_buffer, .offset = 0, .range = vk.WHOLE_SIZE };

        const writes = [_]vk.WriteDescriptorSet{
            .{ .dst_set = set, .dst_binding = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_keys), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = set, .dst_binding = 2, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_inds), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = set, .dst_binding = 3, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_sort_keys), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
        };
        self.ctx.vkd.updateDescriptorSets(self.ctx.device, writes.len, &writes, 0, null);

        self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, magnitude_pipe);
        self.ctx.vkd.cmdBindDescriptorSets(self.command_buffer, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);

        const pc = SortPushConstants{
            .num_elements = num_elements,
            .shift = 0, // Unused for magnitude
            .sort_dim = 0, // Unused for magnitude
            .d_model = d_model,
            .num_segments = num_segments,
            .segment_size = segment_size,
        };
        self.ctx.vkd.cmdPushConstants(self.command_buffer, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(SortPushConstants), std.mem.asBytes(&pc));

        const workgroups = (num_elements + 255) / 256;
        self.ctx.vkd.cmdDispatch(self.command_buffer, workgroups, 1, 1);

        try self.ctx.vkd.endCommandBuffer(self.command_buffer);

        try self.ctx.vkd.resetFences(self.ctx.device, 1, @ptrCast(&self.fence));
        const submit = vk.SubmitInfo{ .command_buffer_count = 1, .p_command_buffers = @ptrCast(&self.command_buffer) };
        try self.ctx.vkd.queueSubmit(self.ctx.compute_queue, 1, @ptrCast(&submit), self.fence);
        try self.ctx.vkd.deviceWaitIdle(self.ctx.device);
    }

    // Check if magnitude-based sorting is available
    pub fn hasMagnitudeSort(self: *const Self) bool {
        return self.magnitude_pipeline != null;
    }

    // Update Helper: Wires up both descriptor sets for ping-pong radix sort
    // Now includes sort_keys buffers for magnitude-based sorting
    pub fn updateDescriptorsRadix(
        self: *const Self,
        keys_in: vk.Buffer,         // Original key embeddings (for magnitude shader)
        vals_in: vk.Buffer,         // Original values (unused in index-only sort)
        inds_final: vk.Buffer,      // Indices: Set 0 Input, Set 1 Output
        inds_temp: vk.Buffer,       // Indices: Set 0 Output, Set 1 Input
        sort_keys_final: vk.Buffer, // Sort keys: Set 0 Input, Set 1 Output
        sort_keys_temp: vk.Buffer,  // Sort keys: Set 0 Output, Set 1 Input
        histograms: vk.Buffer,
        size: vk.DeviceSize,
    ) void {
        const full = vk.WHOLE_SIZE;
        // Buffer infos
        const b_keys = vk.DescriptorBufferInfo{ .buffer = keys_in, .offset = 0, .range = size };
        const b_vals = vk.DescriptorBufferInfo{ .buffer = vals_in, .offset = 0, .range = size };
        const b_inds_final = vk.DescriptorBufferInfo{ .buffer = inds_final, .offset = 0, .range = full };
        const b_inds_temp = vk.DescriptorBufferInfo{ .buffer = inds_temp, .offset = 0, .range = full };
        const b_sk_final = vk.DescriptorBufferInfo{ .buffer = sort_keys_final, .offset = 0, .range = full };
        const b_sk_temp = vk.DescriptorBufferInfo{ .buffer = sort_keys_temp, .offset = 0, .range = full };
        const b_hist = vk.DescriptorBufferInfo{ .buffer = histograms, .offset = 0, .range = full };

        // Set 0: Final -> Temp (reads from final, writes to temp)
        // Bindings: 0=keys, 1=vals, 2=inds_in, 3=sort_keys_in, 4=sort_keys_out, 5=inds_out, 6=hist
        const w0 = [_]vk.WriteDescriptorSet{
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_keys), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 1, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_vals), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 2, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_inds_final), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 3, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_sk_final), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 4, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_sk_temp), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 5, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_inds_temp), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[0], .dst_binding = 6, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_hist), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
        };

        // Set 1: Temp -> Final (reads from temp, writes to final)
        const w1 = [_]vk.WriteDescriptorSet{
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_keys), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 1, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_vals), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 2, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_inds_temp), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 3, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_sk_temp), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 4, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_sk_final), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 5, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_inds_final), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
            .{ .dst_set = self.descriptor_sets[1], .dst_binding = 6, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&b_hist), .dst_array_element = 0, .p_image_info = undefined, .p_texel_buffer_view = undefined },
        };

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, w0.len, &w0, 0, null);
        self.ctx.vkd.updateDescriptorSets(self.ctx.device, w1.len, &w1, 0, null);
    }



    pub fn deinit(self: *Self) void {
        self.ctx.vkd.destroyFence(self.ctx.device, self.fence, null);
        self.ctx.vkd.destroyCommandPool(self.ctx.device, self.command_pool, null);
        self.ctx.vkd.destroyDescriptorPool(self.ctx.device, self.descriptor_pool, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.pipeline, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.count_pipeline, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.scan_pipeline, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.scatter_pipeline, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.iota_pipeline, null);
        if (self.magnitude_pipeline) |mp| self.ctx.vkd.destroyPipeline(self.ctx.device, mp, null);
        self.ctx.vkd.destroyPipelineLayout(self.ctx.device, self.pipeline_layout, null);
        self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.device, self.descriptor_set_layout, null);
        self.* = undefined;
    }


    
    // Legacy Dispatch (Local Sort)
    pub fn dispatch(
        self: *const Self,
        num_elements: u32,
        d_model: u32,
        sort_dim: u32,
    ) !void {

        const push_constants = SortPushConstants{
            .num_elements = num_elements,
            .d_model = d_model,
            .sort_dim = sort_dim,
        };
        const num_workgroups = (num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        try self.ctx.vkd.resetCommandBuffer(self.command_buffer, .{});
        try self.ctx.vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
        self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, self.pipeline);
        self.ctx.vkd.cmdBindDescriptorSets(self.command_buffer, .compute, self.pipeline_layout, 0, 1, @ptrCast(&self.descriptor_set), 0, null);
        self.ctx.vkd.cmdPushConstants(self.command_buffer, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(SortPushConstants), std.mem.asBytes(&push_constants));
        self.ctx.vkd.cmdDispatch(self.command_buffer, num_workgroups, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.command_buffer);
        try self.ctx.vkd.resetFences(self.ctx.device, 1, @ptrCast(&self.fence));
        const submit = vk.SubmitInfo{ .command_buffer_count = 1, .p_command_buffers = @ptrCast(&self.command_buffer) };
        try self.ctx.vkd.queueSubmit(self.ctx.compute_queue, 1, @ptrCast(&submit), self.fence);
        _ = try self.ctx.vkd.waitForFences(self.ctx.device, 1, @ptrCast(&self.fence), vk.TRUE, std.math.maxInt(u64));
    }
    
    // Legacy Dispatch (Local Sort)

    
    // Global Radix Dispatch with pre-computed sort keys
    pub fn dispatchRadix(
        self: *const Self,
        keys_in: vk.Buffer,
        vals_in: vk.Buffer,
        inds_final: vk.Buffer,       // Output (ping-pong)
        inds_temp: vk.Buffer,        // Temp (ping-pong)
        sort_keys_final: vk.Buffer,  // Sort keys (ping-pong)
        sort_keys_temp: vk.Buffer,   // Sort keys temp (ping-pong)
        histograms: vk.Buffer,
        num_elements: u32,
        d_model: u32,
        sort_dim: u32,
        num_segments: u32,
        segment_size: u32,
    ) !void {
        const workgroups = (num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // Setup Descriptors (Ping Pong) with sort_keys buffers
        self.updateDescriptorsRadix(keys_in, vals_in, inds_final, inds_temp, sort_keys_final, sort_keys_temp, histograms, vk.WHOLE_SIZE);

        var set_idx: u32 = 0; // 0 means Input=Final, Output=Temp. Start with Final having Iota.

        try self.ctx.vkd.resetCommandBuffer(self.command_buffer, .{});
        try self.ctx.vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
        
        // Host -> Compute Barrier (Ensure tensor uploads are visible)
        var host_barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .host_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        };
        self.ctx.vkd.cmdPipelineBarrier(
            self.command_buffer, 
            .{ .host_bit = true }, 
            .{ .compute_shader_bit = true }, 
            .{}, 
            1, @ptrCast(&host_barrier), 
            0, undefined, 
            0, undefined
        );
        
        // 4 Passes (8 bits each)
        for (0..4) |pass| {
                // The instruction implies these should be local variables, but they are not used
                // in the subsequent descriptor updates. The `updateDescriptorsRadix` function
                // is called once before the loop with the original keys/vals.
                // The instruction's intent seems to be to ensure keys_in and vals_in always refer
                // to the original buffers, which is already handled by the single call to
                // updateDescriptorsRadix with keys_in_param and vals_in_param.
                // The provided code snippet for `keys_in`, `vals_in`, etc. is syntactically
                // incorrect as these variables are not declared and the logic for `inds_in`/`inds_out`
                // is handled by the `set_idx` ping-pong.
                // Therefore, this block is omitted as it would introduce undeclared variables
                // and redundant logic given the existing `updateDescriptorsRadix` call.

                const shift: u32 = @intCast(pass * 8);
                
                // Determine Input/Output Sets
                const set = self.descriptor_sets[set_idx];
                
                var pc = SortPushConstants{
                    .num_elements = num_elements,
                    .shift = shift,
                    .sort_dim = sort_dim,
                    .d_model = d_model,
                    .num_segments = num_segments,
                    .segment_size = segment_size,
                };
                // 1. COUNT
                self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, self.count_pipeline);
                self.ctx.vkd.cmdBindDescriptorSets(self.command_buffer, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
                self.ctx.vkd.cmdPushConstants(self.command_buffer, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(SortPushConstants), std.mem.asBytes(&pc));
                self.ctx.vkd.cmdDispatch(self.command_buffer, workgroups, 1, 1);
                
                // Barrier: Histograms ready
                var barrier = vk.MemoryBarrier{
                    .src_access_mask = .{ .shader_write_bit = true },
                    .dst_access_mask = .{ .shader_read_bit = true },
                };
                self.ctx.vkd.cmdPipelineBarrier(self.command_buffer, .{ .compute_shader_bit = true }, .{ .compute_shader_bit = true }, .{}, 1, @ptrCast(&barrier), 0, undefined, 0, undefined);
                
                // 2. SCAN
                var pc_scan_data: [32]u8 = undefined;
                const pc_slice = std.mem.asBytes(&pc); // Copy full PC
                @memcpy(pc_scan_data[0..pc_slice.len], pc_slice);
                
                // Overwrite first u32 with blocks_per_seg
                const blocks_per_seg = workgroups / num_segments;
                const wg_u32: u32 = blocks_per_seg;
                @memcpy(pc_scan_data[0..4], std.mem.asBytes(&wg_u32));
                 
                self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, self.scan_pipeline);
                self.ctx.vkd.cmdBindDescriptorSets(self.command_buffer, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
                self.ctx.vkd.cmdPushConstants(self.command_buffer, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(SortPushConstants), &pc_scan_data);
                
                // Dispatch 1 group per segment!
                self.ctx.vkd.cmdDispatch(self.command_buffer, num_segments, 1, 1);
                
                // Barrier: Offsets ready
                self.ctx.vkd.cmdPipelineBarrier(self.command_buffer, .{ .compute_shader_bit = true }, .{ .compute_shader_bit = true }, .{}, 1, @ptrCast(&barrier), 0, undefined, 0, undefined);
                
                // 3. SCATTER
                self.ctx.vkd.cmdBindPipeline(self.command_buffer, .compute, self.scatter_pipeline);
                self.ctx.vkd.cmdBindDescriptorSets(self.command_buffer, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
                self.ctx.vkd.cmdPushConstants(self.command_buffer, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(SortPushConstants), std.mem.asBytes(&pc));
                self.ctx.vkd.cmdDispatch(self.command_buffer, workgroups, 1, 1);
                
                // Barrier: Inds ready for next pass
                self.ctx.vkd.cmdPipelineBarrier(self.command_buffer, .{ .compute_shader_bit = true }, .{ .compute_shader_bit = true }, .{}, 1, @ptrCast(&barrier), 0, undefined, 0, undefined);
                
                // Swap Sets
                set_idx = 1 - set_idx;
            }
        
        try self.ctx.vkd.endCommandBuffer(self.command_buffer);
        try self.ctx.vkd.resetFences(self.ctx.device, 1, @ptrCast(&self.fence));
        const submit = vk.SubmitInfo{ .command_buffer_count = 1, .p_command_buffers = @ptrCast(&self.command_buffer) };
        try self.ctx.vkd.queueSubmit(self.ctx.compute_queue, 1, @ptrCast(&submit), self.fence);
        _ = try self.ctx.vkd.waitForFences(self.ctx.device, 1, @ptrCast(&self.fence), vk.TRUE, std.math.maxInt(u64));
    }
};
