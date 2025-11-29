const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const InstanceDispatch = @import("vulkan_context.zig").InstanceDispatchType;

const log = std.log.scoped(.buffer_manager);

pub const Buffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    mapped: ?*anyopaque,

    const Self = @This();

    pub fn getMappedSlice(self: *const Self, comptime T: type) []T {
        if (self.mapped) |ptr| {
            const count = @divExact(self.size, @sizeOf(T));
            return @as([*]T, @ptrCast(@alignCast(ptr)))[0..count];
        }
        return &[_]T{};
    }
};

pub const BufferManager = struct {
    ctx: *const VulkanContext,
    memory_properties: vk.PhysicalDeviceMemoryProperties,

    const Self = @This();

    pub fn init(ctx: *const VulkanContext) Self {
        const memory_properties = ctx.vki.getPhysicalDeviceMemoryProperties(ctx.physical_device);
        return Self{
            .ctx = ctx,
            .memory_properties = memory_properties,
        };
    }

    pub fn createBuffer(
        self: *const Self,
        size: vk.DeviceSize,
        usage: vk.BufferUsageFlags,
        memory_flags: vk.MemoryPropertyFlags,
    ) !Buffer {
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = null,
        }, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        const mem_requirements = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const memory_type_index = try self.findMemoryType(mem_requirements.memory_type_bits, memory_flags);

        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &.{
            .allocation_size = mem_requirements.size,
            .memory_type_index = memory_type_index,
        }, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);

        // Map if host visible
        var mapped: ?*anyopaque = null;
        if (memory_flags.host_visible_bit) {
            mapped = try self.ctx.vkd.mapMemory(self.ctx.device, memory, 0, size, .{});
        }

        return Buffer{
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .mapped = mapped,
        };
    }

    pub fn destroyBuffer(self: *const Self, buffer: *Buffer) void {
        if (buffer.mapped != null) {
            self.ctx.vkd.unmapMemory(self.ctx.device, buffer.memory);
        }
        self.ctx.vkd.destroyBuffer(self.ctx.device, buffer.buffer, null);
        self.ctx.vkd.freeMemory(self.ctx.device, buffer.memory, null);
        buffer.* = undefined;
    }

    pub fn createDeviceLocalBuffer(self: *const Self, size: vk.DeviceSize) !Buffer {
        return self.createBuffer(
            size,
            .{ .storage_buffer_bit = true, .transfer_dst_bit = true, .transfer_src_bit = true },
            .{ .device_local_bit = true },
        );
    }

    pub fn createStagingBuffer(self: *const Self, size: vk.DeviceSize) !Buffer {
        return self.createBuffer(
            size,
            .{ .transfer_src_bit = true, .transfer_dst_bit = true },
            .{ .host_visible_bit = true, .host_coherent_bit = true },
        );
    }

    // Host-visible storage buffer - works well on integrated GPUs
    pub fn createHostVisibleStorageBuffer(self: *const Self, size: vk.DeviceSize) !Buffer {
        return self.createBuffer(
            size,
            .{ .storage_buffer_bit = true },
            .{ .host_visible_bit = true, .host_coherent_bit = true },
        );
    }

    pub fn copyBuffer(
        self: *const Self,
        command_buffer: vk.CommandBuffer,
        src: vk.Buffer,
        dst: vk.Buffer,
        size: vk.DeviceSize,
    ) void {
        const region = vk.BufferCopy{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        };
        self.ctx.vkd.cmdCopyBuffer(command_buffer, src, dst, 1, @ptrCast(&region));
    }

    fn findMemoryType(self: *const Self, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        for (0..self.memory_properties.memory_type_count) |i| {
            const idx: u5 = @intCast(i);
            if ((type_filter & (@as(u32, 1) << idx)) != 0) {
                const mem_type = self.memory_properties.memory_types[i];
                if (mem_type.property_flags.contains(properties)) {
                    return @intCast(i);
                }
            }
        }
        log.err("Failed to find suitable memory type", .{});
        return error.NoSuitableMemoryType;
    }
};
