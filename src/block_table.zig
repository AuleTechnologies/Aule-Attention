const std = @import("std");
const vk = @import("vulkan");
const BufferManager = @import("buffer_manager.zig").BufferManager;
const Buffer = @import("buffer_manager.zig").Buffer;
const VulkanContext = @import("vulkan_context.zig").VulkanContext;

const log = std.log.scoped(.block_table);

pub const BlockTable = struct {
    // GPU-side storage
    table_buffer: Buffer, // Device-local SSBO
    staging_buffer: Buffer, // Host-visible for uploads

    // CPU-side mirror
    table_cpu: []i32, // [batch_size * max_blocks]

    batch_size: u32,
    max_blocks: u32,

    // References
    ctx: *const VulkanContext,
    buffer_manager: *const BufferManager,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *const VulkanContext,
        buffer_manager: *const BufferManager,
        batch_size: u32,
        max_blocks: u32,
    ) !Self {
        log.info("Creating BlockTable: batch={}, max_blocks={}", .{ batch_size, max_blocks });

        const total_entries = batch_size * max_blocks;
        const byte_size = total_entries * @sizeOf(i32);

        // Allocate CPU-side table
        const table_cpu = try allocator.alloc(i32, total_entries);
        errdefer allocator.free(table_cpu);

        // Initialize with -1 (sentinel)
        @memset(table_cpu, -1);

        // Allocate GPU buffers
        const table_buffer = try buffer_manager.createDeviceLocalBuffer(byte_size);
        errdefer {
            var buf = table_buffer;
            buffer_manager.destroyBuffer(&buf);
        }

        const staging_buffer = try buffer_manager.createStagingBuffer(byte_size);
        errdefer {
            var buf = staging_buffer;
            buffer_manager.destroyBuffer(&buf);
        }

        return Self{
            .table_buffer = table_buffer,
            .staging_buffer = staging_buffer,
            .table_cpu = table_cpu,
            .batch_size = batch_size,
            .max_blocks = max_blocks,
            .ctx = ctx,
            .buffer_manager = buffer_manager,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.table_cpu);
        var buf1 = self.table_buffer;
        self.buffer_manager.destroyBuffer(&buf1);
        var buf2 = self.staging_buffer;
        self.buffer_manager.destroyBuffer(&buf2);
        self.* = undefined;
    }

    pub fn set(self: *Self, request_id: u32, logical_block: u32, physical_block: i32) void {
        std.debug.assert(request_id < self.batch_size);
        std.debug.assert(logical_block < self.max_blocks);

        const idx = request_id * self.max_blocks + logical_block;
        self.table_cpu[idx] = physical_block;

        log.debug("BlockTable[{}][{}] = {}", .{ request_id, logical_block, physical_block });
    }

    pub fn get(self: *const Self, request_id: u32, logical_block: u32) i32 {
        std.debug.assert(request_id < self.batch_size);
        std.debug.assert(logical_block < self.max_blocks);

        const idx = request_id * self.max_blocks + logical_block;
        return self.table_cpu[idx];
    }

    pub fn sync(self: *Self) !void {
        log.debug("Syncing BlockTable to GPU ({} entries)", .{self.table_cpu.len});

        // Copy CPU table to staging buffer
        const staging_slice = self.staging_buffer.getMappedSlice(i32);
        @memcpy(staging_slice, self.table_cpu);

        // For MVP, we use staging buffer directly in shader
        // Production version would use vkCmdCopyBuffer to device-local buffer
    }

    pub fn getBuffer(self: *const Self) vk.Buffer {
        return self.table_buffer.buffer;
    }

    pub fn getStagingBuffer(self: *const Self) vk.Buffer {
        return self.staging_buffer.buffer;
    }
};
