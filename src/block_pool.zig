const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const BufferManager = @import("buffer_manager.zig").BufferManager;
const Buffer = @import("buffer_manager.zig").Buffer;

const log = std.log.scoped(.block_pool);

pub const BlockPoolConfig = struct {
    initial_blocks: u32 = 512, // Start with 512 blocks
    blocks_per_chunk: u32 = 512, // Grow by 512 blocks
    max_blocks: u32 = 8192, // Max 256K tokens
    block_size: u32 = 32, // 32 tokens per block
    num_kv_heads: u32, // From model config
    head_dim: u32, // Typically 64
};

pub const BlockPool = struct {
    // GPU storage for all blocks
    kv_pool_buffer: Buffer,

    // Free block tracking
    free_blocks: std.ArrayList(u32),
    total_blocks: u32,
    config: BlockPoolConfig,

    // References
    buffer_manager: *const BufferManager,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        buffer_manager: *const BufferManager,
        config: BlockPoolConfig,
    ) !Self {
        log.info("Initializing BlockPool: {} initial blocks, {} max", .{
            config.initial_blocks,
            config.max_blocks,
        });

        // Allocate initial pool: [initial_blocks, 2, num_kv_heads, 32, head_dim]
        const kv_pool_size = config.initial_blocks * 2 * config.num_kv_heads *
            config.block_size * config.head_dim * @sizeOf(f32);

        const kv_pool_buffer = try buffer_manager.createDeviceLocalBuffer(kv_pool_size);

        // Initialize free list with all blocks
        var free_blocks = std.ArrayList(u32).init(allocator);
        try free_blocks.ensureTotalCapacity(config.initial_blocks);

        var i: u32 = 0;
        while (i < config.initial_blocks) : (i += 1) {
            try free_blocks.append(i);
        }

        return Self{
            .kv_pool_buffer = kv_pool_buffer,
            .free_blocks = free_blocks,
            .total_blocks = config.initial_blocks,
            .config = config,
            .buffer_manager = buffer_manager,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var buf = self.kv_pool_buffer;
        self.buffer_manager.destroyBuffer(&buf);
        self.free_blocks.deinit();
        self.* = undefined;
    }

    pub fn allocateBlock(self: *Self) !u32 {
        if (self.free_blocks.items.len == 0) {
            // Out of free blocks, try to grow
            try self.growPool();
        }

        if (self.free_blocks.items.len == 0) {
            log.err("Block pool exhausted (max {} blocks)", .{self.config.max_blocks});
            return error.BlockPoolExhausted;
        }

        const block_id = self.free_blocks.pop();
        log.debug("Allocated block {}, {} free remaining", .{ block_id, self.free_blocks.items.len });
        return block_id;
    }

    pub fn freeBlock(self: *Self, block_id: u32) void {
        std.debug.assert(block_id < self.total_blocks);
        self.free_blocks.append(block_id) catch {
            log.err("Failed to free block {}", .{block_id});
            return;
        };
        log.debug("Freed block {}, {} free total", .{ block_id, self.free_blocks.items.len });
    }

    pub fn growPool(self: *Self) !void {
        const new_total = self.total_blocks + self.config.blocks_per_chunk;

        if (new_total > self.config.max_blocks) {
            log.warn("Cannot grow pool beyond max {} blocks", .{self.config.max_blocks});
            return error.MaxBlocksReached;
        }

        log.info("Growing block pool: {} -> {} blocks", .{ self.total_blocks, new_total });

        // Allocate new larger buffer
        const new_size = new_total * 2 * self.config.num_kv_heads *
            self.config.block_size * self.config.head_dim * @sizeOf(f32);

        const new_buffer = try self.buffer_manager.createDeviceLocalBuffer(new_size);

        // TODO: Copy old data to new buffer using vkCmdCopyBuffer
        // For MVP, we don't preserve data (allocations happen at start)

        // Free old buffer
        var old_buf = self.kv_pool_buffer;
        self.buffer_manager.destroyBuffer(&old_buf);

        // Update state
        self.kv_pool_buffer = new_buffer;

        // Add new blocks to free list
        const old_total = self.total_blocks;
        self.total_blocks = new_total;

        var i: u32 = old_total;
        while (i < new_total) : (i += 1) {
            try self.free_blocks.append(i);
        }

        log.info("Block pool grown successfully, {} free blocks", .{self.free_blocks.items.len});
    }

    pub fn getBuffer(self: *const Self) vk.Buffer {
        return self.kv_pool_buffer.buffer;
    }
};
