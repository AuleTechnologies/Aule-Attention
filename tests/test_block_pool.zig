const std = @import("std");
const testing = std.testing;
const BlockPool = @import("block_pool").BlockPool;
const BlockPoolConfig = @import("block_pool").BlockPoolConfig;
const BlockTable = @import("block_table").BlockTable;
const VulkanContext = @import("vulkan_context").VulkanContext;
const BufferManager = @import("buffer_manager").BufferManager;

test "BlockPool: basic allocation and deallocation" {
    const allocator = testing.allocator;

    var ctx = try VulkanContext.init(allocator);
    defer ctx.deinit();

    const buffer_manager = BufferManager.init(&ctx);

    var config = BlockPoolConfig{
        .initial_blocks = 512,
        .blocks_per_chunk = 512,
        .max_blocks = 2048,
        .block_size = 32,
        .num_kv_heads = 8,
        .head_dim = 64,
    };

    var pool = try BlockPool.init(allocator, &buffer_manager, config);
    defer pool.deinit();

    // Allocate 100 blocks
    var blocks: [100]u32 = undefined;
    for (&blocks) |*b| {
        b.* = try pool.allocateBlock();
    }

    // Verify we allocated 100 blocks
    try testing.expectEqual(@as(usize, 412), pool.free_blocks.items.len);

    // Free all blocks
    for (blocks) |b| {
        pool.freeBlock(b);
    }

    // Verify free list restored
    try testing.expectEqual(@as(usize, 512), pool.free_blocks.items.len);
}

test "BlockPool: growth when exhausted" {
    const allocator = testing.allocator;

    var ctx = try VulkanContext.init(allocator);
    defer ctx.deinit();

    const buffer_manager = BufferManager.init(&ctx);

    var config = BlockPoolConfig{
        .initial_blocks = 512,
        .blocks_per_chunk = 512,
        .max_blocks = 2048,
        .block_size = 32,
        .num_kv_heads = 8,
        .head_dim = 64,
    };

    var pool = try BlockPool.init(allocator, &buffer_manager, config);
    defer pool.deinit();

    // Allocate beyond initial capacity
    var blocks = std.ArrayList(u32).init(allocator);
    defer blocks.deinit();

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const block = try pool.allocateBlock();
        try blocks.append(block);
    }

    // Verify pool grew
    try testing.expect(pool.total_blocks > 512);
    try testing.expectEqual(@as(u32, 1024), pool.total_blocks);

    // Free all
    for (blocks.items) |b| {
        pool.freeBlock(b);
    }
}

test "BlockPool: max blocks limit" {
    const allocator = testing.allocator;

    var ctx = try VulkanContext.init(allocator);
    defer ctx.deinit();

    const buffer_manager = BufferManager.init(&ctx);

    var config = BlockPoolConfig{
        .initial_blocks = 512,
        .blocks_per_chunk = 512,
        .max_blocks = 1024, // Low limit for test
        .block_size = 32,
        .num_kv_heads = 8,
        .head_dim = 64,
    };

    var pool = try BlockPool.init(allocator, &buffer_manager, config);
    defer pool.deinit();

    // Allocate up to max
    var blocks = std.ArrayList(u32).init(allocator);
    defer blocks.deinit();

    var i: usize = 0;
    while (i < 1024) : (i += 1) {
        const block = try pool.allocateBlock();
        try blocks.append(block);
    }

    // Next allocation should fail
    try testing.expectError(error.BlockPoolExhausted, pool.allocateBlock());

    // Free all
    for (blocks.items) |b| {
        pool.freeBlock(b);
    }
}

test "BlockTable: basic indexing" {
    const allocator = testing.allocator;

    var ctx = try VulkanContext.init(allocator);
    defer ctx.deinit();

    const buffer_manager = BufferManager.init(&ctx);

    var table = try BlockTable.init(allocator, &ctx, &buffer_manager, 4, 64);
    defer table.deinit();

    // Set some entries
    table.set(0, 5, 123);
    table.set(1, 10, 456);
    table.set(3, 63, 789);

    // Verify
    try testing.expectEqual(@as(i32, 123), table.get(0, 5));
    try testing.expectEqual(@as(i32, 456), table.get(1, 10));
    try testing.expectEqual(@as(i32, 789), table.get(3, 63));

    // Unset entries should be -1
    try testing.expectEqual(@as(i32, -1), table.get(0, 0));
    try testing.expectEqual(@as(i32, -1), table.get(2, 20));
}
