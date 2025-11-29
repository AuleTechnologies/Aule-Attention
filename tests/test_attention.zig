const std = @import("std");
const aule = @import("aule");

const Attention = aule.Attention;
const AttentionRef = aule.AttentionRef;
const attention_ref = @import("aule").AttentionRef;

/// Test configuration
const TestConfig = struct {
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    name: []const u8,
};

/// Test cases covering various dimensions (head_dim <= 64 for current shader)
const test_configs = [_]TestConfig{
    // Small tests for quick validation
    .{ .batch_size = 1, .num_heads = 1, .seq_len = 4, .head_dim = 8, .name = "tiny" },
    .{ .batch_size = 1, .num_heads = 1, .seq_len = 16, .head_dim = 16, .name = "small" },
    .{ .batch_size = 1, .num_heads = 2, .seq_len = 16, .head_dim = 32, .name = "small_multihead" },

    // Medium tests
    .{ .batch_size = 1, .num_heads = 4, .seq_len = 32, .head_dim = 64, .name = "medium" },
    .{ .batch_size = 2, .num_heads = 4, .seq_len = 32, .head_dim = 64, .name = "medium_batch" },

    // Larger tests (still reasonable for integrated GPU)
    .{ .batch_size = 1, .num_heads = 8, .seq_len = 64, .head_dim = 64, .name = "large" },
    .{ .batch_size = 2, .num_heads = 8, .seq_len = 64, .head_dim = 64, .name = "large_batch" },
};

/// Initialize random tensors with normal-ish distribution
fn initRandomTensor(buffer: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (buffer) |*val| {
        // Generate roughly normal values in [-1, 1]
        val.* = (random.float(f32) * 2.0 - 1.0) * 0.5;
    }
}

/// Calculate statistics for comparing outputs
fn compareOutputs(expected: []const f32, actual: []const f32) struct {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    max_rel_diff: f32,
    all_close: bool,
} {
    var max_abs: f32 = 0;
    var sum_abs: f32 = 0;
    var max_rel: f32 = 0;

    for (expected, actual) |e, a| {
        const abs_diff = @abs(e - a);
        max_abs = @max(max_abs, abs_diff);
        sum_abs += abs_diff;

        // Relative difference (avoiding division by zero)
        const denom = @max(@abs(e), 1e-6);
        const rel_diff = abs_diff / denom;
        max_rel = @max(max_rel, rel_diff);
    }

    const mean_abs = sum_abs / @as(f32, @floatFromInt(expected.len));

    // fp32 tolerance: allow small numerical differences
    const abs_tol: f32 = 1e-4;
    const rel_tol: f32 = 1e-3;

    return .{
        .max_abs_diff = max_abs,
        .mean_abs_diff = mean_abs,
        .max_rel_diff = max_rel,
        .all_close = max_abs < abs_tol or max_rel < rel_tol,
    };
}

test "attention_gpu_vs_cpu_reference" {
    const allocator = std.testing.allocator;

    // Initialize GPU attention
    var gpu_attention = Attention.init(allocator) catch |err| {
        std.debug.print("Skipping GPU test - Vulkan init failed: {}\n", .{err});
        return;
    };
    defer gpu_attention.deinit();

    // Initialize CPU reference
    const cpu_ref = AttentionRef.init(allocator);

    for (test_configs) |config| {
        std.debug.print("\nTesting config: {s} (B={}, H={}, N={}, D={})\n", .{
            config.name,
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
        });

        const total_size = config.batch_size * config.num_heads * config.seq_len * config.head_dim;

        // Allocate tensors
        const Q = try allocator.alloc(f32, total_size);
        defer allocator.free(Q);
        const K = try allocator.alloc(f32, total_size);
        defer allocator.free(K);
        const V = try allocator.alloc(f32, total_size);
        defer allocator.free(V);
        const gpu_output = try allocator.alloc(f32, total_size);
        defer allocator.free(gpu_output);
        const cpu_output = try allocator.alloc(f32, total_size);
        defer allocator.free(cpu_output);

        // Initialize with same random data
        initRandomTensor(Q, 42);
        initRandomTensor(K, 123);
        initRandomTensor(V, 456);

        // Run CPU reference
        try cpu_ref.forward(
            Q,
            K,
            V,
            cpu_output,
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
        );

        // Run GPU implementation
        try gpu_attention.forward(
            Q,
            K,
            V,
            gpu_output,
            @intCast(config.batch_size),
            @intCast(config.num_heads),
            @intCast(config.seq_len),
            @intCast(config.head_dim),
        );

        // Compare results
        const stats = compareOutputs(cpu_output, gpu_output);

        std.debug.print("  Max abs diff: {e:.6}\n", .{stats.max_abs_diff});
        std.debug.print("  Mean abs diff: {e:.6}\n", .{stats.mean_abs_diff});
        std.debug.print("  Max rel diff: {e:.6}\n", .{stats.max_rel_diff});
        std.debug.print("  All close: {}\n", .{stats.all_close});

        // Assert results are close enough
        try std.testing.expect(stats.all_close);
    }
}

test "attention_uniform_weights" {
    // When Q=K, attention should produce uniform weights
    // Output should be the average of V across the sequence
    const allocator = std.testing.allocator;

    var gpu_attention = Attention.init(allocator) catch |err| {
        std.debug.print("Skipping GPU test - Vulkan init failed: {}\n", .{err});
        return;
    };
    defer gpu_attention.deinit();

    const batch_size: u32 = 1;
    const num_heads: u32 = 1;
    const seq_len: u32 = 4;
    const head_dim: u32 = 8;
    const total_size: usize = batch_size * num_heads * seq_len * head_dim;

    const Q = try allocator.alloc(f32, total_size);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, total_size);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, total_size);
    defer allocator.free(V);
    const output = try allocator.alloc(f32, total_size);
    defer allocator.free(output);

    // Set Q = K (uniform attention)
    @memset(Q, 0.5);
    @memset(K, 0.5);

    // V has distinct values per position
    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            V[i * head_dim + d] = @as(f32, @floatFromInt(i * head_dim + d));
        }
    }

    try gpu_attention.forward(Q, K, V, output, batch_size, num_heads, seq_len, head_dim);

    // Compute expected average
    var expected_avg: [8]f32 = undefined;
    for (0..head_dim) |d| {
        var sum: f32 = 0;
        for (0..seq_len) |i| {
            sum += V[i * head_dim + d];
        }
        expected_avg[d] = sum / @as(f32, @floatFromInt(seq_len));
    }

    // All output rows should be approximately the average
    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            const actual = output[i * head_dim + d];
            const expected = expected_avg[d];
            const diff = @abs(actual - expected);
            if (diff > 0.01) {
                std.debug.print("Mismatch at [{}, {}]: expected {}, got {}\n", .{ i, d, expected, actual });
            }
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "attention_identity_kv" {
    // When each position only attends to itself (large diagonal in attention matrix),
    // output should approximate V
    const allocator = std.testing.allocator;

    var gpu_attention = Attention.init(allocator) catch |err| {
        std.debug.print("Skipping GPU test - Vulkan init failed: {}\n", .{err});
        return;
    };
    defer gpu_attention.deinit();

    const batch_size: u32 = 1;
    const num_heads: u32 = 1;
    const seq_len: u32 = 4;
    const head_dim: u32 = 8;
    const total_size: usize = batch_size * num_heads * seq_len * head_dim;

    const Q = try allocator.alloc(f32, total_size);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, total_size);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, total_size);
    defer allocator.free(V);
    const output = try allocator.alloc(f32, total_size);
    defer allocator.free(output);

    // Make Q[i] and K[i] have high dot product, Q[i] and K[j] (i!=j) have low
    @memset(Q, 0);
    @memset(K, 0);

    for (0..seq_len) |i| {
        // Position i has 1.0 in dimension i (one-hot like)
        Q[i * head_dim + i] = 10.0; // Large value to dominate softmax
        K[i * head_dim + i] = 10.0;
    }

    // V has distinct values
    for (0..total_size) |i| {
        V[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    try gpu_attention.forward(Q, K, V, output, batch_size, num_heads, seq_len, head_dim);

    // Output should be close to V (since attention is nearly diagonal)
    for (0..total_size) |i| {
        const diff = @abs(output[i] - V[i]);
        // Allow some tolerance since softmax won't be exactly 1.0
        try std.testing.expect(diff < 0.1);
    }
}

test "attention_numerical_stability" {
    // Test with large values to verify numerical stability
    const allocator = std.testing.allocator;

    var gpu_attention = Attention.init(allocator) catch |err| {
        std.debug.print("Skipping GPU test - Vulkan init failed: {}\n", .{err});
        return;
    };
    defer gpu_attention.deinit();

    const cpu_ref = AttentionRef.init(allocator);

    const batch_size: u32 = 1;
    const num_heads: u32 = 1;
    const seq_len: u32 = 16;
    const head_dim: u32 = 16;
    const total_size: usize = batch_size * num_heads * seq_len * head_dim;

    const Q = try allocator.alloc(f32, total_size);
    defer allocator.free(Q);
    const K = try allocator.alloc(f32, total_size);
    defer allocator.free(K);
    const V = try allocator.alloc(f32, total_size);
    defer allocator.free(V);
    const gpu_output = try allocator.alloc(f32, total_size);
    defer allocator.free(gpu_output);
    const cpu_output = try allocator.alloc(f32, total_size);
    defer allocator.free(cpu_output);

    // Use larger values that could cause overflow without proper softmax implementation
    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    for (0..total_size) |i| {
        Q[i] = (random.float(f32) * 2.0 - 1.0) * 5.0; // Values in [-5, 5]
        K[i] = (random.float(f32) * 2.0 - 1.0) * 5.0;
        V[i] = (random.float(f32) * 2.0 - 1.0) * 2.0;
    }

    try cpu_ref.forward(Q, K, V, cpu_output, batch_size, num_heads, seq_len, head_dim);
    try gpu_attention.forward(Q, K, V, gpu_output, batch_size, num_heads, seq_len, head_dim);

    // Verify no NaN or Inf
    for (gpu_output) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    const stats = compareOutputs(cpu_output, gpu_output);
    std.debug.print("Numerical stability test: max_abs={e:.6}, max_rel={e:.6}\n", .{
        stats.max_abs_diff,
        stats.max_rel_diff,
    });
}

test "attention_batch_independence" {
    // Verify that batches don't interfere with each other
    const allocator = std.testing.allocator;

    var gpu_attention = Attention.init(allocator) catch |err| {
        std.debug.print("Skipping GPU test - Vulkan init failed: {}\n", .{err});
        return;
    };
    defer gpu_attention.deinit();

    const num_heads: u32 = 2;
    const seq_len: u32 = 8;
    const head_dim: u32 = 16;
    const single_size: usize = num_heads * seq_len * head_dim;

    // Run single batch
    const Q1 = try allocator.alloc(f32, single_size);
    defer allocator.free(Q1);
    const K1 = try allocator.alloc(f32, single_size);
    defer allocator.free(K1);
    const V1 = try allocator.alloc(f32, single_size);
    defer allocator.free(V1);
    const output1 = try allocator.alloc(f32, single_size);
    defer allocator.free(output1);

    initRandomTensor(Q1, 111);
    initRandomTensor(K1, 222);
    initRandomTensor(V1, 333);

    try gpu_attention.forward(Q1, K1, V1, output1, 1, num_heads, seq_len, head_dim);

    // Run as batch of 2 with different second batch
    var Q2 = try allocator.alloc(f32, single_size * 2);
    defer allocator.free(Q2);
    var K2 = try allocator.alloc(f32, single_size * 2);
    defer allocator.free(K2);
    var V2 = try allocator.alloc(f32, single_size * 2);
    defer allocator.free(V2);
    var output2 = try allocator.alloc(f32, single_size * 2);
    defer allocator.free(output2);

    // First batch same as before
    @memcpy(Q2[0..single_size], Q1);
    @memcpy(K2[0..single_size], K1);
    @memcpy(V2[0..single_size], V1);

    // Second batch different
    initRandomTensor(Q2[single_size..], 444);
    initRandomTensor(K2[single_size..], 555);
    initRandomTensor(V2[single_size..], 666);

    try gpu_attention.forward(Q2, K2, V2, output2, 2, num_heads, seq_len, head_dim);

    // First batch output should match single batch output
    const stats = compareOutputs(output1, output2[0..single_size]);
    std.debug.print("Batch independence: max_diff={e:.6}\n", .{stats.max_abs_diff});
    try std.testing.expect(stats.max_abs_diff < 1e-5);
}
