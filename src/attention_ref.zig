const std = @import("std");
const math = std.math;

/// CPU reference implementation of attention for testing
/// Implements standard scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
pub const AttentionRef = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    /// Compute attention forward pass
    /// Q, K, V: [batch_size, num_heads, seq_len, head_dim] in row-major order
    /// Output: [batch_size, num_heads, seq_len, head_dim]
    pub fn forward(
        self: *const Self,
        Q: []const f32,
        K: []const f32,
        V: []const f32,
        output: []f32,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) !void {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Allocate scratch space for attention scores
        const scores = try self.allocator.alloc(f32, seq_len * seq_len);
        defer self.allocator.free(scores);

        // Process each batch and head independently
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                const base_offset = (b * num_heads + h) * seq_len * head_dim;

                // Step 1: Compute attention scores S = Q @ K^T
                // S[i,j] = sum_d Q[i,d] * K[j,d]
                for (0..seq_len) |i| {
                    for (0..seq_len) |j| {
                        var dot: f32 = 0.0;
                        for (0..head_dim) |d| {
                            const q_val = Q[base_offset + i * head_dim + d];
                            const k_val = K[base_offset + j * head_dim + d];
                            dot += q_val * k_val;
                        }
                        scores[i * seq_len + j] = dot * scale;
                    }
                }

                // Step 2: Apply softmax row-wise
                for (0..seq_len) |i| {
                    const row_start = i * seq_len;

                    // Find max for numerical stability
                    var row_max: f32 = -math.inf(f32);
                    for (0..seq_len) |j| {
                        row_max = @max(row_max, scores[row_start + j]);
                    }

                    // Compute exp and sum
                    var row_sum: f32 = 0.0;
                    for (0..seq_len) |j| {
                        const exp_val = @exp(scores[row_start + j] - row_max);
                        scores[row_start + j] = exp_val;
                        row_sum += exp_val;
                    }

                    // Normalize
                    const inv_sum = 1.0 / row_sum;
                    for (0..seq_len) |j| {
                        scores[row_start + j] *= inv_sum;
                    }
                }

                // Step 3: Compute output O = softmax(S) @ V
                for (0..seq_len) |i| {
                    for (0..head_dim) |d| {
                        var acc: f32 = 0.0;
                        for (0..seq_len) |j| {
                            const attn_weight = scores[i * seq_len + j];
                            const v_val = V[base_offset + j * head_dim + d];
                            acc += attn_weight * v_val;
                        }
                        output[base_offset + i * head_dim + d] = acc;
                    }
                }
            }
        }
    }

    /// Compute attention with causal mask (for autoregressive models)
    /// Position i can only attend to positions 0..i
    pub fn forwardCausal(
        self: *const Self,
        Q: []const f32,
        K: []const f32,
        V: []const f32,
        output: []f32,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) !void {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        const scores = try self.allocator.alloc(f32, seq_len * seq_len);
        defer self.allocator.free(scores);

        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                const base_offset = (b * num_heads + h) * seq_len * head_dim;

                // Compute scores with causal mask
                for (0..seq_len) |i| {
                    for (0..seq_len) |j| {
                        if (j > i) {
                            // Future position - mask out
                            scores[i * seq_len + j] = -math.inf(f32);
                        } else {
                            var dot: f32 = 0.0;
                            for (0..head_dim) |d| {
                                const q_val = Q[base_offset + i * head_dim + d];
                                const k_val = K[base_offset + j * head_dim + d];
                                dot += q_val * k_val;
                            }
                            scores[i * seq_len + j] = dot * scale;
                        }
                    }
                }

                // Softmax (same as non-causal)
                for (0..seq_len) |i| {
                    const row_start = i * seq_len;

                    var row_max: f32 = -math.inf(f32);
                    for (0..seq_len) |j| {
                        row_max = @max(row_max, scores[row_start + j]);
                    }

                    var row_sum: f32 = 0.0;
                    for (0..seq_len) |j| {
                        const exp_val = @exp(scores[row_start + j] - row_max);
                        scores[row_start + j] = exp_val;
                        row_sum += exp_val;
                    }

                    const inv_sum = 1.0 / row_sum;
                    for (0..seq_len) |j| {
                        scores[row_start + j] *= inv_sum;
                    }
                }

                // Output
                for (0..seq_len) |i| {
                    for (0..head_dim) |d| {
                        var acc: f32 = 0.0;
                        for (0..seq_len) |j| {
                            const attn_weight = scores[i * seq_len + j];
                            const v_val = V[base_offset + j * head_dim + d];
                            acc += attn_weight * v_val;
                        }
                        output[base_offset + i * head_dim + d] = acc;
                    }
                }
            }
        }
    }
};

/// Helper to compare two float arrays with tolerance
pub fn compareArrays(expected: []const f32, actual: []const f32, tolerance: f32) bool {
    if (expected.len != actual.len) return false;

    for (expected, actual) |e, a| {
        if (@abs(e - a) > tolerance) {
            return false;
        }
    }
    return true;
}

/// Calculate max absolute difference between arrays
pub fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return math.inf(f32);

    var max_diff: f32 = 0.0;
    for (a, b) |av, bv| {
        max_diff = @max(max_diff, @abs(av - bv));
    }
    return max_diff;
}

/// Calculate mean absolute difference
pub fn meanAbsDiff(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return math.inf(f32);

    var sum: f32 = 0.0;
    for (a, b) |av, bv| {
        sum += @abs(av - bv);
    }
    return sum / @as(f32, @floatFromInt(a.len));
}

test "attention_ref basic" {
    const allocator = std.testing.allocator;
    const ref = AttentionRef.init(allocator);

    // Simple test: batch=1, heads=1, seq=4, dim=8
    const batch_size: usize = 1;
    const num_heads: usize = 1;
    const seq_len: usize = 4;
    const head_dim: usize = 8;
    const total_size = batch_size * num_heads * seq_len * head_dim;

    var Q = try allocator.alloc(f32, total_size);
    defer allocator.free(Q);
    var K = try allocator.alloc(f32, total_size);
    defer allocator.free(K);
    var V = try allocator.alloc(f32, total_size);
    defer allocator.free(V);
    const output = try allocator.alloc(f32, total_size);
    defer allocator.free(output);

    // Initialize with simple pattern
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..total_size) |i| {
        Q[i] = random.float(f32) * 2.0 - 1.0;
        K[i] = random.float(f32) * 2.0 - 1.0;
        V[i] = random.float(f32) * 2.0 - 1.0;
    }

    try ref.forward(Q, K, V, output, batch_size, num_heads, seq_len, head_dim);

    // Verify output is valid (not NaN or Inf)
    for (output) |val| {
        try std.testing.expect(!math.isNan(val));
        try std.testing.expect(!math.isInf(val));
    }

    // Verify output rows sum to reasonable values (weighted average of V)
    // Each output element should be a convex combination of V values
}

test "attention_ref identity" {
    const allocator = std.testing.allocator;
    const ref = AttentionRef.init(allocator);

    // When K=Q and attention is uniform, output should be average of V
    const batch_size: usize = 1;
    const num_heads: usize = 1;
    const seq_len: usize = 2;
    const head_dim: usize = 4;
    const total_size = batch_size * num_heads * seq_len * head_dim;

    var Q = try allocator.alloc(f32, total_size);
    defer allocator.free(Q);
    var K = try allocator.alloc(f32, total_size);
    defer allocator.free(K);
    var V = try allocator.alloc(f32, total_size);
    defer allocator.free(V);
    const output = try allocator.alloc(f32, total_size);
    defer allocator.free(output);

    // Set Q and K to be identical (uniform attention)
    for (0..total_size) |i| {
        Q[i] = 0.5;
        K[i] = 0.5;
    }

    // V = [[1, 2, 3, 4], [5, 6, 7, 8]]
    V[0] = 1.0;
    V[1] = 2.0;
    V[2] = 3.0;
    V[3] = 4.0;
    V[4] = 5.0;
    V[5] = 6.0;
    V[6] = 7.0;
    V[7] = 8.0;

    try ref.forward(Q, K, V, output, batch_size, num_heads, seq_len, head_dim);

    // With uniform Q=K, softmax gives 0.5, 0.5
    // Output should be average: [3, 4, 5, 6] for both rows
    const expected_avg = [_]f32{ 3.0, 4.0, 5.0, 6.0 };

    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            const val = output[i * head_dim + d];
            try std.testing.expectApproxEqAbs(expected_avg[d], val, 0.001);
        }
    }
}
