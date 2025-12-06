const std = @import("std");
const aule = @import("aule");
const Attention = aule.Attention;

pub fn main() !void {
    // Setup allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize Engine
    std.debug.print("Initializing aule-attention...\n", .{});
    var attn = try Attention.init(allocator);
    defer attn.deinit();
    var ctx = &attn.context;

    // Configuration (Back to larger size)
    const batch = 4;
    const heads = 8;
    const seq = 256;
    const dim = 64;
    const shape = [4]u32{batch, heads, seq, dim};
    const total_elements = batch * heads * seq * dim;
    
    std.debug.print("Benchmarking config: B={} H={} S={} D={}\n", .{batch, heads, seq, dim});

    // Alloc Host Memory for initialization
    const host_data = try allocator.alloc(f32, total_elements);
    defer allocator.free(host_data);
    @memset(host_data, 0.1);

    // 1. Setup GPU Tensors (Once)
    std.debug.print("Allocating GPU tensors...\n", .{});
    var q_t = try ctx.createTensor(shape);
    defer ctx.destroyTensor(&q_t);
    var k_t = try ctx.createTensor(shape);
    defer ctx.destroyTensor(&k_t);
    var v_t = try ctx.createTensor(shape);
    defer ctx.destroyTensor(&v_t);
    var o_t = try ctx.createTensor(shape);
    defer ctx.destroyTensor(&o_t);

    // 2. Upload (Once)
    std.debug.print("Uploading data...\n", .{});
    try ctx.upload(&q_t, host_data);
    try ctx.upload(&k_t, host_data);
    try ctx.upload(&v_t, host_data);

    // Warmup
    std.debug.print("Warming up kernel...\n", .{});
    try ctx.attention(&q_t, &k_t, &v_t, &o_t, false);

    // Benchmark Loop (Compute Only)
    const iterations = 50;
    var timer = try std.time.Timer.start();
    
    std.debug.print("Running {} iterations (Compute only)...\n", .{iterations});
    const start = timer.read();
    for (0..iterations) |_| {
        try ctx.attention(&q_t, &k_t, &v_t, &o_t, false);
    }
    const end = timer.read();

    const total_ns = end - start;
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;

    // Calculate TFLOPS
    // Ops = 4 * B * H * S^2 * D
    const ops_per_iter = 4.0 * @as(f64, @floatFromInt(batch)) * 
                             @as(f64, @floatFromInt(heads)) * 
                             @as(f64, @floatFromInt(seq)) * 
                             @as(f64, @floatFromInt(seq)) * 
                             @as(f64, @floatFromInt(dim));
    
    const tflops = (ops_per_iter) / (avg_ms / 1000.0) / 1_000_000_000_000.0;

    std.debug.print("--------------------------------------------------\n", .{});
    std.debug.print("Average Time: {d:.3} ms\n", .{avg_ms});
    std.debug.print("Throughput:   {d:.3} TFLOPS\n", .{tflops});
    std.debug.print("--------------------------------------------------\n", .{});
}
