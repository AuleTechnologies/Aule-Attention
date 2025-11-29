const std = @import("std");
const aule = @import("aule");

test "GPU multiply by 2" {
    var instance = try aule.Aule.init(std.testing.allocator);
    defer instance.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output: [8]f32 = undefined;

    try instance.testMultiply(&input, &output);

    // Verify each element is doubled
    for (input, output) |in_val, out_val| {
        try std.testing.expectApproxEqAbs(in_val * 2.0, out_val, 0.0001);
    }
}

test "GPU multiply larger array" {
    var instance = try aule.Aule.init(std.testing.allocator);
    defer instance.deinit();

    // Test with 1024 elements (4 workgroups of 256)
    var input: [1024]f32 = undefined;
    var output: [1024]f32 = undefined;

    for (&input, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    try instance.testMultiply(&input, &output);

    for (input, output) |in_val, out_val| {
        try std.testing.expectApproxEqAbs(in_val * 2.0, out_val, 0.0001);
    }
}

test "GPU multiply non-aligned count" {
    var instance = try aule.Aule.init(std.testing.allocator);
    defer instance.deinit();

    // Test with 300 elements (not a multiple of workgroup size 256)
    var input: [300]f32 = undefined;
    var output: [300]f32 = undefined;

    for (&input, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.5;
    }

    try instance.testMultiply(&input, &output);

    for (input, output) |in_val, out_val| {
        try std.testing.expectApproxEqAbs(in_val * 2.0, out_val, 0.0001);
    }
}
