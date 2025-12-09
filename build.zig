const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Options
    const enable_hip = b.option(bool, "hip", "Enable HIP backend") orelse false;
    const options = b.addOptions();
    options.addOption(bool, "enable_hip", enable_hip);

    // Vulkan-zig: use the generator to create bindings
    const vulkan_dep = b.dependency("vulkan", .{});
    const vulkan_headers_dep = b.dependency("vulkan_headers", .{});

    // Get the generator executable
    const vk_gen = vulkan_dep.artifact("vulkan-zig-generator");

    // Run generator with vk.xml as input
    const vk_generate_cmd = b.addRunArtifact(vk_gen);
    vk_generate_cmd.addFileArg(vulkan_headers_dep.path("registry/vk.xml"));
    const vulkan_zig = vk_generate_cmd.addOutputFileArg("vk.zig");

    // Create module from generated source
    const vulkan_mod = b.addModule("vulkan", .{
        .root_source_file = vulkan_zig,
    });

    // Attention shader (fp32)
    const attention_f32_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const attention_f32_spv = attention_f32_compile.addOutputFileArg("attention_f32.spv");
    attention_f32_compile.addFileArg(b.path("shaders/attention_f32.comp"));

    // AMD-optimized attention shader (fp32, 64-wide wavefront)
    const attention_amd_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const attention_amd_spv = attention_amd_compile.addOutputFileArg("attention_f32_amd.spv");
    attention_amd_compile.addFileArg(b.path("shaders/attention_f32_amd.comp"));

    // Backward pass shader (fp32)
    const attention_bwd_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const attention_bwd_spv = attention_bwd_compile.addOutputFileArg("attention_backward_f32.spv");
    attention_bwd_compile.addFileArg(b.path("shaders/attention_backward_f32.comp"));

    // Forward pass with LSE output (for backward)
    const attention_fwd_lse_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const attention_fwd_lse_spv = attention_fwd_lse_compile.addOutputFileArg("attention_forward_f32.spv");
    attention_fwd_lse_compile.addFileArg(b.path("shaders/attention_forward_f32.comp"));

    // Main library (shared)
    const lib = b.addSharedLibrary(.{
        .name = "aule",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.root_module.addImport("vulkan", vulkan_mod);
    lib.root_module.addOptions("config", options);
    lib.root_module.addAnonymousImport("attention_f32_spv", .{
        .root_source_file = attention_f32_spv,
    });
    lib.root_module.addAnonymousImport("attention_amd_spv", .{
        .root_source_file = attention_amd_spv,
    });
    lib.root_module.addAnonymousImport("attention_bwd_spv", .{
        .root_source_file = attention_bwd_spv,
    });
    lib.root_module.addAnonymousImport("attention_fwd_lse_spv", .{
        .root_source_file = attention_fwd_lse_spv,
    });
    // Link Vulkan on native builds only - cross-compilation uses runtime dynamic loading
    const is_native = target.query.isNative();
    if (is_native) {
        lib.linkSystemLibrary("vulkan");
    }
    lib.linkLibC();

    b.installArtifact(lib);

    // Static library for testing
    const static_lib = b.addStaticLibrary(.{
        .name = "aule_static",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    static_lib.root_module.addImport("vulkan", vulkan_mod);
    static_lib.root_module.addOptions("config", options);
    static_lib.root_module.addAnonymousImport("attention_f32_spv", .{
        .root_source_file = attention_f32_spv,
    });
    static_lib.root_module.addAnonymousImport("attention_amd_spv", .{
        .root_source_file = attention_amd_spv,
    });
    static_lib.root_module.addAnonymousImport("attention_bwd_spv", .{
        .root_source_file = attention_bwd_spv,
    });
    static_lib.root_module.addAnonymousImport("attention_fwd_lse_spv", .{
        .root_source_file = attention_fwd_lse_spv,
    });
    static_lib.linkSystemLibrary("vulkan");
    static_lib.linkLibC();

    // Tests - attention
    const attention_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    attention_tests.root_module.addImport("vulkan", vulkan_mod);
    attention_tests.root_module.addOptions("config", options);
    attention_tests.root_module.addImport("aule", static_lib.root_module);
    attention_tests.linkSystemLibrary("vulkan");
    attention_tests.linkLibC();

    const run_attention_tests = b.addRunArtifact(attention_tests);
    const attention_test_step = b.step("test-attention", "Run attention tests");
    attention_test_step.dependOn(&run_attention_tests.step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_attention_tests.step);

    // All tests
    const all_test_step = b.step("test-all", "Run all tests");
    all_test_step.dependOn(&run_attention_tests.step);

    // Benchmark
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("tests/benchmark_attention.zig"),
        .target = target,
        .optimize = optimize, // Use build arg
    });
    benchmark.root_module.addImport("aule", static_lib.root_module);
    // Note: static_lib already has vulkan/libc linked, but we might need to ensure transient deps work
    // Ideally we link shared 'lib' or static 'static_lib' module.
    
    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run attention benchmark");
    benchmark_step.dependOn(&run_benchmark.step);
}
