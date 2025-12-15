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

    // Spatial Sort shader
    const spatial_sort_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const spatial_sort_spv = spatial_sort_compile.addOutputFileArg("spatial_sort.spv");
    spatial_sort_compile.addFileArg(b.path("shaders/spatial_sort.comp"));

    // Gravity Attention shader
    const attention_gravity_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const attention_gravity_spv = attention_gravity_compile.addOutputFileArg("attention_gravity.spv");
    attention_gravity_compile.addFileArg(b.path("shaders/attention_gravity.comp"));

    // --- Radix Sort Shaders ---
    const radix_count_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const radix_count_spv = radix_count_compile.addOutputFileArg("radix_count.spv");
    radix_count_compile.addFileArg(b.path("shaders/radix_count.comp"));

    const radix_scan_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const radix_scan_spv = radix_scan_compile.addOutputFileArg("radix_scan.spv");
    radix_scan_compile.addFileArg(b.path("shaders/radix_scan.comp"));

    const radix_scatter_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const radix_scatter_spv = radix_scatter_compile.addOutputFileArg("radix_scatter.spv");
    radix_scatter_compile.addFileArg(b.path("shaders/radix_scatter.comp"));

    const iota_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const iota_spv = iota_compile.addOutputFileArg("iota.spv");
    iota_compile.addFileArg(b.path("shaders/iota.comp"));

    const magnitude_sort_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const magnitude_sort_spv = magnitude_sort_compile.addOutputFileArg("magnitude_sort.spv");
    magnitude_sort_compile.addFileArg(b.path("shaders/magnitude_sort.comp"));

    // --- Optimized FP32 Fast Shader (vectorized, block skipping) ---
    const attention_f32_fast_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const attention_f32_fast_spv = attention_f32_fast_compile.addOutputFileArg("attention_f32_fast.spv");
    attention_f32_fast_compile.addFileArg(b.path("shaders/attention_f32_fast.comp"));

    // --- FP16 Shaders (require GL_EXT_shader_explicit_arithmetic_types_float16) ---
    const attention_f16_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const attention_f16_spv = attention_f16_compile.addOutputFileArg("attention_f16.spv");
    attention_f16_compile.addFileArg(b.path("shaders/attention_f16.comp"));

    const attention_f16_amd_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const attention_f16_amd_spv = attention_f16_amd_compile.addOutputFileArg("attention_f16_amd.spv");
    attention_f16_amd_compile.addFileArg(b.path("shaders/attention_f16_amd.comp"));

    // --- Paged Attention Shader (for PagedAttention feature) ---
    const attention_paged_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const attention_paged_spv = attention_paged_compile.addOutputFileArg("attention_paged.spv");
    attention_paged_compile.addFileArg(b.path("shaders/attention_paged.comp"));

    // K/V copy shader for paged attention
    const copy_kv_paged_compile = b.addSystemCommand(&.{ "glslc", "-O", "--target-env=vulkan1.2", "-o" });
    const copy_kv_paged_spv = copy_kv_paged_compile.addOutputFileArg("copy_kv_to_paged.spv");
    copy_kv_paged_compile.addFileArg(b.path("shaders/copy_kv_to_paged.comp"));
    // --------------------------

    // Main library (shared)
    const lib = b.addSharedLibrary(.{
        .name = "aule",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.root_module.addImport("vulkan", vulkan_mod);
    lib.root_module.addOptions("config", options);
    lib.root_module.addAnonymousImport("attention_f32_spv", .{ .root_source_file = attention_f32_spv });
    lib.root_module.addAnonymousImport("attention_amd_spv", .{ .root_source_file = attention_amd_spv });
    lib.root_module.addAnonymousImport("attention_bwd_spv", .{ .root_source_file = attention_bwd_spv });
    lib.root_module.addAnonymousImport("attention_fwd_lse_spv", .{ .root_source_file = attention_fwd_lse_spv });
    lib.root_module.addAnonymousImport("spatial_sort_spv", .{ .root_source_file = spatial_sort_spv });
    lib.root_module.addAnonymousImport("attention_gravity_spv", .{ .root_source_file = attention_gravity_spv });
    
    // Radix Imports
    lib.root_module.addAnonymousImport("radix_count_spv", .{ .root_source_file = radix_count_spv });
    lib.root_module.addAnonymousImport("radix_scan_spv", .{ .root_source_file = radix_scan_spv });
    lib.root_module.addAnonymousImport("radix_scatter_spv", .{ .root_source_file = radix_scatter_spv });
    lib.root_module.addAnonymousImport("iota_spv", .{ .root_source_file = iota_spv });
    lib.root_module.addAnonymousImport("magnitude_sort_spv", .{ .root_source_file = magnitude_sort_spv });

    // Optimized FP32 fast shader
    lib.root_module.addAnonymousImport("attention_f32_fast_spv", .{ .root_source_file = attention_f32_fast_spv });

    // FP16 shader imports
    lib.root_module.addAnonymousImport("attention_f16_spv", .{ .root_source_file = attention_f16_spv });
    lib.root_module.addAnonymousImport("attention_f16_amd_spv", .{ .root_source_file = attention_f16_amd_spv });

    // Paged attention shaders
    lib.root_module.addAnonymousImport("attention_paged_spv", .{ .root_source_file = attention_paged_spv });
    lib.root_module.addAnonymousImport("copy_kv_to_paged_spv", .{ .root_source_file = copy_kv_paged_spv });

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
    static_lib.root_module.addAnonymousImport("attention_f32_spv", .{ .root_source_file = attention_f32_spv });
    static_lib.root_module.addAnonymousImport("attention_amd_spv", .{ .root_source_file = attention_amd_spv });
    static_lib.root_module.addAnonymousImport("attention_bwd_spv", .{ .root_source_file = attention_bwd_spv });
    static_lib.root_module.addAnonymousImport("attention_fwd_lse_spv", .{ .root_source_file = attention_fwd_lse_spv });
    static_lib.root_module.addAnonymousImport("spatial_sort_spv", .{ .root_source_file = spatial_sort_spv });
    static_lib.root_module.addAnonymousImport("attention_gravity_spv", .{ .root_source_file = attention_gravity_spv });
    
    // Radix Imports
    static_lib.root_module.addAnonymousImport("radix_count_spv", .{ .root_source_file = radix_count_spv });
    static_lib.root_module.addAnonymousImport("radix_scan_spv", .{ .root_source_file = radix_scan_spv });
    static_lib.root_module.addAnonymousImport("radix_scatter_spv", .{ .root_source_file = radix_scatter_spv });
    static_lib.root_module.addAnonymousImport("iota_spv", .{ .root_source_file = iota_spv });
    static_lib.root_module.addAnonymousImport("magnitude_sort_spv", .{ .root_source_file = magnitude_sort_spv });

    // Optimized FP32 fast shader (static)
    static_lib.root_module.addAnonymousImport("attention_f32_fast_spv", .{ .root_source_file = attention_f32_fast_spv });

    // FP16 shader imports (static)
    static_lib.root_module.addAnonymousImport("attention_f16_spv", .{ .root_source_file = attention_f16_spv });
    static_lib.root_module.addAnonymousImport("attention_f16_amd_spv", .{ .root_source_file = attention_f16_amd_spv });

    // Paged attention shaders (static)
    static_lib.root_module.addAnonymousImport("attention_paged_spv", .{ .root_source_file = attention_paged_spv });
    static_lib.root_module.addAnonymousImport("copy_kv_to_paged_spv", .{ .root_source_file = copy_kv_paged_spv });

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
    attention_tests.root_module.addImport("aule", &static_lib.root_module);
    attention_tests.linkSystemLibrary("vulkan");
    attention_tests.linkLibC();

    const run_attention_tests = b.addRunArtifact(attention_tests);
    const attention_test_step = b.step("test-attention", "Run attention tests");
    attention_test_step.dependOn(&run_attention_tests.step);

    // Tests - block pool
    const block_pool_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_block_pool.zig"),
        .target = target,
        .optimize = optimize,
    });
    block_pool_tests.root_module.addImport("vulkan", vulkan_mod);
    block_pool_tests.root_module.addOptions("config", options);
    block_pool_tests.root_module.addImport("aule", &static_lib.root_module);

    // Create modules with vulkan dependency
    const block_pool_mod = b.createModule(.{ .root_source_file = b.path("src/block_pool.zig") });
    block_pool_mod.addImport("vulkan", vulkan_mod);

    const block_table_mod = b.createModule(.{ .root_source_file = b.path("src/block_table.zig") });
    block_table_mod.addImport("vulkan", vulkan_mod);

    const vulkan_context_mod = b.createModule(.{ .root_source_file = b.path("src/vulkan_context.zig") });
    vulkan_context_mod.addImport("vulkan", vulkan_mod);
    vulkan_context_mod.addOptions("config", options);

    const buffer_manager_mod = b.createModule(.{ .root_source_file = b.path("src/buffer_manager.zig") });
    buffer_manager_mod.addImport("vulkan", vulkan_mod);

    block_pool_tests.root_module.addImport("block_pool", block_pool_mod);
    block_pool_tests.root_module.addImport("block_table", block_table_mod);
    block_pool_tests.root_module.addImport("vulkan_context", vulkan_context_mod);
    block_pool_tests.root_module.addImport("buffer_manager", buffer_manager_mod);
    block_pool_tests.linkSystemLibrary("vulkan");
    block_pool_tests.linkLibC();

    const run_block_pool_tests = b.addRunArtifact(block_pool_tests);
    const block_pool_test_step = b.step("test-block-pool", "Run block pool tests");
    block_pool_test_step.dependOn(&run_block_pool_tests.step);

    // Tests - paged attention
    const paged_attention_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_paged_attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    paged_attention_tests.root_module.addImport("vulkan", vulkan_mod);
    paged_attention_tests.root_module.addOptions("config", options);
    paged_attention_tests.root_module.addImport("aule", &static_lib.root_module);
    paged_attention_tests.root_module.addAnonymousImport("attention_f32_spv", .{ .root_source_file = attention_f32_spv });
    paged_attention_tests.root_module.addAnonymousImport("attention_amd_spv", .{ .root_source_file = attention_amd_spv });
    paged_attention_tests.root_module.addAnonymousImport("attention_paged_spv", .{ .root_source_file = attention_paged_spv });
    paged_attention_tests.root_module.addAnonymousImport("copy_kv_to_paged_spv", .{ .root_source_file = copy_kv_paged_spv });
    paged_attention_tests.linkSystemLibrary("vulkan");
    paged_attention_tests.linkLibC();

    const run_paged_attention_tests = b.addRunArtifact(paged_attention_tests);
    const paged_attention_test_step = b.step("test-paged", "Run paged attention tests");
    paged_attention_test_step.dependOn(&run_paged_attention_tests.step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_attention_tests.step);
    test_step.dependOn(&run_block_pool_tests.step);

    // All tests
    const all_test_step = b.step("test-all", "Run all tests");
    all_test_step.dependOn(&run_attention_tests.step);
    all_test_step.dependOn(&run_block_pool_tests.step);
    all_test_step.dependOn(&run_paged_attention_tests.step);

    // Benchmark
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("tests/benchmark_attention.zig"),
        .target = target,
        .optimize = optimize, // Use build arg
    });
    benchmark.root_module.addImport("aule", &static_lib.root_module);
    // Note: static_lib already has vulkan/libc linked, but we might need to ensure transient deps work
    // Ideally we link shared 'lib' or static 'static_lib' module.
    
    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run attention benchmark");
    benchmark_step.dependOn(&run_benchmark.step);
}
