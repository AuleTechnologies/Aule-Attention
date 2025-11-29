const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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

    // Compile SPIR-V shaders at build time
    const shader_compile = b.addSystemCommand(&.{
        "glslc",
        "-O",
        "--target-env=vulkan1.2",
        "-o",
    });
    const spv_path = shader_compile.addOutputFileArg("test.spv");
    shader_compile.addFileArg(b.path("shaders/test.comp"));

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

    // Main library (shared)
    const lib = b.addSharedLibrary(.{
        .name = "aule",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.root_module.addImport("vulkan", vulkan_mod);
    lib.root_module.addAnonymousImport("test_shader_spv", .{
        .root_source_file = spv_path,
    });
    lib.root_module.addAnonymousImport("attention_f32_spv", .{
        .root_source_file = attention_f32_spv,
    });
    lib.root_module.addAnonymousImport("attention_amd_spv", .{
        .root_source_file = attention_amd_spv,
    });
    lib.linkSystemLibrary("vulkan");
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
    static_lib.root_module.addAnonymousImport("test_shader_spv", .{
        .root_source_file = spv_path,
    });
    static_lib.root_module.addAnonymousImport("attention_f32_spv", .{
        .root_source_file = attention_f32_spv,
    });
    static_lib.root_module.addAnonymousImport("attention_amd_spv", .{
        .root_source_file = attention_amd_spv,
    });
    static_lib.linkSystemLibrary("vulkan");
    static_lib.linkLibC();

    // Tests - multiply
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_multiply.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_tests.root_module.addImport("vulkan", vulkan_mod);
    lib_tests.root_module.addImport("aule", static_lib.root_module);
    lib_tests.linkSystemLibrary("vulkan");
    lib_tests.linkLibC();

    const run_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Tests - attention
    const attention_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    attention_tests.root_module.addImport("vulkan", vulkan_mod);
    attention_tests.root_module.addImport("aule", static_lib.root_module);
    attention_tests.linkSystemLibrary("vulkan");
    attention_tests.linkLibC();

    const run_attention_tests = b.addRunArtifact(attention_tests);
    const attention_test_step = b.step("test-attention", "Run attention tests");
    attention_test_step.dependOn(&run_attention_tests.step);

    // All tests
    const all_test_step = b.step("test-all", "Run all tests");
    all_test_step.dependOn(&run_tests.step);
    all_test_step.dependOn(&run_attention_tests.step);
}
