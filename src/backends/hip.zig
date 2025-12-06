//! HIP/ROCm backend for AMD datacenter GPUs (MI300X, MI250, etc.)
//!
//! This backend uses the HIP C API to run attention kernels on AMD GPUs
//! that don't have Vulkan support (datacenter accelerators).
//!
//! Requires ROCm to be installed on the system.

const std = @import("std");
const config = @import("config");

pub const HipError = error{
    InvalidDevice,
    OutOfMemory,
    InvalidValue,
    NotInitialized,
    LaunchFailure,
    Unknown,
    NotSupported,
};

// Conditional compilation based on build option
const use_hip = if (@hasDecl(config, "enable_hip")) config.enable_hip else false;

const c = if (use_hip) @cImport({
    @cInclude("hip/hip_runtime.h");
}) else struct {};

fn checkHipError(err: if (use_hip) c.hipError_t else i32) HipError!void {
    if (!use_hip) return HipError.NotSupported;
    return switch (err) {
        c.hipSuccess => {},
        c.hipErrorInvalidDevice => HipError.InvalidDevice,
        c.hipErrorOutOfMemory => HipError.OutOfMemory,
        c.hipErrorInvalidValue => HipError.InvalidValue,
        c.hipErrorNotInitialized => HipError.NotInitialized,
        c.hipErrorLaunchFailure => HipError.LaunchFailure,
        else => HipError.Unknown,
    };
}

/// GPU buffer allocated via HIP
const HipBuffer = if (use_hip) struct {
    ptr: *anyopaque,
    size: usize,

    pub fn init(size: usize) HipError!HipBuffer {
        var ptr: ?*anyopaque = null;
        try checkHipError(c.hipMalloc(&ptr, size));
        return HipBuffer{
            .ptr = ptr orelse return HipError.OutOfMemory,
            .size = size,
        };
    }

    pub fn deinit(self: *HipBuffer) void {
        _ = c.hipFree(self.ptr);
        self.* = undefined;
    }

    pub fn upload(self: *HipBuffer, data: []const u8) HipError!void {
        if (data.len > self.size) return HipError.InvalidValue;
        try checkHipError(c.hipMemcpy(
            self.ptr,
            data.ptr,
            data.len,
            c.hipMemcpyHostToDevice,
        ));
    }

    pub fn download(self: *const HipBuffer, output: []u8) HipError!void {
        if (output.len > self.size) return HipError.InvalidValue;
        try checkHipError(c.hipMemcpy(
            output.ptr,
            self.ptr,
            output.len,
            c.hipMemcpyDeviceToHost,
        ));
    }
} else struct {};

/// HIP tensor for attention operations
pub const HipTensor = struct {
    buffer: if (use_hip) HipBuffer else void,
    shape: [4]u32,
    element_count: usize,

    pub fn init(shape: [4]u32) HipError!HipTensor {
        if (!use_hip) return HipError.NotSupported;
        
        var count: usize = 1;
        for (shape) |dim| {
            count *= dim;
        }
        const size = count * @sizeOf(f32);
        return HipTensor{
            .buffer = try HipBuffer.init(size),
            .shape = shape,
            .element_count = count,
        };
    }

    pub fn deinit(self: *HipTensor) void {
        if (use_hip) {
            self.buffer.deinit();
        }
        self.* = undefined;
    }

    pub fn upload(self: *HipTensor, data: []const f32) HipError!void {
        if (!use_hip) return HipError.NotSupported;
        if (data.len != self.element_count) return HipError.InvalidValue;
        const bytes = std.mem.sliceAsBytes(data);
        try self.buffer.upload(bytes);
    }

    pub fn download(self: *const HipTensor, output: []f32) HipError!void {
        if (!use_hip) return HipError.NotSupported;
        if (output.len != self.element_count) return HipError.InvalidValue;
        const bytes = std.mem.sliceAsBytes(output);
        try self.buffer.download(bytes);
    }
};

/// HIP attention context
pub const HipAttention = struct {
    module: if (use_hip) c.hipModule_t else void,
    kernel: if (use_hip) c.hipFunction_t else void,
    device_id: c_int,

    const Self = @This();

    /// Initialize HIP backend with embedded kernel
    pub fn init() HipError!Self {
        if (!use_hip) return HipError.NotSupported;

        // Initialize HIP
        try checkHipError(c.hipInit(0));

        // Get device count
        var device_count: c_int = 0;
        try checkHipError(c.hipGetDeviceCount(&device_count));
        if (device_count == 0) return HipError.InvalidDevice;

        // Use first device
        try checkHipError(c.hipSetDevice(0));

        // Load precompiled kernel module (embedded at compile time)
        var module: c.hipModule_t = undefined;

        // Conditional embed: only load kernel binary when HIP is enabled
        const kernel_data = if (use_hip) @embedFile("attention_hip.hsaco") else "";
        
        if (use_hip) {
             try checkHipError(c.hipModuleLoadData(&module, kernel_data.ptr));
        }

        // Get kernel function
        var kernel: c.hipFunction_t = undefined;
        if (use_hip) {
            try checkHipError(c.hipModuleGetFunction(&kernel, module, "attention_forward"));
        }

        return Self{
            .module = module,
            .kernel = kernel,
            .device_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        if (use_hip) {
            _ = c.hipModuleUnload(self.module);
        }
        self.* = undefined;
    }

    /// Compute attention: output = softmax(Q @ K^T / sqrt(d)) @ V
    pub fn forward(
        self: *Self,
        Q: *HipTensor,
        K: *HipTensor,
        V: *HipTensor,
        output: *HipTensor,
    ) HipError!void {
        if (!use_hip) return HipError.NotSupported;

        const batch_size = Q.shape[0];
        const num_heads = Q.shape[1];
        const seq_len = Q.shape[2];
        const head_dim = Q.shape[3];
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Kernel arguments
        var args = [_]?*anyopaque{
            @ptrCast(&Q.buffer.ptr),
            @ptrCast(&K.buffer.ptr),
            @ptrCast(&V.buffer.ptr),
            @ptrCast(&output.buffer.ptr),
            @ptrCast(@constCast(&batch_size)),
            @ptrCast(@constCast(&num_heads)),
            @ptrCast(@constCast(&seq_len)),
            @ptrCast(@constCast(&head_dim)),
            @ptrCast(@constCast(&scale)),
        };

        // Launch kernel
        const block_size: c_uint = 256;
        const grid_size: c_uint = @intCast((batch_size * num_heads * seq_len + block_size - 1) / block_size);

        try checkHipError(c.hipModuleLaunchKernel(
            self.kernel,
            grid_size,
            1,
            1, // grid dimensions
            block_size,
            1,
            1, // block dimensions
            0, // shared memory
            null, // stream
            &args,
            null, // extra
        ));

        // Synchronize
        try checkHipError(c.hipDeviceSynchronize());
    }
};

/// Check if HIP/ROCm is available on this system
pub fn isAvailable() bool {
    if (!use_hip) return false;
    
    var device_count: c_int = 0;
    const err = c.hipGetDeviceCount(&device_count);
    return err == c.hipSuccess and device_count > 0;
}

/// Get device name
pub fn getDeviceName(allocator: std.mem.Allocator) ![]u8 {
    if (!use_hip) return error.NotSupported;

    var props: c.hipDeviceProp_t = undefined;
    try checkHipError(c.hipGetDeviceProperties(&props, 0));

    const name_len = std.mem.indexOfScalar(u8, &props.name, 0) orelse props.name.len;
    const name = try allocator.alloc(u8, name_len);
    @memcpy(name, props.name[0..name_len]);
    return name;
}