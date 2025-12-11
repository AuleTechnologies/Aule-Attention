const std = @import("std");
const vk = @import("vulkan");
const VulkanContext = @import("vulkan_context.zig").VulkanContext;
const BufferManager = @import("buffer_manager.zig").BufferManager;
const Buffer = @import("buffer_manager.zig").Buffer;

/// A tensor that lives on the GPU
/// Data stays on GPU between operations - no copy overhead for repeated use
pub const GpuTensor = struct {
    buffer: Buffer,
    shape: [4]u32, // [batch, heads, seq, dim] or fewer dimensions
    ndim: u8,
    dtype: DType,
    element_count: usize,
    buffer_manager: *const BufferManager,

    pub const DType = enum {
        f32,
        f16,

        pub fn size(self: DType) usize {
            return switch (self) {
                .f32 => 4,
                .f16 => 2,
            };
        }
    };

    const Self = @This();

    /// Create a new GPU tensor with given shape
    pub fn init(
        buffer_manager: *const BufferManager,
        shape: []const u32,
        dtype: DType,
    ) !Self {
        std.debug.print("GpuTensor.init: shape len {}\n", .{shape.len});
        if (shape.len > 4 or shape.len == 0) {
            return error.InvalidShape;
        }

        var element_count: usize = 1;
        var stored_shape: [4]u32 = .{ 1, 1, 1, 1 };
        for (shape, 0..) |dim, i| {
            stored_shape[i] = dim;
            element_count *= dim;
        }

        const byte_size = element_count * dtype.size();
        const buffer = try buffer_manager.createHostVisibleStorageBuffer(@intCast(byte_size));

        return Self{
            .buffer = buffer,
            .shape = stored_shape,
            .ndim = @intCast(shape.len),
            .dtype = dtype,
            .element_count = element_count,
            .buffer_manager = buffer_manager,
        };
    }

    /// Free GPU memory
    pub fn deinit(self: *Self) void {
        var buf = self.buffer;
        self.buffer_manager.destroyBuffer(&buf);
        self.* = undefined;
    }

    /// Upload data from CPU to GPU
    pub fn upload(self: *Self, data: []const f32) !void {
        if (data.len != self.element_count) {
            return error.SizeMismatch;
        }
        if (self.dtype != .f32) {
            return error.DTypeMismatch;
        }

        const gpu_slice = self.buffer.getMappedSlice(f32);
        @memcpy(gpu_slice, data);
    }

    /// Download data from GPU to CPU
    pub fn download(self: *const Self, output: []f32) !void {
        if (output.len != self.element_count) {
            return error.SizeMismatch;
        }
        if (self.dtype != .f32) {
            return error.DTypeMismatch;
        }

        const gpu_slice = self.buffer.getMappedSlice(f32);
        @memcpy(output, gpu_slice);
    }

    /// Get the raw Vulkan buffer handle
    pub fn getBuffer(self: *const Self) vk.Buffer {
        return self.buffer.buffer;
    }

    /// Get buffer size in bytes
    pub fn byteSize(self: *const Self) vk.DeviceSize {
        return self.buffer.size;
    }

    /// Get shape as slice
    pub fn getShape(self: *const Self) []const u32 {
        return self.shape[0..self.ndim];
    }
};

/// Manages multiple GPU tensors and provides attention operations on them
pub const TensorContext = struct {
    ctx: *const VulkanContext,
    buffer_manager: BufferManager,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        const ctx = try allocator.create(VulkanContext);
        ctx.* = try VulkanContext.init(allocator);

        return Self{
            .ctx = ctx,
            .buffer_manager = BufferManager.init(ctx),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var ctx_ptr = @constCast(self.ctx);
        ctx_ptr.deinit();
        self.allocator.destroy(ctx_ptr);
        self.* = undefined;
    }

    /// Create a new tensor on GPU
    pub fn createTensor(self: *Self, shape: []const u32, dtype: GpuTensor.DType) !GpuTensor {
        return GpuTensor.init(&self.buffer_manager, shape, dtype);
    }

    /// Wait for all GPU operations to complete
    pub fn synchronize(self: *const Self) !void {
        try self.ctx.waitIdle();
    }
};
