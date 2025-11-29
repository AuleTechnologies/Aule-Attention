const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.vulkan_context);

/// GPU vendor identification
pub const GpuVendor = enum {
    amd,
    nvidia,
    intel,
    apple,
    other,
};

/// AMD GPU architecture
pub const AmdArch = enum {
    gcn4, // Polaris (RX 400/500)
    gcn5, // Vega
    rdna1, // RX 5000
    rdna2, // RX 6000, 680M, Steam Deck
    rdna3, // RX 7000, 780M
    cdna, // MI100/MI200/MI300 (datacenter)
    unknown,
};

/// GPU capabilities detected at runtime
pub const GpuCapabilities = struct {
    vendor: GpuVendor,
    amd_arch: AmdArch,
    fp16_supported: bool,
    subgroup_size: u32, // Wavefront/warp size
    device_name: [256]u8,

    pub fn isAmd(self: GpuCapabilities) bool {
        return self.vendor == .amd;
    }

    pub fn isRdna(self: GpuCapabilities) bool {
        return self.amd_arch == .rdna1 or self.amd_arch == .rdna2 or self.amd_arch == .rdna3;
    }

    pub fn prefersFp16(self: GpuCapabilities) bool {
        // RDNA 2/3 have native FP16 with 2x throughput
        return self.fp16_supported and (self.amd_arch == .rdna2 or self.amd_arch == .rdna3);
    }

    pub fn getDeviceName(self: *const GpuCapabilities) []const u8 {
        return std.mem.sliceTo(&self.device_name, 0);
    }
};

pub const VulkanContext = struct {
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    compute_queue: vk.Queue,
    queue_family_index: u32,
    device_properties: vk.PhysicalDeviceProperties,
    gpu_caps: GpuCapabilities,

    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        // Load base functions from the Vulkan loader
        const vkb = BaseDispatch.load(getLoaderFn()) catch |err| {
            log.err("Failed to load Vulkan base functions: {}", .{err});
            return error.VulkanLoadFailed;
        };

        // Create instance
        const app_info = vk.ApplicationInfo{
            .p_application_name = "aule-attention",
            .application_version = vk.makeApiVersion(0, 0, 1, 0),
            .p_engine_name = "aule",
            .engine_version = vk.makeApiVersion(0, 0, 1, 0),
            .api_version = vk.API_VERSION_1_2,
        };

        const instance = try vkb.createInstance(&.{
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = null,
            .enabled_extension_count = 0,
            .pp_enabled_extension_names = null,
        }, null);

        const vki = InstanceDispatch.load(instance, vkb.dispatch.vkGetInstanceProcAddr) catch |err| {
            log.err("Failed to load instance functions: {}", .{err});
            return error.VulkanLoadFailed;
        };
        errdefer vki.destroyInstance(instance, null);

        // Select physical device with compute capability
        const physical_device, const queue_family_index = try selectPhysicalDevice(allocator, vki, instance);
        const device_properties = vki.getPhysicalDeviceProperties(physical_device);

        // Detect GPU capabilities
        const gpu_caps = detectGpuCapabilities(device_properties);
        log.info("Selected GPU: {s}", .{gpu_caps.getDeviceName()});
        log.info("  Vendor: {s}, AMD Arch: {s}", .{ @tagName(gpu_caps.vendor), @tagName(gpu_caps.amd_arch) });
        log.info("  FP16: {}, Subgroup size: {}", .{ gpu_caps.fp16_supported, gpu_caps.subgroup_size });

        // Create logical device with compute queue
        const queue_priority: f32 = 1.0;
        const queue_create_info = vk.DeviceQueueCreateInfo{
            .queue_family_index = queue_family_index,
            .queue_count = 1,
            .p_queue_priorities = @ptrCast(&queue_priority),
        };

        const device = try vki.createDevice(physical_device, &.{
            .queue_create_info_count = 1,
            .p_queue_create_infos = @ptrCast(&queue_create_info),
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = null,
            .enabled_extension_count = 0,
            .pp_enabled_extension_names = null,
            .p_enabled_features = null,
        }, null);

        const vkd = DeviceDispatch.load(device, vki.dispatch.vkGetDeviceProcAddr) catch |err| {
            log.err("Failed to load device functions: {}", .{err});
            return error.VulkanLoadFailed;
        };
        errdefer vkd.destroyDevice(device, null);

        const compute_queue = vkd.getDeviceQueue(device, queue_family_index, 0);

        return Self{
            .instance = instance,
            .physical_device = physical_device,
            .device = device,
            .compute_queue = compute_queue,
            .queue_family_index = queue_family_index,
            .device_properties = device_properties,
            .gpu_caps = gpu_caps,
            .vki = vki,
            .vkd = vkd,
        };
    }

    pub fn deinit(self: *Self) void {
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
        self.* = undefined;
    }

    pub fn waitIdle(self: *const Self) !void {
        try self.vkd.deviceWaitIdle(self.device);
    }

    fn selectPhysicalDevice(
        allocator: std.mem.Allocator,
        vki: InstanceDispatch,
        instance: vk.Instance,
    ) !struct { vk.PhysicalDevice, u32 } {
        var device_count: u32 = 0;
        _ = try vki.enumeratePhysicalDevices(instance, &device_count, null);

        if (device_count == 0) {
            log.err("No Vulkan-capable GPU found", .{});
            return error.NoGpuFound;
        }

        const devices = try allocator.alloc(vk.PhysicalDevice, device_count);
        defer allocator.free(devices);
        _ = try vki.enumeratePhysicalDevices(instance, &device_count, devices.ptr);

        // Find device with compute queue, prefer discrete GPU
        var best_device: ?vk.PhysicalDevice = null;
        var best_queue_family: u32 = 0;
        var best_is_discrete = false;

        for (devices[0..device_count]) |pdev| {
            var queue_family_count: u32 = 0;
            vki.getPhysicalDeviceQueueFamilyProperties(pdev, &queue_family_count, null);

            const queue_families = try allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
            defer allocator.free(queue_families);
            vki.getPhysicalDeviceQueueFamilyProperties(pdev, &queue_family_count, queue_families.ptr);

            for (queue_families[0..queue_family_count], 0..) |qf, idx| {
                if (qf.queue_flags.compute_bit) {
                    const props = vki.getPhysicalDeviceProperties(pdev);
                    const is_discrete = props.device_type == .discrete_gpu;

                    if (best_device == null or (is_discrete and !best_is_discrete)) {
                        best_device = pdev;
                        best_queue_family = @intCast(idx);
                        best_is_discrete = is_discrete;
                    }
                    break;
                }
            }
        }

        if (best_device) |pdev| {
            return .{ pdev, best_queue_family };
        }

        log.err("No GPU with compute queue found", .{});
        return error.NoComputeQueue;
    }
};

/// Detect GPU vendor and architecture from device properties
fn detectGpuCapabilities(props: vk.PhysicalDeviceProperties) GpuCapabilities {
    var caps = GpuCapabilities{
        .vendor = .other,
        .amd_arch = .unknown,
        .fp16_supported = false,
        .subgroup_size = 32, // Default
        .device_name = undefined,
    };

    // Copy device name
    @memcpy(&caps.device_name, &props.device_name);

    // Detect vendor from vendor ID
    // AMD: 0x1002, NVIDIA: 0x10DE, Intel: 0x8086, Apple: 0x106B
    switch (props.vendor_id) {
        0x1002 => {
            caps.vendor = .amd;
            caps.subgroup_size = 64; // AMD wavefront is 64 (or 32 in wave32 mode)
            caps.amd_arch = detectAmdArchitecture(props);
            // FP16 native on RDNA2+
            caps.fp16_supported = (caps.amd_arch == .rdna2 or caps.amd_arch == .rdna3);
        },
        0x10DE => {
            caps.vendor = .nvidia;
            caps.subgroup_size = 32; // NVIDIA warp is 32
            caps.fp16_supported = true; // Most modern NVIDIA GPUs support FP16
        },
        0x8086 => {
            caps.vendor = .intel;
            caps.subgroup_size = 16; // Intel EU width varies, 16 is common
            caps.fp16_supported = true;
        },
        0x106B => {
            caps.vendor = .apple;
            caps.subgroup_size = 32;
            caps.fp16_supported = true;
        },
        else => {},
    }

    return caps;
}

/// Detect AMD GPU architecture from device ID
fn detectAmdArchitecture(props: vk.PhysicalDeviceProperties) AmdArch {
    const device_id = props.device_id;
    const name = std.mem.sliceTo(&props.device_name, 0);

    // Check device name for known patterns
    if (std.mem.indexOf(u8, name, "RX 7") != null or
        std.mem.indexOf(u8, name, "780M") != null or
        std.mem.indexOf(u8, name, "Radeon 780") != null)
    {
        return .rdna3;
    }

    if (std.mem.indexOf(u8, name, "RX 6") != null or
        std.mem.indexOf(u8, name, "680M") != null or
        std.mem.indexOf(u8, name, "Radeon 680") != null or
        std.mem.indexOf(u8, name, "Van Gogh") != null or // Steam Deck
        std.mem.indexOf(u8, name, "RADV VANGOGH") != null)
    {
        return .rdna2;
    }

    if (std.mem.indexOf(u8, name, "RX 5") != null or
        std.mem.indexOf(u8, name, "5700") != null or
        std.mem.indexOf(u8, name, "5600") != null or
        std.mem.indexOf(u8, name, "5500") != null)
    {
        return .rdna1;
    }

    if (std.mem.indexOf(u8, name, "Vega") != null or
        std.mem.indexOf(u8, name, "VEGA") != null or
        std.mem.indexOf(u8, name, "gfx900") != null or
        std.mem.indexOf(u8, name, "gfx906") != null)
    {
        return .gcn5;
    }

    if (std.mem.indexOf(u8, name, "RX 5") != null or
        std.mem.indexOf(u8, name, "RX 4") != null or
        std.mem.indexOf(u8, name, "580") != null or
        std.mem.indexOf(u8, name, "570") != null or
        std.mem.indexOf(u8, name, "480") != null or
        std.mem.indexOf(u8, name, "Polaris") != null)
    {
        return .gcn4;
    }

    if (std.mem.indexOf(u8, name, "MI300") != null or
        std.mem.indexOf(u8, name, "MI250") != null or
        std.mem.indexOf(u8, name, "MI200") != null or
        std.mem.indexOf(u8, name, "MI100") != null or
        std.mem.indexOf(u8, name, "gfx90a") != null or
        std.mem.indexOf(u8, name, "gfx942") != null)
    {
        return .cdna;
    }

    // Try to detect from device ID ranges (less reliable)
    // RDNA3: 0x744x, 0x745x
    if (device_id >= 0x7440 and device_id <= 0x745F) return .rdna3;
    // RDNA2: 0x73xx
    if (device_id >= 0x7300 and device_id <= 0x73FF) return .rdna2;
    // RDNA1: 0x731x, 0x7340
    if (device_id >= 0x7310 and device_id <= 0x7350) return .rdna1;

    return .unknown;
}

// Get the vkGetInstanceProcAddr function from the Vulkan loader
var vk_lib: ?std.DynLib = null;

fn getLoaderFn() vk.PfnGetInstanceProcAddr {
    if (vk_lib == null) {
        vk_lib = std.DynLib.open("libvulkan.so.1") catch |err| {
            log.err("Failed to load Vulkan library: {}", .{err});
            @panic("Failed to load Vulkan");
        };
    }
    return vk_lib.?.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse @panic("vkGetInstanceProcAddr not found");
}

// API specification for vulkan-zig wrappers
const apis: []const vk.ApiInfo = &.{
    vk.ApiInfo{
        .base_commands = .{
            .createInstance = true,
            .getInstanceProcAddr = true,
        },
        .instance_commands = .{
            .destroyInstance = true,
            .enumeratePhysicalDevices = true,
            .getPhysicalDeviceProperties = true,
            .getPhysicalDeviceQueueFamilyProperties = true,
            .getPhysicalDeviceMemoryProperties = true,
            .createDevice = true,
            .getDeviceProcAddr = true,
        },
        .device_commands = .{
            .destroyDevice = true,
            .getDeviceQueue = true,
            .deviceWaitIdle = true,
            .createCommandPool = true,
            .destroyCommandPool = true,
            .allocateCommandBuffers = true,
            .freeCommandBuffers = true,
            .beginCommandBuffer = true,
            .endCommandBuffer = true,
            .resetCommandBuffer = true,
            .queueSubmit = true,
            .queueWaitIdle = true,
            .createFence = true,
            .destroyFence = true,
            .waitForFences = true,
            .resetFences = true,
            .createBuffer = true,
            .destroyBuffer = true,
            .getBufferMemoryRequirements = true,
            .allocateMemory = true,
            .freeMemory = true,
            .bindBufferMemory = true,
            .mapMemory = true,
            .unmapMemory = true,
            .flushMappedMemoryRanges = true,
            .invalidateMappedMemoryRanges = true,
            .createShaderModule = true,
            .destroyShaderModule = true,
            .createPipelineLayout = true,
            .destroyPipelineLayout = true,
            .createComputePipelines = true,
            .destroyPipeline = true,
            .createDescriptorSetLayout = true,
            .destroyDescriptorSetLayout = true,
            .createDescriptorPool = true,
            .destroyDescriptorPool = true,
            .allocateDescriptorSets = true,
            .updateDescriptorSets = true,
            .cmdBindPipeline = true,
            .cmdBindDescriptorSets = true,
            .cmdDispatch = true,
            .cmdPipelineBarrier = true,
            .cmdPushConstants = true,
            .cmdCopyBuffer = true,
        },
    },
};

// Vulkan dispatch tables
const BaseDispatch = vk.BaseWrapper(apis);
const InstanceDispatch = vk.InstanceWrapper(apis);
const DeviceDispatch = vk.DeviceWrapper(apis);

pub const InstanceDispatchType = InstanceDispatch;
pub const DeviceDispatchType = DeviceDispatch;
