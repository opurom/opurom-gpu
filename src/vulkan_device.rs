//! Real Vulkan compute execution for display lists. The recorder in
//! `vulkan.rs` stays the byte-deterministic transport contract; this module
//! runs the same ordered pass plan as Metal over the committed SPIR-V
//! kernels, with a memory barrier per pass and a host-visible readback. All
//! kernel arithmetic is integer, so the pixels must match the CPU oracle and
//! the Metal readback byte for byte. The loader is the P1 hand-rolled dlopen
//! probe; no crate dependency is added here.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::vulkan_loader::{
    resolve, PhysicalDeviceFeatures, VulkanLibrary, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_QUEUE_COMPUTE_BIT,
};
use crate::{
    display_byte_count, DisplayList, GpuError, KERNEL_BLUR_V, KERNEL_EFFECT, KERNEL_GAUSSIAN_V,
    PARAM_WORDS,
};

const VK_SUCCESS: i32 = 0;
const VK_API_VERSION_1_0: u32 = 1 << 22;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 46;

const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x20;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_ACCESS_MEMORY_READ_BIT: u32 = 0x8000;
const VK_ACCESS_MEMORY_WRITE_BIT: u32 = 0x10000;
const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x800;
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x20;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x2;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x4;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x1;
const VK_WHOLE_SIZE: u64 = u64::MAX;
const LOCAL_SIZE: u32 = 64;

#[repr(C)]
struct ApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const std::ffi::c_char,
    application_version: u32,
    p_engine_name: *const std::ffi::c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct InstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const ApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const std::ffi::c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const std::ffi::c_char,
}

#[repr(C)]
struct DeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct DeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_create_info_count: u32,
    p_queue_create_infos: *const DeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const std::ffi::c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const std::ffi::c_char,
    p_enabled_features: *const PhysicalDeviceFeatures,
}

/// Prefix of `VkPhysicalDeviceProperties`; the tail keeps the driver's whole
/// struct write inside the allocation.
#[repr(C, align(8))]
struct Properties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; 256],
    pipeline_cache_uuid: [u8; 16],
    tail: [u64; 96],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueFamily {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryType {
    property_flags: u32,
    heap_index: u32,
}

#[repr(C)]
struct MemoryProperties {
    memory_type_count: u32,
    memory_types: [MemoryType; 32],
    memory_heap_count: u32,
    memory_heaps: [MemoryHeap; 16],
}

#[repr(C)]
struct MemoryHeap {
    size: u64,
    flags: u32,
}

#[repr(C)]
struct BufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct MemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}

#[repr(C)]
struct MemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    allocation_size: u64,
    memory_type_index: u32,
}

#[repr(C)]
struct DescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    p_immutable_samplers: *const c_void,
}

#[repr(C)]
struct DescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    p_bindings: *const DescriptorSetLayoutBinding,
}

#[repr(C)]
struct DescriptorPoolSize {
    descriptor_type: u32,
    descriptor_count: u32,
}

#[repr(C)]
struct DescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: *const DescriptorPoolSize,
}

#[repr(C)]
struct DescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    descriptor_pool: *mut c_void,
    descriptor_set_count: u32,
    p_set_layouts: *const *mut c_void,
}

#[repr(C)]
struct DescriptorBufferInfo {
    buffer: *mut c_void,
    offset: u64,
    range: u64,
}

#[repr(C)]
struct WriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: *mut c_void,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    p_image_info: *const c_void,
    p_buffer_info: *const DescriptorBufferInfo,
    p_texel_buffer_view: *const c_void,
}

#[repr(C)]
struct ShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: u64,
    p_code: *const u32,
}

#[repr(C)]
struct PushConstantRange {
    stage_flags: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct PipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    p_set_layouts: *const *mut c_void,
    push_constant_range_count: u32,
    p_push_constant_ranges: *const PushConstantRange,
}

#[repr(C)]
struct PipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: *mut c_void,
    p_name: *const std::ffi::c_char,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct ComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: PipelineShaderStageCreateInfo,
    layout: *mut c_void,
    base_pipeline_handle: *mut c_void,
    base_pipeline_index: i32,
}

#[repr(C)]
struct CommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}

#[repr(C)]
struct CommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    command_pool: *mut c_void,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct CommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
struct SubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const c_void,
    p_wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    p_command_buffers: *const *mut c_void,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const c_void,
}

#[repr(C)]
struct FenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
struct MemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access_mask: u32,
    dst_access_mask: u32,
}

macro_rules! vk_fn {
    ($library:expr, $instance:expr, $name:literal, $ty:ty) => {
        resolve::<$ty>($library, $instance, $name)
            .map_err(|reason| GpuError::Vulkan(reason.to_owned()))?
    };
}

/// Bound buffer plus the host mapping kept for upload and readback.
struct GpuBuffer {
    buffer: *mut c_void,
    memory: *mut c_void,
    mapped: *mut c_void,
    size: usize,
}

struct SessionFns {
    enumerate: unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32,
    get_properties: unsafe extern "system" fn(*mut c_void, *mut Properties),
    get_features: unsafe extern "system" fn(*mut c_void, *mut PhysicalDeviceFeatures),
    get_memory_properties: unsafe extern "system" fn(*mut c_void, *mut MemoryProperties),
    get_queue_families: unsafe extern "system" fn(*mut c_void, *mut u32, *mut QueueFamily),
    create_device: unsafe extern "system" fn(
        *mut c_void,
        *const DeviceCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    get_device_queue: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void),
    create_buffer: unsafe extern "system" fn(
        *mut c_void,
        *const BufferCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    get_buffer_requirements:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut MemoryRequirements),
    allocate_memory: unsafe extern "system" fn(
        *mut c_void,
        *const MemoryAllocateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    bind_buffer_memory:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, u64) -> i32,
    map_memory:
        unsafe extern "system" fn(*mut c_void, *mut c_void, u64, u64, u32, *mut *mut c_void) -> i32,
    unmap_memory: unsafe extern "system" fn(*mut c_void, *mut c_void),
    free_memory: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_buffer: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    create_set_layout: unsafe extern "system" fn(
        *mut c_void,
        *const DescriptorSetLayoutCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    create_pool: unsafe extern "system" fn(
        *mut c_void,
        *const DescriptorPoolCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    allocate_sets: unsafe extern "system" fn(
        *mut c_void,
        *const DescriptorSetAllocateInfo,
        *mut *mut c_void,
    ) -> i32,
    update_sets:
        unsafe extern "system" fn(*mut c_void, u32, *const WriteDescriptorSet, u32, *const c_void),
    create_module: unsafe extern "system" fn(
        *mut c_void,
        *const ShaderModuleCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    create_pipeline_layout: unsafe extern "system" fn(
        *mut c_void,
        *const PipelineLayoutCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    create_pipelines: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        *const ComputePipelineCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    create_command_pool: unsafe extern "system" fn(
        *mut c_void,
        *const CommandPoolCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    allocate_commands: unsafe extern "system" fn(
        *mut c_void,
        *const CommandBufferAllocateInfo,
        *mut *mut c_void,
    ) -> i32,
    begin_command:
        unsafe extern "system" fn(*mut c_void, u32, *const CommandBufferBeginInfo) -> i32,
    end_command: unsafe extern "system" fn(*mut c_void) -> i32,
    bind_pipeline: unsafe extern "system" fn(*mut c_void, u32, *mut c_void),
    bind_sets: unsafe extern "system" fn(
        *mut c_void,
        u32,
        *mut c_void,
        u32,
        u32,
        *const *mut c_void,
        u32,
        *const u32,
    ),
    push_constants:
        unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *const c_void),
    dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32),
    barrier: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        *const MemoryBarrier,
        u32,
        *const c_void,
        u32,
        *const c_void,
    ),
    create_fence: unsafe extern "system" fn(
        *mut c_void,
        *const FenceCreateInfo,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    submit: unsafe extern "system" fn(*mut c_void, u32, *const SubmitInfo, *mut c_void) -> i32,
    wait_fences: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, u32, u64) -> i32,
    destroy_fence: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_command_pool: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_pool: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_set_layout: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_pipeline_layout: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_pipeline: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_module: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void),
    destroy_device: unsafe extern "system" fn(*mut c_void, *const c_void),
    destroy_instance: unsafe extern "system" fn(*mut c_void, *const c_void),
}

/// A live device with the eight committed kernels ready to dispatch.
// SAFETY: the session's raw device handles are only touched under the Mutex
// in `SESSION`, and the loader handle stays alive until the session drops.
unsafe impl Send for VulkanSession {}

pub(crate) struct VulkanSession {
    // Kept alive for the process lifetime: closing the dlopen handle while a
    // device or instance still exists would invalidate resolved entry points.
    _library: VulkanLibrary,
    instance: *mut c_void,
    device: *mut c_void,
    queue: *mut c_void,
    queue_family: u32,
    memory_types: Vec<u32>,
    set_layout: *mut c_void,
    pipeline_layout: *mut c_void,
    pipelines: [*mut c_void; 8],
    functions: SessionFns,
    device_name: String,
}

static SESSION: OnceLock<Result<Mutex<VulkanSession>, String>> = OnceLock::new();

/// Execute one display list with Vulkan compute and return raw RGBA8 bytes.
pub(crate) fn execute(list: &DisplayList) -> Result<Vec<u8>, GpuError> {
    let session = SESSION
        .get_or_init(|| {
            VulkanSession::acquire()
                .map(Mutex::new)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|reason| GpuError::Vulkan(reason.clone()))?;
    let session = session
        .lock()
        .map_err(|_| GpuError::Vulkan("vulkan session poisoned".to_owned()))?;
    session.render(list)
}

impl VulkanSession {
    fn acquire() -> Result<Self, GpuError> {
        let library =
            VulkanLibrary::load().map_err(|reason| GpuError::Vulkan(reason.to_owned()))?;
        // vkCreateInstance is a global command: resolve it with a null
        // instance, then resolve every other entry point with the real
        // instance handle (Vulkan requires this for instance-level commands).
        let create_instance = resolve::<
            unsafe extern "system" fn(
                *const InstanceCreateInfo,
                *const c_void,
                *mut *mut c_void,
            ) -> i32,
        >(&library, std::ptr::null_mut(), "vkCreateInstance")
        .map_err(|reason| GpuError::Vulkan(reason.to_owned()))?;

        let application = ApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: b"opurom-gpu\0".as_ptr().cast::<std::ffi::c_char>(),
            application_version: 0,
            p_engine_name: std::ptr::null(),
            engine_version: 0,
            api_version: VK_API_VERSION_1_0,
        };
        let create_info = InstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            p_application_info: &application,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };
        let mut instance: *mut c_void = std::ptr::null_mut();
        // SAFETY: a valid instance create info with no extensions or layers.
        let result = unsafe { create_instance(&create_info, std::ptr::null(), &mut instance) };
        if result != VK_SUCCESS || instance.is_null() {
            return Err(GpuError::Vulkan(format!(
                "vkCreateInstance failed: {result}"
            )));
        }
        let functions = SessionFns {
            enumerate: vk_fn!(
                &library,
                instance,
                "vkEnumeratePhysicalDevices",
                unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32
            ),
            get_properties: vk_fn!(
                &library,
                instance,
                "vkGetPhysicalDeviceProperties",
                unsafe extern "system" fn(*mut c_void, *mut Properties)
            ),
            get_features: vk_fn!(
                &library,
                instance,
                "vkGetPhysicalDeviceFeatures",
                unsafe extern "system" fn(*mut c_void, *mut PhysicalDeviceFeatures)
            ),
            get_memory_properties: vk_fn!(
                &library,
                instance,
                "vkGetPhysicalDeviceMemoryProperties",
                unsafe extern "system" fn(*mut c_void, *mut MemoryProperties)
            ),
            get_queue_families: vk_fn!(
                &library,
                instance,
                "vkGetPhysicalDeviceQueueFamilyProperties",
                unsafe extern "system" fn(*mut c_void, *mut u32, *mut QueueFamily)
            ),
            create_device: vk_fn!(
                &library,
                instance,
                "vkCreateDevice",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const DeviceCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            get_device_queue: vk_fn!(
                &library,
                instance,
                "vkGetDeviceQueue",
                unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void)
            ),
            create_buffer: vk_fn!(
                &library,
                instance,
                "vkCreateBuffer",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const BufferCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            get_buffer_requirements: vk_fn!(
                &library,
                instance,
                "vkGetBufferMemoryRequirements",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *mut MemoryRequirements)
            ),
            allocate_memory: vk_fn!(
                &library,
                instance,
                "vkAllocateMemory",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const MemoryAllocateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            bind_buffer_memory: vk_fn!(
                &library,
                instance,
                "vkBindBufferMemory",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, u64) -> i32
            ),
            map_memory: vk_fn!(
                &library,
                instance,
                "vkMapMemory",
                unsafe extern "system" fn(
                    *mut c_void,
                    *mut c_void,
                    u64,
                    u64,
                    u32,
                    *mut *mut c_void,
                ) -> i32
            ),
            unmap_memory: vk_fn!(
                &library,
                instance,
                "vkUnmapMemory",
                unsafe extern "system" fn(*mut c_void, *mut c_void)
            ),
            free_memory: vk_fn!(
                &library,
                instance,
                "vkFreeMemory",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_buffer: vk_fn!(
                &library,
                instance,
                "vkDestroyBuffer",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            create_set_layout: vk_fn!(
                &library,
                instance,
                "vkCreateDescriptorSetLayout",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const DescriptorSetLayoutCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            create_pool: vk_fn!(
                &library,
                instance,
                "vkCreateDescriptorPool",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const DescriptorPoolCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            allocate_sets: vk_fn!(
                &library,
                instance,
                "vkAllocateDescriptorSets",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const DescriptorSetAllocateInfo,
                    *mut *mut c_void,
                ) -> i32
            ),
            update_sets: vk_fn!(
                &library,
                instance,
                "vkUpdateDescriptorSets",
                unsafe extern "system" fn(
                    *mut c_void,
                    u32,
                    *const WriteDescriptorSet,
                    u32,
                    *const c_void,
                )
            ),
            create_module: vk_fn!(
                &library,
                instance,
                "vkCreateShaderModule",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const ShaderModuleCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            create_pipeline_layout: vk_fn!(
                &library,
                instance,
                "vkCreatePipelineLayout",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const PipelineLayoutCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            create_pipelines: vk_fn!(
                &library,
                instance,
                "vkCreateComputePipelines",
                unsafe extern "system" fn(
                    *mut c_void,
                    *mut c_void,
                    u32,
                    *const ComputePipelineCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            create_command_pool: vk_fn!(
                &library,
                instance,
                "vkCreateCommandPool",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const CommandPoolCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            allocate_commands: vk_fn!(
                &library,
                instance,
                "vkAllocateCommandBuffers",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const CommandBufferAllocateInfo,
                    *mut *mut c_void,
                ) -> i32
            ),
            begin_command: vk_fn!(
                &library,
                instance,
                "vkBeginCommandBuffer",
                unsafe extern "system" fn(*mut c_void, u32, *const CommandBufferBeginInfo) -> i32
            ),
            end_command: vk_fn!(
                &library,
                instance,
                "vkEndCommandBuffer",
                unsafe extern "system" fn(*mut c_void) -> i32
            ),
            bind_pipeline: vk_fn!(
                &library,
                instance,
                "vkCmdBindPipeline",
                unsafe extern "system" fn(*mut c_void, u32, *mut c_void)
            ),
            bind_sets: vk_fn!(
                &library,
                instance,
                "vkCmdBindDescriptorSets",
                unsafe extern "system" fn(
                    *mut c_void,
                    u32,
                    *mut c_void,
                    u32,
                    u32,
                    *const *mut c_void,
                    u32,
                    *const u32,
                )
            ),
            push_constants: vk_fn!(
                &library,
                instance,
                "vkCmdPushConstants",
                unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *const c_void)
            ),
            dispatch: vk_fn!(
                &library,
                instance,
                "vkCmdDispatch",
                unsafe extern "system" fn(*mut c_void, u32, u32, u32)
            ),
            barrier: vk_fn!(
                &library,
                instance,
                "vkCmdPipelineBarrier",
                unsafe extern "system" fn(
                    *mut c_void,
                    u32,
                    u32,
                    u32,
                    u32,
                    *const MemoryBarrier,
                    u32,
                    *const c_void,
                    u32,
                    *const c_void,
                )
            ),
            create_fence: vk_fn!(
                &library,
                instance,
                "vkCreateFence",
                unsafe extern "system" fn(
                    *mut c_void,
                    *const FenceCreateInfo,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32
            ),
            submit: vk_fn!(
                &library,
                instance,
                "vkQueueSubmit",
                unsafe extern "system" fn(*mut c_void, u32, *const SubmitInfo, *mut c_void) -> i32
            ),
            wait_fences: vk_fn!(
                &library,
                instance,
                "vkWaitForFences",
                unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, u32, u64) -> i32
            ),
            destroy_fence: vk_fn!(
                &library,
                instance,
                "vkDestroyFence",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_command_pool: vk_fn!(
                &library,
                instance,
                "vkDestroyCommandPool",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_pool: vk_fn!(
                &library,
                instance,
                "vkDestroyDescriptorPool",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_set_layout: vk_fn!(
                &library,
                instance,
                "vkDestroyDescriptorSetLayout",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_pipeline_layout: vk_fn!(
                &library,
                instance,
                "vkDestroyPipelineLayout",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_pipeline: vk_fn!(
                &library,
                instance,
                "vkDestroyPipeline",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_module: vk_fn!(
                &library,
                instance,
                "vkDestroyShaderModule",
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void)
            ),
            destroy_device: vk_fn!(
                &library,
                instance,
                "vkDestroyDevice",
                unsafe extern "system" fn(*mut c_void, *const c_void)
            ),
            destroy_instance: vk_fn!(
                &library,
                instance,
                "vkDestroyInstance",
                unsafe extern "system" fn(*mut c_void, *const c_void)
            ),
        };
        let mut session = Self {
            _library: library,
            instance,
            device: std::ptr::null_mut(),
            queue: std::ptr::null_mut(),
            queue_family: 0,
            memory_types: Vec::new(),
            set_layout: std::ptr::null_mut(),
            pipeline_layout: std::ptr::null_mut(),
            pipelines: [std::ptr::null_mut(); 8],
            functions,
            device_name: String::new(),
        };
        session.init_device()?;
        session.init_pipelines()?;
        Ok(session)
    }

    fn init_device(&mut self) -> Result<(), GpuError> {
        let f = &self.functions;
        let mut count = 0u32;
        // SAFETY: the count call writes only the count.
        let result = unsafe { (f.enumerate)(self.instance, &mut count, std::ptr::null_mut()) };
        if result != VK_SUCCESS || count == 0 {
            return Err(GpuError::Vulkan("no Vulkan device on this host".to_owned()));
        }
        let mut devices = vec![std::ptr::null_mut(); count as usize];
        // SAFETY: the vector holds exactly `count` slots.
        let result = unsafe { (f.enumerate)(self.instance, &mut count, devices.as_mut_ptr()) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan(
                "vkEnumeratePhysicalDevices failed".to_owned(),
            ));
        }
        let mut best: Option<(usize, Properties)> = None;
        for (index, &device) in devices.iter().enumerate() {
            let mut properties = std::mem::MaybeUninit::<Properties>::zeroed();
            // SAFETY: the output buffer is larger than the real struct.
            unsafe { (f.get_properties)(device, properties.as_mut_ptr()) };
            let properties = unsafe { properties.assume_init() };
            if best.is_none() || properties.device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
                best = Some((index, properties));
            }
        }
        let (index, properties) = best.expect("at least one device was enumerated");
        let physical = devices[index];
        let bytes = &properties.device_name[..properties
            .device_name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(properties.device_name.len())];
        self.device_name = String::from_utf8_lossy(bytes).into_owned();

        let mut features = std::mem::MaybeUninit::<PhysicalDeviceFeatures>::zeroed();
        // SAFETY: output buffer has the frozen 55-boolean 1.0 layout.
        unsafe { (f.get_features)(physical, features.as_mut_ptr()) };
        let features = unsafe { features.assume_init() };
        if features.shader_int64 == 0 {
            return Err(GpuError::Vulkan(
                "Vulkan device lacks 64-bit integer shaders".to_owned(),
            ));
        }
        let mut enabled: PhysicalDeviceFeatures = unsafe { std::mem::zeroed() };
        enabled.shader_int64 = 1;

        let mut family_count = 0u32;
        // SAFETY: the count call writes only the count.
        unsafe { (f.get_queue_families)(physical, &mut family_count, std::ptr::null_mut()) };
        let mut families = vec![
            QueueFamily {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: [0; 3],
            };
            family_count as usize
        ];
        if family_count > 0 {
            // SAFETY: the vector holds exactly `family_count` slots.
            unsafe { (f.get_queue_families)(physical, &mut family_count, families.as_mut_ptr()) };
        }
        let Some(family) = families.iter().position(|family| {
            family.queue_flags & VK_QUEUE_COMPUTE_BIT != 0 && family.queue_count > 0
        }) else {
            return Err(GpuError::Vulkan(
                "no Vulkan compute queue on this host".to_owned(),
            ));
        };
        self.queue_family = family as u32;

        let mut memory_properties = std::mem::MaybeUninit::<MemoryProperties>::zeroed();
        // SAFETY: output buffer has the full memory-properties layout.
        unsafe { (f.get_memory_properties)(physical, memory_properties.as_mut_ptr()) };
        let memory_properties = unsafe { memory_properties.assume_init() };
        self.memory_types = memory_properties.memory_types
            [..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .filter(|(_, kind)| {
                kind.property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0
                    && kind.property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0
            })
            .map(|(index, _)| index as u32)
            .collect();
        if self.memory_types.is_empty() {
            return Err(GpuError::Vulkan(
                "no host-visible Vulkan memory on this host".to_owned(),
            ));
        }

        let priority = [1.0f32];
        let queue_info = DeviceQueueCreateInfo {
            s_type: 2,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: self.queue_family,
            queue_count: 1,
            p_queue_priorities: priority.as_ptr(),
        };
        let device_info = DeviceCreateInfo {
            s_type: 3,
            p_next: std::ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            p_enabled_features: &enabled,
        };
        // SAFETY: the chosen family exists and shaderInt64 was queried.
        let result = unsafe {
            (f.create_device)(physical, &device_info, std::ptr::null(), &mut self.device)
        };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan(format!("vkCreateDevice failed: {result}")));
        }
        // SAFETY: family and index exist on the created device.
        unsafe { (f.get_device_queue)(self.device, self.queue_family, 0, &mut self.queue) };
        if self.queue.is_null() {
            return Err(GpuError::Vulkan(
                "Vulkan compute queue could not be fetched".to_owned(),
            ));
        }
        Ok(())
    }

    fn init_pipelines(&mut self) -> Result<(), GpuError> {
        let bindings: Vec<DescriptorSetLayoutBinding> = (0..5)
            .map(|binding| DescriptorSetLayoutBinding {
                binding,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: std::ptr::null(),
            })
            .collect();
        let layout_info = DescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let f = &self.functions;
        // SAFETY: five valid storage-buffer bindings for the compute stage.
        let result = unsafe {
            (f.create_set_layout)(
                self.device,
                &layout_info,
                std::ptr::null(),
                &mut self.set_layout,
            )
        };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan(
                "vkCreateDescriptorSetLayout failed".to_owned(),
            ));
        }
        let range = PushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: (PARAM_WORDS * 4) as u32,
        };
        let pipeline_layout_info = PipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &self.set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &range,
        };
        // SAFETY: the set layout exists and the 112-byte push range is in limits.
        let result = unsafe {
            (f.create_pipeline_layout)(
                self.device,
                &pipeline_layout_info,
                std::ptr::null(),
                &mut self.pipeline_layout,
            )
        };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkCreatePipelineLayout failed".to_owned()));
        }

        let blobs: [&[u8]; 8] = [
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/raster_batch.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/snapshot_frame.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/build_mask.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/effect_local.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/blur_horizontal.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/blur_vertical.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/gaussian_horizontal.spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spirv/gaussian_vertical.spv"
            )),
        ];
        for (index, blob) in blobs.iter().enumerate() {
            // Shader modules need aligned u32 words; SPIR-V file bytes are LE.
            let words: Vec<u32> = blob
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let module_info = ShaderModuleCreateInfo {
                s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                code_size: words.len() as u64 * 4,
                p_code: words.as_ptr(),
            };
            let mut module: *mut c_void = std::ptr::null_mut();
            // SAFETY: the words are aligned and were checked as SPIR-V by the
            // artifact test; the driver copies them during module creation.
            let result = unsafe {
                (f.create_module)(self.device, &module_info, std::ptr::null(), &mut module)
            };
            if result != VK_SUCCESS {
                return Err(GpuError::Vulkan(format!(
                    "vkCreateShaderModule {index} failed: {result}"
                )));
            }
            let stage = PipelineShaderStageCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module,
                p_name: b"main\0".as_ptr().cast::<std::ffi::c_char>(),
                p_specialization_info: std::ptr::null(),
            };
            let pipeline_info = ComputePipelineCreateInfo {
                s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                stage,
                layout: self.pipeline_layout,
                base_pipeline_handle: std::ptr::null_mut(),
                base_pipeline_index: 0,
            };
            // SAFETY: a complete compute-stage description with an aligned
            // SPIR-V module and a valid pipeline layout.
            let result = unsafe {
                (f.create_pipelines)(
                    self.device,
                    std::ptr::null_mut(),
                    1,
                    &pipeline_info,
                    std::ptr::null(),
                    &mut self.pipelines[index],
                )
            };
            // SAFETY: the module is no longer needed after pipeline creation.
            unsafe { (f.destroy_module)(self.device, module, std::ptr::null()) };
            if result != VK_SUCCESS {
                return Err(GpuError::Vulkan(format!(
                    "vkCreateComputePipelines {index} failed: {result}"
                )));
            }
        }
        Ok(())
    }

    fn make_buffer(&self, size: usize) -> Result<GpuBuffer, GpuError> {
        let info = BufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size: size as u64,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharing_mode: 0,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };
        let f = &self.functions;
        let mut buffer = std::ptr::null_mut();
        // SAFETY: a valid storage buffer description with no queue sharing.
        let result =
            unsafe { (f.create_buffer)(self.device, &info, std::ptr::null(), &mut buffer) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkCreateBuffer failed".to_owned()));
        }
        let mut requirements = MemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: both handles come from this device.
        unsafe { (f.get_buffer_requirements)(self.device, buffer, &mut requirements) };
        let Some(&memory_type) = self
            .memory_types
            .iter()
            .find(|&&index| requirements.memory_type_bits & (1 << index) != 0)
        else {
            // SAFETY: cleanup the buffer created above before failing.
            unsafe { (f.destroy_buffer)(self.device, buffer, std::ptr::null()) };
            return Err(GpuError::Vulkan(
                "no host-visible memory fits a frame buffer".to_owned(),
            ));
        };
        let allocate = MemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: requirements.size,
            memory_type_index: memory_type,
        };
        let mut memory = std::ptr::null_mut();
        // SAFETY: the allocation size comes from the driver's requirements.
        let result =
            unsafe { (f.allocate_memory)(self.device, &allocate, std::ptr::null(), &mut memory) };
        if result != VK_SUCCESS {
            unsafe { (f.destroy_buffer)(self.device, buffer, std::ptr::null()) };
            return Err(GpuError::Vulkan("vkAllocateMemory failed".to_owned()));
        }
        // SAFETY: the memory type was chosen as host-visible and coherent.
        let result = unsafe { (f.bind_buffer_memory)(self.device, buffer, memory, 0) };
        if result != VK_SUCCESS {
            unsafe {
                (f.free_memory)(self.device, memory, std::ptr::null());
                (f.destroy_buffer)(self.device, buffer, std::ptr::null());
            }
            return Err(GpuError::Vulkan("vkBindBufferMemory failed".to_owned()));
        }
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the whole allocation is host-visible and not yet mapped.
        let result =
            unsafe { (f.map_memory)(self.device, memory, 0, VK_WHOLE_SIZE, 0, &mut mapped) };
        if result != VK_SUCCESS {
            unsafe {
                (f.free_memory)(self.device, memory, std::ptr::null());
                (f.destroy_buffer)(self.device, buffer, std::ptr::null());
            }
            return Err(GpuError::Vulkan("vkMapMemory failed".to_owned()));
        }
        Ok(GpuBuffer {
            buffer,
            memory,
            mapped,
            size,
        })
    }

    fn render(&self, list: &DisplayList) -> Result<Vec<u8>, GpuError> {
        let byte_count = display_byte_count(list.width, list.height)?;
        let (packed, passes) = list.build_passes()?;
        let pixel_count = byte_count / 4;
        let f = &self.functions;

        let frame_a = self.make_buffer(byte_count)?;
        let frame_b = self.make_buffer(byte_count)?;
        let backdrop = self.make_buffer(byte_count)?;
        let mask = self.make_buffer(byte_count)?;
        let command_bytes = packed.len() * 4;
        let commands = self.make_buffer(command_bytes.max(4))?;

        // SAFETY: mapped host-visible allocations with the exact byte sizes.
        unsafe {
            let background = u32::from_le_bytes(list.background);
            for index in 0..pixel_count {
                (frame_a.mapped as *mut u32).add(index).write(background);
            }
            (commands.mapped as *mut u32).copy_from(packed.as_ptr(), packed.len());
        }
        let buffers = [&frame_a, &frame_b, &backdrop, &mask, &commands];

        let pool_size = DescriptorPoolSize {
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 10,
        };
        let pool_info = DescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 2,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut pool = std::ptr::null_mut();
        // SAFETY: two descriptor sets, ten storage-buffer descriptors.
        let result =
            unsafe { (f.create_pool)(self.device, &pool_info, std::ptr::null(), &mut pool) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkCreateDescriptorPool failed".to_owned()));
        }
        let layouts = [self.set_layout, self.set_layout];
        let allocate_sets = DescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 2,
            p_set_layouts: layouts.as_ptr(),
        };
        let mut sets = [std::ptr::null_mut(); 2];
        // SAFETY: the pool can supply exactly two sets of this layout.
        let result = unsafe { (f.allocate_sets)(self.device, &allocate_sets, sets.as_mut_ptr()) };
        if result != VK_SUCCESS {
            unsafe { (f.destroy_pool)(self.device, pool, std::ptr::null()) };
            return Err(GpuError::Vulkan(
                "vkAllocateDescriptorSets failed".to_owned(),
            ));
        }
        let infos: Vec<DescriptorBufferInfo> = buffers
            .iter()
            .map(|buffer| DescriptorBufferInfo {
                buffer: buffer.buffer,
                offset: 0,
                range: buffer.size as u64,
            })
            .collect();
        let mut writes = Vec::with_capacity(10);
        for (set_index, order) in [[0usize, 1], [1, 0]].into_iter().enumerate() {
            for binding in 0..5usize {
                let info_index = match binding {
                    0 => order[0],
                    1 => order[1],
                    other => other,
                };
                writes.push(WriteDescriptorSet {
                    s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: sets[set_index],
                    dst_binding: binding as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: std::ptr::null(),
                    p_buffer_info: &infos[info_index],
                    p_texel_buffer_view: std::ptr::null(),
                });
            }
        }
        // SAFETY: the buffer infos outlive this call; all sets/pool are live.
        unsafe {
            (f.update_sets)(
                self.device,
                writes.len() as u32,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            )
        };

        let pool_info = CommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: self.queue_family,
        };
        let mut command_pool = std::ptr::null_mut();
        // SAFETY: a valid queue family index from device creation.
        let result = unsafe {
            (f.create_command_pool)(self.device, &pool_info, std::ptr::null(), &mut command_pool)
        };
        if result != VK_SUCCESS {
            unsafe { (f.destroy_pool)(self.device, pool, std::ptr::null()) };
            return Err(GpuError::Vulkan("vkCreateCommandPool failed".to_owned()));
        }
        let command_info = CommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool,
            level: 0,
            command_buffer_count: 1,
        };
        let mut command = std::ptr::null_mut();
        // SAFETY: the pool has room for one primary command buffer.
        let result = unsafe { (f.allocate_commands)(self.device, &command_info, &mut command) };
        if result != VK_SUCCESS {
            unsafe {
                (f.destroy_command_pool)(self.device, command_pool, std::ptr::null());
                (f.destroy_pool)(self.device, pool, std::ptr::null());
            }
            return Err(GpuError::Vulkan(
                "vkAllocateCommandBuffers failed".to_owned(),
            ));
        }
        let begin_info = CommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: std::ptr::null(),
        };
        // SAFETY: the command buffer was allocated from a live pool.
        let result = unsafe { (f.begin_command)(command, 0, &begin_info) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkBeginCommandBuffer failed".to_owned()));
        }

        let barrier = MemoryBarrier {
            s_type: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: VK_ACCESS_MEMORY_WRITE_BIT,
            dst_access_mask: VK_ACCESS_MEMORY_READ_BIT,
        };
        let groups = pixel_count.div_ceil(LOCAL_SIZE as usize) as u32;
        let mut current = 0usize;
        for pass in &passes {
            let vertical = pass.kernel == KERNEL_BLUR_V || pass.kernel == KERNEL_GAUSSIAN_V;
            let set = sets[current ^ usize::from(vertical)];
            // SAFETY: all pipelines, descriptor sets and push constants are
            // live; bindings mirror the Metal ping-pong, including the
            // vertical-pass buffer-role reversal.
            unsafe {
                (f.bind_pipeline)(command, 3, self.pipelines[pass.kernel as usize]);
                (f.bind_sets)(
                    command,
                    3,
                    self.pipeline_layout,
                    0,
                    1,
                    &set,
                    0,
                    std::ptr::null(),
                );
                (f.push_constants)(
                    command,
                    self.pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    (PARAM_WORDS * 4) as u32,
                    pass.params.as_ptr().cast::<c_void>(),
                );
                (f.dispatch)(command, groups, 1, 1);
                (f.barrier)(
                    command,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &barrier,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                );
            }
            if pass.kernel == KERNEL_EFFECT {
                current ^= 1;
            }
        }
        // SAFETY: every pass was recorded with at least one dispatch.
        let result = unsafe { (f.end_command)(command) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkEndCommandBuffer failed".to_owned()));
        }

        let fence_info = FenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
        };
        let mut fence = std::ptr::null_mut();
        // SAFETY: an unsignalled fence for this device.
        let result =
            unsafe { (f.create_fence)(self.device, &fence_info, std::ptr::null(), &mut fence) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkCreateFence failed".to_owned()));
        }
        let submit = SubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        // SAFETY: one recorded command buffer on a compute queue, fenced.
        let result = unsafe { (f.submit)(self.queue, 1, &submit, fence) };
        if result == VK_SUCCESS {
            // SAFETY: the fence is the only wait target; no timeout.
            let result = unsafe { (f.wait_fences)(self.device, 1, &fence, 1, u64::MAX) };
            if result != VK_SUCCESS {
                return Err(GpuError::Vulkan("vkWaitForFences failed".to_owned()));
            }
        }
        // SAFETY: the fence was created above and the queue is idle.
        unsafe { (f.destroy_fence)(self.device, fence, std::ptr::null()) };
        if result != VK_SUCCESS {
            return Err(GpuError::Vulkan("vkQueueSubmit failed".to_owned()));
        }

        let source = if current == 0 { &frame_a } else { &frame_b };
        // SAFETY: host-visible coherent mapping, queue idle after the fence.
        let pixels =
            unsafe { std::slice::from_raw_parts(source.mapped as *const u8, byte_count).to_vec() };

        // SAFETY: every resource below was created above on this device.
        unsafe {
            (f.destroy_command_pool)(self.device, command_pool, std::ptr::null());
            (f.destroy_pool)(self.device, pool, std::ptr::null());
            (f.destroy_set_layout)(self.device, self.set_layout, std::ptr::null());
            for buffer in buffers {
                (f.unmap_memory)(self.device, buffer.memory);
                (f.free_memory)(self.device, buffer.memory, std::ptr::null());
                (f.destroy_buffer)(self.device, buffer.buffer, std::ptr::null());
            }
        }
        Ok(pixels)
    }
}

impl Drop for VulkanSession {
    fn drop(&mut self) {
        let f = &self.functions;
        // SAFETY: all objects were created in acquire/init on this device and
        // are destroyed once, before the loader handle drops.
        unsafe {
            for pipeline in self.pipelines {
                if !pipeline.is_null() {
                    (f.destroy_pipeline)(self.device, pipeline, std::ptr::null());
                }
            }
            (f.destroy_pipeline_layout)(self.device, self.pipeline_layout, std::ptr::null());
            (f.destroy_set_layout)(self.device, self.set_layout, std::ptr::null());
            (f.destroy_device)(self.device, std::ptr::null());
            (f.destroy_instance)(self.instance, std::ptr::null());
        }
    }
}
