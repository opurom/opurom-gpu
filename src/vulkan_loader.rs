//! Runtime Vulkan loader probe. The recorder in `vulkan.rs` never touches a
//! driver; this module loads one with `dlopen`/`dlsym` (the house runtime-load
//! pattern), creates a real instance, enumerates physical devices, picks a
//! compute-capable device, checks the `shaderInt64` feature gate and creates a
//! device with one compute queue. It answers one question honestly: can this
//! host load a Vulkan driver, and which one? Pixel execution stays pending
//! until the compute pipeline lands; nothing here records or executes a
//! display list.

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::OnceLock;

// Modern glibc and libSystem export these from libc; no -ldl link.
#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(unix)]
const RTLD_NOW: c_int = 2;
#[cfg(unix)]
const RTLD_LOCAL: c_int = 0;

const VK_SUCCESS: i32 = 0;
const VK_API_VERSION_1_0: u32 = 1 << 22;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub(crate) const VK_QUEUE_COMPUTE_BIT: u32 = 0x2;
pub(crate) const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;

/// What the probe found, cached for the process lifetime.
pub struct VulkanProbe {
    /// A real driver answered: instance, device and compute queue all built.
    pub loaded: bool,
    /// Static reason for the state above; never empty.
    pub note: &'static str,
    pub device_name: String,
    pub api_version: u32,
    pub driver_version: u32,
    pub shader_int64: bool,
    pub compute_queue_family: Option<u32>,
}

impl VulkanProbe {
    /// True when the host could run the pixel path today: driver, device,
    /// compute queue and the 64-bit integer shader gate all present.
    pub fn usable(&self) -> bool {
        self.loaded && self.shader_int64 && self.compute_queue_family.is_some()
    }

    /// One CLI detail line; `None` while no driver answered.
    pub fn detail_line(&self) -> Option<String> {
        if !self.loaded {
            return None;
        }
        Some(format!(
            "vulkan.device={} api={}.{}.{} driver={:#010x} shader_int64={} compute_queue_family={}",
            self.device_name,
            self.api_version >> 22,
            (self.api_version >> 12) & 0x3ff,
            self.api_version & 0xfff,
            self.driver_version,
            self.shader_int64,
            self.compute_queue_family
                .map(|family| family.to_string())
                .unwrap_or_else(|| "-".to_owned()),
        ))
    }
}

static PROBE: OnceLock<VulkanProbe> = OnceLock::new();

/// Probe the host once and cache the answer for the process lifetime.
pub fn vulkan_probe() -> &'static VulkanProbe {
    PROBE.get_or_init(run_probe)
}

#[cfg(not(unix))]
fn run_probe() -> VulkanProbe {
    VulkanProbe {
        loaded: false,
        note: "Vulkan loading requires a Unix host",
        device_name: String::new(),
        api_version: 0,
        driver_version: 0,
        shader_int64: false,
        compute_queue_family: None,
    }
}

#[repr(C)]
struct ApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const c_char,
    application_version: u32,
    p_engine_name: *const c_char,
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
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
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
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
    p_enabled_features: *const c_void,
}

/// Prefix of `VkPhysicalDeviceProperties`; the driver writes the whole struct,
/// so the tail keeps the allocation larger than the real 552-byte layout.
#[repr(C, align(8))]
struct PropertiesBuffer {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; 256],
    pipeline_cache_uuid: [u8; 16],
    tail: [u64; 96],
}

/// The 55 core `VkPhysicalDeviceFeatures` booleans, frozen by Vulkan 1.0.
/// Fields after `shader_int64` are unread padding kept for the real size.
#[repr(C)]
pub(crate) struct PhysicalDeviceFeatures {
    robust_buffer_access: u32,
    full_draw_index_uint32: u32,
    image_cube_array: u32,
    independent_blend: u32,
    geometry_shader: u32,
    tessellation_shader: u32,
    sample_rate_shading: u32,
    dual_src_blend: u32,
    logic_op: u32,
    multi_draw_indirect: u32,
    draw_indirect_first_instance: u32,
    depth_clamp: u32,
    depth_bias_clamp: u32,
    fill_mode_non_solid: u32,
    depth_bounds: u32,
    wide_lines: u32,
    large_points: u32,
    alpha_to_one: u32,
    multi_viewport: u32,
    sampler_anisotropy: u32,
    texture_compression_etc2: u32,
    texture_compression_astc_ldr: u32,
    texture_compression_bc: u32,
    occlusion_query_precise: u32,
    pipeline_statistics_query: u32,
    vertex_pipeline_stores_and_atomics: u32,
    fragment_stores_and_atomics: u32,
    shader_tessellation_and_geometry_point_size: u32,
    shader_image_gather_extended: u32,
    shader_storage_image_extended_formats: u32,
    shader_storage_image_multisample: u32,
    shader_storage_image_read_without_format: u32,
    shader_storage_image_write_without_format: u32,
    shader_uniform_buffer_array_dynamic_indexing: u32,
    shader_sampled_image_array_dynamic_indexing: u32,
    shader_storage_buffer_array_dynamic_indexing: u32,
    shader_storage_image_array_dynamic_indexing: u32,
    shader_clip_distance: u32,
    shader_cull_distance: u32,
    shader_float64: u32,
    pub(crate) shader_int64: u32,
    remaining: [u32; 14],
}

const _: () = assert!(std::mem::size_of::<PhysicalDeviceFeatures>() == 55 * 4);

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}

type GetInstanceProcAddrFn = unsafe extern "system" fn(*mut c_void, *const c_char) -> *mut c_void;
type CreateInstanceFn =
    unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> i32;
type DestroyInstanceFn = unsafe extern "system" fn(*mut c_void, *const c_void);
type EnumeratePhysicalDevicesFn =
    unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32;
type GetPhysicalDevicePropertiesFn = unsafe extern "system" fn(*mut c_void, *mut PropertiesBuffer);
type GetPhysicalDeviceFeaturesFn =
    unsafe extern "system" fn(*mut c_void, *mut PhysicalDeviceFeatures);
type GetPhysicalDeviceQueueFamilyPropertiesFn =
    unsafe extern "system" fn(*mut c_void, *mut u32, *mut QueueFamilyProperties);
type CreateDeviceFn = unsafe extern "system" fn(
    *mut c_void,
    *const DeviceCreateInfo,
    *const c_void,
    *mut *mut c_void,
) -> i32;
type GetDeviceQueueFn = unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void);
type DestroyDeviceFn = unsafe extern "system" fn(*mut c_void, *const c_void);

/// Every entry point the probe needs, resolved through the loader's
/// `vkGetInstanceProcAddr` (instance functions after instance creation).
pub(crate) struct VulkanLibrary {
    pub(crate) handle: *mut c_void,
    pub(crate) get_instance_proc_addr: GetInstanceProcAddrFn,
}

#[cfg(unix)]
fn load_symbol(handle: *mut c_void, name: &str) -> *mut c_void {
    let symbol = CString::new(name).expect("symbol names contain no NUL");
    // SAFETY: dlerror clears state; dlsym reads the loaded handle.
    let _ = unsafe { dlerror() };
    let pointer = unsafe { dlsym(handle, symbol.as_ptr().cast::<c_char>()) };
    if unsafe { dlerror() }.is_null() {
        pointer
    } else {
        std::ptr::null_mut()
    }
}

impl VulkanLibrary {
    /// dlopen the first loader the host provides. MoltenVK ships the loader
    /// inside its library, so it is a candidate like the .so loader names.
    #[cfg(unix)]
    pub(crate) fn load() -> Result<Self, &'static str> {
        for name in [
            "libvulkan.so.1",
            "libvulkan.so",
            "libvulkan.dylib",
            "libMoltenVK.dylib",
        ] {
            let path = CString::new(name).expect("loader names contain no NUL");
            // SAFETY: dlopen with a valid path; the handle is closed once.
            let handle = unsafe { dlopen(path.as_ptr().cast::<c_char>(), RTLD_NOW | RTLD_LOCAL) };
            if handle.is_null() {
                let _ = unsafe { dlerror() };
                continue;
            }
            let entry = load_symbol(handle, "vkGetInstanceProcAddr");
            if entry.is_null() {
                // SAFETY: the handle came from dlopen and is closed exactly once.
                unsafe { dlclose(handle) };
                continue;
            }
            // Only the loader's vkGetInstanceProcAddr is stored here. Every
            // other entry point must be resolved through it with the right
            // instance handle; transmuting this pointer to vkCreateInstance
            // would call the wrong ABI and crash on hosts that have a loader.
            return Ok(Self {
                handle,
                get_instance_proc_addr: unsafe {
                    std::mem::transmute::<*mut c_void, GetInstanceProcAddrFn>(entry)
                },
            });
        }
        Err("no Vulkan loader found on this host")
    }
}

impl Drop for VulkanLibrary {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the handle came from dlopen and is closed exactly once.
        unsafe {
            dlclose(self.handle);
        }
    }
}

/// Resolve one instance-level entry point, failing the probe if missing.
#[cfg(unix)]
pub(crate) fn resolve<T>(
    library: &VulkanLibrary,
    instance: *mut c_void,
    name: &str,
) -> Result<T, &'static str> {
    let symbol = CString::new(name).expect("symbol names contain no NUL");
    let pointer =
        unsafe { (library.get_instance_proc_addr)(instance, symbol.as_ptr().cast::<c_char>()) };
    if pointer.is_null() {
        return Err("Vulkan loader is missing a required entry point");
    }
    // The pointer always holds a function of the requested type; transmute_copy
    // side-steps the generic size proof the plain transmute cannot make.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&pointer) })
}

#[cfg(unix)]
fn device_name_of(buffer: &PropertiesBuffer) -> String {
    let end = buffer
        .device_name
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.device_name.len());
    String::from_utf8_lossy(&buffer.device_name[..end]).into_owned()
}

#[cfg(unix)]
fn run_probe() -> VulkanProbe {
    let unavailable = |note: &'static str| VulkanProbe {
        loaded: false,
        note,
        device_name: String::new(),
        api_version: 0,
        driver_version: 0,
        shader_int64: false,
        compute_queue_family: None,
    };

    let library = match VulkanLibrary::load() {
        Ok(library) => library,
        Err(note) => return unavailable(note),
    };

    let application = ApplicationInfo {
        s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: b"opurom-gpu\0".as_ptr().cast::<c_char>(),
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
    let create_instance: CreateInstanceFn =
        match resolve(&library, std::ptr::null_mut(), "vkCreateInstance") {
            Ok(create) => create,
            Err(note) => return unavailable(note),
        };
    let mut instance: *mut c_void = std::ptr::null_mut();
    // SAFETY: valid create info with no extensions or layers; result checked.
    let result = unsafe { create_instance(&create_info, std::ptr::null(), &mut instance) };
    if result != VK_SUCCESS || instance.is_null() {
        return unavailable("Vulkan instance creation failed");
    }

    let probe = probe_devices(&library, instance);
    // SAFETY: the instance came from vkCreateInstance and is destroyed once.
    unsafe {
        let destroy: DestroyInstanceFn = match resolve(&library, instance, "vkDestroyInstance") {
            Ok(destroy) => destroy,
            Err(note) => return unavailable(note),
        };
        destroy(instance, std::ptr::null());
    }
    probe
}

#[cfg(unix)]
fn probe_devices(library: &VulkanLibrary, instance: *mut c_void) -> VulkanProbe {
    let unavailable = |note: &'static str| VulkanProbe {
        loaded: false,
        note,
        device_name: String::new(),
        api_version: 0,
        driver_version: 0,
        shader_int64: false,
        compute_queue_family: None,
    };

    let enumerate: EnumeratePhysicalDevicesFn =
        match resolve(library, instance, "vkEnumeratePhysicalDevices") {
            Ok(enumerate) => enumerate,
            Err(note) => return unavailable(note),
        };
    let get_properties: GetPhysicalDevicePropertiesFn =
        match resolve(library, instance, "vkGetPhysicalDeviceProperties") {
            Ok(get) => get,
            Err(note) => return unavailable(note),
        };
    let get_features: GetPhysicalDeviceFeaturesFn =
        match resolve(library, instance, "vkGetPhysicalDeviceFeatures") {
            Ok(get) => get,
            Err(note) => return unavailable(note),
        };
    let get_queue_families: GetPhysicalDeviceQueueFamilyPropertiesFn = match resolve(
        library,
        instance,
        "vkGetPhysicalDeviceQueueFamilyProperties",
    ) {
        Ok(get) => get,
        Err(note) => return unavailable(note),
    };
    let create_device: CreateDeviceFn = match resolve(library, instance, "vkCreateDevice") {
        Ok(create) => create,
        Err(note) => return unavailable(note),
    };
    let get_device_queue: GetDeviceQueueFn = match resolve(library, instance, "vkGetDeviceQueue") {
        Ok(get) => get,
        Err(note) => return unavailable(note),
    };
    let destroy_device: DestroyDeviceFn = match resolve(library, instance, "vkDestroyDevice") {
        Ok(destroy) => destroy,
        Err(note) => return unavailable(note),
    };

    let mut count = 0u32;
    // SAFETY: the count call writes only the count; the result is checked.
    let result = unsafe { enumerate(instance, &mut count, std::ptr::null_mut()) };
    if result != VK_SUCCESS || count == 0 {
        return unavailable("no Vulkan device on this host");
    }
    let mut devices: Vec<*mut c_void> = vec![std::ptr::null_mut(); count as usize];
    // SAFETY: the vector holds exactly `count` slots as the driver requires.
    let result = unsafe { enumerate(instance, &mut count, devices.as_mut_ptr()) };
    if result != VK_SUCCESS {
        return unavailable("no Vulkan device on this host");
    }

    // Prefer a discrete GPU; otherwise keep the first device the driver lists.
    let mut best: Option<(usize, PropertiesBuffer)> = None;
    for (index, &device) in devices.iter().enumerate() {
        let mut properties = std::mem::MaybeUninit::<PropertiesBuffer>::zeroed();
        // SAFETY: properties is an output buffer larger than the real struct.
        unsafe { get_properties(device, properties.as_mut_ptr()) };
        let properties = unsafe { properties.assume_init() };
        let preferred = properties.device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if best.is_none() || preferred {
            best = Some((index, properties));
        }
        if preferred {
            break;
        }
    }
    let (index, properties) = best.expect("at least one enumerated device");
    let device = devices[index];
    let device_name = device_name_of(&properties);

    let mut features = std::mem::MaybeUninit::<PhysicalDeviceFeatures>::zeroed();
    // SAFETY: features is an output buffer of the exact frozen 1.0 layout.
    unsafe { get_features(device, features.as_mut_ptr()) };
    let features = unsafe { features.assume_init() };
    let shader_int64 = features.shader_int64 != 0;

    let mut family_count = 0u32;
    // SAFETY: the count call writes only the count; the result is checked.
    unsafe { get_queue_families(device, &mut family_count, std::ptr::null_mut()) };
    let mut families: Vec<QueueFamilyProperties> = vec![
        QueueFamilyProperties {
            queue_flags: 0,
            queue_count: 0,
            timestamp_valid_bits: 0,
            min_image_transfer_granularity: [0; 3],
        };
        family_count as usize
    ];
    if family_count > 0 {
        // SAFETY: the vector holds exactly `family_count` slots.
        unsafe { get_queue_families(device, &mut family_count, families.as_mut_ptr()) };
    }
    let compute_queue_family = families
        .iter()
        .position(|family| family.queue_flags & VK_QUEUE_COMPUTE_BIT != 0 && family.queue_count > 0)
        .map(|index| index as u32);
    let Some(queue_family_index) = compute_queue_family else {
        return VulkanProbe {
            loaded: false,
            note: "no Vulkan compute queue on this host",
            device_name,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            shader_int64,
            compute_queue_family: None,
        };
    };
    if !shader_int64 {
        return VulkanProbe {
            loaded: false,
            note: "Vulkan device lacks 64-bit integer shaders",
            device_name,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            shader_int64: false,
            compute_queue_family: Some(queue_family_index),
        };
    }

    let queue_priority = [1.0f32];
    let queue_info = DeviceQueueCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        queue_family_index,
        queue_count: 1,
        p_queue_priorities: queue_priority.as_ptr(),
    };
    let device_info = DeviceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
        // Features stay disabled for the probe; the pixel path enables
        // shaderInt64 on its own device in the compute-pipeline phase.
        p_enabled_features: std::ptr::null(),
    };
    let mut logical_device: *mut c_void = std::ptr::null_mut();
    // SAFETY: valid queue create info for an existing compute-capable family.
    let result =
        unsafe { create_device(device, &device_info, std::ptr::null(), &mut logical_device) };
    if result != VK_SUCCESS {
        return VulkanProbe {
            loaded: false,
            note: "Vulkan device creation failed",
            device_name,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            shader_int64,
            compute_queue_family: Some(queue_family_index),
        };
    }
    let mut queue: *mut c_void = std::ptr::null_mut();
    // SAFETY: queue family and index exist on this device; result is checked.
    unsafe { get_device_queue(logical_device, queue_family_index, 0, &mut queue) };
    let queue_ready = !queue.is_null();
    // SAFETY: the device came from vkCreateDevice and is destroyed exactly once.
    unsafe { destroy_device(logical_device, std::ptr::null()) };
    if !queue_ready {
        return VulkanProbe {
            loaded: false,
            note: "Vulkan compute queue could not be fetched",
            device_name,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            shader_int64,
            compute_queue_family: Some(queue_family_index),
        };
    }

    VulkanProbe {
        loaded: true,
        note: "Vulkan driver present; recording only; pixel compute execution pending",
        device_name,
        api_version: properties.api_version,
        driver_version: properties.driver_version,
        shader_int64,
        compute_queue_family: Some(queue_family_index),
    }
}
