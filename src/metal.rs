#![cfg(target_os = "macos")]

//! Real Metal compute path using objc2-metal 0.3.2 safe Objective-C bindings.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem::size_of;
use std::slice;
use std::time::Instant;

use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::display::DisplayList;
use crate::{FrameParams, GpuError, GpuFrame};

// MTLCreateSystemDefaultDevice requires CoreGraphics to be linked on macOS.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline uchar3 unpack_rgb(uint packed) {
    return uchar3(
        uchar((packed >> 0u) & 255u),
        uchar((packed >> 8u) & 255u),
        uchar((packed >> 16u) & 255u)
    );
}

static inline uchar mix_channel(uchar from, uchar to, uint t) {
    int value = ((int(to) - int(from)) * int(t) + 512) / 1024 + int(from);
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    return uchar(value);
}

kernel void fill_gradient_rgba8(
    device uchar4 *out [[buffer(0)]],
    constant uint *p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint width = p[0];
    const uint height = p[1];
    const uint inset = p[2];
    const uint solid = p[3];
    const uint top = p[4];
    const uint bottom = p[5];
    const uint count = width * height;
    if (tid >= count) return;

    const uint x = tid % width;
    const uint y = tid / width;
    uchar3 rgb = unpack_rgb(solid);
    if (x >= inset && x < width - inset && y >= inset && y < height - inset) {
        const uint gradient_height = height - 2u * inset;
        const uint last = gradient_height - 1u;
        const uint local_y = y - inset;
        const uint t = (last == 0u) ? 1024u : (local_y * 1024u) / last;
        uchar3 a = unpack_rgb(top);
        uchar3 b = unpack_rgb(bottom);
        rgb = uchar3(
            mix_channel(a.x, b.x, t),
            mix_channel(a.y, b.y, t),
            mix_channel(a.z, b.z, t)
        );
    }
    out[tid] = uchar4(rgb, 255);
}
"#;

// R6 display-list kernel: one thread per viewport pixel walks the exact
// command stream recorded by opurom-render, in draw order.
const DISPLAY_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant long COORD_BIAS = 1073741824L;
constant uint COMMAND_WORDS = 16u;
constant uint KIND_SOLID = 0u;
constant uint KIND_ROUNDED = 1u;
constant uint KIND_GRADIENT = 2u;

static inline uchar4 unpack_rgba(uint packed) {
    return uchar4(
        uchar((packed >> 0u) & 255u),
        uchar((packed >> 8u) & 255u),
        uchar((packed >> 16u) & 255u),
        uchar((packed >> 24u) & 255u)
    );
}

static inline uchar mix_channel(uchar from, uchar to, long t) {
    long value = ((long(to) - long(from)) * t + 512L) / 1024L + long(from);
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    return uchar(value);
}

static inline bool outside_round(long rx, long ry, long rw, long rh, long rad, long x, long y) {
    if (rad <= 0) return false;
    const long cr = rad - 1L;
    const long limit = cr * cr;
    const long dx = x - rx;
    const long cx = (dx < rad) ? (cr - dx) : ((dx >= rw - rad) ? (dx - (rw - rad)) : -1L);
    const long dy = y - ry;
    const long cy = (dy < rad) ? (cr - dy) : ((dy >= rh - rad) ? (dy - (rh - rad)) : -1L);
    return cx >= 0 && cy >= 0 && cx * cx + cy * cy > limit;
}

static inline uchar4 blend_px(uchar4 dst, uchar4 src) {
    if (src.w == 0) return dst;
    if (src.w == 255) return src;
    const uint a = uint(src.w);
    const uint inv = 255u - a;
    return uchar4(
        uchar((uint(src.x) * a + uint(dst.x) * inv) / 255u),
        uchar((uint(src.y) * a + uint(dst.y) * inv) / 255u),
        uchar((uint(src.z) * a + uint(dst.z) * inv) / 255u),
        uchar(255u)
    );
}

kernel void raster_display_list(
    device uchar4 *out [[buffer(0)]],
    device const uint *cmds [[buffer(1)]],
    constant uint *p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint width = p[0];
    const uint height = p[1];
    const uint command_count = p[3];
    const uint count = width * height;
    if (tid >= count) return;

    const uint x = tid % width;
    const uint y = tid / width;
    uchar4 color = unpack_rgba(p[2]);

    for (uint ci = 0; ci < command_count; ++ci) {
        const uint base = ci * COMMAND_WORDS;
        const uint kind = cmds[base];
        const long rx = long(cmds[base + 1u]) - COORD_BIAS;
        const long ry = long(cmds[base + 2u]) - COORD_BIAS;
        const long rw = long(cmds[base + 3u]) - COORD_BIAS;
        const long rh = long(cmds[base + 4u]) - COORD_BIAS;
        const long clipx = long(cmds[base + 5u]) - COORD_BIAS;
        const long clipy = long(cmds[base + 6u]) - COORD_BIAS;
        const long clipw = long(cmds[base + 7u]) - COORD_BIAS;
        const long cliph = long(cmds[base + 8u]) - COORD_BIAS;
        const long rad = long(cmds[base + 9u]) - COORD_BIAS;
        const long px = long(x);
        const long py = long(y);

        if (px < rx || py < ry || px >= rx + rw || py >= ry + rh) continue;
        if (px < clipx || py < clipy || px >= clipx + clipw || py >= clipy + cliph) continue;
        if (rad > 0 && outside_round(rx, ry, rw, rh, rad, px, py)) continue;

        if (kind == KIND_GRADIENT) {
            const long last = rh - 1L;
            const long local_y = py - ry;
            const long t = (last <= 0L) ? 1024L : (local_y * 1024L) / last;
            const uchar4 top = unpack_rgba(cmds[base + 10u]);
            const uchar4 bottom = unpack_rgba(cmds[base + 11u]);
            const uchar alpha = uchar(cmds[base + 12u]);
            const uchar4 src = uchar4(
                mix_channel(top.x, bottom.x, t),
                mix_channel(top.y, bottom.y, t),
                mix_channel(top.z, bottom.z, t),
                alpha
            );
            color = blend_px(color, src);
        } else if (kind == KIND_SOLID || kind == KIND_ROUNDED) {
            color = blend_px(color, unpack_rgba(cmds[base + 10u]));
        }
    }
    out[tid] = color;
}
"#;

fn ns_error(error: &NSError, context: &str) -> GpuError {
    let description = error.localizedDescription();
    let domain = error.domain();
    GpuError::Metal(format!(
        "{context}: {} (domain={}, code={})",
        &*description,
        &*domain,
        error.code()
    ))
}

fn require_system_device<T>(device: Option<T>) -> Result<T, GpuError> {
    device.ok_or_else(|| {
        GpuError::MetalUnavailable("MTLCreateSystemDefaultDevice() returned nil".to_owned())
    })
}

pub(super) fn render(params: &FrameParams, byte_count: usize) -> Result<GpuFrame, GpuError> {
    let device = require_system_device(MTLCreateSystemDefaultDevice())?;
    let name = device.name();
    let device_name = (&*name).to_string();
    if device_name.is_empty() {
        return Err(GpuError::Metal("MTLDevice name was empty".to_owned()));
    }

    let source = NSString::from_str(KERNEL_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| ns_error(&*error, "newLibraryWithSource"))?;
    let function_name = NSString::from_str("fill_gradient_rgba8");
    let function = library.newFunctionWithName(&function_name).ok_or_else(|| {
        GpuError::Metal("newFunctionWithName returned nil for fill_gradient_rgba8".to_owned())
    })?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&*function)
        .map_err(|error| ns_error(&*error, "newComputePipelineStateWithFunction"))?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| GpuError::Metal("newCommandQueue returned nil".to_owned()))?;
    let buffer = device
        .newBufferWithLength_options(byte_count, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| GpuError::Metal("newBufferWithLength returned nil".to_owned()))?;
    if buffer.length() != byte_count {
        return Err(GpuError::Metal(format!(
            "buffer length {} does not match requested {byte_count}",
            buffer.length()
        )));
    }

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| GpuError::Metal("commandBuffer returned nil".to_owned()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| GpuError::Metal("computeCommandEncoder returned nil".to_owned()))?;

    let kernel_words = [
        params.width,
        params.height,
        params.inset(),
        params.solid.packed(),
        params.top.packed(),
        params.bottom.packed(),
    ];
    let kernel_bytes = NonNull::from(&kernel_words).cast::<c_void>();
    let pixel_count = byte_count / 4;
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup().max(1);
    let group_width = pixel_count.min(max_threads);

    encoder.setComputePipelineState(&*pipeline);
    // SAFETY: both buffers stay alive through commit/wait; index and length match
    // the six MSL uint parameters and the validated RGBA8 output length.
    unsafe {
        encoder.setBytes_length_atIndex(kernel_bytes, size_of::<[u32; 6]>(), 1);
        encoder.setBuffer_offset_atIndex(Some(&*buffer), 0, 0);
    }
    // shortcut: one 1D thread per pixel, upgrade to 2D tiling with the full rasterizer in R6+.
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: pixel_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();

    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let dispatch_elapsed = started.elapsed();
    if let Some(error) = command_buffer.error() {
        return Err(ns_error(&*error, "command buffer execution"));
    }

    // SAFETY: shared-storage buffer, command completed, and length was checked.
    let pixels = unsafe {
        slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), byte_count).to_vec()
    };
    if pixels.len() != byte_count {
        return Err(GpuError::Metal(format!(
            "readback length {} does not match requested {byte_count}",
            pixels.len()
        )));
    }

    Ok(GpuFrame {
        device_name,
        pixels,
        dispatch_elapsed,
    })
}

/// Upload one parsed display list to a real Metal compute kernel, dispatch one
/// thread per viewport pixel, and read the RGBA8 result back.
pub(super) fn raster(
    list: &DisplayList,
    packed_commands: &[u32],
    byte_count: usize,
) -> Result<GpuFrame, GpuError> {
    let device = require_system_device(MTLCreateSystemDefaultDevice())?;
    let name = device.name();
    let device_name = (&*name).to_string();
    if device_name.is_empty() {
        return Err(GpuError::Metal("MTLDevice name was empty".to_owned()));
    }

    let source = NSString::from_str(DISPLAY_KERNEL_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| ns_error(&*error, "raster newLibraryWithSource"))?;
    let function_name = NSString::from_str("raster_display_list");
    let function = library.newFunctionWithName(&function_name).ok_or_else(|| {
        GpuError::Metal("newFunctionWithName returned nil for raster_display_list".to_owned())
    })?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&*function)
        .map_err(|error| ns_error(&*error, "raster newComputePipelineStateWithFunction"))?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| GpuError::Metal("newCommandQueue returned nil".to_owned()))?;

    let output = device
        .newBufferWithLength_options(byte_count, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            GpuError::Metal("raster output newBufferWithLength returned nil".to_owned())
        })?;
    if output.length() != byte_count {
        return Err(GpuError::Metal(format!(
            "raster output buffer length {} does not match requested {byte_count}",
            output.length()
        )));
    }

    let command_bytes = packed_commands.len().checked_mul(4).ok_or_else(|| {
        GpuError::InvalidParameters("display-list command bytes overflow".to_owned())
    })?;
    // A zero-command list still needs a buffer slot, so allocate one word.
    let allocation = command_bytes.max(4);
    let commands = device
        .newBufferWithLength_options(allocation, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            GpuError::Metal("raster commands newBufferWithLength returned nil".to_owned())
        })?;
    if commands.length() != allocation {
        return Err(GpuError::Metal(format!(
            "raster command buffer length {} does not match allocation {allocation}",
            commands.length()
        )));
    }
    if !packed_commands.is_empty() {
        // SAFETY: shared storage buffer of `allocation` bytes; `packed_commands`
        // fits the validated allocation and both stay alive through completion.
        unsafe {
            core::ptr::copy_nonoverlapping(
                packed_commands.as_ptr(),
                commands.contents().as_ptr().cast::<u32>(),
                packed_commands.len(),
            );
        }
    }

    let params = [
        list.width,
        list.height,
        crate::display::pack_rgba([
            list.background[0],
            list.background[1],
            list.background[2],
            list.background[3],
        ]),
        (packed_commands.len() / crate::display::COMMAND_WORDS) as u32,
    ];
    let params_bytes = NonNull::from(&params).cast::<c_void>();

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| GpuError::Metal("raster commandBuffer returned nil".to_owned()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| GpuError::Metal("raster computeCommandEncoder returned nil".to_owned()))?;

    let pixel_count = byte_count / 4;
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup().max(1);
    let group_width = pixel_count.min(max_threads);

    encoder.setComputePipelineState(&*pipeline);
    // SAFETY: output and command buffers stay alive through commit/wait, and
    // the four MSL uint parameters match the validated viewport/command layout.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*output), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*commands), 0, 1);
        encoder.setBytes_length_atIndex(params_bytes, size_of::<[u32; 4]>(), 2);
    }
    // shortcut: one 1D thread per pixel, upgrade to 2D tiling when needed.
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: pixel_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();

    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let dispatch_elapsed = started.elapsed();
    if let Some(error) = command_buffer.error() {
        return Err(ns_error(&*error, "raster command buffer execution"));
    }

    // SAFETY: shared-storage buffer, command completed, and length was checked.
    let pixels = unsafe {
        slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), byte_count).to_vec()
    };
    if pixels.len() != byte_count {
        return Err(GpuError::Metal(format!(
            "raster readback length {} does not match requested {byte_count}",
            pixels.len()
        )));
    }

    Ok(GpuFrame {
        device_name,
        pixels,
        dispatch_elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nil_device_maps_to_metal_unavailable() {
        match require_system_device(None::<u8>) {
            Err(error @ GpuError::MetalUnavailable(_)) => {
                assert!(error.to_string().contains("MetalUnavailable"));
                assert!(error.to_string().contains("returned nil"));
            }
            _ => panic!("nil device must produce MetalUnavailable"),
        }
    }
}
