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
