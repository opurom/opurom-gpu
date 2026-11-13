#![cfg(target_os = "macos")]

//! Real Metal compute path using objc2-metal 0.3.2 safe Objective-C bindings.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::mem::size_of;
use std::slice;
use std::time::{Duration, Instant};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::display::{Pass, KERNEL_BLUR_V, KERNEL_EFFECT, PARAM_WORDS};
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

// Raster kernels: batches of regular commands plus real effect passes.
// One thread per viewport pixel; passes run in command order on the frame buffer.
const RASTER_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant long COORD_BIAS = 1073741824L;
constant uint COMMAND_WORDS = 16u;
constant uint KIND_SOLID = 0u;
constant uint KIND_ROUNDED = 1u;
constant uint KIND_GRADIENT = 2u;
constant uint KIND_GRADIENT_H = 3u;

static inline uchar4 unpack_rgba(uint packed) {
    return uchar4(
        uchar((packed >> 0u) & 255u),
        uchar((packed >> 8u) & 255u),
        uchar((packed >> 16u) & 255u),
        uchar((packed >> 24u) & 255u)
    );
}

static inline long dec(uint v) { return long(v) - COORD_BIAS; }

static inline uchar clamp_u8(long v) {
    if (v < 0) return uchar(0);
    if (v > 255) return uchar(255);
    return uchar(v);
}

static inline uchar mix_channel(uchar from, uchar to, long t) {
    long value = ((long(to) - long(from)) * t + 512L) / 1024L + long(from);
    return clamp_u8(value);
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

static inline bool inside_shape(long rx, long ry, long rw, long rh, long rad, long x, long y) {
    if (x < rx || y < ry || x >= rx + rw || y >= ry + rh) return false;
    return !outside_round(rx, ry, rw, rh, rad, x, y);
}

// Regular-command blend: integer /255 without half-up rounding.
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

// Effect blend: integer half-up (src * a + dst * (255 - a) + 127) / 255.
static inline uchar blend_add(uchar s, uchar d, long a) {
    return clamp_u8((long(s) * a + long(d) * (255L - a) + 127L) / 255L);
}

static inline uchar4 blend_fx(uchar4 dst, uchar3 src, long a) {
    if (a <= 0) return dst;
    return uchar4(
        blend_add(src.x, dst.x, a),
        blend_add(src.y, dst.y, a),
        blend_add(src.z, dst.z, a),
        uchar(255)
    );
}

static inline long alpha_u8(long fixed) {
    return long(clamp_u8((fixed * 255L + 512L) / 1024L));
}

// Two-pass separable box blur of the hard mask over its local bounds.
static inline long soft_coverage(
    device const uchar *mask, uint width,
    long bx, long by, long bw, long bh, long r, long x, long y
) {
    const long ix = x - bx;
    const long iy = y - by;
    if (ix < 0 || iy < 0 || ix >= bw || iy >= bh) return 0;
    long left = ix - r; if (left < 0) left = 0;
    long right = ix + r + 1; if (right > bw) right = bw;
    long top = iy - r; if (top < 0) top = 0;
    long bottom = iy + r + 1; if (bottom > bh) bottom = bh;
    const long cx = right - left;
    const long cy = bottom - top;
    if (cx <= 0 || cy <= 0) return 0;
    long sum_y = 0;
    for (long yy = top; yy < bottom; ++yy) {
        long sum_x = 0;
        for (long xx = left; xx < right; ++xx) {
            sum_x += long(mask[(by + yy) * long(width) + (bx + xx)]);
        }
        sum_y += (sum_x + cx / 2) / cx;
    }
    return long(clamp_u8((sum_y + cy / 2) / cy));
}

// Two-pass separable box blur of the full backdrop framebuffer at one pixel.
static inline uchar3 blurred_backdrop(
    device const uchar *src, uint w, uint h, long r, long x, long y
) {
    long left = x - r; if (left < 0) left = 0;
    long right = x + r + 1; if (right > long(w)) right = long(w);
    long top = y - r; if (top < 0) top = 0;
    long bottom = y + r + 1; if (bottom > long(h)) bottom = long(h);
    const long cx = right - left;
    const long cy = bottom - top;
    long rgb[3] = {0L, 0L, 0L};
    if (cx <= 0 || cy <= 0) return uchar3(0, 0, 0);
    for (long k = 0; k < 3; ++k) {
        long sum_y = 0;
        for (long yy = top; yy < bottom; ++yy) {
            long sum_x = 0;
            for (long xx = left; xx < right; ++xx) {
                sum_x += long(src[(yy * long(w) + xx) * 4L + k]);
            }
            sum_y += (sum_x + cx / 2) / cx;
        }
        rgb[k] = clamp_u8((sum_y + cy / 2) / cy);
    }
    return uchar3(uchar(rgb[0]), uchar(rgb[1]), uchar(rgb[2]));
}

kernel void raster_batch(
    device uchar4 *frame [[buffer(0)]],
    device const uint *cmds [[buffer(4)]],
    constant uint *p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint width = p[0];
    const uint height = p[1];
    const uint count = width * height;
    if (tid >= count) return;
    const uint first = p[5];
    const uint command_count = p[6];
    uchar4 color = unpack_rgba(p[4]);
    if (p[2] == 0u) color = frame[tid];
    const long px = long(tid % width);
    const long py = long(tid / width);

    for (uint ci = 0; ci < command_count; ++ci) {
        const uint base = first + ci * COMMAND_WORDS;
        const uint kind = cmds[base];
        const long rx = dec(cmds[base + 1u]);
        const long ry = dec(cmds[base + 2u]);
        const long rw = dec(cmds[base + 3u]);
        const long rh = dec(cmds[base + 4u]);
        const long clipx = dec(cmds[base + 5u]);
        const long clipy = dec(cmds[base + 6u]);
        const long clipw = dec(cmds[base + 7u]);
        const long cliph = dec(cmds[base + 8u]);
        const long rad = dec(cmds[base + 9u]);

        if (px < rx || py < ry || px >= rx + rw || py >= ry + rh) continue;
        if (px < clipx || py < clipy || px >= clipx + clipw || py >= clipy + cliph) continue;
        if (rad > 0 && outside_round(rx, ry, rw, rh, rad, px, py)) continue;

        if (kind == KIND_GRADIENT || kind == KIND_GRADIENT_H) {
            const bool horizontal = kind == KIND_GRADIENT_H;
            const long last = (horizontal ? rw : rh) - 1L;
            const long local = horizontal ? (px - rx) : (py - ry);
            const long t = (last <= 0L) ? 1024L : (local * 1024L) / last;
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
    frame[tid] = color;
}

kernel void snapshot_frame(
    device const uchar *frame [[buffer(0)]],
    device uchar *backdrop [[buffer(2)]],
    constant uint *p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint count = p[0] * p[1];
    if (tid >= count) return;
    const long off = long(tid) * 4L;
    backdrop[off] = frame[off];
    backdrop[off + 1L] = frame[off + 1L];
    backdrop[off + 2L] = frame[off + 2L];
    backdrop[off + 3L] = frame[off + 3L];
}

kernel void build_mask(
    device const uchar *frame [[buffer(0)]],
    device const uchar *backdrop [[buffer(2)]],
    device uchar *mask [[buffer(3)]],
    constant uint *p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint w = p[0];
    const uint h = p[1];
    const uint count = w * h;
    if (tid >= count) return;
    const long x = long(tid % w);
    const long y = long(tid / w);
    const long rad = long(p[11]);
    const long bx = dec(p[13]), by = dec(p[14]), bw = dec(p[15]), bh = dec(p[16]);
    const long rx = dec(p[17]), ry = dec(p[18]), rw = dec(p[19]), rh = dec(p[20]);
    const long cx = dec(p[21]), cy = dec(p[22]), cw = dec(p[23]), ch = dec(p[24]);
    const bool shape = p[25] != 0u;
    uchar v = 0;
    if (x >= bx && y >= by && x < bx + bw && y < by + bh &&
        x >= cx && y >= cy && x < cx + cw && y < cy + ch) {
        if (shape) {
            if (inside_shape(rx, ry, rw, rh, rad, x, y)) v = 255;
        } else {
            const long off = long(tid) * 4L;
            if (frame[off] != backdrop[off] || frame[off + 1L] != backdrop[off + 1L] ||
                frame[off + 2L] != backdrop[off + 2L] || frame[off + 3L] != backdrop[off + 3L]) {
                v = 255;
            }
        }
    }
    mask[tid] = v;
}

kernel void effect_local(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    device const uchar *backdrop [[buffer(2)]],
    device const uchar *mask [[buffer(3)]],
    constant uint *p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint w = p[0];
    const uint h = p[1];
    const uint count = w * h;
    if (tid >= count) return;
    const long x = long(tid % w);
    const long y = long(tid / w);
    const long off = long(tid) * 4L;
    const uchar4 here = uchar4(src[off], src[off + 1L], src[off + 2L], 255);
    const long kind = long(p[2]);
    const uchar3 color = unpack_rgba(p[3]).xyz;
    const long base = alpha_u8(long(p[7]));
    const long r = long(p[8]);
    const long dx = dec(p[9]);
    const long dy = dec(p[10]);
    const long rad = long(p[11]);
    const long denom = long(p[12]);
    const long bx = dec(p[13]), by = dec(p[14]), bw = dec(p[15]), bh = dec(p[16]);
    const long rx = dec(p[17]), ry = dec(p[18]), rw = dec(p[19]), rh = dec(p[20]);
    const long cx = dec(p[21]), cy = dec(p[22]), cw = dec(p[23]), ch = dec(p[24]);
    uchar4 out = here;

    if (kind == 0L || kind == 1L) {
        const long ix = x - bx;
        const long iy = y - by;
        if (ix >= 0 && iy >= 0 && ix < bw && iy < bh) {
            const long mi = (by + iy) * long(w) + (bx + ix);
            if (mask[mi] == 0) {
                const long sx = (kind == 0L) ? (x - dx) : x;
                const long sy = (kind == 0L) ? (y - dy) : y;
                const long cov = soft_coverage(mask, w, bx, by, bw, bh, r, sx, sy);
                long a = (base * cov + 127L) / 255L;
                if (a > 255) a = 255;
                out = blend_fx(here, color, a);
            }
        }
    } else if (kind == 2L) {
        if (x >= cx && y >= cy && x < cx + cw && y < cy + ch &&
            inside_shape(rx, ry, rw, rh, rad, x, y)) {
            const uchar3 blurred = blurred_backdrop(backdrop, w, h, r, x, y);
            out = blend_fx(uchar4(blurred, 255), color, base);
        }
    } else if (kind == 3L) {
        if (x >= cx && y >= cy && x < cx + cw && y < cy + ch &&
            inside_shape(rx, ry, rw, rh, rad, x, y)) {
            const long d = (x - rx) + (y - ry);
            long t = (d * 1024L) / denom;
            if (t < 0) t = 0;
            if (t > 1024) t = 1024;
            long delta = 3L * t - 1024L;
            if (delta < 0) delta = -delta;
            long band = 1024L - delta;
            if (band < 0) band = 0;
            long a = (base * band + 512L) / 1024L;
            if (a > 255) a = 255;
            out = blend_fx(here, color, a);
        }
    }
    dst[off] = out.x;
    dst[off + 1L] = out.y;
    dst[off + 2L] = out.z;
    dst[off + 3L] = 255;
}

kernel void blur_horizontal(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint *p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint w = p[0];
    const uint h = p[1];
    const uint count = w * h;
    if (tid >= count) return;
    const long r = long(p[2]);
    const long x = long(tid % w);
    const long y = long(tid / w);
    long left = x - r; if (left < 0) left = 0;
    long right = x + r + 1; if (right > long(w)) right = long(w);
    const long n = right - left;
    for (long k = 0; k < 4; ++k) {
        long sum = 0;
        for (long xx = left; xx < right; ++xx) {
            sum += long(src[(y * long(w) + xx) * 4L + k]);
        }
        dst[long(tid) * 4L + k] = clamp_u8((sum + n / 2) / n);
    }
}

kernel void blur_vertical(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint *p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint w = p[0];
    const uint h = p[1];
    const uint count = w * h;
    if (tid >= count) return;
    const long r = long(p[2]);
    const long x = long(tid % w);
    const long y = long(tid / w);
    long top = y - r; if (top < 0) top = 0;
    long bottom = y + r + 1; if (bottom > long(h)) bottom = long(h);
    const long n = bottom - top;
    for (long k = 0; k < 4; ++k) {
        long sum = 0;
        for (long yy = top; yy < bottom; ++yy) {
            sum += long(src[(yy * long(w) + x) * 4L + k]);
        }
        dst[long(tid) * 4L + k] = clamp_u8((sum + n / 2) / n);
    }
}
"#;

pub(super) fn ns_error(error: &NSError, context: &str) -> GpuError {
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

/// The live compute output, kept on the Metal device for presentation.
pub(super) struct RasterFrame {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub device_name: String,
    pub byte_count: usize,
    pub dispatch_elapsed: Duration,
}

impl RasterFrame {
    pub fn read_pixels(&self) -> Vec<u8> {
        // SAFETY: shared-storage buffer, committed compute passes completed,
        // and the allocation length was checked when it was created.
        unsafe {
            slice::from_raw_parts(
                self.buffer.contents().as_ptr().cast::<u8>(),
                self.byte_count,
            )
            .to_vec()
        }
    }
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
    // shortcut: one 1D thread per pixel, upgrade to 2D tiling with the full rasterizer.
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

/// Execute one parsed display list as an ordered sequence of real Metal
/// compute passes: regular-command batches, frame snapshots, silhouette masks,
/// local effects and whole-frame blurs. One thread per viewport pixel per pass.
pub(super) fn raster(
    packed_commands: &[u32],
    passes: &[Pass],
    byte_count: usize,
) -> Result<GpuFrame, GpuError> {
    let frame = raster_buffer(packed_commands, passes, byte_count)?;
    let pixels = frame.read_pixels();
    if pixels.len() != byte_count {
        return Err(GpuError::Metal(format!(
            "raster readback length {} does not match requested {byte_count}",
            pixels.len()
        )));
    }
    Ok(GpuFrame {
        device_name: frame.device_name,
        pixels,
        dispatch_elapsed: frame.dispatch_elapsed,
    })
}

/// Execute the same passes but retain the final GPU buffer for a blit.
pub(super) fn raster_buffer(
    packed_commands: &[u32],
    passes: &[Pass],
    byte_count: usize,
) -> Result<RasterFrame, GpuError> {
    let device = require_system_device(MTLCreateSystemDefaultDevice())?;
    let name = device.name();
    let device_name = (&*name).to_string();
    if device_name.is_empty() {
        return Err(GpuError::Metal("MTLDevice name was empty".to_owned()));
    }

    let source = NSString::from_str(RASTER_KERNEL_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| ns_error(&*error, "raster newLibraryWithSource"))?;
    let pipeline = |kernel: &str| -> Result<_, GpuError> {
        let function_name = NSString::from_str(kernel);
        let function = library.newFunctionWithName(&function_name).ok_or_else(|| {
            GpuError::Metal(format!("newFunctionWithName returned nil for {kernel}"))
        })?;
        device
            .newComputePipelineStateWithFunction_error(&*function)
            .map_err(|error| ns_error(&*error, "raster newComputePipelineStateWithFunction"))
    };
    let pipelines = [
        pipeline("raster_batch")?,
        pipeline("snapshot_frame")?,
        pipeline("build_mask")?,
        pipeline("effect_local")?,
        pipeline("blur_horizontal")?,
        pipeline("blur_vertical")?,
    ];
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| GpuError::Metal("newCommandQueue returned nil".to_owned()))?;

    let buffer = |length: usize, what: &str| -> Result<_, GpuError> {
        let created = device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| GpuError::Metal(format!("{what} newBufferWithLength returned nil")))?;
        if created.length() != length {
            return Err(GpuError::Metal(format!(
                "{what} buffer length {} does not match requested {length}",
                created.length()
            )));
        }
        Ok(created)
    };
    let frame_a = buffer(byte_count, "raster frame A")?;
    let frame_b = buffer(byte_count, "raster frame B")?;
    let backdrop = buffer(byte_count, "raster backdrop")?;
    let mask = buffer(byte_count / 4, "raster mask")?;

    let command_bytes = packed_commands.len().checked_mul(4).ok_or_else(|| {
        GpuError::InvalidParameters("display-list command bytes overflow".to_owned())
    })?;
    // A zero-command list still needs a buffer slot, so allocate one word.
    let commands = buffer(command_bytes.max(4), "raster commands")?;
    if !packed_commands.is_empty() {
        // SAFETY: shared storage buffer of the validated allocation;
        // `packed_commands` fits it and both stay alive through completion.
        unsafe {
            core::ptr::copy_nonoverlapping(
                packed_commands.as_ptr(),
                commands.contents().as_ptr().cast::<u32>(),
                packed_commands.len(),
            );
        }
    }

    let pixel_count = byte_count / 4;
    // A separate committed command buffer per pass guarantees that each pass
    // sees the previous pass's writes to the shared-storage frame buffers.
    let started = Instant::now();
    let mut current = 0usize;
    for pass in passes {
        let state = pipelines
            .get(pass.kernel as usize)
            .ok_or_else(|| GpuError::InvalidDisplayList("unknown pass kernel".to_owned()))?;
        let (primary, secondary) = if current == 0 {
            (&*frame_a, &*frame_b)
        } else {
            (&*frame_b, &*frame_a)
        };
        // The vertical blur reads the horizontal pass's output and writes the
        // current frame back, so its buffer roles are reversed.
        let (buf0, buf1) = if pass.kernel == KERNEL_BLUR_V {
            (secondary, primary)
        } else {
            (primary, secondary)
        };
        let command_buffer = queue
            .commandBuffer()
            .ok_or_else(|| GpuError::Metal("raster commandBuffer returned nil".to_owned()))?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            GpuError::Metal("raster computeCommandEncoder returned nil".to_owned())
        })?;
        let max_threads = state.maxTotalThreadsPerThreadgroup().max(1);
        let group_width = pixel_count.min(max_threads);
        let params_bytes = NonNull::from(&pass.params).cast::<c_void>();
        encoder.setComputePipelineState(&**state);
        // SAFETY: all six buffers stay alive through commit/wait; indices and
        // lengths match the MSL kernel arguments and the validated viewport.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf0), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(buf1), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*backdrop), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*mask), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&*commands), 0, 4);
            encoder.setBytes_length_atIndex(params_bytes, size_of::<[u32; PARAM_WORDS]>(), 5);
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
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if let Some(error) = command_buffer.error() {
            return Err(ns_error(&*error, "raster command buffer execution"));
        }
        if pass.kernel == KERNEL_EFFECT {
            current ^= 1;
        }
    }
    let dispatch_elapsed = started.elapsed();

    let final_buffer = if current == 0 { frame_a } else { frame_b };
    Ok(RasterFrame {
        device,
        queue,
        buffer: final_buffer,
        device_name,
        byte_count,
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
