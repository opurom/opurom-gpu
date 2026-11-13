//! Present the live compute buffer into a real `CAMetalLayer`.
//!
//! The platform adapter loads this crate's cdylib at runtime and passes the
//! window layer as an opaque Objective-C object pointer. No CPU image is used
//! as a substitute: the final `MTLBuffer` is copied into the drawable
//! texture with a real `MTLBlitCommandEncoder`, then committed and presented.

use std::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::{display_byte_count, GpuError};

#[cfg(target_os = "macos")]
use crate::{metal, parse_display_list};
#[cfg(target_os = "macos")]
use objc2::runtime::{AnyObject, ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDrawable, MTLOrigin, MTLPixelFormat, MTLSize, MTLTexture,
};
#[cfg(target_os = "macos")]
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
#[cfg(any(target_os = "macos", test))]
use std::ffi::CStr;

pub const PRESENT_OK: c_int = 0;
pub const PRESENT_METAL_UNAVAILABLE: c_int = 10;
pub const PRESENT_NO_DRAWABLE: c_int = 11;
pub const PRESENT_ERROR: c_int = 12;
pub const PRESENT_INVALID_ARGUMENT: c_int = 13;
pub const PRESENT_UNSUPPORTED: c_int = 14;

/// Dimensions and byte strides for one RGBA8 buffer-to-texture blit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlitCopy {
    pub width: usize,
    pub height: usize,
    pub bytes_per_row: usize,
    pub bytes_per_image: usize,
}

/// Compute the exact `copyFromBuffer:...:toTexture:` parameters for a frame.
pub fn blit_copy(width: u32, height: u32) -> Result<BlitCopy, GpuError> {
    let bytes_per_image = display_byte_count(width, height)?;
    let width = usize::try_from(width)
        .map_err(|_| GpuError::InvalidParameters("width does not fit usize".to_owned()))?;
    let height = usize::try_from(height)
        .map_err(|_| GpuError::InvalidParameters("height does not fit usize".to_owned()))?;
    let bytes_per_row = width
        .checked_mul(4)
        .ok_or_else(|| GpuError::InvalidParameters("RGBA8 row byte count overflow".to_owned()))?;
    Ok(BlitCopy {
        width,
        height,
        bytes_per_row,
        bytes_per_image,
    })
}

/// The four ordered steps of a real presentation command buffer.
pub trait PresentationEncoder {
    fn copy_buffer_to_texture(&mut self, copy: BlitCopy) -> Result<(), GpuError>;
    fn present_drawable(&mut self) -> Result<(), GpuError>;
    fn commit(&mut self) -> Result<(), GpuError>;
    fn wait_until_completed(&mut self) -> Result<(), GpuError>;
}

/// Encode a blit, drawable presentation, commit and completion check in order.
pub fn encode_presentation<E: PresentationEncoder>(
    encoder: &mut E,
    copy: BlitCopy,
) -> Result<(), GpuError> {
    encoder.copy_buffer_to_texture(copy)?;
    encoder.present_drawable()?;
    encoder.commit()?;
    encoder.wait_until_completed()
}

/// Classify an error as an honest SKIP status or a real FAIL status.
pub fn status_from_error(error: &GpuError) -> c_int {
    match error {
        GpuError::MetalUnavailable(_) => PRESENT_METAL_UNAVAILABLE,
        GpuError::DrawableUnavailable(_) => PRESENT_NO_DRAWABLE,
        _ => PRESENT_ERROR,
    }
}

fn write_reason(buffer: *mut c_char, capacity: usize, reason: &str) {
    if buffer.is_null() || capacity == 0 {
        return;
    }
    // SAFETY: the caller owns `capacity` writable c_char bytes; the copy is
    // bounded by that length and the last byte is always a NUL terminator.
    unsafe {
        ptr::write(buffer, 0);
        let count = reason.len().min(capacity - 1);
        ptr::copy_nonoverlapping(reason.as_ptr(), buffer.cast::<u8>(), count);
        *buffer.add(count) = 0;
    }
}

/// C ABI used by `opurom-platform` through `dlopen`/`dlsym`.
///
/// Returns one of the `PRESENT_*` statuses and writes a NUL-terminated reason
/// into the caller's buffer. A panic is contained and reported as an error.
#[no_mangle]
pub extern "C" fn opr_present_display_list(
    json: *const c_char,
    layer: *mut c_void,
    reason: *mut c_char,
    reason_capacity: usize,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if json.is_null() {
            write_reason(reason, reason_capacity, "the display list pointer is null");
            return PRESENT_INVALID_ARGUMENT;
        }
        if layer.is_null() {
            write_reason(reason, reason_capacity, "the CAMetalLayer pointer is null");
            return PRESENT_INVALID_ARGUMENT;
        }

        #[cfg(target_os = "macos")]
        {
            let text = match unsafe { CStr::from_ptr(json) }.to_str() {
                Ok(text) => text,
                Err(_) => {
                    write_reason(reason, reason_capacity, "the display list is not UTF-8");
                    return PRESENT_INVALID_ARGUMENT;
                }
            };
            match present_impl(text, layer) {
                Ok(()) => {
                    write_reason(reason, reason_capacity, "");
                    PRESENT_OK
                }
                Err(error) => {
                    let status = status_from_error(&error);
                    write_reason(reason, reason_capacity, &error.to_string());
                    status
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (json, layer);
            write_reason(
                reason,
                reason_capacity,
                "Metal presentation is available only on macOS",
            );
            PRESENT_UNSUPPORTED
        }
    }));
    result.unwrap_or_else(|_| {
        write_reason(reason, reason_capacity, "Metal presentation panicked");
        PRESENT_ERROR
    })
}

#[cfg(target_os = "macos")]
fn present_impl(json: &str, layer: *mut c_void) -> Result<(), GpuError> {
    let list = parse_display_list(json)?;
    let byte_count = display_byte_count(list.width, list.height)?;
    let (packed, passes) = list.build_passes()?;
    let raster = metal::raster_buffer(&packed, &passes, byte_count)?;
    let copy = blit_copy(list.width, list.height)?;

    // SAFETY: the caller retains the live CAMetalLayer for this call; both
    // crates use the same Objective-C object representation.
    let layer = unsafe { &*(layer as *const CAMetalLayer) };
    let size = CGSize::new(f64::from(list.width), f64::from(list.height));
    layer.setFrame(CGRect::new(CGPoint::ZERO, size));
    layer.setContentsScale(1.0);
    layer.setDevice(Some(&*raster.device));
    layer.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    layer.setDrawableSize(size);
    layer.setFramebufferOnly(false);
    layer.setPresentsWithTransaction(true);

    let drawable = layer.nextDrawable().ok_or_else(|| {
        GpuError::DrawableUnavailable(
            "CAMetalLayer nextDrawable() returned nil; no on-screen drawable is available"
                .to_owned(),
        )
    })?;
    let texture = drawable.texture();
    if texture.width() != copy.width
        || texture.height() != copy.height
        || texture.pixelFormat() != MTLPixelFormat::RGBA8Unorm
    {
        return Err(GpuError::Metal(format!(
            "drawable texture is {}x{} format {}, expected {}x{} RGBA8Unorm",
            texture.width(),
            texture.height(),
            texture.pixelFormat().0,
            copy.width,
            copy.height
        )));
    }

    let command_buffer = raster
        .queue
        .commandBuffer()
        .ok_or_else(|| GpuError::Metal("presentation commandBuffer returned nil".to_owned()))?;
    let blit = command_buffer
        .blitCommandEncoder()
        .ok_or_else(|| GpuError::Metal("blitCommandEncoder returned nil".to_owned()))?;
    let drawable_protocol = ProtocolObject::<dyn MTLDrawable>::from_ref(&*drawable);
    let mut encoder = MetalPresentation {
        command_buffer: &command_buffer,
        blit: &blit,
        buffer: &raster.buffer,
        texture: &texture,
        drawable: drawable_protocol,
    };
    encode_presentation(&mut encoder, copy)?;

    // The presented drawable's own IOSurface is exposed as the layer contents:
    // `cacheDisplay(in:to:)` re-renders the layer tree offscreen and
    // cannot see window-server drawable content (it renders blank). This is
    // the exact zero-copy surface the Metal blit wrote and the window server
    // presents, never a CPU re-render or a second drawable substitute.
    let Some(surface) = texture.iosurface() else {
        return Err(GpuError::DrawableUnavailable(
            "the drawable texture is not IOSurface-backed; cacheDisplay cannot capture the presented frame"
                .to_owned(),
        ));
    };
    let contents: &AnyObject = (*surface).as_ref();
    // SAFETY: the surface outlives the retained layer contents, and CALayer
    // accepts its IOSurface-backed contents object.
    unsafe { layer.setContents(Some(contents)) };
    Ok(())
}

#[cfg(target_os = "macos")]
struct MetalPresentation<'a> {
    command_buffer: &'a ProtocolObject<dyn MTLCommandBuffer>,
    blit: &'a ProtocolObject<dyn MTLBlitCommandEncoder>,
    buffer: &'a ProtocolObject<dyn MTLBuffer>,
    texture: &'a ProtocolObject<dyn MTLTexture>,
    drawable: &'a ProtocolObject<dyn MTLDrawable>,
}

#[cfg(target_os = "macos")]
impl PresentationEncoder for MetalPresentation<'_> {
    fn copy_buffer_to_texture(&mut self, copy: BlitCopy) -> Result<(), GpuError> {
        // SAFETY: the compute buffer and drawable texture stay alive through
        // commit; sizes, strides and the RGBA8 format were checked above.
        unsafe {
            self.blit
                .copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    self.buffer,
                    0,
                    copy.bytes_per_row,
                    copy.bytes_per_image,
                    MTLSize {
                        width: copy.width,
                        height: copy.height,
                        depth: 1,
                    },
                    self.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
        }
        self.blit.endEncoding();
        Ok(())
    }

    fn present_drawable(&mut self) -> Result<(), GpuError> {
        self.command_buffer.presentDrawable(self.drawable);
        Ok(())
    }

    fn commit(&mut self) -> Result<(), GpuError> {
        self.command_buffer.commit();
        Ok(())
    }

    fn wait_until_completed(&mut self) -> Result<(), GpuError> {
        self.command_buffer.waitUntilCompleted();
        if let Some(error) = self.command_buffer.error() {
            return Err(metal::ns_error(
                &error,
                "presentation command buffer execution",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuError;

    struct RecordingEncoder {
        steps: Vec<String>,
    }

    impl PresentationEncoder for RecordingEncoder {
        fn copy_buffer_to_texture(&mut self, copy: BlitCopy) -> Result<(), GpuError> {
            self.steps.push(format!(
                "blit:{}x{}:{}:{}",
                copy.width, copy.height, copy.bytes_per_row, copy.bytes_per_image
            ));
            Ok(())
        }

        fn present_drawable(&mut self) -> Result<(), GpuError> {
            self.steps.push("present".to_owned());
            Ok(())
        }

        fn commit(&mut self) -> Result<(), GpuError> {
            self.steps.push("commit".to_owned());
            Ok(())
        }

        fn wait_until_completed(&mut self) -> Result<(), GpuError> {
            self.steps.push("wait".to_owned());
            Ok(())
        }
    }

    #[test]
    fn blit_copy_matches_an_rgba8_frame() {
        let copy = blit_copy(200, 150).unwrap();
        assert_eq!(
            copy,
            BlitCopy {
                width: 200,
                height: 150,
                bytes_per_row: 800,
                bytes_per_image: 120_000,
            }
        );
        assert!(blit_copy(0, 150).is_err());
    }

    #[test]
    fn presentation_encoding_orders_blit_present_commit_and_wait() {
        let mut encoder = RecordingEncoder { steps: Vec::new() };
        encode_presentation(&mut encoder, blit_copy(200, 150).unwrap()).unwrap();
        assert_eq!(
            encoder.steps,
            ["blit:200x150:800:120000", "present", "commit", "wait"]
        );
    }

    #[test]
    fn unavailable_device_and_drawable_are_skip_statuses() {
        let device = GpuError::MetalUnavailable("no Metal device".to_owned());
        let drawable = GpuError::DrawableUnavailable(
            "CAMetalLayer nextDrawable() returned nil; no on-screen drawable is available"
                .to_owned(),
        );
        assert_eq!(status_from_error(&device), PRESENT_METAL_UNAVAILABLE);
        assert_eq!(status_from_error(&drawable), PRESENT_NO_DRAWABLE);
        assert_eq!(
            status_from_error(&GpuError::Metal("bad pipeline".to_owned())),
            PRESENT_ERROR
        );
        assert!(drawable.to_string().contains("DrawableUnavailable"));
        assert!(!drawable.to_string().contains("PASS"));
    }

    #[test]
    fn a_null_display_list_is_never_reported_as_a_pass() {
        let mut reason = [0 as c_char; 128];
        let status = opr_present_display_list(
            ptr::null(),
            ptr::null_mut(),
            reason.as_mut_ptr(),
            reason.len(),
        );
        assert_eq!(status, PRESENT_INVALID_ARGUMENT);
        let reason = unsafe { CStr::from_ptr(reason.as_ptr()) }.to_str().unwrap();
        assert!(reason.contains("display list pointer"));
    }
}
