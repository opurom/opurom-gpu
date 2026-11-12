//! Independent CPU oracle for the Metal compute kernel.
//!
//! All arithmetic is integer and mirrors opurom-render's two-stop vertical
//! gradient: t is 0..=1024, channel lerp adds 512 before truncating division.

use crate::{FrameParams, GpuError};

const UNIT: i64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn packed(self) -> u32 {
        u32::from(self.r) | (u32::from(self.g) << 8) | (u32::from(self.b) << 16)
    }
}

fn mix_channel(from: u8, to: u8, t: u32) -> u8 {
    let value =
        ((i64::from(to) - i64::from(from)) * i64::from(t) + UNIT / 2) / UNIT + i64::from(from);
    value.clamp(0, 255) as u8
}

pub fn render_cpu(params: &FrameParams) -> Result<Vec<u8>, GpuError> {
    let byte_count = params.validate()?;
    let inset = params.inset();
    let gradient_height = params.height - 2 * inset;
    let last = gradient_height - 1;
    let mut pixels = Vec::with_capacity(byte_count);

    for y in 0..params.height {
        for x in 0..params.width {
            let inside =
                x >= inset && x < params.width - inset && y >= inset && y < params.height - inset;
            let rgb = if inside {
                let local_y = y - inset;
                let t = if last == 0 {
                    UNIT as u32
                } else {
                    (local_y * UNIT as u32) / last
                };
                [
                    mix_channel(params.top.r, params.bottom.r, t),
                    mix_channel(params.top.g, params.bottom.g, t),
                    mix_channel(params.top.b, params.bottom.b, t),
                ]
            } else {
                [params.solid.r, params.solid.g, params.solid.b]
            };
            pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }

    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{digest_hex, fnv1a, MAX_DIMENSION, PROBE_PARAMS, SELF_TEST_CASES};

    #[test]
    fn fnv_known_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(digest_hex(b"foobar"), "85944171f73967e8");
        assert_ne!(digest_hex(b"foobar"), digest_hex(b"foobas"));
    }

    #[test]
    fn cpu_reference_vectors() {
        assert_eq!(
            render_cpu(&SELF_TEST_CASES[0]).unwrap(),
            vec![40, 50, 60, 255]
        );
        assert_eq!(
            render_cpu(&SELF_TEST_CASES[1]).unwrap(),
            vec![
                200, 100, 0, 255, 200, 100, 0, 255, 101, 100, 128, 255, 101, 100, 128, 255, 1, 100,
                255, 255, 1, 100, 255, 255,
            ]
        );
        assert_eq!(mix_channel(200, 0, 1024), 1);
        assert_eq!(mix_channel(0, 255, 512), 128);
    }

    #[test]
    fn cpu_reference_has_solid_border_and_gradient_interior() {
        let params = SELF_TEST_CASES[3];
        let pixels = render_cpu(&params).unwrap();
        let row_stride = params.width as usize * 4;
        assert_eq!(&pixels[0..4], &[240, 230, 220, 255]);
        assert_eq!(&pixels[row_stride + 4..row_stride + 8], &[12, 200, 75, 255]);
        assert_eq!(digest_hex(&pixels), "2a702491487682ba");
    }

    #[test]
    fn probe_digest_is_stable() {
        let pixels = render_cpu(&PROBE_PARAMS).unwrap();
        assert_eq!(pixels.len(), 12_288);
        assert_eq!(digest_hex(&pixels), "704ffb40489246c5");
    }

    #[test]
    fn self_test_digest_is_stable() {
        let mut all_bytes = Vec::new();
        for params in SELF_TEST_CASES {
            all_bytes.extend_from_slice(&render_cpu(&params).unwrap());
        }
        assert_eq!(all_bytes.len(), 1_092);
        assert_eq!(digest_hex(&all_bytes), "f8b4088439129f9c");
        assert_eq!(
            digest_hex(&render_cpu(&SELF_TEST_CASES[2]).unwrap()),
            "d98d9da892f31700"
        );
    }

    #[test]
    fn parameter_validation() {
        let red = Rgb::new(1, 2, 3);
        assert_eq!(FrameParams::new(1, 1, red, red, red).validate(), Ok(4));
        assert!(matches!(
            FrameParams::new(0, 1, red, red, red).validate(),
            Err(GpuError::InvalidParameters(_))
        ));
        assert!(matches!(
            FrameParams::new(1, 0, red, red, red).validate(),
            Err(GpuError::InvalidParameters(_))
        ));
        assert!(matches!(
            FrameParams::new(MAX_DIMENSION + 1, 1, red, red, red).validate(),
            Err(GpuError::InvalidParameters(_))
        ));
    }
}
