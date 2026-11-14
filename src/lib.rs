//! Deterministic RGBA8 Metal compute backend for OPUROM.
//!
//! The CPU reference and the MSL kernel share an integer pixel contract. Metal
//! code is compiled only on macOS; other targets return `MetalUnavailable`.
//!
//! `raster <commands.json> <out.ppm>` uploads the independent renderer's exact
//! display list to Metal compute kernels. Six effect command types are
//! supported: blur, Gaussian blur, shadow, glow, glass and liquid execute as
//! real multi-pass MSL kernels with frame snapshots and readback. The
//! presentation path retains the final compute buffer and presents it through a
//! real CAMetalLayer drawable from the platform's live window. Text glyph compositing stays CPU-side: glyphs reach the GPU only as
//! pre-decomposed solid rectangles.

use std::fmt;
use std::time::Duration;

mod display;
#[cfg(target_os = "macos")]
mod metal;
pub mod presentation;
mod reference;

pub use display::{
    gaussian_blur_cpu, parse_display_list, render_cpu_commands, Command, DisplayList, Pass,
    COMMAND_WORDS, COORD_BIAS, GAUSSIAN_MAX_RADIUS, GAUSSIAN_MIN_RADIUS, KERNEL_BATCH,
    KERNEL_BLUR_H, KERNEL_BLUR_V, KERNEL_EFFECT, KERNEL_GAUSSIAN_H, KERNEL_GAUSSIAN_V, KERNEL_MASK,
    KERNEL_SNAPSHOT, MAX_COMMANDS as MAX_DISPLAY_COMMANDS, PARAM_WORDS, PARTICLE_MAX_COUNT,
    PARTICLE_MAX_RADIUS, PARTICLE_MAX_SEED,
};
pub use reference::{render_cpu, Rgb};

/// Bound one host allocation and keeps the MSL thread index in `uint` range.
pub const MAX_DIMENSION: u32 = 4096;

pub const PROBE_PARAMS: FrameParams = FrameParams::new(
    64,
    48,
    Rgb::new(10, 20, 30),
    Rgb::new(220, 40, 20),
    Rgb::new(20, 80, 240),
);

pub const SELF_TEST_CASES: [FrameParams; 4] = [
    FrameParams::new(
        1,
        1,
        Rgb::new(1, 2, 3),
        Rgb::new(10, 20, 30),
        Rgb::new(40, 50, 60),
    ),
    FrameParams::new(
        2,
        3,
        Rgb::new(5, 7, 11),
        Rgb::new(200, 100, 0),
        Rgb::new(0, 100, 255),
    ),
    FrameParams::new(
        9,
        5,
        Rgb::new(17, 34, 51),
        Rgb::new(0, 0, 0),
        Rgb::new(255, 255, 255),
    ),
    FrameParams::new(
        17,
        13,
        Rgb::new(240, 230, 220),
        Rgb::new(12, 200, 75),
        Rgb::new(75, 12, 200),
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameParams {
    pub width: u32,
    pub height: u32,
    pub solid: Rgb,
    pub top: Rgb,
    pub bottom: Rgb,
}

impl FrameParams {
    pub const fn new(width: u32, height: u32, solid: Rgb, top: Rgb, bottom: Rgb) -> Self {
        Self {
            width,
            height,
            solid,
            top,
            bottom,
        }
    }

    pub fn validate(&self) -> Result<usize, GpuError> {
        if self.width == 0 || self.height == 0 {
            return Err(GpuError::InvalidParameters(
                "width and height must be positive".to_owned(),
            ));
        }
        if self.width > MAX_DIMENSION || self.height > MAX_DIMENSION {
            return Err(GpuError::InvalidParameters(format!(
                "width and height must not exceed {MAX_DIMENSION}"
            )));
        }

        let pixels = u64::from(self.width) * u64::from(self.height);
        if pixels > u64::from(u32::MAX) {
            return Err(GpuError::InvalidParameters(
                "thread count exceeds the MSL uint range".to_owned(),
            ));
        }
        let bytes = pixels
            .checked_mul(4)
            .ok_or_else(|| GpuError::InvalidParameters("RGBA8 byte count overflow".to_owned()))?;
        usize::try_from(bytes).map_err(|_| {
            GpuError::InvalidParameters("RGBA8 byte count does not fit usize".to_owned())
        })
    }

    pub fn inset(&self) -> u32 {
        self.width.min(self.height) / 8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    /// Documented fallback when the host cannot expose a Metal device.
    MetalUnavailable(String),
    /// The window is present but Core Animation cannot supply a drawable.
    DrawableUnavailable(String),
    InvalidParameters(String),
    InvalidDisplayList(String),
    Metal(String),
    Mismatch {
        width: u32,
        height: u32,
        gpu_digest: String,
        cpu_digest: String,
    },
}

impl GpuError {
    pub fn metal_unavailable() -> Self {
        Self::MetalUnavailable(
            "Metal compute is available only on macOS; no GPU work was performed".to_owned(),
        )
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetalUnavailable(message) => write!(f, "MetalUnavailable: {message}"),
            Self::DrawableUnavailable(message) => write!(f, "DrawableUnavailable: {message}"),
            Self::InvalidParameters(message) => write!(f, "InvalidParameters: {message}"),
            Self::InvalidDisplayList(message) => write!(f, "InvalidDisplayList: {message}"),
            Self::Metal(message) => write!(f, "Metal error: {message}"),
            Self::Mismatch {
                width,
                height,
                gpu_digest,
                cpu_digest,
            } => write!(
                f,
                "GPU/CPU mismatch at {width}x{height}: gpu={gpu_digest} cpu={cpu_digest}"
            ),
        }
    }
}

impl std::error::Error for GpuError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuFrame {
    pub device_name: String,
    pub pixels: Vec<u8>,
    pub dispatch_elapsed: Duration,
}

impl GpuFrame {
    pub fn byte_count(&self) -> usize {
        self.pixels.len()
    }

    pub fn digest(&self) -> u64 {
        fnv1a(&self.pixels)
    }

    pub fn digest_hex(&self) -> String {
        digest_hex(&self.pixels)
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn digest_hex(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a(bytes))
}

/// Budget boundary, in milliseconds, between the full and the degraded tier.
pub const QUALITY_FULL_MS: i64 = 16;
/// Budget boundary, in milliseconds, between the three-quarters and half tiers.
pub const QUALITY_THREE_QUARTERS_MS: i64 = 8;
/// Budget boundary, in milliseconds, between the half and quarter tiers.
pub const QUALITY_HALF_MS: i64 = 4;

/// The deterministic frame-budget quality tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityTier {
    Full,
    ThreeQuarters,
    Half,
    Quarter,
}

impl QualityTier {
    pub fn parse(s: &str) -> Option<QualityTier> {
        Some(match s {
            "full" => QualityTier::Full,
            "three_quarters" => QualityTier::ThreeQuarters,
            "half" => QualityTier::Half,
            "quarter" => QualityTier::Quarter,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            QualityTier::Full => "full",
            QualityTier::ThreeQuarters => "three_quarters",
            QualityTier::Half => "half",
            QualityTier::Quarter => "quarter",
        }
    }
}

/// Caller-applied quality policy selected from a measured frame budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuQuality {
    pub tier: QualityTier,
    pub particle_count: usize,
    pub effects_enabled: bool,
}

/// Map a measured frame budget to a deterministic quality tier with the same
/// boundaries the renderer's `choose_quality` uses: `>=16 ms` full, `>=8 ms`
/// three quarters, `>=4 ms` half, otherwise one quarter, with effects enabled
/// only at the full tier. `measured_frame_budget_ms` is caller-supplied; this
/// helper never reads a clock, so the Metal and CPU paths select the same tier.
pub fn choose_quality(
    measured_frame_budget_ms: i64,
    particle_count: usize,
    effects_enabled: bool,
) -> GpuQuality {
    let (num, den, tier, effects) = if measured_frame_budget_ms >= QUALITY_FULL_MS {
        (1, 1, QualityTier::Full, effects_enabled)
    } else if measured_frame_budget_ms >= QUALITY_THREE_QUARTERS_MS {
        (3, 4, QualityTier::ThreeQuarters, false)
    } else if measured_frame_budget_ms >= QUALITY_HALF_MS {
        (1, 2, QualityTier::Half, false)
    } else {
        (1, 4, QualityTier::Quarter, false)
    };
    GpuQuality {
        tier,
        particle_count: particle_count.saturating_mul(num) / den,
        effects_enabled: effects,
    }
}

pub fn render(params: &FrameParams) -> Result<GpuFrame, GpuError> {
    let byte_count = params.validate()?;

    #[cfg(target_os = "macos")]
    {
        metal::render(params, byte_count)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = byte_count;
        Err(GpuError::metal_unavailable())
    }
}

/// Validate a display-list viewport and return its RGBA byte count.
pub fn display_byte_count(width: u32, height: u32) -> Result<usize, GpuError> {
    if width == 0 || height == 0 {
        return Err(GpuError::InvalidParameters(
            "display-list width and height must be positive".to_owned(),
        ));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(GpuError::InvalidParameters(format!(
            "display-list dimensions must not exceed {MAX_DIMENSION}"
        )));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > u64::from(u32::MAX) {
        return Err(GpuError::InvalidParameters(
            "thread count exceeds the MSL uint range".to_owned(),
        ));
    }
    pixels
        .checked_mul(4)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            GpuError::InvalidParameters("display-list RGBA byte count overflow".to_owned())
        })
}

/// Execute one parsed display list on the Metal compute device.
pub fn raster_display(list: &DisplayList) -> Result<GpuFrame, GpuError> {
    let byte_count = display_byte_count(list.width, list.height)?;
    let (packed, passes) = list.build_passes()?;

    #[cfg(target_os = "macos")]
    {
        metal::raster(&packed, &passes, byte_count)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (packed, passes, byte_count);
        Err(GpuError::metal_unavailable())
    }
}

/// Render one parsed display list with Metal and encode the readback as P6.
pub fn raster_display_ppm(list: &DisplayList) -> Result<Vec<u8>, GpuError> {
    let frame = raster_display(list)?;
    ppm_bytes(list.width, list.height, &frame.pixels)
}

/// Encode RGBA8 pixels in the renderer's exact P6 PPM format.
pub fn ppm_bytes(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, GpuError> {
    let byte_count = display_byte_count(width, height)?;
    if rgba.len() != byte_count {
        return Err(GpuError::InvalidParameters(format!(
            "RGBA length {} does not match {byte_count}",
            rgba.len()
        )));
    }
    let header = format!("P6\n{width} {height}\n255\n");
    let mut ppm = Vec::with_capacity(header.len() + width as usize * height as usize * 3);
    ppm.extend_from_slice(header.as_bytes());
    for pixel in rgba.chunks_exact(4) {
        ppm.extend_from_slice(&pixel[..3]);
    }
    Ok(ppm)
}

/// Parse and execute `commands.json`, then encode the GPU PPM bytes.
pub fn raster_json_ppm(json: &str) -> Result<Vec<u8>, GpuError> {
    let list = parse_display_list(json)?;
    raster_display_ppm(&list)
}

pub fn probe_output() -> Result<String, GpuError> {
    let frame = render(&PROBE_PARAMS)?;
    let milliseconds = frame.dispatch_elapsed.as_secs_f64() * 1000.0;
    Ok(format!(
        "device={}\nbackend=metal-compute\nsize={}x{}\nbytes={}\ndispatch_ms={:.3}\ndigest={}",
        frame.device_name,
        PROBE_PARAMS.width,
        PROBE_PARAMS.height,
        frame.byte_count(),
        milliseconds,
        frame.digest_hex()
    ))
}

pub fn self_test_output() -> Result<String, GpuError> {
    let mut device_name = String::new();
    let mut all_gpu_bytes = Vec::new();

    for params in SELF_TEST_CASES {
        let frame = render(&params)?;
        let cpu_bytes = render_cpu(&params)?;
        if frame.pixels != cpu_bytes {
            return Err(GpuError::Mismatch {
                width: params.width,
                height: params.height,
                gpu_digest: digest_hex(&frame.pixels),
                cpu_digest: digest_hex(&cpu_bytes),
            });
        }
        if device_name.is_empty() {
            device_name.clone_from(&frame.device_name);
        }
        all_gpu_bytes.extend_from_slice(&frame.pixels);
    }

    if device_name.is_empty() {
        return Err(GpuError::Metal("MTLDevice name was empty".to_owned()));
    }
    Ok(format!(
        "PASS device={} cases={} digest={}",
        device_name,
        SELF_TEST_CASES.len(),
        digest_hex(&all_gpu_bytes)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_error_is_documented() {
        let error = GpuError::metal_unavailable();
        assert!(matches!(error, GpuError::MetalUnavailable(_)));
        assert!(error.to_string().contains("MetalUnavailable"));
        assert!(!error.to_string().contains("PASS"));
    }

    #[test]
    fn mismatch_error_reports_both_digests() {
        let error = GpuError::Mismatch {
            width: 2,
            height: 3,
            gpu_digest: "aaaaaaaaaaaaaaaa".to_owned(),
            cpu_digest: "bbbbbbbbbbbbbbbb".to_owned(),
        };
        let message = error.to_string();
        assert!(message.contains("2x3"));
        assert!(message.contains("aaaaaaaaaaaaaaaa"));
        assert!(message.contains("bbbbbbbbbbbbbbbb"));
    }

    #[test]
    fn quality_tier_boundaries_match_the_renderer_table() {
        // The GPU tier must use the renderer's boundaries: >=16 ms full,
        // >=8 ms three quarters, >=4 ms half, otherwise one quarter.
        for (budget, tier) in [
            (-1, QualityTier::Quarter),
            (0, QualityTier::Quarter),
            (3, QualityTier::Quarter),
            (4, QualityTier::Half),
            (7, QualityTier::Half),
            (8, QualityTier::ThreeQuarters),
            (15, QualityTier::ThreeQuarters),
            (16, QualityTier::Full),
            (33, QualityTier::Full),
        ] {
            let quality = choose_quality(budget, 8, true);
            assert_eq!(quality.tier, tier, "budget {budget}");
        }
        // Effects stay enabled only at the full tier, and only when requested.
        assert!(!choose_quality(0, 8, true).effects_enabled);
        assert!(!choose_quality(4, 8, true).effects_enabled);
        assert!(!choose_quality(8, 8, true).effects_enabled);
        assert!(choose_quality(16, 8, true).effects_enabled);
        assert!(!choose_quality(16, 8, false).effects_enabled);
    }

    #[test]
    fn quality_particle_count_uses_the_same_integer_reduction() {
        assert_eq!(choose_quality(16, 8, true).particle_count, 8);
        assert_eq!(choose_quality(8, 8, true).particle_count, 6);
        assert_eq!(choose_quality(4, 8, true).particle_count, 4);
        assert_eq!(choose_quality(3, 8, true).particle_count, 2);
        assert_eq!(choose_quality(0, 8, true).particle_count, 2);
        // Truncating reduction: 7 * 3 / 4 == 5.
        assert_eq!(choose_quality(8, 7, true).particle_count, 5);
        // Saturating so a huge count cannot overflow.
        assert_eq!(
            choose_quality(16, usize::MAX, true).particle_count,
            usize::MAX
        );
        assert_eq!(
            choose_quality(0, usize::MAX, true).particle_count,
            usize::MAX / 4
        );
    }

    #[test]
    fn quality_tier_selection_is_deterministic_and_monotonic() {
        for budget in 0..=24 {
            assert_eq!(
                choose_quality(budget, 17, true),
                choose_quality(budget, 17, true),
                "the tier must be a pure function of the budget"
            );
        }
        // Tightening the budget never raises the particle count, and effects
        // turn on once the budget reaches the full tier.
        let mut previous_count = 0;
        let mut previous_effects = false;
        for budget in 0..=24 {
            let quality = choose_quality(budget, 17, true);
            assert!(quality.particle_count >= previous_count, "budget {budget}");
            if previous_effects {
                assert!(quality.effects_enabled, "budget {budget}");
            }
            previous_count = quality.particle_count;
            previous_effects = quality.effects_enabled;
        }
    }

    #[test]
    fn quality_tier_is_orthogonal_to_the_cpu_oracle() {
        // The tier is a caller-applied policy, not a pixel input: the CPU
        // reference stays byte identical at every tier, so the Metal and CPU
        // paths cannot disagree on pixels because of a tier choice.
        for budget in [-1, 0, 3, 4, 8, 16, 33] {
            let quality = choose_quality(budget, 8, true);
            let params = SELF_TEST_CASES[0];
            let cpu = render_cpu(&params).unwrap();
            assert!(quality.particle_count <= 8);
            assert_eq!(digest_hex(&cpu), digest_hex(&render_cpu(&params).unwrap()));
        }
    }
}
