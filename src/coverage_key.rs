//! Frozen coverage-cache key shared by the CPU raster path and the GPU
//! backends.
//!
//! The key covers only what changes coverage bytes: canonical geometry,
//! device scale, sub-pixel phase and (when it intersects the edge band) the
//! clip shape. Colour, opacity, blend mode and integer translation never
//! participate, matching Flutter's raster-cache rule that zeroes integral
//! translation (`flow/raster_cache_key.h:85-86`).
//!
//! Serialization is frozen: fixed field order, little-endian, 28 bytes total.
//! `digest()` is FNV-1a 64 over exactly those bytes. Changing the layout or
//! the hash requires bumping `COVERAGE_KEY_VERSION`.

/// Rule version: bump on any normalization or coverage-rule change.
pub const COVERAGE_KEY_VERSION: u8 = 1;

/// Serialized key length in bytes.
pub const COVERAGE_KEY_BYTES: usize = 28;

/// Coverage algorithm tier; the tier is part of the key because it changes
/// edge bytes for identical geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CoverageTier {
    /// Axis-aligned small-radius closed-form corner coverage.
    Fast = 0,
    /// Subdivided polygon trapezoid-area coverage.
    Precise = 1,
    /// No anti-aliasing (binary coverage).
    None = 2,
}

/// Canonical geometry consumed by the key. Callers pass **already normalized**
/// radii: the single implementation lives in
/// `opurom-render/src/geometry.rs` (`RoundingRadii::scaled`), and the GPU never
/// re-normalizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverageGeometry {
    Rect {
        w: i32,
        h: i32,
    },
    /// Corner order is fixed: top-left, top-right, bottom-left, bottom-right;
    /// each corner carries independent x/y radii (elliptical corners allowed).
    RoundRect {
        w: i32,
        h: i32,
        radii: [[i32; 2]; 4],
    },
    /// Convex polygon in device pixels; order is significant and canonical.
    Polygon {
        points: Vec<[i32; 2]>,
    },
    Circle {
        diameter: i32,
    },
    /// Line band used by stroked straight edges.
    Line {
        dx: i32,
        dy: i32,
    },
}

impl CoverageGeometry {
    fn tag(&self) -> u8 {
        match self {
            Self::Rect { .. } => 0,
            Self::RoundRect { .. } => 1,
            Self::Polygon { .. } => 2,
            Self::Circle { .. } => 3,
            Self::Line { .. } => 4,
        }
    }

    /// Canonical little-endian bytes: tag, then fields in declaration order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = vec![self.tag()];
        match self {
            Self::Rect { w, h } => {
                put_i32(&mut out, *w);
                put_i32(&mut out, *h);
            }
            Self::RoundRect { w, h, radii } => {
                put_i32(&mut out, *w);
                put_i32(&mut out, *h);
                for corner in radii {
                    put_i32(&mut out, corner[0]);
                    put_i32(&mut out, corner[1]);
                }
            }
            Self::Polygon { points } => {
                put_u32(&mut out, points.len() as u32);
                for point in points {
                    put_i32(&mut out, point[0]);
                    put_i32(&mut out, point[1]);
                }
            }
            Self::Circle { diameter } => put_i32(&mut out, *diameter),
            Self::Line { dx, dy } => {
                put_i32(&mut out, *dx);
                put_i32(&mut out, *dy);
            }
        }
        out
    }
}

/// One frozen coverage-cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverageKey {
    pub version: u8,
    pub kind: u8,
    pub geom_hash: u64,
    pub scale_q: u32,
    pub subpixel_q: u32,
    pub clip_hash: u64,
    pub aa_rule: u8,
}

/// Fixed-point scale denominator: `scale_q` is `floor(scale * 256)`.
pub const SCALE_ONE: i64 = 256;

/// Fixed-point sub-pixel denominator: one pixel is 256 units.
pub const SUBPIXEL_ONE: i64 = 256;

/// Quantize a positive device scale as `floor(num * 256 / den)`.
pub fn quantize_scale(num: i64, den: i64) -> Option<u32> {
    if num <= 0 || den <= 0 {
        return None;
    }
    let q = (num as i128 * SCALE_ONE as i128) / den as i128;
    u32::try_from(q).ok()
}

/// Pack a sub-pixel phase: fractional parts of `x_256` and `y_256`, each in
/// `0..=255`, as `x | y << 8`. Integral translation is dropped by design.
pub fn quantize_subpixel(x_256: i64, y_256: i64) -> u32 {
    let fx = x_256.rem_euclid(SUBPIXEL_ONE) as u32;
    let fy = y_256.rem_euclid(SUBPIXEL_ONE) as u32;
    fx | (fy << 8)
}

impl CoverageKey {
    /// Build a key from: geometry, device scale, sub-pixel phase, tier and an
    /// optional clip shape. The clip participates only when the caller proved
    /// the clip intersects the edge band; otherwise pass `None` and the
    /// `clip_hash` stays zero.
    pub fn new(
        geometry: &CoverageGeometry,
        scale_q: u32,
        subpixel_q: u32,
        tier: CoverageTier,
        clip: Option<&CoverageGeometry>,
    ) -> Self {
        let geom_hash = fnv1a(&geometry.canonical_bytes());
        let clip_hash = clip.map(|clip| fnv1a(&clip.canonical_bytes())).unwrap_or(0);
        Self {
            version: COVERAGE_KEY_VERSION,
            kind: geometry.tag(),
            geom_hash,
            scale_q,
            subpixel_q,
            clip_hash,
            aa_rule: tier as u8,
        }
    }

    /// Convenience constructor for rounded rectangles whose radii already came
    /// from the render-side normalization.
    pub fn round_rect(
        w: i32,
        h: i32,
        radii: [[i32; 2]; 4],
        scale_q: u32,
        subpixel_q: u32,
        tier: CoverageTier,
        clip: Option<&CoverageGeometry>,
    ) -> Self {
        Self::new(
            &CoverageGeometry::RoundRect { w, h, radii },
            scale_q,
            subpixel_q,
            tier,
            clip,
        )
    }

    /// Frozen 28-byte little-endian layout:
    ///
    /// ```text
    /// 0   version u8
    /// 1   kind    u8
    /// 2   aa_rule u8
    /// 3   reserved 0
    /// 4   geom_hash   u64
    /// 12  scale_q     u32
    /// 16  subpixel_q  u32
    /// 20  clip_hash   u64
    /// ```
    pub fn to_bytes(self) -> [u8; COVERAGE_KEY_BYTES] {
        let mut out = [0u8; COVERAGE_KEY_BYTES];
        out[0] = self.version;
        out[1] = self.kind;
        out[2] = self.aa_rule;
        out[4..12].copy_from_slice(&self.geom_hash.to_le_bytes());
        out[12..16].copy_from_slice(&self.scale_q.to_le_bytes());
        out[16..20].copy_from_slice(&self.subpixel_q.to_le_bytes());
        out[20..28].copy_from_slice(&self.clip_hash.to_le_bytes());
        out
    }

    /// Decode the frozen layout; reserved byte must be zero.
    pub fn from_bytes(bytes: &[u8; COVERAGE_KEY_BYTES]) -> Option<Self> {
        if bytes[3] != 0 {
            return None;
        }
        Some(Self {
            version: bytes[0],
            kind: bytes[1],
            aa_rule: bytes[2],
            geom_hash: u64::from_le_bytes(bytes[4..12].try_into().ok()?),
            scale_q: u32::from_le_bytes(bytes[12..16].try_into().ok()?),
            subpixel_q: u32::from_le_bytes(bytes[16..20].try_into().ok()?),
            clip_hash: u64::from_le_bytes(bytes[20..28].try_into().ok()?),
        })
    }

    /// FNV-1a 64 over the frozen bytes; equal keys always hash equally.
    pub fn digest(self) -> u64 {
        fnv1a(&self.to_bytes())
    }
}

/// FNV-1a 64, the crate's existing digest algorithm.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}
