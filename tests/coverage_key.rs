//! Frozen CoverageKey vectors: canonical little-endian bytes, FNV-1a 64
//! digests and the render-side normalized radius vectors.
//!
//! The golden bytes were produced by an independent reference encoder (same
//! frozen layout), so a layout change here fails instead of drifting.
//! Normalized radii come from `opurom-render/src/geometry.rs`
//! (`RoundingRadii::scaled`, render @ b3610b7), which stays the only
//! normalization implementation.

use opurom_gpu::coverage_key::{
    fnv1a, quantize_scale, quantize_subpixel, CoverageGeometry, CoverageKey, CoverageTier,
    COVERAGE_KEY_BYTES, COVERAGE_KEY_VERSION,
};

fn digest_hex(key: CoverageKey) -> String {
    format!("{:016x}", key.digest())
}

#[test]
fn fnv1a_matches_the_known_answers() {
    assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
}

/// Rows: name, normalized radii (TL, TR, BL, BR), digest.
/// Radii are the render-side `RoundingRadii::scaled` outputs for:
/// all-10 in 40x30; all-100 in 50x30 (global scale floor); the asymmetric
/// corner case; all-30x10 elliptical in 80x20; all-25 in 50x50; a flat corner;
/// and a zero-height box.
const RENDER_VECTORS: [(&str, i32, i32, [[i32; 2]; 4], CoverageTier, u64); 7] = [
    (
        "single_small",
        40,
        30,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
        CoverageTier::Fast,
        0xfe02_f1f8_c3b7_bc27,
    ),
    (
        "radius_gt_side",
        50,
        30,
        [[14, 14], [14, 14], [14, 14], [14, 14]],
        CoverageTier::Fast,
        0xee19_798b_6f31_609d,
    ),
    (
        "asymmetric",
        100,
        60,
        [[20, 10], [5, 5], [0, 0], [30, 40]],
        CoverageTier::Fast,
        0xbef4_0c4c_7ac5_72fd,
    ),
    (
        "elliptical",
        80,
        20,
        [[30, 10], [30, 10], [30, 10], [30, 10]],
        CoverageTier::Precise,
        0xbd6c_4ad3_c161_0f90,
    ),
    (
        "near_circle",
        50,
        50,
        [[25, 25], [25, 25], [25, 25], [25, 25]],
        CoverageTier::Fast,
        0x4824_b143_cd8a_639a,
    ),
    (
        "flat_corner",
        30,
        20,
        [[0, 0], [9, 9], [9, 9], [9, 9]],
        CoverageTier::Fast,
        0x9e99_265b_37cb_de43,
    ),
    (
        "zero_box",
        0,
        20,
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        CoverageTier::Fast,
        0x9afc_5c55_5c13_077e,
    ),
];

#[test]
fn round_rect_key_bytes_and_digest_are_frozen() {
    let cases: [(&str, i32, i32, [[i32; 2]; 4], &str); 4] = [
        (
            "small",
            40,
            30,
            [[10, 10], [10, 10], [10, 10], [10, 10]],
            "010100007a796c106adea38b00010000000000000000000000000000",
        ),
        (
            "oversized",
            50,
            30,
            [[14, 14], [14, 14], [14, 14], [14, 14]],
            "01010000605c1598051a431000010000000000000000000000000000",
        ),
        (
            "asymmetric",
            100,
            60,
            [[20, 10], [5, 5], [0, 0], [30, 40]],
            "010100002cf9c614597b78c400010000000000000000000000000000",
        ),
        (
            "elliptical",
            80,
            20,
            [[30, 10], [30, 10], [30, 10], [30, 10]],
            "0101010028fe944b0b1bdaee00010000000000000000000000000000",
        ),
    ];
    let digests = [
        "fe02f1f8c3b7bc27",
        "ee19798b6f31609d",
        "bef40c4c7ac572fd",
        "bd6c4ad3c1610f90",
    ];
    for (index, (name, w, h, radii, bytes)) in cases.iter().enumerate() {
        let key = CoverageKey::round_rect(
            *w,
            *h,
            *radii,
            256,
            0,
            if *name == "elliptical" {
                CoverageTier::Precise
            } else {
                CoverageTier::Fast
            },
            None,
        );
        assert_eq!(key.version, COVERAGE_KEY_VERSION, "{name}");
        assert_eq!(key.to_bytes().len(), COVERAGE_KEY_BYTES);
        assert_eq!(bytes_of(&key), *bytes, "{name}: frozen byte layout");
        assert_eq!(digest_hex(key), digests[index], "{name}: frozen digest");
    }
}

#[test]
fn other_geometry_kinds_and_clip_participation_are_frozen() {
    let polygon = CoverageGeometry::Polygon {
        points: vec![[0, 0], [10, 0], [0, 10]],
    };
    let polygon_key = CoverageKey::new(&polygon, 256, 0, CoverageTier::Precise, None);
    assert_eq!(
        bytes_of(&polygon_key),
        "01020100065b9c6ffe734a0000010000000000000000000000000000"
    );
    assert_eq!(digest_hex(polygon_key), "824b97d05ab1761f");

    let circle = CoverageGeometry::Circle { diameter: 20 };
    let circle_key = CoverageKey::new(&circle, 256, 0, CoverageTier::Fast, None);
    assert_eq!(
        bytes_of(&circle_key),
        "01030000e67cf6979f22ab7100010000000000000000000000000000"
    );
    assert_eq!(digest_hex(circle_key), "9e99eacb1272b71a");

    let line = CoverageGeometry::Line { dx: 16, dy: 8 };
    let line_key = CoverageKey::new(&line, 256, 0, CoverageTier::None, None);
    assert_eq!(
        bytes_of(&line_key),
        "01040200eb93ddd4751c84a800010000000000000000000000000000"
    );
    assert_eq!(digest_hex(line_key), "f6763d479e0e860f");

    // The clip participates only when the caller proved an edge-band
    // intersection; otherwise the hash stays zero.
    let clip = CoverageGeometry::RoundRect {
        w: 40,
        h: 30,
        radii: [[4, 4], [4, 4], [4, 4], [4, 4]],
    };
    let no_clip = CoverageKey::round_rect(
        40,
        30,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
        256,
        0,
        CoverageTier::Fast,
        None,
    );
    let with_clip = CoverageKey::round_rect(
        40,
        30,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
        256,
        0,
        CoverageTier::Fast,
        Some(&clip),
    );
    assert_eq!(no_clip.clip_hash, 0);
    assert_ne!(with_clip.clip_hash, 0);
    assert_eq!(
        bytes_of(&with_clip),
        "010100007a796c106adea38b0001000000000000fa58f09f44d9ddb0"
    );
    assert_eq!(digest_hex(with_clip), "b519c2d3eab6053a");
}

#[test]
fn quantization_is_deterministic_and_excludes_integral_translation() {
    assert_eq!(quantize_scale(3, 2), Some(384));
    assert_eq!(quantize_scale(1, 3), Some(85));
    assert_eq!(quantize_scale(0, 1), None);
    assert_eq!(quantize_scale(1, 0), None);

    // 1/256 sub-pixel phase: x=7/256, y=9/256 packed as x | y<<8.
    assert_eq!(quantize_subpixel(7, 9), 7 | (9 << 8));
    // Integer translation is dropped: adding whole pixels changes nothing.
    assert_eq!(
        quantize_subpixel(3 * 256 + 7, 2 * 256 + 9),
        quantize_subpixel(7, 9)
    );
    assert_eq!(quantize_subpixel(-1, -257), 255 | (255 << 8));

    let base = CoverageKey::round_rect(
        40,
        30,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
        384,
        7 | (9 << 8),
        CoverageTier::Fast,
        None,
    );
    let shifted = CoverageKey::round_rect(
        40,
        30,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
        384,
        quantize_subpixel(5 * 256 + 7, 4 * 256 + 9),
        CoverageTier::Fast,
        None,
    );
    assert_eq!(base, shifted);
    assert_eq!(digest_hex(base), "80a3e645e8046133");
}

#[test]
fn key_round_trips_through_the_frozen_layout() {
    let key = CoverageKey::round_rect(
        40,
        30,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
        85,
        0,
        CoverageTier::Fast,
        None,
    );
    assert_eq!(digest_hex(key), "1f7f60efc6cd6af9");
    assert_eq!(CoverageKey::from_bytes(&key.to_bytes()), Some(key));
    let mut corrupt = key.to_bytes();
    corrupt[3] = 1;
    assert_eq!(CoverageKey::from_bytes(&corrupt), None);
}

#[test]
fn render_normalized_vectors_are_pinned() {
    // Normalized values captured from opurom-render @ b3610b7
    // (`RoundingRadii::scaled`); the GPU consumes them unchanged. The
    // oversized case proves the radius was globally scaled (100 -> 14), not
    // clamped per corner, and the asymmetric case proves a flat corner
    // normalizes to zero.
    for (name, w, h, radii, tier, digest) in RENDER_VECTORS {
        let key = CoverageKey::round_rect(w, h, radii, 256, 0, tier, None);
        assert_eq!(digest_hex(key), format!("{digest:016x}"), "{name}");
        assert_ne!(key.geom_hash, 0, "{name}");
    }
    let oversized =
        CoverageKey::round_rect(50, 30, [[14, 14]; 4], 256, 0, CoverageTier::Fast, None);
    let per_corner_clamp =
        CoverageKey::round_rect(50, 30, [[15, 15]; 4], 256, 0, CoverageTier::Fast, None);
    assert_ne!(
        oversized.geom_hash, per_corner_clamp.geom_hash,
        "a per-corner clamp would produce a different key than the global scale"
    );
}

fn bytes_of(key: &CoverageKey) -> String {
    key.to_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
