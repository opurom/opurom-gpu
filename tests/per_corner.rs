//! Per-corner radii on the display-list path: parsing, packing and the CPU
//! oracle must agree with the renderer's own containment rule.
//!
//! The renderer owns normalization (`RoundingRadii::scaled`) and the ellipse
//! test (`round_rect_contains` / `corner_contains`, i128 arithmetic with
//! `RADIUS_CLAMP = 1<<21`). This crate never normalizes: it consumes the radii
//! as given, clamps them to `RADIUS_CLAMP_GPU = 1<<15` so the cross-multiplied
//! products stay inside int64, and evaluates the identical predicate.
//!
//! `render_reference_outside` below is a direct transcription of the renderer
//! rule with i128 arithmetic, so the two implementations are compared on every
//! pixel instead of trusting one of them.

use opurom_gpu::{digest_hex, parse_display_list, render_cpu_commands, COMMAND_WORDS, COORD_BIAS};

/// Renderer-side clamp, `opurom-render/src/geometry.rs`.
const RENDER_RADIUS_CLAMP: i128 = 1 << 21;

/// GPU-side protocol bound, `opurom-gpu/src/display.rs`.
const RADIUS_CLAMP_GPU: i64 = 1 << 15;

/// Transcription of the renderer's `corner_contains` with i128 arithmetic.
fn render_corner_contains(
    px: i64,
    py: i64,
    corner: (i64, i64),
    dir: (i64, i64),
    radii: (i64, i64),
) -> bool {
    let rx = (radii.0 as i128).clamp(0, RENDER_RADIUS_CLAMP);
    let ry = (radii.1 as i128).clamp(0, RENDER_RADIUS_CLAMP);
    if rx == 0 || ry == 0 {
        return true;
    }
    let crx = (corner.0 as i128 - px as i128) * dir.0 as i128;
    let cry = (corner.1 as i128 - py as i128) * dir.1 as i128;
    let qx = rx - crx;
    let qy = ry - cry;
    if qx <= 0 || qy <= 0 {
        return true;
    }
    let a = qx * ry;
    let b = qy * rx;
    let rr = rx * ry;
    a * a + b * b <= rr * rr
}

/// Transcription of the renderer's `round_rect_contains`: true when the pixel
/// is inside the rounded rectangle.
fn render_reference_contains(px: i64, py: i64, rect: [i64; 4], radii: [[i64; 2]; 4]) -> bool {
    if px < rect[0] || py < rect[1] || px >= rect[0] + rect[2] || py >= rect[1] + rect[3] {
        return false;
    }
    render_corner_contains(
        px,
        py,
        (rect[0], rect[1]),
        (-1, -1),
        (radii[0][0], radii[0][1]),
    ) && render_corner_contains(
        px,
        py,
        (rect[0] + rect[2], rect[1]),
        (1, -1),
        (radii[1][0], radii[1][1]),
    ) && render_corner_contains(
        px,
        py,
        (rect[0], rect[1] + rect[3]),
        (-1, 1),
        (radii[2][0], radii[2][1]),
    ) && render_corner_contains(
        px,
        py,
        (rect[0] + rect[2], rect[1] + rect[3]),
        (1, 1),
        (radii[3][0], radii[3][1]),
    )
}

fn list(width: u32, height: u32, radii: &str) -> String {
    format!(
        r#"{{"version":1,"width":{width},"height":{height},"background":[0,0,0,255],"commands":[
{{"kind":"round_rect_per_corner","rect":[0,0,{width},{height}],"clip":[0,0,{width},{height}],"radii":{radii},"color":[255,0,0,255]}}
]}}"#
    )
}

fn radius_json(radii: &[[i64; 2]; 4]) -> String {
    format!(
        "[[{},{}],[{},{}],[{},{}],[{},{}]]",
        radii[0][0],
        radii[0][1],
        radii[1][0],
        radii[1][1],
        radii[2][0],
        radii[2][1],
        radii[3][0],
        radii[3][1]
    )
}

/// Render one per-corner command and compare every pixel with the renderer's
/// own containment rule.
fn assert_matches_render_reference(width: u32, height: u32, radii: [[i64; 2]; 4], label: &str) {
    let parsed = parse_display_list(&list(width, height, &radius_json(&radii)))
        .expect("per-corner list must parse");
    let pixels = render_cpu_commands(&parsed).expect("CPU oracle must render");
    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let offset = (y as usize * width as usize + x as usize) * 4;
            let painted =
                pixels[offset] == 255 && pixels[offset + 1] == 0 && pixels[offset + 2] == 0;
            let expected =
                render_reference_contains(x, y, [0, 0, width as i64, height as i64], radii);
            assert_eq!(
                painted, expected,
                "{label}: pixel ({x},{y}) disagrees with the renderer rule"
            );
        }
    }
}

#[test]
fn per_corner_command_parses_with_the_fixed_corner_order() {
    let parsed = parse_display_list(&list(8, 6, "[[1,2],[3,4],[5,6],[7,8]]"))
        .expect("per-corner list must parse");
    assert_eq!(parsed.commands.len(), 1);
    match &parsed.commands[0] {
        opurom_gpu::Command::RoundRectPerCorner { radii, .. } => {
            // Top-left, top-right, bottom-left, bottom-right.
            assert_eq!(*radii, [[1, 2], [3, 4], [5, 6], [7, 8]]);
        }
        other => panic!("unexpected command {other:?}"),
    }
}

#[test]
fn per_corner_packing_uses_the_new_kind_and_raw_radius_words() {
    let parsed = parse_display_list(&list(8, 6, "[[1,2],[3,4],[5,6],[7,8]]"))
        .expect("per-corner list must parse");
    let words = parsed.pack().expect("packing must succeed");
    assert_eq!(words.len(), COMMAND_WORDS);
    // New kind, the scalar radius slot keeps its biased zero.
    assert_eq!(words[0], 5);
    assert_eq!(words[9], COORD_BIAS as u32);
    assert_eq!(
        words[10], 0xff00_00ff,
        "packed RGBA puts alpha in the high byte"
    );
    // Eight raw radius words in top-left, top-right, bottom-left,
    // bottom-right order.
    for (index, expected) in [1i64, 2, 3, 4, 5, 6, 7, 8].iter().enumerate() {
        assert_eq!(
            words[11 + index],
            *expected as u32,
            "radius word {index} must carry the raw radius"
        );
    }
    for index in 19..COMMAND_WORDS {
        assert_eq!(words[index], 0, "word {index} must stay empty");
    }
}

#[test]
fn a_single_rounded_corner_matches_the_render_reference() {
    let mut radii = [[0i64; 2]; 4];
    radii[0] = [6, 6];
    assert_matches_render_reference(24, 18, radii, "top-left only");
}

#[test]
fn every_corner_position_matches_the_render_reference() {
    let cases: [(&str, usize); 4] = [
        ("top-left", 0),
        ("top-right", 1),
        ("bottom-left", 2),
        ("bottom-right", 3),
    ];
    for (name, index) in cases {
        let mut radii = [[0i64; 2]; 4];
        radii[index] = [7, 5];
        assert_matches_render_reference(32, 24, radii, name);
    }
}

/// Swapping a rounded corner between top-left and bottom-right must mirror
/// the two corner sample points, matching the renderer's own
/// `diagonal_corners_produce_mirrored_pixels` criterion.
#[test]
fn diagonal_corners_mirror_the_sampled_pixels() {
    let render = |r: &[[i64; 2]; 4]| {
        let parsed =
            parse_display_list(&list(32, 32, &radius_json(r))).expect("per-corner list must parse");
        render_cpu_commands(&parsed).expect("CPU oracle must render")
    };
    let mut tl = [[0i64; 2]; 4];
    tl[0] = [12, 12];
    let mut br = [[0i64; 2]; 4];
    br[3] = [12, 12];
    let a = render(&tl);
    let b = render(&br);
    let at = |buf: &[u8], x: usize, y: usize| buf[(y * 32 + x) * 4];
    assert_ne!(
        at(&a, 1, 1),
        at(&a, 30, 30),
        "a single rounded corner is visible"
    );
    assert_eq!(
        at(&a, 1, 1),
        at(&b, 30, 30),
        "top-left mirrors bottom-right"
    );
    assert_eq!(
        at(&a, 30, 30),
        at(&b, 1, 1),
        "bottom-right mirrors top-left"
    );
}

/// A single rounded corner must leave the opposite corner untouched. This
/// pins the array order (TL,TR,BL,BR) even though the renderer's corner test
/// is not pixel-exact 180 degrees symmetric: its BR corner sits at `x+w`, one
/// past the last pixel, so a literal mirror of two full frames differs by a
/// few boundary pixels.
#[test]
fn a_rounded_corner_does_not_touch_the_opposite_corner() {
    let mut tl = [[0i64; 2]; 4];
    tl[0] = [10, 10];
    let mut br = [[0i64; 2]; 4];
    br[3] = [10, 10];

    let render = |r: &[[i64; 2]; 4]| {
        let parsed =
            parse_display_list(&list(24, 20, &radius_json(r))).expect("per-corner list must parse");
        render_cpu_commands(&parsed).expect("CPU oracle must render")
    };
    let tl_pixels = render(&tl);
    let br_pixels = render(&br);
    let plain = render(&[[0i64; 2]; 4]);

    // Top-left radius changes only the top-left band; bottom-right radius
    // changes only the bottom-right band. Both agree with the renderer rule.
    for (x, y) in [(0i64, 0i64), (1, 1), (23, 19), (22, 18)] {
        let offset = (y as usize * 24 + x as usize) * 4;
        if x < 10 && y < 10 {
            assert_ne!(tl_pixels[offset], plain[offset], "TL corner at ({x},{y})");
            assert_eq!(
                br_pixels[offset], plain[offset],
                "BR must not affect ({x},{y})"
            );
        } else {
            assert_eq!(
                tl_pixels[offset], plain[offset],
                "TL must not affect ({x},{y})"
            );
            assert_ne!(br_pixels[offset], plain[offset], "BR corner at ({x},{y})");
        }
    }
    assert_matches_render_reference(24, 20, tl, "top-left quadrants");
    assert_matches_render_reference(24, 20, br, "bottom-right quadrants");
}

#[test]
fn asymmetric_and_elliptical_corners_match_the_render_reference() {
    let mut radii = [[0i64; 2]; 4];
    radii[0] = [10, 4];
    radii[1] = [4, 10];
    radii[2] = [0, 0];
    radii[3] = [7, 7];
    assert_matches_render_reference(32, 24, radii, "asymmetric elliptical");
}

/// The protocol bound keeps the cross-multiplied products inside int64:
/// with `rx,ry <= 2^15` each square is at most `2^60` and the sum at most
/// `2^61 < 2^63`. Radii at the bound must still agree with the renderer,
/// whose own i128 arithmetic is exact for every radius it can reach.
#[test]
fn radii_at_the_protocol_bound_match_the_render_reference() {
    for radius in [RADIUS_CLAMP_GPU - 1, RADIUS_CLAMP_GPU] {
        let mut radii = [[0i64; 2]; 4];
        radii[0] = [radius, radius];
        assert_matches_render_reference(12, 12, radii, "bound radius");
    }
}

#[test]
fn radii_above_the_protocol_bound_are_rejected() {
    let json = list(
        8,
        8,
        &format!("[[{},0],[0,0],[0,0],[0,0]]", RADIUS_CLAMP_GPU + 1),
    );
    let error = parse_display_list(&json).expect_err("an over-bound radius must be rejected");
    assert!(
        error.to_string().contains("protocol bound"),
        "unexpected error {error}"
    );
}

#[test]
fn malformed_per_corner_commands_are_rejected() {
    let cases = [
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"round_rect_per_corner","rect":[0,0,8,8],"clip":[0,0,8,8],"radii":[[1,2],[3,4]],"color":[1,2,3,255]}]}"#,
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"round_rect_per_corner","rect":[0,0,8,8],"clip":[0,0,8,8],"radii":[[1],[2,3],[4,5],[6,7]],"color":[1,2,3,255]}]}"#,
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"round_rect_per_corner","rect":[0,0,8,8],"clip":[0,0,8,8],"radii":[[1,2],[3,4],[5,6],[7,8]]}]}"#,
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"round_rect_per_corner","rect":[0,0,8,8],"clip":[0,0,8,8],"radii":[[1,2],[3,4],[5,6],[7,8],[9,10]],"color":[1,2,3,255]}]}"#,
    ];
    for (index, case) in cases.iter().enumerate() {
        let error = parse_display_list(case).expect_err("malformed input must be rejected");
        assert!(
            matches!(error, opurom_gpu::GpuError::InvalidDisplayList(_)),
            "case {index}: {error:?}"
        );
    }
}

/// The real display list emitted by `opurom-render` for the IR with
/// `cornerRadii.topLeft = 40`. The RGBA digest is the renderer CLI's own
/// `digest=` line; the PPM digest is over the RGB bytes the CLI writes.
const REAL_RENDER_COMMANDS: &str = r#"{"version":1,"width":120,"height":120,"background":[247,248,250,255],"commands":[
{"kind":"solid","rect":[0,0,120,120],"clip":[0,0,120,120],"color":[255,255,255,255]},
{"kind":"solid","rect":[1,1,118,118],"clip":[0,0,120,120],"color":[255,255,255,255]},
{"kind":"round_rect_per_corner","rect":[20,20,80,80],"clip":[0,0,120,120],
 "radii":[[40,40],[0,0],[0,0],[0,0]],"color":[0,0,0,255]},
{"kind":"round_rect_per_corner","rect":[21,21,78,78],"clip":[0,0,120,120],
 "radii":[[39,39],[0,0],[0,0],[0,0]],"color":[0,0,0,255]}
]}"#;

#[test]
fn cpu_oracle_matches_the_real_render_display_list() {
    let parsed = parse_display_list(REAL_RENDER_COMMANDS).expect("real command JSON must parse");
    let pixels = render_cpu_commands(&parsed).expect("CPU oracle must render");
    // The renderer CLI prints `digest=c840f5c3f204b3cd` over these RGBA bytes.
    assert_eq!(pixels.len(), 120 * 120 * 4);
    assert_eq!(digest_hex(&pixels), "c840f5c3f204b3cd");
    let ppm = opurom_gpu::ppm_bytes(120, 120, &pixels).expect("PPM must encode");
    assert_eq!(digest_hex(&ppm), "eeec1b895735eaf9");
}

/// The old scalar `rounded` command keeps its bytes and its pixels.
#[test]
fn the_scalar_rounded_command_is_unchanged() {
    let json = r#"{"version":1,"width":8,"height":4,"background":[10,20,30,255],"commands":[
{"kind":"rounded","rect":[4,0,4,4],"clip":[0,0,8,4],"radius":2,"color":[10,60,220,255]}
]}"#;
    let parsed = parse_display_list(json).expect("scalar rounded must parse");
    let words = parsed.pack().expect("packing must succeed");
    assert_eq!(words.len(), COMMAND_WORDS);
    assert_eq!(words[0], 1, "the scalar kind stays 1");
    assert_eq!(
        words[9],
        (2 + COORD_BIAS) as u32,
        "the scalar radius slot stays biased"
    );
    assert_eq!(words[11], 0, "the per-corner radius words stay empty");
    let pixels = render_cpu_commands(&parsed).unwrap();
    assert_eq!(
        digest_hex(&pixels),
        digest_hex(&render_cpu_commands(&parsed).unwrap())
    );
}
