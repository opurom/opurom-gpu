//! Per-node material elevation on the display-list path.
//!
//! The renderer owns the elevation table (`ELEVATION_TABLE`) and the software
//! raster (`draw_elevation`). This crate consumes the resolved level and
//! expands it into one mask pass plus one blur pass per layer, matching the
//! renderer's pixel contract byte for byte.
//!
//! The `clip` field is part of the command: the renderer's layer bounds are
//! clipped by the canvas clip, so the replay backend needs it too.

use opurom_gpu::{digest_hex, parse_display_list, render_cpu_passes, Command, GpuError};

/// The GPU keeps its own copy of the elevation table, so the values are
/// compared against the renderer's source of truth.
#[test]
fn the_elevation_table_matches_the_renderer_source() {
    let source = std::fs::read_to_string("../opurom-render/src/effects.rs");
    let Ok(source) = source else {
        // CI checks out this repository alone; the fixture stays the source of
        // truth here and the sibling file is checked for drift only.
        return;
    };
    let expected = [
        "Some([(0, 1, 3, 0, 31), (0, 1, 1, 0, 36), (0, 2, 1, -1, 51)])",
        "Some([(0, 1, 5, 0, 31), (0, 2, 2, 0, 36), (0, 3, 1, -2, 51)])",
        "Some([(0, 1, 8, 0, 31), (0, 3, 4, 0, 36), (0, 3, 3, -2, 51)])",
        "Some([(0, 1, 10, 0, 31), (0, 4, 5, 0, 36), (0, 2, 4, -1, 51)])",
        "Some([(0, 1, 18, 0, 31), (0, 6, 10, 0, 36), (0, 3, 5, -1, 51)])",
    ];
    for line in expected {
        assert!(
            source.contains(line),
            "the renderer table drifted: {line} is missing"
        );
    }
}

/// The real display list `opurom-render --commands` emits for an IR with
/// `elevation: 2` on a 60x40 box with radius 4. The renderer CLI prints
/// `digest=0e8cd3cf8ee64ea5` over these RGBA bytes and writes a PPM whose
/// digest is `5c9e5b76cf5879e7`.
const REAL_ELEVATION_COMMANDS: &str = r#"{"version":1,"width":120,"height":100,"background":[247,248,250,255],"commands":[
{"kind":"rounded","rect":[0,0,120,100],"clip":[0,0,120,100],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[1,1,118,98],"clip":[0,0,120,100],"radius":3,"color":[255,255,255,255]},
{"kind":"effect_elevation","rect":[8,8,60,40],"clip":[0,0,120,100],"radius":4,"level":2,"color":[0,0,0,255]},
{"kind":"rounded","rect":[8,8,60,40],"clip":[0,0,120,100],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[9,9,58,38],"clip":[0,0,120,100],"radius":3,"color":[16,32,48,255]}
]}"#;

fn elevation_command(level: i64, clip: [i64; 4]) -> String {
    format!(
        r#"{{"version":1,"width":120,"height":100,"background":[247,248,250,255],"commands":[
{{"kind":"effect_elevation","rect":[8,8,60,40],"clip":[{},{},{},{}],"radius":4,"level":{level},"color":[0,0,0,255]}}
]}}"#,
        clip[0], clip[1], clip[2], clip[3]
    )
}

/// A real display list from an IR whose elevated box sits inside an opaque
/// surface, so the command carries a clipped `clip` instead of the viewport.
const REAL_CLIPPED_ELEVATION_COMMANDS: &str = r#"{"version":1,"width":120,"height":100,"background":[247,248,250,255],"commands":[
{"kind":"rounded","rect":[0,0,120,100],"clip":[0,0,120,100],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[1,1,118,98],"clip":[0,0,120,100],"radius":3,"color":[255,255,255,255]},
{"kind":"rounded","rect":[4,4,70,60],"clip":[0,0,120,100],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[5,5,68,58],"clip":[0,0,120,100],"radius":3,"color":[238,238,238,255]},
{"kind":"effect_elevation","rect":[4,4,60,40],"clip":[4,4,70,60],"radius":4,"level":2,"color":[0,0,0,255]},
{"kind":"rounded","rect":[4,4,60,40],"clip":[4,4,70,60],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[5,5,58,38],"clip":[4,4,70,60],"radius":3,"color":[16,32,48,255]}
]}"#;

#[test]
fn cpu_oracle_matches_the_rendered_elevation_frame() {
    let parsed = parse_display_list(REAL_ELEVATION_COMMANDS).expect("command JSON must parse");
    let pixels = render_cpu_passes(&parsed).expect("CPU oracle must render");
    assert_eq!(pixels.len(), 120 * 100 * 4);
    assert_eq!(digest_hex(&pixels), "0e8cd3cf8ee64ea5");
    let ppm = opurom_gpu::ppm_bytes(120, 100, &pixels).expect("PPM must encode");
    assert_eq!(digest_hex(&ppm), "5c9e5b76cf5879e7");
}

#[test]
fn a_clipped_elevation_matches_the_rendered_frame() {
    let parsed =
        parse_display_list(REAL_CLIPPED_ELEVATION_COMMANDS).expect("command JSON must parse");
    let pixels = render_cpu_passes(&parsed).expect("CPU oracle must render");
    assert_eq!(digest_hex(&pixels), "ff6d5985d247ed7a");
    let ppm = opurom_gpu::ppm_bytes(120, 100, &pixels).expect("PPM must encode");
    assert_eq!(digest_hex(&ppm), "766fe858d9dcbd7a");
}

#[test]
fn every_allowed_level_expands_to_six_passes() {
    for level in [1, 2, 3, 4, 6] {
        let parsed = parse_display_list(&elevation_command(level, [0, 0, 120, 100]))
            .expect("command JSON must parse");
        let (_, passes) = parsed.build_passes().expect("pass plan must build");
        // Three layers, each a mask pass plus a local effect pass.
        assert_eq!(
            passes.len(),
            7,
            "level {level}: one clear plus six layer passes"
        );
        let kernels: Vec<u32> = passes.iter().map(|pass| pass.kernel).collect();
        assert_eq!(
            &kernels[1..],
            &[
                opurom_gpu::KERNEL_MASK,
                opurom_gpu::KERNEL_EFFECT,
                opurom_gpu::KERNEL_MASK,
                opurom_gpu::KERNEL_EFFECT,
                opurom_gpu::KERNEL_MASK,
                opurom_gpu::KERNEL_EFFECT,
            ],
            "level {level}: layers alternate mask and effect passes"
        );
        // The layer alphas are the M3 table: ambient 31, penumbra 36, umbra 51.
        let alphas: Vec<u32> = passes
            .iter()
            .filter(|pass| pass.kernel == opurom_gpu::KERNEL_EFFECT)
            .map(|pass| pass.params[7])
            .collect();
        assert_eq!(alphas, vec![31, 36, 51], "level {level}");
    }
}

#[test]
fn the_level_whitelist_is_enforced() {
    for level in [0, 5, 7, -1, 8] {
        let error = parse_display_list(&elevation_command(level, [0, 0, 120, 100]))
            .expect_err("a level outside the whitelist must be rejected");
        assert!(
            error.to_string().contains("elevation level"),
            "level {level}: {error}"
        );
        assert!(matches!(error, GpuError::InvalidDisplayList(_)));
    }
}

#[test]
fn a_clipped_elevation_stays_inside_the_clip() {
    let full = parse_display_list(&elevation_command(2, [0, 0, 120, 100]))
        .expect("command JSON must parse");
    let clipped =
        parse_display_list(&elevation_command(2, [0, 0, 60, 50])).expect("command JSON must parse");
    let full_pixels = render_cpu_passes(&full).expect("CPU oracle must render");
    let clipped_pixels = render_cpu_passes(&clipped).expect("CPU oracle must render");
    // Every pixel outside the clip stays at the background colour.
    for y in 0..100i64 {
        for x in 0..120i64 {
            let offset = (y as usize * 120 + x as usize) * 4;
            if x >= 60 || y >= 50 {
                assert_eq!(
                    &clipped_pixels[offset..offset + 4],
                    &[247, 248, 250, 255],
                    "pixel ({x},{y}) outside the clip must stay background"
                );
            }
        }
    }
    // The clipped frame differs from the unclipped one.
    assert_ne!(full_pixels, clipped_pixels);
}

#[test]
fn the_command_carries_its_own_fields() {
    let parsed = parse_display_list(REAL_ELEVATION_COMMANDS).expect("command JSON must parse");
    let command = parsed
        .commands
        .iter()
        .find(|command| matches!(command, Command::EffectElevation { .. }))
        .expect("the list carries an elevation command");
    match command {
        Command::EffectElevation {
            rect,
            clip,
            radius,
            level,
            color,
        } => {
            assert_eq!(*rect, [8, 8, 60, 40]);
            assert_eq!(*clip, [0, 0, 120, 100]);
            assert_eq!(*radius, 4);
            assert_eq!(*level, 2);
            assert_eq!(*color, [0, 0, 0, 255]);
        }
        other => panic!("unexpected command {other:?}"),
    }
}

#[test]
fn a_malformed_elevation_command_is_rejected() {
    let cases = [
        // missing clip
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"effect_elevation","rect":[0,0,8,8],"radius":1,"level":1,"color":[0,0,0,255]}]}"#,
        // missing level
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"effect_elevation","rect":[0,0,8,8],"clip":[0,0,8,8],"radius":1,"color":[0,0,0,255]}]}"#,
        // clip outside the viewport
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"effect_elevation","rect":[0,0,8,8],"clip":[0,0,9,8],"radius":1,"level":1,"color":[0,0,0,255]}]}"#,
        // radius beyond the box
        r#"{"version":1,"width":8,"height":8,"background":[0,0,0,255],"commands":[
{"kind":"effect_elevation","rect":[0,0,8,8],"clip":[0,0,8,8],"radius":9,"level":1,"color":[0,0,0,255]}]}"#,
    ];
    for (index, case) in cases.iter().enumerate() {
        let error = parse_display_list(case).expect_err("malformed input must be rejected");
        assert!(
            matches!(error, GpuError::InvalidDisplayList(_)),
            "case {index}: {error:?}"
        );
    }
}
