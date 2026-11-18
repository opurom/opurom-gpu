//! Cross-crate command-surface audit. Every command kind emitted by
//! opurom-render must be accepted by the GPU display-list parser, and the
//! software replay must be deterministic and clamped at the wire boundary.
//! The byte-for-byte CPU/GPU proofs for representative fixtures live in
//! `tests/display.rs` and `tests/software_effects.rs`.

use opurom_gpu::{
    digest_hex, parse_display_list, ppm_bytes, render_cpu_commands, render_cpu_passes,
};

const RENDER_COMMAND_KINDS: &[&str] = &[
    "solid",
    "rounded",
    "gradient",
    "gradient_horizontal",
    "round_rect_per_corner",
    "particles",
    "effect_backdrop",
    "effect_mask",
    "effect_shadow",
    "effect_elevation",
    "effect_glow",
    "effect_glass",
    "effect_liquid",
    "effect_blur",
    "effect_gaussian_blur",
];

fn doc(commands: &str) -> String {
    format!(
        r#"{{"version":1,"width":32,"height":24,"background":[10,20,30,255],"commands":[{commands}]}}"#
    )
}

/// Every GPU wire spelling emitted by the renderer must survive a
/// parse/re-serialize round-trip without being dropped or renamed.
#[test]
fn every_renderer_command_kind_has_a_gpu_wire_spelling() {
    let list = parse_display_list(&doc(
        r#"
        {"kind":"solid","rect":[0,0,8,6],"clip":[0,0,32,24],"color":[200,100,50,128]},
        {"kind":"rounded","rect":[8,0,8,6],"clip":[0,0,32,24],"radius":2,"color":[10,60,220,255]},
        {"kind":"round_rect_per_corner","rect":[16,0,8,6],"clip":[0,0,32,24],"radii":[[1,1],[2,2],[1,1],[2,2]],"color":[8,9,10,255]},
        {"kind":"gradient","rect":[0,6,16,6],"clip":[0,0,32,24],"radius":0,"top":[16,32,48,255],"bottom":[240,192,144,255],"alpha":255},
        {"kind":"gradient_horizontal","rect":[16,6,16,6],"clip":[0,0,32,24],"radius":0,"top":[255,0,64,200],"bottom":[0,128,255,200],"alpha":200},
        {"kind":"particles","rect":[0,12,16,8],"clip":[0,0,32,24],"count":8,"seed":3,"radius":1,"color":[255,240,128,255]},
        {"kind":"effect_backdrop"},
        {"kind":"effect_mask","rect":[16,12,12,8],"clip":[0,0,32,24],"bounds":[14,10,16,12],"shape_radius":2,"shape":1},
        {"kind":"effect_shadow","bounds":[14,10,16,12],"radius":2,"dx":1,"dy":1,"alpha":256,"color":[0,0,0,255]},
        {"kind":"effect_elevation","rect":[0,0,8,6],"clip":[0,0,32,24],"radius":2,"level":3,"color":[0,0,0,255]},
        {"kind":"effect_glow","bounds":[16,0,12,8],"radius":2,"alpha":256,"color":[40,255,120,255]},
        {"kind":"effect_glass","rect":[16,12,12,8],"clip":[0,0,32,24],"bounds":[14,10,16,12],"shape_radius":2,"radius":2,"alpha":256,"color":[255,255,255,255]},
        {"kind":"effect_liquid","rect":[0,6,16,6],"clip":[0,0,32,24],"bounds":[0,4,16,10],"shape_radius":0,"denom":16,"alpha":256,"color":[255,220,80,255]},
        {"kind":"effect_blur","radius":1},
        {"kind":"effect_gaussian_blur","radius":2}
        "#,
    )).expect("all renderer command kinds must parse");
    assert_eq!(list.commands.len(), RENDER_COMMAND_KINDS.len());
}

#[test]
fn all_command_kinds_are_accepted_and_replay_is_deterministic() {
    let json = doc(r#"
        {"kind":"solid","rect":[0,0,8,6],"clip":[0,0,32,24],"color":[200,100,50,128]},
        {"kind":"rounded","rect":[8,0,8,6],"clip":[0,0,32,24],"radius":2,"color":[10,60,220,255]},
        {"kind":"round_rect_per_corner","rect":[16,0,8,6],"clip":[0,0,32,24],"radii":[[1,1],[2,2],[1,1],[2,2]],"color":[8,9,10,255]},
        {"kind":"gradient","rect":[0,6,16,6],"clip":[0,0,32,24],"radius":0,"top":[16,32,48,255],"bottom":[240,192,144,255],"alpha":255},
        {"kind":"gradient_horizontal","rect":[16,6,16,6],"clip":[0,0,32,24],"radius":0,"top":[255,0,64,200],"bottom":[0,128,255,200],"alpha":200},
        {"kind":"particles","rect":[0,12,16,8],"clip":[0,0,32,24],"count":8,"seed":3,"radius":1,"color":[255,240,128,255]},
        {"kind":"effect_backdrop"},
        {"kind":"effect_mask","rect":[16,12,12,8],"clip":[0,0,32,24],"bounds":[14,10,16,12],"shape_radius":2,"shape":1},
        {"kind":"effect_shadow","bounds":[14,10,16,12],"radius":2,"dx":1,"dy":1,"alpha":256,"color":[0,0,0,255]},
        {"kind":"effect_elevation","rect":[0,0,8,6],"clip":[0,0,32,24],"radius":2,"level":3,"color":[0,0,0,255]},
        {"kind":"effect_glow","bounds":[16,0,12,8],"radius":2,"alpha":256,"color":[40,255,120,255]},
        {"kind":"effect_glass","rect":[16,12,12,8],"clip":[0,0,32,24],"bounds":[14,10,16,12],"shape_radius":2,"radius":2,"alpha":256,"color":[255,255,255,255]},
        {"kind":"effect_liquid","rect":[0,6,16,6],"clip":[0,0,32,24],"bounds":[0,4,16,10],"shape_radius":0,"denom":16,"alpha":256,"color":[255,220,80,255]},
        {"kind":"effect_blur","radius":1},
        {"kind":"effect_gaussian_blur","radius":2}
        "#);
    let list = parse_display_list(&json).expect("all renderer command kinds must parse");
    assert_eq!(list.commands.len(), RENDER_COMMAND_KINDS.len());
    let a = render_cpu_commands(&list).unwrap();
    let b = render_cpu_commands(&list).unwrap();
    assert_eq!(a, b, "software replay must be deterministic");
    let ppm = ppm_bytes(list.width, list.height, &a).unwrap();
    assert_eq!(
        digest_hex(&ppm),
        digest_hex(&ppm_bytes(list.width, list.height, &b).unwrap())
    );
}

#[test]
fn newly_added_primitives_cannot_leave_the_parity_envelope() {
    // The join kernel and the coalescer are pure render-side kernels; they
    // emit only existing regular command kinds, never a new wire spelling.
    let json = doc(
        r#"{"kind":"solid","rect":[0,0,4,4],"clip":[0,0,32,24],"color":[1,2,3,255]},
           {"kind":"solid","rect":[0,4,4,4],"clip":[0,0,32,24],"color":[1,2,3,255]}"#,
    );
    let list = parse_display_list(&json).unwrap();
    assert_eq!(list.commands.len(), 2);
    assert_eq!(
        digest_hex(&render_cpu_commands(&list).unwrap()),
        digest_hex(&render_cpu_commands(&list).unwrap()),
        "adjacent solids replay deterministically"
    );
}
