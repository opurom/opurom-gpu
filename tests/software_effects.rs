//! Software-backend effect parity: byte-pinned determinism on every host,
//! plus Metal readback and cross-repo byte equality where the host has them.
//!
//! `render_cpu_passes` mirrors the ordered pass plan, so the software backend
//! executes all eight effect kinds. This file pins those bytes for every host,
//! compares them with the Metal readback on macOS, and runs `raster --backend
//! software` against the independent `opurom-render` binary wherever that
//! sibling is built — no GPU is required for that proof.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use opurom_gpu::{
    digest_hex, parse_display_list, ppm_bytes, render_cpu_commands, render_cpu_passes, Backend,
    Command as DrawCommand, DisplayList,
};

/// One display list carrying every regular kind and all eight effect kinds,
/// in the order the renderer emits them: snapshot, silhouette, local effects,
/// then the two whole-frame blurs.
const ALL_EFFECTS: &str = r#"{"version":1,"width":48,"height":32,"background":[10,20,30,255],"commands":[
{"kind":"solid","rect":[-2,-1,20,16],"clip":[0,0,48,32],"color":[200,100,50,128]},
{"kind":"rounded","rect":[8,4,16,12],"clip":[0,0,48,32],"radius":3,"color":[10,60,220,255]},
{"kind":"gradient","rect":[0,16,48,8],"clip":[0,0,48,32],"radius":0,"top":[16,32,48,255],"bottom":[240,192,144,255],"alpha":255},
{"kind":"gradient_horizontal","rect":[24,0,24,10],"clip":[0,0,48,32],"radius":2,"top":[255,0,64,200],"bottom":[0,128,255,200],"alpha":200},
{"kind":"particles","rect":[4,18,40,12],"clip":[0,0,48,32],"count":24,"seed":7,"radius":2,"color":[255,240,128,255]},
{"kind":"effect_backdrop"},
{"kind":"effect_mask","rect":[8,4,16,12],"clip":[0,0,48,32],"bounds":[4,2,24,16],"shape_radius":3,"shape":1},
{"kind":"effect_shadow","bounds":[4,2,24,16],"radius":4,"dx":3,"dy":2,"alpha":768,"color":[0,0,0,255]},
{"kind":"effect_glow","bounds":[24,0,24,12],"radius":3,"alpha":512,"color":[40,255,120,255]},
{"kind":"effect_glass","rect":[8,4,16,12],"clip":[0,0,48,32],"bounds":[4,2,24,16],"shape_radius":3,"radius":3,"alpha":640,"color":[255,255,255,255]},
{"kind":"effect_liquid","rect":[0,16,48,8],"clip":[0,0,48,32],"bounds":[0,14,48,12],"shape_radius":0,"denom":48,"alpha":512,"color":[255,220,80,255]},
{"kind":"effect_blur","radius":2},
{"kind":"effect_gaussian_blur","radius":3}
]}"#;

/// Pinned P6 digest of the fixture above. Any byte drift in the software
/// effect path fails here on every host, GPU or not.
const ALL_EFFECTS_PPM_DIGEST: &str = "b344de17bf75a4b8";

const EFFECT_KINDS: [&str; 8] = [
    "effect_backdrop",
    "effect_mask",
    "effect_shadow",
    "effect_glow",
    "effect_glass",
    "effect_liquid",
    "effect_blur",
    "effect_gaussian_blur",
];

const SIBLING_IR: &str = "../opurom-render/tests/fixtures/effects.opuir.json";

fn skip(message: &str) {
    use std::io::Write;
    // Direct stderr writes keep the SKIP notice visible without --nocapture.
    let _ = writeln!(std::io::stderr().lock(), "SKIP: {message}");
}

fn fixture() -> DisplayList {
    for kind in EFFECT_KINDS {
        assert!(
            ALL_EFFECTS.contains(&format!("\"kind\":\"{kind}\"")),
            "the fixture must carry {kind}"
        );
    }
    parse_display_list(ALL_EFFECTS).expect("all-effect fixture must parse")
}

fn sibling_renderer() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("Opurom_RENDER_BIN") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let path = PathBuf::from("../opurom-render/target/debug/opurom-render");
    path.is_file().then_some(path)
}

/// The whole effect path, pinned on every host: the library oracle, the
/// `raster --backend software` CLI and a repeated render must all produce the
/// same bytes, and those bytes must carry the digest above.
#[test]
fn software_oracle_pins_every_effect_kind_on_any_host() {
    let list = fixture();
    let pixels = render_cpu_passes(&list).expect("software must render effects");
    let again = render_cpu_passes(&list).expect("software must render the fixture twice");
    assert_eq!(again, pixels, "repeated renders must be byte-identical");

    let expected = ppm_bytes(list.width, list.height, &pixels).expect("PPM encoding must work");
    assert_eq!(digest_hex(&expected), ALL_EFFECTS_PPM_DIGEST);

    let backend_ppm = Backend::Software
        .run_ppm(&list)
        .expect("the software backend must render");
    assert_eq!(backend_ppm, expected, "backend and library must agree");

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let commands = dir.join("software-effects.commands.json");
    let output = dir.join("software-effects.ppm");
    fs::write(&commands, ALL_EFFECTS).expect("fixture must be writable");
    let raster = Command::new(env!("CARGO_BIN_EXE_opurom-gpu"))
        .arg("raster")
        .arg("--backend")
        .arg("software")
        .arg(&commands)
        .arg(&output)
        .output()
        .expect("opurom-gpu raster must run");
    assert_eq!(
        raster.status.code(),
        Some(0),
        "software raster must exit 0: {}",
        String::from_utf8_lossy(&raster.stderr)
    );
    assert_eq!(
        fs::read(&output).expect("software PPM must exist"),
        expected,
        "the CLI must write the pinned bytes"
    );
}

/// The regular-command slice of the same list renders identically in both CPU
/// paths, so teaching the software backend about effects changed no plain
/// display-list bytes.
#[test]
fn software_oracle_matches_unit_oracle_without_effects() {
    let list = fixture();
    let plain = DisplayList {
        width: list.width,
        height: list.height,
        background: list.background,
        commands: list
            .commands
            .iter()
            .copied()
            .filter(|command| {
                matches!(
                    command,
                    DrawCommand::Solid { .. }
                        | DrawCommand::Rounded { .. }
                        | DrawCommand::Gradient { .. }
                        | DrawCommand::GradientHorizontal { .. }
                        | DrawCommand::Particles { .. }
                )
            })
            .collect(),
    };
    assert_eq!(
        render_cpu_passes(&plain).expect("software must render"),
        render_cpu_commands(&plain).expect("unit oracle must render"),
        "effect-free lists must be byte-identical in both CPU paths"
    );
}

/// Run both binaries on the same committed effect IR, then `cmp` the software
/// PPM against the independent renderer's PPM. No GPU is involved, so this
/// pixel-alignment proof runs on any host that has the sibling renderer.
#[test]
fn software_ppm_matches_independent_software_renderer() {
    let Some(renderer) = sibling_renderer() else {
        skip(
            "sibling opurom-render binary not found at ../opurom-render/target/debug/opurom-render \
             (set Opurom_RENDER_BIN); cannot run the software cross-repo proof",
        );
        return;
    };
    let ir = Path::new(SIBLING_IR);
    if !ir.is_file() {
        skip("committed effect IR fixture ../opurom-render/tests/fixtures/effects.opuir.json not found");
        return;
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    for (width, height, label) in [
        ("96", "160", "software-96x160"),
        ("96", "64", "software-96x64"),
    ] {
        let cpu_ppm = dir.join(format!("effects-{label}-cpu.ppm"));
        let gpu_ppm = dir.join(format!("effects-{label}-gpu.ppm"));
        let commands = dir.join(format!("effects-{label}.commands.json"));

        let render = Command::new(&renderer)
            .arg(ir)
            .args([width, height])
            .arg(&cpu_ppm)
            // Command streams encode crisp geometry without coverage, so the
            // CPU oracle must raster with edge coverage off for parity.
            .args(["--no-edge-aa", "--commands"])
            .arg(&commands)
            .output()
            .expect("opurom-render must run");
        assert!(
            render.status.success(),
            "{label}: opurom-render failed: {}",
            String::from_utf8_lossy(&render.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&render.stdout).lines().count(),
            2,
            "{label}: --commands must not add stdout lines"
        );

        let raster = Command::new(env!("CARGO_BIN_EXE_opurom-gpu"))
            .arg("raster")
            .arg("--backend")
            .arg("software")
            .arg(&commands)
            .arg(&gpu_ppm)
            .output()
            .expect("opurom-gpu raster must run");
        assert_eq!(
            raster.status.code(),
            Some(0),
            "{label}: raster must exit 0: {}",
            String::from_utf8_lossy(&raster.stderr)
        );

        let cpu_bytes = fs::read(&cpu_ppm).expect("software renderer PPM must exist");
        let gpu_bytes = fs::read(&gpu_ppm).expect("GPU software-backend PPM must exist");
        assert_eq!(
            cpu_bytes,
            gpu_bytes,
            "{label}: software-backend PPM must be byte-identical to the renderer \
             (cpu={} gpu={})",
            digest_hex(&cpu_bytes),
            digest_hex(&gpu_bytes)
        );
        println!(
            "PROOF {label}: cmp=identical bytes={} digest={}",
            cpu_bytes.len(),
            digest_hex(&cpu_bytes)
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_readback_matches_software_oracle_on_every_effect_kind() {
    let list = fixture();
    let cpu = render_cpu_passes(&list).expect("software oracle must render");
    let gpu = opurom_gpu::raster_display(&list).expect("Metal raster must succeed");
    assert_eq!(
        gpu.pixels, cpu,
        "Metal readback must byte-match the software oracle on all eight effect kinds"
    );
    let again = opurom_gpu::raster_display(&list).expect("Metal raster must run twice");
    assert_eq!(
        again.pixels, gpu.pixels,
        "repeated Metal runs must be byte-identical"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_readback_matches_software_oracle_on_every_effect_kind() {
    skip("Metal readback requires macOS; the software bytes stay pinned by the all-hosts test");
}
