//! Cross-repo GPU/CPU visual-equality proof for the five effect kernels.
//!
//! This test runs the independent software renderer on a committed effect IR,
//! dumps its command stream, executes that stream with `opurom-gpu raster`, and
//! `cmp`s both PPMs byte-for-byte. It is not a self-comparison against shared
//! code: the CPU frame comes from the separate opurom-render binary.

#[cfg(target_os = "macos")]
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[cfg(target_os = "macos")]
const EFFECT_IR: &str = "../opurom-render/tests/fixtures/effects.opuir.json";

#[cfg(target_os = "macos")]
fn sibling_renderer() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("OPUROM_RENDER_BIN") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let path = PathBuf::from("../opurom-render/target/debug/opurom-render");
    path.is_file().then_some(path)
}

fn skip(message: &str) {
    use std::io::Write;
    // Direct stderr writes keep the SKIP notice visible without --nocapture.
    let _ = writeln!(std::io::stderr().lock(), "SKIP: {message}");
}

/// Run both binaries on the same IR, viewport, state, theme and effects, then
/// compare the two PPM files with the system `cmp` and with byte equality.
#[cfg(target_os = "macos")]
fn assert_effect_byte_equality(renderer: &Path, ir: &Path, width: &str, height: &str, label: &str) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cpu_ppm = dir.join(format!("effects-{label}-cpu.ppm"));
    let gpu_ppm = dir.join(format!("effects-{label}-gpu.ppm"));
    let commands = dir.join(format!("effects-{label}.commands.json"));

    let render = Command::new(renderer)
        .arg(ir)
        .args([width, height])
        .arg(&cpu_ppm)
        .args(["--commands"])
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

    let command_json = fs::read_to_string(&commands).expect("command stream must exist");
    for kind in [
        "effect_blur",
        "effect_shadow",
        "effect_glow",
        "effect_glass",
        "effect_liquid",
    ] {
        assert!(
            command_json.contains(&format!("\"kind\":\"{kind}\"")),
            "{label}: {kind} command missing from the stream"
        );
    }

    let raster = Command::new(env!("CARGO_BIN_EXE_opurom-gpu"))
        .arg("raster")
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

    let compared = Command::new("cmp")
        .arg(&cpu_ppm)
        .arg(&gpu_ppm)
        .status()
        .expect("system cmp must be available");
    assert!(
        compared.success(),
        "{label}: cmp reported a CPU/GPU PPM difference"
    );
    let cpu_bytes = fs::read(&cpu_ppm).expect("software PPM must exist");
    let gpu_bytes = fs::read(&gpu_ppm).expect("GPU PPM must exist");
    assert_eq!(
        cpu_bytes, gpu_bytes,
        "{label}: GPU PPM must be byte-identical to the software PPM"
    );
    assert!(
        cpu_bytes.starts_with(b"P6\n"),
        "{label}: P6 header required"
    );
    println!(
        "PROOF {label}: cmp=identical bytes={} digest={}",
        cpu_bytes.len(),
        opurom_gpu::digest_hex(&cpu_bytes)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn effect_gpu_ppm_is_byte_identical_to_independent_software_renderer() {
    let Some(renderer) = sibling_renderer() else {
        skip(
            "sibling opurom-render binary not found at ../opurom-render/target/debug/opurom-render \
             (set OPUROM_RENDER_BIN); cannot run the effect byte-equality proof",
        );
        return;
    };
    let ir = Path::new(EFFECT_IR);
    if !ir.is_file() {
        skip("committed effect IR fixture ../opurom-render/tests/fixtures/effects.opuir.json not found");
        return;
    }
    assert_effect_byte_equality(&renderer, ir, "96", "64", "all-five-effects");
    // A second viewport exercises different bounds and clip intersections.
    assert_effect_byte_equality(&renderer, ir, "64", "48", "small-viewport");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn effect_gpu_ppm_is_byte_identical_to_independent_software_renderer() {
    skip("effect byte equality requires macOS Metal; asserting MetalUnavailable instead");
    let json = r#"{"version":1,"width":2,"height":2,"background":[0,0,0,255],"commands":[{"kind":"effect_blur","radius":1}]}"#;
    assert!(matches!(
        opurom_gpu::raster_json_ppm(json),
        Err(opurom_gpu::GpuError::MetalUnavailable(_))
    ));
}
