//! R6 display-list raster tests: Metal/software equality and CLI error paths.
//! The R9 effect equality proof lives in `tests/r9.rs`.
//!
//! The cross-repo test locates the sibling `opurom-render` binary, renders the
//! rect/gradient fixture and the golden counter frame with `--commands`, then
//! runs `opurom-gpu raster` and compares both PPMs byte-for-byte. That is a
//! real GPU/software pixel-consistency proof between two independent crates.

use std::fs;
#[cfg(target_os = "macos")]
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[cfg(target_os = "macos")]
const FIXTURE_IR: &str = "../opurom-render/tests/fixtures/r6_rect_gradient.opuir.json";
#[cfg(target_os = "macos")]
const COUNTER_IR: &str = "../opurom-spec/counter.opuir.json";

#[cfg(target_os = "macos")]
fn sibling_renderer() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("OPUROM_RENDER_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
        return None;
    }
    let path = PathBuf::from("../opurom-render/target/debug/opurom-render");
    path.is_file().then_some(path)
}

fn skip(message: &str) {
    use std::io::Write;
    // Direct stderr writes keep the SKIP notice visible without --nocapture.
    let _ = writeln!(std::io::stderr().lock(), "SKIP: {message}");
}

#[cfg(target_os = "macos")]
fn ppm_digest(bytes: &[u8]) -> String {
    opurom_gpu::digest_hex(bytes)
}

/// Render one IR with the independent software renderer, dump the display list,
/// execute it through `opurom-gpu raster`, and assert byte-identical PPMs.
#[cfg(target_os = "macos")]
fn assert_cross_repo_byte_equality(
    renderer: &Path,
    ir_path: &Path,
    width: &str,
    height: &str,
    label: &str,
) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cpu_ppm = dir.join(format!("r6-{label}-cpu.ppm"));
    let gpu_ppm = dir.join(format!("r6-{label}-gpu.ppm"));
    let commands = dir.join(format!("r6-{label}.commands.json"));

    let render = Command::new(renderer)
        .arg(ir_path)
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

    let cpu_bytes = fs::read(&cpu_ppm).expect("software PPM must exist");
    let gpu_bytes = fs::read(&gpu_ppm).expect("GPU PPM must exist");
    assert_eq!(
        cpu_bytes,
        gpu_bytes,
        "{label}: GPU PPM must be byte-identical to the software PPM (cpu={} gpu={})",
        ppm_digest(&cpu_bytes),
        ppm_digest(&gpu_bytes)
    );
    assert!(
        cpu_bytes.starts_with(b"P6\n"),
        "{label}: P6 header required"
    );
    println!(
        "PROOF {label}: cmp=identical bytes={} digest={}",
        cpu_bytes.len(),
        ppm_digest(&cpu_bytes)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_ppm_is_byte_identical_to_independent_software_renderer() {
    let Some(renderer) = sibling_renderer() else {
        skip(
            "sibling opurom-render binary not found at ../opurom-render/target/debug/opurom-render \
             (set OPUROM_RENDER_BIN); cannot run the cross-repo byte-equality proof",
        );
        return;
    };

    assert_cross_repo_byte_equality(
        &renderer,
        Path::new(FIXTURE_IR),
        "96",
        "64",
        "rect-gradient",
    );
    // The golden frame exercises CPU-decomposed glyph rects on top of shapes.
    assert_cross_repo_byte_equality(&renderer, Path::new(COUNTER_IR), "320", "240", "counter");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_ppm_is_byte_identical_to_independent_software_renderer() {
    skip(
        "cross-repo GPU/software byte equality requires macOS Metal; \
         sibling renderer availability is not evaluated on this host",
    );
    // A valid display list must never fake success off macOS.
    let json = r#"{"version":1,"width":1,"height":1,"background":[0,0,0,255],"commands":[]}"#;
    assert!(matches!(
        opurom_gpu::raster_json_ppm(json),
        Err(opurom_gpu::GpuError::MetalUnavailable(_))
    ));
}

#[test]
fn raster_cli_error_paths_are_readable() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let missing = dir.join("r6-does-not-exist.json");
    let bad_json = dir.join("r6-bad.json");
    fs::write(&bad_json, "{\"version\":1,").unwrap();

    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_opurom-gpu"))
            .args(args)
            .output()
            .unwrap()
    };

    // Missing input file: exit 1 with a readable diagnostic.
    let out = run(&[
        "raster",
        missing.to_str().unwrap(),
        dir.join("out.ppm").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("cannot read display list"), "{stderr}");

    // Malformed display list: rejected before any GPU work, exit 1.
    let out = run(&[
        "raster",
        bad_json.to_str().unwrap(),
        dir.join("out.ppm").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("InvalidDisplayList"), "{stderr}");

    // Wrong arity / unknown command: usage, exit 2.
    let out = run(&["raster"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("Usage:"));
    let out = run(&["frobnicate"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("Usage:"));
}

#[cfg(target_os = "macos")]
#[test]
fn raster_rejects_unwritable_output_path() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let json = dir.join("r6-ok.commands.json");
    fs::write(
        &json,
        r#"{"version":1,"width":2,"height":1,"background":[1,2,3,255],"commands":[]}"#,
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_opurom-gpu"))
        .arg("raster")
        .arg(&json)
        .arg(dir.join("no-such-dir/out.ppm"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("cannot write"), "{stderr}");
}

#[cfg(target_os = "macos")]
#[test]
fn metal_display_raster_matches_cpu_unit_oracle() {
    let json = r#"{"version":1,"width":8,"height":4,"background":[10,20,30,255],"commands":[
{"kind":"solid","rect":[-2,-1,6,4],"clip":[0,0,8,4],"color":[200,100,50,128]},
{"kind":"rounded","rect":[4,0,4,4],"clip":[0,0,8,4],"radius":2,"color":[10,60,220,255]},
{"kind":"gradient","rect":[0,2,8,2],"clip":[0,0,8,4],"radius":0,"top":[16,32,48,255],"bottom":[240,192,144,255],"alpha":255},
{"kind":"gradient","rect":[1,0,6,4],"clip":[0,0,8,4],"radius":2,"top":[255,255,255,128],"bottom":[0,0,0,128],"alpha":128}
]}"#;
    let list = opurom_gpu::parse_display_list(json).expect("fixture must parse");
    let cpu = opurom_gpu::render_cpu_commands(&list).expect("CPU oracle must render");
    let gpu = opurom_gpu::raster_display(&list).expect("Metal raster must succeed");
    assert_eq!(
        gpu.pixels, cpu,
        "Metal readback must byte-match the CPU oracle"
    );
    assert!(!gpu.device_name.is_empty());
    assert_eq!(
        opurom_gpu::raster_display_ppm(&list).unwrap(),
        opurom_gpu::ppm_bytes(list.width, list.height, &gpu.pixels).unwrap()
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_display_raster_reports_metal_unavailable() {
    skip("display-list raster requires macOS Metal; asserting MetalUnavailable instead");
    let json = r#"{"version":1,"width":2,"height":2,"background":[0,0,0,255],"commands":[]}"#;
    assert!(matches!(
        opurom_gpu::raster_json_ppm(json),
        Err(opurom_gpu::GpuError::MetalUnavailable(_))
    ));
}

#[test]
fn ppm_encoding_and_byte_count_validate_inputs() {
    assert_eq!(
        opurom_gpu::ppm_bytes(2, 1, &[1, 2, 3, 255, 4, 5, 6, 255]).unwrap(),
        b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06".to_vec()
    );
    assert!(matches!(
        opurom_gpu::ppm_bytes(2, 1, &[0; 7]),
        Err(opurom_gpu::GpuError::InvalidParameters(_))
    ));
    assert!(matches!(
        opurom_gpu::display_byte_count(0, 1),
        Err(opurom_gpu::GpuError::InvalidParameters(_))
    ));
    assert!(matches!(
        opurom_gpu::display_byte_count(opurom_gpu::MAX_DIMENSION + 1, 1),
        Err(opurom_gpu::GpuError::InvalidParameters(_))
    ));
    assert_eq!(opurom_gpu::display_byte_count(3, 4).unwrap(), 48);
}
