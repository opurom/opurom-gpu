//! Display-list raster tests: Metal/software equality and CLI error paths.
//! The effect equality proof lives in `tests/effects.rs`.
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
const FIXTURE_IR: &str = "../opurom-render/tests/fixtures/rect-gradient.opuir.json";
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
    let cpu_ppm = dir.join(format!("display-{label}-cpu.ppm"));
    let gpu_ppm = dir.join(format!("display-{label}-gpu.ppm"));
    let commands = dir.join(format!("display-{label}.commands.json"));

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
    let missing = dir.join("display-does-not-exist.json");
    let bad_json = dir.join("display-bad.json");
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
    let json = dir.join("display-ok.commands.json");
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

#[cfg(target_os = "macos")]
#[test]
fn metal_horizontal_gradient_matches_cpu_unit_oracle() {
    let json = r#"{"version":1,"width":6,"height":3,"background":[10,20,30,255],"commands":[
{"kind":"gradient_horizontal","rect":[-2,-1,8,5],"clip":[0,0,6,3],"radius":0,"top":[200,100,0,255],"bottom":[0,50,200,255],"alpha":255},
{"kind":"gradient_horizontal","rect":[1,1,4,2],"clip":[2,0,4,3],"radius":0,"top":[255,255,255,128],"bottom":[0,0,0,128],"alpha":128},
{"kind":"rounded","rect":[2,0,3,3],"clip":[0,0,6,3],"radius":1,"color":[9,8,7,255]},
{"kind":"gradient","rect":[0,1,6,2],"clip":[0,0,6,3],"radius":0,"top":[16,32,48,255],"bottom":[240,192,144,255],"alpha":255}
]}"#;
    let list = opurom_gpu::parse_display_list(json).expect("fixture must parse");
    let cpu = opurom_gpu::render_cpu_commands(&list).expect("CPU oracle must render");
    let gpu = opurom_gpu::raster_display(&list).expect("Metal raster must succeed");
    assert_eq!(
        gpu.pixels, cpu,
        "Metal readback must byte-match the CPU oracle"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_horizontal_gradient_matches_cpu_unit_oracle() {
    skip("horizontal gradient raster requires macOS Metal; asserting MetalUnavailable instead");
    let json = r#"{"version":1,"width":4,"height":1,"background":[0,0,0,255],"commands":[{"kind":"gradient_horizontal","rect":[0,0,4,1],"clip":[0,0,4,1],"radius":0,"top":[0,0,0,255],"bottom":[255,255,255,255],"alpha":255}]}"#;
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

#[test]
fn frame_validation_and_display_byte_count_agree_on_boundaries() {
    let red = [1u8, 2, 3];
    let make = |w: u32, h: u32| {
        opurom_gpu::FrameParams::new(
            w,
            h,
            opurom_gpu::Rgb::new(red[0], red[1], red[2]),
            opurom_gpu::Rgb::new(red[0], red[1], red[2]),
            opurom_gpu::Rgb::new(red[0], red[1], red[2]),
        )
    };

    let sizes = [
        (0u32, 1u32),
        (1, 0),
        (1, 1),
        (3, 4),
        (opurom_gpu::MAX_DIMENSION, opurom_gpu::MAX_DIMENSION),
        (opurom_gpu::MAX_DIMENSION + 1, 1),
    ];
    for (width, height) in sizes {
        let via_frame = make(width, height).validate();
        let via_count = opurom_gpu::display_byte_count(width, height);
        assert_eq!(
            via_frame.is_ok(),
            via_count.is_ok(),
            "validation paths disagree for {width}x{height}"
        );
        if let (Ok(frame_bytes), Ok(count_bytes)) = (via_frame, via_count) {
            assert_eq!(frame_bytes, count_bytes, "{width}x{height}");
        }
    }

    let max = opurom_gpu::MAX_DIMENSION as usize;
    assert_eq!(
        opurom_gpu::display_byte_count(opurom_gpu::MAX_DIMENSION, opurom_gpu::MAX_DIMENSION),
        Ok(max * max * 4)
    );
}

/// `DisplayList` has public fields, so callers can bypass `parse_display_list`.
#[test]
fn constructed_display_lists_reject_invalid_geometry_without_panicking() {
    use opurom_gpu::{render_cpu_commands, Command, DisplayList, GpuError};

    fn clean_error<T>(
        index: usize,
        path: &str,
        result: std::thread::Result<Result<T, GpuError>>,
    ) -> GpuError {
        match result {
            Ok(Err(error)) => error,
            Ok(Ok(_)) => panic!("case {index} {path} accepted an invalid list"),
            Err(_) => panic!("case {index} {path} panicked"),
        }
    }

    let cases: [(DisplayList, &str); 6] = [
        (
            DisplayList {
                width: 2,
                height: 1,
                background: [0, 0, 0, 255],
                commands: vec![Command::Solid {
                    rect: [1, 0, i64::MAX, 1],
                    clip: [0, 0, 2, 1],
                    color: [1, 2, 3, 255],
                }],
            },
            "exceeds the integer range",
        ),
        (
            DisplayList {
                width: 2,
                height: 1,
                background: [0, 0, 0, 255],
                commands: vec![Command::Rounded {
                    rect: [0, 0, 2, 1],
                    clip: [0, 0, 2, 1],
                    radius: i64::MAX / 2,
                    color: [1, 2, 3, 255],
                }],
            },
            "exceeds the integer range",
        ),
        (
            DisplayList {
                width: 2,
                height: 1,
                background: [0, 0, 0, 255],
                commands: vec![Command::Gradient {
                    rect: [0, 0, 2, i64::MIN],
                    clip: [0, 0, 2, 1],
                    radius: 0,
                    top: [0, 0, 0, 255],
                    bottom: [1, 2, 3, 255],
                    alpha: 255,
                }],
            },
            "exceeds the integer range",
        ),
        (
            DisplayList {
                width: 8,
                height: 4,
                background: [0, 0, 0, 255],
                commands: vec![Command::Solid {
                    rect: [0, 0, 2, 2],
                    clip: [0, 0, 9, 2],
                    color: [1, 2, 3, 255],
                }],
            },
            "clip extends outside the viewport",
        ),
        (
            DisplayList {
                width: 8,
                height: 4,
                background: [0, 0, 0, 255],
                commands: vec![Command::EffectBlur { radius: 33 }],
            },
            "radius 33 must be 0..=32",
        ),
        (
            DisplayList {
                width: 8,
                height: 4,
                background: [0, 0, 0, 255],
                commands: vec![Command::EffectShadow {
                    bounds: [-1, 0, 4, 4],
                    radius: 1,
                    dx: 0,
                    dy: 0,
                    alpha: 0,
                    color: [0, 0, 0, 255],
                }],
            },
            "bounds must be normalized",
        ),
    ];
    for (index, (list, expected)) in cases.iter().enumerate() {
        let packed = clean_error(
            index,
            "pack",
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| list.pack())),
        );
        let rendered = clean_error(
            index,
            "render",
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| render_cpu_commands(list))),
        );
        for (path, error) in [("pack", packed), ("render", rendered)] {
            assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "case {index} {path}: {error}"
            );
            assert!(
                error.to_string().contains(expected),
                "case {index} {path}: {error} must mention {expected:?}"
            );
        }
    }
}
