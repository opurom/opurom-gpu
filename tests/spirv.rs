//! SPIR-V artifact integrity: the committed `shaders/spirv/*.spv` blobs must
//! exist, carry the SPIR-V magic, and — when a glslangValidator is available
//! on the host — recompile byte-identically from the committed GLSL sources,
//! so a stale artifact fails the suite instead of drifting silently.

use std::path::PathBuf;
use std::process::Command;

const KERNELS: [&str; 8] = [
    "raster_batch",
    "snapshot_frame",
    "build_mask",
    "effect_local",
    "blur_horizontal",
    "blur_vertical",
    "gaussian_horizontal",
    "gaussian_vertical",
];

fn skip(message: &str) {
    use std::io::Write;
    // Direct stderr writes keep the SKIP notice visible without --nocapture.
    let _ = writeln!(std::io::stderr().lock(), "SKIP: {message}");
}

fn validator() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("OPUROM_GLSLANG") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let path = PathBuf::from(env!("HOME"))
        .join("tools")
        .join("glslangValidator");
    path.is_file().then_some(path)
}

fn blob_path(kernel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("spirv")
        .join(format!("{kernel}.spv"))
}

#[test]
fn committed_spirv_blobs_are_valid_and_fresh() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    for kernel in KERNELS {
        let blob = fs::read_bytes(kernel);
        assert_eq!(
            &blob[0..4],
            &[0x03, 0x02, 0x23, 0x07],
            "{kernel}: SPIR-V magic required"
        );
        assert!(blob.len() > 100, "{kernel}: blob is implausibly small");
    }

    let Some(validator) = validator() else {
        skip(
            "no glslangValidator on this host (~/tools/glslangValidator or $OPUROM_GLSLANG); \
             asserting committed blob validity only",
        );
        return;
    };
    for kernel in KERNELS {
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join(format!("{kernel}.comp"));
        let out = dir.join(format!("{kernel}.spv"));
        let compiled = Command::new(&validator)
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.0")
            .arg("-o")
            .arg(&out)
            .arg(&source)
            .output()
            .expect("glslangValidator must run");
        assert!(
            compiled.status.success(),
            "{kernel}: recompile failed: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        assert_eq!(
            std::fs::read(&out).expect("recompiled blob must exist"),
            fs::read_bytes(kernel),
            "{kernel}: committed blob is stale; run scripts/generate_spirv.sh"
        );
    }
}

mod fs {
    pub fn read_bytes(kernel: &str) -> Vec<u8> {
        let path = super::blob_path(kernel);
        std::fs::read(&path).unwrap_or_else(|error| panic!("{kernel}: {error}"))
    }
}
