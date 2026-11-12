#[cfg(not(target_os = "macos"))]
fn skip(message: &str) {
    use std::io::Write;

    // Direct stderr writes keep the SKIP notice visible without --nocapture.
    let _ = writeln!(std::io::stderr().lock(), "SKIP: {message}");
}

#[test]
fn gpu_readback_matches_cpu_reference() {
    #[cfg(target_os = "macos")]
    {
        let output = opurom_gpu::self_test_output().expect("Metal self-test must succeed");
        assert!(output.starts_with("PASS device="));
        assert!(output.contains("cases=4"));
        assert!(output.contains("digest=f8b4088439129f9c"));
        let device = output
            .strip_prefix("PASS device=")
            .and_then(|tail| tail.split(" cases=").next())
            .unwrap_or_default();
        assert!(!device.is_empty());
        assert!(!device.contains("simulated"));
    }

    #[cfg(not(target_os = "macos"))]
    {
        skip("GPU/CPU equality requires macOS Metal; asserting MetalUnavailable instead");
        assert!(matches!(
            opurom_gpu::self_test_output(),
            Err(opurom_gpu::GpuError::MetalUnavailable(_))
        ));
    }
}

#[test]
fn probe_reports_real_device_and_digest() {
    #[cfg(target_os = "macos")]
    {
        let output = opurom_gpu::probe_output().expect("Metal probe must succeed");
        assert!(output.contains("device="));
        assert!(output.contains("backend=metal-compute"));
        assert!(output.contains("size=64x48"));
        assert!(output.contains("bytes=12288"));
        assert!(output.contains("dispatch_ms="));
        assert!(output.contains("digest=704ffb40489246c5"));
        let device = opurom_gpu::render(&opurom_gpu::PROBE_PARAMS)
            .expect("Metal device must be available")
            .device_name;
        assert!(output.contains(&device));
        assert!(!device.is_empty());
    }

    #[cfg(not(target_os = "macos"))]
    {
        skip("probe requires macOS Metal; asserting MetalUnavailable instead");
        assert!(matches!(
            opurom_gpu::probe_output(),
            Err(opurom_gpu::GpuError::MetalUnavailable(_))
        ));
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_commands_never_fake_success() {
    skip("GPU commands are unavailable off macOS; both must return MetalUnavailable");
    assert!(matches!(
        opurom_gpu::probe_output(),
        Err(opurom_gpu::GpuError::MetalUnavailable(_))
    ));
    assert!(matches!(
        opurom_gpu::self_test_output(),
        Err(opurom_gpu::GpuError::MetalUnavailable(_))
    ));
}
