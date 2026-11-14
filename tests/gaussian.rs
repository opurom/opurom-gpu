//! Tests for the `effect_gaussian_blur` display-list command: exact packed
//! words, pinned CPU pixel bytes, radius bounds on both validation paths,
//! panic-free malformed input, parsed/struct parity and real Metal equality.

use opurom_gpu::{
    gaussian_blur_cpu, parse_display_list, render_cpu_commands, Command, DisplayList, GpuError,
    GAUSSIAN_MAX_RADIUS, GAUSSIAN_MIN_RADIUS, KERNEL_BATCH, KERNEL_GAUSSIAN_H, KERNEL_GAUSSIAN_V,
    PARAM_WORDS,
};

fn header(width: u32, height: u32, background: [u8; 4], commands: &str) -> String {
    format!(
        "{{\"version\":1,\"width\":{width},\"height\":{height},\"background\":[{},{},{},{}],\"commands\":[{commands}]}}",
        background[0], background[1], background[2], background[3]
    )
}

const GAUSSIAN_COMMAND: &str = "{\"kind\":\"effect_gaussian_blur\",\"radius\":3}";

#[test]
fn gaussian_blur_parses_and_packs_exact_weight_words() {
    let list = parse_display_list(&header(8, 4, [0, 0, 0, 255], GAUSSIAN_COMMAND)).unwrap();
    assert_eq!(
        list.commands,
        vec![Command::EffectGaussianBlur { radius: 3 }]
    );
    let (words, passes) = list.build_passes().unwrap();
    assert!(words.is_empty(), "an effect emits no regular command words");
    assert_eq!(passes.len(), 3);
    let kernels: Vec<u32> = passes.iter().map(|p| p.kernel).collect();
    assert_eq!(
        kernels,
        vec![KERNEL_BATCH, KERNEL_GAUSSIAN_H, KERNEL_GAUSSIAN_V]
    );
    // Binomial row C(6, 0..=6) = [1, 6, 15, 20, 15, 6, 1], weight sum 64.
    let mut expected = [0u32; PARAM_WORDS];
    expected[0] = 8;
    expected[1] = 4;
    expected[2] = 3;
    expected[3] = 64;
    for (j, weight) in [1u32, 6, 15, 20, 15, 6, 1].iter().enumerate() {
        expected[4 + j] = *weight;
    }
    assert_eq!(passes[1].params, expected);
    assert_eq!(passes[2].params, expected);
}

#[test]
fn gaussian_blur_cpu_pins_exact_pixel_bytes() {
    // A single impulse; radius 1 kernel [1, 2, 1] (sum 4), clamped edges.
    let input = [255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255];
    assert_eq!(
        gaussian_blur_cpu(&input, 3, 1, 1).unwrap(),
        [191, 0, 0, 255, 64, 0, 0, 255, 0, 0, 0, 255].to_vec()
    );
    // A uniform frame is a fixed point of a normalized blur.
    let flat = [7u8, 8, 9, 255].repeat(4 * 3);
    assert_eq!(gaussian_blur_cpu(&flat, 4, 3, 2).unwrap(), flat);
}

#[test]
fn gaussian_blur_radius_bounds_are_checked_on_both_paths() {
    for (radius, expected) in [
        (-1i64, "gaussian radius -1 must be 1..=8"),
        (0, "gaussian radius 0 must be 1..=8"),
        (9, "gaussian radius 9 must be 1..=8"),
    ] {
        let command = format!("{{\"kind\":\"effect_gaussian_blur\",\"radius\":{radius}}}");
        let error = parse_display_list(&header(8, 4, [0, 0, 0, 255], &command))
            .expect_err("out-of-range radius was accepted");
        assert!(
            matches!(error, GpuError::InvalidDisplayList(_)),
            "{error:?}"
        );
        assert!(error.to_string().contains(expected), "{error}");
    }
    for radius in [GAUSSIAN_MIN_RADIUS, GAUSSIAN_MAX_RADIUS] {
        let command = format!("{{\"kind\":\"effect_gaussian_blur\",\"radius\":{radius}}}");
        let list = parse_display_list(&header(8, 4, [0, 0, 0, 255], &command)).unwrap();
        assert_eq!(list.build_passes().unwrap().1.len(), 3);
    }
    // Public fields bypass the parser; DisplayList::validate() rechecks them.
    let direct = DisplayList {
        width: 8,
        height: 4,
        background: [0, 0, 0, 255],
        commands: vec![Command::EffectGaussianBlur { radius: 99 }],
    };
    for result in [
        direct.pack().map(|_| ()),
        render_cpu_commands(&direct).map(|_| ()),
    ] {
        let error = result.expect_err("constructed list accepted an invalid radius");
        assert!(
            error
                .to_string()
                .contains("gaussian radius 99 must be 1..=8"),
            "{error}"
        );
    }
    assert!(gaussian_blur_cpu(&[0u8; 4], 1, 1, 0)
        .unwrap_err()
        .to_string()
        .contains("1..=8"));
}

#[test]
fn gaussian_blur_malformed_commands_never_panic() {
    let cases = [
        "{\"kind\":\"effect_gaussian_blur\"}",
        "{\"kind\":\"effect_gaussian_blur\",\"radius\":}",
        "{\"kind\":\"effect_gaussian_blur\",\"radius\":\"wide\"}",
        "{\"kind\":\"effect_gaussian_blur\",\"radius\":3,\"extra\":1}",
        "{\"kind\":\"effect_gaussian_blur\",\"radius\":3",
        "{\"kind\":\"effect_gaussian_blur\",\"radius\":3.5}",
        "{\"kind\":\"effect_gaussian_blur\",\"radius\":999999999999999999999}",
    ];
    for (index, command) in cases.iter().enumerate() {
        let json = header(8, 4, [0, 0, 0, 255], command);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parse_display_list(&json)));
        match result {
            Ok(Err(error)) => assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "case {index}: {error}"
            ),
            Ok(Ok(_)) => panic!("case {index} was accepted"),
            Err(_) => panic!("case {index} panicked"),
        }
    }
}

#[test]
fn gaussian_blur_parsed_and_constructed_streams_agree() {
    let parsed = parse_display_list(&header(8, 4, [10, 20, 30, 255], GAUSSIAN_COMMAND)).unwrap();
    let constructed = DisplayList {
        width: 8,
        height: 4,
        background: [10, 20, 30, 255],
        commands: vec![Command::EffectGaussianBlur { radius: 3 }],
    };
    assert_eq!(constructed, parsed);
    assert_eq!(
        constructed.build_passes().unwrap(),
        parsed.build_passes().unwrap()
    );
    let blur = |list: &DisplayList| {
        gaussian_blur_cpu(
            &render_cpu_commands(list).unwrap(),
            list.width,
            list.height,
            3,
        )
        .unwrap()
    };
    assert_eq!(blur(&parsed), blur(&constructed));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gaussian_blur_matches_cpu_reference() {
    let prefix = r#"{"version":1,"width":8,"height":4,"background":[10,20,30,255],"commands":[
{"kind":"solid","rect":[0,0,3,4],"clip":[0,0,8,4],"color":[200,40,20,255]},
{"kind":"solid","rect":[5,0,3,4],"clip":[0,0,8,4],"color":[20,40,220,255]}"#;
    for radius in [1i64, 3, 8] {
        let json = format!("{prefix},{{\"kind\":\"effect_gaussian_blur\",\"radius\":{radius}}}]}}");
        let list = opurom_gpu::parse_display_list(&json).expect("fixture must parse");
        let base = opurom_gpu::render_cpu_commands(&list).expect("CPU oracle must render");
        let cpu = opurom_gpu::gaussian_blur_cpu(&base, list.width, list.height, radius)
            .expect("CPU Gaussian reference must render");
        let gpu = opurom_gpu::raster_display(&list).expect("Metal raster must succeed");
        assert!(!gpu.device_name.is_empty());
        assert_eq!(
            gpu.pixels, cpu,
            "Metal Gaussian readback must byte-match the CPU reference at radius {radius}"
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_gaussian_blur_reports_metal_unavailable() {
    let json = r#"{"version":1,"width":2,"height":2,"background":[0,0,0,255],"commands":[{"kind":"effect_gaussian_blur","radius":1}]}"#;
    assert!(matches!(
        opurom_gpu::raster_json_ppm(json),
        Err(opurom_gpu::GpuError::MetalUnavailable(_))
    ));
}
