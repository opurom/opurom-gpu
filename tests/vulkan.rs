//! Headless Vulkan command-recording tests: per-primitive translation, typed
//! validation, deterministic byte encoding, decode round-trip, and parity of
//! the recorded stream against the Metal/CPU reference packing.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use opurom_gpu::{
    decode_stream, encode_stream, parse_display_list, record_stream, Backend, BackendAvailability,
    BackendStatus, Command, DisplayList, VkCommand, VkEffect, VkError, VkPipeline, VkPrimitive,
    VulkanBackend,
};

use opurom_gpu::vulkan::{VULKAN_MAGIC as MAGIC, VULKAN_STREAM_VERSION};

fn header(width: u32, height: u32, background: [u8; 4], commands: &str) -> String {
    format!(
        "{{\"version\":1,\"width\":{width},\"height\":{height},\"background\":[{},{},{},{}],\"commands\":[{commands}]}}",
        background[0], background[1], background[2], background[3]
    )
}

/// A fixture that exercises every regular primitive and every effect pass.
const FULL_STREAM: &str = "\
{\"kind\":\"solid\",\"rect\":[-2,-1,6,4],\"clip\":[0,0,8,4],\"color\":[200,100,50,128]},\
{\"kind\":\"rounded\",\"rect\":[4,0,4,4],\"clip\":[0,0,8,4],\"radius\":2,\"color\":[10,60,220,255]},\
{\"kind\":\"gradient\",\"rect\":[0,2,8,2],\"clip\":[0,0,8,4],\"radius\":0,\"top\":[16,32,48,255],\"bottom\":[240,192,144,255],\"alpha\":255},\
{\"kind\":\"gradient_horizontal\",\"rect\":[1,0,6,4],\"clip\":[0,0,8,4],\"radius\":2,\"top\":[255,255,255,128],\"bottom\":[0,0,0,128],\"alpha\":128},\
{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":6,\"seed\":7,\"radius\":1,\"color\":[240,120,60,255]},\
{\"kind\":\"effect_backdrop\"},\
{\"kind\":\"effect_mask\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[0,0,8,4],\"shape_radius\":1,\"shape\":1},\
{\"kind\":\"effect_shadow\",\"bounds\":[0,0,8,4],\"radius\":1,\"dx\":-2,\"dy\":1,\"alpha\":512,\"color\":[1,2,3,255]},\
{\"kind\":\"effect_glow\",\"bounds\":[0,0,8,4],\"radius\":1,\"alpha\":512,\"color\":[4,5,6,255]},\
{\"kind\":\"effect_glass\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[2,2,4,2],\"shape_radius\":1,\"radius\":1,\"alpha\":384,\"color\":[7,8,9,255]},\
{\"kind\":\"effect_liquid\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[2,2,4,2],\"shape_radius\":1,\"denom\":8,\"alpha\":640,\"color\":[10,11,12,255]},\
{\"kind\":\"effect_blur\",\"radius\":2},\
{\"kind\":\"effect_gaussian_blur\",\"radius\":3}";

fn full_list() -> DisplayList {
    parse_display_list(&header(8, 4, [10, 20, 30, 255], FULL_STREAM)).expect("fixture must parse")
}

fn draws(commands: &[VkCommand]) -> Vec<opurom_gpu::VkDraw> {
    commands
        .iter()
        .filter_map(|command| match command {
            VkCommand::Draw(draw) => Some(*draw),
            _ => None,
        })
        .collect()
}

fn dispatches(commands: &[VkCommand]) -> Vec<VkEffect> {
    commands
        .iter()
        .filter_map(|command| match command {
            VkCommand::Dispatch(dispatch) => Some(dispatch.effect),
            _ => None,
        })
        .collect()
}

#[test]
fn every_primitive_translates_to_a_typed_command() {
    let list = full_list();
    let commands = record_stream(&list).expect("record must succeed");

    // The first two commands initialize the image: bind clear, then copy.
    assert!(matches!(
        commands[0],
        VkCommand::BindPipeline {
            pipeline: VkPipeline::ClearImage
        }
    ));
    match commands[1] {
        VkCommand::CopyBufferToImage(copy) => {
            assert_eq!(copy.extent, [8, 4]);
            assert_eq!(copy.bytes_per_row, 32);
            assert_eq!(copy.bytes_per_image, 128);
            assert_eq!(copy.color, [10, 20, 30, 255]);
        }
        other => panic!("second command must be a buffer-to-image copy: {other:?}"),
    }

    let recorded = draws(&commands);
    assert_eq!(recorded.len(), 5, "five regular commands become draws");
    let pipelines: Vec<VkPipeline> = recorded.iter().map(|draw| draw.pipeline).collect();
    assert_eq!(
        pipelines,
        vec![
            VkPipeline::Solid,
            VkPipeline::Rounded,
            VkPipeline::Gradient,
            VkPipeline::GradientHorizontal,
            VkPipeline::Particles,
        ]
    );
    // Rect primitives draw two triangles (six vertices); particles draw points.
    for draw in &recorded[0..4] {
        assert_eq!(draw.primitive, VkPrimitive::TriangleList);
        assert_eq!(draw.vertex_count, 6);
    }
    assert_eq!(recorded[4].primitive, VkPrimitive::PointList);
    assert_eq!(recorded[4].vertex_count, 6);
    assert_eq!(recorded[4].count, 6);

    // Each draw is preceded by a pipeline bind and a scissor set.
    for window in commands.windows(3) {
        if let VkCommand::Draw(draw) = window[2] {
            assert!(
                matches!(window[0], VkCommand::BindPipeline { pipeline } if pipeline == draw.pipeline)
            );
            assert!(matches!(window[1], VkCommand::SetScissor { rect } if rect == draw.scissor));
        }
    }

    // Every effect kind maps to one compute dispatch.
    assert_eq!(
        dispatches(&commands),
        vec![
            VkEffect::Backdrop,
            VkEffect::Mask,
            VkEffect::Shadow,
            VkEffect::Glow,
            VkEffect::Glass,
            VkEffect::Liquid,
            VkEffect::BlurHorizontal,
            VkEffect::BlurVertical,
            VkEffect::GaussianHorizontal,
            VkEffect::GaussianVertical,
        ]
    );
}

#[test]
fn recorded_geometry_and_color_match_the_display_list() {
    let list = full_list();
    let commands = record_stream(&list).expect("record must succeed");
    let recorded = draws(&commands);
    // Solid rect [-2,-1,6,4], clip [0,0,8,4], color [200,100,50,128].
    let solid = recorded[0];
    assert_eq!(
        (
            solid.rect.x,
            solid.rect.y,
            solid.rect.width,
            solid.rect.height
        ),
        (-2, -1, 6, 4)
    );
    assert_eq!(
        (
            solid.scissor.x,
            solid.scissor.y,
            solid.scissor.width,
            solid.scissor.height
        ),
        (0, 0, 8, 4)
    );
    assert_eq!(solid.color, [200, 100, 50, 128]);
    assert_eq!(solid.alpha, 128);

    // The vertical gradient carries both stops and the alpha field.
    let gradient = recorded[2];
    assert_eq!(gradient.color, [16, 32, 48, 255]);
    assert_eq!(gradient.second_color, Some([240, 192, 144, 255]));
    assert_eq!(gradient.alpha, 255);

    // Particles carry the count and seed verbatim.
    let particles = recorded[4];
    assert_eq!((particles.count, particles.seed), (6, 7));
    assert_eq!(particles.radius, 1);
}

#[test]
fn encoding_is_byte_deterministic_and_framed() {
    let list = full_list();
    let first = encode_stream(&list).expect("encode must succeed");
    let second = encode_stream(&list).expect("encode must succeed");
    assert_eq!(first, second, "identical lists must encode identically");

    assert_eq!(&first[0..8], &MAGIC);
    let version = u32::from_le_bytes(first[8..12].try_into().unwrap());
    assert_eq!(version, VULKAN_STREAM_VERSION);

    // The trailing 8 bytes are an FNV-1a checksum over the body.
    let split = first.len() - 8;
    let checksum = u64::from_le_bytes(first[split..].try_into().unwrap());
    assert_eq!(checksum, opurom_gpu::fnv1a(&first[..split]));

    // Re-recording the same list on a different backend instance agrees.
    let mut other = VulkanBackend::new();
    other.record(&list).expect("record must succeed");
    assert_eq!(other.encode(), first);
}

#[test]
fn encoded_stream_decodes_back_to_the_recorded_commands() {
    let list = full_list();
    let mut backend = VulkanBackend::new();
    let recorded = backend.record(&list).expect("record must succeed");
    let bytes = backend.encode();
    let decoded = decode_stream(&bytes).expect("decode must succeed");
    assert_eq!(decoded.commands(), recorded.as_slice());
    assert_eq!(decoded.viewport(), (8, 4));
    assert_eq!(decoded.background(), [10, 20, 30, 255]);
    assert_eq!(decoded.encode(), bytes, "re-encoding must be identical");
}

#[test]
fn decode_rejects_corruption_without_panicking() {
    let list = full_list();
    let bytes = encode_stream(&list).expect("encode must succeed");

    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    assert!(matches!(
        decode_stream(&flipped),
        Err(VkError::InvalidEncoding(_))
    ));

    let mut truncated = bytes.clone();
    truncated.truncate(bytes.len() / 2);
    assert!(decode_stream(&truncated).is_err());

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    // Recompute a valid checksum so the magic check itself is reached.
    let split = bad_magic.len() - 8;
    let body = bad_magic[..split].to_vec();
    bad_magic[split..].copy_from_slice(&opurom_gpu::fnv1a(&body).to_le_bytes());
    assert!(matches!(
        decode_stream(&bad_magic),
        Err(VkError::InvalidEncoding(_))
    ));
}

/// Parity proof: for the same list the recorded draw count equals the number of
/// regular-command words the Metal/CPU path packs, and each draw's geometry and
/// color equal the display-list command it came from. The expected values are
/// derived from the list itself and from `build_passes`, never hardcoded.
#[test]
fn recorded_stream_matches_the_metal_reference_packing() {
    let list = full_list();
    let commands = record_stream(&list).expect("record must succeed");

    // The number of regular draw commands must equal the number of 16-word
    // command blocks that `build_passes` packed for the Metal batch pass.
    let (words, _passes) = list.build_passes().expect("reference packing must succeed");
    let regular_commands = list
        .commands
        .iter()
        .filter(|command| {
            matches!(
                command,
                Command::Solid { .. }
                    | Command::Rounded { .. }
                    | Command::Gradient { .. }
                    | Command::GradientHorizontal { .. }
                    | Command::Particles { .. }
            )
        })
        .count();
    assert_eq!(words.len(), regular_commands * opurom_gpu::COMMAND_WORDS);
    let recorded = draws(&commands);
    assert_eq!(
        recorded.len(),
        regular_commands,
        "one Vulkan draw per packed Metal regular command"
    );

    // Per-command parity: replay the same geometry from the typed display-list
    // commands the Metal batch packs and compare with the recorded draws.
    let regular: Vec<&Command> = list
        .commands
        .iter()
        .filter(|command| {
            matches!(
                command,
                Command::Solid { .. }
                    | Command::Rounded { .. }
                    | Command::Gradient { .. }
                    | Command::GradientHorizontal { .. }
                    | Command::Particles { .. }
            )
        })
        .collect();
    for (draw, command) in recorded.iter().zip(regular.iter()) {
        match **command {
            Command::Solid { rect, clip, color }
            | Command::Rounded {
                rect, clip, color, ..
            } => {
                assert_eq!(
                    [
                        draw.rect.x as i64,
                        draw.rect.y as i64,
                        draw.rect.width as i64,
                        draw.rect.height as i64
                    ],
                    rect
                );
                assert_eq!(
                    [
                        draw.scissor.x as i64,
                        draw.scissor.y as i64,
                        draw.scissor.width as i64,
                        draw.scissor.height as i64
                    ],
                    clip
                );
                assert_eq!(draw.color, color);
            }
            Command::Gradient {
                rect,
                top,
                bottom,
                alpha,
                ..
            }
            | Command::GradientHorizontal {
                rect,
                top,
                bottom,
                alpha,
                ..
            } => {
                assert_eq!(
                    [
                        draw.rect.x as i64,
                        draw.rect.y as i64,
                        draw.rect.width as i64,
                        draw.rect.height as i64
                    ],
                    rect
                );
                assert_eq!(draw.color, top);
                assert_eq!(draw.second_color, Some(bottom));
                assert_eq!(draw.alpha, alpha);
            }
            Command::Particles {
                rect, count, seed, ..
            } => {
                assert_eq!(
                    [
                        draw.rect.x as i64,
                        draw.rect.y as i64,
                        draw.rect.width as i64,
                        draw.rect.height as i64
                    ],
                    rect
                );
                assert_eq!(draw.count, count as u32);
                assert_eq!(draw.seed, seed as u32);
            }
            _ => unreachable!("filtered to regular commands"),
        }
    }
}

#[test]
fn record_json_matches_record_of_the_parsed_list() {
    let json = header(8, 4, [10, 20, 30, 255], FULL_STREAM);
    let mut backend = VulkanBackend::new();
    let from_json = backend
        .record_json(&json)
        .expect("record_json must succeed");
    let from_list = record_stream(&full_list()).expect("record must succeed");
    assert_eq!(from_json, from_list);
}

#[test]
fn validation_rejects_malformed_input_without_panicking() {
    fn expect_vk_error(list: DisplayList) -> VkError {
        match catch_unwind(AssertUnwindSafe(|| list.build_passes())) {
            Ok(Ok(_)) => {}
            Ok(Err(_)) => {}
            Err(_) => panic!("reference packing panicked"),
        }
        match catch_unwind(AssertUnwindSafe(|| record_stream(&list))) {
            Ok(Err(error)) => error,
            Ok(Ok(_)) => panic!("malformed list was accepted"),
            Err(_) => panic!("record panicked"),
        }
    }

    // Out-of-range rect (negative width).
    let err = expect_vk_error(DisplayList {
        width: 8,
        height: 4,
        background: [0, 0, 0, 255],
        commands: vec![Command::Solid {
            rect: [0, 0, 0, 4],
            clip: [0, 0, 8, 4],
            color: [1, 2, 3, 255],
        }],
    });
    assert!(matches!(err, VkError::OutOfRangeRect { .. }), "{err}");

    // Invalid clip: outside the viewport.
    let err = expect_vk_error(DisplayList {
        width: 8,
        height: 4,
        background: [0, 0, 0, 255],
        commands: vec![Command::Solid {
            rect: [0, 0, 2, 2],
            clip: [0, 0, 9, 2],
            color: [1, 2, 3, 255],
        }],
    });
    assert!(matches!(err, VkError::InvalidClip { .. }), "{err}");

    // Negative radius.
    let err = expect_vk_error(DisplayList {
        width: 8,
        height: 4,
        background: [0, 0, 0, 255],
        commands: vec![Command::Rounded {
            rect: [0, 0, 4, 4],
            clip: [0, 0, 8, 4],
            radius: -1,
            color: [1, 2, 3, 255],
        }],
    });
    assert!(matches!(err, VkError::InvalidRadius { .. }), "{err}");

    // Unbounded particle count.
    let err = expect_vk_error(DisplayList {
        width: 8,
        height: 4,
        background: [0, 0, 0, 255],
        commands: vec![Command::Particles {
            rect: [0, 0, 8, 4],
            clip: [0, 0, 8, 4],
            count: 513,
            seed: 1,
            radius: 1,
            color: [1, 2, 3, 255],
        }],
    });
    assert!(matches!(err, VkError::UnboundedCount { .. }), "{err}");

    // Coordinate past the integer range.
    let err = expect_vk_error(DisplayList {
        width: 8,
        height: 4,
        background: [0, 0, 0, 255],
        commands: vec![Command::Solid {
            rect: [i64::MAX, 0, 2, 2],
            clip: [0, 0, 8, 4],
            color: [1, 2, 3, 255],
        }],
    });
    assert!(matches!(err, VkError::OutOfRangeRect { .. }), "{err}");
}

#[test]
fn record_json_rejects_malformed_json_as_a_typed_error() {
    let mut backend = VulkanBackend::new();
    let result = catch_unwind(AssertUnwindSafe(|| backend.record_json("{\"version\":1,")));
    match result {
        Ok(Err(VkError::MalformedDisplayList(_))) => {}
        other => panic!("malformed JSON must be a typed MalformedDisplayList: {other:?}"),
    }
}

#[test]
fn backend_selector_reports_honest_availability() {
    assert_eq!(
        Backend::Software.available(),
        BackendStatus {
            backend: Backend::Software,
            availability: BackendAvailability::Live,
            note: Backend::Software.available().note,
        }
    );
    // Vulkan availability is probed honestly: hosts with a real driver are
    // live, and every other host reports exactly why it is unavailable.
    assert!(matches!(
        Backend::Vulkan.available().availability,
        BackendAvailability::Live | BackendAvailability::Unavailable
    ));

    // Software runs the CPU oracle and yields a P6 PPM. Vulkan only promises
    // pixels on a host where the probe found a usable driver.
    let list = full_list();
    let ppm = Backend::Software
        .run_ppm(&list)
        .expect("software must render");
    assert!(ppm.starts_with(b"P6\n"));
    let vulkan = Backend::Vulkan.run_ppm(&list);
    if opurom_gpu::vulkan_probe().usable() {
        assert_eq!(
            vulkan.expect("vulkan must execute on a probed driver"),
            ppm,
            "vulkan compute must byte-match the CPU oracle"
        );
    } else {
        assert!(
            vulkan.is_err(),
            "without a driver Vulkan must refuse honestly"
        );
    }

    assert_eq!(Backend::parse("vulkan"), Some(Backend::Vulkan));
    assert_eq!(Backend::parse("metal"), Some(Backend::Metal));
    assert_eq!(Backend::parse("software"), Some(Backend::Software));
    assert_eq!(Backend::parse("d3d"), None);
}

/// The loader probe is cached, honest about the device it found, and drives
/// the backend availability: usable hosts are live, every other host reports
/// why it is unavailable. Recording never touches the driver.
#[test]
fn vulkan_probe_reports_what_the_host_has() {
    let probe = opurom_gpu::vulkan_probe();
    let again = opurom_gpu::vulkan_probe();
    assert_eq!(
        probe.note, again.note,
        "the probe is cached once per process"
    );
    assert!(!probe.note.is_empty());
    if probe.usable() {
        assert!(!probe.device_name.is_empty());
        assert!(probe.shader_int64);
        assert!(probe.compute_queue_family.is_some());
        assert!(
            probe.api_version >= 0x40_0000,
            "api version must be at least Vulkan 1.0"
        );
        assert_eq!(
            Backend::Vulkan.available().availability,
            BackendAvailability::Live
        );
        // Recording stays driver-free and byte-deterministic even on hosts
        // with a live driver.
        let list = full_list();
        assert!(!record_stream(&list)
            .expect("recording must work")
            .is_empty());
    } else {
        assert_eq!(
            Backend::Vulkan.available().availability,
            BackendAvailability::Unavailable
        );
    }
}

/// The `backends` CLI reports all three backends and, on a host with a
/// driver, the probed Vulkan device detail line.
#[test]
fn backends_cli_reports_three_backends_and_vulkan_detail() {
    let out = ProcessCommand::new(env!("CARGO_BIN_EXE_opurom-gpu"))
        .arg("backends")
        .output()
        .expect("backends must run");
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("metal="));
    assert!(stdout.contains("vulkan="));
    assert!(stdout.contains("software="));
    if opurom_gpu::vulkan_probe().usable() {
        assert!(stdout.contains("vulkan.device="));
    }
}

/// A compact display list carrying every regular kind and all eight effect
/// kinds, used for the Vulkan/CPU byte-equality gate.
const EXEC_FIXTURE: &str = r#"{"version":1,"width":32,"height":24,"background":[30,40,50,255],"commands":[
{"kind":"solid","rect":[0,0,32,24],"clip":[0,0,32,24],"color":[30,40,50,255]},
{"kind":"rounded","rect":[4,4,10,8],"clip":[0,0,32,24],"radius":2,"color":[200,80,40,255]},
{"kind":"gradient_horizontal","rect":[16,4,12,8],"clip":[0,0,32,24],"radius":0,"top":[10,20,30,255],"bottom":[220,180,140,255],"alpha":255},
{"kind":"particles","rect":[8,12,16,8],"clip":[0,0,32,24],"count":10,"seed":3,"radius":1,"color":[255,255,255,255]},
{"kind":"effect_backdrop"},
{"kind":"effect_mask","rect":[4,4,10,8],"clip":[0,0,32,24],"bounds":[2,2,14,12],"shape_radius":2,"shape":1},
{"kind":"effect_shadow","bounds":[2,2,14,12],"radius":2,"dx":2,"dy":1,"alpha":512,"color":[0,0,0,255]},
{"kind":"effect_glow","bounds":[16,4,12,8],"radius":2,"alpha":512,"color":[40,255,120,255]},
{"kind":"effect_glass","rect":[4,4,10,8],"clip":[0,0,32,24],"bounds":[2,2,14,12],"shape_radius":2,"radius":2,"alpha":512,"color":[255,255,255,255]},
{"kind":"effect_liquid","rect":[16,4,12,8],"clip":[0,0,32,24],"bounds":[16,4,12,8],"shape_radius":0,"denom":24,"alpha":512,"color":[255,200,40,255]},
{"kind":"effect_blur","radius":1},
{"kind":"effect_gaussian_blur","radius":2}
]}"#;

/// A display list carrying one material elevation layer group, used for the
/// Vulkan/CPU byte-equality gate.
const ELEVATION_EXEC_FIXTURE: &str = r#"{"version":1,"width":120,"height":100,"background":[247,248,250,255],"commands":[
{"kind":"rounded","rect":[0,0,120,100],"clip":[0,0,120,100],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[1,1,118,98],"clip":[0,0,120,100],"radius":3,"color":[255,255,255,255]},
{"kind":"effect_elevation","rect":[8,8,60,40],"clip":[0,0,120,100],"radius":4,"level":2,"color":[0,0,0,255]},
{"kind":"rounded","rect":[8,8,60,40],"clip":[0,0,120,100],"radius":4,"color":[216,220,228,255]},
{"kind":"rounded","rect":[9,9,58,38],"clip":[0,0,120,100],"radius":3,"color":[16,32,48,255]}
]}"#;

fn vulkan_skip(message: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr().lock(), "SKIP: {message}");
}

/// Vulkan executes the same ordered pass plan as Metal over committed SPIR-V.
/// Integer-only kernels make the bytes comparable: a driver arithmetic
/// difference is a real failure, never an "approximately equal" pass.
#[test]
fn vulkan_execute_matches_software_oracle() {
    if !opurom_gpu::vulkan_probe().usable() {
        vulkan_skip("no probed Vulkan compute device; execute gate is CI-verified on lavapipe");
        return;
    }
    let list = parse_display_list(EXEC_FIXTURE).expect("fixture must parse");
    let cpu = Backend::Software
        .run_ppm(&list)
        .expect("software oracle must render");
    let gpu = Backend::Vulkan.run_ppm(&list).expect("vulkan must execute");
    assert_eq!(
        gpu, cpu,
        "vulkan compute must byte-match the software oracle"
    );
    let again = Backend::Vulkan
        .run_ppm(&list)
        .expect("vulkan must execute twice");
    assert_eq!(again, gpu, "repeated vulkan renders must be byte-identical");
    println!(
        "PROOF vulkan-execute: cmp=identical bytes={} digest={}",
        cpu.len(),
        opurom_gpu::digest_hex(&cpu)
    );
}

/// The material elevation layers expand into mask and blur passes; the Vulkan
/// dispatch must produce the same bytes as the software oracle.
#[test]
fn vulkan_elevation_matches_software_oracle() {
    if !opurom_gpu::vulkan_probe().usable() {
        vulkan_skip("no probed Vulkan compute device; execute gate is CI-verified on lavapipe");
        return;
    }
    let list = parse_display_list(ELEVATION_EXEC_FIXTURE).expect("fixture must parse");
    let cpu = Backend::Software
        .run_ppm(&list)
        .expect("software oracle must render");
    let gpu = Backend::Vulkan
        .run_ppm(&list)
        .expect("vulkan must execute the elevation passes");
    assert_eq!(
        gpu, cpu,
        "vulkan elevation compute must byte-match the software oracle"
    );
    println!(
        "PROOF vulkan-elevation: cmp=identical bytes={} digest={}",
        cpu.len(),
        opurom_gpu::digest_hex(&cpu)
    );
}

/// The CLI refuses honestly without a driver and executes with one; the
/// resulting PPM must equal the software backend's bytes.
#[test]
fn raster_vulkan_cli_honest_execution() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let commands = dir.join("vulkan-execute.commands.json");
    let output = dir.join("vulkan-execute.ppm");
    std::fs::write(&commands, EXEC_FIXTURE).expect("fixture must be writable");
    let run = ProcessCommand::new(env!("CARGO_BIN_EXE_opurom-gpu"))
        .arg("raster")
        .arg("--backend")
        .arg("vulkan")
        .arg(&commands)
        .arg(&output)
        .output()
        .expect("raster must run");
    if opurom_gpu::vulkan_probe().usable() {
        assert_eq!(
            run.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let list = parse_display_list(EXEC_FIXTURE).expect("fixture must parse");
        assert_eq!(
            std::fs::read(&output).expect("vulkan PPM must exist"),
            Backend::Software
                .run_ppm(&list)
                .expect("software oracle must render"),
            "the CLI must write the oracle's bytes"
        );
    } else {
        assert_eq!(run.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&run.stderr).contains("unavailable"));
    }
}

/// A loader that answers `vkGetInstanceProcAddr` but cannot create an
/// instance must degrade honestly. This used to call the loader's entry point
/// through a `vkCreateInstance` signature and segfault before any test ran.
#[test]
fn loader_without_instance_degrades_without_crash() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let library = dir.join("libvulkan.so.1");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("fake_vulkan_loader.c");
    let compile = if cfg!(target_os = "macos") {
        ProcessCommand::new("cc")
            .args(["-dynamiclib", "-o"])
            .arg(&library)
            .arg(&source)
            .status()
    } else {
        ProcessCommand::new("cc")
            .args(["-shared", "-fPIC", "-o"])
            .arg(&library)
            .arg(&source)
            .status()
    };
    if !compile.is_ok_and(|status| status.success()) || !library.is_file() {
        vulkan_skip("no C compiler for the fake loader; honest-degradation path untested");
        return;
    }
    let out = ProcessCommand::new(env!("CARGO_BIN_EXE_opurom-gpu"))
        .arg("backends")
        .env("LD_LIBRARY_PATH", &dir)
        .env("DYLD_LIBRARY_PATH", &dir)
        .output()
        .expect("backends must run");
    assert_eq!(
        out.status.code(),
        Some(0),
        "a loader without an instance must never crash: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("vulkan=unavailable"),
        "a failed instance must be reported honestly: {stdout}"
    );
    if stdout.contains("Vulkan instance creation failed") {
        println!("PROOF fake-loader: vkCreateInstance failed honestly, no crash");
    }
}
