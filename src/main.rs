//! CLI: `probe`, `self-test`, `backends`, `vk-record`, and the
//! `raster [--backend metal|software|vulkan] <commands.json> <out.ppm>`
//! execution command. Selecting an unavailable backend fails with a clear
//! diagnostic; the Vulkan backend is record-only and has its own `vk-record`
//! command that writes the encoded command buffer instead of pixels.

use std::{env, fs, process};

use opurom_gpu::{
    encode_stream, parse_display_list, probe_output, raster_json_ppm, self_test_output, Backend,
    BackendAvailability,
};

const USAGE: &str = "Usage: opurom-gpu <probe|self-test|backends> | \
opurom-gpu vk-record <commands.json> <out.bin> | \
opurom-gpu raster [--backend metal|software|vulkan] <commands.json> <out.ppm>";

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "probe" => {
            println!("{}", probe_output().map_err(|e| e.to_string())?);
            Ok(())
        }
        [command] if command == "self-test" => {
            println!("{}", self_test_output().map_err(|e| e.to_string())?);
            Ok(())
        }
        [command] if command == "backends" => {
            for backend in [Backend::Metal, Backend::Vulkan, Backend::Software] {
                let status = backend.available();
                let label = match status.availability {
                    BackendAvailability::Live => "live",
                    BackendAvailability::RecordOnly => "record-only",
                    BackendAvailability::Unavailable => "unavailable",
                };
                println!("{}={} ({})", backend.name(), label, status.note);
            }
            Ok(())
        }
        [mode, commands, out] if mode == "vk-record" => {
            let json = fs::read_to_string(commands)
                .map_err(|e| format!("cannot read display list {commands:?}: {e}"))?;
            let list = parse_display_list(&json).map_err(|e| e.to_string())?;
            let bytes = encode_stream(&list).map_err(|e| e.to_string())?;
            fs::write(out, bytes).map_err(|e| format!("cannot write {out:?}: {e}"))
        }
        [mode, commands, out] if mode == "raster" => raster(Backend::Metal, commands, out),
        [mode, flag, backend_name, commands, out] if mode == "raster" && flag == "--backend" => {
            let backend = Backend::parse(backend_name).ok_or_else(|| USAGE.to_string())?;
            raster(backend, commands, out)
        }
        _ => Err(USAGE.to_string()),
    }
}

fn raster(backend: Backend, commands: &str, out: &str) -> Result<(), String> {
    let status = backend.available();
    if status.availability == BackendAvailability::Unavailable {
        return Err(format!(
            "backend {} is unavailable on this host: {}",
            backend.name(),
            status.note
        ));
    }
    if backend == Backend::Vulkan {
        return Err(
            "the Vulkan backend is record-only; use `vk-record <commands.json> <out.bin>`"
                .to_owned(),
        );
    }
    let json = fs::read_to_string(commands)
        .map_err(|e| format!("cannot read display list {commands:?}: {e}"))?;
    let ppm = match backend {
        Backend::Software => {
            let list = parse_display_list(&json).map_err(|e| e.to_string())?;
            backend.run_ppm(&list).map_err(|e| e.to_string())?
        }
        _ => raster_json_ppm(&json).map_err(|e| e.to_string())?,
    };
    fs::write(out, ppm).map_err(|e| format!("cannot write {out:?}: {e}"))
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(error) if error.starts_with("Usage:") => {
            eprintln!("{error}");
            process::exit(2);
        }
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(1);
        }
    }
}
