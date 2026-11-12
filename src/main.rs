//! CLI: `probe`, `self-test`, and the R6 `raster <commands.json> <out.ppm>`
//! display-list execution command.

use std::{env, fs, process};

use opurom_gpu::{probe_output, raster_json_ppm, self_test_output};

const USAGE: &str =
    "Usage: opurom-gpu <probe|self-test> | opurom-gpu raster <commands.json> <out.ppm>";

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
        [mode, commands, out_ppm] if mode == "raster" => {
            let json = fs::read_to_string(commands)
                .map_err(|e| format!("cannot read display list {commands:?}: {e}"))?;
            let ppm = raster_json_ppm(&json).map_err(|e| e.to_string())?;
            fs::write(out_ppm, ppm).map_err(|e| format!("cannot write {out_ppm:?}: {e}"))
        }
        _ => Err(USAGE.to_string()),
    }
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
