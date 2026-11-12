use std::{env, process};

use opurom_gpu::{probe_output, self_test_output};

fn main() {
    let mut args = env::args().skip(1);
    let result = match (args.next(), args.next()) {
        (Some(command), None) if command == "probe" => probe_output(),
        (Some(command), None) if command == "self-test" => self_test_output(),
        _ => {
            eprintln!("Usage: opurom-gpu <probe|self-test>");
            process::exit(2);
        }
    };

    match result {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(1);
        }
    }
}
