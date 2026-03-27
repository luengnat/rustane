//! Debug: print the exact MIL and try compiling it.

use rustane::mil::programs::dynamic_matmul_mil;
use rustane::wrapper::ANECompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let mil = dynamic_matmul_mil(64, 64);
    println!("=== MIL ({} bytes) ===", mil.len());
    println!("{}", mil);

    let input_bytes = rustane::mil::programs::dynamic_matmul_input_bytes(64, 64);
    let output_bytes = rustane::mil::programs::dynamic_matmul_output_bytes(64, 64);
    println!(
        "\nInput bytes: {}, Output bytes: {}",
        input_bytes, output_bytes
    );

    println!("\n=== Attempting compile... ===");
    match ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes]) {
        Ok(_) => println!("COMPILE SUCCESS!"),
        Err(e) => println!("COMPILE FAILED: {}", e),
    }

    Ok(())
}
