//! Test single-output FP16 to confirm multi-output is the issue

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 128;
    let seq: usize = 16;

    // Simple MIL that just returns the input (identity) - single output
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> out = identity(x=x)[name=tensor<string, []>(\"o0\")];\n",
        dim, seq
    ));
    mil.push_str(&format!("    }} -> (out);\n}}\n"));

    eprintln!("=== GENERATED MIL ===");
    eprintln!("{}", mil);
    eprintln!("=== END MIL ===\n");

    let input_bytes = dim * seq * 2; // FP16
    let output_sizes = vec![dim * seq * 2];

    eprintln!("Input: {} bytes, Output: {:?}", input_bytes, output_sizes);

    let mut compiler = rustane::wrapper::ANECompiler::new();
    let exec = compiler.compile_single(&mil, None, &[input_bytes], &output_sizes);

    match exec {
        Ok(mut executor) => {
            eprintln!("Compile OK");

            let input: Vec<u8> = (0..dim * seq)
                .map(|i| {
                    let v: f32 = ((i % 100) as f32) * 0.01;
                    half::f16::from_f32(v).to_le_bytes()
                })
                .flatten()
                .collect();

            if let Err(e) = executor.write_input(0, &input) {
                eprintln!("write_input failed: {e}");
                std::process::exit(3);
            }
            eprintln!("Input written");

            if let Err(e) = executor.eval() {
                eprintln!("eval FAILED: {e}");
                std::process::exit(4);
            }
            eprintln!("Eval OK");

            let mut buf = vec![0u8; output_sizes[0]];
            if let Err(e) = executor.read_output(0, &mut buf) {
                eprintln!("read_output FAILED: {e}");
            } else {
                eprintln!("read_output OK: {} bytes", output_sizes[0]);
            }

            eprintln!("\nSUCCESS: Single-output FP16 test passed!");
        }
        Err(e) => {
            eprintln!("Compile FAILED: {e}");
            std::process::exit(2);
        }
    }
}
