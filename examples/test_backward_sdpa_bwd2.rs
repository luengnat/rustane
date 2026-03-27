//! SDPA backward pass 2 MIL compile + eval test - single output versions.
//!
//! Usage: ./test_backward_sdpa_bwd2
//! Tests bwd_sdpa_bwd2_dqf_mil, bwd_sdpa_bwd2_dkf_mil compilation and evaluation on ANE.

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 64;
    let seq: usize = 16;
    let heads: usize = 4;
    let head_dim: usize = dim / heads; // 16

    eprintln!(
        "Testing SDPA backward pass 2: dim={} seq={} heads={} head_dim={}",
        dim, seq, heads, head_dim
    );

    let mut all_ok = true;

    // === Test dqf (dQ gradient) ===
    eprintln!("\n=== Testing dqf computation ===");
    {
        let mil = rustane::mil::bwd_sdpa_bwd2_dqf_mil(seq, dim, heads, head_dim);
        eprintln!("=== DQF MIL (first 50 lines) ===");
        for (i, line) in mil.lines().take(50).enumerate() {
            eprintln!("{}: {}", i + 1, line);
        }
        eprintln!("...");
        eprintln!("=== END MIL ===\n");

        let req = rustane::mil::bwd_sdpa_bwd2_dqf_compile_request(seq, dim, heads, head_dim);

        let input_bytes = req.input_sizes[0];
        let output_bytes = req.output_sizes[0];

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            input_bytes, output_bytes
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(&mil, &[], &[], &[], &req.input_sizes, &req.output_sizes);

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                let score_ch = heads * seq;
                let in_ch = 2 * score_ch + 2 * dim;
                let input_elements = in_ch * seq;
                let input: Vec<u8> = (0..input_elements)
                    .map(|i| {
                        let v: f32 = ((i % 100) as f32) * 0.01;
                        v.to_le_bytes()
                    })
                    .flatten()
                    .collect();

                if let Err(e) = executor.write_input(0, &input) {
                    eprintln!("write_input: {e}");
                    all_ok = false;
                } else {
                    eprintln!("Input written");

                    if let Err(e) = executor.eval() {
                        eprintln!("eval FAILED: {e}");
                        all_ok = false;
                    } else {
                        eprintln!("Eval OK");

                        let output_bytes = dim * seq * 4;
                        let mut buf = vec![0u8; output_bytes];
                        if let Err(e) = executor.read_output(0, &mut buf) {
                            eprintln!("read_output FAILED: {e}");
                            all_ok = false;
                        } else {
                            eprintln!("read_output OK: {} bytes", output_bytes);
                            let f32_vals: Vec<f32> = buf
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            let all_zero = f32_vals.iter().all(|&x| x == 0.0);
                            let has_nan = f32_vals.iter().any(|x| x.is_nan() || x.is_infinite());
                            if all_zero {
                                eprintln!("dqf: all zeros (FAIL)");
                                all_ok = false;
                            } else if has_nan {
                                eprintln!("dqf: nan/inf (FAIL)");
                                all_ok = false;
                            } else {
                                eprintln!(
                                    "dqf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
                                    f32_vals[0],
                                    f32_vals[1],
                                    f32_vals[2],
                                    f32_vals[3],
                                    f32_vals.len()
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("compile failed: {e}");
                all_ok = false;
            }
        }
    }

    // === Test dkf (dK gradient) ===
    eprintln!("\n=== Testing dkf computation ===");
    {
        let mil = rustane::mil::bwd_sdpa_bwd2_dkf_mil(seq, dim, heads, head_dim);
        let req = rustane::mil::bwd_sdpa_bwd2_dkf_compile_request(seq, dim, heads, head_dim);

        let input_bytes = req.input_sizes[0];
        let output_bytes = req.output_sizes[0];

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            input_bytes, output_bytes
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(&mil, &[], &[], &[], &req.input_sizes, &req.output_sizes);

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                let score_ch = heads * seq;
                let in_ch = 2 * score_ch + 2 * dim;
                let input_elements = in_ch * seq;
                let input: Vec<u8> = (0..input_elements)
                    .map(|i| {
                        let v: f32 = ((i % 100) as f32) * 0.01;
                        v.to_le_bytes()
                    })
                    .flatten()
                    .collect();

                if let Err(e) = executor.write_input(0, &input) {
                    eprintln!("write_input: {e}");
                    all_ok = false;
                } else {
                    eprintln!("Input written");

                    if let Err(e) = executor.eval() {
                        eprintln!("eval FAILED: {e}");
                        all_ok = false;
                    } else {
                        eprintln!("Eval OK");

                        let output_bytes = dim * seq * 4;
                        let mut buf = vec![0u8; output_bytes];
                        if let Err(e) = executor.read_output(0, &mut buf) {
                            eprintln!("read_output FAILED: {e}");
                            all_ok = false;
                        } else {
                            eprintln!("read_output OK: {} bytes", output_bytes);
                            let f32_vals: Vec<f32> = buf
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            let all_zero = f32_vals.iter().all(|&x| x == 0.0);
                            let has_nan = f32_vals.iter().any(|x| x.is_nan() || x.is_infinite());
                            if all_zero {
                                eprintln!("dkf: all zeros (FAIL)");
                                all_ok = false;
                            } else if has_nan {
                                eprintln!("dkf: nan/inf (FAIL)");
                                all_ok = false;
                            } else {
                                eprintln!(
                                    "dkf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
                                    f32_vals[0],
                                    f32_vals[1],
                                    f32_vals[2],
                                    f32_vals[3],
                                    f32_vals.len()
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("compile failed: {e}");
                all_ok = false;
            }
        }
    }

    if all_ok {
        println!("\nOK - All SDPA backward pass 2 tests passed!");
    } else {
        eprintln!("\nFAILED - Some tests failed");
        std::process::exit(1);
    }
}
