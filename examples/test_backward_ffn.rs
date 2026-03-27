//! FFN backward MIL compile + eval test - single output versions.
//!
//! Usage: ./test_backward_ffn
//! Tests bwd_ffn_dh1_mil, bwd_ffn_dh3_mil, bwd_ffn_dx_mil compilation and evaluation on ANE.

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 64;
    let hidden: usize = 128;
    let seq: usize = 16;

    // Create weight blobs: small random values to avoid fp16 overflow
    // Weight shapes match conv expected format: [out_channels, in_channels, 1, 1]
    let w1t_data: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();
    let w2t_data: Vec<f32> = (0..hidden * dim)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();
    let w3t_data: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();

    let w1t = rustane::ane::WeightBlob::from_f32(&w1t_data, dim, hidden).unwrap();
    let w2t = rustane::ane::WeightBlob::from_f32(&w2t_data, hidden, dim).unwrap();
    let w3t = rustane::ane::WeightBlob::from_f32(&w3t_data, dim, hidden).unwrap();

    eprintln!(
        "Weight blobs created: W1t={} bytes, W2t={} bytes, W3t={} bytes",
        w1t.len(),
        w2t.len(),
        w3t.len()
    );

    let mut all_ok = true;

    // === Test dh1 ===
    eprintln!("\n=== Testing dh1 computation ===");
    {
        let in_ch = dim + 2 * hidden;
        let mil = rustane::mil::bwd_ffn_dh1_mil(seq, dim, hidden);
        eprintln!("=== DH1 MIL ===");
        eprintln!("{}", mil);
        eprintln!("=== END MIL ===\n");
        let input_bytes = in_ch * seq * 4; // FP32
        let output_bytes = hidden * seq * 4; // FP32

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            input_bytes, output_bytes
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(
            &mil,
            &["@model_path/weights/w2t.bin"],
            &[w2t.as_bytes()],
            &[w2t.len()],
            &[input_bytes],
            &[output_bytes],
        );

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                // Create input: pack [dffn, h1, h3] as fp32
                let input: Vec<u8> = (0..in_ch * seq)
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

                        let mut buf = vec![0u8; output_bytes];
                        if let Err(e) = executor.read_output(0, &mut buf) {
                            eprintln!("read_output FAILED: {e}");
                            all_ok = false;
                        } else {
                            eprintln!("read_output OK: {} bytes", output_bytes);
                            // Check for valid output (FP32)
                            let f32_vals: Vec<f32> = buf
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            let all_zero = f32_vals.iter().all(|&x| x == 0.0);
                            let has_nan = f32_vals.iter().any(|x| x.is_nan() || x.is_infinite());
                            if all_zero {
                                eprintln!("dh1: all zeros (FAIL)");
                                all_ok = false;
                            } else if has_nan {
                                eprintln!("dh1: nan/inf (FAIL)");
                                all_ok = false;
                            } else {
                                eprintln!(
                                    "dh1: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
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

    // === Test dh3 ===
    eprintln!("\n=== Testing dh3 computation ===");
    {
        let in_ch = dim + 2 * hidden;
        let mil = rustane::mil::bwd_ffn_dh3_mil(seq, dim, hidden);
        let input_bytes = in_ch * seq * 4; // FP32
        let output_bytes = hidden * seq * 4; // FP32

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            input_bytes, output_bytes
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(
            &mil,
            &["@model_path/weights/w2t.bin"],
            &[w2t.as_bytes()],
            &[w2t.len()],
            &[input_bytes],
            &[output_bytes],
        );

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                // Create input FP32
                let input: Vec<u8> = (0..in_ch * seq)
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
                                eprintln!("dh3: all zeros (FAIL)");
                                all_ok = false;
                            } else if has_nan {
                                eprintln!("dh3: nan/inf (FAIL)");
                                all_ok = false;
                            } else {
                                eprintln!(
                                    "dh3: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
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

    // === Test dx ===
    eprintln!("\n=== Testing dx computation ===");
    {
        let in_ch = 2 * hidden;
        let mil = rustane::mil::bwd_ffn_dx_mil(seq, dim, hidden);
        let input_bytes = in_ch * seq * 4; // FP32
        let output_bytes = dim * seq * 4; // FP32

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            input_bytes, output_bytes
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(
            &mil,
            &["@model_path/weights/w1t.bin", "@model_path/weights/w3t.bin"],
            &[w1t.as_bytes(), w3t.as_bytes()],
            &[w1t.len(), w3t.len()],
            &[input_bytes],
            &[output_bytes],
        );

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                // Input is packed [dh1, dh3] FP32
                let input: Vec<u8> = (0..in_ch * seq)
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
                                eprintln!("dx: all zeros (FAIL)");
                                all_ok = false;
                            } else if has_nan {
                                eprintln!("dx: nan/inf (FAIL)");
                                all_ok = false;
                            } else {
                                eprintln!(
                                    "dx: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
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
        println!("\nOK - All single-output FFN backward tests passed!");
    } else {
        eprintln!("\nFAILED - Some tests failed");
        std::process::exit(1);
    }
}
