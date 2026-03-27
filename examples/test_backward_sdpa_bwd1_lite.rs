//! SDPA backward pass 1 MIL compile + eval test - dvf and pf only.
//!
//! Usage: ./test_backward_sdpa_bwd1_lite
//! Tests bwd_sdpa_bwd1_dvf_mil and bwd_sdpa_bwd1_pf_mil.
//! NOTE: dpf is skipped due to ANE compiler limitation - see docs/DPF_COMPILATION_ISSUE.md

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 64;
    let seq: usize = 16;
    let heads: usize = 4;
    let head_dim: usize = dim / heads;

    eprintln!(
        "Testing SDPA backward pass 1 (lite): dim={} seq={} heads={} head_dim={}",
        dim, seq, heads, head_dim
    );

    let mask_data: Vec<f32> = (0..seq * seq)
        .map(|i| if i % seq >= i / seq { 0.0 } else { -1000.0 })
        .collect();
    let wot_data: Vec<f32> = (0..dim * dim).map(|i| ((i % 100) as f32) * 0.005).collect();
    let wot = rustane::ane::WeightBlob::from_f32(&wot_data, dim, dim).unwrap();
    let mask = rustane::ane::WeightBlob::from_f32(&mask_data, seq, seq).unwrap();

    let mut all_ok = true;

    // === Test dvf (dV gradient) ===
    eprintln!("\n=== Testing dvf computation ===");
    {
        let mil = rustane::mil::bwd_sdpa_bwd1_dvf_mil(seq, dim, heads, head_dim);
        let req =
            rustane::mil::bwd_sdpa_bwd1_dvf_compile_request(seq, dim, heads, head_dim, &wot, &mask);

        let mut weight_paths: Vec<&str> = req.weights.keys().map(|k| k.as_str()).collect();
        weight_paths.sort();
        let weight_datas: Vec<&[u8]> = weight_paths
            .iter()
            .map(|path| req.weights.get(*path).unwrap().as_slice())
            .collect();
        let weight_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            req.input_sizes[0], req.output_sizes[0]
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(
            &mil,
            &weight_paths,
            &weight_datas,
            &weight_lens,
            &req.input_sizes,
            &req.output_sizes,
        );

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                let input_elements = 4 * dim * seq;
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
                } else if let Err(e) = executor.eval() {
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
                            eprintln!("dvf: all zeros (FAIL)");
                            all_ok = false;
                        } else if has_nan {
                            eprintln!("dvf: nan/inf (FAIL)");
                            all_ok = false;
                        } else {
                            eprintln!(
                                "dvf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
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
            Err(e) => {
                eprintln!("compile failed: {e}");
                all_ok = false;
            }
        }
    }

    // === Test pf (probs forward) ===
    eprintln!("\n=== Testing pf computation ===");
    {
        let mil = rustane::mil::bwd_sdpa_bwd1_pf_mil(seq, dim, heads, head_dim);
        let req = rustane::mil::bwd_sdpa_bwd1_pf_compile_request(seq, dim, heads, head_dim, &mask);

        let mut weight_paths: Vec<&str> = req.weights.keys().map(|k| k.as_str()).collect();
        weight_paths.sort();
        let weight_datas: Vec<&[u8]> = weight_paths
            .iter()
            .map(|path| req.weights.get(*path).unwrap().as_slice())
            .collect();
        let weight_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        eprintln!(
            "Input: {} bytes, Output: {} bytes",
            req.input_sizes[0], req.output_sizes[0]
        );

        let mut compiler = rustane::wrapper::ANECompiler::new();
        let exec = compiler.compile_multi(
            &mil,
            &weight_paths,
            &weight_datas,
            &weight_lens,
            &req.input_sizes,
            &req.output_sizes,
        );

        match exec {
            Ok(mut executor) => {
                eprintln!("Compile OK");

                let input_elements = 4 * dim * seq;
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
                } else if let Err(e) = executor.eval() {
                    eprintln!("eval FAILED: {e}");
                    all_ok = false;
                } else {
                    eprintln!("Eval OK");

                    let output_bytes = heads * seq * seq * 4;
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
                            eprintln!("pf: all zeros (FAIL)");
                            all_ok = false;
                        } else if has_nan {
                            eprintln!("pf: nan/inf (FAIL)");
                            all_ok = false;
                        } else {
                            eprintln!(
                                "pf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
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
            Err(e) => {
                eprintln!("compile failed: {e}");
                all_ok = false;
            }
        }
    }

    eprintln!("\n=== NOTE: dpf computation skipped ===");
    eprintln!("dpf (dP gradient) has a known ANE compiler issue.");
    eprintln!("See docs/DPF_COMPILATION_ISSUE.md for details.");

    if all_ok {
        println!("\nOK - SDPA bwd1 lite tests passed (dvf + pf)!");
    } else {
        eprintln!("\nFAILED - Some tests failed");
        std::process::exit(1);
    }
}
