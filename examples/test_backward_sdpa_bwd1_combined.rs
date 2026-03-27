//! SDPA backward pass 1 - combined output test (dvf + pf + dpf concatenated)
//!
//! Usage: ./test_backward_sdpa_bwd1_combined
//! Tests the combined bwd_sdpa_bwd1_combined_mil which returns all three outputs
//! concatenated together, allowing dpf to work around the ANE compiler limitation.

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 64;
    let seq: usize = 16;
    let heads: usize = 4;
    let head_dim: usize = dim / heads;
    let score_ch = heads * seq;
    let out_ch = dim + 2 * score_ch;

    eprintln!(
        "Testing SDPA backward pass 1 (combined): dim={} seq={} heads={} head_dim={}",
        dim, seq, heads, head_dim
    );
    eprintln!(
        "Output: {} channels (dvf={}, pf={}, dpf={})",
        out_ch, dim, score_ch, score_ch
    );

    // Create weight blobs
    let wot_data: Vec<f32> = (0..dim * dim).map(|i| ((i % 100) as f32) * 0.005).collect();
    let mask_data: Vec<f32> = (0..seq * seq)
        .map(|i| if i % seq >= i / seq { 0.0 } else { -1000.0 })
        .collect();
    let wot = rustane::ane::WeightBlob::from_f32(&wot_data, dim, dim).unwrap();
    let mask = rustane::ane::WeightBlob::from_f32(&mask_data, seq, seq).unwrap();

    let mil = rustane::mil::bwd_sdpa_bwd1_combined_mil(seq, dim, heads, head_dim);

    eprintln!("\n=== Combined MIL (first 60 lines) ===");
    for (i, line) in mil.lines().take(60).enumerate() {
        eprintln!("{}: {}", i + 1, line);
    }
    eprintln!("...");

    let req = rustane::mil::bwd_sdpa_bwd1_combined_compile_request(
        seq, dim, heads, head_dim, &wot, &mask,
    );

    let mut compiler = rustane::wrapper::ANECompiler::new();
    let exec = compiler.compile_multi(
        &mil,
        &[
            "@model_path/weights/wot.bin",
            "@model_path/weights/mask.bin",
        ],
        &[wot.as_ref(), mask.as_ref()],
        &[wot.len(), mask.len()],
        &req.input_sizes,
        &req.output_sizes,
    );

    match exec {
        Ok(mut executor) => {
            eprintln!("\nCompile OK");

            // Create input: [1, 4*DIM, 1, SEQ] packed as fp32
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
                std::process::exit(1);
            }
            eprintln!("Input written");

            if let Err(e) = executor.eval() {
                eprintln!("eval FAILED: {e}");
                std::process::exit(1);
            }
            eprintln!("Eval OK");

            // Read concatenated output: [1, DIM+2*SCORE_CH, 1, SEQ]
            let output_bytes = out_ch * seq * 4;
            let mut buf = vec![0u8; output_bytes];
            if let Err(e) = executor.read_output(0, &mut buf) {
                eprintln!("read_output FAILED: {e}");
                std::process::exit(1);
            }
            eprintln!("read_output OK: {} bytes", output_bytes);

            let f32_vals: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Split output into dvf, pf, dpf
            let dvf_end = dim * seq;
            let pf_end = dvf_end + score_ch * seq;

            let dvf_vals = &f32_vals[0..dvf_end];
            let pf_vals = &f32_vals[dvf_end..pf_end];
            let dpf_vals = &f32_vals[pf_end..];

            let all_ok = true;

            // Check dvf
            let dvf_zero = dvf_vals.iter().all(|&x| x == 0.0);
            let dvf_nan = dvf_vals.iter().any(|x| x.is_nan() || x.is_infinite());
            if dvf_zero {
                eprintln!("dvf: all zeros (FAIL)");
            } else if dvf_nan {
                eprintln!("dvf: nan/inf (FAIL)");
            } else {
                eprintln!(
                    "dvf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals) - OK",
                    dvf_vals[0],
                    dvf_vals[1],
                    dvf_vals[2],
                    dvf_vals[3],
                    dvf_vals.len()
                );
            }

            // Check pf
            let pf_zero = pf_vals.iter().all(|&x| x == 0.0);
            let pf_nan = pf_vals.iter().any(|x| x.is_nan() || x.is_infinite());
            if pf_zero {
                eprintln!("pf: all zeros (FAIL)");
            } else if pf_nan {
                eprintln!("pf: nan/inf (FAIL)");
            } else {
                eprintln!(
                    "pf:  [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals) - OK",
                    pf_vals[0],
                    pf_vals[1],
                    pf_vals[2],
                    pf_vals[3],
                    pf_vals.len()
                );
            }

            // Check dpf
            let dpf_zero = dpf_vals.iter().all(|&x| x == 0.0);
            let dpf_nan = dpf_vals.iter().any(|x| x.is_nan() || x.is_infinite());
            if dpf_zero {
                eprintln!("dpf: all zeros (FAIL)");
            } else if dpf_nan {
                eprintln!("dpf: nan/inf (FAIL)");
            } else {
                eprintln!(
                    "dpf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals) - OK",
                    dpf_vals[0],
                    dpf_vals[1],
                    dpf_vals[2],
                    dpf_vals[3],
                    dpf_vals.len()
                );
            }

            if all_ok && !dvf_zero && !dvf_nan && !pf_zero && !pf_nan && !dpf_zero && !dpf_nan {
                println!("\nOK - Combined bwd1 test passed (dvf + pf + dpf)!");
            } else {
                eprintln!("\nFAILED - Some outputs invalid");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    }
}
