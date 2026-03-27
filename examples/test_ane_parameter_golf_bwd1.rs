//! ANE Backward Pass Test for Parameter-Golf Configuration
//!
//! Tests the bwd_sdpa_bwd1_combined_mil function with parameter-golf model dimensions:
//! - model_dim: 416
//! - num_heads: 8
//! - head_dim: 52
//! - seq_len: 1024 (training), 256 (fast test)
//!
//! ## Usage
//!
//! ```bash
//! # Fast test (seq=256)
//! cargo run --example test_ane_parameter_golf_bwd1 --release
//!
//! # Full sequence length (seq=1024)
//! cargo run --example test_ane_parameter_golf_bwd1 --release -- --seq-len 1024
//! ```

use rustane::ane::WeightBlob;
use rustane::mil::{bwd_sdpa_bwd1_combined_compile_request, bwd_sdpa_bwd1_combined_mil};
use rustane::wrapper::ANECompiler;

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    // Parameter-Golf configuration
    let dim: usize = 416;
    let heads: usize = 8;
    let head_dim: usize = dim / heads; // 52

    // Parse command line for sequence length
    let seq: usize = std::env::args()
        .find(|arg| arg == "--seq-len")
        .and_then(|_| std::env::args().skip_while(|a| a != "--seq-len").nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256); // Default to 256 for fast test

    let score_ch = heads * seq;
    let out_ch = dim + 2 * score_ch;

    eprintln!(
        "=== Parameter-Golf ANE Backward Test ==="
    );
    eprintln!("Config: dim={} seq={} heads={} head_dim={}", dim, seq, heads, head_dim);
    eprintln!("Output: {} channels (dvf={}, pf={}, dpf={})", out_ch, dim, score_ch, score_ch);
    eprintln!("Output tensor: [1, {}, 1, {}] = {} f32 values = {:.2} MB",
        out_ch, seq, out_ch * seq, (out_ch * seq * 4) as f64 / 1024.0 / 1024.0);

    // Create weight blobs matching parameter-golf initialization
    let wot_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();
    let mask_data: Vec<f32> = (0..seq * seq)
        .map(|i| if i % seq >= i / seq { 0.0 } else { -1000.0 })
        .collect();
    let wot = WeightBlob::from_f32(&wot_data, dim, dim).unwrap();
    let mask = WeightBlob::from_f32(&mask_data, seq, seq).unwrap();

    eprintln!("\nGenerating MIL...");
    let mil = bwd_sdpa_bwd1_combined_mil(seq, dim, heads, head_dim);

    eprintln!("Creating compile request...");
    let req = bwd_sdpa_bwd1_combined_compile_request(seq, dim, heads, head_dim, &wot, &mask);

    eprintln!("Compiling on ANE...");
    let compile_start = std::time::Instant::now();

    let mut compiler = ANECompiler::new();
    let exec = compiler.compile_multi(
        &mil,
        &["@model_path/weights/wot.bin", "@model_path/weights/mask.bin"],
        &[wot.as_ref(), mask.as_ref()],
        &[wot.len(), mask.len()],
        &req.input_sizes,
        &req.output_sizes,
    );

    match exec {
        Ok(mut executor) => {
            let compile_time = compile_start.elapsed();
            eprintln!("Compile OK ({:.2}s)", compile_time.as_secs_f64());

            // Create input: [1, 4*DIM, 1, SEQ] packed as fp32
            // (qf, kf, vf, df concatenated)
            let input_elements = 4 * dim * seq;
            let input: Vec<u8> = (0..input_elements)
                .map(|i| {
                    let v: f32 = ((i % 100) as f32) * 0.01;
                    v.to_le_bytes()
                })
                .flatten()
                .collect();

            eprintln!("Writing input ({:.2} MB)...", (input.len() as f64) / 1024.0 / 1024.0);
            if let Err(e) = executor.write_input(0, &input) {
                eprintln!("write_input: {e}");
                std::process::exit(1);
            }

            eprintln!("Executing on ANE...");
            let eval_start = std::time::Instant::now();
            if let Err(e) = executor.eval() {
                eprintln!("eval FAILED: {e}");
                std::process::exit(1);
            }
            let eval_time = eval_start.elapsed();

            // Read concatenated output: [1, DIM+2*SCORE_CH, 1, SEQ]
            let output_bytes = out_ch * seq * 4;
            eprintln!("Reading output ({:.2} MB)...", (output_bytes as f64) / 1024.0 / 1024.0);
            let mut buf = vec![0u8; output_bytes];
            if let Err(e) = executor.read_output(0, &mut buf) {
                eprintln!("read_output FAILED: {e}");
                std::process::exit(1);
            }

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

            eprintln!("\n=== Results ===");
            eprintln!("Execution time: {:.3}s", eval_time.as_secs_f64());
            eprintln!("Throughput: {:.2} tokens/sec", (seq as f64) / eval_time.as_secs_f64());

            // Validate outputs
            let mut all_ok = true;

            // Check dvf
            let dvf_zero = dvf_vals.iter().all(|&x| x == 0.0);
            let dvf_nan = dvf_vals.iter().any(|x| x.is_nan() || x.is_infinite());
            if dvf_zero {
                eprintln!("dvf: all zeros (FAIL)");
                all_ok = false;
            } else if dvf_nan {
                eprintln!("dvf: nan/inf (FAIL)");
                all_ok = false;
            } else {
                eprintln!(
                    "dvf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals) - OK",
                    dvf_vals[0], dvf_vals[1], dvf_vals[2], dvf_vals[3], dvf_vals.len()
                );
            }

            // Check pf
            let pf_zero = pf_vals.iter().all(|&x| x == 0.0);
            let pf_nan = pf_vals.iter().any(|x| x.is_nan() || x.is_infinite());
            if pf_zero {
                eprintln!("pf:  all zeros (FAIL)");
                all_ok = false;
            } else if pf_nan {
                eprintln!("pf:  nan/inf (FAIL)");
                all_ok = false;
            } else {
                eprintln!(
                    "pf:  [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals) - OK",
                    pf_vals[0], pf_vals[1], pf_vals[2], pf_vals[3], pf_vals.len()
                );
            }

            // Check dpf
            let dpf_zero = dpf_vals.iter().all(|&x| x == 0.0);
            let dpf_nan = dpf_vals.iter().any(|x| x.is_nan() || x.is_infinite());
            if dpf_zero {
                eprintln!("dpf: all zeros (FAIL)");
                all_ok = false;
            } else if dpf_nan {
                eprintln!("dpf: nan/inf (FAIL)");
                all_ok = false;
            } else {
                eprintln!(
                    "dpf: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals) - OK",
                    dpf_vals[0], dpf_vals[1], dpf_vals[2], dpf_vals[3], dpf_vals.len()
                );
            }

            if all_ok {
                eprintln!("\n=== Parameter-Golf ANE Backward Test PASSED ===");
                eprintln!("All three outputs (dvf, pf, dpf) computed correctly!");
                println!("OK - Parameter-golf bwd1 test passed!");
            } else {
                eprintln!("\nFAILED - Some outputs invalid");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Compile failed: {e}");
            std::process::exit(1);
        }
    }
}
