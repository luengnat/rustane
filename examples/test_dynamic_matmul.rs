//! Test dynamic matmul: weights in input, no recompile needed.
//!
//! This is the key performance optimization: instead of baking weights into
//! the ANE model (requiring recompilation to update), we pack weights into
//! the input tensor alongside activations. Weights change every step by
//! just changing the input IOSurface content — zero recompilation.

use half::f16;
use rustane::mil::programs::{
    dynamic_matmul_input_bytes, dynamic_matmul_mil, dynamic_matmul_output_bytes,
    pack_dynamic_matmul_input,
};
use rustane::wrapper::ANECompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let dim = 64;
    let seq = 64; // ANE may require seq >= 32
                  // Actually try small first to debug
    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = dynamic_matmul_input_bytes(dim, seq);
    let output_bytes = dynamic_matmul_output_bytes(dim, seq);

    println!("=== Dynamic Matmul Test ===");
    println!("Dim={}, Seq={}", dim, seq);
    println!(
        "Input bytes: {} ({:.1}KB)",
        input_bytes,
        input_bytes as f64 / 1024.0
    );
    println!(
        "Output bytes: {} ({:.1}KB)",
        output_bytes,
        output_bytes as f64 / 1024.0
    );

    // Compile with no weights (empty dict)
    let mut exec =
        ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes])?;

    // Create identity weight matrix
    let w_identity: Vec<f32> = (0..dim * dim)
        .map(|i| if i % (dim + 1) == 0 { 1.0 } else { 0.0 })
        .collect();

    // Create 2x identity weight matrix
    let w_2x: Vec<f32> = (0..dim * dim)
        .map(|i| if i % (dim + 1) == 0 { 2.0 } else { 0.0 })
        .collect();

    // Test 1: Identity weights
    println!("\n--- Test 1: Identity weights ---");
    let input: Vec<f32> = (0..dim * seq)
        .map(|i| ((i % 50) as f32 - 25.0) / 100.0)
        .collect();
    let packed = pack_dynamic_matmul_input(&input, &w_identity, dim, seq);
    exec.write_input(0, &packed)?;
    exec.eval()?;

    let raw_out = exec.read_output_vec(0)?;
    let out: Vec<f32> = raw_out[..dim * seq * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let max_err: f32 = input
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max error vs identity: {:.6}", max_err);
    println!(
        "Input[0..3]:  [{:.4}, {:.4}, {:.4}]",
        input[0], input[1], input[2]
    );
    println!(
        "Output[0..3]: [{:.4}, {:.4}, {:.4}]",
        out[0], out[1], out[2]
    );
    println!(
        "{}",
        if max_err < 0.1 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Test 2: 2x weights (NO recompile!)
    println!("\n--- Test 2: 2x weights (no recompile) ---");
    let packed2 = pack_dynamic_matmul_input(&input, &w_2x, dim, seq);
    exec.write_input(0, &packed2)?;
    exec.eval()?;

    let raw_out2 = exec.read_output_vec(0)?;
    let out2: Vec<f32> = raw_out2[..dim * seq * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let max_err2: f32 = input
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a * 2.0 - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max error vs 2x: {:.6}", max_err2);
    println!(
        "Input[0..3]:  [{:.4}, {:.4}, {:.4}]",
        input[0], input[1], input[2]
    );
    println!(
        "Output[0..3]: [{:.4}, {:.4}, {:.4}]",
        out2[0], out2[1], out2[2]
    );
    println!(
        "{}",
        if max_err2 < 0.1 {
            "✅ PASS (weights changed without recompile!)"
        } else {
            "❌ FAIL"
        }
    );

    // Test 3: Speed benchmark — many evals with weight changes
    println!("\n--- Test 3: Speed benchmark (100 steps, no recompile) ---");
    use std::time::Instant;
    let start = Instant::now();
    for step in 0..100 {
        let scale = 1.0 + (step as f32) * 0.01;
        let w_scaled: Vec<f32> = (0..dim * dim)
            .map(|i| if i % (dim + 1) == 0 { scale } else { 0.0 })
            .collect();
        let packed = pack_dynamic_matmul_input(&input, &w_scaled, dim, seq);
        exec.write_input(0, &packed)?;
        exec.eval()?;
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / 100;
    println!("100 steps in {:?}", elapsed);
    println!(
        "Average step: {}μs ({:.1}ms)",
        avg_us,
        avg_us as f64 / 1000.0
    );
    println!("Throughput: {:.0} steps/sec", 100.0 / elapsed.as_secs_f64());

    // Compare: CPU matmul time
    let cpu_start = Instant::now();
    for step in 0..100 {
        let scale = 1.0 + (step as f32) * 0.01;
        let w_scaled: Vec<f32> = (0..dim * dim)
            .map(|i| if i % (dim + 1) == 0 { scale } else { 0.0 })
            .collect();
        let _ = cpu_matmul(&input, &w_scaled, dim, dim, seq);
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_avg_us = cpu_elapsed.as_micros() / 100;
    println!("\nCPU matmul: 100 steps in {:?}", cpu_elapsed);
    println!(
        "CPU average step: {}μs ({:.1}ms)",
        cpu_avg_us,
        cpu_avg_us as f64 / 1000.0
    );
    println!(
        "Speedup: {:.1}x",
        cpu_elapsed.as_secs_f64() / elapsed.as_secs_f64()
    );

    println!("\nOK dynamic_matmul");
    Ok(())
}

fn cpu_matmul(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * n + l] * w[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}
