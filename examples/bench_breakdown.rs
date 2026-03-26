//! Benchmark breakdown: measure write_input, eval, read_output separately.

use std::time::Instant;
use rustane::mil::programs::{
    dynamic_matmul_input_bytes, dynamic_matmul_mil, dynamic_matmul_output_bytes,
    pack_dynamic_matmul_input,
};
use rustane::wrapper::ANECompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let dim = 64;
    let seq = 64;
    let num_steps = 100;

    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = dynamic_matmul_input_bytes(dim, seq);
    let output_bytes = dynamic_matmul_output_bytes(dim, seq);

    let mut exec = ANECompiler::new()
        .compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes])?;

    let w_id: Vec<f32> = (0..dim * dim)
        .map(|i| if i % (dim + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let input: Vec<f32> = (0..dim * seq).map(|i| ((i % 50) as f32 - 25.0) / 100.0).collect();
    let packed = pack_dynamic_matmul_input(&input, &w_id, dim, seq);

    println!("=== Timing Breakdown (D={}, S={}, {} steps) ===", dim, seq, num_steps);
    println!("Input: {} bytes ({:.0}KB)", input_bytes, input_bytes as f64 / 1024.0);
    println!("Output: {} bytes ({:.0}KB)", output_bytes, output_bytes as f64 / 1024.0);

    let mut write_us = 0u128;
    let mut eval_us = 0u128;
    let mut read_us = 0u128;

    for _ in 0..num_steps {
        let t = Instant::now();
        exec.write_input(0, &packed)?;
        write_us += t.elapsed().as_micros();

        let t = Instant::now();
        exec.eval()?;
        eval_us += t.elapsed().as_micros();

        let t = Instant::now();
        let _raw = exec.read_output_vec(0)?;
        read_us += t.elapsed().as_micros();
    }

    let total_us = write_us + eval_us + read_us;
    println!("\nwrite_input: {}μs total, {:.0}μs/step ({:.1}%)",
        write_us, write_us as f64 / num_steps as f64, write_us as f64 / total_us as f64 * 100.0);
    println!("eval:        {}μs total, {:.0}μs/step ({:.1}%)",
        eval_us, eval_us as f64 / num_steps as f64, eval_us as f64 / total_us as f64 * 100.0);
    println!("read_output: {}μs total, {:.0}μs/step ({:.1}%)",
        read_us, read_us as f64 / num_steps as f64, read_us as f64 / total_us as f64 * 100.0);
    println!("total:       {}μs total, {:.0}μs/step",
        total_us, total_us as f64 / num_steps as f64);
    println!("throughput:   {:.0} steps/sec",
        num_steps as f64 / (total_us as f64 / 1_000_000.0));

    // Also measure pack time
    let pack_start = Instant::now();
    for _ in 0..num_steps {
        let _ = pack_dynamic_matmul_input(&input, &w_id, dim, seq);
    }
    let pack_us = pack_start.elapsed().as_micros();
    println!("\npack_input:  {}μs total, {:.0}μs/step",
        pack_us, pack_us as f64 / num_steps as f64);

    Ok(())
}
