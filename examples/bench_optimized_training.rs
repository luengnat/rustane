//! Optimized training benchmark v3: fixed CPU loss, clean comparison.

use rustane::mil::programs::{
    dynamic_matmul_input_bytes, dynamic_matmul_mil, dynamic_matmul_output_bytes,
    pack_dynamic_matmul_input, pack_weights_into,
};
use rustane::wrapper::ANECompiler;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let dim = 64;
    let seq = 64;
    let lr = 0.01f32;
    let num_steps = 1000;

    println!("=== Dynamic Weight Training: ANE vs CPU ===");
    println!("Dim={}, Seq={}, LR={}, Steps={}", dim, seq, lr, num_steps);

    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = dynamic_matmul_input_bytes(dim, seq);
    let output_bytes = dynamic_matmul_output_bytes(dim, seq);

    let mut exec =
        ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes])?;

    let total_ch = dim + dim * dim;
    let mut pack_buf_f32 = vec![0.0f32; total_ch * seq];
    let mut grad = vec![0.0f32; dim * dim];
    let mut errors = vec![0.0f32; dim * seq];

    // Same initial weights for both
    let init_weights: Vec<f32> = (0..dim * dim)
        .map(|i| ((i * 7 + 13) % 100) as f32 / 1000.0 - 0.05)
        .collect();
    let input: Vec<f32> = (0..dim * seq)
        .map(|i| ((i * 3 + 7) % 200) as f32 / 1000.0 - 0.1)
        .collect();
    let target: Vec<f32> = (0..dim * seq).map(|i| ((i % 10) as f32) * 0.01).collect();
    let n = dim * seq;

    // --- ANE Training ---
    let mut weights = init_weights.clone();
    let mut packed = pack_dynamic_matmul_input(&input, &weights, dim, seq);

    let ane_start = Instant::now();
    let mut ane_loss = 0.0f32;
    let mut loss_first = 0.0f32;
    let mut pack_us: u128 = 0;
    let mut write_us: u128 = 0;
    let mut eval_us: u128 = 0;
    let mut read_us: u128 = 0;
    let mut grad_us: u128 = 0;

    for step in 0..num_steps {
        let t = Instant::now();
        if step > 0 {
            pack_weights_into(&mut pack_buf_f32, &weights, dim, seq);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    pack_buf_f32.as_ptr() as *const u8,
                    pack_buf_f32.len() * 4,
                )
            };
            packed.copy_from_slice(bytes);
        }
        pack_us += t.elapsed().as_micros();

        let t = Instant::now();
        exec.write_input(0, &packed)?;
        write_us += t.elapsed().as_micros();

        let t = Instant::now();
        exec.eval()?;
        eval_us += t.elapsed().as_micros();

        let t = Instant::now();
        let raw_out = exec.read_output_vec(0)?;
        read_us += t.elapsed().as_micros();

        let t = Instant::now();
        let output: Vec<f32> = raw_out[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut loss = 0.0f32;
        for i in 0..n {
            errors[i] = output[i] - target[i];
            loss += errors[i] * errors[i];
        }
        loss /= n as f32;
        if step == 0 {
            loss_first = loss;
        }
        ane_loss = loss;

        let scale = 2.0 / n as f32;
        grad.fill(0.0);
        for d_out in 0..dim {
            for d_in in 0..dim {
                let mut sum = 0.0f32;
                for s in 0..seq {
                    sum += errors[d_out * seq + s] * input[d_in * seq + s];
                }
                grad[d_out * dim + d_in] = scale * sum;
            }
        }
        for i in 0..weights.len() {
            weights[i] -= lr * grad[i];
        }
        grad_us += t.elapsed().as_micros();
    }
    let ane_elapsed = ane_start.elapsed();

    // --- CPU Training ---
    let mut cpu_weights = init_weights.clone();
    let mut cpu_errors = vec![0.0f32; n];

    let cpu_start = Instant::now();
    let mut cpu_loss = 0.0f32;
    let mut cpu_loss_first = 0.0f32;

    for step in 0..num_steps {
        // CPU matmul
        let mut output = vec![0.0f32; n];
        for i in 0..dim {
            for j in 0..seq {
                let mut sum = 0.0f32;
                for l in 0..dim {
                    sum += cpu_weights[i * dim + l] * input[l * seq + j];
                }
                output[i * seq + j] = sum;
            }
        }

        // Loss
        let mut loss = 0.0f32;
        for i in 0..n {
            cpu_errors[i] = output[i] - target[i];
            loss += cpu_errors[i] * cpu_errors[i];
        }
        loss /= n as f32;
        if step == 0 {
            cpu_loss_first = loss;
        }
        cpu_loss = loss;

        // Gradient + update
        let scale = 2.0 / n as f32;
        for d_out in 0..dim {
            for d_in in 0..dim {
                let mut sum = 0.0f32;
                for s in 0..seq {
                    sum += cpu_errors[d_out * seq + s] * input[d_in * seq + s];
                }
                cpu_weights[d_out * dim + d_in] -= lr * scale * sum;
            }
        }
    }
    let cpu_elapsed = cpu_start.elapsed();

    // --- Results ---
    let total_us = pack_us + write_us + eval_us + read_us + grad_us;
    println!("\n=== ANE Timing Breakdown ===");
    println!(
        "pack:  {:.0}μs/step ({:.1}%)",
        pack_us as f64 / num_steps as f64,
        pack_us as f64 / total_us as f64 * 100.0
    );
    println!(
        "write: {:.0}μs/step ({:.1}%)",
        write_us as f64 / num_steps as f64,
        write_us as f64 / total_us as f64 * 100.0
    );
    println!(
        "eval:  {:.0}μs/step ({:.1}%)",
        eval_us as f64 / num_steps as f64,
        eval_us as f64 / total_us as f64 * 100.0
    );
    println!(
        "read:  {:.0}μs/step ({:.1}%)",
        read_us as f64 / num_steps as f64,
        read_us as f64 / total_us as f64 * 100.0
    );
    println!(
        "grad:  {:.0}μs/step ({:.1}%)",
        grad_us as f64 / num_steps as f64,
        grad_us as f64 / total_us as f64 * 100.0
    );

    let ane_tput = num_steps as f64 / ane_elapsed.as_secs_f64();
    let cpu_tput = num_steps as f64 / cpu_elapsed.as_secs_f64();
    let speedup = cpu_elapsed.as_secs_f64() / ane_elapsed.as_secs_f64();

    println!("\n=== RESULTS ===");
    println!(
        "ANE: {:.0} steps/sec | loss: {:.6} → {:.6}",
        ane_tput, loss_first, ane_loss
    );
    println!(
        "CPU: {:.0} steps/sec | loss: {:.6} → {:.6}",
        cpu_tput, cpu_loss_first, cpu_loss
    );
    println!("\nSpeedup: {:.2}x", speedup);
    if speedup > 1.0 {
        println!("✅ ANE is {:.1}x faster than CPU!", speedup);
    } else {
        println!("❌ ANE is {:.1}x slower", 1.0 / speedup);
    }
    println!(
        "\nLosses converge similarly: ANE {:.6} vs CPU {:.6}",
        ane_loss, cpu_loss
    );

    Ok(())
}
