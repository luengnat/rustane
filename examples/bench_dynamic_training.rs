//! Benchmark: ANE dynamic-weight training vs CPU training.
//!
//! Measures end-to-end training throughput for a single linear layer:
//!   Forward: y = W @ x (matmul)
//!   Loss: MSE(y, target)
//!   Gradient: dW = dL/dW (computed on CPU)
//!   Update: W -= lr * dW (write to input IOSurface)
//!
//! The ANE version uses dynamic matmul (weights in input tensor).
//! Zero recompilation — weights change by modifying IOSurface content.

use rustane::mil::programs::{
    dynamic_matmul_input_bytes, dynamic_matmul_mil, dynamic_matmul_output_bytes,
    pack_dynamic_matmul_input,
};
use rustane::wrapper::ANECompiler;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let dim = 64;
    let seq = 64;
    let lr = 0.01f32;
    let num_steps = 500;

    println!("=== Dynamic Weight Training Benchmark ===");
    println!("Dim={}, Seq={}, LR={}, Steps={}", dim, seq, lr, num_steps);

    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = dynamic_matmul_input_bytes(dim, seq);
    let output_bytes = dynamic_matmul_output_bytes(dim, seq);

    // --- ANE Setup ---
    println!("\n--- ANE Setup ---");
    let mut exec =
        ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes])?;
    println!("Compiled dynamic matmul program");

    // Initialize: small random weights, small random input, fixed target
    let mut rng = SimpleRng::new(42);
    let mut weights: Vec<f32> = (0..dim * dim)
        .map(|_| rng.next_f32() * 0.1 - 0.05)
        .collect();
    let input: Vec<f32> = (0..dim * seq)
        .map(|_| rng.next_f32() * 0.5 - 0.25)
        .collect();
    let target: Vec<f32> = (0..dim * seq).map(|i| ((i % 10) as f32) * 0.01).collect();

    // --- ANE Training ---
    println!("\n--- ANE Training ({} steps, no recompile) ---", num_steps);
    let ane_start = Instant::now();
    let mut ane_loss = 0.0f32;

    for _step in 0..num_steps {
        // Forward: pack input + weights, run ANE
        let packed = pack_dynamic_matmul_input(&input, &weights, dim, seq);
        exec.write_input(0, &packed)?;
        exec.eval()?;
        let raw_out = exec.read_output_vec(0)?;

        // Read output as fp32
        let output: Vec<f32> = raw_out[..dim * seq * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Loss: MSE
        let loss: f32 = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / (dim * seq) as f32;
        ane_loss = loss;

        // Gradient: dW = 2/N * (output - target) @ input^T
        // output: [D, S], input: [D, S] -> dW: [D, D]
        let scale = 2.0 / (dim * seq) as f32;
        let mut grad = vec![0.0f32; dim * dim];
        for d_out in 0..dim {
            for d_in in 0..dim {
                let mut sum = 0.0f32;
                for s in 0..seq {
                    let err = output[d_out * seq + s] - target[d_out * seq + s];
                    sum += err * input[d_in * seq + s];
                }
                grad[d_out * dim + d_in] = scale * sum;
            }
        }

        // SGD update: W -= lr * grad
        for i in 0..weights.len() {
            weights[i] -= lr * grad[i];
        }
    }
    let ane_elapsed = ane_start.elapsed();
    let ane_throughput = num_steps as f64 / ane_elapsed.as_secs_f64();

    println!("Time: {:.2?}", ane_elapsed);
    println!("Throughput: {:.0} steps/sec", ane_throughput);
    println!("Final loss: {:.6}", ane_loss);

    // --- CPU Training (same logic, CPU matmul) ---
    println!("\n--- CPU Training ({} steps) ---", num_steps);
    let mut rng2 = SimpleRng::new(42);
    let mut cpu_weights: Vec<f32> = (0..dim * dim)
        .map(|_| rng2.next_f32() * 0.1 - 0.05)
        .collect();

    let cpu_start = Instant::now();
    let mut cpu_loss = 0.0f32;

    for _step in 0..num_steps {
        // Forward: CPU matmul
        let output = cpu_matmul(&cpu_weights, &input, dim, dim, seq);

        // Loss
        let loss: f32 = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / (dim * seq) as f32;
        cpu_loss = loss;

        // Gradient
        let scale = 2.0 / (dim * seq) as f32;
        let mut grad = vec![0.0f32; dim * dim];
        for d_out in 0..dim {
            for d_in in 0..dim {
                let mut sum = 0.0f32;
                for s in 0..seq {
                    let err = output[d_out * seq + s] - target[d_out * seq + s];
                    sum += err * input[d_in * seq + s];
                }
                grad[d_out * dim + d_in] = scale * sum;
            }
        }

        // SGD update
        for i in 0..cpu_weights.len() {
            cpu_weights[i] -= lr * grad[i];
        }
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_throughput = num_steps as f64 / cpu_elapsed.as_secs_f64();

    println!("Time: {:.2?}", cpu_elapsed);
    println!("Throughput: {:.0} steps/sec", cpu_throughput);
    println!("Final loss: {:.6}", cpu_loss);

    // --- Comparison ---
    println!("\n=== RESULTS ===");
    let speedup = cpu_elapsed.as_secs_f64() / ane_elapsed.as_secs_f64();
    println!("ANE throughput: {:.0} steps/sec", ane_throughput);
    println!("CPU throughput: {:.0} steps/sec", cpu_throughput);
    println!("Speedup: {:.2}x", speedup);
    println!("ANE loss: {:.6}", ane_loss);
    println!("CPU loss: {:.6}", cpu_loss);
    if speedup > 1.0 {
        println!("✅ ANE is {:.1}x faster than CPU", speedup);
    } else {
        println!(
            "❌ ANE is {:.1}x SLOWER than CPU (need to optimize)",
            1.0 / speedup
        );
    }

    Ok(())
}

fn cpu_matmul(w: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // w: [m, k], x: [k, n] -> [m, n]
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += w[i * k + l] * x[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Simple xorshift RNG for reproducibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state % 1_000_000) as f32) / 1_000_000.0
    }
}
