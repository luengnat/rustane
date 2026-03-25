//! Simple training loop: ANE forward + CPU gradient + delta compilation.
//!
//! Demonstrates a complete training step:
//!   forward (ANE) → loss (CPU) → gradient (CPU) → SGD update → delta compile (reload)
//!
//! Model: 2-layer conv1x1 (64→64→64)
//! Loss: MSE
//! Optimizer: SGD (lr=0.01)
//!
//! Validates TRL-01 (complete training step) and TRL-02 (loss decreases).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example train_simple --release
//! ```

use rustane::ane::WeightBlob;
use rustane::mil::programs::conv1x1_mil;
use rustane::training::DeltaCompiler;
use rustane::wrapper::ANERuntime;
use std::time::Instant;

const DIM: usize = 64;
const SEQ: usize = 16;
const NUM_STEPS: usize = 50;
const LR: f32 = 0.01;
const WEIGHT_NAME: &str = "@model_path/weights/weight.bin";

/// Simple f32 matrix multiply: C = A * B
/// A: [M, K], B: [K, N], C: [M, N]
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose matrix: A^T
/// A: [M, N] -> A^T: [N, M]
fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// MSE loss
fn mse_loss(output: &[f32], target: &[f32]) -> f32 {
    let n = output.len() as f32;
    output
        .iter()
        .zip(target.iter())
        .map(|(o, t)| (o - t) * (o - t))
        .sum::<f32>()
        / n
}

/// Convert f32 slice to bytes (little-endian)
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Convert bytes to f32 slice (little-endian)
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Training Loop (ANE Forward + CPU Gradient) ===\n");

    ANERuntime::init()?;
    println!("✅ ANE runtime initialized");

    // Initialize DeltaCompiler
    let mut dc = DeltaCompiler::new();

    // Model weights: 2 layers, each [DIM, DIM] for conv1x1
    let mut weights: Vec<Vec<f32>> = (0..2)
        .map(|layer| {
            (0..DIM * DIM)
                .map(|i| {
                    // Small random-ish weights, seeded by layer
                    let v = (((i * 7 + layer * 1000 + 13) % 100) as f32 - 50.0) / 200.0;
                    v // range: -0.25 to 0.25
                })
                .collect()
        })
        .collect();

    // Compile layers
    println!("\n=== Compiling 2-layer model ===");
    let input_size = DIM * SEQ * 4; // fp32
    let output_size = DIM * SEQ * 4; // fp32

    for layer in 0..2 {
        let blob = WeightBlob::from_f32(&weights[layer], DIM, DIM)?;
        let mil = conv1x1_mil(SEQ, DIM, DIM);
        let idx = dc.add_program(
            &mil,
            &[WEIGHT_NAME],
            &[blob.as_bytes()],
            &[blob.as_bytes().len()],
            &[input_size],
            &[output_size],
        )?;
        println!("  Layer {} compiled (idx={})", layer, idx);
    }

    println!(
        "  Compile budget: {}/{}",
        dc.compile_count(),
        dc.remaining_budget()
    );

    // Generate synthetic data
    let input: Vec<f32> = (0..DIM * SEQ)
        .map(|i| ((i % 50) as f32 - 25.0) / 100.0)
        .collect();
    let target: Vec<f32> = (0..DIM * SEQ)
        .map(|i| {
            // Target is a simple function of input (learnable)
            let x = input[i];
            x * 0.5 + 0.1
        })
        .collect();

    let input_bytes = f32_to_bytes(&input);

    // Training loop
    println!("\n=== Training ({} steps, lr={}) ===", NUM_STEPS, LR);
    let mut losses = Vec::with_capacity(NUM_STEPS);
    let train_start = Instant::now();

    for step in 0..NUM_STEPS {
        let step_start = Instant::now();

        // --- Forward pass (ANE) ---
        // Layer 0: input -> h0
        dc.executor_mut(0)?.write_input(0, &input_bytes)?;
        dc.executor_mut(0)?.eval()?;
        let mut h0_bytes = vec![0u8; output_size];
        dc.executor_mut(0)?.read_output(0, &mut h0_bytes)?;
        let h0 = bytes_to_f32(&h0_bytes);

        // Layer 1: h0 -> output
        dc.executor_mut(1)?.write_input(0, &h0_bytes)?;
        dc.executor_mut(1)?.eval()?;
        let mut out_bytes = vec![0u8; output_size];
        dc.executor_mut(1)?.read_output(0, &mut out_bytes)?;
        let output = bytes_to_f32(&out_bytes);

        // --- Loss (CPU) ---
        let loss = mse_loss(&output, &target);
        losses.push(loss);

        // --- Gradient (CPU) ---
        // dL/d(output) = 2 * (output - target) / N
        let n = (DIM * SEQ) as f32;
        let dloss: Vec<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t) / n)
            .collect();

        // Reshape: dloss is [DIM, SEQ] (treat as column-major for matmul)
        // grad_W1 = dloss * h0^T  where dloss:[DIM,SEQ], h0:[DIM,SEQ]
        // grad_W1[i][j] = sum_k dloss[i][k] * h0[j][k]
        let h0_t = transpose(&h0, DIM, SEQ); // [SEQ, DIM]
        let grad_w1 = matmul(&dloss, &h0_t, DIM, SEQ, DIM); // [DIM, DIM]

        // dL/dh0 = W1^T * dloss
        let w1_t = transpose(&weights[1], DIM, DIM); // [DIM, DIM]
        let dloss_h0 = matmul(&w1_t, &dloss, DIM, DIM, SEQ); // [DIM, SEQ]

        // grad_W0 = dloss_h0 * input^T
        let input_t = transpose(&input, DIM, SEQ); // [SEQ, DIM]
        let grad_w0 = matmul(&dloss_h0, &input_t, DIM, SEQ, DIM); // [DIM, DIM]

        // --- SGD update (CPU) ---
        for i in 0..weights[0].len() {
            weights[0][i] -= LR * grad_w0[i];
        }
        for i in 0..weights[1].len() {
            weights[1][i] -= LR * grad_w1[i];
        }

        // --- Delta compilation (reload weights) ---
        let blob0 = WeightBlob::from_f32(&weights[0], DIM, DIM)?;
        let blob1 = WeightBlob::from_f32(&weights[1], DIM, DIM)?;
        dc.reload_layer(0, &[(WEIGHT_NAME, blob0.as_bytes())])?;
        dc.reload_layer(1, &[(WEIGHT_NAME, blob1.as_bytes())])?;

        let step_time = step_start.elapsed();

        if step == 0 || step == NUM_STEPS - 1 || step % 10 == 0 {
            println!(
                "  Step {:3}: loss={:.6} time={:.1}ms",
                step,
                loss,
                step_time.as_secs_f64() * 1000.0
            );
        }
    }

    let total_time = train_start.elapsed();
    println!("\n=== Training Complete ===",);
    println!("  Total time: {:?}", total_time);
    println!(
        "  Avg step time: {:.1}ms",
        total_time.as_secs_f64() * 1000.0 / NUM_STEPS as f64
    );

    // TRL-02: Loss should decrease
    let initial_loss = losses[0];
    let final_loss = losses[NUM_STEPS - 1];
    let loss_decreased = final_loss < initial_loss;

    println!("\n=== Loss Analysis ===");
    println!("  Initial loss: {:.6}", initial_loss);
    println!("  Final loss:   {:.6}", final_loss);
    println!(
        "  Reduction:    {:.1}%",
        (1.0 - final_loss / initial_loss) * 100.0
    );

    if loss_decreased {
        println!(
            "  ✅ TRL-02 PASS: loss decreased ({:.6} → {:.6})",
            initial_loss, final_loss
        );
    } else {
        println!(
            "  ❌ TRL-02 FAIL: loss did not decrease ({:.6} → {:.6})",
            initial_loss, final_loss
        );
    }

    // Budget check
    println!("\n=== Compile Budget ===");
    println!("  Compiles used (this session): {}", dc.compiles_used());
    println!(
        "  Total process compiles: {}/{}",
        dc.compile_count(),
        dc.budget_status().limit
    );

    println!("\nOK train_simple");
    Ok(())
}
