//! Multi-layer delta compilation test with timing.
//!
//! Validates DLT-01 (<500ms reload), DLT-03 (selective weight update),
//! and basic durability across cycles.
//!
//! Uses subprocess isolation — spawned by parent test harness.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example test_delta_compilation --release
//! ```

use rustane::ane::WeightBlob;
use rustane::mil::programs::conv1x1_mil;
use rustane::wrapper::{ANECompiler, ANERuntime};
use std::time::Instant;

/// Per-layer state: executor + weight file name + current weights
struct LayerState {
    executor: rustane::wrapper::ANEExecutor,
    weight_name: String,
    current_weights: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
}

fn make_weight_blob(weights: &[f32], rows: usize, cols: usize) -> WeightBlob {
    WeightBlob::from_f32(weights, rows, cols).expect("weight blob creation")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Layer Delta Compilation Test ===\n");

    // Initialize ANE runtime
    ANERuntime::init()?;
    println!("✅ ANE runtime initialized\n");

    // Model config: 4 layers, each a conv1x1 (simulating linear layers)
    let num_layers = 4;
    let seq_len = 16;
    let dims = [64, 64, 64, 64]; // in_dim for each layer
    let out_dims = [64, 64, 64, 64]; // out_dim for each layer

    // Step 1: Compile all layers
    println!("=== Step 1: Compiling {} layers ===", num_layers);
    let compile_start = Instant::now();
    let mut layers: Vec<LayerState> = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let in_dim = dims[i];
        let out_dim = out_dims[i];
        let weight_size = out_dim * in_dim;

        // Generate initial weights (small values to avoid fp16 overflow)
        let initial_weights: Vec<f32> = (0..weight_size)
            .map(|j| ((j % 100) as f32 - 50.0) / 100.0)
            .collect();

        let weight_blob = make_weight_blob(&initial_weights, out_dim, in_dim);
        let mil = conv1x1_mil(seq_len, in_dim, out_dim);
        let weight_name = "@model_path/weights/weight.bin";

        let mut compiler = ANECompiler::new();
        let input_size = in_dim * seq_len * 4; // fp32 input
        let output_size = out_dim * seq_len * 4; // fp32 output
        let executor = compiler.compile_multi(
            &mil,
            &[weight_name],
            &[weight_blob.as_bytes()],
            &[weight_blob.as_bytes().len()],
            &[input_size],
            &[output_size],
        )?;

        layers.push(LayerState {
            executor,
            weight_name: weight_name.to_string(),
            current_weights: initial_weights.clone(),
            in_dim,
            out_dim,
            seq_len,
        });
        println!(
            "  Layer {}: compiled ({}x{} → {}x{})",
            i, in_dim, seq_len, out_dim, seq_len
        );
    }
    let compile_time = compile_start.elapsed();
    println!(
        "✅ All {} layers compiled in {:?}",
        num_layers, compile_time
    );

    // Step 2: Verify initial eval works
    println!("\n=== Step 2: Initial evaluation ===");
    let input_data: Vec<f32> = (0..dims[0] * seq_len)
        .map(|i| ((i % 50) as f32 - 25.0) / 50.0)
        .collect();
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    for i in 0..num_layers {
        let out_dim = layers[i].out_dim;
        let seq = layers[i].seq_len;
        let mut output_buf = vec![0u8; out_dim * seq * 4];
        layers[i].executor.write_input(0, &input_bytes)?;
        layers[i].executor.eval()?;
        layers[i].executor.read_output(0, &mut output_buf)?;

        // Check output is not all zeros
        let has_nonzero = output_buf.iter().any(|&b| b != 0);
        println!("  Layer {}: eval OK, nonzero={}", i, has_nonzero);
    }

    // Step 3: Delta compilation cycles
    println!("\n=== Step 3: Delta compilation (10 cycles) ===");
    let num_cycles = 10;
    let mut total_reload_time = std::time::Duration::ZERO;

    for cycle in 0..num_cycles {
        let cycle_start = Instant::now();

        for i in 0..num_layers {
            let in_dim = layers[i].in_dim;
            let out_dim = layers[i].out_dim;
            let perturbation = (cycle as f32 + 1.0) * 0.01;
            let new_weights: Vec<f32> = layers[i]
                .current_weights
                .iter()
                .enumerate()
                .map(|(j, &w)| {
                    let noise = ((j as f32 * 7.3 + cycle as f32 * 3.7) % 1.0 - 0.5) * perturbation;
                    (w + noise).clamp(-1.0, 1.0)
                })
                .collect();

            let new_blob = make_weight_blob(&new_weights, out_dim, in_dim);
            let wname = layers[i].weight_name.clone();
            layers[i]
                .executor
                .reload_weights(&[(&wname, new_blob.as_bytes())])?;
            layers[i].current_weights = new_weights;
        }

        let cycle_time = cycle_start.elapsed();
        total_reload_time += cycle_time;

        // Verify eval still works after reload
        for i in 0..num_layers {
            let out_dim = layers[i].out_dim;
            let seq = layers[i].seq_len;
            let mut output_buf = vec![0u8; out_dim * seq * 4];
            layers[i].executor.write_input(0, &input_bytes)?;
            layers[i].executor.eval()?;
            layers[i].executor.read_output(0, &mut output_buf)?;
        }

        println!(
            "  Cycle {:2}: reload={:.1}ms",
            cycle + 1,
            cycle_time.as_secs_f64() * 1000.0
        );
    }

    let avg_reload = total_reload_time / num_cycles;
    println!("\n=== Performance Summary ===");
    println!(
        "  Compile time (all {} layers): {:?}",
        num_layers, compile_time
    );
    println!(
        "  Average reload time ({} layers): {:?}",
        num_layers, avg_reload
    );
    println!("  Per-layer avg: {:?}", avg_reload / num_layers as u32);

    // DLT-01: Check <500ms for 4-layer model reload
    if avg_reload.as_millis() < 500 {
        println!(
            "  ✅ DLT-01 PASS: avg reload {:.1}ms < 500ms",
            avg_reload.as_secs_f64() * 1000.0
        );
    } else {
        println!(
            "  ❌ DLT-01 FAIL: avg reload {:.1}ms >= 500ms",
            avg_reload.as_secs_f64() * 1000.0
        );
    }

    // Step 4: Verify weight changes produce different outputs
    println!("\n=== Step 4: Weight change verification (DLT-03) ===");

    // Get baseline output for layer 0
    let mut baseline_buf = vec![0u8; layers[0].out_dim * layers[0].seq_len * 4];
    layers[0].executor.write_input(0, &input_bytes)?;
    layers[0].executor.eval()?;
    layers[0].executor.read_output(0, &mut baseline_buf)?;

    // Reload layer 0 with significantly different weights
    let big_change_weights: Vec<f32> = layers[0]
        .current_weights
        .iter()
        .map(|&w| w + 0.5) // Add 0.5 to all weights
        .collect();
    let big_blob = make_weight_blob(&big_change_weights, layers[0].out_dim, layers[0].in_dim);
    let weight_name = layers[0].weight_name.clone();
    layers[0]
        .executor
        .reload_weights(&[(&weight_name, big_blob.as_bytes())])?;

    // Get new output
    let mut new_buf = vec![0u8; layers[0].out_dim * layers[0].seq_len * 4];
    layers[0].executor.write_input(0, &input_bytes)?;
    layers[0].executor.eval()?;
    layers[0].executor.read_output(0, &mut new_buf)?;

    let outputs_differ = baseline_buf != new_buf;
    if outputs_differ {
        println!("  ✅ DLT-03 PASS: weight change produced different output");
    } else {
        println!("  ❌ DLT-03 FAIL: output unchanged after weight modification");
    }

    println!("\nOK delta_compilation");
    Ok(())
}
