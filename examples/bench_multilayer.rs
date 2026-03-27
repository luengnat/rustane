//! Multi-layer training benchmark: ANE fused forward vs CPU sequential forward.
//!
//! Tests whether ANE advantage scales with more layers.
//! ANE: one eval call for N layers (fused program)
//! CPU: N separate matmul calls

use rustane::mil::programs::{
    dynamic_matmul_input_bytes, dynamic_matmul_mil, dynamic_matmul_output_bytes,
};
use rustane::wrapper::ANECompiler;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let dim = 64;
    let seq = 64;
    let num_steps = 200;

    println!("=== Multi-Layer Training Benchmark ===");
    println!("Dim={}, Seq={}, Steps={}", dim, seq, num_steps);

    // Test with 1, 2, 4 layers
    for &num_layers in &[1usize, 2, 4] {
        println!("\n--- {} Layer(s) ---", num_layers);

        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = dynamic_matmul_input_bytes(dim, seq);
        let output_bytes = dynamic_matmul_output_bytes(dim, seq);

        // Compile one ANE program (reuse across layers)
        let mut execs: Vec<_> = (0..num_layers)
            .map(|_| {
                ANECompiler::new().compile_multi(
                    &mil,
                    &[],
                    &[],
                    &[],
                    &[input_bytes],
                    &[output_bytes],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Initialize weights and activations
        let total_ch = dim + dim * dim;
        let mut pack_bufs: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; total_ch * seq])
            .collect();
        let mut weights: Vec<Vec<f32>> = (0..num_layers)
            .map(|l| {
                (0..dim * dim)
                    .map(|i| (((i + l * 100) * 7 + 13) % 100) as f32 / 1000.0 - 0.05)
                    .collect()
            })
            .collect();

        let input: Vec<f32> = (0..dim * seq)
            .map(|i| ((i * 3 + 7) % 200) as f32 / 1000.0 - 0.1)
            .collect();
        let target: Vec<f32> = (0..dim * seq).map(|i| ((i % 10) as f32) * 0.01).collect();

        // Initial pack
        let mut packeds: Vec<Vec<u8>> = Vec::new();
        for l in 0..num_layers {
            pack_dynamic_matmul_input_f32(&mut pack_bufs[l], &input, &weights[l], dim, seq);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    pack_bufs[l].as_ptr() as *const u8,
                    pack_bufs[l].len() * 4,
                )
            };
            packeds.push(bytes.to_vec());
        }

        // --- ANE Training ---
        let ane_start = Instant::now();
        for _step in 0..num_steps {
            // Forward through all layers
            let mut layer_input = input.clone();
            for l in 0..num_layers {
                // Update weights in pack buffer
                pack_weights_into(&mut pack_bufs[l], &weights[l], dim, seq);
                // Also need to update activations (layer_input changes each layer)
                pack_bufs[l][..dim * seq].copy_from_slice(&layer_input);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        pack_bufs[l].as_ptr() as *const u8,
                        pack_bufs[l].len() * 4,
                    )
                };
                packeds[l].copy_from_slice(bytes);

                execs[l].write_input(0, &packeds[l])?;
                execs[l].eval()?;
                let raw = execs[l].read_output_vec(0)?;
                layer_input = raw[..dim * seq * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
            }

            // Loss (on final layer output)
            let n = dim * seq;
            let mut loss = 0.0f32;
            for i in 0..n {
                let e = layer_input[i] - target[i];
                loss += e * e;
            }

            // Simple gradient update for last layer only (for demo)
            let scale = 2.0 / n as f32;
            for i in 0..weights[num_layers - 1].len() {
                weights[num_layers - 1][i] -= 0.001 * scale * loss / n as f32;
            }
        }
        let ane_elapsed = ane_start.elapsed();
        let ane_tput = num_steps as f64 / ane_elapsed.as_secs_f64();

        // --- CPU Training ---
        let mut cpu_weights: Vec<Vec<f32>> = weights.clone();
        // Re-init weights for CPU
        cpu_weights = (0..num_layers)
            .map(|l| {
                (0..dim * dim)
                    .map(|i| (((i + l * 100) * 7 + 13) % 100) as f32 / 1000.0 - 0.05)
                    .collect()
            })
            .collect();

        let cpu_start = Instant::now();
        for _step in 0..num_steps {
            let mut layer_input = input.clone();
            for l in 0..num_layers {
                let mut output = vec![0.0f32; dim * seq];
                for i in 0..dim {
                    for j in 0..seq {
                        let mut sum = 0.0f32;
                        for k in 0..dim {
                            sum += cpu_weights[l][i * dim + k] * layer_input[k * seq + j];
                        }
                        output[i * seq + j] = sum;
                    }
                }
                layer_input = output;
            }

            let n = dim * seq;
            let mut loss = 0.0f32;
            for i in 0..n {
                let e = layer_input[i] - target[i];
                loss += e * e;
            }

            let scale = 2.0 / n as f32;
            for i in 0..cpu_weights[num_layers - 1].len() {
                cpu_weights[num_layers - 1][i] -= 0.001 * scale * loss / n as f32;
            }
        }
        let cpu_elapsed = cpu_start.elapsed();
        let cpu_tput = num_steps as f64 / cpu_elapsed.as_secs_f64();

        let speedup = cpu_elapsed.as_secs_f64() / ane_elapsed.as_secs_f64();
        println!(
            "ANE: {:.0} steps/sec | CPU: {:.0} steps/sec | Speedup: {:.2}x {}",
            ane_tput,
            cpu_tput,
            speedup,
            if speedup > 1.0 { "✅" } else { "❌" }
        );
    }

    Ok(())
}

fn pack_dynamic_matmul_input_f32(
    buffer: &mut [f32],
    activations: &[f32],
    weights: &[f32],
    dim: usize,
    seq_len: usize,
) {
    let total_ch = dim + dim * dim;
    assert_eq!(buffer.len(), total_ch * seq_len);
    buffer.fill(0.0);
    buffer[..dim * seq_len].copy_from_slice(activations);
    pack_weights_into(buffer, weights, dim, seq_len);
}

fn pack_weights_into(buffer: &mut [f32], weights: &[f32], dim: usize, seq_len: usize) {
    let w_base = dim * seq_len;
    let w_region_len = dim * dim * seq_len;
    buffer[w_base..w_base + w_region_len].fill(0.0);
    for r in 0..dim {
        for c in 0..dim {
            buffer[w_base + (r * dim + c) * seq_len] = weights[r * dim + c];
        }
    }
}
