//! ANE Peak Throughput Benchmark
//!
//! Based on "Inside the M4 Apple Neural Engine" inmem_peak.m
//!
//! Key insights for achieving 19 TFLOPS:
//! 1. Deep conv graphs (32-128 chained ops) - single ops only get ~30% utilization
//! 2. Conv 1x1 is ANE's native fast path - 3x faster than matmul
//! 3. Keep working set in SRAM (~32 MB on M4)
//!
//! This benchmark chains multiple 1x1 convolutions to fill the ANE pipeline.
//! Uses compile_multi() with separate weight blobs per layer.
//!
//! VERIFICATION STATUS (2026-03-26):
//! - Single-layer conv test: PASSED
//! - Multi-layer via compile_multi: PASSED (see ane_peak_multi_blob.rs)

use half::f16;
use rustane::mil::WeightBlob;
use rustane::wrapper::ANECompiler;
use std::time::Instant;

/// Build weight blobs for each conv layer
fn build_weight_blobs(channels: usize, depth: usize) -> Vec<WeightBlob> {
    // Match test_multi_conv_chain.rs - use sqrt(2/channels) scaling
    let scale = (2.0 / channels as f32).sqrt();
    (0..depth)
        .map(|i| {
            let total_weights = channels * channels;
            let weights: Vec<f32> = (0..total_weights)
                .map(|j| {
                    let seed = (i * 1000 + j) as f32;
                    ((seed % 1024.0) / 1024.0 * 2.0 - 1.0) * scale
                })
                .collect();
            WeightBlob::from_fp32(&weights, channels as i32, channels as i32).unwrap()
        })
        .collect()
}

/// Generate MIL for deep conv graph (chained 1x1 convolutions)
///
/// Uses FP16 input/output directly (matches test_residual_ffn.rs pattern)
/// Uses tensor wrapper syntax: tensor<string, []>("...") instead of string("...")
fn gen_mil(channels: usize, spatial: usize, depth: usize) -> String {
    let mut mil = String::new();

    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    // Use FP16 input directly (like test_residual_ffn.rs)
    mil.push_str(&format!("    func main<ios16>(tensor<fp16, [1, {}, 1, {}]> x) {{\n", channels, spatial));

    // Conv config constants
    mil.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // Chain depth convolutions - each weight is a SEPARATE blob
    let mut prev_name = "x".to_string();
    for i in 0..depth {
        mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, 1]> W{} = const()[name = tensor<string, []>(\"W{}\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w{}.bin\"), offset = tensor<uint64, []>(64)))]  ;\n",
            channels, channels, i, i, channels, channels, i));

        let out_name = format!("c{}", i);
        mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> {} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W{}, x = {})[name = tensor<string, []>(\"conv{}\")];\n",
            channels, spatial, out_name, i, prev_name, i));

        prev_name = out_name;
    }

    // Return final output (FP16, caller converts to FP32)
    mil.push_str(&format!("    }} -> ({});\n", prev_name));
    mil.push_str("}\n");

    mil
}

fn benchmark(channels: usize, spatial: usize, depth: usize, _debug: bool) -> Option<(f64, f64, bool)> {
    let mil = gen_mil(channels, spatial, depth);
    let weight_blobs = build_weight_blobs(channels, depth);

    // Calculate tensor sizes (FP16 now)
    let input_bytes = channels * spatial * 2; // FP16
    let output_bytes = input_bytes;

    // Prepare weight names and data for compile_multi
    let weight_names: Vec<String> = (0..depth)
        .map(|i| format!("@model_path/weights/w{}.bin", i))
        .collect();
    let weight_name_refs: Vec<&str> = weight_names.iter().map(|s| s.as_str()).collect();
    let weight_data_refs: Vec<&[u8]> = weight_blobs.iter().map(|b| b.as_bytes()).collect();
    let weight_lens: Vec<usize> = weight_blobs.iter().map(|b| b.len()).collect();

    // Compile with multiple weight blobs
    let compile_start = Instant::now();
    let mut compiler = ANECompiler::new();
    let exec = compiler.compile_multi(
        &mil,
        &weight_name_refs,
        &weight_data_refs,
        &weight_lens,
        &[input_bytes],
        &[output_bytes],
    );
    let compile_time = compile_start.elapsed();

    match exec {
        Ok(mut kernel) => {
            // Warmup
            for _ in 0..5 {
                // Use input=1.0 for better signal propagation
                let input: Vec<f16> = (0..channels * spatial).map(|_| f16::from_f32(1.0)).collect();
                let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
                let _ = kernel.write_input(0, &input_bytes);
                let _ = kernel.eval();
            }

            // Benchmark
            let iterations = 50;
            let exec_start = Instant::now();
            for _ in 0..iterations {
                let input: Vec<f16> = (0..channels * spatial).map(|_| f16::from_f32(1.0)).collect();
                let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
                let _ = kernel.write_input(0, &input_bytes);
                let _ = kernel.eval();
            }
            let exec_time = exec_start.elapsed() / iterations as u32;

            // Verify output
            let mut output = vec![0u8; output_bytes];
            kernel.read_output(0, &mut output).unwrap();

            let output_f32: Vec<f32> = output
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = (chunk[0] as u16) | ((chunk[1] as u16) << 8);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();

            let finite_count = output_f32.iter().filter(|x| x.is_finite()).count();
            let non_zero_count = output_f32.iter().filter(|x| x.abs() > 1e-10).count();
            let verified = finite_count == output_f32.len() && non_zero_count > 0;

            // Calculate GFLOPS for deep conv graph
            // Each conv: 2 * channels * channels * spatial FLOPs
            // Total: depth * 2 * channels^2 * spatial
            let gflops = 2.0 * channels as f64 * channels as f64 * spatial as f64 * depth as f64 / 1e9;
            let ms = exec_time.as_secs_f64() * 1000.0;
            let tflops = gflops / ms;

            Some((tflops, compile_time.as_secs_f64() * 1000.0, verified))
        }
        Err(e) => {
            println!("  Compile failed: {:?}", e);
            None
        }
    }
}

fn main() {
    rustane::init().ok();

    println!("=== ANE Peak Throughput Benchmark ===\n");
    println!("Based on inmem_peak.m from 'Inside the M4 Apple Neural Engine'\n");
    println!("{:<28} {:>7} {:>9} {:>10} {:>8} {:>8}",
             "Config", "W(MB)", "GFLOP", "ms/eval", "TFLOPS", "%peak");
    println!("{}", "-".repeat(80));

    // Test configurations - verified working patterns for multi-layer conv
    // ANE compiler limit: programs with 20+ layers fail with InvalidMILProgram
    let configs = [
        (64, 32, 4),      // Baseline: minimal depth
        (64, 32, 8),      // Moderate depth
        (64, 32, 12),     // Deeper graph
        (64, 32, 16),     // Max working depth for 64ch
        (128, 32, 8),     // Larger channels, moderate depth
        (256, 32, 4),     // Max channels, shallow depth
    ];

    let peak_tflops = 19.0; // M4 ANE peak FP16

    for (channels, spatial, depth) in configs {
        // Weight blob size: depth * (128-byte header + FP16 weights)
        let weight_bytes = depth * (128 + channels * channels * 2);
        let weight_mb = weight_bytes as f64 / (1024.0 * 1024.0);
        let gflops = 2.0 * channels as f64 * channels as f64 * spatial as f64 * depth as f64 / 1e9;

        let config_str = format!("{}x conv {}ch sp{}", depth, channels, spatial);

        if let Some((tflops, compile_ms, verified)) = benchmark(channels, spatial, depth, false) {
            let pct_peak = tflops / peak_tflops * 100.0;
            let ms = gflops / tflops;
            let status = if verified { "✓" } else { "✗ INVALID" };
            println!("{:<28} {:>7.1} {:>9.2} {:>10.3} ms {:>8.2} {:>7.1}% {} (compile: {:.1} ms)",
                     config_str, weight_mb, gflops, ms, tflops, pct_peak, status, compile_ms);
        } else {
            println!("{:<28} {:>7.1} {:>9.2} FAILED", config_str, weight_mb, gflops);
        }
    }

    println!("\n=== Key Insights ===");
    println!("1. Deep graphs (32-128 ops) fill ANE pipeline = 94%%+ utilization");
    println!("2. Single ops get only ~30%% utilization");
    println!("3. Conv 1x1 is ANE's native fast path (3x faster than matmul)");
    println!("4. Keep working set under ~32 MB for SRAM residency");
}
