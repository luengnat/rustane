//! ANE Peak Throughput Benchmark - Single Layer Version
//!
//! This benchmark uses a single large 1x1 convolution to measure ANE throughput.
//! While deep graphs (32-128 chained ops) achieve higher utilization per the
//! "Inside the M4 Apple Neural Engine" analysis, our current ANE bridge only
//! supports single-layer weight blobs.
//!
//! Key insights:
//! - Conv 1x1 is ANE's native fast path (3x faster than matmul)
//! - Keep working set in SRAM (~32 MB on M4) for peak performance
//! - Large tensors = more parallelism
//!
//! VERIFICATION STATUS (2026-03-26):
//! - Uses mil::WeightBlob format (verified working)
//! - Output verification: pending
//! - TFLOPS calculations: VALID formula, may underutilize ANE

use rustane::mil::WeightBlob;
use rustane::wrapper::ANECompiler;
use std::time::Instant;

/// Build weight blob for conv 1x1 with deterministic pseudo-random weights
fn build_weight_blob(channels: usize) -> WeightBlob {
    let total_weights = channels * channels;
    let weights: Vec<f32> = (0..total_weights)
        .map(|j| ((j % 1024) as f32 / 1024.0 * 2.0) - 1.0) // Random-ish FP32 in [-1, 1]
        .collect();

    WeightBlob::from_fp32(&weights, channels as i32, channels as i32).unwrap()
}

/// Generate MIL for single 1x1 convolution
fn gen_mil(channels: usize, spatial: usize) -> String {
    let mut mil = String::new();

    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!("    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n", channels, spatial));

    // Cast input to FP16
    mil.push_str("        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n");
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", channels, spatial));

    // Conv config constants
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");

    // Weight tensor - offset 64 to skip first part of 128-byte header
    // (The ANE compiler accepts offset 64 even though header is 128 bytes)
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        channels, channels, channels, channels));

    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n",
        channels, spatial));

    // Cast output back to FP32
    mil.push_str("        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n");
    mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> out = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", channels, spatial));
    mil.push_str("    } -> (out);\n");
    mil.push_str("}\n");

    mil
}

fn benchmark(channels: usize, spatial: usize) -> Option<(f64, f64, bool)> {
    let mil = gen_mil(channels, spatial);
    let weight_blob = build_weight_blob(channels);

    // Calculate tensor sizes
    let input_bytes = channels * spatial * 4; // FP32
    let output_bytes = input_bytes;

    // Compile with weight blob
    let compile_start = Instant::now();
    let mut compiler = ANECompiler::new();
    let exec = compiler.compile_single(&mil, Some(weight_blob.as_bytes()), &[input_bytes], &[output_bytes]);
    let compile_time = compile_start.elapsed();

    match exec {
        Ok(mut kernel) => {
            // Warmup
            for _ in 0..5 {
                let input = vec![0.1f32; channels * spatial];
                let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
                let _ = kernel.write_input(0, &input_bytes);
                let _ = kernel.eval();
            }

            // Benchmark
            let iterations = 50;
            let exec_start = Instant::now();
            for _ in 0..iterations {
                let input = vec![0.1f32; channels * spatial];
                let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
                let _ = kernel.write_input(0, &input_bytes);
                let _ = kernel.eval();
            }
            let exec_time = exec_start.elapsed() / iterations as u32;

            // Verify output
            let mut output = vec![0u8; output_bytes];
            kernel.read_output(0, &mut output).unwrap();

            let output_f32: Vec<f32> = output
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let finite_count = output_f32.iter().filter(|x| x.is_finite()).count();
            let non_zero_count = output_f32.iter().filter(|x| x.abs() > 1e-10).count();
            let verified = finite_count == output_f32.len() && non_zero_count > 0;

            // Calculate GFLOPS for single conv
            // Conv 1x1: 2 * channels * channels * spatial FLOPs
            let gflops = 2.0 * channels as f64 * channels as f64 * spatial as f64 / 1e9;
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

    println!("=== ANE Peak Throughput Benchmark (Single Layer) ===\n");
    println!("Uses single large conv 1x1 (deep graphs not supported by current ANE bridge)\n");
    println!("{:<24} {:>7} {:>9} {:>10} {:>8} {:>8}",
             "Config", "W(MB)", "GFLOP", "ms/eval", "TFLOPS", "%peak");
    println!("{}", "-".repeat(75));

    // Test configurations - vary channels and spatial to find sweet spot
    // Keep working set under ~32 MB for SRAM residency
    let configs = [
        // Large channels, moderate spatial (maximize compute)
        (512, 64),    // ~256 MB weights (too big for SRAM)
        (256, 64),    // ~64 MB weights
        (256, 32),    // ~64 MB weights, less compute
        (128, 64),    // ~16 MB weights (fits in SRAM)
        (128, 128),   // ~16 MB weights, more compute
        (64, 128),    // ~4 MB weights (well within SRAM)
    ];

    let peak_tflops = 19.0; // M4 ANE peak FP16

    for (channels, spatial) in configs {
        // Weight blob size: 128-byte header + FP16 weights
        let weight_bytes = 128 + channels * channels * 2;
        let weight_mb = weight_bytes as f64 / (1024.0 * 1024.0);
        let gflops = 2.0 * channels as f64 * channels as f64 * spatial as f64 / 1e9;

        let config_str = format!("conv {}ch sp{}", channels, spatial);

        if let Some((tflops, compile_ms, verified)) = benchmark(channels, spatial) {
            let pct_peak = tflops / peak_tflops * 100.0;
            let ms = gflops / tflops;
            let status = if verified { "✓" } else { "✗ INVALID" };
            println!("{:<24} {:>7.1} {:>9.2} {:>10.3} ms {:>8.2} {:>7.1}% {}",
                     config_str, weight_mb, gflops, ms, tflops, pct_peak, status);
            println!("  (Compile time: {:.1} ms)", compile_ms);
        } else {
            println!("{:<24} {:>7.1} {:>9.2} FAILED", config_str, weight_mb, gflops);
        }
    }

    println!("\n=== Notes ===");
    println!("1. Single conv operations typically achieve ~30% ANE utilization");
    println!("2. Deep graphs (32-128 chained ops) achieve 94%+ utilization");
    println!("3. For peak throughput, use deep conv graphs (see ane_peak_throughput.rs)");
    println!("4. Multi-layer conv now supported via compile_multi() API");
}
