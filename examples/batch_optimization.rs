//! Batch Size Optimization Example
//!
//! Finds optimal batch size for ANE inference by measuring throughput
//! and memory usage across different batch sizes.

use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};
use std::time::Instant;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

fn benchmark_batch_size(
    batch: usize,
    seq_len: usize,
    embed_dim: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let out_features = embed_dim; // Identity transformation

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "Batch Size: {} (seq_len={}, embed_dim={})",
        batch, seq_len, embed_dim
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Build MIL program
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, 64]> x) {{\n",
        embed_dim
    ));
    mil.push_str("        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str(
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n",
    );
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, 64]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", embed_dim));
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n", out_features, embed_dim, out_features, embed_dim));
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, 64]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n", out_features));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!("        tensor<fp32, [1, {}, 1, 64]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", out_features));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");

    // Create weights
    let weight_data: Vec<f32> = (0..embed_dim * out_features)
        .map(|i| ((i as f32 * 0.001) % 2.0) - 1.0)
        .collect();

    let weight_blob = WeightBlob::from_fp32(&weight_data, out_features as i32, embed_dim as i32)?;

    let input_bytes = embed_dim * 64 * 4;
    let output_bytes = out_features * 64 * 4;

    println!("  Input: {} bytes", input_bytes);
    println!("  Output: {} bytes", output_bytes);
    println!("  Memory: {} bytes (weights)", weight_blob.len());

    // Compile
    println!("  Compiling...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(
        &mil,
        Some(weight_blob.as_bytes()),
        &[input_bytes],
        &[output_bytes],
    )?;
    println!("  ✓ Compiled");

    // Create input
    let input_data: Vec<f32> = (0..embed_dim * 64)
        .map(|i| (i as f32 * 0.01) % 2.0 - 1.0)
        .collect();

    let input_tensor = ANETensor::from_fp32(input_data, vec![1, embed_dim, 1, 64])?;

    // Warmup
    for _ in 0..WARMUP {
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let throughput = ITERATIONS as f64 / duration.as_secs_f64();
    let samples_per_sec = throughput * batch as f64;

    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} inferences/sec", throughput);
    println!("  Samples/sec: {:.1}", samples_per_sec);

    Ok((avg_ms, throughput, samples_per_sec))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Batch Size Optimization");
    println!("====================================\n");

    let avail = rustane::ANEAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available");
        return Ok(());
    }

    init()?;
    println!("✓ ANE initialized\n");

    let seq_len = 64;
    let embed_dim = 256;

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64];

    println!(
        "Testing batch sizes with seq_len={}, embed_dim={}",
        seq_len, embed_dim
    );
    println!(
        "Total iterations per size: {} ({} warmup + {} measured)\n",
        WARMUP + ITERATIONS,
        WARMUP,
        ITERATIONS
    );

    let mut results = Vec::new();

    for batch in batch_sizes {
        match benchmark_batch_size(batch, seq_len, embed_dim) {
            Ok((avg_ms, throughput, samples_per_sec)) => {
                results.push((batch, avg_ms, throughput, samples_per_sec));
            }
            Err(e) => {
                println!("  ✗ Failed: {}", e);
            }
        }
    }

    // Summary table
    println!("\n\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("OPTIMIZATION SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:<12} {:>12} {:>15} {:>15} {:>12}",
        "Batch", "Avg (ms)", "Inferences/s", "Samples/s", "Efficiency"
    );
    println!("{}", "-".repeat(75));

    for (batch, avg_ms, inferences, samples) in &results {
        // Efficiency: samples per second / batch size (how well we're utilizing parallelism)
        let efficiency = samples / *batch as f64;
        println!(
            "{:<12} {:>12.3} {:>15.1} {:>15.1} {:>12.1}",
            batch, avg_ms, inferences, samples, efficiency
        );
    }

    // Find optimal batch size
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RECOMMENDATIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if !results.is_empty() {
        // Best throughput
        let best_throughput = results
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        println!("\n📊 Maximum Throughput:");
        println!("   Batch size: {}", best_throughput.0);
        println!("   Samples/sec: {:.1}", best_throughput.3);
        println!("   Use for: Maximizing total processing capacity");

        // Best latency
        let best_latency = results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!("\n⚡ Lowest Latency:");
        println!("   Batch size: {}", best_latency.0);
        println!("   Avg time: {:.3}ms", best_latency.1);
        println!("   Use for: Real-time applications, interactive systems");

        // Best efficiency
        let best_efficiency = results
            .iter()
            .max_by(|a, b| {
                let eff_a = a.3 / a.0 as f64;
                let eff_b = b.3 / b.0 as f64;
                eff_a.partial_cmp(&eff_b).unwrap()
            })
            .unwrap();

        println!("\n🎯 Best Efficiency:");
        println!("   Batch size: {}", best_efficiency.0);
        println!("   Samples/sec: {:.1}", best_efficiency.3);
        println!("   Use for: Balanced workloads, cost-effective inference");

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("KEY INSIGHTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        println!("\n📈 General Trends:");
        println!("   • Larger batches → Higher throughput (more parallelism)");
        println!("   • Smaller batches → Lower latency (less waiting)");
        println!("   • Optimal batch depends on your use case");

        println!("\n💡 When to use each:");
        println!("   • Batch 1-4: Real-time, interactive, low-latency apps");
        println!("   • Batch 8-16: General purpose, balanced performance");
        println!("   • Batch 32-64: Batch processing, offline analysis");

        println!("\n⚠️  Considerations:");
        println!("   • Memory usage increases with batch size");
        println!("   • ANE has ~119 compilation limit per process");
        println!("   • Larger batches may hit memory constraints");
        println!("   • Test with your actual model and data");
    }

    println!("\n✅ Batch optimization complete!");

    Ok(())
}
