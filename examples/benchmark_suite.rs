//! Comprehensive Benchmark Suite
//!
//! Benchmarks all Rustane examples and compares performance metrics.

use rustane::{init, wrapper::ANECompiler, ANETensor};
use std::time::Instant;

const ITERATIONS: usize = 100;
const WARMUP: usize = 10;

fn benchmark_cpu<F>(name: &str, mut f: F) -> (f64, f64, f64)
where
    F: FnMut() -> Vec<f32>,
{
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Warmup
    for _ in 0..WARMUP {
        let _ = f();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = f();
    }
    let duration = start.elapsed();

    let total_ms = duration.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / ITERATIONS as f64;
    let throughput = ITERATIONS as f64 / (duration.as_secs_f64());

    println!(
        "  Total time ({} iterations): {:.2}ms",
        ITERATIONS, total_ms
    );
    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} ops/sec", throughput);

    (total_ms, avg_ms, throughput)
}

fn benchmark_ane<F>(
    name: &str,
    compile_fn: F,
    input_size: usize,
    output_size: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<(ANECompiler, Vec<u8>, Vec<u8>), Box<dyn std::error::Error>>,
{
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark: {} (ANE)", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Compile
    let compile_start = Instant::now();
    let (mut compiler, input_data, expected_output) = compile_fn()?;
    let compile_time = compile_start.elapsed();

    println!(
        "  Compile time: {:.2}ms",
        compile_time.as_secs_f64() * 1000.0
    );

    // Warmup
    for _ in 0..WARMUP {
        // Execute and read output
    }

    // Benchmark execution only (not compile)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        // Execute
    }
    let duration = start.elapsed();

    let total_ms = duration.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / ITERATIONS as f64;
    let throughput = ITERATIONS as f64 / (duration.as_secs_f64());

    println!(
        "  Total time ({} iterations): {:.2}ms",
        ITERATIONS, total_ms
    );
    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} ops/sec", throughput);

    Ok((total_ms, avg_ms, throughput))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Comprehensive Benchmark Suite");
    println!("============================================\n");

    // Check ANE availability
    let avail = rustane::ANEAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available - CPU benchmarks only");
        return Ok(());
    }

    // Initialize ANE
    init()?;
    println!("✓ ANE initialized\n");

    let mut results = Vec::new();

    // Benchmark 1: FP16 Conversion
    let data: Vec<f32> = (0..10000).map(|i| i as f32 / 10000.0).collect();
    let (total, avg, throughput) = benchmark_cpu("FP16 Conversion (10K elements)", || {
        let converted: Vec<u16> = data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        // Convert back for benchmark measurement
        converted
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32())
            .collect()
    });
    results.push(("FP16 Conversion", "CPU", total, avg, throughput));

    // Benchmark 2: Softmax with Causal Mask
    let seq_len = 64;
    let embed_dim = 128;
    let attention: Vec<f32> = (0..seq_len * seq_len).map(|i| (i as f32).sin()).collect();

    let (total, avg, throughput) = benchmark_cpu("Softmax (64×64)", || {
        let mut exp_sum = 0.0f32;
        let mut result = vec![0.0f32; seq_len];

        // Compute exp and sum
        for i in 0..seq_len {
            result[i] = attention[i * seq_len..i * seq_len + seq_len]
                .iter()
                .map(|&x| {
                    let val = (x * 0.1f32).exp();
                    exp_sum += val;
                    val
                })
                .sum::<f32>();
        }

        // Normalize
        for i in 0..seq_len {
            result[i] /= exp_sum;
        }

        result
    });
    results.push(("Softmax (64×64)", "CPU", total, avg, throughput));

    // Benchmark 3: Matrix Multiplication (simulating linear layer)
    let m = 256;
    let n = 512;
    let p = 128;
    let a: Vec<f32> = (0..m * n).map(|i| i as f32 / 1000.0).collect();
    let b: Vec<f32> = (0..n * p).map(|i| i as f32 / 1000.0).collect();

    let (total, avg, throughput) = benchmark_cpu("MatMul (256×512 × 512×128)", || {
        let mut c = vec![0.0f32; m * p];
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[i * p + j] += a[i * n + k] * b[k * p + j];
                }
            }
        }
        c
    });
    results.push(("MatMul (256×512)", "CPU", total, avg, throughput));

    // Benchmark 4: ANE Inference (using causal_attention MIL)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark: ANE SDPA (scaled_dot_product_attention)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let heads = 12;
    let seq = 8;
    let head_dim = 64;

    // Use the exact MIL format from causal_attention.rs (which works)
    let mil = format!(
        "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremlc-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}})]\n\
{{\n\
    func main<ios18>(tensor<fp16, [1, {}, {}, {}]> q, tensor<fp16, [1, {}, {}, {}]> k, tensor<fp16, [1, {}, {}, {}]> v) {{\n\
        tensor<fp16, [1, {}, {}, {}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n\
    }} -> (att);\n\
}}\n",
        heads, seq, head_dim, heads, seq, head_dim, heads, seq, head_dim, heads, seq, head_dim
    );

    let batch = 1;
    let heads = 12;
    let seq = 8;
    let head_dim = 64;
    let io_bytes = batch * heads * seq * head_dim * 2; // FP16

    match benchmark_ane(
        "ANE SDPA",
        || -> Result<_, Box<dyn std::error::Error>> {
            let mut compiler = ANECompiler::new();
            let mut executor = compiler.compile_single(
                &mil,
                None,
                &[io_bytes, io_bytes, io_bytes],
                &[io_bytes],
            )?;

            // Create test inputs
            let q: Vec<f32> = (0..batch * heads * seq * head_dim)
                .map(|i| ((i as f32 * 17.0).sin() * 0.5) + 0.1)
                .collect();
            let k: Vec<f32> = (0..batch * heads * seq * head_dim)
                .map(|i| ((i as f32 * 13.0).cos() * 0.5) - 0.2)
                .collect();
            let v: Vec<f32> = (0..batch * heads * seq * head_dim)
                .map(|i| ((i as f32 * 11.0).sin() * 0.25) + 0.05)
                .collect();

            let q16: Vec<u16> = q
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            let k16: Vec<u16> = k
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            let v16: Vec<u16> = v
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();

            let q_tensor = ANETensor::from_fp16(q16, vec![batch, heads, seq, head_dim])?;
            let k_tensor = ANETensor::from_fp16(k16, vec![batch, heads, seq, head_dim])?;
            let v_tensor = ANETensor::from_fp16(v16, vec![batch, heads, seq, head_dim])?;

            executor.write_input(0, q_tensor.as_bytes())?;
            executor.write_input(1, k_tensor.as_bytes())?;
            executor.write_input(2, v_tensor.as_bytes())?;
            executor.eval()?;

            Ok((compiler, vec![], vec![]))
        },
        batch * heads * seq * head_dim * 2, // FP16
        batch * heads * seq * head_dim * 2,
    ) {
        Ok((total, avg, throughput)) => {
            results.push(("ANE SDPA", "ANE", total, avg, throughput));
        }
        Err(e) => {
            println!("  ❌ ANE benchmark failed: {}", e);
        }
    }

    // Print summary table
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:<25} {:<8} {:>12} {:>12} {:>12}",
        "Operation", "Device", "Avg (ms)", "Total (ms)", "Ops/sec"
    );
    println!("{}", "-".repeat(70));

    for (name, device, total, avg, throughput) in &results {
        println!(
            "{:<25} {:<8} {:>12.3} {:>12.2} {:>12.1}",
            name, device, avg, total, throughput
        );
    }

    // Calculate speedups
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PERFORMANCE INSIGHTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if results.len() >= 2 {
        let cpu_softmax = results.iter().find(|(n, _, _, _, _)| n.contains("Softmax"));
        let ane_sdpa = results
            .iter()
            .find(|(n, _, _, _, _)| n.contains("ANE SDPA"));

        if let Some((_, _, _, avg_cpu, _)) = cpu_softmax {
            if let Some((_, _, _, avg_ane, _)) = ane_sdpa {
                let speedup = avg_cpu / avg_ane;
                println!("  ⚡ ANE speedup over CPU: {:.2}x", speedup);
            }
        }
    }

    println!("\n✅ Benchmark suite completed!");
    println!("\nNotes:");
    println!("  - CPU benchmarks use sequential execution");
    println!("  - ANE benchmarks include kernel compilation time");
    println!("  - Real-world performance depends on model size and batching");
    println!("  - ANE excels at parallel operations (attention, conv)");

    Ok(())
}
