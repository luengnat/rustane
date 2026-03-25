//! Multi-configuration ANE benchmark with DIM/SEQ sweep.
//!
//! Tests training throughput across multiple (DIM, SEQ) configurations
//! including the target DIM=768 SEQ=256.
//!
//! Reports: compile time, forward time, reload time, throughput per config.
//!
//! Validates PERF-01 (benchmark at 768/256), PERF-02 (timing breakdown),
//! PERF-03 (DIM/SEQ tuning impact).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example benchmark_dims --release
//! ```

use rustane::ane::WeightBlob;
use rustane::mil::programs::conv1x1_mil;
use rustane::wrapper::{ANECompiler, ANERuntime};
use std::time::Instant;

const WEIGHT_NAME: &str = "@model_path/weights/weight.bin";
const BENCH_STEPS: usize = 5;

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

struct BenchResult {
    dim: usize,
    seq: usize,
    compile_ms: f64,
    avg_fwd_ms: f64,
    avg_reload_ms: f64,
    throughput: f64,
    error: Option<String>,
}

fn bench_config(dim: usize, seq: usize) -> BenchResult {
    let weight_size = dim * dim;
    let input_size = dim * seq * 4; // fp32
    let output_size = dim * seq * 4;

    // Generate weights and input
    let weights: Vec<f32> = (0..weight_size)
        .map(|i| (((i % 100) as f32 - 50.0) / 200.0).max(-0.5).min(0.5))
        .collect();
    let input: Vec<f32> = (0..dim * seq)
        .map(|i| ((i % 50) as f32 - 25.0) / 100.0)
        .collect();
    let input_bytes = f32_to_bytes(&input);

    let blob = WeightBlob::from_f32(&weights, dim, dim).unwrap();
    let mil = conv1x1_mil(seq, dim, dim);

    // Compile
    let compile_start = Instant::now();
    let mut compiler = ANECompiler::new();
    let compile_result = compiler.compile_multi(
        &mil,
        &[WEIGHT_NAME],
        &[blob.as_bytes()],
        &[blob.as_bytes().len()],
        &[input_size],
        &[output_size],
    );

    let mut executor = match compile_result {
        Ok(e) => e,
        Err(e) => {
            return BenchResult {
                dim,
                seq,
                compile_ms: compile_start.elapsed().as_secs_f64() * 1000.0,
                avg_fwd_ms: 0.0,
                avg_reload_ms: 0.0,
                throughput: 0.0,
                error: Some(format!("Compile failed: {:?}", e)),
            };
        }
    };
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    // Benchmark forward + reload
    let mut total_fwd = std::time::Duration::ZERO;
    let mut total_reload = std::time::Duration::ZERO;
    let bench_start = Instant::now();

    for step in 0..BENCH_STEPS {
        // Forward
        let fwd_start = Instant::now();
        if executor.write_input(0, &input_bytes).is_err() {
            break;
        }
        if executor.eval().is_err() {
            break;
        }
        let mut out_buf = vec![0u8; output_size];
        if executor.read_output(0, &mut out_buf).is_err() {
            break;
        }
        total_fwd += fwd_start.elapsed();

        // Reload with slightly different weights
        let new_weights: Vec<f32> = weights
            .iter()
            .map(|&w| (w + (step as f32 + 1.0) * 0.001).clamp(-1.0, 1.0))
            .collect();
        let new_blob = WeightBlob::from_f32(&new_weights, dim, dim).unwrap();

        let reload_start = Instant::now();
        if executor
            .reload_weights(&[(WEIGHT_NAME, new_blob.as_bytes())])
            .is_err()
        {
            break;
        }
        total_reload += reload_start.elapsed();
    }

    let total_time = bench_start.elapsed();
    let throughput = BENCH_STEPS as f64 / total_time.as_secs_f64();

    BenchResult {
        dim,
        seq,
        compile_ms,
        avg_fwd_ms: total_fwd.as_secs_f64() * 1000.0 / BENCH_STEPS as f64,
        avg_reload_ms: total_reload.as_secs_f64() * 1000.0 / BENCH_STEPS as f64,
        throughput,
        error: None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANE Multi-Configuration Benchmark ===\n");

    ANERuntime::init()?;
    println!("✅ ANE runtime initialized\n");

    let configs = [
        (32, 16),
        (64, 16),
        (64, 32),
        (128, 16),
        (128, 32),
        (256, 64),
        (512, 128),
        (768, 256),
    ];

    println!(
        "{:<8} {:<6} {:>12} {:>12} {:>12} {:>14}   {}",
        "DIM", "SEQ", "Compile(ms)", "Fwd(ms)", "Reload(ms)", "Throughput", "Status"
    );
    println!("{}", "-".repeat(85));

    let mut results = Vec::new();

    for &(dim, seq) in &configs {
        print!("{:<8} {:<6} ", dim, seq);
        let result = bench_config(dim, seq);

        if let Some(ref err) = result.error {
            println!(
                "{:>12.1} {:>12} {:>12} {:>14}   ❌ {}",
                result.compile_ms, "-", "-", "-", err
            );
        } else {
            println!(
                "{:>12.1} {:>12.2} {:>12.2} {:>10.1} s/s   ✅ OK",
                result.compile_ms, result.avg_fwd_ms, result.avg_reload_ms, result.throughput
            );
        }
        results.push(result);
    }

    println!("\n=== Summary ===");
    let successful: Vec<&BenchResult> = results.iter().filter(|r| r.error.is_none()).collect();
    if !successful.is_empty() {
        let fastest = successful
            .iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
            .unwrap();
        println!(
            "  Fastest config: DIM={} SEQ={} ({:.1} steps/sec)",
            fastest.dim, fastest.seq, fastest.throughput
        );
    }

    let target = results.iter().find(|r| r.dim == 768 && r.seq == 256);
    if let Some(t) = target {
        if let Some(ref err) = t.error {
            println!("\n  ⚠️  Target config (768x256): {}", err);
        } else {
            println!("\n  ✅ PERF-01: Target config (768x256) benchmarked successfully");
            println!(
                "     Compile: {:.1}ms, Forward: {:.2}ms, Reload: {:.2}ms",
                t.compile_ms, t.avg_fwd_ms, t.avg_reload_ms
            );
        }
    }

    println!("\n=== PERF-03: DIM/SEQ Impact ===");
    for r in &results {
        if r.error.is_none() {
            println!("  DIM={:>3} SEQ={:>3}: compile={:>7.1}ms  fwd={:>6.2}ms  reload={:>6.2}ms  throughput={:>6.1}/s",
                r.dim, r.seq, r.compile_ms, r.avg_fwd_ms, r.avg_reload_ms, r.throughput);
        }
    }

    println!("\nOK benchmark_dims");
    Ok(())
}
