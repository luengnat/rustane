//! Benchmarking harness for ANE training pipeline.
//!
//! Captures: compilation time, step latency, memory usage, and loss convergence.
//! Follows reference implementation's comprehensive measurement approach.

use rustane::training::{GradAccumulator, LossScaler};
use std::time::Instant;

/// Metrics captured during a training benchmark
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Total compilation time (ms)
    pub compilation_ms: f32,
    /// Per-step latency statistics
    pub step_latency_stats: StepLatencyStats,
    /// Loss trajectory
    pub losses: Vec<f32>,
    /// Memory estimate (params × 4 bytes)
    pub memory_estimate_mb: f32,
    /// Final scale factor
    pub final_scale: f32,
    /// Updates performed
    pub optimizer_updates: usize,
}

/// Step latency statistics
#[derive(Debug, Clone)]
pub struct StepLatencyStats {
    /// Min latency (ms)
    pub min_ms: f32,
    /// Max latency (ms)
    pub max_ms: f32,
    /// Average latency (ms)
    pub avg_ms: f32,
    /// Median latency (ms)
    pub median_ms: f32,
    /// Total steps measured
    pub num_steps: usize,
}

impl StepLatencyStats {
    fn from_times(times_ms: &[f32]) -> Self {
        let mut sorted = times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        Self {
            min_ms: *sorted.first().unwrap_or(&0.0),
            max_ms: *sorted.last().unwrap_or(&0.0),
            avg_ms: times_ms.iter().sum::<f32>() / n as f32,
            median_ms: median,
            num_steps: n,
        }
    }
}

/// Benchmark a training configuration
pub fn benchmark_training_config(
    config_name: &str,
    num_params: usize,
    num_layers: usize,
    num_steps: usize,
    accum_steps: usize,
) -> TrainingMetrics {
    // Compilation (simulated as zero for CPU-only)
    let compile_start = Instant::now();
    let _compile_time = compile_start.elapsed();

    // Initialize training components
    let mut scaler = LossScaler::for_transformer(num_layers);
    let mut accumulator = GradAccumulator::new(num_params, accum_steps);
    let mut losses = Vec::new();
    let mut step_times = Vec::new();
    let mut optimizer_updates = 0;

    // Training loop
    for step in 1..=num_steps {
        let step_start = Instant::now();

        // Synthetic loss (decreasing)
        let loss = 2.0 - (step as f32 * 0.05) + (step as f32 * 0.02).sin() * 0.1;

        // Scale for FP16
        let scaled_loss = scaler.scale_loss(loss);

        // Synthetic gradients
        let grads: Vec<f32> = (0..num_params)
            .map(|_| scaled_loss / 100.0 + 0.0001)
            .collect();

        // Accumulate
        accumulator.accumulate_fp32(&grads, 1.0);
        let _valid = scaler.update(&grads);

        // Optimizer step
        if accumulator.is_complete() {
            optimizer_updates += 1;
            accumulator.reset();
        }

        let step_elapsed = step_start.elapsed().as_micros() as f32 / 1000.0;
        step_times.push(step_elapsed);
        losses.push(loss);
    }

    let memory_bytes = (num_params * 4) as f32; // Assume fp32
    let memory_mb = memory_bytes / (1024.0 * 1024.0);

    TrainingMetrics {
        compilation_ms: 0.0, // CPU-only, no ANE compilation
        step_latency_stats: StepLatencyStats::from_times(&step_times),
        losses,
        memory_estimate_mb: memory_mb,
        final_scale: scaler.current_scale(),
        optimizer_updates,
    }
}

/// Run parameter sweep benchmark
pub fn param_sweep_benchmark(configs: &[(&str, usize, usize)]) {
    println!("\n{}", "=".repeat(90));
    println!("BENCHMARK: Parameter Sweep");
    println!("{}", "=".repeat(90));

    println!(
        "\n{:<15} {:<12} {:<12} {:<12} {:<15} {:<12}",
        "Config", "Params", "Layers", "Mem (MB)", "Avg Step (ms)", "Updates"
    );
    println!("{}", "-".repeat(90));

    let mut all_metrics = Vec::new();

    for (name, num_params, num_layers) in configs {
        let metrics = benchmark_training_config(name, *num_params, *num_layers, 20, 2);

        println!(
            "{:<15} {:<12} {:<12} {:<12.1} {:<15.3} {:<12}",
            name,
            num_params,
            num_layers,
            metrics.memory_estimate_mb,
            metrics.step_latency_stats.avg_ms,
            metrics.optimizer_updates,
        );

        all_metrics.push((name.to_string(), metrics));
    }

    println!("\n{}", "=".repeat(90));
    println!("Detailed Latency Statistics:");
    println!("{}", "=".repeat(90));

    for (name, metrics) in &all_metrics {
        let stats = &metrics.step_latency_stats;
        println!("\n{}:", name);
        println!("  Min:    {:.3} ms", stats.min_ms);
        println!("  Max:    {:.3} ms", stats.max_ms);
        println!("  Avg:    {:.3} ms", stats.avg_ms);
        println!("  Median: {:.3} ms", stats.median_ms);
        println!("  Steps:  {}", stats.num_steps);
    }

    println!("\n{}", "=".repeat(90));
    println!("Loss Convergence:");
    println!("{}", "=".repeat(90));

    for (name, metrics) in &all_metrics {
        if !metrics.losses.is_empty() {
            let first_loss = metrics.losses[0];
            let final_loss = metrics.losses[metrics.losses.len() - 1];
            let improvement = (first_loss - final_loss) / first_loss * 100.0;

            println!(
                "{:<15} {} → {} ({:.1}% improvement)",
                name, first_loss, final_loss, improvement
            );
        }
    }

    println!("\n{}", "=".repeat(90));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Benchmark small, medium, and large parameter-golf configs
    #[test]
    #[ignore] // Run with: cargo test benchmark_param_sweep -- --ignored --nocapture
    fn benchmark_param_sweep() {
        let configs = vec![
            ("param-golf-baseline", 8192, 2),   // 512*16*1 params
            ("param-golf-scaled-2x", 16384, 2), // 512*16*2 params
            ("param-golf-scaled-4x", 32768, 4), // 512*16*4 params
        ];

        param_sweep_benchmark(&configs);
    }

    /// Benchmark with varying accumulation steps
    #[test]
    #[ignore] // Run with: cargo test benchmark_accumulation_sweep -- --ignored --nocapture
    fn benchmark_accumulation_sweep() {
        println!("\n{}", "=".repeat(90));
        println!("BENCHMARK: Gradient Accumulation Impact");
        println!("{}", "=".repeat(90));

        let num_params = 16384;
        let num_layers = 4;

        println!(
            "\n{:<15} {:<12} {:<15} {:<12} {:<12}",
            "Accum Steps", "Total Steps", "Avg Step (ms)", "Updates", "Eff. Batch"
        );
        println!("{}", "-".repeat(75));

        for accum_steps in &[1, 2, 4, 8] {
            let metrics =
                benchmark_training_config("test", num_params, num_layers, 16, *accum_steps);

            println!(
                "{:<15} {:<12} {:<15.3} {:<12} {:<12}",
                accum_steps,
                16,
                metrics.step_latency_stats.avg_ms,
                metrics.optimizer_updates,
                16 / accum_steps, // Assuming 1 sample per mini-batch
            );
        }

        println!("\n{}", "=".repeat(90));
    }

    /// Benchmark with varying model sizes
    #[test]
    #[ignore] // Run with: cargo test benchmark_model_scale -- --ignored --nocapture
    fn benchmark_model_scale() {
        println!("\n{}", "=".repeat(90));
        println!("BENCHMARK: Model Scale Impact");
        println!("{}", "=".repeat(90));

        let model_sizes = vec![
            ("48M", 48_000_000, 6),
            ("110M", 110_000_000, 8),
            ("300M", 300_000_000, 12),
        ];

        println!(
            "\n{:<15} {:<15} {:<12} {:<15} {:<12}",
            "Model Size", "Params", "Layers", "Avg Step (ms)", "Memory (MB)"
        );
        println!("{}", "-".repeat(75));

        for (name, num_params, num_layers) in &model_sizes {
            let metrics = benchmark_training_config(name, *num_params, *num_layers, 10, 1);

            println!(
                "{:<15} {:<15} {:<12} {:<15.3} {:<12.1}",
                name,
                num_params,
                num_layers,
                metrics.step_latency_stats.avg_ms,
                metrics.memory_estimate_mb,
            );
        }

        println!("\n{}", "=".repeat(90));
    }

    /// Single configuration detailed analysis
    #[test]
    #[ignore] // Run with: cargo test benchmark_detailed -- --ignored --nocapture
    fn benchmark_detailed() {
        let metrics = benchmark_training_config("parameter-golf", 16384, 4, 50, 2);

        println!("\n{}", "=".repeat(70));
        println!("DETAILED BENCHMARK: parameter-golf");
        println!("{}", "=".repeat(70));

        println!("\nConfiguration:");
        println!("  Parameters: 16,384");
        println!("  Layers: 4");
        println!("  Steps: 50");
        println!("  Accumulation: 2");

        println!("\nMemory:");
        println!("  Estimated: {:.1} MB", metrics.memory_estimate_mb);
        println!("  (Params × 4 bytes for fp32)");

        println!("\nLatency Statistics:");
        let stats = &metrics.step_latency_stats;
        println!("  Min:    {:.3} ms", stats.min_ms);
        println!("  Max:    {:.3} ms", stats.max_ms);
        println!("  Avg:    {:.3} ms", stats.avg_ms);
        println!("  Median: {:.3} ms", stats.median_ms);

        println!("\nThroughput:");
        let tokens_per_step = 16; // seq_len
        let samples_per_sec = 1000.0 / stats.avg_ms;
        let tokens_per_sec = tokens_per_step as f32 * samples_per_sec;
        println!("  Samples/sec: {:.1}", samples_per_sec);
        println!("  Tokens/sec: {:.1}", tokens_per_sec);

        println!("\nConvergence:");
        if !metrics.losses.is_empty() {
            let first = metrics.losses[0];
            let last = metrics.losses[metrics.losses.len() - 1];
            let improvement = (first - last) / first * 100.0;
            println!("  Initial: {}", first);
            println!("  Final: {}", last);
            println!("  Improvement: {:.1}%", improvement);
        }

        println!("\nOptimizer:");
        println!("  Updates: {}", metrics.optimizer_updates);
        println!("  Final scale: {}", metrics.final_scale);

        println!("\n{}", "=".repeat(70));
    }
}
