//! CPU vs ANE Training Benchmark with Loss Tracking
//!
//! Benchmarks transformer models through real training steps, measuring:
//! - Per-step timing (forward + backward + optimizer)
//! - Loss progression over steps (actual convergence)
//! - Gradient norms
//! - Steps/second throughput
//! - CPU vs ANE head speedup
//!
//! Run with: cargo run --example benchmark_cpu_vs_ane

use std::time::Instant;

use rustane::training::{
    AdamOptimizer, ConstantScheduler, CrossEntropyLoss, Model as TrainingModel, TrainerBuilder,
    TransformerANE, TransformerConfig,
};
use rustane::{Batch, Result};

/// A single benchmark run configuration
struct BenchConfig {
    name: String,
    vocab_size: usize,
    dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_layers: usize,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    lr: f32,
    use_ane_head: bool,
}

impl BenchConfig {
    fn to_transformer_config(&self) -> TransformerConfig {
        TransformerConfig::new(
            self.vocab_size,
            self.dim,
            self.hidden_dim,
            self.n_heads,
            self.n_layers,
            self.seq_len,
        )
        .unwrap()
    }
}

/// Results from a single benchmark run
struct BenchResult {
    config_name: String,
    param_count: usize,
    total_time_ms: f64,
    avg_step_ms: f64,
    min_step_ms: f64,
    max_step_ms: f64,
    steps_per_sec: f64,
    initial_loss: f32,
    final_loss: f32,
    loss_decrease_pct: f32,
    ane_enabled: bool,
    steps_completed: usize,
    timed_out: bool,
}

fn run_benchmark(config: &BenchConfig) -> Result<BenchResult> {
    let tf_config = config.to_transformer_config();

    println!("{}", "=".repeat(64));
    println!("  Benchmark: {}", config.name);
    println!(
        "  Architecture: vocab={}, dim={}, hidden={}, heads={}, layers={}",
        config.vocab_size, config.dim, config.hidden_dim, config.n_heads, config.n_layers
    );
    println!(
        "  Training: seq={}, batch={}, steps={}, lr={}",
        config.seq_len, config.batch_size, config.steps, config.lr
    );
    println!(
        "  Forward: {}",
        if config.use_ane_head {
            "ANE accelerated"
        } else {
            "CPU only"
        }
    );

    // Create model
    let mut model = TransformerANE::new(&tf_config)?;
    let param_count = model.param_count();
    let params_str = if param_count > 1_000_000 {
        format!("{:.2}M", param_count as f64 / 1_000_000.0)
    } else {
        format!("{:.1}K", param_count as f64 / 1000.0)
    };
    println!("  Parameters: {}", params_str);

    if config.use_ane_head {
        model.enable_ane_head(true);
    }

    // Create optimizer and scheduler
    let optimizer = AdamOptimizer::new(param_count);
    let scheduler = ConstantScheduler::new(config.lr);

    // Build trainer
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(optimizer)
        .with_scheduler(scheduler)
        .with_loss_fn(CrossEntropyLoss::new())
        .with_grad_clip_norm(1.0)
        .build()?;

    println!(
        "\n  {:>4} | {:>9} | {:>9} | {:>12}",
        "Step", "Loss", "Grad Norm", "Step (ms)"
    );
    println!("  {}", "-".repeat(44));

    let mut initial_loss: f32 = 0.0;
    let mut final_loss: f32 = 0.0;
    let mut step_times = Vec::with_capacity(config.steps);
    let mut timed_out = false;

    let total_start = Instant::now();

    for step in 0..config.steps {
        // Generate synthetic batch: deterministic token sequence
        let tokens: Vec<u32> = (0..config.batch_size * config.seq_len)
            .map(|i| ((i * 7 + step * 13) % config.vocab_size) as u32)
            .collect();
        let batch = Batch::new(tokens, config.batch_size, config.seq_len)?;

        let step_start = Instant::now();
        match trainer.train_step(&batch) {
            Ok(metrics) => {
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                step_times.push(step_ms);

                if step == 0 {
                    initial_loss = metrics.loss;
                }
                final_loss = metrics.loss;

                // Print every step for small runs, every Nth for large
                if config.steps <= 30
                    || step % (config.steps / 10).max(1) == 0
                    || step == config.steps - 1
                {
                    println!(
                        "  {:>4} | {:>9.5} | {:>9.5} | {:>10.2}",
                        step, metrics.loss, metrics.grad_norm, step_ms
                    );
                }
            }
            Err(_) => {
                println!("  {:>4} | ERROR - stopping early", step);
                timed_out = true;
                break;
            }
        }

        // Timeout: if a single step takes > 60 seconds, stop (ANE first step compiles)
        if total_start.elapsed().as_secs() > 60 * (step + 1) as u64 {
            println!("  (timeout - stopping early)");
            timed_out = true;
            break;
        }
    }

    let steps_completed = step_times.len();
    let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let avg_step_ms = if steps_completed > 0 {
        step_times.iter().sum::<f64>() / steps_completed as f64
    } else {
        0.0
    };
    let min_step_ms = if steps_completed > 0 {
        step_times.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        0.0
    };
    let max_step_ms = if steps_completed > 0 {
        step_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    } else {
        0.0
    };
    let steps_per_sec = if avg_step_ms > 0.0 {
        1000.0 / avg_step_ms
    } else {
        0.0
    };

    let loss_decrease_pct = if initial_loss > 0.0 && final_loss < initial_loss {
        ((initial_loss - final_loss) / initial_loss) * 100.0
    } else {
        0.0
    };

    println!(
        "\n  Result: {:.1}ms total | {:.1}ms/step | {:.1} steps/sec | {}/{} steps",
        total_time_ms, avg_step_ms, steps_per_sec, steps_completed, config.steps
    );
    println!(
        "  Loss: {:.5} -> {:.5} ({:.1}% decrease)\n",
        initial_loss, final_loss, loss_decrease_pct
    );

    Ok(BenchResult {
        config_name: config.name.clone(),
        param_count,
        total_time_ms,
        avg_step_ms,
        min_step_ms,
        max_step_ms,
        steps_per_sec,
        initial_loss,
        final_loss,
        loss_decrease_pct,
        ane_enabled: config.use_ane_head,
        steps_completed,
        timed_out,
    })
}

fn print_comparison(results: &[BenchResult]) {
    println!("{}", "=".repeat(86));
    println!("  COMPARISON TABLE");
    println!("{}", "=".repeat(86));
    println!(
        "{:<24} {:>6} {:>5} {:>4} {:>9} {:>9} {:>10}",
        "Config", "Params", "Mode", "Done", "Avg(ms)", "Steps/s", "Loss Drop%"
    );
    println!("{}", "-".repeat(86));

    for r in results {
        let params_str = if r.param_count > 1_000_000 {
            format!("{:.1}M", r.param_count as f64 / 1_000_000.0)
        } else {
            format!("{:.0}K", r.param_count as f64 / 1_000.0)
        };
        let done_str = if r.timed_out {
            format!("{}/{}!", r.steps_completed, r.steps_completed)
        } else {
            format!("{}/{}", r.steps_completed, r.steps_completed)
        };
        println!(
            "{:<24} {:>6} {:>5} {:>4} {:>9.1} {:>9.1} {:>9.1}%",
            r.config_name,
            params_str,
            if r.ane_enabled { "ANE" } else { "CPU" },
            done_str,
            r.avg_step_ms,
            r.steps_per_sec,
            r.loss_decrease_pct
        );
    }
    println!("{}", "=".repeat(86));
}

fn print_speedup(results: &[BenchResult]) {
    println!("\n  CPU vs ANE Speedup:");
    println!("  {}", "-".repeat(50));
    println!(
        "  {:<24} {:>12} {:>12} {:>10}",
        "Model", "CPU (ms)", "ANE (ms)", "Speedup"
    );
    println!("  {}", "-".repeat(50));

    let pairs: Vec<(&str, &str)> = vec![("Small-CPU", "Small-ANE"), ("Medium-CPU", "Medium-ANE")];

    for (cpu_name, ane_name) in pairs {
        let cpu = results.iter().find(|r| r.config_name == cpu_name);
        let ane = results.iter().find(|r| r.config_name == ane_name);
        if let (Some(c), Some(a)) = (cpu, ane) {
            if a.avg_step_ms == 0.0 || a.timed_out {
                println!(
                    "  {:<24} {:>10.1}ms {:>12} {:>10}",
                    cpu_name.replace("-CPU", ""),
                    c.avg_step_ms,
                    "timed out",
                    "N/A"
                );
            } else {
                let speedup = c.avg_step_ms / a.avg_step_ms;
                let status = if speedup > 1.0 {
                    format!("{:.2}x faster", speedup)
                } else {
                    format!("{:.2}x slower", 1.0 / speedup)
                };
                println!(
                    "  {:<24} {:>10.1}ms {:>10.1}ms {:>10} {}",
                    cpu_name.replace("-CPU", ""),
                    c.avg_step_ms,
                    a.avg_step_ms,
                    format!("{:.2}x", speedup),
                    status
                );
            }
        }
    }
}

fn main() -> Result<()> {
    println!();
    println!("  ============================================================");
    println!("  Rustane: CPU vs ANE Training Benchmark with Loss Tracking");
    println!("  Real forward/backward/optimizer steps on TransformerANE");
    println!("  ============================================================");
    println!();

    let mut results = Vec::new();

    // === CPU Benchmarks ===

    results.push(run_benchmark(&BenchConfig {
        name: "Tiny-CPU".into(),
        vocab_size: 256,
        dim: 32,
        hidden_dim: 128,
        n_heads: 2,
        n_layers: 1,
        seq_len: 16,
        batch_size: 4,
        steps: 50,
        lr: 0.001,
        use_ane_head: false,
    })?);

    results.push(run_benchmark(&BenchConfig {
        name: "Small-CPU".into(),
        vocab_size: 512,
        dim: 64,
        hidden_dim: 256,
        n_heads: 4,
        n_layers: 2,
        seq_len: 32,
        batch_size: 4,
        steps: 30,
        lr: 0.001,
        use_ane_head: false,
    })?);

    results.push(run_benchmark(&BenchConfig {
        name: "Medium-CPU".into(),
        vocab_size: 1024,
        dim: 128,
        hidden_dim: 512,
        n_heads: 8,
        n_layers: 4,
        seq_len: 64,
        batch_size: 2,
        steps: 20,
        lr: 0.0005,
        use_ane_head: false,
    })?);

    // === ANE Benchmarks ===
    // Note: ANE forward uses compile-per-step (no cache), so it's slower
    // for small models. The ANE compilation overhead dominates for tiny workloads.

    results.push(run_benchmark(&BenchConfig {
        name: "Small-ANE".into(),
        vocab_size: 512,
        dim: 64,
        hidden_dim: 256,
        n_heads: 4,
        n_layers: 2,
        seq_len: 32,
        batch_size: 4,
        steps: 10,
        lr: 0.001,
        use_ane_head: true,
    })?);

    results.push(run_benchmark(&BenchConfig {
        name: "Medium-ANE".into(),
        vocab_size: 1024,
        dim: 128,
        hidden_dim: 512,
        n_heads: 8,
        n_layers: 4,
        seq_len: 64,
        batch_size: 2,
        steps: 5,
        lr: 0.0005,
        use_ane_head: true,
    })?);

    // Print results
    print_comparison(&results);
    print_speedup(&results);

    println!("\n  NOTES:");
    println!("  - ANE compile cache: compiles once per unique shape, reuses within batch");
    println!("  - QKV and attn_out projections run on ANE (2 of 4 matmul types work)");
    println!("  - ANE final_norm (RMSNorm) and logits head skip ANE (always fail) → CPU");
    println!("  - dual_linear (w1/w3) and w2 projections fail on ANE → CPU fallback");
    println!("  - Cache cleared each step (weights change after optimizer)");
    println!("  - Backward pass is always CPU (not yet ANE-accelerated)");
    println!();

    Ok(())
}
