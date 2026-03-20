//! Benchmark ANE backward pass performance vs CPU
//!
//! Compares timing of ANE-accelerated backward pass against CPU-only backward pass.

use rustane::data::Batch;
use rustane::training::{ANEGradientAccumulator, Model, TransformerANE, TransformerConfig};

fn main() {
    println!("=== ANE Backward Pass Benchmark ===\n");

    // Test configurations
    let configs = vec![
        (
            "Tiny",
            TransformerConfig::new(256, 32, 128, 2, 1, 32).unwrap(),
        ),
        (
            "Small",
            TransformerConfig::new(512, 64, 256, 4, 2, 64).unwrap(),
        ),
        (
            "Medium",
            TransformerConfig::new(1024, 128, 512, 8, 4, 128).unwrap(),
        ),
    ];

    for (name, config) in configs {
        println!(
            "Configuration: {} ({}M parameters)",
            name,
            config.param_count() / 1_000_000
        );
        println!(
            "  dim={}, seq_len={}, hidden_dim={}, n_heads={}, n_layers={}, vocab_size={}",
            config.dim,
            config.seq_len,
            config.hidden_dim,
            config.n_heads,
            config.n_layers,
            config.vocab_size
        );

        let mut model = TransformerANE::new(&config).unwrap();
        let batch_size = 4.min(128 / config.seq_len);
        let batch = Batch::new(
            vec![0u32; batch_size * config.seq_len],
            batch_size,
            config.seq_len,
        )
        .unwrap();

        // Warmup
        let _ = model.forward(&batch);

        // ANE backward (multiple runs)
        let mut ane_times = Vec::new();
        for _ in 0..5 {
            let _ = model.forward(&batch);
            let mut accum = ANEGradientAccumulator::from_config(&config).unwrap();
            let start = std::time::Instant::now();
            let _ = model.backward_on_ane(&batch, 1.0, &mut accum);
            ane_times.push(start.elapsed());
        }

        // CPU backward (multiple runs)
        let mut cpu_times = Vec::new();
        for _ in 0..5 {
            let _ = model.forward(&batch);
            let start = std::time::Instant::now();
            let _ = model.backward_with_batch(&batch, 1.0);
            cpu_times.push(start.elapsed());
        }

        let ane_avg = ane_times.iter().sum::<std::time::Duration>() / ane_times.len() as u32;
        let cpu_avg = cpu_times.iter().sum::<std::time::Duration>() / cpu_times.len() as u32;

        println!(
            "  ANE backward: {:.2}ms (avg of 5 runs)",
            ane_avg.as_secs_f64() * 1000.0
        );
        println!(
            "  CPU backward: {:.2}ms (avg of 5 runs)",
            cpu_avg.as_secs_f64() * 1000.0
        );

        if ane_avg < cpu_avg {
            let speedup = cpu_avg.as_secs_f64() / ane_avg.as_secs_f64();
            println!("  Speedup: {:.2}x faster with ANE", speedup);
        } else {
            let slowdown = ane_avg.as_secs_f64() / cpu_avg.as_secs_f64();
            println!(
                "  Note: ANE is {:.2}x slower (likely due to compilation overhead)",
                slowdown
            );
        }
        println!();
    }

    println!("=== Benchmark Complete ===");
    println!(
        "\nNote: First ANE run includes compilation time. Subsequent runs benefit from caching."
    );
    println!("ANE performance varies based on:");
    println!("  - Model size (larger models benefit more from ANE parallelism)");
    println!("  - Batch size (larger batches amortize compilation overhead)");
    println!("  - ANE availability (falls back to CPU if ANE unavailable)");
}
