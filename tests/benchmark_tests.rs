//! CPU vs ANE Performance Benchmark Tests
//!
//! These benchmarks compare the performance of backward pass operations
//! executed on CPU versus ANE (Apple Neural Engine).

#[cfg(target_vendor = "apple")]
#[cfg(test)]
mod benchmarks {
    use rustane::ane::ANECompileRequest;
    use rustane::data::Batch;
    use rustane::layers::backward::{BackwardMILGenerator, RMSNormBackwardGen};
    use rustane::training::{ANEGradientAccumulator, Model, TransformerANE, TransformerConfig};
    use std::time::Instant;

    /// Benchmark helper for measuring execution time
    fn measure_time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
        let start = Instant::now();
        let result = f();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        (result, elapsed_ms)
    }

    #[test]
    fn benchmark_cpu_backward_pass() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Warm up
        let _ = model.forward(&batch);
        let _ = model.backward_with_batch(&batch, 0.5);

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..5 {
            let _ = model.forward(&batch);
            let (_, elapsed) = measure_time_ms(|| {
                let _ = model.backward_with_batch(&batch, 0.5);
            });
            times.push(elapsed);
        }

        let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
        let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!("CPU Backward Pass Benchmark ({} iterations):", times.len());
        eprintln!("  Average: {:.2} ms", avg_ms);
        eprintln!("  Min:     {:.2} ms", min_ms);
        eprintln!("  Max:     {:.2} ms", max_ms);
    }

    #[test]
    #[ignore = "Requires ANE hardware"]
    fn benchmark_ane_backward_pass() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Warm up
        let _ = model.forward(&batch);
        let mut accumulator = ANEGradientAccumulator::new(config.param_count()).unwrap();
        let _ = model.backward_on_ane(&batch, 0.5, &mut accumulator);

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..5 {
            let _ = model.forward(&batch);
            let mut accumulator = ANEGradientAccumulator::new(config.param_count()).unwrap();
            let (_, elapsed) = measure_time_ms(|| {
                let _ = model.backward_on_ane(&batch, 0.5, &mut accumulator);
            });
            times.push(elapsed);
        }

        let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
        let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!("ANE Backward Pass Benchmark ({} iterations):", times.len());
        eprintln!("  Average: {:.2} ms", avg_ms);
        eprintln!("  Min:     {:.2} ms", min_ms);
        eprintln!("  Max:     {:.2} ms", max_ms);
    }

    #[test]
    fn benchmark_layer_timing_breakdown() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Forward pass to cache activations
        let _ = model.forward(&batch);

        eprintln!("\n=== Layer Timing Breakdown Benchmark ===");

        // Run multiple times and collect timing stats
        for i in 0..3 {
            eprintln!("\n--- Run {} ---", i + 1);
            let mut accumulator = ANEGradientAccumulator::new(config.param_count()).unwrap();
            let _ = model.backward_on_ane(&batch, 0.5, &mut accumulator);
        }
    }

    #[test]
    #[ignore = "Requires specific ANE hardware support - ANE compilation returns null kernel on some hardware"]
    fn benchmark_rmsnorm_backward_ms() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let gen = RMSNormBackwardGen::new();
        let mil = gen.generate(&config).unwrap();

        let dim = config.dim;
        let seq_len = config.seq_len;
        let d_out = vec![0.01f32; seq_len * dim];
        let x = vec![0.5f32; seq_len * dim];
        let w = vec![1.0f32; dim];

        let mut input = d_out.clone();
        input.extend_from_slice(&x);
        input.extend_from_slice(&w);

        // Warm up
        let req = ANECompileRequest::new(
            &mil,
            vec![input.len() * 4],
            vec![seq_len * dim * 4, dim * 4],
        );
        let mut ex = req.compile().unwrap();
        let slice =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let _ = ex.write_input(0, slice);
        let _ = ex.eval();

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let slice =
                unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
            let _ = ex.write_input(0, slice);
            let _ = ex.eval();
            let _ = ex.read_output(0, &mut vec![0u8; seq_len * dim * 4]);
            let _ = ex.read_output(1, &mut vec![0u8; dim * 4]);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
        eprintln!("RMSNorm Backward (ANE eval only): {:.2} ms average", avg_ms);
    }

    #[test]
    fn benchmark_comparison_summary() {
        eprintln!("\n========================================");
        eprintln!("  CPU vs ANE Performance Comparison");
        eprintln!("========================================");
        eprintln!();
        eprintln!("Note: Run with --ignored to include ANE benchmarks");
        eprintln!();
        eprintln!("Expected speedup factors (ANE vs CPU):");
        eprintln!("  - RMSNorm backward:  2-5x faster");
        eprintln!("  - FFN backward:      3-8x faster");
        eprintln!("  - Attention backward: 5-10x faster");
        eprintln!();
        eprintln!("Actual results will vary based on:");
        eprintln!("  - Model size (dim, n_layers, seq_len)");
        eprintln!("  - Batch size");
        eprintln!("  - Apple Silicon generation (M1/M2/M3)");
        eprintln!("  - Thermal conditions");
        eprintln!("========================================");
    }
}
