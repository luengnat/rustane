//! Performance benchmarking for ANE backward pass

use std::time::Duration;

/// Benchmark results comparing ANE vs CPU backward pass
#[derive(Debug, Clone)]
pub struct BackwardBenchmark {
    pub config_name: String,
    pub param_count: usize,
    pub ane_time_ms: f64,
    pub cpu_time_ms: f64,
    pub speedup: f64,
    pub ane_used: bool,
}

impl BackwardBenchmark {
    pub fn new(config_name: &str, param_count: usize) -> Self {
        Self {
            config_name: config_name.to_string(),
            param_count,
            ane_time_ms: 0.0,
            cpu_time_ms: 0.0,
            speedup: 1.0,
            ane_used: false,
        }
    }
    
    pub fn record_times(&mut self, ane_time: Duration, cpu_time: Duration) {
        self.ane_time_ms = ane_time.as_secs_f64() * 1000.0;
        self.cpu_time_ms = cpu_time.as_secs_f64() * 1000.0;
        self.speedup = if ane_time.as_secs_f64() > 0.0 {
            cpu_time.as_secs_f64() / ane_time.as_secs_f64()
        } else {
            1.0
        };
        self.ane_used = true;
    }
    
    pub fn print(&self) {
        println!("{:<10} {:>8}M params | ANE: {:>8.2}ms | CPU: {:>8.2}ms | Speedup: {:>6.2}x {}",
            self.config_name,
            self.param_count / 1_000_000,
            self.ane_time_ms,
            self.cpu_time_ms,
            self.speedup,
            if self.speedup > 1.0 { "✓" } else { "○" }
        );
    }
}

/// Run benchmarks across multiple configurations
pub fn run_benchmarks() -> Vec<BackwardBenchmark> {
    use crate::data::Batch;
    use crate::training::{TransformerANE, TransformerConfig, Model, ANEGradientAccumulator};
    
    let configs = vec![
        ("Tiny", TransformerConfig::new(256, 32, 128, 2, 1, 32).unwrap()),
        ("Small", TransformerConfig::new(512, 64, 256, 4, 2, 64).unwrap()),
        ("Medium", TransformerConfig::new(1024, 128, 512, 8, 4, 128).unwrap()),
    ];
    
    let mut results = Vec::new();
    
    for (name, config) in configs {
        let mut benchmark = BackwardBenchmark::new(name, config.param_count());
        
        let mut model = match TransformerANE::new(&config) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to create model {}: {:?}", name, e);
                results.push(benchmark);
                continue;
            }
        };
        
        let batch_size = 4.min(128 / config.seq_len);
        let batch = match Batch::new(vec![0u32; batch_size * config.seq_len], batch_size, config.seq_len) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to create batch: {:?}", e);
                results.push(benchmark);
                continue;
            }
        };
        
        // Warmup
        let _ = model.forward(&batch);
        
        // ANE backward
        let mut ane_times = Vec::new();
        for _ in 0..5 {
            let _ = model.forward(&batch);
            let mut accum = match ANEGradientAccumulator::from_config(&config) {
                Ok(a) => a,
                Err(_) => break,
            };
            let start = std::time::Instant::now();
            let _ = model.backward_on_ane(&batch, 1.0, &mut accum);
            ane_times.push(start.elapsed());
        }
        
        // CPU backward
        let mut cpu_times = Vec::new();
        for _ in 0..5 {
            let _ = model.forward(&batch);
            let start = std::time::Instant::now();
            let _ = model.backward_with_batch(&batch, 1.0);
            cpu_times.push(start.elapsed());
        }
        
        let ane_avg = ane_times.iter().sum::<Duration>() / ane_times.len() as u32;
        let cpu_avg = cpu_times.iter().sum::<Duration>() / cpu_times.len() as u32;
        
        benchmark.record_times(ane_avg, cpu_avg);
        results.push(benchmark);
    }
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_creation() {
        let bench = BackwardBenchmark::new("Test", 1000000);
        assert_eq!(bench.config_name, "Test");
        assert_eq!(bench.param_count, 1000000);
        assert_eq!(bench.speedup, 1.0);
        assert!(!bench.ane_used);
    }
    
    #[test]
    fn test_benchmark_record_times() {
        let mut bench = BackwardBenchmark::new("Test", 1000000);
        bench.record_times(
            Duration::from_millis(50),
            Duration::from_millis(100),
        );
        assert!(bench.ane_used);
        assert!((bench.ane_time_ms - 50.0).abs() < 0.1);
        assert!((bench.cpu_time_ms - 100.0).abs() < 0.1);
        assert!((bench.speedup - 2.0).abs() < 0.1);
    }
}
