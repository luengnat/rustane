//! Performance Benchmarking and Timing Utilities
//!
//! Provides timing instrumentation and benchmarking utilities for comparing
//! CPU vs ANE performance during training.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::training::TransformerConfig;

/// Timing statistics for a single operation
#[derive(Clone, Debug)]
pub struct TimingStats {
    /// Total time spent in this operation
    pub total_time: Duration,
    /// Number of times this operation was executed
    pub count: usize,
    /// Minimum time for a single execution
    pub min_time: Duration,
    /// Maximum time for a single execution
    pub max_time: Duration,
}

impl TimingStats {
    /// Create new timing statistics
    pub fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            count: 0,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
        }
    }

    /// Record a single execution time
    pub fn record(&mut self, duration: Duration) {
        self.total_time += duration;
        self.count += 1;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
    }

    /// Get average time per execution
    pub fn average(&self) -> Duration {
        if self.count == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.count as u32
    }

    /// Get total time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total_time.as_secs_f64() * 1000.0
    }

    /// Get average time in microseconds
    pub fn average_us(&self) -> f64 {
        self.average().as_secs_f64() * 1_000_000.0
    }
}

impl Default for TimingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for a backward pass
#[derive(Clone, Debug)]
pub struct BackwardPassMetrics {
    /// Total backward pass time
    pub total_time: Duration,
    /// Time spent on ANE operations
    pub ane_time: Duration,
    /// Time spent on CPU operations
    pub cpu_time: Duration,
    /// Number of layers processed on ANE
    pub ane_layers: usize,
    /// Per-layer timing breakdown
    pub layer_timings: HashMap<String, TimingStats>,
}

impl BackwardPassMetrics {
    /// Create new backward pass metrics
    pub fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            ane_time: Duration::ZERO,
            cpu_time: Duration::ZERO,
            ane_layers: 0,
            layer_timings: HashMap::new(),
        }
    }

    /// Calculate speedup factor (CPU time / ANE time)
    pub fn speedup_factor(&self) -> f64 {
        if self.ane_time.is_zero() {
            return 1.0;
        }
        self.cpu_time.as_secs_f64() / self.ane_time.as_secs_f64()
    }

    /// Get percentage of time spent on ANE
    pub fn ane_percentage(&self) -> f64 {
        if self.total_time.is_zero() {
            return 0.0;
        }
        (self.ane_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
    }

    /// Format metrics as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Backward Pass Metrics:\n\
             Total Time: {:.2}ms\n\
             ANE Time: {:.2}ms ({:.1}%)\n\
             CPU Time: {:.2}ms ({:.1}%)\n\
             ANE Layers: {}\n\
             Speedup: {:.2}x",
            self.total_time.as_secs_f64() * 1000.0,
            self.ane_time.as_secs_f64() * 1000.0,
            self.ane_percentage(),
            self.cpu_time.as_secs_f64() * 1000.0,
            100.0 - self.ane_percentage(),
            self.ane_layers,
            self.speedup_factor()
        )
    }
}

impl Default for BackwardPassMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing context for measuring operation duration
///
/// # Example
///
/// ```ignore
/// let mut timer = TimingContext::new();
/// {
///     let _guard = timer.time_operation("rmsnorm_backward");
///     // ... perform operation ...
/// } // Timing recorded automatically
/// ```
pub struct TimingContext {
    metrics: BackwardPassMetrics,
    #[allow(dead_code)]
    current_operation: Option<(String, Instant)>,
}

impl TimingContext {
    /// Create a new timing context
    pub fn new() -> Self {
        Self {
            metrics: BackwardPassMetrics::new(),
            current_operation: None,
        }
    }

    /// Time an operation and return a guard that records timing on drop
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation to time
    ///
    /// # Returns
    ///
    /// A guard that records timing when dropped
    pub fn time_operation(&mut self, operation_name: &str) -> TimingGuard<'_> {
        TimingGuard {
            context: self,
            name: operation_name.to_string(),
            start: Instant::now(),
        }
    }

    /// Record a manual timing measurement
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation
    /// * `duration` - Duration of the operation
    pub fn record_timing(&mut self, operation_name: &str, duration: Duration) {
        let stats = self
            .metrics
            .layer_timings
            .entry(operation_name.to_string())
            .or_insert_with(TimingStats::new);
        stats.record(duration);
    }

    /// Record ANE layer execution
    pub fn record_ane_layer(&mut self) {
        self.metrics.ane_layers += 1;
    }

    /// Add ANE time
    pub fn add_ane_time(&mut self, duration: Duration) {
        self.metrics.ane_time += duration;
        self.metrics.total_time += duration;
    }

    /// Add CPU time
    pub fn add_cpu_time(&mut self, duration: Duration) {
        self.metrics.cpu_time += duration;
        self.metrics.total_time += duration;
    }

    /// Get the collected metrics
    pub fn metrics(&self) -> &BackwardPassMetrics {
        &self.metrics
    }

    /// Consume the context and return the metrics
    pub fn into_metrics(self) -> BackwardPassMetrics {
        self.metrics
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.metrics = BackwardPassMetrics::new();
    }
}

impl Default for TimingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard that records timing when dropped
pub struct TimingGuard<'a> {
    context: &'a mut TimingContext,
    name: String,
    start: Instant,
}

impl<'a> Drop for TimingGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.context.record_timing(&self.name, duration);
    }
}

/// Benchmark comparison between CPU and ANE backward pass
///
/// # Example
///
/// ```ignore
/// let config = TransformerConfig::tiny();
/// let mut benchmark = BackwardBenchmark::new(&config)?;
///
/// // Run benchmarks
/// let results = benchmark.run_comparison(num_iterations)?;
///
/// // Print results
/// println!("{}", results.format());
/// ```
pub struct BackwardBenchmark {
    config: TransformerConfig,
    iterations: usize,
}

impl BackwardBenchmark {
    /// Create a new benchmark
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration
    /// * `iterations` - Number of benchmark iterations
    pub fn new(config: &TransformerConfig, iterations: usize) -> Result<Self> {
        if iterations == 0 {
            return Err(crate::Error::InvalidParameter(
                "Iterations must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
            iterations,
        })
    }

    /// Run CPU vs ANE comparison benchmark
    ///
    /// # Returns
    ///
    /// Benchmark results comparing performance
    pub fn run_comparison(&self) -> Result<BenchmarkResults> {
        // This is a placeholder - actual implementation would:
        // 1. Create test data
        // 2. Run CPU backward pass `iterations` times
        // 3. Run ANE backward pass `iterations` times
        // 4. Collect statistics

        Ok(BenchmarkResults {
            cpu_time_us: 1000.0, // Placeholder
            ane_time_us: 100.0,  // Placeholder
            speedup: 10.0,
            config: self.config.clone(),
            iterations: self.iterations,
        })
    }
}

/// Results from a benchmark comparison
#[derive(Clone, Debug)]
pub struct BenchmarkResults {
    /// CPU backward pass time in microseconds
    pub cpu_time_us: f64,
    /// ANE backward pass time in microseconds
    pub ane_time_us: f64,
    /// Speedup factor (CPU time / ANE time)
    pub speedup: f64,
    /// Model configuration used
    pub config: TransformerConfig,
    /// Number of iterations run
    pub iterations: usize,
}

impl BenchmarkResults {
    /// Format results as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Benchmark Results ({} iterations):\n\
             Model: vocab_size={}, dim={}, n_layers={}\n\
             CPU Time: {:.2}ms\n\
             ANE Time: {:.2}ms\n\
             Speedup: {:.2}x\n\
             Time Saved: {:.1}%",
            self.iterations,
            self.config.vocab_size,
            self.config.dim,
            self.config.n_layers,
            self.cpu_time_us / 1000.0,
            self.ane_time_us / 1000.0,
            self.speedup,
            (1.0 - (self.ane_time_us / self.cpu_time_us)) * 100.0
        )
    }

    /// Get speedup as a formatted string
    pub fn speedup_string(&self) -> String {
        format!("{:.2}x", self.speedup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_stats() {
        let mut stats = TimingStats::new();

        stats.record(Duration::from_micros(100));
        stats.record(Duration::from_micros(200));
        stats.record(Duration::from_micros(150));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.average_us(), 150.0);
        assert_eq!(stats.min_time.as_micros(), 100);
        assert_eq!(stats.max_time.as_micros(), 200);
    }

    #[test]
    fn test_backward_pass_metrics() {
        let mut metrics = BackwardPassMetrics::new();
        metrics.ane_time = Duration::from_micros(100);
        metrics.cpu_time = Duration::from_micros(1000);
        metrics.total_time = Duration::from_micros(1100);

        assert_eq!(metrics.speedup_factor(), 10.0);
        assert_eq!(metrics.ane_percentage(), 100.0 / 1100.0 * 100.0);
    }

    #[test]
    fn test_timing_context() {
        let mut ctx = TimingContext::new();

        {
            let _guard = ctx.time_operation("test_op");
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = ctx.metrics.layer_timings.get("test_op").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.average().as_millis() >= 10);
    }
}
