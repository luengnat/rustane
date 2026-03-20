//! Benchmarking utilities for measuring inference performance

use crate::wrapper::ANEExecutor;
use crate::Result;
use std::time::{Duration, Instant};

/// Results from a benchmark run
#[derive(Clone, Debug)]
pub struct BenchmarkResults {
    /// Total wall-clock time across all benchmark iterations.
    pub total_time: Duration,
    /// Average per-iteration runtime.
    pub avg_time: Duration,
    /// Effective throughput in samples per second.
    pub throughput_samples_per_sec: f64,
    /// Number of timed benchmark iterations.
    pub num_iterations: usize,
}

impl std::fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark Results:")?;
        writeln!(f, "  Iterations: {}", self.num_iterations)?;
        writeln!(f, "  Total time: {:?}", self.total_time)?;
        writeln!(f, "  Avg time: {:?}", self.avg_time)?;
        writeln!(
            f,
            "  Throughput: {:.2} samples/sec",
            self.throughput_samples_per_sec
        )
    }
}

/// Simple timer for measuring execution time
pub struct BenchmarkTimer {
    start: Option<Instant>,
    total_duration: Duration,
}

impl BenchmarkTimer {
    /// Create a new timer
    pub fn new() -> Self {
        Self {
            start: None,
            total_duration: Duration::ZERO,
        }
    }

    /// Start the timer
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop the timer and record elapsed time
    pub fn stop(&mut self) -> Duration {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed();
            self.total_duration += elapsed;
            elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = None;
        self.total_duration = Duration::ZERO;
    }

    /// Get the total recorded duration
    pub fn elapsed(&self) -> Duration {
        self.total_duration
    }
}

impl Default for BenchmarkTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark inference execution
///
/// Runs the executor multiple times and measures performance.
///
/// # Arguments
///
/// * `executor` - ANE executor
/// * `num_iterations` - Number of times to run inference
/// * `inference_fn` - Function that executes one inference step
///
/// # Returns
///
/// Benchmark results with timing and throughput information
///
/// # Example
///
/// ```no_run
/// # use rustane::utils::benchmark_inference;
/// # use rustane::wrapper::ANEExecutor;
/// # let mut executor = ANEExecutor::new(...);
/// let results = benchmark_inference(&mut executor, 10, |exec| {
///     exec.eval().unwrap()
/// }).unwrap();
/// println!("{}", results);
/// ```
pub fn benchmark_inference<F>(
    _executor: &mut ANEExecutor,
    num_iterations: usize,
    inference_fn: F,
) -> Result<BenchmarkResults>
where
    F: Fn(&mut ANEExecutor) -> Result<()>,
{
    let mut timer = BenchmarkTimer::new();
    let mut total_time = Duration::ZERO;

    // Warmup run
    inference_fn(_executor)?;

    // Benchmark runs
    for _ in 0..num_iterations {
        timer.start();
        inference_fn(_executor)?;
        let elapsed = timer.stop();
        total_time += elapsed;
    }

    let avg_time = total_time / num_iterations as u32;
    let throughput_samples_per_sec = if avg_time.as_secs_f64() > 0.0 {
        1.0 / avg_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchmarkResults {
        total_time,
        avg_time,
        throughput_samples_per_sec,
        num_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_timer() {
        let mut timer = BenchmarkTimer::new();

        timer.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed.as_millis() >= 10);
        assert!(timer.elapsed() >= Duration::from_millis(10));
    }

    #[test]
    fn test_benchmark_timer_reset() {
        let mut timer = BenchmarkTimer::new();

        timer.start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        timer.stop();

        timer.reset();
        assert_eq!(timer.elapsed(), Duration::ZERO);
        assert!(timer.start.is_none());
    }

    #[test]
    fn test_benchmark_results_display() {
        let results = BenchmarkResults {
            total_time: Duration::from_millis(100),
            avg_time: Duration::from_millis(10),
            throughput_samples_per_sec: 100.0,
            num_iterations: 10,
        };

        let display = format!("{}", results);
        assert!(display.contains("Benchmark Results:"));
        assert!(display.contains("100.00 samples/sec"));
    }
}
