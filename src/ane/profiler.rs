//! ANE Profiler - Kernel Timing and Performance Analysis
//!
//! This module provides:
//! 1. Per-kernel timing capture (compile, execute, data transfer)
//! 2. Performance analytics (utilization, bottlenecks, speedup)
//! 3. Automatic report generation
//! 4. Timeline visualization
//!
//! # Usage
//!
//! ```no_run
//! use rustane::ane::ANEProfiler;
//!
//! let mut profiler = ANEProfiler::new();
//!
//! // Profile a training step
//! profiler.start_step();
//! profiler.start_kernel("rmsnorm_layer_0");
//! // ... execute kernel
//! profiler.end_kernel("rmsnorm_layer_0", 1024, 512);
//! profiler.end_step();
//!
//! // Generate report
//! profiler.generate_report();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Timing data for a single kernel execution
#[derive(Debug, Clone)]
pub struct KernelTiming {
    /// Kernel name/identifier
    pub kernel_name: String,
    /// Compilation time (if compiled this step)
    pub compile_time: Option<Duration>,
    /// Data transfer time (CPU -> ANE)
    pub h2d_time: Duration,
    /// Kernel execution time on ANE
    pub exec_time: Duration,
    /// Data transfer time (ANE -> CPU)
    pub d2h_time: Duration,
    /// Total time
    pub total_time: Duration,
    /// Input size in bytes
    pub input_bytes: usize,
    /// Output size in bytes
    pub output_bytes: usize,
}

/// Aggregated statistics for a kernel type
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Kernel name
    pub kernel_name: String,
    /// Number of executions
    pub call_count: u64,
    /// Total execution time
    pub total_exec_time: Duration,
    /// Total data transfer time
    pub total_transfer_time: Duration,
    /// Total compile time (usually 0 after caching)
    pub total_compile_time: Duration,
    /// Average execution time
    pub avg_exec_time: Duration,
    /// Minimum execution time
    pub min_exec_time: Duration,
    /// Maximum execution time
    pub max_exec_time: Duration,
    /// Total bytes processed (input + output)
    pub total_bytes: usize,
    /// Throughput in GB/s
    pub throughput_gbps: f64,
    /// Percentage of total step time
    pub pct_of_total: f64,
}

/// Step-level profiling data
#[derive(Debug, Clone)]
pub struct StepProfile {
    /// Step number
    pub step: u64,
    /// Total step duration
    pub total_duration: Duration,
    /// Kernel timings in this step
    pub kernels: Vec<KernelTiming>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Profiling metrics summary
#[derive(Debug, Clone)]
pub struct ProfilerMetrics {
    /// Total steps profiled
    pub total_steps: u64,
    /// Average step time
    pub avg_step_time: Duration,
    /// Total kernels executed
    pub total_kernels: u64,
    /// Total compute time
    pub total_compute_time: Duration,
    /// Total data transfer time
    pub total_transfer_time: Duration,
    /// Total compile time
    pub total_compile_time: Duration,
    /// Average ANE utilization (exec_time / step_time)
    pub avg_ane_utilization: f64,
    /// Overall throughput in GB/s
    pub overall_throughput_gbps: f64,
}

/// ANE profiler for kernel timing and performance analysis
pub struct ANEProfiler {
    /// Current step number
    current_step: u64,
    /// Current step start time
    step_start: Option<Instant>,
    /// Current kernel start time
    kernel_start: Option<Instant>,
    /// Current kernel name
    current_kernel: Option<String>,
    /// Step start time for h2d transfer
    h2d_start: Option<Instant>,
    /// Completed step profiles
    step_profiles: Vec<StepProfile>,
    /// Aggregated kernel statistics
    kernel_stats: HashMap<String, Vec<KernelTiming>>,
    /// Profiler enabled
    enabled: bool,
    /// Maximum steps to retain (0 = unlimited)
    max_retained_steps: usize,
}

impl ANEProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            current_step: 0,
            step_start: None,
            kernel_start: None,
            current_kernel: None,
            h2d_start: None,
            step_profiles: Vec::new(),
            kernel_stats: HashMap::new(),
            enabled: true,
            max_retained_steps: 1000,
        }
    }

    /// Create a disabled profiler (zero overhead)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::new()
        }
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start profiling a training step
    pub fn start_step(&mut self) {
        if !self.enabled {
            return;
        }
        self.step_start = Some(Instant::now());
        self.current_step += 1;
    }

    /// End profiling a training step
    pub fn end_step(&mut self) -> StepProfile {
        if !self.enabled {
            return StepProfile {
                step: self.current_step,
                total_duration: Duration::ZERO,
                kernels: Vec::new(),
                timestamp: Instant::now(),
            };
        }

        let step_start = self.step_start.take().unwrap_or_else(Instant::now);
        let total_duration = step_start.elapsed();

        let profile = StepProfile {
            step: self.current_step,
            total_duration,
            kernels: Vec::new(),
            timestamp: step_start,
        };

        // Retain only last N steps if limit is set
        if self.max_retained_steps > 0 && self.step_profiles.len() >= self.max_retained_steps {
            self.step_profiles.remove(0);
        }

        self.step_profiles.push(profile.clone());
        profile
    }

    /// Start timing a kernel execution
    pub fn start_kernel(&mut self, kernel_name: &str) {
        if !self.enabled {
            return;
        }
        self.kernel_start = Some(Instant::now());
        self.current_kernel = Some(kernel_name.to_string());
    }

    /// Record host-to-device transfer start
    pub fn start_h2d(&mut self) {
        if !self.enabled {
            return;
        }
        self.h2d_start = Some(Instant::now());
    }

    /// Record host-to-device transfer end
    pub fn end_h2d(&mut self) -> Duration {
        if !self.enabled {
            return Duration::ZERO;
        }
        self.h2d_start
            .take()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Record device-to-host transfer start
    pub fn start_d2h(&mut self) {
        if !self.enabled {
            return;
        }
        self.h2d_start = Some(Instant::now());
    }

    /// Record device-to-host transfer end
    pub fn end_d2h(&mut self) -> Duration {
        if !self.enabled {
            return Duration::ZERO;
        }
        self.h2d_start
            .take()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// End timing a kernel execution
    pub fn end_kernel(
        &mut self,
        kernel_name: &str,
        input_bytes: usize,
        output_bytes: usize,
    ) -> KernelTiming {
        // Return zero timing if disabled
        if !self.enabled {
            return KernelTiming {
                kernel_name: kernel_name.to_string(),
                compile_time: None,
                h2d_time: Duration::ZERO,
                exec_time: Duration::ZERO,
                d2h_time: Duration::ZERO,
                total_time: Duration::ZERO,
                input_bytes,
                output_bytes,
            };
        }

        let kernel_start = self.kernel_start.take().unwrap_or_else(Instant::now);
        let exec_time = kernel_start.elapsed();

        let h2d_time = self.end_h2d();
        let d2h_time = self.end_d2h();

        let total_time = h2d_time + exec_time + d2h_time;

        let timing = KernelTiming {
            kernel_name: kernel_name.to_string(),
            compile_time: None,
            h2d_time,
            exec_time,
            d2h_time,
            total_time,
            input_bytes,
            output_bytes,
        };

        // Record timing for aggregation
        self.kernel_stats
            .entry(kernel_name.to_string())
            .or_insert_with(Vec::new)
            .push(timing.clone());

        // Also record in current step if available
        if let Some(last_step) = self.step_profiles.last_mut() {
            last_step.kernels.push(timing.clone());
        }

        timing
    }

    /// Record compilation time for current kernel
    pub fn record_compile_time(&mut self, compile_time: Duration) {
        if !self.enabled {
            return;
        }
        if let Some(kernel_name) = &self.current_kernel {
            if let Some(timings) = self.kernel_stats.get_mut(kernel_name) {
                if let Some(last_timing) = timings.last_mut() {
                    last_timing.compile_time = Some(compile_time);
                }
            }
        }
    }

    /// Get aggregated statistics for all kernels
    pub fn get_stats(&self) -> Vec<KernelStats> {
        let mut stats = Vec::new();
        let total_time: Duration = self
            .kernel_stats
            .values()
            .flatten()
            .map(|t| t.total_time)
            .sum();

        for (kernel_name, timings) in &self.kernel_stats {
            if timings.is_empty() {
                continue;
            }

            let call_count = timings.len() as u64;
            let total_exec_time: Duration = timings.iter().map(|t| t.exec_time).sum();
            let total_transfer_time: Duration =
                timings.iter().map(|t| t.h2d_time + t.d2h_time).sum();
            let total_compile_time: Duration = timings.iter().filter_map(|t| t.compile_time).sum();

            let exec_times: Vec<Duration> = timings.iter().map(|t| t.exec_time).collect();
            let min_exec_time = *exec_times.iter().min().unwrap_or(&Duration::ZERO);
            let max_exec_time = *exec_times.iter().max().unwrap_or(&Duration::ZERO);
            let avg_exec_time = total_exec_time / call_count as u32;

            let total_bytes: usize = timings.iter().map(|t| t.input_bytes + t.output_bytes).sum();
            let total_secs = total_exec_time.as_secs_f64();
            let throughput_gbps = if total_secs > 0.0 {
                (total_bytes as f64 / 1e9) / total_secs
            } else {
                0.0
            };

            let pct_of_total = if total_time.as_secs_f64() > 0.0 {
                (total_exec_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
            } else {
                0.0
            };

            stats.push(KernelStats {
                kernel_name: kernel_name.clone(),
                call_count,
                total_exec_time,
                total_transfer_time,
                total_compile_time,
                avg_exec_time,
                min_exec_time,
                max_exec_time,
                total_bytes,
                throughput_gbps,
                pct_of_total,
            });
        }

        // Sort by total time (descending)
        stats.sort_by(|a, b| b.total_exec_time.cmp(&a.total_exec_time));
        stats
    }

    /// Get overall profiler metrics
    pub fn get_metrics(&self) -> ProfilerMetrics {
        let total_steps = self.step_profiles.len() as u64;
        let total_kernels: u64 = self.kernel_stats.values().map(|v| v.len() as u64).sum();

        let total_compute_time: Duration = self
            .kernel_stats
            .values()
            .flatten()
            .map(|t| t.exec_time)
            .sum();

        let total_transfer_time: Duration = self
            .kernel_stats
            .values()
            .flatten()
            .map(|t| t.h2d_time + t.d2h_time)
            .sum();

        let total_compile_time: Duration = self
            .kernel_stats
            .values()
            .flatten()
            .filter_map(|t| t.compile_time)
            .sum();

        let avg_step_time = if total_steps > 0 {
            let total: Duration = self.step_profiles.iter().map(|s| s.total_duration).sum();
            total / total_steps as u32
        } else {
            Duration::ZERO
        };

        let total_step_time: Duration = self.step_profiles.iter().map(|s| s.total_duration).sum();
        let avg_ane_utilization = if total_step_time.as_secs_f64() > 0.0 {
            (total_compute_time.as_secs_f64() / total_step_time.as_secs_f64()) * 100.0
        } else {
            0.0
        };

        let total_bytes: usize = self
            .kernel_stats
            .values()
            .flatten()
            .map(|t| t.input_bytes + t.output_bytes)
            .sum();
        let overall_throughput_gbps = if total_compute_time.as_secs_f64() > 0.0 {
            (total_bytes as f64 / 1e9) / total_compute_time.as_secs_f64()
        } else {
            0.0
        };

        ProfilerMetrics {
            total_steps,
            avg_step_time,
            total_kernels,
            total_compute_time,
            total_transfer_time,
            total_compile_time,
            avg_ane_utilization,
            overall_throughput_gbps,
        }
    }

    /// Generate a text report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════╗\n");
        report.push_str("║              ANE Profiler Report                        ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════╝\n\n");

        // Overall metrics
        let metrics = self.get_metrics();
        report.push_str("┌──────────────────────────────────────────────────────────┐\n");
        report.push_str("│ Overall Metrics                                          │\n");
        report.push_str("├──────────────────────────────────────────────────────────┤\n");
        report.push_str(&format!(
            "│ Total Steps:           {:>10}                                  │\n",
            metrics.total_steps
        ));
        report.push_str(&format!(
            "│ Total Kernels:         {:>10}                                  │\n",
            metrics.total_kernels
        ));
        report.push_str(&format!(
            "│ Avg Step Time:         {:>10.2} ms                             │\n",
            metrics.avg_step_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "│ Total Compute Time:    {:>10.2} ms                             │\n",
            metrics.total_compute_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "│ Total Transfer Time:   {:>10.2} ms                             │\n",
            metrics.total_transfer_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "│ Total Compile Time:    {:>10.2} ms                             │\n",
            metrics.total_compile_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "│ ANE Utilization:       {:>10.1} %                               │\n",
            metrics.avg_ane_utilization
        ));
        report.push_str(&format!(
            "│ Throughput:            {:>10.2} GB/s                           │\n",
            metrics.overall_throughput_gbps
        ));
        report.push_str("└──────────────────────────────────────────────────────────┘\n\n");

        // Per-kernel statistics
        let stats = self.get_stats();
        if !stats.is_empty() {
            report.push_str("┌──────────────────────────────────────────────────────────┐\n");
            report.push_str("│ Per-Kernel Statistics (sorted by time)                  │\n");
            report.push_str("├──────────────────────────────────────────────────────────┤\n");

            for stat in &stats {
                report.push_str(&format!(
                    "│ {:<20}                                    │\n",
                    stat.kernel_name
                ));
                report.push_str(&format!(
                    "│   Calls: {:>6}   Avg: {:>8.2} ms   Total: {:>8.2} ms ({:>5.1}%)  │\n",
                    stat.call_count,
                    stat.avg_exec_time.as_secs_f64() * 1000.0,
                    stat.total_exec_time.as_secs_f64() * 1000.0,
                    stat.pct_of_total
                ));
                report.push_str(&format!(
                    "│   Throughput: {:>8.2} GB/s   Transfer: {:>8.2} ms                │\n",
                    stat.throughput_gbps,
                    stat.total_transfer_time.as_secs_f64() * 1000.0
                ));

                if stat.min_exec_time != stat.max_exec_time {
                    report.push_str(&format!(
                        "│   Min: {:>8.2} ms   Max: {:>8.2} ms                          │\n",
                        stat.min_exec_time.as_secs_f64() * 1000.0,
                        stat.max_exec_time.as_secs_f64() * 1000.0
                    ));
                }
                report.push_str("│                                                          │\n");
            }

            report.push_str("└──────────────────────────────────────────────────────────┘\n");
        }

        // Bottleneck analysis
        report.push_str("\n┌──────────────────────────────────────────────────────────┐\n");
        report.push_str("│ Bottleneck Analysis                                      │\n");
        report.push_str("├──────────────────────────────────────────────────────────┤\n");

        if let Some(slowest) = stats.first() {
            report.push_str(&format!(
                "│ Slowest Kernel: {}                   │\n",
                truncate(&slowest.kernel_name, 42)
            ));
            report.push_str(&format!(
                "│   Takes {:.1}% of total compute time                   │\n",
                slowest.pct_of_total
            ));
        }

        let transfer_ratio = if metrics.total_compute_time.as_secs_f64() > 0.0 {
            metrics.total_transfer_time.as_secs_f64() / metrics.total_compute_time.as_secs_f64()
        } else {
            0.0
        };

        if transfer_ratio > 0.3 {
            report.push_str(&format!(
                "│ ⚠️  High transfer overhead: {:.1}% of compute time      │\n",
                transfer_ratio * 100.0
            ));
            report.push_str("│   Consider: larger batches, async transfers        │\n");
        } else {
            report.push_str("│ ✓ Transfer overhead is acceptable                      │\n");
        }

        if metrics.avg_ane_utilization < 50.0 {
            report.push_str(&format!(
                "│ ⚠️  Low ANE utilization: {:.1}%                              │\n",
                metrics.avg_ane_utilization
            ));
            report.push_str("│   Consider: kernel fusion, better pipelining         │\n");
        } else if metrics.avg_ane_utilization > 80.0 {
            report.push_str(&format!(
                "│ ✓ Excellent ANE utilization: {:.1}%                        │\n",
                metrics.avg_ane_utilization
            ));
        }

        report.push_str("└──────────────────────────────────────────────────────────┘\n");

        report
    }

    /// Clear all collected profiling data
    pub fn clear(&mut self) {
        self.current_step = 0;
        self.step_start = None;
        self.kernel_start = None;
        self.current_kernel = None;
        self.step_profiles.clear();
        self.kernel_stats.clear();
    }
}

impl Default for ANEProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to truncate strings for display
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// RAII guard for automatic kernel timing
pub struct KernelTimer<'a> {
    profiler: &'a mut ANEProfiler,
    kernel_name: String,
    input_bytes: usize,
    output_bytes: usize,
}

impl<'a> KernelTimer<'a> {
    /// Create a new kernel timer
    pub fn new(profiler: &'a mut ANEProfiler, kernel_name: &str) -> Self {
        profiler.start_kernel(kernel_name);
        Self {
            profiler,
            kernel_name: kernel_name.to_string(),
            input_bytes: 0,
            output_bytes: 0,
        }
    }

    /// Set input/output sizes for throughput calculation
    pub fn with_data(&mut self, input_bytes: usize, output_bytes: usize) -> &mut Self {
        self.input_bytes = input_bytes;
        self.output_bytes = output_bytes;
        self
    }
}

impl<'a> Drop for KernelTimer<'_> {
    fn drop(&mut self) {
        self.profiler
            .end_kernel(&self.kernel_name, self.input_bytes, self.output_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic_timing() {
        let mut profiler = ANEProfiler::new();

        profiler.start_step();
        profiler.start_kernel("test_kernel");
        std::thread::sleep(Duration::from_millis(10));
        let timing = profiler.end_kernel("test_kernel", 1024, 1024);
        profiler.end_step();

        assert_eq!(timing.kernel_name, "test_kernel");
        assert!(timing.exec_time >= Duration::from_millis(10));
        assert!(timing.exec_time < Duration::from_millis(100));
    }

    #[test]
    fn test_profiler_disabled() {
        let mut profiler = ANEProfiler::disabled();

        profiler.start_step();
        profiler.start_kernel("test_kernel");
        profiler.end_kernel("test_kernel", 1024, 1024);
        profiler.end_step();

        assert!(profiler.step_profiles.is_empty());
        assert!(profiler.kernel_stats.is_empty());
    }

    #[test]
    fn test_kernel_timer_guard() {
        let mut profiler = ANEProfiler::new();

        {
            let mut _timer = KernelTimer::new(&mut profiler, "guarded_kernel");
            std::thread::sleep(Duration::from_millis(5));
        } // Timer drops here

        let stats = profiler.get_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].kernel_name, "guarded_kernel");
        assert_eq!(stats[0].call_count, 1);
    }

    #[test]
    fn test_multiple_kernels_aggregation() {
        let mut profiler = ANEProfiler::new();

        for _ in 0..3 {
            profiler.start_step();
            profiler.start_kernel("kernel_a");
            std::thread::sleep(Duration::from_millis(5));
            profiler.end_kernel("kernel_a", 512, 512);

            profiler.start_kernel("kernel_b");
            std::thread::sleep(Duration::from_millis(3));
            profiler.end_kernel("kernel_b", 256, 256);
            profiler.end_step();
        }

        let stats = profiler.get_stats();
        assert_eq!(stats.len(), 2);

        let kernel_a = stats.iter().find(|s| s.kernel_name == "kernel_a").unwrap();
        assert_eq!(kernel_a.call_count, 3);

        let kernel_b = stats.iter().find(|s| s.kernel_name == "kernel_b").unwrap();
        assert_eq!(kernel_b.call_count, 3);
    }

    #[test]
    fn test_generate_report() {
        let mut profiler = ANEProfiler::new();

        profiler.start_step();
        profiler.start_kernel("rmsnorm");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_kernel("rmsnorm", 4096, 4096);
        profiler.end_step();

        let report = profiler.generate_report();
        assert!(report.contains("ANE Profiler Report"));
        assert!(report.contains("Overall Metrics"));
        assert!(report.contains("Per-Kernel Statistics"));
        assert!(report.contains("rmsnorm"));
    }

    #[test]
    fn test_profiler_metrics() {
        let mut profiler = ANEProfiler::new();

        profiler.start_step();
        profiler.start_kernel("test");
        profiler.end_kernel("test", 1024, 1024);
        profiler.end_step();

        let metrics = profiler.get_metrics();
        assert_eq!(metrics.total_steps, 1);
        assert_eq!(metrics.total_kernels, 1);
        assert!(metrics.avg_ane_utilization >= 0.0);
        assert!(metrics.avg_ane_utilization <= 100.0);
    }
}
