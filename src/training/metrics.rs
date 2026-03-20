//! Training Metrics Tracking and Logging
//!
//! Provides comprehensive metrics tracking for training runs with support for:
//! - Console logging (real-time progress)
//! - File logging (persistent records)
//! - In-memory aggregation (for analysis)
//! - Extension points for WandB/MLflow integration
//!
//! # Usage
//!
//! ```ignore
//! use rustane::training::metrics::{MetricsTracker, MetricsLogger, ConsoleLogger};
//!
//! let mut tracker = MetricsTracker::new()
//!     .with_logger(Box::new(ConsoleLogger::new()))
//!     .with_logger(Box::new(FileLogger::new("training.log")));
//!
//! tracker.log("train_loss", 0.5, 1);
//! tracker.log("learning_rate", 0.001, 1);
//! tracker.log("grad_norm", 1.2, 1);
//!
//! tracker.flush(); // Write to all loggers
//! ```

use crate::training::StepMetrics;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for metrics logging backends
pub trait MetricsLogger: Send {
    /// Log a single metric value
    fn log(&mut self, name: &str, value: f64, step: u32);

    /// Log a batch of metrics at once
    fn log_batch(&mut self, metrics: &[(String, f64)], step: u32);

    /// Flush any buffered data
    fn flush(&mut self);

    /// Clone the logger for use in multiple contexts
    fn box_clone(&self) -> Box<dyn MetricsLogger>;
}

impl Clone for Box<dyn MetricsLogger> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Console logger for real-time training progress
pub struct ConsoleLogger {
    show_timestamp: bool,
    min_level: LogLevel,
}

/// Log level for filtering console output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Debug-level messages
    Debug,
    /// Info-level messages
    Info,
    /// Warning-level messages
    Warning,
    /// Error-level messages
    Error,
}

impl ConsoleLogger {
    /// Create a new console logger
    pub fn new() -> Self {
        Self {
            show_timestamp: true,
            min_level: LogLevel::Info,
        }
    }

    /// Create a console logger without timestamps
    pub fn without_timestamps() -> Self {
        Self {
            show_timestamp: false,
            min_level: LogLevel::Info,
        }
    }

    /// Set the minimum log level
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = level;
        self
    }
}

impl Default for ConsoleLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsLogger for ConsoleLogger {
    fn log(&mut self, name: &str, value: f64, step: u32) {
        if self.min_level > LogLevel::Info {
            return;
        }

        let timestamp = if self.show_timestamp {
            format!(
                "[{}] ",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| format!("{:.3}", d.as_secs_f64()))
                    .unwrap_or_else(|_| "?".to_string())
            )
        } else {
            String::new()
        };

        println!("{}Step {:5}: {} = {:.6}", timestamp, step, name, value);
    }

    fn log_batch(&mut self, metrics: &[(String, f64)], step: u32) {
        if self.min_level > LogLevel::Info {
            return;
        }

        let timestamp = if self.show_timestamp {
            format!(
                "[{}] ",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| format!("{:.3}", d.as_secs_f64()))
                    .unwrap_or_else(|_| "?".to_string())
            )
        } else {
            String::new()
        };

        print!("{}Step {:5}: ", timestamp, step);
        for (i, (name, value)) in metrics.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{} = {:.6}", name, value);
        }
        println!();
    }

    fn flush(&mut self) {
        let _ = std::io::stdout().flush();
    }

    fn box_clone(&self) -> Box<dyn MetricsLogger> {
        Box::new(ConsoleLogger {
            show_timestamp: self.show_timestamp,
            min_level: self.min_level,
        })
    }
}

/// File logger for persistent training records
pub struct FileLogger {
    path: PathBuf,
    file: Option<File>,
    buffer: Vec<String>,
    buffer_size: usize,
}

impl FileLogger {
    /// Create a new file logger
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        Self {
            path,
            file: None,
            buffer: Vec::new(),
            buffer_size: 100,
        }
    }

    /// Set the buffer size before flushing to disk
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Ensure the file is open
    fn ensure_file_open(&mut self) -> Result<(), String> {
        if self.file.is_none() {
            self.file = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)
                    .map_err(|e| format!("Failed to open log file: {}", e))?,
            );

            // Write header if file is empty
            if let Ok(metadata) = std::fs::metadata(&self.path) {
                if metadata.len() == 0 {
                    if let Some(ref mut file) = self.file {
                        let _ = file.write_all(b"timestamp,step,name,value\n");
                    }
                }
            }
        }
        Ok(())
    }

    /// Write buffered logs to file
    fn flush_buffer(&mut self) -> Result<(), String> {
        self.ensure_file_open()?;

        if let Some(ref mut file) = self.file {
            for line in &self.buffer {
                writeln!(file, "{}", line)
                    .map_err(|e| format!("Failed to write to log file: {}", e))?;
            }
            file.flush()
                .map_err(|e| format!("Failed to flush log file: {}", e))?;
        }

        self.buffer.clear();
        Ok(())
    }
}

impl MetricsLogger for FileLogger {
    fn log(&mut self, name: &str, value: f64, step: u32) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let line = format!("{},{},{},{}", timestamp, step, name, value);
        self.buffer.push(line);

        if self.buffer.len() >= self.buffer_size {
            let _ = self.flush_buffer();
        }
    }

    fn log_batch(&mut self, metrics: &[(String, f64)], step: u32) {
        for (name, value) in metrics {
            self.log(name, *value, step);
        }
    }

    fn flush(&mut self) {
        let _ = self.flush_buffer();
    }

    fn box_clone(&self) -> Box<dyn MetricsLogger> {
        Box::new(FileLogger {
            path: self.path.clone(),
            file: None,
            buffer: Vec::new(),
            buffer_size: self.buffer_size,
        })
    }
}

/// JSON file logger for structured output (compatible with WandB/MLflow)
pub struct JsonLogger {
    path: PathBuf,
    file: Option<File>,
    buffer: Vec<String>,
    buffer_size: usize,
}

impl JsonLogger {
    /// Create a new JSON logger
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        Self {
            path,
            file: None,
            buffer: Vec::new(),
            buffer_size: 50,
        }
    }

    /// Set the buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    fn ensure_file_open(&mut self) -> Result<(), String> {
        if self.file.is_none() {
            self.file = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)
                    .map_err(|e| format!("Failed to open JSON log file: {}", e))?,
            );

            // Write opening bracket if file is empty
            if let Ok(metadata) = std::fs::metadata(&self.path) {
                if metadata.len() == 0 {
                    if let Some(ref mut file) = self.file {
                        let _ = file.write_all(b"[\n");
                    }
                }
            }
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<(), String> {
        self.ensure_file_open()?;

        let is_continuation = self.is_continuation();

        if let Some(ref mut file) = self.file {
            for (i, line) in self.buffer.iter().enumerate() {
                if i > 0 || is_continuation {
                    let _ = file.write_all(b",\n");
                }
                writeln!(file, "{}", line)
                    .map_err(|e| format!("Failed to write to JSON log: {}", e))?;
            }
            file.flush()
                .map_err(|e| format!("Failed to flush JSON log: {}", e))?;
        }

        self.buffer.clear();
        Ok(())
    }

    fn is_continuation(&self) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.path) {
            metadata.len() > 10 // More than just "[\n"
        } else {
            false
        }
    }
}

impl MetricsLogger for JsonLogger {
    fn log(&mut self, name: &str, value: f64, step: u32) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let json = format!(
            r#"{{"timestamp": {:.3}, "step": {}, "metric": "{}", "value": {}}}"#,
            timestamp, step, name, value
        );
        self.buffer.push(json);

        if self.buffer.len() >= self.buffer_size {
            let _ = self.flush_buffer();
        }
    }

    fn log_batch(&mut self, metrics: &[(String, f64)], step: u32) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let mut metric_obj = String::from("{");
        metric_obj.push_str(&format!(r#"timestamp": {:.3}, "step": {}, "metrics": {{"#, timestamp, step));

        for (i, (name, value)) in metrics.iter().enumerate() {
            if i > 0 {
                metric_obj.push_str(", ");
            }
            metric_obj.push_str(&format!(r#""{}": {}"#, name, value));
        }
        metric_obj.push_str("}}");

        self.buffer.push(metric_obj);

        if self.buffer.len() >= self.buffer_size {
            let _ = self.flush_buffer();
        }
    }

    fn flush(&mut self) {
        let _ = self.flush_buffer();
    }

    fn box_clone(&self) -> Box<dyn MetricsLogger> {
        Box::new(JsonLogger {
            path: self.path.clone(),
            file: None,
            buffer: Vec::new(),
            buffer_size: self.buffer_size,
        })
    }
}

/// Aggregates metrics for statistical analysis
pub struct MetricsAggregator {
    history: HashMap<String, Vec<(u32, f64)>>,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
        }
    }

    /// Add a metric value
    pub fn add(&mut self, name: &str, value: f64, step: u32) {
        self.history
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push((step, value));
    }

    /// Get all values for a metric
    pub fn get(&self, name: &str) -> Option<&[(u32, f64)]> {
        self.history.get(name).map(|v| v.as_slice())
    }

    /// Get statistics for a metric
    pub fn stats(&self, name: &str) -> Option<MetricStats> {
        let values = self.history.get(name)?;

        if values.is_empty() {
            return None;
        }

        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &(_, val) in values {
            sum += val;
            min = min.min(val);
            max = max.max(val);
        }

        let avg = sum / values.len() as f64;

        // Compute variance
        let mut variance_sum = 0.0;
        for &(_, val) in values {
            let diff = val - avg;
            variance_sum += diff * diff;
        }
        let variance = variance_sum / values.len() as f64;
        let std_dev = variance.sqrt();

        Some(MetricStats {
            count: values.len(),
            min,
            max,
            avg,
            std_dev,
        })
    }

    /// Get the last value for a metric
    pub fn last_value(&self, name: &str) -> Option<f64> {
        self.history.get(name)?.last().map(|&(_, v)| v)
    }

    /// Get all metric names
    pub fn metric_names(&self) -> impl Iterator<Item = &str> {
        self.history.keys().map(|s| s.as_str())
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a metric
#[derive(Debug, Clone)]
pub struct MetricStats {
    /// Number of samples
    pub count: usize,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Average (mean) value
    pub avg: f64,
    /// Standard deviation
    pub std_dev: f64,
}

/// Training metrics tracker with multiple logging backends
pub struct MetricsTracker {
    loggers: Vec<Box<dyn MetricsLogger>>,
    aggregator: MetricsAggregator,
    current_step: u32,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            loggers: Vec::new(),
            aggregator: MetricsAggregator::new(),
            current_step: 0,
        }
    }

    /// Add a logger backend
    pub fn with_logger(mut self, logger: Box<dyn MetricsLogger>) -> Self {
        self.loggers.push(logger);
        self
    }

    /// Add console logger
    pub fn with_console(mut self) -> Self {
        self.loggers.push(Box::new(ConsoleLogger::new()));
        self
    }

    /// Add file logger
    pub fn with_file_logger(mut self, path: impl Into<PathBuf>) -> Self {
        self.loggers.push(Box::new(FileLogger::new(path)));
        self
    }

    /// Add JSON logger
    pub fn with_json_logger(mut self, path: impl Into<PathBuf>) -> Self {
        self.loggers.push(Box::new(JsonLogger::new(path)));
        self
    }

    /// Log a single metric
    pub fn log(&mut self, name: &str, value: f64) {
        self.log_at_step(name, value, self.current_step);
    }

    /// Log a metric at a specific step
    pub fn log_at_step(&mut self, name: &str, value: f64, step: u32) {
        // Log to all backends
        for logger in &mut self.loggers {
            logger.log(name, value, step);
        }

        // Add to aggregator
        self.aggregator.add(name, value, step);
    }

    /// Log multiple metrics at once
    pub fn log_metrics(&mut self, metrics: &HashMap<String, f64>) {
        let batch: Vec<(String, f64)> = metrics.iter().map(|(k, v)| (k.clone(), *v)).collect();

        // Log to all backends
        for logger in &mut self.loggers {
            logger.log_batch(&batch, self.current_step);
        }

        // Add to aggregator
        for (name, value) in metrics {
            self.aggregator.add(name, *value, self.current_step);
        }
    }

    /// Log training step metrics from the trainer
    pub fn log_step_metrics(&mut self, metrics: &StepMetrics) {
        self.log_at_step("loss", metrics.loss as f64, metrics.step);
        self.log_at_step("grad_norm", metrics.grad_norm as f64, metrics.step);
        self.log_at_step("learning_rate", metrics.learning_rate as f64, metrics.step);
        self.current_step = metrics.step + 1;
    }

    /// Increment the step counter
    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }

    /// Get the current step
    pub fn current_step(&self) -> u32 {
        self.current_step
    }

    /// Set the current step
    pub fn set_step(&mut self, step: u32) {
        self.current_step = step;
    }

    /// Flush all loggers
    pub fn flush(&mut self) {
        for logger in &mut self.loggers {
            logger.flush();
        }
    }

    /// Get the aggregator for analysis
    pub fn aggregator(&self) -> &MetricsAggregator {
        &self.aggregator
    }

    /// Get mutable aggregator
    pub fn aggregator_mut(&mut self) -> &mut MetricsAggregator {
        &mut self.aggregator
    }

    /// Print a summary of all tracked metrics
    pub fn print_summary(&self) {
        println!("\n=== Metrics Summary ===");
        for name in self.aggregator.metric_names() {
            if let Some(stats) = self.aggregator.stats(name) {
                println!(
                    "{}: count={}, min={:.4}, max={:.4}, avg={:.4}, std={:.4}",
                    name, stats.count, stats.min, stats.max, stats.avg, stats.std_dev
                );
            }
        }
        println!("=======================\n");
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_aggregator() {
        let mut agg = MetricsAggregator::new();

        agg.add("loss", 1.0, 0);
        agg.add("loss", 0.5, 1);
        agg.add("loss", 0.25, 2);

        let stats = agg.stats("loss").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.avg - 0.5833).abs() < 0.01);
        assert_eq!(stats.min, 0.25);
        assert_eq!(stats.max, 1.0);
    }

    #[test]
    fn test_metrics_aggregator_last_value() {
        let mut agg = MetricsAggregator::new();

        agg.add("lr", 0.001, 0);
        agg.add("lr", 0.0005, 1);

        assert_eq!(agg.last_value("lr"), Some(0.0005));
        assert_eq!(agg.last_value("unknown"), None);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new().with_console();

        tracker.log("test", 1.0);
        tracker.log("test", 2.0);

        assert_eq!(tracker.aggregator().last_value("test"), Some(2.0));
    }

    #[test]
    fn test_metrics_tracker_step() {
        let mut tracker = MetricsTracker::new();

        assert_eq!(tracker.current_step(), 0);
        tracker.increment_step();
        assert_eq!(tracker.current_step(), 1);
        tracker.set_step(10);
        assert_eq!(tracker.current_step(), 10);
    }

    #[test]
    fn test_metrics_tracker_batch() {
        let mut tracker = MetricsTracker::new();

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 1.0);
        metrics.insert("lr".to_string(), 0.001);

        tracker.log_metrics(&metrics);

        assert_eq!(tracker.aggregator().last_value("loss"), Some(1.0));
        assert_eq!(tracker.aggregator().last_value("lr"), Some(0.001));
    }

    #[test]
    fn test_console_logger_clone() {
        let logger = ConsoleLogger::new();
        let _logger_clone = logger.box_clone();
    }

    #[test]
    fn test_file_logger_creation() {
        let logger = FileLogger::new("/tmp/test.log");
        assert_eq!(logger.path, PathBuf::from("/tmp/test.log"));
    }

    #[test]
    fn test_json_logger_creation() {
        let logger = JsonLogger::new("/tmp/test.json");
        assert_eq!(logger.path, PathBuf::from("/tmp/test.json"));
    }

    #[test]
    fn test_metrics_tracker_with_json() {
        let mut tracker = MetricsTracker::new().with_json_logger("/tmp/test_metrics.json");

        tracker.log("test_metric", 42.0);
        tracker.flush();

        assert_eq!(tracker.aggregator().last_value("test_metric"), Some(42.0));
    }
}
