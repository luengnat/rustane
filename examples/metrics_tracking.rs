//! Example: Training Metrics Tracking and Logging
//!
//! Demonstrates the metrics tracking system with multiple logging backends:
//! - Console logging (real-time progress)
//! - File logging (persistent records)
//! - JSON logging (structured output for analysis)
//! - In-memory aggregation (statistics and analysis)
//!
//! This provides a foundation for integration with experiment tracking tools
//! like WandB or MLflow.

use rustane::training::metrics::MetricsTracker;
use std::collections::HashMap;

fn main() {
    println!("Rustane Training Metrics Tracking Example");
    println!("=======================================\n");

    // Example 1: Basic console logging
    println!("Example 1: Basic Console Logging");
    println!("---------------------------------");
    basic_console_logging();
    println!();

    // Example 2: File logging
    println!("Example 2: File Logging");
    println!("---------------------");
    file_logging_demo();
    println!();

    // Example 3: JSON logging for analysis
    println!("Example 3: JSON Logging");
    println!("--------------------");
    json_logging_demo();
    println!();

    // Example 4: Aggregated statistics
    println!("Example 4: Metrics Aggregation");
    println!("---------------------------");
    metrics_aggregation_demo();
    println!();

    // Example 5: Simulated training run
    println!("Example 5: Simulated Training Run");
    println!("-------------------------------");
    simulated_training_run();
    println!();

    println!("✓ Example completed!");
    println!("\nKey features:");
    println!("  • Multiple logger backends (console, file, JSON)");
    println!("  • Automatic metrics aggregation and statistics");
    println!("  • Compatible with WandB/MLflow (via JSON export)");
    println!("  • Real-time progress monitoring");
}

/// Demonstrate basic console logging
fn basic_console_logging() {
    let mut tracker = MetricsTracker::new().with_console();

    println!("Logging training metrics:");
    tracker.log("loss", 2.5);
    tracker.log("accuracy", 0.75);
    tracker.log("learning_rate", 0.001);

    tracker.increment_step();
    tracker.log("loss", 2.3);
    tracker.log("accuracy", 0.78);

    println!("\n→ Metrics logged to console in real-time");
}

/// Demonstrate file logging
fn file_logging_demo() {
    let log_path = "/tmp/rustane_training.log";

    let mut tracker = MetricsTracker::new().with_file_logger(log_path);

    println!("Logging to file: {}", log_path);

    // Simulate some training steps
    for step in 0..5 {
        let loss = 2.5 - (step as f64 * 0.1);
        tracker.log_at_step("loss", loss, step);
    }

    tracker.flush();

    println!("→ Metrics persisted to {}", log_path);

    // Show file contents
    if let Ok(contents) = std::fs::read_to_string(log_path) {
        println!("\nFile contents (first 10 lines):");
        for (i, line) in contents.lines().take(10).enumerate() {
            println!("  {}: {}", i + 1, line);
        }
    }
}

/// Demonstrate JSON logging
fn json_logging_demo() {
    let json_path = "/tmp/rustane_training.json";

    let mut tracker = MetricsTracker::new().with_json_logger(json_path);

    println!("Logging JSON to: {}", json_path);

    // Log multiple metrics
    let mut metrics = HashMap::new();
    metrics.insert("loss".to_string(), 2.5);
    metrics.insert("accuracy".to_string(), 0.75);
    metrics.insert("grad_norm".to_string(), 1.2);

    tracker.log_metrics(&metrics);
    tracker.increment_step();

    metrics.insert("loss".to_string(), 2.3);
    metrics.insert("accuracy".to_string(), 0.78);
    metrics.insert("grad_norm".to_string(), 1.1);

    tracker.log_metrics(&metrics);
    tracker.flush();

    println!("→ JSON metrics written to {}", json_path);

    // Show JSON file contents
    if let Ok(contents) = std::fs::read_to_string(json_path) {
        println!("\nJSON contents:");
        println!("{}", contents);
    }
}

/// Demonstrate metrics aggregation
fn metrics_aggregation_demo() {
    let mut tracker = MetricsTracker::new();
    let aggregator = tracker.aggregator_mut();

    println!("Collecting metrics for analysis:");

    // Add some training data
    for step in 0..10 {
        let loss = 2.5 - (step as f64 * 0.15) + (step as f32 * 0.01) as f64;
        aggregator.add("loss", loss, step);

        let accuracy = 0.6 + (step as f64 * 0.03);
        aggregator.add("accuracy", accuracy, step);
    }

    // Get statistics
    if let Some(loss_stats) = aggregator.stats("loss") {
        println!("\nLoss Statistics:");
        println!("  Count: {}", loss_stats.count);
        println!("  Min: {:.4}", loss_stats.min);
        println!("  Max: {:.4}", loss_stats.max);
        println!("  Avg: {:.4}", loss_stats.avg);
        println!("  StdDev: {:.4}", loss_stats.std_dev);
    }

    if let Some(acc_stats) = aggregator.stats("accuracy") {
        println!("\nAccuracy Statistics:");
        println!("  Count: {}", acc_stats.count);
        println!("  Min: {:.4}", acc_stats.min);
        println!("  Max: {:.4}", acc_stats.max);
        println!("  Avg: {:.4}", acc_stats.avg);
        println!("  StdDev: {:.4}", acc_stats.std_dev);
    }

    println!("\n→ Metrics aggregated for analysis");
}

/// Demonstrate a simulated training run
fn simulated_training_run() {
    // Setup tracker with multiple backends
    let mut tracker = MetricsTracker::new()
        .with_console()
        .with_json_logger("/tmp/rustane_full_training.json");

    println!("Simulated training (20 steps):");
    println!("-----------------------------");

    let mut metrics = HashMap::new();

    for step in 0..20 {
        // Simulate training dynamics
        let loss =
            3.0_f64 * (0.95_f32).powi(step) as f64 + 0.1_f64 * (step as f32).sin().abs() as f64;
        let accuracy = 1.0 - (loss / 4.0);
        let grad_norm = 2.0_f64 * (0.98_f32).powi(step) as f64 + 0.05_f64;
        let learning_rate = 0.001_f64 * (0.995_f32).powi(step) as f64;

        metrics.insert("loss".to_string(), loss);
        metrics.insert("accuracy".to_string(), accuracy);
        metrics.insert("grad_norm".to_string(), grad_norm);
        metrics.insert("learning_rate".to_string(), learning_rate);

        tracker.log_metrics(&metrics);

        // Print progress every 5 steps
        if (step + 1) % 5 == 0 {
            println!(
                "Step {:2}: loss={:.4}, acc={:.4}, lr={:.6}",
                step + 1,
                loss,
                accuracy,
                learning_rate
            );
        }
    }

    tracker.flush();

    println!("\n=== Training Summary ===");
    tracker.print_summary();

    println!("\nIntegration with WandB/MLflow:");
    println!("  The JSON log file can be imported into:");
    println!("  - WandB: `wandb sync /tmp/rustane_full_training.json`");
    println!("  - MLflow: Load as a pandas DataFrame for analysis");
    println!("\n→ Structured logs enable post-hoc analysis and visualization");
}
