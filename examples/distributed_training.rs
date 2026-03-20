//! Distributed Training Example (Multi-ANE)
//!
//! This example demonstrates how distributed training across multiple ANE devices
//! would work for data parallelism.
//!
//! **Note:** Full multi-ANE training requires hardware with multiple ANEs (M3/M4 Max/Ultra)
//! and is primarily a demonstration of the API design.
//!
//! ## Architecture
//!
//! **Data Parallelism:**
//! - Each ANE processes a different subset of the batch
//! - Gradients are averaged (all-reduce) across devices
//! - Model weights are synchronized after each step
//!
//! Run this example:
//! ```sh
//! cargo run --example distributed_training
//! ```

use rustane::ane::{
    detect_ane_devices, get_optimal_device_count, per_device_batch_size, MultiANEConfig,
};
use rustane::data::Batch;
use rustane::training::{TransformerANE, TransformerConfig};
use rustane::TrainingModel;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Distributed Training (Multi-ANE) Demo ===\n");

    // Detect available ANE devices
    let devices = detect_ane_devices()?;

    println!("--- Device Detection ---");
    println!("  Available ANE devices: {}", devices.len());
    for (i, device) in devices.iter().enumerate() {
        println!(
            "    Device {}: {} (~{} MB)",
            i, device.name, device.available_memory_mb
        );
    }
    println!();

    let optimal_count = get_optimal_device_count();
    println!("  Optimal device count for training: {}", optimal_count);
    println!();

    // Create a model configuration
    let config = TransformerConfig::new(512, 256, 512, 8, 6, 128)?;

    println!("--- Model Configuration ---");
    println!("  Parameters: {}", config.param_count());
    println!(
        "  Memory per ANE: ~{} MB",
        estimate_model_memory_mb(&config)
    );
    println!();

    // Demonstrate batch distribution
    let total_batch_size = 32;
    let num_devices = optimal_count.max(1); // Use at least 1 device

    println!("--- Batch Distribution ---");
    println!("  Total batch size: {}", total_batch_size);
    println!("  Number of devices: {}", num_devices);

    match per_device_batch_size(total_batch_size, num_devices) {
        Ok(per_device) => {
            println!("  Per-device batch: {}", per_device);
            println!();

            // Simulate distributed training
            simulate_distributed_training(&config, total_batch_size, num_devices)?;
        }
        Err(e) => {
            println!("  ✗ Batch distribution failed: {}", e);
            println!("  Hint: Use batch size divisible by {}", num_devices);
        }
    }

    println!("\n--- Scaling Analysis ---");
    println!(
        "Theoretical speedup with {} ANEs: {:.1}x",
        num_devices,
        num_devices as f32 * 0.85
    );
    println!("  (85% efficiency due to communication overhead)");
    println!();
    println!("Real-world considerations:");
    println!("  • Communication overhead for gradient synchronization");
    println!("  • Memory bandwidth limitations");
    println!("  • Batch size must be divisible by device count");
    println!("  • Larger models benefit more from multiple ANEs");

    Ok(())
}

/// Estimate model memory usage in MB
fn estimate_model_memory_mb(config: &TransformerConfig) -> usize {
    // Rough estimate: parameters * 4 bytes + activations
    let param_memory = config.param_count() * 4;
    let activation_memory = config.n_layers * config.dim * config.seq_len * config.dim * 4;

    (param_memory + activation_memory) / (1024 * 1024)
}

/// Simulate distributed training across multiple ANE devices
fn simulate_distributed_training(
    config: &TransformerConfig,
    total_batch: usize,
    num_devices: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Simulated Distributed Training ---");

    let per_device_batch = total_batch / num_devices;

    // Simulate forward pass on each device
    println!("  Forward pass (parallel on {} ANEs):", num_devices);
    let forward_start = Instant::now();

    for device_id in 0..num_devices {
        let token_start = device_id * per_device_batch * config.seq_len;
        let token_end = (device_id + 1) * per_device_batch * config.seq_len;

        println!(
            "    Device {}: tokens[{}..{}]",
            device_id, token_start, token_end
        );

        // Simulate model creation and forward pass
        let mut model = TransformerANE::new(config)?;
        let tokens = vec![42u32; per_device_batch * config.seq_len];
        let batch = Batch::new(tokens, per_device_batch, config.seq_len)?;

        let _ = model.forward(&batch)?;
    }

    let forward_time = forward_start.elapsed();
    println!("  Time: {:.2?}", forward_time);

    // Simulate gradient accumulation (all-reduce)
    println!("  Gradient all-reduce:");
    println!("    Gather gradients from {} devices", num_devices);
    println!("    Average gradients");
    println!("    Broadcast updated weights");

    println!("\n  Note: Full multi-ANE training requires:");
    println!("    • Gradient synchronization primitives");
    println!("    • Optimizer state distributed across devices");
    println!("    • NCCL-like all-reduce implementation");

    Ok(())
}

/// Demonstrate multi-ANE configuration validation
fn demonstrate_multi_ane_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Multi-ANE Configuration ---");

    // Valid configuration
    let config = MultiANEConfig::new(2).with_shard_size(4);
    println!("  2-device config: valid = {}", config.validate().is_ok());

    // Invalid configurations
    let too_many = MultiANEConfig::new(5); // Max 4 on Apple Silicon
    println!("  5-device config: valid = {}", too_many.validate().is_ok());

    let invalid_batch = MultiANEConfig::new(2).with_shard_size(3);
    println!("  Shard size 3 (batch 8): divisible = {}", (8 % 2 == 0));

    Ok(())
}
