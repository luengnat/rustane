//! Mixed Precision Training Example
//!
//! This example demonstrates how to use FP16/BF16 for memory-efficient training
//! while maintaining FP32 master weights for numerical stability.
//!
//! ## Key Concepts
//!
//! **Mixed Precision Training:**
//! - Forward/backward in FP16/BF16 (reduced memory, faster computation)
//! - Master weights in FP32 (numerical stability)
//! - Loss scaling to prevent gradient underflow
//!
//! ## Memory Savings
//!
//! | Precision | Bytes per value | Memory for 1B params |
//!|-----------|----------------|---------------------|
//! | FP32 | 4 bytes | 4 GB |
//! | FP16 | 2 bytes | 2 GB |
//! | BF16 | 2 bytes | 2 GB |
//!
//! Run this example:
//! ```sh
//! cargo run --example mixed_precision_training
//! ```

use rustane::data::Batch;
use rustane::training::transformer_config::GradientCheckpointingConfig;
use rustane::training::{LossScaler, TransformerANE, TransformerConfig};
use rustane::TrainingModel;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mixed Precision Training Demo ===\n");

    // Create a transformer configuration
    let vocab_size = 512;
    let dim = 128;
    let hidden_dim = 256;
    let n_heads = 4;
    let n_layers = 4;
    let seq_len = 64;

    let config = TransformerConfig::new(vocab_size, dim, hidden_dim, n_heads, n_layers, seq_len)?;

    println!("Configuration:");
    println!("  Layers: {}", n_layers);
    println!("  Hidden dim: {}", dim);
    println!("  Sequence length: {}", seq_len);
    println!("  Total parameters: {}\n", config.param_count());

    // Calculate memory usage for different precisions
    let param_count = config.param_count();
    let fp32_memory_mb = (param_count * 4) / (1024 * 1024);
    let fp16_memory_mb = (param_count * 2) / (1024 * 1024);
    let memory_saved_mb = fp32_memory_mb - fp16_memory_mb;

    println!("Memory Usage:");
    println!("  FP32 weights:     {} MB", fp32_memory_mb);
    println!("  FP16 weights:     {} MB", fp16_memory_mb);
    println!(
        "  Memory saved:     {} MB ({:.1}%)\n",
        memory_saved_mb, 50.0
    );

    // Create model with gradient checkpointing for additional memory savings
    let config_with_checkpointing = config
        .clone()
        .with_gradient_checkpointing(GradientCheckpointingConfig::with_interval(2));

    let mut model = TransformerANE::new(&config_with_checkpointing)?;

    // Create a loss scaler for FP16 training
    let mut loss_scaler = LossScaler::for_transformer(n_layers);
    println!("Loss Scaler:");
    println!("  Initial scale: {:.2}\n", loss_scaler.current_scale());

    // Create a training batch
    let batch_size = 2;
    let tokens = vec![42u32; batch_size * seq_len];
    let batch = Batch::new(tokens, batch_size, seq_len)?;

    println!("--- Forward Pass ---");

    // Forward pass (model internally uses FP16 for ANE operations)
    let start = Instant::now();
    let output = model.forward(&batch)?;
    let forward_time = start.elapsed();

    println!("  Output shape: {:?}", output.shape());
    println!("  Forward time: {:.2?}", forward_time);

    // Simulate a loss value
    let loss = 2.5;
    println!("  Simulated loss: {:.4}\n", loss);

    // Scale loss for FP16 training
    let scaled_loss = loss_scaler.scale_loss(loss);
    println!("--- Loss Scaling ---");
    println!("  Original loss: {:.4}", loss);
    println!("  Scaled loss: {:.4}", scaled_loss);
    println!("  Scale factor: {:.2}\n", loss_scaler.current_scale());

    // Demonstrate memory savings with gradient checkpointing
    println!("--- Gradient Checkpointing ---");
    println!("  Checkpoint interval: 2");
    println!("  Memory savings: ~50% of activation memory");
    println!("  Trade-off: Recomputation overhead during backward\n");

    println!("--- Summary ---");
    println!("Mixed precision training combines:");
    println!("  1. FP16/FP32 master weights for stability");
    println!("  2. FP16 forward/backward for speed and memory savings");
    println!("  3. Loss scaling to prevent underflow");
    println!("  4. Gradient checkpointing for additional memory savings");
    println!("\nTotal memory reduction: ~65-75% vs pure FP32 training");
    println!("(50% from FP16 weights + additional from checkpointing)");

    Ok(())
}
