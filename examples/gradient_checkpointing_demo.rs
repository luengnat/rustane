//! Gradient Checkpointing Demonstration
//!
//! This example demonstrates how gradient checkpointing reduces memory usage
//! during training by selectively storing layer activations.
//!
//! ## Key Concepts
//!
//! **Without checkpointing:** All layer activations are stored during forward pass
//! - High memory usage: O(n_layers × activations_size)
//! - Fast backward pass (no recomputation)
//!
//! **With checkpointing:** Only every Nth layer's activations are stored
//! - Reduced memory usage: O(n_layers / checkpoint_interval × activations_size)
//! - Slower backward pass (recomputes missing activations)
//!
//! ## Trade-offs
//!
//! | Checkpoint Interval | Memory Saved | Recomputation Overhead |
//! |---------------------|--------------|------------------------|
//! | 1 (disabled) | 0% | 0% |
//! | 2 | ~50% | ~1× forward pass |
//! | 4 | ~75% | ~3× forward pass |
//!
//! Run this example:
//! ```sh
//! cargo run --example gradient_checkpointing_demo
//! ```

use rustane::data::Batch;
use rustane::training::transformer_config::GradientCheckpointingConfig;
use rustane::training::{TransformerANE, TransformerConfig};
use rustane::TrainingModel;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gradient Checkpointing Demo ===\n");

    // Create a small transformer configuration
    let vocab_size = 512;
    let dim = 128;
    let hidden_dim = 256;
    let n_heads = 4;
    let n_layers = 6;
    let seq_len = 64;

    let base_config =
        TransformerConfig::new(vocab_size, dim, hidden_dim, n_heads, n_layers, seq_len)?;

    println!("Configuration:");
    println!("  Layers: {}", n_layers);
    println!("  Hidden dim: {}", dim);
    println!("  Sequence length: {}", seq_len);
    println!("  Total parameters: {}\n", base_config.param_count());

    // Test different checkpointing configurations
    let configs = vec![
        ("No checkpointing", GradientCheckpointingConfig::disabled()),
        ("Interval 2", GradientCheckpointingConfig::with_interval(2)),
        ("Interval 3", GradientCheckpointingConfig::with_interval(3)),
    ];

    for (name, gc_config) in configs {
        println!("--- {} ---", name);

        let config = base_config.clone().with_gradient_checkpointing(gc_config);

        // Create model
        let mut model = TransformerANE::new(&config)?;

        // Create a small batch
        let batch_size = 2;
        let tokens = vec![1u32; batch_size * seq_len];
        let batch = Batch::new(tokens, batch_size, seq_len)?;

        // Measure forward pass
        let start = Instant::now();
        let output = model.forward(&batch)?;
        let forward_time = start.elapsed();

        // Estimate memory usage (rough approximation)
        let total_activations = estimate_activation_memory(&config, batch_size, seq_len);
        let memory_savings_pct = config
            .gradient_checkpointing
            .memory_savings_factor(n_layers)
            * 100.0;
        let stored_activations_pct = 100.0 - memory_savings_pct;
        let stored_activations =
            (total_activations as f32 * stored_activations_pct / 100.0) as usize;
        let memory_saved_mb = (total_activations - stored_activations) * 4 / (1024 * 1024);

        println!("  Forward time: {:.2?}", forward_time);
        println!("  Output shape: {:?}", output.shape());
        println!(
            "  Estimated memory saved: {:.1} MB ({:.1}%)",
            memory_saved_mb, memory_savings_pct
        );
        println!();
    }

    println!("--- Summary ---");
    println!("Gradient checkpointing reduces memory at the cost of recomputation.");
    println!("Use interval 2-4 for large models that don't fit in memory.");
    println!("\nNote: Recomputation during backward pass is not yet implemented.");
    println!("For now, disable checkpointing (interval=1) for training.");

    Ok(())
}

/// Estimate activation memory per layer in elements (not bytes)
fn estimate_activation_memory(
    config: &TransformerConfig,
    batch_size: usize,
    seq_len: usize,
) -> usize {
    // Approximate activations stored per layer:
    // - x_attn_in, x_attn_norm, q, k, v, attn_out, attn_probs
    // - x_ffn_in, x_ffn_norm, h1, silu, h3, ffn_hidden
    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let n_heads = config.n_heads;

    // Q, K, V projections
    let qkv_size = 3 * batch_size * seq_len * dim;
    // Attention output and intermediate values
    let attn_size = 2 * batch_size * seq_len * dim + batch_size * n_heads * seq_len * seq_len;
    // FFN intermediate values
    let ffn_size = batch_size * seq_len * (2 * dim + 2 * hidden_dim);

    qkv_size + attn_size + ffn_size
}
