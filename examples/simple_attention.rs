//! Simple attention example
//!
//! This example demonstrates how to use the MultiHeadAttention and SelfAttention layers
//! in Rustane. Note that the forward pass is not yet implemented, but this shows the API.

use rustane::{Layer, MultiHeadAttentionBuilder, SelfAttentionBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Attention Example ===\n");

    // Example 1: Multi-head attention with custom settings
    println!("Example 1: Multi-Head Attention");
    println!("------------------------------");

    let mha = MultiHeadAttentionBuilder::new(64, 4)
        .with_name("my_attention")
        .with_head_dim(16)
        .with_causal(true)
        .with_bias(true)
        .build()?;

    println!("Layer name: {}", mha.name());
    println!("Total parameters: {}", mha.num_parameters());
    println!("Input shape: {:?}", mha.input_shape());
    println!("Output shape: {:?}\n", mha.output_shape());

    // Example 2: Self-attention (convenience wrapper)
    println!("Example 2: Self-Attention");
    println!("-------------------------");

    let sa = SelfAttentionBuilder::new(128, 8)
        .with_dropout(0.1)
        .build()?;

    println!("Layer name: {}", sa.name());
    println!("Total parameters: {}", sa.num_parameters());
    println!("Input shape: {:?}", sa.input_shape());
    println!("Output shape: {:?}\n", sa.output_shape());

    // Example 3: Different configurations
    println!("Example 3: Different Configurations");
    println!("-----------------------------------");

    // Small model
    let small = MultiHeadAttentionBuilder::new(32, 2).build()?;
    println!("Small model: {} params", small.num_parameters());

    // Large model
    let large = MultiHeadAttentionBuilder::new(512, 8).build()?;
    println!("Large model: {} params", large.num_parameters());

    // Custom head dimension
    let custom = MultiHeadAttentionBuilder::new(64, 4)
        .with_head_dim(32)
        .build()?;
    println!("Custom: {} params", custom.num_parameters());

    println!("\n=== API Notes ===");
    println!("- MultiHeadAttention: Full control over attention configuration");
    println!("- SelfAttention: Convenience wrapper with causal masking enabled");
    println!("- Builder pattern: Fluent API for configuration");
    println!("- All layers implement the Layer trait for Sequential models");
    println!("\nNote: Forward pass not yet implemented - this demonstrates the API only.");

    Ok(())
}
