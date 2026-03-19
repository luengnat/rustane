//! Sequential Model Example
//!
//! This example demonstrates how to build a simple neural network
//! using the Sequential model API.

use rustane::{
    layers::{Linear, ReLU, Sequential, SharedLayer},
    Model,
};

fn main() -> rustane::Result<()> {
    println!("=== Sequential Model Example ===\n");

    // Create a shared activation layer
    let shared_relu = SharedLayer::new(ReLU::new());

    // Build a sequential model
    let model = Sequential::new("simple_mlp")
        .add(Box::new(Linear::new(10, 32).build()?))
        .add_shared(shared_relu.clone())
        .add(Box::new(Linear::new(32, 10).build()?))
        .add(Box::new(ReLU::new()))
        .build();

    println!("Model: {}", model.name());
    println!("Number of layers: {}", model.len());
    println!("Total parameters: {}", model.num_parameters());
    println!("Trainable parameters: {}", model.num_trainable_parameters());

    println!("\nModel Summary:");
    println!("{}", model.summary());

    // Demonstrate layer freezing
    println!("\n=== Layer Freezing ===");
    let mut model = model;
    model.freeze_layer(0)?;
    println!("After freezing layer 0:");
    println!("Trainable parameters: {}", model.num_trainable_parameters());

    model.unfreeze_layer(0)?;
    println!("After unfreezing layer 0:");
    println!("Trainable parameters: {}", model.num_trainable_parameters());

    // Demonstrate parameter sharing
    println!("\n=== Parameter Sharing ===");
    let model2 = Sequential::new("model2").add_shared(shared_relu).build();

    println!("Model 2 uses the same shared ReLU layer as Model 1");

    Ok(())
}
