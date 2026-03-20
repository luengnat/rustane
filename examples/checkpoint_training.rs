//! Model Checkpointing and Training Resumption Example
//!
//! This example demonstrates how to:
//! 1. Train a model and save checkpoints periodically
//! 2. Load a checkpoint and resume training
//! 3. Validate checkpoint integrity
//!
//! Run this example:
//! ```sh
//! cargo run --example checkpoint_training
//! ```

use rustane::data::Batch;
use rustane::TrainingModel;
use rustane::training::{
    Checkpoint, LossScalerState, ModelConfig, OptimizerState, TransformerANE,
    TransformerConfig,
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Checkpointing Demo ===\n");

    // Create a transformer configuration
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;

    // Convert to ModelConfig for checkpointing
    let model_config = ModelConfig {
        vocab_size: config.vocab_size,
        dim: config.dim,
        hidden_dim: config.hidden_dim,
        n_heads: config.n_heads,
        n_layers: config.n_layers,
        seq_len: config.seq_len,
    };

    println!("Model Configuration:");
    println!("  Vocab size: {}", model_config.vocab_size);
    println!("  Hidden dim: {}", model_config.dim);
    println!("  Layers: {}", model_config.n_layers);
    println!("  Parameters: {}\n", config.param_count());

    // Create and initialize model
    let mut model = TransformerANE::new(&config)?;

    // Simulate training step 1
    println!("--- Training Step 1 ---");
    let step1_loss = simulate_training_step(&mut model, &config, 1)?;

    // Create checkpoint after step 1
    let weights1 = model.parameters().to_vec();
    let optimizer_state1 = OptimizerState {
        m: Some(vec![0.1f32; weights1.len()]),
        v: Some(vec![0.01f32; weights1.len()]),
        beta1: 0.9,
        beta2: 0.999,
    };
    let loss_scaler_state1 = LossScalerState {
        scale: 256.0,
        steps_since_growth: 100,
    };

    let checkpoint1 = Checkpoint::new(
        weights1,
        1,
        step1_loss,
        0.001,
        model_config.clone(),
    )
    .with_optimizer_state(optimizer_state1)
    .with_loss_scaler_state(loss_scaler_state1);

    // Save checkpoint
    let checkpoint_dir = "checkpoints";
    let checkpoint_path = format!("{}/checkpoint_00001.json", checkpoint_dir);
    checkpoint1.save(&checkpoint_path)?;

    println!("  Loss: {:.4}", step1_loss);
    println!("  Checkpoint saved to: {}\n", checkpoint_path);

    // Simulate training step 2
    println!("--- Training Step 2 ---");
    let step2_loss = simulate_training_step(&mut model, &config, 2)?;

    // Create checkpoint after step 2
    let weights2 = model.parameters().to_vec();
    let checkpoint2 = Checkpoint::new(weights2, 2, step2_loss, 0.0009, model_config);

    let checkpoint_path2 = format!("{}/checkpoint_00002.json", checkpoint_dir);
    checkpoint2.save(&checkpoint_path2)?;

    println!("  Loss: {:.4}", step2_loss);
    println!("  Checkpoint saved to: {}\n", checkpoint_path2);

    // List all checkpoints
    println!("--- Saved Checkpoints ---");
    let entries: Vec<_> = fs::read_dir(checkpoint_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();

    for entry in &entries {
        println!("  {}", entry.file_name().to_string_lossy());
    }
    println!();

    // Demonstrate checkpoint loading and validation
    println!("--- Loading Checkpoint ---");
    let loaded_checkpoint = Checkpoint::load(&checkpoint_path)?;

    println!("  Step: {}", loaded_checkpoint.step);
    println!("  Loss: {:.4}", loaded_checkpoint.loss);
    println!("  Learning rate: {:.6}", loaded_checkpoint.learning_rate);
    println!("  Weights: {} elements", loaded_checkpoint.weights.len());

    if let Some(ref opt_state) = loaded_checkpoint.optimizer_state {
        println!("  Optimizer state:");
        println!("    Beta1: {:.3}", opt_state.beta1);
        println!("    Beta2: {:.3}", opt_state.beta2);
    }

    if let Some(ref scaler_state) = loaded_checkpoint.loss_scaler_state {
        println!("  Loss scaler state:");
        println!("    Scale: {:.2}", scaler_state.scale);
        println!("    Steps since growth: {}", scaler_state.steps_since_growth);
    }
    println!();

    // Validate checkpoint
    let expected_params = config.param_count();
    match loaded_checkpoint.validate(expected_params) {
        Ok(_) => println!("  ✓ Checkpoint validation passed"),
        Err(e) => println!("  ✗ Checkpoint validation failed: {}", e),
    }
    println!();

    // Demonstrate training resumption
    println!("--- Resuming Training from Checkpoint ---");

    // Create a new model and load weights from checkpoint
    let mut new_model = TransformerANE::new(&config)?;
    let loaded_weights = &loaded_checkpoint.weights;

    // Copy weights to new model
    new_model.parameters().copy_from_slice(loaded_weights);

    println!("  Loaded weights from step {}", loaded_checkpoint.step);
    println!("  Model state restored");

    // Resume training from step 3
    let step3_loss = simulate_training_step(&mut new_model, &config, 3)?;
    println!("  Step 3 loss (after resume): {:.4}", step3_loss);

    println!("\n--- Summary ---");
    println!("Checkpointing enables:");
    println!("  1. Periodic model state saving during training");
    println!("  2. Recovery from training interruptions");
    println!("  3. Resumption from any saved step");
    println!("  4. Model deployment from specific checkpoints");
    println!("\nBest practices:");
    println!("  - Save checkpoints every N steps (e.g., 1000)");
    println!("  - Keep last K checkpoints (e.g., last 5)");
    println!("  - Validate checkpoints before long-running training");
    println!("  - Use descriptive checkpoint names for experiments");

    Ok(())
}

/// Simulate a training step
fn simulate_training_step(
    model: &mut TransformerANE,
    config: &TransformerConfig,
    step: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let batch_size = 2;
    let seq_len = config.seq_len;
    let tokens = vec![42u32; batch_size * seq_len];
    let batch = Batch::new(tokens, batch_size, seq_len)?;

    // Forward pass
    let _output = model.forward(&batch)?;

    // Simulate decreasing loss
    let base_loss = 3.0;
    let decay = 0.1 * step as f32;
    Ok((base_loss - decay).max(0.5))
}
