//! Complete Training Example with ANE Backward Pass
//!
//! This example demonstrates:
//! - Running backward validation suite at startup
//! - Training a TransformerANE model with ANE backward pass
//! - Using ANEGradientAccumulator for gradient accumulation
//! - Monitoring training metrics
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_transformer_ane_full
//! ```

use rustane::data::Batch;
use rustane::layers::backward::validation::quick_validate;
use rustane::training::{
    ANEGradientAccumulator, CrossEntropyLoss, LossFn, Model, TransformerANE, TransformerConfig,
};

fn main() -> rustane::Result<()> {
    println!("========================================");
    println!("  Rustane ANE Training Example");
    println!("  Phase 3: ANE Backward Pass");
    println!("========================================\n");

    // Step 1: Run backward validation suite
    println!("Step 1: Running backward validation suite...");
    let validation_report = quick_validate()?;

    println!("  RMSNorm backward: {}", if validation_report.rmsnorm_passed { "✓ PASS" } else { "✗ FAIL" });
    println!("  Attention backward: {}", if validation_report.attention_passed { "✓ PASS" } else { "✗ FAIL" });
    println!("  FFN backward: {}", if validation_report.ffn_passed { "✓ PASS" } else { "✗ FAIL" });
    println!("  Loss backward: {}", if validation_report.loss_passed { "✓ PASS" } else { "✗ FAIL" });
    println!("  Max relative error: {:.2e}", validation_report.max_relative_error);

    if !validation_report.all_passed() {
        eprintln!("\nValidation failed! Aborting training.");
        for msg in &validation_report.error_messages {
            eprintln!("  Error: {}", msg);
        }
        return Ok(());
    }
    println!("  ✓ All validations passed!\n");

    // Step 2: Create model configuration
    println!("Step 2: Creating model configuration...");
    let config = TransformerConfig::new(
        512,   // vocab_size
        128,   // dim
        256,   // hidden_dim
        4,     // n_heads
        2,     // n_layers
        64,    // seq_len
    )?;
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Dimension: {}", config.dim);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Attention heads: {}", config.n_heads);
    println!("  Layers: {}", config.n_layers);
    println!("  Sequence length: {}", config.seq_len);
    println!("  Parameters: {}\n", config.param_count());

    // Step 3: Initialize model
    println!("Step 3: Initializing TransformerANE model...");
    let mut model = TransformerANE::new(&config)?;
    println!("  ✓ Model initialized\n");

    // Step 4: Create gradient accumulator
    println!("Step 4: Creating ANE gradient accumulator...");
    let mut accumulator = ANEGradientAccumulator::from_config(&config)?;
    println!("  ✓ Accumulator ready ({} parameters)\n", accumulator.num_params());

    // Step 5: Generate synthetic training data
    println!("Step 5: Generating synthetic training data...");
    let num_batches = 10;
    let batch_size = 4;
    let seq_len = 32;
    let mut batches = Vec::with_capacity(num_batches);

    for batch_idx in 0..num_batches {
        let tokens: Vec<u32> = (0..(batch_size * seq_len))
            .map(|i| ((i + batch_idx * 100) % config.vocab_size) as u32)
            .collect();
        let batch = Batch::new(tokens, batch_size, seq_len)?;
        batches.push(batch);
    }
    println!("  Generated {} batches (batch_size={}, seq_len={})\n", num_batches, batch_size, seq_len);

    // Step 6: Training loop
    println!("Step 6: Starting training loop...");
    println!("========================================");

    let loss_fn = CrossEntropyLoss::new();
    let mut total_loss = 0.0f32;

    for (step, batch) in batches.iter().enumerate() {
        // Forward pass
        let output = model.forward(batch)?;

        // Compute loss
        let loss = loss_fn.compute(&output, batch)?;

        // Backward pass using ANE gradient accumulator
        accumulator.reset()?;
        model.backward_on_ane(batch, loss, &mut accumulator)?;

        // Get accumulated gradients
        let grads = accumulator.get_accumulated()?;

        // Apply gradients (simple SGD optimizer step)
        let lr = 0.001f32;
        for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }

        total_loss += loss;

        // Print metrics every few steps
        if step % 2 == 0 {
            println!(
                "  Step {:2}: loss={:.4}, grad_norm={:.4e}, accum_count={}",
                step,
                loss,
                accumulator.max_abs_gradient(),
                accumulator.accumulation_count()
            );
        }
    }

    println!("========================================");
    println!("  Training complete!");
    println!("  Average loss: {:.4}", total_loss / num_batches as f32);
    println!("========================================\n");

    // Step 7: Demonstrate gradient accumulator features
    println!("Step 7: Gradient accumulator features:");
    println!("  Max abs gradient: {:.4e}", accumulator.max_abs_gradient());
    println!("  Is empty: {}", accumulator.is_empty());

    // Reset and verify
    accumulator.reset()?;
    println!("  After reset - Is empty: {}\n", accumulator.is_empty());

    println!("✓ Example completed successfully!");

    Ok(())
}
