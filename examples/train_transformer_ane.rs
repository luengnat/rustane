//! Complete training example for TransformerANE
//!
//! Demonstrates the full training pipeline:
//! 1. Configuration setup
//! 2. Synthetic dataset creation
//! 3. Model initialization
//! 4. Trainer setup with optimizer and scheduler
//! 5. Training loop with metrics reporting
//!
//! This example shows how to integrate all components of the training system
//! and can serve as a template for implementing custom training loops.

use rustane::{
    data::{DataLoader, Dataset, SequentialDataset, SequentialSampler},
    training::{ConstantScheduler, CrossEntropyLoss, Optimizer, TrainerBuilder},
    training::{TransformerANE, TransformerConfig},
    Result,
};

/// Simple SGD optimizer for demonstration
///
/// This is a minimal optimizer implementation. Production systems typically use:
/// - Adam (momentum + adaptive learning rates)
/// - AdamW (Adam + weight decay)
/// - SGD with momentum
struct SimpleOptimizer;

impl SimpleOptimizer {
    /// Create a new SGD optimizer
    fn new(_lr: f32) -> Self {
        SimpleOptimizer
    }
}

impl Optimizer for SimpleOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        for (param, &grad) in params.iter_mut().zip(grads.iter()) {
            // SGD update: params = params - lr * grads
            *param -= lr * grad;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("Rustane: TransformerANE Training Example");
    println!("=======================================\n");

    // ============================================================================
    // STEP 1: Configuration
    // ============================================================================
    println!("STEP 1: Configuration");
    println!("---------------------");

    // Create a small transformer configuration for demonstration
    // Parameters: vocab_size=4096, seq_len=256, dim=768, n_heads=8, n_layers=6, hidden_dim=512
    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)?;

    println!("  Vocab size:    {}", config.vocab_size);
    println!("  Sequence len:  {}", config.seq_len);
    println!("  Model dim:     {}", config.dim);
    println!("  Heads:         {}", config.n_heads);
    println!("  Layers:        {}", config.n_layers);
    println!("  Hidden dim:    {}", config.hidden_dim);
    println!(
        "  Total params:  {:.2}M\n",
        config.param_count() as f32 / 1_000_000.0
    );

    // ============================================================================
    // STEP 2: Dataset and DataLoader
    // ============================================================================
    println!("STEP 2: Dataset and DataLoader");
    println!("------------------------------");

    // Create synthetic dataset: 8 samples of 256 tokens each
    let mut samples = Vec::new();
    for sample_id in 0..8 {
        let sample: Vec<u32> = (0..256)
            .map(|i| (sample_id * 1000 + i) as u32 % 4096)
            .collect();
        samples.push(sample);
    }

    let dataset = SequentialDataset::new(samples);
    let dataset_len = dataset.len();
    println!("  Created dataset with {} samples", dataset_len);
    println!("  Tokens per sample: 256");

    // Create sampler: sequential iteration through dataset indices
    let sampler = SequentialSampler::new(dataset_len);

    // Create dataloader: batch samples into groups of 2
    let dataloader = DataLoader::new(dataset, sampler, 2)?;
    println!("  DataLoader configured with batch_size=2");
    println!(
        "  Total batches: {} (8 samples / 2 batch_size)\n",
        dataset_len / 2
    );

    // ============================================================================
    // STEP 3: Model Initialization
    // ============================================================================
    println!("STEP 3: Model Initialization");
    println!("----------------------------");

    let mut model = TransformerANE::new(&config)?;
    println!(
        "  TransformerANE initialized: {:.2}M parameters",
        config.param_count() as f32 / 1_000_000.0
    );
    println!("  Parameters initialized with small random values (0.01)\n");

    // ============================================================================
    // STEP 4: Trainer Setup
    // ============================================================================
    println!("STEP 4: Trainer Setup");
    println!("---------------------");

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    println!("  Optimizer:       SGD (learning_rate=0.001)");
    println!("  Scheduler:       Constant");
    println!("  Loss function:   Cross-Entropy\n");

    // ============================================================================
    // STEP 5: Training Loop
    // ============================================================================
    println!("STEP 5: Training Loop");
    println!("---------------------\n");

    println!("Batch │ Loss      │ Grad Norm │ Learning Rate │");
    println!("------|-----------|-----------|---------------|");

    let mut total_batches = 0;
    let mut total_loss = 0.0;
    let mut total_grad_norm = 0.0;

    for batch_result in dataloader.iter() {
        let batch = batch_result?;

        // Run a single training step:
        // 1. Forward pass: compute logits from input tokens
        // 2. Loss computation: compute cross-entropy loss from logits and targets
        // 3. Backward pass: compute gradients using cached activations
        // 4. Optimizer step: update parameters using gradients and learning rate
        let metrics = trainer.train_step(&batch)?;

        // Print metrics for this step
        println!(
            "{:>5} │ {:.6} │ {:.8} │ {:.6}      │",
            total_batches, metrics.loss, metrics.grad_norm, metrics.learning_rate
        );

        total_batches += 1;
        total_loss += metrics.loss;
        total_grad_norm += metrics.grad_norm;

        // For demonstration, stop after 4 batches
        if total_batches >= 4 {
            break;
        }
    }

    println!();

    // ============================================================================
    // STEP 6: Summary
    // ============================================================================
    println!("STEP 6: Summary");
    println!("---------------");

    let avg_loss = total_loss / total_batches as f32;
    let avg_grad_norm = total_grad_norm / total_batches as f32;

    println!("✓ Training completed successfully!");
    println!();
    println!("  Total batches processed:    {}", total_batches);
    println!("  Average loss:               {:.6}", avg_loss);
    println!("  Average gradient norm:      {:.8}", avg_grad_norm);
    println!();
    println!("Pipeline demonstration:");
    println!("  ✓ Configuration creation (TransformerConfig)");
    println!("  ✓ Data pipeline (SequentialDataset + DataLoader)");
    println!("  ✓ Model initialization (TransformerANE)");
    println!("  ✓ Trainer setup (Optimizer + Scheduler + Loss)");
    println!("  ✓ Training loop execution");
    println!("  ✓ Metrics tracking and reporting");
    println!();
    println!("For production training:");
    println!("  • Implement custom Optimizer (Adam, AdamW with momentum)");
    println!("  • Use WarmupCosineScheduler or WarmupLinearScheduler");
    println!("  • Load real tokenized data via JsonlDataset or TextDataset");
    println!("  • Add checkpointing to save/restore model state");
    println!("  • Implement learning rate warmup (5-10% of total steps)");
    println!("  • Monitor metrics with loss curves and gradient health");

    Ok(())
}
