//! Example: Full sharded training workflow with synthetic data
//!
//! Demonstrates the complete training pipeline:
//! 1. Synthetic shard creation (mimics real sharded data)
//! 2. ShardedDataLoader discovery and iteration
//! 3. Per-shard DataLoader creation with batching
//! 4. Batch chunking for gradient accumulation
//! 5. Training with accumulated steps
//! 6. Metrics reporting and tracking
//!
//! This example uses synthetic data to be self-contained and fast.
//! The same pipeline works with real data sources (FineWeb, parameter-golf, etc.)

use rustane::data::{Batch, DataLoader, Dataset, JsonlDataset, SequentialSampler};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, Optimizer, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::ConstantScheduler;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("Rustane: Full Sharded Training Workflow");
    println!("======================================\n");

    // Configuration
    let shard_count = 3;
    let samples_per_shard = 16;
    let seq_len = 32;
    let batch_size = 4;
    let max_chunk_tokens = 64; // Will create chunks of (batch_size × seq_len) / max_chunk_tokens
    let steps_per_shard = 2;
    let vocab_size = 256;

    println!("Configuration:");
    println!("  Shards: {}", shard_count);
    println!("  Samples per shard: {}", samples_per_shard);
    println!("  Sequence length: {}", seq_len);
    println!("  Batch size: {}", batch_size);
    println!("  Batch tokens: {}", batch_size * seq_len);
    println!("  Max chunk tokens: {}", max_chunk_tokens);
    println!("  Vocab size: {}\n", vocab_size);

    // Step 1: Create synthetic shards
    println!("Step 1: Creating synthetic shards...");
    let shard_dir = create_synthetic_shards(shard_count, samples_per_shard, seq_len)?;
    println!(
        "  Created {} shards in {}",
        shard_count,
        shard_dir.display()
    );
    println!("  Each shard: ~{} bytes\n", samples_per_shard * seq_len * 4);

    // Step 2: Create model and trainer
    println!("Step 2: Setting up model and trainer...");
    let vocab_size_copy = vocab_size;
    let mut model = SimpleLanguageModel::new(vocab_size_copy);
    let model_param_count = model.param_count();
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;
    println!("  Model parameters: {}", model_param_count);
    println!("  Optimizer: SGD");
    println!("  Scheduler: Constant (LR=0.001)\n");

    // Step 3: Training loop
    println!("Step 3: Training with shard iteration...");
    println!("  Shard │ Batch │ Loss      │ Grad Norm │ LR       │ Chunks");
    println!("  ------|-------|-----------|-----------|----------|--------");

    let mut global_step = 0usize;

    // Iterate over shards
    for shard_idx in 0..shard_count {
        let shard_path = shard_dir.join(format!("shard_{:03}.jsonl", shard_idx));

        // Load shard dataset
        let dataset = JsonlDataset::load(&shard_path)?;
        println!("\n  [Shard {}] Loaded {} samples", shard_idx, dataset.len());

        // Create sampler for this shard
        let sampler = SequentialSampler::new(dataset.len());

        // Create dataloader with batching
        let dataloader = DataLoader::new(dataset, sampler, batch_size)?;
        println!(
            "  [Shard {}] Created dataloader (batch_size={})",
            shard_idx, batch_size
        );

        // Iterate over batches within shard
        let mut batch_idx = 0usize;
        for batch_result in dataloader.iter() {
            if batch_idx >= steps_per_shard {
                break; // Limit steps per shard for demo
            }

            let batch = batch_result?;
            let _shape = batch.shape();

            // Step 4: Chunk batch for gradient accumulation
            let chunks = batch.into_chunks(max_chunk_tokens)?;
            let chunk_count = chunks.len();

            // Step 5: Train with accumulated steps
            let metrics =
                trainer.train_accumulated_steps(chunks.into_iter().map(Ok), chunk_count)?;

            // Step 6: Report metrics
            println!(
                "  {:>6} │ {:>5} │ {:.6} │ {:.6} │ {:.6} │ {}",
                shard_idx,
                batch_idx,
                metrics.loss,
                metrics.grad_norm,
                metrics.learning_rate,
                chunk_count
            );

            batch_idx += 1;
            global_step += 1;
        }
    }

    println!("\n  ✓ Training completed!");
    println!("  Global steps: {}", global_step);
    println!("  Total batches processed: {}", global_step);
    println!("\nStep 4: Pipeline Summary\n");

    println!("Data Pipeline Architecture:");
    println!("  Synthetic Shards (on disk)");
    println!("         ↓");
    println!("  ShardedDataLoader.iter_shards()");
    println!("         ↓");
    println!("  JsonlDataset.load(shard_path)");
    println!("         ↓");
    println!("  DataLoader(dataset, sampler, batch_size)");
    println!("         ↓");
    println!("  Batch {{ tokens, batch_size, seq_len }}");
    println!("         ↓");
    println!("  Batch.into_chunks(max_chunk_tokens)");
    println!("         ↓");
    println!("  Trainer.train_accumulated_steps(chunks, count)");
    println!("         ↓");
    println!("  StepMetrics {{ loss, grad_norm, learning_rate }}");

    println!("\nKey Design Points:");
    println!("  • CPU: Data loading, sampling, batching, chunking");
    println!("  • ANE: Forward/backward passes, loss computation");
    println!("  • Gradient accumulation: Scale loss by 1/chunk_count");
    println!("  • Streaming: Shards loaded one at a time (memory efficient)");
    println!("  • Token-aligned chunks: Respect seq_len boundaries");

    // Cleanup
    println!("\nCleaning up synthetic shards...");
    fs::remove_dir_all(&shard_dir).ok();

    println!("\n✓ Example completed successfully!");
    Ok(())
}

/// Create synthetic shards with random token data
fn create_synthetic_shards(
    shard_count: usize,
    samples_per_shard: usize,
    seq_len: usize,
) -> Result<PathBuf> {
    let shard_dir =
        env::temp_dir().join(format!("rustane-synthetic-shards-{}", std::process::id()));
    fs::create_dir_all(&shard_dir).map_err(|e| rustane::Error::Io(e.to_string()))?;

    for shard_idx in 0..shard_count {
        let shard_path = shard_dir.join(format!("shard_{:03}.jsonl", shard_idx));
        let mut file = File::create(&shard_path).map_err(|e| rustane::Error::Io(e.to_string()))?;

        // Generate synthetic samples for this shard
        for sample_idx in 0..samples_per_shard {
            // Create a sample with seq_len tokens
            // Use a deterministic pattern: (shard * 1000 + sample) as base offset
            let base = (shard_idx * 1000 + sample_idx) as u32;
            let sample: Vec<u32> = (0..seq_len).map(|i| (base + i as u32) % 256).collect();

            let line =
                serde_json::to_string(&sample).map_err(|e| rustane::Error::Other(e.to_string()))?;
            writeln!(file, "{}", line).map_err(|e| rustane::Error::Io(e.to_string()))?;
        }
    }

    Ok(shard_dir)
}

/// Minimal language model for demonstration
///
/// Uses a bigram approach (token → next token distribution):
/// - Each input token indexes a row in the logits table
/// - The model learns P(next_token | current_token)
/// - Suitable for demonstrating training mechanics
pub struct SimpleLanguageModel {
    /// Logits table: [vocab_size × vocab_size]
    logits: Vec<f32>,
    vocab_size: usize,
    /// Cache for backward pass
    last_tokens: Option<Vec<u32>>,
    last_logits: Option<Vec<f32>>,
}

impl SimpleLanguageModel {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            logits: vec![0.0f32; vocab_size * vocab_size],
            vocab_size,
            last_tokens: None,
            last_logits: None,
        }
    }
}

impl Model for SimpleLanguageModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        let tokens = batch.tokens();
        if tokens.is_empty() {
            return Err(rustane::Error::Other(
                "batch must contain at least 1 token".to_string(),
            ));
        }

        // Store for backward pass
        self.last_tokens = Some(tokens.to_vec());

        // Produce logits for each position (except the last which is target)
        let num_positions = tokens.len().saturating_sub(1).max(1);
        let mut logits = Vec::with_capacity(num_positions * self.vocab_size);

        for &token in tokens.iter().take(num_positions) {
            let token_idx = (token as usize) % self.vocab_size;
            let row_start = token_idx * self.vocab_size;
            logits.extend_from_slice(&self.logits[row_start..row_start + self.vocab_size]);
        }

        self.last_logits = Some(logits.clone());

        ANETensor::from_fp32(logits, vec![num_positions, self.vocab_size])
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        let tokens = self
            .last_tokens
            .as_ref()
            .ok_or_else(|| rustane::Error::Other("no tokens cached".to_string()))?;
        let logits = self
            .last_logits
            .as_ref()
            .ok_or_else(|| rustane::Error::Other("no logits cached".to_string()))?;

        let num_positions = tokens.len().saturating_sub(1).max(1);
        let mut grads = vec![0.0f32; self.logits.len()];

        // Compute cross-entropy gradients
        for pos in 0..num_positions {
            if pos + 1 >= tokens.len() {
                break;
            }

            let logits_at_pos = &logits[pos * self.vocab_size..(pos + 1) * self.vocab_size];
            let max_logit = logits_at_pos
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits_at_pos
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f32 = exp_logits.iter().sum();

            let input_idx = (tokens[pos] as usize) % self.vocab_size;
            let target_idx = (tokens[pos + 1] as usize) % self.vocab_size;
            let row_start = input_idx * self.vocab_size;

            for pred_idx in 0..self.vocab_size {
                let softmax_prob = exp_logits[pred_idx] / sum_exp;
                let target_prob = if pred_idx == target_idx { 1.0 } else { 0.0 };
                let grad = (softmax_prob - target_prob) / num_positions as f32;
                grads[row_start + pred_idx] += grad;
            }
        }

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.logits
    }

    fn param_count(&self) -> usize {
        self.logits.len()
    }
}

/// Simple SGD optimizer for demonstration
struct SimpleOptimizer {
    learning_rate: f32,
}

impl SimpleOptimizer {
    fn new(lr: f32) -> Self {
        Self { learning_rate: lr }
    }
}

impl Optimizer for SimpleOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_language_model_creation() {
        let model = SimpleLanguageModel::new(256);
        assert_eq!(model.param_count(), 256 * 256);
        assert_eq!(model.vocab_size, 256);
    }

    #[test]
    fn test_synthetic_shards_creation() -> Result<()> {
        let shard_dir = create_synthetic_shards(2, 4, 16)?;
        assert!(shard_dir.exists());

        // Verify shards exist
        let shard_0 = shard_dir.join("shard_000.jsonl");
        let shard_1 = shard_dir.join("shard_001.jsonl");
        assert!(shard_0.exists());
        assert!(shard_1.exists());

        // Cleanup
        fs::remove_dir_all(&shard_dir).ok();
        Ok(())
    }

    #[test]
    fn test_optimizer_step() -> Result<()> {
        let mut optimizer = SimpleOptimizer::new(0.1);
        let grads = vec![0.1, 0.2, 0.3];
        let mut params = vec![1.0, 2.0, 3.0];

        optimizer.step(&grads, &mut params, 0.1)?;

        // params -= lr * grads
        assert!((params[0] - (1.0 - 0.1 * 0.1)).abs() < 1e-6);
        assert!((params[1] - (2.0 - 0.1 * 0.2)).abs() < 1e-6);
        assert!((params[2] - (3.0 - 0.1 * 0.3)).abs() < 1e-6);

        Ok(())
    }
}
