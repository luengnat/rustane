//! Example: Training with Parameter-Golf tokenized data
//!
//! This example demonstrates how to use rustane's sharded training with
//! data tokenized by parameter-golf's SentencePiece tokenizer.
//!
//! Prerequisites:
//! 1. Download parameter-golf FineWeb dataset:
//!    python3 ~/dev/parameter-golf/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
//!
//! 2. The tokenized shards will be at:
//!    ~/dev/parameter-golf/data/datasets/fineweb10B_sp1024/

use rustane::data::{Batch, Dataset, DataLoader, RandomSampler};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::{ConstantScheduler, Optimizer};

fn main() -> Result<()> {
    println!("Rustane + Parameter-Golf Training Example");
    println!("=========================================\n");

    // Configuration matching parameter-golf's standard setup
    let vocab_size = 1024u32;  // SentencePiece sp1024 tokenizer
    let seq_len = 512;         // Sequence length (parameter-golf standard)
    let batch_tokens = 8192;   // Tokens per batch (parameter-golf standard)
    let chunk_tokens = 2048;   // Tokens per accumulation chunk
    let accum_steps = (batch_tokens / chunk_tokens) as u32;
    let num_train_steps = 20;
    let batch_size = batch_tokens / seq_len; // batch_size * seq_len = batch_tokens

    println!("Configuration:");
    println!("  Vocab Size:     {}", vocab_size);
    println!("  Sequence Length: {}", seq_len);
    println!("  Batch Tokens:    {}", batch_tokens);
    println!("  Batch Size:      {}", batch_size);
    println!("  Chunk Tokens:    {}", chunk_tokens);
    println!("  Accum Steps:     {}", accum_steps);
    println!("  Train Steps:     {}\n", num_train_steps);

    // Create dataset and loader
    // In production, this would read from parameter-golf's shard files:
    // let dataset = load_fineweb_shards("/path/to/fineweb10B_sp1024/train_*.jsonl");
    let dataset = SyntheticParameterGolfDataset::new(
        100,  // num_sequences
        seq_len,
        vocab_size,
    );

    let sampler = RandomSampler::new(dataset.len(), 42);
    let mut loader = DataLoader::new(dataset, sampler, batch_size)?;

    // Initialize model and trainer
    let mut model = SimpleParameterGolfModel::new(vocab_size as usize, 256);  // 256-dim hidden
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    println!("Starting training with gradient accumulation");
    println!("Step | Loss    | Grad Norm | LR       | Batches");
    println!("-----|---------|-----------|----------|--------");

    let mut total_batches = 0u64;
    let mut batch_iter = loader.iter();

    for step in 0..num_train_steps {
        // Try to get a batch
        if let Some(Ok(batch)) = batch_iter.next() {
            total_batches += 1;

            // Split into chunks for gradient accumulation
            let chunks = batch.into_chunks(chunk_tokens)?;

            // Train with accumulated gradients
            let metrics = trainer.train_accumulated_steps(
                chunks.into_iter().map(Ok),
                accum_steps,
            )?;

            println!(
                "{:4} | {:.5} | {:.5}    | {:.6} | {}",
                step, metrics.loss, metrics.grad_norm, metrics.learning_rate, total_batches
            );
        } else {
            // Reset iterator if we run out of batches
            let sampler = RandomSampler::new(100, 42);
            let new_loader = DataLoader::new(
                SyntheticParameterGolfDataset::new(100, seq_len, vocab_size),
                sampler,
                batch_size
            )?;
            batch_iter = new_loader.iter();
        }
    }

    println!("\n✓ Training completed!");
    println!("  Total batches processed: {}", total_batches);

    // In a real scenario, you would:
    // 1. Save the model weights
    // 2. Evaluate on parameter-golf's validation set
    // 3. Compute bits-per-byte (BPB) metric
    // 4. Compare against parameter-golf leaderboard

    Ok(())
}

// ===== Model =====

struct SimpleParameterGolfModel {
    // Embedding layer: [vocab_size, hidden_dim]
    embed_weight: Vec<f32>,
    // Output layer: [hidden_dim, vocab_size]
    output_weight: Vec<f32>,
    // Parameters for gradient computation
    vocab_size: usize,
    hidden_dim: usize,
}

impl SimpleParameterGolfModel {
    fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        let embed_size = vocab_size * hidden_dim;
        let output_size = hidden_dim * vocab_size;

        SimpleParameterGolfModel {
            embed_weight: vec![0.01; embed_size],
            output_weight: vec![0.01; output_size],
            vocab_size,
            hidden_dim,
        }
    }

    fn param_count(&self) -> usize {
        self.embed_weight.len() + self.output_weight.len()
    }
}

impl Model for SimpleParameterGolfModel {
    fn forward(&mut self, _batch: &Batch) -> Result<ANETensor> {
        // Simulate embedding lookup + output projection
        // In reality, this would be a transformer layer
        let logits_size = 1024; // Simplified: fixed output size
        let logits = vec![0.0f32; logits_size];

        Ok(ANETensor::from_fp32(logits))
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        // Simple gradient computation: loss * parameter_scale
        let grads = vec![loss * 0.001; self.param_count()];

        // In real implementation:
        // - Compute gradients through embedding and output layers
        // - Use chain rule for proper gradient flow
        // - Handle attention mechanisms (for transformer)

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        // In practice, would concatenate embed_weight and output_weight
        // For this example, just return embed_weight
        &mut self.embed_weight
    }

    fn param_count(&self) -> usize {
        self.param_count()
    }
}

// ===== Dataset =====

struct SyntheticParameterGolfDataset {
    sequences: Vec<Vec<u32>>,
}

impl SyntheticParameterGolfDataset {
    fn new(num_sequences: usize, seq_len: usize, vocab_size: u32) -> Self {
        let sequences = (0..num_sequences)
            .map(|_| {
                (0..seq_len)
                    .map(|i| (i as u32) % vocab_size)
                    .collect()
            })
            .collect();

        SyntheticParameterGolfDataset { sequences }
    }
}

impl Dataset for SyntheticParameterGolfDataset {
    fn len(&self) -> usize {
        self.sequences.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        Ok(self.sequences[idx].clone())
    }
}

// ===== Optimizer =====

struct SimpleOptimizer {
    _lr: f32,
}

impl SimpleOptimizer {
    fn new(lr: f32) -> Self {
        SimpleOptimizer { _lr: lr }
    }
}

impl Optimizer for SimpleOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        // SGD: params -= lr * grads
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        Ok(())
    }
}

// ===== Notes =====

// To use with real parameter-golf data:
//
// 1. Download the SentencePiece tokenizer (sp1024) and vocabulary
// 2. Tokenize the FineWeb dataset using parameter-golf's script
// 3. Create a JsonlDataset or similar to read the sharded .jsonl files
// 4. Use ShardedDataLoader to stream shards from disk
//
// Example with real data:
//   use rustane::data::{ShardedDataLoader, ShardConfig};
//
//   let config = ShardConfig::new(
//       "data/fineweb10B_sp1024/train_*.bin".to_string(),
//       1024,  // vocab_size
//   )?;
//
//   let mut loader = ShardedDataLoader::new(&config)?;
//   for shard in loader.iter_shards()? {
//       // Process shard
//   }
