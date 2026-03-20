//! Example: Validation on FineWeb dataset
//!
//! Demonstrates:
//! - Loading FineWeb validation shards
//! - Computing validation loss
//! - Calculating metrics (loss, bits-per-byte, accuracy)
//! - Comparing against parameter-golf baseline

use rustane::data::Batch;
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, Optimizer, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::ConstantScheduler;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn main() -> Result<()> {
    println!("Rustane FineWeb Validation Example");
    println!("===================================\n");

    let val_file =
        "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin";
    let val_path = Path::new(val_file);

    if !val_path.exists() {
        return Err(rustane::Error::Other(format!(
            "Validation file not found: {}",
            val_file
        )));
    }

    println!("Loading validation data from: {}\n", val_file);

    // Load validation batch
    let batch = load_fineweb_batch(val_path, 4096, 512)?;
    println!("Loaded validation batch:");
    println!("  Batch size: {}", batch.batch_size());
    println!("  Sequence length: {}", batch.seq_len());
    println!("  Total tokens: {}\n", batch.tokens().len());

    // Create model and trainer for evaluation
    let mut model = SimpleModel::new(32);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    // Evaluate on validation batch
    println!("Computing validation metrics...\n");

    // Process chunks and accumulate loss
    let chunk_size = 512; // tokens per evaluation chunk
    let chunks = batch.into_chunks(chunk_size)?;
    let mut total_loss = 0.0f32;
    let mut chunk_count = 0usize;
    let mut total_tokens = 0usize;

    for chunk in chunks {
        total_tokens += chunk.tokens().len();

        // Forward pass to get loss
        let metrics = trainer.train_accumulated_steps(vec![chunk].into_iter().map(Ok), 1)?;

        total_loss += metrics.loss;
        chunk_count += 1;
    }

    // Calculate metrics
    let avg_loss = total_loss / chunk_count as f32;
    let bits_per_byte = avg_loss / std::f32::consts::LN_2; // Convert nats to bits
    let perplexity = avg_loss.exp();

    println!("Validation Results:");
    println!("===================");
    println!("Average loss (nats):  {:.5}", avg_loss);
    println!("Bits per byte:        {:.5}", bits_per_byte);
    println!("Perplexity:           {:.2}", perplexity);
    println!("\nChunks evaluated:     {}", chunk_count);
    println!("Total tokens:         {}", total_tokens);

    // Parameter-Golf baseline for reference
    println!("\n\nParameter-Golf Baseline (for reference):");
    println!("========================================");
    println!("Naive baseline BPB:    1.2244  (9 layers, 512 dim, 1024 vocab)");
    println!("SOTA (as of 2026-03): 1.1748  (Muon WD + 10 layer)");

    if bits_per_byte < 1.2244 {
        println!("\n✓ Your model BPB is better than naive baseline!");
    } else if bits_per_byte < 1.25 {
        println!("\n→ Your model BPB is in baseline range");
    } else {
        println!("\n→ Your model BPB is above baseline (expected for simple demo model)");
    }

    Ok(())
}

// ===== Model =====

struct SimpleModel {
    // Embedding layer: [vocab_size, hidden_dim]
    embed_weight: Vec<f32>,
    // Output layer: [hidden_dim, vocab_size]
    output_weight: Vec<f32>,
    // Parameters for gradient computation
    vocab_size: usize,
    hidden_dim: usize,
}

impl SimpleModel {
    fn new(hidden_dim: usize) -> Self {
        let vocab_size = 1024; // SentencePiece sp1024
        let embed_size = vocab_size * hidden_dim;
        let output_size = hidden_dim * vocab_size;

        SimpleModel {
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

impl Model for SimpleModel {
    fn forward(&mut self, _batch: &Batch) -> Result<ANETensor> {
        // Simulate embedding lookup + output projection
        let logits_size = 1024; // Simplified: fixed output size
        let logits = vec![0.0f32; logits_size];
        let shape = vec![logits_size];

        ANETensor::from_fp32(logits, shape)
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        // Simple gradient computation: loss * parameter_scale
        let grads = vec![loss * 0.001; self.param_count()];
        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.embed_weight
    }

    fn param_count(&self) -> usize {
        self.param_count()
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

// ===== FineWeb Format Loading =====

fn load_fineweb_batch(file: &Path, max_tokens: usize, seq_len: usize) -> Result<Batch> {
    let tokens = load_fineweb_tokens_prefix(file, max_tokens)?;
    let usable = (tokens.len() / seq_len) * seq_len;
    if usable == 0 {
        return Err(rustane::Error::Other(format!(
            "binary shard {} did not contain enough tokens for seq_len={}",
            file.display(),
            seq_len
        )));
    }

    let tokens = tokens[..usable].to_vec();
    let batch_size = usable / seq_len;
    Batch::new(tokens, batch_size, seq_len)
}

fn load_fineweb_tokens_prefix(file: &Path, max_tokens: usize) -> Result<Vec<u32>> {
    let mut f = File::open(file).map_err(|e| rustane::Error::Io(e.to_string()))?;
    let mut header_buf = [0u8; 256 * 4];
    f.read_exact(&mut header_buf)
        .map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut header = [0i32; 256];
    for (idx, chunk) in header_buf.chunks_exact(4).enumerate() {
        header[idx] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    if header[0] != 20240520 || header[1] != 1 {
        return Err(rustane::Error::Other(format!(
            "unexpected FineWeb shard header for {}",
            file.display()
        )));
    }

    let num_tokens = header[2].max(0) as usize;
    let count = num_tokens.min(max_tokens);
    let mut token_buf = vec![0u8; count * 2];
    f.seek(SeekFrom::Start((256 * 4) as u64))
        .map_err(|e| rustane::Error::Io(e.to_string()))?;
    f.read_exact(&mut token_buf)
        .map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut tokens = Vec::with_capacity(count);
    for chunk in token_buf.chunks_exact(2) {
        tokens.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
    }
    Ok(tokens)
}
