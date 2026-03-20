//! Example: Training a toy model with the Trainer
//!
//! Demonstrates:
//! - Creating a model, dataset, and dataloader
//! - Building a Trainer with all components
//! - Running a training loop
//! - Monitoring loss progression

use rustane::data::{Dataset, RandomSampler, Sampler};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, Optimizer, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::data::Batch;
use rustane::ConstantScheduler;

fn main() -> Result<()> {
    println!("Rustane Trainer Example");
    println!("======================\n");

    // 1. Create dataset
    let dataset = ToyDataset::new(100, 32, 512); // 100 samples, 32 tokens each, 512 vocab
    println!("Created dataset with {} samples", dataset.len());

    // 2. Create sampler
    let sampler = RandomSampler::new(dataset.len(), 42);
    println!("Created sampler with random seed\n");

    // 3. Create model
    let mut model = ToyModel::new(512);
    println!("Created toy model with {} parameters\n", model.param_count());

    // 4. Build trainer
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    println!("Trainer built successfully\n");

    // 5. Training loop
    println!("Starting training loop (10 steps):");
    println!("Step | Loss    | Grad Norm | Learning Rate");
    println!("-----|---------|-----------|---------------");

    // Create dataloader and iterate
    let mut sampler = RandomSampler::new(dataset.len(), 42);
    let indices = sampler.sample();
    
    for step in 0..10 {
        // Get batch (batch_size=2, seq_len=32)
        let batch_size = 2;
        let seq_len = 32;
        let mut batch_tokens = Vec::new();
        
        // Collect tokens for this batch
        for i in 0..batch_size {
            if let Ok(sample) = dataset.get(indices[step * batch_size + i]) {
                batch_tokens.extend_from_slice(&sample);
            }
        }
        
        let batch = Batch::new(batch_tokens, batch_size, seq_len)?;

        // Single training step
        let metrics = trainer.train_step(&batch)?;

        println!(
            "{:4} | {:.5} | {:.5}    | {:.6}",
            metrics.step, metrics.loss, metrics.grad_norm, metrics.learning_rate
        );
    }

    println!("\n✓ Training completed successfully!");
    Ok(())
}

// ===== Model Implementation =====

struct ToyModel {
    params: Vec<f32>,
}

impl ToyModel {
    fn new(vocab_size: usize) -> Self {
        ToyModel {
            params: vec![0.01; vocab_size * 2], // Simplified: just 2x vocab_size
        }
    }
}

impl Model for ToyModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        // Placeholder: Create a dummy logits tensor [batch_size, vocab_size]
        let (batch_size, _seq_len) = batch.shape();
        let logits_data = vec![0.0f32; batch_size * self.params.len()];
        ANETensor::from_fp32(logits_data, vec![batch_size, self.params.len()])
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        Ok(self.params.iter().map(|_| loss * 0.001).collect())
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn param_count(&self) -> usize {
        self.params.len()
    }
}

// ===== Supporting Types =====

struct ToyDataset {
    samples: Vec<Vec<u32>>,
}

impl ToyDataset {
    fn new(num_samples: usize, seq_len: usize, vocab_size: u32) -> Self {
        let samples = (0..num_samples)
            .map(|_| (0..seq_len).map(|i| (i as u32) % vocab_size).collect())
            .collect();
        ToyDataset { samples }
    }
}

impl Dataset for ToyDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        if idx >= self.samples.len() {
            return Err(rustane::Error::InvalidParameter(
                format!("Index {} out of bounds", idx),
            ));
        }
        Ok(self.samples[idx].clone())
    }
}

// ===== Optimizer Implementation =====

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
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        Ok(())
    }
}
