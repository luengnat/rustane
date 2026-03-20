//! Integration tests for Trainer with realistic training loop

use rustane::data::{DataLoader, Dataset, RandomSampler};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::ConstantScheduler;

/// Minimal model for testing: two linear layers
///
/// Architecture:
/// - Input: flattened batch tokens
/// - Hidden: 64-dim dense layer
/// - Output: 256 predictions (vocab size)
struct ToyModel {
    w1: Vec<f32>,                  // [256, 64]
    w2: Vec<f32>,                  // [64, 256]
    last_input: Option<Vec<f32>>,  // Cache for backward pass
    last_hidden: Option<Vec<f32>>, // Cache for backward pass
}

impl ToyModel {
    fn new() -> Self {
        let vocab_size = 256;
        let hidden_size = 64;
        let w1_size = vocab_size * hidden_size;
        let w2_size = hidden_size * vocab_size;

        ToyModel {
            w1: vec![0.01; w1_size],
            w2: vec![0.01; w2_size],
            last_input: None,
            last_hidden: None,
        }
    }

    fn forward_linear(input: &[f32], weights: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0; out_dim];
        for i in 0..out_dim {
            let mut sum = 0.0;
            for j in 0..in_dim {
                sum += input[j] * weights[i * in_dim + j];
            }
            output[i] = sum;
        }
        output
    }
}

impl Model for ToyModel {
    fn forward(&mut self, batch: &rustane::Batch) -> Result<ANETensor> {
        // Flatten batch tokens to f32
        let input = batch.tokens().iter().map(|&x| x as f32).collect::<Vec<_>>();

        // Hidden layer: ReLU activation
        let hidden = Self::forward_linear(&input, &self.w1, batch.tokens().len(), 64);
        let hidden = hidden.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();

        // Output layer
        let output = Self::forward_linear(&hidden, &self.w2, 64, 256);

        self.last_input = Some(input);
        self.last_hidden = Some(hidden);

        // Return dummy ANETensor (in production, would be real ANE tensor)
        Ok(ANETensor::from_fp32(output, vec![256])?)
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        let _hidden = self
            .last_hidden
            .as_ref()
            .ok_or_else(|| rustane::Error::Other("hidden layer not cached".to_string()))?;

        let _input = self
            .last_input
            .as_ref()
            .ok_or_else(|| rustane::Error::Other("input not cached".to_string()))?;

        // Simple gradient computation: scale by loss
        let mut grads = vec![0.0; self.w1.len() + self.w2.len()];

        // Gradients for w1
        for i in 0..self.w1.len() {
            grads[i] = loss * 0.001;
        }

        // Gradients for w2
        for i in 0..self.w2.len() {
            grads[self.w1.len() + i] = loss * 0.001;
        }

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        // Return w1 parameters
        &mut self.w1
    }

    fn param_count(&self) -> usize {
        self.w1.len() + self.w2.len()
    }
}

/// Synthetic dataset for testing
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
        self.samples
            .get(idx)
            .cloned()
            .ok_or_else(|| rustane::Error::InvalidParameter(format!("Index {} out of bounds", idx)))
    }
}

/// Minimal optimizer for testing
struct SimpleOptimizer {
    _lr: f32,
}

impl SimpleOptimizer {
    fn new(lr: f32) -> Self {
        SimpleOptimizer { _lr: lr }
    }
}

// Implement rustane's Optimizer trait
impl rustane::training::trainer::Optimizer for SimpleOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        Ok(())
    }
}

#[test]
fn test_toy_model_creation() -> Result<()> {
    let model = ToyModel::new();
    assert_eq!(model.param_count(), 256 * 64 + 64 * 256);
    Ok(())
}

#[test]
fn test_toy_dataset_creation() -> Result<()> {
    let dataset = ToyDataset::new(4, 8, 256);
    assert_eq!(dataset.len(), 4);
    assert_eq!(dataset.get(0)?.len(), 8);
    Ok(())
}

#[test]
fn test_trainer_builder_incomplete() -> Result<()> {
    let mut model = ToyModel::new();
    let builder = TrainerBuilder::new(&mut model);

    // Should fail - missing optimizer, scheduler, and loss_fn
    assert!(builder.build().is_err());
    Ok(())
}

#[test]
fn test_trainer_builder_with_optimizer_only() -> Result<()> {
    let mut model = ToyModel::new();
    let builder = TrainerBuilder::new(&mut model).with_optimizer(SimpleOptimizer::new(0.001));

    // Should fail - missing scheduler and loss_fn
    assert!(builder.build().is_err());
    Ok(())
}

#[test]
fn test_trainer_builder_complete() -> Result<()> {
    let mut model = ToyModel::new();

    // Build complete trainer
    let _trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    Ok(())
}

#[test]
fn test_single_training_step() -> Result<()> {
    let mut model = ToyModel::new();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    // Create a simple batch
    let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let batch = rustane::Batch::new(tokens, 2, 4)?;

    // Perform one training step
    let metrics = trainer.train_step(&batch)?;

    // Verify metrics are valid
    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
    assert!(metrics.learning_rate > 0.0);
    assert_eq!(metrics.step, 0);

    Ok(())
}

#[test]
fn test_multiple_training_steps() -> Result<()> {
    let mut model = ToyModel::new();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    // Create batches
    let batch1 = rustane::Batch::new(vec![0, 1, 2, 3, 4, 5, 6, 7], 2, 4)?;
    let batch2 = rustane::Batch::new(vec![8, 9, 10, 11, 12, 13, 14, 15], 2, 4)?;

    // Train for 2 steps
    let metrics1 = trainer.train_step(&batch1)?;
    assert_eq!(metrics1.step, 0);

    let metrics2 = trainer.train_step(&batch2)?;
    assert_eq!(metrics2.step, 1);

    Ok(())
}

#[test]
fn test_dataloader_with_toy_dataset() -> Result<()> {
    let dataset = ToyDataset::new(4, 8, 256);
    let sampler = RandomSampler::new(dataset.len(), 42);
    let dataloader = DataLoader::new(dataset, sampler, 2)?;

    let mut batch_count = 0;
    for batch_result in dataloader.iter() {
        let batch = batch_result?;
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.seq_len(), 8);
        batch_count += 1;
    }

    assert_eq!(batch_count, 2); // 4 samples / batch_size 2
    Ok(())
}

#[test]
fn test_integration_dataloader_trainer() -> Result<()> {
    let mut model = ToyModel::new();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    // Create dataset and dataloader
    let dataset = ToyDataset::new(4, 8, 256);
    let sampler = RandomSampler::new(dataset.len(), 42);
    let dataloader = DataLoader::new(dataset, sampler, 2)?;

    // Train on all batches
    let mut step_count = 0;
    for batch_result in dataloader.iter() {
        let batch = batch_result?;
        let _metrics = trainer.train_step(&batch)?;
        step_count += 1;
    }

    assert_eq!(step_count, 2); // 4 samples / batch_size 2
    Ok(())
}
