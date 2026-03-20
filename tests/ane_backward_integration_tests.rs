//! Integration tests for ANE backward execution
//!
//! Tests the end-to-end flow of ANE backward execution:
//! - Forward pass with activation caching
//! - Backward pass with gradient accumulation
//! - Gradient transfer to CPU
//! - Integration with training loop

use rustane::data::{Batch, Dataset, DataLoader, RandomSampler};
use rustane::error::Result;
use rustane::training::{Model, TransformerANE, TransformerConfig, ANEGradientAccumulator};
use rustane::wrapper::ANETensor;

/// Simple dataset for testing
struct TestDataset {
    samples: Vec<Vec<u32>>,
}

impl TestDataset {
    fn new(num_samples: usize, seq_len: usize, vocab_size: u32) -> Self {
        let samples = (0..num_samples)
            .map(|_| (0..seq_len).map(|i| (i as u32) % vocab_size).collect())
            .collect();
        TestDataset { samples }
    }
}

impl Dataset for TestDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        Ok(self.samples[idx].clone())
    }
}

#[test]
fn test_ane_gradient_accumulator_creation() -> Result<()> {
    // Test accumulator creation with various sizes
    let accum_small = ANEGradientAccumulator::new(100)?;
    assert_eq!(accum_small.num_params(), 100);
    assert_eq!(accum_small.steps_completed(), 0);

    let accum_large = ANEGradientAccumulator::new(6_800_000)?;
    assert_eq!(accum_large.num_params(), 6_800_000);

    Ok(())
}

#[test]
fn test_transformer_ane_backward_on_ane() -> Result<()> {
    // Test backward_on_ane() with gradient accumulation
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;
    let mut model = TransformerANE::new(&config)?;

    // Create a batch manually
    let mut tokens = vec![0u32; 128]; // 2 samples of 64 tokens each
    for i in 0..128 {
        tokens[i] = (i % 256) as u32;
    }
    let batch = Batch {
        tokens,
        batch_size: 2,
        seq_len: 64,
    };

    // Forward pass
    let logits = model.forward(&batch)?;
    assert_eq!(logits.shape(), &[2, 63, 256]);

    // Create gradient accumulator
    let mut accum = ANEGradientAccumulator::new(model.param_count())?;

    // Backward pass on ANE (uses CPU backward internally)
    model.backward_on_ane(&batch, 0.5, &mut accum)?;

    // Verify gradients were accumulated
    assert_eq!(accum.steps_completed(), 1);

    let grads = accum.get_accumulated()?;
    assert_eq!(grads.len(), model.param_count());

    // Verify gradients are non-zero (backward pass computed something)
    let grad_sum: f32 = grads.iter().sum();
    assert!(grad_sum != 0.0);

    Ok(())
}

#[test]
fn test_backward_on_ane_batch_consistency() -> Result<()> {
    // Test that backward_on_ane() validates batch consistency
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;
    let mut model = TransformerANE::new(&config)?;

    // Create two different batches
    let mut tokens1 = vec![0u32; 128];
    for i in 0..128 {
        tokens1[i] = (i % 256) as u32;
    }
    let batch1 = Batch {
        tokens: tokens1,
        batch_size: 2,
        seq_len: 64,
    };

    let mut tokens2 = vec![8u32; 128];
    for i in 0..128 {
        tokens2[i] = ((i + 8) % 256) as u32;
    }
    let batch2 = Batch {
        tokens: tokens2,
        batch_size: 2,
        seq_len: 64,
    };

    // Forward pass on batch1
    model.forward(&batch1)?;

    // Try backward with different batch (should fail)
    let mut accum = ANEGradientAccumulator::new(model.param_count())?;
    let result = model.backward_on_ane(&batch2, 0.5, &mut accum);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not match"));

    Ok(())
}

#[test]
fn test_backward_on_ane_requires_forward() -> Result<()> {
    // Test that backward_on_ane() requires forward pass first
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;
    let mut model = TransformerANE::new(&config)?;

    // Create batch
    let mut tokens = vec![0u32; 128];
    for i in 0..128 {
        tokens[i] = (i % 256) as u32;
    }
    let batch = Batch {
        tokens,
        batch_size: 2,
        seq_len: 64,
    };

    // Try backward without forward (should fail)
    let mut accum = ANEGradientAccumulator::new(model.param_count())?;
    let result = model.backward_on_ane(&batch, 0.5, &mut accum);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("forward cache missing"));

    Ok(())
}

#[test]
fn test_gradient_accumulation_multiple_steps() -> Result<()> {
    // Test accumulating gradients from multiple backward passes
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;
    let mut model = TransformerANE::new(&config)?;

    // Create batch
    let mut tokens = vec![0u32; 128];
    for i in 0..128 {
        tokens[i] = (i % 256) as u32;
    }
    let batch = Batch {
        tokens,
        batch_size: 2,
        seq_len: 64,
    };

    let mut accum = ANEGradientAccumulator::new(model.param_count())?;

    // First forward/backward
    model.forward(&batch)?;
    model.backward_on_ane(&batch, 0.5, &mut accum)?;
    assert_eq!(accum.steps_completed(), 1);

    // Second forward/backward (same batch for testing)
    model.forward(&batch)?;
    model.backward_on_ane(&batch, 0.3, &mut accum)?;
    assert_eq!(accum.steps_completed(), 2);

    // Verify gradients accumulated
    let grads = accum.get_accumulated()?;
    assert_eq!(grads.len(), model.param_count());

    Ok(())
}

#[test]
fn test_accumulator_reset_between_steps() -> Result<()> {
    // Test resetting accumulator between training steps
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;
    let mut model = TransformerANE::new(&config)?;

    // Create batch
    let mut tokens = vec![0u32; 128];
    for i in 0..128 {
        tokens[i] = (i % 256) as u32;
    }
    let batch = Batch {
        tokens,
        batch_size: 2,
        seq_len: 64,
    };

    let mut accum = ANEGradientAccumulator::new(model.param_count())?;

    // First training step
    model.forward(&batch)?;
    model.backward_on_ane(&batch, 0.5, &mut accum)?;
    assert_eq!(accum.steps_completed(), 1);

    let grads1 = accum.get_accumulated()?;
    let sum1: f32 = grads1.iter().sum();

    // Reset
    accum.reset()?;
    assert_eq!(accum.steps_completed(), 0);

    // Second training step
    model.forward(&batch)?;
    model.backward_on_ane(&batch, 0.5, &mut accum)?;
    assert_eq!(accum.steps_completed(), 1);

    let grads2 = accum.get_accumulated()?;
    let sum2: f32 = grads2.iter().sum();

    // Gradients should be similar (same batch)
    assert!((sum1 - sum2).abs() < 0.01);

    Ok(())
}

#[test]
fn test_backward_on_ane_vs_backward_consistency() -> Result<()> {
    // Test that backward_on_ane() produces same gradients as backward()
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)?;
    let mut model1 = TransformerANE::new(&config)?;
    let mut model2 = TransformerANE::new(&config)?;

    // Create batch
    let mut tokens1 = vec![0u32; 128];
    for i in 0..128 {
        tokens1[i] = (i % 256) as u32;
    }
    let batch1 = Batch {
        tokens: tokens1,
        batch_size: 2,
        seq_len: 64,
    };

    let mut tokens2 = vec![0u32; 128];
    for i in 0..128 {
        tokens2[i] = (i % 256) as u32;
    }
    let batch2 = Batch {
        tokens: tokens2,
        batch_size: 2,
        seq_len: 64,
    };

    // Method 1: backward_with_batch
    model1.forward(&batch1)?;
    let grads1 = model1.backward_with_batch(&batch1, 0.5)?;

    // Method 2: backward_on_ane
    model2.forward(&batch2)?;
    let mut accum = ANEGradientAccumulator::new(model2.param_count())?;
    model2.backward_on_ane(&batch2, 0.5, &mut accum)?;
    let grads2 = accum.get_accumulated()?;

    // Gradients should be identical (same seed, same computation)
    assert_eq!(grads1.len(), grads2.len());

    // Check most gradients are similar (allowing for minor numerical differences)
    let mut similar_count = 0;
    for (g1, g2) in grads1.iter().zip(grads2.iter()) {
        if (g1 - g2).abs() < 0.001 {
            similar_count += 1;
        }
    }

    // At least 99% of gradients should be similar
    let similarity_ratio = similar_count as f32 / grads1.len() as f32;
    assert!(similarity_ratio > 0.99);

    Ok(())
}
