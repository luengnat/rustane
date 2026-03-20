//! Integration tests for sharded training workflows.
//!
//! These tests cover the full path from shard discovery to batch chunking
//! and gradient-accumulated training.

use rustane::data::{Batch, DataLoader, Dataset, JsonlDataset, RandomSampler, ShardedDataLoader, ShardConfig};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, Optimizer, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::ConstantScheduler;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// Minimal synthetic dataset used for integration tests.
struct TestDataset {
    samples: Vec<Vec<u32>>,
}

impl TestDataset {
    fn new(num_samples: usize, seq_len: usize, vocab_size: u32) -> Self {
        let samples = (0..num_samples)
            .map(|_| (0..seq_len).map(|i| (i as u32) % vocab_size).collect())
            .collect();
        Self { samples }
    }
}

impl Dataset for TestDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        self.samples.get(idx).cloned().ok_or_else(|| {
            rustane::Error::InvalidParameter(format!("Index {} out of bounds", idx))
        })
    }
}

/// Minimal model that returns deterministic gradients.
struct TestModel {
    params: Vec<f32>,
}

impl TestModel {
    fn new(param_count: usize) -> Self {
        Self {
            params: vec![0.01; param_count],
        }
    }
}

impl Model for TestModel {
    fn forward(&mut self, _batch: &Batch) -> Result<ANETensor> {
        ANETensor::from_fp32(vec![0.0; 16], vec![16])
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

/// Minimal optimizer for testing.
struct SimpleOptimizer;

impl Optimizer for SimpleOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        Ok(())
    }
}

fn write_jsonl_shard(path: &Path, samples: &[Vec<u32>]) -> Result<()> {
    let mut contents = String::new();
    for sample in samples {
        contents.push_str(&serde_json::to_string(sample).unwrap());
        contents.push('\n');
    }
    fs::write(path, contents).map_err(|e| rustane::Error::Io(e.to_string()))?;
    Ok(())
}

#[test]
fn test_sharded_loader_discovers_temp_files() -> Result<()> {
    let dir = tempdir().map_err(|e| rustane::Error::Io(e.to_string()))?;
    let shard_a = dir.path().join("shard_000.jsonl");
    let shard_b = dir.path().join("shard_001.jsonl");

    write_jsonl_shard(&shard_a, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])?;
    write_jsonl_shard(&shard_b, &[vec![9, 10, 11, 12]])?;

    let pattern = format!("{}/shard_*.jsonl", dir.path().display());
    let config = ShardConfig::new(pattern, 256)?;
    let loader = ShardedDataLoader::new(&config)?;

    assert_eq!(loader.shard_count(), 2);

    let shards: Vec<_> = loader.iter_shards().collect();
    assert_eq!(shards.len(), 2);
    assert_eq!(shards[0].0, 0);
    assert_eq!(shards[1].0, 1);
    assert_eq!(shards[0].1, shard_a);
    assert_eq!(shards[1].1, shard_b);

    Ok(())
}

#[test]
fn test_batch_chunking_respects_seq_len() -> Result<()> {
    let batch = Batch::new(vec![1u32; 96], 3, 32)?;
    let chunks = batch.into_chunks(64)?;

    assert!(!chunks.is_empty());
    assert!(chunks.len() <= 2);
    for chunk in &chunks {
        assert_eq!(chunk.tokens.len() % 32, 0);
        assert_eq!(chunk.seq_len(), 32);
    }

    Ok(())
}

#[test]
fn test_dataloader_with_synthetic_dataset() -> Result<()> {
    let dataset = TestDataset::new(4, 8, 256);
    let sampler = RandomSampler::new(dataset.len(), 7);
    let dataloader = DataLoader::new(dataset, sampler, 2)?;

    let mut iter = dataloader.iter();
    let batch = iter
        .next()
        .ok_or_else(|| rustane::Error::Other("expected a batch".to_string()))??;

    assert_eq!(batch.shape(), (2, 8));
    assert_eq!(batch.tokens().len(), 16);

    Ok(())
}

#[test]
fn test_train_accumulated_steps_from_shard_batches() -> Result<()> {
    let dir = tempdir().map_err(|e| rustane::Error::Io(e.to_string()))?;
    let shard_a = dir.path().join("shard_000.jsonl");
    let shard_b = dir.path().join("shard_001.jsonl");

    write_jsonl_shard(
        &shard_a,
        &[
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9, 10, 11],
            vec![12, 13, 14, 15],
        ],
    )?;
    write_jsonl_shard(&shard_b, &[vec![16, 17, 18, 19]])?;

    let dataset = JsonlDataset::load(&shard_a)?;
    let sampler = RandomSampler::new(dataset.len(), 42);
    let dataloader = DataLoader::new(dataset, sampler, 4)?;

    let mut model = TestModel::new(16);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer)
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    let batch = dataloader
        .iter()
        .next()
        .ok_or_else(|| rustane::Error::Other("expected a batch".to_string()))??;

    let chunks = batch.into_chunks(8)?;
    assert_eq!(chunks.len(), 2);

    let metrics = trainer.train_accumulated_steps(chunks.into_iter().map(Ok), 2)?;

    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
    assert!(metrics.learning_rate > 0.0);
    assert_eq!(metrics.step, 0);

    Ok(())
}
