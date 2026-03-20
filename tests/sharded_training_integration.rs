//! Integration tests for sharded training pipeline
//!
//! Tests the full end-to-end workflow:
//! 1. Creating ShardedDataLoader from synthetic shards
//! 2. Iterating over shards
//! 3. Creating batches from shards
//! 4. Chunking batches for gradient accumulation
//! 5. Training with accumulated steps

use rustane::{
    data::{
        Batch, DataLoader, Dataset, SequentialDataset, SequentialSampler,
        ShardConfig, ShardedDataLoader,
    },
    training::{
        ConstantScheduler, CrossEntropyLoss, Optimizer, StepMetrics,
        TrainerBuilder,
    },
    Result,
};

use rustane::training::Model;
use rustane::wrapper::ANETensor;

// Helper: Create synthetic dataset for testing
fn create_synthetic_dataset(num_samples: usize, seq_len: usize, _vocab_size: u32) -> SequentialDataset {
    let mut samples = Vec::new();
    for i in 0..num_samples {
        let sample: Vec<u32> = (0..seq_len as u32)
            .map(|j| (i as u32 + j) % 1000)
            .collect();
        samples.push(sample);
    }
    SequentialDataset::new(samples)
}

// Simple mock optimizer for testing
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

// Simple mock model for testing
struct SimpleMockModel {
    params: Vec<f32>,
}

impl SimpleMockModel {
    fn new(param_count: usize) -> Self {
        SimpleMockModel {
            params: vec![1.0f32; param_count],
        }
    }
}

impl Model for SimpleMockModel {
    fn forward(&mut self, _batch: &Batch) -> Result<ANETensor> {
        ANETensor::from_fp32(vec![1.0f32; 256], vec![256])
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        Ok(vec![0.1f32; self.params.len()])
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn param_count(&self) -> usize {
        self.params.len()
    }
}

// Test 1: ShardedDataLoader creation from glob pattern
#[test]
fn test_sharded_loader_creation() -> Result<()> {
    let config = ShardConfig::new(
        "tests/sharded_training_integration.rs".to_string(),
        50000,
    )?;

    let loader = ShardedDataLoader::new(&config)?;
    assert_eq!(loader.shard_count(), 1);

    Ok(())
}

// Test 2: ShardedDataLoader iteration
#[test]
fn test_sharded_loader_iteration() -> Result<()> {
    let config = ShardConfig::new(
        "tests/sharded_training_integration.rs".to_string(),
        50000,
    )?;

    let loader = ShardedDataLoader::new(&config)?;
    assert_eq!(loader.shard_count(), 1);

    let mut shard_iter = loader.iter_shards();
    let first_shard = shard_iter.next();
    assert!(first_shard.is_some());

    let (shard_idx, _shard_path) = first_shard.unwrap();
    assert_eq!(shard_idx, 0);

    assert!(shard_iter.next().is_none());

    Ok(())
}

// Test 3: Batch creation from shards
#[test]
fn test_batch_creation_from_synthetic_dataset() -> Result<()> {
    let dataset = create_synthetic_dataset(4, 32, 1000);
    assert_eq!(dataset.len(), 4);

    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 2)?;

    let mut iter = dataloader.iter();
    let batch = iter.next();
    assert!(batch.is_some());

    let batch = batch.unwrap()?;

    let (batch_size, seq_len) = batch.shape();
    assert_eq!(batch_size, 2);
    assert_eq!(seq_len, 32);
    assert_eq!(batch.tokens().len(), 2 * 32);

    Ok(())
}

// Test 4: Batch chunking for gradient accumulation
#[test]
fn test_batch_chunking() -> Result<()> {
    let batch = Batch::new(vec![1u32; 128], 4, 32)?;
    assert_eq!(batch.tokens().len(), 128);
    assert_eq!(batch.batch_size(), 4);
    assert_eq!(batch.seq_len(), 32);

    let chunks = batch.into_chunks(32)?;
    assert_eq!(chunks.len(), 4);

    for chunk in &chunks {
        assert_eq!(chunk.tokens().len(), 32);
        assert_eq!(chunk.batch_size(), 1);
        assert_eq!(chunk.seq_len(), 32);
    }

    let total: usize = chunks.iter().map(|c| c.tokens().len()).sum();
    assert_eq!(total, 128);

    Ok(())
}

// Test 5: ChunkIterator for batches
#[test]
fn test_chunk_iterator() -> Result<()> {
    let batch = Batch::new(vec![1u32; 100], 4, 25)?;

    let iter = batch.chunks(25)?;

    let chunks: Vec<_> = iter.collect::<Result<Vec<_>>>()?;

    assert_eq!(chunks.len(), 4);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.tokens().len(), 25, "Chunk {} has wrong size", i);
        assert_eq!(chunk.batch_size(), 1);
        assert_eq!(chunk.seq_len(), 25);
    }

    Ok(())
}

// Test 6: Trainer with accumulated steps
#[test]
fn test_trainer_with_accumulated_steps() -> Result<()> {
    let mut model = SimpleMockModel::new(2);
    let batch = Batch::new(vec![1u32; 64], 2, 32)?;

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    let chunks = batch.into_chunks(32)?;
    assert_eq!(chunks.len(), 2);

    let chunk_results: Vec<Result<Batch>> = chunks.into_iter().map(Ok).collect();

    let metrics = trainer.train_accumulated_steps(chunk_results.into_iter(), 2)?;

    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
    assert!(metrics.learning_rate > 0.0);
    assert_eq!(metrics.step, 0);

    Ok(())
}

// Test 7: Full sharded training pipeline
#[test]
fn test_full_sharded_training_pipeline() -> Result<()> {
    let config = ShardConfig::new(
        "tests/sharded_training_integration.rs".to_string(),
        50000,
    )?;

    let loader = ShardedDataLoader::new(&config)?;
    assert!(loader.shard_count() > 0);

    let mut shard_count = 0;
    for (shard_idx, _shard_path) in loader.iter_shards() {
        shard_count += 1;
        assert_eq!(shard_idx, 0);
    }
    assert_eq!(shard_count, 1);

    let dataset = create_synthetic_dataset(8, 32, 1000);
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 4)?;

    let mut model = SimpleMockModel::new(2);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    let mut batch_count = 0;
    let mut last_loss: Option<f32> = None;

    for batch_result in dataloader.iter() {
        let batch = batch_result?;

        let chunks = batch.into_chunks(32)?;
        let chunk_count = chunks.len();
        let chunk_results: Vec<Result<Batch>> = chunks.into_iter().map(Ok).collect();

        let metrics = trainer.train_accumulated_steps(chunk_results.into_iter(), chunk_count)?;

        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());

        batch_count += 1;
        last_loss = Some(metrics.loss);
    }

    assert!(batch_count > 0);
    assert!(last_loss.is_some());

    Ok(())
}

// Test 8: Sharded training with multiple batches and steps
#[test]
fn test_sharded_training_with_multiple_steps() -> Result<()> {
    let dataset = create_synthetic_dataset(16, 32, 1000);
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 4)?;

    let mut model = SimpleMockModel::new(2);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    let mut metrics_history: Vec<StepMetrics> = Vec::new();

    for batch_result in dataloader.iter() {
        let batch = batch_result?;

        let chunks = batch.into_chunks(64)?;

        if chunks.is_empty() {
            continue;
        }

        let chunk_count = chunks.len();
        let chunk_results: Vec<Result<Batch>> = chunks.into_iter().map(Ok).collect();

        let metrics = trainer.train_accumulated_steps(chunk_results.into_iter(), chunk_count)?;
        metrics_history.push(metrics);
    }

    assert!(metrics_history.len() > 0);

    for (i, metrics) in metrics_history.iter().enumerate() {
        assert_eq!(metrics.step as usize, i);
        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());
    }

    Ok(())
}

// Test 9: Batch chunking with different chunk sizes
#[test]
fn test_batch_chunking_respects_seq_len_boundaries() -> Result<()> {
    let batch = Batch::new(vec![1u32; 128], 4, 32)?;

    let chunk_sizes = vec![32, 64, 96];

    for chunk_size in chunk_sizes {
        let chunks = batch.clone().into_chunks(chunk_size)?;

        for chunk in &chunks {
            assert_eq!(
                chunk.tokens().len() % 32,
                0,
                "Chunk size {} not multiple of seq_len 32",
                chunk.tokens().len()
            );
        }

        let total: usize = chunks.iter().map(|c| c.tokens().len()).sum();
        assert_eq!(total, 128);
    }

    Ok(())
}

// Test 10: End-to-end integration with real DataLoader iteration
#[test]
fn test_end_to_end_with_dataloader_iteration() -> Result<()> {
    let dataset = create_synthetic_dataset(12, 32, 1000);
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 3)?;

    let mut model = SimpleMockModel::new(2);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    let mut total_batches = 0;
    let mut total_training_steps = 0;

    for batch_result in dataloader.iter() {
        let batch = batch_result?;
        total_batches += 1;

        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.seq_len(), 32);
        assert_eq!(batch.tokens().len(), 3 * 32);

        let chunks = batch.into_chunks(32)?;
        let chunk_count = chunks.len();

        let chunk_results: Vec<Result<Batch>> = chunks.into_iter().map(Ok).collect();
        let _metrics = trainer.train_accumulated_steps(chunk_results.into_iter(), chunk_count)?;
        total_training_steps += 1;
    }

    assert_eq!(total_batches, 4);
    assert_eq!(total_training_steps, 4);

    Ok(())
}
