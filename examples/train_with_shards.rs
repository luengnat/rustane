//! Example: Training on sharded data with gradient accumulation.
//!
//! Demonstrates:
//! - Creating synthetic shard files on disk
//! - Discovering shards with `ShardedDataLoader`
//! - Loading each shard into a `DataLoader`
//! - Chunking batches and training with explicit accumulation steps

use rustane::data::{Batch, DataLoader, Dataset, JsonlDataset, RandomSampler, ShardedDataLoader, ShardConfig};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, Optimizer, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::ConstantScheduler;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("Rustane Sharded Training Example");
    println!("================================\n");

    let shard_dir = create_demo_shards()?;
    let shard_pattern = format!("{}/shard_*.jsonl", shard_dir.display());
    let config = ShardConfig::new(shard_pattern, 256)?;
    let loader = ShardedDataLoader::new(&config)?;

    println!("Discovered {} shard(s)\n", loader.shard_count());

    let mut model = SimpleModel::new(32);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    println!("Step | Shard | Loss    | Grad Norm | LR");
    println!("-----|-------|---------|-----------|--------");

    let mut step = 0usize;
    for (shard_idx, shard_path) in loader.iter_shards() {
        let dataset = JsonlDataset::load(&shard_path)?;
        let sampler = RandomSampler::new(dataset.len(), 42);
        let dataloader = DataLoader::new(dataset, sampler, 4)?;

        for batch_result in dataloader.iter() {
            let batch = batch_result?;
            let chunks = batch.into_chunks(16)?;

            let metrics = trainer.train_accumulated_steps(chunks.into_iter().map(Ok), 2)?;

            println!(
                "{:4} | {:5} | {:.5} | {:.5}    | {:.6}",
                step,
                shard_idx,
                metrics.loss,
                metrics.grad_norm,
                metrics.learning_rate
            );
            step += 1;
        }

        println!("Processed shard: {}", shard_path.display());
    }

    println!("\n✓ Training completed!");
    Ok(())
}

fn create_demo_shards() -> Result<PathBuf> {
    let shard_dir = env::temp_dir().join(format!("rustane-shards-{}", std::process::id()));
    fs::create_dir_all(&shard_dir).map_err(|e| rustane::Error::Io(e.to_string()))?;

    write_shard(
        &shard_dir.join("shard_000.jsonl"),
        &[
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![8, 9, 10, 11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20, 21, 22, 23],
            vec![24, 25, 26, 27, 28, 29, 30, 31],
            vec![32, 33, 34, 35, 36, 37, 38, 39],
            vec![40, 41, 42, 43, 44, 45, 46, 47],
            vec![48, 49, 50, 51, 52, 53, 54, 55],
            vec![56, 57, 58, 59, 60, 61, 62, 63],
        ],
    )?;

    write_shard(
        &shard_dir.join("shard_001.jsonl"),
        &[
            vec![64, 65, 66, 67, 68, 69, 70, 71],
            vec![72, 73, 74, 75, 76, 77, 78, 79],
            vec![80, 81, 82, 83, 84, 85, 86, 87],
            vec![88, 89, 90, 91, 92, 93, 94, 95],
            vec![96, 97, 98, 99, 100, 101, 102, 103],
            vec![104, 105, 106, 107, 108, 109, 110, 111],
            vec![112, 113, 114, 115, 116, 117, 118, 119],
            vec![120, 121, 122, 123, 124, 125, 126, 127],
        ],
    )?;

    Ok(shard_dir)
}

fn write_shard(path: &Path, samples: &[Vec<u32>]) -> Result<()> {
    let mut file = File::create(path).map_err(|e| rustane::Error::Io(e.to_string()))?;
    for sample in samples {
        let line = serde_json::to_string(sample)
            .map_err(|e| rustane::Error::Other(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| rustane::Error::Io(e.to_string()))?;
    }
    Ok(())
}

struct SimpleModel {
    params: Vec<f32>,
}

impl SimpleModel {
    fn new(vocab_size: usize) -> Self {
        Self {
            params: vec![0.01; vocab_size * 2],
        }
    }
}

impl Model for SimpleModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        let (batch_size, _seq_len) = batch.shape();
        let logits = vec![0.0f32; batch_size * self.params.len()];
        ANETensor::from_fp32(logits, vec![batch_size, self.params.len()])
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

struct SimpleOptimizer {
    _lr: f32,
}

impl SimpleOptimizer {
    fn new(lr: f32) -> Self {
        Self { _lr: lr }
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
