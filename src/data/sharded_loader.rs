//! Sharded data loading from disk
//!
//! This module provides types and utilities for loading data from multiple
//! sharded files, enabling distributed training across multiple data sources.

use std::path::PathBuf;
use super::{DataLoader, Dataset, Sampler, SequentialDataset, SequentialSampler};

/// Configuration for loading data from multiple shards
///
/// Specifies the pattern for shard files, vocabulary size, and optional
/// pre-computed metadata about each shard.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Glob pattern to match shard files (e.g., "data/shards/shard_*.jsonl")
    pub shard_pattern: String,
    /// Vocabulary size (max token ID + 1)
    pub vocab_size: u32,
    /// Optional pre-computed metadata for each shard
    pub shard_metadata: Option<Vec<ShardMetadata>>,
}

/// Metadata about a single data shard
///
/// Contains information about where a shard is located and how many
/// tokens it contains, useful for load balancing across shards.
#[derive(Debug, Clone)]
pub struct ShardMetadata {
    /// Index of this shard (0-based)
    pub shard_idx: usize,
    /// Total number of tokens in this shard
    pub token_count: usize,
    /// Path to the shard file
    pub path: String,
}

/// A batch from a single shard
///
/// Represents one shard's data ready for loading as batches.
#[derive(Debug)]
pub struct ShardBatch<D: Dataset, S: Sampler> {
    /// Index of the shard this batch comes from
    pub shard_idx: usize,
    /// Path to the shard file
    pub shard_path: PathBuf,
    /// DataLoader for this shard
    pub loader: DataLoader<D, S>,
    /// Total tokens available in this shard
    pub token_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic ShardConfig creation
    #[test]
    fn test_shard_config_creation() {
        let config = ShardConfig {
            shard_pattern: "data/shards/shard_*.jsonl".to_string(),
            vocab_size: 50000,
            shard_metadata: None,
        };

        assert_eq!(config.vocab_size, 50000);
        assert_eq!(
            config.shard_pattern,
            "data/shards/shard_*.jsonl".to_string()
        );
        assert!(config.shard_metadata.is_none());
    }

    /// Test ShardMetadata creation
    #[test]
    fn test_shard_metadata_creation() {
        let metadata = ShardMetadata {
            shard_idx: 5,
            token_count: 1_000_000,
            path: "data/shards/shard_5.jsonl".to_string(),
        };

        assert_eq!(metadata.shard_idx, 5);
        assert_eq!(metadata.token_count, 1_000_000);
        assert_eq!(metadata.path, "data/shards/shard_5.jsonl");
    }

    /// Test ShardBatch creation with metadata
    #[test]
    fn test_shard_batch_creation() {
        let metadata = ShardMetadata {
            shard_idx: 0,
            token_count: 500_000,
            path: "data/shards/shard_0.jsonl".to_string(),
        };

        // Create a minimal dataset and sampler for the loader
        let samples = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let loader = DataLoader::new(dataset, sampler, 2).unwrap();

        let shard_batch = ShardBatch {
            shard_idx: metadata.shard_idx,
            shard_path: PathBuf::from(&metadata.path),
            loader,
            token_count: metadata.token_count,
        };

        assert_eq!(shard_batch.shard_idx, 0);
        assert_eq!(shard_batch.token_count, 500_000);
        assert_eq!(
            shard_batch.shard_path,
            PathBuf::from("data/shards/shard_0.jsonl")
        );
    }
}
