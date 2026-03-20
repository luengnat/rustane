//! Sharded data loading from disk
//!
//! This module provides types and utilities for loading data from multiple
//! sharded files, enabling distributed training across multiple data sources.

use std::path::PathBuf;
use glob::glob;
use super::{DataLoader, Dataset, Sampler};
use crate::Result;

/// Configuration for loading data from multiple shards
///
/// Specifies the pattern for shard files, vocabulary size, and optional
/// pre-computed metadata about each shard.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardConfig {
    shard_pattern: String,
    vocab_size: u32,
    shard_metadata: Option<Vec<ShardMetadata>>,
}

impl ShardConfig {
    /// Create a new ShardConfig with validation
    ///
    /// # Arguments
    /// - `shard_pattern`: Glob pattern to match shard files (e.g., "data/shards/shard_*.jsonl")
    /// - `vocab_size`: Vocabulary size (max token ID + 1), must be > 0
    ///
    /// # Errors
    /// Returns error if shard_pattern is empty or vocab_size is 0
    pub fn new(shard_pattern: String, vocab_size: u32) -> Result<Self> {
        if shard_pattern.is_empty() {
            return Err(crate::Error::InvalidParameter(
                "shard_pattern cannot be empty".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(crate::Error::InvalidParameter(
                "vocab_size must be greater than 0".to_string(),
            ));
        }
        Ok(ShardConfig {
            shard_pattern,
            vocab_size,
            shard_metadata: None,
        })
    }

    /// Get the shard pattern
    pub fn shard_pattern(&self) -> &str {
        &self.shard_pattern
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Get the shard metadata if available
    pub fn shard_metadata(&self) -> Option<&[ShardMetadata]> {
        self.shard_metadata.as_deref()
    }

    /// Set shard metadata
    pub fn with_metadata(mut self, metadata: Vec<ShardMetadata>) -> Self {
        self.shard_metadata = Some(metadata);
        self
    }
}

/// Metadata about a single data shard
///
/// Contains information about where a shard is located and how many
/// tokens it contains, useful for load balancing across shards.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardMetadata {
    shard_idx: usize,
    token_count: usize,
    path: PathBuf,
}

impl ShardMetadata {
    /// Create new shard metadata with validation
    ///
    /// # Arguments
    /// - `shard_idx`: Index of this shard (0-based)
    /// - `token_count`: Total number of tokens in this shard, must be > 0
    /// - `path`: Path to the shard file
    ///
    /// # Errors
    /// Returns error if token_count is 0
    pub fn new(shard_idx: usize, token_count: usize, path: PathBuf) -> Result<Self> {
        if token_count == 0 {
            return Err(crate::Error::InvalidParameter(
                "token_count must be greater than 0".to_string(),
            ));
        }
        Ok(ShardMetadata {
            shard_idx,
            token_count,
            path,
        })
    }

    /// Get the shard index
    pub fn shard_idx(&self) -> usize {
        self.shard_idx
    }

    /// Get the token count
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Get the path to the shard file
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

/// A batch from a single shard
///
/// Represents one shard's data ready for loading as batches.
#[derive(Debug)]
pub struct ShardBatch<D: Dataset, S: Sampler> {
    shard_idx: usize,
    shard_path: PathBuf,
    loader: DataLoader<D, S>,
    token_count: usize,
}

impl<D: Dataset, S: Sampler> ShardBatch<D, S> {
    /// Create a new ShardBatch
    pub fn new(
        shard_idx: usize,
        shard_path: PathBuf,
        loader: DataLoader<D, S>,
        token_count: usize,
    ) -> Self {
        ShardBatch {
            shard_idx,
            shard_path,
            loader,
            token_count,
        }
    }

    /// Get the shard index
    pub fn shard_idx(&self) -> usize {
        self.shard_idx
    }

    /// Get the shard path
    pub fn shard_path(&self) -> &PathBuf {
        &self.shard_path
    }

    /// Get the token count
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Consume self and return the DataLoader
    pub fn into_loader(self) -> DataLoader<D, S> {
        self.loader
    }

    /// Consume self and return all components
    pub fn into_parts(self) -> (usize, PathBuf, DataLoader<D, S>, usize) {
        (self.shard_idx, self.shard_path, self.loader, self.token_count)
    }
}

/// Loads tokenized data from multiple shard files on disk
///
/// Discovers shard files matching a glob pattern and provides iteration
/// over them. Each shard can be loaded into its own DataLoader for processing.
#[derive(Debug)]
pub struct ShardedDataLoader {
    /// List of shard file paths discovered via glob pattern
    shard_files: Vec<PathBuf>,
    /// Configuration
    #[allow(dead_code)]
    config: ShardConfig,
}

impl ShardedDataLoader {
    /// Create new sharded loader from config
    ///
    /// # Arguments
    /// - `config`: ShardConfig with shard pattern and vocab size
    ///
    /// # Errors
    /// Returns error if glob pattern is invalid or no shards found
    pub fn new(config: &ShardConfig) -> Result<Self> {
        let mut shard_files = Vec::new();

        match glob(config.shard_pattern()) {
            Ok(paths) => {
                for entry in paths {
                    match entry {
                        Ok(path) => shard_files.push(path),
                        Err(e) => {
                            return Err(crate::Error::Other(
                                format!("error reading shard path: {}", e)
                            ))
                        }
                    }
                }
            }
            Err(e) => {
                return Err(crate::Error::Other(
                    format!("invalid glob pattern: {}", e)
                ))
            }
        }

        shard_files.sort();

        if shard_files.is_empty() {
            return Err(crate::Error::Other(
                format!("no shard files found matching pattern: {}", config.shard_pattern())
            ));
        }

        Ok(ShardedDataLoader {
            shard_files,
            config: config.clone(),
        })
    }

    /// Get total number of discovered shards
    pub fn shard_count(&self) -> usize {
        self.shard_files.len()
    }

    /// Create iterator over all shards
    ///
    /// # Returns
    /// Iterator that yields (shard_idx, shard_path) tuples for each shard file
    pub fn iter_shards(&self) -> ShardIterator<'_> {
        ShardIterator {
            parent: self,
            current_idx: 0,
        }
    }
}

/// Iterator over shards
///
/// Yields (shard_idx, shard_path) tuples for each discovered shard.
pub struct ShardIterator<'a> {
    parent: &'a ShardedDataLoader,
    current_idx: usize,
}

impl<'a> Iterator for ShardIterator<'a> {
    type Item = (usize, PathBuf);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.parent.shard_files.len() {
            return None;
        }

        let shard_path = self.parent.shard_files[self.current_idx].clone();
        let shard_idx = self.current_idx;
        self.current_idx += 1;

        Some((shard_idx, shard_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{SequentialDataset, SequentialSampler};

    /// Test basic ShardConfig creation
    #[test]
    fn test_shard_config_creation() {
        let config = ShardConfig::new(
            "data/shards/shard_*.jsonl".to_string(),
            50000,
        ).unwrap();

        assert_eq!(config.vocab_size(), 50000);
        assert_eq!(config.shard_pattern(), "data/shards/shard_*.jsonl");
        assert!(config.shard_metadata().is_none());
    }

    /// Test ShardConfig with empty pattern validation
    #[test]
    fn test_shard_config_empty_pattern() {
        let result = ShardConfig::new("".to_string(), 50000);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::InvalidParameter(msg) => {
                assert!(msg.contains("shard_pattern cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    /// Test ShardConfig with zero vocab_size validation
    #[test]
    fn test_shard_config_zero_vocab_size() {
        let result = ShardConfig::new("data/shards/shard_*.jsonl".to_string(), 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::InvalidParameter(msg) => {
                assert!(msg.contains("vocab_size must be greater than 0"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    /// Test ShardConfig accessors return correct values
    #[test]
    fn test_shard_config_accessors() {
        let config = ShardConfig::new(
            "data/shards/shard_*.jsonl".to_string(),
            25000,
        ).unwrap();

        assert_eq!(config.shard_pattern(), "data/shards/shard_*.jsonl");
        assert_eq!(config.vocab_size(), 25000);
        assert!(config.shard_metadata().is_none());
    }

    /// Test ShardConfig with metadata
    #[test]
    fn test_shard_config_with_metadata() {
        let metadata = vec![
            ShardMetadata::new(0, 1_000_000, PathBuf::from("shard_0.jsonl")).unwrap(),
            ShardMetadata::new(1, 2_000_000, PathBuf::from("shard_1.jsonl")).unwrap(),
        ];

        let config = ShardConfig::new("data/shards/shard_*.jsonl".to_string(), 50000)
            .unwrap()
            .with_metadata(metadata.clone());

        assert!(config.shard_metadata().is_some());
        assert_eq!(config.shard_metadata().unwrap().len(), 2);
    }

    /// Test ShardMetadata creation
    #[test]
    fn test_shard_metadata_creation() {
        let metadata = ShardMetadata::new(
            5,
            1_000_000,
            PathBuf::from("data/shards/shard_5.jsonl"),
        ).unwrap();

        assert_eq!(metadata.shard_idx(), 5);
        assert_eq!(metadata.token_count(), 1_000_000);
        assert_eq!(metadata.path(), &PathBuf::from("data/shards/shard_5.jsonl"));
    }

    /// Test ShardMetadata with zero token_count validation
    #[test]
    fn test_shard_metadata_zero_token_count() {
        let result = ShardMetadata::new(
            0,
            0,
            PathBuf::from("data/shards/shard_0.jsonl"),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::InvalidParameter(msg) => {
                assert!(msg.contains("token_count must be greater than 0"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    /// Test ShardMetadata accessors
    #[test]
    fn test_shard_metadata_accessors() {
        let metadata = ShardMetadata::new(
            3,
            500_000,
            PathBuf::from("data/shards/shard_3.jsonl"),
        ).unwrap();

        assert_eq!(metadata.shard_idx(), 3);
        assert_eq!(metadata.token_count(), 500_000);
        assert_eq!(
            metadata.path(),
            &PathBuf::from("data/shards/shard_3.jsonl")
        );
    }

    /// Test ShardMetadata equality
    #[test]
    fn test_shard_metadata_equality() {
        let metadata1 = ShardMetadata::new(
            1,
            1_000_000,
            PathBuf::from("data/shards/shard_1.jsonl"),
        ).unwrap();

        let metadata2 = ShardMetadata::new(
            1,
            1_000_000,
            PathBuf::from("data/shards/shard_1.jsonl"),
        ).unwrap();

        assert_eq!(metadata1, metadata2);
    }

    /// Test ShardConfig equality
    #[test]
    fn test_shard_config_equality() {
        let config1 = ShardConfig::new(
            "data/shards/shard_*.jsonl".to_string(),
            50000,
        ).unwrap();

        let config2 = ShardConfig::new(
            "data/shards/shard_*.jsonl".to_string(),
            50000,
        ).unwrap();

        assert_eq!(config1, config2);
    }

    /// Test ShardBatch creation
    #[test]
    fn test_shard_batch_creation() {
        // Create a minimal dataset and sampler for the loader
        let samples = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let loader = DataLoader::new(dataset, sampler, 2).unwrap();

        let shard_batch = ShardBatch::new(
            0,
            PathBuf::from("data/shards/shard_0.jsonl"),
            loader,
            500_000,
        );

        assert_eq!(shard_batch.shard_idx(), 0);
        assert_eq!(shard_batch.token_count(), 500_000);
        assert_eq!(
            shard_batch.shard_path(),
            &PathBuf::from("data/shards/shard_0.jsonl")
        );
    }

    /// Test ShardBatch into_loader consumes self
    #[test]
    fn test_shard_batch_into_loader() {
        let samples = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let loader = DataLoader::new(dataset, sampler, 2).unwrap();

        let shard_batch = ShardBatch::new(
            0,
            PathBuf::from("data/shards/shard_0.jsonl"),
            loader,
            500_000,
        );

        let _recovered_loader = shard_batch.into_loader();
        // If we got here without compile error, the method works
    }

    /// Test ShardBatch into_parts consumes self
    #[test]
    fn test_shard_batch_into_parts() {
        let samples = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let loader = DataLoader::new(dataset, sampler, 2).unwrap();

        let shard_batch = ShardBatch::new(
            0,
            PathBuf::from("data/shards/shard_0.jsonl"),
            loader,
            500_000,
        );

        let (shard_idx, shard_path, _loader, token_count) = shard_batch.into_parts();
        assert_eq!(shard_idx, 0);
        assert_eq!(shard_path, PathBuf::from("data/shards/shard_0.jsonl"));
        assert_eq!(token_count, 500_000);
    }

    /// Test ShardedDataLoader creation with non-existent pattern
    #[test]
    fn test_sharded_loader_creation_no_shards() {
        let config = ShardConfig::new("nonexistent/*.bin".to_string(), 50257).unwrap();
        let result = ShardedDataLoader::new(&config);
        assert!(result.is_err());
    }

    /// Test ShardedDataLoader shard count
    #[test]
    fn test_sharded_loader_shard_count() {
        // When no shards found, should return error (empty glob)
        let config = ShardConfig::new("nonexistent/*.bin".to_string(), 50257).unwrap();
        let result = ShardedDataLoader::new(&config);
        assert!(result.is_err());
    }

    /// Test ShardedDataLoader iterator interface
    #[test]
    fn test_sharded_loader_iter_shards() {
        // This test verifies the iterator interface exists and basic structure works
        // We'll use synthetic data with glob pattern matching this file itself
        let config = ShardConfig::new(
            "src/data/sharded_loader.rs".to_string(),
            50257,
        ).unwrap();
        
        let loader = ShardedDataLoader::new(&config).unwrap();
        assert_eq!(loader.shard_count(), 1);

        let mut shard_iter = loader.iter_shards();
        let first_shard = shard_iter.next();
        assert!(first_shard.is_some());

        let (shard_idx, shard_path) = first_shard.unwrap();
        assert_eq!(shard_idx, 0);
        assert!(shard_path.to_string_lossy().contains("sharded_loader.rs"));

        // No more shards
        assert!(shard_iter.next().is_none());
    }
}
