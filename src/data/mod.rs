//! Data loading infrastructure for Rustane
//!
//! This module provides abstractions for loading, sampling, and batching data
//! for training neural networks. It's designed to support both synthetic and
//! real datasets with efficient streaming and memory management.
//!
//! # Core Components
//!
//! - [`Dataset`] - Trait for accessing samples
//! - [`Sampler`] - Trait for sampling indices from datasets
//! - [`DataLoader`] - Iterates over batches with sampling and batching
//! - [`Batch`] - Represents a batch of tokenized data
//!
//! # Example
//!
//! ```no_run
//! use rustane::data::{DataLoader, SequentialDataset, SequentialSampler};
//!
//! // Create in-memory dataset
//! let samples = vec![
//!     vec![0, 1, 2, 3],
//!     vec![4, 5, 6, 7],
//!     vec![8, 9, 10, 11],
//! ];
//! let dataset = SequentialDataset::new(samples);
//!
//! // Create sampler and dataloader
//! let sampler = SequentialSampler::new(dataset.len());
//! let dataloader = DataLoader::new(dataset, sampler, 2)?; // batch_size=2
//!
//! // Iterate over batches
//! for batch in dataloader.iter() {
//!     println!(\"Batch shape: {:?}\", batch?.shape());
//! }
//! # Ok::<(), rustane::Error>(())
//! ```

use crate::Result;
use std::collections::VecDeque;

pub use self::dataset::{Dataset, SequentialDataset};
pub use self::sampler::{RandomSampler, Sampler, SequentialSampler};
pub use self::collate::{Collator, PadCollator, TruncateCollator};

mod dataset;
mod sampler;
mod collate;

/// A batch of tokenized samples, potentially padded or packed
///
/// Contains token IDs arranged in a 2D grid [batch_size, sequence_length].
/// Token IDs are u32 to support large vocabularies (up to 2^32).
#[derive(Debug, Clone)]
pub struct Batch {
    /// Flattened token IDs: [batch_size * seq_len]
    tokens: Vec<u32>,
    /// Batch size
    batch_size: usize,
    /// Sequence length (may vary with packing, but for now assumed constant)
    seq_len: usize,
}

impl Batch {
    /// Create a new batch from tokens and metadata
    ///
    /// # Arguments
    /// - `tokens`: Flattened token IDs
    /// - `batch_size`: Number of samples in batch
    /// - `seq_len`: Length of each sequence
    ///
    /// # Errors
    /// Returns an error if tokens.len() != batch_size * seq_len
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize) -> Result<Self> {
        if tokens.len() != batch_size * seq_len {
            return Err(crate::Error::InvalidParameter(
                format!(
                    "tokens length {} doesn't match batch_size*seq_len {}",
                    tokens.len(),
                    batch_size * seq_len
                ),
            ));
        }
        Ok(Batch {
            tokens,
            batch_size,
            seq_len,
        })
    }

    /// Return the shape of this batch: [batch_size, seq_len]
    pub fn shape(&self) -> (usize, usize) {
        (self.batch_size, self.seq_len)
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get all token IDs as a slice
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Get token ID at [batch_idx, seq_idx]
    pub fn get(&self, batch_idx: usize, seq_idx: usize) -> Option<u32> {
        if batch_idx >= self.batch_size || seq_idx >= self.seq_len {
            return None;
        }
        Some(self.tokens[batch_idx * self.seq_len + seq_idx])
    }
}

/// An iterator over batches from a dataset
pub struct DataLoaderIter<D: Dataset, S: Sampler> {
    dataset: D,
    sampler: S,
    batch_size: usize,
    indices: VecDeque<usize>,
    exhausted: bool,
}

impl<D: Dataset, S: Sampler> Iterator for DataLoaderIter<D, S> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        // Collect indices for this batch
        let mut batch_indices = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(idx) = self.indices.pop_front() {
                batch_indices.push(idx);
            } else {
                // No more indices available
                if batch_indices.is_empty() {
                    self.exhausted = true;
                    return None;
                }
                break; // Partial batch at the end
            }
        }

        // Fetch samples from dataset
        let mut tokens = Vec::new();
        let mut seq_len = 0;

        for idx in batch_indices {
            match self.dataset.get(idx) {
                Ok(sample) => {
                    if seq_len == 0 {
                        seq_len = sample.len();
                    }
                    tokens.extend_from_slice(&sample);
                }
                Err(e) => {
                    return Some(Err(e));
                }
            }
        }

        let batch_size = tokens.len() / seq_len.max(1);

        match Batch::new(tokens, batch_size, seq_len) {
            Ok(batch) => Some(Ok(batch)),
            Err(e) => {
                self.exhausted = true;
                Some(Err(e))
            }
        }
    }
}

/// Data loader that iterates over batches
///
/// Combines a dataset, sampler, and batch size to produce batches of data.
pub struct DataLoader<D: Dataset, S: Sampler> {
    dataset: D,
    sampler: S,
    batch_size: usize,
}

impl<D: Dataset, S: Sampler> DataLoader<D, S> {
    /// Create a new data loader
    ///
    /// # Arguments
    /// - `dataset`: Source of samples
    /// - `sampler`: Produces indices to sample
    /// - `batch_size`: Number of samples per batch
    ///
    /// # Errors
    /// Returns an error if batch_size is 0
    pub fn new(dataset: D, sampler: S, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(crate::Error::InvalidParameter(
                "batch_size must be > 0".to_string(),
            ));
        }
        Ok(DataLoader {
            dataset,
            sampler,
            batch_size,
        })
    }

    /// Get an iterator over batches
    ///
    /// Consumes self and returns an iterator that yields batches
    pub fn iter(mut self) -> DataLoaderIter<D, S> {
        let indices = self.sampler.sample();
        DataLoaderIter {
            dataset: self.dataset,
            sampler: self.sampler,
            batch_size: self.batch_size,
            indices: VecDeque::from(indices),
            exhausted: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let tokens = vec![0, 1, 2, 3, 4, 5];
        let batch = Batch::new(tokens, 2, 3).unwrap();
        assert_eq!(batch.shape(), (2, 3));
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.seq_len(), 3);
    }

    #[test]
    fn test_batch_shape_mismatch() {
        let tokens = vec![0, 1, 2];
        let result = Batch::new(tokens, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_indexing() {
        let tokens = vec![0, 1, 2, 3, 4, 5];
        let batch = Batch::new(tokens, 2, 3).unwrap();
        assert_eq!(batch.get(0, 0), Some(0));
        assert_eq!(batch.get(0, 2), Some(2));
        assert_eq!(batch.get(1, 0), Some(3));
        assert_eq!(batch.get(1, 2), Some(5));
        assert_eq!(batch.get(2, 0), None); // Out of bounds
    }

    #[test]
    fn test_sequential_dataset() {
        let samples = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let dataset = SequentialDataset::new(samples);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0).unwrap(), vec![0, 1, 2]);
        assert_eq!(dataset.get(1).unwrap(), vec![3, 4, 5]);
    }

    #[test]
    fn test_sequential_dataset_out_of_bounds() {
        let samples = vec![vec![0, 1]];
        let dataset = SequentialDataset::new(samples);
        assert!(dataset.get(1).is_err());
    }

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler::new(5);
        let indices = sampler.sample();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sequential_sampler_empty() {
        let mut sampler = SequentialSampler::new(0);
        let indices = sampler.sample();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_dataloader_simple_batch() {
        let samples = vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
            vec![6, 7],
        ];
        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(4);
        let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

        let mut iter = dataloader.iter();
        let batch1 = iter.next().unwrap().unwrap();
        assert_eq!(batch1.shape(), (2, 2));
        assert_eq!(batch1.tokens(), &[0, 1, 2, 3]);

        let batch2 = iter.next().unwrap().unwrap();
        assert_eq!(batch2.shape(), (2, 2));
        assert_eq!(batch2.tokens(), &[4, 5, 6, 7]);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dataloader_partial_batch() {
        let samples = vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
        ];
        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(3);
        let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

        let mut iter = dataloader.iter();
        let batch1 = iter.next().unwrap().unwrap();
        assert_eq!(batch1.shape(), (2, 2));

        let batch2 = iter.next().unwrap().unwrap();
        assert_eq!(batch2.shape(), (1, 2)); // Partial batch

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dataloader_invalid_batch_size() {
        let dataset = SequentialDataset::new(vec![vec![0]]);
        let sampler = SequentialSampler::new(1);
        let result = DataLoader::new(dataset, sampler, 0);
        assert!(result.is_err());
    }
}
