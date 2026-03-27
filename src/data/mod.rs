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
//!     let _batch = batch?;
//!     println!("Batch: {:?}", _batch.shape());
//! }
//! # Ok::<(), rustane::Error>(())
//! ```

use crate::Result;
use std::collections::VecDeque;

/// Compute token-aligned chunk sizes respecting seq_len boundaries
///
/// # Arguments
/// - `total_tokens`: Total number of tokens in batch
/// - `seq_len`: Sequence length (chunk boundaries must be multiples of this)
/// - `max_chunk_tokens`: Desired maximum chunk size
///
/// # Returns
/// Vector of chunk sizes that:
/// - All are multiples of seq_len (except possibly handling edge cases)
/// - Sum to total_tokens
/// - Are at most max_chunk_tokens (rounded down to nearest seq_len multiple)
fn compute_chunk_sizes(total_tokens: usize, seq_len: usize, max_chunk_tokens: usize) -> Vec<usize> {
    if seq_len == 0 {
        return vec![total_tokens];
    }

    // Ensure chunk size is multiple of seq_len
    let usable_chunk = ((max_chunk_tokens / seq_len).max(1)) * seq_len;
    let mut chunks = Vec::new();
    let mut remaining = total_tokens;

    while remaining > 0 {
        let chunk = remaining.min(usable_chunk);
        chunks.push(chunk);
        remaining -= chunk;
    }

    chunks
}

pub use self::collate::{Collator, PadCollator, TruncateCollator};
pub use self::dataset::{Dataset, SequentialDataset};
pub use self::filesystem::{JsonlDataset, TextDataset};
pub use self::loader::{
    count_special_tokens, find_boundaries, load_shard, load_shard_range, BatchConfig,
    DistributedTokenLoader, ShardHeader, TokenStream, SHARD_MAGIC, SHARD_VERSION,
};
pub use self::sampler::{RandomSampler, Sampler, SequentialSampler};
pub use self::sharded_loader::{ShardBatch, ShardConfig, ShardMetadata, ShardedDataLoader};

// Batch and ChunkIterator are defined in this module

mod collate;
mod dataset;
mod filesystem;
pub mod loader;
mod sampler;
mod sharded_loader;

/// A batch of tokenized samples, potentially padded or packed
///
/// Contains token IDs arranged in a 2D grid [batch_size, sequence_length].
/// Token IDs are u32 to support large vocabularies (up to 2^32).
#[derive(Debug, Clone)]
pub struct Batch {
    /// Flattened token IDs: [batch_size * seq_len]
    pub tokens: Vec<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length (may vary with packing, but for now assumed constant)
    pub seq_len: usize,
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
            return Err(crate::Error::InvalidParameter(format!(
                "tokens length {} doesn't match batch_size*seq_len {}",
                tokens.len(),
                batch_size * seq_len
            )));
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

    /// Returns an iterator over token-aligned chunks for gradient accumulation
    ///
    /// # Arguments
    /// - `max_chunk_tokens`: Maximum number of tokens per chunk
    ///
    /// # Behavior
    /// - Chunks respect seq_len boundaries (no sequences are split)
    /// - All chunks except possibly the last will be exactly sized to multiples of seq_len
    /// - Total tokens across chunks equals original batch size
    /// - batch_size is recalculated for each chunk based on actual token count
    ///
    /// # Errors
    /// Returns an error if seq_len is 0 or max_chunk_tokens is 0
    ///
    /// # Example
    /// ```
    /// # use rustane::data::Batch;
    /// let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
    /// let mut chunks = batch.chunks(25).unwrap();
    /// let chunk1 = chunks.next().unwrap().unwrap();
    /// assert_eq!(chunk1.shape(), (1, 25)); // 25 tokens / 25 seq_len = batch_size 1
    /// ```
    pub fn chunks(&self, max_chunk_tokens: usize) -> Result<ChunkIterator<'_>> {
        if self.seq_len == 0 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be > 0".to_string(),
            ));
        }
        if max_chunk_tokens == 0 {
            return Err(crate::Error::InvalidParameter(
                "max_chunk_tokens must be > 0".to_string(),
            ));
        }

        let chunk_sizes = compute_chunk_sizes(self.tokens.len(), self.seq_len, max_chunk_tokens);

        Ok(ChunkIterator {
            original_batch: self,
            chunk_sizes,
            current_chunk_idx: 0,
            current_pos: 0,
        })
    }

    /// Split batch into token-aligned chunks for gradient accumulation (returns Vec)
    ///
    /// # Arguments
    /// - `max_chunk_tokens`: Maximum number of tokens per chunk
    ///
    /// # Behavior
    /// - Chunks respect seq_len boundaries (no sequences are split)
    /// - All chunks except possibly the last will be exactly sized to multiples of seq_len
    /// - Total tokens across chunks equals original batch size
    /// - batch_size is recalculated for each chunk based on actual token count
    ///
    /// # Errors
    /// Returns an error if seq_len is 0 or max_chunk_tokens is 0
    ///
    /// # Example
    /// ```
    /// # use rustane::data::Batch;
    /// let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
    /// let chunks = batch.into_chunks(25).unwrap();
    /// assert_eq!(chunks.len(), 4); // 4 chunks of 25 tokens each
    /// ```
    pub fn into_chunks(self, max_chunk_tokens: usize) -> Result<Vec<Batch>> {
        if self.seq_len == 0 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be > 0".to_string(),
            ));
        }
        if max_chunk_tokens == 0 {
            return Err(crate::Error::InvalidParameter(
                "max_chunk_tokens must be > 0".to_string(),
            ));
        }

        let total_tokens = self.tokens.len();
        if total_tokens <= max_chunk_tokens {
            return Ok(vec![self]);
        }

        let chunk_sizes = compute_chunk_sizes(total_tokens, self.seq_len, max_chunk_tokens);
        let mut chunks = Vec::new();
        let mut pos = 0;

        for chunk_size in chunk_sizes {
            let end = (pos + chunk_size).min(total_tokens);
            let chunk_tokens = self.tokens[pos..end].to_vec();

            let new_batch_size = chunk_tokens.len() / self.seq_len;

            chunks.push(Batch {
                tokens: chunk_tokens,
                batch_size: new_batch_size,
                seq_len: self.seq_len,
            });

            pos = end;
            if pos >= total_tokens {
                break;
            }
        }

        Ok(chunks)
    }
}

/// An iterator over chunks of a batch
#[derive(Debug)]
pub struct ChunkIterator<'a> {
    original_batch: &'a Batch,
    chunk_sizes: Vec<usize>,
    current_chunk_idx: usize,
    current_pos: usize,
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_chunk_idx >= self.chunk_sizes.len() {
            return None;
        }

        let chunk_size = self.chunk_sizes[self.current_chunk_idx];
        let end = (self.current_pos + chunk_size).min(self.original_batch.tokens.len());
        let chunk_tokens = self.original_batch.tokens[self.current_pos..end].to_vec();

        let new_batch_size = chunk_tokens.len() / self.original_batch.seq_len;

        self.current_pos = end;
        self.current_chunk_idx += 1;

        Some(Ok(Batch {
            tokens: chunk_tokens,
            batch_size: new_batch_size,
            seq_len: self.original_batch.seq_len,
        }))
    }
}

/// An iterator over batches from a dataset
pub struct DataLoaderIter<D: Dataset, S: Sampler> {
    dataset: D,
    _sampler: S,
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
#[derive(Debug)]
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
            _sampler: self.sampler,
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
        let samples = vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
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
        let samples = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
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

    #[test]
    fn test_batch_into_chunks_basic() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let chunks = batch.into_chunks(25).unwrap();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].tokens.len(), 25);
    }

    #[test]
    fn test_batch_into_chunks_respects_seq_len() {
        let batch = Batch::new(vec![1u32; 128], 4, 32).unwrap();
        let chunks = batch.into_chunks(64).unwrap();
        for chunk in &chunks {
            assert_eq!(chunk.tokens.len() % 32, 0);
        }
    }

    #[test]
    fn test_batch_chunks_sum_to_original() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let original_len = batch.tokens.len();
        let chunks = batch.into_chunks(25).unwrap();
        let total: usize = chunks.iter().map(|c| c.tokens.len()).sum();
        assert_eq!(total, original_len);
    }

    #[test]
    fn test_chunk_iterator_basic() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let chunk_iter = batch.chunks(25).unwrap();
        let chunks: Vec<_> = chunk_iter.collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(chunks.len(), 4);
        for chunk in &chunks {
            assert_eq!(chunk.tokens.len(), 25);
        }
    }

    #[test]
    fn test_chunk_iterator_respects_seq_len() {
        let batch = Batch::new(vec![1u32; 128], 4, 32).unwrap();
        let chunk_iter = batch.chunks(64).unwrap();
        let chunks: Vec<_> = chunk_iter.collect::<Result<Vec<_>>>().unwrap();
        for chunk in &chunks {
            assert_eq!(chunk.tokens.len() % 32, 0);
        }
    }

    #[test]
    fn test_chunk_iterator_batch_size_calculation() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let chunk_iter = batch.chunks(25).unwrap();
        let chunks: Vec<_> = chunk_iter.collect::<Result<Vec<_>>>().unwrap();

        // Each chunk should have batch_size = tokens / seq_len = 25 / 25 = 1
        for chunk in &chunks {
            assert_eq!(chunk.batch_size(), 1);
            assert_eq!(chunk.seq_len(), 25);
        }
    }

    #[test]
    fn test_chunk_iterator_sum_to_original() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let original_len = batch.tokens.len();
        let chunk_iter = batch.chunks(25).unwrap();
        let chunks: Vec<_> = chunk_iter.collect::<Result<Vec<_>>>().unwrap();
        let total: usize = chunks.iter().map(|c| c.tokens.len()).sum();
        assert_eq!(total, original_len);
    }

    #[test]
    fn test_chunk_iterator_invalid_max_chunk() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let result = batch.chunks(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_into_chunks_batch_size_calculation() {
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
        let chunks = batch.into_chunks(25).unwrap();

        // Each chunk should have batch_size = tokens / seq_len = 25 / 25 = 1
        for chunk in &chunks {
            assert_eq!(chunk.batch_size(), 1);
            assert_eq!(chunk.seq_len(), 25);
        }
    }

    #[test]
    fn test_chunks_guard_seq_len_zero() {
        // CRITICAL: Guard against division by zero in chunks()
        let batch = Batch {
            tokens: vec![1, 2, 3],
            batch_size: 1,
            seq_len: 0, // Invalid, should trigger guard
        };
        let result = batch.chunks(10);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("seq_len must be > 0"));
    }

    #[test]
    fn test_into_chunks_guard_seq_len_zero() {
        // CRITICAL: Guard against division by zero in into_chunks()
        let batch = Batch {
            tokens: vec![1, 2, 3],
            batch_size: 1,
            seq_len: 0, // Invalid, should trigger guard
        };
        let result = batch.into_chunks(10);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("seq_len must be > 0"));
    }

    #[test]
    fn test_chunks_borrows_self() {
        // HIGH: chunks() should borrow &self, not consume self
        let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();

        // Should be able to call chunks() multiple times on the same batch
        let _chunks1 = batch.chunks(25).unwrap();
        let _chunks2 = batch.chunks(50).unwrap(); // Would fail if chunks() took self

        // Batch is still available after both calls
        assert_eq!(batch.batch_size(), 4);
        assert_eq!(batch.seq_len(), 25);
    }
}
