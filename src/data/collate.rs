//! Collation strategies for preparing batches

use super::Batch;
use crate::Result;

/// Trait for collating samples into batches
///
/// A collator defines how to combine multiple samples into a single batch.
/// Common strategies include padding to fixed length, truncating long sequences,
/// or packing variable-length sequences.
pub trait Collator: Send {
    /// Collate samples into a batch
    ///
    /// # Arguments
    /// - `samples`: Vector of token sequences (may have different lengths)
    ///
    /// # Returns
    /// A properly formatted Batch ready for processing
    fn collate(&self, samples: Vec<Vec<u32>>) -> Result<Batch>;
}

/// Pad all sequences to a fixed length
///
/// Shorter sequences are padded with a pad token (default: 0).
/// Longer sequences are NOT truncated - returns error if any sequence
/// exceeds the target length.
///
/// # Example
///
/// ```
/// use rustane::data::{PadCollator, Collator};
///
/// let collator = PadCollator::new(10, 0); // max_len=10, pad_token=0
/// let samples = vec![
///     vec![1, 2, 3],
///     vec![4, 5],
/// ];
/// let batch = collator.collate(samples).unwrap();
/// assert_eq!(batch.seq_len(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct PadCollator {
    max_len: usize,
    pad_token: u32,
}

impl PadCollator {
    /// Create a new pad collator
    ///
    /// # Arguments
    /// - `max_len`: Target sequence length (sequences shorter than this will be padded)
    /// - `pad_token`: Token ID to use for padding
    pub fn new(max_len: usize, pad_token: u32) -> Self {
        PadCollator { max_len, pad_token }
    }
}

impl Collator for PadCollator {
    fn collate(&self, samples: Vec<Vec<u32>>) -> Result<Batch> {
        if samples.is_empty() {
            return Err(crate::Error::InvalidParameter(
                "cannot collate empty samples".to_string(),
            ));
        }

        // Check that no sample exceeds max_len
        for (idx, sample) in samples.iter().enumerate() {
            if sample.len() > self.max_len {
                return Err(crate::Error::InvalidParameter(format!(
                    "sample {} length {} exceeds max_len {}",
                    idx,
                    sample.len(),
                    self.max_len
                )));
            }
        }

        let batch_size = samples.len();
        let mut tokens = Vec::with_capacity(batch_size * self.max_len);

        for sample in samples {
            tokens.extend_from_slice(&sample);
            // Pad with pad_token
            for _ in sample.len()..self.max_len {
                tokens.push(self.pad_token);
            }
        }

        Batch::new(tokens, batch_size, self.max_len)
    }
}

/// Truncate sequences to a maximum length
///
/// Sequences longer than `max_len` are truncated to exactly that length.
/// Shorter sequences are padded with a pad token.
///
/// # Example
///
/// ```
/// use rustane::data::{TruncateCollator, Collator};
///
/// let collator = TruncateCollator::new(5, 0); // max_len=5, pad_token=0
/// let samples = vec![
///     vec![1, 2, 3, 4, 5, 6, 7], // Will be truncated to [1, 2, 3, 4, 5]
///     vec![8, 9],                 // Will be padded
/// ];
/// let batch = collator.collate(samples).unwrap();
/// assert_eq!(batch.seq_len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct TruncateCollator {
    max_len: usize,
    pad_token: u32,
}

impl TruncateCollator {
    /// Create a new truncate collator
    ///
    /// # Arguments
    /// - `max_len`: Maximum sequence length (longer sequences are truncated)
    /// - `pad_token`: Token ID to use for padding shorter sequences
    pub fn new(max_len: usize, pad_token: u32) -> Self {
        TruncateCollator { max_len, pad_token }
    }
}

impl Collator for TruncateCollator {
    fn collate(&self, samples: Vec<Vec<u32>>) -> Result<Batch> {
        if samples.is_empty() {
            return Err(crate::Error::InvalidParameter(
                "cannot collate empty samples".to_string(),
            ));
        }

        let batch_size = samples.len();
        let mut tokens = Vec::with_capacity(batch_size * self.max_len);

        for sample in samples {
            // Truncate to max_len
            let truncated_len = sample.len().min(self.max_len);
            tokens.extend_from_slice(&sample[..truncated_len]);

            // Pad if necessary
            for _ in truncated_len..self.max_len {
                tokens.push(self.pad_token);
            }
        }

        Batch::new(tokens, batch_size, self.max_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_collator_basic() {
        let collator = PadCollator::new(5, 0);
        let samples = vec![vec![1, 2], vec![3, 4, 5]];
        let batch = collator.collate(samples).unwrap();
        assert_eq!(batch.shape(), (2, 5));
        // First sample: [1, 2, 0, 0, 0]
        assert_eq!(batch.get(0, 0), Some(1));
        assert_eq!(batch.get(0, 1), Some(2));
        assert_eq!(batch.get(0, 2), Some(0));
        // Second sample: [3, 4, 5, 0, 0]
        assert_eq!(batch.get(1, 0), Some(3));
        assert_eq!(batch.get(1, 4), Some(0));
    }

    #[test]
    fn test_pad_collator_exact_length() {
        let collator = PadCollator::new(3, 0);
        let samples = vec![vec![1, 2, 3]];
        let batch = collator.collate(samples).unwrap();
        assert_eq!(batch.tokens(), &[1, 2, 3]);
    }

    #[test]
    fn test_pad_collator_exceeds_max_len() {
        let collator = PadCollator::new(3, 0);
        let samples = vec![vec![1, 2, 3, 4]];
        let result = collator.collate(samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_collator_empty() {
        let collator = PadCollator::new(5, 0);
        let result = collator.collate(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncate_collator_basic() {
        let collator = TruncateCollator::new(3, 0);
        let samples = vec![vec![1, 2, 3, 4, 5], vec![6]];
        let batch = collator.collate(samples).unwrap();
        assert_eq!(batch.shape(), (2, 3));
        // First sample truncated: [1, 2, 3]
        assert_eq!(batch.get(0, 0), Some(1));
        assert_eq!(batch.get(0, 1), Some(2));
        assert_eq!(batch.get(0, 2), Some(3));
        // Second sample padded: [6, 0, 0]
        assert_eq!(batch.get(1, 0), Some(6));
        assert_eq!(batch.get(1, 1), Some(0));
    }

    #[test]
    fn test_truncate_collator_no_truncation_needed() {
        let collator = TruncateCollator::new(5, 0);
        let samples = vec![vec![1, 2], vec![3]];
        let batch = collator.collate(samples).unwrap();
        assert_eq!(batch.shape(), (2, 5));
    }

    #[test]
    fn test_truncate_collator_custom_pad_token() {
        let collator = TruncateCollator::new(4, 99);
        let samples = vec![vec![1]];
        let batch = collator.collate(samples).unwrap();
        assert_eq!(batch.tokens(), &[1, 99, 99, 99]);
    }

    #[test]
    fn test_pad_collator_custom_pad_token() {
        let collator = PadCollator::new(4, 255);
        let samples = vec![vec![1, 2]];
        let batch = collator.collate(samples).unwrap();
        assert_eq!(batch.tokens(), &[1, 2, 255, 255]);
    }
}
