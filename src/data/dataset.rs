//! Dataset trait and implementations

use crate::Result;

/// Trait for accessing samples from a dataset
///
/// Implementers should provide methods to access individual samples
/// by index. Samples are represented as sequences of token IDs (u32).
pub trait Dataset: Send {
    /// Get the total number of samples in the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a sample by index
    ///
    /// # Arguments
    /// - `idx`: Index of the sample to retrieve (0-based)
    ///
    /// # Returns
    /// A vector of token IDs representing the sample
    ///
    /// # Errors
    /// Returns an error if the index is out of bounds
    fn get(&self, idx: usize) -> Result<Vec<u32>>;
}

/// A simple in-memory dataset backed by a Vec
///
/// All samples are stored in memory. Useful for small datasets and testing.
/// Each sample is a sequence of token IDs.
///
/// # Example
///
/// ```
/// use rustane::data::{SequentialDataset, Dataset};
///
/// let samples = vec![
///     vec![0, 1, 2],
///     vec![3, 4, 5],
/// ];
/// let dataset = SequentialDataset::new(samples);
/// assert_eq!(dataset.len(), 2);
/// assert_eq!(dataset.get(0).unwrap(), vec![0, 1, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct SequentialDataset {
    samples: Vec<Vec<u32>>,
}

impl SequentialDataset {
    /// Create a new sequential dataset from a vector of samples
    ///
    /// # Arguments
    /// - `samples`: A vector of token sequences
    pub fn new(samples: Vec<Vec<u32>>) -> Self {
        SequentialDataset { samples }
    }

    /// Get the internal samples (for testing/debugging)
    pub fn inner(&self) -> &[Vec<u32>] {
        &self.samples
    }

    /// Consume and return the internal samples
    pub fn into_inner(self) -> Vec<Vec<u32>> {
        self.samples
    }
}

impl Dataset for SequentialDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        self.samples.get(idx).cloned().ok_or_else(|| {
            crate::Error::InvalidParameter(format!(
                "dataset index out of bounds: {} >= {}",
                idx,
                self.samples.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_dataset_creation() {
        let samples = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let dataset = SequentialDataset::new(samples);
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_sequential_dataset_access() {
        let samples = vec![vec![10, 20, 30], vec![40, 50, 60]];
        let dataset = SequentialDataset::new(samples);
        assert_eq!(dataset.get(0).unwrap(), vec![10, 20, 30]);
        assert_eq!(dataset.get(1).unwrap(), vec![40, 50, 60]);
    }

    #[test]
    fn test_sequential_dataset_out_of_bounds() {
        let dataset = SequentialDataset::new(vec![vec![1, 2]]);
        let result = dataset.get(1);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("out of bounds"));
    }

    #[test]
    fn test_sequential_dataset_empty() {
        let dataset: SequentialDataset = SequentialDataset::new(vec![]);
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
    }
}
