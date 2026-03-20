//! Sampler trait and implementations

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Trait for sampling indices from a dataset
///
/// Implementers define the strategy for selecting which indices to access
/// when iterating over a dataset. Common implementations include sequential
/// sampling, random sampling, and distributed sampling.
pub trait Sampler: Send {
    /// Produce a sequence of indices to sample
    ///
    /// Returns a vector of indices that will be used to fetch samples
    /// from the dataset. The sampler can use any strategy (sequential,
    /// random, stratified, etc.) to produce these indices.
    ///
    /// # Returns
    /// A vector of indices (0-based) to sample from the dataset
    fn sample(&mut self) -> Vec<usize>;
}

/// Sequential sampler that returns indices in order
///
/// Iterates through all indices from 0 to num_samples-1 in order.
/// This is useful for deterministic iteration over a dataset, particularly
/// during evaluation or validation.
///
/// # Example
///
/// ```
/// use rustane::data::{SequentialSampler, Sampler};
///
/// let mut sampler = SequentialSampler::new(5);
/// let indices = sampler.sample();
/// assert_eq!(indices, vec![0, 1, 2, 3, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    num_samples: usize,
}

impl SequentialSampler {
    /// Create a new sequential sampler
    ///
    /// # Arguments
    /// - `num_samples`: Total number of samples to sample from
    pub fn new(num_samples: usize) -> Self {
        SequentialSampler { num_samples }
    }
}

impl Sampler for SequentialSampler {
    fn sample(&mut self) -> Vec<usize> {
        (0..self.num_samples).collect()
    }
}

/// Random sampler with deterministic seeding
///
/// Shuffles indices randomly. Useful for training data sampling.
/// Use a fixed seed for reproducibility.
///
/// # Example
///
/// ```
/// use rustane::data::{RandomSampler, Sampler};
///
/// let mut sampler = RandomSampler::new(5, 42); // seed=42
/// let indices = sampler.sample();
/// assert_eq!(indices.len(), 5);
/// // Indices will be a shuffled permutation of [0, 1, 2, 3, 4]
/// ```
#[derive(Debug, Clone)]
pub struct RandomSampler {
    num_samples: usize,
    seed: u64,
}

impl RandomSampler {
    /// Create a new random sampler with a specific seed
    ///
    /// # Arguments
    /// - `num_samples`: Total number of samples to sample from
    /// - `seed`: Random seed for reproducibility
    pub fn new(num_samples: usize, seed: u64) -> Self {
        RandomSampler { num_samples, seed }
    }

    /// Create a new random sampler with a default seed
    pub fn default_seed(num_samples: usize) -> Self {
        RandomSampler::new(num_samples, 0)
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..self.num_samples).collect();
        indices.shuffle(&mut rng);
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler_produces_indices() {
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
    fn test_sequential_sampler_single() {
        let mut sampler = SequentialSampler::new(1);
        let indices = sampler.sample();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_sequential_sampler_large() {
        let mut sampler = SequentialSampler::new(1000);
        let indices = sampler.sample();
        assert_eq!(indices.len(), 1000);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[999], 999);
    }

    #[test]
    fn test_random_sampler_produces_indices() {
        let mut sampler = RandomSampler::new(5, 42);
        let indices = sampler.sample();
        assert_eq!(indices.len(), 5);
        // Check all indices are present (it's a permutation)
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler_deterministic() {
        let mut sampler1 = RandomSampler::new(10, 123);
        let indices1 = sampler1.sample();

        let mut sampler2 = RandomSampler::new(10, 123);
        let indices2 = sampler2.sample();

        // Same seed should produce same shuffled order
        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_random_sampler_different_seeds() {
        let mut sampler1 = RandomSampler::new(10, 123);
        let indices1 = sampler1.sample();

        let mut sampler2 = RandomSampler::new(10, 456);
        let indices2 = sampler2.sample();

        // Different seeds should (very likely) produce different shuffles
        // Although theoretically could be the same, probability is negligible
        assert_ne!(indices1, indices2);
    }

    #[test]
    fn test_random_sampler_empty() {
        let mut sampler = RandomSampler::new(0, 42);
        let indices = sampler.sample();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_random_sampler_single() {
        let mut sampler = RandomSampler::new(1, 42);
        let indices = sampler.sample();
        assert_eq!(indices, vec![0]);
    }
}
