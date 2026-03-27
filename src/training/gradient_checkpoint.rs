//! Gradient Checkpointing for Memory Efficiency
//!
//! Gradient checkpointing (also known as activation checkpointing) reduces memory
//! usage during training by recomputing activations during the backward pass instead
//! of storing all intermediate results.
//!
//! # Memory-Compute Tradeoff
//!
//! - **Standard Training**: Store all activations → O(n) memory, O(1) compute
//! - **Checkpointing**: Store select checkpoints → O(√n) memory, O(2) compute
//!
//! For large models, this can reduce memory usage by 4-8x with only 20-30% compute overhead.
//!
//! # Checkpoint Strategies
//!
//! 1. **Uniform Checkpointing**: Save every Nth layer
//! 2. **Block Checkpointing**: Save at block boundaries (e.g., transformer blocks)
//! 3. **Custom Checkpointing**: User-specified checkpoint locations
//!
//! # Quick Start
//!
//! ```no_run
//! use rustane::training::checkpoint::{CheckpointStrategy, CheckpointManager};
//!
//! // Strategy 1: Save every 4th layer
//! let strategy = CheckpointStrategy::every_n_layers(4);
//!
//! // Strategy 2: Save at block boundaries (e.g., transformer blocks)
//! let strategy = CheckpointStrategy::block_boundaries(vec![0, 4, 8, 12]);
//!
//! // Strategy 3: Custom checkpoint locations
//! let strategy = CheckpointStrategy::custom(vec![0, 2, 5, 9]);
//!
//! // Create checkpoint manager
//! let mut manager = CheckpointManager::new(strategy, 12);
//!
//! // Forward pass with checkpointing
//! let mut ctx = manager.begin_forward();
//! for (layer_idx, layer) in layers.iter() {
//!     let save = ctx.should_save(layer_idx);
//!     let output = layer.forward(&input);
//!     if save {
//!         ctx.save_activation(layer_idx, output.clone());
//!     }
//! }
//! let checkpoints = ctx.finish();
//!
//! // Backward pass with recomputation
//! for (layer_idx, layer) in layers.iter().rev() {
//!     let needs_recompute = !manager.is_checkpoint(layer_idx);
//!     if needs_recompute {
//!         // Recompute activations from last checkpoint
//!         let activations = manager.recompute_range(&checkpoints, layer_idx, &layer);
//!         // Use recomputed activations for gradient
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};

use half::f16;

/// Activation storage for checkpointed tensors
#[derive(Debug, Clone)]
pub struct Activation {
    /// Flattened tensor data (fp16 for memory efficiency)
    pub data: Vec<f16>,
    /// Tensor shape [batch, seq_len, features]
    pub shape: Vec<usize>,
    /// Optional layer metadata
    pub metadata: Option<String>,
}

impl Activation {
    /// Create a new activation from fp32 data (converted to fp16 for storage)
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let fp16_data: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        Self {
            data: fp16_data,
            shape,
            metadata: None,
        }
    }

    /// Create a new activation from fp16 data directly
    pub fn from_f16(data: Vec<f16>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            metadata: None,
        }
    }

    /// Convert back to fp32 for computation
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x.to_f32()).collect()
    }

    /// Get total element count
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// Get memory size in bytes (fp16 = 2 bytes per element)
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 2
    }
}

/// Checkpoint strategy for determining which activations to save
#[derive(Debug, Clone)]
pub enum CheckpointStrategy {
    /// Save every Nth layer (0, N, 2N, ...)
    EveryNthLayer { n: usize },
    /// Save at specific layer indices
    BlockBoundaries { indices: Vec<usize> },
    /// Custom strategy with user-defined logic
    Custom { indices: Vec<usize> },
}

impl CheckpointStrategy {
    /// Create a strategy that saves every Nth layer
    pub fn every_n_layers(n: usize) -> Self {
        Self::EveryNthLayer { n }
    }

    /// Create a strategy that saves at block boundaries
    pub fn block_boundaries(indices: Vec<usize>) -> Self {
        Self::BlockBoundaries { indices }
    }

    /// Create a custom checkpoint strategy
    pub fn custom(indices: Vec<usize>) -> Self {
        Self::Custom { indices }
    }

    /// Check if a layer should be saved as a checkpoint
    pub fn should_save(&self, layer_idx: usize, total_layers: usize) -> bool {
        match self {
            CheckpointStrategy::EveryNthLayer { n } => layer_idx % n == 0,
            CheckpointStrategy::BlockBoundaries { indices }
            | CheckpointStrategy::Custom { indices } => {
                indices.contains(&layer_idx) || layer_idx == total_layers - 1
            }
        }
    }

    /// Get estimated memory savings vs full activation storage
    /// Returns ratio: checkpoint_memory / full_memory (lower = better savings)
    pub fn memory_ratio(&self, total_layers: usize) -> f64 {
        let num_checkpoints = match self {
            CheckpointStrategy::EveryNthLayer { n } => (total_layers + n - 1) / n,
            CheckpointStrategy::BlockBoundaries { indices }
            | CheckpointStrategy::Custom { indices } => {
                indices.len() + 1 // +1 for final layer always saved
            }
        };
        num_checkpoints as f64 / total_layers as f64
    }
}

/// Context for forward pass with checkpointing
pub struct CheckpointContext<'a> {
    /// Strategy for determining checkpoints
    strategy: &'a CheckpointStrategy,
    /// Saved activations
    activations: HashMap<usize, Activation>,
    /// Total number of layers
    total_layers: usize,
    #[allow(dead_code)]
    /// Current layer being processed
    current_layer: usize,
}

impl<'a> CheckpointContext<'a> {
    /// Create a new checkpoint context
    pub fn new(strategy: &'a CheckpointStrategy, total_layers: usize) -> Self {
        Self {
            strategy,
            activations: HashMap::new(),
            total_layers,
            current_layer: 0,
        }
    }

    /// Check if current layer should be saved as checkpoint
    pub fn should_save(&self, layer_idx: usize) -> bool {
        self.strategy.should_save(layer_idx, self.total_layers)
    }

    /// Save an activation for later recomputation
    pub fn save_activation(&mut self, layer_idx: usize, activation: Activation) {
        self.activations.insert(layer_idx, activation);
    }

    /// Get a saved activation if it exists
    pub fn get_activation(&self, layer_idx: usize) -> Option<&Activation> {
        self.activations.get(&layer_idx)
    }

    /// Get all checkpoint indices
    pub fn checkpoint_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.activations.keys().cloned().collect();
        indices.sort();
        indices
    }

    /// Get total memory used by saved activations (bytes)
    pub fn memory_usage(&self) -> usize {
        self.activations.values().map(|a| a.memory_bytes()).sum()
    }

    /// Finish forward pass and return checkpoint data
    pub fn finish(self) -> CheckpointData {
        // Get checkpoint indices before moving activations
        let mut indices: Vec<usize> = self.activations.keys().cloned().collect();
        indices.sort();

        CheckpointData {
            activations: self.activations,
            checkpoint_indices: indices,
        }
    }
}

/// Checkpoint data returned from forward pass
#[derive(Debug, Clone)]
pub struct CheckpointData {
    /// Saved activations at checkpoint locations
    pub activations: HashMap<usize, Activation>,
    /// Indices of saved checkpoints
    pub checkpoint_indices: Vec<usize>,
}

impl CheckpointData {
    /// Get activation at a specific layer
    pub fn get_activation(&self, layer_idx: usize) -> Option<&Activation> {
        self.activations.get(&layer_idx)
    }

    /// Get the last checkpoint before a given layer
    pub fn get_last_checkpoint_before(&self, layer_idx: usize) -> Option<(usize, &Activation)> {
        self.checkpoint_indices
            .iter()
            .filter(|&&idx| idx < layer_idx)
            .last()
            .and_then(|&idx| self.activations.get(&idx).map(|act| (idx, act)))
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.activations.values().map(|a| a.memory_bytes()).sum()
    }
}

/// Manager for gradient checkpointing during training
pub struct CheckpointManager {
    /// Strategy for checkpoint selection
    strategy: CheckpointStrategy,
    /// Total layers in the model
    total_layers: usize,
    /// Set of checkpoint layer indices for fast lookup
    checkpoint_set: HashSet<usize>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(strategy: CheckpointStrategy, total_layers: usize) -> Self {
        let mut checkpoint_set = HashSet::new();

        // Build checkpoint set based on strategy
        for layer_idx in 0..total_layers {
            if strategy.should_save(layer_idx, total_layers) {
                checkpoint_set.insert(layer_idx);
            }
        }

        Self {
            strategy,
            total_layers,
            checkpoint_set,
        }
    }

    /// Check if a layer is a checkpoint
    pub fn is_checkpoint(&self, layer_idx: usize) -> bool {
        self.checkpoint_set.contains(&layer_idx)
    }

    /// Get checkpoint strategy
    pub fn strategy(&self) -> &CheckpointStrategy {
        &self.strategy
    }

    /// Begin a forward pass with checkpointing
    pub fn begin_forward(&self) -> CheckpointContext<'_> {
        CheckpointContext::new(&self.strategy, self.total_layers)
    }

    /// Get recomputation range for a layer
    /// Returns (start_layer, end_layer) that needs recomputation
    pub fn get_recompute_range(
        &self,
        layer_idx: usize,
        checkpoints: &CheckpointData,
    ) -> (usize, usize) {
        match checkpoints.get_last_checkpoint_before(layer_idx) {
            Some((checkpoint_idx, _)) => (checkpoint_idx + 1, layer_idx + 1),
            None => (0, layer_idx + 1), // Recompute from beginning
        }
    }

    /// Recompute activations for a layer (simulated - actual recomputation happens in training loop)
    ///
    /// This method provides the range of layers that need to be recomputed.
    /// The actual recomputation logic is model-specific and should be implemented
    /// by the caller based on their model architecture.
    pub fn prepare_recompute<F, G>(
        &self,
        layer_idx: usize,
        checkpoints: &CheckpointData,
        mut recompute_fn: F,
        mut save_fn: G,
    ) -> Result<Activation, String>
    where
        F: FnMut(usize, &Activation) -> Result<Activation, String>,
        G: FnMut(usize, Activation),
    {
        let (start, end) = self.get_recompute_range(layer_idx, checkpoints);

        // Get starting activation (either from checkpoint or input)
        let start_activation = if start == 0 {
            // Start from model input - caller should provide this
            return Err("Initial input required for recomputation from layer 0".to_string());
        } else {
            checkpoints
                .get_activation(start - 1)
                .cloned()
                .ok_or_else(|| format!("Checkpoint not found at layer {}", start - 1))?
        };

        // Recompute through the range
        let mut current_activation = start_activation;
        for l in start..end {
            current_activation = recompute_fn(l, &current_activation)?;

            // Save intermediate results if they are checkpoints
            if self.is_checkpoint(l) && l < layer_idx {
                save_fn(l, current_activation.clone());
            }
        }

        Ok(current_activation)
    }

    /// Get estimated memory savings vs full activation storage
    pub fn memory_savings_ratio(&self) -> f64 {
        self.strategy.memory_ratio(self.total_layers)
    }

    /// Get expected number of checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoint_set.len()
    }

    /// Get checkpoint indices
    pub fn checkpoint_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.checkpoint_set.iter().cloned().collect();
        indices.sort();
        indices
    }

    /// Print checkpoint configuration and expected savings
    pub fn print_summary(&self) {
        let ratio = self.memory_savings_ratio();
        let memory_reduction = (1.0 - ratio) * 100.0;

        println!("\n=== Gradient Checkpointing Configuration ===\n");
        println!("Total layers: {}", self.total_layers);
        println!("Number of checkpoints: {}", self.num_checkpoints());
        println!("Checkpoint indices: {:?}", self.checkpoint_indices());
        println!("\nMemory Efficiency:");
        println!("  - Activation memory reduction: {:.1}%", memory_reduction);
        println!("  - Memory ratio: {:.2}x (vs full storage)", ratio);
        println!(
            "  - Estimated compute overhead: ~{:.0}%",
            (1.0 - ratio) * 100.0
        );
        println!("\nStrategy: {:?}", self.strategy);
    }
}

/// Builder for configuring gradient checkpointing
pub struct CheckpointBuilder {
    total_layers: usize,
    strategy: Option<CheckpointStrategy>,
}

impl CheckpointBuilder {
    /// Create a new checkpoint builder
    pub fn new(total_layers: usize) -> Self {
        Self {
            total_layers,
            strategy: None,
        }
    }

    /// Set checkpoint strategy to every Nth layer
    pub fn every_n_layers(mut self, n: usize) -> Self {
        self.strategy = Some(CheckpointStrategy::every_n_layers(n));
        self
    }

    /// Set checkpoint strategy to block boundaries
    pub fn block_boundaries(mut self, indices: Vec<usize>) -> Self {
        self.strategy = Some(CheckpointStrategy::block_boundaries(indices));
        self
    }

    /// Set custom checkpoint strategy
    pub fn custom(mut self, indices: Vec<usize>) -> Self {
        self.strategy = Some(CheckpointStrategy::custom(indices));
        self
    }

    /// Build the checkpoint manager
    pub fn build(self) -> Result<CheckpointManager, String> {
        let strategy = self
            .strategy
            .ok_or_else(|| "Checkpoint strategy not specified".to_string())?;
        Ok(CheckpointManager::new(strategy, self.total_layers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let activation = Activation::from_f32(&data, vec![2, 2]);

        assert_eq!(activation.num_elements(), 4);
        assert_eq!(activation.memory_bytes(), 8); // 4 elements * 2 bytes (fp16)

        let recovered = activation.to_f32();
        assert!((recovered[0] - 1.0).abs() < 0.01);
        assert!((recovered[3] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_strategy_every_n() {
        let strategy = CheckpointStrategy::every_n_layers(4);

        assert!(strategy.should_save(0, 12));
        assert!(!strategy.should_save(1, 12));
        assert!(strategy.should_save(4, 12));
        assert!(strategy.should_save(8, 12));

        // Memory ratio should be ~1/4 for every 4th layer
        let ratio = strategy.memory_ratio(12);
        assert!(ratio < 0.35); // ~0.25 + final layer
    }

    #[test]
    fn test_checkpoint_strategy_block_boundaries() {
        let strategy = CheckpointStrategy::block_boundaries(vec![0, 4, 8]);

        assert!(strategy.should_save(0, 12));
        assert!(strategy.should_save(4, 12));
        assert!(strategy.should_save(8, 12));
        assert!(strategy.should_save(11, 12)); // Last layer always saved

        let ratio = strategy.memory_ratio(12);
        assert_eq!(ratio, 4.0 / 12.0); // 3 checkpoints + 1 final = 4 / 12
    }

    #[test]
    fn test_checkpoint_context() {
        let strategy = CheckpointStrategy::every_n_layers(3);
        let mut ctx = CheckpointContext::new(&strategy, 9);

        // Simulate forward pass
        for layer in 0..9 {
            if ctx.should_save(layer) {
                let data = vec![layer as f32; 16];
                ctx.save_activation(layer, Activation::from_f32(&data, vec![4, 4]));
            }
        }

        let checkpoints = ctx.finish();

        // Verify checkpoints at 0, 3, 6
        assert!(checkpoints.get_activation(0).is_some());
        assert!(checkpoints.get_activation(3).is_some());
        assert!(checkpoints.get_activation(6).is_some());
        assert!(checkpoints.get_activation(1).is_none());
    }

    #[test]
    fn test_checkpoint_manager_recompute_range() {
        let strategy = CheckpointStrategy::every_n_layers(4);
        let manager = CheckpointManager::new(strategy, 12);

        // Create mock checkpoint data
        let mut activations = HashMap::new();
        activations.insert(0, Activation::from_f32(&vec![0.0f32; 16], vec![4, 4]));
        activations.insert(4, Activation::from_f32(&vec![0.0f32; 16], vec![4, 4]));
        activations.insert(8, Activation::from_f32(&vec![0.0f32; 16], vec![4, 4]));

        let checkpoints = CheckpointData {
            activations,
            checkpoint_indices: vec![0, 4, 8],
        };

        // Layer 5: last checkpoint before is 4, so recompute range is (4+1, 5+1) = (5, 6)
        let (start, end) = manager.get_recompute_range(5, &checkpoints);
        assert_eq!(start, 5);
        assert_eq!(end, 6);

        // Layer 7: last checkpoint before is 4, so recompute range is (4+1, 7+1) = (5, 8)
        let (start, end) = manager.get_recompute_range(7, &checkpoints);
        assert_eq!(start, 5);
        assert_eq!(end, 8);

        // Layer 2: last checkpoint before is 0, so recompute range is (0+1, 2+1) = (1, 3)
        let (start, end) = manager.get_recompute_range(2, &checkpoints);
        assert_eq!(start, 1);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_memory_savings() {
        // Every 4th layer: ~25% memory usage
        let strategy_4 = CheckpointStrategy::every_n_layers(4);
        let manager_4 = CheckpointManager::new(strategy_4, 16);
        assert!(manager_4.memory_savings_ratio() < 0.35);

        // Every 2nd layer: ~50% memory usage
        let strategy_2 = CheckpointStrategy::every_n_layers(2);
        let manager_2 = CheckpointManager::new(strategy_2, 16);
        assert!(manager_2.memory_savings_ratio() < 0.60);

        // Savings should improve with more layers
        assert!(manager_4.memory_savings_ratio() < manager_2.memory_savings_ratio());
    }

    #[test]
    fn test_checkpoint_builder() {
        let manager = CheckpointBuilder::new(12)
            .every_n_layers(3)
            .build()
            .unwrap();

        // With every_n_layers(3) for 12 layers: 0, 3, 6, 9 = 4 checkpoints
        // (layer 11 is not saved because it's not a multiple of 3)
        assert_eq!(manager.num_checkpoints(), 4);
        assert!(manager.is_checkpoint(0));
        assert!(manager.is_checkpoint(3));
        assert!(!manager.is_checkpoint(1));
    }
}
