//! Model trait for training orchestration

use crate::error::Result;
use crate::wrapper::ANETensor;
use crate::data::Batch;

/// Trait for models used in training
///
/// Models handle forward pass computation and backward pass gradient computation.
/// They hide the complexity of ANE integration and numerical operations.
pub trait Model: Send {
    /// Forward pass: process a batch and return logits/activations
    ///
    /// # Arguments
    /// - `batch`: Tokenized batch [batch_size × seq_len]
    ///
    /// # Returns
    /// ANETensor with logits/activations for loss computation
    ///
    /// # Errors
    /// Returns error if forward pass fails (shape mismatch, ANE failure, etc.)
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor>;

    /// Backward pass: compute gradients given a scalar loss
    ///
    /// # Arguments
    /// - `loss`: Scalar loss value from loss function
    ///
    /// # Returns
    /// Gradients as Vec<f32>, one gradient per parameter.
    /// Length must match parameter_count().
    ///
    /// # Errors
    /// Returns error if backward pass fails (gradient computation, NaN/Inf, etc.)
    fn backward(&mut self, loss: f32) -> Result<Vec<f32>>;

    /// Get mutable reference to model parameters
    ///
    /// Used by optimizer to update weights in-place.
    fn parameters(&mut self) -> &mut [f32];

    /// Total number of trainable parameters
    fn param_count(&self) -> usize;
}
