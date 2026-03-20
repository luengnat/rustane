//! Model trait for training orchestration

use crate::data::Batch;
use crate::error::Result;
use crate::wrapper::ANETensor;

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

    /// Batch-aware backward pass.
    ///
    /// Models that cache forward activations and need target context can override
    /// this hook. The default behavior falls back to `backward(loss)` for older
    /// implementations.
    fn backward_with_batch(&mut self, batch: &Batch, loss: f32) -> Result<Vec<f32>> {
        let _ = batch;
        self.backward(loss)
    }

    /// ANE-accelerated backward pass with gradient accumulation
    ///
    /// # Arguments
    /// - `batch`: Reference to the batch used in forward pass (for cached activations)
    /// - `loss`: Scalar loss value from loss function
    /// - `accumulator`: Gradient accumulator for ANE memory
    ///
    /// # Returns
    /// Unit result - gradients are accumulated in the provided accumulator
    ///
    /// # Errors
    /// Returns error if ANE backward pass fails
    ///
    /// # Default Implementation
    /// Falls back to CPU backward and transfers to accumulator (for compatibility)
    ///
    /// # Phase 3 Feature
    /// This is the Phase 3 ANE backward interface. Models implementing this
    /// will compute gradients directly on ANE using backward MIL kernels.
    fn backward_on_ane(
        &mut self,
        batch: &Batch,
        loss: f32,
        accumulator: &mut crate::training::ANEGradientAccumulator,
    ) -> Result<()> {
        // Default: CPU backward → transfer to accumulator
        let grads = self.backward_with_batch(batch, loss)?;
        accumulator
            .accumulate(&grads)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        Ok(())
    }

    /// Get mutable reference to model parameters
    ///
    /// Used by optimizer to update weights in-place.
    fn parameters(&mut self) -> &mut [f32];

    /// Total number of trainable parameters
    fn param_count(&self) -> usize;
}
