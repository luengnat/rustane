//! Loss functions for model training

use crate::error::Result;
use crate::wrapper::ANETensor;
use crate::data::Batch;

/// Trait for loss computation
///
/// Loss functions take model output and batch targets, compute a scalar loss.
/// They are injected into the Trainer for flexibility.
pub trait LossFn: Send {
    /// Compute scalar loss from model output and batch targets
    ///
    /// # Arguments
    /// - `logits`: Model output tensor from forward pass
    /// - `batch`: Batch containing targets (token IDs)
    ///
    /// # Returns
    /// Scalar loss value (f32)
    ///
    /// # Errors
    /// Returns error if loss computation fails (shape mismatch, invalid values, etc.)
    fn compute(&self, logits: &ANETensor, batch: &Batch) -> Result<f32>;
}

/// Cross-entropy loss for language modeling (next-token prediction)
///
/// Standard loss for autoregressive models: predicting next token given context.
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss function
    pub fn new() -> Self {
        CrossEntropyLoss
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for CrossEntropyLoss {
    fn compute(&self, _logits: &ANETensor, _batch: &Batch) -> Result<f32> {
        // Placeholder: In production, this extracts:
        // 1. Logits shape: [batch_size, seq_len, vocab_size]
        // 2. Target tokens from batch
        // 3. Computes cross-entropy loss per position
        // 4. Returns mean loss
        //
        // For now, return a dummy value to enable testing
        Ok(1.0)
    }
}

/// Mean Squared Error loss for regression tasks
///
/// Useful for non-language-modeling objectives (e.g., value prediction, token embedding)
#[derive(Debug, Clone)]
pub struct MSELoss;

impl MSELoss {
    /// Create a new MSE loss function
    pub fn new() -> Self {
        MSELoss
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for MSELoss {
    fn compute(&self, _logits: &ANETensor, _batch: &Batch) -> Result<f32> {
        // Placeholder: In production, this would:
        // 1. Compare logits to targets element-wise
        // 2. Compute (predicted - target)^2 for each element
        // 3. Return mean squared error
        Ok(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss_creation() {
        let loss = CrossEntropyLoss::new();
        assert_eq!(std::mem::size_of_val(&loss), 0); // Zero-sized type
    }

    #[test]
    fn test_mse_loss_creation() {
        let loss = MSELoss::new();
        assert_eq!(std::mem::size_of_val(&loss), 0); // Zero-sized type
    }
}
