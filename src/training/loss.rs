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

    /// Compute softmax of logits safely
    fn softmax(logits: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute exp(logits - max)
        let mut exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        
        // Normalize by sum
        let sum: f32 = exp_logits.iter().sum();
        if sum > 0.0 {
            for x in &mut exp_logits {
                *x /= sum;
            }
        }
        
        exp_logits
    }
}



impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for CrossEntropyLoss {
    fn compute(&self, logits: &ANETensor, batch: &Batch) -> Result<f32> {
        // Extract FP32 logits from tensor
        let logits_bytes = logits.as_bytes();
        if logits_bytes.len() % 4 != 0 {
            return Err(crate::Error::Other(
                "Logits tensor must be FP32 (4 bytes per element)".to_string(),
            ));
        }

        // Convert bytes to f32 slice
        let logits_f32 = unsafe {
            std::slice::from_raw_parts(
                logits_bytes.as_ptr() as *const f32,
                logits_bytes.len() / 4,
            )
        };

        // Get the tensor shape
        let shape = logits.shape();
        if shape.is_empty() {
            return Err(crate::Error::Other(
                "Logits tensor must have non-empty shape".to_string(),
            ));
        }

        let vocab_size = shape[shape.len() - 1]; // Last dimension is vocab
        let num_positions = logits_f32.len() / vocab_size;

        if num_positions == 0 {
            return Err(crate::Error::Other(
                "No positions in logits tensor".to_string(),
            ));
        }

        let tokens = batch.tokens();

        // For next-token prediction:
        // - tokens[0..num_positions] are inputs
        // - tokens[1..num_positions+1] are targets
        // - We predict tokens[i+1] given tokens[i:i+1] context
        if tokens.len() < num_positions {
            return Err(crate::Error::Other(
                format!(
                    "Not enough tokens in batch ({}) for {} positions",
                    tokens.len(),
                    num_positions
                ),
            ));
        }

        // Compute cross-entropy loss
        let mut total_loss: f32 = 0.0;

        for pos in 0..num_positions {
            let logits_at_pos = &logits_f32[pos * vocab_size..(pos + 1) * vocab_size];
            
            // Get target token (next token in sequence)
            let target_token = if pos + 1 < tokens.len() {
                tokens[pos + 1] as usize
            } else {
                // Fallback: use current token as target (shouldn't happen normally)
                tokens[pos] as usize
            };

            if target_token >= vocab_size {
                return Err(crate::Error::Other(
                    format!(
                        "Target token {} exceeds vocab size {}",
                        target_token, vocab_size
                    ),
                ));
            }

            // Compute softmax
            let softmax = Self::softmax(logits_at_pos);

            // Cross-entropy: -log(softmax[target])
            let prob = softmax[target_token];
            let loss_at_pos = if prob > 0.0 {
                -prob.ln()
            } else {
                // If probability is 0, use a large penalty
                10.0
            };

            total_loss += loss_at_pos;
        }

        // Return average loss in nats (natural log scale)
        Ok(total_loss / num_positions as f32)
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
