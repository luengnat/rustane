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

        let vocab_size = shape[shape.len() - 1];
        if vocab_size == 0 {
            return Err(crate::Error::Other(
                "Logits vocab dimension must be > 0".to_string(),
            ));
        }
        if logits_f32.len() % vocab_size != 0 {
            return Err(crate::Error::Other(
                "Logits buffer does not align with vocab dimension".to_string(),
            ));
        }
        let num_positions = logits_f32.len() / vocab_size;

        if num_positions == 0 {
            return Err(crate::Error::Other(
                "No positions in logits tensor".to_string(),
            ));
        }

        let tokens = batch.tokens();
        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let batched_positions = batch_size.saturating_mul(seq_len.saturating_sub(1));
        let flattened_positions = tokens.len().saturating_sub(1);

        let mut total_loss: f32 = 0.0;

        if num_positions == batched_positions && batch_size > 0 && seq_len > 0 {
            for sample_idx in 0..batch_size {
                let sample_offset = sample_idx * seq_len;
                let row_offset = sample_idx * (seq_len - 1);
                for pos in 0..(seq_len - 1) {
                    let row_idx = row_offset + pos;
                    let logits_at_pos = &logits_f32[row_idx * vocab_size..(row_idx + 1) * vocab_size];
                    let target_token = tokens[sample_offset + pos + 1] as usize;
                    if target_token >= vocab_size {
                        return Err(crate::Error::Other(format!(
                            "Target token {} exceeds vocab size {}",
                            target_token, vocab_size
                        )));
                    }
                    let softmax = Self::softmax(logits_at_pos);
                    let prob = softmax[target_token];
                    total_loss += if prob > 0.0 { -prob.ln() } else { 10.0 };
                }
            }
            Ok(total_loss / batched_positions as f32)
        } else if num_positions == flattened_positions && flattened_positions > 0 {
            for pos in 0..num_positions {
                let logits_at_pos = &logits_f32[pos * vocab_size..(pos + 1) * vocab_size];
                let target_token = tokens[pos + 1] as usize;
                if target_token >= vocab_size {
                    return Err(crate::Error::Other(format!(
                        "Target token {} exceeds vocab size {}",
                        target_token, vocab_size
                    )));
                }
                let softmax = Self::softmax(logits_at_pos);
                let prob = softmax[target_token];
                total_loss += if prob > 0.0 { -prob.ln() } else { 10.0 };
            }
            Ok(total_loss / num_positions as f32)
        } else if num_positions == 1 && !tokens.is_empty() {
            let logits_at_pos = &logits_f32[..vocab_size];
            let target_token = if tokens.len() > 1 {
                tokens[1] as usize
            } else {
                tokens[0] as usize
            };
            if target_token >= vocab_size {
                return Err(crate::Error::Other(format!(
                    "Target token {} exceeds vocab size {}",
                    target_token, vocab_size
                )));
            }
            let softmax = Self::softmax(logits_at_pos);
            let prob = softmax[target_token];
            Ok(if prob > 0.0 { -prob.ln() } else { 10.0 })
        } else {
            Err(crate::Error::Other(format!(
                "Logits rows ({}) do not match expected batched positions ({}) or flattened positions ({})",
                num_positions, batched_positions, flattened_positions
            )))
        }
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
    use crate::wrapper::ANETensor;

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

    #[test]
    fn test_cross_entropy_loss_batched_layout() {
        let loss = CrossEntropyLoss::new();
        let batch = Batch::new(vec![1, 2, 3, 0, 1, 2], 2, 3).unwrap();
        let logits = ANETensor::from_fp32(vec![0.0f32; 16], vec![4, 4]).unwrap();

        let value = loss.compute(&logits, &batch).unwrap();
        assert!((value - 4.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_flattened_layout() {
        let loss = CrossEntropyLoss::new();
        let batch = Batch::new(vec![1, 2, 3, 0], 1, 4).unwrap();
        let logits = ANETensor::from_fp32(vec![0.0f32; 12], vec![3, 4]).unwrap();

        let value = loss.compute(&logits, &batch).unwrap();
        assert!((value - 4.0f32.ln()).abs() < 1e-6);
    }
}
