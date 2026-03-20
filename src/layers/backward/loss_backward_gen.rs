//! Cross-entropy loss backward pass MIL generator
//!
//! Generates MIL code for cross-entropy loss backward pass.
//!
//! # Forward Pass
//! ```text
//! loss = -log(softmax(logits)[target])
//! where softmax(logits)[i] = exp(logits[i]) / sum(exp(logits))
//! ```
//!
//! # Backward Pass
//! Computes gradient of cross-entropy loss w.r.t. logits:
//! ```text
//! dL/dlogits = softmax(logits) - one_hot(target)
//! ```
//!
//! # Mathematical Derivation
//!
//! Given loss `L = -log(p[target])` where `p = softmax(logits)`:
//!
//! ```text
//! // Softmax output
//! p[i] = exp(logits[i]) / sum(exp(logits))
//!
//! // Gradient
//! dL/dlogits[i] = p[i] - 1 if i == target
//!                = p[i]   otherwise
//!
//! // Vector form
//! dL/dlogits = softmax(logits) - one_hot(target)
//! ```
//!
//! This is numerically stable because we compute softmax using max-logit subtraction.

use crate::training::TransformerConfig;
use crate::ane::Result;
use super::BackwardMILGenerator;

/// MIL generator for cross-entropy loss backward pass
#[derive(Debug)]
pub struct LossBackwardGen;

impl LossBackwardGen {
    /// Create new loss backward MIL generator
    pub fn new() -> Self {
        LossBackwardGen
    }

    /// Generate MIL code for loss backward operation
    ///
    /// # MIL Structure
    /// ```text
    /// Inputs:
    ///   - logits: Model output logits [batch_size, seq_len, vocab_size]
    ///   - targets: Ground truth token indices [batch_size, seq_len]
    ///
    /// Outputs:
    ///   - d_logits: Gradient w.r.t. logits [batch_size, seq_len, vocab_size]
    /// ```
    ///
    /// # Numerical Stability
    ///
    /// Softmax uses max-logit subtraction to prevent overflow:
    /// ```text
    /// max_logit = max(logits)
    /// exp_logits = exp(logits - max_logit)
    /// softmax = exp_logits / sum(exp_logits)
    /// ```
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let _batch_size = 1; // Will be dynamic
        let _seq_len = config.seq_len;
        let vocab_size = config.vocab_size;

        format!(r#"
#!irms6
schema loss_backward_schema {{
    input logits: tensor<batch_sizexseq_lenx{vocab_size}xf32> = Input()
    input targets: tensor<batch_sizexseq_lenxi32> = Input()
    output d_logits: tensor<batch_sizexseq_lenx{vocab_size}xf32> = Output()
}}

main loss_backward(
    logits: tensor<batch_sizexseq_lenx{vocab_size}xf32>,
    targets: tensor<batch_sizexseq_lenxi32>
) -> (d_logits: tensor<batch_sizexseq_lenx{vocab_size}xf32>) {{

    // ===== Compute stable softmax =====
    // max_logit = max(logits, axis=-1, keep_dims=true)
    // exp_logits = exp(logits - max_logit)
    // softmax = exp_logits / sum(exp_logits, axis=-1, keep_dims=true)

    let max_logit = reduce_max(logits, axes=[2], keep_dims=true)
    let logits_stable = logits - max_logit
    let exp_logits = exp(logits_stable)
    let sum_exp = reduce_sum(exp_logits, axes=[2], keep_dims=true)
    let softmax = exp_logits / sum_exp

    // ===== Create one-hot encoding for targets =====
    // one_hot[i, j, k] = 1 if targets[i, j] == k else 0

    // Reshape targets for broadcasting
    let targets_expanded = expand_dims(targets, axes=[2])
    let target_range = const_0_to_{vocab_size}()
    let one_hot = equal(targets_expanded, target_range)
    let one_hot_float = cast(one_hot, dtype=float32)

    // ===== Compute gradient =====
    // dL/dlogits = softmax(logits) - one_hot(target)

    let d_logits = softmax - one_hot_float

    // Return gradient
    return (d_logits)
}}

// Helper: Create tensor [0, 1, 2, ..., vocab_size-1]
const_0_to_{vocab_size}(): tensor<{vocab_size}xi32> {{
    return range(0, {vocab_size}, 1)
}}
"#)
    }
}

impl Default for LossBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for LossBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, _config: &TransformerConfig) -> Result<()> {
        // TODO: Implement validation in Phase 3b
        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "loss_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_backward_gen_creation() {
        let gen = LossBackwardGen::new();
        assert_eq!(gen.operation_name(), "loss_backward");
    }

    #[test]
    fn test_loss_backward_gen_default() {
        let gen = LossBackwardGen::default();
        assert_eq!(gen.operation_name(), "loss_backward");
    }

    #[test]
    fn test_loss_backward_generate_mil() {
        let gen = LossBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config);

        assert!(mil_code.is_ok());
        let mil = mil_code.unwrap();
        assert!(mil.contains("loss_backward"));
        assert!(mil.contains("d_logits"));
    }

    #[test]
    fn test_loss_backward_mil_structure() {
        let gen = LossBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        // Verify MIL contains required sections
        assert!(mil_code.contains("schema"));
        assert!(mil_code.contains("input"));
        assert!(mil_code.contains("output"));
        assert!(mil_code.contains("main"));
        assert!(mil_code.contains("return"));

        // Verify mathematical operations
        assert!(mil_code.contains("reduce_max"));
        assert!(mil_code.contains("exp"));
        assert!(mil_code.contains("reduce_sum"));
        assert!(mil_code.contains("equal"));
        assert!(mil_code.contains("cast"));

        // Verify numerical stability
        assert!(mil_code.contains("max_logit"));
        assert!(mil_code.contains("logits_stable"));
    }
}
