//! Feed-forward network backward pass MIL generator
//!
//! Generates MIL code for FFN (SwiGLU) backward pass.
//!
//! # Forward Pass
//! ```text
//! gate = SiLU(X @ W_gate)
//! up = X @ W_up
//! out = gate * up
//! output = out @ W_down
//! ```
//!
//! # Backward Pass
//! Computes gradients for all weights and inputs:
//! - `d_W_gate`, `d_W_up`, `d_W_down`: Gradients w.r.t. projection weights
//! - `d_X`: Gradient w.r.t. input (accumulated from gate and up paths)
//!
//! # Mathematical Derivation
//!
//! Given output gradient `d_out`:
//!
//! ```text
//! // Output projection backward
//! d_out_ffn = d_out @ W_down^T
//! d_W_down = out^T @ d_out
//!
//! // Element-wise multiplication backward
//! d_gate = d_out_ffn * up
//! d_up = d_out_ffn * gate
//!
//! // Up projection backward
//! d_W_up = X^T @ d_up
//! d_X_up = d_up @ W_up^T
//!
//! // Gate projection backward (with SiLU activation)
//! // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//! // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//! //         = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//! d_gate_proj = d_gate * SiLU'(gate_proj)
//! d_W_gate = X^T @ d_gate_proj
//! d_X_gate = d_gate_proj @ W_gate^T
//!
//! // Input gradient (accumulated)
//! d_X = d_X_up + d_X_gate
//! ```

use crate::training::TransformerConfig;
use crate::ane::Result;
use super::BackwardMILGenerator;

/// MIL generator for feed-forward network backward pass
#[derive(Debug)]
pub struct FFNBackwardGen;

impl FFNBackwardGen {
    /// Create new FFN backward MIL generator
    pub fn new() -> Self {
        FFNBackwardGen
    }

    /// Generate MIL code for FFN backward operation
    ///
    /// # MIL Structure
    /// ```text
    /// Inputs (from cached forward pass):
    ///   - d_out: Gradient w.r.t. output [batch_size, seq_len, hidden_dim]
    ///   - X: Input activations [batch_size, seq_len, hidden_dim]
    ///   - gate_proj: Cached gate projection [batch_size, seq_len, ffn_hidden_dim]
    ///   - up_proj: Cached up projection [batch_size, seq_len, ffn_hidden_dim]
    ///   - gate_out: Cached SiLU(gate_proj) [batch_size, seq_len, ffn_hidden_dim]
    ///   - W_gate, W_up, W_down: Projection weights
    ///
    /// Outputs:
    ///   - d_X: Gradient w.r.t. input [batch_size, seq_len, hidden_dim]
    ///   - d_W_gate, d_W_up, d_W_down: Gradients w.r.t. weights
    /// ```
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let _batch_size = 1; // Will be dynamic
        let _seq_len = config.seq_len;
        let hidden_dim = config.dim;
        let ffn_hidden_dim = config.hidden_dim;

        format!(r#"
#!irms6
schema ffn_backward_schema {{
    input d_out: tensor<batch_sizexseq_lenx{hidden_dim}xf32> = Input()
    input X: tensor<batch_sizexseq_lenx{hidden_dim}xf32> = Input()
    input gate_proj: tensor<batch_sizexseq_lenx{ffn_hidden_dim}xf32> = Input()
    input up_proj: tensor<batch_sizexseq_lenx{ffn_hidden_dim}xf32> = Input()
    input gate_out: tensor<batch_sizexseq_lenx{ffn_hidden_dim}xf32> = Input()
    input W_gate: tensor<{hidden_dim}x{ffn_hidden_dim}xf32> = Input()
    input W_up: tensor<{hidden_dim}x{ffn_hidden_dim}xf32> = Input()
    input W_down: tensor<{ffn_hidden_dim}x{hidden_dim}xf32> = Input()

    output d_X: tensor<batch_sizexseq_lenx{hidden_dim}xf32> = Output()
    output d_W_gate: tensor<{hidden_dim}x{ffn_hidden_dim}xf32> = Output()
    output d_W_up: tensor<{hidden_dim}x{ffn_hidden_dim}xf32> = Output()
    output d_W_down: tensor<{ffn_hidden_dim}x{hidden_dim}xf32> = Output()
}}

main ffn_backward(
    d_out: tensor<batch_sizexseq_lenx{hidden_dim}xf32>,
    X: tensor<batch_sizexseq_lenx{hidden_dim}xf32>,
    gate_proj: tensor<batch_sizexseq_lenx{ffn_hidden_dim}xf32>,
    up_proj: tensor<batch_sizexseq_lenx{ffn_hidden_dim}xf32>,
    gate_out: tensor<batch_sizexseq_lenx{ffn_hidden_dim}xf32>,
    W_gate: tensor<{hidden_dim}x{ffn_hidden_dim}xf32>,
    W_up: tensor<{hidden_dim}x{ffn_hidden_dim}xf32>,
    W_down: tensor<{ffn_hidden_dim}x{hidden_dim}xf32>
) -> (d_X: tensor<batch_sizexseq_lenx{hidden_dim}xf32>,
      d_W_gate: tensor<{hidden_dim}x{ffn_hidden_dim}xf32>,
      d_W_up: tensor<{hidden_dim}x{ffn_hidden_dim}xf32>,
      d_W_down: tensor<{ffn_hidden_dim}x{hidden_dim}xf32>) {{

    // ===== Down projection backward =====
    // d_out_ffn = d_out @ W_down^T
    // d_W_down = out^T @ d_out
    // where out = gate_out * up_proj

    let W_down_transposed = transpose(W_down, perm=[1, 0])
    let d_out_ffn = matmul(d_out, W_down_transposed)

    let ffn_out = gate_out * up_proj
    let ffn_out_transposed = transpose(ffn_out, perm=[0, 2, 1])
    let d_W_down_accum = matmul(ffn_out_transposed, d_out)

    // ===== Element-wise multiplication backward =====
    // d_gate = d_out_ffn * up_proj
    // d_up = d_out_ffn * gate_out

    let d_gate = d_out_ffn * up_proj
    let d_up = d_out_ffn * gate_out

    // ===== Up projection backward =====
    // d_W_up = X^T @ d_up
    // d_X_up = d_up @ W_up^T

    let X_transposed = transpose(X, perm=[0, 2, 1])
    let d_W_up_accum = matmul(X_transposed, d_up)

    let W_up_transposed = transpose(W_up, perm=[1, 0])
    let d_X_up = matmul(d_up, W_up_transposed)

    // ===== Gate projection backward (with SiLU derivative) =====
    // SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // d_gate_proj = d_gate * SiLU'(gate_proj)

    let sigmoid_gate_proj = sigmoid(gate_proj)
    let one_minus_sigmoid = 1.0 - sigmoid_gate_proj
    let gate_times_one_minus_sigmoid = gate_proj * one_minus_sigmoid
    let one_plus_term = 1.0 + gate_times_one_minus_sigmoid
    let silu_derivative = sigmoid_gate_proj * one_plus_term

    let d_gate_proj = d_gate * silu_derivative

    // d_W_gate = X^T @ d_gate_proj
    let d_W_gate_accum = matmul(X_transposed, d_gate_proj)

    // d_X_gate = d_gate_proj @ W_gate^T
    let W_gate_transposed = transpose(W_gate, perm=[1, 0])
    let d_X_gate = matmul(d_gate_proj, W_gate_transposed)

    // ===== Input gradient (accumulated) =====
    // d_X = d_X_up + d_X_gate

    let d_X = d_X_up + d_X_gate

    // Return all gradients
    return (d_X, d_W_gate_accum, d_W_up_accum, d_W_down_accum)
}}
"#)
    }
}

impl Default for FFNBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for FFNBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, _config: &TransformerConfig) -> Result<()> {
        // TODO: Implement validation in Phase 3b
        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "ffn_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_backward_gen_creation() {
        let gen = FFNBackwardGen::new();
        assert_eq!(gen.operation_name(), "ffn_backward");
    }

    #[test]
    fn test_ffn_backward_gen_default() {
        let gen = FFNBackwardGen::default();
        assert_eq!(gen.operation_name(), "ffn_backward");
    }

    #[test]
    fn test_ffn_backward_generate_mil() {
        let gen = FFNBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config);

        assert!(mil_code.is_ok());
        let mil = mil_code.unwrap();
        assert!(mil.contains("ffn_backward"));
        assert!(mil.contains("d_X"));
        assert!(mil.contains("d_W_gate"));
        assert!(mil.contains("d_W_up"));
        assert!(mil.contains("d_W_down"));
    }

    #[test]
    fn test_ffn_backward_mil_structure() {
        let gen = FFNBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        // Verify MIL contains required sections
        assert!(mil_code.contains("schema"));
        assert!(mil_code.contains("input"));
        assert!(mil_code.contains("output"));
        assert!(mil_code.contains("main"));
        assert!(mil_code.contains("return"));

        // Verify mathematical operations
        assert!(mil_code.contains("matmul"));
        assert!(mil_code.contains("transpose"));
        assert!(mil_code.contains("sigmoid"));

        // Verify SiLU derivative is present
        assert!(mil_code.contains("silu_derivative"));
    }
}
