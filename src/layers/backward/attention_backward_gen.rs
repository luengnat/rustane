//! Multi-head attention backward pass MIL generator
//!
//! Generates MIL code for scaled dot-product attention backward pass.
//!
//! # Forward Pass
//! ```text
//! Q = X @ W_Q
//! K = X @ W_K
//! V = X @ W_V
//! scores = Q @ K^T / sqrt(d_k)
//! attn = softmax(scores)
//! output = attn @ V
//! ```
//!
//! # Backward Pass
//! Computes gradients for all weights and inputs:
//! - `d_W_Q`, `d_W_K`, `d_W_V`: Gradients w.r.t. projection weights
//! - `d_W_O`: Gradient w.r.t. output projection weights
//! - `d_X`: Gradient w.r.t. input (accumulated from Q, K, V paths)
//!
//! # Mathematical Derivation
//!
//! Given output gradient `d_out`, we backpropagate:
//!
//! ```text
//! // Output projection backward
//! d_V = d_out @ W_O^T
//! d_W_O = attn^T @ d_out
//!
//! // Attention backward
//! d_attn = d_out @ V^T
//! d_scores = d_attn * attn - attn * sum(d_attn * attn, axis=-1)
//!
//! // Q, K backward
//! d_Q = d_scores @ K / sqrt(d_k)
//! d_K = d_scores^T @ Q / sqrt(d_k)
//! d_W_Q = X^T @ d_Q
//! d_W_K = X^T @ d_K
//! d_W_V = X^T @ d_V
//!
//! // Input gradient (accumulated)
//! d_X = d_Q @ W_Q^T + d_K @ W_K^T + d_V @ W_V^T
//! ```

use crate::training::TransformerConfig;
use crate::ane::Result;
use super::BackwardMILGenerator;

/// MIL generator for multi-head attention backward pass
pub struct AttentionBackwardGen;

impl AttentionBackwardGen {
    /// Create new attention backward MIL generator
    pub fn new() -> Self {
        AttentionBackwardGen
    }

    /// Generate MIL code for attention backward operation
    ///
    /// # MIL Structure
    /// ```text
    /// Inputs (from cached forward pass):
    ///   - d_out: Gradient w.r.t. output [batch_size, seq_len, hidden_dim]
    ///   - X: Input activations [batch_size, seq_len, hidden_dim]
    ///   - Q: Cached queries [batch_size, num_heads, seq_len, d_k]
    ///   - K: Cached keys [batch_size, num_heads, seq_len, d_k]
    ///   - V: Cached values [batch_size, num_heads, seq_len, d_k]
    ///   - attn: Cached attention weights [batch_size, num_heads, seq_len, seq_len]
    ///   - W_Q, W_K, W_V, W_O: Projection weights
    ///
    /// Outputs:
    ///   - d_X: Gradient w.r.t. input [batch_size, seq_len, hidden_dim]
    ///   - d_W_Q, d_W_K, d_W_V, d_W_O: Gradients w.r.t. weights
    /// ```
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let batch_size = 1; // Will be dynamic
        let seq_len = config.seq_len;
        let hidden_dim = config.dim;
        let num_heads = config.n_heads;
        let d_k = config.head_dim;
        let scale = (1.0 / (d_k as f32).sqrt());

        format!(r#"
#!irms6
schema attention_backward_schema {{
    input d_out: tensor<batch_sizexseq_lenx{hidden_dim}xf32> = Input()
    input X: tensor<batch_sizexseq_lenx{hidden_dim}xf32> = Input()
    input Q: tensor<batch_sizex{num_heads}xseq_lenx{d_k}xf32> = Input()
    input K: tensor<batch_sizex{num_heads}xseq_lenx{d_k}xf32> = Input()
    input V: tensor<batch_sizex{num_heads}xseq_lenx{d_k}xf32> = Input()
    input attn: tensor<batch_sizex{num_heads}xseq_lenxseq_lenxf32> = Input()
    input W_Q: tensor<{hidden_dim}x{hidden_dim}xf32> = Input()
    input W_K: tensor<{hidden_dim}x{hidden_dim}xf32> = Input()
    input W_V: tensor<{hidden_dim}x{hidden_dim}xf32> = Input()
    input W_O: tensor<{hidden_dim}x{hidden_dim}xf32> = Input()

    output d_X: tensor<batch_sizexseq_lenx{hidden_dim}xf32> = Output()
    output d_W_Q: tensor<{hidden_dim}x{hidden_dim}xf32> = Output()
    output d_W_K: tensor<{hidden_dim}x{hidden_dim}xf32> = Output()
    output d_W_V: tensor<{hidden_dim}x{hidden_dim}xf32> = Output()
    output d_W_O: tensor<{hidden_dim}x{hidden_dim}xf32> = Output()
}}

main attention_backward(
    d_out: tensor<batch_sizexseq_lenx{hidden_dim}xf32>,
    X: tensor<batch_sizexseq_lenx{hidden_dim}xf32>,
    Q: tensor<batch_sizex{num_heads}xseq_lenx{d_k}xf32>,
    K: tensor<batch_sizex{num_heads}xseq_lenx{d_k}xf32>,
    V: tensor<batch_sizex{num_heads}xseq_lenx{d_k}xf32>,
    attn: tensor<batch_sizex{num_heads}xseq_lenxseq_lenxf32>,
    W_Q: tensor<{hidden_dim}x{hidden_dim}xf32>,
    W_K: tensor<{hidden_dim}x{hidden_dim}xf32>,
    W_V: tensor<{hidden_dim}x{hidden_dim}xf32>,
    W_O: tensor<{hidden_dim}x{hidden_dim}xf32>
) -> (d_X: tensor<batch_sizexseq_lenx{hidden_dim}xf32>,
      d_W_Q: tensor<{hidden_dim}x{hidden_dim}xf32>,
      d_W_K: tensor<{hidden_dim}x{hidden_dim}xf32>,
      d_W_V: tensor<{hidden_dim}x{hidden_dim}xf32>,
      d_W_O: tensor<{hidden_dim}x{hidden_dim}xf32>) {{

    // ===== Output projection backward =====
    // d_V = d_out @ W_O^T
    // d_W_O = attn^T @ d_out

    // Reshape for multi-head processing
    let d_out_reshaped = reshape(d_out, shape=[batch_size, seq_len, {num_heads}, {d_k}])
    let d_out_transposed = transpose(d_out_reshaped, perm=[0, 2, 1, 3])

    // Gradient w.r.t. V
    let W_O_transposed = transpose(W_O, perm=[1, 0])
    let d_V_temp = matmul(d_out_transposed, W_O_transposed)

    // Gradient w.r.t. W_O
    let attn_transposed = transpose(attn, perm=[0, 1, 3, 2])
    let d_W_O_accum = matmul(attn_transposed, d_out_reshaped)
    let d_W_O_final = reshape(d_W_O_accum, shape=[{hidden_dim}, {hidden_dim}])

    // ===== Attention backward =====
    // d_attn = d_out @ V^T
    // d_scores = d_attn * attn - attn * sum(d_attn * attn, axis=-1)

    let V_transposed = transpose(V, perm=[0, 1, 3, 2])
    let d_attn = matmul(d_out_transposed, V_transposed)

    // Softmax backward: d_scores = d_attn * attn - attn * sum(d_attn * attn, axis=-1)
    let d_attn_times_attn = d_attn * attn
    let sum_d_attn_times_attn = reduce_sum(d_attn_times_attn, axes=[3], keep_dims=true)
    let d_scores = d_attn_times_attn - (attn * sum_d_attn_times_attn)

    // Scale gradients
    let d_scores_scaled = d_scores * {scale}.0

    // ===== Q, K backward =====
    // d_Q = d_scores @ K
    // d_K = d_scores^T @ Q

    let K_transposed = transpose(K, perm=[0, 1, 3, 2])
    let d_Q = matmul(d_scores_scaled, K_transposed)

    let d_scores_transposed = transpose(d_scores_scaled, perm=[0, 1, 3, 2])
    let d_K = matmul(d_scores_transposed, Q)

    // ===== Weight gradients =====
    // d_W_Q = X^T @ d_Q
    // d_W_K = X^T @ d_K
    // d_W_V = X^T @ d_V

    // Reshape X for projection
    let X_reshaped = reshape(X, shape=[batch_size, seq_len, {num_heads}, {d_k}])
    let X_transposed = transpose(X_reshaped, perm=[0, 2, 1, 3])

    // Project gradients back to hidden dimension
    let d_Q_merged = transpose(d_Q, perm=[0, 2, 1, 3])
    let d_Q_reshaped = reshape(d_Q_merged, shape=[batch_size * seq_len, {hidden_dim}])

    let d_K_merged = transpose(d_K, perm=[0, 2, 1, 3])
    let d_K_reshaped = reshape(d_K_merged, shape=[batch_size * seq_len, {hidden_dim}])

    let d_V_merged = transpose(d_V_temp, perm=[0, 2, 1, 3])
    let d_V_reshaped = reshape(d_V_merged, shape=[batch_size * seq_len, {hidden_dim}])

    let X_flat = reshape(X, shape=[batch_size * seq_len, {hidden_dim}])
    let X_flat_transposed = transpose(X_flat, perm=[1, 0])

    let d_W_Q_final = matmul(X_flat_transposed, d_Q_reshaped)
    let d_W_K_final = matmul(X_flat_transposed, d_K_reshaped)
    let d_W_V_final = matmul(X_flat_transposed, d_V_reshaped)

    // ===== Input gradient (accumulated) =====
    // d_X = d_Q @ W_Q^T + d_K @ W_K^T + d_V @ W_V^T

    let W_Q_transposed = transpose(W_Q, perm=[1, 0])
    let W_K_transposed = transpose(W_K, perm=[1, 0])
    let W_V_transposed = transpose(W_V, perm=[1, 0])

    let d_X_Q = matmul(d_Q_reshaped, W_Q_transposed)
    let d_X_K = matmul(d_K_reshaped, W_K_transposed)
    let d_X_V = matmul(d_V_reshaped, W_V_transposed)

    let d_X_flat = d_X_Q + d_X_K + d_X_V
    let d_X = reshape(d_X_flat, shape=[batch_size, seq_len, {hidden_dim}])

    // Return all gradients
    return (d_X, d_W_Q_final, d_W_K_final, d_W_V_final, d_W_O_final)
}}
"#)
    }
}

impl Default for AttentionBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for AttentionBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        // TODO: Implement validation in Phase 3b
        // For now, return Ok to allow compilation to proceed
        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "attention_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_backward_gen_creation() {
        let gen = AttentionBackwardGen::new();
        assert_eq!(gen.operation_name(), "attention_backward");
    }

    #[test]
    fn test_attention_backward_gen_default() {
        let gen = AttentionBackwardGen::default();
        assert_eq!(gen.operation_name(), "attention_backward");
    }

    #[test]
    fn test_attention_backward_generate_mil() {
        let gen = AttentionBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config);

        assert!(mil_code.is_ok());
        let mil = mil_code.unwrap();
        assert!(mil.contains("attention_backward"));
        assert!(mil.contains("d_X"));
        assert!(mil.contains("d_W_Q"));
        assert!(mil.contains("d_W_K"));
        assert!(mil.contains("d_W_V"));
        assert!(mil.contains("d_W_O"));
    }

    #[test]
    fn test_attention_backward_mil_structure() {
        let gen = AttentionBackwardGen::new();
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
        assert!(mil_code.contains("reshape"));
        assert!(mil_code.contains("reduce_sum"));
    }
}
