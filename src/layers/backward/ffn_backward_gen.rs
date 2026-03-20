//! Feed-forward network backward pass MIL generator
//!
//! Generates MIL code for FFN (SwiGLU) backward pass.
//!
//! # Forward Pass
//! ```text
//! gate = SiLU(W1 @ x)
//! hidden = W3 @ x
//! output = (gate * hidden) @ W2
//! ```
//!
//! # Backward Pass
//! Computes gradients for all three weight matrices.

use super::{validate_mil_structure, BackwardMILGenerator};
use crate::ane::{ANECompileRequest, ANEError, Result};
use crate::training::TransformerConfig;

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
    /// Implements backward pass for SwiGLU FFN:
    /// Forward: output = SiLU(W1 @ x) * (W3 @ x) @ W2
    /// Backward: Computes gradients for W1, W2, W3, and x
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim * 4; // Standard FFN expansion

        format!(
            r#"
#!irms6
schema ffn_backward_schema {{
    input d_out: tensor<seq_lenxdimxf32> = Input()
    input x: tensor<seq_lenxdimxf32> = Input()
    input w1_out: tensor<seq_lenx{hidden_dim}xf32> = Input()
    input w3_out: tensor<seq_lenx{hidden_dim}xf32> = Input()
    output d_x: tensor<seq_lenxdimxf32> = Output()
    output d_w1: tensor<dimx{hidden_dim}xf32> = Output()
    output d_w3: tensor<dimx{hidden_dim}xf32> = Output()
    output d_w2: tensor<{hidden_dim}xdimxf32> = Output()
}}

main ffn_backward(d_out: tensor<seq_lenxdimxf32>,
                  x: tensor<seq_lenxdimxf32>,
                  w1_out: tensor<seq_lenx{hidden_dim}xf32>,
                  w3_out: tensor<seq_lenx{hidden_dim}xf32>) ->
                  (d_x: tensor<seq_lenxdimxf32>,
                   d_w1: tensor<dimx{hidden_dim}xf32>,
                   d_w3: tensor<dimx{hidden_dim}xf32>,
                   d_w2: tensor<{hidden_dim}xdimxf32>) {{
    // Reshape for processing
    let d_out_reshaped = reshape(d_out, shape=[seq_len, dim])
    let x_reshaped = reshape(x, shape=[seq_len, dim])
    let w1_out_reshaped = reshape(w1_out, shape=[seq_len, {hidden_dim}])
    let w3_out_reshaped = reshape(w3_out, shape=[seq_len, {hidden_dim}])

    // SiLU activation: gate = w1_out * sigmoid(w1_out)
    let gate = w1_out_reshaped * sigmoid(w1_out_reshaped)

    // Intermediate activation: gate * w3_out
    let intermediate = gate * w3_out_reshaped

    // d_W2 = intermediate^T @ d_out
    let d_w2 = transpose(intermediate) @ d_out_reshaped

    // d_intermediate = d_out @ W2^T
    let d_intermediate = d_out_reshaped @ transpose(d_w2)

    // d_gate = d_intermediate * w3_out
    let d_gate = d_intermediate * w3_out_reshaped

    // d_w3_out = d_intermediate * gate
    let d_w3_out = d_intermediate * gate

    // d_w1_out = d_gate * sigmoid_derivative(w1_out)
    // SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    let sigmoid_w1 = sigmoid(w1_out_reshaped)
    let silu_derivative = sigmoid_w1 * (1.0 + w1_out_reshaped * (1.0 - sigmoid_w1))
    let d_w1_out = d_gate * silu_derivative

    // Gradient accumulation
    let mut d_x_accum = const_zero(shape=[seq_len, dim], dtype=float32)
    let mut d_w1_accum = const_zero(shape=[dim, {hidden_dim}], dtype=float32)
    let mut d_w3_accum = const_zero(shape=[dim, {hidden_dim}], dtype=float32)

    // Accumulate gradients across sequence
    for i in 0..seq_len {{
        let d_x_i = d_w1_out[i] @ transpose(d_w1) + d_w3_out[i] @ transpose(d_w3)
        d_x_accum[i] = d_x_i
    }}

    // Accumulate weight gradients
    let d_w1_final = transpose(x_reshaped) @ d_w1_out
    let d_w3_final = transpose(x_reshaped) @ d_w3_out

    // Reshape outputs
    let d_x_final = reshape(d_x_accum, shape=[seq_len * dim])

    return (d_x_final, d_w1_final, d_w3_final, d_w2)
}}
"#
        )
    }

    /// Helper function to calculate input size in bytes
    pub fn input_bytes(&self, config: &TransformerConfig) -> usize {
        let seq_len = config.seq_len;
        let dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim * 4;

        // d_out + x + w1_out + w3_out
        (seq_len * dim + seq_len * dim + 2 * seq_len * hidden_dim) * 4
    }

    /// Helper function to calculate output sizes in bytes
    pub fn output_sizes(&self, config: &TransformerConfig) -> Vec<usize> {
        let seq_len = config.seq_len;
        let dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim * 4;

        vec![
            seq_len * dim * 4,           // d_x
            dim * hidden_dim * 4,        // d_w1
            dim * hidden_dim * 4,        // d_w3
            hidden_dim * dim * 4,        // d_w2
        ]
    }

    /// Helper function to run FFN backward on ANE
    pub fn run_on_ane(
        &self,
        config: &TransformerConfig,
        d_out: &[f32],
        x: &[f32],
        w1_out: &[f32],
        w3_out: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mil_code = self.generate_mil_code(config);

        let input_bytes = self.input_bytes(config);
        let output_sizes = self.output_sizes(config);

        // Pack inputs as bytes
        let mut packed_input = Vec::with_capacity(input_bytes);
        for &val in d_out.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }
        for &val in x.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }
        for &val in w1_out.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }
        for &val in w3_out.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }

        let mut request =
            ANECompileRequest::new(&mil_code, vec![input_bytes], output_sizes.clone());
        let mut executor = request
            .compile()
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;

        executor
            .write_input(0, &packed_input)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor.eval().map_err(|e| ANEError::EvalFailed(e.to_string()))?;

        let d_x_len = config.seq_len * config.hidden_dim;
        let d_w1_len = config.hidden_dim * config.hidden_dim * 4;
        let d_w3_len = config.hidden_dim * config.hidden_dim * 4;
        let d_w2_len = config.hidden_dim * 4 * config.hidden_dim;

        let mut d_x_bytes = vec![0u8; d_x_len * 4];
        let mut d_w1_bytes = vec![0u8; d_w1_len * 4];
        let mut d_w3_bytes = vec![0u8; d_w3_len * 4];
        let mut d_w2_bytes = vec![0u8; d_w2_len * 4];

        executor
            .read_output(0, &mut d_x_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(1, &mut d_w1_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(2, &mut d_w3_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(3, &mut d_w2_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;

        let d_x: Vec<f32> = d_x_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_w1: Vec<f32> = d_w1_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_w3: Vec<f32> = d_w3_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_w2: Vec<f32> = d_w2_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok((d_x, d_w1, d_w3, d_w2))
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

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let mil_code = self.generate(config)?;
        validate_mil_structure(
            &mil_code,
            "ffn_backward",
            &["d_out", "x", "w1_out", "w3_out"],
            &["d_x", "d_w1", "d_w3", "d_w2"],
        )?;

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
    fn test_ffn_backward_generate_mil() {
        let gen = FFNBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        assert!(mil_code.contains("ffn_backward"));
    }
}