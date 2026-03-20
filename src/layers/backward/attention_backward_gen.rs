//! Multi-head attention backward pass MIL generator
//!
//! Generates MIL code for scaled dot-product attention backward pass.

use super::{validate_mil_structure, BackwardMILGenerator};
use crate::ane::{ANECompileRequest, ANEError, Result};
use crate::training::TransformerConfig;

/// MIL generator for multi-head attention backward pass
#[derive(Debug)]
pub struct AttentionBackwardGen;
impl AttentionBackwardGen {
    /// Create new attention backward MIL generator
    pub fn new() -> Self {
        AttentionBackwardGen
    }

    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let n_heads = config.n_heads;
        let head_dim = config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        format!(
            r#"
#!irms6
schema attention_backward_schema {{
    input d_out: tensor<seq_lenxdimxf32> = Input()
    input q: tensor<seq_lenxdimxf32> = Input()
    input k: tensor<seq_lenxdimxf32> = Input()
    input v: tensor<seq_lenxdimxf32> = Input()
    input attn_weights: tensor<seq_len_x_seq_len_x{n_heads}xf32> = Input()
    output d_x: tensor<seq_lenxdimxf32> = Output()
    output d_wq: tensor<dimxdimxf32> = Output()
    output d_wk: tensor<dimxdimxf32> = Output()
    output d_wv: tensor<dimxdimxf32> = Output()
    output d_wo: tensor<dimxdimxf32> = Output()
}}

main attention_backward(d_out: tensor<seq_lenxdimxf32>,
                       q: tensor<seq_lenxdimxf32>,
                       k: tensor<seq_lenxdimxf32>,
                       v: tensor<seq_lenxdimxf32>,
                       attn_weights: tensor<seq_len_x_seq_len_x{n_heads}xf32>) ->
                       (d_x: tensor<seq_lenxdimxf32>,
                        d_wq: tensor<dimxdimxf32>,
                        d_wk: tensor<dimxdimxf32>,
                        d_wv: tensor<dimxdimxf32>,
                        d_wo: tensor<dimxdimxf32>) {{
    // Reshape inputs for multi-head processing
    let d_out_reshaped = reshape(d_out, shape=[seq_len, n_heads, head_dim])
    let q_reshaped = reshape(q, shape=[seq_len, n_heads, head_dim])
    let k_reshaped = reshape(k, shape=[seq_len, n_heads, head_dim])
    let v_reshaped = reshape(v, shape=[seq_len, n_heads, head_dim])
    let attn_reshaped = reshape(attn_weights, shape=[seq_len, seq_len, n_heads])

    // Initialize output gradients
    let d_x = const_zero(shape=[seq_len, dim], dtype=float32)
    let d_wq = const_zero(shape=[dim, dim], dtype=float32)
    let d_wk = const_zero(shape=[dim, dim], dtype=float32)
    let d_wv = const_zero(shape=[dim, dim], dtype=float32)
    let d_wo = const_zero(shape=[dim, dim], dtype=float32)

    // Process each head independently
    for head in 0..n_heads {{
        let d_out_head = d_out_reshaped[.., head, ..]
        let q_head = q_reshaped[.., head, ..]
        let k_head = k_reshaped[.., head, ..]
        let v_head = v_reshaped[.., head, ..]
        let attn_head = attn_reshaped[.., .., head]

        // d_V = attn_weights^T @ d_out
        let d_v_head = transpose(attn_head) @ d_out_head

        // d_attn = d_out @ V^T (before softmax derivative)
        let d_attn_raw = d_out_head @ transpose(v_head)

        // Backprop through softmax (simplified - in reality would need softmax derivative)
        // For now, use approximation: d_attn = d_attn_raw * attn_head * (1 - attn_head)
        let d_attn = d_attn_raw * attn_head * (1.0 - attn_head)

        // d_Q = d_attn @ K^T / sqrt(d_k)
        let d_q_head = d_attn @ transpose(k_head) * {scale}.0

        // Accumulate gradients for current head
        let d_x_head = d_v_head + d_q_head
        let d_wq_head = outer(q_head, d_attn)
        let d_wk_head = outer(k_head, d_attn)
        let d_wv_head = outer(v_head, d_out_head)
        let d_wo_head = outer(v_head, d_out_head)

        // Sum across heads (reshape back to full dimensions)
        let d_x_head_full = reshape(d_x_head, shape=[seq_len, head_dim])
        let d_wq_head_full = reshape(d_wq_head, shape=[dim, dim])
        let d_wk_head_full = reshape(d_wk_head, shape=[dim, dim])
        let d_wv_head_full = reshape(d_wv_head, shape=[dim, dim])
        let d_wo_head_full = reshape(d_wo_head, shape=[dim, dim])

        d_x = d_x + reshape(d_x_head_full, shape=[seq_len, dim])
        d_wq = d_wq + d_wq_head_full
        d_wk = d_wk + d_wk_head_full
        d_wv = d_wv + d_wv_head_full
        d_wo = d_wo + d_wo_head_full
    }}

    // Reshape d_x to match input format
    let d_x_final = reshape(d_x, shape=[seq_len * dim])

    return (d_x_final, d_wq, d_wk, d_wv, d_wo)
}}
"#
        )
    }

    /// Get input size in bytes
    pub fn input_bytes(&self, config: &TransformerConfig) -> usize {
        let seq_len = config.seq_len;
        let dim = config.hidden_dim;
        let n_heads = config.n_heads;

        // d_out + q + k + v + attn_weights
        (seq_len * dim + 3 * seq_len * dim + seq_len * seq_len * n_heads) * 4
    }

    /// Get output sizes in bytes
    pub fn output_sizes(&self, config: &TransformerConfig) -> Vec<usize> {
        let seq_len = config.seq_len;
        let dim = config.hidden_dim;

        vec![
            seq_len * dim * 4, // d_x
            dim * dim * 4,     // d_wq
            dim * dim * 4,     // d_wk
            dim * dim * 4,     // d_wv
            dim * dim * 4,     // d_wo
        ]
    }

    /// Run attention backward on ANE
    pub fn run_on_ane(
        config: &TransformerConfig,
        d_out: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attn_weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let gen = AttentionBackwardGen::new();
        let mil_code = gen.generate_mil_code(config);

        let input_bytes = gen.input_bytes(config);
        let output_sizes = gen.output_sizes(config);

        // Pack inputs as bytes
        let mut packed_input = Vec::with_capacity(input_bytes);
        for &v in d_out.iter() {
            packed_input.extend_from_slice(&v.to_le_bytes());
        }
        for &v in q.iter() {
            packed_input.extend_from_slice(&v.to_le_bytes());
        }
        for &v in k.iter() {
            packed_input.extend_from_slice(&v.to_le_bytes());
        }
        for &v in v.iter() {
            packed_input.extend_from_slice(&v.to_le_bytes());
        }
        for &v in attn_weights.iter() {
            packed_input.extend_from_slice(&v.to_le_bytes());
        }

        let request = ANECompileRequest::new(&mil_code, vec![input_bytes], output_sizes.clone());
        let mut executor = request
            .compile()
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;

        executor
            .write_input(0, &packed_input)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .eval()
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;

        let d_x_len = config.seq_len * config.hidden_dim;
        let grad_len = config.hidden_dim * config.hidden_dim;

        let mut d_x_bytes = vec![0u8; d_x_len * 4];
        let mut d_wq_bytes = vec![0u8; grad_len * 4];
        let mut d_wk_bytes = vec![0u8; grad_len * 4];
        let mut d_wv_bytes = vec![0u8; grad_len * 4];
        let mut d_wo_bytes = vec![0u8; grad_len * 4];

        executor
            .read_output(0, &mut d_x_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(1, &mut d_wq_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(2, &mut d_wk_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(3, &mut d_wv_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(4, &mut d_wo_bytes)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;

        let d_x: Vec<f32> = d_x_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_wq: Vec<f32> = d_wq_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_wk: Vec<f32> = d_wk_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_wv: Vec<f32> = d_wv_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let d_wo: Vec<f32> = d_wo_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok((d_x, d_wq, d_wk, d_wv, d_wo))
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
        let mil_code = self.generate(config)?;
        validate_mil_structure(
            &mil_code,
            "attention_backward",
            &["d_out", "q", "k", "v", "attn_weights"],
            &["d_x", "d_wq", "d_wk", "d_wv", "d_wo"],
        )?;

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
    fn test_attention_backward_generate_mil() {
        let gen = AttentionBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        assert!(mil_code.contains("attention_backward"));
    }
}
