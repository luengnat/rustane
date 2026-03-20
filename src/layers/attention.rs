//! Attention mechanism implementations for ANE
//!
//! This module provides multi-head attention and self-attention layers
//! that leverage the Apple Neural Engine (ANE) for matrix multiplication
//! while using CPU fallback for softmax with causal masking.
//!
//! # ANE Limitations
//!
//! - ANE ignores attention masks, so causal masking is done on CPU
//! - ANE doesn't support softmax natively
//! - ~119 compilation limit per process
//!
//! # Architecture
//!
//! - QK^T (query-key transpose): ANE matrix multiplication
//! - Softmax with causal mask: CPU fallback
//! - SV (scaled dot-product result): ANE matrix multiplication

use crate::layers::traits::{Layer, Shape};
use crate::layers::Linear;
use crate::wrapper::ANEExecutor;
use crate::{Error, Result};

/// CPU-based softmax with causal mask
///
/// Computes softmax along the last dimension with causal masking
/// for autoregressive (decoder-side) attention.
pub struct SoftmaxWithCausalMask {
    seq_len: usize,
}

impl SoftmaxWithCausalMask {
    /// Create a new softmax with causal mask
    pub fn new(seq_len: usize) -> Self {
        Self { seq_len }
    }

    /// Apply causal mask to attention scores
    ///
    /// Sets upper triangle (excluding diagonal) to negative infinity
    /// to prevent attending to future tokens.
    fn apply_causal_mask(&self, scores: &mut [f32]) {
        for i in 0..self.seq_len {
            for j in (i + 1)..self.seq_len {
                scores[i * self.seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    /// Numerically stable softmax along the last dimension
    ///
    /// Computes softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    fn softmax(&self, scores: &mut [f32]) {
        let seq_len = self.seq_len;

        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let row = &mut scores[row_start..row_end];

            // Find max for numerical stability
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Subtract max and compute exp
            let mut sum = 0.0;
            for val in row.iter_mut() {
                if val.is_finite() {
                    *val = (*val - max_val).exp();
                    sum += *val;
                } else {
                    *val = 0.0;
                }
            }

            // Normalize
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }
    }

    /// Compute softmax with causal mask
    ///
    /// # Arguments
    ///
    /// * `scores` - Attention scores [seq_len, seq_len]
    ///
    /// # Returns
    ///
    /// Attention weights after softmax and masking
    pub fn compute(&self, mut scores: Vec<f32>) -> Result<Vec<f32>> {
        if scores.len() != self.seq_len * self.seq_len {
            return Err(Error::InvalidParameter(format!(
                "Expected scores length {}, got {}",
                self.seq_len * self.seq_len,
                scores.len()
            )));
        }

        self.apply_causal_mask(&mut scores);
        self.softmax(&mut scores);

        Ok(scores)
    }
}

/// Multi-head attention layer
///
/// Implements scaled dot-product attention with multiple heads.
/// Each head learns different attention patterns.
///
/// # Architecture
///
/// 1. Q, K, V projections (via Linear layers)
/// 2. Reshape for multi-head: [batch, seq_len, num_heads, head_dim]
/// 3. Scaled dot-product attention (native ANE operation)
/// 4. Output projection
///
/// # Current Status
///
/// The layer structure is complete with builder pattern and parameter counting.
/// The forward pass requires custom MIL compilation using the proven
/// `scaled_dot_product_attention` operation (see examples/causal_attention.rs).
pub struct MultiHeadAttention {
    name: String,
    num_heads: usize,
    #[allow(dead_code)]
    embed_dim: usize,
    head_dim: usize,
    #[allow(dead_code)]
    dropout: f32,
    #[allow(dead_code)]
    causal: bool,

    // Q, K, V projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,

    // Output projection
    out_proj: Linear,

    input_shape: Shape,
    output_shape: Shape,
}

impl Layer for MultiHeadAttention {
    fn forward(
        &self,
        _executor: &mut ANEExecutor,
        _input_idx: usize,
        _output_idx: usize,
    ) -> Result<()> {
        // Note: This is a simplified forward pass that demonstrates the API.
        // A complete implementation would require:
        // 1. Compiling Q, K, V projection layers
        // 2. Reshaping tensors for multi-head layout
        // 3. Calling scaled_dot_product_attention
        // 4. Output projection
        //
        // For now, this returns an error to indicate the forward pass
        // needs to be implemented with proper MIL program compilation.
        Err(Error::NotImplemented(
            "MultiHeadAttention forward pass requires custom MIL compilation. \
             Use the causal_attention example for a working SDPA proof-of-life."
                .to_string(),
        ))
    }

    fn input_shape(&self) -> &Shape {
        &self.input_shape
    }

    fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_parameters(&self) -> usize {
        self.q_proj.num_parameters()
            + self.k_proj.num_parameters()
            + self.v_proj.num_parameters()
            + self.out_proj.num_parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MultiHeadAttention {
    /// Build MIL program for scaled dot-product attention
    ///
    /// This generates a MIL program using the native `scaled_dot_product_attention`
    /// operation, which has been verified to work on ANE.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Batch size (typically 1 for inference)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    ///
    /// MIL program string
    pub fn build_sdpa_mil_program(&self, batch_size: usize, seq_len: usize) -> String {
        format!(
            "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremlc-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}})]\n{{\n    func main<ios18>(tensor<fp16, [{b}, {h}, {s}, {d}]> q, tensor<fp16, [{b}, {h}, {s}, {d}]> k, tensor<fp16, [{b}, {h}, {s}, {d}]> v) {{\n        tensor<fp16, [{b}, {h}, {s}, {d}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n    }} -> (att);\n}}\n",
            b = batch_size,
            h = self.num_heads,
            s = seq_len,
            d = self.head_dim
        )
    }

    /// Get the sequence length from input shape
    ///
    /// # Arguments
    ///
    /// * `input_shape` - Input tensor shape [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    ///
    /// # Returns
    ///
    /// Sequence length
    pub fn extract_seq_len(&self, input_shape: &Shape) -> Result<usize> {
        match input_shape.len() {
            2 => Ok(input_shape[0]), // [seq_len, embed_dim]
            3 => Ok(input_shape[1]), // [batch, seq_len, embed_dim]
            _ => Err(Error::InvalidTensorShape(format!(
                "Expected 2D or 3D input, got {}D",
                input_shape.len()
            ))),
        }
    }
}

/// Builder for MultiHeadAttention
pub struct MultiHeadAttentionBuilder {
    name: String,
    embed_dim: usize,
    num_heads: usize,
    head_dim: Option<usize>,
    dropout: f32,
    bias: bool,
    causal: bool,
}

impl MultiHeadAttentionBuilder {
    /// Create a new MultiHeadAttention builder
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of attention heads
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            name: format!("mha_{}_{}", embed_dim, num_heads),
            embed_dim,
            num_heads,
            head_dim: None,
            dropout: 0.0,
            bias: true,
            causal: false,
        }
    }

    /// Set the layer name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the head dimension explicitly
    ///
    /// If not set, defaults to embed_dim / num_heads
    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = Some(head_dim);
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable or disable bias in projection layers
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Enable or disable causal masking
    ///
    /// When true, prevents attending to future tokens
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Build the MultiHeadAttention layer
    pub fn build(self) -> Result<MultiHeadAttention> {
        // Validate: embed_dim must be divisible by num_heads
        if self.embed_dim % self.num_heads != 0 {
            return Err(Error::InvalidParameter(format!(
                "embed_dim {} must be divisible by num_heads {}",
                self.embed_dim, self.num_heads
            )));
        }

        let head_dim = self.head_dim.unwrap_or(self.embed_dim / self.num_heads);

        // Create Q, K, V projections
        let q_proj = Linear::new(self.embed_dim, head_dim * self.num_heads)
            .with_bias(self.bias)
            .with_name(format!("{}_q_proj", self.name))
            .build()?;

        let k_proj = Linear::new(self.embed_dim, head_dim * self.num_heads)
            .with_bias(self.bias)
            .with_name(format!("{}_k_proj", self.name))
            .build()?;

        let v_proj = Linear::new(self.embed_dim, head_dim * self.num_heads)
            .with_bias(self.bias)
            .with_name(format!("{}_v_proj", self.name))
            .build()?;

        // Output projection
        let out_proj = Linear::new(head_dim * self.num_heads, self.embed_dim)
            .with_bias(self.bias)
            .with_name(format!("{}_out_proj", self.name))
            .build()?;

        Ok(MultiHeadAttention {
            name: self.name,
            num_heads: self.num_heads,
            embed_dim: self.embed_dim,
            head_dim,
            dropout: self.dropout,
            causal: self.causal,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            input_shape: vec![1, self.embed_dim], // Will be updated with seq_len
            output_shape: vec![1, self.embed_dim],
        })
    }
}

/// Self-attention convenience layer
///
/// Wrapper around MultiHeadAttention with causal masking enabled by default.
/// Useful for decoder-side attention in transformers.
pub struct SelfAttention {
    inner: MultiHeadAttention,
}

impl Layer for SelfAttention {
    fn forward(
        &self,
        executor: &mut ANEExecutor,
        input_idx: usize,
        output_idx: usize,
    ) -> Result<()> {
        self.inner.forward(executor, input_idx, output_idx)
    }

    fn input_shape(&self) -> &Shape {
        self.inner.input_shape()
    }

    fn output_shape(&self) -> &Shape {
        self.inner.output_shape()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for SelfAttention
pub struct SelfAttentionBuilder {
    embed_dim: usize,
    num_heads: usize,
    head_dim: Option<usize>,
    dropout: f32,
    bias: bool,
}

impl Default for SelfAttentionBuilder {
    fn default() -> Self {
        Self {
            embed_dim: 512,
            num_heads: 8,
            head_dim: None,
            dropout: 0.0,
            bias: true,
        }
    }
}

impl SelfAttentionBuilder {
    /// Create a new SelfAttention builder
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of attention heads
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embed_dim,
            num_heads,
            head_dim: None,
            dropout: 0.0,
            bias: true,
        }
    }

    /// Set the head dimension explicitly
    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = Some(head_dim);
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable or disable bias
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Build the SelfAttention layer
    ///
    /// Always creates a causal attention layer
    pub fn build(self) -> Result<SelfAttention> {
        let inner = MultiHeadAttentionBuilder::new(self.embed_dim, self.num_heads)
            .with_head_dim(self.head_dim.unwrap_or(self.embed_dim / self.num_heads))
            .with_dropout(self.dropout)
            .with_bias(self.bias)
            .with_causal(true) // Self-attention is always causal
            .with_name(format!(
                "self_attention_{}_{}",
                self.embed_dim, self.num_heads
            ))
            .build()?;

        Ok(SelfAttention { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_causal_mask() {
        let softmax = SoftmaxWithCausalMask::new(4);
        let mut scores = vec![1.0f32; 16]; // 4x4

        softmax.apply_causal_mask(&mut scores);

        // Upper triangle should be -inf
        assert_eq!(scores[0 * 4 + 1], f32::NEG_INFINITY);
        assert_eq!(scores[0 * 4 + 2], f32::NEG_INFINITY);
        assert_eq!(scores[0 * 4 + 3], f32::NEG_INFINITY);
        assert_eq!(scores[1 * 4 + 2], f32::NEG_INFINITY);
        assert_eq!(scores[1 * 4 + 3], f32::NEG_INFINITY);
        assert_eq!(scores[2 * 4 + 3], f32::NEG_INFINITY);

        // Diagonal should remain unchanged
        assert_eq!(scores[0 * 4 + 0], 1.0);
        assert_eq!(scores[1 * 4 + 1], 1.0);
        assert_eq!(scores[2 * 4 + 2], 1.0);
        assert_eq!(scores[3 * 4 + 3], 1.0);

        // Lower triangle should remain unchanged
        assert_eq!(scores[1 * 4 + 0], 1.0);
        assert_eq!(scores[2 * 4 + 0], 1.0);
        assert_eq!(scores[2 * 4 + 1], 1.0);
        assert_eq!(scores[3 * 4 + 0], 1.0);
        assert_eq!(scores[3 * 4 + 1], 1.0);
        assert_eq!(scores[3 * 4 + 2], 1.0);
    }

    #[test]
    fn test_softmax_stability() {
        let softmax = SoftmaxWithCausalMask::new(4);
        let mut scores = vec![1000.0f32; 16];

        softmax.softmax(&mut scores);

        // Should not overflow
        assert!(scores.iter().all(|x| x.is_finite()));

        // Should normalize
        for i in 0..4 {
            let row_start = i * 4;
            let row_end = row_start + 4;
            let row_sum: f32 = scores[row_start..row_end].iter().sum();
            assert!((row_sum - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_softmax_compute() {
        let softmax = SoftmaxWithCausalMask::new(3);
        let scores = vec![1.0f32; 9]; // 3x3

        let result = softmax.compute(scores).unwrap();

        // Check dimensions
        assert_eq!(result.len(), 9);

        // Check upper triangle is zero (after softmax of -inf)
        assert_eq!(result[0 * 3 + 1], 0.0);
        assert_eq!(result[0 * 3 + 2], 0.0);
        assert_eq!(result[1 * 3 + 2], 0.0);

        // Check diagonal sums to 1.0 (only one element per row after masking)
        assert_eq!(result[0 * 3 + 0], 1.0);
        assert!((result[1 * 3 + 0] + result[1 * 3 + 1] - 1.0).abs() < 0.001);
        assert!((result[2 * 3 + 0] + result[2 * 3 + 1] + result[2 * 3 + 2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mha_builder_valid() {
        let mha = MultiHeadAttentionBuilder::new(64, 4)
            .with_causal(true)
            .build()
            .unwrap();

        assert_eq!(mha.num_heads, 4);
        assert_eq!(mha.embed_dim, 64);
        assert_eq!(mha.head_dim, 16);
        assert!(mha.causal);
        assert!(mha.name.contains("mha"));
    }

    #[test]
    fn test_mha_builder_invalid_dimensions() {
        let result = MultiHeadAttentionBuilder::new(65, 4).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_mha_builder_custom_head_dim() {
        let mha = MultiHeadAttentionBuilder::new(64, 4)
            .with_head_dim(32)
            .build()
            .unwrap();

        assert_eq!(mha.head_dim, 32);
    }

    #[test]
    fn test_mha_parameters() {
        let mha = MultiHeadAttentionBuilder::new(64, 4)
            .with_bias(false)
            .build()
            .unwrap();

        // Q, K, V projections: 64 * 64 = 4096 each (no bias)
        // Output projection: 64 * 64 = 4096
        // Total: 4096 * 4 = 16384
        assert_eq!(mha.num_parameters(), 16384);
    }

    #[test]
    fn test_self_attention_convenience() {
        let sa = SelfAttentionBuilder::new(64, 4).build().unwrap();

        assert!(sa.inner.causal);
        assert_eq!(sa.num_parameters(), sa.inner.num_parameters());
        assert!(sa.inner.name.contains("self_attention"));
    }

    #[test]
    fn test_self_attention_defaults() {
        let sa = SelfAttentionBuilder::new(128, 8).build().unwrap();

        assert_eq!(sa.inner.embed_dim, 128);
        assert_eq!(sa.inner.num_heads, 8);
        assert_eq!(sa.inner.head_dim, 16);
        assert!(sa.inner.causal);
    }

    #[test]
    fn test_softmax_invalid_length() {
        let softmax = SoftmaxWithCausalMask::new(4);
        let scores = vec![1.0f32; 9]; // Wrong length

        let result = softmax.compute(scores);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }
}
