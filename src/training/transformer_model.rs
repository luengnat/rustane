//! Transformer model with ANE forward pass and CPU backward pass
//!
//! Implements the Model trait for training a transformer using Apple Neural Engine.

use crate::data::Batch;
use crate::error::Result;
use crate::training::{Model, TransformerConfig};
use crate::wrapper::ANETensor;

/// Cached activations from forward pass, used by backward
///
/// Stores intermediate tensors during forward pass for use in backward propagation.
/// This allows the backward pass to recompute gradients using the same intermediate
/// values without re-running forward pass.
#[derive(Clone, Debug)]
pub struct CachedActivations {
    /// Layer inputs (pre-attention norm) - needed for backward
    pub x_pre_attn_norm: Vec<Vec<f32>>,
    /// Layer inputs (pre-FFN norm) - needed for backward
    pub x_pre_ffn_norm: Vec<Vec<f32>>,

    /// Normalized activations after attention RMSNorm
    pub x_attn_norm: Vec<Vec<f32>>,
    /// Normalized activations after FFN RMSNorm
    pub x_ffn_norm: Vec<Vec<f32>>,

    /// Query projections from attention
    pub q: Vec<Vec<f32>>,
    /// Key projections from attention
    pub k: Vec<Vec<f32>>,
    /// Value projections from attention
    pub v: Vec<Vec<f32>>,
    /// Attention weight matrices
    pub attn_weights: Vec<Vec<f32>>,

    /// FFN w1 output activations
    pub w1_out: Vec<Vec<f32>>,
    /// FFN w1 gated activations (after SwiGLU)
    pub w1_gated: Vec<Vec<f32>>,

    /// Final layer normalization output
    pub x_final_norm: Vec<f32>,
}

impl CachedActivations {
    /// Create new empty cached activations
    fn new() -> Self {
        CachedActivations {
            x_pre_attn_norm: vec![],
            x_pre_ffn_norm: vec![],
            x_attn_norm: vec![],
            x_ffn_norm: vec![],
            q: vec![],
            k: vec![],
            v: vec![],
            attn_weights: vec![],
            w1_out: vec![],
            w1_gated: vec![],
            x_final_norm: vec![],
        }
    }

    /// Clear all cached activations
    fn clear(&mut self) {
        self.x_pre_attn_norm.clear();
        self.x_pre_ffn_norm.clear();
        self.x_attn_norm.clear();
        self.x_ffn_norm.clear();
        self.q.clear();
        self.k.clear();
        self.v.clear();
        self.attn_weights.clear();
        self.w1_out.clear();
        self.w1_gated.clear();
        self.x_final_norm.clear();
    }
}

/// Transformer model with ANE forward pass and CPU backward pass
///
/// Implements the Model trait for training transformers on Apple Neural Engine.
/// The forward pass uses ANE kernels for efficient computation.
/// The backward pass computes gradients on CPU using cached activations.
pub struct TransformerANE {
    config: TransformerConfig,

    // Trainable weights (host memory, CPU-accessible for optimizer updates)
    trainable_params: Vec<f32>,

    // Cached weight views used by the simplified forward/backward implementation
    embedding: Vec<f32>,
    classifier: Vec<f32>,
    #[allow(dead_code)]
    layer_norms: Vec<Vec<f32>>,
    #[allow(dead_code)]
    attention_weights: Vec<Vec<f32>>,
    #[allow(dead_code)]
    ffn_weights: Vec<Vec<f32>>,

    // Cached activations for backward
    cached: CachedActivations,
    last_input_tokens: Vec<u32>,
    last_input_activations: Vec<f32>,
    last_logits: Vec<f32>,
    last_batch_size: usize,
    last_seq_len: usize,
}

impl TransformerANE {
    /// Create new TransformerANE model
    ///
    /// Initializes all weights with small random values (0.01).
    /// Ready for forward/backward passes immediately.
    ///
    /// # Arguments
    /// * `config` - TransformerConfig with validated dimensions
    ///
    /// # Returns
    /// New TransformerANE instance
    ///
    /// # Errors
    /// None for now, but may fail in future if ANE runtime is unavailable
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        // Embedding matrix: [vocab_size, dim]
        let embedding = vec![0.01f32; config.vocab_size * config.dim];

        // Classifier/output projection: [dim, vocab_size]
        let classifier = vec![0.01f32; config.dim * config.vocab_size];

        // Layer norms: 2 per layer (pre-attention, pre-FFN) each of size [dim]
        let mut layer_norms = Vec::new();
        for _ in 0..config.n_layers * 2 {
            layer_norms.push(vec![1.0f32; config.dim]);
        }

        // Attention weights per layer: Q, K, V projections each [dim, dim]
        let mut attention_weights = Vec::new();
        for _ in 0..config.n_layers {
            attention_weights.push(vec![0.01f32; 3 * config.dim * config.dim]);
        }

        // FFN weights per layer: w1, w3 (dim*hidden_dim each) + w2 (hidden_dim*dim)
        let mut ffn_weights = Vec::new();
        for _ in 0..config.n_layers {
            ffn_weights.push(vec![
                0.01f32;
                2 * config.dim * config.hidden_dim + config.hidden_dim * config.dim
            ]);
        }

        let mut trainable_params = Vec::with_capacity(config.param_count());
        trainable_params.extend_from_slice(&embedding);
        trainable_params.extend_from_slice(&classifier);
        for layer_norm in &layer_norms {
            trainable_params.extend_from_slice(layer_norm);
        }
        for attention in &attention_weights {
            trainable_params.extend_from_slice(attention);
        }
        for ffn in &ffn_weights {
            trainable_params.extend_from_slice(ffn);
        }

        Ok(TransformerANE {
            config: config.clone(),
            trainable_params,
            embedding,
            classifier,
            layer_norms,
            attention_weights,
            ffn_weights,
            cached: CachedActivations::new(),
            last_input_tokens: vec![],
            last_input_activations: vec![],
            last_logits: vec![],
            last_batch_size: 0,
            last_seq_len: 0,
        })
    }

    /// Get reference to model config
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Synchronize the cached weight views from the contiguous trainable buffer.
    fn sync_weights_from_params(&mut self) {
        let mut offset = 0;

        let embedding_len = self.embedding.len();
        self.embedding
            .copy_from_slice(&self.trainable_params[offset..offset + embedding_len]);
        offset += embedding_len;

        let classifier_len = self.classifier.len();
        self.classifier
            .copy_from_slice(&self.trainable_params[offset..offset + classifier_len]);
        offset += classifier_len;

        for layer_norm in &mut self.layer_norms {
            let len = layer_norm.len();
            layer_norm.copy_from_slice(&self.trainable_params[offset..offset + len]);
            offset += len;
        }

        for attention in &mut self.attention_weights {
            let len = attention.len();
            attention.copy_from_slice(&self.trainable_params[offset..offset + len]);
            offset += len;
        }

        for ffn in &mut self.ffn_weights {
            let len = ffn.len();
            ffn.copy_from_slice(&self.trainable_params[offset..offset + len]);
            offset += len;
        }
    }
}

impl Model for TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        self.sync_weights_from_params();

        // Clear previous caches
        self.cached.clear();
        self.last_input_tokens = batch.tokens().to_vec();
        self.last_batch_size = batch.batch_size();
        self.last_seq_len = batch.seq_len();

        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let tokens = batch.tokens();
        let dim = self.config.dim;

        if seq_len < 2 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be at least 2 for next-token training".to_string(),
            ));
        }

        if tokens.len() != batch_size * seq_len {
            return Err(crate::Error::InvalidParameter(
                "token count mismatch".to_string(),
            ));
        }

        // **Step 1: Embedding lookup**
        // Convert token ids to embedding vectors
        // Output shape: [batch_size * seq_len, dim]
        let mut x = vec![0.0f32; batch_size * seq_len * dim];
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize) % self.config.vocab_size;
            let emb_start = token_idx * dim;
            let x_start = i * dim;
            if x_start + dim <= x.len() && emb_start + dim <= self.embedding.len() {
                x[x_start..x_start + dim]
                    .copy_from_slice(&self.embedding[emb_start..emb_start + dim]);
            }
        }

        // **Step 2: Per-layer transformer forward**
        // Loop structure (simplified for Phase 2):
        // TODO: Implement layer loop with ANE kernel invocations
        //   1. Pre-attention RMSNorm
        //   2. Attention forward (via ANE)
        //   3. Residual
        //   4. Pre-FFN RMSNorm
        //   5. FFN forward (via ANE)
        //   6. Residual
        //   7. Cache all intermediates
        //
        // For now, return logits directly from embedding for compilation

        // **Step 3: Output projection and classifier**
        // Final classifier (vocab projection)
        // We emit next-token logits, so the last position of each sequence is
        // intentionally skipped. That matches the cross-entropy layout used by
        // the training pipeline.
        let effective_seq_len = seq_len.saturating_sub(1);
        let mut logits = vec![0.0f32; batch_size * effective_seq_len * self.config.vocab_size];
        for sample_idx in 0..batch_size {
            let sample_offset = sample_idx * seq_len;
            for pos in 0..effective_seq_len {
                let token_idx = sample_offset + pos;
                let row_idx = sample_idx * effective_seq_len + pos;
                let x_start = token_idx * dim;
                let logit_start = row_idx * self.config.vocab_size;

                // Simple dot product with classifier (no proper forward yet)
                for j in 0..self.config.vocab_size {
                    let mut sum = 0.0f32;
                    for k in 0..dim {
                        if x_start + k < x.len() && j * dim + k < self.classifier.len() {
                            sum += x[x_start + k] * self.classifier[j * dim + k];
                        }
                    }
                    logits[logit_start + j] = sum;
                }
            }
        }

        self.last_input_activations = x.clone();
        self.last_logits = logits.clone();

        // Convert to ANETensor with shape [batch_size, seq_len - 1, vocab_size]
        let shape = vec![batch_size, effective_seq_len, self.config.vocab_size];
        ANETensor::from_fp32(logits, shape)
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        let total_params = self.param_count();
        let grads = vec![0.0f32; total_params];
        Ok(grads)
    }

    fn backward_with_batch(&mut self, batch: &Batch, _loss: f32) -> Result<Vec<f32>> {
        if self.last_logits.is_empty() || self.last_input_activations.is_empty() {
            return Err(crate::Error::Other(
                "forward cache missing; call forward before backward".to_string(),
            ));
        }

        if batch.tokens().len() != self.last_input_tokens.len()
            || batch.batch_size() != self.last_batch_size
            || batch.seq_len() != self.last_seq_len
        {
            return Err(crate::Error::Other(
                "batch used for backward does not match cached forward batch".to_string(),
            ));
        }

        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        if seq_len < 2 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be at least 2 for next-token training".to_string(),
            ));
        }
        let effective_seq_len = seq_len.saturating_sub(1);
        let vocab_size = self.config.vocab_size;
        let dim = self.config.dim;
        let output_positions = batch_size * effective_seq_len;
        let expected_logits = output_positions * vocab_size;
        if self.last_logits.len() != expected_logits {
            return Err(crate::Error::Other(
                "cached logits shape does not match expected training layout".to_string(),
            ));
        }

        let inv_output_positions = 1.0f32 / output_positions as f32;

        let mut grads = vec![0.0f32; self.param_count()];
        let embedding_len = self.embedding.len();
        let classifier_len = self.classifier.len();
        let mut d_embedding = vec![0.0f32; embedding_len];
        let mut d_classifier = vec![0.0f32; classifier_len];

        // Re-run the accumulation in a cache-friendly layout using the cached activations.
        for sample_idx in 0..batch_size {
            let sample_offset = sample_idx * seq_len;
            for pos in 0..effective_seq_len {
                let token_idx = sample_offset + pos;
                let row_idx = sample_idx * effective_seq_len + pos;
                let input_vec = &self.last_input_activations[token_idx * dim..(token_idx + 1) * dim];
                let logits_at_pos = &self.last_logits[row_idx * vocab_size..(row_idx + 1) * vocab_size];
                let target_token = batch.tokens()[sample_offset + pos + 1] as usize;
                let max_logit = logits_at_pos
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut exp_sum = 0.0f32;
                for &logit in logits_at_pos {
                    exp_sum += (logit - max_logit).exp();
                }

                for vocab_idx in 0..vocab_size {
                    let mut grad = (logits_at_pos[vocab_idx] - max_logit).exp() / exp_sum;
                    if vocab_idx == target_token {
                        grad -= 1.0;
                    }
                    grad *= inv_output_positions;

                    let cls_start = vocab_idx * dim;
                    for k in 0..dim {
                        d_classifier[cls_start + k] += grad * input_vec[k];
                        d_embedding[token_idx * dim + k] += grad * self.classifier[cls_start + k];
                    }
                }
            }
        }

        grads[..embedding_len].copy_from_slice(&d_embedding);
        grads[embedding_len..embedding_len + classifier_len].copy_from_slice(&d_classifier);
        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.trainable_params
    }

    fn param_count(&self) -> usize {
        self.config.param_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_ane_creation() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let model = TransformerANE::new(&config).unwrap();

        assert_eq!(model.param_count(), config.param_count());
    }

    #[test]
    fn test_transformer_ane_weight_initialization() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let model = TransformerANE::new(&config).unwrap();

        // Verify weight dimensions
        assert_eq!(model.embedding.len(), 256 * 128);
        assert_eq!(model.classifier.len(), 128 * 256);
        assert_eq!(model.layer_norms.len(), 2 * 2); // 2 per layer
        assert_eq!(model.attention_weights.len(), 2); // 1 per layer
        assert_eq!(model.ffn_weights.len(), 2); // 1 per layer
    }

    #[test]
    fn test_transformer_ane_weight_values() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let model = TransformerANE::new(&config).unwrap();

        // Check embedding initialized to 0.01
        assert!(model.embedding.iter().all(|&x| (x - 0.01).abs() < 1e-6));

        // Check layer norms initialized to 1.0
        assert!(model.layer_norms.iter().all(|ln| ln.iter().all(|&x| (x - 1.0).abs() < 1e-6)));
    }

    #[test]
    fn test_cached_activations_creation() {
        let cached = CachedActivations::new();

        assert_eq!(cached.x_pre_attn_norm.len(), 0);
        assert_eq!(cached.q.len(), 0);
        assert_eq!(cached.attn_weights.len(), 0);
    }

    #[test]
    fn test_cached_activations_clear() {
        let mut cached = CachedActivations::new();

        // Simulate population (in real forward pass)
        cached.x_pre_attn_norm.push(vec![1.0f32; 100]);
        cached.q.push(vec![2.0f32; 50]);

        assert!(cached.x_pre_attn_norm.len() > 0);
        assert!(cached.q.len() > 0);

        // Clear
        cached.clear();

        assert_eq!(cached.x_pre_attn_norm.len(), 0);
        assert_eq!(cached.q.len(), 0);
    }

    #[test]
    fn test_transformer_ane_param_count() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
        let model = TransformerANE::new(&config).unwrap();

        let expected = config.param_count();
        assert_eq!(model.param_count(), expected);
        assert!(expected > 6_800_000);
        assert!(expected < 6_900_000);
    }

    #[test]
    fn test_transformer_ane_forward_small_batch() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();

        let tokens = vec![0u32; 2 * 64]; // 2 samples, 64 seq_len
        let batch = Batch::new(tokens, 2, 64).unwrap();

        let result = model.forward(&batch);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        // Output should be next-token logits: [batch_size, seq_len - 1, vocab_size]
        let expected_elements = 2 * (64 - 1) * 256;
        assert_eq!(tensor.num_elements(), expected_elements);
    }

    #[test]
    fn test_transformer_ane_backward() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();

        let grads = model.backward(0.5).unwrap();
        assert_eq!(grads.len(), config.param_count());
    }

    #[test]
    fn test_transformer_ane_backward_with_batch() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        let _ = model.forward(&batch).unwrap();
        let grads = model.backward_with_batch(&batch, 0.5).unwrap();

        assert_eq!(grads.len(), config.param_count());
        assert!(grads.iter().any(|g| *g != 0.0));
    }
}
