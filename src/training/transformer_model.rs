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

    // Weights (host memory, CPU-accessible for optimizer updates)
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

        Ok(TransformerANE {
            config: config.clone(),
            embedding,
            classifier,
            layer_norms,
            attention_weights,
            ffn_weights,
            cached: CachedActivations::new(),
        })
    }

    /// Get reference to model config
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

impl Model for TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        // Clear previous caches
        self.cached.clear();

        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let tokens = batch.tokens();
        let dim = self.config.dim;

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
        // Final RMSNorm followed by classifier (vocab projection)
        // Logits shape: [batch_size * seq_len, vocab_size]
        let mut logits = vec![0.0f32; batch_size * seq_len * self.config.vocab_size];
        for i in 0..batch_size * seq_len {
            let x_start = i * dim;
            let logit_start = i * self.config.vocab_size;

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

        // Convert to ANETensor with shape [batch_size, seq_len, vocab_size]
        let shape = vec![batch_size, seq_len, self.config.vocab_size];
        ANETensor::from_fp32(logits, shape)
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        // Start from loss gradient
        let total_params = self.param_count();
        let grads = vec![0.0f32; total_params];

        // TODO: Backprop through all layers using cached activations
        // For now, return zero gradients (stub implementation)

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        // Return embedding as primary parameter slice
        // In a full implementation, would flatten all weights into single slice
        &mut self.embedding
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
        // Output should be [batch_size, seq_len, vocab_size]
        let expected_elements = 2 * 64 * 256;
        assert_eq!(tensor.num_elements(), expected_elements);
    }

    #[test]
    fn test_transformer_ane_backward() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();

        let grads = model.backward(0.5).unwrap();
        assert_eq!(grads.len(), config.param_count());
    }
}
