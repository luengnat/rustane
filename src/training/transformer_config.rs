//! Transformer model configuration with validation and parameter counting.

use crate::ane::ANEError;

/// Gradient checkpointing configuration for memory-efficient training.
#[derive(Clone, Debug, Default)]
pub struct GradientCheckpointingConfig {
    /// Enable gradient checkpointing to reduce memory usage.
    /// When enabled, only every `checkpoint_interval` layer activations are stored.
    /// Missing activations are recomputed during the backward pass.
    pub enabled: bool,
    /// Store activations every N layers (default: 2).
    /// For example, with 12 layers and interval 2, only layers 0, 2, 4, 6, 8, 10 are stored.
    /// Lower values = more memory saved but more recomputation overhead.
    pub checkpoint_interval: usize,
}

impl GradientCheckpointingConfig {
    /// Create a new gradient checkpointing configuration.
    pub fn new(enabled: bool, checkpoint_interval: usize) -> Self {
        assert!(checkpoint_interval > 0, "checkpoint_interval must be > 0");
        Self {
            enabled,
            checkpoint_interval,
        }
    }

    /// Disable gradient checkpointing.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            checkpoint_interval: 1,
        }
    }

    /// Enable gradient checkpointing with specified interval.
    pub fn with_interval(checkpoint_interval: usize) -> Self {
        Self::new(true, checkpoint_interval)
    }

    /// Calculate memory savings factor (0.0 = no savings, 1.0 = 100% savings).
    pub fn memory_savings_factor(&self, n_layers: usize) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        // We save (interval - 1) / interval of the layer activations
        let checkpoints = (n_layers + self.checkpoint_interval - 1) / self.checkpoint_interval;
        let saved_layers = n_layers - checkpoints;
        saved_layers as f32 / n_layers as f32
    }
}

/// Transformer model configuration
///
/// Defines the architecture of a transformer model with validation for ANE compatibility.
/// Ensures the model dimensions are internally consistent for training.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    /// Vocabulary size (number of unique tokens)
    pub vocab_size: usize,
    /// Model embedding and output dimension
    pub dim: usize,
    /// Feed-forward network hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per attention head (computed as dim / n_heads)
    pub head_dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub seq_len: usize,
    /// Whether to tie the output projection to the input embedding matrix
    pub tie_embeddings: bool,
    /// Logit softcap used during the final projection
    pub logit_softcap: f32,
    /// Gradient checkpointing configuration for memory-efficient training.
    pub gradient_checkpointing: GradientCheckpointingConfig,
}

impl TransformerConfig {
    /// Create and validate a transformer configuration.
    ///
    /// Validates that:
    /// - `dim` is divisible by `n_heads` (required for multi-head attention)
    /// - `n_heads`, `dim`, and `hidden_dim` are non-zero
    ///
    /// # Arguments
    /// * `vocab_size` - Number of unique tokens in vocabulary
    /// * `dim` - Model embedding dimension
    /// * `hidden_dim` - Feed-forward hidden dimension
    /// * `n_heads` - Number of attention heads
    /// * `n_layers` - Number of transformer layers
    /// * `seq_len` - Maximum sequence length
    ///
    /// # Errors
    /// Returns `ANEError::ConfigError` if validation fails.
    pub fn new(
        vocab_size: usize,
        dim: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_layers: usize,
        seq_len: usize,
    ) -> Result<Self, ANEError> {
        if n_heads == 0 {
            return Err(ANEError::ConfigError(
                "n_heads must be greater than zero".to_string(),
            ));
        }

        if dim == 0 {
            return Err(ANEError::ConfigError(
                "dim must be greater than zero".to_string(),
            ));
        }

        if hidden_dim == 0 {
            return Err(ANEError::ConfigError(
                "hidden_dim must be greater than zero".to_string(),
            ));
        }

        // Validate that dim is divisible by n_heads
        if dim % n_heads != 0 {
            return Err(ANEError::ConfigError(format!(
                "dim {} must be divisible by n_heads {}",
                dim, n_heads
            )));
        }

        let head_dim = dim / n_heads;

        Ok(TransformerConfig {
            vocab_size,
            dim,
            hidden_dim,
            n_heads,
            head_dim,
            n_layers,
            seq_len,
            tie_embeddings: false,
            logit_softcap: 30.0,
            gradient_checkpointing: GradientCheckpointingConfig::disabled(),
        })
    }

    /// Create a small default configuration used by tests and examples.
    pub fn tiny() -> Self {
        Self::new(256, 128, 256, 4, 2, 64).expect("tiny config should be valid")
    }

    /// Set gradient checkpointing configuration.
    pub fn with_gradient_checkpointing(mut self, gradient_checkpointing: GradientCheckpointingConfig) -> Self {
        self.gradient_checkpointing = gradient_checkpointing;
        self
    }

    /// Enable gradient checkpointing with specified interval.
    pub fn with_checkpoint_interval(mut self, checkpoint_interval: usize) -> Self {
        self.gradient_checkpointing = GradientCheckpointingConfig::with_interval(checkpoint_interval);
        self
    }

    /// Enable or disable tied input/output embeddings.
    pub fn with_tie_embeddings(mut self, tie_embeddings: bool) -> Self {
        self.tie_embeddings = tie_embeddings;
        self
    }

    /// Set the final logit softcap.
    pub fn with_logit_softcap(mut self, logit_softcap: f32) -> Self {
        self.logit_softcap = logit_softcap;
        self
    }

    /// Calculate total parameter count for this configuration.
    ///
    /// Counts parameters for:
    /// - Embedding layer: `vocab_size * dim`
    /// - Classifier/output layer: `dim * vocab_size`
    /// - Per transformer layer (x n_layers):
    ///   - Query, Key, Value projections: `3 * dim * dim`
    ///   - Attention output projection: `dim * dim`
    ///   - Feed-forward w1 and w3: `2 * dim * hidden_dim`
    ///   - Feed-forward w2: `hidden_dim * dim`
    ///   - Two RMSNorms: `2 * dim`
    /// - Final RMSNorm: `dim`
    pub fn param_count(&self) -> usize {
        // Embedding and classifier layers
        let embedding = self.vocab_size * self.dim;
        let classifier = if self.tie_embeddings {
            0
        } else {
            self.dim * self.vocab_size
        };

        // Per-layer parameters
        let per_layer = 4 * self.dim * self.dim         // QKV + attention output projection
            + self.dim * self.hidden_dim * 2            // w1, w3 in FFN
            + self.hidden_dim * self.dim                // w2 in FFN
            + 2 * self.dim; // RMSNorms

        embedding + classifier + per_layer * self.n_layers + self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_count_calculation() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).expect("valid config");

        // Manual calculation:
        // embedding: 4096 * 256 = 1,048,576
        // classifier: 256 * 4096 = 1,048,576
        // per_layer: 4 * 256 * 256 + 256 * 768 * 2 + 768 * 256 + 2 * 256
        //          = 262,144 + 393,216 + 196,608 + 512
        //          = 852,480
        // final_norm: 256
        // total: 1,048,576 + 1,048,576 + 852,480 * 6 + 256
        //      = 2,097,152 + 5,114,880 + 256
        //      = 7,212,288

        let expected = 1_048_576 + 1_048_576 + 852_480 * 6 + 256;
        assert_eq!(config.param_count(), expected);
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).expect("valid config");
        assert_eq!(config.head_dim, 32);
    }

    #[test]
    fn test_validation_dim_divisible_by_heads() {
        let result = TransformerConfig::new(4096, 255, 768, 8, 6, 512);
        assert!(result.is_err());
        match result {
            Err(ANEError::ConfigError(msg)) => {
                assert!(msg.contains("divisible by n_heads"));
            }
            _ => panic!("expected ConfigError"),
        }
    }

    #[test]
    fn test_validation_zero_heads() {
        let result = TransformerConfig::new(4096, 256, 768, 0, 6, 512);
        assert!(result.is_err());
        match result {
            Err(ANEError::ConfigError(msg)) => {
                assert!(msg.contains("n_heads"));
            }
            _ => panic!("expected ConfigError"),
        }
    }

    #[test]
    fn test_validation_hidden_dim_nonzero() {
        let result = TransformerConfig::new(4096, 128, 0, 4, 3, 512);
        assert!(result.is_err());
        match result {
            Err(ANEError::ConfigError(msg)) => {
                assert!(msg.contains("hidden_dim"));
            }
            _ => panic!("expected ConfigError"),
        }
    }

    #[test]
    fn test_clone() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).expect("valid config");
        let cloned = config.clone();
        assert_eq!(config.param_count(), cloned.param_count());
        assert_eq!(config.dim, cloned.dim);
    }

    #[test]
    fn test_tied_embeddings_reduce_param_count() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)
            .expect("valid config")
            .with_tie_embeddings(true);
        let untied = TransformerConfig::new(4096, 256, 768, 8, 6, 512).expect("valid config");
        assert_eq!(untied.param_count() - 4096 * 256, config.param_count());
    }

    #[test]
    fn test_logit_softcap_setter() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)
            .expect("valid config")
            .with_logit_softcap(12.5);
        assert_eq!(config.logit_softcap, 12.5);
    }

    #[test]
    fn test_gradient_checkpointing_disabled_by_default() {
        let config = TransformerConfig::tiny();
        assert!(!config.gradient_checkpointing.enabled);
    }

    #[test]
    fn test_gradient_checkpointing_enabled() {
        let config = TransformerConfig::tiny()
            .with_checkpoint_interval(2);
        assert!(config.gradient_checkpointing.enabled);
        assert_eq!(config.gradient_checkpointing.checkpoint_interval, 2);
    }

    #[test]
    fn test_memory_savings_factor() {
        let gc = GradientCheckpointingConfig::with_interval(2);
        // With 12 layers and interval 2: store 0, 2, 4, 6, 8, 10 = 6 checkpoints
        // Save 6 out of 12 = 50% savings
        assert!((gc.memory_savings_factor(12) - 0.5).abs() < 0.01);

        // With interval 1: store all layers = 0% savings
        let gc_all = GradientCheckpointingConfig::with_interval(1);
        assert_eq!(gc_all.memory_savings_factor(12), 0.0);

        // Disabled = 0% savings
        let gc_disabled = GradientCheckpointingConfig::disabled();
        assert_eq!(gc_disabled.memory_savings_factor(12), 0.0);
    }

    #[test]
    fn test_checkpoint_interval_validation() {
        // interval 0 should panic
        let result = std::panic::catch_unwind(|| {
            GradientCheckpointingConfig::new(true, 0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_checkpointing_config_builder() {
        let gc = GradientCheckpointingConfig::new(true, 3);
        assert!(gc.enabled);
        assert_eq!(gc.checkpoint_interval, 3);
    }
}
