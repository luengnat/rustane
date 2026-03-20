//! Transformer model configuration with validation and parameter counting.

use crate::ane::ANEError;

/// Transformer model configuration
///
/// Defines the architecture of a transformer model with validation for ANE compatibility.
/// Ensures all dimensions are properly aligned for efficient computation on Apple Neural Engine.
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
}

impl TransformerConfig {
    /// Create and validate a transformer configuration.
    ///
    /// Validates that:
    /// - `dim` is divisible by `n_heads` (required for multi-head attention)
    /// - `dim` is divisible by 128 (ANE efficiency requirement)
    /// - `hidden_dim` is divisible by 128 (ANE efficiency requirement)
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
        let head_dim = dim / n_heads;

        // Validate that dim is divisible by n_heads
        if dim % n_heads != 0 {
            return Err(ANEError::ConfigError(
                format!("dim {} must be divisible by n_heads {}", dim, n_heads),
            ));
        }

        // Validate that dim is divisible by 128 for ANE efficiency
        if dim % 128 != 0 {
            return Err(ANEError::ConfigError(format!(
                "dim {} must be divisible by 128 for ANE efficiency",
                dim
            )));
        }

        // Validate that hidden_dim is divisible by 128 for ANE efficiency
        if hidden_dim % 128 != 0 {
            return Err(ANEError::ConfigError(format!(
                "hidden_dim {} must be divisible by 128 for ANE efficiency",
                hidden_dim
            )));
        }

        Ok(TransformerConfig {
            vocab_size,
            dim,
            hidden_dim,
            n_heads,
            head_dim,
            n_layers,
            seq_len,
        })
    }

    /// Calculate total parameter count for this configuration.
    ///
    /// Counts parameters for:
    /// - Embedding layer: `vocab_size * dim`
    /// - Classifier/output layer: `dim * vocab_size`
    /// - Per transformer layer (x n_layers):
    ///   - Query, Key, Value projections: `3 * dim * dim`
    ///   - Feed-forward w1 and w3: `2 * dim * hidden_dim`
    ///   - Feed-forward w2: `hidden_dim * dim`
    ///   - Two layer normalizations: `2 * dim`
    pub fn param_count(&self) -> usize {
        // Embedding and classifier layers
        let embedding = self.vocab_size * self.dim;
        let classifier = self.dim * self.vocab_size;

        // Per-layer parameters
        let per_layer = 3 * self.dim * self.dim         // QKV projections
            + self.dim * self.hidden_dim * 2            // w1, w3 in FFN
            + self.hidden_dim * self.dim                // w2 in FFN
            + 2 * self.dim;                             // layer norms

        embedding + classifier + per_layer * self.n_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_count_calculation() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)
            .expect("valid config");

        // Manual calculation:
        // embedding: 4096 * 256 = 1,048,576
        // classifier: 256 * 4096 = 1,048,576
        // per_layer: 3 * 256 * 256 + 256 * 768 * 2 + 768 * 256 + 2 * 256
        //          = 196,608 + 393,216 + 196,608 + 512
        //          = 786,944
        // total: 1,048,576 + 1,048,576 + 786,944 * 6
        //      = 2,097,152 + 4,721,664
        //      = 6,818,816

        let expected = 1_048_576 + 1_048_576 + 786_944 * 6;
        assert_eq!(config.param_count(), expected);
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)
            .expect("valid config");
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
    fn test_validation_dim_divisible_by_128() {
        let result = TransformerConfig::new(4096, 256, 100, 8, 6, 512);
        assert!(result.is_err());
        match result {
            Err(ANEError::ConfigError(msg)) => {
                assert!(msg.contains("hidden_dim"));
                assert!(msg.contains("128"));
            }
            _ => panic!("expected ConfigError"),
        }
    }

    #[test]
    fn test_validation_hidden_dim_divisible_by_128() {
        let result = TransformerConfig::new(4096, 128, 129, 4, 3, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_clone() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)
            .expect("valid config");
        let cloned = config.clone();
        assert_eq!(config.param_count(), cloned.param_count());
        assert_eq!(config.dim, cloned.dim);
    }
}
