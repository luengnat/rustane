#[cfg(test)]
mod transformer_config_tests {
    use rustane::training::TransformerConfig;

    #[test]
    fn test_transformer_config_6_8m_params() {
        let config = TransformerConfig::new(
            4096,  // vocab_size
            256,   // dim
            768,   // hidden_dim
            8,     // n_heads
            6,     // n_layers
            512,   // seq_len
        ).expect("config should be valid");

        // Verify parameter count: ~7.2M
        let embedding_params = 4096 * 256;
        let classifier_params = 256 * 4096;
        let per_layer_params = 4 * 256 * 256 +  // attention: qkv + output
                               256 * 768 * 2 +  // ffn: w1, w3
                               768 * 256 +      // ffn: w2
                               2 * 256;         // layer norms
        let layer_params = per_layer_params * 6;
        let total = embedding_params + classifier_params + layer_params + 256;

        assert!(config.param_count() > 7_000_000);
        assert!(config.param_count() < 7_300_000);
        assert_eq!(config.param_count(), total);
        
        // Print for debugging
        println!("Embedding params: {}", embedding_params);
        println!("Classifier params: {}", classifier_params);
        println!("Per-layer params: {}", per_layer_params);
        println!("Layer params (6 layers): {}", layer_params);
        println!("Expected total: {}", total);
        println!("Actual param_count(): {}", config.param_count());
    }

    #[test]
    fn test_transformer_config_head_dim_computed() {
        let config = TransformerConfig::new(
            4096,  // vocab_size
            256,   // dim
            768,   // hidden_dim
            8,     // n_heads
            6,     // n_layers
            512,   // seq_len
        ).expect("config should be valid");

        assert_eq!(config.head_dim, 32); // 256 / 8
    }

    #[test]
    fn test_transformer_config_validation_dim_divisible_by_n_heads() {
        // Invalid: dim not divisible by n_heads
        let result = TransformerConfig::new(
            4096,  // vocab_size
            255,   // dim (not divisible by 8)
            768,   // hidden_dim
            8,     // n_heads
            6,     // n_layers
            512,   // seq_len
        );
        assert!(result.is_err());
        if let Err(e) = result {
            println!("Error message: {}", e);
        }
    }

    #[test]
    fn test_transformer_config_validation_zero_heads() {
        // Invalid: n_heads must be non-zero
        let result = TransformerConfig::new(
            4096,  // vocab_size
            256,   // dim
            100,   // hidden_dim
            0,     // n_heads
            6,     // n_layers
            512,   // seq_len
        );
        assert!(result.is_err());
        if let Err(e) = result {
            println!("Error message: {}", e);
        }
    }

    #[test]
    fn test_transformer_config_valid_small() {
        let config = TransformerConfig::new(
            512,   // vocab_size
            128,   // dim
            384,   // hidden_dim
            4,     // n_heads
            3,     // n_layers
            256,   // seq_len
        ).expect("config should be valid");

        assert_eq!(config.vocab_size, 512);
        assert_eq!(config.dim, 128);
        assert_eq!(config.hidden_dim, 384);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.head_dim, 32); // 128 / 4
        assert_eq!(config.n_layers, 3);
        assert_eq!(config.seq_len, 256);
        
        // Calculate expected params
        let embedding = 512 * 128;
        let classifier = 128 * 512;
        let per_layer = 4 * 128 * 128 + 128 * 384 * 2 + 384 * 128 + 2 * 128;
        let expected = embedding + classifier + per_layer * 3 + 128;
        
        assert_eq!(config.param_count(), expected);
    }

    #[test]
    fn test_transformer_config_debug_and_clone() {
        let config = TransformerConfig::new(
            4096,  // vocab_size
            256,   // dim
            768,   // hidden_dim
            8,     // n_heads
            6,     // n_layers
            512,   // seq_len
        ).expect("config should be valid");

        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.dim, cloned.dim);
        assert_eq!(config.param_count(), cloned.param_count());
        
        // Ensure Debug trait works
        println!("{:?}", config);
    }
}
