//! Integration tests for TransformerANE model

use rustane::data::Batch;
use rustane::training::{Model, TransformerANE, TransformerConfig};

#[test]
fn test_transformer_ane_forward_pass() {
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();

    // Create dummy batch
    let tokens = vec![0u32; 2 * 64]; // 2 samples, 64 seq_len
    let batch = Batch::new(tokens, 2, 64).unwrap();

    let result = model.forward(&batch);

    // Should either succeed or fail gracefully
    // (ANE may not be available, but no panic)
    match result {
        Ok(_tensor) => {
            // Forward pass succeeded
            assert!(_tensor.num_elements() > 0);
        }
        Err(e) => {
            // Forward pass not available is acceptable
            eprintln!("Forward pass not available: {:?}", e);
        }
    }
}

#[test]
fn test_transformer_ane_implements_model_trait() {
    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
    let model = TransformerANE::new(&config).unwrap();

    let param_count = model.param_count();
    assert!(param_count > 7_000_000);
    assert!(param_count < 7_300_000);

    // Verify the expected count matches config
    assert_eq!(param_count, config.param_count());
}

#[test]
fn test_transformer_ane_backward_pass() {
    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();

    let result = model.backward(0.5);

    match result {
        Ok(grads) => {
            // Gradients should have correct count
            assert_eq!(grads.len(), config.param_count());
        }
        Err(e) => {
            eprintln!("Backward pass not available: {:?}", e);
        }
    }
}

#[test]
fn test_transformer_ane_parameters_access() {
    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();

    let params = model.parameters();
    assert!(!params.is_empty());
    assert_eq!(params.len(), model.param_count());
}

#[test]
fn test_transformer_ane_small_config() {
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
    let model = TransformerANE::new(&config).unwrap();

    let expected_params = config.param_count();
    assert_eq!(model.param_count(), expected_params);
}

#[test]
fn test_transformer_ane_batch_size_one() {
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();

    let tokens = vec![0u32; 1 * 64]; // 1 sample, 64 seq_len
    let batch = Batch::new(tokens, 1, 64).unwrap();

    let result = model.forward(&batch);
    // Should not panic
    let _ = result;
}

#[test]
fn test_transformer_ane_large_batch() {
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();

    let tokens = vec![1u32; 8 * 64]; // 8 samples, 64 seq_len
    let batch = Batch::new(tokens, 8, 64).unwrap();

    let result = model.forward(&batch);
    // Should not panic
    let _ = result;
}
