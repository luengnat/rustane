//! Integration tests for TransformerANE model

use rustane::data::Batch;
use rustane::training::{CrossEntropyLoss, LossFn, Model, TransformerANE, TransformerConfig};

fn tiny_config() -> TransformerConfig {
    TransformerConfig::new(5, 4, 8, 2, 1, 3).unwrap()
}

fn tiny_batch() -> Batch {
    Batch::new(vec![0, 1, 2], 1, 3).unwrap()
}

fn copy_parameters(dst: &mut TransformerANE, src: &mut TransformerANE) {
    let params = src.parameters().to_vec();
    dst.parameters().copy_from_slice(&params);
}

fn loss_for_batch(model: &mut TransformerANE, batch: &Batch) -> f32 {
    let loss_fn = CrossEntropyLoss::new();
    let logits = model.forward(batch).unwrap();
    loss_fn.compute(&logits, batch).unwrap()
}

fn set_deterministic_parameters(model: &mut TransformerANE) {
    let groups = model.parameter_groups();
    let params = model.parameters();

    for group in groups {
        for (offset, value) in params[group.range.clone()].iter_mut().enumerate() {
            *value = match group.kind {
                rustane::training::ParameterGroupKind::Scalar => 1.0 + (offset as f32) * 0.001,
                _ => {
                    let centered = ((group.range.start + offset) % 23) as f32 - 11.0;
                    centered * 0.002
                }
            };
        }
    }
}

fn assert_close(lhs: &[f32], rhs: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            diff <= abs_tol + rel_tol * scale,
            "mismatch at index {}: lhs={} rhs={} diff={} abs_tol={} rel_tol={}",
            idx,
            a,
            b,
            diff,
            abs_tol,
            rel_tol
        );
    }
}

fn parameter_index(model: &TransformerANE, name: &str, offset: usize) -> usize {
    let group = model
        .parameter_groups()
        .into_iter()
        .find(|group| group.name == name)
        .unwrap_or_else(|| panic!("missing parameter group {}", name));
    group.range.start + offset
}

fn finite_difference_gradient(
    model: &mut TransformerANE,
    batch: &Batch,
    index: usize,
    epsilon: f32,
) -> f32 {
    let original = model.parameters()[index];

    model.parameters()[index] = original + epsilon;
    let plus = loss_for_batch(model, batch);

    model.parameters()[index] = original - epsilon;
    let minus = loss_for_batch(model, batch);

    model.parameters()[index] = original;

    (plus - minus) / (2.0 * epsilon)
}

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

#[test]
fn test_transformer_ane_forward_matches_checkpointing() {
    let config = tiny_config();
    let mut baseline = TransformerANE::new(&config).unwrap();
    let mut checkpointed = TransformerANE::new(&config.with_checkpoint_interval(2)).unwrap();
    let batch = tiny_batch();

    set_deterministic_parameters(&mut baseline);
    copy_parameters(&mut checkpointed, &mut baseline);

    let baseline_logits = baseline.forward(&batch).unwrap().to_vec_f32();
    let checkpointed_logits = checkpointed.forward(&batch).unwrap().to_vec_f32();

    assert_close(&baseline_logits, &checkpointed_logits, 1e-6, 1e-6);
}

#[test]
fn test_transformer_ane_backward_matches_checkpointing() {
    let config = tiny_config();
    let mut baseline = TransformerANE::new(&config).unwrap();
    let mut checkpointed = TransformerANE::new(&config.with_checkpoint_interval(2)).unwrap();
    let batch = tiny_batch();

    set_deterministic_parameters(&mut baseline);
    copy_parameters(&mut checkpointed, &mut baseline);

    let _ = baseline.forward(&batch).unwrap();
    let _ = checkpointed.forward(&batch).unwrap();

    let baseline_grads = baseline.backward_with_batch(&batch, 1.0).unwrap();
    let checkpointed_grads = checkpointed.backward_with_batch(&batch, 1.0).unwrap();

    assert_close(&baseline_grads, &checkpointed_grads, 1e-5, 1e-5);
}

#[test]
fn test_transformer_ane_backward_matches_finite_difference() {
    let config = tiny_config();
    let mut model = TransformerANE::new(&config).unwrap();
    let batch = tiny_batch();

    set_deterministic_parameters(&mut model);
    let _ = model.forward(&batch).unwrap();
    let analytic_grads = model.backward_with_batch(&batch, 1.0).unwrap();

    let indices = [
        parameter_index(&model, "layers.0.wq", 0),
        parameter_index(&model, "layers.0.w1", 0),
        parameter_index(&model, "final_norm", 0),
        parameter_index(&model, "lm_head", 0),
    ];

    let mut failures = Vec::new();
    for &index in &indices {
        let numeric = finite_difference_gradient(&mut model, &batch, index, 1e-3);
        let analytic = analytic_grads[index];
        let diff = (numeric - analytic).abs();
        let scale = numeric.abs().max(analytic.abs()).max(1.0);
        if diff > 2e-2 * scale + 2e-3 {
            failures.push((index, analytic, numeric, diff));
        }
    }

    assert!(
        failures.is_empty(),
        "gradient mismatches: {:?}",
        failures
    );
}
