//! Integration Tests for ANE Backward Pass
//!
//! Tests end-to-end backward pass functionality:
//! - Forward → Backward → Optimizer workflow
//! - Gradient accumulation
//! - ANEGradientAccumulator integration
//! - Validation suite integration

use rustane::data::Batch;
use rustane::layers::backward::validation::{quick_validate, BackwardValidationSuite};
use rustane::training::{
    ANEGradientAccumulator, AdamOptimizer, ConstantScheduler, CrossEntropyLoss, LossFn, Model,
    TrainerBuilder, TransformerANE, TransformerConfig,
};

/// Helper to create a test configuration
fn test_config() -> TransformerConfig {
    TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap()
}

/// Helper to create a test batch
fn test_batch(batch_size: usize, seq_len: usize) -> Batch {
    let tokens: Vec<u32> = (0..(batch_size * seq_len) as u32).map(|i| i % 256).collect();
    Batch::new(tokens, batch_size, seq_len).unwrap()
}

#[test]
fn test_backward_validation_suite_quick() {
    // Run quick validation
    let report = quick_validate().unwrap();

    // All validations should pass (placeholder implementation)
    assert!(report.rmsnorm_passed, "RMSNorm backward should pass validation");
    assert!(report.attention_passed, "Attention backward should pass validation");
    assert!(report.ffn_passed, "FFN backward should pass validation");
    assert!(report.loss_passed, "Loss backward should pass validation");
    assert!(report.all_passed(), "All backward validations should pass");
}

#[test]
fn test_backward_validation_suite_with_config() {
    let config = test_config();
    let suite = BackwardValidationSuite::new();

    let report = suite.validate_all(&config).unwrap();

    // Verify report structure
    assert!(report.max_relative_error >= 0.0);
    assert_eq!(report.pass_count(), 4);
    assert_eq!(report.fail_count(), 0);
}

#[test]
fn test_gradient_accumulator_creation() {
    let config = test_config();
    let accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    assert_eq!(accumulator.num_params(), config.param_count());
    assert!(accumulator.is_empty());
    assert_eq!(accumulator.accumulation_count(), 0);
}

#[test]
fn test_gradient_accumulator_accumulation() {
    let config = test_config();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Create dummy gradients
    let grads1: Vec<f32> = (0..config.param_count()).map(|i| i as f32 * 0.01).collect();
    let grads2: Vec<f32> = (0..config.param_count()).map(|i| i as f32 * 0.02).collect();

    // Accumulate twice
    accumulator.accumulate(&grads1).unwrap();
    accumulator.accumulate(&grads2).unwrap();

    assert_eq!(accumulator.accumulation_count(), 2);
    assert!(!accumulator.is_empty());

    // Verify accumulated values
    let accumulated = accumulator.get_accumulated().unwrap();
    for i in 0..config.param_count() {
        let expected = i as f32 * 0.01 + i as f32 * 0.02;
        assert!((accumulated[i] - expected).abs() < 1e-6);
    }
}

#[test]
fn test_gradient_accumulator_reset() {
    let config = test_config();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Accumulate some gradients
    let grads = vec![1.0f32; config.param_count()];
    accumulator.accumulate(&grads).unwrap();
    assert!(!accumulator.is_empty());

    // Reset
    accumulator.reset().unwrap();
    assert!(accumulator.is_empty());
    assert_eq!(accumulator.accumulation_count(), 0);
}

#[test]
fn test_gradient_accumulator_max_abs() {
    let config = test_config();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Accumulate gradients with known max
    let mut grads = vec![0.0f32; config.param_count()];
    grads[10] = 5.0f32;
    grads[20] = -3.0f32;

    accumulator.accumulate(&grads).unwrap();

    assert_eq!(accumulator.max_abs_gradient(), 5.0f32);
}

#[test]
fn test_gradient_accumulator_scale() {
    let config = test_config();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Accumulate gradients
    let grads = vec![1.0f32; config.param_count()];
    accumulator.accumulate(&grads).unwrap();

    // Scale by 0.5
    accumulator.scale(0.5f32);

    let accumulated = accumulator.get_accumulated().unwrap();
    assert!(accumulated.iter().all(|&v| (v - 0.5f32).abs() < 1e-6));
}

#[test]
fn test_transformer_ane_forward_backward_integration() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    // Create batch
    let batch = test_batch(2, 32);

    // Forward pass
    let output = model.forward(&batch).unwrap();
    assert!(output.num_elements() > 0);

    // Backward pass
    let grads = model.backward_with_batch(&batch, 1.0f32).unwrap();
    assert_eq!(grads.len(), config.param_count());

    // Verify gradients are not all zero
    assert!(grads.iter().any(|&g| g != 0.0f32));
}

#[test]
fn test_transformer_ane_backward_on_ane() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Create batch
    let batch = test_batch(2, 32);

    // Forward pass
    let _output = model.forward(&batch).unwrap();

    // Backward on ANE (uses default CPU implementation)
    let result = model.backward_on_ane(&batch, 1.0f32, &mut accumulator);
    assert!(result.is_ok());

    // Verify gradients were accumulated
    assert!(!accumulator.is_empty());
    assert_eq!(accumulator.accumulation_count(), 1);
}

#[test]
fn test_transformer_ane_backward_on_ane_requires_forward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Try backward without forward (should fail)
    let batch = test_batch(2, 32);
    let result = model.backward_on_ane(&batch, 1.0f32, &mut accumulator);

    assert!(result.is_err());
}

#[test]
fn test_transformer_ane_backward_on_ane_requires_matching_batch() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Forward with one batch
    let batch1 = test_batch(2, 32);
    let _output = model.forward(&batch1).unwrap();

    // Backward with different batch (should fail)
    let batch2 = test_batch(4, 16);
    let result = model.backward_on_ane(&batch2, 1.0f32, &mut accumulator);

    assert!(result.is_err());
}

#[test]
fn test_trainer_with_ane_backward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    let optimizer = AdamOptimizer::new(config.param_count());
    let scheduler = ConstantScheduler::new(0.001);
    let loss_fn = CrossEntropyLoss::new();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(optimizer)
        .with_scheduler(scheduler)
        .with_loss_fn(loss_fn)
        .build()
        .unwrap();

    // Training step
    let batch = test_batch(2, 32);
    let metrics = trainer.train_step(&batch).unwrap();

    // Verify metrics
    assert!(metrics.loss >= 0.0);
    assert!(metrics.learning_rate > 0.0);
}

#[test]
fn test_cross_entropy_loss_integration() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    // Forward pass
    let batch = test_batch(2, 32);
    let output = model.forward(&batch).unwrap();

    // Compute loss
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    assert!(loss >= 0.0);
}

#[test]
fn test_gradient_accumulation_multiple_steps() {
    let config = test_config();
    let mut accumulator = ANEGradientAccumulator::from_config(&config).unwrap();

    // Simulate multiple backward steps
    for step in 0..5 {
        let grads: Vec<f32> = (0..config.param_count())
            .map(|i| (i + step) as f32 * 0.001)
            .collect();
        accumulator.accumulate(&grads).unwrap();
    }

    assert_eq!(accumulator.accumulation_count(), 5);

    // Verify accumulated values
    let accumulated = accumulator.get_accumulated().unwrap();
    for i in 0..config.param_count() {
        // Sum of (i + step) * 0.001 for step in 0..5
        let expected: f32 = (0..5).map(|step| (i + step) as f32 * 0.001).sum();
        assert!((accumulated[i] - expected).abs() < 1e-4);
    }
}

#[test]
fn test_model_parameters_update_after_backward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    // Get initial parameters
    let initial_params: Vec<f32> = model.parameters().to_vec();

    // Forward and backward
    let batch = test_batch(2, 32);
    let _output = model.forward(&batch).unwrap();
    let grads = model.backward_with_batch(&batch, 1.0f32).unwrap();

    // Simulate optimizer step (simple SGD)
    let lr = 0.01f32;
    for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
        *param -= lr * grad;
    }

    // Verify parameters changed
    let updated_params: Vec<f32> = model.parameters().to_vec();
    assert_ne!(initial_params, updated_params);
}

#[test]
fn test_batch_size_one_backward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    // Single sample batch
    let batch = test_batch(1, 32);
    let output = model.forward(&batch).unwrap();
    assert!(output.num_elements() > 0);

    let grads = model.backward_with_batch(&batch, 1.0f32).unwrap();
    assert_eq!(grads.len(), config.param_count());
}

#[test]
fn test_large_batch_backward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    // Larger batch
    let batch = test_batch(8, 64);
    let output = model.forward(&batch).unwrap();
    assert!(output.num_elements() > 0);

    let grads = model.backward_with_batch(&batch, 1.0f32).unwrap();
    assert_eq!(grads.len(), config.param_count());
}
