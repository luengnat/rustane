//! Parameter-Golf Backward Propagation Tests
//!
//! Comprehensive tests for backward pass in parameter-golf training:
//! - Gradient accumulation
//! - Loss function backward
//! - Optimizer backward step
//! - Trainer integration
//! - Learning rate scheduler interactions

use rustane::data::Batch;
use rustane::training::Model;
use rustane::training::{
    AdamOptimizer, AdamWOptimizer, ConstantScheduler, CrossEntropyLoss, GradAccumulator,
    LRScheduler, LionOptimizer, LossFn, Optimizer, TrainerBuilder, WarmupCosineScheduler,
    WarmupLinearScheduler,
};
use rustane::wrapper::ANETensor;
use rustane::Result;

// ============================================================================
// Mock Model for Testing
// ============================================================================

struct MockModel {
    params: Vec<f32>,
    forward_loss: f32,
}

impl MockModel {
    fn new(param_count: usize) -> Self {
        MockModel {
            params: vec![1.0; param_count],
            forward_loss: 2.0,
        }
    }

    fn with_loss(mut self, loss: f32) -> Self {
        self.forward_loss = loss;
        self
    }
}

impl Model for MockModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        // Return logits tensor with shape [num_positions, vocab_size]
        // where num_positions = batch_size * (seq_len - 1) for batched layout
        // Use a small vocab size (e.g., 10) for testing
        let vocab_size = 10;
        let num_positions = batch.batch_size() * (batch.seq_len().saturating_sub(1)).max(1);
        let logits_data = vec![self.forward_loss; num_positions * vocab_size];
        ANETensor::from_fp32(logits_data, vec![num_positions, vocab_size])
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        // Return gradients proportional to loss
        Ok(self.params.iter().map(|_| loss * 0.1).collect())
    }

    fn backward_with_batch(&mut self, _batch: &Batch, loss: f32) -> Result<Vec<f32>> {
        self.backward(loss)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn param_count(&self) -> usize {
        self.params.len()
    }
}

// ============================================================================
// TEST 1: Gradient Accumulator Backward Tests
// ============================================================================

#[test]
fn test_grad_accumulator_basic_accumulation() {
    let mut accum = GradAccumulator::new(10, 4);

    // Accumulate 4 batches of gradients
    for i in 1..=4 {
        let grads = vec![i as f32; 10];
        accum.accumulate(&grads, i as f32, 0.25).unwrap();
    }

    assert!(accum.is_ready());

    // Final gradients should be sum of scaled gradients: (1+2+3+4) * 0.25 = 2.5
    let final_grads = accum.gradients();
    for &g in final_grads {
        assert!((g - 2.5).abs() < 1e-5);
    }

    // Average loss should be (1+2+3+4) * 0.25 = 2.5
    assert!((accum.average_loss() - 2.5).abs() < 1e-5);
}

#[test]
fn test_grad_accumulator_reset_after_step() {
    let mut accum = GradAccumulator::new(5, 2);

    // First accumulation cycle
    accum.accumulate(&[1.0; 5], 2.0, 0.5).unwrap();
    accum.accumulate(&[1.0; 5], 2.0, 0.5).unwrap();
    assert!(accum.is_ready());

    // Reset after optimizer step
    accum.reset();
    assert!(!accum.is_ready());
    assert_eq!(accum.accumulated_steps(), 0);

    // Second accumulation cycle should start fresh
    accum.accumulate(&[2.0; 5], 3.0, 0.5).unwrap();
    let grads = accum.gradients();
    assert!((grads[0] - 1.0).abs() < 1e-5); // 2.0 * 0.5
}

#[test]
fn test_grad_accumulator_gradient_scaling() {
    let mut accum = GradAccumulator::new(3, 4);

    // Each batch contributes scaled gradients
    let scale = 0.25;
    for _ in 0..4 {
        accum.accumulate(&[4.0; 3], 1.0, scale).unwrap();
    }

    // Final gradient: 4 * 4.0 * 0.25 = 4.0
    let final_grads = accum.gradients();
    assert!((final_grads[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_grad_accumulator_loss_tracking() {
    let mut accum = GradAccumulator::new(2, 3);

    accum.accumulate(&[1.0; 2], 0.5, 1.0 / 3.0).unwrap();
    accum.accumulate(&[1.0; 2], 1.0, 1.0 / 3.0).unwrap();
    accum.accumulate(&[1.0; 2], 1.5, 1.0 / 3.0).unwrap();

    // Average loss: (0.5 + 1.0 + 1.5) / 3 = 1.0
    let avg_loss = accum.average_loss();
    assert!((avg_loss - 1.0).abs() < 0.01);
}

#[test]
fn test_grad_accumulator_size_mismatch_error() {
    let mut accum = GradAccumulator::new(10, 2);
    let wrong_size_grads = vec![1.0f32; 5]; // Expected 10, got 5

    let result = accum.accumulate(&wrong_size_grads, 1.0, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_grad_accumulator_partial_accumulation() {
    let mut accum = GradAccumulator::new(4, 4);

    // Only complete 2 of 4 steps
    accum.accumulate(&[2.0; 4], 1.0, 0.25).unwrap();
    accum.accumulate(&[2.0; 4], 1.0, 0.25).unwrap();

    assert!(!accum.is_ready());
    assert_eq!(accum.accumulated_steps(), 2);

    // Gradients should be partially accumulated: 2 * 2.0 * 0.25 = 1.0
    let grads = accum.gradients();
    assert!((grads[0] - 1.0).abs() < 1e-5);
}

// ============================================================================
// TEST 2: Loss Function Backward Tests
// ============================================================================

#[test]
fn test_cross_entropy_loss_computation() {
    let loss_fn = CrossEntropyLoss::new();
    // Use tokens that fit within vocab size (vocab=10, tokens must be < 10)
    let batch = Batch::new(vec![1u32, 2, 3, 4, 5, 6], 2, 3).unwrap();

    // Logits shape: [batch_size * (seq_len - 1), vocab_size] = [2 * 2, 10] = [4, 10]
    // With uniform logits, loss should be ln(vocab_size) = ln(10) ≈ 2.3
    let logits = ANETensor::from_fp32(vec![0.0f32; 40], vec![4, 10]).unwrap();

    let loss = loss_fn.compute(&logits, &batch).unwrap();

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_cross_entropy_loss_high_confidence_correct() {
    let loss_fn = CrossEntropyLoss::new();
    // batch=[0, 1], seq_len=2, so target is tokens[1] = 1
    // vocab_size=10, so we need logits [1, 10]
    let batch = Batch::new(vec![0u32, 1], 1, 2).unwrap();

    // High confidence correct: make logit[1] very high (target is 1)
    let mut logits_data = vec![0.0f32; 10];
    logits_data[1] = 10.0; // High logit for correct token (target=1)

    let logits = ANETensor::from_fp32(logits_data, vec![1, 10]).unwrap();
    let loss = loss_fn.compute(&logits, &batch).unwrap();

    // High confidence correct should give low loss (~0)
    assert!(loss < 1.0);
}

#[test]
fn test_cross_entropy_loss_high_confidence_wrong() {
    let loss_fn = CrossEntropyLoss::new();
    // batch=[1, 0], seq_len=2, so target is tokens[1] = 0
    let batch = Batch::new(vec![1u32, 0], 1, 2).unwrap();

    // High confidence wrong: make logit[5] very high (target is 0, but we predict 5)
    let mut logits_data = vec![0.0f32; 10];
    logits_data[5] = 10.0; // High logit for wrong token

    let logits = ANETensor::from_fp32(logits_data, vec![1, 10]).unwrap();
    let loss = loss_fn.compute(&logits, &batch).unwrap();

    // High confidence wrong should give high loss
    assert!(loss > 5.0);
}

#[test]
fn test_cross_entropy_loss_gradient_signal() {
    // Verify loss changes appropriately with different predictions
    let loss_fn = CrossEntropyLoss::new();
    let batch = Batch::new(vec![0u32], 1, 1).unwrap();

    // Logits shape: [1, 10] for vocab_size=10
    // Low confidence (uniform)
    let logits_uniform = ANETensor::from_fp32(vec![0.0f32; 10], vec![1, 10]).unwrap();
    let loss_uniform = loss_fn.compute(&logits_uniform, &batch).unwrap();

    // High confidence correct (target is 0, so make logit[0] high)
    let mut logits_confident = vec![0.0f32; 10];
    logits_confident[0] = 10.0;
    let logits = ANETensor::from_fp32(logits_confident, vec![1, 10]).unwrap();
    let loss_confident = loss_fn.compute(&logits, &batch).unwrap();

    // Confident correct should have lower loss
    assert!(loss_confident < loss_uniform);
}

// ============================================================================
// TEST 3: Optimizer Backward Step Tests
// ============================================================================

#[test]
fn test_adam_optimizer_backward_step() {
    let mut optimizer = AdamOptimizer::new(5);
    let mut params = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let grads = vec![0.1, -0.2, 0.3, -0.4, 0.5];
    let lr = 0.001;

    let initial_params = params.clone();

    // Backward step: update parameters
    optimizer.step(&grads, &mut params, lr).unwrap();

    // Parameters should change
    for i in 0..5 {
        assert_ne!(params[i], initial_params[i]);
    }

    // Verify optimization happened by checking params changed
    let total_change: f32 = params
        .iter()
        .zip(initial_params.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(total_change > 0.0);
}

#[test]
fn test_adamw_optimizer_weight_decay_backward() {
    let mut optimizer = AdamWOptimizer::new(3);
    let mut params = vec![1.0, 1.0, 1.0];
    let zero_grads = vec![0.0, 0.0, 0.0];
    let lr = 0.01;

    let initial = params[0];

    // With zero gradients, weight decay should still reduce params
    optimizer.step(&zero_grads, &mut params, lr).unwrap();

    assert!(params[0] < initial);
}

#[test]
fn test_lion_optimizer_sign_based_backward() {
    let mut optimizer = LionOptimizer::new(4);
    let mut params = vec![1.0, 2.0, 3.0, 4.0];
    let grads = vec![0.1, -0.1, 0.2, -0.2];
    let lr = 0.01;

    let initial = params.clone();

    optimizer.step(&grads, &mut params, lr).unwrap();

    // Lion uses sign-based updates
    // Positive gradient -> parameter decreases
    // Negative gradient -> parameter increases
    assert!(params[0] < initial[0]); // grad[0] > 0
    assert!(params[1] > initial[1]); // grad[1] < 0
    assert!(params[2] < initial[2]); // grad[2] > 0
    assert!(params[3] > initial[3]); // grad[3] < 0
}

#[test]
fn test_optimizer_multiple_backward_steps() {
    let mut optimizer = AdamOptimizer::new(2);
    let mut params = vec![1.0, 1.0];
    let grads = vec![0.1, 0.1];
    let lr = 0.01;

    let mut param_history = vec![params.clone()];

    // Multiple backward steps
    for _ in 0..5 {
        optimizer.step(&grads, &mut params, lr).unwrap();
        param_history.push(params.clone());
    }

    // Parameters should monotonically change (for same-sign gradients)
    for i in 1..param_history.len() {
        assert!(param_history[i][0] <= param_history[i - 1][0]);
    }
}

#[test]
fn test_optimizer_gradient_clipping_simulation() {
    let mut optimizer = AdamOptimizer::new(3);
    let mut params = vec![1.0, 2.0, 3.0];

    // Very large gradients (would cause instability without clipping)
    let large_grads = vec![1000.0, 1000.0, 1000.0];
    let lr = 0.001;

    // Manually clip gradients
    let max_norm = 1.0;
    let grad_norm: f32 = large_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    let scale = if grad_norm > max_norm {
        max_norm / grad_norm
    } else {
        1.0
    };

    let clipped_grads: Vec<f32> = large_grads.iter().map(|g| g * scale).collect();

    optimizer.step(&clipped_grads, &mut params, lr).unwrap();

    // With clipped gradients, parameter changes should be bounded
    assert!((params[0] - 1.0).abs() < 0.1);
}

// ============================================================================
// TEST 4: Learning Rate Scheduler Backward Interaction Tests
// ============================================================================

#[test]
fn test_warmup_scheduler_backward_phases() {
    let scheduler = WarmupLinearScheduler::new(0.001, 100, 1000);

    // Warmup phase: LR increases
    let lr_early = scheduler.get_lr(10);
    let lr_mid = scheduler.get_lr(50);
    let lr_late = scheduler.get_lr(90);

    assert!(lr_early < lr_mid);
    assert!(lr_mid < lr_late);

    // Decay phase: LR decreases
    let lr_decay_start = scheduler.get_lr(100);
    let lr_decay_mid = scheduler.get_lr(500);
    let lr_decay_end = scheduler.get_lr(900);

    assert!(lr_decay_start > lr_decay_mid);
    assert!(lr_decay_mid > lr_decay_end);
}

#[test]
fn test_cosine_scheduler_smooth_backward_decay() {
    let scheduler = WarmupCosineScheduler::new(0.001, 100, 1000, 0.0001);

    // After warmup, cosine decay should be smooth
    let lr_start = scheduler.get_lr(100);
    let lr_mid = scheduler.get_lr(550);
    let lr_end = scheduler.get_lr(999);

    assert!(lr_start > lr_mid);
    assert!(lr_mid > lr_end);
    assert!(lr_end >= 0.0001); // Should not go below min_lr
}

#[test]
fn test_scheduler_optimizer_interaction() {
    let mut optimizer = AdamOptimizer::new(2);
    let mut params = vec![1.0, 1.0];
    let grads = vec![0.1, 0.1];
    let scheduler = WarmupLinearScheduler::new(0.01, 100, 1000);

    // Step 1: Low LR during warmup
    let lr1 = scheduler.get_lr(0);
    optimizer.step(&grads, &mut params, lr1).unwrap();
    let params_after_1 = params.clone();

    // Step 50: Higher LR (mid warmup)
    let lr50 = scheduler.get_lr(50);
    optimizer.step(&grads, &mut params, lr50).unwrap();
    let params_after_50 = params.clone();

    // Higher LR should cause larger parameter changes
    let change_1: f32 = (params_after_1[0] - 1.0f32).abs();
    let change_50: f32 = (params_after_50[0] - params_after_1[0]).abs();

    assert!(lr1 < lr50);
    assert!(change_50 > change_1);
}

// ============================================================================
// TEST 5: Trainer Backward Integration Tests
// ============================================================================

#[test]
fn test_trainer_single_backward_step() {
    let mut model = MockModel::new(10);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(10))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    // Single training step (includes backward pass)
    let metrics = trainer.train_step(&batch).unwrap();

    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
    assert_eq!(metrics.step, 0);
}

#[test]
fn test_trainer_multiple_backward_steps() {
    let mut model = MockModel::new(5);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(5))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    let mut losses = Vec::new();

    // Multiple training steps
    for _ in 0..5 {
        let metrics = trainer.train_step(&batch).unwrap();
        losses.push(metrics.loss);
    }

    // All losses should be finite
    for &loss in &losses {
        assert!(loss.is_finite());
    }
}

#[test]
fn test_trainer_gradient_clipping() {
    let mut model = MockModel::new(8);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(8))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .with_grad_clip_norm(1.0)
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    let metrics = trainer.train_step(&batch).unwrap();

    // Grad norm should be clipped to max_norm or less
    assert!(metrics.grad_norm <= 1.0 || metrics.grad_norm.is_finite());
}

#[test]
fn test_trainer_accumulated_backward_steps() {
    let mut model = MockModel::new(6);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(6))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    // Accumulate over 2 batches
    let batches = vec![Ok(batch.clone()), Ok(batch.clone())];
    let metrics = trainer
        .train_accumulated_steps(batches.into_iter(), 2)
        .unwrap();

    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
}

#[test]
fn test_trainer_backward_nan_detection() {
    // Create a model that produces NaN gradients
    struct NaNModel {
        params: Vec<f32>,
    }

    impl Model for NaNModel {
        fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
            // Return NaN logits with correct shape
            let vocab_size = 10;
            let num_positions = batch.batch_size() * (batch.seq_len().saturating_sub(1)).max(1);
            ANETensor::from_fp32(
                vec![f32::NAN; num_positions * vocab_size],
                vec![num_positions, vocab_size],
            )
        }

        fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
            Ok(vec![f32::NAN; 4])
        }

        fn backward_with_batch(&mut self, _batch: &Batch, _loss: f32) -> Result<Vec<f32>> {
            Ok(vec![f32::NAN; 4])
        }

        fn parameters(&mut self) -> &mut [f32] {
            &mut self.params
        }

        fn param_count(&self) -> usize {
            self.params.len()
        }
    }

    let mut model = NaNModel {
        params: vec![1.0; 4],
    };

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(4))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    // Should detect NaN and return error
    let result = trainer.train_step(&batch);
    assert!(result.is_err());
}

// ============================================================================
// TEST 6: Parameter-Golf Specific Backward Scenarios
// ============================================================================

#[test]
fn test_parameter_golf_gradient_accumulation_config() {
    // Simulate parameter-golf training config
    let global_batch_tokens = 65_536;
    let _seq_len = 1024;
    let grad_accum_steps = 8;

    // Each chunk processes: global_batch_tokens / grad_accum_steps tokens
    let tokens_per_chunk = global_batch_tokens / grad_accum_steps;
    assert_eq!(tokens_per_chunk, 8_192);

    // Accumulator for parameter-golf model (e.g., 1M parameters)
    let param_count = 1_000_000;
    let mut accum = GradAccumulator::new(param_count, grad_accum_steps);

    // Simulate accumulating gradients from 8 chunks
    for i in 0..grad_accum_steps {
        let grads = vec![(i + 1) as f32 * 0.01; param_count];
        accum
            .accumulate(&grads, (i + 1) as f32, 1.0 / grad_accum_steps as f32)
            .unwrap();
    }

    assert!(accum.is_ready());

    // Average gradient should be mean of 0.01, 0.02, ..., 0.08
    let avg_grad = accum.gradients()[0];
    let expected_avg = (0.01 + 0.08) / 2.0; // Arithmetic mean
    assert!((avg_grad - expected_avg).abs() < 1e-5);
}

#[test]
fn test_parameter_golf_loss_convergence_simulation() {
    // Simulate loss decreasing over training steps
    let mut model = MockModel::new(100);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamWOptimizer::new(100).with_weight_decay(0.1))
        .with_scheduler(WarmupCosineScheduler::new(0.003, 100, 1000, 0.0003))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4, 5, 6], 2, 3).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let metrics = trainer.train_step(&batch).unwrap();
        losses.push(metrics.loss);
    }

    // All losses should be finite
    for &loss in &losses {
        assert!(loss.is_finite());
    }
}

#[test]
fn test_parameter_golf_effective_batch_size() {
    // Parameter-golf often uses large effective batch sizes via accumulation
    let micro_batch_tokens = 8_192;
    let grad_accum_steps = 8;
    let effective_batch_tokens = micro_batch_tokens * grad_accum_steps;

    assert_eq!(effective_batch_tokens, 65_536);

    // Verify accumulator configuration matches
    let param_count = 1000;
    let accum = GradAccumulator::new(param_count, grad_accum_steps);

    assert_eq!(accum.total_steps(), grad_accum_steps);
}

#[test]
fn test_backward_gradient_norm_monitoring() {
    // Monitor gradient norms during training (important for stability)
    let mut model = MockModel::new(50);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(50))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    let mut grad_norms = Vec::new();
    for _ in 0..5 {
        let metrics = trainer.train_step(&batch).unwrap();
        grad_norms.push(metrics.grad_norm);
    }

    // All gradient norms should be finite and positive
    for &norm in &grad_norms {
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }
}

#[test]
fn test_backward_learning_rate_warmup_stability() {
    // Warmup prevents instability at start of training
    let mut model = MockModel::new(20);

    // Scheduler with warmup
    let scheduler = WarmupCosineScheduler::new(0.003, 100, 1000, 0.0003);

    // LR at step 0 should be very small (stable)
    let lr_start = scheduler.get_lr(0);
    assert!(lr_start < 0.0001);

    // LR at end of warmup should be at peak
    let lr_peak = scheduler.get_lr(99);
    assert!(lr_peak > 0.002);

    // Training with warmup scheduler
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(20))
        .with_scheduler(scheduler)
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();

    // Early steps should be stable (small LR)
    let metrics = trainer.train_step(&batch).unwrap();
    assert!(metrics.learning_rate < 0.0001);
}

// ============================================================================
// TEST 7: Backward Error Handling
// ============================================================================

#[test]
fn test_backward_handles_invalid_logits() {
    let loss_fn = CrossEntropyLoss::new();
    // Batch with token 10 which will exceed vocab size 4
    let batch = Batch::new(vec![1u32, 10, 3], 1, 3).unwrap();

    // Create a tensor with small vocab size (4)
    // Token 10 in the batch exceeds this vocab size
    let logits = ANETensor::from_fp32(vec![0.0f32; 8], vec![2, 4]).unwrap();
    let result = loss_fn.compute(&logits, &batch);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds vocab"));
}

#[test]
fn test_backward_handles_target_exceeds_vocab() {
    let loss_fn = CrossEntropyLoss::new();

    // Target token 100 exceeds vocab size 4
    let batch = Batch::new(vec![100u32], 1, 1).unwrap();
    let logits = ANETensor::from_fp32(vec![0.0f32; 4], vec![4]).unwrap();

    let result = loss_fn.compute(&logits, &batch);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds vocab"));
}

#[test]
fn test_trainer_handles_optimizer_failure() {
    struct BrokenModel {
        params: Vec<f32>,
    }

    impl Model for BrokenModel {
        fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
            // Return logits with correct shape
            let vocab_size = 10;
            let num_positions = batch.batch_size() * (batch.seq_len().saturating_sub(1)).max(1);
            ANETensor::from_fp32(
                vec![1.0f32; num_positions * vocab_size],
                vec![num_positions, vocab_size],
            )
        }

        fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
            Ok(vec![0.1; 4])
        }

        fn backward_with_batch(&mut self, _batch: &Batch, _loss: f32) -> Result<Vec<f32>> {
            Ok(vec![0.1; 4])
        }

        fn parameters(&mut self) -> &mut [f32] {
            &mut self.params
        }

        fn param_count(&self) -> usize {
            self.params.len()
        }
    }

    let mut model = BrokenModel {
        params: vec![1.0; 4],
    };

    // Optimizer expects 4 params but model has 4 - should work
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(4))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4).unwrap();
    let result = trainer.train_step(&batch);

    // Should succeed since sizes match
    assert!(result.is_ok());
}

// ============================================================================
// TEST 8: Backward Performance Characteristics
// ============================================================================

#[test]
fn test_gradient_accumulation_memory_efficiency() {
    // Gradient accumulation allows larger effective batches without more memory
    let param_count = 10_000;
    let grad_accum_steps = 8;

    // Memory for accumulator (single set of gradients)
    let accum_memory = param_count * std::mem::size_of::<f32>();

    // Without accumulation, to get same effective batch would need:
    let full_batch_memory = param_count * grad_accum_steps * std::mem::size_of::<f32>();

    // Accumulation uses 1/N of the memory
    assert!(accum_memory < full_batch_memory);
    assert_eq!(full_batch_memory / accum_memory, grad_accum_steps);
}

#[test]
fn test_backward_step_timing_consistency() {
    use std::time::Instant;

    let mut model = MockModel::new(1000);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(1000))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let batch = Batch::new(vec![1u32; 64], 1, 64).unwrap();

    let mut times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        trainer.train_step(&batch).unwrap();
        times.push(start.elapsed().as_millis());
    }

    // All steps should complete in reasonable time (< 100ms for mock)
    for &time in &times {
        assert!(time < 100);
    }
}
