//! Training orchestration for models

use std::fmt;
use crate::data::Batch;
use crate::error::Result;
use crate::training::model::Model;
use crate::training::loss::LossFn;
use crate::training::scheduler::LRScheduler;

/// Error type for training failures
#[derive(Debug, Clone)]
pub enum TrainerError {
    /// Model forward pass failed
    ModelForwardFailed(String),

    /// Model backward pass failed
    ModelBackwardFailed(String),

    /// Loss computation failed or returned invalid value
    LossComputationFailed(String),

    /// Invalid tensor shape from model
    InvalidLogitsShape(String),

    /// Optimizer step failed
    OptimizerStepFailed(String),

    /// NaN or Inf detected in gradients
    InvalidGradients(String),

    /// Gradient norm computation failed
    GradientNormInvalid(String),

    /// Builder missing required component
    IncompleteTrainer(String),
}

impl fmt::Display for TrainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainerError::ModelForwardFailed(msg) => write!(f, "Model forward pass failed: {}", msg),
            TrainerError::ModelBackwardFailed(msg) => write!(f, "Model backward pass failed: {}", msg),
            TrainerError::LossComputationFailed(msg) => write!(f, "Loss computation failed: {}", msg),
            TrainerError::InvalidLogitsShape(msg) => write!(f, "Invalid logits shape: {}", msg),
            TrainerError::OptimizerStepFailed(msg) => write!(f, "Optimizer step failed: {}", msg),
            TrainerError::InvalidGradients(msg) => write!(f, "Invalid gradients: {}", msg),
            TrainerError::GradientNormInvalid(msg) => write!(f, "Gradient norm invalid: {}", msg),
            TrainerError::IncompleteTrainer(msg) => write!(f, "Incomplete trainer: {}", msg),
        }
    }
}

impl std::error::Error for TrainerError {}

/// Metrics returned after each training step
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Loss value for this step
    pub loss: f32,

    /// L2 norm of gradients (indicator of gradient magnitude)
    pub grad_norm: f32,

    /// Learning rate used for this step
    pub learning_rate: f32,

    /// Training step number (0-based)
    pub step: u32,
}

impl StepMetrics {
    /// Create a new StepMetrics
    pub fn new(loss: f32, grad_norm: f32, learning_rate: f32, step: u32) -> Self {
        StepMetrics {
            loss,
            grad_norm,
            learning_rate,
            step,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_metrics_creation() {
        let metrics = StepMetrics::new(1.5, 0.5, 0.001, 0);
        assert_eq!(metrics.loss, 1.5);
        assert_eq!(metrics.grad_norm, 0.5);
        assert_eq!(metrics.learning_rate, 0.001);
        assert_eq!(metrics.step, 0);
    }

    #[test]
    fn test_trainer_error_display() {
        let err = TrainerError::ModelForwardFailed("test error".to_string());
        assert_eq!(err.to_string(), "Model forward pass failed: test error");
    }
}

/// Trait for optimizers that update model parameters based on gradients
pub trait Optimizer: Send {
    /// Perform a single optimization step
    ///
    /// # Arguments
    /// - `grads`: Gradient vector (one per parameter)
    /// - `params`: Mutable reference to model parameters
    /// - `lr`: Learning rate for this step
    ///
    /// # Errors
    /// Returns error if optimization step fails
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()>;
}

/// Adam optimizer with bias correction.
///
/// This is a simple CPU implementation intended for training examples and
/// small-to-medium model experiments.
pub struct AdamOptimizer {
    m: Vec<f32>,
    v: Vec<f32>,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer for a parameter vector of the given size.
    pub fn new(param_count: usize) -> Self {
        Self::with_hyperparams(param_count, 0.9, 0.999, 1e-8)
    }

    /// Create Adam with custom hyperparameters.
    pub fn with_hyperparams(param_count: usize, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            beta1,
            beta2,
            eps,
            step: 0,
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        use crate::Error;

        if grads.len() != params.len() || grads.len() != self.m.len() || grads.len() != self.v.len() {
            return Err(Error::Other(format!(
                "adam optimizer state mismatch: grads={}, params={}, m={}, v={}",
                grads.len(),
                params.len(),
                self.m.len(),
                self.v.len()
            )));
        }

        self.step += 1;
        let step = self.step as f32;
        let beta1_correction = 1.0 - self.beta1.powf(step);
        let beta2_correction = 1.0 - self.beta2.powf(step);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / beta1_correction.max(1e-12);
            let v_hat = self.v[i] / beta2_correction.max(1e-12);
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }

        Ok(())
    }
}

/// Builder for Trainer (ensures all required components are set)
pub struct TrainerBuilder<'a, M: Model> {
    model: &'a mut M,
    optimizer: Option<Box<dyn Optimizer>>,
    scheduler: Option<Box<dyn LRScheduler>>,
    loss_fn: Option<Box<dyn LossFn>>,
    grad_clip_norm: Option<f32>,
}

impl<'a, M: Model> TrainerBuilder<'a, M> {
    /// Create a new trainer builder
    pub fn new(model: &'a mut M) -> Self {
        TrainerBuilder {
            model,
            optimizer: None,
            scheduler: None,
            loss_fn: None,
            grad_clip_norm: None,
        }
    }

    /// Set the optimizer
    pub fn with_optimizer<O: Optimizer + 'static>(mut self, opt: O) -> Self {
        self.optimizer = Some(Box::new(opt));
        self
    }

    /// Set the learning rate scheduler
    pub fn with_scheduler<S: LRScheduler + 'static>(mut self, sch: S) -> Self {
        self.scheduler = Some(Box::new(sch));
        self
    }

    /// Set the learning rate scheduler from a boxed trait object.
    pub fn with_scheduler_box(mut self, sch: Box<dyn LRScheduler>) -> Self {
        self.scheduler = Some(sch);
        self
    }

    /// Set the loss function
    pub fn with_loss_fn<L: LossFn + 'static>(mut self, loss: L) -> Self {
        self.loss_fn = Some(Box::new(loss));
        self
    }

    /// Set an optional global gradient norm clipping threshold.
    pub fn with_grad_clip_norm(mut self, grad_clip_norm: f32) -> Self {
        self.grad_clip_norm = Some(grad_clip_norm);
        self
    }

    /// Build trainer, ensuring all components are set
    pub fn build(self) -> Result<Trainer<'a, M>> {
        let optimizer = self.optimizer
            .ok_or_else(|| crate::Error::Other(
                TrainerError::IncompleteTrainer("optimizer not set".to_string()).to_string()
            ))?;

        let scheduler = self.scheduler
            .ok_or_else(|| crate::Error::Other(
                TrainerError::IncompleteTrainer("scheduler not set".to_string()).to_string()
            ))?;

        let loss_fn = self.loss_fn
            .ok_or_else(|| crate::Error::Other(
                TrainerError::IncompleteTrainer("loss function not set".to_string()).to_string()
            ))?;

        Ok(Trainer {
            model: self.model,
            optimizer,
            scheduler,
            loss_fn,
            grad_clip_norm: self.grad_clip_norm,
            current_step: 0,
        })
    }
}

/// Orchestrates training: forward → loss → backward → optimize
pub struct Trainer<'a, M: Model> {
    model: &'a mut M,
    optimizer: Box<dyn Optimizer>,
    scheduler: Box<dyn LRScheduler>,
    loss_fn: Box<dyn LossFn>,
    grad_clip_norm: Option<f32>,
    current_step: u32,
}

impl<'a, M: Model> Trainer<'a, M> {
    /// Single training step
    pub fn train_step(&mut self, batch: &Batch) -> Result<StepMetrics> {
        // 1. Forward: logits = model.forward(batch)
        let logits = self.model.forward(batch)
            .map_err(|_| crate::Error::Other(
                TrainerError::ModelForwardFailed("forward pass failed".to_string()).to_string()
            ))?;

        // 2. Loss: loss = loss_fn.compute(&logits, batch)
        let loss = self.loss_fn.compute(&logits, batch)
            .map_err(|_| crate::Error::Other(
                TrainerError::LossComputationFailed("loss computation failed".to_string()).to_string()
            ))?;

        // 3. Backward: grads = model.backward_with_batch(batch, loss)
        let grads = self.model.backward_with_batch(batch, loss)
            .map_err(|_| crate::Error::Other(
                TrainerError::ModelBackwardFailed("backward pass failed".to_string()).to_string()
            ))?;

        // Verify gradient vector length matches parameter count
        if grads.len() != self.model.param_count() {
            return Err(crate::Error::Other(
                TrainerError::InvalidGradients(
                    format!("gradient count {} != param count {}",
                        grads.len(), self.model.param_count())
                ).to_string()
            ));
        }

        // 4. Metrics: grad_norm = compute_norm(&grads)
        let grad_norm = compute_l2_norm(&grads);

        // Check for NaN/Inf in gradients
        if !grad_norm.is_finite() {
            return Err(crate::Error::Other(
                TrainerError::InvalidGradients(format!("grad_norm is {}", grad_norm)).to_string()
            ));
        }

        for (i, &g) in grads.iter().enumerate() {
            if !g.is_finite() {
                return Err(crate::Error::Other(
                    TrainerError::InvalidGradients(
                        format!("gradient[{}] is {}", i, g)
                    ).to_string()
                ));
            }
        }

        let mut clipped_grads = grads;
        if let Some(max_norm) = self.grad_clip_norm {
            if max_norm > 0.0 && grad_norm > max_norm {
                let scale = max_norm / grad_norm;
                for g in &mut clipped_grads {
                    *g *= scale;
                }
            }
        }

        // 5. LR: lr = scheduler.get_lr(current_step)
        let learning_rate = self.scheduler.get_lr(self.current_step);

        // 6. Optimize: optimizer.step(&grads, model.parameters(), learning_rate)
        self.optimizer.step(&clipped_grads, self.model.parameters(), learning_rate)
            .map_err(|_| crate::Error::Other(
                TrainerError::OptimizerStepFailed("optimizer step failed".to_string()).to_string()
            ))?;

        // 7. Increment: current_step += 1
        self.current_step += 1;

        // 8. Return: StepMetrics
        Ok(StepMetrics::new(loss, grad_norm, learning_rate, self.current_step - 1))
    }

    /// Train with explicit gradient accumulation over chunks
    ///
    /// # Arguments
    /// - `chunks`: Iterator yielding Result<Batch> chunks
    /// - `accumulation_steps`: Number of backward passes before optimizer step
    ///
    /// # Returns
    /// StepMetrics with aggregated loss and grad norm
    ///
    /// # Errors
    /// Returns an error if:
    /// - accumulation_steps is 0
    /// - Forward/backward pass fails
    /// - Chunk count doesn't match accumulation_steps
    /// - Gradients are NaN/Inf
    pub fn train_accumulated_steps<I>(
        &mut self,
        chunks: I,
        accumulation_steps: usize,
    ) -> Result<StepMetrics>
    where
        I: IntoIterator<Item = Result<Batch>>,
    {
        use crate::training::GradAccumulator;

        if accumulation_steps == 0 {
            return Err(crate::Error::Other(
                "accumulation_steps must be > 0".to_string()
            ));
        }

        let mut accum = GradAccumulator::new(self.model.param_count(), accumulation_steps);
        let scale = 1.0 / accumulation_steps as f32;
        let mut chunk_count = 0usize;

        // Process each chunk
        for chunk_result in chunks {
            let chunk = chunk_result?;

            // Forward pass
            let logits = self.model.forward(&chunk)
                .map_err(|_| crate::Error::Other(
                    TrainerError::ModelForwardFailed("forward pass failed".to_string()).to_string()
                ))?;

            // Compute loss
            let loss = self.loss_fn.compute(&logits, &chunk)
                .map_err(|_| crate::Error::Other(
                    TrainerError::LossComputationFailed("loss computation failed".to_string()).to_string()
                ))?;

            // Backward pass
            let grads = self.model.backward_with_batch(&chunk, loss)
                .map_err(|_| crate::Error::Other(
                    TrainerError::ModelBackwardFailed("backward pass failed".to_string()).to_string()
                ))?;

            // Validate gradient count
            if grads.len() != self.model.param_count() {
                return Err(crate::Error::Other(
                    TrainerError::InvalidGradients(
                        format!("gradient count {} != param count {}",
                            grads.len(), self.model.param_count())
                    ).to_string()
                ));
            }

            // Check for NaN/Inf in gradients
            for (i, &g) in grads.iter().enumerate() {
                if !g.is_finite() {
                    return Err(crate::Error::Other(
                        TrainerError::InvalidGradients(
                            format!("gradient[{}] is {}", i, g)
                        ).to_string()
                    ));
                }
            }

            // Accumulate gradients
            accum.accumulate(&grads, loss, scale)?;
            chunk_count += 1;
        }

        // Verify we got the expected number of chunks
        if chunk_count != accumulation_steps {
            return Err(crate::Error::Other(
                format!("expected {} chunks, got {}", accumulation_steps, chunk_count)
            ));
        }

        let mut clipped_grads = accum.gradients().to_vec();
        let grad_norm = compute_l2_norm(&clipped_grads);
        if let Some(max_norm) = self.grad_clip_norm {
            if max_norm > 0.0 && grad_norm > max_norm {
                let scale = max_norm / grad_norm;
                for g in &mut clipped_grads {
                    *g *= scale;
                }
            }
        }

        // Apply accumulated gradients
        let learning_rate = self.scheduler.get_lr(self.current_step);
        self.optimizer.step(&clipped_grads, self.model.parameters(), learning_rate)
            .map_err(|_| crate::Error::Other(
                TrainerError::OptimizerStepFailed("optimizer step failed".to_string()).to_string()
            ))?;

        self.current_step += 1;

        // Return aggregated metrics
        Ok(StepMetrics::new(
            accum.average_loss(),
            grad_norm,
            learning_rate,
            self.current_step - 1,
        ))
    }
}

/// Compute L2 norm of a gradient vector
fn compute_l2_norm(grads: &[f32]) -> f32 {
    grads.iter().map(|g| g * g).sum::<f32>().sqrt()
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use crate::wrapper::ANETensor;

    // Mock types for testing
    struct MockModel {
        params: Vec<f32>,
    }

    impl Model for MockModel {
        fn forward(&mut self, _batch: &Batch) -> Result<ANETensor> {
            Err(crate::Error::NotImplemented("not implemented".to_string()))
        }

        fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
            Ok(vec![0.1, 0.2])
        }

        fn parameters(&mut self) -> &mut [f32] {
            &mut self.params
        }

        fn param_count(&self) -> usize {
            self.params.len()
        }
    }

    /// Minimal optimizer for testing
    struct SimpleOptimizer {
        _lr: f32,
    }

    impl SimpleOptimizer {
        fn new(lr: f32) -> Self {
            SimpleOptimizer { _lr: lr }
        }
    }

    impl Optimizer for SimpleOptimizer {
        fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
            for (param, grad) in params.iter_mut().zip(grads.iter()) {
                *param -= lr * grad;
            }
            Ok(())
        }
    }

    #[test]
    fn test_builder_construction() {
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let builder = TrainerBuilder::new(&mut model);

        // Should fail - missing optimizer
        assert!(builder.build().is_err());
    }

    #[test]
    fn test_builder_missing_component() {
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let builder = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001));

        // Should fail - missing scheduler and loss_fn
        let result = builder.build();
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod trainer_tests {
    use super::*;

    #[test]
    fn test_compute_l2_norm() {
        let grads = vec![3.0, 4.0]; // 3-4-5 triangle
        let norm = compute_l2_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_l2_norm_empty() {
        let grads: Vec<f32> = vec![];
        let norm = compute_l2_norm(&grads);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_compute_l2_norm_single() {
        let grads = vec![7.0];
        let norm = compute_l2_norm(&grads);
        assert!((norm - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_trainer_error_variants() {
        let errors = vec![
            TrainerError::ModelForwardFailed("test".to_string()),
            TrainerError::ModelBackwardFailed("test".to_string()),
            TrainerError::LossComputationFailed("test".to_string()),
            TrainerError::InvalidLogitsShape("test".to_string()),
            TrainerError::OptimizerStepFailed("test".to_string()),
            TrainerError::InvalidGradients("test".to_string()),
            TrainerError::GradientNormInvalid("test".to_string()),
            TrainerError::IncompleteTrainer("test".to_string()),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_step_metrics_properties() {
        let metrics = StepMetrics::new(2.0, 1.5, 0.001, 5);
        assert_eq!(metrics.loss, 2.0);
        assert_eq!(metrics.grad_norm, 1.5);
        assert_eq!(metrics.learning_rate, 0.001);
        assert_eq!(metrics.step, 5);
    }
}


#[cfg(test)]
mod accumulated_steps_tests {
    use super::*;
    use crate::wrapper::ANETensor;

    // Mock types for accumulated steps testing
    struct MockModel {
        params: Vec<f32>,
    }

    impl Model for MockModel {
        fn forward(&mut self, _batch: &Batch) -> Result<ANETensor> {
            // Return a dummy tensor (in tests this isn't actually used)
            ANETensor::from_fp32(vec![1.0f32; 256], vec![256])
        }

        fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
            Ok(vec![0.1, 0.2])
        }

        fn parameters(&mut self) -> &mut [f32] {
            &mut self.params
        }

        fn param_count(&self) -> usize {
            self.params.len()
        }
    }

    /// Minimal optimizer for testing
    struct SimpleOptimizer {
        _lr: f32,
    }

    impl SimpleOptimizer {
        fn new(lr: f32) -> Self {
            SimpleOptimizer { _lr: lr }
        }
    }

    impl Optimizer for SimpleOptimizer {
        fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
            for (param, grad) in params.iter_mut().zip(grads.iter()) {
                *param -= lr * grad;
            }
            Ok(())
        }
    }

    #[test]
    fn test_train_accumulated_steps_basic() -> Result<()> {
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        // Try to call train_accumulated_steps
        let chunks = vec![Ok(batch)];
        let _metrics = trainer.train_accumulated_steps(chunks.into_iter(), 1)?;

        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_single_batch() -> Result<()> {
        // Test with 1 batch, 1 accumulation step (should just call train_step equivalent)
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let metrics = trainer.train_accumulated_steps(
            vec![Ok(batch)].into_iter(),
            1
        )?;

        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());
        assert!(metrics.learning_rate > 0.0);
        assert_eq!(metrics.step, 0);
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_multiple_batches() -> Result<()> {
        // Test accumulating over 2 batches, then 1 optimizer step
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let batches = vec![
            Ok(batch.clone()),
            Ok(batch.clone()),
        ];

        let metrics = trainer.train_accumulated_steps(
            batches.into_iter(),
            2  // Accumulate every 2 batches
        )?;

        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());
        assert_eq!(metrics.step, 0);
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_exact_multiple() -> Result<()> {
        // Test with exact multiple: 4 batches, 2 per step = 2 steps
        // For now, we test a single 4-batch accumulation
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let batches = (0..4)
            .map(|_| Ok(batch.clone()))
            .collect::<Vec<_>>();

        let metrics = trainer.train_accumulated_steps(
            batches.into_iter(),
            4  // 4 batches in one accumulation step
        )?;

        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());
        assert_eq!(metrics.step, 0);  // First step (0-indexed)
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_invalid_accumulation() -> Result<()> {
        // Test with 0 accumulation steps (should error)
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let result = trainer.train_accumulated_steps(
            vec![Ok(batch)].into_iter(),
            0  // Invalid: 0 accumulation steps
        );

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_mismatched_count() -> Result<()> {
        // Test with fewer batches than accumulation_steps (should error)
        let mut model = MockModel { params: vec![1.0, 2.0] };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let result = trainer.train_accumulated_steps(
            vec![Ok(batch)].into_iter(),
            2  // Expect 2 batches, but only provide 1
        );

        assert!(result.is_err());
        Ok(())
    }
}
