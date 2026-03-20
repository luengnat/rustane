//! Training orchestration for models
//!
//! This module provides the `Trainer` and related types for orchestrating
//! the complete training loop: forward pass → loss computation → backward pass → optimizer step.
//!
//! # Example
//!
//! ```no_run
//! use rustane::training::{
//!     Model, TrainerBuilder, AdamOptimizer, WarmupCosineScheduler, CrossEntropyLoss,
//! };
//! use rustane::data::Batch;
//!
//! # struct MyModel;
//! # impl Model for MyModel {
//! #     fn forward(&mut self, _: &Batch) -> rustane::error::Result<rustane::wrapper::ANETensor> { todo!() }
//! #     fn backward(&mut self, _: f32) -> rustane::error::Result<Vec<f32>> { todo!() }
//! #     fn backward_with_batch(&mut self, _: &Batch, _: f32) -> rustane::error::Result<Vec<f32>> { todo!() }
//! #     fn parameters(&mut self) -> &mut [f32] { &mut [] }
//! #     fn param_count(&self) -> usize { 0 }
//! # }
//! # let mut model = MyModel;
//! # let dataloader = vec![Batch::new(vec![0u32], 1, 1).unwrap()];
//! // Create model, optimizer, scheduler, and loss function
//! let optimizer = AdamOptimizer::new(model.param_count());
//! let scheduler = WarmupCosineScheduler::new(0.001, 1000, 10000, 1e-5);
//!
//! // Build trainer with all components
//! let mut trainer = TrainerBuilder::new(&mut model)
//!     .with_optimizer(optimizer)
//!     .with_scheduler(scheduler)
//!     .with_loss_fn(CrossEntropyLoss)
//!     .with_grad_clip_norm(1.0)
//!     .build()
//!     .unwrap();
//!
//! // Training loop
//! for epoch in 0..3 {
//!     for batch in &dataloader {
//!         match trainer.step(batch) {
//!             Ok(metrics) => {
//!                 println!("Step {}: loss={:.4}, grad_norm={:.4}",
//!                     metrics.step, metrics.loss, metrics.grad_norm);
//!             }
//!             Err(e) => {
//!                 eprintln!("Training error: {}", e);
//!                 break;
//!             }
//!         }
//!     }
//! }
//! ```

use crate::data::Batch;
use crate::error::Result;
use crate::training::loss::LossFn;
use crate::training::model::Model;
use crate::training::scheduler::LRScheduler;
use std::fmt;

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
            TrainerError::ModelForwardFailed(msg) => {
                write!(f, "Model forward pass failed: {}", msg)
            }
            TrainerError::ModelBackwardFailed(msg) => {
                write!(f, "Model backward pass failed: {}", msg)
            }
            TrainerError::LossComputationFailed(msg) => {
                write!(f, "Loss computation failed: {}", msg)
            }
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
///
/// Provides comprehensive information about the training state for monitoring
/// and debugging.
///
/// # Fields
///
/// - **loss**: Scalar loss value for this batch (lower is better)
/// - **grad_norm**: L2 norm of gradients (indicator of gradient magnitude)
/// - **learning_rate**: Learning rate used for this step (may vary with scheduler)
/// - **step**: Training step number (0-based, increments each call)
///
/// # Interpreting Metrics
///
/// - **Loss**: Should decrease over time. Spikes may indicate learning rate issues.
/// - **Grad norm**: Typical values are 0.1-10. Very large values (>100) may indicate
///   exploding gradients. Very small values (<0.001) may indicate vanishing gradients.
/// - **Learning rate**: Follows the scheduler pattern (warmup → decay).
///
/// # Example
///
/// ```
/// use rustane::training::StepMetrics;
///
/// let metrics = StepMetrics::new(2.5, 0.8, 0.001, 100);
/// println!("Loss: {:.4}, Grad norm: {:.4}", metrics.loss, metrics.grad_norm);
/// ```
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
    ///
    /// # Arguments
    ///
    /// * `loss`: Scalar loss value
    /// * `grad_norm`: L2 norm of the gradient vector
    /// * `learning_rate`: Learning rate used for this step
    /// * `step`: Current training step number
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
///
/// Optimizers implement the parameter update rule: θ = θ - lr * f(gradients, state)
///
/// # Implementing an Optimizer
///
/// Custom optimizers must implement the [`Optimizer::step`] method, which:
/// 1. Takes gradients and current parameters as input
/// 2. Updates parameters in-place using the optimizer's state
/// 3. Returns an error if the update fails
///
/// # Example
///
/// ```no_run
/// use rustane::training::Optimizer;
/// use rustane::error::Result;
///
/// struct SGDMomentum {
///     velocity: Vec<f32>,
///     momentum: f32,
/// }
///
/// impl Optimizer for SGDMomentum {
///     fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
///         for i in 0..params.len() {
///             self.velocity[i] = self.momentum * self.velocity[i] + grads[i];
///             params[i] -= lr * self.velocity[i];
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait Optimizer: Send {
    /// Perform a single optimization step
    ///
    /// Updates parameters in-place using gradients and the optimizer's internal state.
    ///
    /// # Arguments
    ///
    /// * `grads`: Gradient vector (one gradient per parameter)
    /// * `params`: Mutable reference to model parameters (updated in-place)
    /// * `lr`: Learning rate for this step
    ///
    /// # Errors
    ///
    /// Returns error if optimization step fails (e.g., size mismatch, NaN/Inf)
    ///
    /// # Notes
    ///
    /// - The gradients and parameters slices must have the same length
    /// - Parameters are modified in-place; no copy is returned
    /// - The optimizer should update its internal state (momentum, statistics, etc.)
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()>;
}

/// Adam optimizer with bias correction
///
/// Implements the Adam optimization algorithm as described in [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980).
///
/// Adam combines ideas from RMSProp and momentum:
/// - Maintains moving averages of gradients (m) and squared gradients (v)
/// - Applies bias correction to account for initialization at zero
/// - Adapts learning rates for each parameter individually
///
/// # Hyperparameters
///
/// - **beta1** (default: 0.9): Exponential decay rate for first moment estimates
/// - **beta2** (default: 0.999): Exponential decay rate for second moment estimates
/// - **eps** (default: 1e-8): Small constant for numerical stability
///
/// # Example
///
/// ```no_run
/// use rustane::training::AdamOptimizer;
///
/// // Create with defaults (beta1=0.9, beta2=0.999, eps=1e-8)
/// let optimizer = AdamOptimizer::new(1000);
///
/// // Create with custom hyperparameters
/// let optimizer = AdamOptimizer::with_hyperparams(1000, 0.99, 0.9999, 1e-7);
/// ```
///
/// # Performance Notes
///
/// This is a simple CPU implementation intended for training examples and
/// small-to-medium model experiments. For large-scale training, consider
/// GPU-accelerated optimizers or framework-specific implementations.
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

/// AdamW optimizer with decoupled weight decay
///
/// AdamW modifies Adam by applying weight decay directly to parameters
/// rather than adding it to the gradient. This is the recommended approach
/// for training transformers and has been shown to improve generalization.
///
/// # Difference from Adam
///
/// - **Adam**: `grad = grad + wd * param` (L2 regularization in gradient)
/// - **AdamW**: `param = param - lr * wd * param` (direct weight decay on parameter)
///
/// # References
///
/// - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
pub struct AdamWOptimizer {
    m: Vec<f32>,
    v: Vec<f32>,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer for a parameter vector of the given size.
    ///
    /// # Arguments
    ///
    /// * `param_count` - Number of parameters to optimize
    ///
    /// Uses default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8, wd=0.01
    pub fn new(param_count: usize) -> Self {
        Self::with_hyperparams(param_count, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Create AdamW with custom hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `param_count` - Number of parameters to optimize
    /// * `beta1` - Exponential decay rate for first moment estimate (default: 0.9)
    /// * `beta2` - Exponential decay rate for second moment estimate (default: 0.999)
    /// * `eps` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    pub fn with_hyperparams(
        param_count: usize,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
        }
    }

    /// Get the current weight decay coefficient
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set the weight decay coefficient
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        use crate::Error;

        if grads.len() != params.len() || grads.len() != self.m.len() || grads.len() != self.v.len()
        {
            return Err(Error::Other(format!(
                "adamw optimizer state mismatch: grads={}, params={}, m={}, v={}",
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
            // Update moments (same as Adam)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / beta1_correction.max(1e-12);
            let v_hat = self.v[i] / beta2_correction.max(1e-12);

            // Apply Adam update with decoupled weight decay
            // param = param - lr * (m_hat / sqrt(v_hat) + eps) - lr * wd * param
            let adam_step = lr * m_hat / (v_hat.sqrt() + self.eps);
            let wd_step = lr * self.weight_decay * params[i];
            params[i] -= adam_step + wd_step;
        }

        Ok(())
    }
}

/// Lion optimizer with sign-based updates
///
/// Lion (Symbolic Optimizer) is a simpler optimizer that only uses the sign of gradients
/// rather than their magnitude. It maintains a single momentum vector and applies updates
/// based on the sign of the momentum.
///
/// # Difference from Adam
///
/// - **Adam**: Uses gradient magnitude with adaptive learning rates (m / sqrt(v))
/// - **Lion**: Uses only gradient sign: `update = lr * sign(momentum) + wd * param`
///
/// # Advantages
///
/// - Less memory (single momentum vector vs two for Adam)
/// - Often better generalization
/// - More stable training for large models
///
/// # References
///
/// - [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)
pub struct LionOptimizer {
    m: Vec<f32>,
    beta1: f32,
    weight_decay: f32,
    step: usize,
}

impl LionOptimizer {
    /// Create a new Lion optimizer for a parameter vector of the given size.
    ///
    /// # Arguments
    ///
    /// * `param_count` - Number of parameters to optimize
    ///
    /// Uses default hyperparameters: β₁=0.9, wd=0.01
    pub fn new(param_count: usize) -> Self {
        Self::with_hyperparams(param_count, 0.9, 0.01)
    }

    /// Create Lion with custom hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `param_count` - Number of parameters to optimize
    /// * `beta1` - Exponential decay rate for momentum (default: 0.9)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    pub fn with_hyperparams(param_count: usize, beta1: f32, weight_decay: f32) -> Self {
        Self {
            m: vec![0.0; param_count],
            beta1,
            weight_decay,
            step: 0,
        }
    }

    /// Get the current momentum coefficient
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get the current weight decay coefficient
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set the weight decay coefficient
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the momentum coefficient
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Compute sign of a float value
    #[inline]
    fn sign(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

impl Optimizer for LionOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        use crate::Error;

        if grads.len() != params.len() || grads.len() != self.m.len() {
            return Err(Error::Other(format!(
                "lion optimizer state mismatch: grads={}, params={}, m={}",
                grads.len(),
                params.len(),
                self.m.len()
            )));
        }

        self.step += 1;

        for i in 0..params.len() {
            let g = grads[i];

            // Update momentum: m = β₁ * m + (1 - β₁) * g
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Lion update: param = param - lr * sign(m) - lr * wd * param
            let update = lr * Self::sign(self.m[i]) + lr * self.weight_decay * params[i];
            params[i] -= update;
        }

        Ok(())
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        use crate::Error;

        if grads.len() != params.len() || grads.len() != self.m.len() || grads.len() != self.v.len()
        {
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

/// Builder for [`Trainer`] (ensures all required components are set)
///
/// The builder pattern prevents incomplete trainer configuration by requiring
/// all mandatory components (optimizer, scheduler, loss_fn) before building.
///
/// # Required Components
///
/// - **Optimizer**: Updates parameters based on gradients (e.g., [`AdamOptimizer`])
/// - **Scheduler**: Provides learning rate for each step (e.g., [`WarmupCosineScheduler`])
/// - **Loss function**: Computes loss from model outputs (e.g., [`CrossEntropyLoss`])
///
/// # Optional Components
///
/// - **Gradient clipping**: Caps gradient norm to prevent exploding gradients
///
/// # Example
///
/// ```no_run
/// use rustane::training::{
///     Model, TrainerBuilder, AdamOptimizer, WarmupCosineScheduler, CrossEntropyLoss,
/// };
/// # use rustane::data::Batch;
/// # struct MyModel;
/// # impl Model for MyModel {
/// #     fn forward(&mut self, _: &Batch) -> rustane::error::Result<rustane::wrapper::ANETensor> { todo!() }
/// #     fn backward(&mut self, _: f32) -> rustane::error::Result<Vec<f32>> { todo!() }
/// #     fn backward_with_batch(&mut self, _: &Batch, _: f32) -> rustane::error::Result<Vec<f32>> { todo!() }
/// #     fn parameters(&mut self) -> &mut [f32] { &mut [] }
/// #     fn param_count(&self) -> usize { 0 }
/// # }
/// # let mut model = MyModel;
///
/// // Basic trainer
/// let mut trainer = TrainerBuilder::new(&mut model)
///     .with_optimizer(AdamOptimizer::new(1000))
///     .with_scheduler(WarmupCosineScheduler::new(0.001, 1000, 10000, 1e-5))
///     .with_loss_fn(CrossEntropyLoss)
///     .build()
///     .unwrap();
///
/// // Trainer with gradient clipping (prevents exploding gradients)
/// let mut trainer = TrainerBuilder::new(&mut model)
///     .with_optimizer(AdamOptimizer::new(1000))
///     .with_scheduler(WarmupCosineScheduler::new(0.001, 1000, 10000, 1e-5))
///     .with_loss_fn(CrossEntropyLoss)
///     .with_grad_clip_norm(1.0)  // Clip gradients at L2 norm of 1.0
///     .build()
///     .unwrap();
/// ```
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
        let optimizer = self.optimizer.ok_or_else(|| {
            crate::Error::Other(
                TrainerError::IncompleteTrainer("optimizer not set".to_string()).to_string(),
            )
        })?;

        let scheduler = self.scheduler.ok_or_else(|| {
            crate::Error::Other(
                TrainerError::IncompleteTrainer("scheduler not set".to_string()).to_string(),
            )
        })?;

        let loss_fn = self.loss_fn.ok_or_else(|| {
            crate::Error::Other(
                TrainerError::IncompleteTrainer("loss function not set".to_string()).to_string(),
            )
        })?;

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

/// Orchestrates the complete training loop
///
/// `Trainer` manages the training process by executing four steps for each batch:
/// 1. **Forward pass**: Computes model outputs from inputs
/// 2. **Loss computation**: Calculates loss from outputs and targets
/// 3. **Backward pass**: Computes gradients via backpropagation
/// 4. **Optimizer step**: Updates parameters using gradients
///
/// # Usage
///
/// ```no_run
/// use rustane::training::{TrainerBuilder, AdamOptimizer, WarmupCosineScheduler, CrossEntropyLoss};
/// # use rustane::training::Model;
/// # use rustane::data::Batch;
/// # use rustane::wrapper::ANETensor;
/// # use rustane::error::Result;
/// # struct MyModel;
/// # impl Model for MyModel {
/// #     fn forward(&mut self, _: &Batch) -> Result<ANETensor> { todo!() }
/// #     fn backward(&mut self, _: f32) -> Result<Vec<f32>> { todo!() }
/// #     fn backward_with_batch(&mut self, _: &Batch, _: f32) -> Result<Vec<f32>> { todo!() }
/// #     fn parameters(&mut self) -> &mut [f32] { &mut [] }
/// #     fn param_count(&self) -> usize { 0 }
/// # }
/// # let mut model = MyModel;
///
/// // Build trainer
/// let mut trainer = TrainerBuilder::new(&mut model)
///     .with_optimizer(AdamOptimizer::new(1000))
///     .with_scheduler(WarmupCosineScheduler::new(0.001, 1000, 10000, 1e-5))
///     .with_loss_fn(CrossEntropyLoss)
///     .build()
///     .unwrap();
///
/// // Training loop
/// for batch in dataloader {
///     match trainer.step(&batch) {
///         Ok(metrics) => {
///             if metrics.step % 100 == 0 {
///                 println!("Step {}: loss={:.4}", metrics.step, metrics.loss);
///             }
///         }
///         Err(e) => eprintln!("Error: {}", e),
///     }
/// }
/// ```
///
/// # Gradient Clipping
///
/// If configured with a gradient norm threshold, the trainer will clip gradients
/// before the optimizer step. This prevents exploding gradients and improves stability.
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
        let logits = self.model.forward(batch).map_err(|_| {
            crate::Error::Other(
                TrainerError::ModelForwardFailed("forward pass failed".to_string()).to_string(),
            )
        })?;

        // 2. Loss: loss = loss_fn.compute(&logits, batch)
        let loss = self.loss_fn.compute(&logits, batch).map_err(|_| {
            crate::Error::Other(
                TrainerError::LossComputationFailed("loss computation failed".to_string())
                    .to_string(),
            )
        })?;

        // 3. Backward: grads = model.backward_with_batch(batch, loss)
        let grads = self.model.backward_with_batch(batch, loss).map_err(|_| {
            crate::Error::Other(
                TrainerError::ModelBackwardFailed("backward pass failed".to_string()).to_string(),
            )
        })?;

        // Verify gradient vector length matches parameter count
        if grads.len() != self.model.param_count() {
            return Err(crate::Error::Other(
                TrainerError::InvalidGradients(format!(
                    "gradient count {} != param count {}",
                    grads.len(),
                    self.model.param_count()
                ))
                .to_string(),
            ));
        }

        // 4. Metrics: grad_norm = compute_norm(&grads)
        let grad_norm = compute_l2_norm(&grads);

        // Check for NaN/Inf in gradients
        if !grad_norm.is_finite() {
            return Err(crate::Error::Other(
                TrainerError::InvalidGradients(format!("grad_norm is {}", grad_norm)).to_string(),
            ));
        }

        for (i, &g) in grads.iter().enumerate() {
            if !g.is_finite() {
                return Err(crate::Error::Other(
                    TrainerError::InvalidGradients(format!("gradient[{}] is {}", i, g)).to_string(),
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
        self.optimizer
            .step(&clipped_grads, self.model.parameters(), learning_rate)
            .map_err(|_| {
                crate::Error::Other(
                    TrainerError::OptimizerStepFailed("optimizer step failed".to_string())
                        .to_string(),
                )
            })?;

        // 7. Increment: current_step += 1
        self.current_step += 1;

        // 8. Return: StepMetrics
        Ok(StepMetrics::new(
            loss,
            grad_norm,
            learning_rate,
            self.current_step - 1,
        ))
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
                "accumulation_steps must be > 0".to_string(),
            ));
        }

        let mut accum = GradAccumulator::new(self.model.param_count(), accumulation_steps);
        let scale = 1.0 / accumulation_steps as f32;
        let mut chunk_count = 0usize;

        // Process each chunk
        for chunk_result in chunks {
            let chunk = chunk_result?;

            // Forward pass
            let logits = self.model.forward(&chunk).map_err(|_| {
                crate::Error::Other(
                    TrainerError::ModelForwardFailed("forward pass failed".to_string()).to_string(),
                )
            })?;

            // Compute loss
            let loss = self.loss_fn.compute(&logits, &chunk).map_err(|_| {
                crate::Error::Other(
                    TrainerError::LossComputationFailed("loss computation failed".to_string())
                        .to_string(),
                )
            })?;

            // Backward pass
            let grads = self.model.backward_with_batch(&chunk, loss).map_err(|_| {
                crate::Error::Other(
                    TrainerError::ModelBackwardFailed("backward pass failed".to_string())
                        .to_string(),
                )
            })?;

            // Validate gradient count
            if grads.len() != self.model.param_count() {
                return Err(crate::Error::Other(
                    TrainerError::InvalidGradients(format!(
                        "gradient count {} != param count {}",
                        grads.len(),
                        self.model.param_count()
                    ))
                    .to_string(),
                ));
            }

            // Check for NaN/Inf in gradients
            for (i, &g) in grads.iter().enumerate() {
                if !g.is_finite() {
                    return Err(crate::Error::Other(
                        TrainerError::InvalidGradients(format!("gradient[{}] is {}", i, g))
                            .to_string(),
                    ));
                }
            }

            // Accumulate gradients
            accum.accumulate(&grads, loss, scale)?;
            chunk_count += 1;
        }

        // Verify we got the expected number of chunks
        if chunk_count != accumulation_steps {
            return Err(crate::Error::Other(format!(
                "expected {} chunks, got {}",
                accumulation_steps, chunk_count
            )));
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
        self.optimizer
            .step(&clipped_grads, self.model.parameters(), learning_rate)
            .map_err(|_| {
                crate::Error::Other(
                    TrainerError::OptimizerStepFailed("optimizer step failed".to_string())
                        .to_string(),
                )
            })?;

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
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let builder = TrainerBuilder::new(&mut model);

        // Should fail - missing optimizer
        assert!(builder.build().is_err());
    }

    #[test]
    fn test_builder_missing_component() {
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let builder = TrainerBuilder::new(&mut model).with_optimizer(SimpleOptimizer::new(0.001));

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
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
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
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let metrics = trainer.train_accumulated_steps(vec![Ok(batch)].into_iter(), 1)?;

        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());
        assert!(metrics.learning_rate > 0.0);
        assert_eq!(metrics.step, 0);
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_multiple_batches() -> Result<()> {
        // Test accumulating over 2 batches, then 1 optimizer step
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let batches = vec![Ok(batch.clone()), Ok(batch.clone())];

        let metrics = trainer.train_accumulated_steps(
            batches.into_iter(),
            2, // Accumulate every 2 batches
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
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let batches = (0..4).map(|_| Ok(batch.clone())).collect::<Vec<_>>();

        let metrics = trainer.train_accumulated_steps(
            batches.into_iter(),
            4, // 4 batches in one accumulation step
        )?;

        assert!(metrics.loss.is_finite());
        assert!(metrics.grad_norm.is_finite());
        assert_eq!(metrics.step, 0); // First step (0-indexed)
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_invalid_accumulation() -> Result<()> {
        // Test with 0 accumulation steps (should error)
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let result = trainer.train_accumulated_steps(
            vec![Ok(batch)].into_iter(),
            0, // Invalid: 0 accumulation steps
        );

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_train_accumulated_steps_mismatched_count() -> Result<()> {
        // Test with fewer batches than accumulation_steps (should error)
        let mut model = MockModel {
            params: vec![1.0, 2.0],
        };
        let batch = Batch::new(vec![1u32, 2, 3, 4], 1, 4)?;

        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(crate::training::ConstantScheduler::new(0.001))
            .with_loss_fn(crate::training::CrossEntropyLoss::new())
            .build()?;

        let result = trainer.train_accumulated_steps(
            vec![Ok(batch)].into_iter(),
            2, // Expect 2 batches, but only provide 1
        );

        assert!(result.is_err());
        Ok(())
    }
}

#[cfg(test)]
mod optimizer_tests {
    use super::*;

    #[test]
    fn test_adam_optimizer_creation() {
        let opt = AdamOptimizer::new(100);
        assert_eq!(opt.m.len(), 100);
        assert_eq!(opt.v.len(), 100);
        assert_eq!(opt.step, 0);
    }

    #[test]
    fn test_adam_optimizer_custom_hyperparams() {
        let opt = AdamOptimizer::with_hyperparams(50, 0.99, 0.9999, 1e-7);
        assert_eq!(opt.m.len(), 50);
        assert_eq!(opt.v.len(), 50);
        assert_eq!(opt.beta1, 0.99);
        assert_eq!(opt.beta2, 0.9999);
        assert_eq!(opt.eps, 1e-7);
    }

    #[test]
    fn test_adam_optimizer_step() -> Result<()> {
        let mut opt = AdamOptimizer::new(3);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let lr = 0.001;

        opt.step(&grads, &mut params, lr)?;

        // Parameters should have changed
        assert_ne!(params[0], 1.0);
        assert_ne!(params[1], 2.0);
        assert_ne!(params[2], 3.0);

        // Moments should have been updated
        assert!(opt.m.iter().any(|&x| x != 0.0));
        assert!(opt.v.iter().any(|&x| x != 0.0));
        assert_eq!(opt.step, 1);

        Ok(())
    }

    #[test]
    fn test_adam_optimizer_multiple_steps() -> Result<()> {
        let mut opt = AdamOptimizer::new(2);
        let mut params = vec![1.0, 1.0];
        let grads = vec![0.1, 0.1];
        let lr = 0.01;

        // First step
        opt.step(&grads, &mut params, lr)?;
        let params_after_step1 = params.clone();

        // Second step with same gradients
        opt.step(&grads, &mut params, lr)?;

        // Parameters should continue to change (due to bias correction)
        assert_ne!(params, params_after_step1);
        assert_eq!(opt.step, 2);

        Ok(())
    }

    #[test]
    fn test_adam_optimizer_size_mismatch() {
        let mut opt = AdamOptimizer::new(5);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2]; // Wrong size
        let lr = 0.001;

        let result = opt.step(&grads, &mut params, lr);
        assert!(result.is_err());
    }

    #[test]
    fn test_adamw_optimizer_creation() {
        let opt = AdamWOptimizer::new(100);
        assert_eq!(opt.m.len(), 100);
        assert_eq!(opt.v.len(), 100);
        assert_eq!(opt.weight_decay, 0.01);
        assert_eq!(opt.step, 0);
    }

    #[test]
    fn test_adamw_optimizer_custom_hyperparams() {
        let opt = AdamWOptimizer::with_hyperparams(50, 0.99, 0.9999, 1e-7, 0.001);
        assert_eq!(opt.m.len(), 50);
        assert_eq!(opt.v.len(), 50);
        assert_eq!(opt.beta1, 0.99);
        assert_eq!(opt.beta2, 0.9999);
        assert_eq!(opt.eps, 1e-7);
        assert_eq!(opt.weight_decay, 0.001);
    }

    #[test]
    fn test_adamw_weight_decay_getter() {
        let opt = AdamWOptimizer::new(100);
        assert_eq!(opt.weight_decay(), 0.01);
    }

    #[test]
    fn test_adamw_with_weight_decay_builder() {
        let opt = AdamWOptimizer::new(100).with_weight_decay(0.05);
        assert_eq!(opt.weight_decay(), 0.05);
    }

    #[test]
    fn test_adamw_optimizer_step() -> Result<()> {
        let mut opt = AdamWOptimizer::new(3);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let lr = 0.001;

        opt.step(&grads, &mut params, lr)?;

        // Parameters should have changed
        assert_ne!(params[0], 1.0);
        assert_ne!(params[1], 2.0);
        assert_ne!(params[2], 3.0);

        // Moments should have been updated
        assert!(opt.m.iter().any(|&x| x != 0.0));
        assert!(opt.v.iter().any(|&x| x != 0.0));
        assert_eq!(opt.step, 1);

        Ok(())
    }

    #[test]
    fn test_adamw_multiple_steps() -> Result<()> {
        let mut opt = AdamWOptimizer::new(2);
        let mut params = vec![1.0, 1.0];
        let grads = vec![0.1, 0.1];
        let lr = 0.01;

        // First step
        opt.step(&grads, &mut params, lr)?;
        let params_after_step1 = params.clone();

        // Second step with same gradients
        opt.step(&grads, &mut params, lr)?;

        // Parameters should continue to change (due to bias correction + weight decay)
        assert_ne!(params, params_after_step1);
        assert_eq!(opt.step, 2);

        Ok(())
    }

    #[test]
    fn test_adamw_size_mismatch() {
        let mut opt = AdamWOptimizer::new(5);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2]; // Wrong size
        let lr = 0.001;

        let result = opt.step(&grads, &mut params, lr);
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_vs_adamw_difference() -> Result<()> {
        // Adam and AdamW should produce different results due to decoupled weight decay
        let mut adam = AdamOptimizer::new(3);
        let mut adamw = AdamWOptimizer::with_hyperparams(3, 0.9, 0.999, 1e-8, 0.01);

        let mut params_adam = vec![1.0, 2.0, 3.0];
        let mut params_adamw = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let lr = 0.001;

        adam.step(&grads, &mut params_adam, lr)?;
        adamw.step(&grads, &mut params_adamw, lr)?;

        // AdamW should have smaller parameters due to direct weight decay
        // (params are directly reduced by lr * wd * param)
        for i in 0..3 {
            assert!(params_adamw[i] < params_adam[i],
                "AdamW param {} should be smaller than Adam param due to weight decay", i);
        }

        Ok(())
    }

    #[test]
    fn test_adamw_zero_weight_decay() -> Result<()> {
        // With zero weight decay, AdamW should behave like Adam (approximately)
        let mut adam = AdamOptimizer::new(3);
        let mut adamw_zero_wd = AdamWOptimizer::with_hyperparams(3, 0.9, 0.999, 1e-8, 0.0);

        let mut params_adam = vec![1.0, 2.0, 3.0];
        let mut params_adamw = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let lr = 0.001;

        adam.step(&grads, &mut params_adam, lr)?;
        adamw_zero_wd.step(&grads, &mut params_adamw, lr)?;

        // With zero weight decay, results should be nearly identical
        for i in 0..3 {
            assert!((params_adam[i] - params_adamw[i]).abs() < 1e-6,
                "AdamW with wd=0 should match Adam");
        }

        Ok(())
    }

    #[test]
    fn test_adamw_weight_decay_effect_accumulates() -> Result<()> {
        // Weight decay effect should compound over multiple steps
        let mut opt = AdamWOptimizer::with_hyperparams(2, 0.9, 0.999, 1e-8, 0.1); // High WD
        let mut params = vec![1.0, 1.0];
        let grads = vec![0.0, 0.0]; // Zero gradients
        let lr = 0.01;

        let initial_param = params[0];

        // Multiple steps with zero gradients but weight decay
        for _ in 0..10 {
            opt.step(&grads, &mut params, lr)?;
        }

        // Parameters should shrink due to weight decay even with zero gradients
        assert!(params[0] < initial_param,
            "Weight decay should reduce parameters even with zero gradients");

        Ok(())
    }

    #[test]
    fn test_adam_bias_correction() -> Result<()> {
        // Test that bias correction makes a difference
        let mut opt = AdamOptimizer::new(1);
        let mut params = vec![1.0];
        let grads = vec![1.0];
        let lr = 0.1;

        // First step - significant correction
        opt.step(&grads, &mut params, lr)?;
        let step1_param = params[0];

        // Reset
        let mut opt2 = AdamOptimizer::new(1);
        let mut params2 = vec![1.0];

        // Many steps later - correction diminishes
        for _ in 0..100 {
            opt2.step(&grads, &mut params2, lr)?;
        }

        // Later steps should have different effective learning rates
        // due to bias correction approaching 1.0
        assert_ne!(step1_param, params2[0]);

        Ok(())
    }

    // ===== Lion Optimizer Tests =====

    #[test]
    fn test_lion_optimizer_creation() {
        let opt = LionOptimizer::new(100);
        assert_eq!(opt.m.len(), 100);
        assert_eq!(opt.step, 0);
        assert_eq!(opt.beta1(), 0.9);
        assert_eq!(opt.weight_decay(), 0.01);
    }

    #[test]
    fn test_lion_optimizer_custom_hyperparams() {
        let opt = LionOptimizer::with_hyperparams(50, 0.99, 0.001);
        assert_eq!(opt.m.len(), 50);
        assert_eq!(opt.beta1(), 0.99);
        assert_eq!(opt.weight_decay(), 0.001);
    }

    #[test]
    fn test_lion_optimizer_step() -> Result<()> {
        let mut opt = LionOptimizer::new(3);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let lr = 0.001;

        opt.step(&grads, &mut params, lr)?;

        // Parameters should have changed
        assert_ne!(params[0], 1.0);
        assert_ne!(params[1], 2.0);
        assert_ne!(params[2], 3.0);

        // Momentum should have been updated
        assert!(opt.m.iter().any(|&x| x != 0.0));
        assert_eq!(opt.step, 1);

        Ok(())
    }

    #[test]
    fn test_lion_optimizer_multiple_steps() -> Result<()> {
        let mut opt = LionOptimizer::new(2);
        let mut params = vec![1.0, 1.0];
        let grads = vec![0.1, 0.1];
        let lr = 0.01;

        // First step
        opt.step(&grads, &mut params, lr)?;
        let params_after_step1 = params.clone();

        // Second step with same gradients
        opt.step(&grads, &mut params, lr)?;

        // Parameters should continue to change (due to momentum)
        assert_ne!(params, params_after_step1);
        assert_eq!(opt.step, 2);

        Ok(())
    }

    #[test]
    fn test_lion_optimizer_size_mismatch() {
        let mut opt = LionOptimizer::new(5);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2]; // Wrong size
        let lr = 0.001;

        let result = opt.step(&grads, &mut params, lr);
        assert!(result.is_err());
    }

    #[test]
    fn test_lion_sign_function() {
        // Test positive, negative, and zero
        assert_eq!(LionOptimizer::sign(1.5), 1.0);
        assert_eq!(LionOptimizer::sign(-0.5), -1.0);
        assert_eq!(LionOptimizer::sign(0.0), 0.0);
        assert_eq!(LionOptimizer::sign(f32::INFINITY), 1.0);
        assert_eq!(LionOptimizer::sign(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn test_lion_weight_decay_getter() {
        let opt = LionOptimizer::new(100);
        assert_eq!(opt.weight_decay(), 0.01);
    }

    #[test]
    fn test_lion_beta1_getter() {
        let opt = LionOptimizer::new(100);
        assert_eq!(opt.beta1(), 0.9);
    }

    #[test]
    fn test_lion_with_weight_decay_builder() {
        let opt = LionOptimizer::new(100).with_weight_decay(0.05);
        assert_eq!(opt.weight_decay(), 0.05);
    }

    #[test]
    fn test_lion_with_beta1_builder() {
        let opt = LionOptimizer::new(100).with_beta1(0.95);
        assert_eq!(opt.beta1(), 0.95);
    }

    #[test]
    fn test_lion_weight_decay_effect_with_zero_grads() -> Result<()> {
        // Lion should apply weight decay even with zero gradients
        let mut opt = LionOptimizer::with_hyperparams(3, 0.9, 0.1); // High WD
        let mut params = vec![1.0, 1.0, 1.0];
        let zero_grads = vec![0.0, 0.0, 0.0];
        let lr = 0.01;

        let initial_param = params[0];

        // Multiple steps with zero gradients
        for _ in 0..10 {
            opt.step(&zero_grads, &mut params, lr)?;
        }

        // Parameters should shrink due to weight decay
        assert!(params[0] < initial_param);

        Ok(())
    }

    #[test]
    fn test_lion_vs_adam_difference() -> Result<()> {
        // Lion and Adam should produce different results
        let mut lion = LionOptimizer::new(3);
        let mut adam = AdamOptimizer::new(3);

        let mut params_lion = vec![1.0, 2.0, 3.0];
        let mut params_adam = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let lr = 0.001;

        lion.step(&grads, &mut params_lion, lr)?;
        adam.step(&grads, &mut params_adam, lr)?;

        // Lion uses sign-based updates, Adam uses magnitude
        // Results should be different
        assert_ne!(params_lion, params_adam);

        Ok(())
    }

    #[test]
    fn test_lion_zero_weight_decay_no_reduction() -> Result<()> {
        // With zero weight decay, parameters shouldn't shrink from zero gradients
        let mut opt = LionOptimizer::with_hyperparams(2, 0.9, 0.0);
        let mut params = vec![1.0, 1.0];
        let zero_grads = vec![0.0, 0.0];
        let lr = 0.01;

        let initial_params = params.clone();

        // Multiple steps with zero gradients but no weight decay
        for _ in 0..5 {
            opt.step(&zero_grads, &mut params, lr)?;
        }

        // Parameters should remain very close to initial
        // (only minor changes from momentum decay)
        for i in 0..2 {
            assert!((params[i] - initial_params[i]).abs() < 0.01);
        }

        Ok(())
    }

    #[test]
    fn test_lion_momentum_accumulation() -> Result<()> {
        // Test that momentum accumulates correctly
        let mut opt = LionOptimizer::with_hyperparams(2, 0.9, 0.0);
        let mut params = vec![0.0, 0.0];
        let grads = vec![1.0, 1.0];
        let lr = 0.1;

        // First step
        opt.step(&grads, &mut params, lr)?;
        let momentum_after_step1 = opt.m[0];

        // m = 0.9 * 0 + 0.1 * 1 = 0.1
        assert!((momentum_after_step1 - 0.1).abs() < 1e-6);

        // Second step with same gradient
        opt.step(&grads, &mut params, lr)?;
        let momentum_after_step2 = opt.m[0];

        // m = 0.9 * 0.1 + 0.1 * 1 = 0.19
        assert!((momentum_after_step2 - 0.19).abs() < 1e-6);

        Ok(())
    }
}
