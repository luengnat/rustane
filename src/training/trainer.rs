//! Training orchestration for models

use std::fmt;
use crate::error::Result;
use crate::wrapper::ANETensor;
use crate::data::Batch;
use crate::training::{Model, LossFn};
use crate::training::scheduler::LRScheduler;
use crate::training::grad_accum::GradAccumulator;

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
