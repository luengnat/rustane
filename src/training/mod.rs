//! Training utilities for FP16 models
//!
//! Provides helpers for training neural networks on Apple Neural Engine:
//! - Loss scaling to prevent gradient underflow
//! - Gradient accumulation for larger effective batch sizes
//! - Integration with CPU-side training loops

pub mod grad_accum;
pub mod loss;
pub mod loss_scale;
pub mod model;
pub mod scheduler;
pub mod trainer;

pub use grad_accum::GradAccumulator;
pub use loss::{CrossEntropyLoss, LossFn, MSELoss};
pub use loss_scale::LossScaler;
pub use model::Model;
pub use scheduler::{
    ConstantScheduler, LRScheduler, WarmupCosineScheduler, WarmupLinearScheduler,
};
pub use trainer::{StepMetrics, TrainerError};
