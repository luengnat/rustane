//! Training utilities for FP16 models
//!
//! Provides helpers for training neural networks on Apple Neural Engine:
//! - Loss scaling to prevent gradient underflow
//! - Gradient accumulation for larger effective batch sizes
//! - Integration with CPU-side training loops

pub mod grad_accum;
pub mod loss_scale;
pub mod model;
pub mod scheduler;

pub use grad_accum::GradAccumulator;
pub use loss_scale::LossScaler;
pub use model::Model;
pub use scheduler::{
    ConstantScheduler, LRScheduler, WarmupCosineScheduler, WarmupLinearScheduler,
};
