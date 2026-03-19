//! Training utilities for FP16 models
//!
//! Provides helpers for training neural networks on Apple Neural Engine:
//! - Loss scaling to prevent gradient underflow
//! - Gradient accumulation for larger effective batch sizes
//! - Integration with CPU-side training loops

pub mod grad_accum;
pub mod loss_scale;

pub use grad_accum::GradAccumulator;
pub use loss_scale::LossScaler;
