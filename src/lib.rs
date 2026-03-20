//! # Rustane - Apple Neural Engine Rust Library
//!
//! This library provides safe, idiomatic Rust bindings to Apple's Neural Engine (ANE)
//! via reverse-engineered private APIs.
//!
//! ## Platform Requirements
//!
//! - **OS**: macOS 15+ (Sequoia)
//! - **Hardware**: Apple Silicon with ANE (M1/M2/M3/M4)
//! - **Rust**: 1.70+ with 2021 edition
//!
//! ## Quick Start
//!
//! ```no_run
//! fn main() -> rustane::Result<()> {
//!     // Initialize ANE runtime
//!     rustane::init()?;
//!
//!     // Your ML code here...
//!
//!     Ok(())
//! }
//! ```
//!
//! For more information, see:
//! - [maderix/ANE](https://github.com/maderix/ANE) - Upstream C bridge and research
//! - [hollance/neural-engine](https://github.com/hollance/neural-engine) - ANE documentation

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(missing_docs)]
#![warn(unused_extern_crates)]

mod sys;

pub mod data;
pub mod error;
pub mod layers;
pub mod mil;
pub mod platform;
pub mod training;
pub mod utils;
pub mod wrapper;

// Python bindings (feature-gated)
#[cfg(feature = "python")]
pub mod python;

pub use data::{
    Batch, ChunkIterator, Collator, DataLoader, Dataset, JsonlDataset, PadCollator,
    RandomSampler, Sampler, SequentialDataset, SequentialSampler, ShardConfig,
    ShardMetadata, ShardBatch, ShardedDataLoader, ShardIterator, TextDataset, TruncateCollator,
};
pub use error::{Error, Result};
pub use layers::traits::Shape;
pub use layers::{Conv2d, Layer, LayerBuilder, Linear, ReLU, SiLU, GELU};
pub use layers::{LayerInfo, Model, ModelSummary, Sequential, SequentialBuilder};
pub use layers::{
    MultiHeadAttention, MultiHeadAttentionBuilder, SelfAttention, SelfAttentionBuilder,
};
pub use mil::{rmsnorm_mil, total_leaked_bytes, LinearLayer, MILBuilder, WeightBlob};
pub use platform::ANEAvailability;
pub use training::{
    ConstantScheduler, CrossEntropyLoss, GradAccumulator, LRScheduler, LossFn, LossScaler,
    Model as TrainingModel, Optimizer, StepMetrics, Trainer, TrainerBuilder, TrainerError,
    WarmupCosineScheduler, WarmupLinearScheduler,
};
pub use wrapper::{ANECompiler, ANEExecutor, ANERuntime, ANETensor, KernelCache};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the ANE runtime
///
/// This function must be called before any other ANE operations.
/// It loads the private AppleNeuralEngine framework and resolves
/// the necessary classes and methods.
///
/// # Errors
///
/// Returns an error if:
/// - Running on non-Apple Silicon hardware
/// - The ANE framework cannot be loaded
/// - Required private APIs are not available
///
/// # Example
///
/// ```no_run
/// fn main() -> rustane::Result<()> {
///     rustane::init()?;
///     // Use ANE operations...
///     Ok(())
/// }
/// ```
pub fn init() -> Result<()> {
    let _ = ANERuntime::init()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
