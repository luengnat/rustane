//! High-level neural network layer abstractions
//!
//! This module provides ergonomic, type-safe layer primitives for building
//! neural networks that run on the Apple Neural Engine.

pub mod activations;
pub mod attention;
pub mod checkpoint;
pub mod conv;
pub mod linear;
pub mod model;
pub mod normalization;
pub mod sequential;
pub mod swiglu;
pub mod traits;

pub use activations::{ReLU, SiLU, GELU};
pub use attention::{
    MultiHeadAttention, MultiHeadAttentionBuilder, SelfAttention, SelfAttentionBuilder,
};
pub use checkpoint::{Checkpoint, LayerWeights, ModelMetadata};
pub use conv::Conv2d;
pub use linear::Linear;
pub use model::{LayerInfo, Model, ModelSummary};
pub use normalization::{LayerNorm, LayerNormBuilder, RMSNorm, RMSNormBuilder};
pub use sequential::{Sequential, SequentialBuilder, SharedLayer};
pub use swiglu::{SiLU as SwiGLUSiLU, SwiGLU, SwiGLUBuilder};
pub use traits::{BiasLayer, Layer, LayerBuilder, Shape, WeightsLayer};
