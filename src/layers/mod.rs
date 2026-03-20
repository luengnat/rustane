//! High-level neural network layer abstractions
//!
//! This module provides ergonomic, type-safe layer primitives for building
//! neural networks that run on the Apple Neural Engine.

pub mod activations;
pub mod attention;
pub mod checkpoint;
pub mod conv;
pub mod linear;
pub mod mil_gen;
pub mod model;
pub mod normalization;
pub mod sequential;
pub mod swiglu;
pub mod traits;
pub mod transformer_backward;

pub use activations::{ReLU, SiLU, GELU};
pub use attention::{
    MultiHeadAttention, MultiHeadAttentionBuilder, SelfAttention, SelfAttentionBuilder,
};
pub use checkpoint::{Checkpoint, LayerWeights, ModelMetadata};
pub use conv::Conv2d;
pub use linear::Linear;
pub use mil_gen::MILGenerator;
pub use model::{LayerInfo, Model, ModelSummary};
pub use normalization::{LayerNorm, LayerNormBuilder, RMSNorm, RMSNormBuilder};
pub use sequential::{Sequential, SequentialBuilder, SharedLayer};
pub use swiglu::{SiLU as SwiGLUSiLU, SwiGLU, SwiGLUBuilder};
pub use traits::{BiasLayer, Layer, LayerBuilder, Shape, WeightsLayer};
pub use transformer_backward::{
    rmsnorm_backward, cross_entropy_backward, attention_backward, ffn_backward,
    AttentionConfig, FFNConfig,
};
