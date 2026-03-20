//! Transformer Layer Components and Backward Propagation
//!
//! This module provides MIL code generation and gradient computation for transformer layers.
//!
//! # Phase 2: Forward MIL Generation
//!
//! Generates Model Intermediate Language (MIL) code for ANE-optimized transformer operations:
//! - Scaled dot-product attention
//! - SiLU-gated feed-forward networks
//!
//! # Phase 3: Backward MIL Generation
//!
//! Generates MIL code for ANE backward propagation (gradient computation):
//! - RMSNorm backward
//! - Multi-head attention backward
//! - FFN (SwiGLU) backward
//! - Cross-entropy loss backward
//!
//! # Components
//!
//! ## MIL Code Generation (`mil_gen.rs`)
//!
//! Generates Model Intermediate Language (MIL) code for ANE-optimized transformer operations:
//! - Scaled dot-product attention
//! - SiLU-gated feed-forward networks
//!
//! ```ignore
//! use rustane::layers::MILGenerator;
//! use rustane::training::TransformerConfig;
//!
//! let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)?;
//! let gen = MILGenerator::new(&config);
//!
//! let attn_mil = gen.gen_attention_forward();  // MIL string for ANE compilation
//! let ffn_mil = gen.gen_ffn_forward();
//! ```
//!
//! ## Backward Passes (`transformer_backward.rs`)
//!
//! CPU-based gradient computation for all transformer layers:
//!
//! ### RMSNorm Backward
//!
//! Computes gradients through layer normalization:
//! ```ignore
//! use rustane::layers::rmsnorm_backward;
//!
//! let (d_x, dw) = rmsnorm_backward(&d_out, &x, &w);
//! // d_x: gradients w.r.t. input
//! // dw: gradients w.r.t. weights
//! ```
//!
//! ### Cross-Entropy Backward
//!
//! Implements standard cross-entropy loss gradient (softmax minus one-hot):
//! ```ignore
//! use rustane::layers::cross_entropy_backward;
//!
//! let grads = cross_entropy_backward(&logits, &targets, vocab_size);
//! // grads[i*vocab_size + target[i]] = softmax[i] - 1
//! ```
//!
//! ### Attention Backward
//!
//! Scaled dot-product attention gradient computation:
//! ```ignore
//! use rustane::layers::{attention_backward, AttentionConfig};
//!
//! let config = AttentionConfig {
//!     seq_len: 512,
//!     dim: 256,
//!     n_heads: 8,
//!     head_dim: 32,
//! };
//!
//! let (d_x, dw_q, dw_k, dw_v) = attention_backward(
//!     &d_out, &q, &k, &v, &attn_weights, &config
//! )?;
//! ```
//!
//! ### FFN Backward
//!
//! Feed-forward network with SiLU gating:
//! ```ignore
//! use rustane::layers::{ffn_backward, FFNConfig};
//!
//! let config = FFNConfig {
//!     seq_len: 512,
//!     dim: 256,
//!     hidden_dim: 768,
//! };
//!
//! let (d_x, dw1, dw3, dw2) = ffn_backward(
//!     &d_out, &x, &w1_out, &w1_gated, &config
//! )?;
//! ```
//!
//! # Mathematical Foundations
//!
//! All gradients computed via automatic differentiation:
//!
//! **RMSNorm**: `y = w * x / sqrt(mean(x²) + ε)`
//! - dL/dw = sum(dL/dy * (x / sqrt(mean(x²)))
//! - dL/dx = dL/dy * w / rms - (weighted_sum * x / rms³) / dim
//!
//! **Cross-Entropy**: `CE = -log(softmax(logits)[target])`
//! - dL/dlogits = softmax(logits) - one_hot(target)
//!
//! **Attention**: `attn = softmax(QK^T / √d) V`
//! - Computes dL/dQ, dL/dK, dL/dV via chain rule
//!
//! **FFN**: `out = (x W₁ * SiLU(x W₁)) ⊙ (x W₃) W₂`
//! - Propagates through SiLU gating and projections
//!
//! # Integration with ANE
//!
//! The design enables efficient hybrid computation:
//! - **Forward passes**: Generated as MIL code, executed on ANE
//! - **Backward passes**: CPU-based using cached forward activations
//! - **Gradient accumulation**: Scaled loss for multi-batch training
//! - **Mixed precision**: Quantized weights on ANE, FP32 gradients on CPU
//!
//! This approach maximizes ANE throughput while keeping backward pass flexible
//! for rapid experimentation.
//!
//! # Legacy Components
//!
//! The module also provides various layer abstractions used in earlier development:
//! - `activations`: ReLU, SiLU, GELU activation functions
//! - `normalization`: LayerNorm and RMSNorm implementations
//! - `linear`: Standard linear projection layers
//! - `conv`: Convolutional layers
//! - `attention`: Multi-head attention with builder patterns
//! - `sequential`: Sequential layer composition
//! - `checkpoint`: Layer checkpointing and serialization

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

// Phase 3: ANE backward MIL generators
pub mod backward;

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
    attention_backward, cross_entropy_backward, ffn_backward, rmsnorm_backward, AttentionConfig,
    FFNConfig,
};
