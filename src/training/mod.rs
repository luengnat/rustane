//! Training Infrastructure for Transformer Models on Apple Neural Engine
//!
//! Provides comprehensive training utilities for transformer-style models:
//! - Model trait and loss functions for flexible training
//! - Learning rate scheduling (constant, linear warmup, cosine annealing)
//! - Loss scaling for FP16 gradient stability
//! - Gradient accumulation for larger effective batch sizes
//! - Trainer orchestration for multi-step training
//! - TransformerConfig validation and parameter counting
//! - TransformerANE reference implementation
//!
//! # Architecture
//!
//! The training system follows a modular, composable design:
//!
//! ## Model Trait
//!
//! Central abstraction for any trainable model:
//! ```ignore
//! pub trait Model {
//!     fn forward(&mut self, batch: &Batch) -> Result<ANETensor>;
//!     fn backward(&mut self, loss: f32) -> Result<Vec<f32>>;
//!     fn backward_with_batch(&mut self, batch: &Batch, loss: f32) -> Result<Vec<f32>>;
//!     fn parameters(&mut self) -> &mut [f32];
//!     fn param_count(&self) -> usize;
//! }
//! ```
//!
//! Implementations:
//! - **TransformerANE**: Full transformer with ANE forward, CPU backward
//!
//! ## TransformerConfig
//!
//! Architecture configuration with validation:
//! ```ignore
//! let config = TransformerConfig::new(
//!     vocab_size: 4096,
//!     dim: 256,
//!     hidden_dim: 768,
//!     n_heads: 8,
//!     n_layers: 6,
//!     seq_len: 512
//! )?;
//! ```
//!
//! Validates:
//! - dim divisible by n_heads (multi-head attention requirement)
//! - n_heads, dim, and hidden_dim are non-zero
//!
//! Computes:
//! - head_dim = dim / n_heads
//! - total param_count for initialization
//!
//! ## Learning Rate Schedulers
//!
//! Multiple scheduling strategies for training convergence:
//!
//! ### ConstantScheduler
//! Fixed learning rate throughout training:
//! ```ignore
//! let scheduler = ConstantScheduler::new(0.001);
//! let lr = scheduler.get_lr(step);  // Always 0.001
//! ```
//!
//! ### WarmupLinearScheduler
//! Linear warmup followed by linear decay:
//! ```ignore
//! let scheduler = WarmupLinearScheduler::new(
//!     peak_lr: 0.001,
//!     warmup_steps: 1000,
//!     total_steps: 10000
//! );
//! // 0 → 0.001 over steps 0-1000
//! // 0.001 → 0 over steps 1000-10000
//! ```
//!
//! ### WarmupCosineScheduler
//! Linear warmup followed by cosine annealing:
//! ```ignore
//! let scheduler = WarmupCosineScheduler::new(
//!     peak_lr: 0.001,
//!     warmup_steps: 1000,
//!     total_steps: 10000,
//!     min_lr: 1e-5
//! );
//! // Smooth decay using cosine function
//! // Maintains min_lr to prevent zero gradients
//! ```
//!
//! All schedulers implement:
//! ```ignore
//! pub trait LRScheduler {
//!     fn get_lr(&self, step: usize) -> f32;
//! }
//! ```
//!
//! ## Loss Functions
//!
//! Standard loss implementations:
//!
//! ### CrossEntropyLoss
//! For next-token prediction:
//! ```ignore
//! let loss = CrossEntropyLoss::compute(&logits, &targets)?;
//! ```
//! - Logits shape: [batch_size, seq_len, vocab_size]
//! - Targets shape: [batch_size, seq_len]
//! - Numerically stable softmax computation
//!
//! ### MSELoss
//! For regression tasks:
//! ```ignore
//! let loss = MSELoss::compute(&predictions, &targets)?;
//! ```
//!
//! All losses implement:
//! ```ignore
//! pub trait LossFn {
//!     fn compute(&self, output: &ANETensor, target: &ANETensor) -> Result<f32>;
//! }
//! ```
//!
//! ## Trainer Orchestration
//!
//! High-level training loop management:
//! ```ignore
//! let mut trainer = TrainerBuilder::new(model, optimizer, scheduler)?
//!     .with_loss_scaler(LossScaler::new(16.0))
//!     .with_grad_accumulator(GradAccumulator::new(4))
//!     .build()?;
//!
//! for batch in dataloader {
//!     let metrics = trainer.step(&batch)?;
//!     println!("loss: {}, lr: {}", metrics.loss, metrics.learning_rate);
//! }
//! ```
//!
//! Tracks:
//! - Current step and learning rate
//! - Loss and gradient statistics
//! - Gradient norms for monitoring
//!
//! ## FP16 Training Utilities
//!
//! ### LossScaler
//! Prevents gradient underflow in FP16:
//! ```ignore
//! let mut scaler = LossScaler::new(2048.0);
//! let scaled_loss = scaler.scale(loss);
//! // ... backward pass
//! scaler.unscale(&mut gradients)?;
//! ```
//!
//! ### GradAccumulator
//! Accumulates gradients across multiple batches:
//! ```ignore
//! let accum = GradAccumulator::new(4);  // Accumulate 4 batches
//! accum.add(&grads)?;
//! if accum.is_full() {
//!     let accumulated = accum.get_accumulated();
//!     optimizer.step(&accumulated)?;
//!     accum.reset();
//! }
//! ```
//!
//! # TransformerANE Model
//!
//! Reference transformer implementation:
//! - **Embedding**: vocab_size × dim lookup table
//! - **Transformer Layers**: Attention + FFN with residuals
//! - **RMSNorm**: Pre-normalization at each layer
//! - **Attention**: Scaled dot-product, multi-head
//! - **FFN**: SiLU-gated with parallel projections
//! - **Output**: Classifier for next-token prediction
//!
//! Forward pass caches activations for backward pass.
//! Backward pass uses cached activations to compute gradients.
//!
//! # Phase 2 Architecture
//!
//! Current implementation focuses on:
//! - **ANE Forward Pass**: Efficient computation via MIL kernels
//! - **CPU Backward Pass**: Flexible gradient computation using cached activations
//! - **Stability**: Numerical gradient validation, loss scaling
//! - **Experimentation**: Easy scheduler/optimizer swapping
//!
//! # Phase 3: ANE Backward Kernels
//!
//! Phase 3 adds end-to-end ANE training with backward MIL kernels:
//! - **ANEGradientAccumulator**: Manages gradient accumulation in ANE memory
//! - **backward_on_ane()**: ANE-accelerated backward pass with gradient accumulation
//! - **BackwardValidationSuite**: Startup validation against CPU reference (1e-6 tolerance)
//! - **MIL Generators**: RMSNorm, Attention, FFN, and Loss backward code generation
//!
//! Usage:
//! ```ignore
//! let mut accum = ANEGradientAccumulator::new(param_count)?;
//! model.backward_on_ane(&batch, loss, &mut accum)?;
//! let grads = accum.get_accumulated()?;
//! ```
//!
//! # Integration with Data Pipeline
//!
//! Works seamlessly with `crate::data` module:
//! ```ignore
//! use rustane::data::{DataLoader, SequentialDataset, RandomSampler, PadCollator};
//!
//! let dataset = SequentialDataset::from_vec(tokens)?;
//! let sampler = RandomSampler::new(0);
//! let collator = PadCollator::new(512, 0);
//! let loader = DataLoader::new(dataset, sampler, collator, 32)?;
//!
//! for batch in loader {
//!     trainer.step(&batch)?;
//! }
//! ```
//!
//! # Example: Full Training Loop
//!
//! ```ignore
//! use rustane::training::*;
//! use rustane::data::*;
//!
//! // Configuration
//! let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512)?;
//! let model = TransformerANE::new(&config)?;
//! let scheduler = WarmupCosineScheduler::new(0.001, 1000, 10000, 1e-5);
//! let optimizer = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
//!
//! // Data pipeline
//! let dataset = SequentialDataset::from_vec(tokens)?;
//! let loader = DataLoader::new(
//!     dataset,
//!     RandomSampler::new(42),
//!     PadCollator::new(512, 0),
//!     32
//! )?;
//!
//! // Training
//! let mut trainer = TrainerBuilder::new(model, optimizer, scheduler)?.build()?;
//! for epoch in 0..3 {
//!     for (step, batch) in loader.iter().enumerate() {
//!         let metrics = trainer.step(&batch)?;
//!         if step % 100 == 0 {
//!             println!("Epoch {} Step {}: loss={:.4}", epoch, step, metrics.loss);
//!         }
//!     }
//! }
//! ```

pub mod ane_backward_executor;
pub mod ane_backward_kernel;
pub mod ane_gradient_buffer;
pub mod ane_persistent_buffer;
pub mod backend;
pub mod benchmark;
pub mod checkpoint;
pub mod distributed;
pub mod grad_accum;
pub mod loss;
pub mod loss_scale;
pub mod model;
pub mod scheduler;
pub mod trainer;
pub mod transformer_config;
pub mod transformer_model;

pub use ane_backward_executor::{ANEBackwardModel, ANEGradientAccumulator};
pub use ane_backward_kernel::{ANEBackwardKernel, ANEBackwardKernelCache};
pub use ane_gradient_buffer::ANEGradientBuffer;
pub use backend::{CpuTrainingBackend, TrainingBackend};
pub use benchmark::BackwardBenchmark;
pub use checkpoint::{
    checkpoint_filename, Checkpoint, LossScalerState, ModelConfig, OptimizerState,
};
pub use distributed::{
    AllReduce, DistributedOptimizerState, DistributedSynchronizer, ReduceMode,
};
pub use grad_accum::GradAccumulator;
pub use loss::{CrossEntropyLoss, LossFn, MSELoss};
pub use loss_scale::LossScaler;
pub use model::Model;
pub use scheduler::{ConstantScheduler, LRScheduler, WarmupCosineScheduler, WarmupLinearScheduler};
pub use trainer::{AdamOptimizer, Optimizer, StepMetrics, Trainer, TrainerBuilder, TrainerError};
pub use transformer_config::{MixedPrecisionConfig, Precision, TransformerConfig};
pub use transformer_model::{
    ane_forward_block_summary, ParameterGroup, ParameterGroupKind, TransformerANE,
};
