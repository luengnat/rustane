# Phase 2 Week 2: MVP Trainer Design

**Date:** 2026-03-20
**Status:** Design Approved
**Context:** Phase 1 complete (data pipeline, 39 tests). Phase 2 Week 1 complete (LR schedulers, 9 tests). Total: 186 tests passing.

## Overview

This design specifies an MVP `Trainer` struct that orchestrates the core training loop: forward pass → loss computation → backward pass → optimizer step. The trainer uses a builder pattern, integrates with existing `Optimizer` and `LRScheduler` traits, and introduces two new traits: `Model` and `LossFn`.

**Philosophy:** Trainer is a thin orchestrator that coordinates existing pieces. It doesn't own the model, doesn't implement loss computation, doesn't manage optimizers directly - it delegates to trait objects and focuses on correct sequencing and metrics.

## Architecture

```
┌──────────────────────────────────────┐
│      Trainer (orchestrator)          │
├──────────────────────────────────────┤
│  1. Model::forward(&batch) → logits  │
│  2. LossFn::compute(logits) → loss   │
│  3. Model::backward(loss) → grads    │
│  4. Optimizer::step(grads, lr)       │
│  5. Metrics: loss, grad_norm, lr     │
└──────────────────────────────────────┘
        ↓           ↓           ↓
     Model      LossFn    Optimizer
    (new)      (new)     (exists)
```

## Core Components

### 1. Model Trait

**File:** `src/training/model.rs`

```rust
pub trait Model: Send {
    /// Forward pass: tokenized batch → ANE tensor output
    ///
    /// # Arguments
    /// - `batch`: Tokenized batch [batch_size × seq_len]
    ///
    /// # Returns
    /// ANETensor with logits/activations for loss computation
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor>;

    /// Backward pass: given loss scalar, compute parameter gradients
    ///
    /// # Arguments
    /// - `loss`: Scalar loss value from loss function
    ///
    /// # Returns
    /// Gradients as Vec<f32>, one gradient per parameter
    /// Should match parameter_count() in length
    fn backward(&mut self, loss: f32) -> Result<Vec<f32>>;

    /// Get mutable reference to model parameters
    ///
    /// Used by optimizer to update weights
    fn parameters(&mut self) -> &mut [f32];

    /// Total number of parameters
    fn param_count(&self) -> usize;
}
```

**Design rationale:**
- `forward()` returns `ANETensor` directly (no conversion needed, realistic for ANE integration)
- `backward()` takes scalar loss (model knows how to convert to gradients internally)
- `parameters()` allows optimizer to mutate weights in-place
- Model hides ANE complexity (forward dispatch, gradient computation)

---

### 2. LossFn Trait

**File:** `src/training/loss.rs`

```rust
pub trait LossFn: Send {
    /// Compute scalar loss from model output and batch targets
    ///
    /// # Arguments
    /// - `logits`: Model output tensor from forward pass
    /// - `batch`: Batch containing token IDs (used for target labels)
    ///
    /// # Returns
    /// Scalar loss value (f32)
    fn compute(&self, logits: &ANETensor, batch: &Batch) -> Result<f32>;
}
```

**Implementations to provide:**
- `CrossEntropyLoss` - Standard for language modeling (next-token prediction)
- `MSELoss` - Baseline for regression tasks

**Design rationale:**
- Loss function is injectable (trainer doesn't own it)
- Decoupled from model (different models can use same loss function)
- Takes both logits and batch (may need targets from batch)

---

### 3. StepMetrics Struct

**File:** `src/training/trainer.rs`

```rust
/// Metrics returned after each training step
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
```

**Design rationale:**
- Struct allows future extension (add timing, memory, ANE utilization in Phase 2 Week 3)
- Includes step counter for easy logging
- Includes learning rate for debugging schedule progression

---

### 4. Trainer Struct & Builder

**File:** `src/training/trainer.rs`

```rust
/// Orchestrates training: forward → loss → backward → optimize
pub struct Trainer<'a, M: Model> {
    model: &'a mut M,
    optimizer: Box<dyn Optimizer>,
    scheduler: Box<dyn LRScheduler>,
    loss_fn: Box<dyn LossFn>,
    current_step: u32,
}

/// Builder for Trainer (ensures all required components are set)
pub struct TrainerBuilder<'a, M: Model> {
    model: &'a mut M,
    optimizer: Option<Box<dyn Optimizer>>,
    scheduler: Option<Box<dyn LRScheduler>>,
    loss_fn: Option<Box<dyn LossFn>>,
}

impl<'a, M: Model> TrainerBuilder<'a, M> {
    pub fn new(model: &'a mut M) -> Self;

    pub fn with_optimizer<O: Optimizer + 'static>(mut self, opt: O) -> Self;
    pub fn with_scheduler<S: LRScheduler + 'static>(mut self, sch: S) -> Self;
    pub fn with_loss_fn<L: LossFn + 'static>(mut self, loss: L) -> Self;

    /// Build Trainer, ensuring all components are set
    pub fn build(self) -> Result<Trainer<'a, M>>;
}

impl<'a, M: Model> Trainer<'a, M> {
    /// Single training step
    pub fn train_step(&mut self, batch: &Batch) -> Result<StepMetrics>;
}
```

**train_step() algorithm:**
1. Forward: `logits = model.forward(batch)`
2. Loss: `loss = loss_fn.compute(&logits, batch)`
3. Backward: `grads = model.backward(loss)`
4. Metrics: `grad_norm = compute_norm(&grads)`
5. LR: `lr = scheduler.get_lr(current_step)`
6. Optimize: `optimizer.step(&grads, model.parameters())`
7. Increment: `current_step += 1`
8. Return: `StepMetrics { loss, grad_norm, learning_rate: lr, step: current_step - 1 }`

**Design rationale:**
- Builder pattern ensures all components are set before training starts
- Trainer borrows model (not owned) for flexibility
- `train_step()` is simple, linear, easy to understand
- Error handling at each step (forward, backward, optimizer all return Result)

---

## Error Handling

**Error types (in `src/training/trainer.rs`):**

```rust
pub enum TrainerError {
    /// Model forward pass failed
    ModelForwardFailed(String),

    /// Model backward pass failed
    ModelBackwardFailed(String),

    /// Loss computation failed or invalid
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
```

**Strategy:**
- All fallible operations wrapped in `?`
- Detect NaN/Inf in gradients immediately (indicates loss scale or numerical instability)
- Return descriptive errors for debugging
- Don't skip invalid steps silently

---

## Testing Strategy

### Unit Tests (in `src/training/trainer.rs`)

1. **test_builder_construction** - Builder methods work correctly, all combinations valid
2. **test_builder_missing_component** - Builder::build() fails if optimizer/scheduler/loss missing
3. **test_single_train_step** - Forward→loss→backward→optim succeeds on valid input
4. **test_metrics_correctness** - Returned metrics match expected values
5. **test_gradient_norm_computation** - grad_norm calculation verified
6. **test_learning_rate_progression** - Scheduler LR changes correctly across steps
7. **test_invalid_gradients_detection** - NaN/Inf gradients caught and reported

### Integration Test (new file `tests/trainer_integration.rs`)

1. Create synthetic dataset (10 samples, seq_len=4)
2. Create RandomSampler
3. Create DataLoader (batch_size=2)
4. Create ToyModel (2-layer linear network, 100 params)
5. Create Trainer with:
   - CrossEntropyLoss
   - SimpleOptimizer (SGD)
   - WarmupLinearScheduler (5 warmup, 10 total)
6. Run 10 train steps
7. Verify:
   - All steps return valid metrics
   - Loss generally decreases (no requirement for monotonic, but shouldn't diverge)
   - Grad norms are positive and finite
   - Learning rate follows schedule

### Test Model Implementation

```rust
/// Minimal model for testing: two linear layers
struct ToyModel {
    w1: Vec<f32>,  // [input_dim, hidden_dim]
    w2: Vec<f32>,  // [hidden_dim, vocab_size]
    last_input: Option<Vec<f32>>,  // cache for backward
}

impl Model for ToyModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        // Flatten batch → linear(hidden) → linear(vocab)
        // Returns ANETensor with logits
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        // Simple backprop through two layers
    }

    fn parameters(&mut self) -> &mut [f32] { ... }
    fn param_count(&self) -> usize { ... }
}
```

---

## Dependencies

### Already Exist
- ✅ `Optimizer` trait (from Phase 2 planning)
- ✅ `LRScheduler` trait + implementations (Phase 2 Week 1)
- ✅ `Batch` struct (Phase 1 Week 1)
- ✅ `ANETensor`, `ANERuntime` (wrapper module)

### New (this design)
- `Model` trait
- `LossFn` trait
- `Trainer` struct + builder
- `StepMetrics` struct
- `CrossEntropyLoss`, `MSELoss` implementations
- ToyModel for testing

### Updates Needed
- `src/lib.rs` - Export Trainer, Model, LossFn, StepMetrics, CrossEntropyLoss
- `src/training/mod.rs` - Add new modules and re-exports

---

## Files to Create/Modify

### Create
- `src/training/trainer.rs` (~1200 lines: Trainer, TrainerBuilder, StepMetrics, error types)
- `src/training/model.rs` (~100 lines: Model trait)
- `src/training/loss.rs` (~200 lines: LossFn trait, CrossEntropyLoss, MSELoss)
- `tests/trainer_integration.rs` (~400 lines: integration test + ToyModel)
- `examples/train_toy_model.rs` (~300 lines: full example with all pieces)

### Modify
- `src/training/mod.rs` - Add module declarations and exports
- `src/lib.rs` - Export new types
- `Cargo.toml` - No changes (all dependencies exist)

---

## Success Criteria

- ✅ Code compiles with no errors or warnings
- ✅ 7 unit tests pass
- ✅ Integration test passes (10-step training completes, loss doesn't diverge)
- ✅ Example runs without errors
- ✅ Total: 190+ tests passing (186 existing + 7 trainer + integration example)
- ✅ New code follows rustane patterns (error handling, trait-based, immutable semantics)

---

## Future Extensions (Phase 2 Week 3+)

**In-model profiling** (Week 3):
- Add `profile_step()` variant that returns `(metrics, profile_data)`
- Measure forward/backward/optimizer timing
- Track ANE utilization

**Gradient checkpointing** (Week 4):
- Optional activation checkpointing to reduce memory
- Recompute vs store trade-off

**In-model loss computation** (Week 4):
- Move loss kernel into ANE
- Reduce ANE↔CPU data transfer 100-1000x

**Multi-device training** (Phase 3):
- DistributedTrainer wrapper
- AllReduce synchronization

---

## Notes

- This design is minimal and focused (MVP). No profiling, no checkpointing, no distributed training yet.
- Model trait deliberately abstract (doesn't prescribe internals) to support different architectures
- LossFn is injectable for flexibility
- Builder pattern ensures compile-time safety (can't create incomplete Trainer)
- Integration test uses ToyModel instead of real model (faster testing, clearer what's being tested)
