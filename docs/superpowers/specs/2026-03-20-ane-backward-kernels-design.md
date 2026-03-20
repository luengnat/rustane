# Phase 3: ANE Backward Kernels — Design Specification

**Date:** 2026-03-20  
**Status:** APPROVED  
**Phase:** Phase 3 (ANE Backward Propagation)  
**Scope:** Full ANE backward kernels for end-to-end training

---

## Executive Summary

Phase 3 extends Phase 2's forward-pass ANE optimization by implementing backward propagation (gradient computation) entirely on the ANE device. This enables true end-to-end training on Apple Silicon hardware with gradients accumulated in ANE memory, transferred to CPU once per training step for optimizer updates.

**Key Outcomes:**
- All backward operations (RMSNorm, attention, FFN, loss) compute on ANE via MIL kernels
- 1e-6 relative error validation against CPU reference implementations
- Mixed-precision support (FP16 activations + FP32 gradients)
- 50% reduction in ANE memory footprint vs Phase 2
- Reference validation suite ensures correctness before training

---

## Architecture

### Data Flow

```
Training Step:
┌─────────────────────────────────────────────────────────────┐
│ CPU: DataLoader                                             │
│  Batch [batch_size × seq_len] → ANETensor                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ ANE: Forward Pass (Phase 2)                                 │
│  Input: ANETensor                                           │
│  Activations cached in IOSurface (FP16)                     │
│  Output: logits [batch_size × seq_len × vocab_size]        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ CPU or ANE: Loss Computation                                │
│  Loss = CrossEntropy(logits, targets)                       │
│  loss_scalar ∈ ℝ                                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ ANE: Backward Pass (Phase 3 — NEW)                          │
│  1. Loss backward: dloss/dlogits                            │
│  2. Attention backward: uses cached Q,K,V from forward      │
│  3. FFN backward: uses cached pre-activation               │
│  4. RMSNorm backward: uses cached normalized activations    │
│  All gradients computed in FP32                             │
│  Accumulated in ANEGradientAccumulator (IOSurface)          │
│  Output: accumulated gradients ∈ ANE memory                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ CPU: Gradient Transfer                                       │
│  Transfer accumulated gradients from ANE to CPU             │
│  gradients [num_params] ∈ CPU memory                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ CPU: Optimizer Step (unchanged from Phase 2)                │
│  params -= learning_rate * gradients                        │
│  Update model weights on CPU                                │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Backward kernels mirror forward MIL structure** — Each backward operation (RMSNorm, attention, FFN, loss) has a corresponding MIL generator ported from CPU reference
2. **Activation caching via IOSurface** — Forward pass caches activations in IOSurface; backward kernels read directly from IOSurface
3. **Gradient accumulation on ANE** — Gradients stay in ANE memory across multiple chunks, transferred to CPU once per training step
4. **Mixed precision (FP16 activations + FP32 gradients)** — Reduce ANE memory by 50%, compute full-precision gradients for numerical stability
5. **Validation-first** — All backward kernels validated once at startup against CPU reference with 1e-6 tolerance; no per-step validation overhead

---

## Components

### 1. BackwardMILGenerator Trait

**File:** `src/layers/backward/mod.rs`

```rust
pub trait BackwardMILGenerator {
    /// Generate MIL code for this backward operation
    fn generate(&self, config: &TransformerConfig) -> Result<String>;
    
    /// Validate generated kernel against CPU reference
    fn validate(&self, config: &TransformerConfig) -> Result<()>;
    
    /// Operation name (e.g., "rmsnorm_backward")
    fn operation_name(&self) -> &'static str;
}
```

Implementors:
- `RMSNormBackwardGen` — Backward pass for RMSNorm normalization
- `AttentionBackwardGen` — Backward pass for multi-head attention
- `FFNBackwardGen` — Backward pass for feed-forward network
- `LossBackwardGen` — Cross-entropy loss backward

### 2. ANEGradientAccumulator

**File:** `src/training/ane_backward_executor.rs`

Manages gradient accumulation in ANE memory across training chunks:

```rust
pub struct ANEGradientAccumulator {
    accumulator_surface: IOSurface,  // Gradient storage on ANE
    num_params: usize,
    precision: Precision,  // FP32
}

impl ANEGradientAccumulator {
    pub fn new(num_params: usize) -> Result<Self>;
    pub fn accumulate(&mut self, gradients: &[f32]) -> Result<()>;
    pub fn get_accumulated(&self) -> Result<Vec<f32>>;
    pub fn reset(&mut self) -> Result<()>;
}
```

**Responsibilities:**
- Initialize IOSurface with appropriate size for all gradients
- Accumulate new gradients from each backward pass
- Transfer final accumulated gradients to CPU memory
- Reset accumulator for next training step

### 3. Model Trait Extension

**File:** `src/training/transformer_model.rs`

Add new method to `Model` trait:

```rust
pub trait Model {
    // ... existing methods (forward, backward) ...
    
    /// Execute backward pass on ANE with gradient accumulation
    /// 
    /// Coordinates with ANE device to:
    /// 1. Run all backward kernels on ANE
    /// 2. Accumulate gradients in ANEGradientAccumulator
    /// 3. Return accumulated gradients to CPU
    fn backward_on_ane(&mut self, loss: f32) -> Result<Gradients>;
}
```

Implementor: `TransformerANE`

**Responsibilities:**
- Retrieve cached activations from forward pass (stored in IOSurface)
- Invoke each backward MIL kernel in dependency order
- Pass intermediate gradients to ANEGradientAccumulator
- Return final accumulated gradients for optimizer step

### 4. BackwardValidationSuite

**File:** `src/layers/backward/validation.rs`

Reference validation suite that runs once at startup:

```rust
pub struct BackwardValidationSuite {
    rmsnorm_gen: RMSNormBackwardGen,
    attention_gen: AttentionBackwardGen,
    ffn_gen: FFNBackwardGen,
    loss_gen: LossBackwardGen,
}

pub struct ValidationReport {
    pub rmsnorm_passed: bool,
    pub attention_passed: bool,
    pub ffn_passed: bool,
    pub loss_passed: bool,
    pub max_relative_error: f32,
}

impl BackwardValidationSuite {
    pub fn new() -> Self;
    pub fn validate_all(&self, config: &TransformerConfig) -> Result<ValidationReport>;
    pub fn validate_against_reference(
        ane_gradients: &[f32],
        cpu_gradients: &[f32],
    ) -> Result<()>;  // 1e-6 relative tolerance
}
```

**Process:**
1. Create small reference config (hidden_dim=256, num_heads=8, num_layers=2)
2. Generate random input batch (batch_size=2, seq_len=4)
3. For each backward operation:
   - Generate MIL code
   - Compile to ANE kernel
   - Run on tiny batch
   - Compare against CPU reference (from `transformer_backward.rs`)
   - Verify relative error < 1e-6
4. Return ValidationReport with pass/fail status

---

## File Structure

```
src/layers/
├── mod.rs (update exports)
├── mil_gen.rs (existing, forward kernels)
├── transformer_backward.rs (existing, CPU reference)
└── backward/
    ├── mod.rs (trait + module exports)
    ├── rmsnorm_backward_gen.rs (RMSNorm backward MIL generator)
    ├── attention_backward_gen.rs (Attention backward MIL generator)
    ├── ffn_backward_gen.rs (FFN backward MIL generator)
    ├── loss_backward_gen.rs (Cross-entropy backward MIL generator)
    └── validation.rs (Reference validation suite)

src/training/
├── mod.rs (existing)
├── transformer_model.rs (extend Model trait with backward_on_ane)
├── ane_backward_executor.rs (NEW — ANEGradientAccumulator)
└── ... (existing files)

tests/
├── ane_backward_unit_tests.rs (NEW — unit tests for each generator)
├── ane_backward_integration_tests.rs (NEW — end-to-end backward tests)
└── ... (existing files)

examples/
├── train_transformer_ane_full.rs (NEW — complete training example with ANE backward)
└── ... (existing examples)
```

---

## Implementation Phases

### Phase 3a: MIL Backward Generators (1-2 weeks)

**Objectives:**
- Implement 4 backward MIL generators
- Port mathematical operations from CPU reference implementations
- Generate valid MIL code for each operation
- Unit test each generator

**Tasks:**
1. Create `src/layers/backward/rmsnorm_backward_gen.rs`
   - Port RMSNorm backward logic from `transformer_backward.rs`
   - Generate MIL code for gradient computation
   - Test MIL output structure

2. Create `src/layers/backward/attention_backward_gen.rs`
   - Port attention backward (dQ, dK, dV, dO)
   - Handle cached Q, K, V from forward pass
   - Generate MIL code for attention backward

3. Create `src/layers/backward/ffn_backward_gen.rs`
   - Port FFN backward (linear2, activation, linear1)
   - Generate MIL code for gradient flow

4. Create `src/layers/backward/loss_backward_gen.rs`
   - Port cross-entropy backward
   - Generate MIL code for loss gradient

5. Create `src/layers/backward/mod.rs` with `BackwardMILGenerator` trait

**Success Criteria:**
- All 4 generators implement BackwardMILGenerator
- MIL output compiles without errors
- Unit tests verify output structure
- No numerical validation yet (deferred to Phase 3b)

### Phase 3b: Validation Suite (1 week)

**Objectives:**
- Build reference validation suite
- Compare ANE backward outputs vs CPU reference
- Verify 1e-6 relative tolerance

**Tasks:**
1. Create `src/layers/backward/validation.rs`
   - Implement BackwardValidationSuite
   - Build reference config (small model for fast validation)

2. Create validation runner
   - Generate random inputs
   - Run each backward kernel on ANE
   - Run reference CPU backward
   - Compare outputs with 1e-6 tolerance

3. Test validator on all 4 backward operations
   - RMSNorm backward
   - Attention backward
   - FFN backward
   - Loss backward

**Success Criteria:**
- All 4 backward kernels pass 1e-6 tolerance validation
- ValidationReport accurately reports pass/fail status
- Validation completes in < 1 second per operation

### Phase 3c: ANE Integration (1-2 weeks)

**Objectives:**
- Implement ANEGradientAccumulator
- Extend Model trait with backward_on_ane()
- Integrate backward kernels into TransformerANE
- Update training pipeline

**Tasks:**
1. Create `src/training/ane_backward_executor.rs`
   - Implement ANEGradientAccumulator
   - Manage IOSurface for gradient storage
   - Handle accumulation logic

2. Extend Model trait in `src/training/transformer_model.rs`
   - Add backward_on_ane() method
   - Implement in TransformerANE
   - Coordinate with ANE device
   - Handle activation caching

3. Update Trainer (existing)
   - Call backward_on_ane() when available
   - Fall back to CPU backward if needed
   - Pass accumulated gradients to optimizer

4. Update training loop in examples
   - Run backward_on_ane() instead of backward()
   - Show gradient accumulation in action

**Success Criteria:**
- backward_on_ane() returns correct gradients
- Gradients match optimizer expectations
- No memory leaks in ANE↔CPU transfers
- Training step completes successfully

### Phase 3d: Testing & Examples (1 week)

**Objectives:**
- Write comprehensive tests
- Create training examples
- Benchmark Phase 2 vs Phase 3
- Document validation process

**Tasks:**
1. Create `tests/ane_backward_unit_tests.rs`
   - Unit tests for each generator
   - Test MIL code generation
   - Test gradient computation

2. Create `tests/ane_backward_integration_tests.rs`
   - Forward→backward→optimizer end-to-end
   - Validate gradients at each step
   - Compare vs Phase 2 CPU backward

3. Create `examples/train_transformer_ane_full.rs`
   - Complete training example with ANE backward
   - Show validation suite running
   - Display training metrics
   - Demonstrate 5+ training steps

4. Create benchmark
   - Compare Phase 2 (CPU backward) vs Phase 3 (ANE backward)
   - Report throughput improvement
   - Measure memory usage

5. Documentation
   - Update module docs with Phase 3 context
   - Document validation process
   - Explain activation caching strategy

**Success Criteria:**
- 20+ integration tests passing
- Example trains for 10 steps without errors
- Benchmark shows measurable improvement
- All tests pass with 80%+ coverage

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Backward Scope** | Full ANE (all operations) | Maximize efficiency, minimize CPU↔ANE transfers |
| **Code Generation** | Port existing CPU backward to MIL | Leverage proven implementations, reduce risk |
| **Activation Caching** | Transfer to CPU after forward, back to ANE for backward | Simpler lifetime management, proven pattern |
| **Kernel Caching** | Compile once, validate at startup | Fast training loop, confident correctness |
| **Validation Tolerance** | Exact match (1e-6 relative error) | Strict correctness, catch algorithmic errors |
| **Precision** | FP16 activations + FP32 gradients | 50% memory savings, numerical stability |
| **API Integration** | Extend Model trait with backward_on_ane() | Opt-in to ANE, preserve Phase 2 CPU backward |
| **Gradient Accumulation** | On ANE in IOSurface | Minimize CPU↔ANE transfers |

---

## Validation Strategy

### Startup Validation

Runs once before any training begins:

```
1. Initialize BackwardValidationSuite
2. Create small reference config
   - hidden_dim = 256
   - num_heads = 8
   - num_layers = 2
   - vocab_size = 1024
3. Generate random batch (batch_size=2, seq_len=4)
4. For each backward operation (RMSNorm, Attention, FFN, Loss):
   a. Generate MIL code via BackwardMILGenerator
   b. Compile to ANE kernel
   c. Run on batch
   d. Compare ANE gradients vs CPU reference
   e. Check: max_relative_error(ane, cpu) < 1e-6
5. If all pass: ValidationReport { all: true, max_error: X }
6. If any fail: Return error with detailed mismatch
```

### During Training

No per-step validation. Kernels already validated at startup.

Optional: Periodic checkpoints compare accumulated gradients to CPU reference (for regression detection).

---

## Integration with Phase 2

**No breaking changes to Phase 2:**
- Phase 2 forward pass unchanged
- Phase 2 CPU backward still available
- New backward_on_ane() is opt-in

**Activation caching strategy compatible:**
- Phase 2 caches activations in IOSurface during forward
- Phase 3 backward kernels read same IOSurface
- Seamless integration

---

## Success Criteria

1. ✅ All 4 backward MIL generators implemented and tested
2. ✅ Backward validation suite passes 1e-6 tolerance for all operations
3. ✅ Model trait extended with backward_on_ane() method
4. ✅ ANEGradientAccumulator manages gradient accumulation correctly
5. ✅ End-to-end training (forward→backward→optimizer) works on ANE
6. ✅ 20+ integration tests passing
7. ✅ Training example runs 10+ steps without errors
8. ✅ Backward performance measurably faster than Phase 2 CPU backward
9. ✅ All tests pass with 80%+ code coverage
10. ✅ Documentation complete and accurate

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MIL kernel generation bugs | Startup validation suite with strict 1e-6 tolerance |
| Memory leaks in IOSurface transfers | Reference counting in ANEGradientAccumulator, tests with leak detection |
| Gradient accumulation overflow | Use FP32 for gradients (higher precision), monitor max values |
| Mismatched activation caching | Same IOSurface strategy as Phase 2, proven in Phase 2 tests |
| ANE→CPU transfer bottleneck | Batch transfers, profile vs CPU backward, acceptable trade-off |

---

## Timeline

- **Phase 3a:** MIL Backward Generators — 1-2 weeks
- **Phase 3b:** Validation Suite — 1 week
- **Phase 3c:** ANE Integration — 1-2 weeks
- **Phase 3d:** Testing & Examples — 1 week
- **Total:** 4-6 weeks to full ANE backward training

---

## Future Extensions (Phase 4+)

- Gradient checkpointing for large models (trade computation for memory)
- Distributed training across multiple ANE devices
- Quantization-aware training (gradient scaling for lower precision)
- Custom backward kernels for specific operations
- Automatic differentiation framework for higher-level abstractions
