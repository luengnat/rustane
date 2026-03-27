# Mixed Precision Training Guide

## Overview

Mixed precision training combines FP32 master weights with FP16/BF16 working precision to achieve:
- **Reduced memory usage** (50% savings on weights + activations)
- **Faster computation** (FP16/BF16 matmul is 2-8x faster than FP32 on ANE)
- **Numerical stability** (FP32 master weights prevent accumulation errors)

## Precision Formats

| Format | Bits | Dynamic Range | Use Case |
|--------|------|---------------|----------|
| FP32 | 32-bit | ±10³⁸ | Master weights, numerical stability |
| FP16 | 16-bit | ±65504 | Working precision, requires loss scaling |
| BF16 | 16-bit | ±10³⁸ | Working precision, no loss scaling needed |

## Memory Savings

For a 1B parameter model:

| Component | FP32 Only | Mixed Precision | Savings |
|-----------|-----------|-----------------|---------|
| Weights | 4 GB | 2 GB (working) + 4 GB (master) | 33% |
| Activations | 8 GB | 4 GB | 50% |
| Gradients | 4 GB | 2 GB | 50% |
| **Total** | **16 GB** | **10 GB** | **~37%** |

With gradient checkpointing: **~60-70% total memory reduction**

## Quick Start

### Basic Usage

```rust
use rustane::training::mixed_precision::{MixedPrecisionState, MasterWeights};

// Create mixed precision state (FP16)
let mut state = MixedPrecisionState::new_fp16(num_params, 256.0);

// Or BF16 (no loss scaling needed)
let mut state = MixedPrecisionState::new_bf16(num_params);

// Get working weights for forward pass
let working = &state.working;

// Forward pass using FP16 weights
let output = model.forward_fp16(&input, working.as_fp16())?;

// Backward pass (gradients computed in FP16)
let mut grads_fp16 = model.backward(&output, &target)?;

// Convert gradients to FP32
let mut grads: Vec<f32> = rustane::training::mixed_precision::fp16_slice_to_f32(&grads_fp16);

// Scale loss for FP16 training
let scaled_loss = state.scale_loss(loss);

// Complete training step
let success = state.complete_step(&mut grads, learning_rate)?;
```

### Integration with TransformerConfig

```rust
use rustane::training::{TransformerConfig, MixedPrecisionConfig, Precision};

// Create configuration with FP16 mixed precision
let config = TransformerConfig::new(vocab_size, dim, hidden_dim, n_heads, n_layers, seq_len)?
    .with_mixed_precision(MixedPrecisionConfig::fp16());

// Or use the shortcut
let config = TransformerConfig::new(vocab_size, dim, hidden_dim, n_heads, n_layers, seq_len)?
    .with_fp16();

// BF16 configuration
let config = TransformerConfig::new(vocab_size, dim, hidden_dim, n_heads, n_layers, seq_len)?
    .with_bf16();

// Create model with mixed precision
let model = TransformerANE::new(&config)?;
```

## API Reference

### Precision Conversion Utilities

```rust
use rustane::training::mixed_precision::*;

// Single value conversion
let fp16 = f32_to_fp16(3.14);
let bf16 = f32_to_bf16(3.14);
let back = fp16_to_f32(fp16);

// Slice conversion
let fp16_vec = f32_slice_to_fp16(&fp32_values);
let bf16_vec = f32_slice_to_bf16(&fp32_values);
let back = fp16_slice_to_f32(&fp16_vec);
```

### MasterWeights

FP32 weight storage with optimizer state:

```rust
// Create master weights
let mut master = MasterWeights::new(num_params);

// With Adam optimizer state
let mut master = MasterWeights::new(num_params).with_adam_state();

// Initialize weights
master.xavier_init(fan_in, fan_out);
master.zero();

// Apply gradients
master.apply_sgd(&grads, lr);
master.apply_sgd_momentum(&grads, lr, 0.9);
master.apply_adam(&grads, lr, 0.9, 0.999, 1e-8, step);

// Access weights
let weights: &[f32] = master.as_slice();
let weights_mut: &mut [f32] = master.as_mut_slice();
```

### WorkingWeights

Reduced precision view of master weights:

```rust
// Create from master
let working_fp16 = WorkingWeights::from_master_fp16(&master);
let working_bf16 = WorkingWeights::from_master_bf16(&master);

// Access precision-specific views
let fp16_slice: &[f16] = working.as_fp16();
let bf16_slice: &[bf16] = working.as_bf16();

// Convert to FP32
let fp32_weights = working.to_f32();

// Sync from master (call after gradient update)
working.sync_from_master(&master);
```

### MixedPrecisionState

Complete mixed precision training state:

```rust
// Create FP16 state with loss scaling
let mut state = MixedPrecisionState::new_fp16(num_params, 256.0);

// Create BF16 state (no loss scaling needed)
let mut state = MixedPrecisionState::new_bf16(num_params);

// Check precision type
assert!(state.is_fp16());
assert!(!state.is_bf16());

// Scale loss for FP16 training
let scaled_loss = state.scale_loss(loss);

// Unscale gradients
state.unscale_grads(&mut grads);

// Apply gradients and complete step
let success = state.complete_step(&mut grads, lr)?;

// Get memory usage
let bytes = state.memory_bytes();
let ratio = state.memory_savings_ratio(); // ~0.75 = 25% savings
```

## Training Loop Integration

### Full Training Step

```rust
use rustane::training::mixed_precision::MixedPrecisionState;

struct MixedPrecisionTrainer {
    model: TransformerANE,
    state: MixedPrecisionState,
}

impl MixedPrecisionTrainer {
    fn train_step(&mut self, batch: &Batch, lr: f32) -> Result<f32> {
        // Forward pass (FP16)
        let output = self.model.forward_fp16(batch, self.state.working.as_fp16())?;

        // Compute loss
        let loss = compute_loss(&output, &batch.targets)?;

        // Scale loss for FP16
        let scaled_loss = self.state.scale_loss(loss);

        // Backward pass
        let mut grads_fp16 = self.model.backward_fp16(&scaled_loss)?;

        // Convert gradients to FP32
        let mut grads = fp16_slice_to_f32(&grads_fp16);

        // Complete training step
        let success = self.state.complete_step(&mut grads, lr)?;

        if !success {
            // Gradient overflow - skip this step
            println!("Gradient overflow, skipping step");
        }

        Ok(loss)
    }
}
```

### With Gradient Accumulation

```rust
use rustane::training::{GradAccumulator, mixed_precision::MixedPrecisionState};

struct AccumulatedTrainer {
    model: TransformerANE,
    state: MixedPrecisionState,
    accum: GradAccumulator,
    accum_steps: usize,
}

impl AccumulatedTrainer {
    fn train_step(&mut self, batch: &Batch, lr: f32) -> Result<f32> {
        // Forward pass
        let output = self.model.forward(batch)?;
        let loss = compute_loss(&output, &batch.targets)?;
        let scaled_loss = self.state.scale_loss(loss);

        // Backward pass
        let grads_fp16 = self.model.backward(&scaled_loss)?;
        let mut grads = fp16_slice_to_f32(&grads_fp16);
        self.state.unscale_grads(&mut grads);

        // Accumulate gradients
        self.accum.add(&grads)?;

        // Apply gradients every accum_steps
        if self.accum.is_full() {
            let accumulated = self.accum.get_accumulated()?;
            self.state.apply_gradients(&accumulated, lr);
            self.state.sync_working_weights();
            self.accum.reset();
        }

        Ok(loss)
    }
}
```

## Loss Scaling for FP16

FP16 has limited dynamic range (±65504). Small gradients can underflow to zero.

### How Loss Scaling Works

1. **Scale up**: Multiply loss by scale factor (e.g., 256)
2. **Backward pass**: Gradients are also scaled up
3. **Unscale**: Divide gradients by scale factor
4. **Adjust**: Increase scale if gradients valid, decrease on overflow

### LossScaler Configuration

```rust
use rustane::training::LossScaler;

// Default configuration
let mut scaler = LossScaler::new(256.0);

// Transformer-specific (scales with model depth)
let mut scaler = LossScaler::for_transformer(n_layers);

// Custom parameters
let mut scaler = LossScaler::new(initial_scale)
    .with_growth_params(
        2.0,    // growth_factor: double on success
        0.5,    // backoff_factor: halve on overflow
        2000,   // growth_interval: try growth every 2000 steps
    );
```

### BF16 Advantage

BF16 has the same dynamic range as FP32 (±10³⁸), so **loss scaling is not required**:

```rust
// BF16 doesn't need loss scaling
let state = MixedPrecisionState::new_bf16(num_params);
assert!(state.loss_scaler().is_none());
```

## Performance Guidelines

### When to Use Mixed Precision

**Use FP16 when:**
- Memory is constrained
- ANE FP16 throughput > FP32 (typically 2-8x faster)
- Model is large enough to benefit

**Use BF16 when:**
- Available on hardware (M2/M3/M4 chips)
- Want FP16 benefits without loss scaling complexity
- Training unstable with FP16

**Stick with FP32 when:**
- Small models (< 10M parameters)
- Numerical precision is critical
- Debugging/troubleshooting

### Optimal Configuration

For typical transformer training on Apple Silicon:

```rust
let config = TransformerConfig::new(vocab_size, dim, hidden_dim, n_heads, n_layers, seq_len)?
    .with_fp16()  // or .with_bf16()
    .with_gradient_checkpointing(GradientCheckpointingConfig::with_interval(2));
```

Expected results:
- **Memory reduction**: ~60-70%
- **Speedup**: 1.5-3x faster training
- **Similar convergence**: to FP32 training

## Best Practices

1. **Monitor gradient health**
   - Watch for `complete_step()` returning `false` (FP16 overflow)
   - Adjust initial loss scale if frequent overflows

2. **Warm up loss scale**
   - Start with conservative scale (64-256)
   - Let dynamic scaling find optimal value

3. **Sync working weights**
   - Always call `sync_working_weights()` after applying gradients
   - Ensures next forward pass uses updated weights

4. **Combine with other optimizations**
   - Gradient checkpointing for additional memory savings
   - Gradient accumulation for larger effective batch sizes
   - Operator fusion for reduced compile overhead

## Troubleshooting

### FP16 Gradient Overflow

**Symptom**: `complete_step()` returns `false` frequently

**Solutions**:
1. Reduce initial loss scale: `new_fp16(num_params, 64.0)`
2. Use BF16 instead: `new_bf16(num_params)`
3. Check for numerical issues in model

### Slow Convergence

**Symptom**: Mixed precision trains slower than FP32

**Solutions**:
1. Verify FP32 master weights are being updated correctly
2. Check learning rate (may need adjustment)
3. Try BF16 for better numerical properties

### Out of Memory

**Symptom**: Still running out of memory with mixed precision

**Solutions**:
1. Enable gradient checkpointing
2. Increase checkpoint frequency
3. Reduce batch size
4. Use gradient accumulation

## Related Documentation

- `docs/GRADIENT_CHECKPOINTING.md` - Memory optimization via activation checkpointing
- `docs/ANE_PROFILER_GUIDE.md` - Performance profiling
- `docs/ANE_OPERATOR_FUSION.md` - Operator fusion for compile savings
- `docs/ANE_TRAINING_ARCHITECTURE.md` - Overall training system design
