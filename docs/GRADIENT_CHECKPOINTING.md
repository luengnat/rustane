# Gradient Checkpointing Guide

## Overview

Gradient checkpointing (also known as activation checkpointing) is a memory optimization technique that reduces activation memory usage during training by recomputing intermediate results during the backward pass instead of storing them all.

## Memory-Compute Tradeoff

| Approach | Memory | Compute |
|----------|--------|---------|
| Standard Training | O(n) | O(1) |
| Gradient Checkpointing | O(√n) | O(2) |

For large models, gradient checkpointing can reduce memory usage by **4-8x** with only **20-30% compute overhead**.

## When to Use

**Use Gradient Checkpointing when:**
- Training large models that don't fit in memory
- Memory bandwidth is the bottleneck
- You can afford modest compute overhead for memory savings
- Batch size is limited by activation memory

**Avoid Gradient Checkpointing when:**
- Memory is not a constraint
- Compute is already the bottleneck
- Training small models (< 100M parameters)

## Checkpoint Strategies

### 1. Every Nth Layer

Saves activations at regular intervals:

```rust
use rustane::training::gradient_checkpoint::CheckpointStrategy;

// Save every 4th layer: 0, 4, 8, 12, ...
let strategy = CheckpointStrategy::every_n_layers(4);
```

**Memory ratio**: ~1/N of full storage

### 2. Block Boundaries

Saves at transformer block boundaries:

```rust
// Save at specific layer indices
let strategy = CheckpointStrategy::block_boundaries(vec![0, 4, 8, 12]);
```

**Use case**: When you want to checkpoint at natural architectural boundaries

### 3. Custom Strategy

User-defined checkpoint locations:

```rust
// Custom checkpoint pattern
let strategy = CheckpointStrategy::custom(vec![0, 2, 5, 9]);
```

## Usage

### Basic Example

```rust
use rustane::training::gradient_checkpoint::{CheckpointManager, CheckpointStrategy, Activation};

// Create checkpoint manager for 12-layer model
let strategy = CheckpointStrategy::every_n_layers(4);
let manager = CheckpointManager::new(strategy, 12);

// Forward pass with checkpointing
let mut ctx = manager.begin_forward();

for layer_idx in 0..12 {
    let output = layers[layer_idx].forward(&input)?;

    if ctx.should_save(layer_idx) {
        // Convert to fp16 for memory-efficient storage
        let activation = Activation::from_f32(&output, output_shape);
        ctx.save_activation(layer_idx, activation);
    }

    input = output;
}

let checkpoints = ctx.finish();
println!("Saved {} checkpoints", checkpoints.checkpoint_indices.len());
```

### Backward Pass with Recomputation

```rust
// Backward pass (reverse order)
for layer_idx in (0..12).rev() {
    if !manager.is_checkpoint(layer_idx) {
        // Need to recompute this layer's activation
        let (start, end) = manager.get_recompute_range(layer_idx, &checkpoints);

        // Recompute activations from start to end
        let mut current_activation = get_starting_activation(&checkpoints, start)?;
        for l in start..end {
            current_activation = layers[l].forward(&current_activation)?;
        }

        // Use recomputed activation for gradient computation
        let grad = compute_gradient(layer_idx, &current_activation)?;
    } else {
        // Activation is available from checkpoint
        let activation = checkpoints.get_activation(layer_idx).unwrap();
        let grad = compute_gradient(layer_idx, activation)?;
    }
}
```

### Using CheckpointBuilder

```rust
use rustane::training::gradient_checkpoint::CheckpointBuilder;

// Fluent builder API
let manager = CheckpointBuilder::new(12)
    .every_n_layers(4)
    .build()?;

// Print configuration summary
manager.print_summary();
```

Example output:
```
=== Gradient Checkpointing Configuration ===

Total layers: 12
Number of checkpoints: 4
Checkpoint indices: [0, 4, 8, 11]

Memory Efficiency:
  - Activation memory reduction: 66.7%
  - Memory ratio: 0.33x (vs full storage)
  - Estimated compute overhead: ~67%

Strategy: EveryNthLayer { n: 4 }
```

## API Reference

### Activation

Storage container for checkpointed tensors:

```rust
pub struct Activation {
    pub data: Vec<f16>,      // fp16 for memory efficiency
    pub shape: Vec<usize>,   // Tensor dimensions
    pub metadata: Option<String>,
}

impl Activation {
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self;
    pub fn from_f16(data: Vec<f16>, shape: Vec<usize>) -> Self;
    pub fn to_f32(&self) -> Vec<f32>;
    pub fn num_elements(&self) -> usize;
    pub fn memory_bytes(&self) -> usize;
}
```

### CheckpointStrategy

Determines which layers to checkpoint:

```rust
pub enum CheckpointStrategy {
    EveryNthLayer { n: usize },
    BlockBoundaries { indices: Vec<usize> },
    Custom { indices: Vec<usize> },
}

impl CheckpointStrategy {
    pub fn every_n_layers(n: usize) -> Self;
    pub fn block_boundaries(indices: Vec<usize>) -> Self;
    pub fn custom(indices: Vec<usize>) -> Self;
    pub fn should_save(&self, layer_idx: usize, total_layers: usize) -> bool;
    pub fn memory_ratio(&self, total_layers: usize) -> f64;
}
```

### CheckpointManager

Manages checkpoint lifecycle:

```rust
pub struct CheckpointManager {
    // ...
}

impl CheckpointManager {
    pub fn new(strategy: CheckpointStrategy, total_layers: usize) -> Self;
    pub fn is_checkpoint(&self, layer_idx: usize) -> bool;
    pub fn begin_forward(&self) -> CheckpointContext<'_>;
    pub fn get_recompute_range(&self, layer_idx: usize, checkpoints: &CheckpointData) -> (usize, usize);
    pub fn memory_savings_ratio(&self) -> f64;
    pub fn num_checkpoints(&self) -> usize;
    pub fn print_summary(&self);
}
```

### CheckpointContext

Forward pass checkpointing context:

```rust
pub struct CheckpointContext<'a> {
    // ...
}

impl<'a> CheckpointContext<'a> {
    pub fn new(strategy: &'a CheckpointStrategy, total_layers: usize) -> Self;
    pub fn should_save(&self, layer_idx: usize) -> bool;
    pub fn save_activation(&mut self, layer_idx: usize, activation: Activation);
    pub fn get_activation(&self, layer_idx: usize) -> Option<&Activation>;
    pub fn memory_usage(&self) -> usize;
    pub fn finish(self) -> CheckpointData;
}
```

### CheckpointData

Returned from forward pass, used during backward pass:

```rust
pub struct CheckpointData {
    pub activations: HashMap<usize, Activation>,
    pub checkpoint_indices: Vec<usize>,
}

impl CheckpointData {
    pub fn get_activation(&self, layer_idx: usize) -> Option<&Activation>;
    pub fn get_last_checkpoint_before(&self, layer_idx: usize) -> Option<(usize, &Activation)>;
    pub fn memory_bytes(&self) -> usize;
}
```

## Performance Analysis

### Memory Savings

For a 12-layer transformer with hidden_dim=768, seq_len=256, batch_size=32:

| Strategy | Checkpoints | Memory Usage | Savings |
|----------|-------------|--------------|---------|
| Full storage | 12 | 22.5 MB | - |
| Every 2nd layer | 6 | 11.3 MB | 50% |
| Every 4th layer | 4 | 7.5 MB | 67% |
| Every 6th layer | 3 | 5.6 MB | 75% |

### Compute Overhead

| Strategy | Recompute Ratio | Estimated Overhead |
|----------|-----------------|-------------------|
| Every 2nd layer | ~50% | +25% |
| Every 4th layer | ~75% | +37% |
| Every 6th layer | ~83% | +42% |

### Optimal Strategy

For **memory-bound training** (typical for large models):
- Use `every_n_layers(4)` for good balance
- Memory savings: ~67%
- Compute overhead: ~30-40%

For **compute-bound training**:
- Use `every_n_layers(2)` or disable checkpointing
- Memory savings: ~50%
- Compute overhead: ~20-25%

## Integration with Training Loop

```rust
use rustane::training::{Trainer, gradient_checkpoint::{CheckpointManager, CheckpointStrategy}};

struct CheckpointedTrainer {
    trainer: Trainer,
    checkpoint_manager: CheckpointManager,
}

impl CheckpointedTrainer {
    fn new(trainer: Trainer, num_layers: usize) -> Self {
        let strategy = CheckpointStrategy::every_n_layers(4);
        let checkpoint_manager = CheckpointManager::new(strategy, num_layers);
        Self { trainer, checkpoint_manager }
    }

    fn train_step(&mut self, batch: &Batch) -> Result<f32> {
        // Forward pass with checkpointing
        let mut ctx = self.checkpoint_manager.begin_forward();

        let mut hidden = self.embed(batch)?;
        for layer_idx in 0..self.num_layers {
            hidden = self.layers[layer_idx].forward(&hidden)?;

            if ctx.should_save(layer_idx) {
                ctx.save_activation(layer_idx, Activation::from_f32(
                    &hidden,
                    hidden.shape()
                ));
            }
        }

        let loss = self.compute_loss(&hidden, batch)?;
        let checkpoints = ctx.finish();

        // Backward pass with recomputation
        self.backward_with_recompute(&checkpoints)?;

        Ok(loss)
    }
}
```

## Best Practices

1. **Choose checkpoint interval based on memory pressure**
   - Start with `every_n_layers(4)`
   - Reduce to `every_n_layers(2)` if compute-bound
   - Increase to `every_n_layers(6)` if still memory-bound

2. **Profile memory usage**
   - Use `CheckpointManager::print_summary()` to see expected savings
   - Monitor actual memory during training

3. **Consider block boundaries**
   - For transformer models, checkpoint at block boundaries
   - Aligns with natural computation boundaries

4. **Combine with other memory optimizations**
   - Gradient accumulation
   - Mixed precision training
   - Activation compression (fp16 storage)

## Related Documentation

- `docs/ANE_TRAINING_ARCHITECTURE.md` - Training system overview
- `docs/ANE_PROFILER_GUIDE.md` - Performance profiling
- `docs/ANE_OPERATOR_FUSION.md` - Operator fusion for compile savings
