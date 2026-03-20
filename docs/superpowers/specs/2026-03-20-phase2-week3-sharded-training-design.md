# Phase 2 Week 3: Efficient Sharded Training with Gradient Accumulation and Token Chunking

**Date:** 2026-03-20
**Status:** Design Approved
**Context:** Phase 1 complete (data pipeline, 39 tests). Phase 2 Week 1 complete (LR schedulers, 9 tests). Phase 2 Week 2 complete (MVP Trainer, 38 tests). Total: 197 tests passing.

## Overview

This design implements three composable layers for efficient training on 200+ tokenized data shards:
1. **ShardedDataLoader** - Stream shard files from disk without loading all into memory
2. **Batch Chunking** - Split batches into token-aligned sub-batches for gradient accumulation
3. **Enhanced GradAccumulator** - Track and scale gradients across multiple backward passes

Inspired by parameter-golf's approach, these layers work together to enable efficient training across large datasets while maintaining the MVP Trainer's simple, unchanged API.

**Philosophy:** Composable layers below the Trainer. User explicitly orchestrates the training loop using high-level abstractions (shards, chunks, accumulation), while the Trainer remains a simple, single-step orchestrator.

## Architecture

```
Training Loop (user code):
    for shard in loader.iter_shards()?
        for chunk in shard.chunks(max_tokens)?
            metrics = trainer.train_accumulated_steps(chunks, accum_steps)?

Components:
    ShardedDataLoader (disk → shard iterators)
        ↓
    Batch.chunks() (batch → token-aligned sub-batches)
        ↓
    trainer.train_accumulated_steps() (orchestrates accumulation)
        ↓
    GradAccumulator (scales and accumulates gradients)
        ↓
    Optimizer.step() (applies accumulated gradients)
```

## Core Components

### 1. ShardedDataLoader

**File:** `src/data/sharded_loader.rs`

```rust
/// Configuration for shard loading
pub struct ShardConfig {
    /// Glob pattern for shard files (e.g., "data/shards/*.bin")
    pub shard_pattern: String,

    /// Expected vocabulary size (for validation)
    pub vocab_size: u32,

    /// Optional metadata about each shard
    pub shard_metadata: Option<Vec<ShardMetadata>>,
}

pub struct ShardMetadata {
    pub shard_idx: usize,
    pub token_count: usize,
    pub path: String,
}

/// Loads tokenized data from multiple shard files on disk
pub struct ShardedDataLoader {
    shard_files: Vec<PathBuf>,
    current_shard_idx: usize,
    current_shard_loader: Option<DataLoader>,
    config: ShardConfig,
}

impl ShardedDataLoader {
    /// Create new sharded loader from glob pattern
    pub fn new(shard_pattern: &str, config: ShardConfig) -> Result<Self>;

    /// Get total number of shards
    pub fn shard_count(&self) -> usize;

    /// Iterate over all shards
    pub fn iter_shards(&mut self) -> Result<ShardIterator>;
}

pub struct ShardIterator {
    parent: &mut ShardedDataLoader,
    done: bool,
}

impl Iterator for ShardIterator {
    type Item = Result<ShardBatch>;
}

pub struct ShardBatch {
    pub shard_idx: usize,
    pub shard_path: PathBuf,
    pub loader: DataLoader,
    pub token_count: usize,
}
```

**Design rationale:**
- Discovers shard files at creation time using glob pattern
- Loads one shard at a time (never all in memory)
- Iterator interface for clean control flow
- Tracks shard metadata for progress/logging
- Transparent to user: they just iterate

---

### 2. Batch Chunking (Batch Enhancement)

**File:** `src/data/mod.rs` (extend existing Batch)

```rust
impl Batch {
    /// Split batch into token-aligned chunks
    ///
    /// # Arguments
    /// - `max_chunk_tokens`: Maximum tokens per chunk (e.g., 2048)
    ///
    /// # Returns
    /// Vec<Batch> where each has <= max_chunk_tokens, aligned to seq_len
    pub fn into_chunks(self, max_chunk_tokens: usize) -> Result<Vec<Batch>>;

    /// Lazy iterator over chunks (memory-efficient)
    pub fn chunks(&self, max_chunk_tokens: usize) -> Result<ChunkIterator>;
}

pub struct ChunkIterator {
    original_batch: Batch,
    chunk_size: usize,
    current_pos: usize,
}

impl Iterator for ChunkIterator {
    type Item = Result<Batch>;
}

/// Helper: compute token-aligned chunk sizes
fn compute_chunk_sizes(
    total_tokens: usize,
    seq_len: usize,
    max_chunk_tokens: usize,
) -> Vec<usize>;
```

**Algorithm:**
1. `usable_chunk = (max_chunk_tokens / seq_len) * seq_len` (align to seq_len)
2. Divide batch into chunks of `usable_chunk` size
3. Respect token boundaries (don't split sequences)
4. Return iterator yielding Batch objects

**Design rationale:**
- Respects sequence length alignment (critical for transformer models)
- Iterator variant for memory efficiency with huge batches
- Transparent: user just calls `.chunks()`

---

### 3. Enhanced GradAccumulator

**File:** `src/training/grad_accum.rs` (extend existing)

```rust
/// Gradient accumulator with multi-step tracking
pub struct GradAccumulator {
    accumulated_grads: Vec<f32>,
    steps_completed: u32,
    total_steps: u32,
    accumulated_loss: f32,
}

impl GradAccumulator {
    /// Create new accumulator for N gradient accumulation steps
    pub fn new(param_count: usize, accumulation_steps: u32) -> Self;

    /// Accumulate gradients from one backward pass
    ///
    /// # Arguments
    /// - `grads`: Gradient vector from model.backward()
    /// - `loss`: Loss value (for averaging)
    /// - `scale`: Scaling factor (usually 1.0 / accumulation_steps)
    pub fn accumulate(&mut self, grads: &[f32], loss: f32, scale: f32) -> Result<()>;

    /// Check if accumulation is complete
    pub fn is_ready(&self) -> bool;

    /// Get accumulated gradients (for optimizer)
    pub fn gradients(&self) -> &[f32];

    /// Get average loss
    pub fn average_loss(&self) -> f32;

    /// Reset for next accumulation cycle
    pub fn reset(&mut self);

    /// Get progress (completed, total)
    pub fn progress(&self) -> (u32, u32);
}
```

**Design rationale:**
- Scales gradients during accumulation (not after)
- Tracks loss separately for averaging
- Clear `.is_ready()` signal for optimizer step
- Caller controls flow (when to apply optimizer)

---

### 4. Trainer Enhancement

**File:** `src/training/trainer.rs` (extend existing Trainer)

Add one new method:

```rust
impl<'a, M: Model> Trainer<'a, M> {
    /// Train with explicit gradient accumulation over chunks
    ///
    /// # Arguments
    /// - `chunks`: Iterator yielding Batch chunks
    /// - `accumulation_steps`: Number of steps before optimizer
    ///
    /// # Returns
    /// StepMetrics with aggregated loss and grad norms
    pub fn train_accumulated_steps<I>(
        &mut self,
        chunks: I,
        accumulation_steps: u32,
    ) -> Result<StepMetrics>
    where
        I: IntoIterator<Item = Result<Batch>>,
    {
        let mut accum = GradAccumulator::new(
            self.model.param_count(),
            accumulation_steps,
        );
        let scale = 1.0 / accumulation_steps as f32;
        let mut metrics_list = Vec::new();

        for chunk in chunks {
            let chunk = chunk?;
            let metrics = self.train_step(&chunk)?;

            // Get gradients from model (note: need to expose last_grads or similar)
            let grads = self.model.backward(metrics.loss)?;
            accum.accumulate(&grads, metrics.loss, scale)?;
            metrics_list.push(metrics);
        }

        if !accum.is_ready() {
            return Err(Error::Other(
                format!("incomplete accumulation: {}/{} steps",
                    accum.progress().0, accum.progress().1)
            ));
        }

        // Apply accumulated gradients
        let lr = self.scheduler.get_lr(self.current_step);
        self.optimizer.step(accum.gradients(), self.model.parameters(), lr)?;
        self.current_step += 1;

        // Return aggregated metrics
        Ok(StepMetrics::new(
            accum.average_loss(),
            compute_accumulated_grad_norm(accum.gradients()),
            lr,
            self.current_step - 1,
        ))
    }
}
```

**Design rationale:**
- Single new method (minimal Trainer changes)
- Takes iterator for memory efficiency
- Handles gradient accumulation internally
- Returns aggregated StepMetrics

---

## User-Facing API

### Example: Train on 200 Shards with Chunking

```rust
use rustane::data::{ShardedDataLoader, ShardConfig};
use rustane::training::{GradAccumulator, Trainer, TrainerBuilder};

fn main() -> Result<()> {
    // Setup
    let config = ShardConfig {
        shard_pattern: "data/shards/*.bin".to_string(),
        vocab_size: 50257,
        shard_metadata: None,
    };
    let mut loader = ShardedDataLoader::new("data/shards/*.bin", config)?;

    let mut model = MyModel::new();
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(0.001))
        .with_scheduler(WarmupCosineScheduler::new(0.001, 500, 5000, 1e-5))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    let chunk_tokens = 2048;
    let grad_accum_steps = 4;

    // Training loop
    for shard in loader.iter_shards()? {
        let shard = shard?;

        // Iterate through batches in this shard
        for batch in shard.loader.iter_batches()? {
            // Split batch into chunks
            let chunks = batch.into_chunks(chunk_tokens)?;

            // Train with accumulation
            let metrics = trainer.train_accumulated_steps(
                chunks.into_iter().map(Ok),
                grad_accum_steps,
            )?;

            println!("Loss: {:.4}, LR: {:.6}", metrics.loss, metrics.learning_rate);
        }

        println!("Shard {} complete", shard.shard_idx);
    }

    Ok(())
}
```

---

## Error Handling

New error variants in codebase:

```rust
pub enum Error {
    // ... existing ...

    /// Shard file not found or unreadable
    ShardLoadFailed(String),

    /// Chunk computation error
    ChunkingError(String),

    /// Gradient accumulation state error
    AccumulationError(String),
}
```

---

## Testing Strategy

### Unit Tests

1. **ShardedDataLoader:**
   - `test_shard_discovery` - Glob pattern finds all shards
   - `test_shard_iteration` - Iterator yields shards in order
   - `test_shard_count` - Correct number of shards

2. **Batch Chunking:**
   - `test_chunks_alignment` - Chunks respect seq_len boundaries
   - `test_chunks_exact_division` - Chunks sum to original batch
   - `test_chunks_respects_max_tokens` - No chunk exceeds max
   - `test_chunks_iterator` - Iterator produces same result as vec

3. **GradAccumulator:**
   - `test_accumulation_scaling` - Gradients scaled correctly
   - `test_loss_averaging` - Loss averaged properly
   - `test_ready_signal` - `.is_ready()` at right step
   - `test_reset` - Reset clears state

4. **Trainer Integration:**
   - `test_train_accumulated_steps` - Single accumulation cycle
   - `test_train_multiple_chunks` - Accumulation over chunks
   - `test_gradient_scaling` - Accumulated gradients scaled

### Integration Tests

1. **test_shard_to_trainer** - Full pipeline (loader → chunks → trainer)
2. **test_200_shard_training** - Synthetic test with many shards
3. **test_convergence** - Loss decreases over epochs

### Example Tests

1. `examples/train_with_shards.rs` - Full working example

---

## Dependencies

### Already Exist
- ✅ `Batch` struct (Phase 1)
- ✅ `DataLoader` (Phase 1)
- ✅ `Dataset` trait (Phase 1)
- ✅ `Model` trait (Phase 2 Week 2)
- ✅ `LossFn` trait (Phase 2 Week 2)
- ✅ `Trainer` struct (Phase 2 Week 2)
- ✅ `Optimizer` trait (existing)
- ✅ `LRScheduler` trait (Phase 2 Week 1)

### New (This Design)
- `ShardedDataLoader` struct
- `ShardConfig` struct
- `ShardMetadata` struct
- `ChunkIterator` struct
- Enhanced `Batch` with `.chunks()`
- Enhanced `GradAccumulator` with accumulation tracking
- Enhanced `Trainer` with `.train_accumulated_steps()`
- Error types for sharding/chunking

### Updates Needed
- `src/data/mod.rs` - Add sharded_loader module, update Batch
- `src/training/mod.rs` - Export new types
- `src/lib.rs` - Export new types

---

## Files to Create/Modify

### Create
- `src/data/sharded_loader.rs` (~300 lines: ShardedDataLoader, ShardBatch, iterator)
- `src/data/chunk_iterator.rs` (~150 lines: ChunkIterator, compute_chunk_sizes)
- `tests/sharded_training_integration.rs` (~400 lines: integration tests)
- `examples/train_with_shards.rs` (~350 lines: full example with synthetic shards)

### Modify
- `src/data/mod.rs` - Add modules, add `into_chunks()` and `chunks()` to Batch
- `src/data/batch.rs` - Add chunking methods
- `src/training/grad_accum.rs` - Enhance with accumulation tracking
- `src/training/trainer.rs` - Add `train_accumulated_steps()`
- `src/lib.rs` - Export new types

---

## Success Criteria

- ✅ Code compiles with no errors or warnings
- ✅ 25+ new tests pass (unit + integration)
- ✅ Example runs without errors
- ✅ Total: 220+ tests passing (197 existing + 25 new)
- ✅ Backward compatible: all existing tests still pass
- ✅ New code follows rustane patterns (error handling, trait-based, immutable)

---

## Future Extensions (Phase 2 Week 4+)

**ANE Profiler:**
- Add `profile_step()` variant returning `(metrics, profile_data)`
- Track forward/backward/optimizer timing per step
- Measure ANE utilization and memory

**Checkpointing:**
- Save/load training state (model weights, optimizer state, step counter)
- Enable training resumption mid-epoch

**Data Preprocessing:**
- Optional pre-tokenization verification
- Shard integrity checking

---

## Notes

- This design maintains 100% backward compatibility with MVP Trainer
- ShardedDataLoader works with existing DataLoader infrastructure
- Chunking is transparent to model (just splits Batch)
- GradAccumulator is orthogonal (can be used independently)
- Design inspired by parameter-golf's efficient training patterns
- Focus on user ergonomics: high-level abstractions that compose naturally
