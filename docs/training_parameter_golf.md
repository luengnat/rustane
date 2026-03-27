# Training on Parameter-Golf Data

This document describes how to train GPT models on the parameter-golf dataset using rustane.

## Overview

The training infrastructure integrates:
- **Data Loader**: `DistributedTokenLoader` for streaming tokenized shards
- **Model**: `TransformerANE` with train_gpt.py-compatible architecture
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing with linear warmup
- **Trainer**: Full training loop with gradient accumulation

## Quick Start

```bash
# Basic training (will fail if data not found)
cargo run --example train_parameter_golf --release

# With custom data directory
cargo run --example train_parameter_golf --release -- \
    --data-dir /path/to/parameter-golf/data

# Full options
cargo run --example train_parameter_golf --release -- \
    --data-dir ~/dev/parameter-golf/data \
    --seq-len 1024 \
    --batch-tokens 65536 \
    --steps 10000 \
    --warmup 1000 \
    --lr 0.0003 \
    --grad-accum 8
```

## Architecture Configuration

The default transformer configuration matches train_gpt.py:

| Parameter | Value | Description |
|-----------|-------|-------------|
| vocab_size | 1024 | SentencePiece BPE vocabulary |
| model_dim | 416 | Hidden dimension |
| hidden_dim | 832 | MLP expansion (2x) |
| num_heads | 8 | Attention heads |
| num_layers | 11 | Transformer blocks |
| seq_len | 1024 | Sequence length |

Total parameters: ~13.8M

## Data Format

The parameter-golf dataset uses:
- **Shard format**: 256 int32 header + uint16 little-endian tokens
- **Magic number**: 20240520
- **Version**: 1
- **Token range**: 0-1023 (vocab_size)

Expected file pattern:
```
datasets/fineweb10B_sp1024/fineweb_train_*.bin
```

## Training Configuration

### Batch Size

Total batch tokens are split across gradient accumulation steps:

```
effective_batch = batch_tokens * grad_accum_steps
```

Default: 65,536 * 8 = 524,288 tokens per update

### Learning Rate Schedule

Cosine annealing with linear warmup:
- **Warmup**: Linear increase from 0 to peak_lr over warmup_steps
- **Decay**: Cosine annealing from peak_lr to min_lr
- **Default peak_lr**: 3e-4
- **Default min_lr**: 3e-5

### Gradient Clipping

Default gradient clipping norm: 1.0

## Training Loop

```rust
use rustane::training::{
    TrainerBuilder, AdamWOptimizer, WarmupCosineScheduler, CrossEntropyLoss,
    TransformerANE, TransformerConfig,
};
use rustane::data::DistributedTokenLoader;

// Create model
let config = TransformerConfig::new(1024, 416, 832, 8, 11, 1024)?;
let mut model = TransformerANE::new(&config)?;

// Create optimizer
let optimizer = AdamWOptimizer::new(model.param_count())
    .with_weight_decay(0.1);

// Create scheduler
let scheduler = WarmupCosineScheduler::new(
    3e-4,  // peak_lr
    1000,  // warmup_steps
    10000, // total_steps
    3e-5,  // min_lr
);

// Build trainer
let mut trainer = TrainerBuilder::new(&mut model)
    .with_optimizer(optimizer)
    .with_scheduler(scheduler)
    .with_loss_fn(CrossEntropyLoss)
    .with_grad_clip_norm(1.0)
    .build()?;

// Training loop
for batch in dataloader {
    let metrics = trainer.train_step(&batch)?;
    println!("Loss: {}", metrics.loss);
}
```

## Output Format

Training progress is logged every 100 steps:

```
Epoch 0 | Step 100 | Loss: 4.5234 | LR: 0.000030 | Grad Norm: 1.2345 | 12500 tok/s | ETA: 15.2m
```

Fields:
- **Loss**: Cross-entropy loss (per-token)
- **LR**: Current learning rate
- **Grad Norm**: Gradient norm (after clipping)
- **tok/s**: Tokens processed per second
- **ETA**: Estimated time to completion

## Troubleshooting

### Data Not Found

```
Warning: Data directory does not exist
```

Solution: Set the correct data directory:
```bash
cargo run --example train_parameter_golf --release -- \
    --data-dir /path/to/parameter-golf/data
```

### Out of Memory

If training fails with memory errors:
1. Reduce `--batch-tokens`
2. Increase `--grad-accum` to maintain effective batch size
3. Reduce `--seq-len`

Example:
```bash
cargo run --example train_parameter_golf --release -- \
    --batch-tokens 32768 \
    --grad-accum 16 \
    --seq-len 512
```

### NaN Loss

If loss becomes NaN:
1. Reduce learning rate
2. Enable loss scaling (mixed precision training)
3. Check for data corruption

## Performance Notes

- **ANE Acceleration**: Forward pass runs on Apple Neural Engine
- **CPU Backward**: Gradients computed on CPU (Phase 2)
- **Gradient Accumulation**: Enables larger effective batches
- **Token Streaming**: Efficient sequential data loading

## Next Steps

1. **Monitor Training**: Watch for loss convergence
2. **Checkpointing**: Save model weights periodically
3. **Validation**: Evaluate on held-out data
4. **Hyperparameter Tuning**: Adjust lr, batch size, etc.

## Related Files

- `examples/train_parameter_golf.rs` - Training example
- `src/data/loader.rs` - Data loading infrastructure
- `src/training/trainer.rs` - Training orchestration
- `src/training/transformer_model.rs` - Transformer implementation
- `docs/gpt_model.md` - GPT model architecture
