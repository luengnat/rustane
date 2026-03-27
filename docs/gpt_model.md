# GPT Model Implementation

## Overview

This directory contains the GPT model implementation for rustane, based on the `train_gpt.py` reference implementation from the parameter-golf project.

## Architecture

The implementation follows the train_gpt.py architecture exactly:

| Parameter | Value | Description |
|-----------|-------|-------------|
| vocab_size | 1024 | SentencePiece BPE vocabulary |
| num_layers | 11 | Transformer blocks |
| model_dim | 416 | Hidden dimension (d_model) |
| num_heads | 8 | Query attention heads |
| num_kv_heads | 4 | KV heads (GQA ratio 2:1) |
| mlp_mult | 2 | MLP expansion multiplier |
| tie_embeddings | true | Tie token embeddings and LM head |
| rope_base | 10000.0 | RoPE frequency base |
| logit_softcap | 30.0 | Logit scaling via tanh |
| qk_gain_init | 1.5 | QK gain initialization |

## Components

### `GptConfig`

Configuration struct with all hyperparameters. Includes:
- `num_params()` - Total parameter count
- `head_dim()` - Dimension per attention head
- `validate()` - Configuration validation

### `build_transformer_block()`

Creates a MIL graph for a single transformer block containing:
- Pre-attention RMSNorm
- Q, K, V projections (with GQA)
- Attention output projection
- Pre-MLP RMSNorm
- MLP with relu² activation (or SwiGLU)
- Residual connections

**Note**: Current implementation is simplified - RoPE and attention softmax require custom kernels not yet available in Graph IR.

### `build_gpt_model()`

Creates the full model graph:
- Token embeddings
- Transformer blocks (num_layers)
- Final RMSNorm
- Output projection (tied to embeddings)
- Logit softcap (pending)

### `print_model_summary()`

Displays model architecture and parameter count.

## Parameter Count

Default configuration: **13,781,336 parameters**
- bf16 size: ~26 MB
- f32 size: ~52 MB

## Memory Usage

| Batch Size | Seq Len | Activation Memory |
|------------|---------|-------------------|
| 1 | 64 | 1.12 MB |
| 32 | 64 | 35.75 MB |

## Implementation Status

### Complete
- [x] GptConfig with train_gpt.py defaults
- [x] Parameter counting
- [x] Configuration validation
- [x] MIL graph construction
- [x] RMSNorm support in Graph IR
- [x] Module exports and integration
- [x] QKV projections with GQA
- [x] MLP with relu² activation
- [x] Residual connections
- [x] Logit softcap (tanh scaling)
- [x] RoPE placeholder (reshape ops, rotary pending)

### Pending
- [ ] Full RoPE implementation - requires sin/cos tables and element-wise ops
- [ ] Full attention (Q@K^T, softmax, @V) - requires proper head separation
- [ ] Encoder/decoder skip connections
- [ ] Complete forward pass with ANE integration
- [ ] Muon optimizer integration
- [ ] Int8 quantization for export

## Usage

### Example

```rust
use rustane::model::{GptConfig, build_gpt_model, print_model_summary};

let config = GptConfig::default();
print_model_summary(&config);

// Validate configuration
config.validate()?;

// Build model
let graph = build_gpt_model(&config, seq_len: 1024)?;
```

### CLI Demo

```bash
# Show model summary and build graphs
cargo run --example gpt_compile --release
```

## Files

- `src/model/gpt.rs` - Main GPT implementation
- `src/model/mod.rs` - Module exports
- `examples/gpt_compile.rs` - Compilation demo

## References

- `train_gpt.py` - Reference implementation
- `docs/data_analysis_summary.md` - Dataset analysis
- `src/data/loader.rs` - Compatible data loader
