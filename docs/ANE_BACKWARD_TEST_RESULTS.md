# ANE Backward Pass Test Results

## Overview

This document summarizes the ANE (Apple Neural Engine) backward pass testing for SDPA (Scaled Dot-Product Attention) gradient computation, including the solution to the dpf compilation issue and validation with parameter-golf configuration.

## Problem Solved

The standalone `bwd_sdpa_bwd1_dpf_mil` function (computing dP gradient) failed to compile with:
```
ane_bridge: ANE compile failed: ... CompilationFailure / InvalidMILProgram
```

**Root Cause**: ANE compiler rejects `matmul -> reshape` pattern in standalone single-output programs.

**Solution**: Use `bwd_sdpa_bwd1_combined_mil` which concatenates dvf+pf+dpf outputs into a single tensor, matching the reference implementation (`stories_mil.h`).

## Test Results

### 1. Basic Validation (dim=64, seq=16, heads=4)

| Test | dvf | pf | dpf | Status |
|------|-----|----|----|--------|
| Combined (bwd1) | [0.2732, 0.5791, ...] | [0.0567, 0.0577, ...] | [64.8750, 66.0625, ...] | PASS |
| BWD2 (dqf+dkf) | [-0.0166, -1.0088, ...] | [-3.4023, -3.4590, ...] | N/A | PASS |
| PF Standalone | [0.0567, 0.0577, ...] | - | - | PASS |
| DVF Standalone | [0.2732, 0.5791, ...] | - | - | PASS |
| DPF Standalone | - | - | - | FAIL (expected) |

### 2. Parameter-Golf Configuration (dim=416, heads=8, head_dim=52)

| Test | Seq Len | dvf | pf | dpf | Status |
|------|---------|-----|----|----|--------|
| Synthetic Data | 256 | [0.2632, 0.5513, ...] | [0.0055, 0.0057, ...] | [1250.0, 1275.0, ...] | PASS |
| Synthetic Data | 1024 | [0.0464, 0.0970, ...] | [0.0010, 0.0010, ...] | [1265.0, 1290.0, ...] | PASS |
| Real Tokens | 256 | [0.0192, 0.0438, ...] | [0.0039, 0.0039, ...] | [12.64, 16.55, ...] | PASS |

### 3. Performance Metrics

| Configuration | Seq Len | Compile Time | Execution Time | Throughput |
|---------------|---------|--------------|----------------|------------|
| Small (dim=64) | 16 | 0.05s | 0.0003s | ~53K tok/s |
| Parameter-Golf | 256 | 0.14s | 0.001s | ~470K tok/s |
| Parameter-Golf | 1024 | 0.11s | 0.008s | ~128K tok/s |
| Real Data | 256 | 0.21s | 0.001s | ~470K tok/s |

## Test Files

| File | Purpose | Status |
|------|---------|--------|
| `examples/test_backward_sdpa_bwd1_combined.rs` | Combined dvf+pf+dpf validation | PASS |
| `examples/test_backward_sdpa_bwd1_lite.rs` | DVF+PF standalone validation | PASS |
| `examples/test_backward_sdpa_bwd2.rs` | DQF+DKF validation | PASS |
| `examples/test_ane_parameter_golf_bwd1.rs` | Parameter-golf config test | PASS |
| `examples/test_ane_with_real_data.rs` | Real token sequences test | PASS |

## Usage Examples

### Basic Test
```bash
cargo run --example test_backward_sdpa_bwd1_combined --release
```

### Parameter-Golf Test (Synthetic)
```bash
# Fast test (seq=256)
cargo run --example test_ane_parameter_golf_bwd1 --release

# Full sequence (seq=1024)
cargo run --example test_ane_parameter_golf_bwd1 --release -- --seq-len 1024
```

### Real Data Test
```bash
# Uses default: ~/dev/parameter-golf/data
cargo run --example test_ane_with_real_data --release

# Custom path
PARAMETER_GOLF_DATA=/custom/path cargo run --example test_ane_with_real_data --release
```

## API Usage

```rust
use rustane::mil::{bwd_sdpa_bwd1_combined_mil, bwd_sdpa_bwd1_combined_compile_request};
use rustane::wrapper::ANECompiler;

// Parameter-golf configuration
let dim = 416;
let seq = 256;
let heads = 8;
let head_dim = dim / heads; // 52

// Generate MIL
let mil = bwd_sdpa_bwd1_combined_mil(seq, dim, heads, head_dim);

// Create weight blobs
let wot = WeightBlob::from_f32(&wot_data, dim, dim)?;
let mask = WeightBlob::from_f32(&mask_data, seq, seq)?;

// Create compile request
let req = bwd_sdpa_bwd1_combined_compile_request(seq, dim, heads, head_dim, &wot, &mask);

// Compile and execute
let mut compiler = ANECompiler::new();
let executor = compiler.compile_multi(
    &mil,
    &["@model_path/weights/wot.bin", "@model_path/weights/mask.bin"],
    &[wot.as_ref(), mask.as_ref()],
    &[wot.len(), mask.len()],
    &req.input_sizes,
    &req.output_sizes,
)?;

// Read output: [1, DIM + 2*SCORE_CH, 1, SEQ]
// Split into: dvf[..dim*seq], pf[dim*seq..dim*seq+score_ch*seq], dpf[...]
```

## Output Tensor Layout

```
Combined Output Shape: [1, DIM + 2*SCORE_CH, 1, SEQ]

Where:
- DIM = model dimension (416 for parameter-golf)
- SCORE_CH = heads * seq_len (8 * 256 = 2048 for parameter-golf seq=256)
- Total channels = 416 + 2*2048 = 4512 (for parameter-golf seq=256)

Layout:
[0 .. dim*seq]           -> dvf (dV gradient)
[dim*seq .. dim*seq + score_ch*seq] -> pf (attention probs)
[dim*seq + score_ch*seq ..] -> dpf (dP gradient)
```

## Integration Status

The `bwd_sdpa_bwd1_combined_mil` function is:
- Implemented in `src/mil/programs.rs`
- Exported from `src/mil/mod.rs`
- Fully tested with parameter-golf configuration
- Validated with both synthetic and real token data

### Current Training Pipeline

The existing ANE backward pass in `src/training/transformer_model.rs` uses `AttentionBackwardGen` which computes weight gradients (d_wq, d_wk, d_wv, d_wo) directly. The `bwd_sdpa_bwd1_combined_mil` function provides the intermediate gradients (dvf, pf, dpf) that can be used for:

1. Fine-grained gradient computation
2. Debugging and validation
3. Alternative backward pass implementations
4. Memory-efficient training with gradient checkpointing

## References

- `docs/DPF_COMPILATION_ISSUE.md` - Detailed dpf issue analysis
- `src/mil/programs.rs:1191` - `bwd_sdpa_bwd1_combined_mil` implementation
- `examples/test_ane_parameter_golf_bwd1.rs` - Parameter-golf test
- `examples/test_ane_with_real_data.rs` - Real data test
