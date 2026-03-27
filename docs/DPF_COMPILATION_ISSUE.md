# DPF Compilation Issue

## Problem

The `bwd_sdpa_bwd1_dpf_mil` function (computing dP gradient) fails to compile as a standalone ANE program with error:

```
ane_bridge: ANE compile failed: ... CompilationFailure / InvalidMILProgram
```

## Working Functions

The following SDPA backward functions compile and execute correctly:
- `bwd_sdpa_bwd1_dvf_mil` - computes dV gradient
- `bwd_sdpa_bwd1_pf_mil` - computes forward attention probs
- `bwd_sdpa_bwd2_dqf_mil` - computes dQ gradient
- `bwd_sdpa_bwd2_dkf_mil` - computes dK gradient

## Pattern Analysis

**PF (works)**: `softmax -> reshape -> cast`
**DVF (works)**: `matmul -> transpose -> reshape -> cast`
**DPF (fails standalone)**: `matmul -> reshape -> cast`

The key difference is that DPF attempts to reshape a matmul output directly, while DVF has an intermediate transpose.

## Solution: Combined Output

**WORKING**: Use `bwd_sdpa_bwd1_combined_mil` which concatenates dvf+pf+dpf outputs together.

The ANE compiler accepts the matmul->reshape pattern when outputs are concatenated in a single program, matching the reference implementation (`stories_mil.h`).

### Usage Example

```rust
use rustane::mil::{bwd_sdpa_bwd1_combined_mil, bwd_sdpa_bwd1_combined_compile_request};

let dim = 64;
let seq = 16;
let heads = 4;
let head_dim = dim / heads;

// Create weight blobs
let wot = /* ... */;
let mask = /* ... */;

// Generate MIL and compile request
let mil = bwd_sdpa_bwd1_combined_mil(seq, dim, heads, head_dim);
let req = bwd_sdpa_bwd1_combined_compile_request(seq, dim, heads, head_dim, &wot, &mask);

// Compile
let mut compiler = ANECompiler::new();
let executor = compiler.compile_multi(
    &mil,
    &["@model_path/weights/wot.bin", "@model_path/weights/mask.bin"],
    &[wot.as_ref(), mask.as_ref()],
    &[wot.len(), mask.len()],
    &req.input_sizes,
    &req.output_sizes,
)?;

// Execute and read concatenated output [1, DIM + 2*SCORE_CH, 1, SEQ]
let output = read_output(&executor);

// Split into individual outputs
let dvf = &output[0..dim * seq];           // [1, DIM, 1, SEQ]
let pf = &output[dim * seq..dim * seq + score_ch * seq];  // [1, SCORE_CH, 1, SEQ]
let dpf = &output[dim * seq + score_ch * seq..];  // [1, SCORE_CH, 1, SEQ]
```

**Test Results (combined)**:
```
dvf: [ 0.2732, 0.5791, 0.9214, 1.3076 ] - OK
pf:  [ 0.0567, 0.0577, 0.0588, 0.0599 ] - OK
dpf: [ 64.8750, 66.0625, 67.2500, 68.4375 ] - OK
```

## Attempts (Standalone dpf - All Failed)

Multiple approaches were attempted for standalone dpf without success:
1. Adding transpose between matmul and reshape (matching DVF pattern)
2. Using identity operations (mul by 1.0, add 0.0) to break matmul->reshape dependency
3. Different transpose permutations ([0,1,3,2], [0,2,1,3])
4. Manual V transpose before matmul instead of using transpose_y=bT
5. Intermediate reshape steps ([1,H,S,S] -> [1,S,H,S] -> [H,S,S] -> [1,H*S,1,S])
6. Various combinations of the above

## Root Cause

The ANE compiler has constraints on single-output programs with matmul->reshape patterns. The reference implementation works because it concatenates dvf+pf+dpf outputs together, which changes how the compiler handles intermediate tensor memory layouts.

## Workaround Options

1. **Combined output (RECOMMENDED)**: Use `bwd_sdpa_bwd1_combined_mil` - all three outputs work correctly
2. **Skip dpf**: Compute dP gradient through alternative means or omit if not needed
3. **CPU fallback**: Compute dpf on CPU using the cpu_fallback module

## Test Files

- `examples/test_backward_sdpa_bwd1_combined.rs` - Tests combined dvf+pf+dpf (ALL PASS)
- `examples/test_backward_sdpa_bwd1_lite.rs` - Tests dvf + pf (both pass)
- `examples/test_backward_sdpa_bwd2.rs` - Tests dqf + dkf (both pass)
- `examples/test_pf.rs` - Tests pf standalone (passes)
- `examples/test_ane_parameter_golf_bwd1.rs` - Tests combined with parameter-golf config (PASS)

## Test Results Summary

```
dim=64, seq=16, heads=4, head_dim=16

Standalone functions:
  dvf: [ 0.2732, 0.5791, 0.9214, 1.3076 ] - PASS
  pf:  [ 0.0567, 0.0577, 0.0588, 0.0599 ] - PASS
  dqf: [ -0.0166, -1.0088, -4.3086, -11.1641 ] - PASS
  dkf: [ -3.4023, -3.4590, -3.5195, -3.5762 ] - PASS
  dpf: CompilationFailure - FAIL

Combined function:
  dvf+pf+dpf: All outputs valid - PASS

Parameter-Golf configuration (dim=416, heads=8, head_dim=52):
  seq=256:  dvf=[0.2632, 0.5513, ...], pf=[0.0055, 0.0057, ...], dpf=[1250.0, 1275.0, ...] - PASS
  seq=1024: dvf=[0.0464, 0.0970, ...], pf=[0.0010, 0.0010, ...], dpf=[1265.0, 1290.0, ...] - PASS
```
