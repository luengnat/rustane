# ANE Constraint Testing Session — Findings & Roadmap

**Date**: 2026-03-25
**Status**: In progress — root cause of all test failures identified, fix pending

---

## Executive Summary

We are building a Rust library to leverage Apple's Neural Engine (ANE) for transformer training, following the Orion paper architecture (arXiv:2603.06728). The immediate goal is to empirically test all 20 ANE constraints from the Orion paper before building fused programs.

**The entire test suite was failing due to a single syntax bug in the MIL header.** The bug has been identified and characterized at the byte level. The fix is trivial — a one-line constant change.

---

## Root Cause: ANE MIL buildInfo Brace Doubling

### The Bug

In `examples/test_ane_constraint.rs`, the `MIL_HEADER` constant has **every** key-value pair wrapped in doubled braces `{{ }}`, but the ANE compiler only expects doubled braces on the **outermost** dict wrapper:

```diff
- [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, ...)]
+ [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, ...)]
```

### Byte-Level Proof

```
MIL_HEADER (BROKEN):  9 open braces, 8 close braces  ← UNBALANCED
programs.rs (WORKING): 5 open braces, 5 close braces  ← BALANCED
```

The broken header produces:
```
({{"key1", "val1"}}, {{"key2", "val2"}}, {{"key3", "val3"}}, {{"key4", "val4"}})
```

The correct header produces:
```
({{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}, {"key4", "val4"}})
```

### Why This Happened

In Rust `const &str` (not `format!`), `{{` is literally two `{` characters — it is NOT an escape. So writing `{{\"key\"` in a `const &str` produces `{{"key"` in the output string. The `programs.rs` code uses `push_str()` where the intent was clearer, and only doubles the outermost braces.

### The Fix

Change `MIL_HEADER` in `test_ane_constraint.rs` line 145-147 from:
```rust
const MIL_HEADER: &str = "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}})]\n\
{\n";
```

To (copy exact line from `programs.rs` line 214):
```rust
const MIL_HEADER: &str = "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n\
{\n";
```

### What Else Was Checked

- **Function body braces** (`{` / `}`): Both the test file and `programs.rs` use single braces for function bodies, program scope, and return statements. This is correct — ANE only requires doubled braces in the buildInfo dict.
- **`mil_fp16_program` and `mil_fp32_program` helpers**: Their function body uses `write!` with `{{\n` which in `format!`/`write!` produces a single `{`. This is correct.
- **`conv1x1_mil` helper**: Same pattern, correct.

---

## Key Discoveries This Session

### 1. ANE MIL ≠ CoreML MIL (Critical)

The ANE compiler requires `{{` and `}}` in the buildInfo dict, not single `{` and `}`. This was proven by byte-level comparison of `programs.rs` output (which works) vs test file output (which didn't).

**Implication**: Any MIL generated for ANE must use the ANE dialect, not standard CoreML MIL. The differences are subtle but cause hard failures.

### 2. Weight Name Matching

`with_weight_blob()` requires the **full BLOBFILE path** as the key name:
- Correct: `"@model_path/weights/W.bin"`
- Wrong: `"W.bin"`

The ANE framework's `modelWithMILText_weights_optionsPlist` matches BLOBFILE paths against the weight dictionary keys exactly.

### 3. fp32 I/O with Internal fp16 Cast Works

The `programs.rs` pattern of `fp32 input → cast to fp16 → compute → cast back to fp32 → fp32 output` compiles and runs successfully. The reference ANE code (`~/dev/ANE/training/`) uses fp16 I/O directly — both work, but fp32 I/O is what we've verified.

### 4. Input/Output Sizes Are in Bytes

`ANECompileRequest::new(mil, input_sizes, output_sizes)` — sizes are **byte counts** used to create IOSurfaces, not element counts. fp32 = `× 4`, fp16 = `× 2`.

### 5. Reference Code Architecture

`~/dev/ANE/training/stories_mil.h` contains working fused MIL generators:
- `gen_sdpa_fwd_taps()` — Full SDPA with RMSNorm, QKV projection, attention, Wo projection, concat taps for backward
- `gen_ffn_fwd_taps()` — FFN with RMSNorm, W1/W3 conv, SiLU gating, W2 conv, concat taps
- `gen_ffn_bwd()` — FFN backward: slice taps → SiLU derivative → transposed convolutions → gradient concat
- `gen_qkvb()` — QKV backward: slice → transposed convolutions → gradient sum
- `gen_sdpa_bwd1()` — SDPA backward part 1: Wo^T + attention gradient

All use fp16 I/O directly (no fp32 cast), single braces for function/program scope, `{{ }}` only in buildInfo.

---

## Files Written This Session (Uncommitted)

| File | Purpose | Status |
|------|---------|--------|
| `examples/test_ane_constraints.rs` | Orchestrator — runs test_ane_constraint as subprocess | Compiles, not tested |
| `examples/test_ane_constraint.rs` | Worker — 30+ test functions for ANE constraints | Compiles, all tests fail (MIL_HEADER bug) |
| `examples/test_minimal.rs` | Debug file used to discover the `{{`/`}` issue | Should be deleted |

---

## Orion Paper 20 Constraints — Test Coverage Map

| # | Constraint | Test Name | Expected Result | Status |
|---|-----------|-----------|-----------------|--------|
| 1 | `concat` rejected | `concat_basic` | FAIL (Orion says rejected) | Not run |
| 2 | `gelu` not valid | `gelu_basic` | FAIL (Orion says invalid) | Not run |
| 3 | ~119 compile limit | (manual test) | N/A | Not tested |
| 4 | Min ~49KB IOSurface | `min_surface_tiny` (seq=1) | FAIL at eval | Not run |
| 5 | Uniform I/O sizes | `multi_output_nonuniform` | FAIL at eval | Not run |
| 6 | Empty weight dict | (handled in runtime) | N/A | Always passes |
| 7 | milText as NSData | (handled in FFI) | N/A | Always passes |
| 8 | Batch dim must be 1 | (all tests use batch=1) | N/A | Always passes |
| 9 | Only fp16/fp32 | (all tests use fp16/fp32) | N/A | Always passes |
| 10 | Multi-output alpha order | `multi_output_alpha`, `multi_output_reverse` | VERIFY ordering | Not run |
| 11 | Multi-input alpha order | `multi_input_add` | VERIFY ordering | Not run |
| 12 | Packed shape data | (implicit in all tests) | N/A | Not run |
| 13 | SDPA causal masks ignored | (no SDPA test yet) | TODO | Not written |
| 14 | BLOBFILE offset = 64 | `blobfile_offset_64`, `blobfile_offset_128` | 64 works, 128 may fail | Not run |
| 15 | Weights baked at compile | (known from prior work) | N/A | Known |
| 16 | Matmul transpose as named const | `matmul_transpose_named` | VERIFY | Not run |
| 17 | Conv no bias | `conv_bias_basic` | FAIL (Orion says rejected) | Not run |
| 18 | Dead names in return | (no test) | TODO | Not written |
| 19 | Dispatch overhead | (benchmarked in prior work) | ~2-5ms | Known |
| 20 | SRAM cliff | (benchmarked in prior work) | N/A | Known |

**Additional tests beyond Orion:**
- `sigmoid_basic` — test sigmoid (used in SiLU)
- `layer_norm_basic`, `layer_norm_with_weight` — test layer_norm op
- `rmsnorm_trick_basic`, `rmsnorm_trick_with_weight` — ANEMLL RMSNorm trick
- `rmsnorm_manual_basic` — manual RMSNorm (mul/reduce_sum/pow)
- `softmax_basic` — test softmax
- `dual_conv_basic` — two convs in one program
- `qkv_fused_basic` — QKV projection fused with concat
- `fused_ffn_small`, `fused_ffn_medium`, `fused_ffn_taps_small` — full FFN forward
- `conv_64x64`, `matmul_64x64`, etc. — conv vs matmul comparison
- `conv_4k_channels`, `conv_16k_channels`, `conv_32k_channels` — large channel tests

---

## Execution Roadmap

### Phase 0: Fix the Bug (5 min)
1. Fix `MIL_HEADER` constant in `test_ane_constraint.rs`
2. Build: `cargo build --example test_ane_constraint`
3. Quick smoke test: `RUSTANE_TEST_NAME=sigmoid_basic ./test_ane_constraint`
4. If sigmoid passes → bug is fixed, proceed to Phase 1

### Phase 1: Run Constraint Tests (~10 min)
1. `cargo build --example test_ane_constraints && cargo run --example test_ane_constraints`
2. Each test runs in a subprocess to isolate compile failures
3. Capture all pass/fail results
4. **Critical question**: Does `concat` actually fail on this hardware/firmware? Orion says it's rejected, but `stories_mil.h` uses `concat` extensively in working code. This may be a firmware version difference.

### Phase 2: Analyze Results (~15 min)
1. Compare results against Orion paper predictions
2. Identify any surprises (ops that work but shouldn't, or vice versa)
3. Update `docs/ane_constraints.md` with empirical findings
4. Determine which constraints are hard blockers vs. soft limits

### Phase 3: Build Fused MIL Generators (~2-4 hours)
Based on test results, build Rust equivalents of `stories_mil.h` generators:
1. `fwd_ffn_mil()` — FFN forward with taps (following `gen_ffn_fwd_taps`)
2. `fwd_sdpa_mil()` — SDPA forward with taps (following `gen_sdpa_fwd_taps`)
3. `bwd_ffn_mil()` — FFN backward (following `gen_ffn_bwd`)
4. `bwd_qkv_mil()` — QKV backward (following `gen_qkvb`)
5. `bwd_sdpa_mil()` — SDPA backward (following `gen_sdpa_bwd1`)

These go in `src/mil/programs.rs` alongside the existing `conv1x1_mil()` and `rmsnorm_mil()`.

### Phase 4: Delta Compilation (~1-2 hours)
1. Implement weight patching: write updated weights to temp file, reload model
2. Benchmark: patch+reload vs full recompile
3. Target: <500ms per weight update (Orion achieved 494ms)
4. Handle compile limit: track compile count, warn at ~100

### Phase 5: Training Loop Integration (~2-4 hours)
1. Wire fused programs into training loop
2. Forward pass: x → RMSNorm → SDPA → add_residual → RMSNorm → FFN → add_residual
3. Backward pass: taps → gradients → weight updates → delta reload
4. Loss computation on CPU (ANE doesn't do cross-entropy well)
5. End-to-end training test on synthetic data

### Phase 6: Benchmark & Optimize (~1-2 hours)
1. Compare ANE training throughput vs CPU baseline
2. Profile: compile time, eval time, dispatch overhead, data transfer
3. Tune: sequence length padding, channel dimensions, fusion boundaries
4. Target: measurable speedup over CPU for parameter-golf model

---

## Key Open Questions

1. **Does `concat` work on M4?** Orion says it's rejected, but `stories_mil.h` uses it. May be firmware-dependent. The test will tell us.

2. **Does `conv` support bias?** Orion says no. Our `conv_bias_basic` test will verify.

3. **Does `matmul` work on ANE at all?** Prior testing showed eval failures for matmul. `matmul_64x64` and `matmul_transpose_named` will test this with the corrected MIL syntax.

4. **What's the actual minimum IOSurface size?** Orion says ~49KB. Our `min_surface_tiny` (seq=1, 64×1×1×1 = 256 bytes) should fail, `min_surface_small` (seq=8, 64×1×1×8 = 2048 bytes) may fail, `min_surface_ok` (seq=16, 64×1×1×16 = 4096 bytes) should work. But the real limit may be different.

5. **fp16 I/O vs fp32 I/O?** `programs.rs` uses fp32 I/O with internal cast. `stories_mil.h` uses fp16 I/O directly. Both should work. fp16 avoids the cast overhead.

---

## Architecture Reference

```
Forward pass (per transformer layer):
┌─────────────────────────────────────────────────┐
│ x_in [1, DIM, 1, SEQ]                          │
│   │                                             │
│   ├─ RMSNorm ───────────────────── xn           │
│   │    │                                        │
│   │    ├─ conv(Wq) ── q                        │
│   │    ├─ conv(Wk) ── k        ┌── taps ────────┤
│   │    ├─ conv(Wv) ── v        │  (q,k,v,attn,  │
│   │    │                       │   xn for bwd)  │
│   │    └─ SDPA(q,k,v) ─ af    │                │
│   │         │                  │                │
│   │         └─ conv(Wo) ── oo ─┘                │
│   │                        │                    │
│   ├─ add(x_in, oo) ── x2                       │
│   │    │                                        │
│   │    ├─ RMSNorm ───────────────── x2n         │
│   │    │                                        │
│   │    ├─ conv(W1) ── h1        ┌── taps ──────┤
│   │    ├─ conv(W3) ── h3        │  (ffn_out,    │
│   │    │    │                   │   h1, h3,     │
│   │    │    ├─ sigmoid(h1)      │   silu, x2n)  │
│   │    │    ├─ h1 * sig = silu  │               │
│   │    │    └─ silu * h3 = gate │               │
│   │    │         │              │               │
│   │    │         └─ conv(W2) ── ┘               │
│   │    │                        │               │
│   │    └─ add(x2, ffn_out) ── x_out            │
│                                                 │
│   taps → used by backward pass for gradients    │
└─────────────────────────────────────────────────┘

Backward pass:
  - Receives taps (intermediate activations) from forward
  - Computes gradients through each operation in reverse
  - Outputs: dX (input gradient), dW1..dWn (weight gradients)
  - Weight gradients → optimizer (CPU) → delta compilation reload
```
