# ANE Constraints Reference

**Last updated**: 2026-03-25
**Source**: Combined findings from [Orion](https://github.com/mechramc/Orion) (20 constraints) and rustane empirical testing.

This document catalogs Apple Neural Engine hardware and software constraints discovered through reverse-engineering and empirical testing. Constraints are organized by when they manifest (compile-time vs. runtime) and whether they cause hard failures or silent corruption.

---

## Empirical Test Results (2026-03-25)

Full test suite run on Apple M4, macOS. 30 tests, subprocess isolation per test.

### Summary

| Result | Count | Tests |
|--------|-------|-------|
| ✅ PASS | 10 | sigmoid, min_surface_ok, blobfile_offset_64, multi_output_uniform, multi_output_alpha, conv_4k/16k/32k_channels, multi_input_add |
| ✗ Compile error | 11 | concat, gelu, conv_bias, layer_norm, softmax, reduce_sum, fused_ffn_taps (concat), qkv_fused (concat), blobfile_offset_128, matmul_64x64, matmul_128x384 |
| ✗ Inference error | 3 | min_surface_tiny (seq=1), min_surface_small (seq=8), multi_output_nonuniform |
| ⚠️ NaN/Inf (data corruption) | 6 | conv_64x64 (seq=32), conv_128x384 (seq=32), fused_ffn_small (seq=32), fused_ffn_medium (seq=32), dual_conv_basic (seq=32), multi_output_reverse (non-alpha) |

### Key Discoveries Beyond Orion

1. **`layer_norm` is rejected** — Orion doesn't mention this! This blocks the ANEMLL RMSNorm trick (`concat([x, -x]) → layer_norm → slice`). The reference code (`stories_mil.h`) uses `layer_norm` in working programs — possible firmware version difference.
2. **`softmax` is rejected** — Orion doesn't mention this. All softmax ops fail with `InvalidMILProgram`. The reference code uses manual attention (matmul + scale + mask + softmax), which suggests softmax may need special handling or a different MIL op name.
3. **`reduce_sum` is rejected** — Used in manual RMSNorm. Fails with `InvalidMILProgram`.
4. **seq=32 produces nan/inf** — seq=16 works fine for all passing tests. seq=32 causes data corruption (not compile/eval failure). The reference code uses various seq lengths — need to investigate the exact boundary.
5. **No channel limit found** — 4K, 16K, and 32K output channels all work fine for conv.
6. **Multi-input works** — Two IOSurface inputs to one program works correctly.
7. **BLOBFILE offset=128 fails** — Confirmed offset must be 64.

### Orion Constraint Verification

| # | Constraint | Orion Says | Empirical | Match? |
|---|-----------|------------|-----------|--------|
| 1 | concat rejected | Yes | ✅ InvalidMILProgram | ✅ Yes |
| 2 | gelu invalid | Yes | ✅ InvalidMILProgram | ✅ Yes |
| 3 | ~119 compile limit | Yes | Not tested (needs many compiles) | — |
| 4 | Min ~49KB IOSurface | Yes | ✅ seq=1 fails, seq=8 fails, seq=16 passes | ✅ Yes |
| 5 | Uniform I/O sizes | Yes | ✅ non-uniform fails at eval | ✅ Yes |
| 6 | Empty weight dict | Yes | Not directly tested (runtime handles) | — |
| 7 | milText as NSData | Yes | Handled in FFI layer | — |
| 8 | Batch dim = 1 | Yes | All tests use batch=1 | — |
| 9 | Only fp16/fp32 | Yes | All tests use fp16/fp32 | — |
| 10 | Multi-output alpha order | Yes | ⚠️ Non-alpha produces nan/inf (not error) | ⚠️ Partial |
| 11 | Multi-input alpha order | Yes | Not yet verified (need ordered test) | — |
| 12 | Packed shape data | Yes | Not directly tested | — |
| 13 | SDPA causal masks ignored | Yes | Not tested (no SDPA test) | — |
| 14 | BLOBFILE offset = 64 | Yes | ✅ offset=64 works, offset=128 fails | ✅ Yes |
| 15 | Weights baked at compile | Yes | Known from prior work | — |
| 16 | Matmul transpose named consts | Yes | ⚠️ Test has code bug (WeightBlob dims) | — |
| 17 | Conv no bias | Yes | ✅ InvalidMILProgram | ✅ Yes |
| 18 | Dead names in return | Yes | Not tested | — |
| 19 | Dispatch overhead | Yes | ~0.2ms per eval (faster than expected) | ⚠️ Lower |
| 20 | SRAM cliff | Yes | Not benchmarked | — |

### ANE Ops Confirmed Working

| Op | Test | Notes |
|----|------|-------|
| `cast` (fp32→fp16, fp16→fp32) | All tests | Required for fp32 I/O pattern |
| `conv` (1x1, no bias) | min_surface_ok, conv_4k/16k/32k | Core operation, works at all channel counts |
| `sigmoid` | sigmoid_basic | Used in SiLU activation |
| `mul` | multi_input_add | Element-wise multiply |
| `add` | multi_input_add | Element-wise add |
| `const` (scalar, tensor) | All tests | Constants including BLOBFILE weights |

### ANE Ops Confirmed Failing

| Op | Test | Error |
|----|------|-------|
| `concat` | concat_basic | `InvalidMILProgram` |
| `gelu` | gelu_basic | `InvalidMILProgram` |
| `conv` (with bias) | conv_bias_basic | `InvalidMILProgram` |
| `layer_norm` | layer_norm_basic | `InvalidMILProgram` |
| `softmax` | softmax_basic | `InvalidMILProgram` |
| `reduce_sum` | rmsnorm_manual_basic | `InvalidMILProgram` |
| `matmul` | matmul_64x64 | `InvalidMILProgram` |
| `linear` | matmul_128x384 | `InvalidMILProgram` |

### Critical Implications for stories_mil.h Port

The reference code (`stories_mil.h`) uses ops that **fail on our hardware**:
- `layer_norm` — used in SDPA forward RMSNorm and ANEMLL trick
- `reduce_sum` — used in manual RMSNorm
- `concat` — used for taps output and ANEMLL trick
- `softmax` — used in attention
- `matmul` — used in attention QK^T and AV

**This means we cannot directly port stories_mil.h.** We need to find ANE-compatible alternatives or decompose these ops. This is a significant architectural finding that changes the roadmap.

---

## Overview

| Category | Count | Severity |
|----------|-------|----------|
| Compile/eval failures | 10 | Hard error |
| Silent wrong data | 7 | Silent corruption |
| Compiler-level | 3 | Hard error |
| **Total** | **20** | |

---

## Compile/Eval Failures (Hard Errors)

These constraints cause compilation or execution to fail with an error message.

### 1. `concat` Op Rejected

**Constraint**: ANE rejects the `concat` MIL op entirely.

**Error**: `ANE does not support concat op`

**Workaround**: Use multi-output programs instead. Generate multiple outputs from a single program rather than concatenating results.

**Source**: Orion #1

---

### 2. `gelu` Is Not a Valid MIL Op

**Constraint**: `gelu` is not recognized by ANE MIL parser.

**Error**: Invalid MIL program

**Workaround**: Decompose GELU into tanh approximation:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

**Source**: Orion #2

---

### 3. ~119 Compile Limit Per Process

**Constraint**: Each process can compile ~119 ANE programs before hitting a hard limit.

**Error**: Compilation fails after ~119 compiles

**Workaround**: **Delta compilation** (Orion's key innovation):
- Compile programs once at startup
- For weight updates: unload → patch weight file on disk → reload
- Avoids recompilation entirely
- Reduces weight update time from 4,200ms to 494ms (8.5x faster)

**Source**: Orion #3

---

### 4. Minimum ~49KB IOSurface Allocation

**Constraint**: `seq_len=1` compiles but fails at eval time due to insufficient IOSurface size.

**Error**: Eval fails for small tensors

**Workaround**: Pad tensors to minimum ~49KB allocation size.

**Source**: Orion #4

---

### 5. Uniform IOSurface Sizes for Multi-I/O

**Constraint**: Multi-output AND multi-input programs require **uniform IOSurface allocation sizes** — all surfaces must be padded to the maximum size.

**Error**: Eval fails with mismatched surface sizes

**Workaround**: Pad all I/O surfaces to the largest tensor dimension in the program.

**Source**: Orion #5

---

### 6. Empty Weight Dict Required

**Constraint**: Weight dict must be `@{}` (empty dict), not `nil`, for weight-free programs.

**Error**: Compilation fails with nil weight dict

**Workaround**: Always pass `@{}` (empty NSDictionary) instead of `nil`.

**Source**: Orion #6

---

### 7. `milText` Must Be `NSData*`

**Constraint**: `milText` parameter must be `NSData*` (UTF-8 bytes), not `NSString*`.

**Error**: Compilation fails

**Workaround**: Convert NSString to NSData using UTF-8 encoding before passing to compiler.

**Source**: Orion #7

---

### 8. Batch Dimension Must Be 1

**Constraint**: ANE requires batch dimension of 1 (first dimension).

**Error**: `ANE requires batch dimension of 1`

**Workaround**: Restructure tensors to `[1, C, H, W]` or `[1, C, 1, S]` layout.

**Source**: Orion + rustane

---

### 9. Only FP16/FP32 Supported

**Constraint**: ANE supports only fp16 and fp32 dtypes for compute.

**Error**: `ANE supports only fp16/fp32`

**Workaround**: Cast int32/bool/string tensors to fp32 before ANE operations.

**Source**: Orion + rustane

---

### 10. Multi-Output Ordering (Alphabetical)

**Constraint**: Multi-output surfaces are ordered **alphabetically by MIL variable name**, not by return tuple order.

**Error**: Outputs appear in wrong order, causing silent data corruption if undetected

**Workaround**: Name outputs to achieve desired ordering, or reorder after eval based on alphabetical name matching.

**Source**: Orion #8

---

## Silent Wrong Data (Corruption)

These constraints cause incorrect results without error messages.

### 11. Multi-Input Ordering (Alphabetical)

**Constraint**: Multi-input surfaces are also ordered **alphabetically by MIL parameter name**.

**Symptom**: Inputs bound to wrong parameters

**Workaround**: Name inputs carefully to ensure alphabetical ordering matches expected binding order.

**Source**: Orion #9

---

### 12. Packed Shape Data (No Stride Adjustment)

**Constraint**: ANE reads flat buffer as **packed shape data** — no stride adjustment for oversized surfaces.

**Symptom**: Data corruption when using oversized IOSurfaces

**Workaround**: Ensure tensor data is packed contiguously; do not rely on strides for padding.

**Source**: Orion #10

---

### 13. SDPA Causal Masks Ignored

**Constraint**: SDPA (scaled dot-product attention) causal masks are silently ignored.

**Symptom**: Attention leaks future information

**Workaround**: Manually decompose attention into separate operations with explicit masking.

**Source**: Orion #11

---

### 14. BLOBFILE Offset Is 64

**Constraint**: BLOBFILE offset is `uint64(64)` (chunk header), not 0 or 128.

**Symptom**: Weights loaded incorrectly, garbage outputs

**Workaround**: Always use offset 64 for BLOBFILE weight data.

**Source**: Orion #12

---

### 15. Weights Baked at Compile Time

**Constraint**: Weights are baked into compiled ANE programs. Overwriting BLOBFILE on disk doesn't change outputs.

**Symptom**: Weight updates have no effect

**Workaround**: Use delta compilation (unload → patch → reload) instead of expecting hot reload.

**Source**: Orion #13

---

### 16. MatMul Transpose Args Must Be Named Consts

**Constraint**: ANE MIL requires named const refs for matmul `transpose_x`/`transpose_y` — inline `true`/`false` literals are rejected.

**Error**: Compilation fails or silent incorrect transpose

**Workaround**: Define transpose flags as named constants in MIL:
```
const transpose_x = false;
var out = mb.matmul(..., transpose_x=transpose_x, ...);
```

**Source**: Orion #14

---

### 17. Conv Does Not Support Bias Parameter

**Constraint**: ANE MIL `conv` op does NOT support `bias=` parameter.

**Error**: Compilation fails

**Workaround**: Bias must be a separate `add` op after convolution.

**Source**: Orion #15

---

### 18. Dead Names in Return Tuples Cause InvalidMILProgram

**Constraint**: Output variable names in return tuple must reference live nodes.

**Error**: `InvalidMILProgram` if return tuple contains dead/eliminated node names

**Workaround**: Update return tuple after optimization passes to only include live nodes.

**Source**: Orion #16

---

## Runtime/Resource Constraints

### 19. Dispatch Overhead

**Constraint**: ANE dispatch has significant overhead (~2-5ms per launch).

**Impact**: Small kernels are latency-bound by dispatch, not compute

**Workaround**: Fuse operations into larger kernels to amortize dispatch overhead.

**Source**: Orion (maderix research)

---

### 20. SRAM Cliff

**Constraint**: ANE has limited SRAM (~several MB). Exceeding it causes fallback to slower DRAM.

**Impact**: Performance drops precipitously when tensors exceed SRAM capacity

**Workaround**: Use optimization passes to annotate and prioritize SRAM residency for frequently-reused intermediates.

**Source**: Orion (maderix research)

---

## rustane-Specific Constraints

### 21. Rectangular MatMul Requirement

**Constraint**: ANE requires rectangular matrices for matmul (not square-only, but specific aspect ratios may be optimal).

**Source**: rustane empirical testing

---

### 22. Tile Size Constraints

**Constraint**: ANE has specific tile size requirements for efficient matmul (typically multiples of 32 or 64).

**Source**: rustane benchmark testing

---

## Optimization Pass Implications

Based on these constraints, the following optimization passes are essential:

| Pass | Purpose | Constraints Addressed |
|------|---------|----------------------|
| **Uniform Output Padding** | Pad all I/O to max size | #5 |
| **DCE (Dead Code Elimination)** | Remove unused nodes | #18 |
| **Cast Fusion** | Fuse redundant casts | #9 |
| **SRAM Annotation** | Mark high-reuse tensors | #20 |
| **Identity Elimination** | Remove no-ops | Performance |
| **ANE Validation** | Check constraints pre-compile | #1, #8, #9, #16, #17 |

---

## Delta Compilation Protocol

To work around constraint #3 (compile limit):

```rust
// 1. Compile once at startup
let program = ane_compile(mil_text, weights, "model_layer0");

// 2. For weight updates, use delta reload:
//    - Unload current program
//    - Update BLOBFILE on disk
//    - Reload (NOT recompile)
// Time: ~494ms vs ~4,200ms for full recompile
ane_program_reload_weights(&mut program, new_weights);

// 3. NEVER exceed 119 compiles per process
//    - Track compile count
//    - Use program cache
//    - Restart process if limit approached
```

---

## References

- [Orion README.md](https://github.com/mechramc/Orion) - Comprehensive constraint documentation
- [maderix/ANE](https://github.com/maderix/ANE) - Foundational reverse-engineering
- [maderix blog Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) - Hardware characterization
- [maderix blog Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) - 38 TOPS debunk, compile limit
- [hollance/neural-engine](https://github.com/hollance/neural-engine) - Community documentation
