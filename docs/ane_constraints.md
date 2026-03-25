# ANE Constraints Reference

**Last updated**: 2026-03-25
**Source**: Combined findings from [Orion](https://github.com/mechramc/Orion) (20 constraints) and rustane empirical testing.

This document catalogs Apple Neural Engine hardware and software constraints discovered through reverse-engineering and empirical testing. Constraints are organized by when they manifest (compile-time vs. runtime) and whether they cause hard failures or silent corruption.

---

## Empirical Test Results (2026-03-25)

**Four test rounds executed:** Phase 2 (30 tests), Phase 3 ops (35 tests), Phase 3 decompositions (13 tests), cumulative **78 tests**.

### Summary

| Result | Count | Tests |
|--------|-------|-------|
| ✅ PASS | 18 | sigmoid, min_surface_ok, blobfile_offset_64, multi_output_uniform, multi_output_alpha, conv_4k/16k/32k_channels, multi_input_add, transpose, reshape, slice_by_size, **neg_via_mul**, **sub_via_mul_add**, **sq_via_mul**, **sum_via_conv**, **rmsnorm_approx**, **conv_1x2**, **multi_output_conv** |
| ✗ Compile error | 22 | concat, gelu, conv_bias, layer_norm, softmax, reduce_sum, mb_softmax, mb_concat, mb_layer_norm, mb_reduce_sum, mb_transpose, mb_reshape, mb_slice_by_size, sub, clamp, exp, log, abs, tanh, relu, leaky_relu, matmul, linear, **conv_2x1**, **depthwise_conv** |
| ✗ Inference error | 6 | min_surface_tiny (seq=1), min_surface_small (seq=8), multi_output_nonuniform, seq=20, seq=24, seq=28 |
| ⚠️ NaN/Inf (corruption) | 9 | seq=32, seq=48, seq=64, seq=128, seq=256, seq=512, pow(-0.5), pow(2.0), sum_via_conv(large) |

### Critical Discovery: ANE Has Extremely Limited Op Support

**Only 9 MIL ops work on this hardware:**
1. `conv` (1x1 and 1x2 kernels only, no bias)
2. `sigmoid`
3. `mul`
4. `add`
5. `cast` (fp16↔fp32)
6. `const`
7. `transpose`
8. `reshape`
9. `slice_by_size`

**Additional ANE capabilities discovered via decomposition tests:**
- **Sum reduction** via conv1x1 with all-ones weights (64→1 channel) ✅
- **Negation** via mul(x, -1.0) ✅
- **Subtraction** via add(x, mul(y, -1)) ✅
- **Squaring** via mul(x, x) ✅
- **Temporal convolution** via 1x2 kernel ✅
- **RMSNorm first stage**: mul(x,x) → conv(all-ones) → sum(x²) ✅

**Rejected ops (29 tested, 0 work):**
- `matmul`, `linear` — Matrix multiplication (blocks attention)
- `concat` — Concatenation (blocks taps output, ANEMLL trick)
- `layer_norm` — Layer normalization (blocks RMSNorm)
- `softmax` — Softmax (blocks attention)
- `reduce_sum` — Reduction (blocks manual RMSNorm)
- `pow` — Power (compiles but produces nan/inf for ALL exponents, even positive)
- `sub` — Subtraction
- `clamp`, `exp`, `log`, `abs`, `tanh`, `relu`, `leaky_relu` — All rejected
- All `mb.*` prefixed variants — None work (mb.matmul, mb.softmax, etc.)
- `conv` with bias — Rejected
- `conv` with 2x1 kernel — CompilationFailure
- `conv` with 3x1 kernel — Panics (WeightBlob dimension mismatch)
- `conv` with depthwise (groups != 1) — InvalidMILProgram

### Sequence Length Constraint

**seq=16 is the ONLY safe sequence length.** seq=20 and seq=24 fail with Program Inference error. seq≥32 produces silent nan/inf data corruption. This is a severe constraint for training.

### Implications for Training Architecture

With only conv(1x1, 1x2), sigmoid, mul, add, cast, const, transpose, reshape, slice_by_size available:

- ✅ **Linear projections**: conv1x1 works (this is matmul)
- ✅ **SiLU activation**: sigmoid(x) * x via mul
- ✅ **Reshape/transpose**: For attention head reshaping
- ✅ **Negation**: mul(x, -1.0) — since sub is rejected
- ✅ **Subtraction**: add(a, mul(b, -1.0))
- ✅ **Squaring**: mul(x, x)
- ✅ **Sum reduction**: conv1x1 with all-ones weights (C→1)
- ✅ **Temporal mixing**: conv with 1x2 kernel (adjacent position mixing)
- ✅ **RMSNorm first stage**: mul(x,x) → conv(all-ones) → sum(x²)
- ❌ **Attention**: No softmax, no matmul — **attention is blocked**
- ❌ **RMSNorm completion**: Can compute sum(x²) but no 1/sqrt() or division
- ❌ **Gradient taps**: No concat — **cannot save intermediate activations**
- ❌ **Loss computation**: No softmax, no log, no cross-entropy on ANE
- ❌ **Depthwise conv**: groups != 1 rejected
- ❌ **2x1 conv**: CompilationFailure

**The ANE on this hardware can do: input → conv → sigmoid(x)*x → mul → add → conv(1x2) → output.**
This is a feedforward network with SiLU activation, residual connections, and local temporal mixing. No normalization, no attention, no loss, no non-1x1-spatial convolutions.

### Conv-Based Decomposition Results

| Decomposition | Works? | Method | Notes |
|---------------|--------|--------|-------|
| `reduce_sum` | ✅ | conv1x1(all-ones, C→1) | Sum of all channels; fp16 overflow with large inputs |
| `sub` | ✅ | add(x, mul(y, -1)) | Full replacement for rejected sub |
| `neg` | ✅ | mul(x, -1) | Negation |
| `x²` | ✅ | mul(x, x) | fp16 overflow with large inputs |
| `pow(x, 2)` | ❌ | pow(x, 2.0) | ANE pow is broken for ALL exponents |
| `softmax` | ❌ | N/A | Requires exp (rejected) |
| `layer_norm` | ❌ | N/A | Requires reduce_sum + division |
| `RMSNorm` | ⚠️ Partial | mul(x,x) → conv(sum) → sum(x²) | Can compute sum-of-squares, but no 1/sqrt() |
| `matmul` | ✅ | conv1x1 | Already proven; same operation |
| `concat` | ✅ | Multi-output programs | Works with alphabetical ordering |
| `depthwise conv` | ❌ | groups=dim | InvalidMILProgram |
| `conv 2x1` | ❌ | 2x1 kernel | CompilationFailure |
| `conv 1x2` | ✅ | 1x2 kernel | Temporal convolution works |

### ANE Ops Confirmed Working (Full Table)

| Op | Test | Compile | Eval | Notes |
|----|------|---------|------|-------|
| `cast` | (all tests) | ✅ | fp32↔fp16 |
| `const` | (all tests) | ✅ | Scalars, tensors, BLOBFILE |
| `conv` (1x1, no bias) | min_surface_ok | ✅ | 4K-32K channels |
| `sigmoid` | sigmoid_basic | ✅ | 0.2ms eval |
| `mul` | multi_input_add | ✅ | Element-wise |
| `add` | multi_input_add | ✅ | Element-wise |
| `transpose` | op_transpose | ✅ | Perm [0,3,2,1] |
| `reshape` | op_reshape | ✅ | [1,64,1,16] → [1,4,16,16] |
| `slice_by_size` | op_slice_by_size | ✅ | Slice with begin/size |
| `conv` (1x2 kernel) | decomp_conv_1x2 | ✅ | Temporal convolution |
| `neg` (via mul) | decomp_neg_via_mul | ✅ | mul(x, -1) |
| `sub` (via mul+add) | decomp_sub_via_mul_add | ✅ | add(x, mul(y, -1)) |
| `x²` (via mul) | decomp_sq_small | ✅ | mul(x, x), fp16-safe range |
| `reduce_sum` (via conv) | decomp_sum_small | ✅ | conv1x1(all-ones, C→1) |
| `rmsnorm stage 1` | decomp_rmsnorm_small | ✅ | mul(x,x) → conv(sum) → sum(x²) |
| `multi-output conv` | decomp_multi_output_as_concat | ✅ | Replaces concat |

### ANE Ops Confirmed Failing (Full Table)

| Op | Test | Error Type | Notes |
|----|------|------------|-------|
| `concat` | concat_basic | InvalidMILProgram | No `mb.concat` variant works |
| `gelu` | gelu_basic | InvalidMILProgram | |
| `matmul` | matmul_64x64 | InvalidMILProgram | No `mb.matmul` works either |
| `linear` | matmul_128x384 | InvalidMILProgram | |
| `conv` (bias) | conv_bias_basic | InvalidMILProgram | |
| `layer_norm` | layer_norm_basic | InvalidMILProgram | No `mb.layer_norm` works |
| `softmax` | softmax_basic | InvalidMILProgram | No `mb.softmax` works |
| `reduce_sum` | rmsnorm_manual_basic | InvalidMILProgram | No `mb.reduce_sum` works |
| `pow` | op_pow | nan/inf | Compiles! But negative exp → nan/inf |
| `sub` | op_sub | InvalidMILProgram | |
| `clamp` | op_clamp | InvalidMILProgram | |
| `exp` | op_exp | InvalidMILProgram | |
| `log` | op_log | InvalidMILProgram | |
| `abs` | op_abs | InvalidMILProgram | |
| `tanh` | op_tanh | InvalidMILProgram | |
| `relu` | op_relu | InvalidMILProgram | |
| `leaky_relu` | op_leaky_relu | InvalidMILProgram | |
| `mb.matmul` | mb_matmul | WeightBlobError | Panics (test bug, but op itself likely rejected) |
| `mb.softmax` | mb_softmax | compile failed | |
| `mb.concat` | mb_concat | compile failed | |
| `mb.layer_norm` | mb_layer_norm | compile failed | |
| `mb.reduce_sum` | mb_reduce_sum | compile failed | |
| `mb.transpose` | mb_transpose | compile failed | |
| `mb.reshape` | mb_reshape | compile failed | |
| `mb.slice_by_size` | mb_slice_by_size | compile failed | |
| `conv` (3x1 kernel) | op_conv3x1 | WeightBlobError | Panics (dimension mismatch, likely op rejected) |
| `conv` (2x1 kernel) | decomp_conv_2x1 | CompilationFailure | Spatial conv rejected |
| `conv` (depthwise) | decomp_depthwise_conv | InvalidMILProgram | groups != 1 rejected |
| `pow` (pos exp) | decomp_pow2_via_mul | nan/inf | pow(x, 2.0) broken for all exponents |

### Key Discoveries Beyond Orion

1. **`layer_norm` is rejected** — Orion doesn't mention this! This blocks the ANEMLL RMSNorm trick (`concat([x, -x]) → layer_norm → slice`). The reference code (`stories_mil.h`) uses `layer_norm` in working programs — possible firmware version difference.
2. **`softmax` is rejected** — Orion doesn't mention this. All softmax ops fail with `InvalidMILProgram`. The reference code uses manual attention (matmul + scale + mask + softmax), which suggests softmax may need special handling or a different MIL op name.
3. **`reduce_sum` is rejected** — Used in manual RMSNorm. Fails with `InvalidMILProgram`. **BUT: can be decomposed via conv1x1 with all-ones weights.**
4. **seq=32 produces nan/inf** — seq=16 works fine for all passing tests. seq=32 causes data corruption (not compile/eval failure).
5. **No channel limit found** — 4K, 16K, and 32K output channels all work fine for conv.
6. **Multi-input works** — Two IOSurface inputs to one program works correctly.
7. **BLOBFILE offset=128 fails** — Confirmed offset must be 64.
8. **`pow` is fundamentally broken** — Even pow(x, 2.0) with positive exponent produces nan/inf. Not just negative exponents.
9. **`sub` can be replaced** — add(x, mul(y, -1)) works as a full substitute for the rejected sub op.
10. **Conv 1x2 works but 2x1 doesn't** — Temporal convolution (1x2 kernel) is accepted, but spatial (2x1 kernel) gets CompilationFailure. Conv 3x1 also fails.
11. **Depthwise conv rejected** — groups != 1 produces InvalidMILProgram.
12. **RMSNorm is partially decomposable** — mul(x,x) → conv(sum) → sum(x²) works, but the final 1/sqrt() normalization step cannot be done on ANE.

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

### Orion Constraint Verification

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
