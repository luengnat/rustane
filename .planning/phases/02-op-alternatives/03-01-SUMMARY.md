---
phase: 02-op-alternatives
plan: 01
subsystem: ane, testing
tags: [ane, mil, conv, decomposition, iosurface, subprocess-testing]

# Dependency graph
requires:
  - phase: 01-constraint-testing/02-01
    provides: "Phase 2 constraint results (30 tests, 8 working ops), test infrastructure, ane_constraints.md"
provides:
  - "Comprehensive ANE op compatibility table (78 tests, 9 working ops + 7 decompositions)"
  - "Conv-based decomposition alternatives for reduce_sum, sub, negation, squaring"
  - "Updated ane_constraints.md with decomposition results and clean tables"
  - "13 new decomposition test functions in test_ane_constraint.rs"
affects: [fused-mil-generators, training-loop, cpu-ane-hybrid]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conv1x1 with all-ones weights for sum reduction (C channels → 1)"
    - "mul(x, -1) for negation, add(x, mul(y, -1)) for subtraction"
    - "mul(x, x) for squaring (fp16-safe range required)"
    - "Small-value inputs (0.00-0.99) to avoid fp16 overflow in multiplicative ops"
    - "Multi-output programs as concat replacement (alphabetical ordering)"

key-files:
  created:
    - ".planning/phases/02-op-alternatives/03-01-SUMMARY.md"
  modified:
    - "examples/test_ane_constraint.rs"
    - "docs/ane_constraints.md"

key-decisions:
  - "RMSNorm cannot be completed on ANE (no 1/sqrt or division) — must be CPU-only"
  - "Attention cannot be done on ANE (no softmax, no matmul) — must be CPU-only"
  - "Conv 1x2 works for temporal mixing but 2x1 spatial conv is rejected"
  - "pow() is fundamentally broken on ANE for ALL exponents, not just negative"
  - "fp16 overflow is a real concern — inputs must be in safe range for mul(x,x)"

patterns-established:
  - "Decomposition testing pattern: test with large values first, then small values to distinguish op failure from fp16 overflow"

# Metrics
duration: 45min
completed: 2026-03-25
---

# Phase 3 Plan 1: Investigate All ANE Op Alternatives Summary

**78 empirical tests across 4 rounds confirm 9 native ANE ops + 7 conv-based decompositions, but attention, normalization, and loss remain impossible on ANE.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-03-25
- **Completed:** 2026-03-25
- **Tasks:** 5
- **Files modified:** 2

## Accomplishments

- **Comprehensive op compatibility table**: 78 tests covering all MIL ops, `mb.*` variants, basic ops, seq boundaries, and conv-based decompositions
- **7 working decompositions discovered**: sum reduction via conv, negation via mul(-1), subtraction via mul+add, squaring via mul(x,x), temporal conv 1x2, multi-output conv, RMSNorm first stage
- **Definitive ANE capability boundary**: The ANE can do feedforward nets with SiLU + residual + local temporal mixing, but NOT attention, normalization, or loss computation
- **Clean documentation**: Removed duplicate tables, added decomposition results, updated ane_constraints.md as single source of truth

## Task Commits

1. **Task 1-2: mb.* variants + basic ops** - `15061ce` (feat)
   - 10 mb.* tests (all fail), 12 basic op tests (transpose/reshape/slice pass, sub/clamp/exp/log/abs/tanh/relu/leaky_relu fail)
2. **Task 3: Conv-based decompositions** - `4b3adc5` (feat)
   - 13 decomposition tests: 7 pass, 3 fail (conv 2x1, depthwise, pow), 3 nan/inf (large input overflow)
3. **Task 4: Sequence length boundary** - `15061ce` (feat, same commit)
   - seq=16 only safe length; seq=20/24 inference error; seq≥32 nan/inf
4. **Task 5: Documentation** - `326f3ad` (docs)
   - Updated ane_constraints.md with all Phase 3 results, decomposition table, clean tables

## Files Created/Modified

- `examples/test_ane_constraint.rs` — Added 25+ test functions: mb.* variants, basic ops, seq boundaries, decomposition tests. Now 1900+ lines with 78+ test functions.
- `docs/ane_constraints.md` — Updated summary table (78 tests), added decomposition results table, removed duplicate Phase 2 tables, added 5 new key discoveries, updated implications section.

## Decisions Made

1. **RMSNorm must be CPU-only** — Can compute sum(x²) on ANE but cannot do 1/sqrt() or division. The final normalization step requires CPU.
2. **Attention must be CPU-only** — No softmax, no matmul. Conv-based local attention via 1x2 kernel is possible but not global attention.
3. **Sub decomposition via mul+add is production-viable** — add(x, mul(y, -1)) is a drop-in replacement for the rejected sub op.
4. **pow() is unreliable on ANE** — Even pow(x, 2.0) with positive exponent produces nan/inf. Must use mul(x, x) instead.
5. **fp16 overflow is a real constraint** — Input values > ~255 can overflow when squared. Training data must be normalized or clamped before ANE ops.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed fp16 overflow false negatives in decomposition tests**
- **Found during:** Task 3 (conv-based decompositions)
- **Issue:** `decomp_sq_via_mul` and `decomp_sum_via_conv` produced nan/inf, initially appearing to fail. Root cause: input_bytes() generates values up to 3024, and 3024² = 9.1M exceeds fp16 max of 65504.
- **Fix:** Added `small_input_bytes()` helper generating values 0.00-0.99. Created follow-up tests `decomp_sq_small`, `decomp_sum_small`, `decomp_rmsnorm_small` that pass.
- **Files modified:** examples/test_ane_constraint.rs
- **Verification:** Small-value tests pass, confirming mul(x,x) and conv(sum) work correctly.
- **Committed in:** `4b3adc5` (part of Task 3 commit)

**2. [Rule 1 - Bug] Fixed WeightBlob dimension mismatch for conv_2x1 and conv_1x2**
- **Found during:** Task 3 (conv kernel tests)
- **Issue:** `make_blob(&w, out_dim * dim * 2, dim * 2)` expected `rows * cols = 524288` elements but got 4096. The rows/cols parameters don't need to match the MIL weight shape.
- **Fix:** Changed to `make_blob(&w, 1, out_dim * dim * 2)` — rows * cols just needs to equal data length.
- **Files modified:** examples/test_ane_constraint.rs
- **Verification:** conv_1x2 now compiles and passes. conv_2x1 correctly fails with CompilationFailure (not WeightBlobError).
- **Committed in:** `4b3adc5` (part of Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs — test data and test setup)
**Impact on plan:** Both were test infrastructure issues, not plan changes. Results are more accurate because of the fixes.

## Issues Encountered

- **pow() is more broken than expected**: Plan assumed pow(x, 2.0) would work (positive exponent), but ANE produces nan/inf for ALL exponents. This means mul(x, x) is the only viable squaring method.
- **Conv 2x1 kernel rejected**: CompilationFailure (different from InvalidMILProgram). This means the ANE compiler crashes rather than gracefully rejecting spatial convolutions > 1x1. Only 1x1 and 1x2 kernels work.
- **Depthwise conv rejected**: groups != 1 is rejected with InvalidMILProgram, blocking channel-wise operations.

## Next Phase Readiness

### What's ready:
- Complete ANE op compatibility table with 78 empirical tests
- Working decompositions for: sub, negation, squaring, sum reduction, RMSNorm first stage
- Clear understanding of what CAN run on ANE: conv1x1, conv1x2, sigmoid, mul, add, cast, const, transpose, reshape, slice_by_size
- Multi-output programs verified as concat replacement

### Blockers for next phases:
- **Attention**: Must be CPU-only (no softmax, no matmul)
- **RMSNorm**: Final step (1/sqrt) must be CPU-only
- **Loss computation**: Must be CPU-only (no softmax, no log, no cross-entropy)
- **seq=16 limit**: Severe constraint for any real training
- ** stories_mil.h port**: Blocked — reference code uses rejected ops (concat, matmul, softmax, layer_norm, reduce_sum)

### Recommended next investigation (from user's "do them all"):
1. **Option A: _ANEModel API** — Check if `_ANEModel` API accepts ops that raw MIL text doesn't
2. **Option C: CPU+ANE hybrid** — Use ANE only for conv1x1 projections, everything else on CPU
3. **Option D: Firmware difference** — Why does stories_mil.h work on reference hardware but not ours?

---
*Phase: 02-op-alternatives*
*Completed: 2026-03-25*

## Self-Check: PASSED

- ✅ `.planning/phases/02-op-alternatives/03-01-SUMMARY.md` exists
- ✅ `docs/ane_constraints.md` exists and updated
- ✅ `examples/test_ane_constraint.rs` exists with 25+ new tests
- ✅ Commit `4b3adc5` (decomposition tests) found
- ✅ Commit `326f3ad` (documentation) found
- ✅ Commit `15061ce` (Phase 3 ops + seq boundary) found
