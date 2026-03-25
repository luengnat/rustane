---
phase: 04-backward-pass-correctness
plan: 01
subsystem: ane, mil, backward
tags: [ane, mil, backward, ffn, sigmoid, slice_by_size, multi-output, conv1x1]

# Dependency graph
requires:
  - phase: 02-op-alternatives/03-01
    provides: "9 ANE ops + 7 decompositions, multi-output as concat replacement"
  - phase: 01-constraint-testing/01-01
    provides: "MIL_HEADER buildInfo syntax, subprocess test pattern"
provides:
  - "bwd_ffn_mil() — ANE-compatible FFN backward MIL generator"
  - "bwd_qkv_mil() — ANE-compatible QKV backward MIL generator"
  - "Sub decomposition pattern: add(x, mul(y, -1)) replaces rejected sub"
  - "Multi-output return pattern replaces rejected concat"
  - "Packed input + slice_by_size unpacking pattern for backward programs"
affects: [04-03, 04-04, training-loop, delta-compilation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Packed single-input backward programs: concat(dffn, h1, h3) → slice_by_size unpack"
    - "Multi-output returns (alphabetical order) replace concat for backward outputs"
    - "Sub decomposition: add(x, mul(y, const(-1.0))) for one_minus_sigmoid"
    - "CONV_CONST boilerplate shared across all backward conv ops"

key-files:
  created:
    - ".planning/phases/04-backward-pass-correctness/04-01-SUMMARY.md"
    - "examples/test_backward_ffn.rs"
  modified:
    - "src/mil/programs.rs"
    - "src/mil/mod.rs"

key-decisions:
  - "Multi-output return (alphabetical ordering) replaces concat for backward program outputs"
  - "sub(x, y) decomposed as add(x, mul(y, const(-1.0))) throughout all backward generators"
  - "Packed input via slice_by_size instead of multi-input (ANE only supports single input)"

patterns-established:
  - "Backward MIL programs accept single packed fp16 input, use slice_by_size to unpack intermediate activations"
  - "Multi-output names must be alphabetically sorted for ANE multi-output ordering"

# Metrics
duration: 15min
completed: 2026-03-26
---

# Phase 4 Plan 1: FFN + QKV Backward MIL Generators Summary

**Ported FFN and QKV backward passes from stories_mil.h to Rust with ANE adaptations: sub→add+mul decomposition, concat→multi-output, packed input via slice_by_size**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-26T17:39:03Z
- **Completed:** 2026-03-26
- **Tasks:** 3 (bwd_ffn_mil, bwd_qkv_mil, test binaries)
- **Files modified:** 4

## Accomplishments

- **bwd_ffn_mil()**: Full SwiGLU FFN backward — sigmoid gating, SiLU derivative, W1t/W2t/W3t weight gradients
- **bwd_qkv_mil()**: QKV projection backward — simplest case: 3 transposed convolutions summed
- **Test binaries**: Subprocess-isolated compile+eval tests for both generators
- **Verified**: Generated MIL contains zero instances of rejected `sub()` or `concat()` ops

## Task Commits

1. **bwd_ffn_mil + bwd_qkv_mil implementation** - `c84ac37` (feat)
2. **FFN backward test binary** - `fd43e56` (feat)
3. **QKV backward test binary** - `7c4b7af` (feat)

## Files Created/Modified

- `src/mil/programs.rs` — Added bwd_ffn_mil(), bwd_qkv_mil(), CONV_CONST_STR, compile_request helpers (~250 lines)
- `src/mil/mod.rs` — Added public exports for backward generators
- `examples/test_backward_ffn.rs` — FFN backward compile+eval test (subprocess-safe)
- `examples/test_backward_qkv.rs` — QKV backward compile+eval test (subprocess-safe)

## Decisions Made

1. **Multi-output return replaces concat** — ANE rejects concat, but multi-output programs work (M1 discovery). Backward outputs returned as separate named outputs in alphabetical order.
2. **Sub decomposition: add(x, mul(y, const(-1.0)))** — Consistent pattern used throughout all generators. Single `nm1 = const(-1.0)` declared once, reused.
3. **CONV_CONST shared constant** — Extracted conv boilerplate (pt, st, pd, dl, gr) into a `const CONV_CONST_STR` used by all backward generators.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-existing build errors in src/ane/{tiling,training_architecture,trainer}.rs (unresolved import `super::mil_generator`) — not caused by this plan's changes.

## Next Phase Readiness

- FFN and QKV backward generators ready for ANE execution
- SDPA backward generators (plan 03) depend on multi-output pattern validated here
- Numerical gradient verification (plan 04) can validate correctness without ANE hardware

---
*Phase: 04-backward-pass-correctness*
*Completed: 2026-03-26*
