---
phase: 04-backward-pass-correctness
plan: 03
subsystem: ane, mil, backward, attention
tags: [ane, mil, sdpa, attention, backward, softmax, matmul, reshape, transpose]

# Dependency graph
requires:
  - phase: 04-backward-pass-correctness/04-01
    provides: "Multi-output pattern, sub decomposition, packed input pattern"
provides:
  - "bwd_sdpa_bwd1_mil() — SDPA backward part 1: recompute attention, compute dV"
  - "bwd_sdpa_bwd2_mil() — SDPA backward part 2: softmax backward, compute dQ and dK"
  - "Complete backward pass pipeline: FFN + QKV + SDPA"
affects: [training-loop, delta-compilation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-part SDPA backward: bwd1 (dV + probs) feeds into bwd2 (dQ + dK)"
    - "Softmax backward via reduce_sum + sub decomposition on attention scores"
    - "Attention recomputation in backward pass: Q@K^T * scale + mask → softmax"

key-files:
  created:
    - ".planning/phases/04-backward-pass-correctness/04-03-SUMMARY.md"
  modified:
    - "src/mil/programs.rs"
    - "src/mil/mod.rs"

key-decisions:
  - "SDPA backward split into two MIL programs (matching stories_mil.h pattern)"
  - "SCORE_CH = HEADS * SEQ for flattened attention score tensors"
  - "Causal mask loaded as BLOBFILE weight (same as forward pass)"

patterns-established:
  - "Complex backward passes can be split into multiple ANE programs"
  - "Attention recomputation in backward is required (probs needed for dQ/dK)"

# Metrics
duration: 10min
completed: 2026-03-26
---

# Phase 4 Plan 3: SDPA Backward MIL Generators Summary

**Complete SDPA backward pass (dV, dQ, dK) via two MIL programs using verified ops: matmul, softmax, reduce_sum, reshape, transpose, with sub decomposition and multi-output returns**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-26
- **Completed:** 2026-03-26
- **Tasks:** 2 (bwd_sdpa_bwd1_mil, bwd_sdpa_bwd2_mil)
- **Files modified:** 2

## Accomplishments

- **bwd_sdpa_bwd1_mil()**: Recomputes attention probabilities from saved Q/K/V, computes dV (via probs^T @ dAttn) and partial dP (dAttn @ V)
- **bwd_sdpa_bwd2_mil()**: Softmax backward (reduce_sum + sub decomposition), computes dQ (ds @ K) and dK (ds^T @ Q)
- **Complete backward pipeline**: FFN + QKV + SDPA all ported from stories_mil.h to Rust

## Task Commits

1. **SDPA backward generators** - `e2ab56d` (feat)

## Files Created/Modified

- `src/mil/programs.rs` — Added bwd_sdpa_bwd1_mil(), bwd_sdpa_bwd2_mil(), compile_request helpers (~340 lines)
- `src/mil/mod.rs` — Added SDPA backward exports

## Decisions Made

1. **Two-program split for SDPA backward** — Following stories_mil.h exactly. bwd1 produces dV + attention data (probs, dp), bwd2 produces dQ + dK. This avoids a single massive MIL program.
2. **SCORE_CH = HEADS * SEQ** — Attention scores [HEADS, SEQ, SEQ] are flattened to [HEADS*SEQ, 1, SEQ] for ANE 4D tensor compatibility.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Format string bug with `n` variable scope in the QF/KF reshape loop — fixed by using `n = &name[0..1]` slice instead of full string reference.

## Next Phase Readiness

- All three backward generators (FFN, QKV, SDPA) implemented and ANE-compatible
- Ready for ANE execution to validate inference correctness at realistic sizes
- Numerical gradient verification can confirm mathematical correctness
- Training loop integration (Phase 6) can now use all backward generators

---
*Phase: 04-backward-pass-correctness*
*Completed: 2026-03-26*
