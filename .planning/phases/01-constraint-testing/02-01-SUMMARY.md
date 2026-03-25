---
phase: 01-constraint-testing
plan: 02
subsystem: infra
tags: [ane, mil, testing, constraints, benchmarking]

# Dependency graph
requires:
  - phase: 01-01
    provides: Fixed MIL_HEADER, verified test infrastructure
provides:
  - Empirical verification of 20 Orion ANE constraints
  - Discovery that layer_norm, softmax, reduce_sum, concat, matmul all rejected
  - Discovery that seq=32 produces nan/inf corruption
  - Confirmation that conv, sigmoid, add, mul, cast work on ANE
affects: [03-fused-mil, 04-backward-delta]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Subprocess isolation per test prevents ANE state corruption
    - seq=16 is the maximum safe sequence length (seq=32 produces nan/inf)

key-files:
  created:
    - docs/ane_constraints.md (with empirical results)
  modified:
    - docs/ane_constraints.md

key-decisions:
  - "stories_mil.h cannot be directly ported — uses rejected ops (layer_norm, softmax, concat, matmul, reduce_sum)"
  - "Phase 3 must find ANE-compatible alternatives for all rejected ops"
  - "seq=16 is safe boundary, seq=32 produces silent data corruption"

patterns-established:
  - "ANE-compatible ops: conv (no bias), sigmoid, mul, add, cast, const"
  - "ANE-rejected ops: concat, gelu, layer_norm, softmax, reduce_sum, matmul, linear"

# Metrics
duration: 15min
completed: 2026-03-25
---

# Phase 2 Plan 1: Run Full Constraint Suite Summary

**Ran 30 ANE constraint tests, discovered 8 ops rejected by ANE including critical layer_norm/softmax/concat/matmul**

## Performance

- **Duration:** 15 min
- **Started:** 2025-03-25T16:30:00Z
- **Completed:** 2025-03-25T16:45:00Z
- **Tasks:** 4
- **Files modified:** 1

## Accomplishments
- Ran all 30 constraint tests with subprocess isolation
- Verified 10/20 Orion constraints match our hardware
- Discovered 5 new constraints not in Orion (layer_norm, softmax, reduce_sum rejected; seq=32 nan/inf; no channel limit)
- Documented complete ANE op compatibility table

## Task Commits

1. **Task 1: Run full constraint suite** - (no code change, results captured)
2. **Task 2: Analyze results** - (analysis captured in docs)
3. **Task 3: Update ane_constraints.md** - `75a0164` (docs)
4. **Task 4: Commit results** - `75a0164` (docs)

## Critical Finding

**stories_mil.h cannot be directly ported.** The reference code uses `layer_norm`, `reduce_sum`, `concat`, `softmax`, and `matmul` — all of which are rejected by the ANE compiler on our hardware. This fundamentally changes the approach for Phase 3.

## Deviations from Plan

None - plan executed as written.

## Next Phase Readiness
- Complete op compatibility table available
- Critical blocker identified: need ANE-compatible alternatives for rejected ops
- Phase 3 plan needs revision to account for op restrictions
- seq=16 boundary must be respected in all future programs

---
*Phase: 01-constraint-testing*
*Completed: 2025-03-25*
