# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** Phase 3 planning needed — critical finding changes approach

## Current Position

**Milestone:** M1: ANE Foundation
**Phase:** 2 — ANE Constraint Testing (COMPLETE)
**Plan:** 02-01 — Run full constraint suite (COMPLETE)
**Status:** Phase complete — awaiting Phase 3 plan revision

## Progress

```
[██████████] 100% — M1: ANE Foundation
```

## Decisions

| Decision | Rationale | Added |
|----------|-----------|-------|
| MIL_HEADER uses programs.rs exact buildInfo pattern (line 214) | Byte-level proof that {{ }} only on outermost dict wrapper | Phase 1 |
| Subprocess isolation for each ANE constraint test | Compile failures can corrupt ANE state | Phase 1 |
| stories_mil.h cannot be directly ported | layer_norm, softmax, concat, matmul, reduce_sum all rejected | Phase 2 |
| seq=16 is safe boundary | seq=32 produces nan/inf data corruption | Phase 2 |
| Phase 3 plan needs revision | Original plan assumed stories_mil.h ops would work | Phase 2 |

## Blockers

- **CRITICAL**: ANE rejects layer_norm, softmax, reduce_sum, concat, matmul — need ANE-compatible alternatives
- seq=32 produces nan/inf — need to understand exact boundary (seq=16 works, seq=32 doesn't)

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 1 | 01-01 | 5min | 4 | 2 |
| 2 | 02-01 | 15min | 4 | 1 |

## Session History

| Date | Activity | Stopped At |
|------|----------|------------|
| 2026-03-25 | Project initialized, roadmap created | Ready for Phase 1 execution |
| 2026-03-25 | Phase 1 complete: fixed MIL_HEADER, smoke tests pass | Ready for Phase 2 execution |
| 2026-03-25 | Phase 2 complete: 30 tests run, critical finding — 8 ANE ops rejected | Phase 3 needs plan revision |
