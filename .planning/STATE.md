# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** Phase 3 complete — ANE capability boundary fully mapped, need to investigate alternatives

## Current Position

**Milestone:** M1: ANE Foundation
**Phase:** 3 — ANE Op Alternatives (COMPLETE)
**Plan:** 03-01 — Investigate all ANE op alternatives (COMPLETE)
**Status:** Phase complete — ANE capability boundary fully characterized. 9 native ops + 7 decompositions. Attention, normalization, loss must be CPU-only.

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
| RMSNorm must be CPU-only | Can compute sum(x²) on ANE but no 1/sqrt or division | Phase 3 |
| Attention must be CPU-only | No softmax, no matmul on ANE | Phase 3 |
| Sub decomposition via mul+add is production-viable | add(x, mul(y, -1)) replaces rejected sub | Phase 3 |
| pow() is fundamentally broken on ANE | Even pow(x, 2.0) with positive exp produces nan/inf; use mul(x,x) instead | Phase 3 |
| fp16 overflow is a real constraint | Inputs > ~255 can overflow when squared; training data must be normalized | Phase 3 |

## Blockers

- **CRITICAL**: ANE cannot do attention, normalization, or loss — must investigate hybrid CPU+ANE approach
- seq=16 is the only safe sequence length — severe constraint for real training
- stories_mil.h port blocked — reference code uses ops rejected on this hardware
- Conv 2x1 and depthwise conv rejected — limits kernel flexibility

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 1 | 01-01 | 5min | 4 | 2 |
| 2 | 02-01 | 15min | 4 | 1 |
| 3 | 03-01 | 45min | 5 | 2 |

## Session History

| Date | Activity | Stopped At |
|------|----------|------------|
| 2026-03-25 | Project initialized, roadmap created | Ready for Phase 1 execution |
| 2026-03-25 | Phase 1 complete: fixed MIL_HEADER, smoke tests pass | Ready for Phase 2 execution |
| 2026-03-25 | Phase 2 complete: 30 tests run, critical finding — 8 ANE ops rejected | Phase 3 needs plan revision |
| 2026-03-25 | Phase 3 complete: 78 tests total, 9 ops + 7 decompositions. Attention/norm/loss must be CPU-only. | Ready for alternative investigation |
