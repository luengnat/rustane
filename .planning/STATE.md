# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** Phase 3.5 breakthrough — MIL syntax fixes unlock reduce_sum, softmax, pow, full SDPA

## Current Position

**Milestone:** M1: ANE Foundation
**Phase:** 3 — ANE Op Alternatives (COMPLETE with BREAKTHROUGH)
**Plan:** 03-01 — Investigate all ANE op alternatives (COMPLETE)
**Status:** BREAKTHROUGH — most "rejected" ops were caused by incorrect MIL syntax, not hardware limits. Full SDPA pipeline compiles. Need to resolve inference errors and concat.

## Progress

```
[██████████] 100% — M1: ANE Foundation
```

## Decisions

| Decision | Rationale | Added |
|----------|-----------|-------|
| MIL_HEADER uses programs.rs exact buildInfo pattern (line 214) | Byte-level proof that {{ }} only on outermost dict wrapper | Phase 1 |
| Subprocess isolation for each ANE constraint test | Compile failures can corrupt ANE state | Phase 1 |
| ANE MIL parser requires exact stories_mil.h syntax | Named constants for all params, values=() for concat, bool for matmul transpose | Phase 3.5 |
| reduce_sum/softmax/pow/RMSNorm/SDPA compile on ANE | Corrected MIL syntax unlocks these ops | Phase 3.5 |
| matmul works between activations, not with BLOBFILE weights | Reference code uses conv1x1 for weight mult, matmul for QK^T/AV | Phase 3.5 |
| concat is the only truly rejected op | values=(...) syntax still fails | Phase 3.5 |
| Inference errors are size-related, not op-related | Reference uses DIM=768 SEQ=256, we test dim=64 seq=16 | Phase 3.5 |
| fp16 overflow is a real constraint | Inputs > ~255 can overflow when squared; training data must be normalized | Phase 3 |
| sub decomposition via mul+add is production-viable | add(x, mul(y, -1)) replaces rejected sub | Phase 3 |

## Blockers

- **HIGH**: Inference errors on newly-compiled ops — need larger tensor sizes (DIM≥768, SEQ≥256)
- **MEDIUM**: concat is truly rejected — blocks gradient taps output and ANEMLL trick
- **LOW**: seq=16 is the only safe sequence length — severe for real training

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 1 | 01-01 | 5min | 4 | 2 |
| 2 | 02-01 | 15min | 4 | 1 |
| 3 | 03-01 | 90min | 7 | 2 |
| 3.5 | (investigation) | 30min | 7 | 1 |

## Session History

| Date | Activity | Stopped At |
|------|----------|------------|
| 2026-03-25 | Project initialized, roadmap created | Ready for Phase 1 execution |
| 2026-03-25 | Phase 1 complete: fixed MIL_HEADER, smoke tests pass | Ready for Phase 2 execution |
| 2026-03-25 | Phase 2 complete: 30 tests run, critical finding — 8 ANE ops rejected | Phase 3 needs plan revision |
| 2026-03-25 | Phase 3 complete: 78 tests, 9 ops + 7 decompositions | Breakthrough investigation |
| 2026-03-25 | BREAKTHROUGH: MIL syntax fixes unlock reduce_sum, softmax, pow, full SDPA | Need larger tensors for eval |
