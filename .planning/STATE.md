# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** Fused ANE programs that train transformers faster than CPU — not individual op benchmarks, but end-to-end training throughput improvement.
**Current focus:** Phase 4 — Backward Pass Correctness

## Current Position

**Milestone:** M2: Fused Training
**Phase:** 4 of 7 (Backward Pass Correctness)
**Plan:** 0 of ? in current phase
**Status:** Ready to plan
**Last activity:** 2026-03-26 — M2 roadmap created

## Progress

```
[██████████] 100% — M1: ANE Foundation (COMPLETE)
[░░░░░░░░░░]   0% — M2: Fused Training — Phase 4 ready to plan
```

## Accumulated Context from M1

### Key Decisions

| Decision | Rationale | Source |
|----------|-----------|--------|
| MIL_HEADER uses programs.rs exact buildInfo pattern (line 214) | Byte-level proof that {{ }} only on outermost dict wrapper | Phase 1 |
| Subprocess isolation for each ANE constraint test | Compile failures can corrupt ANE state | Phase 1 |
| ANE MIL parser requires exact stories_mil.h syntax | Named constants for all params, values=() for concat, bool for matmul transpose | Phase 3 |
| reduce_sum/softmax/pow/RMSNorm/SDPA compile on ANE | Corrected MIL syntax unlocks these ops | Phase 3 |
| matmul works between activations, not with BLOBFILE weights | Reference code uses conv1x1 for weight mult, matmul for QK^T/AV | Phase 3 |
| concat is the only truly rejected op | values=(...) syntax still fails | Phase 3 |
| Inference errors are size-related, not op-related | Reference uses DIM=768 SEQ=256, we test dim=64 seq=16 | Phase 3 |

### Validated Capabilities

- 9 ANE ops confirmed working (conv2d, transpose, matmul, add, mul, reduce_sum, softmax, pow, cast)
- 7 decomposition strategies (sub, div, exp, log, sqrt, abs, relu)
- Full SDPA MIL pipeline compiles on ANE
- RMSNorm MIL generator works on ANE
- conv1x1 for weight multiplication, matmul for QK^T and AV

### Carried Blockers

- **HIGH**: Inference errors on newly-compiled ops — need larger tensor sizes (DIM≥768, SEQ≥256)
- **HIGH**: concat is truly rejected — blocks gradient taps output; backward generators must work without concat
- **LOW**: seq=16 is the only safe sequence length — backward pass must validate at realistic sizes first

## Session Continuity

Last session: 2026-03-26
Stopped at: M2 roadmap created, ready to plan Phase 4
Resume file: None
