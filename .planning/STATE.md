# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** M2 Fused Training — backward pass, delta compilation, end-to-end training loop

## Current Position

**Milestone:** M2: Fused Training
**Phase:** Not started (defining requirements)
**Plan:** —
**Status:** Defining requirements
**Last activity:** 2026-03-26 — Milestone M2 started

## Progress

```
[██████████] 100% — M1: ANE Foundation (COMPLETE)
[░░░░░░░░░░]   0% — M2: Fused Training
```

## Accumulated Context from M1

### Key Decisions

| Decision | Rationale | Source |
|----------|-----------|--------|
| MIL_HEADER uses programs.rs exact buildInfo pattern (line 214) | Byte-level proof that {{ }} only on outermost dict wrapper | Phase 1 |
| Subprocess isolation for each ANE constraint test | Compile failures can corrupt ANE state | Phase 1 |
| ANE MIL parser requires exact stories_mil.h syntax | Named constants for all params, values=() for concat, bool for matmul transpose | Phase 3.5 |
| reduce_sum/softmax/pow/RMSNorm/SDPA compile on ANE | Corrected MIL syntax unlocks these ops | Phase 3.5 |
| matmul works between activations, not with BLOBFILE weights | Reference code uses conv1x1 for weight mult, matmul for QK^T/AV | Phase 3.5 |
| concat is the only truly rejected op | values=(...) syntax still fails | Phase 3.5 |
| Inference errors are size-related, not op-related | Reference uses DIM=768 SEQ=256, we test dim=64 seq=16 | Phase 3.5 |
| fp16 overflow is a real constraint | Inputs > ~255 can overflow when squared; training data must be normalized | Phase 3 |
| sub decomposition via mul+add is production-viable | add(x, mul(y, -1)) replaces rejected sub | Phase 3 |

### Validated Capabilities

- 9 ANE ops confirmed working (conv2d, transpose, matmul, add, mul, reduce_sum, softmax, pow, cast)
- 7 decomposition strategies (sub, div, exp, log, sqrt, abs, relu)
- Full SDPA MIL pipeline compiles on ANE
- RMSNorm MIL generator works on ANE
- conv1x1 for weight multiplication, matmul for QK^T and AV

### Carried Blockers

- **HIGH**: Inference errors on newly-compiled ops — need larger tensor sizes (DIM≥768, SEQ≥256)
- **MEDIUM**: concat is truly rejected — blocks gradient taps output and ANEMLL trick
- **LOW**: seq=16 is the only safe sequence length — severe for real training

## Session Continuity

Last session: 2026-03-26
Stopped at: Milestone M2 started, defining requirements
Resume file: —
