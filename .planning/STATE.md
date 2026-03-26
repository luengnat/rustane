# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** Dynamic weight research (post-M2)

## Current Position

**Milestone:** M2: Fused Training — COMPLETE
**Phase:** Post-M2 research — dynamic weight ANE training
**Plan:** N/A (independent investigation)
**Status:** Dynamic matmul WORKS. ANE compute is ~3x faster than CPU matmul. Full training step at parity. Next: fused programs.
**Last activity:** 2026-03-26 — Dynamic matmul fixed and benchmarked

## Progress

```
[██████████] 100% — M1: ANE Foundation (COMPLETE)
[██████████] 100% — M2: Fused Training (COMPLETE)
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
| Multi-output return replaces concat for backward programs | ANE returns multiple named outputs in alphabetical order | Phase 4 |
| Sub decomposition: add(x, mul(y, const(-1.0))) | Consistent pattern for replacing rejected sub op | Phase 4 |
| Packed single-input + slice_by_size for backward programs | ANE only supports single input; activations packed and unpacked | Phase 4 |
| SDPA backward split into two MIL programs | bwd1 (dV + probs) feeds bwd2 (dQ + dK) | Phase 4 |
| DeltaCompiler owns ANEExecutor instances via RAII | Drop frees ANE resources automatically | Phase 5 |
| CompileBudgetMonitor via delegation (not inheritance) | Simpler API, no trait boilerplate | Phase 5 |
| memory_pool module commented out (untracked, broken) | Pre-existing compile errors, out of scope | Phase 5 |
| ANE forward + CPU gradient pattern for training loop | ANE backward needs larger tensors; CPU gradient is correct and sufficient | Phase 6 |
| Analytical MSE gradient (not numerical) for training | Faster and more accurate than finite differences | Phase 6 |
| DIM/SEQ sweep benchmark with graceful error handling | Large configs may fail on ANE — report results instead of crashing | Phase 7 |

### Validated Capabilities

- 9 ANE ops confirmed working (conv2d, transpose, matmul, add, mul, reduce_sum, softmax, pow, cast)
- 7 decomposition strategies (sub, div, exp, log, sqrt, abs, relu)
- Full SDPA MIL pipeline compiles on ANE
- RMSNorm MIL generator works on ANE
- conv1x1 for weight multiplication, matmul for QK^T and AV
- bwd_ffn_mil() — SwiGLU FFN backward with sigmoid gating and SiLU derivative
- bwd_qkv_mil() — QKV projection backward (3 transposed convolutions)
- bwd_sdpa_bwd1_mil() + bwd_sdpa_bwd2_mil() — Full SDPA backward (dV, dQ, dK)
- DeltaCompiler — Multi-layer program management with budget tracking
- reload_weights() verified across 20+ cycles without state corruption
- Compile count non-increase during reloads (DLT-02 verified)
- End-to-end training loop: ANE forward → CPU loss → CPU gradient → SGD → delta reload
- Loss decreases over 50 training steps on synthetic data (TRL-02 verified)
- ANE vs CPU throughput benchmark with timing breakdown
- Multi-config benchmark: (32,16) to (768,256) DIM/SEQ sweep

### Carried Blockers

- **HIGH**: Inference errors on newly-compiled ops — need larger tensor sizes (DIM≥768, SEQ≥256)
- **LOW**: seq=16 is the only safe sequence length — backward pass must validate at realistic sizes first

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 1 | 01-01 | 5min | 4 | 2 |
| 2 | 02-01 | 15min | 4 | 1 |
| 3 | 03-01 | 90min | 7 | 2 |
| 3.5 | (investigation) | 30min | 7 | 1 |
| 4 | 01-04 | 5min | 3 | 4 |
| 4 | 02-04 | — | 2 | 1 |
| 4 | 03-04 | — | 2 | 2 |
| 4 | 04-04 | — | 1 | 1 |
| 5 | 01-05 | 10min | 1 | 2 |
| 5 | 02-05 | 8min | 1 | 2 |
| 5 | 03-05 | 5min | 1 | 1 |
| 6 | 01-06 | 8min | 1 | 1 |
| 6 | 02-06 | 5min | 1 | 1 |
| 7 | 01-07 | 5min | 1 | 1 |

## Session History

| Date | Activity | Stopped At |
|------|----------|------------|
| 2026-03-25 | Project initialized, roadmap created | Ready for Phase 1 execution |
| 2026-03-25 | Phase 1 complete: fixed MIL_HEADER, smoke tests pass | Ready for Phase 2 execution |
| 2026-03-25 | Phase 2 complete: 30 tests run, critical finding — 8 ANE ops rejected | Phase 3 needs plan revision |
| 2026-03-25 | Phase 3 complete: 78 tests, 9 ops + 7 decompositions | Breakthrough investigation |
| 2026-03-25 | BREAKTHROUGH: MIL syntax fixes unlock reduce_sum, softmax, pow, full SDPA | Need larger tensors for eval |
| 2026-03-26 | M2 milestone started, roadmap created (4 phases) | Phase 4 planning |
| 2026-03-26 | Phase 4 complete: all backward MIL generators ported from stories_mil.h | Ready for Phase 5 delta compilation |
| 2026-03-26 | Phase 5 complete: DeltaCompiler, multi-layer tests, state survival verified | Ready for Phase 6 training loop |
| 2026-03-26 | Phase 6 complete: training loop + ANE vs CPU benchmark | Ready for Phase 7 benchmarking |
| 2026-03-26 | Phase 7 complete: multi-config benchmark | M2 COMPLETE |
| 2026-03-26 | Dynamic weights research: fixed reload, discovered dynamic input approach | Dynamic mul 2x faster than CPU |
| 2026-03-26 | Fixed dynamic matmul (const order + sw1 size bug), benchmarked training | ANE compute 3x faster, step at parity |

## Session Continuity

Last session: 2026-03-26
Stopped at: Dynamic matmul working, benchmarks show ANE compute 3x faster than CPU
Resume file: None

## Post-M2 Dynamic Weight Discoveries

### Dynamic Weight Approaches (from reference test_weight_patch.m)
| Approach | Status | Speed |
|----------|--------|-------|
| 1. Disk patch weights.bin | ❌ Fails | N/A |
| 2. unload/update/load weights | ❌ Fails | N/A |
| 3. _ANEWeight objects | ❌ Crashes | N/A |
| 4. IOSurface as weightsBuffer | ❌ Ignored | N/A |
| 5. Element-wise mul (weights in input) | ✅ Works | 14,282 steps/sec (2x CPU) |
| 6. Dynamic matmul (weights in input) | ✅ Works | ANE compute 3x faster |

### Key Bugs Found and Fixed
1. **ANE const declaration order matters**: `ws=[1,1,D,D]` must be declared before `sw1=[1,D*D,1,1]` or CompilationFailure
2. **sw1 slice size was wrong**: Must be `[1,D*D,1,1]` not `[1,1,1,1]` — captures all weight channels
3. **Reshape target for activations**: Must be `[1,1,D,S]` (ObjC pattern) not `[1,1,S,D]`

### ANE Training Performance (D=64, S=64)
| Component | Time | % of step |
|-----------|------|-----------|
| Pack weights (f32→bytes) | 28μs | 10.0% |
| Write to IOSurface | 17μs | 6.2% |
| ANE eval (matmul) | 95μs | 33.9% |
| Read from IOSurface | 2μs | 0.5% |
| CPU gradient + SGD | 137μs | 49.3% |
| **Total** | **280μs** | **100%** |

CPU matmul alone: ~280μs. ANE matmul: 95μs → **3x faster compute**.
Full step: ANE 3,558 steps/sec vs CPU 4,159 steps/sec → gradient on CPU is bottleneck.
Path to faster: fuse forward+activation into one ANE program to reduce overhead.
