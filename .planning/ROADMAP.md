# Roadmap: Rustane ANE Training

## Overview

Build ANE-accelerated transformer training by first empirically validating all known ANE constraints, then constructing fused MIL programs for forward and backward passes, implementing delta compilation for weight updates, and wiring it all into a training loop that beats CPU baseline.

## Milestones

- ✅ **M1: ANE Foundation** — Phases 1-3 (shipped 2026-03-26)
- ✅ **M2: Fused Training** — Phases 4-7 (shipped 2026-03-27)
- ✅ **M3: Production Readiness** — Phases 8-11 (shipped 2026-03-27)

## Phases

<details>
<summary>✅ M1: ANE Foundation (Phases 1-3) — SHIPPED 2026-03-26</summary>

### Phase 1: Fix & Smoke Test
**Goal**: Fix the MIL_HEADER buildInfo bug and verify constraint test infrastructure works

Plans:
- [x] 01-01: Fix MIL_HEADER, build, smoke test

### Phase 2: ANE Constraint Testing
**Goal**: Run all 30+ constraint tests and document which ANE constraints hold on this hardware

Plans:
- [x] 02-01: Run full constraint suite, analyze results, update docs

### Phase 3: Fused MIL Generators
**Goal**: Build Rust implementations of the fused forward programs from stories_mil.h

Plans:
- [x] 03-01: Build fwd_ffn_mil() with taps
- [x] 03-02: Build fwd_sdpa_mil() with taps

</details>

<details>
<summary>✅ M2: Fused Training (Phases 4-7) — SHIPPED 2026-03-27</summary>

### Phase 4: Backward Pass Correctness
**Goal**: All backward MIL generators produce correct gradients verified against numerical checks at realistic tensor sizes

Plans:
- [x] 04-01: Port FFN backward from stories_mil.h, ANE-compatible
- [x] 04-02: Port QKV backward from stories_mil.h
- [x] 04-03: Port SDPA backward parts 1+2 from stories_mil.h
- [x] 04-04: Numerical gradient verification for bwd_ffn and bwd_qkv

### Phase 5: Delta Compilation
**Goal**: Weight updates via delta compilation (patch + reload) work reliably within the ANE compile budget

Plans:
- [x] 05-01: Multi-layer delta compilation test with timing
- [x] 05-02: DeltaCompiler abstraction with compile budget tracking
- [x] 05-03: State survival verification across reload cycles

### Phase 6: Training Loop Integration
**Goal**: End-to-end training loop runs on synthetic data with ANE-accelerated forward and backward passes

Plans:
- [x] 06-01: ANE forward + CPU gradient training loop
- [x] 06-02: ANE vs CPU throughput benchmark

### Phase 7: Performance Benchmarking
**Goal**: Document final ANE training performance and tuning parameters for the parameter-golf model

Plans:
- [x] 07-01: Multi-config benchmark (DIM/SEQ sweep including target 768/256)

</details>

<details>
<summary>✅ M3: Production Readiness (Phases 8-11) — SHIPPED 2026-03-27</summary>

### Phase 8: Inference Correctness at Production Sizes
**Goal**: ANE inference produces correct results at DIM≥768, SEQ≥256
**Success Criteria**: ✅ All met
- FFN avg relative error <0.1% at D=768..2048
- QKV projection correct at all tested sizes (2.5-7.9% avg rel error)
- Documented accuracy at D=768/1024/2048, SEQ=256/512

### Phase 9: fp16 Overflow Mitigation
**Goal**: Training works correctly at D=1024+ without fp16 overflow corruption
**Success Criteria**: ✅ Non-issue — no mitigation needed
- Zero overflow at D=256..2048 with proper init (scale=0.02)
- All values stay well within fp16 range (±65504)
- Max observed value ~0.1 across all intermediates

### Phase 10: CPU Attention Optimization
**Goal**: Reduce CPU attention bottleneck
**Success Criteria**: ✅ Already optimal — no optimization needed
- BLAS cblas_sgemm dominates (not softmax loop)
- 12-head attention: ~1.6ms/layer, only 19% of 100ms step
- Softmax fusion variants are slightly slower, not faster

### Phase 11: Final Integration Benchmark
**Goal**: Document final production-ready performance numbers
**Success Criteria**: ✅ All met
- End-to-end benchmark at D=768, SP=256, 12 layers (113M params)
- Correctness verified at all benchmarked sizes (loss stable, no NaN/Inf)
- Forward: 31ms (2.1x speedup), Step: 175ms (1.2x speedup), 5.7 steps/sec

</details>

## Progress

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| 1. Fix & Smoke Test | M1 | ✅ Complete | 2026-03-26 |
| 2. Constraint Testing | M1 | ✅ Complete | 2026-03-26 |
| 3. Fused MIL Generators | M1 | ✅ Complete | 2026-03-26 |
| 4. Backward Pass | M2 | ✅ Complete | 2026-03-26 |
| 5. Delta Compilation | M2 | ✅ Complete | 2026-03-26 |
| 6. Training Loop | M2 | ✅ Complete | 2026-03-26 |
| 7. Benchmarking | M2 | ✅ Complete | 2026-03-27 |
| 8. Inference Correctness | M3 | ✅ Complete | 2026-03-27 |
| 9. fp16 Overflow | M3 | ✅ Complete | 2026-03-27 |
| 10. CPU Attention | M3 | ✅ Complete | 2026-03-27 |
| 11. Final Benchmark | M3 | ✅ Complete | 2026-03-27 |
