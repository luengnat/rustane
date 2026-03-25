# Roadmap: Rustane ANE Training

## Overview

Build ANE-accelerated transformer training by first empirically validating all known ANE constraints, then constructing fused MIL programs for forward and backward passes, implementing delta compilation for weight updates, and wiring it all into a training loop that beats CPU baseline.

## Milestones

- ✅ **M1: ANE Foundation** - Phases 1-3 (shipped 2026-03-26)
- 🚧 **M2: Fused Training** - Phases 4-7 (in progress)

## Phases

<details>
<summary>✅ M1: ANE Foundation (Phases 1-3) — SHIPPED 2026-03-26</summary>

### Phase 1: Fix & Smoke Test
**Goal**: Fix the MIL_HEADER buildInfo bug and verify constraint test infrastructure works
**Plans**: 1 plan

Plans:
- [x] 01-01: Fix MIL_HEADER, build, smoke test

### Phase 2: ANE Constraint Testing
**Goal**: Run all 30+ constraint tests and document which ANE constraints hold on this hardware
**Plans**: 1 plan

Plans:
- [x] 02-01: Run full constraint suite, analyze results, update docs

### Phase 3: Fused MIL Generators
**Goal**: Build Rust implementations of the fused forward programs from stories_mil.h
**Plans**: 2 plans

Plans:
- [x] 03-01: Build fwd_ffn_mil() with taps
- [x] 03-02: Build fwd_sdpa_mil() with taps

</details>

### 🚧 M2: Fused Training (In Progress)

**Milestone Goal:** Build fused backward MIL programs, delta compilation, and wire end-to-end training loop that's faster than CPU.

#### Phase 4: Backward Pass Correctness
**Goal**: All backward MIL generators produce correct gradients verified against numerical checks at realistic tensor sizes
**Depends on**: Phase 3 (M1)
**Requirements**: BWD-01, BWD-02, BWD-03, BWD-04, BWD-05
**Success Criteria** (what must be TRUE):
  1. `bwd_ffn_mil()` gradients match numerical gradient check within ±1% tolerance at DIM=768, SEQ=256
  2. `bwd_qkv_mil()` produces correct gradients for Q, K, V projections verified against numerical check
  3. `bwd_sdpa_mil()` produces correct dK, dV, dQ gradients verified against numerical check
  4. All backward generators compile using only ANE-verified ops (no concat) with decomposition strategies applied where needed
  5. Gradient taps saved from forward pass are correctly wired as inputs to backward MIL programs
**Plans**: 4 plans

Plans:
- [ ] 04-01: Port FFN backward from stories_mil.h, ANE-compatible (sub→decomp, concat→multi-output)
- [ ] 04-02: Port QKV backward from stories_mil.h (simplest: 3 conv + 2 add)
- [ ] 04-03: Port SDPA backward parts 1+2 from stories_mil.h (softmax backward, matmul gradients)
- [ ] 04-04: Numerical gradient verification for bwd_ffn and bwd_qkv

#### Phase 5: Delta Compilation
**Goal**: Weight updates via delta compilation (patch + reload) work reliably within the ANE compile budget
**Depends on**: Phase 3 (M1) — builds on forward pass infrastructure, not backward
**Requirements**: DLT-01, DLT-02, DLT-03, DLT-04
**Success Criteria** (what must be TRUE):
  1. Delta compilation (unload → patch weights → reload) completes in under 500ms for a 4-layer model
  2. Weight patch correctly updates only changed weights; unchanged weights retain their compiled state
  3. Compile count is tracked per process with a warning emitted when approaching the ~100 limit
  4. ANE program state (IOSurface buffers, compile handles) survives across multiple delta compilation cycles
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

#### Phase 6: Training Loop Integration
**Goal**: End-to-end training loop runs on synthetic data with ANE-accelerated forward and backward passes
**Depends on**: Phase 4, Phase 5
**Requirements**: TRL-01, TRL-02, TRL-03, TRL-04
**Success Criteria** (what must be TRUE):
  1. One complete training step (forward → backward → SGD weight update → delta compile) runs without errors on synthetic data
  2. Loss decreases measurably over 100 training steps on synthetic data, demonstrating learning
  3. ANE training throughput (steps/sec) exceeds CPU-only baseline (even 1.1x counts as success)
  4. Training handles the ~119 compile limit gracefully via delta compilation, continuing without crashes or errors
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

#### Phase 7: Performance Benchmarking
**Goal**: Document final ANE training performance and tuning parameters for the parameter-golf model
**Depends on**: Phase 6
**Requirements**: PERF-01, PERF-02, PERF-03
**Success Criteria** (what must be TRUE):
  1. Benchmark report shows ANE vs CPU throughput (steps/sec) for DIM=768, SEQ=256 model
  2. Compile time, eval time, and dispatch overhead are individually measured and documented
  3. Sequence length and channel dimension tuning results documented with measured impact on throughput
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 4 → 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Fix & Smoke Test | M1 | 1/1 | Complete | 2026-03-26 |
| 2. Constraint Testing | M1 | 1/1 | Complete | 2026-03-26 |
| 3. Fused MIL Generators | M1 | 2/2 | Complete | 2026-03-26 |
| 4. Backward Pass | M2 | 0/? | Not started | - |
| 5. Delta Compilation | M2 | 0/? | Not started | - |
| 6. Training Loop | M2 | 0/? | Not started | - |
| 7. Benchmarking | M2 | 0/? | Not started | - |
