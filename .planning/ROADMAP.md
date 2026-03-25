# Roadmap: Rustane ANE Training

## Overview

Build ANE-accelerated transformer training by first empirically validating all known ANE constraints, then constructing fused MIL programs for forward and backward passes, implementing delta compilation for weight updates, and wiring it all into the existing training loop.

## Milestones

- 🚧 **M1: ANE Foundation** - Phases 1-3 (in progress)
- 📋 **M2: Fused Training** - Phases 4-6 (planned)

## Phases

### Phase 1: Fix & Smoke Test
**Goal**: Fix the MIL_HEADER buildInfo bug and verify constraint test infrastructure works
**Depends on**: Nothing (first phase)
**Requirements**: ANE MIL syntax correctness
**Success Criteria** (what must be TRUE):
  1. `sigmoid_basic` test compiles and evals successfully (proves MIL syntax is correct)
  2. Constraint test orchestrator runs all tests in subprocess isolation without crashes
  3. Build compiles cleanly with no warnings
**Plans**: 1 plan

Plans:
- [ ] 01-01: Fix MIL_HEADER, build, smoke test

### Phase 2: ANE Constraint Testing
**Goal**: Run all 30+ constraint tests and document which ANE constraints hold on this hardware
**Depends on**: Phase 1
**Requirements**: All 20 Orion constraints empirically tested
**Success Criteria** (what must be TRUE):
  1. Every constraint test produces a clear PASS or FAIL result
  2. Results documented with compile times, eval times, and error messages
  3. `docs/ane_constraints.md` updated with empirical findings (not just Orion claims)
  4. Surprising results flagged (ops that work but shouldn't, or vice versa)
**Plans**: 1 plan

Plans:
- [ ] 02-01: Run full constraint suite, analyze results, update docs

### Phase 3: Fused MIL Generators
**Goal**: Build Rust implementations of the fused forward programs from stories_mil.h
**Depends on**: Phase 2
**Requirements**: Fused forward MIL programs, ANEMLL RMSNorm trick
**Success Criteria** (what must be TRUE):
  1. `fwd_ffn_mil()` compiles and evals with correct output (verified against CPU reference)
  2. `fwd_sdpa_mil()` compiles and evals with correct attention output
  3. Taps (intermediate activations for backward) are accessible as additional outputs
  4. All generators use correct ANE MIL syntax (verified by constraint test results)
**Plans**: 2 plans

Plans:
- [ ] 03-01: Build fwd_ffn_mil() with taps
- [ ] 03-02: Build fwd_sdpa_mil() with taps

### Phase 4: Backward Pass & Delta Compilation
**Goal**: Implement backward pass MIL programs and delta compilation for weight updates
**Depends on**: Phase 3
**Requirements**: Fused backward MIL programs, delta compilation
**Success Criteria** (what must be TRUE):
  1. `bwd_ffn_mil()` produces gradients matching numerical gradient check (±1%)
  2. `bwd_qkv_mil()` and `bwd_sdpa_mil()` produce correct attention gradients
  3. Delta compilation (unload → patch → reload) completes in <500ms
  4. Compile count tracked; warning emitted at ~100 compiles
**Plans**: 2 plans

Plans:
- [ ] 04-01: Build backward MIL generators (ffn, qkv, sdpa)
- [ ] 04-02: Implement delta compilation with compile limit tracking

### Phase 5: Training Loop Integration
**Goal**: Wire fused programs into end-to-end training loop with synthetic data
**Depends on**: Phase 4
**Requirements**: End-to-end training faster than CPU
**Success Criteria** (what must be TRUE):
  1. One training step (forward + backward + weight update) completes without errors
  2. Loss decreases over 100 steps on synthetic data
  3. ANE training throughput > CPU baseline (even 1.1x counts)
**Plans**: 1 plan

Plans:
- [ ] 05-01: Wire training loop, end-to-end test, benchmark vs CPU

### Phase 6: Optimize & Benchmark
**Goal**: Tune for maximum ANE throughput and document final performance
**Depends on**: Phase 5
**Requirements**: Performance optimization
**Success Criteria** (what must be TRUE):
  1. Sequence length padding strategy documented and benchmarked
  2. Channel dimension tuning documented
  3. Final benchmark: ANE vs CPU throughput for parameter-golf model
  4. Performance report with compile times, eval times, dispatch overhead breakdown
**Plans**: 1 plan

Plans:
- [ ] 06-01: Benchmark, tune, document final performance

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Fix & Smoke Test | M1 | 0/1 | Not started | - |
| 2. Constraint Testing | M1 | 0/1 | Not started | - |
| 3. Fused MIL Generators | M1 | 0/2 | Not started | - |
| 4. Backward & Delta | M2 | 0/2 | Not started | - |
| 5. Training Loop | M2 | 0/1 | Not started | - |
| 6. Optimize | M2 | 0/1 | Not started | - |
