# Requirements: M2 Fused Training

## Milestone Goal

Build fused backward MIL programs, implement delta compilation for weight updates, wire end-to-end training loop with synthetic data, and demonstrate ANE training speedup over CPU baseline.

## BWD — Backward Pass MIL Generators

- [x] **BWD-01**: `bwd_ffn_mil()` produces gradients matching numerical gradient check (±1% tolerance) for a 2-layer FFN
- [x] **BWD-02**: `bwd_qkv_mil()` produces correct gradients for query, key, value projections in attention
- [x] **BWD-03**: `bwd_sdpa_mil()` produces correct attention gradients (dK, dV, dQ from dAttn)
- [x] **BWD-04**: Backward generators use only ANE-verified ops (no concat) with decomposition strategies where needed
- [x] **BWD-05**: Gradient taps (intermediate activations saved from forward pass) are correctly passed to backward MIL programs

## DLT — Delta Compilation

- [x] **DLT-01**: Delta compilation (unload → patch weights → reload) completes in under 500ms for a 4-layer model
- [x] **DLT-02**: Compile count is tracked per process; warning emitted when approaching ~100 compile limit
- [x] **DLT-03**: Weight patch correctly updates only changed weights in the existing ANE program without full recompilation
- [x] **DLT-04**: ANE program state (IOSurface buffers, compile handles) survives across delta compilation cycles

## TRL — Training Loop Integration

- [x] **TRL-01**: One complete training step (forward + backward + weight update via SGD) runs without errors on synthetic data
- [x] **TRL-02**: Loss decreases over 100 training steps on synthetic data, demonstrating learning
- [x] **TRL-03**: ANE training throughput (steps/sec) exceeds CPU-only baseline (even 1.1x counts)
- [x] **TRL-04**: Training loop handles the compile limit gracefully — either delta compilation or subprocess rotation

## PERF — Performance Benchmarking

- [x] **PERF-01**: Final benchmark reports ANE vs CPU throughput for parameter-golf model (DIM=768, SEQ=256)
- [x] **PERF-02**: Compile time, eval time, and dispatch overhead are measured and documented
- [x] **PERF-03**: Sequence length and channel dimension tuning are documented with measured impact on throughput

## Traceability

| REQ-ID | Phase | Plan | Status |
|--------|-------|------|--------|
| BWD-01 | Phase 4 | 04-04 | ✅ Complete |
| BWD-02 | Phase 4 | 04-04 | ✅ Complete |
| BWD-03 | Phase 4 | 04-04 | ✅ Complete |
| BWD-04 | Phase 4 | 04-01..04-03 | ✅ Complete |
| BWD-05 | Phase 4 | 04-01..04-03 | ✅ Complete |
| DLT-01 | Phase 5 | 05-01 | ✅ Complete |
| DLT-02 | Phase 5 | 05-02 | ✅ Complete |
| DLT-03 | Phase 5 | 05-01 | ✅ Complete |
| DLT-04 | Phase 5 | 05-03 | ✅ Complete |
| TRL-01 | Phase 6 | 06-01 | ✅ Complete |
| TRL-02 | Phase 6 | 06-01 | ✅ Complete |
| TRL-03 | Phase 6 | 06-02 | ✅ Complete |
| TRL-04 | Phase 6 | 06-01 | ✅ Complete |
| PERF-01 | Phase 7 | 07-01 | ✅ Complete |
| PERF-02 | Phase 7 | 07-01 | ✅ Complete |
| PERF-03 | Phase 7 | 07-01 | ✅ Complete |
