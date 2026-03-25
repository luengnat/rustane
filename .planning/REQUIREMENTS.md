# Requirements: M2 Fused Training

## Milestone Goal

Build fused backward MIL programs, implement delta compilation for weight updates, wire end-to-end training loop with synthetic data, and demonstrate ANE training speedup over CPU baseline.

## BWD — Backward Pass MIL Generators

- [ ] **BWD-01**: `bwd_ffn_mil()` produces gradients matching numerical gradient check (±1% tolerance) for a 2-layer FFN
- [ ] **BWD-02**: `bwd_qkv_mil()` produces correct gradients for query, key, value projections in attention
- [ ] **BWD-03**: `bwd_sdpa_mil()` produces correct attention gradients (dK, dV, dQ from dAttn)
- [ ] **BWD-04**: Backward generators use only ANE-verified ops (no concat) with decomposition strategies where needed
- [ ] **BWD-05**: Gradient taps (intermediate activations saved from forward pass) are correctly passed to backward MIL programs

## DLT — Delta Compilation

- [ ] **DLT-01**: Delta compilation (unload → patch weights → reload) completes in under 500ms for a 4-layer model
- [ ] **DLT-02**: Compile count is tracked per process; warning emitted when approaching ~100 compile limit
- [ ] **DLT-03**: Weight patch correctly updates only changed weights in the existing ANE program without full recompilation
- [ ] **DLT-04**: ANE program state (IOSurface buffers, compile handles) survives across delta compilation cycles

## TRL — Training Loop Integration

- [ ] **TRL-01**: One complete training step (forward + backward + weight update via SGD) runs without errors on synthetic data
- [ ] **TRL-02**: Loss decreases over 100 training steps on synthetic data, demonstrating learning
- [ ] **TRL-03**: ANE training throughput (steps/sec) exceeds CPU-only baseline (even 1.1x counts)
- [ ] **TRL-04**: Training loop handles the compile limit gracefully — either delta compilation or subprocess rotation

## PERF — Performance Benchmarking

- [ ] **PERF-01**: Final benchmark reports ANE vs CPU throughput for parameter-golf model (DIM=768, SEQ=256)
- [ ] **PERF-02**: Compile time, eval time, and dispatch overhead are measured and documented
- [ ] **PERF-03**: Sequence length and channel dimension tuning are documented with measured impact on throughput

## Future Requirements (Deferred)

- [ ] concat workaround strategy (alternative to rejected concat op for gradient taps)
- [ ] Larger tensor inference validation (DIM≥768, SEQ≥256 correctness)
- [ ] ANEMLL RMSNorm trick verification
- [ ] Multi-layer model support (beyond 4 layers)
- [ ] Learning rate scheduling
- [ ] Gradient clipping

## Out of Scope

- CoreML model conversion (`.mlmodel`/`.mlpackage`) — we use raw MIL text directly
- GPU/MPS fallback — ANE or CPU only
- Training on real datasets (FineWeb etc.) — synthetic data for validation
- Multi-device ANE — single ANE only
- Model serving/inference — training only
- Optimizer beyond SGD — Adam/AdamW deferred

## Traceability

| REQ-ID | Phase | Plan | Status |
|--------|-------|------|--------|
| BWD-01 | Phase 4 | — | Pending |
| BWD-02 | Phase 4 | — | Pending |
| BWD-03 | Phase 4 | — | Pending |
| BWD-04 | Phase 4 | — | Pending |
| BWD-05 | Phase 4 | — | Pending |
| DLT-01 | Phase 5 | — | Pending |
| DLT-02 | Phase 5 | — | Pending |
| DLT-03 | Phase 5 | — | Pending |
| DLT-04 | Phase 5 | — | Pending |
| TRL-01 | Phase 6 | — | Pending |
| TRL-02 | Phase 6 | — | Pending |
| TRL-03 | Phase 6 | — | Pending |
| TRL-04 | Phase 6 | — | Pending |
| PERF-01 | Phase 7 | — | Pending |
| PERF-02 | Phase 7 | — | Pending |
| PERF-03 | Phase 7 | — | Pending |
