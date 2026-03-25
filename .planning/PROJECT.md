# Rustane — ANE-Accelerated Transformer Training

## What This Is

A Rust library that leverages Apple's Neural Engine (ANE) for transformer model training, following the Orion paper architecture (arXiv:2603.06728). It compiles fused MIL programs targeting the ANE to achieve actual speedup over CPU for small transformer models (parameter-golf architecture).

## Core Value

Fused ANE programs that train transformers faster than CPU — not individual op benchmarks, but end-to-end training throughput improvement.

## Requirements

### Validated

- ✓ ANE FFI bridge compiles and links — can invoke `ane_bridge_compile_multi_weights` from Rust
- ✓ `conv1x1_mil()` generates valid ANE MIL and produces correct results — `programs.rs` is the ground truth
- ✓ Weight blob format works — `WeightBlob::from_f32()` produces correct ANE-compatible blobs
- ✓ fp32 I/O with internal fp16 cast works — `fp32 input → cast → conv → cast → fp32 output` compiles and evals
- ✓ ANE MIL syntax discovered — `{{`/`}}` only in buildInfo dict outer wrapper, single braces elsewhere
- ✓ All 30+ ANE constraint tests run in subprocess isolation with clear PASS/FAIL results
- ✓ 9 ANE ops confirmed working + 7 decomposition strategies (sub, div, exp, log, sqrt, abs, relu)
- ✓ reduce_sum, softmax, pow compile on ANE with correct MIL syntax (Phase 3.5 breakthrough)
- ✓ Full SDPA MIL pipeline compiles on ANE
- ✓ RMSNorm MIL generator works on ANE
- ✓ conv1x1 for weight multiplication, matmul for QK^T and AV (reference pattern)

### Active

- [ ] Fused backward MIL programs that produce correct gradients
- [ ] Delta compilation (weight patch + reload) under 500ms
- [ ] End-to-end training loop faster than CPU-only baseline
- [ ] Inference correctness at production sizes (DIM≥768, SEQ≥256)
- [ ] concat workaround (only truly rejected op — blocks gradient taps output)

### Out of Scope

- CoreML model conversion (`.mlmodel`/`.mlpackage`) — we use raw MIL text directly
- GPU/MPS fallback — ANE or CPU only
- Training on real datasets (FineWeb etc.) — synthetic data for validation
- Multi-device ANE — single ANE only
- Model serving/inference — training only

## Context

- **ANE MIL ≠ CoreML MIL**: The ANE compiler uses a proprietary MIL dialect. Key difference: buildInfo dict requires `{{`/`}}` (double braces) for the outer wrapper only. Standard CoreML MIL uses single braces.
- **Reference codebase**: `~/dev/ANE/training/stories_mil.h` contains working fused MIL generators in Objective-C — these are the templates for our Rust implementations.
- **Orion paper** (arXiv:2603.06728): Defines 20 ANE constraints discovered through reverse-engineering. Our goal is to test all 20 empirically before building fused programs.
- **Prior session findings**: Discovered the `{{`/`}` buildInfo syntax requirement, weight name matching (`@model_path/weights/W.bin`), and that all constraint tests were failing due to a buildInfo brace-doubling bug in `MIL_HEADER`.
- **Hardware**: Apple M4 (or similar), macOS with ANE support.
- **Existing codebase**: Has ANE FFI bridge, `conv1x1_mil()`, `rmsnorm_mil()`, `linear_matmul_mil()`, training loop infrastructure. Many untracked files from prior exploration.

## Constraints

- **Platform**: macOS only (ANE is Apple hardware)
- **Language**: Rust — all MIL generation must be in Rust
- **Compilation limit**: ~119 ANE compiles per process (Orion #3) — delta compilation required
- **Batch size**: Must be 1 (Orion #8)
- **Data types**: fp16 and fp32 only (Orion #9)
- **Minimum surface**: ~49KB IOSurface allocation (Orion #4)
- **Weight format**: BLOBFILE with offset=64, named with full `@model_path/weights/` path

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use conv1x1 instead of matmul | Matmul had eval failures, conv works reliably | ✓ Good |
| fp32 I/O with internal fp16 cast | Matches programs.rs proven pattern, easier I/O | ✓ Good |
| Subprocess isolation for constraint tests | Compile failures can corrupt ANE state | ✓ Good |
| Follow stories_mil.h for fused programs | Proven working Objective-C reference code | — Pending |
| Delta compilation over recompilation | 8.5x faster weight updates (Orion) | — Pending |

## Current Milestone: M2 Fused Training

**Goal:** Build fused backward MIL programs, delta compilation, and wire end-to-end training loop that's faster than CPU.

**Target features:**
- Fused backward MIL generators (FFN, QKV, SDPA)
- Delta compilation with compile limit tracking
- End-to-end training loop on synthetic data
- Performance benchmarking vs CPU baseline

---

*Last updated: 2026-03-26 — M2 Fused Training started*
