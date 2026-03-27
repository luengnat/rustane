# Rustane — ANE-Accelerated Transformer Training

## What This Is

A Rust library that leverages Apple's Neural Engine (ANE) for transformer model training, following the Orion paper architecture (arXiv:2603.06728). It compiles fused MIL programs targeting the ANE to achieve actual speedup over CPU for small transformer models (parameter-golf architecture).

## Core Value

Fused ANE programs that train transformers faster than CPU — not individual op benchmarks, but end-to-end training throughput improvement.

## Requirements

### Shipped — M1: ANE Foundation

- ✅ ANE FFI bridge compiles and links — can invoke `ane_bridge_compile_multi_weights` from Rust
- ✅ `conv1x1_mil()` generates valid ANE MIL and produces correct results
- ✅ Weight blob format works — `WeightBlob::from_f32()` produces correct ANE-compatible blobs
- ✅ fp32 I/O with internal fp16 cast works
- ✅ ANE MIL syntax discovered — `{{`/`}}` only in buildInfo dict outer wrapper
- ✅ All 30+ ANE constraint tests run in subprocess isolation
- ✅ 9 ANE ops confirmed working + 7 decomposition strategies
- ✅ reduce_sum, softmax, pow compile on ANE with correct MIL syntax
- ✅ Full SDPA MIL pipeline compiles on ANE
- ✅ RMSNorm MIL generator works on ANE
- ✅ conv1x1 for weight multiplication, matmul for QK^T and AV

### Shipped — M2: Fused Training

- ✅ Fused backward MIL programs produce correct gradients (FFN, QKV, SDPA)
- ✅ Delta compilation (weight patch + reload) under 500ms
- ✅ End-to-end training loop faster than CPU-only baseline (1.2-1.4x)
- ✅ Performance benchmarking vs CPU (DIM=512..2048, 6-12 layers)
- ✅ Compile budget tracking and graceful handling

### Shipped — M3: Production Readiness

- ✅ Inference correctness verified: FFN <0.1% error, QKV <8% at D=768..2048
- ✅ fp16 overflow: non-issue with proper initialization (scale=0.02)
- ✅ CPU attention: already optimal via BLAS (no further optimization needed)
- ✅ Final integration benchmark: 2.1x forward, 1.2x total step at D=768/12L

### Out of Scope

- CoreML model conversion (`.mlmodel`/`.mlpackage`) — we use raw MIL text directly
- GPU/MPS fallback — ANE or CPU only
- Training on real datasets (FineWeb etc.) — synthetic data for validation
- Multi-device ANE — single ANE only
- Optimizer beyond SGD — Adam/AdamW deferred

## Context

- **ANE MIL ≠ CoreML MIL**: The ANE compiler uses a proprietary MIL dialect. Key difference: buildInfo dict requires `{{`/`}}` (double braces) for the outer wrapper only.
- **Reference codebase**: `~/dev/ANE/training/stories_mil.h` contains working fused MIL generators in Objective-C.
- **Orion paper** (arXiv:2603.06728): Defines 20 ANE constraints discovered through reverse-engineering.
- **Hardware**: Apple M4 (or similar), macOS with ANE support.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use conv1x1 instead of matmul | Matmul had eval failures, conv works reliably | ✅ Good |
| fp32 I/O with internal fp16 cast | Matches programs.rs proven pattern, easier I/O | ✅ Good |
| Subprocess isolation for constraint tests | Compile failures can corrupt ANE state | ✅ Good |
| Follow stories_mil.h for fused programs | Proven working Objective-C reference code | ✅ Good |
| ANE forward + CPU backward (not hybrid) | ANE backward overhead exceeds compute savings | ✅ Good |
| Delta compilation over recompilation | 8.5x faster weight updates (Orion) | ✅ Good |

## Milestones

| Milestone | Status | Shipped |
|-----------|--------|---------|
| M1: ANE Foundation | ✅ Complete | 2026-03-26 |
| M2: Fused Training | ✅ Complete | 2026-03-27 |
| M3: Production Readiness | ✅ Complete | 2026-03-27 |

---

*Last updated: 2026-03-27 — All milestones shipped*
