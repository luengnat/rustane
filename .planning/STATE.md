# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** Dynamic weight research (post-M2)

## Current Position

**Milestone:** M2: Fused Training — COMPLETE
**Phase:** Post-M2 — ANE training feasibility analysis COMPLETE
**Plan:** N/A (investigation concluded)
**Status:** ANE training is NOT faster than CPU BLAS. Definitive analysis complete.
**Last activity:** 2026-03-26 — Batched training benchmark, Orion paper analysis, backward pass exploration

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
Stopped at: Multi-layer ANE inference pipeline — 13.4x speedup at 24 layers
Resume file: None

## ANE Training Feasibility — Definitive Analysis (THIS SESSION)

### Why ANE Training Is NOT Faster Than CPU

The backward pass dominates training time and ANE cannot accelerate it.

**Timing breakdown at D=768, SP=256 (FFN layer, BLAS CPU):**

| Component | ANE | CPU (BLAS) | Notes |
|-----------|-----|-----------|-------|
| Forward pass | 0.75ms | 4.2ms | **5.6x faster** on ANE |
| Backward pass | N/A | 26.5ms | ANE cannot do backward ops |
| Weight reload | 73ms | 0ms | ANE must reload after weight update |
| **Total/step** | **~104ms** | **~30.7ms** | **ANE 3.4x SLOWER** |

**Even with zero-cost reload**: ANE would be 27.25ms vs CPU 30.7ms = only **1.13x faster** (13%).

The forward pass saves only 3.4ms, but backward costs 26.5ms. The ANE simply cannot do backward.

### ANE Ops That FAIL (error 0x1d, statusType=0x9)

These are essential backward pass operations:
- `transpose` — needed for W^T @ dY
- `matmul` (as matmul) — needed for gradient computation
- `add` (2 inputs) — needed for residual gradient
- `reduce_mean` — needed for loss gradient

### ANE Ops That WORK
- `conv1x1` (is matmul with const weights), `softmax`, `sigmoid`, `mul`, `concat`, `cast`

### Conv1x1 Gradient Trick — Analyzed and Rejected

Can conv1x1 compute backward matmuls? `dL/dW = dY @ X^T`:
- Mathematically: conv1x1(dY_as_weight, X_layout_as_input) = dY @ X^T ✓
- But dY changes every step → must reload as weight → ~73ms per reload
- CPU BLAS does the same matmul in ~4ms
- **18x slower than CPU** due to reload overhead

### Batched Training — Analyzed and Rejected

Amortize reload over N CPU steps: break-even at N > 73ms / 3.4ms ≈ 22 steps.

| Batch Size | ANE Total | Per Step | Speedup vs CPU |
|-----------|-----------|----------|----------------|
| 1 | 10,119ms | 101ms | 0.31x |
| 5 | 4,133ms | 41ms | 0.76x |
| 10 | 3,739ms | 37ms | 0.84x |
| 20 | 3,945ms | 39ms | 0.79x |
| 50 | 3,274ms | 33ms | 0.95x |
| 100 | 2,921ms | 29ms | 1.07x (noise) |

Batch=100 appears 7% faster but ANE forward output is NOT used for training — it's pure CPU + 1 reload. The "speedup" is CPU timing variance.

### Orion Paper Comparison (arXiv:2603.06728)

The Orion paper does NOT claim ANE training is faster than CPU:
- Their "3.8x speedup" is vs naive full-recompile-every-step (4,200ms/step)
- Their reload: 494ms vs our 73ms — **we're 6.7x faster at reload**
- Their training: 1,000 steps in 22 min (1.32s/step) — our CPU BLAS: 30ms/step (**44x faster**)
- Orion's value: enables on-device training via private APIs, not speed advantage
- Orion's LoRA adapter-as-input: inject LoRA via IOSurface input, no recompilation needed

### ANE Value Proposition

| Use Case | ANE Speedup | Status |
|----------|------------|--------|
| **Inference** (eval) | **4,576x** vs CPU | ✅ Incredible value |
| **Inference** (incl compile) | **14.4x** vs CPU | ✅ Strong value |
| **Training** (per-step) | **0.3x** vs CPU BLAS | ❌ Slower |
| **Training** (batched) | **1.0x** vs CPU BLAS | ❌ No benefit |

### Path Forward Options

1. **✅ DONE: Multi-layer ANE inference pipeline** — 13.4x speedup at 24 layers
2. **LoRA adapter-as-input** — Orion's approach: inject LoRA via IOSurface input, no recompilation
3. **Hybrid inference/training** — ANE for inference, CPU for training (current state)
4. **Accept limitation** — ANE is for inference, not training (honest conclusion)

## Multi-Layer ANE Inference Results

### inference_pipeline.rs — Multi-layer FFN benchmark

| Layers | Params | ANE Total | CPU Total | Speedup | Throughput |
|--------|--------|-----------|-----------|---------|------------|
| 6 | 42.5M | 2.3ms | 26.7ms | **11.7x** | 112K tok/s |
| 12 | 84.9M | 4.5ms | 48.7ms | **10.8x** | 57K tok/s |
| 24 | 169.9M | 9.3ms | 124.7ms | **13.4x** | 28K tok/s |

- Consistent ~0.38ms per layer regardless of depth
- All layers compile successfully (~80ms/layer)
- Correctness: 0.4% avg relative error (fp16 precision)
- Speedup *increases* with more layers (CPU slows down, ANE stays constant)
- No ANE memory issues at 24 layers (170M params)

## Hybird-Batch-Prefill-on-ANE Discoveries (THIS SESSION)

### 47. `_ANEInMemoryModel` requires `program(1.3)`, NOT `program(1.0)`
- Hybird uses `program(1.0)` because it compiles via CoreML's public API
- Our `_ANEInMemoryModel` direct compile requires `program(1.3)` with full buildInfo
- `program(1.0)` → InvalidMILProgram on our runtime (works on hybird's)
- `program(1.3)` → compiles successfully on our runtime
- This was THE critical fix — previous benchmarks may have been invalid due to wrong MIL version

### 48. Weight dict offset must be `NSNumber numberWithInt:0`, not `numberWithUnsignedChar:0`
- Our code used `NSNumber::new_u8(0)` → ANE compiler ignores it
- Hybird uses `[NSNumber numberWithInt:0]` → works correctly
- Fixed to `NSNumber::new_i32(0)` in `create_weight_dictionary`

### 49. Weight dict keys must be full BLOBFILE path `@model_path/weights/{name}.bin`
- ANE compiler looks up BLOBFILE path in weight dictionary
- Key must match MIL's `BLOBFILE(path = tensor<string, []>("@model_path/weights/W.bin"), ...)`
- Hybird constructs: `snprintf(buf, ..., "@model_path/weights/%s.bin", name)`

### 50. BLOBFILE MIL closing syntax: `)]  ;` not `)]];`
- ANE MIL parser requires exactly `)]  ;` (with two spaces before semicolon) to close const with BLOBFILE
- `)]];` causes InvalidMILProgram — discovered through binary comparison

### 51. ALL 7 ANE correctness tests pass with 100% accuracy (D=64, SP=32)
| Test | Op | Time | Accuracy |
|------|-----|------|----------|
| 10 | Conv1x1 | 56μs | 100% |
| 11 | Softmax | 92μs | 100% |
| 12 | Fused 2-Conv | 48μs | 100% |
| 13 | Fused 3-Conv QKV | 87μs | 100% |
| 14 | Fused FFN SwiGLU | 85μs | 100% |
| 15 | Batch Prefill | 78μs | 100% |
| 16 | Dim Sweep (32-1024) | 57-65μs | 100% |

### 52. Softmax WORKS on ANE (contradicts prior belief)
- Previously believed softmax crashes on ANE (SIGSEGV)
- Softmax standalone passes with 100% accuracy at 92μs
- Softmax in fused context may still fail (mega SDPA test)
- All softmax sum≈1 verified (32/32)

### 53. Concat WORKS in fused programs
- Fused 2-conv + concat: 48μs, 100% accuracy
- Fused 3-conv + concat: 87μs, 100% accuracy
- Previously concat was "the only truly rejected op" — now confirmed working

### 54. Dimension sweep shows consistent ~60μs latency across D=32-1024
- Suggests minimum fixed overhead ~60μs per ANE eval call
- At D=1024: 1.03 TFLOPS (still far from peak but correct)
- Need larger spatial dimensions (SP) and fused programs for better throughput

## Maderix Pattern Discoveries (PREVIOUS SESSION)

### 41. Spatial packing [1,IC,1,SEQ+OC] is 5× faster than channel packing [1,D+D*D,1,S] at D=768
- At D=64: channel 58us vs spatial 68us (channel wins at small dims)
- At D=512: channel 433us vs spatial 87us (5× spatial wins at large dims)
- At D=768: channel 687us vs spatial 136us (5× spatial wins)
- Spatial scales with compute, channel is bottlenecked by slice/reshape overhead

### 42. Fused QKV projection: 3 matmuls in 128us at D=768, S=256 (7 TFLOPS)
- 43μs per matmul when fused (vs 136μs for single matmul = 3.2× fusion win)
- Approaches ANE peak (6.6 TFLOPS M4, 19 TFLOPS FP16 theoretical)

### 43. Fused FFN (SwiGLU): 3 matmuls + SiLU + gate + residual in 520us at D=768, S=256
- sigmoid, mul, add all work on ANE (only softmax/gelu crash)
- 7.0 TFLOPS sustained throughput

### 44. fp16 direct I/O 11% faster than fp32 cast I/O at D=768
- fp16: 149us (2.0 TFLOPS), fp32: 167us (1.8 TFLOPS)
- Difference shrinks at large dims (compute dominates)

### 45. Mega SDPA (QKV + attention + softmax) COMPILE FAIL
- Softmax in fused context causes InvalidMILProgram
- Need CPU fallback for softmax (matches maderix approach)

### 46. Fusion scaling: 4+ matmuls with spatial packing COMPILE FAIL
- Too many const declarations overwhelm MIL compiler
- Solution: split into separate kernels (maderix uses 10 per layer)

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
