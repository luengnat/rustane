# State: Rustane ANE Training

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** Fused ANE programs that train transformers faster than CPU
**Current focus:** Backward correctness verified, training benchmarks complete

## Current Position

**Milestone:** M2: Fused Training — COMPLETE
**Phase:** Post-M2 — Backward correctness verified and attention forward bug fixed
**Plan:** N/A (investigation concluded)
**Status:** Backward verified via numerical gradient checking; attention mm→mm_abt fix applied
**Last activity:** 2026-03-27 — Fixed attention forward bug (mm→mm_abt), verified backward correctness, benchmarked up to D=2048

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

### Backward Verification Results (2026-03-27)

**Root cause bug found:** `cpu_attention` used `mm(&attn, sp, sp, v, d)` which BLAS interprets V with stride `d` instead of `sp`. When `d ≠ sp`, this computes a scrambled matrix multiply instead of `attn @ V^T`. Fixed with `mm_abt(&attn, sp, sp, v, d)`.

**Verification suite** (`examples/verify_backward.rs`):
- Pure f64 reference: all rel errors < 5e-8 (math proven correct)
- Linear W@x → MSE: ✅ (rel ~1e-2)
- W@x + x (residual) → MSE: ✅ (rel ~2e-3)
- Attention-only (dWq/dWk/dWv/dWo): ✅ (rel ~7e-4)
- FFN-only (dWg/dWu/dWd): ✅ (rel ~1e-8, near-perfect)
- Full transformer layer (7 weights + dx): ✅ (rel ~3e-3)
- Gradient descent sanity check: ✅ (loss decreases)

**Training benchmarks after fix:**

| Config | Layers | Params | Fwd Speedup | Total Speedup |
|--------|--------|--------|-------------|---------------|
| D=512 6L | 6 | 25.2M | 2.08x | 1.28x |
| D=768 12L | 12 | 113.2M | 3.99x | 1.36x |
| D=1024 6L | 6 | 100.7M | 3.84x | 1.15x |
| D=1024 12L | 12 | 201.3M | 3.98x | 1.22x |
| D=2048 6L | 6 | 402.7M | 3.12x | 1.45x |

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
| 2026-03-27 | Backward correctness verified: found+fixed attention mm→mm_abt bug | All tests pass, benchmarks up to D=2048 |

## Session Continuity

Last session: 2026-03-27
Stopped at: Backward cleanup complete — 12L total speedup 1.21x→1.44x
Resume file: None

## Training Strategy — FINAL RECOMMENDATION (2026-03-26)

### Three strategies compared:

| Strategy | ANE Compiles | D=512 12L | D=768 12L | D=1024 12L |
|----------|-------------|-----------|-----------|------------|
| **ANE fwd + CPU bwd** | 1/layer | **1.42x** | **1.22x** | **1.22x** |
| Hybrid (ANE fwd+bwd) | 3/layer | 1.15x | 1.04x | 1.17x |
| Pure CPU | 0 | 1.00x | 1.00x | 1.00x |

### Why ANE backward is NOT worthwhile:
- ANE backward saves ~0.43ms/layer on 2 matmuls (micro-benchmark confirmed)
- But each ANE program call costs ~0.5-0.9ms (write_input + eval + read_output + fp16 conversion)
- 2 backward programs × ~0.7ms overhead = ~1.4ms overhead, saving only ~0.43ms compute
- Net: ANE backward LOSES ~1ms per layer (0.83x vs CPU)
- Plus wastes 2 ANE compiles per layer (critical given ~119 compile limit)

### Why ANE forward + CPU backward is optimal:
- ANE forward: 2.5-2.7x speedup, consistent across D=512-1024
- Only 1 ANE compile per layer (can support ~119 layers)
- CPU backward is already fast (BLAS-optimized)
- Total speedup: 1.22-1.42x depending on model size

### ANE I/O overhead breakdown (micro-benchmark, D=768):
| Component | Time |
|-----------|------|
| write_input (d*sp fp16) | 0.006ms |
| eval (ANE compute) | 0.188ms |
| read_output (inter*sp fp16) | 0.031ms |
| from_fp16 (inter*sp) | **0.264ms** |
| **Total ANE call** | **0.520ms** |
| CPU cblas_sgemm (same op) | 0.702ms |

The `from_fp16` conversion (0.264ms) is the largest single overhead — it's CPU-bound conversion of 786K fp16 values to fp32.

## End-to-End Training Step — Measured (FINAL)

### train_hybrid_step.rs results (D=768, SP=256)

| Layers | Hybrid Forward | CPU Forward | Fwd Speedup | Hybrid Backward | CPU Backward | Bwd Speedup | Hybrid Total | CPU Total | Total Speedup |
|--------|---------------|-------------|-------------|-----------------|-------------|-------------|-------------|-----------|---------------|
| 1       | 1.35ms        | 3.49ms      | 2.59x       | 8.61ms          | 10.66ms     | 1.24x       | 10.07ms      | 14.18ms   | 1.41x         |
| 6       | 7.52ms        | 21.26ms     | 2.83x       | 50.44ms         | 49.13ms     | 0.97x       | 58.07ms      | 70.41ms   | 1.21x         |
| 12      | 14.89ms       | 58.66ms     | 3.94x       | 111.39ms        | 116.51ms    | 1.05x       | 126.39ms     | 175.19ms  | 1.39x         |

### Why Backward Speedup is Modest

- ANE saves ~1.6ms/layer on input-gradient matmuls (3.6x faster than CPU)
- But write_input/read_output overhead: ~0.6ms per layer (2 programs × ~0.3ms I/O)
- CPU weight gradients: ~3ms/layer (can't use ANE — activations change every step)
- Net: ANE saves 1.6ms but pays 0.6ms overhead = 1.0ms net saving vs ~8ms total = ~12% improvement
- Forward scales much better: 1 program per layer, single I/O, larger matmuls dominate

### Key Insight

**ANE training IS faster than CPU (1.2-1.4x for FFN-only), but the speedup is modest.**
The real win is in inference: 3-4x forward speedup that scales with depth.
For training, the backward pass is dominated by weight gradients which must stay on CPU.

## ANE Backward Pass — NOW WORKING (BREAKTHROUGH)

### Strategy: Split backward — ANE does matmuls, CPU does element-wise

ANE can't handle too many ops in one program, and element-wise ops have negligible cost.
Solution: ANE handles the expensive matmuls via conv1x1, CPU handles SiLU', mul, add.

**Program A (ANE):** dfused = Wd^T @ dy — single conv1x1, 1 weight
**CPU:** SiLU'(gate), dsilu = dfused*up, dup = dfused*silu, dgate = dsilu*SiLU'
**Program B (ANE):** dx_partial = [WgT|WuT] @ concat(dgate, dup) — concat + conv1x1
**CPU:** dx = dx_partial + dy

### Performance Results

| D | inter | SP | CPU backward | ANE matmuls | Hybrid | Speedup |
|---|-------|-----|-------------|-------------|--------|---------|
| 256 | 1024 | 256 | 0.92ms | 0.25ms | 0.93ms | 1.0x |
| 512 | 2048 | 256 | 2.50ms | 0.47ms | 1.59ms | 1.6x |
| 768 | 3072 | 256 | 4.92ms | 1.08ms | 2.69ms | 1.8x |
| 1024 | 4096 | 256 | 7.77ms | 1.64ms | 4.18ms | 1.9x |

ANE matmul speedup: ~3x vs CPU BLAS for D≥512.

### Accuracy

At D=768, SP=256: ALL EXCELLENT (avg_rel < 0.1%)
- dfused: 0.089% avg relative error
- dgate: 0.089%
- dL/dx: 0.10%

### Limitations

- **fp16 overflow at D=1024**: dx values exceed fp16 range (~65504), causing poor accuracy
- **fp16 subnormal at small scales**: Values < 6e-5 lose precision; need gradient magnitude > 0.01
- **Two ANE programs per layer**: Each has ~60μs overhead; limits benefit at small D

### Key Fixes This Session

1. **mm_at parameter swap bug**: cpu_backward_ffn had k and m swapped in all 3 mm_at calls
2. **conv1x1 weight shape**: Must be [out_channels, in_channels, 1, 1] = [inter, d, 1, 1]
3. **Weight blob layout**: Must match MIL shape — [inter, d] row-major, not [d, inter]
4. **Too many ops in one program**: SiLU' (8 ops) crashes ANE; split into matmul-only programs
5. **compile_multi arg count**: Missing output_sizes parameter and &[u8] vs Vec<u8] type mismatches

## ANE Training Feasibility — Revised Analysis (PREVIOUS)

### Why ANE Training IS Now Faster Than CPU (REVISED)

**Previous analysis was wrong** — we assumed ANE couldn't do backward at all.
New approach: ANE handles matmuls (conv1x1), CPU handles element-wise ops.

**Updated timing at D=768, SP=256 (FFN layer):**

| Component | ANE+CPU Hybrid | Pure CPU (BLAS) | Speedup |
|-----------|---------------|-----------------|---------|
| Forward pass | 0.75ms | 4.2ms | 5.6x |
| Backward (matmul) | 1.08ms | 3.88ms | 3.6x |
| Backward (element-wise) | 1.86ms | 1.86ms | 1.0x |
| **Total backward** | **2.93ms** | **5.74ms** | **2.0x** |
| **Total layer (fwd+bwd)** | **3.68ms** | **9.94ms** | **2.7x** |

**For a 6-layer transformer with CPU attention:**
- Forward: ANE FFN 5.6x, CPU attention unchanged → ~2-3x overall
- Backward: ANE FFN 2.0x, CPU attention unchanged → ~1.3x overall
- **Net training step: ~1.5-2x faster than CPU**

### ANE Ops That FAIL (error 0x1d, statusType=0x9)

These are essential backward pass operations:
- `transpose` — needed for W^T @ dY
- `matmul` (as matmul) — needed for gradient computation
- `add` (2 inputs) — needed for residual gradient
- `reduce_mean` — needed for loss gradient

### ANE Ops That WORK
- `conv1x1` (is matmul with const weights), `softmax`, `sigmoid`, `mul`, `concat`, `cast`

### Conv1x1 Gradient Trick — NOW WORKING

Can conv1x1 compute backward matmuls? YES!
- W^T @ dY: conv1x1(W^T_as_weight, dY_as_input) = W^T @ dY ✓
- Key insight: weights DON'T change during backward (only activations change)
- Split into 2 programs: one for Wd^T@dy, one for [WgT|WuT]@concat(dgate,dup)
- ANE matmul: 3.6x faster than CPU BLAS
- No weight reload needed (weights are const in the MIL program)

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
| **Training** (FFN only, fwd+CPU bwd) | **1.22-1.42x** vs CPU | ✅ Modest but real |
| **Training** (full transformer) | **1.21-1.43x** vs CPU | ✅ Realistic speedup |
| **Training** (full hybrid) | **1.04-1.17x** vs CPU | ⚠️ Marginal |

### Full Transformer Training Results (train_transformer.rs)

Realistic benchmark with attention + FFN per layer:
- ANE: QKV projection, output projection, FFN (3 programs/layer)
- CPU: multi-head attention, all backward passes, weight gradients

| Config | Layers | Params | Fwd Speedup | Bwd | Total Speedup |
|--------|--------|--------|-------------|-----|---------------|
| D=512 6L | 6 | 25.2M | 2.47x | 0.99x | **1.33x** |
| D=768 6L | 6 | 56.6M | 3.25x | 1.02x | **1.42x** |
| D=768 12L | 12 | 113.2M | 3.36x | 1.02x | **1.44x** |

Forward speedup consistent ~3.0-3.4x across configs.
Backward now at CPU parity (1.0x) after dead code cleanup — was 0.82x at 12L due to 6 wasted BLAS calls.
Fwd/Bwd ratio: ~18-23% forward, ~77-82% backward — limits total speedup.

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

### Hybrid Transformer Inference Results

#### inference_transformer.rs — Full transformer (ANE linear + CPU attention)

| Layers | Params | Programs | ANE | CPU | Speedup | Throughput |
|--------|--------|----------|-----|-----|---------|------------|
| 6 | 56.6M | 18 | 12.3ms | 41.8ms | **3.4x** | 20.7K tok/s |
| 12 | 113.2M | 36 | 24.6ms | 71.9ms | **2.9x** | 10.4K tok/s |

Per-layer time breakdown at 12 layers:
- ANE QKV projection: 0.244ms (3 conv1x1 fused)
- CPU attention (BLAS): 0.484ms (24% of layer — the bottleneck)
- ANE out proj + residual: 0.372ms
- ANE FFN + residual: 0.404ms

**Key insight**: CPU attention is now the bottleneck (24% of layer time).
Further speedup requires either ANE attention (needs transpose/matmul which fail)
 or multi-head parallel attention on CPU.

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
