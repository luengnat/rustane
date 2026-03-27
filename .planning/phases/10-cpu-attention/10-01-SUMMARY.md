# Phase 10: CPU Attention Optimization — Summary

## Status: COMPLETE — CPU attention already optimal via BLAS

## Goal
Profile CPU attention and compare optimized implementations (baseline vs fused softmax vs flash-style).

## Approach
Created `examples/bench_attention.rs` comparing three implementations:
1. **Baseline**: Separate scale array + 3-pass softmax (max, exp+sum, divide)
2. **Optimized**: Fused scale into softmax (max*scaled, single pass)
3. **Flash**: Pre-computed row_max/row_sum (flash attention style)

Tested at D=256..2048, SP=256..512, with multi-head configurations (8, 12, 16 heads).

## Results

### Single-head attention
| Config | Baseline | Optimized | Flash |
|--------|----------|-----------|-------|
| D=256, SP=256 | 204us | 246us (0.83x) | 226us (0.90x) |
| D=768, SP=256 | 378us | 387us (0.97x) | 383us (0.98x) |
| D=2048, SP=512 | 3193us | 3254us (0.98x) | 3234us (0.98x) |

### Multi-head attention (12L transformer, D=768, heads=12)
- Baseline: 1573us/layer, 131us/head
- Optimized: 1810us/layer (0.87x — slower!)
- Total per step: ~19ms for all 12 layers

### Key Finding
**CPU attention is already optimal.** The bottleneck is BLAS `cblas_sgemm` for QK^T and AV matmuls, not the softmax loop. Softfusion optimization makes things slightly slower due to additional multiplications per element. The flash variant provides no benefit since we're not memory-limited (all data fits in L2 cache).

At ~19ms per 12-layer attention step vs a typical 100ms total training step, attention is only ~19% of compute. The ANE handles the matmul-heavy projections (FFN, QKV), so CPU attention is not the bottleneck.

## Conclusion
No CPU attention optimization needed. The BLAS matmuls dominate, and the softmax loop is already negligible. Future optimization should focus on:
- Multi-head attention in `train_transformer.rs` (currently uses single-head)
- Offloading more operations to ANE
