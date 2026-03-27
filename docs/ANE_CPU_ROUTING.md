# ANE vs CPU Operation Routing Guide

## Overview

This document describes which operations should run on the ANE (Apple Neural Engine) versus CPU, based on hardware capabilities and limitations discovered through extensive testing.

## Hardware Characteristics

### ANE (Apple Neural Engine)

| Property | Value |
|----------|-------|
| **FP16 Peak** | 18.6 TOPS (M4) |
| **INT8 Peak** | 35.1 TOPS (M4) |
| **Memory** | Shared with CPU (IOSurface) |
| **Compile Limit** | ~119 compilations per process |
| **Best For** | Large matrix operations, convolutions |

### CPU

| Property | Value |
|----------|-------|
| **FP32 Native** | Yes (no conversion overhead) |
| **Flexible Ops** | All operations supported |
| **Best For** | Control flow, small ops, unsupported operations |

## Operation Routing Matrix

### GREEN - ANE Native (Use ANE)

| Operation | MIL Op | Size Constraints | Notes |
|-----------|--------|------------------|-------|
| **MatMul** | `mb.matmul` | Min: 32x32, Max: 16384 elems | 7-10x speedup, primary ANE workload |
| **Conv1x1** | `nn.convolution` | Same as MatMul | Linear layers, excellent performance |
| **RMSNorm** | `mb.rms_norm` | Any valid size | Native ANE support |
| **ReLU** | `mb.relu` | Any valid size | Element-wise activation |
| **Sigmoid** | `mb.sigmoid` | Any valid size | Element-wise activation |
| **Tanh** | `mb.tanh` | Any valid size | Element-wise activation |
| **Softmax** | `mb.softmax` | Any valid size | Specify axis attribute |
| **Add** | `mb.add` | Min: 1024 elems | Element-wise addition |
| **Mul** | `mb.mul` | Min: 1024 elems | Element-wise multiplication |
| **Sub** | `mb.sub` | Min: 1024 elems | Element-wise subtraction |
| **Reshape** | `mb.reshape` | Any | Tensor reshaping |
| **Transpose** | `mb.transpose` | Any | Dimension reordering |
| **Slice** | `mb.slice_by_index` | Any | For RoPE, attention |
| **Concat** | `mb.concat` | Any | Tensor concatenation |
| **ReduceSum** | `mb.reduce_sum` | Any | Sum reduction |
| **ReduceMax** | `mb.reduce_max` | Any | Max reduction |
| **Pow** | `mb.pow` | Any | Power operation |
| **Sqrt** | `mb.sqrt` | Any | Square root |
| **Rsqrt** | `mb.rsqrt` | Any | Reciprocal square root |
| **Exp** | `mb.exp` | Any | Exponential |

### RED - CPU Fallback Required (Never ANE)

| Operation | Reason | Alternative |
|-----------|--------|-------------|
| **reduce_mean** | Not supported by ANE | `reduce_sum` + multiply by reciprocal |
| **LayerNorm** | Not supported | Use RMSNorm or CPU |
| **GELU** | Not in ANE op set | Approximate with tanh or CPU |
| **Embedding Lookup** | Complex indexing | CPU table lookup |
| **Multi-input programs** | ANE only accepts single input | Split into multiple kernels |
| **Backward pass (full)** | Requires multiple activations | CPU with cblas/accelerate |
| **Gradient computation** | No backprop support | CPU fallback |
| **Dropout** | Stochastic operations | CPU during training |
| **Adam/Lion optimizer** | Scalar ops, momentum | CPU (fast enough) |
| **Control flow** | Static compute graph only | CPU |

### YELLOW - Size Dependent (Use CPU if too small)

| Operation | Min Size for ANE | Recommendation |
|-----------|------------------|----------------|
| **MatMul** | 32x32 (1,024 elems) | Use CPU for < 768 dims |
| **Conv1x1** | 32x32 | Use CPU for small filters |
| **Element-wise** | 1,024 elements | CPU faster for small tensors |
| **ReduceSum** | 1,024 elements | CPU for small reductions |

## Size Constraints (Hard Limits)

| Constraint | Value | Behavior |
|------------|-------|----------|
| **Minimum** | 1,024 elements (32x32) | Smaller fails with inference error |
| **Maximum** | 16,384 elements | Larger requires tiling |
| **Alignment** | 16-byte | Memory alignment requirement |
| **Layout** | `[N, C, H, W]` or `[1, C, 1, S]` | Channel-first required |

## Recommended Routing Strategy

```rust
pub enum ExecutionTarget {
    /// Run on ANE hardware
    Ane,
    /// Run on CPU
    Cpu,
    /// ANE preferred, CPU fallback if too small
    AneOrCpu { min_elements: usize },
}

impl ExecutionTarget {
    pub fn for_operation(op: &Operation, tensor_size: usize) -> Self {
        match op {
            // Always CPU
            Op::ReduceMean | Op::LayerNorm | Op::Gelu | Op::Embedding => {
                ExecutionTarget::Cpu
            }

            // Always ANE (if size permits)
            Op::MatMul | Op::Conv1x1 => {
                if tensor_size >= 1024 {
                    ExecutionTarget::Ane
                } else {
                    ExecutionTarget::Cpu
                }
            }

            // ANE for large tensors
            Op::Add | Op::Mul | Op::Sub | Op::ReduceSum => {
                ExecutionTarget::AneOrCpu { min_elements: 1024 }
            }

            // Native ANE ops
            Op::RMSNorm | Op::ReLU | Op::Sigmoid | Op::Tanh => {
                ExecutionTarget::Ane
            }
        }
    }
}
```

## Transformer Layer Routing Example

For a standard transformer layer:

```
┌─────────────────────────────────────────────────────────┐
│ INPUT EMBEDDING          → CPU (table lookup)           │
├─────────────────────────────────────────────────────────┤
│ FOR EACH LAYER:                                        │
│   RMSNorm (input)        → ANE ✅                       │
│   QKV Projection         → ANE ✅ (MatMul)              │
│   RoPE                   → ANE ✅ (Slice/Mul/Add)       │
│   Attention Scores       → CPU ❌ (multi-input)         │
│   Attention Output       → ANE ✅ (MatMul)              │
│   Output Projection      → ANE ✅ (MatMul)              │
│   RMSNorm (ffn)          → ANE ✅                       │
│   FFN Up                 → ANE ✅ (MatMul)              │
│   SiLU Activation        → CPU ❌ (not in ANE)          │
│   FFN Down               → ANE ✅ (MatMul)              │
├─────────────────────────────────────────────────────────┤
│ BACKWARD PASS (ALL CPU):                               │
│   All gradients          → CPU ❌ (multi-input req)     │
│   Optimizer step         → CPU ❌ (scalar ops)          │
└─────────────────────────────────────────────────────────┘
```

## Compile Budget Management

The ANE has a hard limit of ~119 compilations per process:

```rust
// At startup: pre-compile all kernels
let kernels = vec![
    KernelTemplate::RmsNorm { channels: 768 },
    KernelTemplate::MatMul { m: 768, k: 768, n: 768 },
    // ... all kernels for your model
];

let registry = KernelRegistry::new(kernels, budget=10)?;

// During training: reuse kernels (NO compilation)
for step in 0..max_steps {
    if registry.should_restart() {
        checkpoint_and_exit();  // Fresh budget on restart
    }
    train_step(&registry);  // Uses pre-compiled kernels
}
```

## MIL Syntax Requirements

ANE requires specific MIL format:

```mil
// CORRECT - ANE compatible
func main<ios18>(tensor<fp16, [1, 768, 1, 256]> x) {
    tensor<fp16, [1,768,1,256]> out = mul(x=x, y=w)[name=string("out")];
} -> (out);

// WRONG - Will fail
main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
    let y = x + x;
    return (y);
}
```

**Required format:**
1. `func main<ios18>` declaration
2. `tensor<dtype, [N,C,H,W]>` tensor types
3. `[name=string("...")]` named operations
4. `-> (output_names)` return syntax
5. Weights as `BLOBFILE` constants

## Performance Guidelines

### When ANE Wins (7-10x speedup)
- Large MatMul (768x768+)
- Conv1x1 layers
- RMSNorm on large tensors
- Batched element-wise ops

### When CPU Wins
- Small matrices (< 768 dims)
- Single element-wise operations
- Complex indexing/gather
- Control flow

### Break-even Point
- ~1,024 elements (32x32 matrix)
- Below this: CPU faster
- Above this: ANE faster

## Fallback Implementation Pattern

```rust
pub fn reduce_mean_cpu(x: &[f32], axis: usize, keep_dims: bool) -> Vec<f32> {
    // Use Accelerate framework for best CPU performance
    let sum = x.iter().sum::<f32>();
    let count = x.len() as f32;
    vec![sum / count]
}

pub fn route_reduce_mean(
    x: &[f32],
    axis: usize,
    keep_dims: bool
) -> Result<Vec<f32>> {
    // ANE doesn't support reduce_mean - always use CPU
    Ok(reduce_mean_cpu(x, axis, keep_dims))
}
```

## References

- `docs/ANE_TRAINING_ARCHITECTURE.md` - Compile budget management
- `docs/ANE_BACKWARD_LIMITATION.md` - Why backward pass requires CPU
- `docs/ANE_MIL_SYNTAX.md` - Working MIL format
- `docs/ANE_FAILURE_REPORT.md` - Documented failures
- `docs/ANE_IOSURFACE_BEST_PRACTICES.md` - Memory management
