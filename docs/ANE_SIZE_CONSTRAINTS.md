# ANE Size Constraints and Automatic Routing

## Overview

The Apple Neural Engine (ANE) has specific size requirements for efficient operation. This document describes the size constraints and the automatic routing mechanism that directs operations to ANE or CPU based on tensor size and operation type.

## Size Constraints

### Hard Limits

| Constraint | Value | Behavior |
|------------|-------|----------|
| **Minimum** | 1,024 elements (32×32) | Smaller tensors fail with inference error |
| **Maximum** | 16,384 elements per tensor | Larger tensors require tiling |
| **Alignment** | 16-byte | Memory alignment requirement |

### Break-even Point

- **~1,024 elements**: Below this, CPU is faster; above this, ANE is faster
- This is the default threshold used by `ExecutionTarget::for_size()`

## Automatic Routing

### ExecutionTarget Enum

```rust
pub enum ExecutionTarget {
    /// Execute on ANE hardware
    Ane,
    /// Execute on CPU
    Cpu,
}

impl ExecutionTarget {
    pub fn for_size(op_name: &str, num_elements: usize) -> Self {
        match op_name {
            // Operations ANE never supports
            "reduce_mean" | "layer_norm" | "gelu" | "silu" | "embedding" => {
                ExecutionTarget::Cpu
            }
            _ => {
                // Size-based routing for other operations
                if num_elements >= ANE_MIN_ELEMENTS {
                    ExecutionTarget::Ane
                } else {
                    ExecutionTarget::Cpu
                }
            }
        }
    }
}
```

### Routing Logic

| Operation Type | Size Requirement | Target |
|----------------|------------------|--------|
| `reduce_mean`, `layer_norm`, `gelu`, `silu`, `embedding` | Any | **CPU** (unsupported on ANE) |
| `matmul`, `conv1x1`, element-wise ops | ≥ 1,024 elements | **ANE** |
| `matmul`, `conv1x1`, element-wise ops | < 1,024 elements | **CPU** |
| `rms_norm`, `relu`, `sigmoid`, `tanh` | Any valid size | **ANE** (native support) |

## MILBuilder Size Validation

The `MILBuilder` provides helper methods for size validation:

### check_size()

Check if a tensor meets minimum ANE size requirements:

```rust
let builder = MILBuilder::new();

// Returns false - 256 elements is too small
assert!(!builder.check_size(&[1, 1, 16, 16]));

// Returns true - 196,608 elements is good for ANE
assert!(builder.check_size(&[1, 768, 1, 256]));
```

### validate_opSize()

Get detailed validation result with recommendation:

```rust
let builder = MILBuilder::new();

// Check matmul operation
let result = builder.validate_op_size("matmul", &[1, 768, 1, 256]);
assert!(result.should_use_ane);
assert!(result.recommendation.contains("ANE acceleration"));

// Check unsupported operation
let result = builder.validate_op_size("reduce_mean", &[1, 768, 1, 256]);
assert!(!result.should_use_ane);
assert!(result.recommendation.contains("CPU fallback"));
```

### SizeValidationResult

```rust
pub struct SizeValidationResult {
    /// true if ANE should be used
    pub should_use_ane: bool,
    /// Total number of elements
    pub num_elements: usize,
    /// Human-readable recommendation
    pub recommendation: &'static str,
}
```

## CPU Fallback Implementations

The following CPU implementations are provided for operations that ANE doesn't support or where CPU is more efficient:

| Function | Purpose |
|----------|---------|
| `reduce_mean_cpu()` | Mean reduction (ANE doesn't support) |
| `layer_norm_cpu()` | Full LayerNorm (ANE only has RMSNorm) |
| `gelu_cpu()` | GELU activation (not in ANE op set) |
| `silu_cpu()` | SiLU/Swish activation (not in ANE op set) |
| `embedding_lookup_cpu()` | Embedding table lookup (complex indexing) |
| `rms_norm_cpu()` | RMSNorm reference/slower than ANE |
| `rope_cpu()` | RoPE for small tensors |

## Usage Pattern

```rust
use rustane::mil::{MILBuilder, ExecutionTarget};

// Determine execution target
let num_elements = 768 * 256; // 196,608
let target = ExecutionTarget::for_size("matmul", num_elements);

match target {
    ExecutionTarget::Ane => {
        // Build and execute ANE kernel
        let mil = MILBuilder::new()
            .add_input("x", "fp16", &[1, 768, 1, 256])
            .add_matmul("out", "x", "weight", false)
            .build();
        // ... execute on ANE
    }
    ExecutionTarget::Cpu => {
        // Use CPU fallback
        // ... execute with Accelerate framework
    }
}
```

## Performance Guidelines

### When ANE Wins (7-10x speedup)

- Large MatMul (768×768+)
- Conv1x1 layers
- RMSNorm on large tensors
- Batched element-wise ops

### When CPU Wins

- Small matrices (< 768 dims, < 1,024 elements)
- Single element-wise operations
- Complex indexing/gather
- Control flow

## Examples

### Example 1: Size Check Before ANE Execution

```rust
let builder = MILBuilder::new();
let shape = [1, 512, 1, 128]; // 65,536 elements

if builder.check_size(&shape) {
    // Safe to use ANE
    let mil = builder
        .add_input("x", "fp16", &shape)
        .add_matmul("out", "x", "weight", false)
        .build();
} else {
    // Use CPU fallback
}
```

### Example 2: Operation-Specific Routing

```rust
let builder = MILBuilder::new();

// GELU is never supported on ANE
let result = builder.validate_op_size("gelu", &[1, 4096, 1, 1]);
assert!(!result.should_use_ane); // Always CPU

// MatMul depends on size
let result = builder.validate_op_size("matmul", &[1, 16, 1, 16]);
assert!(!result.should_use_ane); // Too small

let result = builder.validate_op_size("matmul", &[1, 768, 1, 768]);
assert!(result.should_use_ane); // Good for ANE
```

## Related Documentation

- `docs/ANE_CPU_ROUTING.md` - Complete operation routing matrix
- `docs/ANE_MIL_SYNTAX.md` - MIL format requirements
- `docs/ANE_TRAINING_ARCHITECTURE.md` - Compile budget management
