# ANE Training System - Complete Documentation

## Overview

A production-ready ANE training system with automatic tiling, compile budget management, and CPU fallback.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Process                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. CONFIGURATION                                                 │
│    - Define model dimensions (dim, layers, heads, seq_len)      │
│    - Check size constraints (≤ 16,384 elements per tensor)      │
│    - Generate kernel templates                                   │
├─────────────────────────────────────────────────────────────────┤
│ 2. TILING (Automatic)                                            │
│    - Split large operations into ANE-compatible chunks          │
│    - Strategy: spatial tiling first, then channel tiling        │
│    - Each tile: ≤ 16,384 elements                               │
├─────────────────────────────────────────────────────────────────┤
│ 3. COMPILE BUDGET CHECK                                          │
│    - ANE limit: ~119 compiles per process                       │
│    - Pre-compile all kernels at startup                         │
│    - Reserve budget for emergencies                             │
├─────────────────────────────────────────────────────────────────┤
│ 4. PRE-COMPILATION                                               │
│    - Compile all tiled kernels                                  │
│    - Store in KernelRegistry                                    │
│    - CPU fallback for incompatible kernels                      │
├─────────────────────────────────────────────────────────────────┤
│ 5. TRAINING LOOP (NO compilation!)                              │
│    ANE Operations:                                               │
│      - Forward pass (pre-compiled kernels)                      │
│      - Backward dx (pre-compiled kernels)                       │
│    CPU Operations:                                               │
│      - Backward dW (cblas_sgemm)                                │
│      - Adam optimizer                                           │
│      - Loss computation                                         │
├─────────────────────────────────────────────────────────────────┤
│ 6. CHECKPOINT & RESTART (when budget low)                       │
│    - Save weights and optimizer state                           │
│    - Exit process                                               │
│    - Restart with fresh compile budget                          │
└─────────────────────────────────────────────────────────────────┘
```

## Size Constraints

ANE hardware limitations:

| Constraint | Value | Notes |
|------------|-------|-------|
| **Min elements** | 1,024 (32×32) | Smaller fails with inference error |
| **Max elements** | 16,384 | Hard limit per tensor |
| **Alignment** | 16-byte | Memory alignment requirement |
| **Data layout** | NCHW [1,C,1,S] | Channel-first for 1D sequences |
| **Data type** | FP16 | 2 bytes per element |

## Tiling Strategy

When operations exceed 16,384 elements:

```rust
// Example: 768×256 = 196,608 elements (needs tiling)
// Strategy: Split along spatial dimension first

// Calculate tiles needed:
max_elements_per_tile = 16,384
channels = 768
spatial = 256

// Spatial tiling:
elements_per_spatial_slice = channels = 768
max_spatial_per_tile = 16,384 / 768 ≈ 21

// Need: ceil(256 / 21) = 13 spatial tiles
// Each tile: 768 × 20 = 15,360 elements ✓
```

### Tiling Algorithm

1. **Check if tiling needed**: `total_elements > 16,384?`
2. **Try spatial tiling first** (usually seq_len):
   - Calculate max spatial per tile: `16,384 / channels`
   - If ≥ 1: use spatial tiling only
3. **Fallback to channel tiling** if spatial insufficient:
   - Split channels across tiles
   - Each tile gets subset of channels
4. **Generate tile kernels** for each chunk

### Tile Kernel Generation

Each tile is a separate kernel with:
- Fixed channel/spatial dimensions
- Slice offsets for input extraction
- Accumulation for output concatenation

Example MIL for tiled RMSNorm:
```mil
// Tile 1: channels 0-768, spatial 0-20
func main<ios18>(tensor<fp16, [1, 768, 1, 20]> x) {
  // RMSNorm computation on tile
  tensor<fp16, [1,768,1,20]> out = ...;
} -> (out);

// Tile 2: channels 0-768, spatial 20-40
// ... etc
```

## Dynamic Weight Packing

**Problem**: ANE compiles weights as constants. Weight updates require recompilation.

**Solution**: Pack weights into input spatial dimension:

```
Input: [1, C, 1, SEQ_LEN + WEIGHT_SIZE]
  [0:SEQ_LEN]                = activations
  [SEQ_LEN:SEQ_LEN+WEIGHT_SIZE] = weights
```

MIL extracts weights at runtime:
```mil
// Slice activations
tensor<int32, [4]> b_a = const()[val=[0,0,0,0]];
tensor<int32, [4]> s_a = const()[val=[1,C,1,SEQ]];
tensor<fp16, [1,C,1,SEQ]> act = slice_by_size(x=input, begin=b_a, size=s_a);

// Slice weights
tensor<int32, [4]> b_w = const()[val=[0,0,0,SEQ]];
tensor<int32, [4]> s_w = const()[val=[1,C,1,W_SIZE]];
tensor<fp16, [1,C,1,W_SIZE]> W = slice_by_size(x=input, begin=b_w, size=s_w);
```

**Benefits**:
- No recompilation for weight updates
- Same kernel works for all steps
- Avoids the 119 compile limit

## Compile Budget Management

ANE has a ~119 compile limit per process due to memory leaks in private APIs.

### Budget Allocation

```rust
const ANE_COMPILE_BUDGET: i32 = 110;  // Leave margin from 119
const ANE_RESERVE_BUDGET: i32 = 10;   // For emergencies

// Available for training kernels: 100 compiles
```

### Budget Tracking

```rust
let budget = CompileBudget::new(10);  // Reserve 10

// Request compilation of 50 kernels
if budget.request_compile(50) {
    // Safe to compile
} else {
    // Would exceed budget
    // TRIGGER CHECKPOINT + RESTART
}
```

### Checkpoint/Restart Workflow

```bash
# training_wrapper.sh
while true; do
    cargo run --example train_ane
    
    if [ $? -eq 42 ]; then  # Special "restart needed" code
        echo "Compile budget exhausted, restarting..."
        continue
    fi
    
    break  # Normal exit
done
```

## Performance Characteristics

Based on maderix/ANE and our implementation:

### Throughput

| Model | Params | Time/Step | Steps/sec | Platform |
|-------|--------|-----------|-----------|----------|
| Stories110M | 109M | 91ms | 11 | M4 |
| Our 256×64 | ~110M | ~100ms | 10 | M4 (estimated) |
| Qwen3-0.6B | 596M | 412ms | 2.4 | M4 |

### ANE Peak Performance (M4)

| Precision | Peak TOPS | Configuration |
|-----------|-----------|---------------|
| FP16 | 18.6 | 128×conv 512ch 64×64 |
| INT8 | 35.1 | 128×conv 512ch 64×64 |

**INT8 provides 1.88x speedup** over FP16.

### Compute Distribution

For a typical training step:
- **ANE (70%)**: Forward pass, backward dx
- **CPU (30%)**: Backward dW, Adam optimizer, loss

## Usage Example

```rust
use rustane::ane::{
    ANETrainingConfig, TiledTrainingConfig, 
    KernelRegistry, CompileBudget
};

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Configure model
    let config = ANETrainingConfig {
        dim: 256,
        n_layers: 8,
        n_heads: 8,
        seq_len: 64,  // 256×64 = 16,384 (at limit)
        ..Default::default()
    };
    
    // 2. Generate tiled kernels
    let tiled = TiledTrainingConfig::from_config(config);
    println!("Total kernels: {}", tiled.total_kernel_count);
    
    // 3. Check budget
    let budget = CompileBudget::new(10);
    if !tiled.fits_budget(&budget) {
        panic!("Insufficient compile budget!");
    }
    
    // 4. Pre-compile all kernels
    let kernels = tiled.get_all_kernels();
    let registry = KernelRegistry::new(kernels, 10)?;
    
    // 5. Train (NO compilation here!)
    for step in 0..1_000_000 {
        if registry.should_restart() {
            save_checkpoint()?;
            return Ok(()); // Exit for restart
        }
        
        train_step(&registry, &config)?;
    }
    
    Ok(())
}
```

## Implementation Files

| File | Purpose |
|------|---------|
| `mil_generator.rs` | MIL program generation with correct syntax |
| `training_architecture.rs` | Compile budget, kernel registry, templates |
| `tiling.rs` | Automatic tiling for large tensors |
| `trainer.rs` | Kernel caching, execution management |
| `runtime.rs` | Low-level ANE compilation and execution |

## Key Design Decisions

1. **Pre-compilation**: All kernels compiled at startup
   - Pros: No runtime surprises, predictable performance
   - Cons: Longer startup time, requires planning

2. **Dynamic weights**: Pack weights in input tensor
   - Pros: No recompilation, efficient weight updates
   - Cons: Higher memory bandwidth (weights in every input)

3. **Hybrid execution**: ANE + CPU
   - Pros: Best of both worlds
   - Cons: Data transfer overhead between ANE and CPU

4. **Automatic tiling**: Transparent to user
   - Pros: Simple API, handles size constraints
   - Cons: More kernels to compile, potential overhead

## Limitations

1. **~119 compile limit**: Requires checkpoint/restart
2. **16,384 element limit**: Large models need extensive tiling
3. **Single-input constraint**: Multi-input kernels fail (0x1d error)
4. **SDPA masking**: Causal attention needs CPU fallback
5. **FP16 underflow**: Backward pass may underflow (needs loss scaling)

## Future Improvements

1. **INT8 quantization**: 1.88x speedup
2. **Kernel disk caching**: Save compiled kernels between restarts
3. **Pipeline parallelism**: Overlap ANE and CPU execution
4. **GQA support**: Grouped Query Attention for efficiency
5. **Multi-device**: Use multiple ANE cores

## References

- maderix/ANE: https://github.com/maderix/ANE
- Tiled Matrix Multiplication: https://penny-xu.github.io/blog/tiled-matrix-multiplication
- ANE Optimization Guidelines: docs/ANE_OPTIMIZATION_GUIDELINES.md
