# ANE Training Architecture

## The ~119 Compile Limit Problem

ANE has a critical limitation: **~119 kernel compilations per process** before memory corruption occurs. This is due to resource leaks in Apple's private ANE compiler APIs.

### Solution: Compile Budget Management

1. **Pre-compile all kernels at startup**
2. **Reuse kernels throughout training** (dynamic weight packing)
3. **Monitor budget consumption**
4. **Checkpoint + restart when budget exhausted**

## Architecture Overview

```
Training Process Lifecycle:
┌─────────────────────────────────────────────────────────────┐
│ 1. STARTUP                                                  │
│    - Define all KernelTemplates needed for model            │
│    - Pre-compile all kernels (respect 119 limit)            │
│    - Store in KernelRegistry                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAINING LOOP (NO compilation here!)                     │
│    - Load batch                                             │
│    - ANE forward (pre-compiled kernels)                     │
│    - ANE backward dx (pre-compiled kernels)                 │
│    - CPU backward dW (cblas)                                │
│    - CPU Adam optimizer                                     │
│    - Update weights (dynamic packing)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CHECKPOINT (when budget low)                             │
│    - Save model weights                                     │
│    - Save optimizer state                                   │
│    - Save training step                                     │
│    - EXIT PROCESS                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. RESTART                                                  │
│    - Load checkpoint                                        │
│    - Re-initialize (fresh compile budget!)                  │
│    - Continue training                                      │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. CompileBudget

Tracks ANE compilation usage to prevent exceeding the ~119 limit.

```rust
let budget = CompileBudget::new(10); // Reserve 10 for emergencies

// Request compilation of 50 kernels
if budget.request_compile(50) {
    // Safe to compile
} else {
    // Would exceed budget - need restart
}
```

### 2. KernelTemplate

Defines kernel types without compiling:

```rust
enum KernelTemplate {
    RmsNorm { channels: usize, seq_len: usize },
    DynamicLinear { in_features: usize, out_features: usize, seq_len: usize },
    // ... etc
}
```

### 3. KernelRegistry

Pre-compiles all kernels at initialization:

```rust
let kernels = vec![
    KernelTemplate::RmsNorm { channels: 768, seq_len: 256 },
    KernelTemplate::DynamicLinear { in_features: 768, out_features: 768, seq_len: 256 },
    // ... all kernels for your model
];

let registry = KernelRegistry::new(kernels, 10)?;
// All kernels now compiled and ready to use
```

### 4. Dynamic Weight Packing

**Problem**: If weights are compiled as constants, you need to recompile for every weight update.

**Solution**: Pack weights into the spatial dimension of the input tensor:

```
Input tensor: [1, C, 1, SEQ + WEIGHT_SIZE]
  [0:SEQ]           = activations
  [SEQ:SEQ+W]       = weights
```

MIL extracts weights at runtime using `slice_by_size`:

```mil
tensor<int32, [4]> begin = const()[val=tensor<int32, [4]>([0,0,0,SEQ])];
tensor<int32, [4]> size = const()[val=tensor<int32, [4]>([1,C,1,WEIGHT_SIZE])];
tensor<fp16, [1,C,1,W]> weights = slice_by_size(x=input, begin=begin, size=size);
```

**Benefits**:
- No recompilation when weights change
- Same kernel works for all steps
- Avoids the 119 compile limit

## Usage Example

```rust
use rustane::ane::{ANETrainingConfig, KernelRegistry, TrainingCheckpoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure training
    let config = ANETrainingConfig {
        dim: 768,
        n_layers: 12,
        seq_len: 256,
        ..Default::default()
    };
    
    // 2. Generate all kernels needed
    let kernels = config.generate_kernels();
    println!("Need {} kernels", kernels.len());
    
    // 3. Pre-compile all kernels
    let registry = KernelRegistry::new(kernels, 10)?;
    println!("Compiled {} kernels", registry.available_count());
    
    // 4. Training loop (NO compilation!)
    let mut checkpoint = TrainingCheckpoint::new(1000);
    
    loop {
        // Check budget
        if registry.should_restart() {
            println!("Budget low! Checkpoint and restart.");
            save_checkpoint()?;
            break; // Exit process
        }
        
        // Training step uses pre-compiled kernels
        train_step_ane(&registry,&config)?;
        
        if checkpoint.step() {
            save_checkpoint()?;
        }
    }
    
    Ok(())
}
```

## Performance Characteristics

From maderix/ANE project:

| Model | Params | Time/Step | Throughput |
|-------|--------|-----------|------------|
| Stories110M | 109M | 91ms | ~11 steps/sec |
| Qwen3-0.6B | 596M | 412ms | ~2.4 steps/sec |

**ANE Peak Performance** (M4):
- FP16: 18.6 TOPS
- INT8: 35.1 TOPS

**Compute Distribution**:
- ANE: Forward + backward dx (70% of compute)
- CPU: Backward dW + Adam (30% of compute)

## Size Constraints

ANE has strict size limits:

| Metric | Value | Notes |
|--------|-------|-------|
| Minimum | 32x32 (1,024 elements) | Smaller fails with inference error |
| Maximum | 16,384 elements | Larger requires tiling |
| Alignment | 16-byte | Memory alignment requirement |
| Layout | NCHW [1,C,1,S] | Channel-first for 1D sequences |

**Tiling Strategy** for large tensors:
1. Split along channel or spatial dimension
2. Process each tile separately
3. Concatenate results

## Checkpoint/Restart Workflow

When compile budget runs low:

```bash
# Training process detects low budget
# 1. Saves checkpoint
# 2. Exits with special code

# Wrapper script detects exit code
# 3. Restarts process
# 4. Loads checkpoint
# 5. Continues training with fresh budget
```

Example wrapper:

```bash
#!/bin/bash
while true; do
    cargo run --example train_ane
    
    if [ $? -eq 42 ]; then  # Special code for "restart needed"
        echo "Restarting with fresh compile budget..."
        continue
    fi
    
    break  # Normal exit
done
```

## Best Practices

1. **Plan kernels upfront**: Know exactly how many kernels you need
2. **Test ANE compatibility**: Ensure all kernels fit within size limits
3. **Use dynamic weights**: Never compile weights as constants
4. **Monitor budget**: Check `registry.remaining_budget()` periodically
5. **Checkpoint frequently**: Save progress before budget exhaustion
6. **Implement restart logic**: Your training script should handle restarts

## Limitations

1. **~119 compile limit**: Hard constraint, requires restart
2. **Size limits**: 16,384 elements max per tensor
3. **Single-input**: Multi-input kernels cause 0x1d errors
4. **SDPA masking**: Causal attention needs CPU fallback
5. **FP16 underflow**: Backward pass may underflow, needs loss scaling

## Future Improvements

1. **Kernel caching to disk**: Save compiled kernels between restarts
2. **INT8 quantization**: 1.88x speedup (35 TOPS vs 18 TOPS)
3. **GQA support**: Grouped Query Attention for larger models
4. **Pipeline parallelism**: Overlap ANE and CPU compute
5. **Multi-device**: Use multiple ANE cores if available

## References

- maderix/ANE: https://github.com/maderix/ANE
- Parameter-golf dataset: ~/dev/parameter-golf
- ANE MIL syntax: docs/ANE_MIL_SYNTAX.md
