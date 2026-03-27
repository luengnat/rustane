# ANE Compile Limit Mitigation Strategies

## The ~119 Compile Limit Problem

The ANE compiler has a memory leak that causes failures after approximately 119 compilations in a single process. This is a hardware/firmware limitation that affects long-running training jobs.

## Mitigation Strategies

### Strategy 1: Pre-compile All Kernels (Recommended)

Pre-compile all needed MIL programs at startup, then reuse handles:

```rust
use rustane::ane::{CompileBudget, ANERuntime};

// Initialize compile budget monitor
let budget = CompileBudget::new(110); // Reserve 9 for safety

// Pre-compile all layer kernels at startup
let mut kernel_handles = Vec::new();
for layer_id in 0..9 {
    let fwd_mil = generate_forward_mil(layer_id);
    let bwd_mil = generate_backward_mil(layer_id);
    
    let fwd_handle = runtime.compile(&fwd_mil)?;
    let bwd_handle = runtime.compile(&bwd_mil)?;
    
    kernel_handles.push((fwd_handle, bwd_handle));
    
    // Check budget
    if budget.remaining() < 10 {
        panic!("Compile budget exhausted during startup");
    }
}

// During training, just reuse handles
for step in 0..iterations {
    for layer_id in 0..9 {
        let (fwd_handle, bwd_handle) = &kernel_handles[layer_id];
        runtime.execute(fwd_handle, inputs)?;
        runtime.execute(bwd_handle, grads)?;
    }
}
```

**Pros**: Simple, no process restart needed
**Cons**: Must know all kernels upfront; limited to ~55 layer pairs

### Strategy 2: Checkpoint/Restart (For Large Models)

When budget is exhausted, save state and restart process:

```rust
use rustane::ane::{TrainingCheckpoint, CompileBudget};

struct TrainingState {
    step: usize,
    model_weights: Vec<Tensor>,
    optimizer_state: OptimizerState,
}

fn train_with_restart() {
    let budget = CompileBudget::new(110);
    let mut state = load_initial_state();
    
    loop {
        // Train until budget warning
        while budget.remaining() > 10 {
            train_step(&mut state);
            state.step += 1;
        }
        
        // Save checkpoint
        let checkpoint = TrainingCheckpoint {
            step: state.step,
            weights: state.model_weights.clone(),
            optimizer: state.optimizer_state.clone(),
        };
        checkpoint.save("checkpoint.bin");
        
        // Exit - external script will restart
        println!("CHECKPOINT_REACHED: step={}", state.step);
        std::process::exit(0);
    }
}
```

With wrapper script:

```bash
#!/bin/bash
# train_wrapper.sh

while true; do
    ./train_ane
    
    if [ $? -eq 0 ]; then
        echo "Training completed successfully"
        break
    fi
    
    # Check if checkpoint was reached
    if grep -q "CHECKPOINT_REACHED" training.log; then
        echo "Restarting from checkpoint..."
        continue
    else
        echo "Training failed with error"
        break
    fi
done
```

**Pros**: Can train arbitrarily large models
**Cons**: Requires external orchestration; restart overhead (~1-2s)

### Strategy 3: Kernel Registry with Dynamic Loading

Use a persistent compilation service:

```rust
use rustane::ane::{KernelRegistry, KernelTemplate};

// Initialize registry
let registry = KernelRegistry::new();

// Register kernel templates (not compiled yet)
for layer_id in 0..9 {
    registry.register(KernelTemplate {
        name: format!("layer_{}_fwd", layer_id),
        mil_generator: || generate_forward_mil(layer_id),
    });
}

// Registry compiles on-demand and caches
let kernel = registry.get("layer_0_fwd")?;
```

**Pros**: Lazy compilation, automatic caching
**Cons**: Still limited by total unique kernels

### Strategy 4: Weight Swapping (Orion T084)

Use single compiled kernel, swap weights via IOSurface:

```rust
use rustane::ane::{ANEProgramCache, WeightsIdGenerator};

// Compile once
let cache = ANEProgramCache::new();
let program = cache.compile_program(base_mil)?;

// Create weight surfaces for all layers
let weight_generator = WeightsIdGenerator::new();
let mut weight_surfaces = Vec::new();

for layer_id in 0..9 {
    let weights = load_layer_weights(layer_id);
    let surface = weight_generator.create_surface(&weights)?;
    weight_surfaces.push(surface);
}

// Execute with different weights
for layer_id in 0..9 {
    let weights = &weight_surfaces[layer_id];
    cache.execute_with_weights(program, weights, inputs)?;
}
```

**Pros**: Single compilation for all layers
**Cons**: Requires careful IOSurface management; weight format must match exactly

### Strategy 5: Hybrid ANE+CPU (Current rustane Approach)

Use ANE for compute-heavy ops, CPU for parameter updates:

```rust
// ANE: Forward and backward passes
let activations = ane.forward(&inputs)?;
let gradients = ane.backward(&activations, &deltas)?;

// CPU: Optimizer step (Adam/Muon)
for (param, grad) in params.iter_mut().zip(gradients) {
    adam.update(param, grad)?;
}
```

**Pros**: Avoids optimizer compilation entirely
**Cons**: Data transfer overhead ANE↔CPU

## Recommended Approach for Parameter-Golf

Given the model architecture (9 layers, 1024 seq, 512 dim):

**Phase 1**: Pre-compile all kernels (18 total: 9 fwd + 9 bwd)
- 18 compiles << 119 limit ✓
- No restart needed

**Phase 2**: If adding more layers
- Use weight swapping for layers beyond 9
- Or implement checkpoint/restart

**Phase 3**: For production training
- Hybrid approach with weight swapping
- Compile budget reserved for specialized kernels (attention variants, etc.)

## Monitoring Compile Budget

```rust
use rustane::ane::CompileBudget;

let budget = CompileBudget::new(110);

// Log remaining budget periodically
if step % 100 == 0 {
    println!("Compile budget: {}/{} remaining", 
             budget.remaining(), 
             budget.total());
}

// Alert when low
if budget.remaining() < 20 {
    eprintln!("WARNING: Compile budget running low!");
}
```

## Compilation Caching

ANE automatically caches E5 binaries:

```
~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/
  └── <build>/<hash>/
      └── H16G.bundle/           # H16G = M4 ANE
          ├── H16G.e5            # Compiled binary
          └── main/main_ane/
              └── model.anehash
```

First compile: ~20-40ms  
Cache hit: effectively free

## Future: Avoiding Compilation Entirely

The ultimate solution is extracting HWX files from CoreML (Option 4):

1. Convert PyTorch model to CoreML
2. Extract pre-compiled HWX files
3. Load HWX directly with `ANEKernel::from_hwx()`

This bypasses the MIL compiler entirely - no ~119 limit!

See: `docs/HWX_INTEGRATION.md`
