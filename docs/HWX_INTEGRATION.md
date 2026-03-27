# HWX Integration Guide

## Overview

HWX files are pre-compiled ANE binaries extracted from CoreML models. Using HWX bypasses the MIL compiler entirely, avoiding the ~119 compile limit and reducing startup time.

## What is HWX?

HWX files are:
- **Mach-O binaries** with ANE operations at offset 0x4000
- **E5 format** (FlatBuffer-based, ~2-3KB per kernel)
- **Hardware-specific** (H16G for M4, H13G for M3, etc.)
- **Architecture-independent** within same ANE generation

## Workflow

```
PyTorch Model → CoreML Conversion → Compilation → HWX Extraction → ANE Runtime
```

## Step 1: Convert PyTorch to CoreML

Use the provided script:

```bash
python scripts/pytorch_to_coreml_hwx.py \
    --model-path ./checkpoints/model.pt \
    --output-dir ./models/coreml
```

This creates:
- `transformer.mlpackage` - CoreML model
- `compiled/*.hwx` - Extracted ANE binaries (if available)

## Step 2: Extract HWX from CoreML (Manual)

If automatic extraction fails:

### Method A: coremlcompiler (Recommended)

```bash
# Find coremlcompiler
COREML_COMPILER=$(xcrun -f coremlcompiler)

# Compile model
$COREML_COMPILER compile transformer.mlpackage ./compiled

# Find HWX files
find ./compiled -name "*.hwx"
```

### Method B: Runtime Extraction

Monitor ANE cache during model loading:

```bash
# Terminal 1: Run your model
python -c "import coremltools as ct; model = ct.models.MLModel('transformer.mlpackage'); ..."

# Terminal 2: Monitor filesystem
sudo fs_usage -w | grep -i "hwx\|ane\|espresso"
```

### Method C: Cache Directory

Check ANE compilation cache:

```bash
# ANE cache locations
~/Library/Caches/com.apple.e5rt.e5bundlecache/
~/Library/Caches/com.apple.espresso/
~/Library/Caches/ANECompiler/

# Find recent HWX files
find ~/Library/Caches -name "*.hwx" -mtime -1
```

## Step 3: Load HWX in Rustane

```rust
use rustane::ane::{HWXLoader, HWXProgram};

// Create loader
let mut loader = HWXLoader::new();
loader.add_search_path("./models/coreml/compiled");

// Load HWX program
let program: HWXProgram = loader.load("layer_0_fwd.hwx")?;

// Convert to kernel
let kernel = loader.to_kernel(&program)?;

// Execute
kernel.write_input(0, &input_data)?;
kernel.eval()?;
let output = kernel.read_output(0)?;
```

## HWX File Structure

```rust
pub struct HWXProgram {
    pub name: String,
    pub data: Vec<u8>,           // Raw Mach-O binary
    pub operations: Vec<ANEOperation>,  // Parsed ops
    pub metadata: HWXMetadata,
}

pub struct ANEOperation {
    pub index: usize,
    pub data: Vec<u8>,           // 0x300 bytes per op
    pub op_type: String,         // "conv", "relu", "matmul", etc.
}
```

## Advanced: Modifying HWX at Runtime

Like tinygrad, you can modify HWX binaries to change parameters:

```rust
use rustane::ane::hwx_loader::{HWXLoader, ANEOperation};

// Load and modify
let mut program = loader.load("matmul.hwx")?;

// Modify operation parameters
for op in &mut program.operations {
    if op.op_type == "conv" {
        // Modify kernel size at offset 0x144
        op.data[0x144] = new_kernel_size as u8;
    }
}

// Re-compile (not needed - HWX is pre-compiled!)
// Just use directly
```

## Integration with Training

### Option A: Full HWX (No MIL Compilation)

```rust
use rustane::ane::{HWXLoader, ANETrainingRuntime};

// Load all layer HWX files
let loader = HWXLoader::new();
let layers: Vec<HWXProgram> = loader.load_directory("./models/hwx")?;

// Training loop - no compilation needed
for step in 0..iterations {
    for (layer_id, layer) in layers.iter().enumerate() {
        let kernel = loader.to_kernel(layer)?;
        
        // Forward
        let activations = kernel.execute(&inputs)?;
        
        // Backward (separate HWX file)
        let bwd_kernel = loader.to_kernel(&bwd_layers[layer_id])?;
        let grads = bwd_kernel.execute(&activations)?;
    }
}
```

### Option B: Hybrid (MIL + HWX)

Use HWX for static kernels, MIL for dynamic:

```rust
// Static kernels from HWX (no compile limit)
let attn_kernel = loader.load("attention.hwx")?;

// Dynamic kernels compiled from MIL
let custom_op = runtime.compile(custom_mil)?;

// Use both
let attn_out = attn_kernel.execute(input)?;
let custom_out = custom_op.execute(attn_out)?;
```

## Performance Comparison

| Approach | Startup Time | Runtime Overhead | Compile Limit |
|----------|-------------|------------------|---------------|
| MIL Compilation | 20-40ms/kernel | None | ~119 |
| HWX Loading | 1-2ms/kernel | None | Unlimited |
| CoreML Direct | 100-200ms | High | N/A |

## Troubleshooting

### HWX Not Found

```rust
// Check search paths
let loader = HWXLoader::new();
println!("Search paths: {:?}", loader.search_paths);

// Verify file exists
assert!(std::path::Path::new("models/layer.hwx").exists());
```

### Invalid HWX Format

```rust
// Enable debug output
export RUST_LOG=debug

// Check Mach-O magic
let data = std::fs::read("layer.hwx")?;
assert_eq!(&data[0..4], b"\xcf\xfa\xed\xfe");  // MH_CIGAM_64
```

### Architecture Mismatch

HWX files are hardware-specific:
- H16G = M4 (16-core ANE)
- H13G = M3 (varies)
- H11G = M1/M2

```rust
// Check metadata
let program = loader.load("layer.hwx")?;
println!("Architecture: {}", program.metadata.architecture);
// Should match your device
```

## Best Practices

1. **Pre-extract HWX** during build process, not at runtime
2. **Version HWX files** with architecture suffix: `layer_0_fwd.H16G.hwx`
3. **Cache loaded programs** - HWXLoader has built-in cache
4. **Fallback to MIL** if HWX not available:

```rust
let kernel = match loader.load("layer.hwx") {
    Ok(program) => loader.to_kernel(&program)?,
    Err(_) => {
        // Fallback to MIL compilation
        let mil = generate_mil();
        runtime.compile(&mil)?
    }
};
```

## Future: Direct HWX Generation

Instead of PyTorch→CoreML→HWX, future work could:
1. Generate E5 binaries directly from MIL
2. Skip CoreML entirely
3. Enable training-specific optimizations

See: `tinygrad/extra/accel/ane/` for reverse-engineered format details.

## References

- [maderix ANE Research](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [tinygrad ANE Implementation](https://github.com/tinygrad/tinygrad/tree/master/extra/accel/ane)
- [hollance Neural Engine Docs](https://github.com/hollance/neural-engine)
