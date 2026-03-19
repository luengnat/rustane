# Rustane - Apple Neural Engine Rust Library

[![Rust](https://img.shields.io/badge/rust-2021--edition-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: macOS 15+](https://img.shields.io/badge/platform-macOS%2015%2B-lightgrey)](https://www.apple.com/macos/)

**Rustane** is a Rust library providing safe, idiomatic APIs for the Apple Neural Engine (ANE). It wraps private ANE bridge APIs with FFI bindings so Rust programs can run high-performance inference on Apple Silicon.

## Status

Rustane is experimental and depends on private, undocumented Apple APIs. Interface details may change with future macOS releases.

## Features

- **Safe Rust wrappers** around ANE private APIs
- **FFI bindings** to the C bridge from [maderix/ANE](https://github.com/maderix/ANE)
- **MIL program builder** for constructing neural network graphs
- **Weight blob utilities** for FP32→FP16 conversion and matrix operations
- **Type-safe tensors** with shape validation and bounds checking
- **Examples** demonstrating inference workflows
- **Platform detection** for ANE availability

## Platform Requirements

- **OS**: macOS 15+ (Sequoia)
- **Hardware**: Apple Silicon with ANE (M1/M2/M3/M4)
- **Rust**: 1.70+ with 2021 edition

## Quick Start

### Prerequisites

1. Build the ANE bridge library from [maderix/ANE](https://github.com/maderix/ANE):
   ```bash
   cd ~/dev/ANE/bridge
   make
   ```

2. Set the library path (or install to system location):
   ```bash
   export ANE_BRIDGE_LIB_PATH=~/dev/ANE/bridge
   export ANE_BRIDGE_INCLUDE_PATH=~/dev/ANE/bridge
   ```

### Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
rustane = "0.1.0"
```

Initialize and use the ANE:
```rust
use rustane::{init, mil::{MILBuilder, WeightBlob}, wrapper::{ANECompiler, ANETensor}};

fn main() -> rustane::Result<()> {
    // Check platform compatibility
    let avail = rustane::ANEAvailability::check();
    if !avail.is_available() {
        eprintln!("ANE not available: {}", avail.describe());
        return Ok(());
    }

    // Initialize ANE runtime
    init()?;

    // Create a simple linear layer
    let mil = MILBuilder::new()
        .add_linear("fc1", "input", "weights", 512)
        .build();

    // Prepare weights
    let weights = vec![1.0f32; 256 * 512];
    let blob = WeightBlob::from_fp32(&weights, 256, 512)?;

    // Compile and execute
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(
        &mil,
        Some(blob.as_bytes()),
        &[256 * 4],
        &[512 * 4],
    )?;

    // Run inference
    let input = ANETensor::from_fp32(vec![0.0; 256], vec![1, 256])?;
    executor.write_input(0, input.as_bytes())?;
    executor.eval()?;

    // Read output
    let mut output = vec![0u8; 512 * 4];
    executor.read_output(0, &mut output)?;

    Ok(())
}
```

## Examples

The library includes several examples demonstrating different use cases:

```bash
# Simple inference with a linear layer
cargo run --example simple_inference

# Packed matmul benchmark
cargo run --example ane_dynamic_matmul_benchmark

# Rectangular tiled benchmark
cargo run --example ane_tiled_rectangular_matmul_benchmark
```

### Example: Simple Linear Layer

For a runnable example, see [`examples/simple_inference.rs`](examples/simple_inference.rs).
It shows the end-to-end flow for checking ANE availability, compiling a MIL program,
writing inputs, and reading outputs.

## Benchmarks

The current matmul benchmark of record is the packed dynamic layout in
[`examples/ane_dynamic_matmul_benchmark.rs`](examples/ane_dynamic_matmul_benchmark.rs).
For rectangular projections, see
[`examples/ane_tiled_rectangular_matmul_benchmark.rs`](examples/ane_tiled_rectangular_matmul_benchmark.rs).

For a side-by-side comparison with MLX, see:
- [`MLX_MATMUL_COMPARISON.md`](MLX_MATMUL_COMPARISON.md)
- [`examples/mlx_matmul_benchmark.py`](examples/mlx_matmul_benchmark.py)

Current measured packed-matmul results on this machine:
- Rustane ANE: `0.070 ms` average for `64×64×64`, `0.091 ms` average for `128×128×64`
- MLX: `0.560 ms` average for `64×64×64`, `0.518 ms` average for `128×128×64`

## Project Structure

```
rustane/
├── src/
│   ├── lib.rs          # Public API surface
│   ├── sys.rs          # Raw FFI bindings (generated via bindgen)
│   ├── error.rs        # Error types
│   ├── wrapper/        # Safe wrapper types
│   │   ├── runtime.rs  # ANE runtime initialization
│   │   ├── tensor.rs   # Type-safe tensor wrapper
│   │   ├── compiler.rs # MIL program compiler
│   │   └── executor.rs # Kernel executor & I/O
│   ├── mil/            # MIL utilities
│   │   ├── builder.rs  # MIL program builder
│   │   ├── programs.rs # Predefined layer programs
│   │   └── util.rs     # Weight blob utilities
│   └── platform.rs     # Platform detection
├── examples/           # Usage examples
├── build.rs            # Build script for FFI bindings
└── tests/              # Integration and parity tests
```

## Documentation

- [Docs Index](docs/README.md) - Public entry point for docs and notes
- [CI Workflow](.github/workflows/ci.yml) - macOS 15 build and test pipeline
- [Release Validation](.github/workflows/release.yml) - tag-based dry-run publish check
- [API Documentation](https://docs.rs/rustane) - Coming soon
- [Examples](examples/) - Working inference examples
- [ANE Architecture](https://github.com/hollance/neural-engine) - Comprehensive ANE documentation
- [Upstream ANE Project](https://github.com/maderix/ANE) - Research project with C bridge

### API Overview

**Platform Detection**
```rust
use rustane::ANEAvailability;

let avail = ANEAvailability::check();
if avail.is_available() {
    println!("ANE is ready!");
}
```

**MIL Program Builder**
```rust
use rustane::MILBuilder;

let mil = MILBuilder::new()
    .add_input("data", &[1, 3, 224, 224])
    .add_convolution("conv1", "data", "weights", 64, [7, 7], [2, 2])
    .build();
```

**Weight Management**
```rust
use rustane::WeightBlob;

let blob = WeightBlob::from_fp32(&weights, rows, cols)?;
compiler.compile_single(&mil, Some(blob.as_bytes()), ...)?;
```

**Tensor Operations**
```rust
use rustane::ANETensor;

let tensor = ANETensor::from_fp32(data, vec![1, 256])?;
executor.write_input(0, tensor.as_bytes())?;
```

## Model Composition

Rustane provides high-level APIs for building neural networks from layer primitives.

### Sequential Models

Build simple feed-forward networks using the `Sequential` API:

```rust
use rustane::layers::{Sequential, Linear, ReLU};

let model = Sequential::new("my_mlp")
    .add(Box::new(Linear::new(784, 256).build()?))
    .add(Box::new(ReLU::new()))
    .add(Box::new(Linear::new(256, 10).build()?))
    .build();

println!("{}", model.summary());
```

### Parameter Sharing

Share layers between multiple models using `SharedLayer`:

```rust
use rustane::layers::{Sequential, SharedLayer, ReLU};

let shared_relu = SharedLayer::new(ReLU::new());

let model1 = Sequential::new("model1")
    .add_shared(shared_relu.clone())
    .build();

let model2 = Sequential::name("model2")
    .add_shared(shared_relu)
    .build();

// Both models share the same ReLU layer
```

## Known Limitations

### Platform Requirements
- **macOS 15+**: Private ANE APIs require macOS 15 (Sequoia) or later
- **Apple Silicon**: ANE hardware only available on M1/M2/M3/M4 chips
- **Private APIs**: Uses undocumented APIs that may break with macOS updates

### Technical Limitations
- **~119 compile limit**: ANE compiler leaks resources; requires exec() restart workaround
- **Single-input constraint**: Multi-input ANE requests require packing into spatial dimensions
- **SDPA causal masking**: ANE hardware ignores attn_mask; requires CPU fallback for causal attention
- **FP16 underflow**: Very deep or wide fp16 workloads may require careful scaling

### C Bridge Limitations
- **INT8 support**: `ane_bridge_build_weight_blob_int8` not yet in compiled library
- **Quantization**: `ane_bridge_build_weight_blob_quantized` not yet available
- **Memory cleanup**: `ane_bridge_free_blob` not yet implemented (memory leaks on blob creation)

### Performance Notes
- Utilization is low (~5-9% of peak) for many operations
- Some element-wise operations fall back to CPU
- Optimal performance requires careful MIL program design

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

Based on research and code from:
- [maderix/ANE](https://github.com/maderix/ANE) - Reverse-engineered ANE bridge implementation
- [hollance/neural-engine](https://github.com/hollance/neural-engine) - ANE architecture documentation

## Disclaimer

This project uses Apple's private, undocumented APIs. These are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture. This project is not affiliated with or endorsed by Apple Inc.

---

**Built with Rust + Claude**
