# Rustane Examples Gallery

Comprehensive collection of examples demonstrating all features of the Rustane ANE training library.

## 🚀 Getting Started

### Quick Start Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [`simple_inference.rs`](simple_inference.rs) | Basic inference with a trained model | Beginner |
| [`train_toy_model.rs`](train_toy_model.rs) | Train a simple toy model | Beginner |
| [`load_synthetic_data.rs`](load_synthetic_data.rs) | Generate synthetic training data | Beginner |
| [`learning_rate_schedules.rs`](learning_rate_schedules.rs) | Compare different LR schedulers | Beginner |

## 🤖 Training Examples

### Basic Training

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`train_transformer_ane.rs`](train_transformer_ane.rs) | Simple transformer training on ANE | Forward/backward on ANE |
| [`train_transformer_ane_full.rs`](train_transformer_ane_full.rs) | Complete training loop with validation | Loss tracking, validation |
| [`train_transformer_ane_hardware.rs`](train_transformer_ane_hardware.rs) | Hardware-specific ANE execution | Low-level ANE integration |
| [`train_with_shards.rs`](train_with_shards.rs) | Sharded data loading | Gradient accumulation |
| [`train_with_shards_synthetic.rs`](train_with_shards_synthetic.rs) | Sharded training with synthetic data | Memory-efficient training |
| [`train_with_parameter_golf_data.rs`](train_with_parameter_golf_data.rs) | Real dataset training | Filesystem data loading |
| [`memory_profile_training.rs`](memory_profile_training.rs) | Memory profiling during training | Optimization insights |

### Advanced Training

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`train_toy_model.rs`](train_toy_model.rs) | Toy model training loop | Optimizer integration |
| [`mlp_classifier.rs`](mlp_classifier.rs) | MLP for classification | Multi-layer training |
| [`gpt_generation.rs`](gpt_generation.rs) | GPT-style text generation | Autoregressive sampling |

## 📊 Data Loading Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`load_synthetic_data.rs`](load_synthetic_data.rs) | Generate random training data | Synthetic data |
| [`load_filesystem_data.rs`](load_filesystem_data.rs) | Load data from filesystem | Real datasets |
| [`collate_batches.rs`](collate_batches.rs) | Batch collation strategies | Padding, truncation |
| [`random_sampling.rs`](random_sampling.rs) | Random data sampling | Dataset shuffling |
| [`batch_optimization.rs`](batch_optimization.rs) | Batch size optimization | Performance tuning |

## 🔧 Layer Components

### Attention Mechanisms

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`simple_attention.rs`](simple_attention.rs) | Basic self-attention | Attention weights |
| [`causal_attention.rs`](causal_attention.rs) | Causal (masked) attention | Autoregressive models |
| [`multi_head_attention.rs`](multi_head_attention.rs) | Multi-head attention | Parallel heads |
| [`decomposed_causal_attention.rs`](decomposed_causal_attention.rs) | Optimized causal attention | Flash attention style |
| [`fused_qkv.rs`](fused_qkv.rs) | Fused QKV projection | Memory optimization |

### Normalization & Activation

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`rmsnorm_pol.rs`](rmsnorm_pol.rs) | RMSNorm with POL | Numerical stability |
| [`convolution.rs`](convolution.rs) | 1D convolution | Feature extraction |

### Complete Layers

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`transformer_layer_pol.rs`](transformer_layer_pol.rs) | Full transformer layer (POL) | Attention + FFN |
| [`transformer_ffn_pol.rs`](transformer_ffn_pol.rs) | Feed-forward network (POL) | SwiGLU activation |
| [`transformer_block_pol.rs`](transformer_block_pol.rs) | Transformer block (POL) | Complete forward pass |
| [`multi_layer.rs`](multi_layer.rs) | Multi-layer transformer | Layer stacking |
| [`sequential_model.rs`](sequential_model.rs) | Sequential model composition | Layer chaining |

## 🎯 Specialized Models

| Example | Description | Use Case |
|---------|-------------|----------|
| [`sentiment_analysis.rs`](sentiment_analysis.rs) | Sentiment classification | NLP tasks |
| [`bert_encoder.rs`](bert_encoder.rs) | BERT-style encoder | Pre-training |
| [`qkt_4d.rs`](qkt_4d.rs) | 4D tensor operations | Advanced models |
| [`sv_4d.rs`](sv_4d.rs) | Singular value decomposition | Linear algebra |

## 🔬 Benchmarking & Profiling

### Performance Benchmarks

| Example | Description | Metrics |
|---------|-------------|---------|
| [`ane_matmul_benchmark.rs`](ane_matmul_benchmark.rs) | Matrix multiplication benchmark | FLOPS, latency |
| [`ane_dynamic_matmul_benchmark.rs`](ane_dynamic_matmul_benchmark.rs) | Dynamic shape matmul | Variable sizes |
| [`ane_tiled_rectangular_matmul_benchmark.rs`](ane_tiled_rectangular_matmul_benchmark.rs) | Tiled matmul optimization | Cache efficiency |
| [`benchmark_suite.rs`](benchmark_suite.rs) | Comprehensive benchmark suite | Multiple operations |
| [`benchmark_backward_performance.rs`](benchmark_backward_performance.rs) | Backward pass benchmarking | CPU vs ANE |

### Testing & Validation

| Example | Description | Purpose |
|---------|-------------|---------|
| [`validate_on_fineweb.rs`](validate_on_fineweb.rs) | Validate on Fineweb dataset | Accuracy testing |
| [`validate_with_parametergolf.py`](validate_with_parametergolf.py) | Python validation script | Cross-language testing |
| [`validate_fineweb_with_torch.py`](validate_fineweb_with_torch.py) | PyTorch comparison | Numerical validation |
| [`test_ane_linear_minimal.rs`](test_ane_linear_minimal.rs) | Minimal linear test | Debugging |
| [`test_conv_mil.rs`](test_conv_mil.rs) | Convolution MIL test | MIL validation |

## 📈 Model Utilities

### Serialization

| Example | Description | Features |
|---------|-------------|----------|
| [`model_serialization.rs`](model_serialization.rs) | Save/load model weights | Checkpointing |
| [`quantization.rs`](quantization.rs) | Model quantization | FP8, INT8 precision |

## 🛠️ Error Handling (Phase 4)

| Example | Description | Features |
|---------|-------------|----------|
| [`error_handling_recovery.rs`](error_handling_recovery.rs) | **NEW** - Comprehensive error handling | Retry, fallback, logging |

### Error Handling Features

- **Error Diagnostics**: Automatic categorization and root cause analysis
- **Automatic Retry**: Exponential backoff with batch size reduction
- **Graceful Fallback**: Automatic CPU degradation when ANE fails
- **Per-Layer Tracking**: Disable ANE for problematic layers
- **Structured Logging**: Severity-based error logging with aggregation

## 📚 Learning Path

### Beginner (New to Rustane)

1. Start with [`simple_inference.rs`](simple_inference.rs) to understand basic inference
2. Try [`train_toy_model.rs`](train_toy_model.rs) for a simple training loop
3. Explore [`load_synthetic_data.rs`](load_synthetic_data.rs) for data loading
4. Read [`learning_rate_schedules.rs`](learning_rate_schedules.rs) for optimization concepts

### Intermediate (Familiar with basics)

1. Study [`train_transformer_ane.rs`](train_transformer_ane.rs) for ANE training
2. Examine [`multi_layer.rs`](multi_layer.rs) for model composition
3. Review [`causal_attention.rs`](causal_attention.rs) for attention mechanisms
4. Practice with [`error_handling_recovery.rs`](error_handling_recovery.rs) for production code

### Advanced (Ready for optimization)

1. Analyze [`train_with_shards.rs`](train_with_shards.rs) for memory optimization
2. Study [`transformer_layer_pol.rs`](transformer_layer_pol.rs) for low-level optimization
3. Benchmark with [`ane_matmul_benchmark.rs`](ane_matmul_benchmark.rs)
4. Profile with [`memory_profile_training.rs`](memory_profile_training.rs)

## 🔍 Finding Examples by Feature

### By Training Component

| Component | Examples |
|-----------|----------|
| **Data Loading** | `load_synthetic_data.rs`, `load_filesystem_data.rs`, `collate_batches.rs` |
| **Model Definition** | `mlp_classifier.rs`, `sentiment_analysis.rs`, `bert_encoder.rs` |
| **Training Loop** | `train_transformer_ane.rs`, `train_with_shards.rs` |
| **Optimization** | `learning_rate_schedules.rs`, `train_toy_model.rs` |
| **Error Handling** | `error_handling_recovery.rs` |

### By ANE Feature

| Feature | Examples |
|---------|----------|
| **Forward Pass** | `simple_inference.rs`, `causal_attention.rs` |
| **Backward Pass** | `train_transformer_ane.rs`, `benchmark_backward_performance.rs` |
| **MIL Generation** | `transformer_layer_pol.rs`, `test_conv_mil.rs` |
| **Memory Optimization** | `train_with_shards.rs`, `fused_qkv.rs` |
| **Error Recovery** | `error_handling_recovery.rs` |

### By Model Architecture

| Architecture | Examples |
|--------------|----------|
| **MLP** | `mlp_classifier.rs` |
| **Transformer** | `bert_encoder.rs`, `gpt_generation.rs` |
| **Attention-only** | `simple_attention.rs`, `multi_head_attention.rs` |
| **Convolutional** | `convolution.rs` |
| **Custom** | `sequential_model.rs`, `multi_layer.rs` |

## 💡 Tips for Running Examples

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15+ (Sequoia)
- Rust toolchain (`cargo`)

### Running Examples

```bash
# Run a specific example
cargo run --example simple_inference

# Run with logging
RUST_LOG=debug cargo run --example train_transformer_ane

# Run ANE benchmarks
cargo run --example ane_matmul_benchmark

# Run error handling demo
cargo run --example error_handling_recovery
```

### Common Issues

1. **ANE not available**: Ensure you're on Apple Silicon with macOS 15+
2. **Out of memory**: Reduce batch size or model dimensions
3. **MIL compilation errors**: Check tensor shapes in the MIL code
4. **Slow performance**: Verify ANE is being used (not CPU fallback)

## 🤝 Contributing Examples

When adding new examples:

1. **Choose a descriptive name** that reflects the example's purpose
2. **Add comprehensive comments** explaining key concepts
3. **Include usage examples** in the doc comment
4. **Update this gallery** with a brief description
5. **Test on real hardware** to ensure ANE execution works

### Example Template

```rust
//! Brief description of what this example demonstrates
//!
//! # Purpose
//!
//! Explain the learning objective or feature being demonstrated
//!
//! # Key Concepts
//!
//! - Concept 1
//! - Concept 2
//!
//! # Usage
//!
//! ```bash
//! cargo run --example example_name
//! ```

use rustane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example code here
    Ok(())
}
```

## 📖 Further Reading

- [API Documentation](https://docs.rs/rustane)
- [Phase 4 Summary](../ROADMAP_SUMMARY.md)
- [Training Guide](../docs/training-guide.md)
- [ANE Internals](../docs/ane-internals.md)
