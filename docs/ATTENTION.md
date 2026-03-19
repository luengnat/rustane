# Attention Mechanism Implementation

## Overview

Rustane provides multi-head attention layer implementations with support for:
- **MultiHeadAttention**: Full-featured attention layer with builder pattern
- **SelfAttention**: Convenience wrapper with causal masking
- **CPU Softmax**: Numerically stable softmax with causal mask
- **MIL Program Generation**: Native `scaled_dot_product_attention` support

## Current Status

✅ **Implemented**:
- Layer structure and builder pattern
- CPU softmax with causal masking
- Parameter counting and validation
- 10 passing tests
- Clean API following Rustane patterns

⚠️ **Known Limitations**:
- Forward pass not yet implemented (requires MIL program compilation)
- `scaled_dot_product_attention` MIL operation has validation issues
- ANE compiler rejects current MIL program format (InvalidMILProgram)

## Usage

### Basic MultiHeadAttention

```rust
use rustane::{Layer, MultiHeadAttentionBuilder};

let mha = MultiHeadAttentionBuilder::new(64, 4)
    .with_causal(true)
    .build()?;

println!("Parameters: {}", mha.num_parameters());
println!("Input shape: {:?}", mha.input_shape());
```

### Self-Attention (Convenience Wrapper)

```rust
use rustane::{Layer, SelfAttentionBuilder};

let sa = SelfAttentionBuilder::new(128, 8)
    .with_dropout(0.1)
    .build()?;

// Self-attention has causal masking enabled by default
```

### CPU Softmax with Causal Mask

```rust
use rustane::layers::attention::SoftmaxWithCausalMask;

let softmax = SoftmaxWithCausalMask::new(4);
let scores = vec![1.0f32; 16]; // 4x4
let result = softmax.compute(scores)?;
```

## API Reference

### MultiHeadAttention

**Builder Methods:**
- `new(embed_dim, num_heads)` - Create builder
- `with_name(name)` - Set layer name
- `with_head_dim(dim)` - Set head dimension (default: embed_dim / num_heads)
- `with_dropout(rate)` - Set dropout probability
- `with_bias(enabled)` - Enable/disable bias
- `with_causal(enabled)` - Enable causal masking
- `build()` - Build the layer

**Layer Trait Methods:**
- `name()` - Get layer name
- `num_parameters()` - Count total parameters
- `input_shape()` - Get input shape
- `output_shape()` - Get output shape
- `forward()` - Execute forward pass (not yet implemented)

### SelfAttention

**Builder Methods:**
- `new(embed_dim, num_heads)` - Create builder
- `with_head_dim(dim)` - Set head dimension
- `with_dropout(rate)` - Set dropout probability
- `with_bias(enabled)` - Enable/disable bias
- `build()` - Build the layer (always causal)

## Examples

### Example 1: Simple Attention API

```bash
cargo run --example simple_attention
```

Demonstrates the attention layer API without execution.

### Example 2: Multi-Head Attention

```bash
cargo run --example multi_head_attention
```

Shows how to use MultiHeadAttention with SDPA MIL program (currently has compilation issues).

### Example 3: Causal Attention (SDPA)

```bash
cargo run --example causal_attention
```

Runs scaled_dot_product_attention on ANE with CPU reference verification (currently has compilation issues).

## Technical Details

### Architecture

The MultiHeadAttention layer follows this architecture:

```
Input [batch, seq_len, embed_dim]
  ↓
Q, K, V Projections (Linear layers)
  ↓
Reshape [batch, num_heads, seq_len, head_dim]
  ↓
Scaled Dot-Product Attention
  ↓
Reshape [batch, seq_len, embed_dim]
  ↓
Output Projection (Linear layer)
  ↓
Output [batch, seq_len, embed_dim]
```

### CPU Softmax Implementation

The CPU softmax implementation:
1. Applies causal mask (upper triangle = -inf)
2. Numerically stable softmax (subtract max before exp)
3. Normalizes along sequence dimension
4. Handles edge cases (overflow, underflow)

### MIL Program Generation

The `build_sdpa_mil_program()` method generates MIL programs using the native `scaled_dot_product_attention` operation:

```rust
let mil = mha.build_sdpa_mil_program(batch_size, seq_len);
```

**Current Issue**: The ANE compiler rejects the generated MIL program with `InvalidMILProgram`.

## Testing

### Test Coverage

- `test_softmax_causal_mask` - Verifies causal mask application
- `test_softmax_stability` - Tests numerical stability
- `test_softmax_compute` - Integration test for softmax
- `test_mha_builder_valid` - Validates builder creates correct instances
- `test_mha_builder_invalid_dimensions` - Tests embed_dim % num_heads validation
- `test_mha_builder_custom_head_dim` - Tests custom head dimension
- `test_mha_parameters` - Verifies parameter counting
- `test_self_attention_convenience` - Tests SelfAttention wrapper
- `test_self_attention_defaults` - Tests default values
- `test_softmax_invalid_length` - Tests error handling

**Total**: 10 tests, all passing

### Running Tests

```bash
# Run all attention tests
cargo test --lib attention

# Run specific test
cargo test test_softmax_causal_mask

# Run with output
cargo test --lib attention -- --nocapture
```

## Performance Considerations

### ANE Limitations

1. **Compile Limit**: ~119 compilations per process
   - Mitigation: Compile caching, kernel reuse

2. **MIL Validation**: ANE compiler has strict MIL syntax requirements
   - Current status: `scaled_dot_product_attention` fails validation
   - Need to investigate exact MIL format requirements

3. **Memory Constraints**: Large sequence lengths may exhaust IOSurface memory
   - Mitigation: Chunked processing, CPU fallback

### Performance Optimization

Future optimizations:
- Fused QKV projection (single ANE execution instead of 3)
- Compile caching to avoid hitting ~119 limit
- Batch processing for multiple sequences
- FP16 conversion optimizations

## Known Issues

### Issue 1: MIL Program Compilation Failure

**Error**: `InvalidMILProgram` when compiling `scaled_dot_product_attention`

**Status**: Under investigation

**Workaround**: Use CPU-based attention for now

**Impact**: Forward pass cannot execute on ANE

### Issue 2: Causal Masking on ANE

**Limitation**: ANE ignores attention masks (from upstream documentation)

**Solution**: CPU softmax with manual causal masking (implemented)

**Impact**: Hybrid CPU/ANE approach required

## Future Work

### Short Term

1. Fix MIL program compilation issues
2. Implement complete forward pass
3. Add integration tests with Sequential models
4. Performance benchmarking

### Long Term

1. Fused QKV projection optimization
2. Compile caching infrastructure
3. Support for larger sequence lengths
4. Integration with transformer models

## References

- [Scaled Dot-Product Attention](https://arxiv.org/abs/1706.03762) - Paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [maderix/ANE](https://github.com/maderix/ANE) - Upstream ANE research
- [hollance/neural-engine](https://github.com/hollance/neural-engine) - ANE documentation

## Contributing

When working on attention mechanisms:

1. Run tests: `cargo test --lib attention`
2. Check examples: `cargo run --example simple_attention`
3. Follow existing patterns (builder pattern, Layer trait)
4. Document MIL format requirements
5. Add tests for new features

## Changelog

### Phase 3.2.1 (2025-03-19)

- ✅ Added `SoftmaxWithCausalMask` (CPU-based)
- ✅ Added `MultiHeadAttention` layer structure
- ✅ Added `SelfAttention` convenience wrapper
- ✅ Added builder pattern for both layers
- ✅ Added 10 comprehensive tests
- ✅ Added MIL program generation method
- ⚠️ Forward pass not yet implemented (MIL compilation issues)
- ⚠️ SDPA examples fail with InvalidMILProgram

### Next Phase

- Fix MIL program compilation
- Implement working forward pass
- Add performance benchmarks
- Integration with Sequential models
