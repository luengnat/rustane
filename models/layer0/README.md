# Layer 0 MIL Programs - Parameter-Golf Compatible

This directory contains MIL (Model Intermediate Language) programs for Layer 0 of a transformer model, compatible with the [parameter-golf](https://github.com/jxbz/parameter-golf) training setup.

## Model Architecture (Matches train_gpt.py Baseline)

```
Configuration:
- vocab_size: 1024
- dim: 512
- num_heads: 8 (query), num_kv_heads: 4 (key/value) - GQA
- head_dim: 64 (512 / 8)
- kv_dim: 256 (4 * 64)
- mlp_hidden: 1024 (512 * 2)
- seq_len: 1024
```

## Key Features

### 1. RMSNorm Without Weight
```python
# Standard RMSNorm
output = x / sqrt(mean(x²) + eps)
```
No learnable weight parameter, unlike LayerNorm.

### 2. GQA (Grouped Query Attention)
- **Q heads**: 8 (full attention heads)
- **K/V heads**: 4 (grouped, shared across Q heads)
- **Head dim**: 64 per head
- Reduces memory bandwidth for KV cache

### 3. RoPE (Rotary Position Embeddings)
- Applied to Q and K before attention
- Position-dependent rotation matrices
- No learned positional embeddings needed

### 4. Learnable Parameters

#### QK Gain
```python
q = q * q_gain[None, :, None, None]
```
Per-head learnable scaling factor for queries.

#### ResidMix
```python
x = mix[0] * x + mix[1] * x0
```
Learnable mixing between current activation and initial residual.

#### Scales
```python
x = x + attn_scale * attn_out
x = x + mlp_scale * mlp_out
```
Learnable scaling for attention and MLP outputs.

### 5. SwiGLU MLP
```python
gate = SiLU(x @ W1) * (x @ W3)
output = gate @ W2
```
- W1: [512, 1024]
- W3: [512, 1024]
- W2: [1024, 512]

## Files

### layer_0_fwd.mil
Forward pass MIL program implementing:
1. ResidMix application
2. RMSNorm (pre-attention)
3. GQA with RoPE
4. SDPA (Scaled Dot Product Attention)
5. Output projection with attn_scale
6. Residual connection
7. RMSNorm (pre-FFN)
8. SwiGLU FFN with mlp_scale
9. Final residual connection

**Inputs:**
- `x`: [1, 512, 1, 1024] - input activations
- `q_gain`: [8] - QK gain per head
- `resid_mix_0`: [512] - residual mix component 0
- `resid_mix_1`: [512] - residual mix component 1
- `attn_scale`: [512] - attention output scale
- `mlp_scale`: [512] - MLP output scale

**Output:**
- `out`: [1, 512, 1, 1024] - output activations

### layer_0_bwd.mil
Backward pass MIL program computing gradients for all learnable parameters.

**Input:**
- `dy`: [1, 512, 1, 1024] - gradient from next layer

**Output:**
- `dx`: [1, 512, 1, 1024] - gradient for previous layer

## Weight Files Required

Place these files in `models/layer0/weights/`:

```
layer0_wq.bin     [512, 512]   - Q projection (8 heads * 64 dim = 512)
layer0_wk.bin     [256, 512]   - K projection (4 heads * 64 dim = 256)
layer0_wv.bin     [256, 512]   - V projection (4 heads * 64 dim = 256)
layer0_wo.bin     [512, 512]   - Output projection
layer0_w1.bin     [1024, 512]  - FFN W1
layer0_w2.bin     [512, 1024]  - FFN W2
layer0_w3.bin     [1024, 512]  - FFN W3
```

### Weight Blob Format

Each weight file must be a valid ANE FP16 blob:
```
[0:64]   Header (version markers)
[64:68]  Magic (0xDEADBEEF)
[68:72]  Flags
[72:76]  Payload size (uint32)
[80:84]  Data offset (128)
[128:]   FP16 weight data (little-endian)
```

Total size: 128 + (rows × cols × 2) bytes

## Usage

### Loading and Executing

```rust
use rustane::ane::mil_loader::load_mil_model;

// Load forward pass
let fwd_executor = load_mil_model(
    "models/layer0/layer_0_fwd.mil",
    &[512 * 1024 * 2],  // Input size in bytes (FP16)
    &[512 * 1024 * 2],  // Output size in bytes (FP16)
)?;

// Prepare inputs
let x = vec![0.0f32; 512 * 1024];
let q_gain = vec![1.5f32; 8];
let resid_mix_0 = vec![1.0f32; 512];
let resid_mix_1 = vec![0.0f32; 512];
let attn_scale = vec![1.0f32; 512];
let mlp_scale = vec![1.0f32; 512];

// Execute (note: current MIL only takes x as input, 
// other params are loaded as constants)
let output = fwd_executor.execute_f32(0, 0, &x)?;
```

### Integration with Training Runtime

```rust
use rustane::ane::ANETrainingRuntime;

let config = ANETrainingConfig {
    vocab_size: 1024,
    dim: 512,
    n_layers: 9,  // 4 encoder + 5 decoder
    n_heads: 8,
    seq_len: 1024,
    // ... other params
};

let mut runtime = ANETrainingRuntime::new(config);
runtime.init()?;

// Training loop
for step in 0..iterations {
    let timing = runtime.train_step(&batch_data)?;
    println!("Step {}: loss={:.4}", step, timing.loss);
}
```

## Comparison with Parameter-Golf

| Feature | Our MIL | Parameter-Golf PyTorch |
|---------|---------|------------------------|
| Norm | RMSNorm no weight | RMSNorm no weight |
| Attention | GQA (8 Q, 4 KV) | GQA (8 Q, 4 KV) |
| Position | RoPE (simplified) | RoPE (full) |
| QK Scaling | Learnable q_gain | Learnable q_gain |
| Residual | ResidMix | ResidMix |
| Skip Conn | Simple | Encoder-decoder |
| MLP | SwiGLU | SwiGLU / ReLU² |
| Scale | attn_scale, mlp_scale | attn_scale, mlp_scale |

## ANE Constraints

### Memory
- Maximum tensor size: 16,384 elements per IOSurface
- This model uses: 512 × 1024 = 524,288 elements per tensor
- **Requires tiling** to fit within ANE limits

### Compilation
- ANE has ~119 compile budget per process
- Use delta compilation for weight updates (~50ms vs ~1000ms)
- Pre-compile all kernels at initialization

### Performance
- FP16 throughput: ~11 TOPS on ANE
- Expected speedup vs CPU: 3-5x for this model size
- Memory bandwidth bound for attention

## MIL Syntax Notes

### Common Patterns

**Expand dims for broadcasting:**
```mil
tensor<fp16, [1, 512, 1, 1024]> scale_expanded = expand_dims(x=scale, axes=[0,2,3]);
```

**Element-wise multiplication:**
```mil
tensor<fp16, [1, 512, 1, 1024]> out = mul(x=a, y=b);
```

**RMSNorm:**
```mil
tensor<fp16, [1, 1, 1, 1024]> rms = rsqrt(x=sum);
tensor<fp16, [1, 512, 1, 1024]> out = mul(x=x, y=rms);
```

**Reshape:**
```mil
tensor<fp16, [1, 8, 64, 1024]> reshaped = reshape(shape=[1,8,64,1024], x=input);
```

**Tile for GQA head expansion:**
```mil
tensor<fp16, [1, 8, 1024, 64]> expanded = tile(x=kv, reps=[1,2,1,1]);
```

## Testing

Run the MIL loader tests:

```bash
cargo test --lib mil_loader
```

Verify MIL files exist:

```bash
ls -la models/layer0/*.mil
```

## References

- Parameter-Golf: https://github.com/jxbz/parameter-golf
- Orion ANE Training Framework
- ANE MIL Specification (Apple)
- "Attention Is All You Need" (Transformer)
- "GLU Variants Improve Transformer" (SwiGLU)
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- GQA: "GQA: Training Generalized Multi-Query Transformer Models"
