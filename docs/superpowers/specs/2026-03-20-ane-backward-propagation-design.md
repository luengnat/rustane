# ANE-Accelerated Backward Propagation for Rustane

> **Goal:** Implement production-ready backward propagation using Apple Neural Engine via `objc2` Rust bindings to private ANE framework, enabling full transformer training on Apple Silicon.

> **Architecture:** Three-layer integration — (1) safe Rust objc2 bindings to private ANE framework, (2) MIL code generation for transformer operations, (3) CPU-based backward pass (refactored for Rust idioms) with cached activations. Forward pass on ANE, backward on CPU, all within rustane core.

> **Tech Stack:** Rust, `objc2`, private `AppleNeuralEngine.framework`, IOSurface, MIL (Model Intermediate Language).

---

## Brainstorming Decisions (Architectural Rationale)

The following key decisions were validated through the brainstorming process:

1. **Use ANE for forward pass, CPU for backward pass** (not full ANE backward)
   - Rationale: ANE gradient API is undocumented and harder to work with. CPU backward is proven (works in ~/dev/ANE/training/), numerically stable, and fast enough.
   - Trade-off: ~30% more latency per step vs 10x easier to debug and maintain

2. **Implement in Rust via objc2 (not use C bridge)**
   - Rationale: Direct Rust bindings are safer (no FFI overhead), more idiomatic, leverage type system for correctness
   - Trade-off: More initial development vs long-term maintainability

3. **Include activation caching strategy**
   - Rationale: Enables CPU backward without recomputing forward pass; memory acceptable for 7.2M model
   - Trade-off: ~2x memory for ~10x backward speedup

4. **Refactor C code to idiomatic Rust (not line-for-line port)**
   - Rationale: C patterns don't translate directly; Rust idioms (Result, iterators, no pointers) improve safety and readability
   - Trade-off: More analysis of C code needed upfront

5. **Module location: rustane core (src/ane/, src/layers/)**
   - Rationale: ANE bindings are low-level infrastructure; transformer backward is a general layer feature; both should be core
   - Trade-off: No isolation boundary, but closer integration with training loop

---

## Target Transformer Configuration

From brainstorming decision: implement 7.2M parameter transformer per existing plan.

**Calculated spec** (matching 7.2M param constraint):

```rust
pub struct TransformerConfig {
    pub vocab_size: usize,     // 4096 (token vocabulary)
    pub dim: usize,            // 256 (model dimension)
    pub hidden_dim: usize,     // 768 (FFN intermediate, 3x expansion)
    pub n_heads: usize,        // 8 (attention heads)
    pub head_dim: usize,       // 32 (dim / n_heads)
    pub n_layers: usize,       // 6 (transformer layers)
    pub seq_len: usize,        // 512 (max sequence length)
}
```

**Parameter breakdown:**
- Embedding: 4096 × 256 = 1.049M
- Classifier (LM head): 256 × 4096 = 1.049M
- Per layer: (3×256×256 for attention + 256×768×2 for FFN + 2×256 for norms) ≈ 838.5k
- 6 layers: 5.031M
- **Total: 7.129M parameters** ✅

---

## Problem Statement

Rustane currently has a Model trait with `forward()` and `backward()` methods, but backward propagation is not implemented. The goal is to:

1. Use ANE for fast forward passes (10-20x faster than CPU)
2. Implement mathematically correct backward passes
3. Integrate with existing rustane training infrastructure (Trainer, DataLoader, optimizers)
4. Achieve production-ready performance (~100-150 ms/step for ~100M param transformer)

---

## Design Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        rustane/src/                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ training/                                                │  │
│  │  • trainer.rs (existing Trainer)                         │  │
│  │  • transformer_model.rs (NEW: ANE-based Model impl)      │  │
│  │  • model.rs (existing Model trait)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↑                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ane/ (NEW: ANE module)                                   │  │
│  │  • mod.rs (public API)                                   │  │
│  │  • runtime.rs (objc2 bindings)                           │  │
│  │  • kernel.rs (ANEKernel wrapper)                         │  │
│  │  • weight_blob.rs (blob builders)                        │  │
│  │  • io_surface.rs (IOSurface RAII)                        │  │
│  │  • error.rs (ANEError type)                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│          ↑                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ layers/                                                  │  │
│  │  • mil_gen.rs (NEW: MIL code generation)                 │  │
│  │  • transformer_backward.rs (NEW: backward pass)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────┐
        │  objc2 (Rust ↔ Objective-C bridge)   │
        └───────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────┐
    │  Private AppleNeuralEngine.framework         │
    │  • _ANEInMemoryModel                         │
    │  • _ANEInMemoryModelDescriptor               │
    │  • _ANERequest                               │
    │  • _ANEIOSurfaceObject                       │
    └──────────────────────────────────────────────┘
```

### Data Flow: Forward + Backward

```
Forward Pass (on ANE):
  Batch [tokens] → Embedding → Layer0 (attn+ffn) → ... → LayerN → Logits

Activations cached during forward:
  ├─ Per-layer hidden states (after norm, before attn/ffn)
  ├─ Q, K, V projections
  ├─ Attention scores (softmax output)
  └─ FFN hidden states

Backward Pass (on CPU, using cached activations):
  Loss gradient → dLogits
  dLogits → dEmbedding + dLayer0 + ... + dLayerN gradients

Weight gradients accumulated for each parameter:
  ├─ dW for all W matrices (attention, FFN)
  ├─ dRMSNorm for all layer norms
  ├─ dEmbed for embedding layer
  └─ Passed to optimizer for parameter update
```

---

## Component Design

### 1. ANE Runtime (`src/ane/runtime.rs`)

**Responsibility:** Safe Rust wrapper around private ANE framework via `objc2`.

**Interface:**

```rust
/// Initialize ANE runtime (load private framework, resolve classes)
pub fn ane_init() -> Result<()>

/// Compile MIL program with weights into ANE kernel
pub struct ANECompileRequest {
    pub mil_text: String,
    pub weights: HashMap<String, WeightBlob>,  // "@model_path/weights/name.bin" → blob
    pub input_sizes: Vec<usize>,                // byte sizes for each input
    pub output_sizes: Vec<usize>,               // byte sizes for each output
}

impl ANECompileRequest {
    pub fn compile(self) -> Result<ANEKernel>
}
```

**Implementation Details:**

- Use `objc2` to call private methods on `_ANEInMemoryModel`, `_ANEInMemoryModelDescriptor`
- Handle QoS (Quality of Service) level 21 for training workloads
- Manage temporary directories for MIL + weight files (required by ANE compiler)
- Return strongly-typed `ANEKernel` wrapper

**Error Handling:**

- Compilation errors from ANE compiler
- Framework loading failures
- Invalid QoS parameters

---

### 2. ANE Kernel (`src/ane/kernel.rs`)

**Responsibility:** Manage lifecycle of a compiled ANE kernel and its I/O.

**Interface:**

```rust
pub struct ANEKernel {
    model: /* _ANEInMemoryModel via objc2 */,
    request: /* _ANERequest via objc2 */,
    io_inputs: Vec<IOSurface>,
    io_outputs: Vec<IOSurface>,
    input_sizes: Vec<usize>,
    output_sizes: Vec<usize>,
}

impl ANEKernel {
    /// Evaluate kernel on ANE (execute computation)
    pub fn eval(&mut self) -> Result<()>

    /// Write input data to IOSurface
    pub fn write_input(&mut self, idx: usize, data: &[f32]) -> Result<()>

    /// Read output data from IOSurface
    pub fn read_output(&mut self, idx: usize) -> Result<Vec<f32>>
}

impl Drop for ANEKernel {
    fn drop(&mut self) {
        // Unload kernel from ANE, free IOSurfaces
    }
}
```

**Implementation Details:**

- IOSurface creation with correct width/height (packed in spatial dimension)
- Lock/unlock for thread-safe access
- Proper cleanup on drop (unload from ANE)

---

### 3. Weight Blob Builders (`src/ane/weight_blob.rs`)

**Responsibility:** Build ANE-formatted weight blobs from Rust arrays.

**ANE Blob Format:**

```
[0-64):    Global header (magic, version, etc.)
[64-128):  Chunk header (magic, data_size, offset)
[128-...): FP16 weight data (f32 converted to _Float16)
```

**Interface:**

```rust
pub struct WeightBlob(Vec<u8>);

impl WeightBlob {
    /// Build from FP32 weights [rows × cols]
    pub fn from_f32(weights: &[f32], rows: usize, cols: usize) -> Self

    /// Build from FP16 weights
    pub fn from_f16(weights: &[half::f16], rows: usize, cols: usize) -> Self

    /// Build from quantized int8 (per-channel scale)
    pub fn from_i8_quantized(
        weights: &[i8],
        scale: f32,
        rows: usize,
        cols: usize
    ) -> Self

    /// Quantize FP32 to int8 and build blob
    /// Returns (blob, per_channel_scales)
    pub fn quantize_f32(
        weights: &[f32],
        rows: usize,
        cols: usize
    ) -> (Self, Vec<f32>)
}

impl AsRef<[u8]> for WeightBlob { ... }
```

**Implementation Details:**

- Use `half` crate for FP32 ↔ FP16 conversion
- Per-channel quantization: scale = max(abs(row)) / 127
- Proper endianness for ANE hardware

---

### 4. IOSurface Management (`src/ane/io_surface.rs`)

**Responsibility:** Safe RAII wrapper around IOSurface.

**Interface:**

```rust
pub struct IOSurface {
    surface: IOSurfaceRef,
}

impl IOSurface {
    /// Create IOSurface for given byte capacity
    pub fn new(bytes: usize) -> Result<Self>

    /// Write data to surface (locks, copies, unlocks)
    pub fn write(&mut self, data: &[u8]) -> Result<()>

    /// Read data from surface (locks, copies, unlocks)
    pub fn read(&self) -> Result<Vec<u8>>

    /// Get base address for direct access (advanced)
    pub fn with_lock<F, R>(&mut self, f: F) -> Result<R>
    where F: FnOnce(*mut u8) -> R
}

impl Drop for IOSurface {
    fn drop(&mut self) {
        CFRelease(self.surface)
    }
}
```

---

### 5. MIL Code Generation (`src/layers/mil_gen.rs`)

**Responsibility:** Generate MIL (Model Intermediate Language) code for transformer operations.

**Architecture:**

- Port MIL generators from `~/dev/ANE/training/stories_mil.h`
- Refactor from C macros to Rust builder pattern
- Generate text strings that ANE compiler consumes

**Interface:**

```rust
pub struct MILGenerator {
    config: TransformerConfig,  // dim, hidden_dim, seq_len, n_heads, etc.
}

impl MILGenerator {
    pub fn new(config: TransformerConfig) -> Self

    /// Generate MIL for attention forward
    /// Input: [1, dim, 1, seq + q_dim + kv_dim + kv_dim] (x + Wq + Wk + Wv)
    /// Output: [1, output_dim, 1, seq] (attention + residual)
    pub fn gen_attention_forward(&self) -> String

    /// Generate MIL for FFN forward
    /// Input: [1, dim, 1, seq + hidden_dim + hidden_dim] (x + W1 + W3)
    /// Output: [1, output_dim, 1, seq] (FFN + residual)
    pub fn gen_ffn_forward(&self) -> String

    /// Generate MIL for attention backward (if supported by ANE)
    /// Currently: fallback to CPU (see transformer_backward.rs)
    pub fn gen_attention_backward(&self) -> String

    /// Generate MIL for FFN backward
    pub fn gen_ffn_backward(&self) -> String
}
```

**Key Decisions:**

- Use MIL matmul op (more efficient than conv workaround)
- Activate weights via spatial dimension (IOSurface packing)
- FP16 computation with FP32 I/O conversion

---

### 6. Transformer Backward Pass (`src/layers/transformer_backward.rs`)

**Responsibility:** Implement backward propagation for transformer layers.

**Architecture:**

- Refactored from C to Rust: idiomatic error handling, iterators, no manual memory
- CPU-based (backward on CPU is proven, fast enough for most use cases)
- Reuses cached activations from forward pass

**Interface:**

```rust
/// Backward for scaled dot-product attention
///
/// # Inputs
/// - `d_out`: Gradient w.r.t. attention output [seq_len, dim]
/// - `q`, `k`, `v`: Forward-pass projections [seq_len, dim] (cached)
/// - `attn_weights`: Softmax(QK^T/√d_k) [seq_len, seq_len] (cached, per head)
/// - `config`: Attention config (n_heads, head_dim, dim)
///
/// # Returns
/// - `(d_x, dw_q, dw_k, dw_v)`: Gradients w.r.t. input, query/key/value weights
pub fn attention_backward(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_weights: &[f32],  // Required: softmax scores
    config: &AttentionConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>

/// Backward for feed-forward network (SiLU gating: W1*W3 style)
///
/// # Inputs
/// - `d_out`: Gradient w.r.t. FFN output [seq_len, dim]
/// - `x`: Pre-FFN activation [seq_len, dim] (from cache: x_ffn_norm)
/// - `w1_out`: W1(x) before SiLU [seq_len, hidden_dim] (cached)
/// - `w1_gated`: W1(x) * SiLU(W1(x)) [seq_len, hidden_dim] (cached)
/// - `config`: FFN config (dim, hidden_dim)
///
/// # Returns
/// - `(d_x, dw1, dw3, dw2)`: Gradients w.r.t. input and all weight matrices
pub fn ffn_backward(
    d_out: &[f32],
    x: &[f32],
    w1_out: &[f32],
    w1_gated: &[f32],
    config: &FFNConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>

/// Backward for RMSNorm
///
/// # Inputs
/// - `d_out`: Gradient w.r.t. normalized output [seq_len, dim]
/// - `x`: Pre-normalized input [seq_len, dim] (from cache: x_pre_*_norm)
/// - `w`: Scale parameter [dim]
///
/// # Returns
/// - `(d_x, dw)`: Gradients w.r.t. input and scale weight
pub fn rmsnorm_backward(
    d_out: &[f32],
    x: &[f32],
    w: &[f32],
) -> (Vec<f32>, Vec<f32>)

/// Cross-entropy loss backward
///
/// Computes gradient: softmax(logits) - one_hot(target)
pub fn cross_entropy_backward(
    logits: &[f32],
    targets: &[u32],
    vocab_size: usize,
) -> Vec<f32>
```

**Key Refactorings:**

- Use `ndarray` or manual slicing for clarity (no raw pointer arithmetic)
- Iterator-based loops instead of C-style for loops
- Type-safe dimension tracking (seq_len, dim, hidden_dim as parameters)
- Numerical stability (max subtraction in softmax, safe division)
- No malloc; use Vec with proper capacity pre-allocation

---

### 7. Transformer Model (`src/training/transformer_model.rs`)

**Responsibility:** Implement Model trait for full transformer with ANE forward + CPU backward.

**Interface:**

```rust
pub struct TransformerANE {
    // Configuration
    config: TransformerConfig,

    // ANE kernels (one per layer, two per layer: attn + ffn)
    attn_kernels: Vec<ANEKernel>,
    ffn_kernels: Vec<ANEKernel>,

    // Weights (host memory, copied to ANE on each forward)
    embedding: Vec<f32>,        // [vocab_size, dim]
    weights: Vec<LayerWeights>, // one per layer
    final_norm: Vec<f32>,       // [dim]
    classifier: Vec<f32>,       // [dim, vocab_size]

    // Cached activations for backward
    cached: CachedActivations,
}

pub struct CachedActivations {
    // Layer inputs (pre-norm) - needed for backward
    x_pre_attn_norm: Vec<Vec<f32>>,        // [nlayers][seq_len, dim] input to attention norm
    x_pre_ffn_norm: Vec<Vec<f32>>,         // [nlayers][seq_len, dim] input to FFN norm

    // Normalized activations after RMSNorm (inputs to attention/FFN)
    x_attn_norm: Vec<Vec<f32>>,            // [nlayers][seq_len, dim] RMSNorm output before attention
    x_ffn_norm: Vec<Vec<f32>>,             // [nlayers][seq_len, dim] RMSNorm output before FFN

    // Attention components (needed for attention_backward)
    q: Vec<Vec<f32>>,                      // [nlayers][seq_len, dim] query projections
    k: Vec<Vec<f32>>,                      // [nlayers][seq_len, dim] key projections
    v: Vec<Vec<f32>>,                      // [nlayers][seq_len, dim] value projections
    attn_weights: Vec<Vec<f32>>,           // [nlayers][seq_len, seq_len] softmax attention scores

    // FFN components (needed for ffn_backward)
    w1_out: Vec<Vec<f32>>,                 // [nlayers][seq_len, hidden_dim] W1(x) before SiLU gating
    w1_gated: Vec<Vec<f32>>,               // [nlayers][seq_len, hidden_dim] W1(x) * SiLU(W1(x))

    // Final layer
    x_final_norm: Vec<f32>,                // [seq_len, dim] Final RMSNorm output
}

impl Model for TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        // 1. Embedding lookup
        // 2. Per-layer: RMSNorm → ANE attention forward → ANE FFN forward
        // 3. Cache activations during forward
        // 4. Final RMSNorm + classifier
        // 5. Return logits as ANETensor
    }

    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        // 1. Cross-entropy backward from loss
        // 2. Per-layer (reverse): classifier backward, norm backward, FFN backward, attn backward
        // 3. Accumulate weight gradients
        // 4. Return all gradients as flat Vec<f32>
    }

    fn parameters(&mut self) -> &mut [f32] {
        // Return mutable view of all weights (embedding + layers + classifier)
    }

    fn param_count(&self) -> usize {
        // Return total parameter count
    }
}
```

**Activation Caching Strategy:**

- During forward: cache all intermediate activations at each layer
- During backward: use cached values instead of recomputing
- Clear cache after backward to free memory
- Trade: ~2x memory for ~10x backward speedup

---

## Integration Points

### 1. With existing Model trait
- `TransformerANE` implements `Model` directly
- No changes to Trainer, optimizer, scheduler, data loader
- Works with existing training loop

### 2. With Batch structure
```rust
impl TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        let tokens = batch.token_ids;  // &[u32]
        let seq_len = batch.seq_len;
        // ... forward pass ...
    }
}
```

### 3. With ANETensor

**Data flow in TransformerANE::forward():**

```rust
impl Model for TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        // ... forward pass computes logits as Vec<f32> ...

        // Convert Vec<f32> → ANETensor
        let logits: Vec<f32> = /* ... */;
        let shape = vec![batch.batch_size, batch.seq_len, self.config.vocab_size];

        // ANETensor stores as f32 (not bytes) - see wrapper.rs
        Ok(ANETensor::from_fp32(logits, shape)?)
    }
}
```

**Data flow in TransformerANE::backward():**

```rust
impl Model for TransformerANE {
    fn backward(&mut self, loss: f32) -> Result<Vec<f32>> {
        // loss is scalar from trainer

        // Cross-entropy backward: softmax(logits) - one_hot(target)
        let logits = /* extract from cached forward */;
        let targets = /* extract from batch */;
        let d_logits = cross_entropy_backward(&logits, &targets, vocab_size);

        // Propagate through transformer layers...
        // All backward functions work with Vec<f32> directly

        Ok(all_accumulated_gradients)
    }
}
```

**Note:** ANETensor stores data as f32 internally (`wrapper.rs`). No byte conversion needed for forward/backward integration.

---

## Safety & Private API Justification

**objc2 Safety:**

- `objc2` crate provides safe Rust abstractions over Objective-C runtime
- No unsafe Rust in the ANE module (all safety checks delegated to objc2)
- Memory lifecycle: objc2 handles reference counting and retain/release
- Private API exposure: limited to required classes/methods only

**Private ANE Framework Usage:**

Why use private APIs instead of public Core ML?
- Public Core ML has no backward pass support (training API was rejected)
- Private ANE APIs are proven: used in ~/dev/ANE/training/ (production code)
- Stability: Apple hardware supports these APIs consistently (M1-M4 all work)
- Mitigation: periodically test on new OS versions; fallback to CPU mode if framework unavailable

**Constraints on objc2 Usage:**

- Only call methods that exist in ~/dev/ANE/training/ (proven patterns)
- Pin minimum OS to macOS 15+ (Sequoia) when private framework stabilized
- Version check at init time: return error if private APIs unavailable
- Comprehensive unit tests catch API breakage immediately

---

## Error Handling

```rust
pub enum ANEError {
    FrameworkNotFound,
    CompileFailed(String),
    EvalFailed(String),
    IOSurfaceError(String),
    InvalidShape { expected: String, got: String },
    WeightBlobError(String),
}

impl From<ANEError> for rustane::Error {
    fn from(e: ANEError) -> Self {
        rustane::Error::Custom(format!("{:?}", e))
    }
}
```

All ANE operations return `Result<T, ANEError>` which propagates to trainer.

---

## Testing Strategy

### Unit Tests

**Weight blob builders:**
- Correct blob format (header + data)
- Round-trip: build blob → parse → verify binary layout
- Quantization correctness

**MIL generation:**
- Valid MIL syntax (parses with ANE compiler)
- Correct dimensions in tensor ops
- Weight offset calculations

**IOSurface:**
- Create/destroy lifecycle
- Write/read round-trip data
- Thread-safe locking

**Backward pass:**
- Numerical gradient checking (finite differences vs. analytic)
- Compare against simple reference implementations
- Dimension validation

### Integration Tests

**Small kernel:**
- Compile simple linear layer MIL → ANEKernel
- Forward pass on ANE
- Verify output shapes

**Single transformer block:**
- Forward + backward on single layer
- Gradient shapes match parameter shapes
- Loss decreases after optimizer step

**Full transformer:**
- Forward + backward on 2-layer transformer
- Training loop: loss decreases over steps
- Gradient norms stay finite

### Example

```rust
// examples/train_transformer_ane.rs
fn main() -> Result<()> {
    // Load or initialize transformer
    let mut model = TransformerANE::new(config)?;

    // Create trainer
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamOptimizer::new(0.001))
        .with_scheduler(WarmupCosineScheduler::new(0.001, 500, 10000))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    // Training loop
    for step in 0..1000 {
        let batch = dataloader.next_batch(32, 512)?;
        let metrics = trainer.train_step(&batch)?;

        if step % 100 == 0 {
            println!("Step {}: loss={:.4}, lr={:.6}",
                step, metrics.loss, metrics.learning_rate);
        }
    }

    Ok(())
}
```

---

## Dependencies

**New dependencies:**

```toml
objc2 = "0.5"
objc2-foundation = "0.5"
half = "2.4"     # FP16 conversion
```

**Existing dependencies:**

- `rustane` core types (Batch, Model, ANETensor, etc.)
- `rand` (if needed for initialization)

---

## Performance Expectations

Based on `~/dev/ANE/training/` benchmarks:

| Model | Layers | Dim | Params | ms/step | TFLOPS |
|-------|--------|-----|--------|---------|--------|
| Stories110M | 12 | 768 | 109M | ~115 | 0.87 |
| Qwen3-0.6B | 28 | 1024 | 596M | ~412 | 1.15 |
| Target: 7.2M transformer (from TRANSFORMER_PLAN.md) | 6 | 256 | 7.2M | ~10-15 | — |

For small models (7.2M params), expected: **5-10 ms/step on M4 Max**, dominated by compile overhead on first step.

---

## Success Criteria

- [ ] All ANE module unit tests pass (weight blobs, IOSurface, MIL generation)
- [ ] Backward pass numerical gradient checks pass (< 1e-4 error)
- [ ] Full training loop completes without errors
- [ ] Loss decreases over 100 training steps
- [ ] All 249+ existing rustane tests still pass (no regressions)
- [ ] Example runs and produces reasonable metrics
- [ ] Gradient norms stay finite (no NaN/Inf)
- [ ] Performance: < 50 ms/step for 7.2M param model on M4 Max

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Private ANE API changes in future OS | Use stable private APIs from existing implementation; monitor Apple updates |
| objc2 safety | Comprehensive unit tests; no unsafe blocks in Rust layer; objc2 handles memory safety |
| Activation memory overhead | Profile with dashboard; implement streaming cache if needed |
| MIL compilation complexity | Use proven MIL patterns from existing implementation; thorough generation tests |
| Numerical stability | Match C implementation exactly; add stability tests (max subtraction in softmax, etc.) |

---

## Timeline

**Phase 1: ANE bindings (Week 1)**
- ANE runtime (objc2 wrapper)
- ANEKernel + IOSurface
- Weight blob builders
- Comprehensive tests

**Phase 2: Backward implementation (Week 2)**
- Backward pass for all layers
- Numerical gradient checks
- Integration with Model trait

**Phase 3: Integration & tuning (Week 2-3)**
- TransformerANE implementation
- Full training loop
- Example + documentation
- Performance profiling

---

## References

- `~/dev/ANE/training/` - Existing production ANE training implementation
- `~/dev/ANE/bridge/ane_bridge.h` - C bridge interface (reference)
- TRANSFORMER_PLAN.md - Target transformer architecture
- Phase 2 Week 3 - Existing trainer infrastructure

---

**Ready to review?**
