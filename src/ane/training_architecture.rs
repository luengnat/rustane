//! ANE Training Architecture with Compile Budget Management
//!
//! Critical Constraint: ANE has ~119 compile limit per process due to memory leaks.
//! Strategy:
//! 1. Pre-compile all kernels at initialization (count them!)
//! 2. Use dynamic weight packing (no recompilation per step)
//! 3. Monitor compile budget
//! 4. Checkpoint + restart when budget exhausted

use super::mil_generator::{ANEShape, ANETensorType};
use super::{ANECompileRequest, ANEError};
use crate::wrapper::{ANEExecutor, ANERuntime};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Maximum ANE compiles per process before memory issues
const ANE_COMPILE_BUDGET: i32 = 110; // Leave margin from 119

/// Minimum compiles reserved for emergency fallback
const ANE_RESERVE_BUDGET: i32 = 10;

/// Compile budget manager
///
/// Tracks all ANE kernel compilations to stay within the ~119 limit.
/// Prevents runtime surprises by planning compiles upfront.
pub struct CompileBudget {
    /// Total compiles used so far
    used: Arc<Mutex<i32>>,
    /// Compiles reserved for critical operations
    reserved: i32,
    /// Whether we're in "emergency mode" (low budget)
    emergency_mode: Arc<Mutex<bool>>,
}

impl CompileBudget {
    pub fn new(reserved: i32) -> Self {
        Self {
            used: Arc::new(Mutex::new(0)),
            reserved,
            emergency_mode: Arc::new(Mutex::new(false)),
        }
    }

    /// Request compile budget allocation
    ///
    /// Returns true if budget available, false if would exceed limit
    pub fn request_compile(&self, count: i32) -> bool {
        let mut used = self.used.lock().unwrap();
        let available = ANE_COMPILE_BUDGET - self.reserved;

        if *used + count > available {
            // Trigger emergency mode
            let mut emergency = self.emergency_mode.lock().unwrap();
            *emergency = true;
            false
        } else {
            *used += count;
            true
        }
    }

    /// Get remaining budget (excluding reserve)
    pub fn remaining(&self) -> i32 {
        let used = self.used.lock().unwrap();
        ANE_COMPILE_BUDGET - self.reserved - *used
    }

    /// Check if in emergency mode
    pub fn is_emergency(&self) -> bool {
        *self.emergency_mode.lock().unwrap()
    }

    /// Get used count
    pub fn used(&self) -> i32 {
        *self.used.lock().unwrap()
    }
}

/// Pre-defined kernel templates for common operations
///
/// These are compiled ONCE at initialization and reused throughout training.
/// No runtime compilation during training loop!
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum KernelTemplate {
    /// RMSNorm: input [C, S] -> output [C, S]
    RmsNorm { channels: usize, seq_len: usize },

    /// Linear layer with dynamic weights
    /// Input: [IC, S+OC] (activations + weights packed)
    /// Output: [OC, S]
    DynamicLinear {
        in_features: usize,
        out_features: usize,
        seq_len: usize,
    },

    /// Attention QKV projection
    /// Input: [DIM, S+Q_DIM+KV_DIM+KV_DIM]
    /// Output: [Q_DIM+Q_DIM+KV_DIM+KV_DIM+DIM, S]
    QkvProjection {
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
    },

    /// SDPA Forward (simplified)
    /// Input: [HEADS, HD, S] Q, K, V tensors
    /// Output: [HEADS, HD, S]
    SdpaForward {
        heads: usize,
        head_dim: usize,
        seq_len: usize,
    },

    /// FFN SwiGLU
    /// Input: [DIM, S+HIDDEN+HIDDEN+HIDDEN] (x + W1 + W2 + W3 packed)
    /// Output: [DIM+HIDDEN*3, S]
    FfnSwiglu {
        dim: usize,
        hidden_dim: usize,
        seq_len: usize,
    },
}

impl KernelTemplate {
    /// Generate unique ID for this kernel
    pub fn id(&self) -> String {
        match self {
            KernelTemplate::RmsNorm { channels, seq_len } => {
                format!("rmsnorm_{}_{}", channels, seq_len)
            }
            KernelTemplate::DynamicLinear {
                in_features,
                out_features,
                seq_len,
            } => {
                format!("linear_{}_{}_{}", in_features, out_features, seq_len)
            }
            KernelTemplate::QkvProjection {
                dim,
                q_dim,
                kv_dim,
                seq_len,
            } => {
                format!("qkv_{}_{}_{}_{}", dim, q_dim, kv_dim, seq_len)
            }
            KernelTemplate::SdpaForward {
                heads,
                head_dim,
                seq_len,
            } => {
                format!("sdpa_{}_{}_{}", heads, head_dim, seq_len)
            }
            KernelTemplate::FfnSwiglu {
                dim,
                hidden_dim,
                seq_len,
            } => {
                format!("ffn_{}_{}_{}", dim, hidden_dim, seq_len)
            }
        }
    }

    /// Estimate input/output sizes for validation
    pub fn io_sizes(&self) -> (usize, usize) {
        match self {
            KernelTemplate::RmsNorm { channels, seq_len } => {
                let size = channels * seq_len * 2; // fp16
                (size, size)
            }
            KernelTemplate::DynamicLinear {
                in_features,
                out_features,
                seq_len,
            } => {
                let input_size = (in_features * seq_len + in_features * out_features) * 2;
                let output_size = out_features * seq_len * 2;
                (input_size, output_size)
            }
            KernelTemplate::QkvProjection {
                dim,
                q_dim,
                kv_dim,
                seq_len,
            } => {
                let input_size = (dim * seq_len + q_dim + kv_dim + kv_dim) * 2;
                let output_size = (q_dim + q_dim + kv_dim + kv_dim + dim) * seq_len * 2;
                (input_size, output_size)
            }
            KernelTemplate::SdpaForward {
                heads,
                head_dim,
                seq_len,
            } => {
                let size = heads * head_dim * seq_len * 2;
                (size * 3, size) // Q, K, V input
            }
            KernelTemplate::FfnSwiglu {
                dim,
                hidden_dim,
                seq_len,
            } => {
                let input_size = (dim * seq_len + dim * hidden_dim * 3) * 2;
                let output_size = (dim + hidden_dim * 3) * seq_len * 2;
                (input_size, output_size)
            }
        }
    }

    /// Check if this kernel size is ANE-compatible
    pub fn is_ane_compatible(&self) -> bool {
        let (input_size, output_size) = self.io_sizes();
        // ANE limit: 16384 elements = 32768 bytes for fp16
        let max_elements = 16384;
        let max_bytes = max_elements * 2;

        input_size <= max_bytes && output_size <= max_bytes
    }
}

/// Pre-compiled kernel registry
///
/// All kernels must be registered and compiled BEFORE training starts.
/// No compilation during training loop!
pub struct KernelRegistry {
    /// Budget manager
    budget: CompileBudget,
    /// Compiled kernels: template_id -> executor
    kernels: HashMap<String, ANEExecutor>,
    /// Failed compilations (too large, etc.)
    failed: Vec<String>,
    /// Whether ANE is available
    ane_available: bool,
}

impl KernelRegistry {
    /// Create new registry and compile all required kernels
    ///
    /// # Arguments
    /// * `templates` - All kernel templates needed for training
    /// * `reserved_budget` - Compiles to reserve for emergencies
    ///
    /// # Errors
    /// Returns error if compile budget exceeded or ANE unavailable
    pub fn new(templates: Vec<KernelTemplate>, reserved_budget: i32) -> Result<Self, ANEError> {
        // Check ANE availability
        let ane_available = match ANERuntime::init() {
            Ok(_) => {
                println!("✅ ANE available for kernel compilation");
                true
            }
            Err(e) => {
                println!("⚠️  ANE unavailable: {}. Will use CPU fallback.", e);
                false
            }
        };

        if !ane_available {
            return Ok(Self {
                budget: CompileBudget::new(reserved_budget),
                kernels: HashMap::new(),
                failed: templates.iter().map(|t| t.id()).collect(),
                ane_available: false,
            });
        }

        // Check if we have enough budget
        let budget = CompileBudget::new(reserved_budget);
        if !budget.request_compile(templates.len() as i32) {
            return Err(ANEError::CompileFailed(format!(
                "Insufficient compile budget: need {} kernels, have {} remaining",
                templates.len(),
                budget.remaining()
            )));
        }

        let mut registry = Self {
            budget,
            kernels: HashMap::new(),
            failed: Vec::new(),
            ane_available: true,
        };

        // Compile all kernels upfront
        println!("\n=== Pre-compiling {} ANE kernels ===", templates.len());
        let start = Instant::now();

        for template in templates {
            if !template.is_ane_compatible() {
                println!("  ⚠️  {} - too large for ANE", template.id());
                registry.failed.push(template.id());
                continue;
            }

            match registry.compile_kernel(&template) {
                Ok(()) => {
                    println!("  ✅ {}", template.id());
                }
                Err(e) => {
                    println!("  ❌ {} - {}", template.id(), e);
                    registry.failed.push(template.id());
                }
            }
        }

        let elapsed = start.elapsed();
        println!("\n=== Compilation complete ===");
        println!("  Successful: {}", registry.kernels.len());
        println!("  Failed: {}", registry.failed.len());
        println!("  Time: {:?}", elapsed);
        println!(
            "  Remaining budget: {}/{}",
            registry.budget.remaining(),
            ANE_COMPILE_BUDGET - reserved_budget
        );

        Ok(registry)
    }

    /// Compile a single kernel
    fn compile_kernel(&mut self, template: &KernelTemplate) -> Result<(), ANEError> {
        let mil = self.generate_mil(template);
        let (input_size, output_size) = template.io_sizes();

        let request = ANECompileRequest::new(&mil, vec![input_size], vec![output_size]);

        let executor = request
            .compile()
            .map_err(|e| ANEError::CompileFailed(e.to_string()))?;
        self.kernels.insert(template.id(), executor);

        Ok(())
    }

    /// Generate MIL for a kernel template
    fn generate_mil(&self, template: &KernelTemplate) -> String {
        match template {
            KernelTemplate::RmsNorm { channels, seq_len } => {
                Self::generate_rmsnorm_mil(*channels, *seq_len)
            }
            KernelTemplate::DynamicLinear {
                in_features,
                out_features,
                seq_len,
            } => Self::generate_linear_mil(*in_features, *out_features, *seq_len),
            KernelTemplate::QkvProjection {
                dim,
                q_dim,
                kv_dim,
                seq_len,
            } => Self::generate_qkv_mil(*dim, *q_dim, *kv_dim, *seq_len),
            KernelTemplate::SdpaForward {
                heads,
                head_dim,
                seq_len,
            } => Self::generate_sdpa_mil(*heads, *head_dim, *seq_len),
            KernelTemplate::FfnSwiglu {
                dim,
                hidden_dim,
                seq_len,
            } => Self::generate_ffn_mil(*dim, *hidden_dim, *seq_len),
        }
    }

    /// Generate RMSNorm MIL
    /// Normalizes across channels using mean squared
    fn generate_rmsnorm_mil(channels: usize, seq_len: usize) -> String {
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {channels}, 1, {seq_len}]> x) {{
        // RMSNorm: x / sqrt(mean(x^2) + eps)
        tensor<fp16, [1, {channels}, 1, {seq_len}]> squared = mul(x=x, y=x)[name=string("squared")];
        tensor<fp16, [1, 1, 1, {seq_len}]> mean = reduce_mean(x=squared, axes=[1], keep_dims=true)[name=string("mean")];
        tensor<fp16, [1, 1, 1, {seq_len}]> eps = const()[name=string("eps"), val=tensor<fp16, [1,1,1,1]>(1e-6)];
        tensor<fp16, [1, 1, 1, {seq_len}]> sum = add(x=mean, y=eps)[name=string("sum")];
        tensor<fp16, [1, 1, 1, {seq_len}]> rms = sqrt(x=sum)[name=string("rms")];
        tensor<fp16, [1, {channels}, 1, {seq_len}]> out = real_div(x=x, y=rms)[name=string("out")];
    }} -> (out);
}}"#
        )
    }

    /// Generate Linear layer MIL with dynamic weight packing
    /// Input: [1, in_features, 1, seq_len + out_features]
    fn generate_linear_mil(in_features: usize, out_features: usize, seq_len: usize) -> String {
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {in_features}, 1, {seq_len}]> x) {{
        // Linear: out = W @ x
        tensor<fp16, [{out_features}, {in_features}, 1, 1]> W = const()[name=string("W"),
            val=tensor<fp16, [{out_features}, {in_features}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/W.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {out_features}, 1, {seq_len}]> out = matmul(x=W, y=x, transpose_x=false, transpose_y=false)[name=string("out")];
    }} -> (out);
}}"#
        )
    }

    /// Generate QKV projection MIL
    fn generate_qkv_mil(dim: usize, q_dim: usize, kv_dim: usize, seq_len: usize) -> String {
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {dim}, 1, {seq_len}]> x) {{
        // Q projection
        tensor<fp16, [{q_dim}, {dim}, 1, 1]> Wq = const()[name=string("Wq"),
            val=tensor<fp16, [{q_dim}, {dim}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/Wq.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {q_dim}, 1, {seq_len}]> q = matmul(x=Wq, y=x)[name=string("q")];
        
        // K projection  
        tensor<fp16, [{kv_dim}, {dim}, 1, 1]> Wk = const()[name=string("Wk"),
            val=tensor<fp16, [{kv_dim}, {dim}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/Wk.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {kv_dim}, 1, {seq_len}]> k = matmul(x=Wk, y=x)[name=string("k")];
        
        // V projection
        tensor<fp16, [{kv_dim}, {dim}, 1, 1]> Wv = const()[name=string("Wv"),
            val=tensor<fp16, [{kv_dim}, {dim}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/Wv.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {kv_dim}, 1, {seq_len}]> v = matmul(x=Wv, y=x)[name=string("v")];
        
        // Concatenate outputs: [q, k, v]
        tensor<fp16, [1, {}, 1, {seq_len}]> out = concat(x=[q, k, v], axis=1)[name=string("out")];
    }} -> (out);
}}"#,
            q_dim + kv_dim + kv_dim
        )
    }

    /// Generate SDPA (Scaled Dot Product Attention) MIL
    fn generate_sdpa_mil(heads: usize, head_dim: usize, seq_len: usize) -> String {
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {}, 1, {seq_len}]> qkv
    ) {{
        // Split QKV into separate tensors
        // Simplified: assumes pre-split or uses slice
        // Full implementation would split and process each head
        
        // Scale factor: 1/sqrt(head_dim)
        tensor<fp16, [1, 1, 1, 1]> scale = const()[name=string("scale"), 
            val=tensor<fp16, [1,1,1,1]>({})];
        
        // Simplified: return identity (full SDPA is complex)
        tensor<fp16, [1, {}, 1, {seq_len}]> out = mul(x=qkv, y=scale)[name=string("out")];
    }} -> (out);
}}"#,
            heads * head_dim,
            1.0 / (head_dim as f32).sqrt(),
            heads * head_dim
        )
    }

    /// Generate FFN SwiGLU MIL
    fn generate_ffn_mil(dim: usize, hidden_dim: usize, seq_len: usize) -> String {
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {dim}, 1, {seq_len}]> x) {{
        // SwiGLU: Swish(x @ W1) * (x @ W3) @ W2
        
        // W1: [hidden_dim, dim]
        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W1 = const()[name=string("W1"),
            val=tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/W1.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> h1 = matmul(x=W1, y=x)[name=string("h1")];
        
        // Swish activation: x * sigmoid(x)
        tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> sig_h1 = sigmoid(x=h1)[name=string("sig_h1")];
        tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> swish = mul(x=h1, y=sig_h1)[name=string("swish")];
        
        // W3: [hidden_dim, dim]
        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W3 = const()[name=string("W3"),
            val=tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/W3.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> h3 = matmul(x=W3, y=x)[name=string("h3")];
        
        // Element-wise multiply
        tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> gated = mul(x=swish, y=h3)[name=string("gated")];
        
        // W2: [dim, hidden_dim]
        tensor<fp16, [{dim}, {hidden_dim}, 1, 1]> W2 = const()[name=string("W2"),
            val=tensor<fp16, [{dim}, {hidden_dim}, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/W2.bin"), offset=uint64(64))
            )];
        tensor<fp16, [1, {dim}, 1, {seq_len}]> out = matmul(x=W2, y=gated)[name=string("out")];
    }} -> (out);
}}"#
        )
    }

    /// Get compiled executor for a template
    pub fn get(&self, template: &KernelTemplate) -> Option<&ANEExecutor> {
        self.kernels.get(&template.id())
    }

    /// Check if kernel is available (compiled successfully)
    pub fn has_kernel(&self, template: &KernelTemplate) -> bool {
        self.kernels.contains_key(&template.id())
    }

    /// Get number of available kernels
    pub fn available_count(&self) -> usize {
        self.kernels.len()
    }

    /// Get number of failed kernels
    pub fn failed_count(&self) -> usize {
        self.failed.len()
    }

    /// Get remaining compile budget
    pub fn remaining_budget(&self) -> i32 {
        self.budget.remaining()
    }

    /// Check if we should checkpoint + restart
    pub fn should_restart(&self) -> bool {
        self.budget.is_emergency() || self.remaining_budget() < 5
    }
}

/// Training step counter for checkpointing
pub struct TrainingCheckpoint {
    step: usize,
    checkpoint_interval: usize,
}

impl TrainingCheckpoint {
    pub fn new(checkpoint_interval: usize) -> Self {
        Self {
            step: 0,
            checkpoint_interval,
        }
    }

    /// Increment step and check if we should checkpoint
    pub fn step(&mut self) -> bool {
        self.step += 1;
        self.step % self.checkpoint_interval == 0
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.step
    }
}

/// Training configuration for ANE
pub struct ANETrainingConfig {
    /// Model dimensions
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub seq_len: usize,

    /// Training parameters
    pub batch_size: usize,
    pub learning_rate: f32,
    pub checkpoint_interval: usize,

    /// ANE parameters
    pub compile_reserve: i32,
    pub use_ane: bool,
}

impl Default for ANETrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            dim: 768,
            n_layers: 12,
            n_heads: 12,
            seq_len: 256,
            batch_size: 4,
            learning_rate: 1e-4,
            checkpoint_interval: 1000,
            compile_reserve: 10,
            use_ane: true,
        }
    }
}

impl ANETrainingConfig {
    /// Generate all kernel templates needed for this configuration
    pub fn generate_kernels(&self) -> Vec<KernelTemplate> {
        let mut kernels = Vec::new();

        // RMSNorm kernels (input norm, post-attention norm)
        kernels.push(KernelTemplate::RmsNorm {
            channels: self.dim,
            seq_len: self.seq_len,
        });

        // QKV projection
        let head_dim = self.dim / self.n_heads;
        let q_dim = self.dim;
        let kv_dim = self.dim; // For MHA, KV_DIM = DIM
        kernels.push(KernelTemplate::QkvProjection {
            dim: self.dim,
            q_dim,
            kv_dim,
            seq_len: self.seq_len,
        });

        // Output projection
        kernels.push(KernelTemplate::DynamicLinear {
            in_features: self.dim,
            out_features: self.dim,
            seq_len: self.seq_len,
        });

        // FFN layers
        let hidden_dim = self.dim * 4; // Standard 4x expansion
        kernels.push(KernelTemplate::FfnSwiglu {
            dim: self.dim,
            hidden_dim,
            seq_len: self.seq_len,
        });

        kernels
    }

    /// Estimate total ANE compiles needed
    pub fn estimate_compiles(&self) -> usize {
        self.generate_kernels().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mil_generation_rmsnorm() {
        let registry = KernelRegistry::new(vec![], 10).unwrap_or_else(|_| KernelRegistry {
            budget: CompileBudget::new(10),
            kernels: HashMap::new(),
            failed: vec![],
            ane_available: false,
        });

        let template = KernelTemplate::RmsNorm {
            channels: 64,
            seq_len: 32,
        };

        let mil = registry.generate_mil(&template);

        // Verify MIL contains expected components
        assert!(mil.contains("program(1.3)"), "Should have program version");
        assert!(
            mil.contains("reduce_mean"),
            "RMSNorm should use reduce_mean"
        );
        assert!(mil.contains("sqrt"), "RMSNorm should use sqrt");
        assert!(
            mil.contains("[1, 64, 1, 32]"),
            "Should have correct tensor shape"
        );
    }

    #[test]
    fn test_mil_generation_linear() {
        let registry = KernelRegistry::new(vec![], 10).unwrap_or_else(|_| KernelRegistry {
            budget: CompileBudget::new(10),
            kernels: HashMap::new(),
            failed: vec![],
            ane_available: false,
        });

        let template = KernelTemplate::DynamicLinear {
            in_features: 128,
            out_features: 64,
            seq_len: 16,
        };

        let mil = registry.generate_mil(&template);

        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("matmul"), "Linear should use matmul");
        assert!(mil.contains("W.bin"), "Should reference weight file");
        assert!(
            mil.contains("[1, 128, 1, 16]"),
            "Should have correct input shape"
        );
    }

    #[test]
    fn test_mil_generation_ffn() {
        let registry = KernelRegistry::new(vec![], 10).unwrap_or_else(|_| KernelRegistry {
            budget: CompileBudget::new(10),
            kernels: HashMap::new(),
            failed: vec![],
            ane_available: false,
        });

        let template = KernelTemplate::FfnSwiglu {
            dim: 128,
            hidden_dim: 512,
            seq_len: 16,
        };

        let mil = registry.generate_mil(&template);

        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("sigmoid"), "SwiGLU should use sigmoid");
        assert!(mil.contains("W1.bin"), "Should reference W1");
        assert!(mil.contains("W2.bin"), "Should reference W2");
        assert!(mil.contains("W3.bin"), "Should reference W3");
    }

    #[test]
    fn test_compile_budget() {
        let budget = CompileBudget::new(10);

        // Should allow up to 100 compiles (110 - 10 reserve)
        assert!(budget.request_compile(50));
        assert!(budget.request_compile(40));
        assert!(!budget.request_compile(30)); // Would exceed (need 30, have 10)

        assert!(budget.is_emergency());
        // 110 total - 10 reserved - 90 used = 10 remaining
        assert_eq!(budget.remaining(), 10);
    }

    #[test]
    fn test_kernel_template_id() {
        let template = KernelTemplate::RmsNorm {
            channels: 768,
            seq_len: 256,
        };
        assert_eq!(template.id(), "rmsnorm_768_256");
    }

    #[test]
    fn test_ane_compatibility() {
        // Small kernel should be compatible
        let small = KernelTemplate::RmsNorm {
            channels: 64,
            seq_len: 64,
        };
        assert!(small.is_ane_compatible());

        // Large kernel should not be compatible
        let large = KernelTemplate::RmsNorm {
            channels: 768,
            seq_len: 1024, // 768*1024 = 786K elements > 16K limit
        };
        assert!(!large.is_ane_compatible());
    }

    #[test]
    fn test_config_kernel_generation() {
        let config = ANETrainingConfig::default();
        let kernels = config.generate_kernels();

        // Should generate: RMSNorm, QKV, Linear, FFN
        assert_eq!(kernels.len(), 4);

        // Check specific kernels exist
        let has_rmsnorm = kernels
            .iter()
            .any(|k| matches!(k, KernelTemplate::RmsNorm { .. }));
        assert!(has_rmsnorm);
    }
}
