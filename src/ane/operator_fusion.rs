//! ANE Operator Fusion
//!
//! Fuse common operation patterns into single kernels to:
//! 1. Reduce compilation overhead (fewer kernels to compile)
//! 2. Reduce data transfer (intermediate results stay on ANE)
//! 3. Improve performance (eliminate memory round-trips)
//!
//! # Fused Patterns
//!
//! - **RMSNorm + Linear** - Layer input normalization + projection
//! - **Linear + Activation** - FFN up/down projections with SiLU
//! - **QKV Projection** - Single kernel for all three projections
//! - **Attention Output** - Project + residual add

use std::collections::HashMap;

/// Fused kernel types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusedKernelType {
    /// RMSNorm + Linear projection
    RmsnormLinear {
        channels: usize,
        seq_len: usize,
        out_features: usize,
    },
    /// Linear + Activation (SiLU/ReLU)
    LinearActivation {
        in_features: usize,
        out_features: usize,
        seq_len: usize,
        activation: ActivationType,
    },
    /// QKV projection (single kernel for Q, K, V)
    QkvProjection {
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
    },
    /// Linear + Residual Add
    LinearResidual {
        in_features: usize,
        out_features: usize,
        seq_len: usize,
    },
    /// Gated Linear Unit (SwiGLU)
    Swiglu {
        dim: usize,
        hidden_dim: usize,
        seq_len: usize,
    },
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationType {
    /// Rectified Linear Unit
    Relu,
    /// Sigmoid Linear Unit (Swish)
    Silu,
    /// Gaussian Error Linear Unit
    Gelu,
}

impl ActivationType {
    /// Get MIL code for activation
    pub fn mil_code(&self, input: &str, output: &str) -> String {
        match self {
            ActivationType::Relu => {
                format!("var {} = mb.relu(x=\"{}\", name=\"relu\");", output, input)
            }
            ActivationType::Silu => {
                // SiLU = x * sigmoid(x)
                // ANE doesn't have native SiLU, use approximation
                format!(
                    "var {sig} = mb.sigmoid(x=\"{inp}\", name=\"silu_sig\");
        var {out} = mb.mul(x=\"{inp}\", y={sig}, name=\"silu\");",
                    inp = input,
                    sig = format!("{}_sig", output),
                    out = output
                )
            }
            ActivationType::Gelu => {
                // GELU uses tanh approximation
                format!(
                    "// GELU approximation for {} -> {}
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        var {} = gelu_approx(x=\"{}\", name=\"gelu\");",
                    input, output, output, input
                )
            }
        }
    }
}

impl FusedKernelType {
    /// Generate unique ID for this fused kernel
    pub fn id(&self) -> String {
        match self {
            FusedKernelType::RmsnormLinear {
                channels,
                seq_len,
                out_features,
            } => format!(
                "fused_rmsnorm_linear_{}_{}_{}",
                channels, seq_len, out_features
            ),

            FusedKernelType::LinearActivation {
                in_features,
                out_features,
                seq_len,
                activation,
            } => format!(
                "fused_linear_{:?}_{}_{}_{}",
                activation, in_features, out_features, seq_len
            ),

            FusedKernelType::QkvProjection {
                dim,
                q_dim,
                kv_dim,
                seq_len,
            } => format!("fused_qkv_{}_{}_{}_{}", dim, q_dim, kv_dim, seq_len),

            FusedKernelType::LinearResidual {
                in_features,
                out_features,
                seq_len,
            } => format!(
                "fused_linear_residual_{}_{}_{}",
                in_features, out_features, seq_len
            ),

            FusedKernelType::Swiglu {
                dim,
                hidden_dim,
                seq_len,
            } => format!("fused_swiglu_{}_{}_{}", dim, hidden_dim, seq_len),
        }
    }

    /// Generate MIL code for this fused kernel
    pub fn generate_mil(&self) -> String {
        match self {
            FusedKernelType::RmsnormLinear {
                channels,
                seq_len,
                out_features,
            } => self.mil_rmsnorm_linear(*channels, *seq_len, *out_features),

            FusedKernelType::LinearActivation {
                in_features,
                out_features,
                seq_len,
                activation,
            } => self.mil_linear_activation(*in_features, *out_features, *seq_len, *activation),

            FusedKernelType::QkvProjection {
                dim,
                q_dim,
                kv_dim,
                seq_len,
            } => self.mil_qkv_projection(*dim, *q_dim, *kv_dim, *seq_len),

            FusedKernelType::LinearResidual {
                in_features,
                out_features,
                seq_len,
            } => self.mil_linear_residual(*in_features, *out_features, *seq_len),

            FusedKernelType::Swiglu {
                dim,
                hidden_dim,
                seq_len,
            } => self.mil_swiglu(*dim, *hidden_dim, *seq_len),
        }
    }

    /// RMSNorm + Linear fused kernel
    /// Input: [1, C, 1, S] (activation)
    /// Output: [1, Out, 1, S]
    fn mil_rmsnorm_linear(&self, channels: usize, seq_len: usize, out_features: usize) -> String {
        let invd = 1.0 / (channels as f32);

        format!(
            r#"program(1.3)
[buildInfo=dict<string,string>("target_os"="ios","target_version"="18")] {{
    func main<ios18>(tensor<fp16, [1,{c},1,{s}]> x, tensor<fp16, [1,{c},1,1]> rms_w, tensor<fp16, [{c},{o}]> linear_w) {{
        // RMSNorm: compute RMS
        tensor<fp16, [1,{c},1,{s}]> sq = mb.mul(x=x, y=x, name="sq");
        tensor<int32, [1]> rax = const<tensor<int32, [1]>>(val=[1]);
        bool kd = const<bool>(val=true);
        tensor<fp16, [1,1,1,{s}]> ss = mb.reduce_sum(x=sq, axes=rax, keep_dims=kd, name="ss");
        fp16 invd = const<fp16>(val={invd});
        tensor<fp16, [1,1,1,{s}]> ss2 = mb.mul(x=ss, y=invd, name="ss2");
        fp16 eps = const<fp16>(val=0.00001);
        tensor<fp16, [1,1,1,{s}]> ss3 = mb.add(x=ss2, y=eps, name="ss3");
        fp16 nhalf = const<fp16>(val=-0.5);
        tensor<fp16, [1,1,1,{s}]> rrms = mb.pow(x=ss3, y=nhalf, name="rrms");
        tensor<fp16, [1,{c},1,{s}]> xr = mb.mul(x=x, y=rrms, name="xr");
        tensor<fp16, [1,{c},1,{s}]> normed = mb.mul(x=xr, y=rms_w, name="normed");

        // Linear projection via matmul
        tensor<fp16, [1,{c},1,{s}]> normed_t = mb.transpose(x=normed, perm=[0,2,3,1], name="normed_t");
        tensor<fp16, [1,{s},{c}]> linear_in = mb.reshape(x=normed_t, shape=[1,{s},{c}], name="linear_in");
        tensor<fp16, [1,{s},{o}]> y = mb.matmul(x=linear_in, y=linear_w, name="linear_out");
        tensor<fp16, [1,{o},1,{s}]> out = mb.transpose(x=y, perm=[0,2,1], name="out");
    }} -> (out);
}}"#,
            c = channels,
            s = seq_len,
            o = out_features,
            invd = invd
        )
    }

    /// Linear + Activation fused kernel
    fn mil_linear_activation(
        &self,
        in_features: usize,
        out_features: usize,
        seq_len: usize,
        activation: ActivationType,
    ) -> String {
        format!(
            r#"program(1.3)
[buildInfo=dict<string,string>("target_os"="ios","target_version"="18")] {{
    func main<ios18>(tensor<fp16, [1,{ic},1,{s}]> x, tensor<fp16, [{ic},{oc}]> w) {{
        // Reshape input for matmul
        tensor<fp16, [1,{s},{ic}]> x_reshaped = mb.reshape(x=x, shape=[1,{s},{ic}], name="x_reshaped");

        // Matmul: [1,S,IC] @ [IC,OC] = [1,S,OC]
        tensor<fp16, [1,{s},{oc}]> y = mb.matmul(x=x_reshaped, y=w, name="linear");

        // Activation
        {activation_mil}

        // Reshape back to [1,OC,1,S]
        tensor<fp16, [1,{oc},1,{s}]> out = mb.transpose(x=activated, perm=[0,2,1], name="out");
    }} -> (out);
}}"#,
            ic = in_features,
            oc = out_features,
            s = seq_len,
            activation_mil = activation.mil_code("y", "activated")
        )
    }

    /// QKV projection fused kernel
    fn mil_qkv_projection(
        &self,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
    ) -> String {
        let total_out = q_dim * 2 + kv_dim * 2;

        format!(
            r#"program(1.3)
[buildInfo=dict<string,string>("target_os"="ios","target_version"="18")] {{
    func main<ios18>(tensor<fp16, [1,{dim},1,{s}]> x, tensor<fp16, [{dim},{qkv_out}]> w_qkv) {{
        // Reshape input
        tensor<fp16, [1,{s},{dim}]> x_reshaped = mb.reshape(x=x, shape=[1,{s},{dim}], name="x_reshaped");

        // Single matmul for Q, K, V
        tensor<fp16, [1,{s},{qkv_out}]> qkv = mb.matmul(x=x_reshaped, y=w_qkv, name="qkv_proj");

        // Split into Q, K, V (simplified - actual split depends on layout)
        tensor<fp16, [1,{s},{q_dim}]> q = mb.slice_by_index(x=qkv, begin=[0,0,0], end=[1,{s},{q_dim}], name="q");
        tensor<fp16, [1,{s},{kv_dim}]> k = mb.slice_by_index(x=qkv, begin=[0,0,{q_dim}], end=[1,{s},{q_dim}+{kv_dim}], name="k");
        tensor<fp16, [1,{s},{kv_dim}]> v = mb.slice_by_index(x=qkv, begin=[0,0,{q_dim}+{kv_dim}], end=[1,{s},{qkv_out}], name="v");

        // Concatenate outputs for return
        tensor<fp16, [1,{s},{qkv_out}]> out_reshaped = mb.concat(values=[q, k, v], axis=2, name="out_concat");
        tensor<fp16, [1,{qkv_out},1,{s}]> out = mb.transpose(x=out_reshaped, perm=[0,2,1], name="out");
    }} -> (out);
}}"#,
            dim = dim,
            s = seq_len,
            q_dim = q_dim,
            kv_dim = kv_dim,
            qkv_out = total_out
        )
    }

    /// Linear + Residual fused kernel
    fn mil_linear_residual(
        &self,
        in_features: usize,
        out_features: usize,
        seq_len: usize,
    ) -> String {
        format!(
            r#"program(1.3)
[buildInfo=dict<string,string>("target_os"="ios","target_version"="18")] {{
    func main<ios18>(tensor<fp16, [1,{ic},1,{s}]> x, tensor<fp16, [{ic},{oc}]> w, tensor<fp16, [1,{oc},1,{s}]> residual) {{
        // Reshape for matmul
        tensor<fp16, [1,{s},{ic}]> x_reshaped = mb.reshape(x=x, shape=[1,{s},{ic}], name="x_reshaped");

        // Linear
        tensor<fp16, [1,{s},{oc}]> y = mb.matmul(x=x_reshaped, y=w, name="linear");
        tensor<fp16, [1,{oc},1,{s}]> y_transposed = mb.transpose(x=y, perm=[0,2,1], name="y_t");

        // Residual add
        tensor<fp16, [1,{oc},1,{s}]> out = mb.add(x=y_transposed, y=residual, name="residual_add");
    }} -> (out);
}}"#,
            ic = in_features,
            oc = out_features,
            s = seq_len
        )
    }

    /// SwiGLU fused kernel
    /// Implements: SwiGLU(x) = SiLU(x @ W1) * (x @ W2) @ W3
    fn mil_swiglu(&self, dim: usize, hidden_dim: usize, seq_len: usize) -> String {
        format!(
            r#"program(1.3)
[buildInfo=dict<string,string>("target_os"="ios","target_version"="18")] {{
    func main<ios18>(tensor<fp16, [1,{dim},1,{s}]> x, tensor<fp16, [{dim},{hidden}]> w1, tensor<fp16, [{dim},{hidden}]> w2, tensor<fp16, [{hidden},{dim}]> w3) {{
        // Reshape input
        tensor<fp16, [1,{s},{dim}]> x_reshaped = mb.reshape(x=x, shape=[1,{s},{dim}], name="x_reshaped");

        // First projection with SiLU activation (gate)
        tensor<fp16, [1,{s},{hidden}]> h1_linear = mb.matmul(x=x_reshaped, y=w1, name="h1_linear");
        tensor<fp16, [1,{s},{hidden}]> h1_silu = mb.sigmoid(x=h1_linear, name="h1_sig");
        tensor<fp16, [1,{s},{hidden}]> h1 = mb.mul(x=h1_linear, y=h1_silu, name="h1_swish");

        // Second projection (no activation)
        tensor<fp16, [1,{s},{hidden}]> h2 = mb.matmul(x=x_reshaped, y=w2, name="h2_linear");

        // Element-wise multiply (gating)
        tensor<fp16, [1,{s},{hidden}]> h = mb.mul(x=h1, y=h2, name="h_gated");

        // Output projection
        tensor<fp16, [1,{s},{dim}]> y = mb.matmul(x=h, y=w3, name="out_proj");
        tensor<fp16, [1,{dim},1,{s}]> out = mb.transpose(x=y, perm=[0,2,1], name="out");
    }} -> (out);
}}"#,
            dim = dim,
            hidden = hidden_dim,
            s = seq_len
        )
    }

    /// Get estimated memory savings (bytes) vs unfused approach
    pub fn memory_savings(&self) -> usize {
        match self {
            FusedKernelType::RmsnormLinear {
                channels,
                seq_len,
                out_features: _,
            } => {
                // Unfused: intermediate [1, C, 1, S] needs to be written/read
                channels * seq_len * 2 // fp16
            }
            FusedKernelType::LinearActivation {
                out_features,
                seq_len,
                ..
            } => {
                // Unfused: intermediate [1, OC, 1, S]
                out_features * seq_len * 2
            }
            FusedKernelType::QkvProjection {
                q_dim,
                kv_dim,
                seq_len,
                ..
            } => {
                // Unfused: Q and K intermediates
                (q_dim + kv_dim) * seq_len * 2
            }
            FusedKernelType::LinearResidual {
                out_features,
                seq_len,
                ..
            } => {
                // Unfused: intermediate output
                out_features * seq_len * 2
            }
            FusedKernelType::Swiglu {
                hidden_dim,
                seq_len,
                ..
            } => {
                // Unfused: h1 and h2 intermediates
                hidden_dim * seq_len * 2 * 2 // Two intermediates
            }
        }
    }

    /// Get estimated compile budget savings
    pub fn compile_savings(&self) -> i32 {
        match self {
            // Fused: 1 kernel vs Unfused: 2 kernels (norm + linear)
            FusedKernelType::RmsnormLinear { .. } => 1,
            // Fused: 1 kernel vs Unfused: 2 kernels (linear + activation)
            FusedKernelType::LinearActivation { .. } => 1,
            // Fused: 1 kernel vs Unfused: 3 kernels (Q + K + V separate)
            FusedKernelType::QkvProjection { .. } => 2,
            // Fused: 1 kernel vs Unfused: 2 kernels
            FusedKernelType::LinearResidual { .. } => 1,
            // Fused: 1 kernel vs Unfused: 4 kernels (W1 + SiLU + W2 + W3)
            FusedKernelType::Swiglu { .. } => 3,
        }
    }
}

/// Fused kernel registry
///
/// Manages creation and lookup of fused kernels
pub struct FusedKernelRegistry {
    registry: HashMap<FusedKernelType, String>,
    total_compile_savings: i32,
}

impl FusedKernelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            total_compile_savings: 0,
        }
    }

    /// Register a fused kernel type
    pub fn register(&mut self, kernel_type: FusedKernelType) -> &str {
        let id = kernel_type.id();
        let _mil = kernel_type.generate_mil();
        let savings = kernel_type.compile_savings();

        self.total_compile_savings += savings;
        self.registry.entry(kernel_type).or_insert(id)
    }

    /// Get or create a fused kernel
    pub fn get_or_create(&mut self, kernel_type: &FusedKernelType) -> String {
        if let Some(id) = self.registry.get(kernel_type) {
            id.clone()
        } else {
            self.register(kernel_type.clone());
            kernel_type.id()
        }
    }

    /// Get total compile budget saved
    pub fn compile_savings(&self) -> i32 {
        self.total_compile_savings
    }

    /// Get total memory saved per step (bytes)
    pub fn memory_savings_per_step(&self) -> usize {
        self.registry.keys().map(|k| k.memory_savings()).sum()
    }

    /// Print registry report
    pub fn print_report(&self) {
        println!("\n=== Fused Kernel Registry Report ===\n");
        println!("Total fused kernels: {}", self.registry.len());
        println!(
            "Compile budget saved: {} kernels",
            self.total_compile_savings
        );
        println!(
            "Memory saved per step: {:.2} KB",
            self.memory_savings_per_step() as f64 / 1024.0
        );
        println!("\nRegistered kernels:");
        for (kernel_type, id) in &self.registry {
            println!("  - {} (id: {})", kernel_type.id(), id);
        }
    }
}

impl Default for FusedKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_kernel_id_generation() {
        let kernel = FusedKernelType::RmsnormLinear {
            channels: 768,
            seq_len: 256,
            out_features: 768,
        };

        let id = kernel.id();
        assert!(id.contains("fused_rmsnorm_linear"));
        assert!(id.contains("768"));
        assert!(id.contains("256"));
    }

    #[test]
    fn test_fused_kernel_mil_generation() {
        let kernel = FusedKernelType::LinearActivation {
            in_features: 512,
            out_features: 512,
            seq_len: 128,
            activation: ActivationType::Relu,
        };

        let mil = kernel.generate_mil();
        assert!(mil.contains("func main<ios18>"));
        assert!(mil.contains("mb.matmul"));
        assert!(mil.contains("mb.relu"));
    }

    #[test]
    fn test_compile_savings() {
        let swiglu = FusedKernelType::Swiglu {
            dim: 768,
            hidden_dim: 2048,
            seq_len: 256,
        };
        assert_eq!(swiglu.compile_savings(), 3);

        let qkv = FusedKernelType::QkvProjection {
            dim: 768,
            q_dim: 768,
            kv_dim: 768,
            seq_len: 256,
        };
        assert_eq!(qkv.compile_savings(), 2);
    }

    #[test]
    fn test_registry_tracks_savings() {
        let mut registry = FusedKernelRegistry::new();

        registry.register(FusedKernelType::RmsnormLinear {
            channels: 768,
            seq_len: 256,
            out_features: 768,
        });

        registry.register(FusedKernelType::Swiglu {
            dim: 768,
            hidden_dim: 2048,
            seq_len: 256,
        });

        assert_eq!(registry.compile_savings(), 4); // 1 + 3
    }
}
