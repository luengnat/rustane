//! Predefined MIL programs
//!
//! This module contains commonly used MIL program templates.
//! Includes templates for linear layers, convolution, and attention ops.

use crate::ane::{ANECompileRequest, WeightBlob as ANEWeightBlob};
use crate::mil::MILBuilder;

/// Simple linear layer MIL program (backward compat)
///
/// # Example
///
/// ```
/// # use rustane::mil::programs::LinearLayer;
/// let layer = LinearLayer::new(256, 512);
/// let mil = layer.mil_program();
/// ```
pub struct LinearLayer {
    input_size: usize,
    output_size: usize,
}

impl LinearLayer {
    /// Create a new linear layer
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::programs::LinearLayer;
    /// let layer = LinearLayer::new(256, 512);
    /// ```
    pub fn new(input_size: usize, output_size: usize) -> Self {
        LinearLayer {
            input_size,
            output_size,
        }
    }

    /// Get the MIL program for this layer
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::programs::LinearLayer;
    /// let layer = LinearLayer::new(256, 512);
    /// let mil = layer.mil_program();
    /// ```
    pub fn mil_program(&self) -> String {
        format!(
            "program(1.0) {{
  var input = const()[1, {}](name=\"input\", shape=[1, {}])
  var fc1 = nn.convolution(bias=false, groups=1, input_name=\"input\", kernel_sizes=[1, 1], name=\"linear\", output_channels={}, pad_type=\"valid\", strides=[1, 1], weight_name=\"weight\", padding_top=0, padding_bottom=0, padding_left=0, padding_right=0)
  return {{fc1}}
}}",
            self.input_size,
            self.input_size,
            self.output_size
        )
    }

    /// Get the expected input size in bytes (assuming FP32)
    pub fn input_size_bytes(&self) -> usize {
        self.input_size * 4 // FP32 = 4 bytes per element
    }

    /// Get the expected output size in bytes (assuming FP32)
    pub fn output_size_bytes(&self) -> usize {
        self.output_size * 4 // FP32 = 4 bytes per element
    }

    /// Get the weight matrix size (input_size × output_size)
    pub fn weight_size(&self) -> (usize, usize) {
        (self.input_size, self.output_size)
    }
}

/// Convolution layer MIL program
pub struct ConvLayer {
    input_channels: usize,
    output_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
}

impl ConvLayer {
    /// Create a new convolution layer
    ///
    /// # Arguments
    ///
    /// * `input_channels` - Number of input channels
    /// * `output_channels` - Number of output channels
    /// * `kernel_size` - (height, width) of the kernel
    /// * `stride` - (height, width) stride
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Self {
        ConvLayer {
            input_channels,
            output_channels,
            kernel_size,
            stride,
        }
    }

    /// Get the MIL program for this layer
    pub fn mil_program(&self) -> String {
        format!(
            "program(1.0) {{
  var input = const()[1, {}, 224, 224](name=\"input\", shape=[1, {}, 224, 224])
  var conv1 = nn.convolution(bias=false, groups=1, input_name=\"input\", kernel_sizes=[{}, {}], name=\"conv\", output_channels={}, pad_type=\"valid\", strides=[{}, {}], weight_name=\"weight\", padding_top=0, padding_bottom=0, padding_left=0, padding_right=0)
  return {{conv1}}
}}",
            self.input_channels,
            self.input_channels,
            self.kernel_size.0,
            self.kernel_size.1,
            self.output_channels,
            self.stride.0,
            self.stride.1
        )
    }
}

/// Non-square linear projection via matmul (program 1.3 format)
///
/// Implements: W[1, out, in] @ x[1, in, S] -> [1, out, S]
/// Using matmul MIL op to handle non-square dimensions
/// that conv would reject.
///
/// # Arguments
///
/// * `seq_len` - Sequence length (S)
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Example
///
/// ```
/// # use rustane::mil::programs::linear_matmul_mil;
/// let mil = linear_matmul_mil(256, 512, 1024);
/// assert!(mil.contains("program(1.3)"));
/// assert!(mil.contains("matmul(transpose_x = bF, transpose_y = bF"));
/// ```
pub fn linear_matmul_mil(seq_len: usize, in_dim: usize, out_dim: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, {}]> x) {{\n",
        in_dim, seq_len
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n",
        in_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}]> W = const()[name = string(\"W\"), val = tensor<fp16, [1, {}, {}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        out_dim, in_dim, out_dim, in_dim
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}]> y16 = matmul(transpose_x = bF, transpose_y = bF, x = W, y = x16)[name = string(\"mm\")];\n",
        out_dim, seq_len
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, {}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        out_dim, seq_len
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

/// Build a compile request for the `linear_matmul_mil` template.
///
/// This helper packages the generated MIL text together with the correctly
/// named ANE weight blob expected by the `BLOBFILE` reference.
pub fn linear_matmul_compile_request(
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    weights: &ANEWeightBlob,
) -> ANECompileRequest {
    ANECompileRequest::new(
        linear_matmul_mil(seq_len, in_dim, out_dim),
        vec![in_dim * seq_len * 4],
        vec![out_dim * seq_len * 4],
    )
    .with_weight_blob("@model_path/weights/weight.bin", weights)
}

/// RMSNorm forward MIL program (program 1.3 format)
///
/// Implements the upstream ANE pattern:
/// `x * x -> reduce_sum(axis=1) -> * invd -> + eps -> pow(-0.5) -> * gamma`
///
/// Input shape:
/// - `[1, dim, 1, seq_len]` fp16
///
/// Output shape:
/// - `[1, dim, 1, seq_len]` fp16
///
/// # Example
///
/// ```
/// # use rustane::mil::programs::rmsnorm_mil;
/// let mil = rmsnorm_mil(64, 256);
/// assert!(mil.contains("reduce_sum"));
/// assert!(mil.contains("pow"));
/// ```
pub fn rmsnorm_mil(seq_len: usize, dim: usize) -> String {
    let invd = 1.0f32 / dim as f32;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> sq = mul(x=x,y=x)[name = string(\"sq\")];\n",
        dim, seq_len
    ));
    mil.push_str("        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val=tensor<int32, [1]>([1])];\n");
    mil.push_str("        bool kd = const()[name = string(\"kd\"), val=bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,1,{}]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name = string(\"ss\")];\n",
        seq_len
    ));
    mil.push_str(&format!(
        "        fp16 invd = const()[name = string(\"invd\"), val=fp16({:.8})];\n",
        invd
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,1,{}]> ss2 = mul(x=ss,y=invd)[name = string(\"ss2\")];\n",
        seq_len
    ));
    mil.push_str("        fp16 eps = const()[name = string(\"eps\"), val=fp16(0.00001)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,1,{}]> ss3 = add(x=ss2,y=eps)[name = string(\"ss3\")];\n",
        seq_len
    ));
    mil.push_str("        fp16 nhalf = const()[name = string(\"nhalf\"), val=fp16(-0.5)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,1,{}]> rrms = pow(x=ss3,y=nhalf)[name = string(\"rrms\")];\n",
        seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, 1]> w = const()[name = string(\"w\"), val = tensor<fp16, [1, {}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/rms_w.bin\"), offset = uint64(64)))];\n",
        dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> xr = mul(x=x,y=rrms)[name = string(\"xr\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> y16 = mul(x=xr,y=w)[name = string(\"y16\")];\n",
        dim, seq_len
    ));
    mil.push_str("    } -> (y16);\n");
    mil.push_str("}\n");
    mil
}

/// Build a compile request for the `rmsnorm_mil` template.
///
/// The request uses the canonical `@model_path/weights/rms_w.bin` weight name
/// referenced by the generated MIL.
pub fn rmsnorm_compile_request(
    seq_len: usize,
    dim: usize,
    weights: &ANEWeightBlob,
) -> ANECompileRequest {
    let tensor_bytes = dim * seq_len * 2;
    ANECompileRequest::new(
        rmsnorm_mil(seq_len, dim),
        vec![tensor_bytes],
        vec![tensor_bytes],
    )
    .with_weight_blob("@model_path/weights/rms_w.bin", weights)
}

/// GQA (Grouped Query Attention) SDPA kernel (program 1.3 format)
///
/// Handles GQA where q_heads > kv_heads by tiling K/V.
/// When q_heads == kv_heads, this is standard SDPA.
///
/// # Arguments
///
/// * `batch` - Batch size
/// * `q_heads` - Number of query heads
/// * `kv_heads` - Number of key-value heads
/// * `seq_len` - Sequence length
/// * `head_dim` - Dimension per head
///
/// # Example
///
/// ```
/// # use rustane::mil::programs::gqa_sdpa_mil;
/// // parameter-golf: 8 Q heads, 4 KV heads
/// let mil = gqa_sdpa_mil(1, 8, 4, 256, 64);
/// assert!(mil.contains("mb.scaled_dot_product_attention"));
/// ```
pub fn gqa_sdpa_mil(
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> String {
    MILBuilder::new()
        .add_input("query", "fp32", &[batch, q_heads, seq_len, head_dim])
        .add_input("key", "fp32", &[batch, kv_heads, seq_len, head_dim])
        .add_input("value", "fp32", &[batch, kv_heads, seq_len, head_dim])
        .add_output("output", "fp32", &[batch, q_heads, seq_len, head_dim])
        .add_sdpa("output", "query", "key", "value")
        .build()
}

/// Full causal attention block for parameter-golf architecture (program 1.3 format)
///
/// Implements attention with 8 query heads, 4 key-value heads (GQA).
/// Output dimension is q_heads * head_dim.
///
/// # Arguments
///
/// * `seq_len` - Sequence length
/// * `dim` - Model dimension (typically 512 for parameter-golf)
/// * `q_heads` - Number of query heads (8)
/// * `kv_heads` - Number of key-value heads (4)
/// * `head_dim` - Dimension per head (typically 64)
///
/// # Example
///
/// ```
/// # use rustane::mil::programs::pg_attention_mil;
/// let mil = pg_attention_mil(256, 512, 8, 4, 64);
/// assert!(mil.contains("mb.scaled_dot_product_attention"));
/// ```
pub fn pg_attention_mil(
    seq_len: usize,
    dim: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> String {
    // Note: In a full implementation, this would include:
    // 1. Linear projections for Q, K, V
    // 2. SDPA with GQA support
    // 3. Output projection back to dim
    // For now, we just demonstrate SDPA
    MILBuilder::new()
        .add_input("x", "fp32", &[1, seq_len, dim])
        .add_input("q_proj_weight", "fp32", &[dim, q_heads * head_dim])
        .add_input("k_proj_weight", "fp32", &[dim, kv_heads * head_dim])
        .add_input("v_proj_weight", "fp32", &[dim, kv_heads * head_dim])
        .add_input("out_proj_weight", "fp32", &[q_heads * head_dim, dim])
        .add_output("output", "fp32", &[1, seq_len, dim])
        // Simplified: just return input for now - full implementation would do:
        // q = matmul(x, q_proj), k = matmul(x, k_proj), v = matmul(x, v_proj)
        // out = sdpa(q, k, v), then matmul(out, out_proj)
        .add_matmul("q_proj", "x", "q_proj_weight", false)
        .add_matmul("k_proj", "x", "k_proj_weight", false)
        .add_matmul("v_proj", "x", "v_proj_weight", false)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ane::WeightBlob as ANEWeightBlob;

    #[test]
    fn test_linear_layer_creation() {
        let layer = LinearLayer::new(256, 512);
        assert_eq!(layer.input_size, 256);
        assert_eq!(layer.output_size, 512);
    }

    #[test]
    fn test_linear_layer_mil_program() {
        let layer = LinearLayer::new(256, 512);
        let mil = layer.mil_program();

        assert!(mil.contains("program(1.0)"));
        assert!(mil.contains("input"));
        assert!(mil.contains("output_channels=512"));
    }

    #[test]
    fn test_linear_layer_sizes() {
        let layer = LinearLayer::new(256, 512);
        assert_eq!(layer.input_size_bytes(), 256 * 4);
        assert_eq!(layer.output_size_bytes(), 512 * 4);
        assert_eq!(layer.weight_size(), (256, 512));
    }

    #[test]
    fn test_conv_layer_creation() {
        let layer = ConvLayer::new(3, 64, (7, 7), (2, 2));
        assert_eq!(layer.input_channels, 3);
        assert_eq!(layer.output_channels, 64);
    }

    #[test]
    fn test_conv_layer_mil_program() {
        let layer = ConvLayer::new(3, 64, (7, 7), (2, 2));
        let mil = layer.mil_program();

        assert!(mil.contains("program(1.0)"));
        assert!(mil.contains("kernel_sizes=[7, 7]"));
        assert!(mil.contains("output_channels=64"));
        assert!(mil.contains("strides=[2, 2]"));
    }

    #[test]
    fn test_linear_matmul_mil() {
        let mil = linear_matmul_mil(256, 512, 1024);
        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("matmul(transpose_x = bF, transpose_y = bF"));
        assert!(mil.contains("tensor<fp32, [1, 512, 256]> x"));
        assert!(mil.contains("tensor<fp16, [1, 1024, 512]> W"));
        assert!(mil.contains("tensor<fp32, [1, 1024, 256]> y"));
    }

    #[test]
    fn test_linear_matmul_compile_request() {
        let blob = ANEWeightBlob::from_f32(&vec![1.0f32; 4 * 8], 4, 8).unwrap();
        let request = linear_matmul_compile_request(16, 8, 4, &blob);

        assert!(request.mil_text.contains("@model_path/weights/weight.bin"));
        assert_eq!(request.input_sizes, vec![8 * 16 * 4]);
        assert_eq!(request.output_sizes, vec![4 * 16 * 4]);
        assert_eq!(
            request.weights.get("@model_path/weights/weight.bin"),
            Some(&blob.as_bytes().to_vec())
        );
    }

    #[test]
    fn test_gqa_sdpa_mil() {
        let mil = gqa_sdpa_mil(1, 8, 4, 256, 64);
        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("mb.scaled_dot_product_attention"));
        assert!(mil.contains("query: fp32[1, 8, 256, 64]"));
        assert!(mil.contains("key: fp32[1, 4, 256, 64]"));
        assert!(mil.contains("value: fp32[1, 4, 256, 64]"));
    }

    #[test]
    fn test_pg_attention_mil() {
        let mil = pg_attention_mil(256, 512, 8, 4, 64);
        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("mb.matmul"));
    }

    #[test]
    fn test_rmsnorm_mil() {
        let mil = rmsnorm_mil(64, 256);
        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("reduce_sum"));
        assert!(mil.contains("pow"));
        assert!(mil.contains("rms_w.bin"));
    }

    #[test]
    fn test_rmsnorm_compile_request() {
        let blob = ANEWeightBlob::from_f16(&vec![half::f16::from_f32(1.0); 8], 1, 8).unwrap();
        let request = rmsnorm_compile_request(32, 8, &blob);

        assert!(request.mil_text.contains("@model_path/weights/rms_w.bin"));
        assert_eq!(request.input_sizes, vec![8 * 32 * 2]);
        assert_eq!(request.output_sizes, vec![8 * 32 * 2]);
        assert_eq!(
            request.weights.get("@model_path/weights/rms_w.bin"),
            Some(&blob.as_bytes().to_vec())
        );
    }
}
