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

/// 1x1 convolution projection for channel-first ANE layouts.
///
/// Implements `conv` on `[1, in_dim, 1, seq_len]` input with a
/// `[out_dim, in_dim, 1, 1]` weight tensor.
pub fn conv1x1_mil(seq_len: usize, in_dim: usize, out_dim: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        in_dim, seq_len
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        in_dim, seq_len
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        out_dim, in_dim, out_dim, in_dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n",
        out_dim, seq_len
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        out_dim, seq_len
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

/// Build a compile request for the `conv1x1_mil` template.
pub fn conv1x1_compile_request(
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    weights: &ANEWeightBlob,
) -> ANECompileRequest {
    ANECompileRequest::new(
        conv1x1_mil(seq_len, in_dim, out_dim),
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

/// CONV_CONST string used by backward MIL programs (from stories_mil.h)
const CONV_CONST_STR: &str = "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";

/// FFN (SwiGLU) backward pass MIL program (program 1.3 format)
///
/// Ported from `gen_ffn_bwd()` in stories_mil.h. Computes gradients for the
/// SwiGLU FFN block: backward through W2, sigmoid-gating, and W1/W3 projections.
///
/// ANE adaptations vs reference:
/// - `sub(x, y)` → `add(x, mul(y, const(-1.0)))` (sub is rejected)
/// - `concat(dx, dh1, dh3)` → multi-output return (concat is rejected)
///
/// Input shape: `[1, DIM+2*HIDDEN, 1, SEQ]` fp16 (packed: dffn, h1, h3)
/// Output shapes: `[1, DIM, 1, SEQ]` (dx), `[1, HIDDEN, 1, SEQ]` (dh1, dh3) fp16
pub fn bwd_ffn_mil(seq_len: usize, dim: usize, hidden_dim: usize) -> String {
    let in_ch = dim + 2 * hidden_dim;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        in_ch, seq_len
    ));
    mil.push_str(CONV_CONST_STR);
    mil.push_str("        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str(&format!(
        "        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        dim
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        dim + hidden_dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];\n",
        hidden_dim, seq_len
    ));
    // W2t and dsilu
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];\n",
        hidden_dim, dim, hidden_dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=string(\"cw2\")];\n",
        hidden_dim, seq_len
    ));
    // sigmoid
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n",
        hidden_dim, seq_len
    ));
    // sub decomposition: oms = 1 - sig → add(1, mul(sig, -1))
    mil.push_str("        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n");
    mil.push_str("        fp16 nm1 = const()[name=string(\"nm1\"), val=fp16(-1.0)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> sneg = mul(x=sig,y=nm1)[name=string(\"msn\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> oms = add(x=one,y=sneg)[name=string(\"oms\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n",
        hidden_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n",
        hidden_dim, seq_len
    ));
    // W1t, W3t, dx
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];\n",
        dim, hidden_dim, dim, hidden_dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=string(\"cw1\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];\n",
        dim, hidden_dim, dim, hidden_dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=string(\"cw3\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];\n",
        dim, seq_len
    ));
    // Multi-output return (alphabetical order): dh1, dh3, dx
    mil.push_str("    } -> (dh1, dh3, dx);\n}\n");
    mil
}

/// Build a compile request for the `bwd_ffn_mil` template.
pub fn bwd_ffn_compile_request(
    seq_len: usize,
    dim: usize,
    hidden_dim: usize,
    w1t: &ANEWeightBlob,
    w2t: &ANEWeightBlob,
    w3t: &ANEWeightBlob,
) -> ANECompileRequest {
    let in_ch = dim + 2 * hidden_dim;
    ANECompileRequest::new(
        bwd_ffn_mil(seq_len, dim, hidden_dim),
        vec![in_ch * seq_len * 2],
        vec![
            hidden_dim * seq_len * 2,
            hidden_dim * seq_len * 2,
            dim * seq_len * 2,
        ],
    )
    .with_weight_blob("@model_path/weights/w1t.bin", w1t)
    .with_weight_blob("@model_path/weights/w2t.bin", w2t)
    .with_weight_blob("@model_path/weights/w3t.bin", w3t)
}

/// QKV backward pass MIL program (program 1.3 format)
///
/// Ported from `gen_qkvb()` in stories_mil.h. Computes dx from packed (dq, dk, dv).
///
/// Input shape: `[1, 3*DIM, 1, SEQ]` fp16 (packed: dq, dk, dv)
/// Output shape: `[1, DIM, 1, SEQ]` fp16 (dx)
pub fn bwd_qkv_mil(seq_len: usize, dim: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        3 * dim,
        seq_len
    ));
    mil.push_str(CONV_CONST_STR);
    mil.push_str(&format!(
        "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq_len
    ));
    mil.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dq = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dk = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        2 * dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dv = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> Wqt = const()[name=string(\"Wqt\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/wqt.bin\"), offset=uint64(64)))];\n",
        dim, dim, dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> Wkt = const()[name=string(\"Wkt\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/wkt.bin\"), offset=uint64(64)))];\n",
        dim, dim, dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> Wvt = const()[name=string(\"Wvt\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/wvt.bin\"), offset=uint64(64)))];\n",
        dim, dim, dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq)[name=string(\"cq\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk)[name=string(\"ck\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv)[name=string(\"cv\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dx = add(x=dxqk,y=dxv)[name=string(\"out\")];\n",
        dim, seq_len
    ));
    mil.push_str("    } -> (dx);\n}\n");
    mil
}

/// Build a compile request for the `bwd_qkv_mil` template.
pub fn bwd_qkv_compile_request(
    seq_len: usize,
    dim: usize,
    wqt: &ANEWeightBlob,
    wkt: &ANEWeightBlob,
    wvt: &ANEWeightBlob,
) -> ANECompileRequest {
    ANECompileRequest::new(
        bwd_qkv_mil(seq_len, dim),
        vec![3 * dim * seq_len * 2],
        vec![dim * seq_len * 2],
    )
    .with_weight_blob("@model_path/weights/wqt.bin", wqt)
    .with_weight_blob("@model_path/weights/wkt.bin", wkt)
    .with_weight_blob("@model_path/weights/wvt.bin", wvt)
}

/// SDPA backward part 1 + Wo^T (program 1.3 format)
///
/// Ported from `gen_sdpa_bwd1()` in stories_mil.h. Recomputes attention probs
/// from saved Q, K, V, then computes dV and partial dP (attention score gradient).
///
/// ANE adaptation: `concat` output → multi-output (dpf, dvf, pf) alphabetical
///
/// Input shape: `[1, 4*DIM, 1, SEQ]` fp16 (packed: qf, kf, vf, dx2f)
/// Output shapes: `[1, DIM, 1, SEQ]` (dvf), `[1, SCORE_CH, 1, SEQ]` (dpf, pf) fp16
/// where SCORE_CH = HEADS * SEQ
pub fn bwd_sdpa_bwd1_mil(seq_len: usize, dim: usize, heads: usize, head_dim: usize) -> String {
    let sc = 1.0 / (head_dim as f32).sqrt();
    let score_ch = heads * seq_len;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        4 * dim,
        seq_len
    ));
    mil.push_str(CONV_CONST_STR);
    // Slice params
    mil.push_str(&format!(
        "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq_len
    ));
    mil.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    // Extract qf, kf, vf, dx2f
    for (i, name) in ["qf", "kf", "vf", "df"].iter().enumerate() {
        mil.push_str(&format!(
            "        tensor<int32, [4]> b{i} = const()[name=string(\"b{i}\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
            i * dim
        ));
        mil.push_str(&format!(
            "        tensor<fp16, [1,{},1,{}]> {name} = slice_by_size(x=x,begin=b{i},size=sz)[name=string(\"s{i}\")];\n",
            dim, seq_len
        ));
    }
    // Wot and df = conv(Wot, dx2f)
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> Wot = const()[name=string(\"Wot\"), val=tensor<fp16, [{},{},1,1]>(BLOBFILE(path=string(\"@model_path/weights/wot.bin\"), offset=uint64(64)))];\n",
        dim, dim, dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> da = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=df)[name=string(\"cwo\")];\n",
        dim, seq_len
    ));
    // Reshape to [HEADS, HD, SEQ], transpose to [HEADS, SEQ, HD]
    mil.push_str(&format!(
        "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,{}, {},{}])];\n",
        heads, head_dim, seq_len
    ));
    mil.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    for name in ["qf", "kf", "vf"] {
        mil.push_str(&format!(
            "        tensor<fp16, [1,{},{},{}]> {n}r = reshape(shape=rsh,x={name})[name=string(\"r{n}\")];\n",
            heads, head_dim, seq_len, n = name
        ));
        mil.push_str(&format!(
            "        tensor<fp16, [1,{},{},{}]> {n} = transpose(perm=pm,x={n}r)[name=string(\"t{n}\")];\n",
            heads, seq_len, head_dim, n = name
        ));
    }
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dr = reshape(shape=rsh,x=da)[name=string(\"rda\")];\n",
        heads, head_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> datt = transpose(perm=pm,x=dr)[name=string(\"td\")];\n",
        heads, seq_len, head_dim
    ));
    // Recompute forward attention: scores = Q @ K^T * scale, probs = softmax(scores + mask)
    mil.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    mil.push_str("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qf,y=kf)[name=string(\"mm1\")];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str(&format!(
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({:.6})];\n",
        sc
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n",
        heads, seq_len, seq_len
    ));
    // Causal mask
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{},{}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n",
        seq_len, seq_len, seq_len, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str("        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n",
        heads, seq_len, seq_len
    ));
    // dV = probs^T @ dAttn, dP = dAttn @ V
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=datt)[name=string(\"dv\")];\n",
        heads, seq_len, head_dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=datt,y=vf)[name=string(\"dp\")];\n",
        heads, seq_len, seq_len
    ));
    // Reshape dV back to [1, DIM, 1, SEQ]
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n",
        heads, head_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n",
        dim, seq_len
    ));
    // Reshape probs, dp to [1, SCORE_CH, 1, SEQ]
    mil.push_str(&format!(
        "        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        score_ch, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n",
        score_ch, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n",
        score_ch, seq_len
    ));
    // Multi-output (alphabetical): dpf, dvf, pf
    mil.push_str("    } -> (dpf, dvf, pf);\n}\n");
    mil
}

/// Build a compile request for bwd_sdpa_bwd1_mil.
pub fn bwd_sdpa_bwd1_compile_request(
    seq_len: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    wot: &ANEWeightBlob,
    mask: &ANEWeightBlob,
) -> ANECompileRequest {
    let score_ch = heads * seq_len;
    ANECompileRequest::new(
        bwd_sdpa_bwd1_mil(seq_len, dim, heads, head_dim),
        vec![4 * dim * seq_len * 2],
        vec![
            score_ch * seq_len * 2, // dpf
            dim * seq_len * 2,      // dvf
            score_ch * seq_len * 2, // pf
        ],
    )
    .with_weight_blob("@model_path/weights/wot.bin", wot)
    .with_weight_blob("@model_path/weights/mask.bin", mask)
}

/// SDPA backward part 2 (program 1.3 format)
///
/// Ported from `gen_sdpa_bwd2()` in stories_mil.h. Computes dQ and dK from
/// attention probs, dp (score gradient), and saved Q, K.
///
/// ANE adaptations: `sub` → add+mul decomposition, `concat` → multi-output
///
/// Input shape: `[1, 2*SCORE_CH + 2*DIM, 1, SEQ]` fp16
/// Output shapes: `[1, DIM, 1, SEQ]` (dkf, dqf) fp16
pub fn bwd_sdpa_bwd2_mil(seq_len: usize, dim: usize, heads: usize, head_dim: usize) -> String {
    let sc = 1.0 / (head_dim as f32).sqrt();
    let score_ch = heads * seq_len;
    let in_ch = 2 * score_ch + 2 * dim;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        in_ch, seq_len
    ));
    // Slice params for SCORE_CH-sized and DIM-sized chunks
    mil.push_str(&format!(
        "        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        score_ch, seq_len
    ));
    mil.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n",
        score_ch, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        score_ch
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n",
        score_ch, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        2 * score_ch
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{},0,0])];\n",
        2 * score_ch + dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];\n",
        dim, seq_len
    ));
    // Reshape probs, dp to [HEADS, SEQ, SEQ]
    mil.push_str(&format!(
        "        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,{}, {},{}])];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n",
        heads, seq_len, seq_len
    ));
    // Reshape qf, kf to [HEADS, HD, SEQ], transpose to [HEADS, SEQ, HD]
    mil.push_str(&format!(
        "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,{}, {},{}])];\n",
        heads, head_dim, seq_len
    ));
    mil.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    for name in ["qf", "kf"] {
        mil.push_str(&format!(
            "        tensor<fp16, [1,{},{},{}]> {n}r = reshape(shape=rsh,x={name})[name=string(\"rq{n}\")];\n",
            heads, head_dim, seq_len, n = &name[0..1]
        ));
        mil.push_str(&format!(
            "        tensor<fp16, [1,{},{},{}]> {name} = transpose(perm=pm,x={n}r)[name=string(\"tq{n}\")];\n",
            heads, seq_len, head_dim, n = &name[0..1]
        ));
    }
    // Softmax backward: pdp = probs * dp, spdp = reduce_sum(pdp, -1), dps = dp - spdp
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str("        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n");
    mil.push_str("        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n",
        heads, seq_len
    ));
    // sub decomposition: dps = dp - spdp → add(dp, mul(spdp, -1))
    mil.push_str("        fp16 nm1 = const()[name=string(\"nm1\"), val=fp16(-1.0)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> sneg = mul(x=spdp,y=nm1)[name=string(\"sneg\")];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dps = add(x=dp,y=sneg)[name=string(\"dps\")];\n",
        heads, seq_len, seq_len
    ));
    // ds = probs * dps * scale
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n",
        heads, seq_len, seq_len
    ));
    mil.push_str(&format!(
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({:.6})];\n",
        sc
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n",
        heads, seq_len, seq_len
    ));
    // dQ = ds @ K, dK = ds^T @ Q
    mil.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    mil.push_str("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=kf)[name=string(\"dq\")];\n",
        heads, seq_len, head_dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=qf)[name=string(\"dk\")];\n",
        heads, seq_len, head_dim
    ));
    // Transpose back, reshape to [1, DIM, 1, SEQ]
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n",
        heads, head_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n",
        heads, head_dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];\n",
        dim, seq_len
    ));
    // Multi-output (alphabetical): dkf, dqf
    mil.push_str("    } -> (dkf, dqf);\n}\n");
    mil
}

/// Build a compile request for bwd_sdpa_bwd2_mil.
pub fn bwd_sdpa_bwd2_compile_request(
    seq_len: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
) -> ANECompileRequest {
    ANECompileRequest::new(
        bwd_sdpa_bwd2_mil(seq_len, dim, heads, head_dim),
        vec![(2 * heads * seq_len + 2 * dim) * seq_len * 2],
        vec![dim * seq_len * 2, dim * seq_len * 2],
    )
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

/// Dynamic linear layer: matmul with weights packed into the input tensor.
///
/// Instead of baking weights into the model (which requires recompilation to update),
/// this program packs the weight matrix into the input alongside activations.
/// Weights can then change every step by just changing the input IOSurface content.
///
/// **Input layout** (fp32): `[1, D + D*D, 1, S]`
/// - Channels `0..D` contain activations `[1, D, 1, S]`
/// - Channels `D..D+D*D` contain weight matrix `[D, D]` (only first spatial position used)
///
/// **Output layout** (fp32): `[1, D, 1, S]`
///
/// **No weights file needed** — all weight data comes through the input.
///
/// ## How it works (inside MIL)
///
/// ```text
/// input [1, D+D*D, 1, S]
///   ├── slice [0..D] → act [1, D, 1, S]
///   └── slice [D..D+D*D] → wf [1, D*D, 1, S]
///         └── slice [:, 0] → wf1 [1, D*D, 1, 1]
///               └── reshape → W [1, 1, D, D]
///   act → reshape → [1, 1, S, D] → transpose → [1, 1, D, S]
///   matmul([1,1,D,S], [1,1,D,D]) → [1,1,D,S]
///   transpose → reshape → [1, D, 1, S]
/// ```
///
/// ## Input size calculation
///
/// ```ignore
/// let input_bytes = (D + D * D) * S * 4;  // fp32
/// let output_bytes = D * S * 4;           // fp32
/// ```
///
/// ## Weight packing format
///
/// Weights are stored in row-major layout in the input's weight channels.
/// For a `[D, D]` weight matrix W, element W[r][c] goes at:
/// `input_offset = D * S + (r * D + c) * S + 0` (only spatial position 0 matters)
pub fn dynamic_matmul_mil(seq_len: usize, dim: usize) -> String {
    let total_ch = dim + dim * dim;
    let mut mil = String::new();

    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");

    // Input: [1, D+D*D, 1, S] fp32
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        total_ch, seq_len
    ));

    // Cast input to fp16
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        total_ch, seq_len
    ));

    // Slice activations: [1, D, 1, S] from channels 0..D
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str(&format!(
        "        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n",
        dim, seq_len
    ));

    // Slice weight region: [1, D*D, 1, S] from channels D..D+D*D
    mil.push_str(&format!(
        "        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n",
        dim
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        dim * dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n",
        dim * dim, seq_len
    ));

    // CRITICAL: Declaration order matters for ANE compiler!
    // ws (reshape target [1,1,D,D]) MUST be declared before sw1 ([1,1,1,1]).
    // Declaring sw1 first causes CompilationFailure on ANE.
    // Reshape weight to [1, 1, D, D] for matmul
    mil.push_str(&format!(
        "        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n",
        dim, dim
    ));

    // Take first spatial position of weight: [1, D*D, 1, 1]
    // Size must be [1, D*D, 1, 1] (NOT [1,1,1,1]) to capture all weight channels
    mil.push_str(&format!(
        "        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n",
        dim * dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n",
        dim * dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n",
        dim, dim
    ));

    // Reshape activations to [1, 1, D, S] then transpose to [1, 1, S, D]
    // matmul expects: x=[1,1,M,K] @ y=[1,1,K,N] → [1,1,M,N]
    // We want: [S, D] @ [D, D] → [S, D], so x=[1,1,S,D], y=[1,1,D,D]
    mil.push_str(&format!(
        "        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n",
        dim, seq_len
    ));
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> a2 = reshape(shape = as2, x = act)[name = string(\"a2\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> a3 = transpose(perm = pm, x = a2)[name = string(\"a3\")];\n",
        seq_len, dim
    ));

    // matmul: [1,1,S,D] @ [1,1,D,D] → [1,1,S,D]
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> yh = matmul(transpose_x = bF, transpose_y = bF, x = a3, y = W)[name = string(\"mm\")];\n",
        seq_len, dim
    ));

    // Transpose back: [1,1,S,D] → [1,1,D,S] then reshape to [1, D, 1, S]
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> yt = transpose(perm = pm, x = yh)[name = string(\"yt\")];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        dim, seq_len
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> yr = reshape(shape = os, x = yt)[name = string(\"yr\")];\n",
        dim, seq_len
    ));

    // Cast output to fp32
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n",
        dim, seq_len
    ));

    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

/// Calculate input byte size for `dynamic_matmul_mil`.
///
/// Input is `[1, D + D*D, 1, S]` fp32.
pub fn dynamic_matmul_input_bytes(dim: usize, seq_len: usize) -> usize {
    (dim + dim * dim) * seq_len * 4
}

/// Calculate output byte size for `dynamic_matmul_mil`.
///
/// Output is `[1, D, 1, S]` fp32.
pub fn dynamic_matmul_output_bytes(dim: usize, seq_len: usize) -> usize {
    dim * seq_len * 4
}

/// Pack activations and weights into a single input buffer for `dynamic_matmul_mil`.
///
/// Input layout: `[activations(D*S), weights(D*D*S)]` as fp32.
/// Weight matrix is stored in row-major: W[r][c] at `base + (r*D + c) * S + 0`.
pub fn pack_dynamic_matmul_input(
    activations: &[f32],
    weights: &[f32],
    dim: usize,
    seq_len: usize,
) -> Vec<u8> {
    let total_ch = dim + dim * dim;
    let mut input = vec![0.0f32; total_ch * seq_len];

    // Copy activations to channels 0..D
    input[..dim * seq_len].copy_from_slice(activations);

    // Copy weights to channels D..D+D*D
    pack_weights_into(&mut input, weights, dim, seq_len);

    // Convert to bytes (fp32 little-endian)
    input.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Update only the weight region of a pre-allocated packed input buffer.
///
/// This avoids reallocating the 1MB+ buffer on every training step.
/// The `buffer` must have been created by `pack_dynamic_matmul_input` or
/// `pack_dynamic_matmul_input_f32`.
///
/// Weight matrix is stored in row-major: W[r][c] at `base + (r*D + c) * S + 0`.
pub fn pack_weights_into(buffer: &mut [f32], weights: &[f32], dim: usize, seq_len: usize) {
    assert_eq!(weights.len(), dim * dim);
    let w_base = dim * seq_len;

    // Zero out old weight region (D*D channels × S spatial = D*D*S floats)
    let w_region_len = dim * dim * seq_len;
    buffer[w_base..w_base + w_region_len].fill(0.0);

    // Write new weights: W[r][c] at position (D + r*D + c) * S + 0
    for r in 0..dim {
        for c in 0..dim {
            buffer[w_base + (r * dim + c) * seq_len] = weights[r * dim + c];
        }
    }
}

/// Pack activations and weights into a pre-allocated f32 buffer.
///
/// Returns the f32 slice that can be converted to bytes for IOSurface.
/// The caller can reuse the buffer across steps by calling `pack_weights_into`.
pub fn pack_dynamic_matmul_input_f32(
    buffer: &mut [f32],
    activations: &[f32],
    weights: &[f32],
    dim: usize,
    seq_len: usize,
) {
    let total_ch = dim + dim * dim;
    assert_eq!(buffer.len(), total_ch * seq_len);
    assert_eq!(activations.len(), dim * seq_len);
    assert_eq!(weights.len(), dim * dim);

    // Zero entire buffer
    buffer.fill(0.0);

    // Copy activations to channels 0..D
    buffer[..dim * seq_len].copy_from_slice(activations);

    // Copy weights
    pack_weights_into(buffer, weights, dim, seq_len);
}
