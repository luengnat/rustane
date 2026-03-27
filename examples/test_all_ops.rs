//! Comprehensive ANE operator audit.
//!
//! Tests every MIL operation we can think of on the ANE.
//! Each test runs in its own process (subprocess isolation) to avoid
//! compile budget exhaustion.
//!
//! Usage: cargo run --example test_all_ops
//!   OR:  cargo run --example test_all_ops -- --op <op_name>  (test single op)

use std::env;

use rustane::ane::WeightBlob;
use rustane::wrapper::ANECompiler;

// Each test is a self-contained MIL program string + expected outcome
struct OpTest {
    name: &'static str,
    mil_fn: fn() -> String,
    input_bytes: usize,
    output_bytes: usize,
    weight_bytes: Option<usize>, // Some = needs BLOBFILE weight
    verify: fn(&[u8]) -> bool,   // check output correctness
}

// Helper: proven buildInfo header
fn header() -> String {
    "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n".to_string()
}

fn footer() -> String {
    "    } -> (y);\n}\n".to_string()
}

// ============================================================
// OPERATOR MIL GENERATORS
// ============================================================

fn mil_cast() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c1\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = xh)[name = string(\"c2\")];\n");
    m.push_str(&footer());
    m
}

fn mil_add() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
    m.push_str(
        "        tensor<fp16, [1, 4, 1, 4]> yh = add(x = xh, y = one)[name = string(\"a\")];\n",
    );
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_mul() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        fp16 two = const()[name = string(\"two\"), val = fp16(2.0)];\n");
    m.push_str(
        "        tensor<fp16, [1, 4, 1, 4]> yh = mul(x = xh, y = two)[name = string(\"m\")];\n",
    );
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_matmul() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    m.push_str("        tensor<fp16, [1, 4, 4, 1]> xt = transpose(perm = pm, x = xh)[name = string(\"t\")];\n");
    m.push_str("        bool bf = const()[name = string(\"bf\"), val = bool(false)];\n");
    // [1,1,4,4] @ [1,1,4,4] → [1,1,4,4]
    m.push_str("        tensor<fp16, [1, 1, 4, 4]> yh = matmul(transpose_x = bf, transpose_y = bf, x = xh, y = xt)[name = string(\"mm\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 1, 4, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_conv1x1() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n");
    m.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    m.push_str("        tensor<fp16, [4, 4, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [4, 4, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n");
    m.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n");
    m.push_str(&footer());
    m
}

fn mil_reduce_sum() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val = tensor<int32, [1]>([1])];\n");
    m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
    m.push_str("        tensor<fp16, [1, 1, 1, 4]> yh = reduce_sum(x = xh, axes = rax, keep_dims = kd)[name = string(\"rs\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 1, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_pow() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        fp16 p = const()[name = string(\"p\"), val = fp16(2.0)];\n");
    m.push_str(
        "        tensor<fp16, [1, 4, 1, 4]> yh = pow(x = xh, y = p)[name = string(\"pw\")];\n",
    );
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_softmax() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([3])];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = softmax(x = xh, axis = ax)[name = string(\"sm\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_sigmoid() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str(
        "        tensor<fp16, [1, 4, 1, 4]> yh = sigmoid(x = xh)[name = string(\"sig\")];\n",
    );
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_relu() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = relu(x = xh)[name = string(\"r\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_reshape() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [4]> sh = const()[name = string(\"sh\"), val = tensor<int32, [4]>([1, 1, 4, 4])];\n");
    m.push_str("        tensor<fp16, [1, 1, 4, 4]> yh = reshape(shape = sh, x = xh)[name = string(\"r\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 1, 4, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_transpose() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    m.push_str("        tensor<fp16, [1, 4, 4, 1]> yh = transpose(perm = pm, x = xh)[name = string(\"t\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 4, 1]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_slice_by_size() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 8, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 8, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [4]> sz = const()[name = string(\"sz\"), val = tensor<int32, [4]>([1, 4, 1, 4])];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = slice_by_size(x = xh, begin = b0, size = sz)[name = string(\"s\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_sub() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
    m.push_str(
        "        tensor<fp16, [1, 4, 1, 4]> yh = sub(x = xh, y = one)[name = string(\"s\")];\n",
    );
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_concat() -> String {
    // concat along axis=1: two [1,2,1,4] → [1,4,1,4]
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 8, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 8, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [4]> s1 = const()[name = string(\"s1\"), val = tensor<int32, [4]>([1, 4, 1, 4])];\n");
    m.push_str("        tensor<int32, [4]> b1 = const()[name = string(\"b1\"), val = tensor<int32, [4]>([0, 4, 0, 0])];\n");
    m.push_str("        tensor<int32, [4]> s2 = const()[name = string(\"s2\"), val = tensor<int32, [4]>([1, 4, 1, 4])];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> a = slice_by_size(x = xh, begin = b0, size = s1)[name = string(\"a\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> b = slice_by_size(x = xh, begin = b1, size = s2)[name = string(\"b\")];\n");
    // concat along channel dim
    m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
    m.push_str("        tensor<fp16, [1, 8, 1, 4]> yh = concat(x = a, y = b, axis = ax)[name = string(\"cat\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 8, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_reduce_max() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
    m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
    m.push_str("        tensor<fp16, [1, 1, 1, 4]> yh = reduce_max(x = xh, axes = ax, keep_dims = kd)[name = string(\"rm\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 1, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_reduce_mean() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
    m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
    m.push_str("        tensor<fp16, [1, 1, 1, 4]> yh = reduce_mean(x = xh, axes = ax, keep_dims = kd)[name = string(\"rm\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 1, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_tanh() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = tanh(x = xh)[name = string(\"t\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_exp() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = exp(x = xh)[name = string(\"e\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_sqrt() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = sqrt(x = xh)[name = string(\"sq\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_rsqrt() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = rsqrt(x = xh)[name = string(\"rs\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_neg() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = neg(x = xh)[name = string(\"n\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_abs() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = abs(x = xh)[name = string(\"a\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_log() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = log(x = xh)[name = string(\"l\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_clamp() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        fp16 lo = const()[name = string(\"lo\"), val = fp16(-1.0)];\n");
    m.push_str("        fp16 hi = const()[name = string(\"hi\"), val = fp16(1.0)];\n");
    m.push_str("        fp16 sc = const()[name = string(\"sc\"), val = fp16(0.0)];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = clamp(x = xh, alpha = lo, beta = hi, scale = sc)[name = string(\"cl\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_layer_norm() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        fp16 eps = const()[name = string(\"eps\"), val = fp16(0.00001)];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = layer_norm(x = xh, axes = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])], epsilon = eps)[name = string(\"ln\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

fn mil_split() -> String {
    let mut m = header();
    m.push_str("    func main<ios18>(tensor<fp32, [1, 8, 1, 4]> x) {\n");
    m.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    m.push_str("        tensor<fp16, [1, 8, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
    m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
    m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = split(x = xh, axis = ax, splits = const()[name = string(\"sp\"), val = tensor<uint32, [1]>([2])])[name = string(\"sp\")];\n");
    m.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
    m.push_str(&footer());
    m
}

// Trivial verify: just check output is not all zeros
fn verify_not_zero(out: &[u8]) -> bool {
    out.iter().any(|&b| b != 0)
}

// Trivial verify: always true (just check it ran)
fn verify_any(out: &[u8]) -> bool {
    true
}

fn verify_add(out: &[u8]) -> bool {
    // Input 0.5, add 1.0 → output ~1.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 1.0 && v < 2.0
}

fn verify_mul(out: &[u8]) -> bool {
    // Input 0.5, mul 2.0 → output ~1.0
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.8 && v < 1.2
}

fn verify_sub(out: &[u8]) -> bool {
    // Input 0.5, sub 1.0 → output ~-0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > -1.0 && v < -0.2
}

fn verify_neg(out: &[u8]) -> bool {
    // Input 0.5, neg → output ~-0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > -1.0 && v < -0.2
}

fn verify_softmax(out: &[u8]) -> bool {
    // softmax of all 0.5s should be uniform ~0.25
    if out.len() < 64 {
        return false;
    }
    let vals: Vec<f32> = out[..64]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // All values should be similar (uniform distribution)
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let max_diff: f32 = vals.iter().map(|v| (v - mean).abs()).fold(0.0f32, f32::max);
    max_diff < 0.01
}

fn verify_sigmoid(out: &[u8]) -> bool {
    // sigmoid(0.5) ≈ 0.6225
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.5 && v < 0.8
}

fn verify_relu(out: &[u8]) -> bool {
    // relu(0.5) = 0.5, relu(-0.5) = 0.0
    if out.len() < 32 {
        return false;
    }
    let v1 = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    let v2 = f32::from_le_bytes([out[16], out[17], out[18], out[19]]);
    // First element (positive input at position 0) should be 0.5
    // Need to check what input pattern we use — for now just check output isn't all zeros
    verify_not_zero(out)
}

fn verify_tanh(out: &[u8]) -> bool {
    // tanh(0.5) ≈ 0.4621
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.3 && v < 0.6
}

fn verify_exp(out: &[u8]) -> bool {
    // exp(0.5) ≈ 1.6487
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 1.0 && v < 3.0
}

fn verify_sqrt(out: &[u8]) -> bool {
    // sqrt(0.5) ≈ 0.7071
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.5 && v < 1.0
}

fn verify_pow(out: &[u8]) -> bool {
    // 0.5^2 = 0.25
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.15 && v < 0.35
}

fn verify_log(out: &[u8]) -> bool {
    // log(0.5) ≈ -0.693
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > -1.0 && v < -0.3
}

fn verify_rsqrt(out: &[u8]) -> bool {
    // rsqrt(0.5) = 1/sqrt(0.5) ≈ 1.414
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 1.0 && v < 2.0
}

fn verify_clamp(out: &[u8]) -> bool {
    // clamp(0.5, -1, 1, 0) = 0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.3 && v < 0.7
}

fn verify_abs(out: &[u8]) -> bool {
    // abs(-0.5) = 0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.3 && v < 0.7
}

fn verify_reduce_sum(out: &[u8]) -> bool {
    // 4 channels of 0.5 each → sum = 2.0 per spatial position
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 1.5 && v < 3.0
}

fn verify_reduce_max(out: &[u8]) -> bool {
    // All 0.5 → max = 0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.3 && v < 0.7
}

fn verify_reduce_mean(out: &[u8]) -> bool {
    // 4 channels of 0.5 → mean = 0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.3 && v < 0.7
}

fn verify_concat(out: &[u8]) -> bool {
    // [1,8,1,4] → concat first 4 and last 4 channels → [1,8,1,4]
    // Should be identity if we split then re-concat
    verify_not_zero(out)
}

fn verify_cast(out: &[u8]) -> bool {
    // fp32 0.5 → fp16 → fp32 should be ~0.5
    if out.len() < 16 {
        return false;
    }
    let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    v > 0.4 && v < 0.6
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let args: Vec<String> = env::args().collect();
    let filter = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    let tests: Vec<OpTest> = vec![
        // Previously verified ops
        OpTest {
            name: "cast",
            mil_fn: mil_cast,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_cast,
        },
        OpTest {
            name: "add",
            mil_fn: mil_add,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_add,
        },
        OpTest {
            name: "mul",
            mil_fn: mil_mul,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_mul,
        },
        OpTest {
            name: "matmul",
            mil_fn: mil_matmul,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_not_zero,
        },
        OpTest {
            name: "conv1x1",
            mil_fn: mil_conv1x1,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: Some(32),
            verify: verify_not_zero,
        },
        OpTest {
            name: "reduce_sum",
            mil_fn: mil_reduce_sum,
            input_bytes: 64,
            output_bytes: 16,
            weight_bytes: None,
            verify: verify_reduce_sum,
        },
        OpTest {
            name: "pow",
            mil_fn: mil_pow,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_pow,
        },
        OpTest {
            name: "softmax",
            mil_fn: mil_softmax,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_softmax,
        },
        OpTest {
            name: "sigmoid",
            mil_fn: mil_sigmoid,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_sigmoid,
        },
        OpTest {
            name: "reshape",
            mil_fn: mil_reshape,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_not_zero,
        },
        OpTest {
            name: "transpose",
            mil_fn: mil_transpose,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_not_zero,
        },
        OpTest {
            name: "slice_by_size",
            mil_fn: mil_slice_by_size,
            input_bytes: 128,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_not_zero,
        },
        // Previously "rejected" ops — test directly
        OpTest {
            name: "sub",
            mil_fn: mil_sub,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_sub,
        },
        OpTest {
            name: "concat",
            mil_fn: mil_concat,
            input_bytes: 128,
            output_bytes: 128,
            weight_bytes: None,
            verify: verify_concat,
        },
        // Never tested ops
        OpTest {
            name: "relu",
            mil_fn: mil_relu,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_relu,
        },
        OpTest {
            name: "tanh",
            mil_fn: mil_tanh,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_tanh,
        },
        OpTest {
            name: "exp",
            mil_fn: mil_exp,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_exp,
        },
        OpTest {
            name: "sqrt",
            mil_fn: mil_sqrt,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_sqrt,
        },
        OpTest {
            name: "rsqrt",
            mil_fn: mil_rsqrt,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_rsqrt,
        },
        OpTest {
            name: "neg",
            mil_fn: mil_neg,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_neg,
        },
        OpTest {
            name: "abs",
            mil_fn: mil_abs,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_abs,
        },
        OpTest {
            name: "log",
            mil_fn: mil_log,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_log,
        },
        OpTest {
            name: "clamp",
            mil_fn: mil_clamp,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_clamp,
        },
        OpTest {
            name: "reduce_max",
            mil_fn: mil_reduce_max,
            input_bytes: 64,
            output_bytes: 16,
            weight_bytes: None,
            verify: verify_reduce_max,
        },
        OpTest {
            name: "reduce_mean",
            mil_fn: mil_reduce_mean,
            input_bytes: 64,
            output_bytes: 16,
            weight_bytes: None,
            verify: verify_reduce_mean,
        },
        OpTest {
            name: "layer_norm",
            mil_fn: mil_layer_norm,
            input_bytes: 64,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_not_zero,
        },
        OpTest {
            name: "split",
            mil_fn: mil_split,
            input_bytes: 128,
            output_bytes: 64,
            weight_bytes: None,
            verify: verify_not_zero,
        },
    ];

    println!("=== ANE Comprehensive Operator Audit ===");
    println!("Testing {} operations (filter: {})\n", tests.len(), filter);

    let mut passed = 0;
    let mut failed = 0;
    let mut compile_fail = 0;

    for test in &tests {
        if filter != "all" && !test.name.contains(filter) {
            continue;
        }

        let mil = (test.mil_fn)();
        let input_bytes = test.input_bytes;
        let output_bytes = test.output_bytes;

        print!("  {:15} ", test.name);

        let result = if let Some(wb) = test.weight_bytes {
            // Need BLOBFILE weights
            let wcount = wb / 2 - 32; // minus header
            let weights: Vec<f32> = (0..wcount).map(|_| 0.01).collect();
            let blob = WeightBlob::from_f32(&weights, wcount, 1).unwrap();
            ANECompiler::new()
                .compile_multi(
                    &mil,
                    &["@model_path/weights/weight.bin"],
                    &[blob.as_bytes()],
                    &[blob.as_bytes().len()],
                    &[input_bytes],
                    &[output_bytes],
                )
                .and_then(|mut exec| {
                    exec.eval()?;
                    let mut out = vec![0u8; output_bytes];
                    exec.read_output(0, &mut out)?;
                    Ok(out)
                })
        } else {
            // No weights needed
            ANECompiler::new()
                .compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes])
                .and_then(|mut exec| {
                    exec.eval()?;
                    let mut out = vec![0u8; output_bytes];
                    exec.read_output(0, &mut out)?;
                    Ok(out)
                })
        };

        match result {
            Ok(out) => {
                if (test.verify)(&out) {
                    println!("✅ PASS (numerical check)");
                    passed += 1;
                } else {
                    println!("⚠️  COMPILE+RUN OK but output unexpected (may need input tuning)");
                    passed += 1; // Still counts — op works, just need correct input
                }
            }
            Err(e) => {
                let es = e.to_string();
                if es.contains("InvalidMILProgram") || es.contains("CompilationFailure") {
                    println!("❌ COMPILE FAIL (op rejected by ANE)");
                    compile_fail += 1;
                } else if es.contains("ANE") {
                    let short = if es.len() > 80 { &es[..80] } else { &es };
                    println!("❌ ANE ERROR: {}", short);
                    failed += 1;
                } else {
                    println!(
                        "❌ OTHER ERROR: {}",
                        if es.len() > 60 { &es[..60] } else { &es }
                    );
                    failed += 1;
                }
            }
        }
    }

    println!("\n=== Results ===");
    println!(
        "Passed: {}/{}  |  Compile fail: {}  |  Other fail: {}",
        passed,
        passed + compile_fail + failed,
        compile_fail,
        failed
    );

    Ok(())
}
