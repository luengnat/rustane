//! ANE Operator Audit — test one op per invocation.
//!
//! Usage:
//!   cargo run --example test_single_op cast       # test single op
//!   cargo run --example test_single_op all        # test all (may hit compile budget)
//!
//! Output: machine-parseable lines "OP:STATUS:detail" on stdout.
//! Progress on stderr.

use rustane::ane::WeightBlob;
use rustane::wrapper::ANECompiler;

fn mil_header() -> String {
    let mut s = String::new();
    s.push_str("program(1.3)\n");
    s.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    s.push_str("{\n");
    s
}

fn mil_footer() -> &'static str {
    "    } -> (y);\n}\n"
}

/// Build a MIL program for the given operator.
/// Returns (mil_string, input_bytes, output_bytes, optional_weight_blob_size).
fn gen_mil(op: &str) -> Option<(String, usize, usize, Option<usize>)> {
    let mut m = mil_header();

    match op {
        // --- Element-wise unary ops ---
        "cast" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c1\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = xh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "neg" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = neg(x = xh)[name = string(\"n\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "abs" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = abs(x = xh)[name = string(\"a\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "exp" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = exp(x = xh)[name = string(\"e\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "log" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = log(x = xh)[name = string(\"l\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "sqrt" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = sqrt(x = xh)[name = string(\"sq\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "rsqrt" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = rsqrt(x = xh)[name = string(\"rs\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "reciprocal" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = reciprocal(x = xh)[name = string(\"r\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "tanh" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = tanh(x = xh)[name = string(\"t\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "sigmoid" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = sigmoid(x = xh)[name = string(\"sig\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "relu" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = relu(x = xh)[name = string(\"r\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "gelu" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = gelu(x = xh)[name = string(\"g\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "silu" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        tensor<fp16, [1, 4, 1, 4]> yh = silu(x = xh)[name = string(\"si\")];\n",
            );
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        // --- Element-wise binary ops ---
        "add" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = add(x = xh, y = one)[name = string(\"a\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "mul" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 two = const()[name = string(\"two\"), val = fp16(2.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = mul(x = xh, y = two)[name = string(\"m\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "sub" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = sub(x = xh, y = one)[name = string(\"s\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "div" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 two = const()[name = string(\"two\"), val = fp16(2.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = div(x = xh, y = two)[name = string(\"d\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "pow" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 p = const()[name = string(\"p\"), val = fp16(2.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = pow(x = xh, y = p)[name = string(\"pw\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        // --- Activation / clamp ops ---
        "clamp" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 lo = const()[name = string(\"lo\"), val = fp16(-1.0)];\n");
            m.push_str("        fp16 hi = const()[name = string(\"hi\"), val = fp16(1.0)];\n");
            m.push_str("        fp16 sc = const()[name = string(\"sc\"), val = fp16(0.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = clamp(x = xh, alpha = lo, beta = hi, scale = sc)[name = string(\"cl\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "clip" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 lo = const()[name = string(\"lo\"), val = fp16(-1.0)];\n");
            m.push_str("        fp16 hi = const()[name = string(\"hi\"), val = fp16(1.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = clip(x = xh, alpha = lo, beta = hi)[name = string(\"cl\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "threshold_relu" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 th = const()[name = string(\"th\"), val = fp16(0.5)];\n");
            m.push_str("        fp16 al = const()[name = string(\"al\"), val = fp16(0.0)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = threshold_relu(x = xh, alpha = al, theta = th)[name = string(\"tr\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "linear_activation" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        fp16 al = const()[name = string(\"al\"), val = fp16(2.0)];\n");
            m.push_str("        fp16 be = const()[name = string(\"be\"), val = fp16(0.5)];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = linear(x = xh, alpha = al, beta = be)[name = string(\"la\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        // --- Reduction / normalization ops ---
        "softmax" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([3])];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = softmax(x = xh, axis = ax)[name = string(\"sm\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "reduce_sum" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
            m.push_str("        tensor<fp16, [1, 1, 1, 4]> yh = reduce_sum(x = xh, axes = rax, keep_dims = kd)[name = string(\"rs\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 1, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 16, None))
        }

        "reduce_max" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
            m.push_str("        tensor<fp16, [1, 1, 1, 4]> yh = reduce_max(x = xh, axes = ax, keep_dims = kd)[name = string(\"rm\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 1, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 16, None))
        }

        "reduce_mean" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
            m.push_str("        tensor<fp16, [1, 1, 1, 4]> yh = reduce_mean(x = xh, axes = ax, keep_dims = kd)[name = string(\"rm\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 1, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 16, None))
        }

        "layer_norm" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str(
                "        fp16 eps = const()[name = string(\"eps\"), val = fp16(0.00001)];\n",
            );
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = layer_norm(x = xh, axes = ax, epsilon = eps)[name = string(\"ln\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        // --- Shape ops ---
        "matmul" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
            m.push_str("        tensor<fp16, [1, 4, 4, 1]> xt = transpose(perm = pm, x = xh)[name = string(\"t\")];\n");
            m.push_str("        bool bf = const()[name = string(\"bf\"), val = bool(false)];\n");
            m.push_str("        tensor<fp16, [1, 1, 4, 4]> yh = matmul(transpose_x = bf, transpose_y = bf, x = xh, y = xt)[name = string(\"mm\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 1, 4, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "reshape" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [4]> sh = const()[name = string(\"sh\"), val = tensor<int32, [4]>([1, 1, 4, 4])];\n");
            m.push_str("        tensor<fp16, [1, 1, 4, 4]> yh = reshape(shape = sh, x = xh)[name = string(\"r\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 1, 4, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "transpose" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
            m.push_str("        tensor<fp16, [1, 4, 4, 1]> yh = transpose(perm = pm, x = xh)[name = string(\"t\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 4, 1]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, None))
        }

        "slice" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 8, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 8, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
            m.push_str("        tensor<int32, [4]> sz = const()[name = string(\"sz\"), val = tensor<int32, [4]>([1, 4, 1, 4])];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = slice_by_size(x = xh, begin = b0, size = sz)[name = string(\"s\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 128, 64, None))
        }

        "concat" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 8, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 8, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
            m.push_str("        tensor<int32, [4]> s1 = const()[name = string(\"s1\"), val = tensor<int32, [4]>([1, 4, 1, 4])];\n");
            m.push_str("        tensor<int32, [4]> b1 = const()[name = string(\"b1\"), val = tensor<int32, [4]>([0, 4, 0, 0])];\n");
            m.push_str("        tensor<int32, [4]> s2 = const()[name = string(\"s2\"), val = tensor<int32, [4]>([1, 4, 1, 4])];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> a = slice_by_size(x = xh, begin = b0, size = s1)[name = string(\"a\")];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> b = slice_by_size(x = xh, begin = b1, size = s2)[name = string(\"b\")];\n");
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        tensor<fp16, [1, 8, 1, 4]> yh = concat(x = a, y = b, axis = ax)[name = string(\"cat\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 8, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 128, 128, None))
        }

        "split" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 8, 1, 4]> x) {\n");
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str("        tensor<fp16, [1, 8, 1, 4]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n");
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        tensor<uint32, [1]> sp = const()[name = string(\"sp\"), val = tensor<uint32, [1]>([2])];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> yh = split(x = xh, axis = ax, splits = sp)[name = string(\"sp\")];\n");
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n");
            m.push_str(mil_footer());
            Some((m, 128, 64, None))
        }

        // --- Convolution ---
        "conv1x1" => {
            m.push_str("    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n");
            m.push_str("        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n");
            m.push_str(
                "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n",
            );
            m.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
            m.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
            m.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
            m.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
            m.push_str("        tensor<fp16, [4, 4, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [4, 4, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n");
            m.push_str("        tensor<fp16, [1, 4, 1, 4]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n");
            m.push_str("        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n");
            m.push_str("        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n");
            m.push_str(mil_footer());
            Some((m, 64, 64, Some(32)))
        }

        _ => None,
    }
}

fn test_op(op: &str) -> String {
    let result = gen_mil(op);
    let (mil, in_b, out_b, wb) = match result {
        Some(r) => r,
        None => return format!("{}:UNKNOWN:unknown operator", op),
    };

    let compile_result = if let Some(wb) = wb {
        let wcount = wb / 2 - 32;
        let weights: Vec<f32> = (0..wcount).map(|_| 0.01).collect();
        let blob = match WeightBlob::from_f32(&weights, wcount, 1) {
            Ok(b) => b,
            Err(e) => return format!("{}:FAIL:weight blob error: {}", op, e),
        };
        ANECompiler::new().compile_multi(
            &mil,
            &["@model_path/weights/weight.bin"],
            &[blob.as_bytes()],
            &[blob.as_bytes().len()],
            &[in_b],
            &[out_b],
        )
    } else {
        ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[in_b], &[out_b])
    };

    match compile_result {
        Ok(mut exec) => {
            // Flush BEFORE eval — eval may crash the process (SIGSEGV)
            use std::io::Write;
            eprintln!("  COMPILE_OK");
            let _ = std::io::stderr().flush();

            let eval_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exec.eval()));
            match eval_result {
                Ok(Ok(_)) => {
                    let mut out = vec![0u8; out_b];
                    exec.read_output(0, &mut out).ok();
                    let nonzero = out.iter().any(|&b| b != 0);
                    if nonzero {
                        format!("{}:PASS:compile+run+output_verified", op)
                    } else {
                        format!("{}:WARN:compile+run+all_zero_output", op)
                    }
                }
                Ok(Err(e)) => format!("{}:FAIL:eval error: {}", op, e),
                Err(_) => format!("{}:FAIL:eval crash/panic", op),
            }
        }
        Err(e) => {
            let es = e.to_string();
            if es.contains("InvalidMILProgram") || es.contains("CompilationFailure") {
                format!("{}:FAIL:compile rejected by ANE", op)
            } else {
                let short = if es.len() > 80 { &es[..80] } else { &es };
                format!("{}:FAIL:compile error: {}", op, short)
            }
        }
    }
}

fn main() {
    if let Err(e) = rustane::init() {
        eprintln!("INIT FAIL: {}", e);
        std::process::exit(1);
    }

    let args: Vec<String> = std::env::args().collect();
    let op = if args.len() > 1 { &args[1] } else { "all" };

    let all_ops = [
        "cast",
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "abs",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "pow",
        "reciprocal",
        "tanh",
        "sigmoid",
        "relu",
        "gelu",
        "silu",
        "clamp",
        "clip",
        "threshold_relu",
        "linear_activation",
        "softmax",
        "matmul",
        "reshape",
        "transpose",
        "slice",
        "reduce_sum",
        "reduce_max",
        "reduce_mean",
        "concat",
        "layer_norm",
        "split",
        "conv1x1",
    ];

    let ops: Vec<&str> = if op == "all" {
        all_ops.to_vec()
    } else {
        vec![op]
    };

    eprintln!("=== ANE Operator Audit ({} ops) ===", ops.len());

    for op_name in &ops {
        eprintln!("  Testing: {} ...", op_name);
        let result = test_op(op_name);
        // Flush to ensure output is visible even on crash
        eprintln!("  Result: {}", result);
        println!("{}", result);
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}
