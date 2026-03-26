//! Systematic ANE operator test — understand what works and what crashes.
//! Tests ONE operation at ONE size per invocation (designed for subprocess use).
//!
//! Usage:
//!   cargo run --example test_size_sensitivity cast 1 64     # cast [1,1,1,64]
//!   cargo run --example test_size_sensitivity add 64 64    # add [1,64,1,64]
//!   cargo run --example test_size_sensitivity matmul 64 64 # matmul dim=64 seq=64

use rustane::wrapper::ANECompiler;

fn mil_header() -> String {
    let mut s = String::new();
    s.push_str("program(1.3)\n");
    s.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    s.push_str("{\n");
    s
}

fn main() {
    if let Err(e) = rustane::init() {
        eprintln!("INIT_FAIL: {}", e);
        std::process::exit(1);
    }

    let args: Vec<String> = std::env::args().collect();
    let op = if args.len() > 1 { &args[1] } else { "cast" };
    let a: usize = if args.len() > 2 {
        args[2].parse().unwrap_or(64)
    } else {
        64
    };
    let b: usize = if args.len() > 3 {
        args[3].parse().unwrap_or(64)
    } else {
        64
    };

    let (mil, in_bytes, out_bytes) = match op {
        "cast" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c1\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = xh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "add" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yh = add(x = xh, y = one)[name = string(\"a\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "mul" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str("        fp16 two = const()[name = string(\"two\"), val = fp16(2.0)];\n");
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yh = mul(x = xh, y = two)[name = string(\"m\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "relu" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(&format!(
                "        tensor<fp16, [1, {}, 1, {}]> yh = relu(x = xh)[name = string(\"r\")];\n",
                ch, w
            ));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "tanh" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(&format!(
                "        tensor<fp16, [1, {}, 1, {}]> yh = tanh(x = xh)[name = string(\"t\")];\n",
                ch, w
            ));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "sigmoid" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yh = sigmoid(x = xh)[name = string(\"sig\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "exp" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(&format!(
                "        tensor<fp16, [1, {}, 1, {}]> yh = exp(x = xh)[name = string(\"e\")];\n",
                ch, w
            ));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "softmax" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([3])];\n");
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yh = softmax(x = xh, axis = ax)[name = string(\"sm\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "reduce_mean" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
            m.push_str(&format!("        tensor<fp16, [1, 1, 1, {}]> yh = reduce_mean(x = xh, axes = ax, keep_dims = kd)[name = string(\"rm\")];\n", w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, 1, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, 1 * 1 * w * 4) // output is reduced
        }

        "layer_norm" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(
                "        fp16 eps = const()[name = string(\"eps\"), val = fp16(0.00001)];\n",
            );
            m.push_str("        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yh = layer_norm(x = xh, axes = ax, epsilon = eps)[name = string(\"ln\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "gelu" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(&format!(
                "        tensor<fp16, [1, {}, 1, {}]> yh = gelu(x = xh)[name = string(\"g\")];\n",
                ch, w
            ));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "silu" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str(&format!(
                "        tensor<fp16, [1, {}, 1, {}]> yh = silu(x = xh)[name = string(\"si\")];\n",
                ch, w
            ));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        "sub" => {
            let (ch, w) = (a, b);
            let mut m = mil_header();
            m.push_str(&format!(
                "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
                ch, w
            ));
            m.push_str(
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
            );
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"c\")];\n", ch, w));
            m.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
            m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yh = sub(x = xh, y = one)[name = string(\"s\")];\n", ch, w));
            m.push_str(
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
            );
            m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", ch, w));
            m.push_str("    } -> (y);\n}\n");
            (m, ch * w * 4, ch * w * 4)
        }

        _ => {
            eprintln!("Unknown op: {}", op);
            std::process::exit(1);
        }
    };

    // Print what we're testing for the shell script
    eprintln!("TEST: {} [1, {}, 1, {}] ({}B in)", op, a, b, in_bytes);

    match ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[in_bytes], &[out_bytes]) {
        Ok(mut exec) => {
            eprintln!("COMPILE_OK");
            match exec.eval() {
                Ok(_) => {
                    let mut out = vec![0u8; out_bytes];
                    exec.read_output(0, &mut out).ok();
                    let nonzero = out.iter().any(|&b| b != 0);
                    if nonzero {
                        println!("{}:PASS", op);
                    } else {
                        println!("{}:WARN:zero_output", op);
                    }
                }
                Err(e) => {
                    println!("{}:FAIL:eval_error", op);
                    eprintln!("EVAL_ERR: {}", e);
                }
            }
        }
        Err(e) => {
            let es = e.to_string();
            if es.contains("InvalidMILProgram") || es.contains("CompilationFailure") {
                println!("{}:FAIL:compile_rejected", op);
            } else {
                println!("{}:FAIL:compile_error", op);
                eprintln!("COMPILE_ERR: {}", &es[..es.len().min(120)]);
            }
        }
    }
}
