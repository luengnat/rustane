use half::f16;
pub use serde_json::json;
use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

pub fn from_fp16_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn run_python_value(script: &str, payload: &Value) -> Value {
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn python3");

    {
        let stdin = child.stdin.as_mut().expect("missing python stdin");
        stdin
            .write_all(payload.to_string().as_bytes())
            .expect("failed to write python payload");
    }

    let output = child.wait_with_output().expect("failed to wait on python");
    if !output.status.success() {
        panic!(
            "python helper failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice::<Value>(&output.stdout).expect("python output was not valid JSON")
}

pub fn run_python_vec(script: &str, payload: &Value) -> Vec<f32> {
    serde_json::from_value(run_python_value(script, payload))
        .expect("python output was not a JSON float array")
}

pub fn mps_is_available() -> bool {
    let script = r#"
import json
import torch
print(json.dumps(bool(torch.backends.mps.is_available())))
"#;
    run_python_value(script, &json!({}))
        .as_bool()
        .unwrap_or(false)
}

pub fn assert_close(expected: &[f32], actual: &[f32], tol: f32) {
    assert_eq!(expected.len(), actual.len(), "length mismatch");
    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (a, b) in expected.iter().zip(actual.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        mean_diff += diff;
    }
    mean_diff /= expected.len() as f32;
    assert!(
        max_diff <= tol,
        "max diff {} exceeded tolerance {} (mean diff {})",
        max_diff,
        tol,
        mean_diff
    );
}

pub fn build_fused_qkv_mil(dim: usize, seq: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq}]> x) {{\n"
    ));
    mil.push_str("        string d1 = const()[name = string(\"d1\"), val = string(\"fp16\")];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = d1, x = x)[name = string(\"cx\")];\n"
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    for name in ["Wq", "Wk", "Wv"] {
        mil.push_str(&format!(
            "        tensor<fp16, [{dim}, {dim}, 1, 1]> {name} = const()[name = string(\"{name}\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{lower}.bin\"), offset = uint64(64)))];\n",
            dim = dim,
            name = name,
            lower = name.to_lowercase()
        ));
    }
    for (out, weight) in [("q", "Wq"), ("k", "Wk"), ("v", "Wv")] {
        mil.push_str(&format!(
            "        tensor<fp16, [1, {dim}, 1, {seq}]> {out} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {weight}, x = x16)[name = string(\"c{out}\")];\n",
            dim = dim,
            seq = seq,
            out = out,
            weight = weight
        ));
    }
    mil.push_str("        int32 ax = const()[name = string(\"ax\"), val = int32(1)];\n");
    mil.push_str("        bool inter = const()[name = string(\"il\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> qkv = concat(axis = ax, interleave = inter, values = (q, k, v))[name = string(\"cat\")];\n",
        dim * 3,
        seq
    ));
    mil.push_str("        string d2 = const()[name = string(\"d2\"), val = string(\"fp32\")];\n");
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = d2, x = qkv)[name = string(\"co\")];\n",
        dim * 3,
        seq
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

pub fn build_qkt_mil(heads: usize, seq: usize, hd: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {heads}, {seq}, {hd}]> q, tensor<fp16, [1, {heads}, {seq}, {hd}]> k) {{\n"
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str("        bool bT = const()[name = string(\"bT\"), val = bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {heads}, {seq}, {seq}]> scores = matmul(transpose_x = bF, transpose_y = bT, x = q, y = k)[name = string(\"qkt\")];\n"
    ));
    mil.push_str("    } -> (scores);\n");
    mil.push_str("}\n");
    mil
}

pub fn build_sv_mil(heads: usize, seq: usize, hd: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {heads}, {seq}, {seq}]> probs, tensor<fp16, [1, {heads}, {seq}, {hd}]> v) {{\n"
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {heads}, {seq}, {hd}]> out = matmul(transpose_x = bF, transpose_y = bF, x = probs, y = v)[name = string(\"sv\")];\n"
    ));
    mil.push_str("    } -> (out);\n");
    mil.push_str("}\n");
    mil
}
