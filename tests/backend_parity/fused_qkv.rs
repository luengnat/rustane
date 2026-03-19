use crate::common::*;
use rustane::{init, require_ane};
use std::process::Command;

fn python_mlx_fused_qkv(dim: usize, seq: usize, input: &[f32]) -> Vec<f32> {
    let script = format!(
        r#"
import json
import numpy as np
import mlx.core as mx

payload = json.load(__import__("sys").stdin)
x = np.array(payload["input"], dtype=np.float32).astype(np.float16).astype(np.float32)
w = np.eye({dim}, dtype=np.float32).astype(np.float16)
x = mx.array(x.reshape({dim}, {seq}), dtype=mx.float16)
w = mx.array(w, dtype=mx.float16)

q = mx.matmul(w, x)
k = mx.matmul(w, x)
v = mx.matmul(w, x)
y = mx.concatenate([q, k, v], axis=0)
mx.eval(y)
print(json.dumps(np.asarray(y, dtype=np.float32).reshape(-1).tolist()))
"#,
        dim = dim,
        seq = seq
    );
    run_python_vec(&script, &json!({ "input": input }))
}

fn python_mps_fused_qkv(dim: usize, seq: usize, input: &[f32]) -> Option<Vec<f32>> {
    if !mps_is_available() {
        return None;
    }

    let script = format!(
        r#"
import json
import numpy as np
import torch

payload = json.load(__import__("sys").stdin)
x = np.array(payload["input"], dtype=np.float32).astype(np.float16).astype(np.float32)
x = torch.tensor(x.reshape({dim}, {seq}), dtype=torch.float32, device="mps")
w = torch.eye({dim}, dtype=torch.float32, device="mps")

q = torch.matmul(w, x)
k = torch.matmul(w, x)
v = torch.matmul(w, x)
y = torch.cat([q, k, v], dim=0)
print(json.dumps(y.detach().cpu().reshape(-1).tolist()))
"#,
        dim = dim,
        seq = seq
    );
    Some(run_python_vec(&script, &json!({ "input": input })))
}

#[test]
fn fused_qkv_matches_mlx_and_mps() {
    let _guard = lock_tests();
    require_ane!();
    init().expect("ANE init");

    let dim = 64;
    let seq = 32;
    let input: Vec<f32> = (0..dim * seq).map(|i| i as f32).collect();

    let mil = build_fused_qkv_mil(dim, seq);
    assert!(mil.contains("conv"));
    assert!(mil.contains("concat"));

    let example = Command::new("cargo")
        .args(["run", "--example", "fused_qkv", "--quiet"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run fused_qkv example");
    assert!(
        example.status.success(),
        "fused_qkv example failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&example.stdout),
        String::from_utf8_lossy(&example.stderr)
    );
    let stdout = String::from_utf8_lossy(&example.stdout);
    assert!(
        stdout.contains("6144 / 6144 values correct"),
        "unexpected fused_qkv output:\n{}",
        stdout
    );

    let mlx = python_mlx_fused_qkv(dim, seq, &input);
    let mut expected = Vec::with_capacity(dim * seq * 3);
    expected.extend_from_slice(&input);
    expected.extend_from_slice(&input);
    expected.extend_from_slice(&input);
    assert_close(&expected, &mlx, 0.01);

    if let Some(mps) = python_mps_fused_qkv(dim, seq, &input) {
        assert_close(&expected, &mps, 0.01);
    }
}
