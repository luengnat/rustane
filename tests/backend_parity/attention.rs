use crate::common::*;
use rustane::{init, require_ane, ANECompiler, ANETensor};

fn python_mlx_causal_attention(
    heads: usize,
    seq: usize,
    hd: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let script = format!(
        r#"
import json
import numpy as np
import mlx.core as mx

payload = json.load(__import__("sys").stdin)
heads = {heads}
seq = {seq}
hd = {hd}
scale = 1.0 / np.sqrt(hd)
q = np.array(payload["q"], dtype=np.float32).astype(np.float16).astype(np.float32).reshape(heads, seq, hd)
k = np.array(payload["k"], dtype=np.float32).astype(np.float16).astype(np.float32).reshape(heads, seq, hd)
v = np.array(payload["v"], dtype=np.float32).astype(np.float16).astype(np.float32).reshape(heads, seq, hd)

q = mx.array(q, dtype=mx.float16)
k = mx.array(k, dtype=mx.float16)
v = mx.array(v, dtype=mx.float16)
scores = mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale
mask = np.triu(np.full((seq, seq), -1.0e9, dtype=np.float32), k=1)
scores = scores + mx.array(mask, dtype=mx.float16)
probs = mx.softmax(scores, axis=-1)
out = mx.matmul(probs, v)
mx.eval(out)
print(json.dumps(np.asarray(out, dtype=np.float32).reshape(-1).tolist()))
"#,
        heads = heads,
        seq = seq,
        hd = hd
    );
    run_python_vec(&script, &json!({ "q": q, "k": k, "v": v }))
}

fn python_mps_causal_attention(
    heads: usize,
    seq: usize,
    hd: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Option<Vec<f32>> {
    if !mps_is_available() {
        return None;
    }

    let script = format!(
        r#"
import json
import numpy as np
import torch

payload = json.load(__import__("sys").stdin)
heads = {heads}
seq = {seq}
hd = {hd}
scale = 1.0 / np.sqrt(hd)
q = np.array(payload["q"], dtype=np.float32).astype(np.float16).astype(np.float32).reshape(heads, seq, hd)
k = np.array(payload["k"], dtype=np.float32).astype(np.float16).astype(np.float32).reshape(heads, seq, hd)
v = np.array(payload["v"], dtype=np.float32).astype(np.float16).astype(np.float32).reshape(heads, seq, hd)

q = torch.tensor(q, dtype=torch.float32, device="mps")
k = torch.tensor(k, dtype=torch.float32, device="mps")
v = torch.tensor(v, dtype=torch.float32, device="mps")
scores = torch.matmul(q, k.transpose(-1, -2)) * scale
mask = torch.triu(torch.full((seq, seq), -1.0e9, dtype=torch.float32, device="mps"), diagonal=1)
scores = scores + mask
probs = torch.softmax(scores, dim=-1)
out = torch.matmul(probs, v)
print(json.dumps(out.detach().cpu().reshape(-1).tolist()))
"#,
        heads = heads,
        seq = seq,
        hd = hd
    );
    Some(run_python_vec(&script, &json!({ "q": q, "k": k, "v": v })))
}

fn build_qkt_mil(heads: usize, seq: usize, hd: usize) -> String {
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

fn build_sv_mil(heads: usize, seq: usize, hd: usize) -> String {
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

fn ane_causal_attention(
    heads: usize,
    seq: usize,
    hd: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let qkt_mil = build_qkt_mil(heads, seq, hd);
    let sv_mil = build_sv_mil(heads, seq, hd);
    assert!(qkt_mil.contains("matmul"));
    assert!(sv_mil.contains("matmul"));

    let q16 = to_fp16_bits(q);
    let k16 = to_fp16_bits(k);
    let v16 = to_fp16_bits(v);
    let q_tensor = ANETensor::from_fp16(q16.clone(), vec![1, heads, seq, hd]).expect("q tensor");
    let k_tensor = ANETensor::from_fp16(k16.clone(), vec![1, heads, seq, hd]).expect("k tensor");
    let v_tensor = ANETensor::from_fp16(v16.clone(), vec![1, heads, seq, hd]).expect("v tensor");

    let qkt_bytes = heads * seq * seq * 2;
    let io_bytes = heads * seq * hd * 2;

    let mut qkt_compiler = ANECompiler::new();
    let mut qkt_exec = qkt_compiler
        .compile_single(&qkt_mil, None, &[io_bytes, io_bytes], &[qkt_bytes])
        .expect("compile qkt");

    let mut sv_compiler = ANECompiler::new();
    let mut sv_exec = sv_compiler
        .compile_single(&sv_mil, None, &[qkt_bytes, io_bytes], &[io_bytes])
        .expect("compile sv");

    qkt_exec
        .write_input(0, q_tensor.as_bytes())
        .expect("q input");
    qkt_exec
        .write_input(1, k_tensor.as_bytes())
        .expect("k input");
    qkt_exec.eval().expect("qkt eval");

    let mut scores_buf = vec![0u8; qkt_bytes];
    qkt_exec.read_output(0, &mut scores_buf).expect("read qkt");
    let mut probs = from_fp16_bytes(&scores_buf);

    let scale = 1.0f32 / (hd as f32).sqrt();
    for h in 0..heads {
        for t in 0..seq {
            let row = &mut probs[h * seq * seq + t * seq..h * seq * seq + (t + 1) * seq];
            let mut max_logit = f32::NEG_INFINITY;
            for t2 in 0..seq {
                if t2 > t {
                    row[t2] = -1.0e30;
                } else {
                    row[t2] *= scale;
                }
                if row[t2] > max_logit {
                    max_logit = row[t2];
                }
            }
            let mut denom = 0.0f32;
            for t2 in 0..seq {
                if t2 > t {
                    row[t2] = 0.0;
                } else {
                    row[t2] = (row[t2] - max_logit).exp();
                    denom += row[t2];
                }
            }
            for t2 in 0..=t {
                row[t2] /= denom;
            }
        }
    }
    let probs_tensor =
        ANETensor::from_fp16(to_fp16_bits(&probs), vec![1, heads, seq, seq]).expect("probs tensor");

    sv_exec
        .write_input(0, probs_tensor.as_bytes())
        .expect("probs input");
    sv_exec
        .write_input(1, v_tensor.as_bytes())
        .expect("v input");
    sv_exec.eval().expect("sv eval");

    let mut out_buf = vec![0u8; io_bytes];
    sv_exec.read_output(0, &mut out_buf).expect("read sv");
    from_fp16_bytes(&out_buf)
}

#[test]
fn causal_attention_matches_mlx_and_mps() {
    let _guard = lock_tests();
    require_ane!();
    init().expect("ANE init");

    let heads = 12;
    let seq = 64;
    let hd = 64;
    let mut q = vec![0.0f32; heads * seq * hd];
    let mut k = vec![0.0f32; heads * seq * hd];
    let mut v = vec![0.0f32; heads * seq * hd];
    for (i, item) in q.iter_mut().enumerate() {
        *item = ((i as f32 * 0.17).sin() * 0.2) + 0.03;
    }
    for (i, item) in k.iter_mut().enumerate() {
        *item = ((i as f32 * 0.13).cos() * 0.2) - 0.04;
    }
    for (i, item) in v.iter_mut().enumerate() {
        *item = ((i as f32 * 0.11).sin() * 0.15) + 0.02;
    }

    let mlx = python_mlx_causal_attention(heads, seq, hd, &q, &k, &v);
    let ane = ane_causal_attention(heads, seq, hd, &q, &k, &v);
    assert_close(&mlx, &ane, 0.01);

    if let Some(mps) = python_mps_causal_attention(heads, seq, hd, &q, &k, &v) {
        assert_close(&mlx, &mps, 0.01);
    }
}
