//! Transformer layer proof of life
//!
//! Runs a tiny pre-norm transformer layer through ANE:
//! 1. QKV projection via 1x1 conv
//! 2. scaled_dot_product_attention
//! 3. Output projection via 1x1 conv
//! 4. Gated FFN via 1x1 convs
//!
//! The goal is not model quality. The goal is to verify that a full
//! transformer-shaped layer still holds together on ANE when we combine the
//! already-verified attention and FFN paths.

use half::f16;
use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};

const HEADS: usize = 12;
const SEQ: usize = 32;
const HEAD_DIM: usize = 64;
const EMBED_DIM: usize = HEADS * HEAD_DIM;
const QKV_DIM: usize = EMBED_DIM * 3;
const BATCH: usize = 1;

fn build_info() -> &'static str {
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]"
}

fn qkv_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {e}, 1, {s}]> x) {{\n",
        e = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {e}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        e = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    for name in ["Wq", "Wk", "Wv"] {
        mil.push_str(&format!(
            "        tensor<fp16, [{e}, {e}, 1, 1]> {n} = const()[name = string(\"{n}\"), val = tensor<fp16, [{e}, {e}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{file}.bin\"), offset = uint64(64)))];\n",
            e = EMBED_DIM,
            n = name,
            file = name.to_lowercase(),
        ));
    }
    for (out_name, weight_name) in [("q", "Wq"), ("k", "Wk"), ("v", "Wv")] {
        mil.push_str(&format!(
            "        tensor<fp16, [1, {e}, 1, {s}]> {out} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {w}, x = x16)[name = string(\"c{out}\")];\n",
            e = EMBED_DIM,
            s = SEQ,
            out = out_name,
            w = weight_name,
        ));
    }
    mil.push_str("        int32 ax = const()[name = string(\"ax\"), val = int32(1)];\n");
    mil.push_str("        bool inter = const()[name = string(\"il\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {qkv}, 1, {s}]> qkv = concat(axis = ax, interleave = inter, values = (q, k, v))[name = string(\"cat\")];\n",
        qkv = QKV_DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {qkv}, 1, {s}]> y = cast(dtype = to_fp32, x = qkv)[name = string(\"cast_out\")];\n",
        qkv = QKV_DIM,
        s = SEQ,
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn sdpa_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {h}, {s}, {d}]> q, tensor<fp16, [1, {h}, {s}, {d}]> k, tensor<fp16, [1, {h}, {s}, {d}]> v) {{\n",
        h = HEADS,
        s = SEQ,
        d = HEAD_DIM,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {h}, {s}, {d}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n",
        h = HEADS,
        s = SEQ,
        d = HEAD_DIM,
    ));
    mil.push_str("    } -> (att);\n");
    mil.push_str("}\n");
    mil
}

fn out_proj_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {e}, 1, {s}]> x) {{\n",
        e = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {e}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        e = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{e}, {e}, 1, 1]> W = const()[name = string(\"Wout\"), val = tensor<fp16, [{e}, {e}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        e = EMBED_DIM,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {e}, 1, {s}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"proj\")];\n",
        e = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {e}, 1, {s}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        e = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn ffn_proj_mil(name: &str) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {d}, 1, {s}]> x) {{\n",
        d = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {d}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        d = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{d}, {d}, 1, 1]> W = const()[name = string(\"{name}\"), val = tensor<fp16, [{d}, {d}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        d = EMBED_DIM,
        name = name,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {d}, 1, {s}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"proj\")];\n",
        d = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {d}, 1, {s}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        d = EMBED_DIM,
        s = SEQ,
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn f32_to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn fp32_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn rms_norm(x: &[f32], embed_dim: usize, seq: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..seq {
        let mut ss = 0.0f32;
        for c in 0..embed_dim {
            let v = x[c * seq + t];
            ss += v * v;
        }
        let scale = 1.0f32 / ((ss / embed_dim as f32) + eps).sqrt();
        for c in 0..embed_dim {
            out[c * seq + t] = x[c * seq + t] * scale;
        }
    }
    out
}

fn linear_channels_first(
    input: &[f32],
    in_channels: usize,
    out_channels: usize,
    seq: usize,
    weights: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_channels * seq];
    for o in 0..out_channels {
        for t in 0..seq {
            let mut sum = 0.0f32;
            for i in 0..in_channels {
                sum += weights[o * in_channels + i] * input[i * seq + t];
            }
            out[o * seq + t] = sum;
        }
    }
    out
}

fn split_qkv(qkv: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let span = EMBED_DIM * SEQ;
    (
        qkv[0..span].to_vec(),
        qkv[span..2 * span].to_vec(),
        qkv[2 * span..3 * span].to_vec(),
    )
}

fn to_head_layout(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; HEADS * SEQ * HEAD_DIM];
    for c in 0..EMBED_DIM {
        let h = c / HEAD_DIM;
        let d = c % HEAD_DIM;
        for t in 0..SEQ {
            out[(h * SEQ + t) * HEAD_DIM + d] = x[c * SEQ + t];
        }
    }
    out
}

fn from_head_layout(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; EMBED_DIM * SEQ];
    for c in 0..EMBED_DIM {
        let h = c / HEAD_DIM;
        let d = c % HEAD_DIM;
        for t in 0..SEQ {
            out[c * SEQ + t] = x[(h * SEQ + t) * HEAD_DIM + d];
        }
    }
    out
}

fn cpu_sdpa(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; HEADS * SEQ * HEAD_DIM];
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    for h in 0..HEADS {
        for t in 0..SEQ {
            let mut scores = [0.0f32; SEQ];
            let mut max_score = f32::NEG_INFINITY;
            for t2 in 0..SEQ {
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM {
                    let q_idx = ((h * SEQ + t) * HEAD_DIM) + d;
                    let k_idx = ((h * SEQ + t2) * HEAD_DIM) + d;
                    s += q[q_idx] * k[k_idx];
                }
                s *= scale;
                scores[t2] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut denom = 0.0f32;
            for t2 in 0..SEQ {
                scores[t2] = (scores[t2] - max_score).exp();
                denom += scores[t2];
            }
            for t2 in 0..SEQ {
                scores[t2] /= denom;
            }

            for d in 0..HEAD_DIM {
                let mut val = 0.0f32;
                for t2 in 0..SEQ {
                    let v_idx = ((h * SEQ + t2) * HEAD_DIM) + d;
                    val += scores[t2] * v[v_idx];
                }
                out[((h * SEQ + t) * HEAD_DIM) + d] = val;
            }
        }
    }

    out
}

fn max_mean_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_diff = max_diff.max(d);
        mean_diff += d;
    }
    (max_diff, mean_diff / a.len() as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Transformer Layer Proof of Life");
    println!("===========================================\n");

    let avail = rustane::ANEAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available");
        return Ok(());
    }
    println!();

    init()?;
    println!("✓ ANE runtime initialized\n");

    // Deterministic input.
    let mut x = vec![0.0f32; EMBED_DIM * SEQ];
    for (i, item) in x.iter_mut().enumerate() {
        *item = ((i as f32 * 0.03125).sin() * 0.4) + ((i as f32 * 0.0078125).cos() * 0.2);
    }
    let x_norm = rms_norm(&x, EMBED_DIM, SEQ, 1e-6);

    // Attention weights.
    let mut q = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    let mut k = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    let mut v = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    let mut wo = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    for i in 0..EMBED_DIM {
        q[i * EMBED_DIM + i] = 1.0;
        k[i * EMBED_DIM + i] = 0.5;
        v[i * EMBED_DIM + i] = -0.25;
        wo[i * EMBED_DIM + i] = 1.0;
    }
    let wq = WeightBlob::from_fp32(&q, EMBED_DIM as i32, EMBED_DIM as i32)?;
    let wk = WeightBlob::from_fp32(&k, EMBED_DIM as i32, EMBED_DIM as i32)?;
    let wv = WeightBlob::from_fp32(&v, EMBED_DIM as i32, EMBED_DIM as i32)?;
    let wo_blob = WeightBlob::from_fp32(&wo, EMBED_DIM as i32, EMBED_DIM as i32)?;

    // FFN weights.
    let mut w1 = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    let mut w3 = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    let mut w2 = vec![0.0f32; EMBED_DIM * EMBED_DIM];
    for i in 0..EMBED_DIM {
        w1[i * EMBED_DIM + i] = 1.0;
        w3[i * EMBED_DIM + i] = 0.5;
        w2[i * EMBED_DIM + i] = 1.0;
    }
    let w1_blob = WeightBlob::from_fp32(&w1, EMBED_DIM as i32, EMBED_DIM as i32)?;
    let w3_blob = WeightBlob::from_fp32(&w3, EMBED_DIM as i32, EMBED_DIM as i32)?;
    let w2_blob = WeightBlob::from_fp32(&w2, EMBED_DIM as i32, EMBED_DIM as i32)?;

    // Compile kernels.
    let qkv_mil = qkv_mil();
    let qkv_input_bytes = EMBED_DIM * SEQ * 4;
    let qkv_output_bytes = QKV_DIM * SEQ * 4;
    let mut qkv_compiler = ANECompiler::new();
    let mut qkv_exec = qkv_compiler.compile_multi(
        &qkv_mil,
        &[
            "@model_path/weights/wq.bin",
            "@model_path/weights/wk.bin",
            "@model_path/weights/wv.bin",
        ],
        &[wq.as_bytes(), wk.as_bytes(), wv.as_bytes()],
        &[wq.len(), wk.len(), wv.len()],
        &[qkv_input_bytes],
        &[qkv_output_bytes],
    )?;

    let sdpa_mil = sdpa_mil();
    let sdpa_io_bytes = HEADS * SEQ * HEAD_DIM * 2;
    let mut sdpa_compiler = ANECompiler::new();
    let mut sdpa_exec = sdpa_compiler.compile_single(
        &sdpa_mil,
        None,
        &[sdpa_io_bytes, sdpa_io_bytes, sdpa_io_bytes],
        &[sdpa_io_bytes],
    )?;

    let out_mil = out_proj_mil();
    let out_input_bytes = EMBED_DIM * SEQ * 4;
    let out_output_bytes = EMBED_DIM * SEQ * 4;
    let mut out_compiler = ANECompiler::new();
    let mut out_exec = out_compiler.compile_single(
        &out_mil,
        Some(wo_blob.as_bytes()),
        &[out_input_bytes],
        &[out_output_bytes],
    )?;

    let w1_mil = ffn_proj_mil("w1");
    let w3_mil = ffn_proj_mil("w3");
    let w2_mil = ffn_proj_mil("w2");
    let mut w1_compiler = ANECompiler::new();
    let mut w1_exec = w1_compiler.compile_single(
        &w1_mil,
        Some(w1_blob.as_bytes()),
        &[out_input_bytes],
        &[out_output_bytes],
    )?;
    let mut w3_compiler = ANECompiler::new();
    let mut w3_exec = w3_compiler.compile_single(
        &w3_mil,
        Some(w3_blob.as_bytes()),
        &[out_input_bytes],
        &[out_output_bytes],
    )?;
    let mut w2_compiler = ANECompiler::new();
    let mut w2_exec = w2_compiler.compile_single(
        &w2_mil,
        Some(w2_blob.as_bytes()),
        &[out_input_bytes],
        &[out_output_bytes],
    )?;

    // CPU reference path.
    let q_cpu_proj = linear_channels_first(&x_norm, EMBED_DIM, EMBED_DIM, SEQ, &q);
    let k_cpu_proj = linear_channels_first(&x_norm, EMBED_DIM, EMBED_DIM, SEQ, &k);
    let v_cpu_proj = linear_channels_first(&x_norm, EMBED_DIM, EMBED_DIM, SEQ, &v);
    let mut qkv_cpu = vec![0.0f32; QKV_DIM * SEQ];
    qkv_cpu[0..EMBED_DIM * SEQ].copy_from_slice(&q_cpu_proj);
    qkv_cpu[EMBED_DIM * SEQ..2 * EMBED_DIM * SEQ].copy_from_slice(&k_cpu_proj);
    qkv_cpu[2 * EMBED_DIM * SEQ..3 * EMBED_DIM * SEQ].copy_from_slice(&v_cpu_proj);
    let (q_cpu, k_cpu, v_cpu) = split_qkv(&qkv_cpu);
    let q_cpu_h = to_head_layout(&q_cpu);
    let k_cpu_h = to_head_layout(&k_cpu);
    let v_cpu_h = to_head_layout(&v_cpu);
    let attn_cpu = cpu_sdpa(&q_cpu_h, &k_cpu_h, &v_cpu_h);
    let attn_cpu_flat = from_head_layout(&attn_cpu);
    let out_cpu = linear_channels_first(&attn_cpu_flat, EMBED_DIM, EMBED_DIM, SEQ, &wo);
    let attn_resid_cpu: Vec<f32> = x_norm
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();
    let ffn_norm_cpu = rms_norm(&attn_resid_cpu, EMBED_DIM, SEQ, 1e-6);
    let h1_cpu = linear_channels_first(&ffn_norm_cpu, EMBED_DIM, EMBED_DIM, SEQ, &w1);
    let h3_cpu = linear_channels_first(&ffn_norm_cpu, EMBED_DIM, EMBED_DIM, SEQ, &w3);
    let silu_cpu: Vec<f32> = h1_cpu.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let gate_cpu: Vec<f32> = silu_cpu
        .iter()
        .zip(h3_cpu.iter())
        .map(|(a, b)| a * b)
        .collect();
    let ffn_cpu = linear_channels_first(&gate_cpu, EMBED_DIM, EMBED_DIM, SEQ, &w2);
    let final_cpu: Vec<f32> = ffn_norm_cpu
        .iter()
        .zip(ffn_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();

    // ANE attention path.
    let x_tensor = ANETensor::from_fp32(x_norm.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;
    println!("Running ANE fused QKV...");
    qkv_exec.write_input(0, x_tensor.as_bytes())?;
    qkv_exec.eval()?;
    println!("✓ QKV complete");
    let mut qkv_buf = vec![0u8; qkv_output_bytes];
    qkv_exec.read_output(0, &mut qkv_buf)?;
    let qkv_ane = fp32_bytes_to_f32(&qkv_buf);
    let (qkv_max, qkv_mean) = max_mean_diff(&qkv_ane, &qkv_cpu);

    let (q_ane, k_ane, v_ane) = split_qkv(&qkv_ane);
    let q_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(&q_ane)),
        vec![BATCH, HEADS, SEQ, HEAD_DIM],
    )?;
    let k_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(&k_ane)),
        vec![BATCH, HEADS, SEQ, HEAD_DIM],
    )?;
    let v_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(&v_ane)),
        vec![BATCH, HEADS, SEQ, HEAD_DIM],
    )?;

    println!("Running ANE SDPA...");
    sdpa_exec.write_input(0, q_ane_tensor.as_bytes())?;
    sdpa_exec.write_input(1, k_ane_tensor.as_bytes())?;
    sdpa_exec.write_input(2, v_ane_tensor.as_bytes())?;
    sdpa_exec.eval()?;
    println!("✓ SDPA complete");
    let mut sdpa_buf = vec![0u8; sdpa_io_bytes];
    sdpa_exec.read_output(0, &mut sdpa_buf)?;
    let sdpa_ane = fp16_bytes_to_f32(&sdpa_buf);
    let (sdpa_max, sdpa_mean) = max_mean_diff(&sdpa_ane, &attn_cpu);

    let sdpa_ane_flat = from_head_layout(&sdpa_ane);
    let out_input_tensor =
        ANETensor::from_fp32(sdpa_ane_flat.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;
    println!("Running ANE output projection...");
    out_exec.write_input(0, out_input_tensor.as_bytes())?;
    out_exec.eval()?;
    println!("✓ Output projection complete");
    let mut out_buf = vec![0u8; out_output_bytes];
    out_exec.read_output(0, &mut out_buf)?;
    let out_ane = fp32_bytes_to_f32(&out_buf);
    let attn_resid_ane: Vec<f32> = x_norm
        .iter()
        .zip(out_ane.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (out_max, out_mean) = max_mean_diff(&out_ane, &out_cpu);
    let (attn_resid_max, attn_resid_mean) = max_mean_diff(&attn_resid_ane, &attn_resid_cpu);

    // ANE FFN path.
    let ffn_norm_ane = rms_norm(&attn_resid_ane, EMBED_DIM, SEQ, 1e-6);
    let ffn_input_tensor =
        ANETensor::from_fp32(ffn_norm_ane.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;

    println!("Running ANE FFN W1...");
    w1_exec.write_input(0, ffn_input_tensor.as_bytes())?;
    w1_exec.eval()?;
    println!("✓ W1 complete");
    let mut h1_buf = vec![0u8; out_output_bytes];
    w1_exec.read_output(0, &mut h1_buf)?;
    let h1_ane = fp32_bytes_to_f32(&h1_buf);

    println!("Running ANE FFN W3...");
    w3_exec.write_input(0, ffn_input_tensor.as_bytes())?;
    w3_exec.eval()?;
    println!("✓ W3 complete");
    let mut h3_buf = vec![0u8; out_output_bytes];
    w3_exec.read_output(0, &mut h3_buf)?;
    let h3_ane = fp32_bytes_to_f32(&h3_buf);

    let (h1_max, h1_mean) = max_mean_diff(&h1_ane, &h1_cpu);
    let (h3_max, h3_mean) = max_mean_diff(&h3_ane, &h3_cpu);

    let silu_ane: Vec<f32> = h1_ane.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let gate_ane: Vec<f32> = silu_ane
        .iter()
        .zip(h3_ane.iter())
        .map(|(a, b)| a * b)
        .collect();
    let gate_tensor = ANETensor::from_fp32(gate_ane.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;

    println!("Running ANE FFN W2...");
    w2_exec.write_input(0, gate_tensor.as_bytes())?;
    w2_exec.eval()?;
    println!("✓ W2 complete");
    let mut y_buf = vec![0u8; out_output_bytes];
    w2_exec.read_output(0, &mut y_buf)?;
    let ffn_ane = fp32_bytes_to_f32(&y_buf);
    let final_ane: Vec<f32> = ffn_norm_ane
        .iter()
        .zip(ffn_ane.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (y_max, y_mean) = max_mean_diff(&ffn_ane, &ffn_cpu);
    let (final_max, final_mean) = max_mean_diff(&final_ane, &final_cpu);

    println!("Verification:");
    println!("  QKV projection max diff:  {:.6}", qkv_max);
    println!("  QKV projection mean diff: {:.6}", qkv_mean);
    println!("  SDPA max diff:            {:.6}", sdpa_max);
    println!("  SDPA mean diff:           {:.6}", sdpa_mean);
    println!("  Out proj max diff:        {:.6}", out_max);
    println!("  Out proj mean diff:       {:.6}", out_mean);
    println!("  Attn resid max diff:      {:.6}", attn_resid_max);
    println!("  Attn resid mean diff:     {:.6}", attn_resid_mean);
    println!("  W1 max diff:              {:.6}", h1_max);
    println!("  W1 mean diff:             {:.6}", h1_mean);
    println!("  W3 max diff:              {:.6}", h3_max);
    println!("  W3 mean diff:             {:.6}", h3_mean);
    println!("  W2 max diff:              {:.6}", y_max);
    println!("  W2 mean diff:             {:.6}", y_mean);
    println!("  Final layer max diff:     {:.6}", final_max);
    println!("  Final layer mean diff:    {:.6}", final_mean);

    if final_max < 0.05 {
        println!("\n✅ Transformer layer proof-of-life completed successfully!");
    } else {
        println!("\n⚠️  Layer ran, but the diff was larger than expected.");
    }

    Ok(())
}
