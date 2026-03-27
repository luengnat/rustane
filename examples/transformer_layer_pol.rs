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
    mil::{rmsnorm_mil, WeightBlob},
    wrapper::{ANECompiler, ANEExecutor, ANETensor},
};
use std::env;

#[derive(Clone, Copy)]
struct ShapeCfg {
    heads: usize,
    seq: usize,
    head_dim: usize,
    embed_dim: usize,
    attn_head_dim: usize,
    attn_embed_dim: usize,
    qkv_dim: usize,
    batch: usize,
}

fn shape_cfg() -> ShapeCfg {
    let heads = env::var("TRANSFORMER_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let seq = env::var("TRANSFORMER_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let head_dim = env::var("TRANSFORMER_HEAD_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let embed_dim = env::var("TRANSFORMER_EMBED_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(heads * head_dim);
    assert_eq!(
        embed_dim,
        heads * head_dim,
        "TRANSFORMER_EMBED_DIM must equal TRANSFORMER_HEADS * TRANSFORMER_HEAD_DIM"
    );
    let attn_head_dim = if head_dim % 64 == 0 {
        head_dim
    } else {
        ((head_dim + 63) / 64) * 64
    };
    let attn_embed_dim = heads * attn_head_dim;
    ShapeCfg {
        heads,
        seq,
        head_dim,
        embed_dim,
        attn_head_dim,
        attn_embed_dim,
        qkv_dim: attn_embed_dim * 3,
        batch: 1,
    }
}

fn build_info() -> &'static str {
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]"
}

fn qkv_mil(cfg: ShapeCfg) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {e}, 1, {s}]> x) {{\n",
        e = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {e}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        e = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    for name in ["Wq", "Wk", "Wv"] {
        mil.push_str(&format!(
        "        tensor<fp16, [{a}, {e}, 1, 1]> {n} = const()[name = string(\"{n}\"), val = tensor<fp16, [{a}, {e}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{file}.bin\"), offset = uint64(64)))];\n",
            a = cfg.attn_embed_dim,
            e = cfg.embed_dim,
            n = name,
            file = name.to_lowercase(),
        ));
    }
    for (out_name, weight_name) in [("q", "Wq"), ("k", "Wk"), ("v", "Wv")] {
        mil.push_str(&format!(
        "        tensor<fp16, [1, {e}, 1, {s}]> {out} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {w}, x = x16)[name = string(\"c{out}\")];\n",
            e = cfg.attn_embed_dim,
            s = cfg.seq,
            out = out_name,
            w = weight_name,
        ));
    }
    mil.push_str("        int32 ax = const()[name = string(\"ax\"), val = int32(1)];\n");
    mil.push_str("        bool inter = const()[name = string(\"il\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {qkv}, 1, {s}]> qkv = concat(axis = ax, interleave = inter, values = (q, k, v))[name = string(\"cat\")];\n",
        qkv = cfg.qkv_dim,
        s = cfg.seq,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {qkv}, 1, {s}]> y = cast(dtype = to_fp32, x = qkv)[name = string(\"cast_out\")];\n",
        qkv = cfg.qkv_dim,
        s = cfg.seq,
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn sdpa_mil(cfg: ShapeCfg) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {h}, {s}, {d}]> q, tensor<fp16, [1, {h}, {s}, {d}]> k, tensor<fp16, [1, {h}, {s}, {d}]> v) {{\n",
        h = cfg.heads,
        s = cfg.seq,
        d = cfg.attn_head_dim,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {h}, {s}, {d}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n",
        h = cfg.heads,
        s = cfg.seq,
        d = cfg.attn_head_dim,
    ));
    mil.push_str("    } -> (att);\n");
    mil.push_str("}\n");
    mil
}

fn qkt_mil(cfg: ShapeCfg) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {h}, {s}, {d}]> q, tensor<fp16, [1, {h}, {s}, {d}]> k) {{\n",
        h = cfg.heads,
        s = cfg.seq,
        d = cfg.attn_head_dim,
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str("        bool bT = const()[name = string(\"bT\"), val = bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {h}, {s}, {s}]> scores = matmul(transpose_x = bF, transpose_y = bT, x = q, y = k)[name = string(\"qkt\")];\n",
        h = cfg.heads,
        s = cfg.seq,
    ));
    mil.push_str("    } -> (scores);\n");
    mil.push_str("}\n");
    mil
}

fn sv_mil(cfg: ShapeCfg) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {h}, {s}, {s}]> probs, tensor<fp16, [1, {h}, {s}, {d}]> v) {{\n",
        h = cfg.heads,
        s = cfg.seq,
        d = cfg.attn_head_dim,
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {h}, {s}, {d}]> out = matmul(transpose_x = bF, transpose_y = bF, x = probs, y = v)[name = string(\"sv\")];\n",
        h = cfg.heads,
        s = cfg.seq,
        d = cfg.attn_head_dim,
    ));
    mil.push_str("    } -> (out);\n");
    mil.push_str("}\n");
    mil
}

fn out_proj_mil(cfg: ShapeCfg) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {a}, 1, {s}]> x) {{\n",
        a = cfg.attn_embed_dim,
        s = cfg.seq,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {a}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        a = cfg.attn_embed_dim,
        s = cfg.seq,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{o}, {a}, 1, 1]> W = const()[name = string(\"Wout\"), val = tensor<fp16, [{o}, {a}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        o = cfg.embed_dim,
        a = cfg.attn_embed_dim,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {o}, 1, {s}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"proj\")];\n",
        o = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {e}, 1, {s}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        e = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn ffn_proj_mil(cfg: ShapeCfg, name: &str) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {d}, 1, {s}]> x) {{\n",
        d = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {d}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        d = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{d}, {d}, 1, 1]> W = const()[name = string(\"{name}\"), val = tensor<fp16, [{d}, {d}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        d = cfg.embed_dim,
        name = name,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {d}, 1, {s}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"proj\")];\n",
        d = cfg.embed_dim,
        s = cfg.seq,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {d}, 1, {s}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        d = cfg.embed_dim,
        s = cfg.seq,
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

fn ane_rms_norm(
    input: &[f32],
    gamma: &[f32],
    cfg: ShapeCfg,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mil = rmsnorm_mil(cfg.seq, cfg.embed_dim);
    let gamma_blob = WeightBlob::from_fp32(gamma, 1, cfg.embed_dim as i32)?;
    let io_bytes = cfg.embed_dim * cfg.seq * 2;
    let input_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(input),
        vec![cfg.batch, cfg.embed_dim, 1, cfg.seq],
    )?;

    let mut compiler = ANECompiler::new();
    let mut executor: ANEExecutor = compiler.compile_multi(
        &mil,
        &["@model_path/weights/rms_w.bin"],
        &[gamma_blob.as_bytes()],
        &[gamma_blob.len()],
        &[io_bytes],
        &[io_bytes],
    )?;

    executor.write_input(0, input_tensor.as_bytes())?;
    executor.eval()?;

    let mut out_buf = vec![0u8; io_bytes];
    executor.read_output(0, &mut out_buf)?;
    Ok(fp16_bytes_to_f32(&out_buf))
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

fn split_qkv(qkv: &[f32], cfg: ShapeCfg) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let span = cfg.attn_embed_dim * cfg.seq;
    (
        qkv[0..span].to_vec(),
        qkv[span..2 * span].to_vec(),
        qkv[2 * span..3 * span].to_vec(),
    )
}

fn to_head_layout(x: &[f32], heads: usize, seq: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * seq * head_dim];
    let embed_dim = heads * head_dim;
    for c in 0..embed_dim {
        let h = c / head_dim;
        let d = c % head_dim;
        for t in 0..seq {
            out[(h * seq + t) * head_dim + d] = x[c * seq + t];
        }
    }
    out
}

fn from_head_layout(x: &[f32], heads: usize, seq: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * head_dim * seq];
    let embed_dim = heads * head_dim;
    for c in 0..embed_dim {
        let h = c / head_dim;
        let d = c % head_dim;
        for t in 0..seq {
            out[c * seq + t] = x[(h * seq + t) * head_dim + d];
        }
    }
    out
}

fn cpu_sdpa(q: &[f32], k: &[f32], v: &[f32], cfg: ShapeCfg) -> Vec<f32> {
    let mut out = vec![0.0f32; cfg.heads * cfg.seq * cfg.attn_head_dim];
    let scale = 1.0f32 / (cfg.attn_head_dim as f32).sqrt();

    for h in 0..cfg.heads {
        for t in 0..cfg.seq {
            let mut scores = vec![0.0f32; cfg.seq];
            let mut max_score = f32::NEG_INFINITY;
            for t2 in 0..cfg.seq {
                let mut s = 0.0f32;
                for d in 0..cfg.attn_head_dim {
                    let q_idx = ((h * cfg.seq + t) * cfg.attn_head_dim) + d;
                    let k_idx = ((h * cfg.seq + t2) * cfg.attn_head_dim) + d;
                    s += q[q_idx] * k[k_idx];
                }
                s *= scale;
                scores[t2] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut denom = 0.0f32;
            for t2 in 0..cfg.seq {
                scores[t2] = (scores[t2] - max_score).exp();
                denom += scores[t2];
            }
            for t2 in 0..cfg.seq {
                scores[t2] /= denom;
            }

            for d in 0..cfg.attn_head_dim {
                let mut val = 0.0f32;
                for t2 in 0..cfg.seq {
                    let v_idx = ((h * cfg.seq + t2) * cfg.attn_head_dim) + d;
                    val += scores[t2] * v[v_idx];
                }
                out[((h * cfg.seq + t) * cfg.attn_head_dim) + d] = val;
            }
        }
    }

    out
}

fn cpu_qkt(q: &[f32], k: &[f32], cfg: ShapeCfg) -> Vec<f32> {
    let mut out = vec![0.0f32; cfg.heads * cfg.seq * cfg.seq];
    let scale = 1.0f32 / (cfg.attn_head_dim as f32).sqrt();
    for h in 0..cfg.heads {
        for t in 0..cfg.seq {
            for t2 in 0..cfg.seq {
                let mut s = 0.0f32;
                for d in 0..cfg.attn_head_dim {
                    let q_idx = ((h * cfg.seq + t) * cfg.attn_head_dim) + d;
                    let k_idx = ((h * cfg.seq + t2) * cfg.attn_head_dim) + d;
                    s += q[q_idx] * k[k_idx];
                }
                out[h * cfg.seq * cfg.seq + t * cfg.seq + t2] = s * scale;
            }
        }
    }
    out
}

fn softmax_rows(scores: &[f32], cfg: ShapeCfg) -> Vec<f32> {
    let mut probs = scores.to_vec();
    for h in 0..cfg.heads {
        for t in 0..cfg.seq {
            let row = &mut probs
                [h * cfg.seq * cfg.seq + t * cfg.seq..h * cfg.seq * cfg.seq + (t + 1) * cfg.seq];
            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let mut denom = 0.0f32;
            for value in row.iter_mut() {
                *value = (*value - max_logit).exp();
                denom += *value;
            }
            for value in row.iter_mut() {
                *value /= denom;
            }
        }
    }
    probs
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

    let avail = rustane::HardwareAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available");
        return Ok(());
    }
    println!();

    init()?;
    println!("✓ ANE runtime initialized\n");

    let cfg = shape_cfg();
    if cfg.attn_head_dim != cfg.head_dim {
        println!(
            "Internal attention padding: head_dim {} -> {} (attn_embed_dim = {})\n",
            cfg.head_dim, cfg.attn_head_dim, cfg.attn_embed_dim
        );
    }

    // Deterministic input.
    let mut x = vec![0.0f32; cfg.embed_dim * cfg.seq];
    for (i, item) in x.iter_mut().enumerate() {
        *item = ((i as f32 * 0.03125).sin() * 0.4) + ((i as f32 * 0.0078125).cos() * 0.2);
    }
    let gamma = vec![1.0f32; cfg.embed_dim];
    let x_norm_cpu = rms_norm(&x, cfg.embed_dim, cfg.seq, 1e-6);
    let x_norm_ane = ane_rms_norm(&x, &gamma, cfg)?;

    // Attention weights.
    let mut q = vec![0.0f32; cfg.attn_embed_dim * cfg.embed_dim];
    let mut k = vec![0.0f32; cfg.attn_embed_dim * cfg.embed_dim];
    let mut v = vec![0.0f32; cfg.attn_embed_dim * cfg.embed_dim];
    let mut wo = vec![0.0f32; cfg.embed_dim * cfg.attn_embed_dim];
    for i in 0..cfg.embed_dim {
        q[i * cfg.embed_dim + i] = 1.0;
        k[i * cfg.embed_dim + i] = 0.5;
        v[i * cfg.embed_dim + i] = -0.25;
        wo[i * cfg.attn_embed_dim + i] = 1.0;
    }
    let wq = WeightBlob::from_fp32(&q, cfg.attn_embed_dim as i32, cfg.embed_dim as i32)?;
    let wk = WeightBlob::from_fp32(&k, cfg.attn_embed_dim as i32, cfg.embed_dim as i32)?;
    let wv = WeightBlob::from_fp32(&v, cfg.attn_embed_dim as i32, cfg.embed_dim as i32)?;
    let wo_blob = WeightBlob::from_fp32(&wo, cfg.embed_dim as i32, cfg.attn_embed_dim as i32)?;

    // FFN weights.
    let mut w1 = vec![0.0f32; cfg.embed_dim * cfg.embed_dim];
    let mut w3 = vec![0.0f32; cfg.embed_dim * cfg.embed_dim];
    let mut w2 = vec![0.0f32; cfg.embed_dim * cfg.embed_dim];
    for i in 0..cfg.embed_dim {
        w1[i * cfg.embed_dim + i] = 1.0;
        w3[i * cfg.embed_dim + i] = 0.5;
        w2[i * cfg.embed_dim + i] = 1.0;
    }
    let w1_blob = WeightBlob::from_fp32(&w1, cfg.embed_dim as i32, cfg.embed_dim as i32)?;
    let w3_blob = WeightBlob::from_fp32(&w3, cfg.embed_dim as i32, cfg.embed_dim as i32)?;
    let w2_blob = WeightBlob::from_fp32(&w2, cfg.embed_dim as i32, cfg.embed_dim as i32)?;

    // Compile kernels.
    let qkv_mil = qkv_mil(cfg);
    let qkv_input_bytes = cfg.embed_dim * cfg.seq * 4;
    let qkv_output_bytes = cfg.qkv_dim * cfg.seq * 4;
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

    let qkt_mil = qkt_mil(cfg);
    let sv_mil = sv_mil(cfg);
    let qkt_bytes = cfg.heads * cfg.seq * cfg.seq * 2;
    let sdpa_io_bytes = cfg.heads * cfg.seq * cfg.attn_head_dim * 2;
    let mut qkt_compiler = ANECompiler::new();
    let mut qkt_exec = qkt_compiler.compile_single(
        &qkt_mil,
        None,
        &[sdpa_io_bytes, sdpa_io_bytes],
        &[qkt_bytes],
    )?;
    let mut sv_compiler = ANECompiler::new();
    let mut sv_exec =
        sv_compiler.compile_single(&sv_mil, None, &[qkt_bytes, sdpa_io_bytes], &[sdpa_io_bytes])?;

    let out_mil = out_proj_mil(cfg);
    let attn_out_input_bytes = cfg.attn_embed_dim * cfg.seq * 4;
    let out_output_bytes = cfg.embed_dim * cfg.seq * 4;
    let mut out_compiler = ANECompiler::new();
    let mut out_exec = out_compiler.compile_single(
        &out_mil,
        Some(wo_blob.as_bytes()),
        &[attn_out_input_bytes],
        &[out_output_bytes],
    )?;

    let w1_mil = ffn_proj_mil(cfg, "w1");
    let w3_mil = ffn_proj_mil(cfg, "w3");
    let w2_mil = ffn_proj_mil(cfg, "w2");
    let mut w1_compiler = ANECompiler::new();
    let mut w1_exec = w1_compiler.compile_single(
        &w1_mil,
        Some(w1_blob.as_bytes()),
        &[cfg.embed_dim * cfg.seq * 4],
        &[out_output_bytes],
    )?;
    let mut w3_compiler = ANECompiler::new();
    let mut w3_exec = w3_compiler.compile_single(
        &w3_mil,
        Some(w3_blob.as_bytes()),
        &[cfg.embed_dim * cfg.seq * 4],
        &[out_output_bytes],
    )?;
    let mut w2_compiler = ANECompiler::new();
    let mut w2_exec = w2_compiler.compile_single(
        &w2_mil,
        Some(w2_blob.as_bytes()),
        &[cfg.embed_dim * cfg.seq * 4],
        &[out_output_bytes],
    )?;

    // CPU reference path.
    let q_cpu_proj =
        linear_channels_first(&x_norm_cpu, cfg.embed_dim, cfg.attn_embed_dim, cfg.seq, &q);
    let k_cpu_proj =
        linear_channels_first(&x_norm_cpu, cfg.embed_dim, cfg.attn_embed_dim, cfg.seq, &k);
    let v_cpu_proj =
        linear_channels_first(&x_norm_cpu, cfg.embed_dim, cfg.attn_embed_dim, cfg.seq, &v);
    let mut qkv_cpu = vec![0.0f32; cfg.qkv_dim * cfg.seq];
    qkv_cpu[0..cfg.attn_embed_dim * cfg.seq].copy_from_slice(&q_cpu_proj);
    qkv_cpu[cfg.attn_embed_dim * cfg.seq..2 * cfg.attn_embed_dim * cfg.seq]
        .copy_from_slice(&k_cpu_proj);
    qkv_cpu[2 * cfg.attn_embed_dim * cfg.seq..3 * cfg.attn_embed_dim * cfg.seq]
        .copy_from_slice(&v_cpu_proj);
    let (q_cpu, k_cpu, v_cpu) = split_qkv(&qkv_cpu, cfg);
    let q_cpu_h = to_head_layout(&q_cpu, cfg.heads, cfg.seq, cfg.attn_head_dim);
    let k_cpu_h = to_head_layout(&k_cpu, cfg.heads, cfg.seq, cfg.attn_head_dim);
    let v_cpu_h = to_head_layout(&v_cpu, cfg.heads, cfg.seq, cfg.attn_head_dim);
    let qkt_cpu = cpu_qkt(&q_cpu_h, &k_cpu_h, cfg);
    let probs_cpu = softmax_rows(&qkt_cpu, cfg);
    let attn_cpu = cpu_sdpa(&q_cpu_h, &k_cpu_h, &v_cpu_h, cfg);
    let attn_cpu_flat = from_head_layout(&attn_cpu, cfg.heads, cfg.seq, cfg.attn_head_dim);
    let out_cpu = linear_channels_first(
        &attn_cpu_flat,
        cfg.attn_embed_dim,
        cfg.embed_dim,
        cfg.seq,
        &wo,
    );
    let attn_resid_cpu: Vec<f32> = x_norm_cpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();
    let ffn_norm_cpu = rms_norm(&attn_resid_cpu, cfg.embed_dim, cfg.seq, 1e-6);
    let h1_cpu = linear_channels_first(&ffn_norm_cpu, cfg.embed_dim, cfg.embed_dim, cfg.seq, &w1);
    let h3_cpu = linear_channels_first(&ffn_norm_cpu, cfg.embed_dim, cfg.embed_dim, cfg.seq, &w3);
    let silu_cpu: Vec<f32> = h1_cpu.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let gate_cpu: Vec<f32> = silu_cpu
        .iter()
        .zip(h3_cpu.iter())
        .map(|(a, b)| a * b)
        .collect();
    let ffn_cpu = linear_channels_first(&gate_cpu, cfg.embed_dim, cfg.embed_dim, cfg.seq, &w2);
    let final_cpu: Vec<f32> = ffn_norm_cpu
        .iter()
        .zip(ffn_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();

    // ANE attention path.
    let x_tensor = ANETensor::from_fp32(
        x_norm_ane.clone(),
        vec![cfg.batch, cfg.embed_dim, 1, cfg.seq],
    )?;
    println!("Running ANE fused QKV...");
    qkv_exec.write_input(0, x_tensor.as_bytes())?;
    qkv_exec.eval()?;
    println!("✓ QKV complete");
    let mut qkv_buf = vec![0u8; qkv_output_bytes];
    qkv_exec.read_output(0, &mut qkv_buf)?;
    let qkv_ane = fp32_bytes_to_f32(&qkv_buf);
    let (qkv_max, qkv_mean) = max_mean_diff(&qkv_ane, &qkv_cpu);

    let (q_ane, k_ane, v_ane) = split_qkv(&qkv_ane, cfg);
    let q_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(
            &q_ane,
            cfg.heads,
            cfg.seq,
            cfg.attn_head_dim,
        )),
        vec![cfg.batch, cfg.heads, cfg.seq, cfg.attn_head_dim],
    )?;
    let k_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(
            &k_ane,
            cfg.heads,
            cfg.seq,
            cfg.attn_head_dim,
        )),
        vec![cfg.batch, cfg.heads, cfg.seq, cfg.attn_head_dim],
    )?;
    let v_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(
            &v_ane,
            cfg.heads,
            cfg.seq,
            cfg.attn_head_dim,
        )),
        vec![cfg.batch, cfg.heads, cfg.seq, cfg.attn_head_dim],
    )?;

    println!("Running ANE QK^T...");
    qkt_exec.write_input(0, q_ane_tensor.as_bytes())?;
    qkt_exec.write_input(1, k_ane_tensor.as_bytes())?;
    qkt_exec.eval()?;
    println!("✓ QK^T complete");
    let mut qkt_buf = vec![0u8; qkt_bytes];
    qkt_exec.read_output(0, &mut qkt_buf)?;
    let scale = 1.0f32 / (cfg.attn_head_dim as f32).sqrt();
    let qkt_ane = fp16_bytes_to_f32(&qkt_buf);
    let mut qkt_ane_scaled = qkt_ane.clone();
    for value in &mut qkt_ane_scaled {
        *value *= scale;
    }
    let mut probs = qkt_ane_scaled.clone();
    for h in 0..cfg.heads {
        for t in 0..cfg.seq {
            let row = &mut probs
                [h * cfg.seq * cfg.seq + t * cfg.seq..h * cfg.seq * cfg.seq + (t + 1) * cfg.seq];
            let mut max_logit = f32::NEG_INFINITY;
            for t2 in 0..cfg.seq {
                row[t2] *= scale;
                if row[t2] > max_logit {
                    max_logit = row[t2];
                }
            }
            let mut denom = 0.0f32;
            for t2 in 0..cfg.seq {
                row[t2] = (row[t2] - max_logit).exp();
                denom += row[t2];
            }
            for t2 in 0..cfg.seq {
                row[t2] /= denom;
            }
        }
    }
    let (qkt_max, qkt_mean) = max_mean_diff(&qkt_ane_scaled, &qkt_cpu);
    let (sdpa_max, sdpa_mean) = max_mean_diff(&probs, &probs_cpu);

    let probs_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&probs),
        vec![cfg.batch, cfg.heads, cfg.seq, cfg.seq],
    )?;
    println!("Running ANE scores@V...");
    sv_exec.write_input(0, probs_tensor.as_bytes())?;
    sv_exec.write_input(1, v_ane_tensor.as_bytes())?;
    sv_exec.eval()?;
    println!("✓ scores@V complete");
    let mut sdpa_buf = vec![0u8; sdpa_io_bytes];
    sv_exec.read_output(0, &mut sdpa_buf)?;
    let sdpa_ane = fp16_bytes_to_f32(&sdpa_buf);
    let (sdpa_out_max, sdpa_out_mean) = max_mean_diff(&sdpa_ane, &attn_cpu);

    let sdpa_ane_flat = from_head_layout(&sdpa_ane, cfg.heads, cfg.seq, cfg.attn_head_dim);
    let out_input_tensor = ANETensor::from_fp32(
        sdpa_ane_flat.clone(),
        vec![cfg.batch, cfg.attn_embed_dim, 1, cfg.seq],
    )?;
    println!("Running ANE output projection...");
    out_exec.write_input(0, out_input_tensor.as_bytes())?;
    out_exec.eval()?;
    println!("✓ Output projection complete");
    let mut out_buf = vec![0u8; out_output_bytes];
    out_exec.read_output(0, &mut out_buf)?;
    let out_ane = fp32_bytes_to_f32(&out_buf);
    let attn_resid_ane: Vec<f32> = x_norm_ane
        .iter()
        .zip(out_ane.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (xnorm_max, xnorm_mean) = max_mean_diff(&x_norm_ane, &x_norm_cpu);
    let (out_max, out_mean) = max_mean_diff(&out_ane, &out_cpu);
    let (attn_resid_max, attn_resid_mean) = max_mean_diff(&attn_resid_ane, &attn_resid_cpu);

    // ANE FFN path.
    let ffn_norm_ane = ane_rms_norm(&attn_resid_ane, &gamma, cfg)?;
    let ffn_input_tensor = ANETensor::from_fp32(
        ffn_norm_ane.clone(),
        vec![cfg.batch, cfg.embed_dim, 1, cfg.seq],
    )?;

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
    let gate_tensor =
        ANETensor::from_fp32(gate_ane.clone(), vec![cfg.batch, cfg.embed_dim, 1, cfg.seq])?;

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
    println!("  RMSNorm input max diff:   {:.6}", xnorm_max);
    println!("  RMSNorm input mean diff:  {:.6}", xnorm_mean);
    println!("  QKV projection max diff:  {:.6}", qkv_max);
    println!("  QKV projection mean diff: {:.6}", qkv_mean);
    println!("  QK^T max diff:            {:.6}", qkt_max);
    println!("  QK^T mean diff:           {:.6}", qkt_mean);
    println!("  probs max diff:           {:.6}", sdpa_max);
    println!("  probs mean diff:          {:.6}", sdpa_mean);
    println!("  scores@V max diff:        {:.6}", sdpa_out_max);
    println!("  scores@V mean diff:       {:.6}", sdpa_out_mean);
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

    if final_max < 0.1 {
        println!("\n✅ Transformer layer proof-of-life completed successfully!");
    } else {
        println!("\n⚠️  Layer ran, but the diff was larger than expected.");
    }

    Ok(())
}
