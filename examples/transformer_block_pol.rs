//! Transformer block proof of life
//!
//! Runs a tiny pre-norm attention block through ANE in three stages:
//! 1. QKV projection via 1x1 conv
//! 2. scaled_dot_product_attention
//! 3. Output projection via 1x1 conv
//!
//! The goal is not model quality. The goal is to verify that the block-shaped
//! path we would want for Parameter Golf actually holds together on ANE.

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
    // x is [channels, seq]; output is [heads, seq, head_dim] flattened.
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
    println!("🍎 Rustane - Transformer Block Proof of Life");
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

    // Build a single deterministic input and normalize it on CPU.
    let mut x = vec![0.0f32; EMBED_DIM * SEQ];
    for (i, item) in x.iter_mut().enumerate() {
        *item = ((i as f32 * 0.03125).sin() * 0.4) + ((i as f32 * 0.0078125).cos() * 0.2);
    }
    let x_norm = rms_norm(&x, EMBED_DIM, SEQ, 1e-6);

    // Prepare Q, K, V, and output projection weights.
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

    // Compile the fused QKV projection kernel.
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

    // Compile the SDPA kernel.
    let sdpa_mil = sdpa_mil();
    let sdpa_io_bytes = HEADS * SEQ * HEAD_DIM * 2;
    let mut sdpa_compiler = ANECompiler::new();
    let mut sdpa_exec = sdpa_compiler.compile_single(
        &sdpa_mil,
        None,
        &[sdpa_io_bytes, sdpa_io_bytes, sdpa_io_bytes],
        &[sdpa_io_bytes],
    )?;

    // Compile the output projection kernel.
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
    let final_cpu: Vec<f32> = x_norm
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();

    // ANE QKV projection.
    let x_tensor = ANETensor::from_fp32(x_norm.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;
    println!("Running ANE fused QKV...");
    qkv_exec.write_input(0, x_tensor.as_bytes())?;
    qkv_exec.eval()?;
    println!("✓ QKV complete");
    let mut qkv_buf = vec![0u8; qkv_output_bytes];
    qkv_exec.read_output(0, &mut qkv_buf)?;
    let qkv_ane = fp32_bytes_to_f32(&qkv_buf);
    let (qkv_max, qkv_mean) = max_mean_diff(&qkv_ane, &qkv_cpu);

    // Feed Q/K/V to SDPA.
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

    // Output projection.
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
    let final_ane: Vec<f32> = x_norm
        .iter()
        .zip(out_ane.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (out_max, out_mean) = max_mean_diff(&out_ane, &out_cpu);
    let (final_max, final_mean) = max_mean_diff(&final_ane, &final_cpu);

    println!("Verification:");
    println!("  QKV projection max diff:  {:.6}", qkv_max);
    println!("  QKV projection mean diff: {:.6}", qkv_mean);
    println!("  SDPA max diff:            {:.6}", sdpa_max);
    println!("  SDPA mean diff:           {:.6}", sdpa_mean);
    println!("  Output proj max diff:     {:.6}", out_max);
    println!("  Output proj mean diff:    {:.6}", out_mean);
    println!("  Final block max diff:     {:.6}", final_max);
    println!("  Final block mean diff:    {:.6}", final_mean);

    // Run the same block a second time to prove sequential composition works too.
    let x2_norm = rms_norm(&final_ane, EMBED_DIM, SEQ, 1e-6);
    let q2_cpu_proj = linear_channels_first(&x2_norm, EMBED_DIM, EMBED_DIM, SEQ, &q);
    let k2_cpu_proj = linear_channels_first(&x2_norm, EMBED_DIM, EMBED_DIM, SEQ, &k);
    let v2_cpu_proj = linear_channels_first(&x2_norm, EMBED_DIM, EMBED_DIM, SEQ, &v);
    let mut qkv2_cpu = vec![0.0f32; QKV_DIM * SEQ];
    qkv2_cpu[0..EMBED_DIM * SEQ].copy_from_slice(&q2_cpu_proj);
    qkv2_cpu[EMBED_DIM * SEQ..2 * EMBED_DIM * SEQ].copy_from_slice(&k2_cpu_proj);
    qkv2_cpu[2 * EMBED_DIM * SEQ..3 * EMBED_DIM * SEQ].copy_from_slice(&v2_cpu_proj);
    let (q2_cpu, k2_cpu, v2_cpu) = split_qkv(&qkv2_cpu);
    let q2_cpu_h = to_head_layout(&q2_cpu);
    let k2_cpu_h = to_head_layout(&k2_cpu);
    let v2_cpu_h = to_head_layout(&v2_cpu);
    let attn2_cpu = cpu_sdpa(&q2_cpu_h, &k2_cpu_h, &v2_cpu_h);
    let attn2_cpu_flat = from_head_layout(&attn2_cpu);
    let out2_cpu = linear_channels_first(&attn2_cpu_flat, EMBED_DIM, EMBED_DIM, SEQ, &wo);
    let final2_cpu: Vec<f32> = x2_norm
        .iter()
        .zip(out2_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();

    let x2_tensor = ANETensor::from_fp32(x2_norm.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;
    println!("Running ANE fused QKV (pass 2)...");
    qkv_exec.write_input(0, x2_tensor.as_bytes())?;
    qkv_exec.eval()?;
    println!("✓ QKV pass 2 complete");
    let mut qkv2_buf = vec![0u8; qkv_output_bytes];
    qkv_exec.read_output(0, &mut qkv2_buf)?;
    let qkv2_ane = fp32_bytes_to_f32(&qkv2_buf);
    let (qkv2_max, qkv2_mean) = max_mean_diff(&qkv2_ane, &qkv2_cpu);

    let (q2_ane, k2_ane, v2_ane) = split_qkv(&qkv2_ane);
    let q2_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(&q2_ane)),
        vec![BATCH, HEADS, SEQ, HEAD_DIM],
    )?;
    let k2_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(&k2_ane)),
        vec![BATCH, HEADS, SEQ, HEAD_DIM],
    )?;
    let v2_ane_tensor = ANETensor::from_fp16(
        f32_to_fp16_bits(&to_head_layout(&v2_ane)),
        vec![BATCH, HEADS, SEQ, HEAD_DIM],
    )?;

    println!("Running ANE SDPA (pass 2)...");
    sdpa_exec.write_input(0, q2_ane_tensor.as_bytes())?;
    sdpa_exec.write_input(1, k2_ane_tensor.as_bytes())?;
    sdpa_exec.write_input(2, v2_ane_tensor.as_bytes())?;
    sdpa_exec.eval()?;
    println!("✓ SDPA pass 2 complete");
    let mut sdpa2_buf = vec![0u8; sdpa_io_bytes];
    sdpa_exec.read_output(0, &mut sdpa2_buf)?;
    let sdpa2_ane = fp16_bytes_to_f32(&sdpa2_buf);
    let (sdpa2_max, sdpa2_mean) = max_mean_diff(&sdpa2_ane, &attn2_cpu);

    let sdpa2_ane_flat = from_head_layout(&sdpa2_ane);
    let out2_input_tensor =
        ANETensor::from_fp32(sdpa2_ane_flat.clone(), vec![BATCH, EMBED_DIM, 1, SEQ])?;
    println!("Running ANE output projection (pass 2)...");
    out_exec.write_input(0, out2_input_tensor.as_bytes())?;
    out_exec.eval()?;
    println!("✓ Output projection pass 2 complete");
    let mut out2_buf = vec![0u8; out_output_bytes];
    out_exec.read_output(0, &mut out2_buf)?;
    let out2_ane = fp32_bytes_to_f32(&out2_buf);
    let final2_ane: Vec<f32> = x2_norm
        .iter()
        .zip(out2_ane.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (out2_max, out2_mean) = max_mean_diff(&out2_ane, &out2_cpu);
    let (final2_max, final2_mean) = max_mean_diff(&final2_ane, &final2_cpu);

    println!("\nSecond pass:");
    println!("  QKV projection max diff:  {:.6}", qkv2_max);
    println!("  QKV projection mean diff: {:.6}", qkv2_mean);
    println!("  SDPA max diff:            {:.6}", sdpa2_max);
    println!("  SDPA mean diff:           {:.6}", sdpa2_mean);
    println!("  Output proj max diff:     {:.6}", out2_max);
    println!("  Output proj mean diff:    {:.6}", out2_mean);
    println!("  Final block max diff:     {:.6}", final2_max);
    println!("  Final block mean diff:    {:.6}", final2_mean);

    if final_max < 0.05 && final2_max < 0.05 {
        println!("\n✅ Transformer block stack proof-of-life completed successfully!");
    } else {
        println!("\n⚠️  One of the block passes drifted beyond the expected tolerance.");
    }

    Ok(())
}
