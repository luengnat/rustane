//! Hybrid transformer layer: ANE for linear ops, CPU for attention.
//!
//! Architecture:
//!   ANE Program 1: QKV projection (3x conv1x1 → concat)
//!   CPU:            Attention (softmax + matmul, since transpose/matmul don't work on ANE)
//!   ANE Program 2: Output projection + residual (conv1x1 + concat(x) + conv1x1([Wo|I]))
//!   ANE Program 3: FFN + residual (gate + up + SiLU + fused + concat(x) + conv1x1([Wd|I]))
//!
//! This is one complete transformer layer using the concat+conv1x1 identity hack
//! for residual connections (since `add` doesn't work on ANE).
//!
//! Usage: cargo run --example test_hybrid_layer -- [D] [num_heads] [SP]

use half::f16;
use std::env;
use std::time::Instant;

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

// ═══════════════════════════════════════════════════════════════════════════
// MIL GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

/// ANE Program 1: QKV projection
/// Input: x [1, D, 1, SP]
/// Output: qkv [1, 3*D, 1, SP]  (Q, K, V concatenated)
fn mil_qkv(d: usize, sp: usize) -> String {
    let total = 3 * d;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    // Three weight constants: Wq, Wk, Wv — each [D, D, 1, 1]
    for wn in ["Wq", "Wk", "Wv"] {
        m.push_str("        tensor<fp16, [");
        m.push_str(&d.to_string());
        m.push_str(", ");
        m.push_str(&d.to_string());
        m.push_str(", 1, 1]> ");
        m.push_str(wn);
        m.push_str(" = const()[name = tensor<string, []>(\"");
        m.push_str(wn);
        m.push_str("\"), val = tensor<fp16, [");
        m.push_str(&d.to_string());
        m.push_str(", ");
        m.push_str(&d.to_string());
        m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/");
        m.push_str(wn);
        m.push_str(".bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    }

    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // Three conv1x1 ops
    for (i, wn) in ["Wq", "Wk", "Wv"].iter().enumerate() {
        m.push_str("        tensor<fp16, [1, ");
        m.push_str(&d.to_string());
        m.push_str(", 1, ");
        m.push_str(&sp.to_string());
        m.push_str("]> q");
        m.push_str(&i.to_string());
        m.push_str(
            " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
        );
        m.push_str(wn);
        m.push_str(", x = x)[name = tensor<string, []>(\"c");
        m.push_str(&i.to_string());
        m.push_str("\")];\n");
    }

    // Concat Q, K, V
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> qkv = concat(values = (q0, q1, q2), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");

    m.push_str("    } -> (qkv);\n}\n");
    m
}

/// ANE Program 2: Output projection + residual
/// Input: attn_out [1, D, 1, SP], x [1, D, 1, SP]
/// Output: out [1, D, 1, SP] = Wo @ attn_out + x
///
/// Uses concat(attn_out, x) → conv1x1([Wo | I]) for residual add.
/// Weight [Wo | I] is [D, 2D, 1, 1].
fn mil_out_proj_residual(d: usize, sp: usize) -> String {
    let total_ic = 2 * d;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    // Two inputs: attn_out and x
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> attn_out, tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    // WoI weight: [D, 2D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]> WoI = const()[name = tensor<string, []>(\"WoI\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WoI.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // Concat attn_out with x
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (attn_out, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");

    // Output projection + residual
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WoI, x = cat)[name = tensor<string, []>(\"co\")];\n");

    m.push_str("    } -> (y);\n}\n");
    m
}

/// ANE Program 3: FFN with residual (same as test_residual_ffn)
/// Input: x [1, D, 1, SP]
/// Output: y [1, D, 1, SP]
fn mil_ffn_residual(d: usize, sp: usize) -> String {
    let inter = d * 4;
    let total_ic = inter + d;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    // Weights: Wg, Wu, WdI
    for (wn, oc, ic) in [("Wg", inter, d), ("Wu", inter, d), ("WdI", d, total_ic)] {
        m.push_str("        tensor<fp16, [");
        m.push_str(&oc.to_string());
        m.push_str(", ");
        m.push_str(&ic.to_string());
        m.push_str(", 1, 1]> ");
        m.push_str(wn);
        m.push_str(" = const()[name = tensor<string, []>(\"");
        m.push_str(wn);
        m.push_str("\"), val = tensor<fp16, [");
        m.push_str(&oc.to_string());
        m.push_str(", ");
        m.push_str(&ic.to_string());
        m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/");
        m.push_str(wn);
        m.push_str(".bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    }

    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // Gate + Up convs
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> gate = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n");

    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> up = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wu, x = x)[name = tensor<string, []>(\"cu\")];\n");

    // SiLU = gate * sigmoid(gate)
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n");

    // Fused = silu * up
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n");

    // Concat fused with x
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (fused, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");

    // Down projection + residual
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdI, x = cat)[name = tensor<string, []>(\"cd\")];\n");

    m.push_str("    } -> (y);\n}\n");
    m
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

fn build_blob(weights: &[f32]) -> Vec<u8> {
    let wsize = weights.len() * 2;
    let total = 128 + wsize;
    let mut buf = vec![0u8; total];
    buf[0] = 0x01;
    buf[4] = 0x02;
    buf[64] = 0xEF;
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 0x01;
    buf[72..76].copy_from_slice(&(wsize as u32).to_le_bytes());
    buf[80..84].copy_from_slice(&128u32.to_le_bytes());
    for (i, &w) in weights.iter().enumerate() {
        let b = f16::from_f32(w).to_bits();
        buf[128 + i * 2] = (b & 0xFF) as u8;
        buf[128 + i * 2 + 1] = (b >> 8) as u8;
    }
    buf
}

fn write_fp16_scattered(input: &[f32], dim: usize, sp: usize) -> Vec<u8> {
    let mut buf = vec![0u8; dim * sp * 2];
    for col in 0..dim {
        for s in 0..sp {
            let b = f16::from_f32(input[col * sp + s]).to_bits();
            let i = (col * sp + s) * 2;
            buf[i] = (b & 0xFF) as u8;
            buf[i + 1] = (b >> 8) as u8;
        }
    }
    buf
}

fn read_fp16_scattered(raw: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; raw.len() / 2];
    for i in 0..out.len() {
        let b = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
        out[i] = f16::from_bits(b).to_f32();
    }
    out
}

fn rand_weight(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u64 * 2654435761).wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * scale
        })
        .collect()
}

fn cpu_matmul(w: &[f32], wr: usize, wc: usize, inp: &[f32], sp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; wr * sp];
    for r in 0..wr {
        for s in 0..sp {
            let mut acc = 0.0f32;
            for c in 0..wc {
                acc += w[r * wc + c] * inp[c * sp + s];
            }
            out[r * sp + s] = acc;
        }
    }
    out
}

fn cpu_softmax(inp: &[f32], dim: usize, sp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim * sp];
    for s in 0..sp {
        let mut mx = f32::NEG_INFINITY;
        for d in 0..dim {
            mx = mx.max(inp[d * sp + s]);
        }
        let mut sm = 0.0f32;
        for d in 0..dim {
            let e = (inp[d * sp + s] - mx).exp();
            out[d * sp + s] = e;
            sm += e;
        }
        for d in 0..dim {
            out[d * sp + s] /= sm;
        }
    }
    out
}

/// CPU attention: Q @ K^T → softmax → @ V
/// Q, K, V are [D, SP] each. Returns [D, SP].
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, sp: usize) -> Vec<f32> {
    // Q @ K^T → [SP, SP]
    let mut scores = vec![0.0f32; sp * sp];
    for i in 0..sp {
        for j in 0..sp {
            let mut dot = 0.0f32;
            for h in 0..d {
                dot += q[h * sp + i] * k[h * sp + j];
            }
            // Scale by 1/sqrt(d)
            scores[i * sp + j] = dot / (d as f32).sqrt();
        }
    }

    // Softmax over each row
    let mut attn = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let mut mx = f32::NEG_INFINITY;
        for j in 0..sp {
            mx = mx.max(scores[i * sp + j]);
        }
        let mut sm = 0.0f32;
        for j in 0..sp {
            let e = (scores[i * sp + j] - mx).exp();
            attn[i * sp + j] = e;
            sm += e;
        }
        for j in 0..sp {
            attn[i * sp + j] /= sm;
        }
    }

    // attn @ V → [D, SP]
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            let mut acc = 0.0f32;
            for j in 0..sp {
                acc += attn[i * sp + j] * v[h * sp + j];
            }
            out[h * sp + i] = acc;
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// ANE RUNNER
// ═══════════════════════════════════════════════════════════════════════════

fn run_ane_program(
    mil: &str,
    blobs: &[(&str, &[u8])],
    input_sizes: &[usize],
    output_sizes: &[usize],
    inputs: &[&[u8]],
) -> Result<(Vec<Vec<f32>>, u128), String> {
    let full_names: Vec<String> = blobs
        .iter()
        .map(|(n, _)| format!("@model_path/weights/{}.bin", n))
        .collect();
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();
    let datas: Vec<&[u8]> = blobs.iter().map(|(_, d)| d.as_ref()).collect();
    let lens: Vec<usize> = blobs.iter().map(|(_, d)| d.len()).collect();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(mil, &name_refs, &datas, &lens, input_sizes, output_sizes)
        .map_err(|e| format!("COMPILE: {}", e))?;

    for (i, inp) in inputs.iter().enumerate() {
        exec.write_input(i, *inp)
            .map_err(|e| format!("WRITE[{}]: {}", i, e))?;
    }

    // Warmup
    exec.eval().map_err(|e| format!("WARMUP: {}", e))?;

    let start = Instant::now();
    for _ in 0..2 {
        exec.eval().map_err(|e| format!("EVAL: {}", e))?;
    }
    let us = start.elapsed().as_micros() / 3;

    let mut outputs = Vec::new();
    for i in 0..output_sizes.len() {
        let raw = exec
            .read_output_vec(i)
            .map_err(|e| format!("READ[{}]: {}", i, e))?;
        outputs.push(read_fp16_scattered(&raw));
    }

    Ok((outputs, us))
}

fn compare(ane: &[f32], cpu: &[f32]) -> (f32, f32, usize, usize) {
    let n = ane.len().min(cpu.len());
    let mut mx = 0.0f32;
    let mut sm = 0.0f32;
    let mut ok = 0usize;
    for i in 0..n {
        let d = (ane[i] - cpu[i]).abs();
        let sc = cpu[i].abs().max(0.01f32);
        mx = mx.max(d);
        sm += d;
        if d <= 0.05 * sc + 0.01 {
            ok += 1;
        }
    }
    (mx, sm / n as f32, ok, n)
}

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID LAYER TEST
// ═══════════════════════════════════════════════════════════════════════════

fn test_hybrid_layer(d: usize, num_heads: usize, sp: usize) {
    let inter = d * 4;
    let head_dim = d / num_heads;
    println!(
        "=== Hybrid Transformer Layer (D={}, heads={}, head_dim={}, SP={}, inter={}) ===",
        d, num_heads, head_dim, sp, inter
    );
    println!("  Pipeline: ANE(QKV) → CPU(attn) → ANE(out_proj+res) → ANE(FFN+res)");
    println!();

    // ── Create weights ──
    let wq = rand_weight(d * d, 0.02);
    let wk = rand_weight(d * d, 0.02);
    let wv = rand_weight(d * d, 0.02);
    let wo = rand_weight(d * d, 0.02);
    let wg = rand_weight(inter * d, 0.02);
    let wu = rand_weight(inter * d, 0.02);
    let wd = rand_weight(d * inter, 0.02);

    // WoI = [Wo | I] for output projection + residual: [D, 2D, 1, 1]
    let mut woi = vec![0.0f32; d * (2 * d)];
    for r in 0..d {
        for c in 0..d {
            woi[r * 2 * d + c] = wo[r * d + c];
        }
        woi[r * 2 * d + d + r] = 1.0;
    }

    // WdI = [Wd | I] for FFN + residual: [D, 5D, 1, 1]
    let mut wdi = vec![0.0f32; d * (inter + d)];
    for r in 0..d {
        for c in 0..inter {
            wdi[r * (inter + d) + c] = wd[r * inter + c];
        }
        wdi[r * (inter + d) + inter + r] = 1.0;
    }

    // ── Build blobs ──
    let bq = build_blob(&wq);
    let bk = build_blob(&wk);
    let bv = build_blob(&wv);
    let boi = build_blob(&woi);
    let bg = build_blob(&wg);
    let bu = build_blob(&wu);
    let bdi = build_blob(&wdi);

    // ── Input ──
    let inp: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 * 0.1 + 1.0)).collect();
    let inp16 = write_fp16_scattered(&inp, d, sp);

    // ═══════════════════════════════════════════════════════════════════
    // ANE Program 1: QKV projection
    // ═══════════════════════════════════════════════════════════════════
    let mil_qkv_str = mil_qkv(d, sp);
    let (qkv_ane, t_qkv) = run_ane_program(
        &mil_qkv_str,
        &[("Wq", &bq), ("Wk", &bk), ("Wv", &bv)],
        &[d * sp * 2],
        &[3 * d * sp * 2],
        &[&inp16],
    )
    .map_err(|e| {
        println!("  ANE QKV FAILED: {}", e);
        e
    })
    .unwrap();

    let qkv_ane = &qkv_ane[0];
    let q_ane = &qkv_ane[0..d * sp];
    let k_ane = &qkv_ane[d * sp..2 * d * sp];
    let v_ane = &qkv_ane[2 * d * sp..3 * d * sp];

    // CPU reference for QKV
    let q_cpu = cpu_matmul(&wq, d, d, &inp, sp);
    let k_cpu = cpu_matmul(&wk, d, d, &inp, sp);
    let v_cpu = cpu_matmul(&wv, d, d, &inp, sp);

    let (mx, mn, ok, n) = compare(q_ane, &q_cpu);
    println!(
        "  ANE QKV: {}us | Q: max={:.4} mean={:.6} | {}/{} ({:.1}%)",
        t_qkv,
        mx,
        mn,
        ok,
        n,
        100.0 * ok as f32 / n as f32
    );

    // ═══════════════════════════════════════════════════════════════════
    // CPU: Attention
    // ═══════════════════════════════════════════════════════════════════
    let t_attn_start = Instant::now();
    let attn_out_cpu = cpu_attention(&q_cpu, &k_cpu, &v_cpu, d, sp);
    let t_attn = t_attn_start.elapsed().as_micros();

    // For ANE path: use ANE-computed QKV for attention (to test end-to-end)
    let attn_out_ane = cpu_attention(q_ane, k_ane, v_ane, d, sp);

    println!(
        "  CPU attention: {}us (Q@K^T → softmax → @V, D={})",
        t_attn, d
    );

    // Verify ANE QKV feeds into attention correctly
    let (mx, mn, ok, n) = compare(&attn_out_ane, &attn_out_cpu);
    println!(
        "  Attn(ANE QKV) vs Attn(CPU QKV): max={:.4} mean={:.6} | {}/{} ({:.1}%)",
        mx,
        mn,
        ok,
        n,
        100.0 * ok as f32 / n as f32
    );

    let attn_out = attn_out_ane; // Use ANE-computed attention output for rest of pipeline
    let attn_out16 = write_fp16_scattered(&attn_out, d, sp);

    // ═══════════════════════════════════════════════════════════════════
    // ANE Program 2: Output projection + residual
    // ═══════════════════════════════════════════════════════════════════
    let mil_out_str = mil_out_proj_residual(d, sp);
    let (out2_ane, t_out) = run_ane_program(
        &mil_out_str,
        &[("WoI", &boi)],
        &[d * sp * 2, d * sp * 2], // Two inputs: attn_out and x
        &[d * sp * 2],
        &[&attn_out16, &inp16],
    )
    .map_err(|e| {
        println!("  ANE out_proj FAILED: {}", e);
        e
    })
    .unwrap();

    let out2_ane = &out2_ane[0];

    // CPU reference for out_proj + residual
    let out_proj_cpu = cpu_matmul(&wo, d, d, &attn_out, sp);
    let mut out2_cpu = vec![0.0f32; d * sp];
    for i in 0..d * sp {
        out2_cpu[i] = out_proj_cpu[i] + inp[i]; // Residual
    }

    let (mx, mn, ok, n) = compare(out2_ane, &out2_cpu);
    println!(
        "  ANE out_proj+res: {}us | max={:.4} mean={:.6} | {}/{} ({:.1}%)",
        t_out,
        mx,
        mn,
        ok,
        n,
        100.0 * ok as f32 / n as f32
    );

    // ═══════════════════════════════════════════════════════════════════
    // ANE Program 3: FFN with residual
    // ═══════════════════════════════════════════════════════════════════
    let out2_ane16 = write_fp16_scattered(out2_ane, d, sp);
    let mil_ffn_str = mil_ffn_residual(d, sp);
    let (ffn_ane, t_ffn) = run_ane_program(
        &mil_ffn_str,
        &[("Wg", &bg), ("Wu", &bu), ("WdI", &bdi)],
        &[d * sp * 2],
        &[d * sp * 2],
        &[&out2_ane16],
    )
    .map_err(|e| {
        println!("  ANE FFN FAILED: {}", e);
        e
    })
    .unwrap();

    let ffn_ane = &ffn_ane[0];

    // CPU reference for FFN + residual
    let gate_cpu = cpu_matmul(&wg, inter, d, &out2_cpu, sp);
    let up_cpu = cpu_matmul(&wu, inter, d, &out2_cpu, sp);
    let silu_cpu: Vec<f32> = gate_cpu
        .iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let fused_cpu: Vec<f32> = silu_cpu
        .iter()
        .zip(up_cpu.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let down_cpu = cpu_matmul(&wd, d, inter, &fused_cpu, sp);
    let mut ffn_cpu = vec![0.0f32; d * sp];
    for i in 0..d * sp {
        ffn_cpu[i] = down_cpu[i] + out2_cpu[i]; // Residual
    }

    let (mx, mn, ok, n) = compare(ffn_ane, &ffn_cpu);
    println!(
        "  ANE FFN+res: {}us | max={:.4} mean={:.6} | {}/{} ({:.1}%)",
        t_ffn,
        mx,
        mn,
        ok,
        n,
        100.0 * ok as f32 / n as f32
    );

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ane_us = t_qkv + t_out + t_ffn;
    let total_hybrid_us = total_ane_us + t_attn;

    // CPU-only reference: full layer
    let t_cpu_start = Instant::now();
    let cpu_q = cpu_matmul(&wq, d, d, &inp, sp);
    let cpu_k = cpu_matmul(&wk, d, d, &inp, sp);
    let cpu_v = cpu_matmul(&wv, d, d, &inp, sp);
    let cpu_attn = cpu_attention(&cpu_q, &cpu_k, &cpu_v, d, sp);
    let cpu_op = cpu_matmul(&wo, d, d, &cpu_attn, sp);
    let mut cpu_after_attn = vec![0.0f32; d * sp];
    for i in 0..d * sp {
        cpu_after_attn[i] = cpu_op[i] + inp[i];
    }
    let cpu_gate = cpu_matmul(&wg, inter, d, &cpu_after_attn, sp);
    let cpu_up = cpu_matmul(&wu, inter, d, &cpu_after_attn, sp);
    let cpu_silu: Vec<f32> = cpu_gate
        .iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let cpu_fused: Vec<f32> = cpu_silu
        .iter()
        .zip(cpu_up.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let cpu_down = cpu_matmul(&wd, d, inter, &cpu_fused, sp);
    let mut cpu_final = vec![0.0f32; d * sp];
    for i in 0..d * sp {
        cpu_final[i] = cpu_down[i] + cpu_after_attn[i];
    }
    let t_cpu = t_cpu_start.elapsed().as_micros();

    let speedup = t_cpu as f64 / total_hybrid_us as f64;

    println!();
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │                  TIMING BREAKDOWN                   │");
    println!("  ├─────────────────────────────────────────────────────┤");
    println!(
        "  │  ANE QKV projection:     {:>8}us                  │",
        t_qkv
    );
    println!(
        "  │  CPU Attention:           {:>8}us                  │",
        t_attn
    );
    println!(
        "  │  ANE Out proj + residual: {:>8}us                  │",
        t_out
    );
    println!(
        "  │  ANE FFN + residual:      {:>8}us                  │",
        t_ffn
    );
    println!("  ├─────────────────────────────────────────────────────┤");
    println!(
        "  │  Total hybrid:            {:>8}us                  │",
        total_hybrid_us
    );
    println!(
        "  │  CPU-only reference:      {:>8}us                  │",
        t_cpu
    );
    println!(
        "  │  Hybrid speedup:          {:>8.1}x                  │",
        speedup
    );
    println!("  └─────────────────────────────────────────────────────┘");

    // End-to-end correctness: ANE hybrid vs CPU-only
    let (mx, mn, ok, n) = compare(ffn_ane, &cpu_final);
    let pct = 100.0 * ok as f32 / n as f32;
    let sym = if pct > 95.0 {
        "PASS"
    } else if pct > 50.0 {
        "WARN"
    } else {
        "FAIL"
    };
    println!(
        "  End-to-end: {} max={:.4} mean={:.6} | {}/{} ({:.1}%)",
        sym, mx, mn, ok, n, pct
    );

    // First few values
    println!("  First 8 values: ANE hybrid vs CPU-only");
    for i in 0..8.min(d * sp) {
        println!(
            "    [{:3}] ANE={:10.4}  CPU={:10.4}  diff={:10.6}",
            i,
            ffn_ane[i],
            cpu_final[i],
            (ffn_ane[i] - cpu_final[i]).abs()
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let heads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sp: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    test_hybrid_layer(d, heads, sp);
}
