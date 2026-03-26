//! Fully-ANE transformer layer for inference (prefill).
//!
//! For prefill (processing entire prompt at once):
//!   Program 1: QKV projections (fused conv1x1 × 3 + concat)
//!   [CPU: extract K, V from QKV output, build weight blobs]
//!   Program 2: ANE attention (conv1x1 K → softmax → conv1x1 V) + out_proj + residual
//!   Program 3: FFN + residual
//!
//! For autoregressive decode (KV-cache):
//!   Program 1: QKV projections for new token
//!   [CPU: append to KV-cache, build weight blobs from cache]
//!   Program 2: ANE attention (cached K, V as weights) + out_proj + residual
//!   Program 3: FFN + residual
//!
//! Limitation: Program 2 must be recompiled when K/V change (new token or prefill).
//! This tests whether compile+eval is still faster than CPU attention.
//!
//! Usage: cargo run --example test_full_ane_layer -- [D] [num_heads] [SP]

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

/// Program 1: QKV projections (same as before)
fn mil_qkv(d: usize, sp: usize) -> String {
    let total = 3 * d;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
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
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
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

/// Program 2: ANE attention + out_proj + residual
/// Inputs: Q [1, D, 1, SP], residual x [1, D, 1, SP]
/// K and V are baked as const weights (recompiled per forward pass for self-attn)
/// Output: [1, D, 1, SP] = Wo @ attn_out + residual
fn mil_attn_out_residual(d: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> q, tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> residual) {\n");

    // K as conv1x1 weight: [SP, D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> WK = const()[name = tensor<string, []>(\"WK\"), val = tensor<fp16, [");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WK.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    // V as conv1x1 weight: [D, SP, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, 1]> WV = const()[name = tensor<string, []>(\"WV\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WV.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    // WoI = [Wo | I]: [D, 2D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&(2 * d).to_string());
    m.push_str(", 1, 1]> WoI = const()[name = tensor<string, []>(\"WoI\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&(2 * d).to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WoI.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // QK scores: conv1x1(K_weight, Q) → [1, SP, 1, SP]
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> scores = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WK, x = q)[name = tensor<string, []>(\"qk\")];\n");

    // Softmax over channels
    m.push_str("        tensor<int32, []> ax1 = const()[name = tensor<string, []>(\"ax1\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> attn = softmax(axis = ax1, x = scores)[name = tensor<string, []>(\"sm\")];\n");

    // attn @ V: conv1x1(V_weight, attn) → [1, D, 1, SP]
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> attn_out = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WV, x = attn)[name = tensor<string, []>(\"av\")];\n");

    // Concat attn_out with residual for output projection + residual
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax2 = const()[name = tensor<string, []>(\"ax2\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&(2 * d).to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (attn_out, residual), axis = ax2, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");

    // Out projection + residual via conv1x1([Wo|I])
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WoI, x = cat)[name = tensor<string, []>(\"co\")];\n");

    m.push_str("    } -> (y);\n}\n");
    m
}

/// Program 3: FFN + residual (same as test_residual_ffn)
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
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
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
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n");
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (fused, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");
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

fn write_fp16(input: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; input.len() * 2];
    for (i, &w) in input.iter().enumerate() {
        let b = f16::from_f32(w).to_bits();
        buf[i * 2] = (b & 0xFF) as u8;
        buf[i * 2 + 1] = (b >> 8) as u8;
    }
    buf
}

fn read_fp16(raw: &[u8]) -> Vec<f32> {
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

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, sp: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; sp * sp];
    for i in 0..sp {
        for j in 0..sp {
            let mut dot = 0.0f32;
            for h in 0..d {
                dot += q[h * sp + i] * k[h * sp + j];
            }
            scores[i * sp + j] = dot * scale;
        }
    }
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

fn run_ane(
    mil: &str,
    blobs: &[(&str, &[u8])],
    in_sizes: &[usize],
    out_sizes: &[usize],
    inputs: &[&[u8]],
) -> Result<(Vec<f32>, u128, u128), String> {
    let full_names: Vec<String> = blobs
        .iter()
        .map(|(n, _)| format!("@model_path/weights/{}.bin", n))
        .collect();
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();
    let datas: Vec<&[u8]> = blobs.iter().map(|(_, d)| d.as_ref()).collect();
    let lens: Vec<usize> = blobs.iter().map(|(_, d)| d.len()).collect();
    let t_compile = Instant::now();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(mil, &name_refs, &datas, &lens, in_sizes, out_sizes)
        .map_err(|e| format!("COMPILE: {}", e))?;
    let compile_us = t_compile.elapsed().as_micros();
    for (i, inp) in inputs.iter().enumerate() {
        exec.write_input(i, *inp)
            .map_err(|e| format!("WRITE[{}]: {}", i, e))?;
    }
    exec.eval().map_err(|e| format!("WARMUP: {}", e))?;
    let t_eval = Instant::now();
    for _ in 0..2 {
        exec.eval().map_err(|e| format!("EVAL: {}", e))?;
    }
    let eval_us = t_eval.elapsed().as_micros() / 3;
    let raw = exec
        .read_output_vec(0)
        .map_err(|e| format!("READ: {}", e))?;
    Ok((read_fp16(&raw), eval_us, compile_us))
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
// FULLY-ANE LAYER TEST
// ═══════════════════════════════════════════════════════════════════════════

fn test_full_ane_layer(d: usize, _num_heads: usize, sp: usize) {
    let inter = d * 4;
    println!(
        "=== Fully-ANE Transformer Layer (D={}, SP={}, inter={}) ===",
        d, sp, inter
    );
    println!(
        "  Pipeline: ANE(QKV) → [CPU: build K/V weights] → ANE(attn+out_proj+res) → ANE(FFN+res)"
    );
    println!();

    // Weights
    let wq = rand_weight(d * d, 0.02);
    let wk = rand_weight(d * d, 0.02);
    let wv = rand_weight(d * d, 0.02);
    let wo = rand_weight(d * d, 0.02);
    let wg = rand_weight(inter * d, 0.02);
    let wu = rand_weight(inter * d, 0.02);
    let wd = rand_weight(d * inter, 0.02);

    // WoI = [Wo | I]: [D, 2D, 1, 1]
    let mut woi = vec![0.0f32; d * 2 * d];
    for r in 0..d {
        for c in 0..d {
            woi[r * 2 * d + c] = wo[r * d + c];
        }
        woi[r * 2 * d + d + r] = 1.0;
    }

    // WdI = [Wd | I]: [D, 5D, 1, 1]
    let mut wdi = vec![0.0f32; d * (inter + d)];
    for r in 0..d {
        for c in 0..inter {
            wdi[r * (inter + d) + c] = wd[r * inter + c];
        }
        wdi[r * (inter + d) + inter + r] = 1.0;
    }

    let bq = build_blob(&wq);
    let bk = build_blob(&wk);
    let bv = build_blob(&wv);
    let boi = build_blob(&woi);
    let bg = build_blob(&wg);
    let bu = build_blob(&wu);
    let bdi = build_blob(&wdi);

    // Input
    let inp: Vec<f32> = (0..d * sp).map(|i| (i % d) as f32 * 0.1 + 1.0).collect();
    let inp16 = write_fp16(&inp);

    // ── Program 1: QKV projections (compiled once per layer) ──
    let mil1 = mil_qkv(d, sp);
    let (qkv_ane, t_qkv_eval, t_qkv_compile) = run_ane(
        &mil1,
        &[("Wq", &bq), ("Wk", &bk), ("Wv", &bv)],
        &[d * sp * 2],
        &[3 * d * sp * 2],
        &[&inp16],
    )
    .unwrap();

    let q_ane = &qkv_ane[0..d * sp];
    let k_ane = &qkv_ane[d * sp..2 * d * sp];
    let v_ane = &qkv_ane[2 * d * sp..3 * d * sp];

    println!(
        "  Program 1 (QKV): compile={}us eval={}us",
        t_qkv_compile, t_qkv_eval
    );

    // ── Build K, V as conv1x1 weights ──
    let t_build = Instant::now();
    let scale = 1.0 / (d as f32).sqrt();
    // WK[c, d_idx] = K[d_idx, c] * scale  →  [SP, D]
    let mut wk_conv = vec![0.0f32; sp * d];
    for c in 0..sp {
        for d_idx in 0..d {
            wk_conv[c * d + d_idx] = k_ane[d_idx * sp + c] * scale;
        }
    }
    // WV[d, j] = V[d, j]  →  [D, SP]
    let mut wv_conv = vec![0.0f32; d * sp];
    for d_idx in 0..d {
        for j in 0..sp {
            wv_conv[d_idx * sp + j] = v_ane[d_idx * sp + j];
        }
    }
    let bk_conv = build_blob(&wk_conv);
    let bv_conv = build_blob(&wv_conv);
    let t_build_us = t_build.elapsed().as_micros();
    println!("  Build K/V weights: {}us", t_build_us);

    // ── Program 2: ANE attention + out_proj + residual (recompiled per forward) ──
    let mil2 = mil_attn_out_residual(d, sp);
    let q16 = write_fp16(q_ane);
    let (after_attn_ane, t_attn_eval, t_attn_compile) = run_ane(
        &mil2,
        &[("WK", &bk_conv), ("WV", &bv_conv), ("WoI", &boi)],
        &[d * sp * 2, d * sp * 2],
        &[d * sp * 2],
        &[&q16, &inp16],
    )
    .unwrap();

    println!(
        "  Program 2 (attn+out): compile={}us eval={}us",
        t_attn_compile, t_attn_eval
    );

    // ── Program 3: FFN + residual (compiled once per layer) ──
    let mil3 = mil_ffn_residual(d, sp);
    let after_attn16 = write_fp16(&after_attn_ane);
    let (ffn_ane, t_ffn_eval, t_ffn_compile) = run_ane(
        &mil3,
        &[("Wg", &bg), ("Wu", &bu), ("WdI", &bdi)],
        &[d * sp * 2],
        &[d * sp * 2],
        &[&after_attn16],
    )
    .unwrap();

    println!(
        "  Program 3 (FFN+res): compile={}us eval={}us",
        t_ffn_compile, t_ffn_eval
    );

    // ── CPU reference ──
    let t_cpu_start = Instant::now();
    let q_cpu = cpu_matmul(&wq, d, d, &inp, sp);
    let k_cpu = cpu_matmul(&wk, d, d, &inp, sp);
    let v_cpu = cpu_matmul(&wv, d, d, &inp, sp);
    let attn_cpu = cpu_attention(&q_cpu, &k_cpu, &v_cpu, d, sp);
    let mut op_cpu = cpu_matmul(&wo, d, d, &attn_cpu, sp);
    for i in 0..d * sp {
        op_cpu[i] += inp[i];
    }
    let gate_cpu = cpu_matmul(&wg, inter, d, &op_cpu, sp);
    let up_cpu = cpu_matmul(&wu, inter, d, &op_cpu, sp);
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
    let mut cpu_final = vec![0.0f32; d * sp];
    for i in 0..d * sp {
        cpu_final[i] = down_cpu[i] + op_cpu[i];
    }
    let t_cpu = t_cpu_start.elapsed().as_micros();

    // ── Summary ──
    let total_eval = t_qkv_eval + t_attn_eval + t_ffn_eval;
    let total_compile = t_qkv_compile + t_attn_compile + t_ffn_compile;
    let total_ane = total_eval + total_compile as u128 + t_build_us;
    // For prefill: QKV compile once, attn compile every time, FFN compile once
    // For subsequent tokens (decode): QKV compile once, attn compile every time, FFN compile once
    let decode_overhead = t_attn_compile + t_build_us; // per-token overhead

    let speedup_total = t_cpu as f64 / total_ane as f64;
    let speedup_eval = t_cpu as f64 / total_eval as f64;

    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │                 FULLY-ANE LAYER TIMING                  │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │  ANE eval only:          {:>8}us  ({:.1}x vs CPU)    │",
        total_eval, speedup_eval
    );
    println!(
        "  │  ANE compile total:      {:>8}us                      │",
        total_compile
    );
    println!(
        "  │  Build K/V weights:      {:>8}us                      │",
        t_build_us
    );
    println!(
        "  │  Total (compile+eval):   {:>8}us  ({:.1}x vs CPU)    │",
        total_ane, speedup_total
    );
    println!(
        "  │  CPU-only reference:     {:>8}us                      │",
        t_cpu
    );
    println!(
        "  │  Decode overhead/token:  {:>8}us                      │",
        decode_overhead
    );
    println!("  └─────────────────────────────────────────────────────────┘");

    // Correctness
    let (mx, mn, ok, n) = compare(&ffn_ane, &cpu_final);
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

    println!("  First 8: ANE vs CPU");
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
    test_full_ane_layer(d, heads, sp);
}
