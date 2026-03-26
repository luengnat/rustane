//! Multi-layer hybrid transformer forward pass.
//!
//! Stacks N transformer layers, each using:
//!   ANE(QKV) → CPU(attention) → ANE(out_proj+res) → ANE(FFN+res)
//!
//! Each layer has its own weights. ANE programs are compiled once per layer.
//! Tests correctness end-to-end and measures total throughput.
//!
//! Usage: cargo run --example test_multilayer -- [D] [num_heads] [SP] [num_layers]

use half::f16;
use std::env;
use std::time::Instant;

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

// ═══════════════════════════════════════════════════════════════════════════
// MIL GENERATORS (same as test_hybrid_layer but with unique weight names)
// ═══════════════════════════════════════════════════════════════════════════

fn mil_qkv(d: usize, sp: usize, layer: usize) -> String {
    let total = 3 * d;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    for wn in ["Wq", "Wk", "Wv"] {
        let full_name = format!("L{}{}", layer, wn);
        m.push_str("        tensor<fp16, [");
        m.push_str(&d.to_string());
        m.push_str(", ");
        m.push_str(&d.to_string());
        m.push_str(", 1, 1]> ");
        m.push_str(&full_name);
        m.push_str(" = const()[name = tensor<string, []>(\"");
        m.push_str(&full_name);
        m.push_str("\"), val = tensor<fp16, [");
        m.push_str(&d.to_string());
        m.push_str(", ");
        m.push_str(&d.to_string());
        m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/");
        m.push_str(&full_name);
        m.push_str(".bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    }

    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    let names = ["Wq", "Wk", "Wv"];
    for i in 0..3 {
        let full_name = format!("L{}{}", layer, names[i]);
        m.push_str("        tensor<fp16, [1, ");
        m.push_str(&d.to_string());
        m.push_str(", 1, ");
        m.push_str(&sp.to_string());
        m.push_str("]> q");
        m.push_str(&i.to_string());
        m.push_str(
            " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
        );
        m.push_str(&full_name);
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

fn mil_out_proj_residual(d: usize, sp: usize, layer: usize) -> String {
    let total_ic = 2 * d;
    let wname = format!("L{}WoI", layer);
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> attn_out, tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]> ");
    m.push_str(&wname);
    m.push_str(" = const()[name = tensor<string, []>(\"");
    m.push_str(&wname);
    m.push_str("\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/");
    m.push_str(&wname);
    m.push_str(".bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (attn_out, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");

    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str(
        "]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
    );
    m.push_str(&wname);
    m.push_str(", x = cat)[name = tensor<string, []>(\"co\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn mil_ffn_residual(d: usize, sp: usize, layer: usize) -> String {
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
        let full_name = format!("L{}{}", layer, wn);
        m.push_str("        tensor<fp16, [");
        m.push_str(&oc.to_string());
        m.push_str(", ");
        m.push_str(&ic.to_string());
        m.push_str(", 1, 1]> ");
        m.push_str(&full_name);
        m.push_str(" = const()[name = tensor<string, []>(\"");
        m.push_str(&full_name);
        m.push_str("\"), val = tensor<fp16, [");
        m.push_str(&oc.to_string());
        m.push_str(", ");
        m.push_str(&ic.to_string());
        m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/");
        m.push_str(&full_name);
        m.push_str(".bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    }

    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    let wg_name = format!("L{}Wg", layer);
    let wu_name = format!("L{}Wu", layer);
    let wdi_name = format!("L{}WdI", layer);

    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> gate = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ");
    m.push_str(&wg_name);
    m.push_str(", x = x)[name = tensor<string, []>(\"cg\")];\n");

    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> up = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ");
    m.push_str(&wu_name);
    m.push_str(", x = x)[name = tensor<string, []>(\"cu\")];\n");

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
    m.push_str(
        "]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
    );
    m.push_str(&wdi_name);
    m.push_str(", x = cat)[name = tensor<string, []>(\"cd\")];\n");
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

fn write_fp16_scattered(input: &[f32], dim: usize, _sp: usize) -> Vec<u8> {
    let mut buf = vec![0u8; dim * input.len() / dim * 2];
    let sp = input.len() / dim;
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

fn rand_weight(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u64 * 2654435761)
                .wrapping_add(seed)
                .wrapping_mul(0x9E3779B97F4A7C15)
                >> 33) as f32
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
    let mut scores = vec![0.0f32; sp * sp];
    for i in 0..sp {
        for j in 0..sp {
            let mut dot = 0.0f32;
            for h in 0..d {
                dot += q[h * sp + i] * k[h * sp + j];
            }
            scores[i * sp + j] = dot / (d as f32).sqrt();
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

// ═══════════════════════════════════════════════════════════════════════════
// ANE RUNNER
// ═══════════════════════════════════════════════════════════════════════════

fn run_ane(
    mil: &str,
    blobs: &[(&str, &[u8])],
    input_sizes: &[usize],
    output_sizes: &[usize],
    inputs: &[&[u8]],
) -> Result<(Vec<f32>, u128), String> {
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
    exec.eval().map_err(|e| format!("WARMUP: {}", e))?;
    let start = Instant::now();
    for _ in 0..2 {
        exec.eval().map_err(|e| format!("EVAL: {}", e))?;
    }
    let us = start.elapsed().as_micros() / 3;
    let raw = exec
        .read_output_vec(0)
        .map_err(|e| format!("READ: {}", e))?;
    Ok((read_fp16_scattered(&raw), us))
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
// MULTI-LAYER TEST
// ═══════════════════════════════════════════════════════════════════════════

struct LayerWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    woi: Vec<f32>,
    wdi: Vec<f32>,
    blobs_qkv: Vec<(String, Vec<u8>)>,
    blob_woi: Vec<u8>,
    blobs_ffn: Vec<(String, Vec<u8>)>,
}

fn make_layer_weights(d: usize, layer: usize) -> LayerWeights {
    let inter = d * 4;
    let seed = (layer + 1) as u64 * 12345;
    let wq = rand_weight(d * d, 0.02, seed);
    let wk = rand_weight(d * d, 0.02, seed + 1);
    let wv = rand_weight(d * d, 0.02, seed + 2);
    let wo = rand_weight(d * d, 0.02, seed + 3);
    let wg = rand_weight(inter * d, 0.02, seed + 4);
    let wu = rand_weight(inter * d, 0.02, seed + 5);
    let wd = rand_weight(d * inter, 0.02, seed + 6);

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

    let blobs_qkv = vec![
        (format!("L{}Wq", layer), build_blob(&wq)),
        (format!("L{}Wk", layer), build_blob(&wk)),
        (format!("L{}Wv", layer), build_blob(&wv)),
    ];
    let blob_woi = build_blob(&woi);
    let blobs_ffn = vec![
        (format!("L{}Wg", layer), build_blob(&wg)),
        (format!("L{}Wu", layer), build_blob(&wu)),
        (format!("L{}WdI", layer), build_blob(&wdi)),
    ];

    LayerWeights {
        wq,
        wk,
        wv,
        wo,
        wg,
        wu,
        wd,
        woi,
        wdi,
        blobs_qkv,
        blob_woi,
        blobs_ffn,
    }
}

fn test_multilayer(d: usize, num_heads: usize, sp: usize, num_layers: usize) {
    let inter = d * 4;
    println!(
        "=== Multi-Layer Hybrid Transformer (D={}, heads={}, SP={}, layers={}) ===",
        d, num_heads, sp, num_layers
    );
    println!();

    // Create per-layer weights
    let layers: Vec<LayerWeights> = (0..num_layers).map(|l| make_layer_weights(d, l)).collect();

    // Compile ANE programs for each layer
    println!("  Compiling {} ANE programs...", num_layers * 3);
    let mut qkv_mils: Vec<String> = Vec::new();
    let mut out_mils: Vec<String> = Vec::new();
    let mut ffn_mils: Vec<String> = Vec::new();
    for l in 0..num_layers {
        qkv_mils.push(mil_qkv(d, sp, l));
        out_mils.push(mil_out_proj_residual(d, sp, l));
        ffn_mils.push(mil_ffn_residual(d, sp, l));
    }
    println!("  Compiled.");

    // Input
    let inp: Vec<f32> = (0..d * sp).map(|i| (i % d) as f32 * 0.1 + 1.0).collect();
    let inp16 = write_fp16_scattered(&inp, d, sp);

    // ── ANE hybrid forward pass ──
    let mut current_ane = inp.clone();
    let mut current_ane16 = inp16;
    let mut total_ane_us: u128 = 0;
    let mut total_attn_us: u128 = 0;

    for l in 0..num_layers {
        // ANE QKV
        let lw = &layers[l];
        let qkv_blobs: Vec<(&str, &[u8])> = lw
            .blobs_qkv
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect();
        let (qkv, t) = run_ane(
            &qkv_mils[l],
            &qkv_blobs,
            &[d * sp * 2],
            &[3 * d * sp * 2],
            &[&current_ane16],
        )
        .unwrap();
        total_ane_us += t;
        let q = &qkv[0..d * sp];
        let k = &qkv[d * sp..2 * d * sp];
        let v = &qkv[2 * d * sp..3 * d * sp];

        // CPU Attention
        let t_attn_start = Instant::now();
        let attn_out = cpu_attention(q, k, v, d, sp);
        let t_attn = t_attn_start.elapsed().as_micros();
        total_attn_us += t_attn;

        // ANE Out proj + residual
        let attn_out16 = write_fp16_scattered(&attn_out, d, sp);
        let woi_blobs: Vec<(&str, &[u8])> = vec![(lw.blobs_qkv[0].0.as_str(), &[] as &[u8])]; // dummy
        let woi_name = format!("L{}WoI", l);
        let (after_attn, t) = run_ane(
            &out_mils[l],
            &[(woi_name.as_str(), lw.blob_woi.as_slice())],
            &[d * sp * 2, d * sp * 2],
            &[d * sp * 2],
            &[&attn_out16, &current_ane16],
        )
        .unwrap();
        total_ane_us += t;

        // ANE FFN + residual
        let after_attn16 = write_fp16_scattered(&after_attn, d, sp);
        let ffn_blobs: Vec<(&str, &[u8])> = lw
            .blobs_ffn
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect();
        let (layer_out, t) = run_ane(
            &ffn_mils[l],
            &ffn_blobs,
            &[d * sp * 2],
            &[d * sp * 2],
            &[&after_attn16],
        )
        .unwrap();
        total_ane_us += t;

        current_ane = layer_out;
        current_ane16 = write_fp16_scattered(&current_ane, d, sp);

        println!(
            "  Layer {}: QKV={}us Attn={}us OutProj={}us FFN={}us",
            l,
            qkv_mils.len(),
            t_attn,
            t,
            t
        );
    }

    // ── CPU-only reference ──
    let t_cpu_start = Instant::now();
    let mut current_cpu = inp.clone();
    for l in 0..num_layers {
        let lw = &layers[l];
        let q = cpu_matmul(&lw.wq, d, d, &current_cpu, sp);
        let k = cpu_matmul(&lw.wk, d, d, &current_cpu, sp);
        let v = cpu_matmul(&lw.wv, d, d, &current_cpu, sp);
        let attn = cpu_attention(&q, &k, &v, d, sp);
        let mut out_proj = cpu_matmul(&lw.wo, d, d, &attn, sp);
        for i in 0..d * sp {
            out_proj[i] += current_cpu[i];
        } // residual
        let gate = cpu_matmul(&lw.wg, inter, d, &out_proj, sp);
        let up = cpu_matmul(&lw.wu, inter, d, &out_proj, sp);
        let silu: Vec<f32> = gate
            .iter()
            .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
            .collect();
        let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
        let down = cpu_matmul(&lw.wd, d, inter, &fused, sp);
        current_cpu = down
            .iter()
            .zip(out_proj.iter())
            .map(|(&d, &r)| d + r)
            .collect();
    }
    let t_cpu = t_cpu_start.elapsed().as_micros();

    let total_hybrid_us = total_ane_us + total_attn_us;
    let speedup = t_cpu as f64 / total_hybrid_us as f64;

    println!();
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!(
        "  │              {}-LAYER RESULTS                       │",
        num_layers
    );
    println!("  ├─────────────────────────────────────────────────────┤");
    println!(
        "  │  Total ANE time:        {:>10}us ({} evals)     │",
        total_ane_us,
        num_layers * 3
    );
    println!(
        "  │  Total CPU attention:   {:>10}us ({} evals)     │",
        total_attn_us, num_layers
    );
    println!(
        "  │  Total hybrid:          {:>10}us                  │",
        total_hybrid_us
    );
    println!(
        "  │  CPU-only reference:    {:>10}us                  │",
        t_cpu
    );
    println!(
        "  │  Hybrid speedup:        {:>10.1}x                  │",
        speedup
    );
    println!("  └─────────────────────────────────────────────────────┘");

    // End-to-end correctness
    let (mx, mn, ok, n) = compare(&current_ane, &current_cpu);
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

    println!("  First 8 values: ANE hybrid vs CPU-only");
    for i in 0..8.min(d * sp) {
        println!(
            "    [{:3}] ANE={:10.4}  CPU={:10.4}  diff={:10.6}",
            i,
            current_ane[i],
            current_cpu[i],
            (current_ane[i] - current_cpu[i]).abs()
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let heads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sp: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let nl: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    test_multilayer(d, heads, sp, nl);
}
