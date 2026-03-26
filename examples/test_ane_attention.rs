//! Test ANE-only attention using conv1x1 as matmul.
//!
//! Attention = softmax(Q^T @ K / sqrt(d)) @ V
//!
//! Using conv1x1 as matmul:
//!   1. QK_scores = conv1x1(K_as_weight, Q)  →  K^T @ Q, stored as [SP, SP]
//!      conv1x1 output[c, s] = sum_d K[c, d] * Q[d, s]
//!      This gives output[c, s] = sum_d K[c, d] * Q[d, s] = scores[s, c] (transposed)
//!
//!   2. softmax over channels → attn_weights[c, s] = softmax_c(scores[s, c])
//!      But we need softmax over j for each query i, i.e., softmax over c for each s.
//!      conv output layout: [1, SP_out, 1, SP_spatial] = [1, SP, 1, SP]
//!      output[c, s] = scores[s, c]. We want softmax(scores[s, :]) for each s.
//!      That's softmax over channel axis (axis=1) for each spatial position s.
//!      Wait — output[c, s] = scores[s, c]. Softmax over channels means:
//!      for each s, normalize output[:, s] which is scores[s, :] — yes, that's correct!
//!
//!   3. attn_out = conv1x1(V_as_weight, attn_weights)  →  V^T @ attn_weights
//!      conv1x1 output[d, s] = sum_j V^T[d, j] * attn_weights[j, s]
//!                            = sum_j V[j, d] * attn_weights[j, s]
//!      We want: output[d, s] = sum_j attn_weights[j, s] * V[d, j]  ✓ PERFECT!
//!
//! Wait — attn_weights[j, s] after softmax is softmax over j of scores[s, j].
//! scores[s, j] = sum_d Q[d, s] * K[d, j].
//! So attn_weights[j, s] = softmax_j(scores[s, :]) = softmax_j(Q[:, s] · K[:, j]).
//! That's softmax over keys for query at position s. ✓
//!
//! And output[d, s] = sum_j attn_weights[j, s] * V[d, j] = sum_j attn(s, j) * V[d, j] ✓
//!
//! Usage: cargo run --example test_ane_attention -- [D] [SP]

use half::f16;
use std::env;
use std::time::Instant;

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

/// ANE attention: QKV projection + attention + output projection + residual
/// All in a single fused program.
fn mil_ane_attention(d: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x, tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> residual) {\n");

    // ── Weights: Wq, Wk, Wv [D, D, 1, 1] each ──
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

    // ── Conv params ──
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // ── Q, K, V projections ──
    for (i, wn) in ["Wq", "Wk", "Wv"].iter().enumerate() {
        let out_name = ["q", "k", "v"][i];
        m.push_str("        tensor<fp16, [1, ");
        m.push_str(&d.to_string());
        m.push_str(", 1, ");
        m.push_str(&sp.to_string());
        m.push_str("]> ");
        m.push_str(out_name);
        m.push_str(
            " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
        );
        m.push_str(wn);
        m.push_str(", x = x)[name = tensor<string, []>(\"");
        m.push_str(out_name);
        m.push_str("\")];\n");
    }

    // ── QK scores = K^T @ Q via conv1x1 ──
    // K_as_weight is stored in a separate weight blob: [SP, D, 1, 1]
    // But K is computed, not a const. We can't use conv1x1 with a non-const weight.
    //
    // ALTERNATIVE: Use the "transpose trick"
    // conv1x1(Q_weight_transposed, K_input) won't work because weights must be const.
    //
    // CONCLUSION: We CANNOT do Q @ K^T entirely on ANE with conv1x1 because
    // conv1x1 weights must be constants, but Q and K are both computed from input.
    //
    // The only ANE ops that take two computed tensors as inputs are:
    //   mul (element-wise), concat, softmax, sigmoid
    // None of these do matrix multiplication.
    //
    // So attention REQUIRES either matmul op (which doesn't work on ANE) or CPU.

    // Placeholder: just output q for now
    m.push_str("    } -> (q);\n}\n");
    m
}

/// Full ANE attention via conv1x1 trick: pre-compute K as weight, Q as input.
/// This works when K is fixed (e.g., cached key-value pairs).
/// For self-attention where both Q and K depend on input, this doesn't work.
fn mil_qk_via_conv1x1(d: usize, sp: usize) -> String {
    // WK_qk is [SP, D, 1, 1] — K values arranged as conv1x1 weight
    // Input is Q [1, D, 1, SP]
    // Output: [1, SP, 1, SP] = K^T @ Q = scores transposed
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> q) {\n");

    // WK_qk weight: [SP, D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> WK = const()[name = tensor<string, []>(\"WK\"), val = tensor<fp16, [");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WK.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    // Scale factor for attention: 1/sqrt(d)
    // We can apply this by scaling the weight: WK_scaled = WK / sqrt(d)
    // Or use mul after conv1x1

    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // QK scores via conv1x1
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> scores = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WK, x = q)[name = tensor<string, []>(\"qk\")];\n");

    // Softmax over channels (axis=1) for each spatial position
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> attn = softmax(axis = ax, x = scores)[name = tensor<string, []>(\"sm\")];\n");

    // attn @ V via conv1x1: Wv_attn is [D, SP, 1, 1], attn is [1, SP, 1, SP]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, 1]> WV = const()[name = tensor<string, []>(\"WV\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WV.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> out = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WV, x = attn)[name = tensor<string, []>(\"av\")];\n");

    m.push_str("    } -> (out);\n}\n");
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
// TEST: ANE attention using conv1x1 for QK^T and attn@V
// ═══════════════════════════════════════════════════════════════════════════

fn test_ane_attention(d: usize, sp: usize) {
    println!("=== ANE Attention via conv1x1 (D={}, SP={}) ===", d, sp);
    println!("  Strategy: K stored as conv1x1 weight [SP, D, 1, 1]");
    println!("            Q as input [1, D, 1, SP]");
    println!("            conv1x1 → K^T @ Q → [1, SP, 1, SP]");
    println!("            softmax → attn weights");
    println!("            conv1x1(V_weight, attn) → attn @ V");
    println!();

    // Create K, V as "pre-computed" values (simulating cached KV)
    let k_values = rand_weight(d * sp, 0.02); // K [D, SP]
    let v_values = rand_weight(d * sp, 0.02); // V [D, SP]
    let q_values = rand_weight(d * sp, 0.02); // Q [D, SP]

    // Create K as conv1x1 weight: [SP, D, 1, 1]
    // WK[c, d_idx] = K[d_idx, c]  (transpose K for conv1x1)
    let scale = 1.0 / (d as f32).sqrt();
    let mut wk = vec![0.0f32; sp * d];
    for c in 0..sp {
        for d_idx in 0..d {
            wk[c * d + d_idx] = k_values[d_idx * sp + c] * scale;
        }
    }

    // Create V as conv1x1 weight: [D, SP, 1, 1]
    // WV[d, j] = V[d, j]
    let mut wv = vec![0.0f32; d * sp];
    for d_idx in 0..d {
        for j in 0..sp {
            wv[d_idx * sp + j] = v_values[d_idx * sp + j];
        }
    }

    let bk = build_blob(&wk);
    let bv = build_blob(&wv);

    let mil = mil_qk_via_conv1x1(d, sp);
    let q16 = write_fp16_scattered(&q_values, d, sp);

    match run_ane(
        &mil,
        &[("WK", &bk), ("WV", &bv)],
        &[d * sp * 2],
        &[d * sp * 2],
        &[&q16],
    ) {
        Ok((ane_out, us)) => {
            // CPU reference
            let cpu_out = cpu_attention(&q_values, &k_values, &v_values, d, sp);

            let (mx, mn, ok, n) = compare(&ane_out, &cpu_out);
            let pct = 100.0 * ok as f32 / n as f32;
            let sym = if pct > 95.0 {
                "PASS"
            } else if pct > 50.0 {
                "WARN"
            } else {
                "FAIL"
            };
            println!(
                "  {:6} ane_attention | {:6}us | max={:8.4} mean={:10.6} | {}/{} ({:5.1}%)",
                sym, us, mx, mn, ok, n, pct
            );

            // Print first few values
            println!("  First 8: ANE vs CPU");
            for i in 0..8.min(d * sp) {
                println!(
                    "    [{:3}] ANE={:10.4}  CPU={:10.4}  diff={:10.6}",
                    i,
                    ane_out[i],
                    cpu_out[i],
                    (ane_out[i] - cpu_out[i]).abs()
                );
            }
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    test_ane_attention(d, sp);
}
