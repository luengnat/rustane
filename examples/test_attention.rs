//! Single-head attention on ANE — tests transpose + matmul ops.
//!
//! Pattern:
//!   x [1, D, 1, SP] → conv1x1 × 3 → Q, K, V each [1, D, 1, SP]
//!   → transpose Q,K,V to [1, 1, D, SP]  (perm 0,2,1,3)
//!   → transpose Q to [1, 1, SP, D]      (perm 0,1,3,2)
//!   → matmul(Q_T, K) → [1, 1, SP, SP]
//!   → scale + softmax(axis=3)
//!   → matmul(scores, V, transpose_y=true) → [1, 1, SP, D]
//!   → transpose back to [1, D, 1, SP]
//!   → conv1x1 output projection → [1, D, 1, SP]
//!
//! Usage: cargo run --example test_attention -- [D] [SP]

use half::f16;
use std::env;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// MIL GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

fn mil_attention(d: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    // ── Weight consts: Wq, Wk, Wv, Wo each [D, D, 1, 1] ──
    for wn in &["Wq", "Wk", "Wv", "Wo"] {
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

    // ── Conv params (shared) ──
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // ── QKV projections: conv1x1 each [1, D, 1, SP] ──
    for (wn, nm) in [("Wq", "q"), ("Wk", "k"), ("Wv", "v")] {
        m.push_str("        tensor<fp16, [1, ");
        m.push_str(&d.to_string());
        m.push_str(", 1, ");
        m.push_str(&sp.to_string());
        m.push_str("]> ");
        m.push_str(nm);
        m.push_str(
            " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
        );
        m.push_str(wn);
        m.push_str(", x = x)[name = tensor<string, []>(\"");
        m.push_str(nm);
        m.push_str("\")];\n");
    }

    // ── Transpose perm consts ──
    // perm (0,2,1,3): [1,D,1,SP] → [1,1,D,SP]
    m.push_str("        tensor<int32, [4]> p0213 = const()[name = tensor<string, []>(\"p0213\"), val = tensor<int32, [4]>([0, 2, 1, 3])];\n");
    // perm (0,1,3,2): [1,1,D,SP] → [1,1,SP,D]
    m.push_str("        tensor<int32, [4]> p0132 = const()[name = tensor<string, []>(\"p0132\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");

    // ── Transpose Q,K,V from [1,D,1,SP] to [1,1,D,SP] ──
    for nm in &["q", "k", "v"] {
        m.push_str("        tensor<fp16, [1, 1, ");
        m.push_str(&d.to_string());
        m.push_str(", ");
        m.push_str(&sp.to_string());
        m.push_str("]> ");
        m.push_str(nm);
        m.push_str("t = transpose(x = ");
        m.push_str(nm);
        m.push_str(", perm = p0213)[name = tensor<string, []>(\"");
        m.push_str(nm);
        m.push_str("t\")];\n");
    }

    // ── Transpose Q from [1,1,D,SP] to [1,1,SP,D] ──
    m.push_str("        tensor<fp16, [1, 1, ");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str("]> qt2 = transpose(x = qt, perm = p0132)[name = tensor<string, []>(\"qt2\")];\n");

    // ── Matmul: Q_T @ K → [1, 1, SP, SP] ──
    m.push_str("        tensor<bool, []> f = const()[name = tensor<string, []>(\"f\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<fp16, [1, 1, ");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str("]> sc = matmul(x = qt2, y = kt, transpose_x = f, transpose_y = f)[name = tensor<string, []>(\"sc\")];\n");

    // ── Scale by 1/sqrt(D) ──
    let scale_val = (1.0 / (d as f64).sqrt()) as f32;
    m.push_str("        tensor<fp16, []> sfs = const()[name = tensor<string, []>(\"sfs\"), val = tensor<fp16, []>(");
    m.push_str(&scale_val.to_string());
    m.push_str(")];\n");
    m.push_str("        tensor<fp16, [1, 1, ");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str("]> scs = mul(x = sc, y = sfs)[name = tensor<string, []>(\"scs\")];\n");

    // ── Softmax over last dim ──
    m.push_str("        tensor<int32, []> ax3 = const()[name = tensor<string, []>(\"ax3\"), val = tensor<int32, []>(3)];\n");
    m.push_str("        tensor<fp16, [1, 1, ");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str("]> att = softmax(axis = ax3, x = scs)[name = tensor<string, []>(\"att\")];\n");

    // ── Matmul: attn @ V^T → [1, 1, SP, D] ──
    m.push_str("        tensor<bool, []> t = const()[name = tensor<string, []>(\"t\"), val = tensor<bool, []>(true)];\n");
    m.push_str("        tensor<fp16, [1, 1, ");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str("]> ao = matmul(x = att, y = vt, transpose_x = f, transpose_y = t)[name = tensor<string, []>(\"ao\")];\n");

    // ── Transpose back: [1,1,SP,D] → [1,1,D,SP] → [1,D,1,SP] ──
    m.push_str("        tensor<fp16, [1, 1, ");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str("]> ao2 = transpose(x = ao, perm = p0132)[name = tensor<string, []>(\"ao2\")];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> ao3 = transpose(x = ao2, perm = p0213)[name = tensor<string, []>(\"ao3\")];\n");

    // ── Output projection: conv1x1 ──
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wo, x = ao3)[name = tensor<string, []>(\"out\")];\n");

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

fn rand_weight(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u64 * 2654435761).wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * scale
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU REFERENCE
// ═══════════════════════════════════════════════════════════════════════════

/// Conv1x1 = matmul: output[w, s] = sum_c weight[w, c] * input[c, s]
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

/// CPU single-head attention:
///   Q = Wq @ x, K = Wk @ x, V = Wv @ x   (each [D, SP])
///   scores = Q^T @ K / sqrt(D)             (each [SP, SP])
///   attn = softmax(scores, dim=1)
///   out = attn @ V^T                        ([SP, D])
///   result = Wo @ out^T                     ([D, SP])
fn cpu_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    d: usize,
    sp: usize,
) -> Vec<f32> {
    // QKV projections: [D, SP]
    let q = cpu_matmul(wq, d, d, x, sp);
    let k = cpu_matmul(wk, d, d, x, sp);
    let v = cpu_matmul(wv, d, d, x, sp);

    // Attention scores: Q^T @ K = [SP, D]^T @ [D, SP] = [SP, SP]
    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; sp * sp];
    for i in 0..sp {
        for j in 0..sp {
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += q[c * sp + i] * k[c * sp + j];
            }
            scores[i * sp + j] = dot * scale;
        }
    }

    // Softmax over dim=1 (columns)
    for i in 0..sp {
        let mut mx = f32::NEG_INFINITY;
        for j in 0..sp {
            mx = mx.max(scores[i * sp + j]);
        }
        let mut sm = 0.0f32;
        for j in 0..sp {
            let e = (scores[i * sp + j] - mx).exp();
            scores[i * sp + j] = e;
            sm += e;
        }
        for j in 0..sp {
            scores[i * sp + j] /= sm;
        }
    }

    // attn @ V^T = [SP, SP] @ [SP, D] = [SP, D]
    let mut attn_out = vec![0.0f32; sp * d];
    for i in 0..sp {
        for c in 0..d {
            let mut dot = 0.0f32;
            for j in 0..sp {
                dot += scores[i * sp + j] * v[c * sp + j];
            }
            attn_out[i * d + c] = dot;
        }
    }

    // Transpose attn_out: [SP, D] → [D, SP]
    let mut attn_out_t = vec![0.0f32; d * sp];
    for c in 0..d {
        for s in 0..sp {
            attn_out_t[c * sp + s] = attn_out[s * d + c];
        }
    }

    // Output projection: Wo @ attn_out_t = [D, D] @ [D, SP] = [D, SP]
    cpu_matmul(wo, d, d, &attn_out_t, sp)
}

// ═══════════════════════════════════════════════════════════════════════════
// ANE RUNNER
// ═══════════════════════════════════════════════════════════════════════════

fn run_ane(
    mil: &str,
    blobs: &[(&str, &[u8])],
    in_b: usize,
    out_b: usize,
    inp16: &[u8],
) -> Result<(Vec<f32>, u128), String> {
    let full_names: Vec<String> = blobs
        .iter()
        .map(|(n, _)| format!("@model_path/weights/{}.bin", n))
        .collect();
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();
    let datas: Vec<&[u8]> = blobs.iter().map(|(_, d)| d.as_ref()).collect();
    let lens: Vec<usize> = blobs.iter().map(|(_, d)| d.len()).collect();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(mil, &name_refs, &datas, &lens, &[in_b], &[out_b])
        .map_err(|e| format!("COMPILE: {}", e))?;
    exec.write_input(0, inp16)
        .map_err(|e| format!("WRITE: {}", e))?;
    // Warmup
    exec.eval().map_err(|e| format!("WARMUP: {}", e))?;
    let start = Instant::now();
    for _ in 0..2 {
        exec.eval().map_err(|e| format!("EVAL: {}", e))?;
    }
    let us = start.elapsed().as_micros() / 3;
    let raw = exec
        .read_output_vec(0)
        .map_err(|e| format!("READ: {}", e))?;
    let mut out = vec![0.0f32; raw.len() / 2];
    for i in 0..out.len() {
        let b = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
        out[i] = f16::from_bits(b).to_f32();
    }
    Ok((out, us))
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
// TEST
// ═══════════════════════════════════════════════════════════════════════════

fn test_attention(d: usize, sp: usize) {
    println!("=== Single-Head Attention (D={}, SP={}) ===", d, sp);
    println!("  Ops: 4×conv1x1 + 4×transpose + 2×matmul + softmax + scale");
    println!();

    // Use small deterministic weights for testability
    let wq = rand_weight(d * d, 0.05);
    let wk = rand_weight(d * d, 0.05);
    let wv = rand_weight(d * d, 0.05);
    let wo = rand_weight(d * d, 0.05);
    let bq = build_blob(&wq);
    let bk = build_blob(&wk);
    let bv = build_blob(&wv);
    let bo = build_blob(&wo);

    // Input
    let inp: Vec<f32> = (0..d * sp)
        .map(|i| ((i % d) as f32 * 0.1 + (i / d) as f32 * 0.05))
        .collect();

    // CPU reference
    let cpu_out = cpu_attention(&inp, &wq, &wk, &wv, &wo, d, sp);

    // ANE
    let mil = mil_attention(d, sp);
    let in_b = d * sp * 2;
    let out_b = d * sp * 2;
    let inp16 = write_fp16_scattered(&inp, d, sp);

    match run_ane(
        &mil,
        &[("Wq", &bq), ("Wk", &bk), ("Wv", &bv), ("Wo", &bo)],
        in_b,
        out_b,
        &inp16,
    ) {
        Ok((ane_out, us)) => {
            let (mx, mn, ok, n) = compare(&ane_out, &cpu_out);
            let pct = 100.0 * ok as f32 / n as f32;
            let sym = if pct > 95.0 {
                "PASS"
            } else if pct > 50.0 {
                "WARN"
            } else {
                "FAIL"
            };
            let flops: f64 = 4.0 * d as f64 * d as f64 * sp as f64 // QKV + output proj
                + 2.0 * d as f64 * sp as f64 * sp as f64 // Q@K + attn@V
                + 2.0 * sp as f64 * sp as f64; // softmax
            let tflops = flops as f64 / (us as f64 * 1e-6) / 1e12;
            println!(
                "  {:6} attention | {:6}us | max={:8.4} mean={:10.6} | {}/{} ({:5.1}%) | {:.2} TFLOPS",
                sym, us, mx, mn, ok, n, pct, tflops
            );

            // Print a few values for debugging
            println!("  First 8 values (ANE vs CPU):");
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
        Err(e) => {
            println!("  FAIL: {}", e);
            // Try to identify which op failed
            if e.contains("COMPILE") {
                println!("  → Compilation failed — MIL syntax error or unsupported op");
            } else if e.contains("EVAL") {
                println!("  → Execution failed — runtime error on ANE");
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    test_attention(d, sp);
}
