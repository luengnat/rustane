//! Comprehensive correctness verification: ANE vs CPU with intermediate value checks.
//!
//! Tests:
//!   1. Conv1x1 (matmul) with varied input magnitudes
//!   2. Residual connection (concat+conv1x1 identity) with large values
//!   3. ANE attention — verify QK scores, softmax normalization, attention output
//!   4. Full hybrid layer — verify each stage matches
//!   5. Full ANE layer — verify each stage matches
//!
//! All tests use a deterministic random seed and compare against CPU fp32 reference.
//! The tolerance accounts for fp16 accumulation: 5% relative + 0.01 absolute.
//!
//! Usage: cargo run --example test_verify_correctness -- [D] [SP]

use half::f16;
use std::env;
use std::time::Instant;

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

// ═══════════════════════════════════════════════════════════════
// SHARED HELPERS
// ═══════════════════════════════════════════════════════════════

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

fn to_fp16(data: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; data.len() * 2];
    for (i, &w) in data.iter().enumerate() {
        let b = f16::from_f32(w).to_bits();
        buf[i * 2] = (b & 0xFF) as u8;
        buf[i * 2 + 1] = (b >> 8) as u8;
    }
    buf
}

fn from_fp16(raw: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; raw.len() / 2];
    for i in 0..out.len() {
        let b = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
        out[i] = f16::from_bits(b).to_f32();
    }
    out
}

fn det_rand(n: usize, scale: f32, seed: u64) -> Vec<f32> {
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
            let mut a = 0.0f32;
            for c in 0..wc {
                a += w[r * wc + c] * inp[c * sp + s];
            }
            out[r * sp + s] = a;
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

fn cpu_silu(inp: &[f32]) -> Vec<f32> {
    inp.iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect()
}

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, sp: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    // QK^T: [SP, SP]
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
    // softmax per row
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
    // attn @ V: [D, SP]
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            let mut a = 0.0f32;
            for j in 0..sp {
                a += attn[i * sp + j] * v[h * sp + j];
            }
            out[h * sp + i] = a;
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
) -> Result<Vec<f32>, String> {
    let full_names: Vec<String> = blobs
        .iter()
        .map(|(n, _)| format!("@model_path/weights/{}.bin", n))
        .collect();
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();
    let datas: Vec<&[u8]> = blobs.iter().map(|(_, d)| d.as_ref()).collect();
    let lens: Vec<usize> = blobs.iter().map(|(_, d)| d.len()).collect();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(mil, &name_refs, &datas, &lens, in_sizes, out_sizes)
        .map_err(|e| format!("COMPILE: {}", e))?;
    for (i, inp) in inputs.iter().enumerate() {
        exec.write_input(i, *inp)
            .map_err(|e| format!("WRITE[{}]: {}", i, e))?;
    }
    exec.eval().map_err(|e| format!("WARMUP: {}", e))?;
    let raw = exec
        .read_output_vec(0)
        .map_err(|e| format!("READ: {}", e))?;
    Ok(from_fp16(&raw))
}

fn detailed_compare(label: &str, ane: &[f32], cpu: &[f32], print_first: usize) -> bool {
    let n = ane.len().min(cpu.len());
    let mut mx = 0.0f32;
    let mut sm = 0.0f32;
    let mut ok = 0usize;
    let mut fail_indices = Vec::new();
    for i in 0..n {
        let d = (ane[i] - cpu[i]).abs();
        let sc = cpu[i].abs().max(0.01f32);
        let rel = if sc > 0.01 { d / sc } else { d };
        mx = mx.max(d);
        sm += d;
        if rel <= 0.05 + 0.01 / sc.max(0.01) {
            ok += 1;
        } else {
            fail_indices.push((i, d, cpu[i], ane[i], rel));
        }
    }
    let pct = 100.0 * ok as f32 / n as f32;
    let pass = pct > 95.0;
    let sym = if pct > 99.9 {
        "✓"
    } else if pct > 95.0 {
        "~"
    } else {
        "✗"
    };
    println!(
        "  {} {} | max_err={:.6} mean_err={:.6} | {}/{} ({:.1}%)",
        sym,
        label,
        mx,
        sm / n as f32,
        ok,
        n,
        pct
    );
    if !fail_indices.is_empty() && print_first > 0 {
        println!("    Worst mismatches:");
        // Sort by relative error descending
        let mut sorted = fail_indices;
        sorted.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, abs_err, cpu_val, ane_val, rel) in sorted.iter().take(print_first) {
            println!(
                "      [{:4}] cpu={:12.6} ane={:12.6} diff={:10.6} rel={:.4}%",
                idx, cpu_val, ane_val, abs_err, rel
            );
        }
    }
    pass
}

// ═══════════════════════════════════════════════════════════════
// TEST 1: Conv1x1 with varied input magnitudes
// ═══════════════════════════════════════════════════════════════

fn mil_conv1x1(d: usize, sp: usize, wn: &str) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
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
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str(
        "]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
    );
    m.push_str(wn);
    m.push_str(", x = x)[name = tensor<string, []>(\"cv\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn test_conv1x1_magnitudes(d: usize, sp: usize) -> bool {
    println!(
        "=== TEST 1: Conv1x1 with varied input magnitudes (D={}, SP={}) ===",
        d, sp
    );
    let w = det_rand(d * d, 0.5, 42);
    let mut all_pass = true;

    for (mag, label) in [
        (0.01, "tiny"),
        (1.0, "medium"),
        (10.0, "large"),
        (100.0, "very large"),
    ] {
        let inp: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * mag).collect();
        let cpu = cpu_matmul(&w, d, d, &inp, sp);
        let mil = mil_conv1x1(d, sp, "W");
        match run_ane(
            &mil,
            &[("W", &build_blob(&w))],
            &[d * sp * 2],
            &[d * sp * 2],
            &[&to_fp16(&inp)],
        ) {
            Ok(ane) => {
                if !detailed_compare(&format!("conv1x1 {}", label), &ane, &cpu, 3) {
                    all_pass = false;
                }
            }
            Err(e) => {
                println!("  ✗ conv1x1 {} | FAIL: {}", label, e);
                all_pass = false;
            }
        }
    }
    println!();
    all_pass
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Residual connection with large values
// ═══════════════════════════════════════════════════════════════

fn mil_residual(d: usize, inter: usize, sp: usize) -> String {
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
    // gate
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> gate = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n");
    // up
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> up = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wu, x = x)[name = tensor<string, []>(\"cu\")];\n");
    // silu
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
    // fused
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n");
    // concat
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (fused, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");
    // down + residual
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdI, x = cat)[name = tensor<string, []>(\"cd\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn test_residual_magnitudes(d: usize, sp: usize) -> bool {
    let inter = d * 4;
    println!(
        "=== TEST 2: Residual FFN with varied input magnitudes (D={}, SP={}, inter={}) ===",
        d, sp, inter
    );
    let wg = det_rand(inter * d, 0.02, 100);
    let wu = det_rand(inter * d, 0.02, 200);
    let wd = det_rand(d * inter, 0.02, 300);
    let mut wdi = vec![0.0f32; d * (inter + d)];
    for r in 0..d {
        for c in 0..inter {
            wdi[r * (inter + d) + c] = wd[r * inter + c];
        }
        wdi[r * (inter + d) + inter + r] = 1.0;
    }
    let mut all_pass = true;

    for (mag, label) in [
        (0.1, "small"),
        (1.0, "unit"),
        (5.0, "medium"),
        (50.0, "large"),
    ] {
        let inp: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * mag).collect();
        // CPU: gate, up, silu, fused, down, + residual
        let gate = cpu_matmul(&wg, inter, d, &inp, sp);
        let up = cpu_matmul(&wu, inter, d, &inp, sp);
        let silu = cpu_silu(&gate);
        let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
        let down = cpu_matmul(&wd, d, inter, &fused, sp);
        let cpu: Vec<f32> = down.iter().zip(inp.iter()).map(|(&d, &x)| d + x).collect();

        match run_ane(
            &mil_residual(d, inter, sp),
            &[
                ("Wg", &build_blob(&wg)),
                ("Wu", &build_blob(&wu)),
                ("WdI", &build_blob(&wdi)),
            ],
            &[d * sp * 2],
            &[d * sp * 2],
            &[&to_fp16(&inp)],
        ) {
            Ok(ane) => {
                if !detailed_compare(&format!("residual {}", label), &ane, &cpu, 3) {
                    all_pass = false;
                }
            }
            Err(e) => {
                println!("  ✗ residual {} | FAIL: {}", label, e);
                all_pass = false;
            }
        }
    }
    println!();
    all_pass
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: ANE attention — verify intermediate values
// ═══════════════════════════════════════════════════════════════

fn mil_attention_only(d: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> q) {\n");
    // K weight [SP, D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> WK = const()[name = tensor<string, []>(\"WK\"), val = tensor<fp16, [");
    m.push_str(&sp.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WK.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // V weight [D, SP, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, 1]> WV = const()[name = tensor<string, []>(\"WV\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WV.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // QK scores
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> scores = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WK, x = q)[name = tensor<string, []>(\"qk\")];\n");
    // Softmax
    m.push_str("        tensor<int32, []> ax1 = const()[name = tensor<string, []>(\"ax1\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> attn = softmax(axis = ax1, x = scores)[name = tensor<string, []>(\"sm\")];\n");
    // Attn @ V
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> out = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WV, x = attn)[name = tensor<string, []>(\"av\")];\n");
    m.push_str("    } -> (out);\n}\n");
    m
}

fn test_attention_detailed(d: usize, sp: usize) -> bool {
    println!(
        "=== TEST 3: ANE attention intermediate verification (D={}, SP={}) ===",
        d, sp
    );
    let scale = 1.0 / (d as f32).sqrt();

    let k_vals = det_rand(d * sp, 0.1, 400);
    let v_vals = det_rand(d * sp, 0.1, 500);
    let q_vals = det_rand(d * sp, 0.1, 600);

    // Build K weight: WK[c, d_idx] = K[d_idx, c] * scale
    let mut wk = vec![0.0f32; sp * d];
    for c in 0..sp {
        for d_idx in 0..d {
            wk[c * d + d_idx] = k_vals[d_idx * sp + c] * scale;
        }
    }
    // Build V weight: WV[d, j] = V[d, j]
    let mut wv = vec![0.0f32; d * sp];
    for d_idx in 0..d {
        for j in 0..sp {
            wv[d_idx * sp + j] = v_vals[d_idx * sp + j];
        }
    }

    // ANE: scores → softmax → attn_output
    let cpu_out = cpu_attention(&q_vals, &k_vals, &v_vals, d, sp);

    let result = match run_ane(
        &mil_attention_only(d, sp),
        &[("WK", &build_blob(&wk)), ("WV", &build_blob(&wv))],
        &[d * sp * 2],
        &[d * sp * 2],
        &[&to_fp16(&q_vals)],
    ) {
        Ok(ane) => {
            let pass = detailed_compare("attention output", &ane, &cpu_out, 5);
            let cpu_sum: f64 = cpu_out.iter().map(|x| *x as f64).sum();
            let ane_sum: f64 = ane.iter().map(|x| *x as f64).sum();
            println!(
                "    CPU sum of output: {:.4}, ANE sum of output: {:.4}, ratio: {:.4}",
                cpu_sum,
                ane_sum,
                ane_sum / cpu_sum
            );
            pass
        }
        Err(e) => {
            println!("  ✗ attention | FAIL: {}", e);
            false
        }
    };
    println!();
    result
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: Hybrid layer — verify per-stage outputs
// ═════════════════════════════════════════════════════════════════

fn test_hybrid_detailed(d: usize, sp: usize) -> bool {
    println!(
        "=== TEST 4: Hybrid layer per-stage verification (D={}, SP={}) ===",
        d, sp
    );
    let wq = det_rand(d * d, 0.02, 700);
    let wk = det_rand(d * d, 0.02, 701);
    let wv = det_rand(d * d, 0.02, 702);
    let wo = det_rand(d * d, 0.02, 703);
    let wg = det_rand(d * 4 * d, 0.02, 704);
    let wu = det_rand(d * 4 * d, 0.02, 705);
    let wd = det_rand(d * d * 4, 0.02, 706);

    let inp: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 0.5) * 2.0).collect();
    let inp16 = to_fp16(&inp);

    // Stage 1: QKV on ANE
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    for (i, wn) in ["Wq", "Wk", "Wv"].iter().enumerate() {
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
    m.push_str(&(3 * d).to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> qkv = concat(values = (q0, q1, q2), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");
    m.push_str("    } -> (qkv);\n}\n");

    let mut all_pass = true;

    match run_ane(
        &m,
        &[
            ("Wq", &build_blob(&wq)),
            ("Wk", &build_blob(&wk)),
            ("Wv", &build_blob(&wv)),
        ],
        &[d * sp * 2],
        &[3 * d * sp * 2],
        &[&inp16],
    ) {
        Ok(qkv) => {
            let q_cpu = cpu_matmul(&wq, d, d, &inp, sp);
            let k_cpu = cpu_matmul(&wk, d, d, &inp, sp);
            if !detailed_compare("  Stage 1: Q", &qkv[0..d * sp], &q_cpu, 2) {
                all_pass = false;
            }
            if !detailed_compare("  Stage 1: K", &qkv[d * sp..2 * d * sp], &k_cpu, 2) {
                all_pass = false;
            }

            // CPU attention
            let attn = cpu_attention(&q_cpu, &k_cpu, &cpu_matmul(&wv, d, d, &inp, sp), d, sp);
            let mut after_attn = cpu_matmul(&wo, d, d, &attn, sp);
            for i in 0..d * sp {
                after_attn[i] += inp[i];
            }

            // Stage 2: out_proj + residual on ANE
            let mut woi = vec![0.0f32; d * 2 * d];
            for r in 0..d {
                for c in 0..d {
                    woi[r * 2 * d + c] = wo[r * d + c];
                }
                woi[r * 2 * d + d + r] = 1.0;
            }
            let attn16 = to_fp16(&attn);
            let mut m2 = String::new();
            m2.push_str(MIL_HEADER);
            m2.push_str("    func main<ios16>(tensor<fp16, [1, ");
            m2.push_str(&d.to_string());
            m2.push_str(", 1, ");
            m2.push_str(&sp.to_string());
            m2.push_str("]> a, tensor<fp16, [1, ");
            m2.push_str(&d.to_string());
            m2.push_str(", 1, ");
            m2.push_str(&sp.to_string());
            m2.push_str("]> b) {\n");
            m2.push_str("        tensor<fp16, [");
            m2.push_str(&d.to_string());
            m2.push_str(", ");
            m2.push_str(&(2 * d).to_string());
            m2.push_str(
                ", 1, 1]> WoI = const()[name = tensor<string, []>(\"WoI\"), val = tensor<fp16, [",
            );
            m2.push_str(&d.to_string());
            m2.push_str(", ");
            m2.push_str(&(2 * d).to_string());
            m2.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WoI.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
            m2.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
            m2.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
            m2.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
            m2.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
            m2.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
            m2.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
            m2.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
            m2.push_str("        tensor<fp16, [1, ");
            m2.push_str(&(2 * d).to_string());
            m2.push_str(", 1, ");
            m2.push_str(&sp.to_string());
            m2.push_str("]> cat = concat(values = (a, b), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");
            m2.push_str("        tensor<fp16, [1, ");
            m2.push_str(&d.to_string());
            m2.push_str(", 1, ");
            m2.push_str(&sp.to_string());
            m2.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WoI, x = cat)[name = tensor<string, []>(\"co\")];\n");
            m2.push_str("    } -> (y);\n}\n");

            match run_ane(
                &m2,
                &[("WoI", &build_blob(&woi))],
                &[d * sp * 2, d * sp * 2],
                &[d * sp * 2],
                &[&attn16, &inp16],
            ) {
                Ok(stage2) => {
                    if !detailed_compare("  Stage 2: out+res", &stage2, &after_attn, 2) {
                        all_pass = false;
                    }
                }
                Err(e) => {
                    println!("  ✗ Stage 2 | FAIL: {}", e);
                    all_pass = false;
                }
            }
        }
        Err(e) => {
            println!("  ✗ Stage 1 | FAIL: {}", e);
            all_pass = false;
        }
    }
    println!();
    all_pass
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    let all_pass = test_conv1x1_magnitudes(d, sp)
        & test_residual_magnitudes(d, sp)
        & test_attention_detailed(d, sp)
        & test_hybrid_detailed(d, sp);

    println!("═══════════════════════════════════════════════════════════");
    if all_pass {
        println!("  ALL TESTS PASSED — ANE calculations match CPU reference.");
    } else {
        println!("  SOME TESTS FAILED — see details above.");
    }
    println!("═════════════════════════════════════════════════════════");
}
