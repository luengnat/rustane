//! Residual FFN on ANE using concat + conv1x1 identity weight hack.
//!
//! Since `add` is not supported on ANE, we implement residual connections by:
//!   concat(ffn_out, x) → conv1x1([Wd | I]) → Wd @ ffn_out + I @ x = residual
//!
//! The weight [Wd | I] combines the down projection with an identity matrix,
//! so a single conv1x1 computes both the projection AND the residual add.
//!
//! Full FFN + residual in one fused ANE program:
//!   x → conv1x1(Wg) → conv1x1(Wu) → sigmoid → mul → mul → concat(x) → conv1x1([Wd|I])
//!
//! Usage: cargo run --example test_residual_ffn -- [D] [SP]

use half::f16;
use std::env;
use std::time::Instant;

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

/// Generate MIL for FFN with residual via concat + conv1x1 identity hack.
///
/// Weight WdI is [D, 4D+D, 1, 1] = [Wd | I] where:
///   - Wd (left block) = [D, 4D] down projection weights
///   - I (right block) = [D, D] identity matrix for residual
fn mil_residual_ffn(d: usize, sp: usize) -> String {
    let inter = d * 4;
    let total_ic = inter + d; // 4D + D input channels for the final conv1x1
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");

    // ── FFN weights: Wg, Wu, WdI ──
    // Wg: [4D, D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> Wg = const()[name = tensor<string, []>(\"Wg\"), val = tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/Wg.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // Wu: [4D, D, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> Wu = const()[name = tensor<string, []>(\"Wu\"), val = tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/Wu.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // WdI: [D, 4D+D, 1, 1] = [Wd | I]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]> WdI = const()[name = tensor<string, []>(\"WdI\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WdI.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");

    // ── Conv params (shared) ──
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");

    // ── Gate conv ──
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> gate = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n");

    // ── Up conv ──
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> up = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wu, x = x)[name = tensor<string, []>(\"cu\")];\n");

    // ── SiLU = gate * sigmoid(gate) ──
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

    // ── Fused = silu * up ──
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n");

    // ── Concat fused with input x for residual ──
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (fused, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");

    // ── Down projection + residual via conv1x1([Wd | I]) ──
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
// ═════════════════════════════════════════════════════════════════════════

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

fn cpu_ffn_residual(
    x: &[f32],
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    d: usize,
    sp: usize,
) -> Vec<f32> {
    let inter = d * 4;
    let gate = cpu_matmul(wg, inter, d, x, sp);
    let up = cpu_matmul(wu, inter, d, x, sp);
    let silu: Vec<f32> = gate
        .iter()
        .map(|&g| {
            let s = 1.0 / (1.0 + (-g).exp());
            g * s
        })
        .collect();
    let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
    let down = cpu_matmul(wd, d, inter, &fused, sp);
    // Residual: down + x
    down.iter().zip(x.iter()).map(|(&d, &x)| d + x).collect()
}

// ═════════════════════════════════════════════════════════════════════════
// ANE RUNNER
// ═════════════════════════════════════════════════════════════════════════

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

// ═════════════════════════════════════════════════════════════════════════
// TEST
// ═════════════════════════════════════════════════════════════════════════

fn test_residual_ffn(d: usize, sp: usize) {
    let inter = d * 4;
    println!(
        "=== Residual FFN via concat+conv1x1 identity (D={}, SP={}, inter={}) ===",
        d, sp, inter
    );
    println!("  Pattern: x → gate → up → SiLU → fused → concat(x) → conv1x1([Wd|I]) → residual");
    println!();

    // Create weights
    let wg = rand_weight(inter * d, 0.02);
    let wu = rand_weight(inter * d, 0.02);
    let wd = rand_weight(d * inter, 0.02);

    // Create [Wd | I] weight: [D, 4D+D, 1, 1]
    // Row r, columns 0..4D = Wd[r, :] (down projection of fused)
    // Row r, column 4D+r = 1.0 (identity: pass through x[r])
    // All other columns = 0
    let mut wd_i = vec![0.0f32; d * (inter + d)];
    for r in 0..d {
        for c in 0..inter {
            wd_i[r * (inter + d) + c] = wd[r * inter + c]; // copy Wd into left block
        }
        wd_i[r * (inter + d) + inter + r] = 1.0; // identity diagonal in right block
    }

    let bg = build_blob(&wg);
    let bu = build_blob(&wu);
    let bwdi = build_blob(&wd_i);

    // Input
    let inp: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 * 0.1 + 1.0)).collect();

    // CPU reference
    let cpu_out = cpu_ffn_residual(&inp, &wg, &wu, &wd, d, sp);

    // ANE
    let mil = mil_residual_ffn(d, sp);
    let in_b = d * sp * 2;
    let out_b = d * sp * 2;
    let inp16 = write_fp16_scattered(&inp, d, sp);

    match run_ane(
        &mil,
        &[("Wg", &bg), ("Wu", &bu), ("WdI", &bwdi)],
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
            let flops = 2.0 * (inter as f64 * d as f64 * 2.0 + d as f64 * inter as f64);
            let tflops = flops / (us as f64 * 1e-6) / 1e12;
            println!(
                "  {:6} residual_ffn | {:6}us | max={:8.4} mean={:10.6} | {}/{} ({:5.1}%) | {:.2} TFLOPS",
                sym, us, mx, mn, ok, n, pct, tflops
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
    test_residual_ffn(d, sp);
}
