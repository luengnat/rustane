//! ANE vs CPU benchmark at Stories110M dimensions with correct MIL.
//!
//! Stories110M: D=768, num_heads=12, head_dim=64, FFN_inter=3072
//! SP (spatial/batch) = 256 for prefill
//!
//! Usage: cargo run --example bench_ane_vs_cpu

use half::f16;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// MIL GENERATORS (push_str only, program(1.3) + ios16 + BLOBFILE)
// ═══════════════════════════════════════════════════════════════════════════

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

fn mil_conv1x1(ic: usize, oc: usize, sp: usize, wn: &str) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
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
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&oc.to_string());
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

fn mil_fused_3conv(a_out: usize, b_out: usize, c_out: usize, ic: usize, sp: usize) -> String {
    let total = a_out + b_out + c_out;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    for (wn, oc) in [("Wa", a_out), ("Wb", b_out), ("Wc", c_out)] {
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
    for (i, (wn, oc, nm)) in [
        ("Wa", a_out, "c0"),
        ("Wb", b_out, "c1"),
        ("Wc", c_out, "c2"),
    ]
    .iter()
    .enumerate()
    {
        m.push_str("        tensor<fp16, [1, ");
        m.push_str(&oc.to_string());
        m.push_str(", 1, ");
        m.push_str(&sp.to_string());
        m.push_str("]> y");
        m.push_str(&i.to_string());
        m.push_str(
            " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
        );
        m.push_str(wn);
        m.push_str(", x = x)[name = tensor<string, []>(\"");
        m.push_str(nm);
        m.push_str("\")];\n");
    }
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = concat(values = (y0, y1, y2), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn mil_fused_ffn(dim: usize, inter: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&dim.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    for (wn, oc_r, ic_r) in [("Wg", inter, dim), ("Wu", inter, dim), ("Wd", dim, inter)] {
        m.push_str("        tensor<fp16, [");
        m.push_str(&oc_r.to_string());
        m.push_str(", ");
        m.push_str(&ic_r.to_string());
        m.push_str(", 1, 1]> ");
        m.push_str(wn);
        m.push_str(" = const()[name = tensor<string, []>(\"");
        m.push_str(wn);
        m.push_str("\"), val = tensor<fp16, [");
        m.push_str(&oc_r.to_string());
        m.push_str(", ");
        m.push_str(&ic_r.to_string());
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
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&dim.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> out = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wd, x = fused)[name = tensor<string, []>(\"cd\")];\n");
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

fn cpu_matmul_fp32(w: &[f32], wr: usize, wc: usize, inp: &[f32], sp: usize) -> Vec<f32> {
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

fn run_ane(
    mil: &str,
    blobs: &[(&str, &[u8])],
    in_b: usize,
    out_b: usize,
    inp16: &[u8],
) -> Result<u128, String> {
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
    for _ in 0..3 {
        exec.eval().map_err(|e| format!("WARMUP: {}", e))?;
    }
    let start = Instant::now();
    for _ in 0..10 {
        exec.eval().map_err(|e| format!("EVAL: {}", e))?;
    }
    Ok(start.elapsed().as_micros() / 10)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ANE vs CPU Benchmark — Stories110M Dimensions (correct MIL)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  ANE: conv1x1 with BLOBFILE weights, fp16 I/O, 10 iters (3 warmup)");
    println!("  CPU: naive f32 matmul, 10 runs");
    println!();

    let configs: Vec<(&str, usize, usize)> = vec![
        ("Small  (D=128, SP=32)", 128, 32),
        ("Med    (D=256, SP=64)", 256, 64),
        ("Med-L  (D=512, SP=128)", 512, 128),
        ("S110M  (D=768, SP=256)", 768, 256),
        ("S110M  (D=768, SP=512)", 768, 512),
        ("Large  (D=1024, SP=256)", 1024, 256),
    ];

    // Header: 7 columns
    println!(
        "  {:40} | {:>10} | {:>8} | {:>8} | {:>7} | {:>7}",
        "Config", "Op", "D", "ANE(μs)", "CPU(μs)", "Speed"
    );
    println!(
        "  {0:-<40}-+-{1:-<10}-+-{2:-<8}-+-{3:-<8}-+-{4:-<7}-+-{5:-<7}",
        "", "", "", "", "", ""
    );

    for (label, dim, sp) in &configs {
        let dim = *dim;
        let sp_val = *sp; // avoid shadowing sp with speedup ratio
        let inp = rand_weight(dim * sp_val, 1.0);

        // ── Single conv1x1 ──
        let w = rand_weight(dim * dim, 0.02);
        let blob = build_blob(&w);
        let mil = mil_conv1x1(dim, dim, sp_val, "W");
        let inp16 = write_fp16_scattered(&inp, dim, sp_val);
        let inp_c = inp.clone();
        let cpu_us: u128 = {
            let s = Instant::now();
            for _ in 0..10 {
                let _ = cpu_matmul_fp32(&w, dim, dim, &inp_c, sp_val);
            }
            s.elapsed().as_micros() / 10
        };
        match run_ane(
            &mil,
            &[("W", &blob)],
            dim * sp_val * 2,
            dim * sp_val * 2,
            &inp16,
        ) {
            Ok(ane_us) => {
                let speedup = cpu_us as f64 / ane_us as f64;
                let tf =
                    2.0 * dim as f64 * dim as f64 * sp_val as f64 / (ane_us as f64 * 1e-6) / 1e12;
                println!(
                    "  {:40} | {:>10} | {:>8} | {:>8}us | {:>6}us | {:>5.1}x [{:.2} TF]",
                    label, "conv1x1", dim, ane_us, cpu_us, speedup, tf
                );
            }
            Err(e) => println!(
                "  {:40} | {:>10} | {:>8} | {:>6}us | {:>6}us | FAIL: {}",
                label, "conv1x1", dim, "-", cpu_us, e
            ),
        }

        // ── Fused QKV ──
        let wa = rand_weight(dim * dim, 0.02);
        let wb = rand_weight(dim * dim, 0.02);
        let wc = rand_weight(dim * dim, 0.02);
        let ba = build_blob(&wa);
        let bb = build_blob(&wb);
        let bc = build_blob(&wc);
        let mil = mil_fused_3conv(dim, dim, dim, dim, sp_val);
        let inp16 = write_fp16_scattered(&inp, dim, sp_val);
        let inp_c = inp.clone();
        let cpu_us: u128 = {
            let s = Instant::now();
            for _ in 0..10 {
                let _ = cpu_matmul_fp32(&wa, dim, dim, &inp_c, sp_val);
                let _ = cpu_matmul_fp32(&wb, dim, dim, &inp_c, sp_val);
                let _ = cpu_matmul_fp32(&wc, dim, dim, &inp_c, sp_val);
            }
            s.elapsed().as_micros() / 10
        };
        match run_ane(
            &mil,
            &[("wa", &ba), ("wb", &bb), ("wc", &bc)],
            dim * sp_val * 2,
            dim * 3 * sp_val * 2,
            &inp16,
        ) {
            Ok(ane_us) => {
                let speedup = cpu_us as f64 / ane_us as f64;
                let tf = 2.0 * dim as f64 * dim as f64 * sp_val as f64 * 3.0
                    / (ane_us as f64 * 1e-6)
                    / 1e12;
                println!(
                    "  {:40} | {:>10} | {:>8} | {:>8}us | {:>6}us | {:>5.1}x [{:.2} TF]",
                    label, "fused_QKV", dim, ane_us, cpu_us, speedup, tf
                );
            }
            Err(e) => println!(
                "  {:40} | {:>10} | {:>8} | {:>6}us | {:>6}us | FAIL: {}",
                label, "fused_QKV", dim, "-", cpu_us, e
            ),
        }

        // ── Fused FFN ──
        let inter = dim * 4;
        let wg = rand_weight(inter * dim, 0.02);
        let wu = rand_weight(inter * dim, 0.02);
        let wd = rand_weight(dim * inter, 0.02);
        let bg = build_blob(&wg);
        let bu = build_blob(&wu);
        let bd = build_blob(&wd);
        let mil = mil_fused_ffn(dim, inter, sp_val);
        let inp16 = write_fp16_scattered(&inp, dim, sp_val);
        let inp_c = inp.clone();
        let cpu_us: u128 = {
            let s = Instant::now();
            for _ in 0..10 {
                let gate = cpu_matmul_fp32(&wg, inter, dim, &inp_c, sp_val);
                let up = cpu_matmul_fp32(&wu, inter, dim, &inp_c, sp_val);
                let silu: Vec<f32> = gate.iter().map(|&g| g / (1.0 + (-g).exp())).collect();
                let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
                let _down = cpu_matmul_fp32(&wd, dim, inter, &fused, sp_val);
            }
            s.elapsed().as_micros() / 10
        };
        match run_ane(
            &mil,
            &[("wg", &bg), ("wu", &bu), ("wd", &bd)],
            dim * sp_val * 2,
            dim * sp_val * 2,
            &inp16,
        ) {
            Ok(ane_us) => {
                let speedup = cpu_us as f64 / ane_us as f64;
                let tf = 2.0 * (dim as f64 * inter as f64 * 2.0 + inter as f64 * dim as f64)
                    / (ane_us as f64 * 1e-6)
                    / 1e12;
                println!(
                    "  {:40} | {:>10} | {:>8} | {:>8}us | {:>6}us | {:>5.1}x [{:.2} TF]",
                    label, "fused_FFN", dim, ane_us, cpu_us, speedup, tf
                );
            }
            Err(e) => println!(
                "  {:40} | {:>10} | {:>8} | {:>6}us | {:>6}us | FAIL: {}",
                label, "fused_FFN", dim, "-", cpu_us, e
            ),
        }

        println!();
    }
}
