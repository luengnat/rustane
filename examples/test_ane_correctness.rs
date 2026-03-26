//! ANE correctness + performance tests using conv1x1 patterns.
//!
//! Key patterns:
//!   - conv1x1 instead of matmul (3x faster on ANE)
//!   - program(1.3) with tensor<string, []>("name") syntax (required for _ANEInMemoryModel)
//!   - ios16 qualifier
//!   - BLOBFILE weights with 128-byte blob header (64+64)
//!   - IOSurface layout: [1, dim, 1, SP] with fp16 scattered
//!   - Numerical verification against CPU reference
//!
//! Usage:
//!   cargo run --example test_ane_correctness -- <test_id> [args...]
//!
//!   10  — Conv1x1 correctness: ANE vs CPU matmul
//!   11  — Softmax standalone
//!   12  — Fused 2-conv (QK projection)
//!   13  — Fused 3-conv (QKV projection)
//!   14  — Fused FFN (gate+up+down conv + sigmoid + mul)
//!   15  — Batch prefill (batch in spatial dim)
//!   16  — Dimension sweep with correctness check

use half::f16;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test_ane_correctness <test_id> [args...]");
        eprintln!("  10=conv1x1  11=softmax  12=fused_2  13=fused_3");
        eprintln!("  14=fused_ffn  15=batch  16=dim_sweep");
        std::process::exit(1);
    }
    let tid: u32 = args[1].parse().unwrap_or(0);
    let a2: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let a3: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    match tid {
        10 => test_conv1x1(a2, a3),
        11 => test_softmax(a2),
        12 => test_fused_2(a2, a3),
        13 => test_fused_3(a2, a3),
        14 => test_fused_ffn(a2),
        15 => test_batch(a2, a3),
        16 => test_dim_sweep(),
        _ => {
            eprintln!("Unknown: {}", tid);
            std::process::exit(1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MIL GENERATORS (r#"..."# templates — safe, no escaping issues)
// ═══════════════════════════════════════════════════════════════════════════

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

fn mil_conv1x1(ic: usize, oc: usize, sp: usize, wn: &str) -> String {
    // Use raw string with NO format! — substitute dimensions manually
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    // Weight const
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
    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // Conv op
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

fn mil_softmax(dim: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&dim.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&dim.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = softmax(axis = ax, x = x)[name = tensor<string, []>(\"sm\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn mil_fused_2conv(a_out: usize, b_out: usize, ic: usize, sp: usize) -> String {
    let total = a_out + b_out;
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    // Wa weight
    m.push_str("        tensor<fp16, [");
    m.push_str(&a_out.to_string());
    m.push_str(", ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, 1]> Wa = const()[name = tensor<string, []>(\"Wa\"), val = tensor<fp16, [");
    m.push_str(&a_out.to_string());
    m.push_str(", ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wa.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // Wb weight
    m.push_str("        tensor<fp16, [");
    m.push_str(&b_out.to_string());
    m.push_str(", ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, 1]> Wb = const()[name = tensor<string, []>(\"Wb\"), val = tensor<fp16, [");
    m.push_str(&b_out.to_string());
    m.push_str(", ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wb.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // Conv ops
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&a_out.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> ya = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wa, x = x)[name = tensor<string, []>(\"ca\")];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&b_out.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> yb = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wb, x = x)[name = tensor<string, []>(\"cb\")];\n");
    // Concat
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&total.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = concat(values = (ya, yb), axis = ax, interleave = ci)[name = tensor<string, []>(\"cc\")];\n");
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
    // Three weights
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
    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // Three conv ops
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
    // Concat
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
    // Three weights: Wg, Wu, Wd
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
    // Conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // gate conv
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> gate = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n");
    // up conv
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
    // fused = silu * up
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n");
    // down conv
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&dim.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> out = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wd, x = fused)[name = tensor<string, []>(\"cd\")];\n");
    m.push_str("    } -> (out);\n}\n");
    m
}

// ═══════════════════════════════════════════════════════════════════════════
// WEIGHT BLOB (hybird pattern: 64-byte outer + 64-byte chunk header + data)
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

// ═══════════════════════════════════════════════════════════════════════════
// CPU REFERENCES
// ═══════════════════════════════════════════════════════════════════════════

fn cpu_matmul(w: &[f32], wr: usize, wc: usize, inp: &[f32], sp: usize) -> Vec<f32> {
    assert_eq!(wc, inp.len() / sp);
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

fn cpu_ffn(
    x: &[f32],
    dim: usize,
    sp: usize,
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    inter: usize,
) -> Vec<f32> {
    let gate = cpu_matmul(wg, inter, dim, x, sp);
    let up = cpu_matmul(wu, inter, dim, x, sp);
    let silu: Vec<f32> = gate
        .iter()
        .map(|&g| {
            let s = 1.0 / (1.0 + (-g).exp());
            g * s
        })
        .collect();
    let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
    cpu_matmul(wd, dim, inter, &fused, sp)
}

// ═══════════════════════════════════════════════════════════════════════════
// ANE COMPILE + EVAL
// ═══════════════════════════════════════════════════════════════════════════

fn ane_run(
    mil: &str,
    blobs: &[(&str, &[u8])],
    in_b: usize,
    out_b: usize,
    inp16: &[u8],
) -> Result<(Vec<f32>, u128), String> {
    // Weight dict keys must match the BLOBFILE path in MIL exactly.
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
    exec.eval().map_err(|e| format!("EVAL: {}", e))?;
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

fn print_cmp(label: &str, mx: f32, mean: f32, ok: usize, n: usize, us: u128) {
    let pct = 100.0 * ok as f32 / n as f32;
    let sym = if pct > 95.0 {
        "PASS"
    } else if pct > 50.0 {
        "WARN"
    } else {
        "FAIL"
    };
    println!(
        "  {:6} {:12} | {:6}us | max={:8.4} mean={:10.6} | {}/{} ({:5.1}%)",
        sym, label, us, mx, mean, ok, n, pct
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

fn test_conv1x1(dim: usize, sp: usize) {
    println!(
        "=== Test 10: Conv1x1 Correctness (D={}, SP={}) ===",
        dim, sp
    );
    let mut w = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w[i * dim + i] = 1.0;
        if i + 1 < dim {
            w[i * dim + i + 1] = 0.5;
        }
    }
    let blob = build_blob(&w);
    let mil = mil_conv1x1(dim, dim, sp, "W");
    let in_b = dim * sp * 2;
    let out_b = in_b;
    let inp: Vec<f32> = (0..dim * sp)
        .map(|i| (i % dim) as f32 * 0.1 + 1.0)
        .collect();
    let inp16 = write_fp16_scattered(&inp, dim, sp);
    let cpu = cpu_matmul(&w, dim, dim, &inp, sp);
    match ane_run(&mil, &[("W", &blob)], in_b, out_b, &inp16) {
        Ok((ane, us)) => {
            let (mx, mn, ok, n) = compare(&ane, &cpu);
            print_cmp("conv1x1", mx, mn, ok, n, us);
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn test_softmax(dim: usize) {
    let sp = 32;
    println!("=== Test 11: Softmax (D={}, SP={}) ===", dim, sp);
    let mil = mil_softmax(dim, sp);
    let in_b = dim * sp * 2;
    let inp: Vec<f32> = (0..dim * sp)
        .map(|i| {
            let c = i / sp;
            let s = i % sp;
            ((c * 7 + s * 3) % 20) as f32 * 0.5 - 5.0
        })
        .collect();
    let inp16 = write_fp16_scattered(&inp, dim, sp);
    let cpu = cpu_softmax(&inp, dim, sp);
    match ane_run(&mil, &[], in_b, in_b, &inp16) {
        Ok((ane, us)) => {
            let (mx, mn, ok, n) = compare(&ane, &cpu);
            print_cmp("softmax", mx, mn, ok, n, us);
            // Verify sums
            let mut sums_ok = 0usize;
            for s in 0..sp {
                let s2: f32 = (0..dim).map(|d| ane[d * sp + s]).sum();
                if (s2 - 1.0).abs() < 0.01 {
                    sums_ok += 1;
                }
            }
            println!("  Softmax sum~=1: {}/{}", sums_ok, sp);
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn test_fused_2(dim: usize, sp: usize) {
    println!("=== Test 12: Fused 2-Conv (D={}, SP={}) ===", dim, sp);
    let mut wa = vec![0.0f32; dim * dim];
    let mut wb = vec![0.0f32; dim * dim];
    for i in 0..dim {
        wa[i * dim + i] = 1.0;
        wb[i * dim + (dim - 1 - i)] = 1.0;
    }
    let ba = build_blob(&wa);
    let bb = build_blob(&wb);
    let mil = mil_fused_2conv(dim, dim, dim, sp);
    let in_b = dim * sp * 2;
    let out_b = 2 * dim * sp * 2;
    let inp: Vec<f32> = (0..dim * sp)
        .map(|i| (i % dim) as f32 * 0.1 + 1.0)
        .collect();
    let inp16 = write_fp16_scattered(&inp, dim, sp);
    let mut cpu = cpu_matmul(&wa, dim, dim, &inp, sp);
    let cb = cpu_matmul(&wb, dim, dim, &inp, sp);
    cpu.extend_from_slice(&cb);
    match ane_run(&mil, &[("wa", &ba), ("wb", &bb)], in_b, out_b, &inp16) {
        Ok((ane, us)) => {
            let (mx, mn, ok, n) = compare(&ane, &cpu);
            print_cmp("fused_2", mx, mn, ok, n, us);
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn test_fused_3(dim: usize, sp: usize) {
    println!("=== Test 13: Fused 3-Conv QKV (D={}, SP={}) ===", dim, sp);
    let mut wa = vec![0.0f32; dim * dim];
    let mut wb = vec![0.0f32; dim * dim];
    let mut wc = vec![0.0f32; dim * dim];
    for i in 0..dim {
        wa[i * dim + i] = 1.0;
        wb[i * dim + i] = 0.5;
        wc[i * dim + i] = 0.3;
    }
    let ba = build_blob(&wa);
    let bb = build_blob(&wb);
    let bc = build_blob(&wc);
    let mil = mil_fused_3conv(dim, dim, dim, dim, sp);
    let in_b = dim * sp * 2;
    let out_b = 3 * dim * sp * 2;
    let inp: Vec<f32> = (0..dim * sp)
        .map(|i| (i % dim) as f32 * 0.1 + 1.0)
        .collect();
    let inp16 = write_fp16_scattered(&inp, dim, sp);
    let mut cpu = cpu_matmul(&wa, dim, dim, &inp, sp);
    let cb = cpu_matmul(&wb, dim, dim, &inp, sp);
    cpu.extend_from_slice(&cb);
    let cc = cpu_matmul(&wc, dim, dim, &inp, sp);
    cpu.extend_from_slice(&cc);
    match ane_run(
        &mil,
        &[("wa", &ba), ("wb", &bb), ("wc", &bc)],
        in_b,
        out_b,
        &inp16,
    ) {
        Ok((ane, us)) => {
            let (mx, mn, ok, n) = compare(&ane, &cpu);
            print_cmp("fused_3", mx, mn, ok, n, us);
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn test_fused_ffn(dim: usize) {
    let sp = 32;
    let inter = dim * 4;
    println!(
        "=== Test 14: Fused FFN SwiGLU (D={}, H={}, SP={}) ===",
        dim, inter, sp
    );
    let mut wg = vec![0.01f32; inter * dim];
    let mut wu = vec![0.01f32; inter * dim];
    let mut wd = vec![0.01f32; dim * inter];
    for i in 0..inter.min(dim) {
        wg[i * dim + i] = 0.5;
        wu[i * dim + i] = 0.3;
    }
    for i in 0..dim.min(inter) {
        wd[i * inter + i] = 0.2;
    }
    let bg = build_blob(&wg);
    let bu = build_blob(&wu);
    let bd = build_blob(&wd);
    let mil = mil_fused_ffn(dim, inter, sp);
    let in_b = dim * sp * 2;
    let out_b = dim * sp * 2;
    let inp: Vec<f32> = (0..dim * sp)
        .map(|i| (i % dim) as f32 * 0.05 + 0.5)
        .collect();
    let inp16 = write_fp16_scattered(&inp, dim, sp);
    let cpu = cpu_ffn(&inp, dim, sp, &wg, &wu, &wd, inter);
    match ane_run(
        &mil,
        &[("wg", &bg), ("wu", &bu), ("wd", &bd)],
        in_b,
        out_b,
        &inp16,
    ) {
        Ok((ane, us)) => {
            let (mx, mn, ok, n) = compare(&ane, &cpu);
            print_cmp("fused_ffn", mx, mn, ok, n, us);
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn test_batch(dim: usize, batch: usize) {
    println!("=== Test 15: Batch Prefill (D={}, B={}) ===", dim, batch);
    let sp = batch;
    let mut w = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w[i * dim + i] = 1.0;
    }
    let blob = build_blob(&w);
    let mil = mil_conv1x1(dim, dim, sp, "W");
    let in_b = dim * sp * 2;
    let out_b = in_b;
    let inp: Vec<f32> = (0..dim * sp)
        .map(|i| {
            let c = i / sp;
            let s = i % sp;
            (c as f32 + 1.0) * (s as f32 + 1.0) * 0.1
        })
        .collect();
    let inp16 = write_fp16_scattered(&inp, dim, sp);
    let cpu = cpu_matmul(&w, dim, dim, &inp, sp);
    match ane_run(&mil, &[("W", &blob)], in_b, out_b, &inp16) {
        Ok((ane, us)) => {
            let (mx, mn, ok, n) = compare(&ane, &cpu);
            print_cmp("batch", mx, mn, ok, n, us);
            let mut sok = 0usize;
            for s in 0..sp {
                let se: f32 = (0..dim)
                    .map(|d| cpu[d * sp + s].abs())
                    .sum::<f32>()
                    .max(0.001);
                let _ae: f32 = (0..dim)
                    .map(|d| ane[d * sp + s].abs())
                    .sum::<f32>()
                    .max(0.001);
                let err: f32 = (0..dim)
                    .map(|d| (cpu[d * sp + s] - ane[d * sp + s]).abs())
                    .sum::<f32>()
                    / se;
                if err < 0.05 {
                    sok += 1;
                }
            }
            println!("  Per-sample <5% error: {}/{}", sok, sp);
        }
        Err(e) => println!("  FAIL: {}", e),
    }
}

fn test_dim_sweep() {
    println!("=== Test 16: Dimension Sweep ===");
    println!(
        "  {:>6} | {:>4} | {:>6}us | {:>8} | {:>8} | {:>6}",
        "D", "SP", "Time", "MaxErr", "Acc", "TFLOPS"
    );
    println!(
        "  {}+{}+{}+{}+{}+{}",
        "-".repeat(6),
        "-".repeat(4),
        "-".repeat(6),
        "-".repeat(8),
        "-".repeat(8),
        "-".repeat(6)
    );
    for &d in &[32, 64, 128, 256, 512, 768, 1024] {
        let sp = 32;
        let mut w = vec![0.0f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
            if i + 1 < d {
                w[i * d + i + 1] = 0.5;
            }
        }
        let blob = build_blob(&w);
        let mil = mil_conv1x1(d, d, sp, "W");
        let in_b = d * sp * 2;
        let out_b = in_b;
        let inp: Vec<f32> = (0..d * sp).map(|i| (i % d) as f32 * 0.1 + 1.0).collect();
        let inp16 = write_fp16_scattered(&inp, d, sp);
        let cpu = cpu_matmul(&w, d, d, &inp, sp);
        match ane_run(&mil, &[("W", &blob)], in_b, out_b, &inp16) {
            Ok((ane, us)) => {
                let (mx, _mn, ok, n) = compare(&ane, &cpu);
                let flops = 2.0 * d as f64 * d as f64 * sp as f64;
                let tflops = flops / (us as f64 * 1e-6) / 1e12;
                println!(
                    "  {:>6} | {:>4} | {:>6}us | {:>8.4} | {:>6.1}% | {:>6.2}",
                    d,
                    sp,
                    us,
                    mx,
                    100.0 * ok as f32 / n as f32,
                    tflops
                );
            }
            Err(e) => println!("  {:>6} | {:>4} | {}", d, sp, e),
        }
    }
}
