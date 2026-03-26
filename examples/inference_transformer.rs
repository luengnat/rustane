//! Multi-Layer Hybrid Transformer Inference Pipeline
//!
//! Complete transformer with attention + FFN per layer:
//!   ANE: QKV projection (3x conv1x1)
//!   CPU: Attention (QK^T, softmax, AV) — transpose/matmul fail on ANE
//!   ANE: Output projection + residual (concat + conv1x1)
//!   ANE: FFN + residual (SwiGLU, concat + conv1x1)
//!
//! Per layer: 3 ANE programs + 1 CPU attention call
//! Chains N layers sequentially with IOSurface I/O between programs.
//!
//! Usage: cargo run --example inference_transformer -- [D] [SP] [layers] [runs]

use half::f16;
use std::env;
use std::time::Instant;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}
const ROW: i32 = 101;
const NOTRANS: i32 = 111;
const TRANS: i32 = 112;

fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NOTRANS,
            NOTRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}
fn mm_at(a: &[f32], k: usize, m: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            TRANS,
            NOTRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

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
fn rand_matrix(rows: usize, cols: usize, scale: f32, seed: u64) -> Vec<f32> {
    (0..rows * cols)
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

// ═══════════════════════════════════════════════════════════════
// MIL GENERATORS
// ═══════════════════════════════════════════════════════════════

const MIL_HEADER: &str = "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n";

/// QKV projection: 3 conv1x1 → concat → [1, 3D, 1, SP]
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

/// Output projection + residual: concat(attn_out, x) → conv1x1([Wo|I])
fn mil_out_proj(d: usize, sp: usize) -> String {
    let total_ic = 2 * d;
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
    m.push_str(", 1, 1]> WoI = const()[name = tensor<string, []>(\"WoI\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&total_ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WoI.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
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
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WoI, x = cat)[name = tensor<string, []>(\"co\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

/// FFN + residual (SwiGLU): gate, up, silu, fused, concat(x), down
fn mil_ffn(d: usize, inter: usize, sp: usize) -> String {
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

// ═══════════════════════════════════════════════════════════════
// CPU ATTENTION (BLAS-accelerated)
// ═══════════════════════════════════════════════════════════════

/// CPU attention: Q, K, V are [D, SP]. Returns [D, SP].
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, sp: usize) -> Vec<f32> {
    // Q^T @ K → [SP, SP]  (scores[i,j] = sum_d Q[d,i] * K[d,j])
    let scores = mm_at(q, d, sp, k, sp);

    // Scale by 1/sqrt(d)
    let scale = 1.0 / (d as f32).sqrt();
    let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();

    // Softmax per row (each query position)
    let mut attn = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let mut mx = f32::NEG_INFINITY;
        for j in 0..sp {
            mx = mx.max(scaled[i * sp + j]);
        }
        let mut sm = 0.0_f32;
        for j in 0..sp {
            let e = (scaled[i * sp + j] - mx).exp();
            attn[i * sp + j] = e;
            sm += e;
        }
        for j in 0..sp {
            attn[i * sp + j] /= sm;
        }
    }

    // attn @ V → [SP, D] (then transpose to [D, SP])
    let av = mm(&attn, sp, sp, v, d); // [SP, D]
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            out[h * sp + i] = av[i * d + h]; // transpose [SP,D] → [D,SP]
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════
// TRANSFORMER LAYER WEIGHTS
// ═══════════════════════════════════════════════════════════════

struct TransformerLayer {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    d: usize,
    inter: usize,
}
impl TransformerLayer {
    fn new(d: usize, inter: usize, seed: u64) -> Self {
        TransformerLayer {
            wq: rand_matrix(d, d, 0.02, seed),
            wk: rand_matrix(d, d, 0.02, seed + 1),
            wv: rand_matrix(d, d, 0.02, seed + 2),
            wo: rand_matrix(d, d, 0.02, seed + 3),
            wg: rand_matrix(inter, d, 0.02, seed + 4),
            wu: rand_matrix(inter, d, 0.02, seed + 5),
            wd: rand_matrix(d, inter, 0.02, seed + 6),
            d,
            inter,
        }
    }
    fn build_woi(&self) -> Vec<f32> {
        let t = 2 * self.d;
        let mut woi = vec![0.0f32; self.d * t];
        for r in 0..self.d {
            for c in 0..self.d {
                woi[r * t + c] = self.wo[r * self.d + c];
            }
            woi[r * t + self.d + r] = 1.0; // identity for residual
        }
        woi
    }
    fn build_wdi(&self) -> Vec<f32> {
        let t = self.inter + self.d;
        let mut wdi = vec![0.0f32; self.d * t];
        for r in 0..self.d {
            for c in 0..self.inter {
                wdi[r * t + c] = self.wd[r * self.inter + c];
            }
            wdi[r * t + self.inter + r] = 1.0; // identity for residual
        }
        wdi
    }
    /// Full CPU forward pass for baseline
    fn forward_cpu(&self, x: &[f32], sp: usize) -> Vec<f32> {
        let d = self.d;
        // QKV projections
        let q = mm(&self.wq, d, d, x, sp);
        let k = mm(&self.wk, d, d, x, sp);
        let v = mm(&self.wv, d, d, x, sp);
        // Attention
        let attn_out = cpu_attention(&q, &k, &v, d, sp);
        // Output projection + residual
        let out = mm(&self.wo, d, d, &attn_out, sp);
        let mut y = vec![0.0f32; d * sp];
        for i in 0..y.len() {
            y[i] = out[i] + x[i];
        }
        // FFN
        let gate = mm(&self.wg, self.inter, d, &y, sp);
        let up = mm(&self.wu, self.inter, d, &y, sp);
        let mut fused = vec![0.0f32; self.inter * sp];
        for i in 0..fused.len() {
            let s = 1.0 / (1.0 + (-gate[i]).exp());
            fused[i] = gate[i] * s * up[i];
        }
        let down = mm(&self.wd, d, self.inter, &fused, sp);
        let mut out2 = vec![0.0f32; d * sp];
        for i in 0..out2.len() {
            out2[i] = down[i] + y[i];
        }
        out2
    }
}

// ═══════════════════════════════════════════════════════════════
// ANE EXECUTOR HELPERS
// ═══════════════════════════════════════════════════════════════

fn compile_ane(
    mil: &str,
    names: &[&str],
    blobs: &[Vec<u8>],
    in_c: usize,
    out_c: usize,
    sp: usize,
) -> Option<rustane::wrapper::ANEExecutor> {
    let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
    let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    let bpe = 2usize; // fp16
    rustane::wrapper::ANECompiler::new()
        .compile_multi(
            mil,
            names,
            &brefs,
            &lens,
            &[in_c * sp * bpe],
            &[out_c * sp * bpe],
        )
        .ok()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let num_layers: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let num_runs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
    let inter = d * 4;

    println!("============================================================");
    println!("  Hybrid Transformer Inference (ANE linear + CPU attention)");
    println!(
        "  D={} SP={} inter={} layers={} runs={}",
        d, sp, inter, num_layers, num_runs
    );
    println!("  Per layer: 3 ANE programs (QKV, out_proj, FFN) + CPU attention");
    println!("============================================================\n");

    let layers: Vec<TransformerLayer> = (0..num_layers)
        .map(|i| TransformerLayer::new(d, inter, 42 + i as u64 * 100))
        .collect();

    let x_data: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * 0.01).collect();
    let x16 = to_fp16(&x_data);

    // Params per layer: 4*D*D (attn) + 2*D*4D + D*4D (FFN) = 4D^2 + 12D^2 = 16D^2
    let params_per_layer = 4 * d * d + 3 * d * inter;
    let total_params = params_per_layer * num_layers;
    println!(
        "  Parameters: {:.1}M per layer, {:.1}M total",
        params_per_layer as f64 / 1e6,
        total_params as f64 / 1e6
    );

    // ── Compile ANE programs ──
    println!("\n=== Compiling ANE programs ===");
    let mil_qkv = mil_qkv(d, sp);
    let mil_out = mil_out_proj(d, sp);
    let mil_ffn_s = mil_ffn(d, inter, sp);

    let qkv_names: Vec<&str> = vec![
        "@model_path/weights/Wq.bin",
        "@model_path/weights/Wk.bin",
        "@model_path/weights/Wv.bin",
    ];
    let out_names: Vec<&str> = vec!["@model_path/weights/WoI.bin"];
    let ffn_names: Vec<&str> = vec![
        "@model_path/weights/Wg.bin",
        "@model_path/weights/Wu.bin",
        "@model_path/weights/WdI.bin",
    ];

    struct LayerPrograms {
        qkv: rustane::wrapper::ANEExecutor,
        out_proj: rustane::wrapper::ANEExecutor,
        ffn: rustane::wrapper::ANEExecutor,
    }

    let mut all_progs = Vec::new();
    let t_compile = Instant::now();
    let mut compile_fail = false;

    for (li, layer) in layers.iter().enumerate() {
        // QKV program
        let qkv_blobs = vec![
            build_blob(&layer.wq),
            build_blob(&layer.wk),
            build_blob(&layer.wv),
        ];
        let qkv_exec = match compile_ane(&mil_qkv, &qkv_names, &qkv_blobs, d, 3 * d, sp) {
            Some(e) => e,
            None => {
                println!("  Layer {:2}: QKV compile FAILED", li + 1);
                compile_fail = true;
                break;
            }
        };

        // Out proj program
        let out_blobs = vec![build_blob(&layer.build_woi())];
        let out_exec = {
            let lens: Vec<usize> = out_blobs.iter().map(|b| b.len()).collect();
            let brefs: Vec<&[u8]> = out_blobs.iter().map(|b| b.as_slice()).collect();
            let bpe = 2usize;
            let result = rustane::wrapper::ANECompiler::new().compile_multi(
                &mil_out,
                &out_names,
                &brefs,
                &lens,
                &[d * sp * bpe, d * sp * bpe],
                &[d * sp * bpe],
            );
            match result {
                Ok(e) => e,
                Err(_) => {
                    println!("  Layer {:2}: out_proj compile FAILED", li + 1);
                    compile_fail = true;
                    break;
                }
            }
        };

        // FFN program
        let ffn_blobs = vec![
            build_blob(&layer.wg),
            build_blob(&layer.wu),
            build_blob(&layer.build_wdi()),
        ];
        let ffn_exec = match compile_ane(&mil_ffn_s, &ffn_names, &ffn_blobs, d, d, sp) {
            Some(e) => e,
            None => {
                println!("  Layer {:2}: FFN compile FAILED", li + 1);
                compile_fail = true;
                break;
            }
        };

        println!("  Layer {:2}: 3 programs compiled OK", li + 1);
        all_progs.push(LayerPrograms {
            qkv: qkv_exec,
            out_proj: out_exec,
            ffn: ffn_exec,
        });
    }

    if compile_fail || all_progs.is_empty() {
        println!("  ABORTING: compile failure");
        return;
    }
    let actual_layers = all_progs.len();
    let total_progs = actual_layers * 3;
    println!(
        "  Total: {} programs in {:.1}ms ({:.1}ms/prog)",
        total_progs,
        t_compile.elapsed().as_secs_f64() * 1000.0,
        t_compile.elapsed().as_secs_f64() * 1000.0 / total_progs as f64
    );

    // Warmup
    for progs in &mut all_progs {
        progs.qkv.write_input(0, &x16).expect("w");
        progs.qkv.eval().expect("e");
        progs.out_proj.write_input(0, &x16).expect("w");
        progs.out_proj.write_input(1, &x16).expect("w");
        progs.out_proj.eval().expect("e");
        progs.ffn.write_input(0, &x16).expect("w");
        progs.ffn.eval().expect("e");
    }

    // ── ANE+CPU Hybrid Inference ──
    println!(
        "\n=== Hybrid Inference ({} layers x 3 ANE + CPU attn, {} runs) ===",
        actual_layers, num_runs
    );

    let mut ane_times = Vec::new();
    let mut ane_qkv_t = 0.0_f64;
    let mut cpu_attn_t = 0.0_f64;
    let mut ane_out_t = 0.0_f64;
    let mut ane_ffn_t = 0.0_f64;
    let mut total_calls = 0usize;

    for _run in 0..num_runs {
        let mut current_fp16 = x16.clone();
        let mut current_f32 = x_data.clone();
        let t_total = Instant::now();

        for progs in &mut all_progs {
            let d = layers[0].d;

            // ANE: QKV projection
            let t = Instant::now();
            progs.qkv.write_input(0, &current_fp16).expect("w");
            progs.qkv.eval().expect("e");
            let qkv_raw = progs.qkv.read_output_vec(0).expect("r");
            ane_qkv_t += t.elapsed().as_secs_f64() * 1000.0;

            // Parse Q, K, V from concatenated output [1, 3D, 1, SP]
            let qkv = from_fp16(&qkv_raw);
            let (q, rest) = qkv.split_at(d * sp);
            let (k, v) = rest.split_at(d * sp);

            // CPU: Attention
            let t = Instant::now();
            let attn_out = cpu_attention(q, k, v, d, sp);
            cpu_attn_t += t.elapsed().as_secs_f64() * 1000.0;

            // ANE: Output projection + residual
            let t = Instant::now();
            let attn_fp16 = to_fp16(&attn_out);
            progs.out_proj.write_input(0, &attn_fp16).expect("w");
            progs.out_proj.write_input(1, &current_fp16).expect("w");
            progs.out_proj.eval().expect("e");
            let out_raw = progs.out_proj.read_output_vec(0).expect("r");
            ane_out_t += t.elapsed().as_secs_f64() * 1000.0;

            let out = from_fp16(&out_raw);
            current_fp16 = out_raw;

            // ANE: FFN + residual
            let t = Instant::now();
            progs.ffn.write_input(0, &current_fp16).expect("w");
            progs.ffn.eval().expect("e");
            let ffn_raw = progs.ffn.read_output_vec(0).expect("r");
            ane_ffn_t += t.elapsed().as_secs_f64() * 1000.0;

            current_f32 = from_fp16(&ffn_raw);
            current_fp16 = ffn_raw;
            total_calls += 1;
        }

        ane_times.push(t_total.elapsed().as_secs_f64() * 1000.0);
    }

    drop(all_progs);

    let ane_avg = ane_times.iter().sum::<f64>() / num_runs as f64;
    let ane_min = ane_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let n = num_runs * actual_layers;

    println!("\n  Per-layer breakdown (avg over {} runs):", num_runs);
    println!("  {:>20} {:>10} {:>10}", "", "Total(ms)", "Per(ms)");
    println!(
        "  {:>20} {:>10} {:>10}",
        "--------------------", "--------", "--------"
    );
    println!(
        "  {:>20} {:>10.2} {:>10.3}",
        "ANE QKV proj",
        ane_qkv_t,
        ane_qkv_t / n as f64
    );
    println!(
        "  {:>20} {:>10.2} {:>10.3}",
        "CPU attention",
        cpu_attn_t,
        cpu_attn_t / n as f64
    );
    println!(
        "  {:>20} {:>10.2} {:>10.3}",
        "ANE out proj+res",
        ane_out_t,
        ane_out_t / n as f64
    );
    println!(
        "  {:>20} {:>10.2} {:>10.3}",
        "ANE FFN+res",
        ane_ffn_t,
        ane_ffn_t / n as f64
    );
    let ane_total_t = ane_qkv_t + ane_out_t + ane_ffn_t;
    let cpu_total_t = cpu_attn_t;
    println!(
        "  {:>20} {:>10.2} {:>10.3}",
        "ANE subtotal",
        ane_total_t,
        ane_total_t / n as f64
    );
    println!(
        "  {:>20} {:>10.2} {:>10.3}",
        "CPU subtotal",
        cpu_total_t,
        cpu_total_t / n as f64
    );
    println!(
        "  {:>20} {:.1}%",
        "ANE fraction",
        ane_total_t / (ane_total_t + cpu_total_t) * 100.0
    );

    // ── CPU Baseline ──
    println!("\n=== CPU Baseline (BLAS, {} layers) ===", actual_layers);
    let mut cpu_times = Vec::new();
    for _ in 0..num_runs {
        let mut x = x_data.clone();
        let t = Instant::now();
        for layer in layers.iter().take(actual_layers) {
            x = layer.forward_cpu(&x, sp);
        }
        cpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let cpu_avg = cpu_times.iter().sum::<f64>() / num_runs as f64;
    let cpu_min = cpu_times.iter().cloned().fold(f64::INFINITY, f64::min);

    // ── Results ──
    println!("\n============================================================");
    println!(
        "  RESULTS: {}-layer Transformer at D={}, SP={}",
        actual_layers, d, sp
    );
    println!("============================================================");
    let speedup = cpu_avg / ane_avg;
    let speedup_best = cpu_min / ane_min;
    let cpu_per_layer = cpu_avg / actual_layers as f64;
    let hybrid_per_layer = ane_avg / actual_layers as f64;

    println!(
        "  {:>20} {:>12} {:>12} {:>10}",
        "", "Hybrid", "CPU", "Speedup"
    );
    println!(
        "  {:>20} {:>12} {:>12} {:>10}",
        "--------------------", "----------", "----------", "--------"
    );
    println!(
        "  {:>20} {:>12.1} {:>12.1} {:>9.1}x",
        "Total latency (ms)", ane_avg, cpu_avg, speedup
    );
    println!(
        "  {:>20} {:>12.1} {:>12.1} {:>9.1}x",
        "Best run (ms)", ane_min, cpu_min, speedup_best
    );
    println!(
        "  {:>20} {:>12.3} {:>12.3} {:>9.1}x",
        "Per layer (ms)",
        hybrid_per_layer,
        cpu_per_layer,
        cpu_per_layer / hybrid_per_layer
    );

    let ane_tps = 1000.0 / ane_avg * sp as f64;
    let cpu_tps = 1000.0 / cpu_avg * sp as f64;
    println!("\n  Throughput:");
    println!("    Hybrid: {:.0} tokens/sec", ane_tps);
    println!("    CPU:    {:.0} tokens/sec", cpu_tps);
    println!("    Speedup: {:.1}x", ane_tps / cpu_tps);

    println!("\n  Per-layer time budget:");
    println!("    ANE QKV proj:       {:.3}ms", ane_qkv_t / n as f64);
    println!(
        "    CPU attention:      {:.3}ms  ({:.0}% of hybrid layer)",
        cpu_attn_t / n as f64,
        cpu_attn_t / (ane_avg / num_runs as f64) * 100.0
    );
    println!("    ANE out proj+res:   {:.3}ms", ane_out_t / n as f64);
    println!("    ANE FFN+res:        {:.3}ms", ane_ffn_t / n as f64);

    println!(
        "\n  CONCLUSION: Hybrid transformer is {:.1}x faster than CPU BLAS",
        speedup
    );
}
