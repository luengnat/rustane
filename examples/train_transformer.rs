//! Full Transformer Training: ANE Forward + CPU Backward
//!
//! Realistic training benchmark with attention + FFN per layer:
//!   ANE: QKV projection, output projection + residual, FFN + residual (3 programs/layer)
//!   CPU: Multi-head attention (QK^T, softmax, AV), all backward passes, weight gradients
//!
//! This measures the realistic ANE training speedup for a complete transformer.
//!
//! Usage: cargo run --release --example train_transformer -- [D] [heads] [SP] [layers] [steps]

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
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
}
const ROW: i32 = 101;
const NT: i32 = 111;
const TR: i32 = 112;

fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            NT,
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
            TR,
            NT,
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
fn mm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            TR,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

fn build_blob(w: &[f32]) -> Vec<u8> {
    let ws = w.len() * 2;
    let t = 128 + ws;
    let mut b = vec![0u8; t];
    b[0] = 1;
    b[4] = 2;
    b[64] = 0xEF;
    b[65] = 0xBE;
    b[66] = 0xAD;
    b[67] = 0xDE;
    b[68] = 1;
    b[72..76].copy_from_slice(&(ws as u32).to_le_bytes());
    b[80..84].copy_from_slice(&128u32.to_le_bytes());
    for (i, &v) in w.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        b[128 + i * 2] = (h & 0xFF) as u8;
        b[128 + i * 2 + 1] = (h >> 8) as u8;
    }
    b
}
fn to_fp16_inplace(src: &[f32], dst: &mut [u8]) {
    for (i, &v) in src.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        dst[i * 2] = (h & 0xFF) as u8;
        dst[i * 2 + 1] = (h >> 8) as u8;
    }
}
fn from_fp16_inplace(src: &[u8], dst: &mut [f32]) {
    for i in 0..dst.len() {
        let h = (src[i * 2] as u16) | ((src[i * 2 + 1] as u16) << 8);
        dst[i] = f16::from_bits(h).to_f32();
    }
}
fn rand_m(r: usize, c: usize, s: f32, seed: u64) -> Vec<f32> {
    (0..r * c)
        .map(|i| {
            let x = ((i as u64 * 2654435761)
                .wrapping_add(seed)
                .wrapping_mul(0x9E3779B97F4A7C15)
                >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * s
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// MIL GENERATORS
// ═══════════════════════════════════════════════════════════════

fn mil_header() -> &'static str {
    "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
}

/// QKV projection: x → [Q, K, V] concatenated → [1, 3D, 1, SP]
fn mil_qkv(d: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(mil_header());
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
    m.push_str(&(3 * d).to_string());
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
    m.push_str(mil_header());
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

/// FFN + residual (SwiGLU): 3 conv1x1 + sigmoid + 2 mul + concat
fn mil_ffn(d: usize, inter: usize, sp: usize) -> String {
    let total_ic = inter + d;
    let mut m = String::new();
    m.push_str(mil_header());
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
    // gate = Wg @ x, up = Wu @ x
    for (wn, oc, on, nm) in [("Wg", inter, "gate", "cg"), ("Wu", inter, "up", "cu")] {
        m.push_str("        tensor<fp16, [1, ");
        m.push_str(&oc.to_string());
        m.push_str(", 1, ");
        m.push_str(&sp.to_string());
        m.push_str("]> ");
        m.push_str(on);
        m.push_str(
            " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = ",
        );
        m.push_str(wn);
        m.push_str(", x = x)[name = tensor<string, []>(\"");
        m.push_str(nm);
        m.push_str("\")];\n");
    }
    // silu = gate * sigmoid(gate), fused = silu * up
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
    // concat + down with residual
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
// CPU ATTENTION
// ═══════════════════════════════════════════════════════════════

/// CPU multi-head attention: Q, K, V are [D, SP]. Returns (attn_out [D, SP], attn_weights [SP, SP])
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, sp: usize) -> (Vec<f32>, Vec<f32>) {
    let scores = mm_at(q, d, sp, k, sp); // [SP, SP]
    let scale = 1.0 / (d as f32).sqrt();
    let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();
    // Softmax per row
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
    // attn @ V → [SP, D] → transpose to [D, SP]
    let av = mm(&attn, sp, sp, v, d);
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            out[h * sp + i] = av[i * d + h];
        }
    }
    (out, attn)
}

// ═══════════════════════════════════════════════════════════════
// TRANSFORMER LAYER
// ═══════════════════════════════════════════════════════════════

struct TransformerLayer {
    // Attention weights
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    // FFN weights
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    d: usize,
    inter: usize,
    sp: usize,
    // ANE executors (forward only)
    ane_qkv: Option<rustane::wrapper::ANEExecutor>,
    ane_out: Option<rustane::wrapper::ANEExecutor>,
    ane_ffn: Option<rustane::wrapper::ANEExecutor>,
    // Pre-allocated buffers
    x16: Vec<u8>,
    qkv16: Vec<u8>,
    out_in16: Vec<u8>, // 2 inputs for out_proj
    ffn_x16: Vec<u8>,
    out16: Vec<u8>,
    ffn16: Vec<u8>,
    // Cached activations for backward
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,     // [SP, SP]
    attn_out: Vec<f32>, // [D, SP]
    attn_in: Vec<f32>,  // input to attention block = x
    ffn_in: Vec<f32>,   // input to FFN block = out_proj output
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl TransformerLayer {
    fn new(d: usize, inter: usize, sp: usize, seed: u64) -> Self {
        let wq = rand_m(d, d, 0.02, seed);
        let wk = rand_m(d, d, 0.02, seed + 1);
        let wv = rand_m(d, d, 0.02, seed + 2);
        let wo = rand_m(d, d, 0.02, seed + 3);
        let wg = rand_m(inter, d, 0.02, seed + 4);
        let wu = rand_m(inter, d, 0.02, seed + 5);
        let wd = rand_m(d, inter, 0.02, seed + 6);

        // WoI = [Wo | I] for residual
        let mut woi = vec![0.0f32; d * 2 * d];
        for r in 0..d {
            for c in 0..d {
                woi[r * 2 * d + c] = wo[r * d + c];
            }
            woi[r * 2 * d + d + r] = 1.0;
        }
        // WdI = [Wd | I] for residual
        let mut wdi = vec![0.0f32; d * (inter + d)];
        for r in 0..d {
            for c in 0..inter {
                wdi[r * (inter + d) + c] = wd[r * inter + c];
            }
            wdi[r * (inter + d) + inter + r] = 1.0;
        }

        // Compile ANE programs
        let blob_wq = build_blob(&wq);
        let blob_wk = build_blob(&wk);
        let blob_wv = build_blob(&wv);
        let blob_woi = build_blob(&woi);
        let blob_wg = build_blob(&wg);
        let blob_wu = build_blob(&wu);
        let blob_wdi = build_blob(&wdi);

        let ane_qkv = rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_qkv(d, sp),
                &[
                    "@model_path/weights/Wq.bin",
                    "@model_path/weights/Wk.bin",
                    "@model_path/weights/Wv.bin",
                ],
                &[&blob_wq[..], &blob_wk[..], &blob_wv[..]],
                &[blob_wq.len(), blob_wk.len(), blob_wv.len()],
                &[d * sp * 2],
                &[3 * d * sp * 2],
            )
            .expect("qkv compile");

        let ane_out = rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_out_proj(d, sp),
                &["@model_path/weights/WoI.bin"],
                &[&blob_woi[..]],
                &[blob_woi.len()],
                &[d * sp * 2, d * sp * 2],
                &[d * sp * 2],
            )
            .expect("out compile");

        let ane_ffn = rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_ffn(d, inter, sp),
                &[
                    "@model_path/weights/Wg.bin",
                    "@model_path/weights/Wu.bin",
                    "@model_path/weights/WdI.bin",
                ],
                &[&blob_wg[..], &blob_wu[..], &blob_wdi[..]],
                &[blob_wg.len(), blob_wu.len(), blob_wdi.len()],
                &[d * sp * 2],
                &[d * sp * 2],
            )
            .expect("ffn compile");

        TransformerLayer {
            wq,
            wk,
            wv,
            wo,
            wg,
            wu,
            wd,
            d,
            inter,
            sp,
            ane_qkv: Some(ane_qkv),
            ane_out: Some(ane_out),
            ane_ffn: Some(ane_ffn),
            x16: vec![0u8; d * sp * 2],
            qkv16: vec![0u8; 3 * d * sp * 2],
            out_in16: vec![0u8; 2 * d * sp * 2],
            ffn_x16: vec![0u8; d * sp * 2],
            out16: vec![0u8; d * sp * 2],
            ffn16: vec![0u8; d * sp * 2],
            q: vec![0.0f32; d * sp],
            k: vec![0.0f32; d * sp],
            v: vec![0.0f32; d * sp],
            attn: vec![0.0f32; sp * sp],
            attn_out: vec![0.0f32; d * sp],
            attn_in: vec![0.0f32; d * sp],
            ffn_in: vec![0.0f32; d * sp],
            gate: vec![0.0f32; inter * sp],
            up: vec![0.0f32; inter * sp],
        }
    }

    /// CPU-only layer (no ANE compilation — for pure CPU baseline)
    fn new_cpu_only(d: usize, inter: usize, sp: usize, seed: u64) -> Self {
        let wq = rand_m(d, d, 0.02, seed);
        let wk = rand_m(d, d, 0.02, seed + 1);
        let wv = rand_m(d, d, 0.02, seed + 2);
        let wo = rand_m(d, d, 0.02, seed + 3);
        let wg = rand_m(inter, d, 0.02, seed + 4);
        let wu = rand_m(inter, d, 0.02, seed + 5);
        let wd = rand_m(d, inter, 0.02, seed + 6);
        TransformerLayer {
            wq,
            wk,
            wv,
            wo,
            wg,
            wu,
            wd,
            d,
            inter,
            sp,
            ane_qkv: None,
            ane_out: None,
            ane_ffn: None,
            x16: vec![0u8; d * sp * 2],
            qkv16: vec![0u8; 3 * d * sp * 2],
            out_in16: vec![0u8; 2 * d * sp * 2],
            ffn_x16: vec![0u8; d * sp * 2],
            out16: vec![0u8; d * sp * 2],
            ffn16: vec![0u8; d * sp * 2],
            q: vec![0.0f32; d * sp],
            k: vec![0.0f32; d * sp],
            v: vec![0.0f32; d * sp],
            attn: vec![0.0f32; sp * sp],
            attn_out: vec![0.0f32; d * sp],
            attn_in: vec![0.0f32; d * sp],
            ffn_in: vec![0.0f32; d * sp],
            gate: vec![0.0f32; inter * sp],
            up: vec![0.0f32; inter * sp],
        }
    }

    /// ANE forward: x → y (caches activations for backward)
    fn forward_ane(&mut self, x: &[f32]) -> Vec<f32> {
        let d = self.d;
        let sp = self.sp;

        // Cache input
        self.attn_in.copy_from_slice(x);

        // ANE QKV projection
        let ane_qkv = self
            .ane_qkv
            .as_mut()
            .expect("forward_ane requires ANE executors");
        to_fp16_inplace(x, &mut self.x16);
        ane_qkv.write_input(0, &self.x16).unwrap();
        ane_qkv.eval().unwrap();
        ane_qkv.read_output(0, &mut self.qkv16).unwrap();
        let mut qkv = vec![0.0f32; 3 * d * sp];
        from_fp16_inplace(&self.qkv16, &mut qkv);
        let (q, rest) = qkv.split_at(d * sp);
        let (k, v) = rest.split_at(d * sp);
        self.q.copy_from_slice(q);
        self.k.copy_from_slice(k);
        self.v.copy_from_slice(v);

        // CPU attention
        let (attn_out, attn_weights) = cpu_attention(&self.q, &self.k, &self.v, d, sp);
        self.attn_out.copy_from_slice(&attn_out);
        self.attn.copy_from_slice(&attn_weights);

        // ANE output projection + residual
        let ane_out = self
            .ane_out
            .as_mut()
            .expect("forward_ane requires ANE executors");
        to_fp16_inplace(&self.attn_out, &mut self.out_in16[..d * sp * 2]);
        to_fp16_inplace(x, &mut self.out_in16[d * sp * 2..]);
        ane_out
            .write_input(0, &self.out_in16[..d * sp * 2])
            .unwrap();
        ane_out
            .write_input(1, &self.out_in16[d * sp * 2..])
            .unwrap();
        ane_out.eval().unwrap();
        ane_out.read_output(0, &mut self.out16).unwrap();
        let mut ffn_in = vec![0.0f32; d * sp];
        from_fp16_inplace(&self.out16, &mut ffn_in);
        self.ffn_in.copy_from_slice(&ffn_in);

        // ANE FFN + residual
        let ane_ffn = self
            .ane_ffn
            .as_mut()
            .expect("forward_ane requires ANE executors");
        to_fp16_inplace(&ffn_in, &mut self.ffn_x16);
        ane_ffn.write_input(0, &self.ffn_x16).unwrap();
        ane_ffn.eval().unwrap();
        ane_ffn.read_output(0, &mut self.ffn16).unwrap();
        let mut y = vec![0.0f32; d * sp];
        from_fp16_inplace(&self.ffn16, &mut y);
        y
    }

    /// Full CPU forward (for baseline benchmark)
    fn forward_cpu(&self, x: &[f32]) -> Vec<f32> {
        let d = self.d;
        let sp = self.sp;
        let q = mm(&self.wq, d, d, x, sp);
        let k = mm(&self.wk, d, d, x, sp);
        let v = mm(&self.wv, d, d, x, sp);
        let (attn_out, _) = cpu_attention(&q, &k, &v, d, sp);
        let out = mm(&self.wo, d, d, &attn_out, sp);
        let mut y = vec![0.0f32; d * sp];
        for i in 0..y.len() {
            y[i] = out[i] + x[i];
        }
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

    /// CPU backward pass (called after forward_ane cached activations)
    /// Returns gradients for all weights and input gradient
    fn backward_cpu(
        &mut self,
        dy: &[f32], // [D, SP] gradient from above
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let d = self.d;
        let sp = self.sp;
        let inter = self.inter;

        // ═══ FFN backward ═══
        // dL/dWd = dy @ fused^T  (fused = silu * up)
        let mut fused = vec![0.0f32; inter * sp];
        for i in 0..inter * sp {
            let s = 1.0 / (1.0 + (-self.gate[i]).exp());
            fused[i] = self.gate[i] * s * self.up[i];
        }
        let dwd = mm_abt(dy, d, sp, &fused, inter);

        // dfused = Wd^T @ dy
        let dfused = mm_at(&self.wd, d, inter, dy, sp);

        // dL/d(up) = dfused * silu
        let mut dup = vec![0.0f32; inter * sp];
        for i in 0..inter * sp {
            let s = 1.0 / (1.0 + (-self.gate[i]).exp());
            dup[i] = dfused[i] * self.gate[i] * s;
        }
        // dL/dWu = dup @ ffn_in^T
        let dwu = mm_abt(&dup, inter, sp, &self.ffn_in, d);

        // dL/d(gate) = dfused * up * SiLU'(gate)
        let mut dgate = vec![0.0f32; inter * sp];
        for i in 0..inter * sp {
            let g = self.gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            dgate[i] = dfused[i] * self.up[i] * s * (1.0 + g * (1.0 - s));
        }
        let dwg = mm_abt(&dgate, inter, sp, &self.ffn_in, d);

        // dL/d(ffn_in) = Wg^T @ dgate + Wu^T @ dup + dy  (residual)
        let dx_ffn_g = mm_at(&self.wg, inter, d, &dgate, sp);
        let dx_ffn_u = mm_at(&self.wu, inter, d, &dup, sp);
        let mut dffn_in = vec![0.0f32; d * sp];
        for i in 0..d * sp {
            dffn_in[i] = dx_ffn_g[i] + dx_ffn_u[i] + dy[i];
        }

        // ═══ Output projection backward ═══
        // dL/dWo = dffn_in @ attn_out^T  (note: dffn_in already includes residual gradient)
        // Actually: out = Wo @ attn_out + x, so dout/dWo = attn_out^T
        // But dffn_in is the gradient of the residual output, not the Wo output.
        // We need to separate: dffn_in = d_outproj_out + d_residual
        // d_outproj_out = dffn_in (since residual is identity, gradient just passes through)
        // dL/d(attn_out) = Wo^T @ dffn_in
        // dL/dWo = dffn_in @ attn_out^T
        let dattn_out = mm_at(&self.wo, d, d, &dffn_in, sp);
        let dwo = mm_abt(&dffn_in, d, sp, &self.attn_out, d);

        // dL/d(attn_in) = dffn_in (residual gradient passes through)
        // dL/d(x_before_attn) = dattn_out (gradient through attention) + dffn_in (residual)
        // But wait — attn_in is the input to attention block. The residual connection means:
        // ffn_in = Wo @ attn_out + attn_in  (attn_in = x)
        // So dL/d(attn_in) = dffn_in (the residual gradient)

        // ═══ Attention backward ═══
        // attn_out = V @ attn^T  (transposed from [SP, D] to [D, SP])
        // So in our layout: attn_out[h, i] = sum_j attn[i, j] * V[h, j]
        // dL/d(attn) = V^T @ dattn_out^T  → [SP, SP]
        // dL/dV = dattn_out^T @ attn  → [D, SP]
        let dv = mm(&dattn_out, d, sp, &self.attn, sp); // dattn_out [D,SP] @ attn [SP,SP]^T = [D, SP]
                                                        // Actually: attn_out = transpose(attn @ V), so:
                                                        // attn_out[h, i] = sum_j attn[i, j] * V[h, j]
                                                        // dL/dV[h, j] = sum_i dattn_out[h, i] * attn[i, j] = dattn_out @ attn
                                                        // dL/dattn[i, j] = sum_h dattn_out[h, i] * V[h, j] = dattn_out^T @ V

        // Simpler: our cpu_attention computes out = transpose(attn @ V)
        // So dL/dattn = dattn_out^T @ V  (where dattn_out is [D, SP])
        // dL/dV = dattn_out @ attn^T

        let dattn = mm_at(&dattn_out, d, sp, &self.v, sp); // [D,SP]^T @ [D,SP] = [SP,SP]
        let dv2 = mm(&dattn_out, d, sp, &self.attn, sp); // [D,SP] @ [SP,SP] = [D,SP]

        // Softmax backward: dscaled = attn * (dattn - (dattn * attn).row_sum)
        let scale = 1.0 / (d as f32).sqrt();
        let mut dscaled = vec![0.0f32; sp * sp];
        for i in 0..sp {
            let mut dot = 0.0_f32;
            for j in 0..sp {
                dot += dattn[i * sp + j] * self.attn[i * sp + j];
            }
            for j in 0..sp {
                dscaled[i * sp + j] = self.attn[i * sp + j] * (dattn[i * sp + j] - dot) * scale;
            }
        }

        // dscores = dscaled  (already scaled)
        // dL/dQ^T = dscores @ K  → dQ = K @ dscores^T  → [D, SP]
        // dL/dK^T = Q @ dscores^T  → dK = dscores @ Q^T  → [D, SP]
        let dq = mm_at(&self.k, d, sp, &dscaled, sp); // K^T @ dscaled = [D, SP]
        let dk = mm(&dscaled, sp, sp, &self.q, d); // dscaled @ Q^T = [SP, D] → need transpose
                                                   // Actually dk should be [D, SP]: dL/dK[h, j] = sum_i dscaled[i, j] * Q[h, i]
        let dk2 = mm_at(&self.q, d, sp, &dscaled, sp); // Q^T @ dscaled = [D, SP]
                                                       // Hmm, let me think more carefully:
                                                       // scores = Q^T @ K  (using mm_at: Q is [D, SP], K is [D, SP], scores = Q^T @ K = [SP, SP])
                                                       // dL/dQ = K @ dscores^T = K @ dscores^T  where dscores is [SP, SP]
                                                       // dL/dK = Q @ dscores  where Q is [D, SP] and dscores is [SP, SP] → [D, SP]
        let dq_correct = mm(&self.k, d, sp, &dscaled, sp); // [D, SP] @ [SP, SP] = [D, SP] — wait no

        // scores[i,j] = sum_h Q[h,i] * K[h,j]
        // dL/dQ[h,i] = sum_j dscaled[i,j] * K[h,j] = K @ dscaled^T... no
        // dL/dQ[h,i] = sum_j K[h,j] * dscaled[i,j] = (K @ dscaled^T)[h, i]?
        // No: K is [D, SP], dscaled is [SP, SP]
        // K @ dscaled^T: [D, SP] @ [SP, SP]^T = [D, SP] @ [SP, SP] — dimensions don't work for ^T
        // K @ dscaled: [D, SP] @ [SP, SP] = [D, SP] ✓ — this is (K @ dscaled)[h, i] = sum_j K[h,j] * dscaled[j, i]
        // But we want sum_j dscaled[i,j] * K[h,j] = sum_j K[h,j] * dscaled[i,j]
        // = (dscaled @ K^T)[i, h] → need to transpose → K @ dscaled^T

        // OK let me just be careful:
        // scores = Q^T @ K where Q,K are [D, SP], scores is [SP, SP]
        // dL/dQ: for each h,i: sum_j dscores[i,j] * d(scores[i,j])/d(Q[h,i]) = sum_j dscores[i,j] * K[h,j]
        // This is (dscores @ K^T)[i, h], transposed to [D, SP] = (K @ dscores^T)[h, i]
        // Using mm: mm(K, D, SP, dscores_t, SP, SP) — but dscores_t is [SP, SP], mm does [D,SP]@[SP,SP]=[D,SP]... no
        // mm(K, D, SP, X, SP, SP) = K[D,SP] @ X[SP,SP] = [D,SP] ← wrong, that's K @ X
        // We want K @ dscores^T = [D,SP] @ [SP,SP] — but mm_abt does A @ B^T
        // mm_abt(K, D, SP, dscaled, SP, SP) = K[D,SP] @ dscaled[SP,SP]^T = K[D,SP] @ dscaled^T[SP,SP] = [D,SP] ✓

        let dq_final = mm_abt(&self.k, d, sp, &dscaled, sp); // K @ dscaled^T = [D, SP]
                                                             // dL/dK: for each h,j: sum_i dscores[i,j] * d(scores[i,j])/d(K[h,j]) = sum_i dscores[i,j] * Q[h,i]
                                                             // = (Q @ dscores)[h, j] ← Q[D,SP] @ dscores[SP,SP] = [D,SP] ✓
        let dk_final = mm(&self.q, d, sp, &dscaled, sp); // Q @ dscores = [D, SP]
                                                         // dL/dV: for each h,j: sum_i dattn_out[h,i] * d(attn_out[h,i])/d(V[h,j])
                                                         // attn_out[h,i] = sum_j attn[i,j] * V[h,j]
                                                         // dL/dV[h,j] = sum_i dattn_out[h,i] * attn[i,j] = (dattn_out @ attn)[h, j]
        let dv_final = mm(&dattn_out, d, sp, &self.attn, sp); // [D, SP] @ [SP, SP] = [D, SP]

        // Weight gradients for attention
        let dwq = mm_abt(&dq_final, d, sp, &self.attn_in, d);
        let dwk = mm_abt(&dk_final, d, sp, &self.attn_in, d);
        let dwv = mm_abt(&dv_final, d, sp, &self.attn_in, d);

        // dL/d(attn_in) = Q^T^T @ dscores @ K^T ... no, simpler:
        // dL/d(x) where x is the input to QKV projections:
        // dL/dx = Wq^T @ dq + Wk^T @ dk + Wv^T @ dv + dffn_in (residual)
        let dx_q = mm_at(&self.wq, d, d, &dq_final, sp);
        let dx_k = mm_at(&self.wk, d, d, &dk_final, sp);
        let dx_v = mm_at(&self.wv, d, d, &dv_final, sp);
        let mut dx = vec![0.0f32; d * sp];
        for i in 0..d * sp {
            dx[i] = dx_q[i] + dx_k[i] + dx_v[i] + dffn_in[i];
        }

        // Note: we don't use gate/up from ANE forward for backward because
        // the ANE FFN computes internally. We need to recompute on CPU.
        // Actually we DO cache gate/up in forward_ane — but wait, the ANE FFN
        // program doesn't return gate/up. It only returns y. So we need to
        // recompute gate/up from ffn_in on CPU for backward.
        self.gate = mm(&self.wg, inter, d, &self.ffn_in, sp);
        self.up = mm(&self.wu, inter, d, &self.ffn_in, sp);

        (dx, dwq, dwk, dwv, dwo, dwg, dwu, dwd)
    }

    /// SGD update
    fn sgd_update(
        &mut self,
        dwq: &[f32],
        dwk: &[f32],
        dwv: &[f32],
        dwo: &[f32],
        dwg: &[f32],
        dwu: &[f32],
        dwd: &[f32],
        lr: f32,
    ) {
        unsafe {
            cblas_saxpy(
                self.wq.len() as i32,
                -lr,
                dwq.as_ptr(),
                1,
                self.wq.as_mut_ptr(),
                1,
            );
            cblas_saxpy(
                self.wk.len() as i32,
                -lr,
                dwk.as_ptr(),
                1,
                self.wk.as_mut_ptr(),
                1,
            );
            cblas_saxpy(
                self.wv.len() as i32,
                -lr,
                dwv.as_ptr(),
                1,
                self.wv.as_mut_ptr(),
                1,
            );
            cblas_saxpy(
                self.wo.len() as i32,
                -lr,
                dwo.as_ptr(),
                1,
                self.wo.as_mut_ptr(),
                1,
            );
            cblas_saxpy(
                self.wg.len() as i32,
                -lr,
                dwg.as_ptr(),
                1,
                self.wg.as_mut_ptr(),
                1,
            );
            cblas_saxpy(
                self.wu.len() as i32,
                -lr,
                dwu.as_ptr(),
                1,
                self.wu.as_mut_ptr(),
                1,
            );
            cblas_saxpy(
                self.wd.len() as i32,
                -lr,
                dwd.as_ptr(),
                1,
                self.wd.as_mut_ptr(),
                1,
            );
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let _heads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let sp: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let layers: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    let steps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);
    let inter = d * 4;
    let lr = 0.001_f32;

    println!("============================================================");
    println!("  Full Transformer Training Benchmark");
    println!(
        "  D={} SP={} inter={} layers={} steps={}",
        d, sp, inter, layers, steps
    );
    println!("  ANE: QKV + out_proj + FFN (3 programs/layer)");
    println!("  CPU: attention + all backward + weight gradients");
    println!("============================================================\n");

    let params_per_layer = 4 * d * d + 3 * d * inter;
    let total_params = params_per_layer * layers;
    println!(
        "  Parameters: {:.1}M per layer, {:.1}M total",
        params_per_layer as f64 / 1e6,
        total_params as f64 / 1e6
    );
    println!(
        "  ANE compiles: {} (3 per layer × {} layers)\n",
        3 * layers,
        layers
    );

    // Create layers
    let mut ane_layers: Vec<TransformerLayer> = (0..layers)
        .map(|i| TransformerLayer::new(d, inter, sp, 42 + i as u64 * 1000))
        .collect();

    // Input and target
    let x: Vec<f32> = rand_m(d, sp, 0.1, 9999);
    let target_w = rand_m(d, d, 0.01, 12345);
    let target_out = mm(&target_w, d, d, &x, sp);

    // Warmup
    let mut tmp_x = x.clone();
    for layer in &mut ane_layers {
        tmp_x = layer.forward_ane(&tmp_x);
    }
    let dy = vec![0.0f32; d * sp];
    for i in (0..layers).rev() {
        let _ = ane_layers[i].backward_cpu(&dy);
    }

    // ── Benchmark: ANE Forward + CPU Backward ──
    let mut fwd_times = Vec::new();
    let mut bwd_times = Vec::new();
    let mut step_times = Vec::new();
    let mut loss_sum = 0.0_f64;

    for _step in 0..steps {
        let mut tmp_x = x.clone();
        let mut layer_inputs: Vec<Vec<f32>> = Vec::new();
        layer_inputs.push(tmp_x.clone());

        let t_step = Instant::now();
        let t_fwd = Instant::now();

        for layer in &mut ane_layers {
            tmp_x = layer.forward_ane(&tmp_x);
            layer_inputs.push(tmp_x.clone());
        }

        let fwd_t = t_fwd.elapsed().as_secs_f64() * 1000.0;

        // MSE loss
        let n = tmp_x.len() as f32;
        let mut loss = 0.0_f32;
        let mut dy = vec![0.0f32; tmp_x.len()];
        for i in 0..tmp_x.len() {
            let diff = tmp_x[i] - target_out[i];
            loss += diff * diff;
            dy[i] = 2.0 * diff / n;
        }
        loss /= n;
        loss_sum += loss as f64;

        let t_bwd = Instant::now();
        let mut grad = dy;
        for i in (0..layers).rev() {
            let (dx, dwq, dwk, dwv, dwo, dwg, dwu, dwd) = ane_layers[i].backward_cpu(&grad);
            ane_layers[i].sgd_update(&dwq, &dwk, &dwv, &dwo, &dwg, &dwu, &dwd, lr);
            grad = dx;
        }
        let bwd_t = t_bwd.elapsed().as_secs_f64() * 1000.0;
        let step_t = t_step.elapsed().as_secs_f64() * 1000.0;

        fwd_times.push(fwd_t);
        bwd_times.push(bwd_t);
        step_times.push(step_t);
    }

    let avg_fwd = fwd_times.iter().sum::<f64>() / steps as f64;
    let avg_bwd = bwd_times.iter().sum::<f64>() / steps as f64;
    let avg_step = step_times.iter().sum::<f64>() / steps as f64;

    // ── Benchmark: Pure CPU ──
    let mut cpu_layers: Vec<TransformerLayer> = (0..layers)
        .map(|i| TransformerLayer::new_cpu_only(d, inter, sp, 42 + i as u64 * 1000))
        .collect();

    let mut cpu_fwd_times = Vec::new();
    let mut cpu_bwd_times = Vec::new();
    let mut cpu_step_times = Vec::new();

    for _step in 0..steps {
        let mut tmp_x = x.clone();
        let mut layer_inputs: Vec<Vec<f32>> = Vec::new();
        layer_inputs.push(tmp_x.clone());

        let t_step = Instant::now();
        let t_fwd = Instant::now();

        for layer in &mut cpu_layers {
            let y = layer.forward_cpu(&tmp_x);
            // Manually cache activations for backward
            layer.attn_in = tmp_x.clone();
            let q = mm(&layer.wq, d, d, &tmp_x, sp);
            let k = mm(&layer.wk, d, d, &tmp_x, sp);
            let v = mm(&layer.wv, d, d, &tmp_x, sp);
            layer.q = q;
            layer.k = k;
            layer.v = v;
            let (ao, aw) = cpu_attention(&layer.q, &layer.k, &layer.v, d, sp);
            layer.attn_out = ao;
            layer.attn = aw;
            let mut out = mm(&layer.wo, d, d, &layer.attn_out, sp);
            for j in 0..out.len() {
                out[j] += tmp_x[j];
            }
            layer.ffn_in = out.clone();
            layer.gate = mm(&layer.wg, inter, d, &out, sp);
            layer.up = mm(&layer.wu, inter, d, &out, sp);
            tmp_x = y;
            layer_inputs.push(tmp_x.clone());
        }

        let fwd_t = t_fwd.elapsed().as_secs_f64() * 1000.0;

        let n = tmp_x.len() as f32;
        let mut dy = vec![0.0f32; tmp_x.len()];
        for i in 0..tmp_x.len() {
            let diff = tmp_x[i] - target_out[i];
            dy[i] = 2.0 * diff / n;
        }

        let t_bwd = Instant::now();
        let mut grad = dy;
        for i in (0..layers).rev() {
            let (dx, dwq, dwk, dwv, dwo, dwg, dwu, dwd) = cpu_layers[i].backward_cpu(&grad);
            cpu_layers[i].sgd_update(&dwq, &dwk, &dwv, &dwo, &dwg, &dwu, &dwd, lr);
            grad = dx;
        }
        let bwd_t = t_bwd.elapsed().as_secs_f64() * 1000.0;
        let step_t = t_step.elapsed().as_secs_f64() * 1000.0;

        cpu_fwd_times.push(fwd_t);
        cpu_bwd_times.push(bwd_t);
        cpu_step_times.push(step_t);
    }

    let cpu_avg_fwd = cpu_fwd_times.iter().sum::<f64>() / steps as f64;
    let cpu_avg_bwd = cpu_bwd_times.iter().sum::<f64>() / steps as f64;
    let cpu_avg_step = cpu_step_times.iter().sum::<f64>() / steps as f64;

    // ── Results ──
    println!(
        "=== Full Transformer Training ({} layers, {:.1}M params) ===\n",
        layers,
        total_params as f64 / 1e6
    );
    println!(
        "  {:25} {:>10} {:>10} {:>10}",
        "", "ANE+CPU", "CPU", "Speedup"
    );
    println!(
        "  {:25} {:>10} {:>10} {:>10}",
        "─".repeat(25),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10)
    );
    println!(
        "  {:25} {:>8}ms {:>8}ms {:>8}x",
        "Forward (ANE linear + CPU attn)",
        format!("{:.1}", avg_fwd),
        format!("{:.1}", cpu_avg_fwd),
        format!("{:.2}", cpu_avg_fwd / avg_fwd)
    );
    println!(
        "  {:25} {:>8}ms {:>8}ms {:>8}x",
        "Backward (all CPU)",
        format!("{:.1}", avg_bwd),
        format!("{:.1}", cpu_avg_bwd),
        format!("{:.2}", cpu_avg_bwd / avg_bwd)
    );
    println!(
        "  {:25} {:>8}ms {:>8}ms {:>8}x",
        "Total step",
        format!("{:.1}", avg_step),
        format!("{:.1}", cpu_avg_step),
        format!("{:.2}", cpu_avg_step / avg_step)
    );
    println!(
        "\n  Loss: {:.6} (avg over {} steps)",
        loss_sum / steps as f64,
        steps
    );
    println!(
        "\n  Fwd/Bwd ratio: {:.0}% fwd, {:.0}% bwd",
        avg_fwd / avg_step * 100.0,
        avg_bwd / avg_step * 100.0
    );
    println!("  ANE compiles used: {}/~119", 3 * layers);
}
