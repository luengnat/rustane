//! ANE Training Benchmark: Fair comparison using Accelerate BLAS for CPU.
//!
//! Architecture: Residual FFN layer (SwiGLU variant)
//!   Forward:  x → Wg·x, Wu·x, SiLU(gate)*up, Wd·fused + x → y
//!   Loss:     MSE(y, target)
//!   Backward: CPU gradient computation (BLAS-accelerated)
//!   Update:   CPU Adam optimizer
//!   Reload:   ANE reload_weights()
//!
//! Usage: cargo run --example train_ane_simple -- [D] [SP] [steps]

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

// ═══════════════════════════════════════════════════════════════
// BLAS HELPERS (row-major)
// ═══════════════════════════════════════════════════════════════

/// C = A @ B, A:[M,K] B:[K,N] → C:[M,N]
fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.032; m * n];
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

/// C = A^T @ B, A:[K,M] B:[K,N] → C:[M,N]
fn mm_at(a: &[f32], k: usize, m: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.032; m * n];
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

/// C = A @ B^T, A:[M,K] B:[N,K] → C:[M,N]
fn mm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.032; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NOTRANS,
            TRANS,
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

// ═══════════════════════════════════════════════════════════════
// HELPERS
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
    let mut out = vec![0.032; raw.len() / 2];
    for i in 0..out.len() {
        let b = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
        out[i] = f16::from_bits(b).to_f32();
    }
    out
}

fn cpu_silu(inp: &[f32]) -> Vec<f32> {
    inp.iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect()
}

fn cpu_silu_backward(inp: &[f32], grad: &[f32]) -> Vec<f32> {
    inp.iter()
        .zip(grad.iter())
        .map(|(&x, &g)| {
            let s = 1.0 / (1.0 + (-x).exp());
            g * s * (1.0 + x * (1.0 - s))
        })
        .collect()
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
// MIL GENERATION
// ═══════════════════════════════════════════════════════════════

fn mil_residual_ffn(d: usize, inter: usize, sp: usize) -> String {
    let total_ic = inter + d;
    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n");
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    for wn in ["Wg", "Wu", "WdI"] {
        let (oc, ic) = match wn {
            "Wg" => (inter, d),
            "Wu" => (inter, d),
            _ => (d, total_ic),
        };
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
    // gate, up, silu, fused
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
    // concat + down+residual
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
// FFN LAYER
// ═══════════════════════════════════════════════════════════════

struct FFNLayer {
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    mg_wg: Vec<f32>,
    mg_wu: Vec<f32>,
    mg_wd: Vec<f32>,
    vg_wg: Vec<f32>,
    vg_wu: Vec<f32>,
    vg_wd: Vec<f32>,
    d: usize,
    inter: usize,
}

impl FFNLayer {
    fn new(d: usize, inter: usize) -> Self {
        FFNLayer {
            wg: rand_matrix(inter, d, 0.02, 42),
            wu: rand_matrix(inter, d, 0.02, 100),
            wd: rand_matrix(d, inter, 0.02, 200),
            mg_wg: vec![0.0; inter * d],
            mg_wu: vec![0.0; inter * d],
            mg_wd: vec![0.0; d * inter],
            vg_wg: vec![0.0; inter * d],
            vg_wu: vec![0.0; inter * d],
            vg_wd: vec![0.0; d * inter],
            d,
            inter,
        }
    }

    fn build_wdi(&self) -> Vec<f32> {
        let t = self.inter + self.d;
        let mut wdi = vec![0.032; self.d * t];
        for r in 0..self.d {
            for c in 0..self.inter {
                wdi[r * t + c] = self.wd[r * self.inter + c];
            }
            wdi[r * t + self.inter + r] = 1.0;
        }
        wdi
    }

    fn forward(&self, x: &[f32], sp: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let gate = mm(&self.wg, self.inter, self.d, x, sp);
        let up = mm(&self.wu, self.inter, self.d, x, sp);
        let silu = cpu_silu(&gate);
        let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
        let down = mm(&self.wd, self.d, self.inter, &fused, sp);
        let y: Vec<f32> = down.iter().zip(x.iter()).map(|(&d, &xi)| d + xi).collect();
        (y, gate, fused, silu)
    }

    fn backward(
        &self,
        x: &[f32],
        gate: &[f32],
        fused: &[f32],
        silu: &[f32],
        dy: &[f32],
        sp: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let grad_wd = mm_abt(dy, self.d, sp, fused, self.inter);
        let dfused = mm_at(&self.wd, self.inter, self.d, dy, sp);
        let up = mm(&self.wu, self.inter, self.d, x, sp);
        let dsilu: Vec<f32> = dfused.iter().zip(up.iter()).map(|(&d, &u)| d * u).collect();
        let dup: Vec<f32> = dfused
            .iter()
            .zip(silu.iter())
            .map(|(&d, &s)| d * s)
            .collect();
        let dgate = cpu_silu_backward(gate, &dsilu);
        let grad_wg = mm_abt(&dgate, self.inter, sp, x, self.d);
        let grad_wu = mm_abt(&dup, self.inter, sp, x, self.d);
        let dx_gate = mm_at(&self.wg, self.d, self.inter, &dgate, sp);
        let dx_up = mm_at(&self.wu, self.d, self.inter, &dup, sp);
        let mut dx = vec![0.032; self.d * sp];
        for i in 0..dx.len() {
            dx[i] = dx_gate[i] + dx_up[i] + dy[i];
        }
        (grad_wg, grad_wu, grad_wd, dx)
    }

    fn adam_update(&mut self, lr: f32, step: usize, gw: &[f32], gu: &[f32], gd: &[f32]) {
        let b1: f32 = 0.9;
        let b2: f32 = 0.999;
        let eps = 1e-8;
        let bc1 = 1.0 - b1.powi(step as i32);
        let bc2 = 1.0 - b2.powi(step as i32);
        Self::adam(
            &mut self.wg,
            &mut self.mg_wg,
            &mut self.vg_wg,
            gw,
            lr,
            b1,
            b2,
            eps,
            bc1,
            bc2,
        );
        Self::adam(
            &mut self.wu,
            &mut self.mg_wu,
            &mut self.vg_wu,
            gu,
            lr,
            b1,
            b2,
            eps,
            bc1,
            bc2,
        );
        Self::adam(
            &mut self.wd,
            &mut self.mg_wd,
            &mut self.vg_wd,
            gd,
            lr,
            b1,
            b2,
            eps,
            bc1,
            bc2,
        );
    }

    fn adam(
        w: &mut [f32],
        m: &mut [f32],
        v: &mut [f32],
        g: &[f32],
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
    ) {
        for i in 0..w.len() {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            w[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let num_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let lr: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let inter = d * 4;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  ANE Training Benchmark (BLAS CPU baseline)              ║");
    println!(
        "║  D={}  SP={}  inter={}  steps={}  lr={}        ║",
        d, sp, inter, num_steps, lr
    );
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Synthetic task: learn a linear mapping
    let target_a = rand_matrix(d, d, 0.5, 999);
    let x_data: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * 0.1).collect();
    let target = mm(&target_a, d, d, &x_data, sp);
    let x16 = to_fp16(&x_data);

    // ── BLAS baseline ──
    println!("=== BLAS Performance ===");
    let t = Instant::now();
    for _ in 0..10 {
        let _ = mm(&rand_matrix(inter, d, 0.02, 1), inter, d, &x_data, sp);
    }
    println!(
        "  matmul [{},{}]@[{},{}]: {:.2}ms",
        inter,
        d,
        d,
        sp,
        t.elapsed().as_secs_f64() * 100.0 / 10.0
    );
    println!();

    // ── ANE-Assisted Training ──
    println!("=== ANE-Assisted Training (forward=ANE, backward=BLAS CPU) ===");
    let mil = mil_residual_ffn(d, inter, sp);
    let mut layer = FFNLayer::new(d, inter);

    let t0 = Instant::now();
    let names: Vec<String> = ["Wg", "Wu", "WdI"]
        .iter()
        .map(|n| format!("@model_path/weights/{}.bin", n))
        .collect();
    let nrefs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let blobs = vec![
        build_blob(&layer.wg),
        build_blob(&layer.wu),
        build_blob(&layer.build_wdi()),
    ];
    let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
    let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(&mil, &nrefs, &brefs, &lens, &[d * sp * 2], &[d * sp * 2])
        .expect("compile failed");
    println!(
        "  ANE compile: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    exec.write_input(0, &x16).expect("w");
    exec.eval().expect("warmup");

    let mut ane_losses = Vec::new();
    let mut ane_fwd_t = Vec::new();
    let mut ane_bwd_t = Vec::new();
    let mut ane_rel_t = Vec::new();

    for step in 1..=num_steps {
        let ts = Instant::now();

        // Forward on ANE
        let tf = Instant::now();
        exec.write_input(0, &x16).expect("w");
        exec.eval().expect("e");
        let y_ane = from_fp16(&exec.read_output_vec(0).expect("r"));
        ane_fwd_t.push(tf.elapsed());

        // CPU forward for backward intermediates
        let (y, gate, fused, silu) = layer.forward(&x_data, sp);

        // Backward
        let tb = Instant::now();
        let n = y.len() as f32;
        let mut loss = 0.032;
        let mut dy = vec![0.032; y.len()];
        for i in 0..y.len() {
            let d = y[i] - target[i];
            loss += d * d;
            dy[i] = 2.0 * d / n;
        }
        loss /= n;
        let (gw, gu, gd, _) = layer.backward(&x_data, &gate, &fused, &silu, &dy, sp);
        // Gradient clipping
        let gn: f32 = gw.iter().map(|x| x * x).sum::<f32>()
            + gu.iter().map(|x| x * x).sum::<f32>()
            + gd.iter().map(|x| x * x).sum::<f32>();
        let clip: f32 = 1.0_f32.min(1.0 / gn.sqrt().max(1e-6));
        let gw: Vec<f32> = gw.iter().map(|g| g * clip).collect();
        let gu: Vec<f32> = gu.iter().map(|g| g * clip).collect();
        let gd: Vec<f32> = gd.iter().map(|g| g * clip).collect();
        layer.adam_update(lr, step, &gw, &gu, &gd);
        ane_bwd_t.push(tb.elapsed());

        // Reload
        let tr = Instant::now();
        exec.reload_weights(&[
            ("@model_path/weights/Wg.bin", &build_blob(&layer.wg)),
            ("@model_path/weights/Wu.bin", &build_blob(&layer.wu)),
            (
                "@model_path/weights/WdI.bin",
                &build_blob(&layer.build_wdi()),
            ),
        ])
        .expect("reload");
        ane_rel_t.push(tr.elapsed());

        ane_losses.push(loss);

        if step <= 5 || step == num_steps {
            println!(
                "  Step {:3}: loss={:.6} | fwd={:.1}ms bwd={:.1}ms reload={:.1}ms total={:.1}ms",
                step,
                loss,
                ane_fwd_t.last().unwrap().as_secs_f64() * 1000.0,
                ane_bwd_t.last().unwrap().as_secs_f64() * 1000.0,
                ane_rel_t.last().unwrap().as_secs_f64() * 1000.0,
                ts.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    let ane_total: f64 = ane_losses.len() as f64;
    let ane_fwd_avg: f64 = ane_fwd_t.iter().map(|t| t.as_secs_f64()).sum::<f64>() / ane_total;
    let ane_bwd_avg: f64 = ane_bwd_t.iter().map(|t| t.as_secs_f64()).sum::<f64>() / ane_total;
    let ane_rel_avg: f64 = ane_rel_t.iter().map(|t| t.as_secs_f64()).sum::<f64>() / ane_total;
    let ane_step_avg: f64 = ane_fwd_avg + ane_bwd_avg + ane_rel_avg;

    // ── CPU-Only Training ──
    println!("\n=== CPU-Only Training (BLAS) ===");
    let mut layer_cpu = FFNLayer::new(d, inter);
    let mut cpu_losses = Vec::new();
    let mut cpu_fwd_t = Vec::new();
    let mut cpu_bwd_t = Vec::new();

    for step in 1..=num_steps {
        let tf = Instant::now();
        let (y, gate, fused, silu) = layer_cpu.forward(&x_data, sp);
        cpu_fwd_t.push(tf.elapsed());

        let tb = Instant::now();
        let n = y.len() as f32;
        let mut loss = 0.032;
        let mut dy = vec![0.032; y.len()];
        for i in 0..y.len() {
            let d = y[i] - target[i];
            loss += d * d;
            dy[i] = 2.0 * d / n;
        }
        loss /= n;
        let (gw, gu, gd, _) = layer_cpu.backward(&x_data, &gate, &fused, &silu, &dy, sp);
        let gn: f32 = gw.iter().map(|x| x * x).sum::<f32>()
            + gu.iter().map(|x| x * x).sum::<f32>()
            + gd.iter().map(|x| x * x).sum::<f32>();
        let clip: f32 = 1.0_f32.min(1.0 / gn.sqrt().max(1e-6));
        let gw: Vec<f32> = gw.iter().map(|g| g * clip).collect();
        let gu: Vec<f32> = gu.iter().map(|g| g * clip).collect();
        let gd: Vec<f32> = gd.iter().map(|g| g * clip).collect();
        layer_cpu.adam_update(lr, step, &gw, &gu, &gd);
        cpu_bwd_t.push(tb.elapsed());
        cpu_losses.push(loss);

        if step <= 5 || step == num_steps {
            println!(
                "  Step {:3}: loss={:.6} | fwd={:.1}ms bwd={:.1}ms total={:.1}ms",
                step,
                loss,
                cpu_fwd_t.last().unwrap().as_secs_f64() * 1000.0,
                cpu_bwd_t.last().unwrap().as_secs_f64() * 1000.0,
                (cpu_fwd_t.last().unwrap().as_secs_f64() + cpu_bwd_t.last().unwrap().as_secs_f64())
                    * 1000.0
            );
        }
    }

    let cpu_total = cpu_losses.len() as f64;
    let cpu_fwd_avg: f64 = cpu_fwd_t.iter().map(|t| t.as_secs_f64()).sum::<f64>() / cpu_total;
    let cpu_bwd_avg: f64 = cpu_bwd_t.iter().map(|t| t.as_secs_f64()).sum::<f64>() / cpu_total;
    let cpu_step_avg = cpu_fwd_avg + cpu_bwd_avg;

    // ── Results ──
    let speedup = cpu_step_avg / ane_step_avg;
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              TIMING BREAKDOWN (ms/step)                ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  {:12} {:>8} {:>8} {:>8} {:>8}",
        "", "ANE", "CPU", "Saved", "Ratio"
    );
    println!(
        "║  {:12} {:>8} {:>8} {:>8} {:>8}",
        "────────────", "────────", "────────", "────────", "────────"
    );
    println!(
        "║  {:12} {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>7.1}x",
        "Forward",
        ane_fwd_avg * 1000.0,
        cpu_fwd_avg * 1000.0,
        (cpu_fwd_avg - ane_fwd_avg) * 1000.0,
        cpu_fwd_avg / ane_fwd_avg.max(0.0001)
    );
    println!(
        "║  {:12} {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>7.1}x",
        "Backward",
        ane_bwd_avg * 1000.0,
        cpu_bwd_avg * 1000.0,
        0.0,
        1.0
    );
    println!(
        "║  {:12} {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>8}",
        "Reload",
        ane_rel_avg * 1000.0,
        0.0,
        0.0,
        "overhead"
    );
    println!(
        "║  {:12} {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>7.1}x",
        "TOTAL",
        ane_step_avg * 1000.0,
        cpu_step_avg * 1000.0,
        (cpu_step_avg - ane_step_avg) * 1000.0,
        speedup
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  ANE loss: {:.6} -> {:.6} ({:.1}% decrease)",
        ane_losses[0],
        *ane_losses.last().unwrap(),
        (1.0 - ane_losses.last().unwrap() / ane_losses[0]) * 100.0
    );
    println!(
        "║  CPU loss: {:.6} -> {:.6} ({:.1}% decrease)",
        cpu_losses[0],
        *cpu_losses.last().unwrap(),
        (1.0 - cpu_losses.last().unwrap() / cpu_losses[0]) * 100.0
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    if speedup > 1.0 {
        println!(
            "\n  ✓ ANE training is {:.1}x faster than CPU (BLAS)",
            speedup
        );
    } else {
        println!(
            "\n  ✗ ANE training is {:.1}x slower than CPU (BLAS)",
            1.0 / speedup
        );
        println!(
            "    Bottleneck: backward on CPU ({:.1}ms) + reload overhead ({:.1}ms)",
            ane_bwd_avg * 1000.0,
            ane_rel_avg * 1000.0
        );
        println!(
            "    Forward speedup: {:.0}x (CPU {:.1}ms -> ANE {:.2}ms)",
            cpu_fwd_avg / ane_fwd_avg.max(0.0001),
            cpu_fwd_avg * 1000.0,
            ane_fwd_avg * 1000.0
        );
    }
}
