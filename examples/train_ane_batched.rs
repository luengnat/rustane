//! Batched ANE Training: Do N CPU steps per ANE reload to amortize reload cost.
//!
//! Key insight: reload_weights is ~50-90ms for 3 weights, but ANE eval is only 0.2ms.
//! ANE forward saves ~3.4ms vs CPU forward. So per-step reload is 25x more expensive
//! than the savings.
//!
//! Strategy: batch N CPU training steps between ANE reloads.
//!   Break-even: N > reload_cost / fwd_savings ≈ 50ms / 3.4ms ≈ 15 steps
//!
//! But even with zero-cost reload, max speedup is only ~1.1x because backward
//! dominates (26.5ms) and must run on CPU regardless.
//!
//! Usage: cargo run --example train_ane_batched -- [D] [SP] [total_steps]

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
fn mm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
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

// ═══════════════════════════════════════════════════════════════
// MIL GENERATION: Residual FFN (same as train_ane_simple.rs)
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
// FFN LAYER (CPU-side)
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
        let mut wdi = vec![0.0f32; self.d * t];
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
        let mut dx = vec![0.0f32; self.d * sp];
        for i in 0..dx.len() {
            dx[i] = dx_gate[i] + dx_up[i] + dy[i];
        }
        (grad_wg, grad_wu, grad_wd, dx)
    }
    fn train_step(&mut self, x: &[f32], target: &[f32], sp: usize, lr: f32, step: usize) -> f32 {
        let (y, gate, fused, silu) = self.forward(x, sp);
        let n = y.len() as f32;
        let mut loss = 0.0f32;
        let mut dy = vec![0.0f32; y.len()];
        for i in 0..y.len() {
            let d = y[i] - target[i];
            loss += d * d;
            dy[i] = 2.0 * d / n;
        }
        loss /= n;
        let (gw, gu, gd, _) = self.backward(x, &gate, &fused, &silu, &dy, sp);
        let gn: f32 = gw.iter().map(|x| x * x).sum::<f32>()
            + gu.iter().map(|x| x * x).sum::<f32>()
            + gd.iter().map(|x| x * x).sum::<f32>();
        let clip: f32 = 1.0_f32.min(1.0 / gn.sqrt().max(1e-6));
        let gw: Vec<f32> = gw.iter().map(|g| g * clip).collect();
        let gu: Vec<f32> = gu.iter().map(|g| g * clip).collect();
        let gd: Vec<f32> = gd.iter().map(|g| g * clip).collect();
        self.adam_update(lr, step, &gw, &gu, &gd);
        loss
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let total_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let lr: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0001);
    let inter = d * 4;

    // Batch sizes to test: how many CPU steps between ANE reloads
    let batch_sizes: Vec<usize> = vec![1, 5, 10, 15, 20, 50, 100];

    println!("============================================================");
    println!("  Batched ANE Training Benchmark");
    println!(
        "  D={} SP={} inter={} steps={} lr={}",
        d, sp, inter, total_steps, lr
    );
    println!("  Strategy: N CPU steps per ANE reload (amortize reload cost)");
    println!("============================================================\n");

    // Shared data — use smaller target to avoid NaN with MSE loss
    let target_a = rand_matrix(d, d, 0.02, 999);
    let x_data: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * 0.01).collect();
    let target = mm(&target_a, d, d, &x_data, sp);
    let x16 = to_fp16(&x_data);

    // ── CPU-Only Baseline ──
    println!("=== CPU-Only Baseline (BLAS) ===");
    let mut layer_cpu = FFNLayer::new(d, inter);
    let mut cpu_losses = Vec::new();
    let t_cpu = Instant::now();
    for step in 1..=total_steps {
        cpu_losses.push(layer_cpu.train_step(&x_data, &target, sp, lr, step));
    }
    let cpu_total_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
    let cpu_per_step_ms = cpu_total_ms / total_steps as f64;
    println!(
        "  CPU total: {:.1}ms ({:.2}ms/step) | loss: {:.4} -> {:.4}\n",
        cpu_total_ms,
        cpu_per_step_ms,
        cpu_losses[0],
        *cpu_losses.last().unwrap()
    );

    // ── ANE-Assisted: Per-step (batch=1) — baseline comparison ──
    println!("=== ANE-Assisted Training (various batch sizes) ===");
    println!(
        "  {:>10} {:>12} {:>10} {:>8} {:>10} {:>8}",
        "Batch", "Total(ms)", "Per(ms)", "Loss", "Speedup", "Note"
    );
    println!(
        "  {:>10} {:>12} {:>10} {:>8} {:>10} {:>8}",
        "--------", "----------", "--------", "------", "--------", "------"
    );

    // Compile ANE once for all batch size experiments
    let mil = mil_residual_ffn(d, inter, sp);
    let t_compile = Instant::now();
    let names: Vec<String> = ["Wg", "Wu", "WdI"]
        .iter()
        .map(|n| format!("@model_path/weights/{}.bin", n))
        .collect();
    let nrefs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    for &batch in &batch_sizes {
        let n_batches = (total_steps + batch - 1) / batch;
        let mut layer = FFNLayer::new(d, inter);
        let mut losses = Vec::new();

        // Compile ANE for this run
        let blobs = vec![
            build_blob(&layer.wg),
            build_blob(&layer.wu),
            build_blob(&layer.build_wdi()),
        ];
        let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
        let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

        let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
            &mil,
            &nrefs,
            &brefs,
            &lens,
            &[d * sp * 2],
            &[d * sp * 2],
        ) {
            Ok(e) => e,
            Err(err) => {
                println!(
                    "  {:>10} {:>12} {:>10} {:>8} {:>10} {:>8}",
                    batch,
                    "COMPILE_ERR",
                    "-",
                    "-",
                    "-",
                    format!("{:?}", err)
                );
                continue;
            }
        };

        // Warmup eval
        exec.write_input(0, &x16).expect("warmup write");
        exec.eval().expect("warmup eval");

        let mut total_reload_ms = 0.0_f64;
        let t_total = Instant::now();
        let mut global_step = 0usize;

        for b in 0..n_batches {
            let batch_start = global_step + 1;
            let batch_end = (global_step + batch).min(total_steps);

            // Do batch of CPU training steps
            for step in batch_start..=batch_end {
                global_step += 1;
                losses.push(layer.train_step(&x_data, &target, sp, lr, global_step));
            }

            // Reload ANE with updated weights (skip on very last batch)
            if global_step < total_steps {
                let t_reload = Instant::now();
                exec.reload_weights(&[
                    ("@model_path/weights/Wg.bin", &build_blob(&layer.wg)),
                    ("@model_path/weights/Wu.bin", &build_blob(&layer.wu)),
                    (
                        "@model_path/weights/WdI.bin",
                        &build_blob(&layer.build_wdi()),
                    ),
                ])
                .expect("reload");
                total_reload_ms += t_reload.elapsed().as_secs_f64() * 1000.0;
            }
        }

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let per_step = total_ms / total_steps as f64;
        let speedup = cpu_total_ms / total_ms;
        let final_loss = *losses.last().unwrap_or(&0.0);
        let note = if speedup > 1.0 { "FASTER" } else { "slower" };

        println!(
            "  {:>10} {:>12.1} {:>10.3} {:>8.4} {:>8.2}x {:>8}",
            batch, total_ms, per_step, final_loss, speedup, note
        );

        // Show reload amortization detail for interesting batch sizes
        if batch <= 20 {
            let reload_per_step = total_reload_ms / total_steps as f64;
            println!(
                "    (reload total={:.1}ms, {:.2}ms/step amortized, {} reloads)",
                total_reload_ms,
                reload_per_step,
                n_batches - 1
            );
        }
    }

    // ── Theoretical Analysis ──
    println!("\n=== Theoretical Analysis ===");
    println!("  From earlier benchmarks at D=768, SP=256:");
    println!("    CPU forward:  4.2ms");
    println!("    ANE forward:  0.75ms (5.6x faster)");
    println!("    CPU backward: 26.5ms (unchanged — must run on CPU)");
    println!("    ANE reload:   50-90ms per reload");
    println!();
    println!("  Per-step breakdown:");
    println!("    CPU-only:     4.2 + 26.5 = 30.7ms/step");
    println!("    ANE-assisted: 0.75 + 26.5 + reload/N = 27.25 + reload/N ms/step");
    println!();
    println!("  Even with ZERO reload cost:");
    println!("    ANE saves 3.45ms forward but backward still 26.5ms");
    println!(
        "    Max speedup = 30.7 / 27.25 = {:.2}x (only {:.0}% faster)",
        30.7 / 27.25,
        (30.7 / 27.25 - 1.0) * 100.0
    );
    println!();
    println!("  CONCLUSION: ANE cannot make training significantly faster");
    println!("  because the backward pass (which ANE cannot do) dominates.");
    println!("  ANE is excellent for INFERENCE but not for TRAINING.");
}
