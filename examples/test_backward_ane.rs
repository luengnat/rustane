//! ANE Backward Pass — ANE does matmuls, CPU does element-wise.
//!
//! Strategy: ANE excels at matmuls (conv1x1) but fails with too many ops.
//! Split: ANE handles Wd^T@dy and WgT@dgate+WuT@dup, CPU handles SiLU' etc.
//!
//! Program A (ANE): dfused = Wd^T @ dy
//!   Input: dy [D, SP]
//!   Output: dfused [inter, SP]
//!
//! CPU: Compute SiLU'(gate), dsilu = dfused*up, dup = dfused*silu, dgate = dsilu*SiLU'
//!
//! Program B (ANE): dx_partial = [WgT|WuT] @ concat(dgate, dup)
//!   Input: dgate [inter, SP], dup [inter, SP]
//!   Output: dx_partial [D, SP]
//!
//! CPU: dx = dx_partial + dy
//!
//! Usage: cargo run --release --example test_backward_ane -- [D] [SP]

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
fn to_fp16(d: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; d.len() * 2];
    for (i, &v) in d.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        b[i * 2] = (h & 0xFF) as u8;
        b[i * 2 + 1] = (h >> 8) as u8;
    }
    b
}
fn from_fp16(r: &[u8]) -> Vec<f32> {
    let mut o = vec![0.0f32; r.len() / 2];
    for i in 0..o.len() {
        let h = (r[i * 2] as u16) | ((r[i * 2 + 1] as u16) << 8);
        o[i] = f16::from_bits(h).to_f32();
    }
    o
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

fn cpu_backward_ffn(
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    x: &[f32],
    gate: &[f32],
    up: &[f32],
    fused: &[f32],
    dy: &[f32],
    d: usize,
    inter: usize,
    sp: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dfused = mm_at(wd, d, inter, dy, sp);
    let silu: Vec<f32> = gate
        .iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let dup: Vec<f32> = dfused
        .iter()
        .zip(silu.iter())
        .map(|(&d, &s)| d * s)
        .collect();
    let dsilu: Vec<f32> = dfused.iter().zip(up.iter()).map(|(&d, &u)| d * u).collect();
    let dgate: Vec<f32> = gate
        .iter()
        .zip(dsilu.iter())
        .map(|(&g, &ds)| {
            let s = 1.0 / (1.0 + (-g).exp());
            ds * s * (1.0 + g * (1.0 - s))
        })
        .collect();
    let dx_g = mm_at(wg, inter, d, &dgate, sp);
    let dx_u = mm_at(wu, inter, d, &dup, sp);
    let mut dx = vec![0.0f32; d * sp];
    for i in 0..dx.len() {
        dx[i] = dx_g[i] + dx_u[i] + dy[i];
    }
    (dx, dgate, dup)
}

fn mil_wdt_conv(d: usize, inter: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n");
    m.push_str("    func main<ios16>(\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dy\n");
    m.push_str("    ) {\n");
    // Wd^T weight: conv1x1 weight is [out_channels, in_channels, 1, 1]
    // out_channels = inter, in_channels = d
    m.push_str("        tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> WdT = const()[name = tensor<string, []>(\"WdT\"), val = tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WdT.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // dfused = Wd^T @ dy
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dfused = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdT, x = dy);\n");
    m.push_str("    } -> (dfused);\n}\n");
    m
}

fn mil_dx_partial(d: usize, inter: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}})]\n{\n");
    m.push_str("    func main<ios16>(\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dgate,\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dup\n");
    m.push_str("    ) {\n");
    // RW [D, 2*inter, 1, 1]
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&(2 * inter).to_string());
    m.push_str(", 1, 1]> RW = const()[name = tensor<string, []>(\"RW\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&(2 * inter).to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/RW.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&(2 * inter).to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (dgate, dup), axis = ax, interleave = ci);\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dx = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = RW, x = cat);\n");
    m.push_str("    } -> (dx);\n}\n");
    m
}

fn compare(ane: &[f32], cpu: &[f32], label: &str) {
    let n = ane.len().min(cpu.len());
    if n == 0 {
        println!("  {:20} EMPTY", label);
        return;
    }
    let mut max_diff = 0.0_f32;
    let mut sum_rel = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let diff = (ane[i] - cpu[i]).abs();
        let denom = cpu[i].abs().max(1e-6_f32);
        let rel = (diff / denom) as f64;
        max_diff = max_diff.max(diff);
        sum_rel += rel;
        max_rel = max_rel.max(rel);
    }
    let avg_rel = sum_rel / n as f64;
    let tag = if avg_rel < 0.01 {
        "EXCELLENT"
    } else if avg_rel < 0.05 {
        "GOOD"
    } else if avg_rel < 0.10 {
        "OK"
    } else {
        "POOR"
    };
    println!(
        "  {:20} max_diff={:.6} avg_rel={:.4}% max_rel={:.1}%  {}",
        label,
        max_diff,
        avg_rel * 100.0,
        max_rel * 100.0,
        tag
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let inter = d * 4;

    println!("============================================================");
    println!("  ANE Backward: Two-Program Split Approach");
    println!("  D={} SP={} inter={}", d, sp, inter);
    println!("============================================================\n");

    let wg = rand_m(inter, d, 0.02, 42);
    let wu = rand_m(inter, d, 0.02, 100);
    let wd = rand_m(d, inter, 0.02, 200);
    let x: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * 0.1).collect();
    let gate = mm(&wg, inter, d, &x, sp);
    let up = mm(&wu, inter, d, &x, sp);
    let silu: Vec<f32> = gate
        .iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
    let down = mm(&wd, d, inter, &fused, sp);
    let mut y = vec![0.0f32; d * sp];
    for i in 0..y.len() {
        y[i] = down[i] + x[i];
    }
    let target = rand_m(d, d, 0.01, 999);
    let target_out = mm(&target, d, d, &x, sp);
    let n = (d * sp) as f32;
    let mut y = vec![0.0f32; d * sp];
    for i in 0..y.len() {
        y[i] = down[i] + x[i];
    }
    // Use synthetic dy with reasonable magnitude (not divided by n)
    let mut dy = vec![0.0f32; y.len()];
    for i in 0..y.len() {
        dy[i] = 2.0 * (y[i] - target_out[i]);
    }

    let (cpu_dx, cpu_dgate, cpu_dup) =
        cpu_backward_ffn(&wg, &wu, &wd, &x, &gate, &up, &fused, &dy, d, inter, sp);
    println!(
        "  CPU dL/dx norm: {:.4}, dgate norm: {:.4}",
        cpu_dx.iter().map(|v| v * v).sum::<f32>().sqrt(),
        cpu_dgate.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    // ── Build ANE weights ──
    // WdT: for conv1x1 weight [inter, d, 1, 1] (out=inter, in=d)
    // blob[i*d + j] = WdT[i][j] = wd[j*inter + i]  (wd is [d x inter])
    let mut wdt = vec![0.0f32; inter * d];
    for i in 0..inter {
        for j in 0..d {
            wdt[i * d + j] = wd[j * inter + i];
        }
    }
    // RW = [WgT | WuT] concatenated along columns: [d x 2*inter]
    let mut wgt = vec![0.0f32; d * inter];
    for r in 0..d {
        for c in 0..inter {
            wgt[r * inter + c] = wg[c * d + r];
        }
    }
    let mut wut = vec![0.0f32; d * inter];
    for r in 0..d {
        for c in 0..inter {
            wut[r * inter + c] = wu[c * d + r];
        }
    }
    let mut rw = vec![0.0f32; d * 2 * inter];
    for r in 0..d {
        for c in 0..inter {
            rw[r * (2 * inter) + c] = wgt[r * inter + c];
        }
    }
    for r in 0..d {
        for c in 0..inter {
            rw[r * (2 * inter) + inter + c] = wut[r * inter + c];
        }
    }

    // ── Compile Program A: dfused = Wd^T @ dy ──
    println!("=== Program A: Wd^T @ dy ===");
    let mil_a = mil_wdt_conv(d, inter, sp);
    let t_a = Instant::now();
    let blob_wdt = build_blob(&wdt);
    let mut exec_a = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil_a,
        &["@model_path/weights/WdT.bin"],
        &[&blob_wdt[..]],
        &[blob_wdt.len()],
        &[d * sp * 2],
        &[inter * sp * 2],
    ) {
        Ok(e) => e,
        Err(e) => {
            println!("  COMPILE FAILED: {:?}", e);
            return;
        }
    };
    println!("  Compile: {:.1}ms", t_a.elapsed().as_secs_f64() * 1000.0);

    // ── Compile Program B: dx_partial = [WgT|WuT] @ concat(dgate, dup) ──
    println!("\n=== Program B: [WgT|WuT] @ concat(dgate, dup) ===");
    let mil_b = mil_dx_partial(d, inter, sp);
    let t_b = Instant::now();
    let blob_rw = build_blob(&rw);
    let mut exec_b = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil_b,
        &["@model_path/weights/RW.bin"],
        &[&blob_rw[..]],
        &[blob_rw.len()],
        &[inter * sp * 2, inter * sp * 2],
        &[d * sp * 2],
    ) {
        Ok(e) => e,
        Err(e) => {
            println!("  COMPILE FAILED: {:?}", e);
            return;
        }
    };
    println!("  Compile: {:.1}ms", t_b.elapsed().as_secs_f64() * 1000.0);

    // Warmup
    exec_a.write_input(0, &to_fp16(&dy)).expect("w");
    exec_a.eval().expect("e");
    exec_a.read_output_vec(0).expect("r");
    exec_b.write_input(0, &to_fp16(&cpu_dgate[..])).expect("w");
    exec_b.write_input(1, &to_fp16(&cpu_dup[..])).expect("w");
    exec_b.eval().expect("e");
    exec_b.read_output_vec(0).expect("r");
    println!("  Warmup OK");

    // ── Run accuracy test ──
    println!("\n=== Accuracy ===");
    let dy16 = to_fp16(&dy);

    // Step 1: ANE computes dfused = Wd^T @ dy
    exec_a.write_input(0, &dy16).expect("w");
    exec_a.eval().expect("e");
    let ane_dfused = from_fp16(&exec_a.read_output_vec(0).expect("r"));
    // CPU reference: dfused = Wd^T @ dy
    let cpu_dfused = mm_at(&wd, d, inter, &dy, sp);
    compare(&ane_dfused, &cpu_dfused, "ANE dfused");

    // Step 2: CPU computes dgate and dup from dfused
    let silu: Vec<f32> = gate
        .iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let dsilu: Vec<f32> = ane_dfused
        .iter()
        .zip(up.iter())
        .map(|(&d, &u)| d * u)
        .collect();
    let dup: Vec<f32> = ane_dfused
        .iter()
        .zip(silu.iter())
        .map(|(&d, &s)| d * s)
        .collect();
    let dgate: Vec<f32> = gate
        .iter()
        .zip(dsilu.iter())
        .map(|(&g, &ds)| {
            let s = 1.0 / (1.0 + (-g).exp());
            ds * s * (1.0 + g * (1.0 - s))
        })
        .collect();
    compare(&dgate, &cpu_dgate, "ANE-based dgate");
    compare(&dup, &cpu_dup, "ANE-based dup");

    // Step 3: ANE computes dx_partial = [WgT|WuT] @ concat(dgate, dup)
    let dgate16 = to_fp16(&dgate);
    let dup16 = to_fp16(&dup);
    exec_b.write_input(0, &dgate16).expect("w");
    exec_b.write_input(1, &dup16).expect("w");
    exec_b.eval().expect("e");
    let ane_dx_partial = from_fp16(&exec_b.read_output_vec(0).expect("r"));

    // Step 4: CPU adds dy
    let mut ane_dx = vec![0.0f32; d * sp];
    for i in 0..ane_dx.len() {
        ane_dx[i] = ane_dx_partial[i] + dy[i];
    }
    compare(&ane_dx, &cpu_dx, "ANE dL/dx");

    // ── Timing ──
    println!("\n=== Timing ===");
    let n_runs = 20;
    let mut cpu_full_t = 0.0_f64;
    let mut cpu_matmul_t = 0.0_f64;
    for _ in 0..n_runs {
        let t = Instant::now();
        let _ = cpu_backward_ffn(&wg, &wu, &wd, &x, &gate, &up, &fused, &dy, d, inter, sp);
        cpu_full_t += t.elapsed().as_secs_f64() * 1000.0;
    }
    for _ in 0..n_runs {
        let t = Instant::now();
        // Just the 3 matmuls: Wd^T@dy, WgT@dgate, WuT@dup
        let _df = mm_at(&wd, d, inter, &dy, sp);
        let _dg = mm_at(&wg, inter, d, &cpu_dgate, sp);
        let _du = mm_at(&wu, inter, d, &cpu_dup, sp);
        cpu_matmul_t += t.elapsed().as_secs_f64() * 1000.0;
    }
    cpu_full_t /= n_runs as f64;
    cpu_matmul_t /= n_runs as f64;

    let mut ane_a_t = 0.0_f64;
    let mut ane_b_t = 0.0_f64;
    for _ in 0..n_runs {
        exec_a.write_input(0, &dy16).expect("w");
        let t = Instant::now();
        exec_a.eval().expect("e");
        exec_a.read_output_vec(0).expect("r");
        ane_a_t += t.elapsed().as_secs_f64() * 1000.0;

        exec_b.write_input(0, &dgate16).expect("w");
        exec_b.write_input(1, &dup16).expect("w");
        let t = Instant::now();
        exec_b.eval().expect("e");
        exec_b.read_output_vec(0).expect("r");
        ane_b_t += t.elapsed().as_secs_f64() * 1000.0;
    }
    ane_a_t /= n_runs as f64;
    ane_b_t /= n_runs as f64;
    let ane_matmul_total = ane_a_t + ane_b_t;

    // CPU element-wise time (everything except the 3 matmuls)
    let cpu_elem_t = cpu_full_t - cpu_matmul_t;
    // Hybrid: ANE matmuls + CPU element-wise
    let hybrid_t = ane_matmul_total + cpu_elem_t;

    println!("  {:30} {:.2}ms", "CPU backward (full):", cpu_full_t);
    println!("  {:30} {:.2}ms", "  CPU matmuls only:", cpu_matmul_t);
    println!("  {:30} {:.2}ms", "  CPU element-wise only:", cpu_elem_t);
    println!(
        "  {:30} {:.2}ms  (Wd^T: {:.2}ms, RW: {:.2}ms)",
        "ANE matmuls:", ane_matmul_total, ane_a_t, ane_b_t
    );
    println!(
        "  {:30} {:.2}ms  ({:.1}x vs CPU matmul)",
        "ANE matmul speedup:",
        ane_matmul_total,
        cpu_matmul_t / ane_matmul_total
    );
    println!(
        "  {:30} {:.2}ms  ({:.1}x vs CPU full)",
        "Hybrid (ANE+CPU elem):",
        hybrid_t,
        cpu_full_t / hybrid_t
    );
}
