//! Test: Can ANE handle 8-op SiLU' + dgate computation?
//!
//! Previous failure was likely due to wrong WdT weight shape [d,inter] instead of [inter,d].
//! The inference_pipeline works with 8 ops, so this should too if shapes are correct.
//!
//! SiLU'(g) = σ(g)(1 + g - g·σ(g)) = conv1x1([1,1,-1], concat(σ(g), g·σ(g), g·σ(g)·σ(g)))
//!
//! Usage: cargo run --release --example test_silu_prime_program -- [D] [SP]

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
const NT: i32 = 111;
const TR: i32 = 112;

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

/// 8-op SiLU' program: dfused + SiLU' + dgate all in one ANE program
/// This FAILED before due to wrong WdT weight shape [d,inter] instead of [inter,d]
fn mil_silu_prime_full(d: usize, inter: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n");
    m.push_str("    func main<ios16>(\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dy,\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> gate,\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> up\n");
    m.push_str("    ) {\n");
    // WdT: [inter, d, 1, 1] — CORRECT shape (out=inter, in=d)
    m.push_str("        tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> WdT = const()[name = tensor<string, []>(\"WdT\"), val = tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WdT.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // SPW: [inter, 3*inter, 1, 1] — for SiLU' via conv1x1
    m.push_str("        tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&(3 * inter).to_string());
    m.push_str(", 1, 1]> SPW = const()[name = tensor<string, []>(\"SPW\"), val = tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&(3 * inter).to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/SPW.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    // conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // Op 1: dfused = Wd^T @ dy
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dfused = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdT, x = dy);\n");
    // Op 2: sig = sigmoid(gate)
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> sig = sigmoid(x = gate);\n");
    // Op 3: silu = gate * sig
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> sx = mul(x = gate, y = sig);\n");
    // Op 4: sxs = sx * sig = gate*sig*sig
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> sxs = mul(x = sx, y = sig);\n");
    // Op 5: concat(sig, sx, sxs)
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&(3 * inter).to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> s3 = concat(values = (sig, sx, sxs), axis = ax, interleave = ci);\n");
    // Op 6: silu_prime = conv1x1(SPW, s3) = SiLU'(gate)
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> silu_prime = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = SPW, x = s3);\n");
    // Op 7: dsilu = dfused * up
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dsilu = mul(x = dfused, y = up);\n");
    // Op 8: dgate = dsilu * silu_prime
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dgate = mul(x = dsilu, y = silu_prime);\n");
    m.push_str("    } -> (dgate);\n}\n");
    m
}

/// 10-op version: SiLU' + dgate + dup (adds dup = dfused * silu)
fn mil_silu_prime_with_dup(d: usize, inter: usize, sp: usize) -> String {
    let mut m = mil_silu_prime_full(d, inter, sp);
    // Add dup = dfused * sx (before the closing brace)
    // Find the last line before "} -> (dgate);" and insert dup
    let close = "    } -> (dgate);\n}\n";
    let dup_code = format!(
        "        tensor<fp16, [1, {inter}, 1, {sp}]> dup = mul(x = dfused, y = sx);\n",
        inter = inter,
        sp = sp
    );
    // Replace closing with dup + dual output
    m = m.replace(
        close,
        &format!(
            "{}\n    {} -> (dgate, dup);\n}}\n",
            dup_code,
            close.replace("dgate", "dgate, dup").trim()
        ),
    );
    m
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let inter = d * 4;

    println!("============================================================");
    println!("  SiLU' ANE Program Test");
    println!("  D={} SP={} inter={}", d, sp, inter);
    println!("============================================================\n");

    // Generate data
    let wg = rand_m(inter, d, 0.02, 42);
    let wu = rand_m(inter, d, 0.02, 100);
    let wd = rand_m(d, inter, 0.02, 200);
    let x: Vec<f32> = rand_m(d, sp, 0.1, 9999);
    let gate = {
        let mut g = mm(&wg, inter, d, &x, sp);
        // Clamp to avoid fp16 overflow
        for v in g.iter_mut() {
            *v = v.max(-200.0).min(200.0);
        }
        g
    };
    let up = {
        let mut u = mm(&wu, inter, d, &x, sp);
        for v in u.iter_mut() {
            *v = v.max(-200.0).min(200.0);
        }
        u
    };
    let target = rand_m(d, d, 0.01, 12345);
    let target_out = mm(&target, d, d, &x, sp);
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
    let dy: Vec<f32> = y
        .iter()
        .zip(target_out.iter())
        .map(|(&a, &b)| 2.0 * (a - b))
        .collect();

    // CPU reference
    let cpu_dfused = mm_at(&wd, d, inter, &dy, sp);
    let silu: Vec<f32> = gate
        .iter()
        .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let dup: Vec<f32> = cpu_dfused
        .iter()
        .zip(silu.iter())
        .map(|(&d, &s)| d * s)
        .collect();
    let dsilu: Vec<f32> = cpu_dfused
        .iter()
        .zip(up.iter())
        .map(|(&d, &u)| d * u)
        .collect();
    let dgate: Vec<f32> = gate
        .iter()
        .zip(dsilu.iter())
        .map(|(&g, &ds)| {
            let s = 1.0 / (1.0 + (-g).exp());
            ds * s * (1.0 + g * (1.0 - s))
        })
        .collect();

    println!(
        "  CPU dgate norm: {:.4}, dup norm: {:.4}",
        dgate.iter().map(|v| v * v).sum::<f32>().sqrt(),
        dup.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    // ── Test 1: 8-op SiLU' program (dgate only) ──
    println!("\n=== Test 1: 8-op SiLU' + dgate ===");
    let mil_8op = mil_silu_prime_full(d, inter, sp);

    // Build weights
    let mut wdt = vec![0.0f32; inter * d];
    for i in 0..inter {
        for j in 0..d {
            wdt[i * d + j] = wd[j * inter + i];
        }
    }
    let mut spw = vec![0.0f32; inter * 3 * inter];
    for i in 0..inter {
        for j in 0..inter {
            spw[i * 3 * inter + j] = 1.0;
        } // σ(g) coeff
        for j in 0..inter {
            spw[i * 3 * inter + inter + j] = 1.0;
        } // g·σ(g) coeff
        for j in 0..inter {
            spw[i * 3 * inter + 2 * inter + j] = -1.0;
        } // -g·σ(g)·σ(g) coeff
    }

    println!("  SPW blob size: {:.1} MB", spw.len() as f64 / 1e6);
    println!("  WdT blob size: {:.1} KB", wdt.len() as f64 * 4.0 / 1024.0);

    let blob_wdt = build_blob(&wdt);
    let blob_spw = build_blob(&spw);

    let t = Instant::now();
    match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil_8op,
        &["@model_path/weights/WdT.bin", "@model_path/weights/SPW.bin"],
        &[&blob_wdt[..], &blob_spw[..]],
        &[blob_wdt.len(), blob_spw.len()],
        &[d * sp * 2, inter * sp * 2, inter * sp * 2],
        &[inter * sp * 2],
    ) {
        Ok(mut exec) => {
            println!("  COMPILE OK: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
            // Run
            exec.write_input(0, &to_fp16(&dy)).expect("w");
            exec.write_input(1, &to_fp16(&gate)).expect("w");
            exec.write_input(2, &to_fp16(&up)).expect("w");
            exec.eval().expect("e");
            let ane_dgate = from_fp16(&exec.read_output_vec(0).expect("r"));
            compare(&ane_dgate, &dgate, "ANE dgate (8-op)");

            // Timing
            let mut times = Vec::new();
            for _ in 0..20 {
                exec.write_input(0, &to_fp16(&dy)).expect("w");
                exec.write_input(1, &to_fp16(&gate)).expect("w");
                exec.write_input(2, &to_fp16(&up)).expect("w");
                let t = Instant::now();
                exec.eval().expect("e");
                exec.read_output_vec(0).expect("r");
                times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let avg = times.iter().sum::<f64>() / 20.0;
            println!("  Avg eval+read: {:.2}ms", avg);
        }
        Err(e) => {
            println!("  COMPILE FAILED: {:?}", e);
            println!("  (SPW weight may be too large for ANE at this dimension)");
        }
    }

    // ── Test 2: Check SPW weight size feasibility ──
    println!("\n=== Weight Size Analysis ===");
    let spw_elements = inter * 3 * inter;
    let spw_bytes = spw_elements * 2;
    println!(
        "  SPW: {} elements = {:.1} MB fp16",
        spw_elements,
        spw_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  WdT: {} elements = {:.1} KB fp16",
        inter * d,
        (inter * d * 2) as f64 / 1024.0
    );

    // Compare with known-working weight sizes
    let wdi_size = d * (inter + d) * 2;
    let rw_size = d * 2 * inter * 2;
    println!(
        "  WdI (works): {:.1} MB",
        wdi_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  RW (works):   {:.1} MB",
        rw_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  SPW:         {:.1} MB ({:.1}x WdI)",
        spw_bytes as f64 / (1024.0 * 1024.0),
        spw_bytes as f64 / wdi_size as f64
    );
}
