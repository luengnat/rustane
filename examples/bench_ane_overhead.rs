//! Measure ANE program call overhead (write + eval + read) vs actual compute
//! to understand why backward hybrid doesn't beat CPU.

use half::f16;
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

fn mil_header() -> &'static str {
    "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
}

/// Simple single conv1x1: W @ x → y
fn mil_conv(oc: usize, ic: usize, sp: usize, wname: &str) -> String {
    let mut m = String::new();
    m.push_str(mil_header());
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    m.push_str("        tensor<fp16, [");
    m.push_str(&oc.to_string());
    m.push_str(", ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [");
    m.push_str(&oc.to_string());
    m.push_str(", ");
    m.push_str(&ic.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/W.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&oc.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x);\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn main() {
    let d: usize = 768;
    let inter = d * 4; // 3072
    let sp: usize = 256;

    println!("=== ANE Overhead Micro-Benchmark ===");
    println!("  D={} inter={} SP={}", d, inter, sp);

    // ── Test 1: Wd^T @ dy (backward A) — conv1x1 [inter, d] @ [d, sp] ──
    println!(
        "\n--- Test 1: Wd^T @ dy  (shape: [{}, {}] @ [{}, {}]) ---",
        inter, d, d, sp
    );

    // Create weight: WdT = wd^T, shape [inter, d]
    let mut wdt = vec![0.0f32; inter * d];
    for i in 0..inter {
        for j in 0..d {
            wdt[i * d + j] = ((i * d + j) as u32 as f32 * 0.001).sin();
        }
    }
    let blob_wdt = build_blob(&wdt);

    let mil_a = mil_conv(inter, d, sp, "WdT");
    let mut exec_a = rustane::wrapper::ANECompiler::new()
        .compile_multi(
            &mil_a,
            &["@model_path/weights/W.bin"],
            &[&blob_wdt[..]],
            &[blob_wdt.len()],
            &[d * sp * 2],
            &[inter * sp * 2],
        )
        .expect("compile a");

    // Input: dy [d, sp]
    let dy: Vec<f32> = (0..d * sp).map(|i| (i as f32 * 0.01).cos()).collect();
    let dy16 = to_fp16(&dy);

    // Warmup
    for _ in 0..3 {
        exec_a.write_input(0, &dy16).unwrap();
        exec_a.eval().unwrap();
        let _ = exec_a.read_output_vec(0).unwrap();
    }

    // Measure: full call (write + eval + read + convert)
    let n_iter = 50;
    let mut total_full = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        exec_a.write_input(0, &dy16).unwrap();
        exec_a.eval().unwrap();
        let raw = exec_a.read_output_vec(0).unwrap();
        let _result = from_fp16(&raw);
        total_full += t.elapsed().as_secs_f64() * 1000.0;
    }

    // Measure: eval only
    exec_a.write_input(0, &dy16).unwrap();
    let mut total_eval = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        exec_a.eval().unwrap();
        total_eval += t.elapsed().as_secs_f64() * 1000.0;
    }
    let _ = exec_a.read_output_vec(0).unwrap(); // drain

    // Measure: write_input only
    let mut total_write = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        exec_a.write_input(0, &dy16).unwrap();
        total_write += t.elapsed().as_secs_f64() * 1000.0;
    }

    // Measure: read_output only
    let mut total_read = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        let _ = exec_a.read_output_vec(0).unwrap();
        total_read += t.elapsed().as_secs_f64() * 1000.0;
    }

    // Measure: fp16 conversion
    let mut total_to_fp16 = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        let _ = to_fp16(&dy);
        total_to_fp16 += t.elapsed().as_secs_f64() * 1000.0;
    }
    let out_raw = exec_a.read_output_vec(0).unwrap();
    let mut total_from_fp16 = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        let _ = from_fp16(&out_raw);
        total_from_fp16 += t.elapsed().as_secs_f64() * 1000.0;
    }

    // CPU baseline
    let mut total_cpu = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        let _ = mm_at(&wdt, d, inter, &dy, sp);
        total_cpu += t.elapsed().as_secs_f64() * 1000.0;
    }

    println!("\n  Per-call breakdown ({} iterations):", n_iter);
    println!(
        "  {:30} {:>10}ms",
        "write_input (d*sp fp16)",
        format!("{:.3}", total_write / n_iter as f64)
    );
    println!(
        "  {:30} {:>10}ms",
        "eval (ANE compute)",
        format!("{:.3}", total_eval / n_iter as f64)
    );
    println!(
        "  {:30} {:>10}ms",
        "read_output (inter*sp fp16)",
        format!("{:.3}", total_read / n_iter as f64)
    );
    println!(
        "  {:30} {:>10}ms",
        "to_fp16 (d*sp)",
        format!("{:.3}", total_to_fp16 / n_iter as f64)
    );
    println!(
        "  {:30} {:>10}ms",
        "from_fp16 (inter*sp)",
        format!("{:.3}", total_from_fp16 / n_iter as f64)
    );
    println!("  {:30} {:>10}ms", "───", "───");
    println!(
        "  {:30} {:>10}ms",
        "Total ANE (full call)",
        format!("{:.3}", total_full / n_iter as f64)
    );
    println!(
        "  {:30} {:>10}ms",
        "CPU cblas_sgemm",
        format!("{:.3}", total_cpu / n_iter as f64)
    );
    let ane_time = total_full / n_iter as f64;
    let cpu_time = total_cpu / n_iter as f64;
    println!(
        "  {:30} {:>10}",
        "ANE vs CPU",
        if ane_time < cpu_time {
            format!("{:.2}x FASTER", cpu_time / ane_time)
        } else {
            format!("{:.2}x SLOWER", cpu_time / ane_time)
        }
    );

    // ── Test 2: Wg^T @ dgate (backward B, single matmul) ──
    println!(
        "\n--- Test 2: Wg^T @ dgate  (shape: [{}, {}] @ [{}, {}]) ---",
        d, inter, inter, sp
    );

    let mut wgt = vec![0.0f32; d * inter];
    for i in 0..d {
        for j in 0..inter {
            wgt[i * inter + j] = ((i * inter + j) as u32 as f32 * 0.001).cos();
        }
    }
    let blob_wgt = build_blob(&wgt);

    let mil_b = mil_conv(d, inter, sp, "WgT");
    let mut exec_b = rustane::wrapper::ANECompiler::new()
        .compile_multi(
            &mil_b,
            &["@model_path/weights/W.bin"],
            &[&blob_wgt[..]],
            &[blob_wgt.len()],
            &[inter * sp * 2],
            &[d * sp * 2],
        )
        .expect("compile b");

    let dgate: Vec<f32> = (0..inter * sp).map(|i| (i as f32 * 0.01).sin()).collect();
    let dgate16 = to_fp16(&dgate);

    // Warmup
    for _ in 0..3 {
        exec_b.write_input(0, &dgate16).unwrap();
        exec_b.eval().unwrap();
        let _ = exec_b.read_output_vec(0).unwrap();
    }

    let mut total_full_b = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        exec_b.write_input(0, &dgate16).unwrap();
        exec_b.eval().unwrap();
        let raw = exec_b.read_output_vec(0).unwrap();
        let _result = from_fp16(&raw);
        total_full_b += t.elapsed().as_secs_f64() * 1000.0;
    }

    // CPU baseline for Wg^T @ dgate
    let mut total_cpu_b = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        let _ = mm_at(&wgt, inter, d, &dgate, sp);
        total_cpu_b += t.elapsed().as_secs_f64() * 1000.0;
    }

    let ane_time_b = total_full_b / n_iter as f64;
    let cpu_time_b = total_cpu_b / n_iter as f64;
    println!(
        "  {:30} {:>10}ms",
        "Total ANE (full call)",
        format!("{:.3}", ane_time_b)
    );
    println!(
        "  {:30} {:>10}ms",
        "CPU cblas_sgemm",
        format!("{:.3}", cpu_time_b)
    );
    println!(
        "  {:30} {:>10}",
        "ANE vs CPU",
        if ane_time_b < cpu_time_b {
            format!("{:.2}x FASTER", cpu_time_b / ane_time_b)
        } else {
            format!("{:.2}x SLOWER", cpu_time_b / ane_time_b)
        }
    );

    // ── Summary ──
    println!("\n=== Backward Pass Analysis (single layer) ===");
    println!("  Backward has 2 ANE calls + CPU element-wise + 3 CPU BLAS calls");
    println!(
        "  ANE call 1 (Wd^T @ dy):    {:.3}ms (saves vs CPU: {:.3}ms)",
        ane_time,
        cpu_time - ane_time
    );
    println!(
        "  ANE call 2 (Wg^T @ dgate):  {:.3}ms (saves vs CPU: {:.3}ms)",
        ane_time_b,
        cpu_time_b - ane_time_b
    );
    let total_ane_saving = (cpu_time - ane_time) + (cpu_time_b - ane_time_b);
    println!("  Total ANE saving:          {:.3}ms", total_ane_saving);
    println!(
        "  Total ANE overhead:        {:.3}ms (2 × ~{:.3}ms fixed overhead)",
        ane_time + ane_time_b,
        (ane_time + ane_time_b) / 2.0
    );
    if total_ane_saving > 0.0 {
        println!(
            "  NET BENEFIT:                {:.3}ms per layer",
            total_ane_saving
        );
    } else {
        println!(
            "  NET LOSS:                   {:.3}ms per layer (ANE backward not worthwhile)",
            -total_ane_saving
        );
    }

    // Also measure the 2-input concat program overhead
    println!("\n--- Test 3: 2-input program overhead (concat+conv) ---");
    let mut rw = vec![0.0f32; d * 2 * inter];
    for i in 0..d * 2 * inter {
        rw[i] = ((i as f32) * 0.001).sin();
    }
    let blob_rw = build_blob(&rw);

    // Build a 2-input program
    let mut mil_c = String::new();
    mil_c.push_str(mil_header());
    mil_c.push_str("    func main<ios16>(\n");
    mil_c.push_str("        tensor<fp16, [1, ");
    mil_c.push_str(&inter.to_string());
    mil_c.push_str(", 1, ");
    mil_c.push_str(&sp.to_string());
    mil_c.push_str("]> a,\n");
    mil_c.push_str("        tensor<fp16, [1, ");
    mil_c.push_str(&inter.to_string());
    mil_c.push_str(", 1, ");
    mil_c.push_str(&sp.to_string());
    mil_c.push_str("]> b\n");
    mil_c.push_str("    ) {\n");
    mil_c.push_str("        tensor<fp16, [");
    mil_c.push_str(&d.to_string());
    mil_c.push_str(", ");
    mil_c.push_str(&(2 * inter).to_string());
    mil_c.push_str(", 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [");
    mil_c.push_str(&d.to_string());
    mil_c.push_str(", ");
    mil_c.push_str(&(2 * inter).to_string());
    mil_c.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/W.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    mil_c.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    mil_c.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil_c.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil_c.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil_c.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    mil_c.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    mil_c.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    mil_c.push_str("        tensor<fp16, [1, ");
    mil_c.push_str(&(2 * inter).to_string());
    mil_c.push_str(", 1, ");
    mil_c.push_str(&sp.to_string());
    mil_c.push_str("]> cat = concat(values = (a, b), axis = ax, interleave = ci);\n");
    mil_c.push_str("        tensor<fp16, [1, ");
    mil_c.push_str(&d.to_string());
    mil_c.push_str(", 1, ");
    mil_c.push_str(&sp.to_string());
    mil_c.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = cat);\n");
    mil_c.push_str("    } -> (y);\n}\n");

    let mut exec_c = rustane::wrapper::ANECompiler::new()
        .compile_multi(
            &mil_c,
            &["@model_path/weights/W.bin"],
            &[&blob_rw[..]],
            &[blob_rw.len()],
            &[inter * sp * 2, inter * sp * 2],
            &[d * sp * 2],
        )
        .expect("compile c");

    // Warmup
    for _ in 0..3 {
        exec_c.write_input(0, &dgate16).unwrap();
        exec_c.write_input(1, &dgate16).unwrap();
        exec_c.eval().unwrap();
        let _ = exec_c.read_output_vec(0).unwrap();
    }

    let mut total_full_c = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        exec_c.write_input(0, &dgate16).unwrap();
        exec_c.write_input(1, &dgate16).unwrap();
        exec_c.eval().unwrap();
        let raw = exec_c.read_output_vec(0).unwrap();
        let _result = from_fp16(&raw);
        total_full_c += t.elapsed().as_secs_f64() * 1000.0;
    }

    // CPU baseline for [WgT|WuT] @ concat(dgate, dup) = WgT@dgate + WuT@dup
    let mut total_cpu_c = 0.0_f64;
    for _ in 0..n_iter {
        let t = Instant::now();
        // Equivalent CPU: two matmuls
        let _r1 = mm_at(&wgt, inter, d, &dgate, sp);
        let _r2 = mm_at(&wgt, inter, d, &dgate, sp);
        total_cpu_c += t.elapsed().as_secs_f64() * 1000.0;
    }

    let ane_time_c = total_full_c / n_iter as f64;
    let cpu_time_c = total_cpu_c / n_iter as f64;
    println!(
        "  {:30} {:>10}ms",
        "Total ANE (2 inputs + concat + conv)",
        format!("{:.3}", ane_time_c)
    );
    println!(
        "  {:30} {:>10}ms",
        "CPU (2x cblas_sgemm)",
        format!("{:.3}", cpu_time_c)
    );
    println!(
        "  {:30} {:>10}",
        "ANE vs CPU",
        if ane_time_c < cpu_time_c {
            format!("{:.2}x FASTER", cpu_time_c / ane_time_c)
        } else {
            format!("{:.2}x SLOWER", cpu_time_c / ane_time_c)
        }
    );

    // Final analysis
    println!("\n=== FINAL ANALYSIS ===");
    println!("  Per backward layer, ANE does 2 program calls:");
    println!(
        "    1. Wd^T @ dy:          ANE={:.3}ms  CPU={:.3}ms  diff={:+.3}ms",
        ane_time,
        cpu_time,
        cpu_time - ane_time
    );
    println!(
        "    2. [WgT|WuT]@cat:      ANE={:.3}ms  CPU={:.3}ms  diff={:+.3}ms",
        ane_time_c,
        cpu_time_c,
        cpu_time_c - ane_time_c
    );
    let net = (cpu_time - ane_time) + (cpu_time_c - ane_time_c);
    println!("    Net ANE benefit:        {:+.3}ms per layer", net);
    if net > 0.0 {
        println!("    → ANE backward IS worthwhile: saves {:.3}ms/layer", net);
    } else {
        println!(
            "    → ANE backward NOT worthwhile: loses {:.3}ms/layer",
            -net
        );
        println!("    → Recommendation: Use ANE forward only, CPU backward");
    }
}
