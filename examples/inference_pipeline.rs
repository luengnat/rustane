//! Multi-Layer ANE Inference Pipeline
//!
//! Chains N FFN layers as separate ANE programs, measuring total inference
//! latency vs CPU BLAS baseline. Proves that ANE inference speedup scales
//! with model depth.
//!
//! Key question: Does the ~60μs per-program overhead dominate at many layers?
//!
//! Usage: cargo run --example inference_pipeline -- [D] [SP] [layers] [runs]

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
// MIL GENERATION: Single FFN layer (SwiGLU + residual)
// ═══════════════════════════════════════════════════════════════

fn mil_ffn_layer(d: usize, inter: usize, sp: usize) -> String {
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
    // gate + up + silu + fused
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
// FFN LAYER (CPU-side, for baseline comparison)
// ═══════════════════════════════════════════════════════════════

struct FFNLayer {
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    d: usize,
    inter: usize,
}
impl FFNLayer {
    fn new(d: usize, inter: usize, seed: u64) -> Self {
        FFNLayer {
            wg: rand_matrix(inter, d, 0.02, seed),
            wu: rand_matrix(inter, d, 0.02, seed + 1000),
            wd: rand_matrix(d, inter, 0.02, seed + 2000),
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
            wdi[r * t + self.inter + r] = 1.0; // identity for residual
        }
        wdi
    }
    fn forward(&self, x: &[f32], sp: usize) -> Vec<f32> {
        let gate = mm(&self.wg, self.inter, self.d, x, sp);
        let up = mm(&self.wu, self.inter, self.d, x, sp);
        // SiLU(gate) * up
        let mut fused = vec![0.0f32; self.inter * sp];
        for i in 0..fused.len() {
            let s = 1.0 / (1.0 + (-gate[i]).exp());
            fused[i] = gate[i] * s * up[i];
        }
        // down projection + residual
        let down = mm(&self.wd, self.d, self.inter, &fused, sp);
        let mut y = vec![0.0f32; self.d * sp];
        for i in 0..y.len() {
            y[i] = down[i] + x[i];
        }
        y
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let num_layers: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let num_runs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
    let inter = d * 4;
    let bytes_per_elem = 2; // fp16

    println!("============================================================");
    println!("  Multi-Layer ANE Inference Pipeline");
    println!(
        "  D={} SP={} inter={} layers={} runs={}",
        d, sp, inter, num_layers, num_runs
    );
    println!("============================================================\n");

    // Create layers
    let layers: Vec<FFNLayer> = (0..num_layers)
        .map(|i| FFNLayer::new(d, inter, 42 + i as u64 * 100))
        .collect();

    // Input data
    let x_data: Vec<f32> = (0..d * sp).map(|i| ((i % d) as f32 + 1.0) * 0.01).collect();
    let x16 = to_fp16(&x_data);

    // Weight stats
    let params_per_layer = inter * d + inter * d + d * inter; // wg + wu + wd
    let total_params = params_per_layer * num_layers;
    println!(
        "  Parameters: {:.1}M per layer, {:.1}M total",
        params_per_layer as f64 / 1e6,
        total_params as f64 / 1e6
    );

    // ── Compile ANE programs ──
    println!("\n=== Compiling {} ANE programs ===", num_layers);
    let mil = mil_ffn_layer(d, inter, sp);
    let names: Vec<String> = ["Wg", "Wu", "WdI"]
        .iter()
        .map(|n| format!("@model_path/weights/{}.bin", n))
        .collect();
    let nrefs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let mut execs = Vec::new();
    let t_compile_total = Instant::now();

    for (i, layer) in layers.iter().enumerate() {
        let t = Instant::now();
        let blobs = vec![
            build_blob(&layer.wg),
            build_blob(&layer.wu),
            build_blob(&layer.build_wdi()),
        ];
        let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
        let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

        match rustane::wrapper::ANECompiler::new().compile_multi(
            &mil,
            &nrefs,
            &brefs,
            &lens,
            &[d * sp * bytes_per_elem],
            &[d * sp * bytes_per_elem],
        ) {
            Ok(exec) => {
                let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
                println!("  Layer {:2}: compiled in {:.1}ms", i + 1, compile_ms);
                execs.push(exec);
            }
            Err(err) => {
                println!("  Layer {:2}: COMPILE FAILED: {:?}", i + 1, err);
                println!("  Stopping at {} layers due to compile failure", i);
                break;
            }
        }
    }

    let actual_layers = execs.len();
    if actual_layers == 0 {
        println!("  ERROR: No programs compiled successfully");
        return;
    }
    let compile_total_ms = t_compile_total.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Total compile: {:.1}ms ({:.1}ms/layer)",
        compile_total_ms,
        compile_total_ms / actual_layers as f64
    );

    // Warmup
    for exec in &mut execs {
        exec.write_input(0, &x16).expect("warmup write");
        exec.eval().expect("warmup eval");
    }

    // ── ANE Inference Benchmark ──
    println!(
        "\n=== ANE Inference ({} layers, {} runs) ===",
        actual_layers, num_runs
    );

    let mut ane_times = Vec::new();
    let mut layer_times: Vec<Vec<f64>> = vec![Vec::new(); actual_layers];

    for run in 0..num_runs {
        let mut current_input = x16.clone();
        let t_total = Instant::now();

        for (i, exec) in execs.iter_mut().enumerate() {
            let t_layer = Instant::now();
            exec.write_input(0, &current_input).expect("write");
            exec.eval().expect("eval");
            let output_raw = exec.read_output_vec(0).expect("read");
            let layer_ms = t_layer.elapsed().as_secs_f64() * 1000.0;
            layer_times[i].push(layer_ms);
            current_input = output_raw;
        }

        ane_times.push(t_total.elapsed().as_secs_f64() * 1000.0);
    }

    // Drop executors to free ANE resources before CPU benchmark
    drop(execs);

    let ane_avg: f64 = ane_times.iter().sum::<f64>() / num_runs as f64;
    let ane_min: f64 = ane_times.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Per-layer breakdown (average over {} runs):", num_runs);
    println!("  {:>8} {:>12} {:>12}", "Layer", "Avg(ms)", "Min(ms)");
    println!("  {:>8} {:>12} {:>12}", "------", "--------", "--------");
    for (i, times) in layer_times.iter().enumerate() {
        let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  {:>8} {:>12.3} {:>12.3}", i + 1, avg, min);
    }
    println!("  {:>8} {:>12.3} {:>12.3}", "TOTAL", ane_avg, ane_min);

    // Verify correctness (compare first ANE output with CPU)
    // Re-create executors for verification
    println!("\n=== Correctness Verification ===");
    let mut cpu_x = x_data.clone();
    let mut cpu_layer_times = Vec::new();

    for (i, layer) in layers.iter().enumerate().take(actual_layers) {
        let t = Instant::now();
        cpu_x = layer.forward(&cpu_x, sp);
        cpu_layer_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // ANE output (re-run once for verification)
    let mut ane_x = x_data.clone();
    for (i, layer) in layers.iter().enumerate().take(actual_layers) {
        let blobs = vec![
            build_blob(&layer.wg),
            build_blob(&layer.wu),
            build_blob(&layer.build_wdi()),
        ];
        let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
        let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

        if let Ok(mut exec) = rustane::wrapper::ANECompiler::new().compile_multi(
            &mil,
            &nrefs,
            &brefs,
            &lens,
            &[d * sp * bytes_per_elem],
            &[d * sp * bytes_per_elem],
        ) {
            exec.write_input(0, &to_fp16(&ane_x)).expect("write");
            exec.eval().expect("eval");
            let output = from_fp16(&exec.read_output_vec(0).expect("read"));
            ane_x = output;
        } else {
            println!("  Verification compile failed at layer {}", i + 1);
            break;
        }
    }

    // Compare CPU vs ANE
    let mut max_diff = 0.0_f32;
    let mut rel_errors = Vec::new();
    for i in 0..cpu_x.len().min(ane_x.len()) {
        let diff = (cpu_x[i] - ane_x[i]).abs();
        let denom = cpu_x[i].abs().max(1e-6_f32);
        let rel = (diff / denom) as f64;
        rel_errors.push(rel);
        max_diff = max_diff.max(diff);
    }
    let avg_rel: f64 = rel_errors.iter().sum::<f64>() / rel_errors.len() as f64;
    let max_rel: f64 = rel_errors.iter().cloned().fold(0.0_f64, f64::max);
    let p99_rel: f64 = {
        let mut sorted = rel_errors.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[(sorted.len() as f64 * 0.99) as usize]
    };

    println!("  CPU vs ANE output comparison ({} elements):", cpu_x.len());
    println!("    Max absolute diff: {:.6}", max_diff);
    println!("    Avg relative error: {:.4}%", avg_rel * 100.0);
    println!("    Max relative error: {:.4}%", max_rel * 100.0);
    println!("    P99 relative error: {:.4}%", p99_rel * 100.0);
    let accuracy = if avg_rel < 0.01 {
        "EXCELLENT"
    } else if avg_rel < 0.05 {
        "GOOD"
    } else if avg_rel < 0.10 {
        "OK"
    } else {
        "POOR"
    };
    println!("    Accuracy: {}", accuracy);

    // ── CPU Baseline ──
    println!("\n=== CPU Baseline (BLAS, {} layers) ===", actual_layers);
    let mut cpu_total_times = Vec::new();

    for _ in 0..num_runs {
        let mut x = x_data.clone();
        let t = Instant::now();
        for layer in layers.iter().take(actual_layers) {
            x = layer.forward(&x, sp);
        }
        cpu_total_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let cpu_avg: f64 = cpu_total_times.iter().sum::<f64>() / num_runs as f64;
    let cpu_min: f64 = cpu_total_times
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let cpu_per_layer: f64 = cpu_avg / actual_layers as f64;

    println!(
        "  CPU total: {:.1}ms ({:.2}ms/layer)",
        cpu_avg, cpu_per_layer
    );
    println!("  CPU best:  {:.1}ms", cpu_min);

    // ── Speedup Summary ──
    println!("\n============================================================");
    println!(
        "  RESULTS: {}-layer FFN at D={}, SP={}",
        actual_layers, d, sp
    );
    println!("============================================================");
    let speedup_avg = cpu_avg / ane_avg;
    let speedup_best = cpu_min / ane_min;
    let ane_per_layer = ane_avg / actual_layers as f64;

    println!("  {:>20} {:>12} {:>12} {:>12}", "", "ANE", "CPU", "Speedup");
    println!(
        "  {:>20} {:>12} {:>12} {:>12}",
        "--------------------", "----------", "----------", "--------"
    );
    println!(
        "  {:>20} {:>12.3} {:>12.1} {:>11.1}x",
        "Total latency", ane_avg, cpu_avg, speedup_avg
    );
    println!(
        "  {:>20} {:>12.3} {:>12.2} {:>11.1}x",
        "Per layer",
        ane_per_layer,
        cpu_per_layer,
        cpu_per_layer / ane_per_layer
    );
    println!(
        "  {:>20} {:>12.3} {:>12.1} {:>11.1}x",
        "Best run", ane_min, cpu_min, speedup_best
    );
    println!();
    println!("  Per-layer ANE breakdown:");
    println!(
        "    {:.3}ms total (includes write_input + eval + read_output)",
        ane_per_layer
    );

    // Throughput
    let ane_tps = 1000.0 / ane_avg * sp as f64; // tokens per second
    let cpu_tps = 1000.0 / cpu_avg * sp as f64;
    println!();
    println!("  Throughput (tokens/sec):");
    println!("    ANE: {:.0} tokens/sec", ane_tps);
    println!("    CPU: {:.0} tokens/sec", cpu_tps);

    // Theoretical scaling
    println!();
    println!("  Theoretical scaling (at 60us/layer overhead):");
    for &n in &[1, 2, 4, 6, 8, 12, 16, 24, 32] {
        let ane_est = n as f64 * ane_per_layer;
        let cpu_est = n as f64 * cpu_per_layer;
        let speedup = cpu_est / ane_est;
        let marker = if n == actual_layers {
            " <-- measured"
        } else {
            ""
        };
        println!(
            "    {:>2} layers: {:.1}ms ANE, {:.1}ms CPU, {:.0}x speedup{}",
            n, ane_est, cpu_est, speedup, marker
        );
    }

    println!(
        "\n  CONCLUSION: ANE inference is {:.0}x faster than CPU BLAS for {}-layer FFN",
        speedup_avg, actual_layers
    );
}
