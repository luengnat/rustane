//! Test reload_weights() on ANEExecutor — the key to ANE training.
//!
//! Measures:
//!   1. Initial compile time
//!   2. reload_weights time (recompile with new weights)
//!   3. eval time after reload
//!   4. Correctness: output changes when weights change
//!   5. Compile count impact
//!
//! Usage: cargo run --example test_reload_weights -- [D] [SP]

use half::f16;
use std::env;
use std::time::Instant;

const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

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

fn det_rand(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    (0..n)
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

fn cpu_matmul(w: &[f32], wr: usize, wc: usize, inp: &[f32], sp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; wr * sp];
    for r in 0..wr {
        for s in 0..sp {
            let mut a = 0.0f32;
            for c in 0..wc {
                a += w[r * wc + c] * inp[c * sp + s];
            }
            out[r * sp + s] = a;
        }
    }
    out
}

fn mil_conv1x1(d: usize, sp: usize, wn: &str) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
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
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    println!("=== reload_weights() Test (D={}, SP={}) ===\n", d, sp);

    let inp: Vec<f32> = det_rand(d * sp, 1.0, 42);
    let inp16 = to_fp16(&inp);
    let mil = mil_conv1x1(d, sp, "W");

    // ── Step 1: Initial compile ──
    println!("Step 1: Initial compile");
    let w1 = det_rand(d * d, 0.5, 100);
    let t0 = Instant::now();
    let full_names: Vec<String> = vec![format!("@model_path/weights/W.bin")];
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(
            &mil,
            &name_refs,
            &[build_blob(&w1).as_slice()],
            &[build_blob(&w1).len()],
            &[d * sp * 2],
            &[d * sp * 2],
        )
        .expect("initial compile failed");
    let compile_time = t0.elapsed();
    println!("  Compile: {:?}", compile_time);

    // Warmup eval
    exec.write_input(0, &inp16).expect("write");
    exec.eval().expect("warmup eval");

    // Baseline output
    let out1 = from_fp16(&exec.read_output_vec(0).expect("read"));
    let cpu1 = cpu_matmul(&w1, d, d, &inp, sp);
    let mut max_err = 0.0f32;
    for i in 0..out1.len() {
        max_err = max_err.max((out1[i] - cpu1[i]).abs());
    }
    println!(
        "  Output 1 vs CPU: max_err={:.6} (should be small)",
        max_err
    );

    // ── Step 2: reload_weights with different weights ──
    println!("\nStep 2: reload_weights with modified weights");
    let w2 = det_rand(d * d, 0.5, 200);

    let t1 = Instant::now();
    exec.reload_weights(&[("@model_path/weights/W.bin", build_blob(&w2).as_slice())])
        .expect("reload failed");
    let reload_time = t1.elapsed();
    println!("  Reload: {:?}", reload_time);

    // Eval after reload
    exec.write_input(0, &inp16).expect("write after reload");
    exec.eval().expect("eval after reload");
    let out2 = from_fp16(&exec.read_output_vec(0).expect("read after reload"));
    let cpu2 = cpu_matmul(&w2, d, d, &inp, sp);
    max_err = 0.0f32;
    for i in 0..out2.len() {
        max_err = max_err.max((out2[i] - cpu2[i]).abs());
    }
    println!(
        "  Output 2 vs CPU: max_err={:.6} (should be small)",
        max_err
    );

    // Verify outputs actually changed
    let mut diff_count = 0usize;
    for i in 0..out1.len().min(out2.len()) {
        if (out1[i] - out2[i]).abs() > 0.001 {
            diff_count += 1;
        }
    }
    println!(
        "  Output changed: {}/{} values differ (>0.001)",
        diff_count,
        out1.len()
    );

    // ── Step 3: Multiple reloads (measure consistency) ──
    println!("\nStep 3: Multiple reloads (5 iterations)");
    let mut reload_times = Vec::new();
    for step in 0..5 {
        let ws = det_rand(d * d, 0.5, 300 + step as u64);
        let t = Instant::now();
        exec.reload_weights(&[("@model_path/weights/W.bin", build_blob(&ws).as_slice())])
            .expect(&format!("reload step {} failed", step));
        reload_times.push(t.elapsed());

        exec.write_input(0, &inp16).expect("write");
        exec.eval().expect("eval");

        let out = from_fp16(&exec.read_output_vec(0).expect("read"));
        let cpu = cpu_matmul(&ws, d, d, &inp, sp);
        let mut me = 0.0f32;
        for i in 0..out.len() {
            me = me.max((out[i] - cpu[i]).abs());
        }
        println!(
            "  Reload {}: {:?} | max_err={:.6}",
            step,
            reload_times.last().unwrap(),
            me
        );
    }

    let avg_reload = reload_times.iter().sum::<std::time::Duration>() / reload_times.len() as u32;
    let min_reload = reload_times.iter().min().unwrap();
    let max_reload = reload_times.iter().max().unwrap();
    println!(
        "\n  Reload timing: avg={:?} min={:?} max={:?}",
        avg_reload, min_reload, max_reload
    );

    // ── Summary ──
    println!("\n═══════════════════════════════════════════════");
    println!("  Initial compile: {:?}", compile_time);
    println!("  Avg reload:      {:?}", avg_reload);
    println!(
        "  Speedup:         {:.1}x",
        compile_time.as_secs_f64() / avg_reload.as_secs_f64()
    );
    println!("═══════════════════════════════════════════════");

    // Check if reload is actually faster
    if avg_reload >= compile_time {
        println!("\n  ⚠️  WARNING: reload_weights is NOT faster than initial compile!");
        println!("  This means ANE training will be bottlenecked by recompilation.");
    } else {
        println!("\n  ✓ reload_weights is faster — ANE training is viable.");
    }
}
