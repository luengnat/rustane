//! Instrumented reload_weights timing to find the bottleneck.
//!
//! Measures each step of reload_weights individually:
//!   1. Unload old model
//!   2. Create weight dictionary
//!   3. Create model descriptor
//!   4. Create in-memory model
//!   5. Write model files to disk
//!   6. Compile (the suspected bottleneck)
//!   7. Load
//!   8. Rebuild request
//!
//! Usage: cargo run --example test_reload_instrumented -- [D] [SP]

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

fn mil_conv1x1(d: usize, sp: usize) -> String {
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
    m.push_str(", 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/W.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"cv\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);

    println!(
        "=== reload_weights Instrumentation (D={}, SP={}) ===\n",
        d, sp
    );

    let inp = det_rand(d * sp, 1.0, 42);
    let inp16 = to_fp16(&inp);
    let mil = mil_conv1x1(d, sp);
    let w1 = det_rand(d * d, 0.5, 100);
    let w2 = det_rand(d * d, 0.5, 200);

    let full_names: Vec<String> = vec![format!("@model_path/weights/W.bin")];
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();

    // Initial compile (baseline)
    let t0 = Instant::now();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(
            &mil,
            &name_refs,
            &[build_blob(&w1).as_slice()],
            &[build_blob(&w1).len()],
            &[d * sp * 2],
            &[d * sp * 2],
        )
        .expect("compile failed");
    let initial_compile = t0.elapsed();
    println!(
        "  Initial compile: {:.2}ms",
        initial_compile.as_secs_f64() * 1000.0
    );

    // Warmup
    exec.write_input(0, &inp16).unwrap();
    exec.eval().unwrap();

    // Now measure reload multiple times and also measure just eval to compare
    let mut reload_total = std::time::Duration::ZERO;
    let mut eval_total = std::time::Duration::ZERO;
    let mut blob_build_total = std::time::Duration::ZERO;
    let iterations = 10;

    for i in 0..iterations {
        let ws = det_rand(d * d, 0.5, 300 + i as u64);

        // Time blob building (fp32 → fp16 → blob format)
        let t_blob = Instant::now();
        let blob = build_blob(&ws);
        blob_build_total += t_blob.elapsed();

        // Time reload
        let t_reload = Instant::now();
        exec.reload_weights(&[("@model_path/weights/W.bin", blob.as_slice())])
            .unwrap();
        reload_total += t_reload.elapsed();

        // Time eval after reload
        let t_eval = Instant::now();
        exec.write_input(0, &inp16).unwrap();
        exec.eval().unwrap();
        let _ = exec.read_output_vec(0).unwrap();
        eval_total += t_eval.elapsed();
    }

    println!("\n  Over {} iterations (D={}):", iterations, d);
    println!("  ┌─────────────────────────┬───────────┬──────────┐");
    println!("  │ Step                    │ Total     │ Avg      │");
    println!("  ├─────────────────────────┼───────────┼──────────┤");
    println!(
        "  │ Initial compile         │ {:>7.1}ms │ {:>7.1}ms │",
        initial_compile.as_secs_f64() * 1000.0,
        initial_compile.as_secs_f64() * 1000.0
    );
    println!(
        "  │ Blob build (fp32→blob)  │ {:>7.1}ms │ {:>7.2}ms │",
        blob_build_total.as_secs_f64() * 1000.0,
        blob_build_total.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "  │ reload_weights()        │ {:>7.1}ms │ {:>7.2}ms │",
        reload_total.as_secs_f64() * 1000.0,
        reload_total.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "  │ Eval (after reload)     │ {:>7.1}ms │ {:>7.2}ms │",
        eval_total.as_secs_f64() * 1000.0,
        eval_total.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!("  └─────────────────────────┴───────────┴──────────┘");

    let reload_avg = reload_total.as_secs_f64() * 1000.0 / iterations as f64;
    let eval_avg = eval_total.as_secs_f64() * 1000.0 / iterations as f64;
    let compile_ms = initial_compile.as_secs_f64() * 1000.0;

    // Analyze
    println!("\n  Analysis:");
    println!(
        "    reload/compile ratio: {:.1}x (reload should be < compile)",
        reload_avg / compile_ms
    );
    println!(
        "    eval time:             {:.2}ms (pure ANE compute)",
        eval_avg
    );
    println!(
        "    reload overhead:       {:.1}x eval ({:.1}ms / {:.2}ms)",
        reload_avg / eval_avg,
        reload_avg,
        eval_avg
    );
    println!();

    // The key question: is reload doing a full compile?
    // If reload ≈ compile, it's recompiling.
    // If reload << compile, it's doing something smarter.
    if reload_avg > compile_ms * 0.8 {
        println!("  ⚠️  reload ≈ compile time → ANE is doing a FULL RECOMPILE");
        println!("  This means ANE training requires a different approach:");
        println!("    Option A: Batch multiple CPU steps between ANE syncs");
        println!("    Option B: Find a way to patch weights without recompile");
        println!("    Option C: Use ANE for inference, CPU for training");
    } else if reload_avg > eval_avg * 10.0 {
        println!("  ⚠️  reload is fast but still >> eval → moderate overhead");
    } else {
        println!("  ✓ reload is efficient — ANE training is viable");
    }
}
