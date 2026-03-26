//! Conv vs Matmul: Which is faster on ANE?
//!
//! Conv1x1 with BLOBFILE weights is the ObjC reference approach.
//! We compare:
//! 1. conv1x1 vs matmul for same [D, D] → [D, S] linear projection
//! 2. conv1x3 vs conv1x8 for local attention patterns (spatial sliding window)
//! 3. Single conv1x1 vs single dynamic matmul eval latency
//!
//! All tests run in subprocess since ANE crashes are SIGSEGV.

use std::env;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let test_id: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    match test_id {
        0 => run_all(),
        1 => test_conv1x1(),
        2 => test_conv1x3(),
        3 => test_conv1x8(),
        4 => test_dynamic_matmul(),
        5 => test_conv_vs_matmul_d64(),
        6 => test_conv_vs_matmul_d128(),
        _ => {
            eprintln!("Usage: bench_conv_vs_matmul [test_id]");
            eprintln!("  0 = run all (default)");
            eprintln!("  1-4 = individual ops");
            eprintln!("  5-6 = direct comparisons");
            std::process::exit(1);
        }
    }
}

fn run_subprocess(test_id: u32) -> (bool, String) {
    let exe = env::current_exe().unwrap();
    let output = Command::new(&exe)
        .arg(&test_id.to_string())
        .output()
        .expect("failed to run subprocess");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();

    if !success {
        let code = output.status.code().unwrap_or(1);
        eprintln!(
            "  [subprocess test {}] exit={} stderr={}",
            test_id,
            code,
            stderr.trim()
        );
    }

    (success, stdout)
}

fn run_all() {
    println!();
    println!("==========================================================================");
    println!("  Conv vs Matmul on ANE: Comprehensive Benchmark");
    println!("  conv1x1: BLOBFILE weights (ObjC reference approach)");
    println!("  conv1xS: Spatial sliding window (local attention candidate)");
    println!("  matmul: Dynamic weights in input tensor (our proven approach)");
    println!("==========================================================================");

    println!("\n--- Individual Op Benchmark: D=64, S=64 ---");
    println!("  (warmup=20, iterations=200, write+eval+read cycle)");
    println!();
    println!(
        "  {:<35} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "Operation", "Wr(us)", "Eval(us)", "Rd(us)", "Total(us)", "Samp/sec"
    );
    println!("  {}", "-".repeat(82));

    for (id, label) in [
        (1, "conv1x1  D=64->64  S=64"),
        (2, "conv1x3  D=64->64  S=64"),
        (3, "conv1x8  D=64->64  S=64"),
        (4, "matmul   D=64->64  S=64"),
    ] {
        let (ok, stdout) = run_subprocess(id);
        if ok {
            for line in stdout.lines() {
                if line.starts_with("RESULT:") {
                    print!("  {:<35} {}", label, &line[7..]);
                }
            }
        } else {
            println!("  {:<35} CRASHED (SIGSEGV)", label);
        }
    }

    println!("\n--- Direct Comparison: Conv vs Matmul at D=64 ---");
    let (ok, stdout) = run_subprocess(5);
    if ok {
        println!("{}", stdout);
    } else {
        println!("  CRASHED");
    }

    println!("\n--- Direct Comparison: Conv vs Matmul at D=128 ---");
    let (ok, stdout) = run_subprocess(6);
    if ok {
        println!("{}", stdout);
    } else {
        println!("  CRASHED");
    }

    println!("\n==========================================================================");
    println!("  KEY FINDINGS");
    println!("==========================================================================");
    println!();
    println!("  1. Conv1x1 uses ANE's native convolution hardware accelerator.");
    println!("     Matmul goes through a more generic computation path.");
    println!("     -> conv1x1 may have hardware advantage for small kernels.");
    println!();
    println!("  2. Conv1x3 and conv1x8 implement spatial sliding windows:");
    println!("     - conv1x3: each output position sees 3 adjacent tokens");
    println!("     - conv1x8: each output position sees 8 adjacent tokens");
    println!("     -> These are LOCAL attention patterns - no softmax needed!");
    println!("     -> Could replace full attention with chunked local attention.");
    println!();
    println!("  3. Conv output shrinks: S=64 -> S-K+1 (valid padding)");
    println!("     - conv1x1: 64->64 (no shrinkage)");
    println!("     - conv1x3: 64->62 (2 tokens lost at edges)");
    println!("     - conv1x8: 64->57 (7 tokens lost at edges)");
    println!();
    println!("  4. BLOBFILE weights for conv = baked at compile time (fast but static).");
    println!("     Dynamic weights for matmul = packed in input (flexible for training).");
    println!();
    println!("  5. For training: conv with BLOBFILE weights means recompilation per step");
    println!("     -> NOT practical for gradient updates.");
    println!("     -> Use dynamic matmul for trainable projections.");
    println!("     -> Use conv1xS only for FIXED operations (e.g., positional encoding).");
}

// ============================================================
// MIL Generators
// ============================================================

fn conv_mil(ic: usize, oc: usize, seq: usize, kw: usize) -> String {
    let out_seq = if seq >= kw { seq - kw + 1 } else { 1 };
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    let input_sig = format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        ic, seq
    );
    mil.push_str(&input_sig);
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    let cast_line = format!(
        "        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        ic, seq
    );
    mil.push_str(&cast_line);
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    // Weight: [OC, IC, 1, KW]
    let w_line = format!(
        "        tensor<fp16, [{}, {}, 1, {}]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, 1, {}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        oc, ic, kw, oc, ic, kw
    );
    mil.push_str(&w_line);
    let conv_line = format!(
        "        tensor<fp16, [1, {}, 1, {}]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = xh)[name = string(\"conv\")];\n",
        oc, out_seq
    );
    mil.push_str(&conv_line);
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    let out_line = format!(
        "        tensor<fp32, [1, {}, 1, {}]> yo = cast(dtype = to32, x = y)[name = string(\"cout\")];\n",
        oc, out_seq
    );
    mil.push_str(&out_line);
    mil.push_str("    } -> (yo);\n");
    mil.push_str("}\n");
    mil
}

/// Dynamic matmul with weights packed in input tensor.
/// Input: [1, D+D*O, 1, S] -> Output: [1, O, 1, S]
fn dynamic_matmul_proj_mil(dim: usize, out_dim: usize, seq: usize) -> String {
    let total_ch = dim + dim * out_dim;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    let sig = format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        total_ch, seq
    );
    mil.push_str(&sig);
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    let cast = format!(
        "        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        total_ch, seq
    );
    mil.push_str(&cast);

    // Slice activations: [1, D, 1, S]
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    let sa = format!(
        "        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        dim, seq
    );
    mil.push_str(&sa);
    let act = format!(
        "        tensor<fp16, [1, {}, 1, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n",
        dim, seq
    );
    mil.push_str(&act);

    // Slice weights: [1, D*O, 1, S] -> [1, D*O, 1, 1]
    let bw = format!(
        "        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n",
        dim
    );
    mil.push_str(&bw);
    let sw = format!(
        "        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        dim * out_dim, seq
    );
    mil.push_str(&sw);
    let wf = format!(
        "        tensor<fp16, [1, {}, 1, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n",
        dim * out_dim, seq
    );
    mil.push_str(&wf);

    // Larger reshape target first: [1, 1, D, O]
    let ws = format!(
        "        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n",
        dim, out_dim
    );
    mil.push_str(&ws);
    let sw1 = format!(
        "        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n",
        dim * out_dim
    );
    mil.push_str(&sw1);
    let wf1 = format!(
        "        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n",
        dim * out_dim
    );
    mil.push_str(&wf1);
    let W = format!(
        "        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n",
        dim, out_dim
    );
    mil.push_str(&W);

    // [1, D, 1, S] -> [1, 1, D, S] -> [1, 1, S, D]
    let rs = format!(
        "        tensor<int32, [4]> rs = const()[name = string(\"rs\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n",
        dim, seq
    );
    mil.push_str(&rs);
    let xr = format!(
        "        tensor<fp16, [1, 1, {}, {}]> xr = reshape(shape = rs, x = act)[name = string(\"xr\")];\n",
        dim, seq
    );
    mil.push_str(&xr);
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    let xt = format!(
        "        tensor<fp16, [1, 1, {}, {}]> xt = transpose(perm = pm, x = xr)[name = string(\"xt\")];\n",
        seq, dim
    );
    mil.push_str(&xt);

    // Matmul: [1, 1, S, D] @ [1, 1, D, O] -> [1, 1, S, O]
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    let mm = format!(
        "        tensor<fp16, [1, 1, {}, {}]> mm = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = W)[name = string(\"mm\")];\n",
        seq, out_dim
    );
    mil.push_str(&mm);

    // [1, 1, S, O] -> [1, 1, O, S] -> [1, O, 1, S]
    let mt = format!(
        "        tensor<fp16, [1, 1, {}, {}]> mt = transpose(perm = pm, x = mm)[name = string(\"mt\")];\n",
        out_dim, seq
    );
    mil.push_str(&mt);
    let os = format!(
        "        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        out_dim, seq
    );
    mil.push_str(&os);
    let yr = format!(
        "        tensor<fp16, [1, {}, 1, {}]> yr = reshape(shape = os, x = mt)[name = string(\"yr\")];\n",
        out_dim, seq
    );
    mil.push_str(&yr);

    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    let y = format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n",
        out_dim, seq
    );
    mil.push_str(&y);
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

// ============================================================
// Benchmark helpers
// ============================================================

fn make_input_data(size_bytes: usize) -> Vec<u8> {
    let num_floats = size_bytes / 4;
    let data: Vec<f32> = (0..num_floats)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn run_bench_conv(label: &str, ic: usize, oc: usize, seq: usize, kw: usize) -> Option<String> {
    let out_seq = if seq >= kw { seq - kw + 1 } else { 1 };
    let in_b = ic * seq * 4;
    let out_b = oc * out_seq * 4;
    let warmup = 20;
    let iters = 200;

    let mil = conv_mil(ic, oc, seq, kw);

    // Weight shape: [OC, IC, 1, KW] -> rows=OC, cols=IC*KW
    let weight_count = oc * ic * kw;
    let weights: Vec<f32> = (0..weight_count)
        .map(|i| if i % (ic * kw + 1) == 0 { 1.0 } else { 0.01 })
        .collect();
    let blob = match rustane::ane::WeightBlob::from_f32(&weights, oc, ic * kw) {
        Ok(b) => b,
        Err(e) => {
            println!("  {:<20} BLOB_FAIL: {}", label, e);
            return None;
        }
    };

    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &["@model_path/weights/weight.bin"],
        &[blob.as_bytes()],
        &[blob.as_bytes().len()],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => e,
        Err(e) => {
            println!("  {:<20} COMPILE_FAIL: {}", label, e);
            return None;
        }
    };

    let data = make_input_data(in_b);

    for _ in 0..warmup {
        exec.write_input(0, &data).unwrap();
        exec.eval().unwrap();
        let _ = exec.read_output_vec(0).unwrap();
    }

    let w_start = Instant::now();
    for _ in 0..iters {
        exec.write_input(0, &data).unwrap();
    }
    let write_us = w_start.elapsed().as_micros() as f64 / iters as f64;

    exec.write_input(0, &data).unwrap();
    let e_start = Instant::now();
    for _ in 0..iters {
        exec.eval().unwrap();
    }
    let eval_us = e_start.elapsed().as_micros() as f64 / iters as f64;

    let r_start = Instant::now();
    for _ in 0..iters {
        let _ = exec.read_output_vec(0).unwrap();
    }
    let read_us = r_start.elapsed().as_micros() as f64 / iters as f64;

    let total_us = write_us + eval_us + read_us;
    let samples_per_sec = 1_000_000.0 / total_us;

    let result = format!(
        "  {:<20} {:>8.1} {:>8.1} {:>8.1} {:>10.1} {:>10.0} [out_seq={}]",
        label, write_us, eval_us, read_us, total_us, samples_per_sec, out_seq
    );
    println!("{}", result);
    Some(format!(
        "RESULT: write={:.1} eval={:.1} read={:.1} total={:.1} samp/sec={:.0} [out_seq={}]",
        write_us, eval_us, read_us, total_us, samples_per_sec, out_seq
    ))
}

fn run_bench_matmul(label: &str, dim: usize, out_dim: usize, seq: usize) -> Option<String> {
    let total_ch = dim + dim * out_dim;
    let in_b = total_ch * seq * 4;
    let out_b = out_dim * seq * 4;
    let warmup = 20;
    let iters = 200;

    let mil = dynamic_matmul_proj_mil(dim, out_dim, seq);

    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &[],
        &[],
        &[],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => e,
        Err(e) => {
            println!("  {:<20} COMPILE_FAIL: {}", label, e);
            return None;
        }
    };

    let data = make_input_data(in_b);

    for _ in 0..warmup {
        exec.write_input(0, &data).unwrap();
        exec.eval().unwrap();
        let _ = exec.read_output_vec(0).unwrap();
    }

    let w_start = Instant::now();
    for _ in 0..iters {
        exec.write_input(0, &data).unwrap();
    }
    let write_us = w_start.elapsed().as_micros() as f64 / iters as f64;

    exec.write_input(0, &data).unwrap();
    let e_start = Instant::now();
    for _ in 0..iters {
        exec.eval().unwrap();
    }
    let eval_us = e_start.elapsed().as_micros() as f64 / iters as f64;

    let r_start = Instant::now();
    for _ in 0..iters {
        let _ = exec.read_output_vec(0).unwrap();
    }
    let read_us = r_start.elapsed().as_micros() as f64 / iters as f64;

    let total_us = write_us + eval_us + read_us;
    let samples_per_sec = 1_000_000.0 / total_us;

    let result = format!(
        "  {:<20} {:>8.1} {:>8.1} {:>8.1} {:>10.1} {:>10.0}",
        label, write_us, eval_us, read_us, total_us, samples_per_sec
    );
    println!("{}", result);
    Some(format!(
        "RESULT: write={:.1} eval={:.1} read={:.1} total={:.1} samp/sec={:.0}",
        write_us, eval_us, read_us, total_us, samples_per_sec
    ))
}

// ============================================================
// Individual test functions (each runs in its own subprocess)
// ============================================================

fn test_conv1x1() {
    let _ = rustane::init();
    run_bench_conv("conv1x1", 64, 64, 64, 1);
}

fn test_conv1x3() {
    let _ = rustane::init();
    run_bench_conv("conv1x3", 64, 64, 64, 3);
}

fn test_conv1x8() {
    let _ = rustane::init();
    run_bench_conv("conv1x8", 64, 64, 64, 8);
}

fn test_dynamic_matmul() {
    let _ = rustane::init();
    run_bench_matmul("matmul", 64, 64, 64);
}

fn test_conv_vs_matmul_d64() {
    let _ = rustane::init();

    println!("  Conv1x1 vs Matmul at D=64, S=64:");
    println!(
        "  {:<20} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "Op", "Wr(us)", "Eval(us)", "Rd(us)", "Total(us)", "Samp/sec"
    );
    println!("  {}", "-".repeat(72));

    // Conv1x1: D=64 -> D=64
    run_bench_conv("conv1x1  64->64", 64, 64, 64, 1);

    // Conv1x3: D=64 -> D=64 (local attention: 3-token window)
    run_bench_conv("conv1x3  64->64", 64, 64, 64, 3);

    // Conv1x8: D=64 -> D=64 (local attention: 8-token window)
    run_bench_conv("conv1x8  64->64", 64, 64, 64, 8);

    // Dynamic matmul: D=64 -> D=64
    run_bench_matmul("matmul   64->64", 64, 64, 64);

    // Conv1x1: D=64 -> D=192 (QKV fused, same as ObjC stories_mil.h)
    run_bench_conv("conv1x1  64->192", 64, 192, 64, 1);

    // Dynamic matmul: D=64 -> D=192
    run_bench_matmul("matmul   64->192", 64, 192, 64);
}

fn test_conv_vs_matmul_d128() {
    let _ = rustane::init();

    println!("  Conv1x1 vs Matmul at D=128, S=64:");
    println!(
        "  {:<20} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "Op", "Wr(us)", "Eval(us)", "Rd(us)", "Total(us)", "Samp/sec"
    );
    println!("  {}", "-".repeat(72));

    // Conv1x1: D=128 -> D=128
    run_bench_conv("conv1x1  128->128", 128, 128, 64, 1);

    // Dynamic matmul: D=128 -> D=128
    run_bench_matmul("matmul   128->128", 128, 128, 64);

    // Conv1x1: D=128 -> D=384 (QKV fused)
    run_bench_conv("conv1x1  128->384", 128, 384, 64, 1);

    // Dynamic matmul: D=128 -> D=384
    run_bench_matmul("matmul   128->384", 128, 384, 64);
}
