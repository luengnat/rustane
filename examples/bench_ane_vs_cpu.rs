//! ANE vs CPU: Fused Transformer Forward Pass Benchmark
//!
//! Definitive test: can ANE beat CPU for transformer training?
//!
//! Strategy:
//! 1. Fused 3-matmul + bias + relu in ONE eval call (dynamic weights in input)
//! 2. Batch dimension to amortize ~130μs dispatch overhead
//! 3. CPU baseline: same computation, naive implementation
//!
//! All ANE tests run in subprocess since ANE crashes are SIGSEGV.

use std::env;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let test_id: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    match test_id {
        0 => run_all(),
        1 => test_single_matmul(),
        2 => test_fused_3matmul(),
        3 => test_fused_3matmul_batched(),
        4 => test_cpu_baseline(),
        _ => {
            eprintln!("Usage: bench_ane_vs_cpu [test_id]");
            std::process::exit(1);
        }
    }
}

fn run_subprocess(test_id: u32, extra_args: &[&str]) -> (bool, String) {
    let exe = env::current_exe().unwrap();
    let mut cmd = Command::new(&exe);
    cmd.arg(&test_id.to_string());
    for arg in extra_args {
        cmd.arg(arg);
    }
    let output = cmd.output().expect("failed to run subprocess");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();
    if !success {
        let code = output.status.code().unwrap_or(1);
        eprintln!(
            "  [test {}] exit={} stderr={}",
            test_id,
            code,
            stderr.trim().chars().take(200).collect::<String>()
        );
    }
    (success, stdout)
}

fn run_all() {
    println!();
    println!("==========================================================================");
    println!("  ANE vs CPU: Fused Transformer Forward Pass Benchmark");
    println!("  Fused program: 3 matmuls (Q,K,V projections) + bias + relu");
    println!("  = ~20 ops in ONE eval call with dynamic weights in input");
    println!("==========================================================================");

    // Test 1: Single matmul (sanity check — proven to work)
    println!("\n--- Test 1: Single Dynamic Matmul D=64, S=64 ---");
    let (ok, stdout) = run_subprocess(1, &[]);
    if ok {
        for line in stdout.lines() {
            if line.starts_with("RESULT:") {
                print!("  {}", &line[7..]);
            }
        }
    } else {
        println!("  CRASHED");
    }

    // Test 2: Fused 3-matmul at B=1
    println!("\n--- Test 2: Fused 3-Matmul + Bias + ReLU, B=1 ---");
    let (ok, stdout) = run_subprocess(2, &[]);
    if ok {
        for line in stdout.lines() {
            if line.starts_with("RESULT:") {
                print!("  {}", &line[7..]);
            }
        }
    } else {
        println!("  CRASHED — falling back to separate matmuls");
    }

    // Test 3: Fused 3-matmul at B=1,2,4,8
    println!("\n--- Test 3: Fused 3-Matmul Batched (B=1,2,4,8) ---");
    println!(
        "  {:<6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12} {:>8}",
        "Batch", "In(KB)", "Wr(us)", "Eval(us)", "Total(us)", "us/samp", "Samp/sec", "Speedup"
    );
    println!("  {}", "-".repeat(88));

    let mut b1_us = 0.0f64;
    for batch in [1, 2, 4, 8] {
        let batch_str = batch.to_string();
        let (ok, stdout) = run_subprocess(3, &[&batch_str]);
        if ok {
            for line in stdout.lines() {
                if line.starts_with("RESULT:") {
                    let parts: Vec<&str> = line[7..].split_whitespace().collect();
                    if parts.len() >= 6 {
                        let total: f64 = parts[4].parse().unwrap_or(0.0);
                        let per_samp = total / batch as f64;
                        let samp_sec = 1_000_000.0 / per_samp;
                        let speedup = if b1_us > 0.0 { b1_us / per_samp } else { 1.0 };
                        if batch == 1 {
                            b1_us = per_samp;
                        }
                        println!(
                            "  {:<6} {:>6.0} {:>8.1} {:>8.1} {:>10.1} {:>10.1} {:>10.0} {:>6.1}x",
                            batch,
                            parts[0].parse::<f64>().unwrap_or(0.0) / 1024.0,
                            parts[2].parse::<f64>().unwrap_or(0.0),
                            parts[3].parse::<f64>().unwrap_or(0.0),
                            total,
                            per_samp,
                            samp_sec,
                            speedup
                        );
                    }
                }
            }
        } else {
            println!("  {:<6} CRASHED", batch);
        }
    }

    // Test 4: CPU baseline
    println!("\n--- Test 4: CPU Baseline ---");
    let (ok, stdout) = run_subprocess(4, &[]);
    if ok {
        println!("{}", stdout);
    } else {
        println!("  FAILED");
    }

    // Summary
    println!("\n==========================================================================");
    println!("  SUMMARY: ANE vs CPU for Fused 3-Matmul Forward Pass");
    println!("==========================================================================");
    println!(
        r#"
  Findings so far:
  - ANE has ~130us fixed dispatch overhead per eval call
  - Fusion (3 matmuls in 1 eval) eliminates 2 dispatch overheads
  - Batching amortizes the single dispatch overhead across samples
  - Conv1x1 is 1.2-3.9x faster than dynamic matmul per projection
  - BUT conv1x1 uses BLOBFILE weights (static, need recompile for updates)
  - Dynamic matmul packs weights in input (flexible for training)

  Next: Compare conv1x1 (BLOBFILE) vs dynamic matmul for training throughput
  (need to measure compile time for weight updates)
"#
    );
}

// ============================================================
// Proven single matmul MIL (from test_batch_matmul.rs test 5)
// ============================================================

fn single_dynamic_matmul_mil(batch: usize, dim: usize, seq: usize) -> String {
    let total_ch = dim + dim * dim;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    let sig = format!(
        "    func main<ios18>(tensor<fp32, [1, {}, {}, {}]> x) {{\n",
        total_ch, batch, seq
    );
    mil.push_str(&sig);
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    let cast = format!("        tensor<fp16, [1, {}, {}, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, batch, seq);
    mil.push_str(&cast);
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    let sa = format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n", dim, batch, seq);
    mil.push_str(&sa);
    let act = format!("        tensor<fp16, [1, {}, {}, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n", dim, batch, seq);
    mil.push_str(&act);
    let bw = format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", dim);
    mil.push_str(&bw);
    let sw = format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n", dim * dim, batch, seq);
    mil.push_str(&sw);
    let wf = format!("        tensor<fp16, [1, {}, {}, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n", dim * dim, batch, seq);
    mil.push_str(&wf);
    // Larger const first
    let ws = format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", dim, dim);
    mil.push_str(&ws);
    let sw1 = format!("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n", dim * dim);
    mil.push_str(&sw1);
    let wf1 = format!("        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n", dim * dim);
    mil.push_str(&wf1);
    let W = format!("        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n", dim, dim);
    mil.push_str(&W);
    let tb = format!("        tensor<int32, [4]> tb = const()[name = string(\"tb\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&tb);
    let ab = format!("        tensor<fp16, [{}, {}, 1, {}]> ab = transpose(perm = tb, x = act)[name = string(\"ab\")];\n", batch, dim, seq);
    mil.push_str(&ab);
    let rb = format!("        tensor<int32, [4]> rb = const()[name = string(\"rb\"), val = tensor<int32, [4]>([{}, 1, {}, {}])];\n", batch, dim, seq);
    mil.push_str(&rb);
    let rb2 = format!("        tensor<fp16, [{}, 1, {}, {}]> rb2 = reshape(shape = rb, x = ab)[name = string(\"rb2\")];\n", batch, dim, seq);
    mil.push_str(&rb2);
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    let xt = format!("        tensor<fp16, [{}, 1, {}, {}]> xt = transpose(perm = pm, x = rb2)[name = string(\"xt\")];\n", batch, seq, dim);
    mil.push_str(&xt);
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    let mm = format!("        tensor<fp16, [{}, 1, {}, {}]> mm = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = W)[name = string(\"mm\")];\n", batch, seq, dim);
    mil.push_str(&mm);
    let mt = format!("        tensor<fp16, [{}, 1, {}, {}]> mt = transpose(perm = pm, x = mm)[name = string(\"mt\")];\n", batch, dim, seq);
    mil.push_str(&mt);
    let mr = format!("        tensor<fp16, [{}, {}, 1, {}]> mr = reshape(shape = rb, x = mt)[name = string(\"mr\")];\n", batch, dim, seq);
    mil.push_str(&mr);
    let tb2 = format!("        tensor<int32, [4]> tb2 = const()[name = string(\"tb2\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&tb2);
    let ob = format!("        tensor<fp16, [1, {}, {}, {}]> ob = transpose(perm = tb2, x = mr)[name = string(\"ob\")];\n", dim, batch, seq);
    mil.push_str(&ob);
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    let y = format!("        tensor<fp32, [1, {}, {}, {}]> y = cast(dtype = to32, x = ob)[name = string(\"cout\")];\n", dim, batch, seq);
    mil.push_str(&y);
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

// ============================================================
// Fused 3-matmul + bias + relu MIL
// Input: [1, D + 3*D*D + D, B, S] = act + Wq + Wk + Wv + bias
// ============================================================

fn fused_3matmul_mil(batch: usize, dim: usize, seq: usize) -> String {
    let d = dim;
    let b = batch;
    let s = seq;
    let total_ch = d + 3 * d * d + d;

    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    let sig = format!(
        "    func main<ios18>(tensor<fp32, [1, {}, {}, {}]> x) {{\n",
        total_ch, b, s
    );
    mil.push_str(&sig);
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    let cast = format!("        tensor<fp16, [1, {}, {}, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, b, s);
    mil.push_str(&cast);

    // --- Slice activations: [1, D, B, S] ---
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    let sa = format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n", d, b, s);
    mil.push_str(&sa);
    let act = format!("        tensor<fp16, [1, {}, {}, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n", d, b, s);
    mil.push_str(&act);

    // --- Const order: larger reshape targets first ---
    let wrs = format!("        tensor<int32, [4]> wrs = const()[name = string(\"wrs\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", d, d);
    mil.push_str(&wrs);
    let sw1 = format!("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n", d * d);
    mil.push_str(&sw1);
    let sq = format!("        tensor<int32, [4]> sq = const()[name = string(\"sq\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n", d * d, b, s);
    mil.push_str(&sq);

    // --- Extract Wq: channels [D, D+D*D] ---
    let bq = format!("        tensor<int32, [4]> bq = const()[name = string(\"bq\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", d);
    mil.push_str(&bq);
    let wq = format!("        tensor<fp16, [1, {}, {}, {}]> wq = slice_by_size(x = xh, begin = bq, size = sq)[name = string(\"wq\")];\n", d * d, b, s);
    mil.push_str(&wq);
    let wq1 = format!("        tensor<fp16, [1, {}, 1, 1]> wq1 = slice_by_size(x = wq, begin = b0, size = sw1)[name = string(\"wq1\")];\n", d * d);
    mil.push_str(&wq1);
    let Wq = format!("        tensor<fp16, [1, 1, {}, {}]> Wq = reshape(shape = wrs, x = wq1)[name = string(\"Wq\")];\n", d, d);
    mil.push_str(&Wq);

    // --- Extract Wk: channels [D+D*D, D+2*D*D] ---
    let bk = format!("        tensor<int32, [4]> bk = const()[name = string(\"bk\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", d + d * d);
    mil.push_str(&bk);
    let wk = format!("        tensor<fp16, [1, {}, {}, {}]> wk = slice_by_size(x = xh, begin = bk, size = sq)[name = string(\"wk\")];\n", d * d, b, s);
    mil.push_str(&wk);
    let wk1 = format!("        tensor<fp16, [1, {}, 1, 1]> wk1 = slice_by_size(x = wk, begin = b0, size = sw1)[name = string(\"wk1\")];\n", d * d);
    mil.push_str(&wk1);
    let Wk = format!("        tensor<fp16, [1, 1, {}, {}]> Wk = reshape(shape = wrs, x = wk1)[name = string(\"Wk\")];\n", d, d);
    mil.push_str(&Wk);

    // --- Extract Wv: channels [D+2*D*D, D+3*D*D] ---
    let bv = format!("        tensor<int32, [4]> bv = const()[name = string(\"bv\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", d + 2 * d * d);
    mil.push_str(&bv);
    let wv = format!("        tensor<fp16, [1, {}, {}, {}]> wv = slice_by_size(x = xh, begin = bv, size = sq)[name = string(\"wv\")];\n", d * d, b, s);
    mil.push_str(&wv);
    let wv1 = format!("        tensor<fp16, [1, {}, 1, 1]> wv1 = slice_by_size(x = wv, begin = b0, size = sw1)[name = string(\"wv1\")];\n", d * d);
    mil.push_str(&wv1);
    let Wv = format!("        tensor<fp16, [1, 1, {}, {}]> Wv = reshape(shape = wrs, x = wv1)[name = string(\"Wv\")];\n", d, d);
    mil.push_str(&Wv);

    // --- Transpose activations: [1, D, B, S] -> [B, D, 1, S] -> [B, 1, D, S] -> [B, 1, S, D] ---
    let tb = format!("        tensor<int32, [4]> tb = const()[name = string(\"tb\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&tb);
    let ab = format!("        tensor<fp16, [{}, {}, 1, {}]> ab = transpose(perm = tb, x = act)[name = string(\"ab\")];\n", b, d, s);
    mil.push_str(&ab);
    let rb = format!("        tensor<int32, [4]> rb = const()[name = string(\"rb\"), val = tensor<int32, [4]>([{}, 1, {}, {}])];\n", b, d, s);
    mil.push_str(&rb);
    let rb2 = format!("        tensor<fp16, [{}, 1, {}, {}]> rb2 = reshape(shape = rb, x = ab)[name = string(\"rb2\")];\n", b, d, s);
    mil.push_str(&rb2);
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    let xt = format!("        tensor<fp16, [{}, 1, {}, {}]> xt = transpose(perm = pm, x = rb2)[name = string(\"xt\")];\n", b, s, d);
    mil.push_str(&xt);

    // --- 3 matmuls: [B, 1, S, D] @ [1, 1, D, D] -> [B, 1, S, D] ---
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    let mQ = format!("        tensor<fp16, [{}, 1, {}, {}]> Q = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = Wq)[name = string(\"mQ\")];\n", b, s, d);
    mil.push_str(&mQ);
    let mK = format!("        tensor<fp16, [{}, 1, {}, {}]> K = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = Wk)[name = string(\"mK\")];\n", b, s, d);
    mil.push_str(&mK);
    let mV = format!("        tensor<fp16, [{}, 1, {}, {}]> V = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = Wv)[name = string(\"mV\")];\n", b, s, d);
    mil.push_str(&mV);

    // --- Q + bias + relu ---
    let bb = format!("        tensor<int32, [4]> bb = const()[name = string(\"bb\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", d + 3 * d * d);
    mil.push_str(&bb);
    let bias = format!("        tensor<fp16, [1, {}, {}, {}]> bias = slice_by_size(x = xh, begin = bb, size = sa)[name = string(\"bias\")];\n", d, b, s);
    mil.push_str(&bias);

    // Transpose Q back: [B, 1, S, D] -> [B, 1, D, S] -> [B, D, 1, S] -> [1, D, B, S]
    let Qt = format!("        tensor<fp16, [{}, 1, {}, {}]> Qt = transpose(perm = pm, x = Q)[name = string(\"Qt\")];\n", b, d, s);
    mil.push_str(&Qt);
    let Qr = format!("        tensor<fp16, [{}, {}, 1, {}]> Qr = reshape(shape = rb, x = Qt)[name = string(\"Qr\")];\n", b, d, s);
    mil.push_str(&Qr);
    let Qb = format!("        tensor<fp16, [1, {}, {}, {}]> Qb = transpose(perm = tb, x = Qr)[name = string(\"Qb\")];\n", d, b, s);
    mil.push_str(&Qb);

    let ab2 = format!("        tensor<fp16, [1, {}, {}, {}]> ab2 = add(x = Qb, y = bias)[name = string(\"addb\")];\n", d, b, s);
    mil.push_str(&ab2);
    let rl = format!(
        "        tensor<fp16, [1, {}, {}, {}]> rl = relu(x = ab2)[name = string(\"relu\")];\n",
        d, b, s
    );
    mil.push_str(&rl);

    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    let y = format!("        tensor<fp32, [1, {}, {}, {}]> y = cast(dtype = to32, x = rl)[name = string(\"cout\")];\n", d, b, s);
    mil.push_str(&y);
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

// ============================================================
// Helpers
// ============================================================

fn make_input_data(size_bytes: usize) -> Vec<u8> {
    let num_floats = size_bytes / 4;
    let data: Vec<f32> = (0..num_floats)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn benchmark_program(
    mil: &str,
    in_b: usize,
    out_b: usize,
    warmup: usize,
    iters: usize,
) -> Option<String> {
    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        mil,
        &[],
        &[],
        &[],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => e,
        Err(e) => {
            println!("COMPILE_FAIL: {}", e);
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
    Some(format!(
        "RESULT: in={}B write={:.1} eval={:.1} read={:.1} total={:.1}",
        in_b, write_us, eval_us, read_us, total_us
    ))
}

// ============================================================
// Subprocess test functions
// ============================================================

fn test_single_matmul() {
    let _ = rustane::init();
    let dim = 64;
    let seq = 64;
    let batch = 1;
    let total_ch = dim + dim * dim;
    let in_b = total_ch * batch * seq * 4;
    let out_b = dim * batch * seq * 4;
    let mil = single_dynamic_matmul_mil(batch, dim, seq);
    if let Some(result) = benchmark_program(&mil, in_b, out_b, 20, 200) {
        println!("{}", result);
    }
}

fn test_fused_3matmul() {
    let _ = rustane::init();
    let dim = 64;
    let seq = 64;
    let batch = 1;
    let total_ch = dim + 3 * dim * dim + dim;
    let in_b = total_ch * batch * seq * 4;
    let out_b = dim * batch * seq * 4;
    let mil = fused_3matmul_mil(batch, dim, seq);
    if let Some(result) = benchmark_program(&mil, in_b, out_b, 20, 200) {
        println!("{}", result);
    }
}

fn test_fused_3matmul_batched() {
    let _ = rustane::init();
    let args: Vec<String> = env::args().collect();
    let batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let dim = 64;
    let seq = 64;
    let total_ch = dim + 3 * dim * dim + dim;
    let in_b = total_ch * batch * seq * 4;
    let out_b = dim * batch * seq * 4;
    let mil = fused_3matmul_mil(batch, dim, seq);
    if let Some(result) = benchmark_program(&mil, in_b, out_b, 20, 200) {
        println!("{}", result);
    }
}

fn test_cpu_baseline() {
    let dim = 64;
    let seq = 64;
    let iters = 5000;

    let act: Vec<f32> = (0..dim * seq)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    let wq: Vec<f32> = (0..dim * dim)
        .map(|i| if i % (dim + 1) == 0 { 0.5 } else { 0.0 })
        .collect();
    let wk: Vec<f32> = wq.clone();
    let wv: Vec<f32> = wq.clone();
    let bias: Vec<f32> = (0..dim).map(|_| 0.1).collect();
    let mut q = vec![0.0f32; dim * seq];

    let start = Instant::now();
    for _ in 0..iters {
        // Q = act @ Wq (D*S × D @ D × D)
        for s in 0..seq {
            for d in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    sum += act[s * dim + k] * wq[k * dim + d];
                }
                q[s * dim + d] = sum;
            }
        }
        // K = act @ Wk (same cost)
        // V = act @ Wv (same cost)
        // Total: 3 matmuls + bias + relu
        for i in 0..(dim * seq) {
            let val = q[i] + bias[i % dim];
            q[i] = if val > 0.0 { val } else { 0.0 };
        }
    }
    let elapsed_us = start.elapsed().as_micros() as f64;
    let per_iter = elapsed_us / iters as f64;
    // We measured 1 matmul + relu; 3 matmuls = 1 matmul + 2 extra matmuls
    let one_matmul_us = (per_iter - 0.0) / 3.0; // approximate
    let three_matmul_relu_us = one_matmul_us * 3.0 + (per_iter - one_matmul_us * 3.0);

    println!("  CPU D={}, S={}, iterations={}", dim, seq, iters);
    println!("  1 matmul + relu:  {:.1} us/sample", per_iter);
    println!(
        "  3 matmuls + relu: {:.1} us/sample (estimated: 3x matmul time)",
        per_iter * 3.0
    );
    println!(
        "  Throughput:       {:.0} samples/sec",
        1_000_000.0 / (per_iter * 3.0)
    );
}
