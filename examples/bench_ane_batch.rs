//! ANE Batch Throughput Benchmark
//!
//! Core question: Can we avoid per-invocation overhead by batching?
//!
//! ANE tensors are 4D: [batch, channels, height, width].
//! Instead of processing one sample at [1, D, 1, S] per eval call,
//! we pack B samples into [B, D, 1, S] and process in ONE eval call.
//!
//! The ~130μs dispatch overhead is amortized over B samples.
//!
//! This tests:
//! 1. Does ANE actually support batch > 1?
//! 2. How does eval time scale with batch size?
//! 3. What's the optimal batch size for throughput?
//! 4. Write/read I/O cost scaling with batch size

use rustane::wrapper::ANECompiler;
use std::time::Instant;

// ─── MIL: Element-wise batched operation ────────────────────────────────────
// Input: [B, C, 1, S] fp32 → add(1.0) → relu → [B, C, 1, S] fp32
// Tests pure element-wise ops with batch dimension.

fn batched_elementwise_mil(batch: usize, channels: usize, spatial: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [{}, {}, 1, {}]> x) {{\n",
        batch, channels, spatial
    ));
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        batch, channels, spatial
    ));
    mil.push_str("        fp16 one = const()[name = string(\"one\"), val = fp16(1.0)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> a = add(x = xh, y = one)[name = string(\"add\")];\n",
        batch, channels, spatial
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> r = relu(x = a)[name = string(\"relu\")];\n",
        batch, channels, spatial
    ));
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [{}, {}, 1, {}]> y = cast(dtype = to32, x = r)[name = string(\"cout\")];\n",
        batch, channels, spatial
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

// ─── MIL: Batched matmul ───────────────────────────────────────────────────
// Input: [B, D+D*D, 1, S] fp32
// Slice activations [B, D, 1, S], weights [B, D*D, 1, S]
// Reshape and matmul: [1,1,B*D,S] @ [1,1,D,B*D] → not quite right...
//
// Actually, matmul on ANE with batch is tricky because the weight matrix
// must be the same for all batch items. So we use a different approach:
// - Input: [B, D, 1, S] activations (no weights in input)
// - Weights: compiled as const (baked into the program)
// - Output: [B, D, 1, S]
//
// BUT: we discovered weights can't be baked (can't update without recompile).
// So for training, we need dynamic weights.
//
// The REAL batch question is: can we do [B, D, 1, S] @ [1, D, 1, D] → [B, D, 1, S]?
// ANE broadcast: the weight [1, D, 1, D] is broadcast across B.
// This means we only need one copy of weights, and process B samples!

fn batched_matmul_const_mil(batch: usize, dim: usize, seq: usize) -> String {
    // Weight matrix as a const (fp16, [1, 1, dim, dim])
    // Input: [B, dim, 1, seq] fp32 activations only
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [{}, {}, 1, {}]> x) {{\n",
        batch, dim, seq
    ));
    // Cast to fp16
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        batch, dim, seq
    ));
    // Identity weight matrix [1, 1, D, D]
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> W = const()[name = string(\"W\"), val = tensor<fp16, [1, 1, {}, {}]>(", dim, dim, dim, dim
    ));
    // Build identity matrix values
    for i in 0..dim {
        for j in 0..dim {
            let val = if i == j { "1.0" } else { "0.0" };
            if i > 0 || j > 0 {
                mil.push_str(", ");
            }
            mil.push_str(val);
        }
    }
    mil.push_str(")];\n");
    // Reshape activations: [B, D, 1, S] → [B, 1, D, S] → transpose → [B, 1, S, D]
    mil.push_str(&format!(
        "        tensor<int32, [4]> rs = const()[name = string(\"rs\"), val = tensor<int32, [4]>([{}, 1, {}, {}])];\n",
        batch, dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> xr = reshape(shape = rs, x = xh)[name = string(\"xr\")];\n",
        batch, dim, seq
    ));
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> xt = transpose(perm = pm, x = xr)[name = string(\"xt\")];\n",
        batch, seq, dim
    ));
    // Matmul: [B, 1, S, D] @ [1, 1, D, D] → [B, 1, S, D] (W broadcasts across B)
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> yh = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = W)[name = string(\"mm\")];\n",
        batch, seq, dim
    ));
    // Transpose back: [B, 1, S, D] → [B, 1, D, S] → [B, D, 1, S]
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> yt = transpose(perm = pm, x = yh)[name = string(\"yt\")];\n",
        batch, dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([{}, {}, 1, {}])];\n",
        batch, dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> yr = reshape(shape = os, x = yt)[name = string(\"yr\")];\n",
        batch, dim, seq
    ));
    // Cast to fp32
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [{}, {}, 1, {}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n",
        batch, dim, seq
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

// ─── MIL: Batched matmul with DYNAMIC weights via input packing ────────────
// Input: [1, D + D*D, B, S] — activations and weights packed along channel dim
// The B batch items share the same weights (broadcast across spatial=B)
// Output: [1, D, B, S]
//
// This is the key architecture for training:
// - Weights are in the input tensor (dynamic, updatable per step)
// - Multiple samples processed in one eval call
// - The ~130μs dispatch overhead is amortized over B samples

fn batched_dynamic_matmul_mil(batch: usize, dim: usize, seq: usize) -> String {
    let total_ch = dim + dim * dim;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    // Input: [1, D+D*D, B, S] — batch is in the HEIGHT dimension!
    // ANE 4D = [batch, channels, height, width]
    // We use batch=1, channels=D+D*D, height=B, width=S
    // This way all B samples and weights are in one tensor
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, {}, {}]> x) {{\n",
        total_ch, batch, seq
    ));
    // Cast to fp16
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        total_ch, batch, seq
    ));
    // Slice activations: [1, D, B, S]
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str(&format!(
        "        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n",
        dim, batch, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n",
        dim, batch, seq
    ));
    // Slice weights: [1, D*D, B, S] — weights are replicated across B spatial positions
    mil.push_str(&format!(
        "        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n",
        dim
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n",
        dim * dim, batch, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n",
        dim * dim, batch, seq
    ));
    // Take first spatial position of weights: [1, D*D, 1, 1]
    // (All B positions have the same weight data)
    mil.push_str(&format!(
        "        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n",
        dim, dim
    ));
    mil.push_str(&format!(
        "        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n",
        dim * dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n",
        dim * dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n",
        dim, dim
    ));
    // Reshape activations: [1, D, B, S] → [1, 1, D*B, S] → transpose → [1, 1, S, D*B]
    mil.push_str(&format!(
        "        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n",
        dim * batch, seq
    ));
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> a2 = reshape(shape = as2, x = act)[name = string(\"a2\")];\n",
        dim * batch, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> a3 = transpose(perm = pm, x = a2)[name = string(\"a3\")];\n",
        seq, dim * batch
    ));
    // Matmul: [1,1,S,D*B] @ [1,1,D*D,D] — NO, this doesn't work for per-sample matmul
    // The matmul would mix samples. We need a different approach.
    //
    // Actually, ANE matmul with batch is:
    // [B, 1, S, D] @ [1, 1, D, D] → [B, 1, S, D] (broadcast)
    // But our data layout is [1, D, B, S], not [B, D, 1, S].
    //
    // Let's reshape to proper batch layout:
    // [1, D, B, S] → [B, D, 1, S] via transpose [2, 1, 3, 0] → no, dims don't match
    // Actually: transpose(perm=[2,1,3,0]) on [1,D,B,S] gives [B,D,S,1] — not right
    // transpose(perm=[2,1,0,3]) on [1,D,B,S] gives [B,D,1,S] — YES!
    //
    // Then reshape [B, D, 1, S] → [B, 1, D, S] → transpose → [B, 1, S, D]
    // matmul [B, 1, S, D] @ [1, 1, D, D] → [B, 1, S, D]
    // transpose → [B, 1, D, S] → reshape → [B, D, 1, S]
    // transpose back → [1, D, B, S]

    // Undo the reshape above — we'll redo properly
    // Actually let me just rewrite from activations:
    // act is [1, D, B, S], transpose to [B, D, 1, S]
    mil.push_str("        tensor<int32, [4]> tb = const()[name = string(\"tb\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> ab = transpose(perm = tb, x = act)[name = string(\"ab\")];\n",
        batch, dim, seq
    ));
    // Reshape: [B, D, 1, S] → [B, 1, D, S]
    mil.push_str(&format!(
        "        tensor<int32, [4]> rb = const()[name = string(\"rb\"), val = tensor<int32, [4]>([{}, 1, {}, {}])];\n",
        batch, dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> rb2 = reshape(shape = rb, x = ab)[name = string(\"rb2\")];\n",
        batch, dim, seq
    ));
    // Transpose: [B, 1, D, S] → [B, 1, S, D]
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> xt = transpose(perm = pm, x = rb2)[name = string(\"xt\")];\n",
        batch, seq, dim
    ));
    // Matmul: [B, 1, S, D] @ [1, 1, D, D] → [B, 1, S, D] (W broadcasts)
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> mm = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = W)[name = string(\"mm\")];\n",
        batch, seq, dim
    ));
    // Transpose: [B, 1, S, D] → [B, 1, D, S]
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> mt = transpose(perm = pm, x = mm)[name = string(\"mt\")];\n",
        batch, dim, seq
    ));
    // Reshape: [B, 1, D, S] → [B, D, 1, S]
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> mr = reshape(shape = rb, x = mt)[name = string(\"mr\")];\n",
        batch, dim, seq
    ));
    // Transpose back: [B, D, 1, S] → [1, D, B, S]
    mil.push_str("        tensor<int32, [4]> tb2 = const()[name = string(\"tb2\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> ob = transpose(perm = tb2, x = mr)[name = string(\"ob\")];\n",
        dim, batch, seq
    ));
    // Cast output to fp32
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, {}, {}]> y = cast(dtype = to32, x = ob)[name = string(\"cout\")];\n",
        dim, batch, seq
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn make_input_data(size_bytes: usize) -> Vec<u8> {
    let num_floats = size_bytes / 4;
    let data: Vec<f32> = (0..num_floats)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn print_section(title: &str) {
    println!("\n{}", "=".repeat(72));
    println!("  {}", title);
    println!("{}", "=".repeat(72));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let warmup = 10;
    let iterations = 500;
    let dim = 64;
    let seq = 64;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║        ANE BATCH STREAMING BENCHMARK                               ║");
    println!("║                                                                    ║");
    println!("║  Question: Can we avoid per-invocation overhead by batching?       ║");
    println!("║  Approach: Pack B samples into one tensor, process in 1 eval call  ║");
    println!(
        "║  Dim={}, Seq={}, Warmup={}, Iters={}                                  ║",
        dim, seq, warmup, iterations
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 1: Batched element-wise — does ANE support batch > 1?
    // ═══════════════════════════════════════════════════════════════════════════
    print_section("TEST 1: Batched Element-Wise (add+relu) — Does batch > 1 work?");
    println!(
        "\n  {:<8} {:>10} {:>8} {:>8} {:>8} {:>12} {:>10} {:>14}",
        "Batch", "Input(KB)", "Wr(μs)", "Eval(μs)", "Rd(μs)", "Tot(μs)", "μs/sample", "Samples/sec"
    );
    println!("  {}", "-".repeat(92));

    let batch_sizes = [1, 2, 4, 8, 16, 32];
    let mut last_total_us = 0.0_f64;

    for &batch in &batch_sizes {
        let ch = dim;
        let in_b = batch * ch * seq * 4;
        let out_b = batch * ch * seq * 4;
        let mil = batched_elementwise_mil(batch, ch, seq);
        let data = make_input_data(in_b);

        let compile_start = Instant::now();
        let mut exec =
            match ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[in_b], &[out_b]) {
                Ok(e) => e,
                Err(e) => {
                    println!("  {:<8} COMPILE FAIL: {}", batch, e);
                    continue;
                }
            };
        let compile_ms = compile_start.elapsed().as_millis();

        // Warmup
        for _ in 0..warmup {
            exec.write_input(0, &data).unwrap();
            exec.eval().unwrap();
            let _ = exec.read_output_vec(0).unwrap();
        }

        // Measure write
        let w_start = Instant::now();
        for _ in 0..iterations {
            exec.write_input(0, &data).unwrap();
        }
        let write_us = w_start.elapsed().as_micros() as f64 / iterations as f64;

        // Measure eval
        exec.write_input(0, &data).unwrap();
        let e_start = Instant::now();
        for _ in 0..iterations {
            exec.eval().unwrap();
        }
        let eval_us = e_start.elapsed().as_micros() as f64 / iterations as f64;

        // Measure read
        let r_start = Instant::now();
        for _ in 0..iterations {
            let _ = exec.read_output_vec(0).unwrap();
        }
        let read_us = r_start.elapsed().as_micros() as f64 / iterations as f64;

        let total_us = write_us + eval_us + read_us;
        let per_sample = total_us / batch as f64;
        let samples_per_sec = 1_000_000.0 / per_sample;
        let speedup = if last_total_us > 0.0 {
            last_total_us / per_sample
        } else {
            1.0
        };

        println!(
            "  {:<8} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10.1} {:>10.1} {:>12.0}  [{:>3}ms, {:>4.1}x vs B=1]",
            batch, in_b as f64 / 1024.0,
            write_us, eval_us, read_us, total_us, per_sample, samples_per_sec,
            compile_ms, speedup
        );

        if batch == 1 {
            last_total_us = per_sample;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 2: Batched matmul with const weights
    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 2: SKIPPED — batched matmul with const weights crashes (SIGSEGV)
    // The inline fp16 identity matrix values in MIL cause eval crash.
    // See Test 3 for batched matmul with dynamic weights (the training approach).
    // ═══════════════════════════════════════════════════════════════════════════
    print_section("TEST 2: SKIPPED — const-weight matmul crashes on ANE eval");
    println!("  (batched dynamic matmul in Test 3 is the relevant test for training)");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 3: Batched dynamic matmul (weights in input)
    // ═══════════════════════════════════════════════════════════════════════════
    print_section("TEST 3: Batched Dynamic Matmul (weights in input tensor)");
    println!("\n  Input: [1, D+D*D, B, S] — B samples share weights, one eval call");
    println!(
        "  {:<8} {:>10} {:>8} {:>8} {:>8} {:>12} {:>10} {:>14}",
        "Batch", "Input(KB)", "Wr(μs)", "Eval(μs)", "Rd(μs)", "Tot(μs)", "μs/sample", "Samples/sec"
    );
    println!("  {}", "-".repeat(92));

    for &batch in &batch_sizes {
        let total_ch = dim + dim * dim;
        let in_b = total_ch * batch * seq * 4;
        let out_b = dim * batch * seq * 4;
        let mil = batched_dynamic_matmul_mil(batch, dim, seq);
        let data = make_input_data(in_b);

        let compile_start = Instant::now();
        let mut exec =
            match ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[in_b], &[out_b]) {
                Ok(e) => e,
                Err(e) => {
                    println!("  {:<8} COMPILE FAIL: {}", batch, e);
                    continue;
                }
            };
        let compile_ms = compile_start.elapsed().as_millis();

        // Warmup
        for _ in 0..warmup {
            exec.write_input(0, &data).unwrap();
            exec.eval().unwrap();
            let _ = exec.read_output_vec(0).unwrap();
        }

        let w_start = Instant::now();
        for _ in 0..iterations {
            exec.write_input(0, &data).unwrap();
        }
        let write_us = w_start.elapsed().as_micros() as f64 / iterations as f64;

        exec.write_input(0, &data).unwrap();
        let e_start = Instant::now();
        for _ in 0..iterations {
            exec.eval().unwrap();
        }
        let eval_us = e_start.elapsed().as_micros() as f64 / iterations as f64;

        let r_start = Instant::now();
        for _ in 0..iterations {
            let _ = exec.read_output_vec(0).unwrap();
        }
        let read_us = r_start.elapsed().as_micros() as f64 / iterations as f64;

        let total_us = write_us + eval_us + read_us;
        let per_sample = total_us / batch as f64;
        let samples_per_sec = 1_000_000.0 / per_sample;

        println!(
            "  {:<8} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10.1} {:>10.1} {:>12.0}  [{:>3}ms]",
            batch,
            in_b as f64 / 1024.0,
            write_us,
            eval_us,
            read_us,
            total_us,
            per_sample,
            samples_per_sec,
            compile_ms
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 4: B=1 comparison — single-sample throughput baseline
    // ═══════════════════════════════════════════════════════════════════════════
    print_section("TEST 4: B=1 Baseline — How fast is single-sample processing?");
    println!("\n  Processing 1000 samples one at a time vs batched:");
    println!();

    let n_samples = 1000;

    // B=1: 1000 eval calls
    let b1_mil = batched_elementwise_mil(1, dim, seq);
    let b1_in = dim * seq * 4;
    let b1_out = dim * seq * 4;
    let b1_data = make_input_data(b1_in);
    let mut b1_exec =
        ANECompiler::new().compile_multi(&b1_mil, &[], &[], &[], &[b1_in], &[b1_out])?;

    let start = Instant::now();
    for _ in 0..n_samples {
        b1_exec.write_input(0, &b1_data).unwrap();
        b1_exec.eval().unwrap();
        let _ = b1_exec.read_output_vec(0).unwrap();
    }
    let b1_total_ms = start.elapsed().as_millis();
    let b1_per_sample_us = b1_total_ms as f64 * 1000.0 / n_samples as f64;

    println!(
        "  B=1 ({} eval calls):   {:.0}ms total, {:.1}μs/sample, {:.0} samples/sec",
        n_samples,
        b1_total_ms as f64,
        b1_per_sample_us,
        1_000_000.0 / b1_per_sample_us
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════
    print_section("SUMMARY: Batch Streaming Analysis");
    println!(
        r#"
  The ANE ~130μs dispatch overhead dominates everything.
  Batching amortizes this cost across B samples.

  ┌────────────────────────────────────────────────────────────────────┐
  │  B=1:   ~130μs/sample  (dispatch overhead dominates)              │
  │  B=8:   ~16μs/sample?   (if ANE supports batch > 1)              │
  │  B=32:  ~4μs/sample?    (8× more data but 32× amortization)     │
  │                                                                    │
  │  I/O scales linearly: B× more data = B× write/read time          │
  │  But I/O is cheap (~1-14μs) vs eval (~130μs)                     │
  │                                                                    │
  │  Key: if eval doesn't increase much with B, batching is huge     │
  └────────────────────────────────────────────────────────────────────┘

  Training implications:
  • During training, process B sequences per forward pass
  • Gradient accumulation across batch = 1 eval for B samples
  • Effective throughput = B × (samples per eval call)
  • At B=8 with fusion: potentially 50,000+ samples/sec
"#
    );

    Ok(())
}
