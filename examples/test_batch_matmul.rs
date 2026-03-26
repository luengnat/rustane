//! Minimal test: does batched matmul work on ANE?
//! Tests progressively more complex batched matmul programs.
//! Run each in subprocess since ANE crashes are SIGSEGV.
//!
//! Usage: cargo run --example test_batch_matmul

use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test_batch_matmul <test_id> [dim] [seq] [batch]");
        eprintln!(
            "  test_id: 1=compile_only, 2=compile+eval, 3=batched_elementwise, 4=batched_matmul_v1"
        );
        std::process::exit(1);
    }

    let test_id: u32 = args[1].parse().unwrap_or(0);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let seq: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let batch: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    match test_id {
        1 => test_compile_only(dim, seq, batch),
        2 => test_compile_and_eval(dim, seq, batch),
        3 => test_batched_elementwise(dim, seq, batch),
        4 => test_batched_matmul_simple(dim, seq, batch),
        5 => test_batched_dynamic_matmul(dim, seq, batch),
        _ => {
            eprintln!("Unknown test_id: {}", test_id);
            std::process::exit(1);
        }
    }
}

fn test_compile_only(dim: usize, seq: usize, batch: usize) {
    // Test 1: Just compile, don't eval — isolates compile vs eval crash
    let total_ch = dim + dim * dim;
    let in_b = total_ch * batch * seq * 4;
    let out_b = dim * batch * seq * 4;

    let mil = batched_dynamic_matmul_mil(batch, dim, seq);
    eprintln!(
        "Test 1: Compile-only B={}, D={}, S={}, input={}KB",
        batch,
        dim,
        seq,
        in_b as f64 / 1024.0
    );

    match rustane::wrapper::ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[in_b], &[out_b])
    {
        Ok(_) => {
            eprintln!("COMPILE_OK");
            println!("COMPILE_OK: B={}, D={}, S={}", batch, dim, seq);
        }
        Err(e) => {
            eprintln!("COMPILE_FAIL: {}", e);
            println!("COMPILE_FAIL: {}", e);
        }
    }
}

fn test_compile_and_eval(dim: usize, seq: usize, batch: usize) {
    // Test 2: Compile + single eval
    let total_ch = dim + dim * dim;
    let in_b = total_ch * batch * seq * 4;
    let out_b = dim * batch * seq * 4;

    let mil = batched_dynamic_matmul_mil(batch, dim, seq);
    eprintln!(
        "Test 2: Compile+Eval B={}, D={}, S={}, input={}KB",
        batch,
        dim,
        seq,
        in_b as f64 / 1024.0
    );

    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &[],
        &[],
        &[],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => {
            eprintln!("COMPILE_OK");
            e
        }
        Err(e) => {
            eprintln!("COMPILE_FAIL: {}", e);
            println!("COMPILE_FAIL: {}", e);
            return;
        }
    };

    let data = make_input_data(in_b);
    eprintln!("Writing input...");
    exec.write_input(0, &data).unwrap();
    eprintln!("Evaluating...");
    let start = Instant::now();
    match exec.eval() {
        Ok(_) => {
            let elapsed = start.elapsed().as_micros();
            eprintln!("EVAL_OK: {}μs", elapsed);
            println!(
                "EVAL_OK: B={}, D={}, S={}, eval={}μs",
                batch, dim, seq, elapsed
            );
        }
        Err(e) => {
            eprintln!("EVAL_FAIL: {}", e);
            println!("EVAL_FAIL: {}", e);
        }
    }
}

fn test_batched_elementwise(dim: usize, seq: usize, batch: usize) {
    // Test 3: Simple batched element-wise (known to work)
    let in_b = batch * dim * seq * 4;
    let out_b = batch * dim * seq * 4;

    let mil = batched_elementwise_mil(batch, dim, seq);
    eprintln!(
        "Test 3: Batched element-wise B={}, D={}, S={}",
        batch, dim, seq
    );

    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &[],
        &[],
        &[],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => {
            eprintln!("COMPILE_OK");
            e
        }
        Err(e) => {
            eprintln!("COMPILE_FAIL: {}", e);
            println!("COMPILE_FAIL: {}", e);
            return;
        }
    };

    let data = make_input_data(in_b);
    exec.write_input(0, &data).unwrap();

    let start = Instant::now();
    exec.eval().unwrap();
    let eval_us = start.elapsed().as_micros();
    let _out = exec.read_output_vec(0).unwrap();

    // Check output is non-zero (we wrote non-zero input)
    let non_zero = _out.iter().filter(|&&b| b != 0).count();
    let per_sample = eval_us as f64 / batch as f64;

    eprintln!(
        "EVAL_OK: {}μs total, {:.1}μs/sample, {} non-zero bytes",
        eval_us, per_sample, non_zero
    );
    println!(
        "OK: B={}, eval={}μs, {:.1}μs/sample, {} non-zero",
        batch, eval_us, per_sample, non_zero
    );
}

fn test_batched_matmul_simple(dim: usize, seq: usize, batch: usize) {
    // Test 4: Simple batched matmul using [B, D, 1, S] input with weights as blobfile
    // This uses a WeightBlob for weights instead of packing them in input
    let in_b = batch * dim * seq * 4; // activations only
    let out_b = batch * dim * seq * 4;

    // Create identity weight blob
    let mut weights = vec![0.0f32; dim * dim];
    for i in 0..dim {
        weights[i * dim + i] = 1.0;
    }
    let weight_blob = rustane::ane::WeightBlob::from_f32(&weights, dim, dim).unwrap();
    let weight_bytes = weight_blob.as_bytes();

    let mil = batched_matmul_blob_mil(batch, dim, seq);
    eprintln!(
        "Test 4: Batched matmul with blob weights B={}, D={}, S={}",
        batch, dim, seq
    );

    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &["@model_path/weights/w.bin"],
        &[&weight_bytes],
        &[weight_bytes.len()],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => {
            eprintln!("COMPILE_OK");
            e
        }
        Err(e) => {
            eprintln!("COMPILE_FAIL: {}", e);
            println!("COMPILE_FAIL: {}", e);
            return;
        }
    };

    let data = make_input_data(in_b);
    exec.write_input(0, &data).unwrap();

    let start = Instant::now();
    match exec.eval() {
        Ok(_) => {
            let eval_us = start.elapsed().as_micros();
            let out = exec.read_output_vec(0).unwrap();
            let non_zero = out.iter().filter(|&&b| b != 0).count();
            eprintln!("EVAL_OK: {}μs, {} non-zero bytes", eval_us, non_zero);
            println!(
                "OK: B={}, eval={}μs, {:.1}μs/sample, {} non-zero",
                batch,
                eval_us,
                eval_us as f64 / batch as f64,
                non_zero
            );
        }
        Err(e) => {
            eprintln!("EVAL_FAIL: {}", e);
            println!("EVAL_FAIL: {}", e);
        }
    }
}

fn test_batched_dynamic_matmul(dim: usize, seq: usize, batch: usize) {
    // Test 5: Batched dynamic matmul with weights in input tensor
    let total_ch = dim + dim * dim;
    let in_b = total_ch * batch * seq * 4;
    let out_b = dim * batch * seq * 4;

    let mil = batched_dynamic_matmul_mil(batch, dim, seq);
    eprintln!(
        "Test 5: Batched dynamic matmul B={}, D={}, S={}, input={}KB",
        batch,
        dim,
        seq,
        in_b as f64 / 1024.0
    );

    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &[],
        &[],
        &[],
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => {
            eprintln!("COMPILE_OK");
            e
        }
        Err(e) => {
            eprintln!("COMPILE_FAIL: {}", e);
            println!("COMPILE_FAIL: {}", e);
            return;
        }
    };

    let data = make_input_data(in_b);
    exec.write_input(0, &data).unwrap();

    let start = Instant::now();
    match exec.eval() {
        Ok(_) => {
            let eval_us = start.elapsed().as_micros();
            let out = exec.read_output_vec(0).unwrap();
            let non_zero = out.iter().filter(|&&b| b != 0).count();
            eprintln!("EVAL_OK: {}μs, {} non-zero bytes", eval_us, non_zero);
            println!(
                "OK: B={}, eval={}μs, {:.1}μs/sample, {} non-zero",
                batch,
                eval_us,
                eval_us as f64 / batch as f64,
                non_zero
            );
        }
        Err(e) => {
            eprintln!("EVAL_FAIL: {}", e);
            println!("EVAL_FAIL: {}", e);
        }
    }
}

// ─── MIL Generators ────────────────────────────────────────────────────────

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
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", batch, channels, spatial));
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
    mil.push_str(&format!("        tensor<fp32, [{}, {}, 1, {}]> y = cast(dtype = to32, x = r)[name = string(\"cout\")];\n", batch, channels, spatial));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

/// Batched matmul with weights from blobfile (const weights).
/// Input: [B, D, 1, S] activations
/// Weights: [1, 1, D, D] from blobfile (broadcast across batch)
/// Output: [B, D, 1, S]
fn batched_matmul_blob_mil(batch: usize, dim: usize, seq: usize) -> String {
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
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", batch, dim, seq));

    // Load weights from blobfile
    mil.push_str(&format!(
        "        tensor<fp16, [1, 1, {}, {}]> W = const()[name = string(\"W\"), val = blobfile(name = string(\"@model_path/weights/w.bin\"))];\n",
        dim, dim
    ));

    // Reshape: [B, D, 1, S] → [B, 1, D, S]
    mil.push_str(&format!(
        "        tensor<int32, [4]> rs = const()[name = string(\"rs\"), val = tensor<int32, [4]>([{}, 1, {}, {}])];\n",
        batch, dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> xr = reshape(shape = rs, x = xh)[name = string(\"xr\")];\n",
        batch, dim, seq
    ));

    // Transpose: [B, 1, D, S] → [B, 1, S, D]
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{}, 1, {}, {}]> xt = transpose(perm = pm, x = xr)[name = string(\"xt\")];\n",
        batch, seq, dim
    ));

    // Matmul: [B, 1, S, D] @ [1, 1, D, D] → [B, 1, S, D]
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
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
        "        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([{}, {}, 1, {}])];\n",
        batch, dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, {}]> yr = reshape(shape = os, x = mt)[name = string(\"yr\")];\n",
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

/// Batched dynamic matmul with weights packed in input.
/// Input: [1, D+D*D, B, S] — batch in height dim, weights shared
/// Output: [1, D, B, S]
fn batched_dynamic_matmul_mil(batch: usize, dim: usize, seq: usize) -> String {
    let total_ch = dim + dim * dim;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n",
    );
    mil.push_str("{\n");
    // Layout: batch=1, channels=D+D*D, height=B, width=S
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, {}, {}]> x) {{\n",
        total_ch, batch, seq
    ));

    // Cast
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!("        tensor<fp16, [1, {}, {}, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, batch, seq));

    // Slice activations: [1, D, B, S]
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n", dim, batch, seq));
    mil.push_str(&format!("        tensor<fp16, [1, {}, {}, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n", dim, batch, seq));

    // Slice weights: [1, D*D, B, S] — take first batch item: [1, D*D, 1, 1]
    mil.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", dim));
    mil.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, {}, {}])];\n", dim * dim, batch, seq));
    mil.push_str(&format!("        tensor<fp16, [1, {}, {}, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n", dim * dim, batch, seq));

    // Take first spatial position of weights: [1, D*D, 1, 1]
    mil.push_str(&format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", dim, dim));
    mil.push_str(&format!("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n", dim * dim));
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n", dim * dim));
    mil.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n", dim, dim));

    // Transpose activations: [1, D, B, S] → [B, D, 1, S]
    mil.push_str("        tensor<int32, [4]> tb = const()[name = string(\"tb\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, {}]> ab = transpose(perm = tb, x = act)[name = string(\"ab\")];\n", batch, dim, seq));

    // Reshape: [B, D, 1, S] → [B, 1, D, S]
    mil.push_str(&format!("        tensor<int32, [4]> rb = const()[name = string(\"rb\"), val = tensor<int32, [4]>([{}, 1, {}, {}])];\n", batch, dim, seq));
    mil.push_str(&format!("        tensor<fp16, [{}, 1, {}, {}]> rb2 = reshape(shape = rb, x = ab)[name = string(\"rb2\")];\n", batch, dim, seq));

    // Transpose: [B, 1, D, S] → [B, 1, S, D]
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    mil.push_str(&format!("        tensor<fp16, [{}, 1, {}, {}]> xt = transpose(perm = pm, x = rb2)[name = string(\"xt\")];\n", batch, seq, dim));

    // Matmul: [B, 1, S, D] @ [1, 1, D, D] → [B, 1, S, D]
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!("        tensor<fp16, [{}, 1, {}, {}]> mm = matmul(transpose_x = bF, transpose_y = bF, x = xt, y = W)[name = string(\"mm\")];\n", batch, seq, dim));

    // Transpose: [B, 1, S, D] → [B, 1, D, S]
    mil.push_str(&format!("        tensor<fp16, [{}, 1, {}, {}]> mt = transpose(perm = pm, x = mm)[name = string(\"mt\")];\n", batch, dim, seq));

    // Reshape: [B, 1, D, S] → [B, D, 1, S]
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, {}]> mr = reshape(shape = rb, x = mt)[name = string(\"mr\")];\n", batch, dim, seq));

    // Transpose back: [B, D, 1, S] → [1, D, B, S]
    mil.push_str("        tensor<int32, [4]> tb2 = const()[name = string(\"tb2\"), val = tensor<int32, [4]>([2, 1, 0, 3])];\n");
    mil.push_str(&format!("        tensor<fp16, [1, {}, {}, {}]> ob = transpose(perm = tb2, x = mr)[name = string(\"ob\")];\n", dim, batch, seq));

    // Cast to fp32
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!("        tensor<fp32, [1, {}, {}, {}]> y = cast(dtype = to32, x = ob)[name = string(\"cout\")];\n", dim, batch, seq));

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
