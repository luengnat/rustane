//! Single ANE constraint test — run via subprocess from test_ane_constraints.
//!
//! Usage: RUSTANE_TEST_NAME=concat_basic ./test_ane_constraint
//! Prints "OK detail" on success, error to stderr on failure.

use std::fmt::Write;
use std::time::Instant;

fn main() {
    let test_name = std::env::var("RUSTANE_TEST_NAME").unwrap_or_default();

    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    match test_name.as_str() {
        "concat_basic" => test_concat(),
        "gelu_basic" => test_gelu(),
        "matmul_transpose_named" => test_matmul_transpose(),
        "conv_bias_basic" => test_conv_bias(),
        "min_surface_tiny" => test_min_surface(1),
        "min_surface_small" => test_min_surface(8),
        "min_surface_ok" => test_min_surface(16),
        "blobfile_offset_64" => test_blobfile_offset(64),
        "blobfile_offset_128" => test_blobfile_offset(128),
        "multi_output_uniform" => test_multi_output(true),
        "multi_output_nonuniform" => test_multi_output(false),
        "multi_output_alpha" => test_multi_output_order(true),
        "multi_output_reverse" => test_multi_output_order(false),
        "conv_4k_channels" => test_large_channels(4096),
        "conv_16k_channels" => test_large_channels(16384),
        "conv_32k_channels" => test_large_channels(32768),
        "conv_64x64" => test_conv_vs_matmul(64, 64, "conv"),
        "matmul_64x64" => test_conv_vs_matmul(64, 64, "matmul"),
        "conv_128x384" => test_conv_vs_matmul(128, 384, "conv"),
        "matmul_128x384" => test_conv_vs_matmul(128, 384, "matmul"),
        "fused_ffn_small" => test_fused_ffn(64, 128, false),
        "fused_ffn_medium" => test_fused_ffn(128, 256, false),
        "fused_ffn_taps_small" => test_fused_ffn(64, 128, true),
        "layer_norm_basic" => test_layer_norm(false),
        "layer_norm_with_weight" => test_layer_norm(true),
        "rmsnorm_trick_basic" => test_rmsnorm_trick(false),
        "rmsnorm_trick_with_weight" => test_rmsnorm_trick(true),
        "rmsnorm_manual_basic" => test_rmsnorm_manual(),
        "softmax_basic" => test_softmax(),
        "sigmoid_basic" => test_sigmoid(),
        "dual_conv_basic" => test_dual_conv(),
        "qkv_fused_basic" => test_qkv_fused(),
        "multi_input_add" => test_multi_input_add(),
        // Phase 3: op variant tests
        "mb_matmul" => test_mb_matmul(),
        "mb_softmax" => test_mb_softmax(),
        "mb_concat" => test_mb_concat(),
        "mb_layer_norm" => test_mb_layer_norm(),
        "mb_reduce_sum" => test_mb_reduce_sum(),
        "mb_transpose" => test_mb_transpose(),
        "mb_reshape" => test_mb_reshape(),
        "mb_slice_by_size" => test_mb_slice_by_size(),
        "op_sub" => test_sub(),
        "op_pow" => test_pow(),
        "op_transpose" => test_transpose(),
        "op_reshape" => test_reshape(),
        "op_slice_by_size" => test_slice_by_size(),
        "op_clamp" => test_clamp(),
        "op_exp" => test_exp(),
        "op_log" => test_log(),
        "op_abs" => test_abs(),
        "op_tanh" => test_tanh(),
        "op_relu" => test_relu(),
        "op_leaky_relu" => test_leaky_relu(),
        "op_conv_nobias" => test_conv_no_bias(),
        "op_conv3x1" => test_conv3x1(),
        "seq_20" => test_seq_boundary(20),
        "seq_24" => test_seq_boundary(24),
        "seq_28" => test_seq_boundary(28),
        "seq_32" => test_seq_boundary(32),
        "seq_48" => test_seq_boundary(48),
        "seq_64" => test_seq_boundary(64),
        "seq_128" => test_seq_boundary(128),
        "seq_256" => test_seq_boundary(256),
        "seq_512" => test_seq_boundary(512),
        _ => {
            eprintln!("Unknown test: {}", test_name);
            std::process::exit(1);
        }
    }
}

use rustane::ane::WeightBlob;

fn make_blob(data: &[f32], rows: usize, cols: usize) -> WeightBlob {
    WeightBlob::from_f32(data, rows, cols).unwrap()
}

fn make_weight_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 100) as f32) * 0.01 - 0.5).collect()
}

/// Generate fp32 input bytes (4 bytes per element).
fn input_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i * 3) as f32).to_le_bytes())
        .flatten()
        .collect()
}

fn compile_eval(
    mil: &str,
    weights: &[(&str, &WeightBlob)],
    input_sizes: &[usize],
    output_sizes: &[usize],
    input_data: Option<&[u8]>,
) -> (f64, f64) {
    let mut req = rustane::ane::ANECompileRequest::new(mil, input_sizes, output_sizes);
    for (name, blob) in weights {
        req = req.with_weight_blob(*name, blob);
    }

    let t0 = Instant::now();
    let mut executor = match req.compile() {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(2);
        }
    };
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if let Some(data) = input_data {
        if let Err(e) = executor.write_input(0, data) {
            eprintln!("write_input: {e}");
            std::process::exit(3);
        }
    }

    let t1 = Instant::now();
    if let Err(e) = executor.eval() {
        eprintln!("eval: {e}");
        std::process::exit(4);
    }
    let eval_ms = t1.elapsed().as_secs_f64() * 1000.0;

    if !output_sizes.is_empty() {
        let mut out_bytes = vec![0u8; output_sizes[0]];
        if let Err(e) = executor.read_output(0, &mut out_bytes) {
            eprintln!("read_output: {e}");
            std::process::exit(5);
        }
        let out: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if out.iter().all(|&x| x == 0.0) {
            eprintln!("all zeros");
            std::process::exit(6);
        }
        if out.iter().any(|x| x.is_nan() || x.is_infinite()) {
            eprintln!("nan/inf");
            std::process::exit(7);
        }
        eprintln!(
            "sample: [{:.4}, {:.4}, {:.4}, {:.4}]",
            out[0], out[1], out[2], out[3]
        );
    }

    (compile_ms, eval_ms)
}

// ============================================================
// MIL helpers — build strings without format! escaping issues
// ============================================================

/// Standard MIL program header (program 1.3 with buildInfo).
/// ANE requires {{ and }} only on the outermost dict wrapper.
/// Inner key-value pairs use single braces. Matches programs.rs line 214.
const MIL_HEADER: &str = "program(1.3)\n\
[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n\
{\n";

/// Standard conv1x1 boilerplate parameters (shared across all conv tests).
const CONV_PARAMS: &str = "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n\
        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n\
        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n\
        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n";

/// Cast to fp16 at function entry and back to fp32 at exit.
const CONV_FP32_IO: &str =
    "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n\
         string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n";

/// Build a complete conv1x1 MIL program using String::push_str (no format! escaping).
/// Uses fp32 I/O with internal fp16 cast (matching programs.rs pattern).
fn conv1x1_mil(seq: usize, in_dim: usize, out_dim: usize, wname: &str, suffix: &str) -> String {
    let mut mil = String::new();
    mil.push_str(MIL_HEADER);
    write!(
        mil,
        "    func main<ios18>(tensor<fp32, [1, {in_dim}, 1, {seq}]> x) {{\n",
    )
    .unwrap();
    mil.push_str(CONV_FP32_IO);
    write!(
        mil,
        "        tensor<fp16, [1, {in_dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        mil,
        "        tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W = const()[name = string(\"{wname}\"), val = tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{wname}.bin\"), offset = uint64(64)))];\n",
    )
    .unwrap();
    mil.push_str(CONV_PARAMS);
    write!(
        mil,
        "        tensor<fp16, [1, {out_dim}, 1, {seq}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"{wname}\")];\n",
    )
    .unwrap();
    if !suffix.is_empty() {
        mil.push_str(suffix);
    } else {
        write!(
            mil,
            "        tensor<fp32, [1, {out_dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        )
        .unwrap();
    }
    mil.push_str("    } -> (y);\n}\n");
    mil
}

/// Build a MIL program with a custom function body (fp16 I/O, no weights).
fn mil_fp16_program(input_sig: &str, body: &str, outputs: &str) -> String {
    let mut mil = String::new();
    mil.push_str(MIL_HEADER);
    write!(mil, "    func main<ios18>({input_sig}) {{\n").unwrap();
    mil.push_str(body);
    write!(mil, "    }} -> ({outputs});\n").unwrap();
    mil.push_str("}\n");
    mil
}

/// Build a MIL program with fp32 I/O (cast to fp16 inside).
fn mil_fp32_program(input_sig: &str, body: &str, outputs: &str) -> String {
    let mut mil = String::new();
    mil.push_str(MIL_HEADER);
    write!(mil, "    func main<ios18>({input_sig}) {{\n").unwrap();
    mil.push_str(CONV_FP32_IO);
    mil.push_str(body);
    write!(mil, "    }} -> ({outputs});\n").unwrap();
    mil.push_str("}\n");
    mil
}

// ============================================================
// Tests — MIL IR Restrictions
// ============================================================

fn test_concat() {
    let dim = 64;
    let seq = 32;
    // Use fp32 I/O like programs.rs — concat doesn't need weights
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> ones = const()[name = string(\"ones\"), val = tensor<fp16, [1, {dim}, 1, {seq}]>(1.0)];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {doubled}, 1, {seq}]> cat16 = concat(axis = 1, x16, ones)[name = string(\"concat\")];\n",
        doubled = dim * 2,
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {doubled}, 1, {seq}]> y = cast(dtype = to_fp32, x = cat16)[name = string(\"cast_out\")];\n",
        doubled = dim * 2,
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[dim * 2 * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_gelu() {
    let dim = 64;
    let seq = 32;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = gelu(x = x16)[name = string(\"gelu\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_matmul_transpose() {
    let dim = 32;
    let seq = 32;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        int32 tr = const()[name = string(\"tr\"), val = int32(1)];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, 1, {dim}, {dim}]> W = const()[name = string(\"W\"), val = tensor<fp16, [{dim}, 1, {dim}, {dim}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = matmul(x = x16, y = W, transpose_y = tr)[name = string(\"mm\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim * dim, dim);
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[dim * seq * 4],
        &[dim * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_conv_bias() {
    let in_dim = 64;
    let out_dim = 32;
    let seq = 32;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {in_dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {out_dim}, 1, 1]> b = const()[name = string(\"b\"), val = tensor<fp16, [1, {out_dim}, 1, 1]>(0.0)];\n",
    )
    .unwrap();
    body.push_str(CONV_PARAMS);
    write!(
        body,
        "        tensor<fp16, [1, {out_dim}, 1, {seq}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, bias = b, weight = W, x = x16)[name = string(\"conv\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {out_dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {in_dim}, 1, {seq}]> x"),
        &body,
        "y",
    );
    let w = make_weight_data(out_dim * in_dim);
    let blob = make_blob(&w, out_dim, in_dim);
    let input = input_bytes(in_dim * seq);
    compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[in_dim * seq * 4],
        &[out_dim * seq * 4],
        Some(&input),
    );
}

// ============================================================
// Memory and I/O Constraints
// ============================================================

fn test_min_surface(seq: usize) {
    let dim = 64;
    let mil = conv1x1_mil(seq, dim, dim, "W", "");
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim, dim);
    // fp32 I/O: 4 bytes per element
    let input_sz = dim * seq * 4;
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/W.bin", &blob)],
        &[input_sz],
        &[input_sz],
        Some(&input),
    );
    println!(
        "OK seq={} {}bytes compile={:.0}ms eval={:.1}ms",
        seq, input_sz, c, e
    );
}

fn test_blobfile_offset(offset: u64) {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({offset})))];\n",
    )
    .unwrap();
    body.push_str(CONV_PARAMS);
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim, dim);
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[dim * seq * 4],
        &[dim * seq * 4],
        Some(&input),
    );
    println!("OK offset={} compile={:.0}ms eval={:.1}ms", offset, c, e);
}

fn test_multi_output(uniform: bool) {
    let dim = 64;
    let seq = 16;
    let out1 = dim * seq * 4;
    let out2 = if uniform { out1 } else { 32 * seq * 4 };
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
    )
    .unwrap();
    body.push_str(CONV_PARAMS);
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> a16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"a\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> b16 = mul(x = a16, y = a16)[name = string(\"b\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> a = cast(dtype = to_fp32, x = a16)[name = string(\"cast_a\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> b = cast(dtype = to_fp32, x = b16)[name = string(\"cast_b\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"),
        &body,
        "a, b",
    );
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim, dim);
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[dim * seq * 4],
        &[out1, out2],
        Some(&input),
    );
    println!(
        "OK uniform={} out1={} out2={} compile={:.0}ms eval={:.1}ms",
        uniform, out1, out2, c, e
    );
}

fn test_multi_output_order(alphabetical: bool) {
    let dim = 64;
    let seq = 16;
    let out_size = dim * seq * 4;
    let (name1, name2) = if alphabetical { ("a", "z") } else { ("z", "a") };
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
    )
    .unwrap();
    body.push_str(CONV_PARAMS);
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> {name1}16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"out1\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> {name2}16 = mul(x = {name1}16, y = {name1}16)[name = string(\"out2\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> {name1} = cast(dtype = to_fp32, x = {name1}16)[name = string(\"cast1\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> {name2} = cast(dtype = to_fp32, x = {name2}16)[name = string(\"cast2\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"),
        &body,
        &format!("{name1}, {name2}"),
    );
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim, dim);
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[dim * seq * 4],
        &[out_size, out_size],
        Some(&input),
    );
    println!(
        "OK alpha=({},{}) compile={:.0}ms eval={:.1}ms",
        name1, name2, c, e
    );
}

// ============================================================
// Performance Characteristics
// ============================================================

fn test_large_channels(out_ch: usize) {
    let in_ch = 64;
    let seq = 16;
    let mil = conv1x1_mil(seq, in_ch, out_ch, "W", "");
    let w = make_weight_data(out_ch * in_ch);
    let blob = make_blob(&w, out_ch, in_ch);
    let input = input_bytes(in_ch * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/W.bin", &blob)],
        &[in_ch * seq * 4],
        &[out_ch * seq * 4],
        Some(&input),
    );
    println!("OK out_ch={} compile={:.0}ms eval={:.1}ms", out_ch, c, e);
}

fn test_conv_vs_matmul(in_dim: usize, out_dim: usize, op: &str) {
    let seq = 32;
    let mil = if op == "conv" {
        conv1x1_mil(seq, in_dim, out_dim, "W", "")
    } else {
        let mut body = String::new();
        write!(
            body,
            "        tensor<fp16, [1, {in_dim}, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        )
        .unwrap();
        write!(
            body,
            "        tensor<fp16, [{out_dim}, {in_dim}]> W = const()[name = string(\"W\"), val = tensor<fp16, [{out_dim}, {in_dim}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        )
        .unwrap();
        write!(
            body,
            "        tensor<fp16, [1, {out_dim}, {seq}]> y16 = linear(x = x16, weight = W)[name = string(\"mm\")];\n",
        )
        .unwrap();
        write!(
            body,
            "        tensor<fp32, [1, {out_dim}, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        )
        .unwrap();
        mil_fp32_program(&format!("tensor<fp32, [1, {in_dim}, {seq}]> x"), &body, "y")
    };
    let w = make_weight_data(out_dim * in_dim);
    let blob = make_blob(&w, out_dim, in_dim);
    let input = input_bytes(in_dim * seq);
    let out_bytes = out_dim * seq * 4;
    let wname = if op == "conv" {
        "@model_path/weights/W.bin"
    } else {
        "@model_path/weights/weight.bin"
    };
    let (c, e) = compile_eval(
        &mil,
        &[(wname, &blob)],
        &[input.len()],
        &[out_bytes],
        Some(&input),
    );
    println!(
        "OK {}x{} {} compile={:.0}ms eval={:.1}ms",
        in_dim, out_dim, op, c, e
    );
}

// ============================================================
// Fused Programs (Orion-style)
// ============================================================

fn test_fused_ffn(dim: usize, hidden: usize, with_taps: bool) {
    let seq = 32;
    let total_out = dim + hidden + hidden;
    let mut body = String::new();
    // Cast input to fp16
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    // Weight constants
    write!(
        body,
        "        tensor<fp16, [{hidden}, {dim}, 1, 1]> W1 = const()[name = string(\"W1\"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w1.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [{hidden}, {dim}, 1, 1]> W3 = const()[name = string(\"W3\"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w3.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {hidden}, 1, 1]> W2 = const()[name = string(\"W2\"), val = tensor<fp16, [{dim}, {hidden}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w2.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    body.push_str(CONV_PARAMS);
    // Conv W1 and W3
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> h1 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W1, x = x16)[name = string(\"h1\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> h3 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W3, x = x16)[name = string(\"h3\")];\n",
    ).unwrap();
    // SiLU: sigmoid(h1) * h1
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> sig = sigmoid(x = h1)[name = string(\"sig\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> silu = mul(x = h1, y = sig)[name = string(\"silu\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> gate = mul(x = silu, y = h3)[name = string(\"gate\")];\n",
    ).unwrap();
    // Conv W2
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W2, x = gate)[name = string(\"out\")];\n",
    ).unwrap();

    let (outputs, out_bytes) = if with_taps {
        write!(
            body,
            "        tensor<fp16, [1, {total_out}, 1, {seq}]> taps16 = concat(axis = 1, y16, h1, gate)[name = string(\"taps\")];\n",
        ).unwrap();
        write!(
            body,
            "        tensor<fp32, [1, {total_out}, 1, {seq}]> taps = cast(dtype = to_fp32, x = taps16)[name = string(\"cast_out\")];\n",
        ).unwrap();
        ("taps", total_out * seq * 4)
    } else {
        write!(
            body,
            "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        ).unwrap();
        ("y", dim * seq * 4)
    };

    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"),
        &body,
        outputs,
    );

    let w1 = make_weight_data(hidden * dim);
    let w3 = make_weight_data(hidden * dim);
    let w2 = make_weight_data(dim * hidden);
    let b1 = make_blob(&w1, hidden, dim);
    let b3 = make_blob(&w3, hidden, dim);
    let b2 = make_blob(&w2, dim, hidden);
    let input = input_bytes(dim * seq);
    let weights = [
        ("@model_path/weights/w1.bin", &b1),
        ("@model_path/weights/w2.bin", &b2),
        ("@model_path/weights/w3.bin", &b3),
    ];
    let (c, e) = compile_eval(&mil, &weights, &[dim * seq * 4], &[out_bytes], Some(&input));
    let taps = if with_taps { " (taps)" } else { "" };
    println!(
        "OK d={} h={}{} compile={:.0}ms eval={:.1}ms",
        dim, hidden, taps, c, e
    );
}

fn test_layer_norm(with_weight: bool) {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();

    if with_weight {
        write!(
            body,
            "        tensor<fp16, [1, {dim}, 1, 1]> gamma = const()[name = string(\"gamma\"), val = tensor<fp16, [1, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/gamma.bin\"), offset = uint64(64)))];\n",
        ).unwrap();
        write!(
            body,
            "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = layer_norm(x = x16, weight = gamma, bias = gamma, axes = [2])[name = string(\"ln\")];\n",
        ).unwrap();
    } else {
        write!(
            body,
            "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = layer_norm(x = x16, axes = [2])[name = string(\"ln\")];\n",
        ).unwrap();
    }
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");

    let input = input_bytes(dim * seq);
    if with_weight {
        let gamma: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let blob = make_blob(&gamma, 1, dim);
        let (c, e) = compile_eval(
            &mil,
            &[("@model_path/weights/gamma.bin", &blob)],
            &[dim * seq * 4],
            &[dim * seq * 4],
            Some(&input),
        );
        println!("OK with_weight compile={:.0}ms eval={:.1}ms", c, e);
    } else {
        let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
        println!("OK no_weight compile={:.0}ms eval={:.1}ms", c, e);
    }
}

fn test_rmsnorm_trick(with_weight: bool) {
    let dim = 64;
    let seq = 16;
    let doubled = 2 * dim;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    // neg_x = x * -1.0
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> neg_x = mul(x = x16, y = -1.0)[name = string(\"neg\")];\n",
    ).unwrap();
    // concat [x, -x]
    write!(
        body,
        "        tensor<fp16, [1, {doubled}, 1, {seq}]> cat = concat(axis = 1, x16, neg_x)[name = string(\"cat\")];\n",
    ).unwrap();

    if with_weight {
        write!(
            body,
            "        tensor<fp16, [1, {dim}, 1, 1]> gamma = const()[name = string(\"gamma\"), val = tensor<fp16, [1, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/gamma.bin\"), offset = uint64(64)))];\n",
        ).unwrap();
        write!(
            body,
            "        tensor<fp16, [1, {doubled}, 1, {seq}]> normed = layer_norm(x = cat, weight = gamma, bias = gamma, axes = [2])[name = string(\"ln\")];\n",
        ).unwrap();
    } else {
        write!(
            body,
            "        tensor<fp16, [1, {doubled}, 1, {seq}]> normed = layer_norm(x = cat, axes = [2])[name = string(\"ln\")];\n",
        ).unwrap();
    }
    // slice first half
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = slice_by_size(x = normed, begin = [0, 0, 0, 0], size = [1, {dim}, 1, {seq}])[name = string(\"slice\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");

    let input = input_bytes(dim * seq);
    if with_weight {
        let gamma: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let blob = make_blob(&gamma, 1, dim);
        let (c, e) = compile_eval(
            &mil,
            &[("@model_path/weights/gamma.bin", &blob)],
            &[dim * seq * 4],
            &[dim * seq * 4],
            Some(&input),
        );
        println!("OK with_weight compile={:.0}ms eval={:.1}ms", c, e);
    } else {
        let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
        println!("OK no_weight compile={:.0}ms eval={:.1}ms", c, e);
    }
}

fn test_rmsnorm_manual() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> sq = mul(x = x16, y = x16)[name = string(\"sq\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, 1, 1, {seq}]> ss = reduce_sum(x = sq, axes = [1], keep_dims = true)[name = string(\"ss\")];\n",
    ).unwrap();
    let invd = 1.0 / dim as f32;
    write!(
        body,
        "        tensor<fp16, [1, 1, 1, 1]> invd = const()[name = string(\"invd\"), val = tensor<fp16, [1, 1, 1, 1]>({invd})];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, 1, 1, {seq}]> ss2 = mul(x = ss, y = invd)[name = string(\"ss2\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, 1, 1, {seq}]> eps = const()[name = string(\"eps\"), val = tensor<fp16, [1, 1, 1, 1]>(1e-5)];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, 1, 1, {seq}]> ss3 = add(x = ss2, y = eps)[name = string(\"ss3\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, 1, 1, {seq}]> rrms = pow(x = ss3, y = -0.5)[name = string(\"rrms\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = mul(x = x16, y = rrms)[name = string(\"y16\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();

    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");

    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_softmax() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = softmax(x = x16, axis = -1)[name = string(\"sm\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_sigmoid() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = sigmoid(x = x16)[name = string(\"sig\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

// ============================================================
// Multi-Weight Programs
// ============================================================

fn test_dual_conv() {
    let dim = 64;
    let hidden = 128;
    let seq = 32;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{hidden}, {dim}, 1, 1]> W1 = const()[name = string(\"W1\"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w1.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [{hidden}, {dim}, 1, 1]> W3 = const()[name = string(\"W3\"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w3.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    body.push_str(CONV_PARAMS);
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> h116 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W1, x = x16)[name = string(\"h1\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {hidden}, 1, {seq}]> h316 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W3, x = x16)[name = string(\"h3\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {hidden}, 1, {seq}]> h1 = cast(dtype = to_fp32, x = h116)[name = string(\"cast_h1\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {hidden}, 1, {seq}]> h3 = cast(dtype = to_fp32, x = h316)[name = string(\"cast_h3\")];\n",
    ).unwrap();

    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"),
        &body,
        "h1, h3",
    );
    let w1 = make_weight_data(hidden * dim);
    let w3 = make_weight_data(hidden * dim);
    let b1 = make_blob(&w1, hidden, dim);
    let b3 = make_blob(&w3, hidden, dim);
    let input = input_bytes(dim * seq);
    let out_bytes = hidden * seq * 4;
    let (c, e) = compile_eval(
        &mil,
        &[
            ("@model_path/weights/w1.bin", &b1),
            ("@model_path/weights/w3.bin", &b3),
        ],
        &[dim * seq * 4],
        &[out_bytes, out_bytes],
        Some(&input),
    );
    println!(
        "OK d={} h={} compile={:.0}ms eval={:.1}ms",
        dim, hidden, c, e
    );
}

fn test_qkv_fused() {
    let dim = 64;
    let seq = 32;
    let out_dim = 3 * dim;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> Wq = const()[name = string(\"Wq\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wq.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> Wk = const()[name = string(\"Wk\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wk.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> Wv = const()[name = string(\"Wv\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wv.bin\"), offset = uint64(64)))];\n",
    ).unwrap();
    body.push_str(CONV_PARAMS);
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> q = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wq, x = x16)[name = string(\"q\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> k = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wk, x = x16)[name = string(\"k\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> v = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wv, x = x16)[name = string(\"v\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {out_dim}, 1, {seq}]> qkv16 = concat(axis = 1, q, k, v)[name = string(\"qkv\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {out_dim}, 1, {seq}]> qkv = cast(dtype = to_fp32, x = qkv16)[name = string(\"cast_out\")];\n",
    ).unwrap();

    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"),
        &body,
        "qkv",
    );
    let w = make_weight_data(dim * dim);
    let bq = make_blob(&w, dim, dim);
    let bk = make_blob(&w, dim, dim);
    let bv = make_blob(&w, dim, dim);
    let input = input_bytes(dim * seq);
    let out_bytes = out_dim * seq * 4;
    let (c, e) = compile_eval(
        &mil,
        &[
            ("@model_path/weights/wq.bin", &bq),
            ("@model_path/weights/wk.bin", &bk),
            ("@model_path/weights/wv.bin", &bv),
        ],
        &[dim * seq * 4],
        &[out_bytes],
        Some(&input),
    );
    println!(
        "OK d={} 3d={} compile={:.0}ms eval={:.1}ms",
        dim, out_dim, c, e
    );
}

fn test_multi_input_add() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> a16 = cast(dtype = to_fp16, x = a)[name = string(\"cast_a\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> b16 = cast(dtype = to_fp16, x = b)[name = string(\"cast_b\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = add(x = a16, y = b16)[name = string(\"add\")];\n",
    ).unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    ).unwrap();
    let mil = mil_fp32_program(
        &format!("tensor<fp32, [1, {dim}, 1, {seq}]> a, tensor<fp32, [1, {dim}, 1, {seq}]> b"),
        &body,
        "y",
    );
    let input = input_bytes(dim * seq);
    let out_bytes = dim * seq * 4;
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[input.len(), input.len()],
        &[out_bytes],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

// ============================================================
// Phase 3: mb.* prefixed op variants
// ============================================================

/// Helper: test a simple unary op with no weights.
fn test_unary_op(op_mil: &str, dim: usize, seq: usize) {
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = {op_mil};\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

/// Helper: test a binary op with no weights.
fn test_binary_op(op_mil: &str, dim: usize, seq: usize) {
    let mut body = String::new();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> ones = const()[name = string(\"ones\"), val = tensor<fp16, [1, {dim}, 1, {seq}]>(1.0)];\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = {op_mil};\n",
    )
    .unwrap();
    write!(
        body,
        "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
    )
    .unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_matmul() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(
        body,
        "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
    )
    .unwrap();
    write!(body, "        tensor<fp16, [{dim}, {dim}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = mb.matmul(transpose_x = bF, transpose_y = bF, x = W, y = x16)[name = string(\"mm\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim * dim, dim);
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[dim * seq * 4],
        &[dim * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_softmax() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(
        body,
        "        int32 sax = const()[name = string(\"sax\"), val = int32(-1)];\n"
    )
    .unwrap();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = mb.softmax(axis = sax, x = x16)[name = string(\"sm\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_concat() {
    let dim = 64;
    let seq = 16;
    let doubled = dim * 2;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> ones = const()[name = string(\"ones\"), val = tensor<fp16, [1, {dim}, 1, {seq}]>(1.0)];\n").unwrap();
    write!(
        body,
        "        int32 cax = const()[name = string(\"cax\"), val = int32(1)];\n"
    )
    .unwrap();
    write!(
        body,
        "        bool cid = const()[name = string(\"cid\"), val = bool(false)];\n"
    )
    .unwrap();
    write!(body, "        tensor<fp16, [1, {doubled}, 1, {seq}]> cat16 = mb.concat(axis = cax, interleave = cid, values = (x16, ones))[name = string(\"cat\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {doubled}, 1, {seq}]> y = cast(dtype = to_fp32, x = cat16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[doubled * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_layer_norm() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val = tensor<int32, [1]>([1])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = mb.layer_norm(x = x16, axes = rax)[name = string(\"ln\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_reduce_sum() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val = tensor<int32, [1]>([1])];\n").unwrap();
    write!(
        body,
        "        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n"
    )
    .unwrap();
    write!(body, "        tensor<fp16, [1, 1, 1, {seq}]> y16 = mb.reduce_sum(x = x16, axes = rax, keep_dims = kd)[name = string(\"rs\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, 1, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[1 * 1 * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_transpose() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 3, 2, 1])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {seq}, 1, {dim}]> y16 = mb.transpose(perm = pm, x = x16)[name = string(\"tr\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {seq}, 1, {dim}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[seq * 1 * dim * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_reshape() {
    let dim = 64;
    let seq = 16;
    let heads = 4;
    let hd = dim / heads;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [4]> sh = const()[name = string(\"sh\"), val = tensor<int32, [4]>([1, {heads}, {hd}, {seq}])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {heads}, {hd}, {seq}]> y16 = mb.reshape(shape = sh, x = x16)[name = string(\"rs\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {heads}, {hd}, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[heads * hd * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_mb_slice_by_size() {
    let dim = 64;
    let seq = 16;
    let half = dim / 2;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [4]> bd = const()[name = string(\"bd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n").unwrap();
    write!(body, "        tensor<int32, [4]> sz = const()[name = string(\"sz\"), val = tensor<int32, [4]>([1, {half}, 1, {seq}])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {half}, 1, {seq}]> y16 = mb.slice_by_size(x = x16, begin = bd, size = sz)[name = string(\"sl\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {half}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[half * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

// ============================================================
// Phase 3: basic ops not yet tested
// ============================================================

fn test_sub() {
    test_binary_op(
        "tensor<fp16, [1, 64, 1, 16]> y16 = sub(x = x16, y = ones)[name = string(\"sub\")];\n",
        64,
        16,
    );
}

fn test_pow() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(
        body,
        "        fp16 nhalf = const()[name = string(\"nhalf\"), val = fp16(-0.5)];\n"
    )
    .unwrap();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> y16 = pow(x = x16, y = nhalf)[name = string(\"pw\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[dim * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_transpose() {
    let dim = 64;
    let seq = 16;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 3, 2, 1])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {seq}, 1, {dim}]> y16 = transpose(perm = pm, x = x16)[name = string(\"tr\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {seq}, 1, {dim}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[seq * 1 * dim * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_reshape() {
    let dim = 64;
    let seq = 16;
    let heads = 4;
    let hd = dim / heads;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [4]> sh = const()[name = string(\"sh\"), val = tensor<int32, [4]>([1, {heads}, {hd}, {seq}])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {heads}, {hd}, {seq}]> y16 = reshape(shape = sh, x = x16)[name = string(\"rs\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {heads}, {hd}, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[],
        &[dim * seq * 4],
        &[heads * hd * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_slice_by_size() {
    let dim = 64;
    let seq = 16;
    let half = dim / 2;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<int32, [4]> bd = const()[name = string(\"bd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n").unwrap();
    write!(body, "        tensor<int32, [4]> sz = const()[name = string(\"sz\"), val = tensor<int32, [4]>([1, {half}, 1, {seq}])];\n").unwrap();
    write!(body, "        tensor<fp16, [1, {half}, 1, {seq}]> y16 = slice_by_size(x = x16, begin = bd, size = sz)[name = string(\"sl\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {half}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(&mil, &[], &[dim * seq * 4], &[half * seq * 4], Some(&input));
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_clamp() {
    test_unary_op("tensor<fp16, [1, 64, 1, 16]> y16 = clamp(x = x16, alpha = 0.0, beta = 1.0)[name = string(\"cl\")];\n", 64, 16);
}

fn test_exp() {
    test_unary_op(
        "tensor<fp16, [1, 64, 1, 16]> y16 = exp(x = x16)[name = string(\"ex\")];\n",
        64,
        16,
    );
}

fn test_log() {
    test_unary_op(
        "tensor<fp16, [1, 64, 1, 16]> y16 = log(x = x16)[name = string(\"lg\")];\n",
        64,
        16,
    );
}

fn test_abs() {
    test_unary_op(
        "tensor<fp16, [1, 64, 1, 16]> y16 = abs(x = x16)[name = string(\"ab\")];\n",
        64,
        16,
    );
}

fn test_tanh() {
    test_unary_op(
        "tensor<fp16, [1, 64, 1, 16]> y16 = tanh(x = x16)[name = string(\"th\")];\n",
        64,
        16,
    );
}

fn test_relu() {
    test_unary_op(
        "tensor<fp16, [1, 64, 1, 16]> y16 = relu(x = x16)[name = string(\"rl\")];\n",
        64,
        16,
    );
}

fn test_leaky_relu() {
    test_unary_op("tensor<fp16, [1, 64, 1, 16]> y16 = leaky_relu(x = x16, alpha = 0.01)[name = string(\"lr\")];\n", 64, 16);
}

fn test_conv_no_bias() {
    let mil = conv1x1_mil(16, 64, 64, "W", "");
    let w = make_weight_data(64 * 64);
    let blob = make_blob(&w, 64, 64);
    let input = input_bytes(64 * 16);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/W.bin", &blob)],
        &[64 * 16 * 4],
        &[64 * 16 * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

fn test_conv3x1() {
    let dim = 64;
    let seq = 16;
    let out_dim = 32;
    let mut body = String::new();
    write!(body, "        tensor<fp16, [1, {dim}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n").unwrap();
    write!(body, "        tensor<fp16, [{out_dim}, {dim}, 3, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{out_dim}, {dim}, 3, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n").unwrap();
    body.push_str(CONV_PARAMS);
    write!(body, "        tensor<fp16, [1, {out_dim}, 1, {seq}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n").unwrap();
    write!(body, "        tensor<fp32, [1, {out_dim}, 1, {seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n").unwrap();
    let mil = mil_fp32_program(&format!("tensor<fp32, [1, {dim}, 1, {seq}]> x"), &body, "y");
    let w = make_weight_data(out_dim * dim * 3);
    let blob = make_blob(&w, out_dim * dim * 3, dim * 3);
    let input = input_bytes(dim * seq);
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/weight.bin", &blob)],
        &[dim * seq * 4],
        &[out_dim * seq * 4],
        Some(&input),
    );
    println!("OK compile={:.0}ms eval={:.1}ms", c, e);
}

// ============================================================
// Phase 3: sequence length boundary
// ============================================================

fn test_seq_boundary(seq: usize) {
    let dim = 64;
    let mil = conv1x1_mil(seq, dim, dim, "W", "");
    let w = make_weight_data(dim * dim);
    let blob = make_blob(&w, dim, dim);
    let input = input_bytes(dim * seq);
    let input_sz = dim * seq * 4;
    let (c, e) = compile_eval(
        &mil,
        &[("@model_path/weights/W.bin", &blob)],
        &[input_sz],
        &[input_sz],
        Some(&input),
    );
    println!(
        "OK seq={} {}bytes compile={:.0}ms eval={:.1}ms",
        seq, input_sz, c, e
    );
}
