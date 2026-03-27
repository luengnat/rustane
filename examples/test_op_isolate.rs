//! Isolate which ANE op fails: transpose, matmul, or softmax.
//!
//! Usage:
//!   cargo run --example test_op_isolate -- 1   # test transpose only
//!   cargo run --example test_op_isolate -- 2   # test matmul only
//!   cargo run --example test_op_isolate -- 3   # test transpose + matmul
//!   cargo run --example test_op_isolate -- 4   # test full attention

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

fn write_fp16(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|&v| {
            let b = f16::from_f32(v).to_bits();
            vec![(b & 0xFF) as u8, (b >> 8) as u8]
        })
        .flatten()
        .collect()
}

fn rand_weight(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u64 * 2654435761).wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * scale
        })
        .collect()
}

fn run_ane(
    mil: &str,
    blobs: &[(&str, &[u8])],
    in_b: usize,
    out_b: usize,
    inp16: &[u8],
) -> Result<(Vec<f32>, u128), String> {
    let full_names: Vec<String> = blobs
        .iter()
        .map(|(n, _)| format!("@model_path/weights/{}.bin", n))
        .collect();
    let name_refs: Vec<&str> = full_names.iter().map(|s| s.as_str()).collect();
    let datas: Vec<&[u8]> = blobs.iter().map(|(_, d)| d.as_ref()).collect();
    let lens: Vec<usize> = blobs.iter().map(|(_, d)| d.len()).collect();
    let mut exec = rustane::wrapper::ANECompiler::new()
        .compile_multi(mil, &name_refs, &datas, &lens, &[in_b], &[out_b])
        .map_err(|e| format!("COMPILE: {}", e))?;
    exec.write_input(0, inp16)
        .map_err(|e| format!("WRITE: {}", e))?;
    exec.eval().map_err(|e| format!("WARMUP: {}", e))?;
    let start = Instant::now();
    for _ in 0..2 {
        exec.eval().map_err(|e| format!("EVAL: {}", e))?;
    }
    let us = start.elapsed().as_micros() / 3;
    let raw = exec
        .read_output_vec(0)
        .map_err(|e| format!("READ: {}", e))?;
    let mut out = vec![0.0f32; raw.len() / 2];
    for i in 0..out.len() {
        let b = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
        out[i] = f16::from_bits(b).to_f32();
    }
    Ok((out, us))
}

// ── Test 1: Transpose only ──
fn mil_transpose(d: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HEADER);
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    // Conv1x1 to produce known output first
    let wn = "W";
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
    // Conv1x1
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> c = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"c\")];\n");
    // Transpose [1, D, 1, SP] → [1, SP, 1, D]
    m.push_str("        tensor<int32, [4]> pm = const()[name = tensor<string, []>(\"pm\"), val = tensor<int32, [4]>([0, 3, 2, 1])];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&sp.to_string());
    m.push_str(", 1, ");
    m.push_str(&d.to_string());
    m.push_str("]> y = transpose(x = c, perm = pm)[name = tensor<string, []>(\"t\")];\n");
    m.push_str("    } -> (y);\n}\n");
    m
}

// ── Test 2: Matmul only (no transpose, use 2D tensors) ──
fn mil_matmul_2d(m: usize, k: usize, n: usize) -> String {
    let mut m2 = String::new();
    m2.push_str(MIL_HEADER);
    // Two 2D inputs
    m2.push_str("    func main<ios16>(tensor<fp16, [");
    m2.push_str(&m.to_string());
    m2.push_str(", ");
    m2.push_str(&k.to_string());
    m2.push_str("]> a, tensor<fp16, [");
    m2.push_str(&k.to_string());
    m2.push_str(", ");
    m2.push_str(&n.to_string());
    m2.push_str("]> b) {\n");
    m2.push_str("        tensor<bool, []> f = const()[name = tensor<string, []>(\"f\"), val = tensor<bool, []>(false)];\n");
    m2.push_str("        tensor<fp16, [");
    m2.push_str(&m.to_string());
    m2.push_str(", ");
    m2.push_str(&n.to_string());
    m2.push_str("]> y = matmul(x = a, y = b, transpose_x = f, transpose_y = f)[name = tensor<string, []>(\"mm\")];\n");
    m2.push_str("    } -> (y);\n}\n");
    m2
}

// ── Test 3: Transpose + matmul (no softmax) ──
fn mil_transpose_matmul(d: usize, sp: usize) -> String {
    let mut m3 = String::new();
    m3.push_str(MIL_HEADER);
    m3.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m3.push_str(&d.to_string());
    m3.push_str(", 1, ");
    m3.push_str(&sp.to_string());
    m3.push_str("]> x) {\n");
    // Single conv1x1 for both "Q" and "K" (use same weight)
    let wn = "W";
    m3.push_str("        tensor<fp16, [");
    m3.push_str(&d.to_string());
    m3.push_str(", ");
    m3.push_str(&d.to_string());
    m3.push_str(", 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [");
    m3.push_str(&d.to_string());
    m3.push_str(", ");
    m3.push_str(&d.to_string());
    m3.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/W.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    m3.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m3.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m3.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m3.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m3.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    // Conv to get Q and K (same weight, so Q = K)
    m3.push_str("        tensor<fp16, [1, ");
    m3.push_str(&d.to_string());
    m3.push_str(", 1, ");
    m3.push_str(&sp.to_string());
    m3.push_str("]> q = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"q\")];\n");
    m3.push_str("        tensor<fp16, [1, ");
    m3.push_str(&d.to_string());
    m3.push_str(", 1, ");
    m3.push_str(&sp.to_string());
    m3.push_str("]> k = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"k\")];\n");
    // Transpose Q: [1,D,1,SP] → [1,1,D,SP] → [1,1,SP,D]
    m3.push_str("        tensor<int32, [4]> p0213 = const()[name = tensor<string, []>(\"p0213\"), val = tensor<int32, [4]>([0, 2, 1, 3])];\n");
    m3.push_str("        tensor<int32, [4]> p0132 = const()[name = tensor<string, []>(\"p0132\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    m3.push_str("        tensor<fp16, [1, 1, ");
    m3.push_str(&d.to_string());
    m3.push_str(", ");
    m3.push_str(&sp.to_string());
    m3.push_str("]> qt = transpose(x = q, perm = p0213)[name = tensor<string, []>(\"qt\")];\n");
    m3.push_str("        tensor<fp16, [1, 1, ");
    m3.push_str(&d.to_string());
    m3.push_str(", ");
    m3.push_str(&sp.to_string());
    m3.push_str("]> kt = transpose(x = k, perm = p0213)[name = tensor<string, []>(\"kt\")];\n");
    m3.push_str("        tensor<fp16, [1, 1, ");
    m3.push_str(&sp.to_string());
    m3.push_str(", ");
    m3.push_str(&d.to_string());
    m3.push_str("]> qt2 = transpose(x = qt, perm = p0132)[name = tensor<string, []>(\"qt2\")];\n");
    // Matmul: [1,1,SP,D] @ [1,1,D,SP] → [1,1,SP,SP]
    m3.push_str("        tensor<bool, []> f = const()[name = tensor<string, []>(\"f\"), val = tensor<bool, []>(false)];\n");
    m3.push_str("        tensor<fp16, [1, 1, ");
    m3.push_str(&sp.to_string());
    m3.push_str(", ");
    m3.push_str(&sp.to_string());
    m3.push_str("]> y = matmul(x = qt2, y = kt, transpose_x = f, transpose_y = f)[name = tensor<string, []>(\"mm\")];\n");
    m3.push_str("    } -> (y);\n}\n");
    m3
}

fn main() {
    let tid: u32 = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    match tid {
        1 => {
            // Test transpose: conv1x1 → transpose [1,D,1,SP] → [1,SP,1,D]
            let d = 8;
            let sp = 4;
            println!("=== Test 1: Transpose (D={}, SP={}) ===", d, sp);
            let w = rand_weight(d * d, 0.05);
            let inp = rand_weight(d * sp, 1.0);
            let mil = mil_transpose(d, sp);
            let in_b = d * sp * 2;
            let out_b = sp * d * 2; // transposed: [1, SP, 1, D]
            let inp16 = write_fp16(&inp);
            match run_ane(&mil, &[("W", &build_blob(&w))], in_b, out_b, &inp16) {
                Ok((out, us)) => {
                    println!("  PASS: {}us, {} output values", us, out.len());
                    println!("  First 8: {:?}", &out[..8.min(out.len())]);
                }
                Err(e) => println!("  FAIL: {}", e),
            }
        }
        2 => {
            // Test matmul with 2D tensors: [M,K] @ [K,N] → [M,N]
            let m = 4;
            let k = 6;
            let n = 4;
            println!("=== Test 2: Matmul 2D ({}×{} @ {}×{}) ===", m, k, k, n);
            let a = rand_weight(m * k, 0.1);
            let b = rand_weight(k * n, 0.1);
            let mil = mil_matmul_2d(m, k, n);
            let in_b = m * k * 2;
            // Two inputs
            let a16 = write_fp16(&a);
            let b16 = write_fp16(&b);
            // Concatenate inputs for the ANE (write_input(0) = a, write_input(1) = b)
            let full_names: Vec<String> = vec![];
            let name_refs: Vec<&str> = vec![];
            let datas: Vec<&[u8]> = vec![];
            let lens: Vec<usize> = vec![];
            let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
                &mil,
                &name_refs,
                &datas,
                &lens,
                &[in_b, k * n * 2],
                &[m * n * 2],
            ) {
                Ok(e) => e,
                Err(e) => {
                    println!("  COMPILE FAIL: {}", e);
                    return;
                }
            };
            if let Err(e) = exec.write_input(0, &a16) {
                println!("  WRITE 0 FAIL: {}", e);
                return;
            }
            if let Err(e) = exec.write_input(1, &b16) {
                println!("  WRITE 1 FAIL: {}", e);
                return;
            }
            match exec.eval() {
                Ok(()) => {
                    let start = Instant::now();
                    for _ in 0..2 {
                        exec.eval().unwrap();
                    }
                    let us = start.elapsed().as_micros() / 3;
                    match exec.read_output_vec(0) {
                        Ok(raw) => {
                            let mut out = vec![0.0f32; raw.len() / 2];
                            for i in 0..out.len() {
                                let b2 = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
                                out[i] = f16::from_bits(b2).to_f32();
                            }
                            println!("  PASS: {}us, {} output values", us, out.len());
                            println!("  First 8: {:?}", &out[..8.min(out.len())]);
                            // Verify against CPU
                            let mut cpu = vec![0.0f32; m * n];
                            for i in 0..m {
                                for j in 0..n {
                                    for c in 0..k {
                                        cpu[i * n + j] += a[i * k + c] * b[c * n + j];
                                    }
                                }
                            }
                            let mut mx = 0.0f32;
                            for i in 0..out.len().min(cpu.len()) {
                                mx = mx.max((out[i] - cpu[i]).abs());
                            }
                            println!("  CPU ref first 8: {:?}", &cpu[..8.min(cpu.len())]);
                            println!("  Max error vs CPU: {:.6}", mx);
                        }
                        Err(e) => println!("  READ FAIL: {}", e),
                    }
                }
                Err(e) => println!("  EVAL FAIL: {}", e),
            }
        }
        3 => {
            // Test transpose + matmul (no softmax)
            let d = 8;
            let sp = 4;
            println!("=== Test 3: Transpose + Matmul (D={}, SP={}) ===", d, sp);
            let w = rand_weight(d * d, 0.05);
            let inp = rand_weight(d * sp, 1.0);
            let mil = mil_transpose_matmul(d, sp);
            let in_b = d * sp * 2;
            let out_b = sp * sp * 2; // [1, 1, SP, SP]
            let inp16 = write_fp16(&inp);
            match run_ane(&mil, &[("W", &build_blob(&w))], in_b, out_b, &inp16) {
                Ok((out, us)) => {
                    println!("  PASS: {}us, {} output values", us, out.len());
                    println!("  First 8: {:?}", &out[..8.min(out.len())]);
                }
                Err(e) => println!("  FAIL: {}", e),
            }
        }
        4 => {
            // Test add with 2 inputs
            let d = 8;
            let sp = 4;
            println!("=== Test 4: Add (2 inputs, D={}, SP={}) ===", d, sp);
            let mut m4 = String::new();
            m4.push_str(MIL_HEADER);
            m4.push_str("    func main<ios16>(tensor<fp16, [1, ");
            m4.push_str(&d.to_string());
            m4.push_str(", 1, ");
            m4.push_str(&sp.to_string());
            m4.push_str("]> a, tensor<fp16, [1, ");
            m4.push_str(&d.to_string());
            m4.push_str(", 1, ");
            m4.push_str(&sp.to_string());
            m4.push_str("]> b) {\n");
            m4.push_str("        tensor<fp16, [1, ");
            m4.push_str(&d.to_string());
            m4.push_str(", 1, ");
            m4.push_str(&sp.to_string());
            m4.push_str("]> y = add(x = a, y = b)[name = tensor<string, []>(\"add\")];\n");
            m4.push_str("    } -> (y);\n}\n");
            let inp_a = rand_weight(d * sp, 1.0);
            let inp_b = rand_weight(d * sp, 0.5);
            let in_b = d * sp * 2;
            let out_b = d * sp * 2;
            let a16 = write_fp16(&inp_a);
            let b16 = write_fp16(&inp_b);
            let full_names: Vec<String> = vec![];
            let name_refs: Vec<&str> = vec![];
            let datas: Vec<&[u8]> = vec![];
            let lens: Vec<usize> = vec![];
            let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
                &m4,
                &name_refs,
                &datas,
                &lens,
                &[in_b, in_b],
                &[out_b],
            ) {
                Ok(e) => e,
                Err(e) => {
                    println!("  COMPILE FAIL: {}", e);
                    return;
                }
            };
            if let Err(e) = exec.write_input(0, &a16) {
                println!("  WRITE 0 FAIL: {}", e);
                return;
            }
            if let Err(e) = exec.write_input(1, &b16) {
                println!("  WRITE 1 FAIL: {}", e);
                return;
            }
            match exec.eval() {
                Ok(()) => {
                    let start = Instant::now();
                    for _ in 0..2 {
                        exec.eval().unwrap();
                    }
                    let us = start.elapsed().as_micros() / 3;
                    match exec.read_output_vec(0) {
                        Ok(raw) => {
                            let mut out = vec![0.0f32; raw.len() / 2];
                            for i in 0..out.len() {
                                let b2 = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
                                out[i] = f16::from_bits(b2).to_f32();
                            }
                            // Verify: out = a + b
                            let mut mx = 0.0f32;
                            for i in 0..out.len() {
                                let expected = inp_a[i] + inp_b[i];
                                mx = mx.max((out[i] - expected).abs());
                            }
                            println!(
                                "  PASS: {}us, {} values, max error: {:.6}",
                                us,
                                out.len(),
                                mx
                            );
                        }
                        Err(e) => println!("  READ FAIL: {}", e),
                    }
                }
                Err(e) => println!("  EVAL FAIL: {}", e),
            }
        }
        5 => {
            // Test reduce_mean
            let d = 8;
            let sp = 4;
            println!("=== Test 5: reduce_mean (D={}, SP={}) ===", d, sp);
            let mut m5 = String::new();
            m5.push_str(MIL_HEADER);
            m5.push_str("    func main<ios16>(tensor<fp16, [1, ");
            m5.push_str(&d.to_string());
            m5.push_str(", 1, ");
            m5.push_str(&sp.to_string());
            m5.push_str("]> x) {\n");
            m5.push_str("        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n");
            m5.push_str("        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n");
            m5.push_str("        tensor<fp16, [1, 1, 1, ");
            m5.push_str(&sp.to_string());
            m5.push_str("]> y = reduce_mean(x = x, axes = ax, keep_dims = kd)[name = tensor<string, []>(\"rm\")];\n");
            m5.push_str("    } -> (y);\n}\n");
            let inp = rand_weight(d * sp, 1.0);
            let in_b = d * sp * 2;
            let out_b = 1 * sp * 2;
            let inp16 = write_fp16(&inp);
            match run_ane(&m5, &[], in_b, out_b, &inp16) {
                Ok((out, us)) => {
                    // CPU reference: mean along channel dim
                    let mut cpu = vec![0.0f32; sp];
                    for s in 0..sp {
                        let mut sum = 0.0f32;
                        for c in 0..d {
                            sum += inp[c * sp + s];
                        }
                        cpu[s] = sum / d as f32;
                    }
                    let mut mx = 0.0f32;
                    for i in 0..out.len().min(cpu.len()) {
                        mx = mx.max((out[i] - cpu[i]).abs());
                    }
                    println!(
                        "  PASS: {}us, {} values, max error: {:.6}",
                        us,
                        out.len(),
                        mx
                    );
                    println!("  ANE: {:?}", &out[..out.len().min(8)]);
                    println!("  CPU: {:?}", &cpu[..cpu.len().min(8)]);
                }
                Err(e) => println!("  FAIL: {}", e),
            }
        }
        _ => {
            println!("Usage: test_op_isolate <test_id>");
            println!("  1 = transpose    2 = matmul 2D    3 = transpose + matmul");
            println!("  4 = add (2 in)   5 = reduce_mean");
        }
    }
}
