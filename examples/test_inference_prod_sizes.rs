//! Phase 8 Plan 1: FFN Inference Correctness at Production Sizes
//!
//! Tests ANE FFN inference at DIM=768,1024,2048 × SEQ=256,512 against CPU BLAS baseline.
//! Measures accuracy (avg relative error) and flags configs where ANE fails to compile
//! or produces inaccurate results.
//!
//! Usage: cargo run --release --example test_inference_prod_sizes

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

fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            NT,
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

fn to_fp16(data: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        buf[i * 2] = (h & 0xFF) as u8;
        buf[i * 2 + 1] = (h >> 8) as u8;
    }
    buf
}

fn from_fp16(raw: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; raw.len() / 2];
    for i in 0..out.len() {
        let h = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
        out[i] = f16::from_bits(h).to_f32();
    }
    out
}

fn rand_m(r: usize, c: usize, s: f32, seed: u64) -> Vec<f32> {
    (0..r * c)
        .map(|i| {
            let x = ((i as u64 * 2654435761)
                .wrapping_add(seed)
                .wrapping_mul(0x9E3779B97F4A7C15)
                >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * s
        })
        .collect()
}

fn mil_ffn_layer(d: usize, inter: usize, sp: usize) -> String {
    let total_ic = inter + d;
    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n");
    m.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        d, sp
    ));
    for (wn, oc, ic) in [("Wg", inter, d), ("Wu", inter, d), ("WdI", d, total_ic)] {
        m.push_str(&format!(
            "        tensor<fp16, [{}, {}, 1, 1]> {} = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/{}.bin\"), offset = tensor<uint64, []>(64)))]  ;\n",
            oc, ic, wn, wn, oc, ic, wn
        ));
    }
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    for (wn, nm, on) in [("Wg", "cg", "gate"), ("Wu", "cu", "up")] {
        m.push_str(&format!(
            "        tensor<fp16, [1, {}, 1, {}]> {} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {}, x = x)[name = tensor<string, []>(\"{}\")];\n",
            inter, sp, on, wn, nm
        ));
    }
    m.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n", inter, sp
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n", inter, sp
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n", inter, sp
    ));
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> cat = concat(values = (fused, x), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n",
        total_ic, sp
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdI, x = cat)[name = tensor<string, []>(\"cd\")];\n",
        d, sp
    ));
    m.push_str("    } -> (y);\n}\n");
    m
}

fn cpu_ffn_forward(
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    x: &[f32],
    d: usize,
    inter: usize,
    sp: usize,
) -> Vec<f32> {
    let gate = mm(wg, inter, d, x, sp);
    let up = mm(wu, inter, d, x, sp);
    let mut fused = vec![0.0f32; inter * sp];
    for i in 0..fused.len() {
        let s = 1.0 / (1.0 + (-gate[i]).exp());
        fused[i] = gate[i] * s * up[i];
    }
    let down = mm(wd, d, inter, &fused, sp);
    let mut y = vec![0.0f32; d * sp];
    for i in 0..y.len() {
        y[i] = down[i] + x[i];
    }
    y
}

fn compare(cpu: &[f32], ane: &[f32], label: &str) {
    if cpu.len() != ane.len() {
        println!(
            "  {:30} SIZE MISMATCH cpu={} ane={}",
            label,
            cpu.len(),
            ane.len()
        );
        return;
    }
    let mut max_abs = 0.0_f32;
    let mut sum_rel = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut zero_denom = 0usize;
    for i in 0..cpu.len() {
        let diff = (cpu[i] - ane[i]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if ane[i].is_nan() {
            nan_count += 1;
            continue;
        }
        if ane[i].is_infinite() {
            inf_count += 1;
            continue;
        }
        let denom = cpu[i].abs().max(ane[i].abs()).max(1e-6_f32);
        let rel = (diff / denom) as f64;
        sum_rel += rel;
        if rel > max_rel {
            max_rel = rel;
        }
        if cpu[i].abs() < 1e-6 && ane[i].abs() < 1e-6 {
            zero_denom += 1;
        }
    }
    let avg_rel = sum_rel / cpu.len() as f64;
    let pct99 = {
        let mut rels: Vec<f64> = (0..cpu.len())
            .map(|i| {
                let diff = (cpu[i] - ane[i]).abs();
                let denom = cpu[i].abs().max(ane[i].abs()).max(1e-6_f32);
                (diff / denom) as f64
            })
            .collect();
        rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rels[(rels.len() as f64 * 0.99) as usize]
    };

    let accuracy = if avg_rel < 0.01 {
        "EXCELLENT"
    } else if avg_rel < 0.05 {
        "GOOD"
    } else if avg_rel < 0.10 {
        "OK"
    } else {
        "POOR"
    };

    println!(
        "  {:30} avg_rel={:.4}% max_rel={:.4}% p99={:.4}% max_abs={:.6} {}",
        label,
        avg_rel * 100.0,
        max_rel * 100.0,
        pct99 * 100.0,
        max_abs,
        accuracy
    );
    if nan_count > 0 {
        println!("    ⚠ {} NaN values in ANE output", nan_count);
    }
    if inf_count > 0 {
        println!("    ⚠ {} Inf values in ANE output", inf_count);
    }
    if zero_denom > 0 {
        println!("    ℹ {} near-zero elements skipped", zero_denom);
    }
}

fn test_ffn(d: usize, sp: usize) {
    let inter = d * 4;
    let params = inter * d * 2 + d * inter;
    println!(
        "\n--- FFN D={} SP={} inter={} ({:.1}M params) ---",
        d,
        sp,
        inter,
        params as f64 / 1e6
    );

    let wg = rand_m(inter, d, 0.02, 42);
    let wu = rand_m(inter, d, 0.02, 43);
    let wd = rand_m(d, inter, 0.02, 44);

    let mut wdi = vec![0.0f32; d * (inter + d)];
    for r in 0..d {
        for c in 0..inter {
            wdi[r * (inter + d) + c] = wd[r * inter + c];
        }
        wdi[r * (inter + d) + inter + r] = 1.0;
    }

    let x = rand_m(d, sp, 0.1, 9999);
    let mil = mil_ffn_layer(d, inter, sp);
    let names: Vec<String> = ["Wg", "Wu", "WdI"]
        .iter()
        .map(|n| format!("@model_path/weights/{}.bin", n))
        .collect();
    let nrefs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let blobs = vec![build_blob(&wg), build_blob(&wu), build_blob(&wdi)];
    let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
    let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

    let t = Instant::now();
    match rustane::wrapper::ANECompiler::new().compile_multi(
        &mil,
        &nrefs,
        &brefs,
        &lens,
        &[d * sp * 2],
        &[d * sp * 2],
    ) {
        Ok(mut exec) => {
            let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t_eval = Instant::now();
            exec.write_input(0, &to_fp16(&x)).expect("write");
            exec.eval().expect("eval");
            let output_raw = exec.read_output_vec(0).expect("read");
            let eval_ms = t_eval.elapsed().as_secs_f64() * 1000.0;
            let ane_out = from_fp16(&output_raw);
            let cpu_out = cpu_ffn_forward(&wg, &wu, &wd, &x, d, inter, sp);

            println!("  Compile: {:.1}ms  Eval: {:.3}ms", compile_ms, eval_ms);
            compare(&cpu_out, &ane_out, &format!("D={} SP={}", d, sp));
        }
        Err(e) => {
            println!("  COMPILE FAILED: {:?}", e);
        }
    }
}

fn test_qkv(d: usize, sp: usize) {
    println!("\n--- QKV D={} SP={} ---", d, sp);

    let wq = rand_m(d, d, 0.02, 42);
    let wk = rand_m(d, d, 0.02, 43);
    let wv = rand_m(d, d, 0.02, 44);
    let x = rand_m(d, sp, 0.1, 9999);

    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n");
    m.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        d, sp
    ));
    for wn in ["Wq", "Wk", "Wv"] {
        m.push_str(&format!(
            "        tensor<fp16, [{}, {}, 1, 1]> {} = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/{}.bin\"), offset = tensor<uint64, []>(64)))]  ;\n",
            d, d, wn, wn, d, d, wn
        ));
    }
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    for (i, wn) in ["Wq", "Wk", "Wv"].iter().enumerate() {
        m.push_str(&format!(
            "        tensor<fp16, [1, {}, 1, {}]> q{} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {}, x = x)[name = tensor<string, []>(\"c{}\")];\n",
            d, sp, i, wn, i
        ));
    }
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> qkv = concat(values = (q0, q1, q2), axis = ax, interleave = ci)[name = tensor<string, []>(\"ct\")];\n",
        3 * d, sp
    ));
    m.push_str("    } -> (qkv);\n}\n");

    let names: Vec<String> = ["Wq", "Wk", "Wv"]
        .iter()
        .map(|n| format!("@model_path/weights/{}.bin", n))
        .collect();
    let nrefs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let blobs = vec![build_blob(&wq), build_blob(&wk), build_blob(&wv)];
    let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
    let brefs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();

    match rustane::wrapper::ANECompiler::new().compile_multi(
        &m,
        &nrefs,
        &brefs,
        &lens,
        &[d * sp * 2],
        &[3 * d * sp * 2],
    ) {
        Ok(mut exec) => {
            exec.write_input(0, &to_fp16(&x)).expect("write");
            exec.eval().expect("eval");
            let output_raw = exec.read_output_vec(0).expect("read");
            let ane_out = from_fp16(&output_raw);

            let cpu_q = mm(&wq, d, d, &x, sp);
            let cpu_k = mm(&wk, d, d, &x, sp);
            let cpu_v = mm(&wv, d, d, &x, sp);
            let mut cpu_qkv = vec![0.0f32; 3 * d * sp];
            cpu_qkv[..d * sp].copy_from_slice(&cpu_q);
            cpu_qkv[d * sp..2 * d * sp].copy_from_slice(&cpu_k);
            cpu_qkv[2 * d * sp..].copy_from_slice(&cpu_v);

            compare(&cpu_qkv, &ane_out, &format!("QKV D={} SP={}", d, sp));
            let q_slice = &ane_out[..d * sp];
            let k_slice = &ane_out[d * sp..2 * d * sp];
            let v_slice = &ane_out[2 * d * sp..];
            compare(&cpu_q, q_slice, &format!("Q D={} SP={}", d, sp));
            compare(&cpu_k, k_slice, &format!("K D={} SP={}", d, sp));
            compare(&cpu_v, v_slice, &format!("V D={} SP={}", d, sp));
        }
        Err(e) => {
            println!("  COMPILE FAILED: {:?}", e);
        }
    }
}

fn main() {
    let _runtime = rustane::wrapper::ANERuntime::init().expect("ANE runtime");

    println!("============================================================");
    println!("  Phase 8: Inference Correctness at Production Sizes");
    println!("============================================================");

    println!("\n=== QKV Projection Sweep ===");
    for &d in &[256, 512, 768, 1024, 2048] {
        for &sp in &[256, 512] {
            test_qkv(d, sp);
        }
    }

    println!("\n=== FFN Inference Sweep (D>=512, SP>=256) ===");
    for &d in &[512, 768, 1024, 2048] {
        for &sp in &[256, 512] {
            test_ffn(d, sp);
        }
    }

    println!("\n============================================================");
    println!("  Phase 8 Complete — check results above");
    println!("============================================================");
}
