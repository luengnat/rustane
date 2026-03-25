//! QKV backward MIL compile + eval test.
//!
//! Usage: ./test_backward_qkv
//! Tests bwd_qkv_mil() compilation and evaluation on ANE.
//! Subprocess-isolated (standalone binary).

use std::time::Instant;

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 64;
    let seq: usize = 16;

    let mil = rustane::mil::bwd_qkv_mil(seq, dim);

    // Verify no rejected ops
    if mil.contains("sub(") && !mil.contains("sub = ") {
        eprintln!("MIL contains rejected sub() op");
        std::process::exit(1);
    }
    if mil.contains("concat(") {
        eprintln!("MIL contains rejected concat() op");
        std::process::exit(1);
    }
    if !mil.contains("slice_by_size") {
        eprintln!("MIL missing slice_by_size");
        std::process::exit(1);
    }

    // Create weight blobs
    let w_data = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 100) as f32) * 0.005).collect() };

    let wqt = rustane::ane::WeightBlob::from_f32(&w_data(dim * dim), dim * dim, 1).unwrap();
    let wkt = rustane::ane::WeightBlob::from_f32(&w_data(dim * dim), dim * dim, 1).unwrap();
    let wvt = rustane::ane::WeightBlob::from_f32(&w_data(dim * dim), dim * dim, 1).unwrap();

    let req = rustane::mil::bwd_qkv_compile_request(seq, dim, &wqt, &wkt, &wvt);

    let t0 = Instant::now();
    let mut executor = match req.compile() {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(2);
        }
    };
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Create input: pack [dq, dk, dv] as fp16
    let input_f16: Vec<u8> = (0..3 * dim * seq)
        .map(|i| {
            let v: f32 = ((i % 100) as f32) * 0.01;
            half::f16::from_f32(v).to_le_bytes()
        })
        .flatten()
        .collect();

    if let Err(e) = executor.write_input(0, &input_f16) {
        eprintln!("write_input: {e}");
        std::process::exit(3);
    }

    let t1 = Instant::now();
    if let Err(e) = executor.eval() {
        eprintln!("eval: {e}");
        std::process::exit(4);
    }
    let eval_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Read output: dx
    let mut buf = vec![0u8; dim * seq * 2];
    if let Err(e) = executor.read_output(0, &mut buf) {
        eprintln!("read_output: {e}");
        std::process::exit(5);
    }
    let fp16_vals: Vec<half::f16> = buf
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
        .collect();
    let f32_vals: Vec<f32> = fp16_vals.iter().map(|h| h.to_f32()).collect();

    let all_zero = f32_vals.iter().all(|&x| x == 0.0);
    let has_nan = f32_vals.iter().any(|x| x.is_nan() || x.is_infinite());

    if all_zero {
        eprintln!("dx: all zeros");
        std::process::exit(6);
    }
    if has_nan {
        eprintln!("dx: nan/inf");
        std::process::exit(7);
    }

    println!(
        "OK bwd_qkv compile={:.0}ms eval={:.1}ms dim={} seq={} sample=[{:.4},{:.4},{:.4},{:.4}]",
        compile_ms, eval_ms, dim, seq, f32_vals[0], f32_vals[1], f32_vals[2], f32_vals[3]
    );
}
