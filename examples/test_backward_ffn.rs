//! FFN backward MIL compile + eval test.
//!
//! Usage: ./test_backward_ffn
//! Tests bwd_ffn_mil() compilation and evaluation on ANE.
//! Subprocess-isolated (standalone binary).

use std::time::Instant;

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 64;
    let hidden: usize = 128;
    let seq: usize = 16;

    let mil = rustane::mil::bwd_ffn_mil(seq, dim, hidden);

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
        eprintln!("MIL missing slice_by_size for input unpacking");
        std::process::exit(1);
    }
    if !mil.contains("-> (dh1, dh3, dx)") {
        eprintln!("MIL missing multi-output return");
        std::process::exit(1);
    }

    // Create weight blobs: small random values to avoid fp16 overflow
    let w1t_data: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();
    let w2t_data: Vec<f32> = (0..hidden * dim)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();
    let w3t_data: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();

    let w1t = rustane::ane::WeightBlob::from_f32(&w1t_data, dim * hidden, 1).unwrap();
    let w2t = rustane::ane::WeightBlob::from_f32(&w2t_data, hidden * dim, 1).unwrap();
    let w3t = rustane::ane::WeightBlob::from_f32(&w3t_data, dim * hidden, 1).unwrap();

    let req = rustane::mil::bwd_ffn_compile_request(seq, dim, hidden, &w1t, &w2t, &w3t);

    let t0 = Instant::now();
    let mut executor = match req.compile() {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(2);
        }
    };
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Create input: pack [dffn, h1, h3] as fp16
    let in_ch = dim + 2 * hidden;
    let input_f16: Vec<u8> = (0..in_ch * seq)
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

    // Read outputs: 0=dh1, 1=dh3, 2=dx (alphabetical order)
    let mut ok = true;
    for (idx, name) in ["dh1", "dh3", "dx"].iter().enumerate() {
        let expected_ch = if idx < 2 {
            hidden * seq * 2
        } else {
            dim * seq * 2
        };
        let mut buf = vec![0u8; expected_ch];
        if let Err(e) = executor.read_output(idx, &mut buf) {
            eprintln!("read_output({}): {e}", name);
            ok = false;
            continue;
        }
        let fp16_vals: Vec<half::f16> = buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let f32_vals: Vec<f32> = fp16_vals.iter().map(|h| h.to_f32()).collect();

        let all_zero = f32_vals.iter().all(|&x| x == 0.0);
        let has_nan = f32_vals.iter().any(|x| x.is_nan() || x.is_infinite());

        if all_zero {
            eprintln!("{}: all zeros", name);
            ok = false;
        } else if has_nan {
            eprintln!("{}: nan/inf", name);
            ok = false;
        } else {
            eprintln!(
                "{}: [ {:.4}, {:.4}, {:.4}, {:.4} ] ({} vals)",
                name,
                f32_vals[0],
                f32_vals[1],
                f32_vals[2],
                f32_vals[3],
                f32_vals.len()
            );
        }
    }

    if ok {
        println!(
            "OK bwd_ffn compile={:.0}ms eval={:.1}ms dim={} hidden={} seq={}",
            compile_ms, eval_ms, dim, hidden, seq
        );
    } else {
        std::process::exit(5);
    }
}
