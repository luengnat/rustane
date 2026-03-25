//! Single ANE conv1x1 test — run via subprocess from test_ane_ops.
//!
//! Usage: RUSTANE_TEST_S=32 RUSTANE_TEST_I=64 RUSTANE_TEST_O=192 ./test_ane_single
//! Prints "OK compile_ms eval_ms" on success, or error message to stderr.

use std::time::Instant;

fn main() {
    let s: usize = std::env::var("RUSTANE_TEST_S")
        .unwrap_or_default()
        .parse()
        .unwrap_or(32);
    let i: usize = std::env::var("RUSTANE_TEST_I")
        .unwrap_or_default()
        .parse()
        .unwrap_or(64);
    let o: usize = std::env::var("RUSTANE_TEST_O")
        .unwrap_or_default()
        .parse()
        .unwrap_or(192);
    let op: &str = std::env::var("RUSTANE_TEST_OP")
        .map(|v| Box::leak(v.into_boxed_str()) as &str)
        .unwrap_or("conv1x1");

    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    use rustane::ane::WeightBlob;

    // Create weight blob: [out_dim, in_dim] (row-major)
    let w: Vec<f32> = (0..o * i)
        .map(|n| ((n % 100) as f32) * 0.01 - 0.5)
        .collect();

    let t0 = Instant::now();

    let req = if op == "matmul" {
        let blob = WeightBlob::from_f32(&w, o, i).unwrap();
        rustane::mil::linear_matmul_compile_request(s, i, o, &blob)
    } else {
        // conv1x1: weight is [out_dim, in_dim, 1, 1]
        let blob = WeightBlob::from_f32(&w, o, i).unwrap();
        rustane::mil::conv1x1_compile_request(s, i, o, &blob)
    };

    let mut executor = match req.compile() {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(2);
        }
    };
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Create input: [positions, in_dim] row-major, then reshape for ANE
    let input_flat: Vec<f32> = (0..i * s)
        .map(|n| ((n * 7 + 3) % 200) as f32 * 0.01)
        .collect();

    let input_bytes: Vec<u8> = if op == "matmul" {
        // matmul: transpose to [in_dim, positions], shape [1, in_dim, positions]
        let mut transposed = vec![0.0f32; input_flat.len()];
        for r in 0..s {
            for c in 0..i {
                transposed[c * s + r] = input_flat[r * i + c];
            }
        }
        transposed.iter().flat_map(|f| f.to_le_bytes()).collect()
    } else {
        // conv1x1: reshape to [1, in_dim, 1, seq_len] = just the flat data in channel-first order
        // input_flat is [seq_len, in_dim] row-major
        // We need [1, in_dim, 1, seq_len] = channel-first = [in_dim, seq_len]
        let mut transposed = vec![0.0f32; input_flat.len()];
        for pos in 0..s {
            for ch in 0..i {
                transposed[ch * s + pos] = input_flat[pos * i + ch];
            }
        }
        transposed.iter().flat_map(|f| f.to_le_bytes()).collect()
    };

    if let Err(e) = executor.write_input(0, &input_bytes) {
        eprintln!("write_input: {e}");
        std::process::exit(3);
    }

    let t1 = Instant::now();
    if let Err(e) = executor.eval() {
        eprintln!("eval: {e}");
        std::process::exit(4);
    }
    let eval_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mut output_bytes = vec![0u8; o * s * 4];
    if let Err(e) = executor.read_output(0, &mut output_bytes) {
        eprintln!("read_output: {e}");
        std::process::exit(5);
    }

    // Check output isn't degenerate
    let output: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if output.iter().all(|&x| x == 0.0) {
        eprintln!("all zeros");
        std::process::exit(6);
    }
    if output.iter().any(|x| x.is_nan() || x.is_infinite()) {
        eprintln!("nan/inf");
        std::process::exit(7);
    }

    // Print first few output values for sanity
    eprintln!(
        "sample output: [{:.4}, {:.4}, {:.4}, {:.4}]",
        output[0], output[1], output[2], output[3]
    );

    println!("OK {:.0} {:.1}", compile_ms, eval_ms);
}
