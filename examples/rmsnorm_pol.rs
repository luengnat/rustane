//! RMSNorm proof-of-life example
//!
//! Runs an upstream-style RMSNorm forward graph on ANE and checks the result
//! against a CPU reference.

use half::f16;
use rustane::{
    init,
    mil::{rmsnorm_mil, WeightBlob},
    wrapper::{ANECompiler, ANETensor},
};

const DIM: usize = 256;
const SEQ: usize = 64;

fn f32_to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn fp16_bits_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect()
}

fn cpu_rmsnorm(x: &[f32], gamma: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..SEQ {
        let mut ss = 0.0f32;
        for c in 0..DIM {
            let v = x[c * SEQ + t];
            ss += v * v;
        }
        let scale = 1.0f32 / ((ss / DIM as f32) + 1e-5).sqrt();
        for c in 0..DIM {
            out[c * SEQ + t] = x[c * SEQ + t] * scale * gamma[c];
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - RMSNorm Proof of Life");
    println!("====================================\n");

    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    let mil = rmsnorm_mil(SEQ, DIM);
    println!("✓ MIL graph built\n");

    println!("Preparing input tensor and gamma weights...");
    let mut x = vec![0.0f32; DIM * SEQ];
    for (i, item) in x.iter_mut().enumerate() {
        *item = ((i as f32 * 0.013).sin() * 0.75) + ((i as f32 * 0.007).cos() * 0.25);
    }
    let x_bits = f32_to_fp16_bits(&x);
    let x_q = fp16_bits_to_f32(&x_bits);
    let gamma = vec![1.0f32; DIM];
    let gamma_blob = WeightBlob::from_fp32(&gamma, 1, DIM as i32)?;
    let input_tensor = ANETensor::from_fp16(x_bits, vec![1, DIM, 1, SEQ])?;
    println!("✓ Inputs prepared\n");

    let io_bytes = DIM * SEQ * 2;
    println!("Compiling RMSNorm kernel...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_multi(
        &mil,
        &["@model_path/weights/rms_w.bin"],
        &[gamma_blob.as_bytes()],
        &[gamma_blob.len()],
        &[io_bytes],
        &[io_bytes],
    )?;
    println!("✓ RMSNorm kernel compiled\n");

    println!("Executing RMSNorm on ANE...");
    executor.write_input(0, input_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    let mut out_buf = vec![0u8; io_bytes];
    executor.read_output(0, &mut out_buf)?;
    let out = fp16_bits_to_f32(
        &out_buf
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>(),
    );

    println!("Computing CPU reference...");
    let cpu_ref = cpu_rmsnorm(&x_q, &gamma);
    println!("✓ CPU reference computed\n");

    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (a, b) in out.iter().zip(cpu_ref.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        mean_diff += diff;
    }
    mean_diff /= out.len() as f32;

    println!("Verification:");
    println!("  max diff:  {:.6}", max_diff);
    println!("  mean diff: {:.6}", mean_diff);

    if max_diff < 0.05 {
        println!("\n✅ RMSNorm proof-of-life completed successfully!");
    } else {
        println!("\n⚠️  RMSNorm ran, but the diff is larger than expected.");
    }

    Ok(())
}
