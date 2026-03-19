//! Attention proof-of-life example
//!
//! Runs `scaled_dot_product_attention` on ANE and checks the result against
//! a CPU reference implementation.

use half::f16;
use rustane::{
    init,
    wrapper::{ANECompiler, ANETensor},
};

const HEADS: usize = 12;
const SEQ: usize = 8;
const HD: usize = 64;

fn build_sdpa_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, {}, {}]> q, tensor<fp16, [1, {}, {}, {}]> k, tensor<fp16, [1, {}, {}, {}]> v) {{\n",
        HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n",
        HEADS, SEQ, HD
    ));
    mil.push_str("    } -> (att);\n");
    mil.push_str("}\n");
    mil
}

fn cpu_sdpa(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; HEADS * SEQ * HD];
    let scale = 1.0f32 / (HD as f32).sqrt();

    for h in 0..HEADS {
        for t in 0..SEQ {
            let mut scores = [0.0f32; SEQ];
            let mut max_score = f32::NEG_INFINITY;
            for t2 in 0..SEQ {
                let mut s = 0.0f32;
                for d in 0..HD {
                    let q_idx = ((h * SEQ + t) * HD) + d;
                    let k_idx = ((h * SEQ + t2) * HD) + d;
                    s += q[q_idx] * k[k_idx];
                }
                s *= scale;
                scores[t2] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut denom = 0.0f32;
            for t2 in 0..SEQ {
                scores[t2] = (scores[t2] - max_score).exp();
                denom += scores[t2];
            }
            for t2 in 0..SEQ {
                scores[t2] /= denom;
            }

            for d in 0..HD {
                let mut val = 0.0f32;
                for t2 in 0..SEQ {
                    let v_idx = ((h * SEQ + t2) * HD) + d;
                    val += scores[t2] * v[v_idx];
                }
                out[((h * SEQ + t) * HD) + d] = val;
            }
        }
    }

    out
}

fn f32_to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Attention Proof of Life");
    println!("====================================\n");

    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    let mil = build_sdpa_mil();
    println!("✓ MIL graph built\n");

    println!("Preparing input tensors...");
    let mut q = vec![0.0f32; HEADS * SEQ * HD];
    let mut k = vec![0.0f32; HEADS * SEQ * HD];
    let mut v = vec![0.0f32; HEADS * SEQ * HD];
    for (i, item) in q.iter_mut().enumerate() {
        *item = ((i as f32 * 17.0).sin() * 0.5) + 0.1;
    }
    for (i, item) in k.iter_mut().enumerate() {
        *item = ((i as f32 * 13.0).cos() * 0.5) - 0.2;
    }
    for (i, item) in v.iter_mut().enumerate() {
        *item = ((i as f32 * 11.0).sin() * 0.25) + 0.05;
    }
    let q16 = f32_to_fp16_bits(&q);
    let k16 = f32_to_fp16_bits(&k);
    let v16 = f32_to_fp16_bits(&v);
    let q_tensor = ANETensor::from_fp16(q16, vec![1, HEADS, SEQ, HD])?;
    let k_tensor = ANETensor::from_fp16(k16, vec![1, HEADS, SEQ, HD])?;
    let v_tensor = ANETensor::from_fp16(v16, vec![1, HEADS, SEQ, HD])?;
    println!("✓ Inputs prepared\n");

    let io_bytes = HEADS * SEQ * HD * 2;
    println!("Compiling SDPA kernel...");
    let mut compiler = ANECompiler::new();
    let mut executor =
        compiler.compile_single(&mil, None, &[io_bytes, io_bytes, io_bytes], &[io_bytes])?;
    println!("✓ SDPA kernel compiled\n");

    println!("Executing SDPA on ANE...");
    executor.write_input(0, q_tensor.as_bytes())?;
    executor.write_input(1, k_tensor.as_bytes())?;
    executor.write_input(2, v_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    let mut out_buf = vec![0u8; io_bytes];
    executor.read_output(0, &mut out_buf)?;
    let out = fp16_bytes_to_f32(&out_buf);

    println!("Computing CPU reference...");
    let cpu_ref = cpu_sdpa(&q, &k, &v);
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
        println!("\n✅ Attention proof-of-life completed successfully!");
    } else {
        println!("\n⚠️  SDPA ran, but the diff is larger than expected.");
    }

    Ok(())
}
