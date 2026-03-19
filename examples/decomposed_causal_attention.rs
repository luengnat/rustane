//! Decomposed causal attention proof-of-life
//!
//! Runs QK^T on ANE, applies causal mask + softmax on CPU, then runs scores@V
//! on ANE. The layouts mirror the upstream ANE tests.

use half::f16;
use rustane::{
    init,
    wrapper::{ANECompiler, ANETensor},
};

const HEADS: usize = 12;
const SEQ: usize = 64;
const HD: usize = 64;

fn qkt_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, {}, {}]> q, tensor<fp16, [1, {}, {}, {}]> k) {{\n",
        HEADS, SEQ, HD, HEADS, SEQ, HD
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str("        bool bT = const()[name = string(\"bT\"), val = bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> scores = matmul(transpose_x = bF, transpose_y = bT, x = q, y = k)[name = string(\"qkt\")];\n",
        HEADS, SEQ, SEQ
    ));
    mil.push_str("    } -> (scores);\n");
    mil.push_str("}\n");
    mil
}

fn sv_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, {}, {}]> probs, tensor<fp16, [1, {}, {}, {}]> v) {{\n",
        HEADS, SEQ, SEQ, HEADS, SEQ, HD
    ));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> out = matmul(transpose_x = bF, transpose_y = bF, x = probs, y = v)[name = string(\"sv\")];\n",
        HEADS, SEQ, HD
    ));
    mil.push_str("    } -> (out);\n");
    mil.push_str("}\n");
    mil
}

fn to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn from_fp16_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn cpu_reference(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; HEADS * SEQ * HD];
    let scale = 1.0f32 / (HD as f32).sqrt();

    for h in 0..HEADS {
        for t in 0..SEQ {
            let mut logits = [0.0f32; SEQ];
            let mut max_logit = f32::NEG_INFINITY;
            for t2 in 0..SEQ {
                let mut s = 0.0f32;
                for d in 0..HD {
                    let q_idx = ((h * SEQ + t) * HD) + d;
                    let k_idx = ((h * SEQ + t2) * HD) + d;
                    s += q[q_idx] * k[k_idx];
                }
                if t2 > t {
                    s = -1.0e30;
                } else {
                    s *= scale;
                }
                logits[t2] = s;
                if s > max_logit {
                    max_logit = s;
                }
            }

            let mut denom = 0.0f32;
            for t2 in 0..SEQ {
                if t2 > t {
                    logits[t2] = 0.0;
                } else {
                    logits[t2] = (logits[t2] - max_logit).exp();
                    denom += logits[t2];
                }
            }
            for t2 in 0..=t {
                logits[t2] /= denom;
            }

            for d in 0..HD {
                let mut val = 0.0f32;
                for t2 in 0..=t {
                    let v_idx = ((h * SEQ + t2) * HD) + d;
                    val += logits[t2] * v[v_idx];
                }
                out[((h * SEQ + t) * HD) + d] = val;
            }
        }
    }

    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Decomposed Causal Attention");
    println!("=========================================\n");

    init()?;
    println!("✓ ANE runtime initialized\n");

    let mut q = vec![0.0f32; HEADS * SEQ * HD];
    let mut k = vec![0.0f32; HEADS * SEQ * HD];
    let mut v = vec![0.0f32; HEADS * SEQ * HD];
    for (i, item) in q.iter_mut().enumerate() {
        *item = ((i as f32 * 0.017).sin() * 0.05) + 0.01;
    }
    for (i, item) in k.iter_mut().enumerate() {
        *item = ((i as f32 * 0.013).cos() * 0.05) - 0.02;
    }
    for (i, item) in v.iter_mut().enumerate() {
        *item = ((i as f32 * 0.011).sin() * 0.05) + 0.01;
    }

    let q16 = to_fp16_bits(&q);
    let k16 = to_fp16_bits(&k);
    let v16 = to_fp16_bits(&v);
    let q_ref: Vec<f32> = q16
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();
    let k_ref: Vec<f32> = k16
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();
    let v_ref: Vec<f32> = v16
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();
    let q_tensor = ANETensor::from_fp16(q16, vec![1, HEADS, SEQ, HD])?;
    let k_tensor = ANETensor::from_fp16(k16, vec![1, HEADS, SEQ, HD])?;
    let v_tensor = ANETensor::from_fp16(v16, vec![1, HEADS, SEQ, HD])?;

    let qkt_mil = qkt_mil();
    let sv_mil = sv_mil();
    let qkt_bytes = HEADS * SEQ * SEQ * 2;
    let io_bytes = HEADS * SEQ * HD * 2;

    println!("Compiling QK^T kernel...");
    let mut qkt_compiler = ANECompiler::new();
    let mut qkt_exec =
        qkt_compiler.compile_single(&qkt_mil, None, &[io_bytes, io_bytes], &[qkt_bytes])?;
    println!("✓ QK^T kernel compiled");

    println!("Compiling scores@V kernel...");
    let mut sv_compiler = ANECompiler::new();
    let mut sv_exec =
        sv_compiler.compile_single(&sv_mil, None, &[qkt_bytes, io_bytes], &[io_bytes])?;
    println!("✓ scores@V kernel compiled\n");

    println!("Executing QK^T on ANE...");
    qkt_exec.write_input(0, q_tensor.as_bytes())?;
    qkt_exec.write_input(1, k_tensor.as_bytes())?;
    qkt_exec.eval()?;
    println!("✓ QK^T complete");

    let mut scores_buf = vec![0u8; qkt_bytes];
    qkt_exec.read_output(0, &mut scores_buf)?;
    let mut probs = from_fp16_bytes(&scores_buf);

    let scale = 1.0f32 / (HD as f32).sqrt();
    for h in 0..HEADS {
        for t in 0..SEQ {
            let row = &mut probs[h * SEQ * SEQ + t * SEQ..h * SEQ * SEQ + (t + 1) * SEQ];
            let mut max_logit = f32::NEG_INFINITY;
            for t2 in 0..SEQ {
                if t2 > t {
                    row[t2] = -1.0e30;
                } else {
                    row[t2] *= scale;
                }
                if row[t2] > max_logit {
                    max_logit = row[t2];
                }
            }
            let mut denom = 0.0f32;
            for t2 in 0..SEQ {
                if t2 > t {
                    row[t2] = 0.0;
                } else {
                    row[t2] = (row[t2] - max_logit).exp();
                    denom += row[t2];
                }
            }
            for t2 in 0..=t {
                row[t2] /= denom;
            }
        }
    }
    let probs_tensor = ANETensor::from_fp16(to_fp16_bits(&probs), vec![1, HEADS, SEQ, SEQ])?;

    println!("Executing scores@V on ANE...");
    sv_exec.write_input(0, probs_tensor.as_bytes())?;
    sv_exec.write_input(1, v_tensor.as_bytes())?;
    sv_exec.eval()?;
    println!("✓ scores@V complete");

    let mut out_buf = vec![0u8; io_bytes];
    sv_exec.read_output(0, &mut out_buf)?;
    let out = from_fp16_bytes(&out_buf);
    let cpu = cpu_reference(&q_ref, &k_ref, &v_ref);

    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (a, b) in out.iter().zip(cpu.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        mean_diff += diff;
    }
    mean_diff /= out.len() as f32;

    println!("\nVerification:");
    println!("  max diff:  {:.6}", max_diff);
    println!("  mean diff: {:.6}", mean_diff);

    if max_diff < 0.05 {
        println!("\n✅ Decomposed causal attention example completed successfully!");
    } else {
        println!("\n⚠️  Example ran, but the diff is larger than expected.");
    }

    Ok(())
}
