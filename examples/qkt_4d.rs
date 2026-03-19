//! 4D QK^T proof-of-life
//!
//! Compiles and runs a single 4D attention-score matmul kernel using the same
//! layout as the upstream ANE causal-attention tests.

use half::f16;
use rustane::{
    init,
    wrapper::{ANECompiler, ANETensor},
};

const HEADS: usize = 12;
const SEQ: usize = 64;
const HD: usize = 64;

fn build_qkt_mil() -> String {
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

fn to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn from_fp16_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - 4D QK^T Proof of Life");
    println!("===================================\n");

    init()?;
    println!("✓ ANE runtime initialized\n");

    let qkt_mil = build_qkt_mil();
    println!("✓ MIL graph built\n");

    let mut q = vec![0.0f32; HEADS * SEQ * HD];
    let mut k = vec![0.0f32; HEADS * SEQ * HD];
    for (i, item) in q.iter_mut().enumerate() {
        *item = ((i as f32 * 0.017).sin() * 0.5) + 0.1;
    }
    for (i, item) in k.iter_mut().enumerate() {
        *item = ((i as f32 * 0.013).cos() * 0.5) - 0.2;
    }

    let q_tensor = ANETensor::from_fp16(to_fp16_bits(&q), vec![1, HEADS, SEQ, HD])?;
    let k_tensor = ANETensor::from_fp16(to_fp16_bits(&k), vec![1, HEADS, SEQ, HD])?;
    let io_bytes = HEADS * SEQ * SEQ * 2;

    println!("Compiling QK^T kernel...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(
        &qkt_mil,
        None,
        &[HEADS * SEQ * HD * 2, HEADS * SEQ * HD * 2],
        &[io_bytes],
    )?;
    println!("✓ QK^T kernel compiled\n");

    println!("Executing QK^T on ANE...");
    executor.write_input(0, q_tensor.as_bytes())?;
    executor.write_input(1, k_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    let mut out_buf = vec![0u8; io_bytes];
    executor.read_output(0, &mut out_buf)?;
    let out = from_fp16_bytes(&out_buf);
    println!("✓ Read {} output values", out.len());
    println!("  first few: {:?}", &out[..8.min(out.len())]);

    println!("\n✅ 4D QK^T example completed successfully!");
    Ok(())
}
