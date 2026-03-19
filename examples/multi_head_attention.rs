//! Multi-head attention example
//!
//! This example demonstrates how to use the MultiHeadAttention layer
//! with the verified scaled_dot_product_attention operation on ANE.

use half::f16;
use rustane::{
    init,
    layers::{Layer, MultiHeadAttentionBuilder},
    wrapper::{ANECompiler, ANETensor},
};

const BATCH: usize = 1;
const HEADS: usize = 12;
const SEQ: usize = 8;
const HEAD_DIM: usize = 64;
const EMBED_DIM: usize = HEADS * HEAD_DIM; // 768

fn build_sdpa_mil_working() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, {}, {}]> q, tensor<fp16, [1, {}, {}, {}]> k, tensor<fp16, [1, {}, {}, {}]> v) {{\n",
        HEADS, SEQ, HEAD_DIM, HEADS, SEQ, HEAD_DIM, HEADS, SEQ, HEAD_DIM
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n",
        HEADS, SEQ, HEAD_DIM
    ));
    mil.push_str("    } -> (att);\n");
    mil.push_str("}\n");
    mil
}

fn build_sdpa_mil() -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, {}, {}]> q, tensor<fp16, [1, {}, {}, {}]> k, tensor<fp16, [1, {}, {}, {}]> v) {{\n",
        HEADS, SEQ, HEAD_DIM, HEADS, SEQ, HEAD_DIM, HEADS, SEQ, HEAD_DIM
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"sdpa\")];\n",
        HEADS, SEQ, HEAD_DIM
    ));
    mil.push_str("    } -> (att);\n");
    mil.push_str("}\n");
    mil
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
    println!("🍎 Rustane - Multi-Head Attention Example");
    println!("========================================\n");

    // Create the MultiHeadAttention layer
    println!("Creating MultiHeadAttention layer...");
    let mha = MultiHeadAttentionBuilder::new(EMBED_DIM, HEADS)
        .with_name("mha_example")
        .build()?;

    println!("✓ Layer created");
    println!("  Name: {}", mha.name());
    println!("  Parameters: {}\n", mha.num_parameters());

    // Initialize ANE runtime
    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    // Build MIL program for SDPA
    println!("Building MIL program...");
    // Use the same MIL as causal_attention for now
    let mil = build_sdpa_mil_working();
    println!("✓ MIL program built\n");
    println!("DEBUG - MIL program:\n{}", mil);

    // Prepare input tensors (Q, K, V)
    // In a real scenario, these would come from the Q, K, V projections
    println!("Preparing input tensors...");
    let mut q = vec![0.0f32; BATCH * HEADS * SEQ * HEAD_DIM];
    let mut k = vec![0.0f32; BATCH * HEADS * SEQ * HEAD_DIM];
    let mut v = vec![0.0f32; BATCH * HEADS * SEQ * HEAD_DIM];

    // Fill with test data
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

    let q_tensor = ANETensor::from_fp16(q16, vec![BATCH, HEADS, SEQ, HEAD_DIM])?;
    let k_tensor = ANETensor::from_fp16(k16, vec![BATCH, HEADS, SEQ, HEAD_DIM])?;
    let v_tensor = ANETensor::from_fp16(v16, vec![BATCH, HEADS, SEQ, HEAD_DIM])?;
    println!("✓ Inputs prepared: [1, {}, {}, {}]\n", HEADS, SEQ, HEAD_DIM);

    // Calculate I/O sizes
    let io_bytes = BATCH * HEADS * SEQ * HEAD_DIM * 2; // FP16 = 2 bytes

    // Compile and execute
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
    println!("✓ Execution complete\n");

    // Read output
    let mut out_buf = vec![0u8; io_bytes];
    executor.read_output(0, &mut out_buf)?;
    let out = fp16_bytes_to_f32(&out_buf);

    println!("Output statistics:");
    println!("  Shape: [1, {}, {}, {}]", HEADS, SEQ, HEAD_DIM);
    println!("  Total elements: {}", out.len());
    println!(
        "  Min: {:.6}",
        out.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "  Max: {:.6}",
        out.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("  Mean: {:.6}", out.iter().sum::<f32>() / out.len() as f32);

    println!("\n✅ Multi-head attention example completed successfully!");
    println!("\nNote: This example uses pre-projected Q, K, V tensors.");
    println!("A complete implementation would include:");
    println!("  1. Q, K, V projections via Linear layers");
    println!("  2. Reshape operations for multi-head layout");
    println!("  3. SDPA operation (demonstrated here)");
    println!("  4. Output projection");

    Ok(())
}
