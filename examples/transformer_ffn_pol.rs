//! Transformer FFN proof of life
//!
//! Runs a tiny square 1x1-conv MLP branch on ANE:
//! 1. First projection
//! 2. Gated SiLU computed on CPU
//! 3. Second projection
//!
//! This is the MLP half of a transformer block, kept intentionally square so we
//! stay inside the conv shapes we have already proven on this bridge.

use half::f16;
use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};

const SEQ: usize = 32;
const DIM: usize = 768;
const BATCH: usize = 1;

fn build_info() -> &'static str {
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]"
}

fn proj_mil(name: &str) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str(build_info());
    mil.push_str("\n{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {d}, 1, {s}]> x) {{\n",
        d = DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {d}, 1, {s}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        d = DIM,
        s = SEQ,
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [{d}, {d}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{d}, {d}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        d = DIM,
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {d}, 1, {s}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"proj\")];\n",
        d = DIM,
        s = SEQ,
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {d}, 1, {s}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        d = DIM,
        s = SEQ,
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn f32_to_fp16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn fp32_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..SEQ {
        let mut ss = 0.0f32;
        for c in 0..DIM {
            let v = x[c * SEQ + t];
            ss += v * v;
        }
        let scale = 1.0f32 / ((ss / DIM as f32) + eps).sqrt();
        for c in 0..DIM {
            out[c * SEQ + t] = x[c * SEQ + t] * scale;
        }
    }
    out
}

fn linear_channels_first(input: &[f32], weights: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; DIM * SEQ];
    for o in 0..DIM {
        for t in 0..SEQ {
            let mut sum = 0.0f32;
            for i in 0..DIM {
                sum += weights[o * DIM + i] * input[i * SEQ + t];
            }
            out[o * SEQ + t] = sum;
        }
    }
    out
}

fn max_mean_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_diff = max_diff.max(d);
        mean_diff += d;
    }
    (max_diff, mean_diff / a.len() as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Transformer FFN Proof of Life");
    println!("=========================================\n");

    init()?;
    println!("✓ ANE runtime initialized\n");

    let mut x = vec![0.0f32; DIM * SEQ];
    for (i, item) in x.iter_mut().enumerate() {
        *item = ((i as f32 * 0.021).sin() * 0.3) + ((i as f32 * 0.009).cos() * 0.1);
    }
    let x_norm = rms_norm(&x, 1e-6);

    let mut w1 = vec![0.0f32; DIM * DIM];
    let mut w3 = vec![0.0f32; DIM * DIM];
    let mut w2 = vec![0.0f32; DIM * DIM];
    for i in 0..DIM {
        w1[i * DIM + i] = 1.0;
        w3[i * DIM + i] = 0.5;
        w2[i * DIM + i] = 1.0;
    }
    let blob1 = WeightBlob::from_fp32(&w1, DIM as i32, DIM as i32)?;
    let blob3 = WeightBlob::from_fp32(&w3, DIM as i32, DIM as i32)?;
    let blob2 = WeightBlob::from_fp32(&w2, DIM as i32, DIM as i32)?;

    let proj1_mil = proj_mil("w1");
    let proj3_mil = proj_mil("w3");
    let proj2_mil = proj_mil("w2");

    let io_bytes = DIM * SEQ * 4;
    let mut c1 = ANECompiler::new();
    let mut e1 = c1.compile_single(&proj1_mil, Some(blob1.as_bytes()), &[io_bytes], &[io_bytes])?;
    let mut c3 = ANECompiler::new();
    let mut e3 = c3.compile_single(&proj3_mil, Some(blob3.as_bytes()), &[io_bytes], &[io_bytes])?;
    let mut c2 = ANECompiler::new();
    let mut e2 = c2.compile_single(&proj2_mil, Some(blob2.as_bytes()), &[io_bytes], &[io_bytes])?;

    let h1_cpu = linear_channels_first(&x_norm, &w1);
    let h3_cpu = linear_channels_first(&x_norm, &w3);
    let silu_cpu: Vec<f32> = h1_cpu.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let gate_cpu: Vec<f32> = silu_cpu
        .iter()
        .zip(h3_cpu.iter())
        .map(|(a, b)| a * b)
        .collect();
    let y_cpu = linear_channels_first(&gate_cpu, &w2);
    let final_cpu: Vec<f32> = x_norm
        .iter()
        .zip(y_cpu.iter())
        .map(|(a, b)| a + b)
        .collect();

    let input = ANETensor::from_fp32(x_norm.clone(), vec![BATCH, DIM, 1, SEQ])?;
    println!("Running ANE W1...");
    e1.write_input(0, input.as_bytes())?;
    e1.eval()?;
    let mut h1_buf = vec![0u8; io_bytes];
    e1.read_output(0, &mut h1_buf)?;
    let h1_ane = fp32_bytes_to_f32(&h1_buf);

    println!("Running ANE W3...");
    e3.write_input(0, input.as_bytes())?;
    e3.eval()?;
    let mut h3_buf = vec![0u8; io_bytes];
    e3.read_output(0, &mut h3_buf)?;
    let h3_ane = fp32_bytes_to_f32(&h3_buf);

    let (h1_max, h1_mean) = max_mean_diff(&h1_ane, &h1_cpu);
    let (h3_max, h3_mean) = max_mean_diff(&h3_ane, &h3_cpu);

    let silu_ane: Vec<f32> = h1_ane.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let gate_ane: Vec<f32> = silu_ane
        .iter()
        .zip(h3_ane.iter())
        .map(|(a, b)| a * b)
        .collect();
    let gate_tensor = ANETensor::from_fp32(gate_ane.clone(), vec![BATCH, DIM, 1, SEQ])?;

    println!("Running ANE W2...");
    e2.write_input(0, gate_tensor.as_bytes())?;
    e2.eval()?;
    let mut y_buf = vec![0u8; io_bytes];
    e2.read_output(0, &mut y_buf)?;
    let y_ane = fp32_bytes_to_f32(&y_buf);
    let final_ane: Vec<f32> = x_norm
        .iter()
        .zip(y_ane.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (y_max, y_mean) = max_mean_diff(&y_ane, &y_cpu);
    let (final_max, final_mean) = max_mean_diff(&final_ane, &final_cpu);

    println!("Verification:");
    println!("  W1 max diff:    {:.6}", h1_max);
    println!("  W1 mean diff:   {:.6}", h1_mean);
    println!("  W3 max diff:    {:.6}", h3_max);
    println!("  W3 mean diff:   {:.6}", h3_mean);
    println!("  W2 max diff:    {:.6}", y_max);
    println!("  W2 mean diff:   {:.6}", y_mean);
    println!("  Final max diff: {:.6}", final_max);
    println!("  Final mean diff:{:.6}", final_mean);

    if final_max < 0.05 {
        println!("\n✅ Transformer FFN proof-of-life completed successfully!");
    } else {
        println!("\n⚠️  FFN ran, but the diff was larger than expected.");
    }

    Ok(())
}
