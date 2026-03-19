//! Fused QKV projection example
//!
//! Demonstrates a small transformer-shaped ANE kernel:
//! three 1x1 conv projections run in one MIL graph, then concatenated.

use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Fused QKV Example");
    println!("================================\n");

    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    let dim = 64;
    let seq = 32;

    println!("Building fused QKV MIL graph (dim={}, seq={})...", dim, seq);
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        dim, seq
    ));
    mil.push_str("        string d1 = const()[name = string(\"d1\"), val = string(\"fp16\")];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> x16 = cast(dtype = d1, x = x)[name = string(\"cx\")];\n",
        dim, seq
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");

    for name in ["Wq", "Wk", "Wv"] {
        mil.push_str(&format!(
            "        tensor<fp16, [{d}, {d}, 1, 1]> {n} = const()[name = string(\"{n}\"), val = tensor<fp16, [{d}, {d}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{name}.bin\"), offset = uint64(64)))];\n",
            d = dim,
            n = name,
            name = name.to_lowercase()
        ));
    }

    for (out_name, weight_name) in [("q", "Wq"), ("k", "Wk"), ("v", "Wv")] {
        mil.push_str(&format!(
            "        tensor<fp16, [1, {d}, 1, {s}]> {out} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {w}, x = x16)[name = string(\"c{out}\")];\n",
            d = dim,
            s = seq,
            out = out_name,
            w = weight_name
        ));
    }

    mil.push_str("        int32 ax = const()[name = string(\"ax\"), val = int32(1)];\n");
    mil.push_str("        bool inter = const()[name = string(\"il\"), val = bool(false)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> qkv = concat(axis = ax, interleave = inter, values = (q, k, v))[name = string(\"cat\")];\n",
        dim * 3,
        seq
    ));
    mil.push_str("        string d2 = const()[name = string(\"d2\"), val = string(\"fp32\")];\n");
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = d2, x = qkv)[name = string(\"co\")];\n",
        dim * 3,
        seq
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    println!("✓ MIL graph built\n");

    println!("Preparing identity weights...");
    let mut q = vec![0.0f32; dim * dim];
    let mut k = vec![0.0f32; dim * dim];
    let mut v = vec![0.0f32; dim * dim];
    for i in 0..dim {
        q[i * dim + i] = 1.0;
        k[i * dim + i] = 1.0;
        v[i * dim + i] = 1.0;
    }
    let wq = WeightBlob::from_fp32(&q, dim as i32, dim as i32)?;
    let wk = WeightBlob::from_fp32(&k, dim as i32, dim as i32)?;
    let wv = WeightBlob::from_fp32(&v, dim as i32, dim as i32)?;
    println!("✓ Weight blobs prepared\n");

    println!("Compiling ANE kernel...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_multi(
        &mil,
        &[
            "@model_path/weights/wq.bin",
            "@model_path/weights/wk.bin",
            "@model_path/weights/wv.bin",
        ],
        &[wq.as_bytes(), wk.as_bytes(), wv.as_bytes()],
        &[wq.len(), wk.len(), wv.len()],
        &[dim * seq * 4],
        &[dim * 3 * seq * 4],
    )?;
    println!("✓ Kernel compiled\n");

    println!("Preparing input tensor...");
    let input_data: Vec<f32> = (0..dim * seq).map(|i| i as f32).collect();
    let input_tensor = ANETensor::from_fp32(input_data.clone(), vec![1, dim, 1, seq])?;
    println!("✓ Input tensor created");

    println!("\nExecuting fused QKV kernel...");
    executor.write_input(0, input_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    let mut output_buf = vec![0u8; dim * 3 * seq * 4];
    executor.read_output(0, &mut output_buf)?;
    let output_data: Vec<f32> = output_buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut correct = 0usize;
    for channel in 0..dim {
        for t in 0..seq {
            let idx = channel * seq + t;
            if (output_data[idx] - input_data[idx]).abs() < 0.01 {
                correct += 1;
            }
            let k_idx = dim * seq + idx;
            if (output_data[k_idx] - input_data[idx]).abs() < 0.01 {
                correct += 1;
            }
            let v_idx = dim * 2 * seq + idx;
            if (output_data[v_idx] - input_data[idx]).abs() < 0.01 {
                correct += 1;
            }
        }
    }
    println!("✓ {} / {} values correct", correct, dim * seq * 3);
    println!("\n✅ Fused QKV example completed successfully!");
    Ok(())
}
