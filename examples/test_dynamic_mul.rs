//! Test Approach 5: dynamic weights via element-wise multiply (simpler, proven to work).
//! Input: [1, D*2, 1, S] — first D channels = data, next D channels = weight scale
//! Output: [1, D, 1, S] — data * weight

use rustane::wrapper::ANECompiler;

fn dynamic_mul_mil(dim: usize, seq: usize) -> String {
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    // Input: [1, D*2, 1, S] fp32
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        dim * 2,
        seq
    ));
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n",
        dim * 2, seq
    ));
    // Slice data: channels 0..D
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str(&format!(
        "        tensor<int32, [4]> s0 = const()[name = string(\"s0\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n",
        dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> data = slice_by_size(x = xh, begin = b0, size = s0)[name = string(\"data\")];\n",
        dim, seq
    ));
    // Slice weight: channels D..2*D
    mil.push_str(&format!(
        "        tensor<int32, [4]> b1 = const()[name = string(\"b1\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n",
        dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> wt = slice_by_size(x = xh, begin = b1, size = s0)[name = string(\"wt\")];\n",
        dim, seq
    ));
    // Multiply
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> yh = mul(x = data, y = wt)[name = string(\"mul\")];\n",
        dim, seq
    ));
    // Cast to fp32
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n",
        dim, seq
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let dim = 256; // Match reference test
    let seq = 64;
    let mil = dynamic_mul_mil(dim, seq);
    let input_bytes = dim * 2 * seq * 4;
    let output_bytes = dim * seq * 4;

    println!("=== Dynamic Element-wise Multiply Test ===");
    println!("Dim={}, Seq={}", dim, seq);

    // Compile with no weights
    let mut exec =
        ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes])?;

    // Test: data * 2.0
    let input: Vec<f32> = (0..dim * seq)
        .map(|i| ((i % 100) as f32) * 0.01_f32)
        .collect();
    let weight = 2.0_f32;

    // Pack: [data | weight] where weight is replicated across spatial dim
    let mut packed = vec![0.0f32; dim * 2 * seq];
    packed[..dim * seq].copy_from_slice(&input);
    for i in 0..(dim * seq) {
        packed[dim * seq + i] = weight;
    }
    let packed_bytes: Vec<u8> = packed.iter().flat_map(|f| f.to_le_bytes()).collect();

    exec.write_input(0, &packed_bytes)?;
    exec.eval()?;
    let raw = exec.read_output_vec(0)?;
    let out: Vec<f32> = raw[..dim * seq * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Check: out[i] should be input[i] * 2.0
    let max_err: f32 = input
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a * weight - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "Input[0..3]:  [{:.4}, {:.4}, {:.4}]",
        input[0], input[1], input[2]
    );
    println!(
        "Output[0..3]: [{:.4}, {:.4}, {:.4}]",
        out[0], out[1], out[2]
    );
    println!(
        "Max error: {:.6} {}",
        max_err,
        if max_err < 0.1 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Speed test: 1000 weight changes
    use std::time::Instant;
    let start = Instant::now();
    for step in 0..1000 {
        let w = 1.0 + (step as f32) * 0.001;
        for i in 0..(dim * seq) {
            packed[dim * seq + i] = w;
        }
        let packed_bytes: Vec<u8> = packed.iter().flat_map(|f| f.to_le_bytes()).collect();
        exec.write_input(0, &packed_bytes)?;
        exec.eval()?;
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / 1000;
    println!("\n1000 steps: {:?}", elapsed);
    println!("Average: {}μs ({:.2}ms)", avg_us, avg_us as f64 / 1000.0);
    println!(
        "Throughput: {:.0} steps/sec",
        1000.0 / elapsed.as_secs_f64()
    );

    Ok(())
}
