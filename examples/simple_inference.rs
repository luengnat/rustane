//! Simple inference example
//!
//! Demonstrates basic ANE inference with a linear layer.

use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Simple Inference Example");
    println!("=====================================\n");

    // Initialize ANE runtime
    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    // Define layer dimensions
    let channels = 256;
    let spatial = 64;

    // Create a simple linear layer MIL program
    println!(
        "Creating MIL program for channel mixer (channels={}, spatial={})...",
        channels, spatial
    );
    let mut mil_program = String::new();
    mil_program.push_str("program(1.3)\n");
    mil_program.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil_program.push_str("{\n");
    mil_program.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        channels, spatial
    ));
    mil_program.push_str("        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n");
    mil_program.push_str("        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil_program.push_str("        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil_program.push_str("        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil_program.push_str(
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n",
    );
    mil_program.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil_program.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
        channels, spatial
    ));
    mil_program.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        channels, channels, channels, channels
    ));
    mil_program.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n",
        channels, spatial
    ));
    mil_program.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil_program.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        channels, spatial
    ));
    mil_program.push_str("    } -> (y);\n");
    mil_program.push_str("}\n");
    println!("✓ MIL program created\n");

    // Prepare weights (identity matrix for demonstration)
    println!("Preparing weights...");
    let mut weight_data = vec![0.0f32; channels * channels];
    // Create identity-like weights (first input_size rows are identity)
    for i in 0..channels {
        weight_data[i * channels + i] = 1.0; // Identity diagonal
    }
    println!("✓ Weights prepared ({} x {})", channels, channels);

    // Create weight blob
    println!("Building ANE weight blob...");
    let weight_blob = WeightBlob::from_fp32(&weight_data, channels as i32, channels as i32)?;
    println!("✓ Weight blob created ({} bytes)\n", weight_blob.len());

    // Compile the kernel
    println!("Compiling ANE kernel...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(
        &mil_program,
        Some(weight_blob.as_bytes()),
        &[channels * spatial * 4], // FP32 = 4 bytes per element
        &[channels * spatial * 4],
    )?;
    println!("✓ Kernel compiled\n");

    // Prepare input tensor
    println!("Preparing input tensor...");
    let input_data: Vec<f32> = (0..channels * spatial).map(|i| i as f32).collect();
    let input_tensor = ANETensor::from_fp32(input_data.clone(), vec![1, channels, 1, spatial])?;
    println!("✓ Input tensor created");
    println!("  Input: {:?}...", &input_data[..5]);

    // Write input and execute
    println!("\nExecuting kernel...");
    executor.write_input(0, input_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    // Read output
    println!("\nReading output...");
    let mut output_buf = vec![0u8; channels * spatial * 4];
    executor.read_output(0, &mut output_buf)?;

    // Convert output to FP32
    let output_data: Vec<f32> = output_buf
        .chunks_exact(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)
        })
        .collect();

    println!("✓ Output read");
    println!("  Output: {:?}...", &output_data[..5]);

    // Verify results (should match input due to identity weights)
    println!("\nVerifying results...");
    let mut correct = 0;
    for i in 0..input_data.len() {
        if (output_data[i] - input_data[i]).abs() < 0.01 {
            correct += 1;
        }
    }
    println!("✓ {}/{} values correct", correct, input_data.len());

    println!("\n✅ Inference completed successfully!");
    Ok(())
}
