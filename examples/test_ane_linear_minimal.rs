//! Minimal test to isolate ANE linear compilation issue

use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};

fn test_exact_simple_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test 1: Exact simple_inference.rs configuration");
    println!("  channels=256, spatial=64");

    let channels = 256;
    let spatial = 64;

    let mut mil_program = String::new();
    mil_program.push_str("program(1.3)\n");
    mil_program.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
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

    let mut weight_data = vec![0.0f32; channels * channels];
    for i in 0..channels {
        weight_data[i * channels + i] = 1.0;
    }

    let weight_blob = WeightBlob::from_fp32(&weight_data, channels as i32, channels as i32)?;

    let input_bytes = channels * spatial * 4;
    let output_bytes = channels * spatial * 4;

    println!("  Compiling...");
    let mut compiler = ANECompiler::new();
    match compiler.compile_single(
        &mil_program,
        Some(weight_blob.as_bytes()),
        &[input_bytes],
        &[output_bytes],
    ) {
        Ok(_) => println!("  ✓ SUCCESS"),
        Err(e) => println!("  ✗ FAILED: {}", e),
    }

    Ok(())
}

fn test_rectangular_weights() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nTest 2: Rectangular weights (256→512)");
    println!("  in_feat=256, out_feat=512, spatial=64");

    let in_feat = 256;
    let out_feat = 512;
    let spatial = 64;

    let mut mil_program = String::new();
    mil_program.push_str("program(1.3)\n");
    mil_program.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil_program.push_str("{\n");
    mil_program.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        in_feat, spatial
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
        in_feat, spatial
    ));
    mil_program.push_str(&format!(
        "        tensor<fp16, [{}, {}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        out_feat, in_feat, out_feat, in_feat
    ));
    mil_program.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n",
        out_feat, spatial
    ));
    mil_program.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil_program.push_str(&format!(
        "        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n",
        out_feat, spatial
    ));
    mil_program.push_str("    } -> (y);\n");
    mil_program.push_str("}\n");

    let weight_data: Vec<f32> = (0..in_feat * out_feat)
        .map(|i| ((i as f32 * 0.001) % 2.0) - 1.0)
        .collect();

    let weight_blob = WeightBlob::from_fp32(&weight_data, out_feat as i32, in_feat as i32)?;

    let input_bytes = in_feat * spatial * 4;
    let output_bytes = out_feat * spatial * 4;

    println!("  Compiling...");
    let mut compiler = ANECompiler::new();
    match compiler.compile_single(
        &mil_program,
        Some(weight_blob.as_bytes()),
        &[input_bytes],
        &[output_bytes],
    ) {
        Ok(_) => println!("  ✓ SUCCESS"),
        Err(e) => println!("  ✗ FAILED: {}", e),
    }

    Ok(())
}

fn test_small_spatial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nTest 3: Small spatial dimension (256→256, spatial=1)");

    let channels = 256;
    let spatial = 1;

    let mut mil_program = String::new();
    mil_program.push_str("program(1.3)\n");
    mil_program.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
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

    let mut weight_data = vec![0.0f32; channels * channels];
    for i in 0..channels {
        weight_data[i * channels + i] = 1.0;
    }

    let weight_blob = WeightBlob::from_fp32(&weight_data, channels as i32, channels as i32)?;

    let input_bytes = channels * spatial * 4;
    let output_bytes = channels * spatial * 4;

    println!("  Compiling...");
    let mut compiler = ANECompiler::new();
    match compiler.compile_single(
        &mil_program,
        Some(weight_blob.as_bytes()),
        &[input_bytes],
        &[output_bytes],
    ) {
        Ok(_) => println!("  ✓ SUCCESS"),
        Err(e) => println!("  ✗ FAILED: {}", e),
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 ANE Linear Compilation Test");
    println!("==============================\n");

    let avail = rustane::ANEAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available");
        return Ok(());
    }

    init()?;
    println!("✓ ANE initialized\n");

    test_exact_simple_inference()?;
    test_rectangular_weights()?;
    test_small_spatial()?;

    println!("\n✅ Tests complete!");

    Ok(())
}
