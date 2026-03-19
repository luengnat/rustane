//! Convolution layer example
//!
//! Demonstrates ANE inference with a non-1x1 convolutional layer.

use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Convolution Layer Example");
    println!("=====================================\n");

    let avail = rustane::ANEAvailability::check();
    println!("Platform check: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE is not available on this platform");
        return Ok(());
    }
    println!();

    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    let batch_size = 1;
    let input_channels = 3;
    let output_channels = 64;
    let input_size = 224;
    let kernel_size = 7;
    let stride = 2;
    let output_size = (input_size - kernel_size) / stride + 1;

    println!("Convolution parameters:");
    println!(
        "  Input: [{} × {} × {}]",
        batch_size, input_channels, input_size
    );
    println!(
        "  Kernel: {} × {} × {} × {}",
        output_channels, input_channels, kernel_size, kernel_size
    );
    println!(
        "  Output: [{} × {} × {}]",
        batch_size, output_channels, output_size
    );
    println!();

    println!("Creating MIL program...");
    let build_info = "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]";
    let mil_program = format!(
        "program(1.3)\n\
{build_info}\n\
{{\n    func main<ios18>(tensor<fp32, [{b}, {c}, {i}, {i}]> input) {{\n        string d1 = const()[name = string(\"d1\"), val = string(\"fp16\")];\n        tensor<fp16, [{b}, {c}, {i}, {i}]> input16 = cast(dtype = d1, x = input)[name = string(\"cast_input\")];\n        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([{st}, {st}])];\n        tensor<fp16, [{oc}, {c}, {k}, {k}]> weight = const()[name = string(\"weight\"), val = tensor<fp16, [{oc}, {c}, {k}, {k}]>(BLOBFILE(path = string(\"@model_path/weights/conv.bin\"), offset = uint64(64)))];\n        tensor<fp16, [{b}, {oc}, {o}, {o}]> conv = conv(dilations = [1, 1], groups = 1, pad = pd, pad_type = pt, strides = st, weight = weight, x = input16)[name = string(\"conv\")];\n        string d2 = const()[name = string(\"d2\"), val = string(\"fp32\")];\n        tensor<fp32, [{b}, {oc}, {o}, {o}]> y = cast(dtype = d2, x = conv)[name = string(\"cast_output\")];\n    }} -> (y);\n}}\n",
        build_info = build_info,
        b = batch_size,
        c = input_channels,
        i = input_size,
        oc = output_channels,
        k = kernel_size,
        o = output_size,
        st = stride
    );
    println!("✓ MIL program created\n");

    println!("Preparing convolution weights...");
    let weight_elements = output_channels * input_channels * kernel_size * kernel_size;
    let mut weight_data = vec![0.0f32; weight_elements];
    for c in 0..input_channels {
        for y in 0..kernel_size {
            for x in 0..kernel_size {
                let idx =
                    (0 * input_channels + c) * kernel_size * kernel_size + y * kernel_size + x;
                weight_data[idx] = 1.0 / (kernel_size * kernel_size) as f32;
            }
        }
    }
    println!(
        "✓ Weights prepared ({} × {} × {} × {})",
        output_channels, input_channels, kernel_size, kernel_size
    );

    println!("Building ANE weight blob...");
    let weight_blob = WeightBlob::from_fp32(&weight_data, weight_elements as i32, 4 as i32)?;
    println!("✓ Weight blob created ({} bytes)\n", weight_blob.len());

    let input_bytes = batch_size * input_channels * input_size * input_size * 4;
    let output_bytes = batch_size * output_channels * output_size * output_size * 4;

    println!("Compiling ANE kernel...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(
        &mil_program,
        Some(weight_blob.as_bytes()),
        &[input_bytes],
        &[output_bytes],
    )?;
    println!("✓ Kernel compiled\n");

    println!("Preparing input tensor...");
    let input_elements = batch_size * input_channels * input_size * input_size;
    let input_data: Vec<f32> = (0..input_elements)
        .map(|i| ((i % 256) as f32) / 255.0)
        .collect();
    let input_tensor = ANETensor::from_fp32(
        input_data.clone(),
        vec![batch_size, input_channels, input_size, input_size],
    )?;
    println!("✓ Input tensor created");
    println!("  Input shape: {:?}\n", input_tensor.shape());

    println!("Executing convolution...");
    executor.write_input(0, input_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    let mut output_buf = vec![0u8; output_bytes];
    executor.read_output(0, &mut output_buf)?;
    let output_data: Vec<f32> = output_buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("\nReading output...");
    println!("✓ Output read");
    println!(
        "  Output shape: [{} × {} × {}]",
        batch_size, output_channels, output_size
    );
    println!(
        "  Output range: [{:.4}, {:.4}]",
        output_data.iter().cloned().fold(f32::INFINITY, f32::min),
        output_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    println!("\n✅ Convolution inference completed successfully!");
    Ok(())
}
