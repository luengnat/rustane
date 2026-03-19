//! Minimal convolution test

use rustane::{init, wrapper::ANECompiler, ANETensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;

    // Try the exact format from fused_qkv
    let mil = r#"
program(1.3)
[buildInfo = dict<string, string>({"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremlc-component-milinternal", ""}, {"coremltools-version", "9.0"})]
{
    func main<ios18>(tensor<fp32, [1, 3, 224, 224]> input) {
        string d1 = const()[name = string("d1"), val = string("fp16")];
        tensor<fp16, [1, 3, 224, 224]> input16 = cast(dtype = d1, x = input)[name = string("cx")];
        tensor<fp16, [64, 3, 7, 7]> weight = const()[name = string("weight"), val = tensor<fp16, [64, 3, 7, 7]>(BLOBFILE(path = string("@model_path/weights/conv.bin"), offset = uint64(0)))];
        string pt = const()[name = string("pt"), val = string("valid")];
        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([2, 2])];
        tensor<fp16, [1, 64, 109, 109]> conv = conv(dilations = [1, 1], groups = 1, pad = pd, pad_type = pt, strides = st, weight = weight, x = input16)[name = string("mm")];
        string d2 = const()[name = string("d2"), val = string("fp32")];
        tensor<fp32, [1, 64, 109, 109]> y = cast(dtype = d2, x = conv)[name = string("co")];
    } -> (y);
}
"#;

    println!("Testing minimal conv MIL...");
    let mut compiler = ANECompiler::new();

    // Try compiling with dummy weight blob
    let weight_data = vec![0.0f32; 64 * 3 * 7 * 7];
    let weight_blob =
        rustane::mil::WeightBlob::from_fp32(&weight_data, weight_data.len() as i32, 4)?;

    match compiler.compile_single(
        mil,
        Some(weight_blob.as_bytes()),
        &[1 * 3 * 224 * 224 * 4],
        &[1 * 64 * 109 * 109 * 4],
    ) {
        Ok(mut exec) => {
            println!("✓ Minimal conv MIL compiled!");
            return Ok(());
        }
        Err(e) => {
            println!("❌ Failed: {}", e);
            Err(e.into())
        }
    }
}
