//! Test simple FP16 conv which we know works from rmsnorm_pol.rs

use rustane::{
    ane::WeightBlob,
    init,
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;

    let dim: usize = 128;
    let seq: usize = 16;

    // Simple MIL: conv 1x1 with identity weights
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, seq
    ));
    mil.push_str("        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"), val=tensor<string, []>(\"valid\")];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"), val=tensor<int32, [2]>([1,1])];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"), val=tensor<int32, [2]>([1,1])];\n");
    mil.push_str("        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"), val=tensor<int32, []>(1)];\n");

    // Weight: [dim, dim, 1, 1] identity-ish
    mil.push_str(&format!("        tensor<fp16, [{}, {}, 1, 1]> W = const()[name=tensor<string, []>(\"W\"), val=tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w.bin\"), offset=tensor<uint64, []>(64)))];\n",
        dim, dim, dim, dim));

    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=tensor<string, []>(\"conv0\")];\n",
        dim, seq));

    mil.push_str(&format!("    }} -> (out);\n}}\n"));

    eprintln!("MIL generated, creating weight blob...");

    // Create identity-ish weights
    let weights: Vec<f32> = (0..dim * dim)
        .map(|i| if i % (dim + 1) == 0 { 1.0 } else { 0.0 }) // Diagonal identity
        .collect();
    let blob = WeightBlob::from_f32(&weights, dim, dim)?;

    eprintln!("Weight blob created: {} bytes", blob.len());

    let input_bytes = dim * seq * 2; // FP16
    let output_bytes = dim * seq * 2; // FP16

    eprintln!(
        "Compiling... Input={} bytes, Output={} bytes",
        input_bytes, output_bytes
    );

    let mut compiler = ANECompiler::new();
    let exec = compiler.compile_multi(
        &mil,
        &["@model_path/weights/w.bin"],
        &[blob.as_bytes()],
        &[blob.len()],
        &[input_bytes],
        &[output_bytes],
    );

    match exec {
        Ok(mut executor) => {
            eprintln!("Compile OK");

            // Create input tensor
            let input: Vec<u16> = (0..dim * seq)
                .map(|i| {
                    let v: f32 = ((i % 100) as f32) * 0.01;
                    half::f16::from_f32(v).to_bits()
                })
                .collect();
            let input_bytes: Vec<u8> = input.iter().flat_map(|b| b.to_le_bytes()).collect();

            if let Err(e) = executor.write_input(0, &input_bytes) {
                eprintln!("write_input failed: {e}");
                std::process::exit(3);
            }
            eprintln!("Input written");

            if let Err(e) = executor.eval() {
                eprintln!("eval FAILED: {e}");
                std::process::exit(4);
            }
            eprintln!("Eval OK");

            let mut buf = vec![0u8; output_bytes];
            if let Err(e) = executor.read_output(0, &mut buf) {
                eprintln!("read_output FAILED: {e}");
            } else {
                eprintln!("read_output OK: {} bytes", output_bytes);
            }

            eprintln!("\nSUCCESS: FP16 conv test passed!");
        }
        Err(e) => {
            eprintln!("Compile FAILED: {e}");
            std::process::exit(2);
        }
    }

    Ok(())
}
