//! Test simple conv with weights - minimal backward-like program

use rustane::{
    ane::WeightBlob,
    init,
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;

    let dim: usize = 64;
    let hidden: usize = 128;
    let seq: usize = 16;

    // Simple MIL: slice + conv only
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim + 2 * hidden,
        seq
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1,1])];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1,1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");

    // Slice dffn
    mil.push_str("        tensor<int32, [4]> bd = const()[name = string(\"bd\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str(&format!(
        "        tensor<int32, [4]> sd = const()[name = string(\"sd\"), val = tensor<int32, [4]>([1,{},1,{}])];\n",
        dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name = string(\"s0\")];\n",
        dim, seq
    ));

    // Conv
    mil.push_str(&format!(
        "        tensor<fp16, [{},{},1,1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{},{},1,1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(64)))];\n",
        hidden, dim, hidden, dim
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=dffn)[name = string(\"conv\")];\n",
        hidden, seq
    ));

    mil.push_str("    } -> (out);\n}\n");

    eprintln!("=== MIL ===");
    eprintln!("{}", mil);
    eprintln!("=== END MIL ===\n");

    // Create weight blob
    let weights: Vec<f32> = (0..hidden * dim)
        .map(|i| ((i % 100) as f32) * 0.005)
        .collect();
    let blob = WeightBlob::from_f32(&weights, hidden, dim).unwrap();
    eprintln!("Weight blob: {} bytes", blob.len());

    let input_bytes = (dim + 2 * hidden) * seq * 2;
    let output_bytes = hidden * seq * 2;

    eprintln!(
        "Input: {} bytes, Output: {} bytes",
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

            // Create input FP16
            let input: Vec<u16> = (0..(dim + 2 * hidden) * seq)
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
                let fp16_vals: Vec<half::f16> = buf
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let f32_vals: Vec<f32> = fp16_vals.iter().map(|h| h.to_f32()).collect();
                eprintln!(
                    "First 4 outputs: [{:.4}, {:.4}, {:.4}, {:.4}]",
                    f32_vals[0], f32_vals[1], f32_vals[2], f32_vals[3]
                );
            }

            eprintln!("\nSUCCESS!");
        }
        Err(e) => {
            eprintln!("Compile FAILED: {e}");
            std::process::exit(2);
        }
    }

    Ok(())
}
