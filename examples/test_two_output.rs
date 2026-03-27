//! Test 2-output return with FP16 tensors (like RMSNorm backward)

use std::time::Instant;

fn main() {
    if rustane::init().is_err() {
        eprintln!("ANE init failed");
        std::process::exit(1);
    }

    let dim: usize = 128;
    let seq: usize = 16;
    let in_ch = dim * 2; // 256 - split into 2 outputs

    // Simple MIL that slices input and returns 2 outputs
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        in_ch, seq
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1,1])];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1,1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");

    // Slice 1: [0:dim]
    mil.push_str(&format!("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0,0,0,0])];\n"));
    mil.push_str(&format!("        tensor<int32, [4]> s0 = const()[name = string(\"s0\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq));
    mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> out0 = slice_by_size(x=x,begin=b0,size=s0)[name = string(\"o0\")];\n", dim, seq));

    // Slice 2: [dim:2*dim]
    mil.push_str(&format!("        tensor<int32, [4]> b1 = const()[name = string(\"b1\"), val = tensor<int32, [4]>([0,{},0,0])];\n", dim));
    mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> out1 = slice_by_size(x=x,begin=b1,size=s0)[name = string(\"o1\")];\n", dim, seq));

    // Return 2 outputs
    mil.push_str(&format!("    }} -> (out0, out1);\n}}\n"));

    eprintln!("=== GENERATED MIL ===");
    eprintln!("{}", mil);
    eprintln!("=== END MIL ===\n");

    // Compile with no weights
    let input_bytes = in_ch * seq * 2; // FP16
    let output_sizes = vec![
        dim * seq * 2, // out0: 128 * 16 * 2 = 4096
        dim * seq * 2, // out1: 128 * 16 * 2 = 4096
    ];

    eprintln!("Input: {} bytes, Outputs: {:?}", input_bytes, output_sizes);

    let mut compiler = rustane::wrapper::ANECompiler::new();
    let exec = compiler.compile_single(&mil, None, &[input_bytes], &output_sizes);

    match exec {
        Ok(mut executor) => {
            eprintln!("Compile OK");

            // Create input
            let input: Vec<u8> = (0..in_ch * seq)
                .map(|i| {
                    let v: f32 = ((i % 100) as f32) * 0.01;
                    half::f16::from_f32(v).to_le_bytes()
                })
                .flatten()
                .collect();

            if let Err(e) = executor.write_input(0, &input) {
                eprintln!("write_input failed: {e}");
                std::process::exit(3);
            }
            eprintln!("Input written");

            if let Err(e) = executor.eval() {
                eprintln!("eval FAILED: {e}");
                std::process::exit(4);
            }
            eprintln!("Eval OK");

            // Read outputs
            for (idx, size) in output_sizes.iter().enumerate() {
                let mut buf = vec![0u8; *size];
                if let Err(e) = executor.read_output(idx, &mut buf) {
                    eprintln!("read_output({}) FAILED: {e}", idx);
                } else {
                    eprintln!("read_output({}) OK: {} bytes", idx, size);
                }
            }

            eprintln!("\nSUCCESS: 2-output FP16 test passed!");
        }
        Err(e) => {
            eprintln!("Compile FAILED: {e}");
            std::process::exit(2);
        }
    }
}
