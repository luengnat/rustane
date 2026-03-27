//! Test multi-output return with simple RMSNorm-like operations

use rustane::{
    ane::WeightBlob,
    init,
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;

    let dim: usize = 128;
    let seq: usize = 16;

    // Simple MIL: square and reduce_sum (2 outputs)
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, seq
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> sq = mul(x=x,y=x)[name = string(\"sq\")];\n",
        dim, seq
    ));
    mil.push_str("        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val = tensor<int32, [1]>([1])];\n");
    mil.push_str("        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,1,{}]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name = string(\"ss\")];\n",
        seq
    ));
    // Return 2 outputs
    mil.push_str(&format!("    }} -> (sq, ss);\n}}\n"));

    eprintln!("=== GENERATED MIL ===");
    eprintln!("{}", mil);
    eprintln!("=== END MIL ===\n");

    let input_bytes = dim * seq * 2; // FP16
    let output_sizes = vec![
        dim * seq * 2, // sq: same shape as input
        seq * 2,       // ss: [1,1,1,seq]
    ];

    eprintln!("Input: {} bytes, Outputs: {:?}", input_bytes, output_sizes);

    let mut compiler = ANECompiler::new();
    let exec = compiler.compile_multi(&mil, &[], &[], &[], &[input_bytes], &output_sizes);

    match exec {
        Ok(mut executor) => {
            eprintln!("Compile OK");

            // Create input FP16
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

            // Read outputs
            for (idx, size) in output_sizes.iter().enumerate() {
                let mut buf = vec![0u8; *size];
                if let Err(e) = executor.read_output(idx, &mut buf) {
                    eprintln!("read_output({}) FAILED: {e}", idx);
                } else {
                    eprintln!("read_output({}) OK: {} bytes", idx, size);
                }
            }

            eprintln!("\nSUCCESS: 2-output RMS test passed!");
        }
        Err(e) => {
            eprintln!("Compile FAILED: {e}");
            std::process::exit(2);
        }
    }

    Ok(())
}
