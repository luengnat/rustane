//! Test: compile the EXACT ObjC reference MIL string to isolate the issue.

use rustane::wrapper::ANECompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let D: usize = 64;
    let S: usize = 64;
    let total_ch = D + D * D;

    // Build the ObjC reference MIL piece by piece to avoid format escaping issues
    let mut objc_mil = String::new();
    objc_mil.push_str("program(1.3)\n");
    objc_mil.push_str("[buildInfo = dict<string, string>({\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"})]\n");
    objc_mil.push_str("{\n");
    objc_mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        total_ch, S
    ));
    objc_mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    objc_mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, S));
    // NOTE: ObjC has NO spaces around = in op arguments, and uses [0,0,0,0] not [0, 0, 0, 0]
    objc_mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
    objc_mil.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D, S));
    objc_mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=xh,begin=b0,size=sa)[name=string(\"act\")];\n", D, S));
    objc_mil.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,{},0,0])];\n", D));
    objc_mil.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D*D, S));
    objc_mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> wf = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wf\")];\n", D*D, S));
    // ObjC declares ws BEFORE sw1
    objc_mil.push_str(&format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", D, D));
    objc_mil.push_str(&format!("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1,{},1,1])];\n", D*D));
    objc_mil.push_str(&format!("        tensor<fp16, [1,{},1,1]> wf1 = slice_by_size(x=wf,begin=b0,size=sw1)[name=string(\"wf1\")];\n", D*D));
    objc_mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W = reshape(shape=ws,x=wf1)[name=string(\"W\")];\n",
        D, D
    ));
    // ObjC reshapes act to [1,1,D,S] not [1,1,S,D]
    objc_mil.push_str(&format!("        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", D, S));
    objc_mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    objc_mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape=as2,x=act)[name=string(\"a2\")];\n",
        D, S
    ));
    objc_mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
        S, D
    ));
    objc_mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    objc_mil.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> yh = matmul(transpose_x = bF, transpose_y = bF, x = a3, y = W)[name = string(\"mm\")];\n", S, D));
    objc_mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n",
        D, S
    ));
    objc_mil.push_str(&format!("        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D, S));
    objc_mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> yr = reshape(shape=os,x=yt)[name=string(\"yr\")];\n",
        D, S
    ));
    objc_mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    objc_mil.push_str(&format!("        tensor<fp32, [1,{},1,{}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", D, S));
    objc_mil.push_str("    } -> (y);\n");
    objc_mil.push_str("}\n");

    println!("=== ObjC Reference MIL ({} bytes) ===", objc_mil.len());
    println!("{}", objc_mil);

    let input_bytes = total_ch * S * 4;
    let output_bytes = D * S * 4;
    println!(
        "\nInput: {} bytes, Output: {} bytes",
        input_bytes, output_bytes
    );

    println!("\n=== Compiling ObjC reference MIL... ===");
    match ANECompiler::new().compile_multi(
        &objc_mil,
        &[],
        &[],
        &[],
        &[input_bytes],
        &[output_bytes],
    ) {
        Ok(_) => println!("COMPILE SUCCESS!"),
        Err(e) => {
            let e_str = e.to_string();
            if e_str.len() > 800 {
                println!("COMPILE FAILED: {}...", &e_str[..800]);
            } else {
                println!("COMPILE FAILED: {}", e_str);
            }
        }
    }

    Ok(())
}
