//! Test: compile the Rust MIL with ObjC-style spacing (no spaces in shapes, no spaces around =)

use rustane::wrapper::ANECompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    // Rust logic but with ObjC spacing style
    let D: usize = 64;
    let S: usize = 64;
    let total_ch = D + D * D;

    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1,{},1,{}]> x) {{\n",
        total_ch, S
    ));
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, S));
    mil.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D, S));
    mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=xh,begin=b0,size=sa)[name=string(\"act\")];\n", D, S));
    mil.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,{},0,0])];\n", D));
    mil.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D*D, S));
    mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> wf = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wf\")];\n", D*D, S));
    // Use Rust order (sw1 before ws) to test if order matters
    mil.push_str("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1,1,1,1])];\n");
    mil.push_str(&format!("        tensor<fp16, [1,{},1,1]> wf1 = slice_by_size(x=wf,begin=b0,size=sw1)[name=string(\"wf1\")];\n", D*D));
    mil.push_str(&format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1,1,{},{}])];\n", D, D));
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W = reshape(shape=ws,x=wf1)[name=string(\"W\")];\n",
        D, D
    ));
    // Use Rust's reshape order: [1,1,S,D]
    mil.push_str(&format!("        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1,1,{},{}])];\n", S, D));
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n");
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape=as2,x=act)[name=string(\"a2\")];\n",
        S, D
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
        D, S
    ));
    mil.push_str("        bool bf = const()[name = string(\"bf\"), val = bool(false)];\n");
    mil.push_str(&format!("        tensor<fp16, [1,1,{},{}]> yh = matmul(transpose_x = bf, transpose_y = bf, x = a3, y = W)[name = string(\"mm\")];\n", S, D));
    mil.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n",
        D, S
    ));
    mil.push_str(&format!("        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D, S));
    mil.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> yr = reshape(shape=os,x=yt)[name=string(\"yr\")];\n",
        D, S
    ));
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!("        tensor<fp32, [1,{},1,{}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", D, S));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");

    println!("=== Test A: Rust logic, ObjC spacing (order: sw1 before ws, reshape: S,D) ===");
    let input_bytes = total_ch * S * 4;
    let output_bytes = D * S * 4;
    match ANECompiler::new().compile_multi(&mil, &[], &[], &[], &[input_bytes], &[output_bytes]) {
        Ok(_) => println!("COMPILE SUCCESS!"),
        Err(e) => println!("COMPILE FAILED"),
    }

    // Now test B: with Rust-style spacing (spaces in shapes, spaces around =)
    let mut mil_b = String::new();
    mil_b.push_str("program(1.3)\n");
    mil_b.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil_b.push_str("{\n");
    mil_b.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        total_ch, S
    ));
    mil_b.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil_b.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, S));
    mil_b.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil_b.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n", D, S));
    mil_b.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n", D, S));
    mil_b.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", D));
    mil_b.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n", D*D, S));
    mil_b.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n", D*D, S));
    mil_b.push_str("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, 1, 1, 1])];\n");
    mil_b.push_str(&format!("        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n", D*D));
    mil_b.push_str(&format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", D, D));
    mil_b.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n", D, D));
    mil_b.push_str(&format!("        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", S, D));
    mil_b.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
    mil_b.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> a2 = reshape(shape = as2, x = act)[name = string(\"a2\")];\n", S, D));
    mil_b.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> a3 = transpose(perm = pm, x = a2)[name = string(\"a3\")];\n", D, S));
    mil_b.push_str("        bool bf = const()[name = string(\"bf\"), val = bool(false)];\n");
    mil_b.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> yh = matmul(transpose_x = bf, transpose_y = bf, x = a3, y = W)[name = string(\"mm\")];\n", S, D));
    mil_b.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> yt = transpose(perm = pm, x = yh)[name = string(\"yt\")];\n", D, S));
    mil_b.push_str(&format!("        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n", D, S));
    mil_b.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yr = reshape(shape = os, x = yt)[name = string(\"yr\")];\n", D, S));
    mil_b.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil_b.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", D, S));
    mil_b.push_str("    } -> (y);\n");
    mil_b.push_str("}\n");

    println!("\n=== Test B: Rust logic, Rust spacing (order: sw1 before ws, reshape: S,D) ===");
    match ANECompiler::new().compile_multi(&mil_b, &[], &[], &[], &[input_bytes], &[output_bytes]) {
        Ok(_) => println!("COMPILE SUCCESS!"),
        Err(e) => println!("COMPILE FAILED"),
    }

    Ok(())
}
