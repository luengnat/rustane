//! Test: isolate whether declaration order matters.

use rustane::wrapper::ANECompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    let D: usize = 64;
    let S: usize = 64;
    let total_ch = D + D * D;
    let input_bytes = total_ch * S * 4;
    let output_bytes = D * S * 4;

    let header = "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n";
    let footer = "    } -> (y);\n}\n";

    // Test C: ObjC logic (ws before sw1, reshape D,S), Rust spacing
    {
        let mut m = String::new();
        m.push_str(header);
        m.push_str(&format!(
            "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
            total_ch, S
        ));
        m.push_str(
            "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
        );
        m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, S));
        m.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
        m.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n", D, S));
        m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> act = slice_by_size(x = xh, begin = b0, size = sa)[name = string(\"act\")];\n", D, S));
        m.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0, {}, 0, 0])];\n", D));
        m.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n", D*D, S));
        m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> wf = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wf\")];\n", D*D, S));
        // ws BEFORE sw1 (ObjC order)
        m.push_str(&format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", D, D));
        m.push_str(&format!("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1, {}, 1, 1])];\n", D*D));
        m.push_str(&format!("        tensor<fp16, [1, {}, 1, 1]> wf1 = slice_by_size(x = wf, begin = b0, size = sw1)[name = string(\"wf1\")];\n", D*D));
        m.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> W = reshape(shape = ws, x = wf1)[name = string(\"W\")];\n", D, D));
        // ObjC reshape: [1,1,D,S]
        m.push_str(&format!("        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1, 1, {}, {}])];\n", D, S));
        m.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n");
        m.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> a2 = reshape(shape = as2, x = act)[name = string(\"a2\")];\n", D, S));
        m.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> a3 = transpose(perm = pm, x = a2)[name = string(\"a3\")];\n", S, D));
        m.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
        m.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> yh = matmul(transpose_x = bF, transpose_y = bF, x = a3, y = W)[name = string(\"mm\")];\n", S, D));
        m.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> yt = transpose(perm = pm, x = yh)[name = string(\"yt\")];\n", D, S));
        m.push_str(&format!("        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1, {}, 1, {}])];\n", D, S));
        m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> yr = reshape(shape = os, x = yt)[name = string(\"yr\")];\n", D, S));
        m.push_str(
            "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
        );
        m.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", D, S));
        m.push_str(footer);

        println!("=== Test C: ObjC logic (ws before sw1, reshape D,S), Rust spacing ===");
        match ANECompiler::new().compile_multi(&m, &[], &[], &[], &[input_bytes], &[output_bytes]) {
            Ok(_) => println!("COMPILE SUCCESS!"),
            Err(_) => println!("COMPILE FAILED"),
        }
    }

    // Test D: Rust order (sw1 before ws), ObjC reshape (D,S), ObjC spacing
    {
        let mut m = String::new();
        m.push_str(header);
        m.push_str(&format!(
            "    func main<ios18>(tensor<fp32, [1,{},1,{}]> x) {{\n",
            total_ch, S
        ));
        m.push_str(
            "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
        );
        m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", total_ch, S));
        m.push_str("        tensor<int32, [4]> b0 = const()[name = string(\"b0\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
        m.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D, S));
        m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=xh,begin=b0,size=sa)[name=string(\"act\")];\n", D, S));
        m.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,{},0,0])];\n", D));
        m.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D*D, S));
        m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> wf = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wf\")];\n", D*D, S));
        // sw1 BEFORE ws (Rust order)
        m.push_str("        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1,1,1,1])];\n");
        m.push_str(&format!("        tensor<fp16, [1,{},1,1]> wf1 = slice_by_size(x=wf,begin=b0,size=sw1)[name=string(\"wf1\")];\n", D*D));
        m.push_str(&format!("        tensor<int32, [4]> ws = const()[name = string(\"ws\"), val = tensor<int32, [4]>([1,1,{},{}])];\n", D, D));
        m.push_str(&format!(
            "        tensor<fp16, [1,1,{},{}]> W = reshape(shape=ws,x=wf1)[name=string(\"W\")];\n",
            D, D
        ));
        // ObjC reshape: [1,1,D,S]
        m.push_str(&format!("        tensor<int32, [4]> as2 = const()[name = string(\"as2\"), val = tensor<int32, [4]>([1,1,{},{}])];\n", D, S));
        m.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n");
        m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape=as2,x=act)[name=string(\"a2\")];\n", D, S));
        m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", S, D));
        m.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
        m.push_str(&format!("        tensor<fp16, [1, 1, {}, {}]> yh = matmul(transpose_x = bF, transpose_y = bF, x = a3, y = W)[name = string(\"mm\")];\n", S, D));
        m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", D, S));
        m.push_str(&format!("        tensor<int32, [4]> os = const()[name = string(\"os\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", D, S));
        m.push_str(&format!(
            "        tensor<fp16, [1,{},1,{}]> yr = reshape(shape=os,x=yt)[name=string(\"yr\")];\n",
            D, S
        ));
        m.push_str(
            "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
        );
        m.push_str(&format!("        tensor<fp32, [1,{},1,{}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", D, S));
        m.push_str(footer);

        println!("\n=== Test D: Rust order (sw1 before ws), ObjC reshape (D,S), ObjC spacing ===");
        match ANECompiler::new().compile_multi(&m, &[], &[], &[], &[input_bytes], &[output_bytes]) {
            Ok(_) => println!("COMPILE SUCCESS!"),
            Err(_) => println!("COMPILE FAILED"),
        }
    }

    Ok(())
}
