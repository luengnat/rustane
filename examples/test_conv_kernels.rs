//! Test which conv kernel sizes ANE supports.

use rustane::wrapper::ANECompiler;

fn conv_mil(ic: usize, oc: usize, h: usize, w: usize, kh: usize, kw: usize) -> String {
    let oh = if h >= kh { h - kh + 1 } else { 1 };
    let ow = if w >= kw { w - kw + 1 } else { 1 };
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, {}, {}]> x) {{\n",
        ic, h, w
    ));
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!("        tensor<fp16, [1, {}, {}, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", ic, h, w));
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, {}, {}]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, {}, {}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        oc, ic, kh, kw, oc, ic, kh, kw
    ));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> yh = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = xh)[name = string(\"conv\")];\n",
        oc, oh, ow
    ));
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, {}, {}]> y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n", oc, oh, ow
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    println!("=== ANE Conv Kernel Size Tests ===\n");

    let configs: Vec<(usize, usize, usize, usize, usize, usize, &str)> = vec![
        (64, 64, 64, 64, 1, 1, "conv1x1 [64→64] baseline"),
        (64, 192, 64, 64, 1, 1, "fused QKV conv1x1 [64→192]"),
        (64, 64, 64, 64, 1, 3, "conv1x3 (3-pos attention)"),
        (64, 64, 64, 64, 3, 3, "conv3x3 (3x3 neighborhood)"),
        (64, 64, 64, 64, 1, 8, "conv1x8 (8-pos attention)"),
        (64, 64, 64, 64, 1, 16, "conv1x16 (16-pos attention)"),
        (64, 64, 64, 64, 1, 32, "conv1x32 (32-pos attention)"),
        (64, 64, 64, 64, 1, 64, "conv1x64 (FULL SEQ attention!)"),
        (16, 16, 16, 16, 1, 16, "conv1x16 small [16,seq=16]"),
        (16, 16, 32, 32, 1, 32, "conv1x32 small [16,seq=32]"),
        (32, 32, 32, 32, 1, 32, "conv1x32 med [32,seq=32]"),
        (32, 32, 64, 64, 1, 64, "conv1x64 med [32,seq=64]"),
    ];

    for (ic, oc, h, w, kh, kw, desc) in &configs {
        let mil = conv_mil(*ic, *oc, *h, *w, *kh, *kw);
        let in_bytes = 1 * ic * h * w * 4;
        let oh = if *h >= *kh { *h - *kh + 1 } else { 1 };
        let ow = if *w >= *kw { *w - *kw + 1 } else { 1 };
        let out_bytes = 1 * oc * oh * ow * 4;
        let weight_bytes = oc * ic * kh * kw * 2;

        print!(
            "{:50} k={}x{} w={}KB ... ",
            desc,
            kh,
            kw,
            weight_bytes / 1024
        );
        use std::io::Write;
        std::io::stdout().flush().ok();

        let weights = vec![0u8; weight_bytes];
        let wname = "@model_path/weights/weight.bin";
        let wlen = weight_bytes;

        match ANECompiler::new().compile_multi(
            &mil,
            &[wname],
            &[weights.as_slice()],
            &[wlen],
            &[in_bytes],
            &[out_bytes],
        ) {
            Ok(mut exec) => match exec.eval() {
                Ok(_) => println!("✅"),
                Err(_) => println!("❌ RUN FAIL"),
            },
            Err(_) => println!("❌ COMPILE FAIL"),
        }
    }

    Ok(())
}
