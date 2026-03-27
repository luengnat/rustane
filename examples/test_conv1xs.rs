//! Test which conv kernel sizes ANE supports.
//! Uses WeightBlob (with proper 64-byte header) for weight data.

use rustane::ane::WeightBlob;
use rustane::wrapper::ANECompiler;

fn conv_mil(ic: usize, oc: usize, seq: usize, kh: usize, kw: usize) -> String {
    let oh = seq - kh + 1;
    let ow = seq - kw + 1;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        ic, seq
    ));
    mil.push_str(
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", ic, seq));
    mil.push_str("        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n");
    mil.push_str("        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    mil.push_str("        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    mil.push_str("        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n");
    // Weight: [oc, ic, kh, kw] flattened to [oc, ic*kh*kw] for WeightBlob
    let w_rows = oc;
    let w_cols = ic * kh * kw;
    mil.push_str(&format!(
        "        tensor<fp16, [{}, {}, {}, {}]> W = const()[name = string(\"W\"), val = tensor<fp16, [{}, {}, {}, {}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
        oc, ic, kh, kw, oc, ic, kh, kw
    ));
    mil.push_str(&format!(
        "        tensor<fp16, [1, {}, {}, {}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"conv\")];\n",
        oc, oh, ow
    ));
    mil.push_str(
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!(
        "        tensor<fp32, [1, {}, {}, {}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", oc, oh, ow
    ));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn test_one(ic: usize, oc: usize, seq: usize, kh: usize, kw: usize, desc: &str) -> bool {
    let mil = conv_mil(ic, oc, seq, kh, kw);
    let in_bytes = ic * seq * 4;
    let oh = seq - kh + 1;
    let ow = seq - kw + 1;
    let out_bytes = oc * oh * ow * 4;

    // WeightBlob expects [rows, cols] = [oc, ic*kh*kw]
    let w_rows = oc;
    let w_cols = ic * kh * kw;
    let weight_count = w_rows * w_cols;
    let weights: Vec<f32> = (0..weight_count).map(|i| 0.01).collect();
    let blob = WeightBlob::from_f32(&weights, w_rows, w_cols).unwrap();

    print!(
        "  {:45} k={}x{} w={:>4}KB ... ",
        desc,
        kh,
        kw,
        blob.as_bytes().len() / 1024
    );
    use std::io::Write;
    std::io::stdout().flush().ok();

    match ANECompiler::new().compile_multi(
        &mil,
        &["@model_path/weights/weight.bin"],
        &[blob.as_bytes()],
        &[blob.as_bytes().len()],
        &[in_bytes],
        &[out_bytes],
    ) {
        Ok(mut exec) => match exec.eval() {
            Ok(_) => {
                println!("✅");
                true
            }
            Err(_) => {
                println!("❌ run fail");
                false
            }
        },
        Err(_) => {
            println!("❌ compile fail");
            false
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;
    println!("=== ANE Conv Kernel Size Tests ===\n");

    println!("--- Baseline ---");
    test_one(64, 64, 64, 1, 1, "conv1x1 [64→64, seq=64]");

    println!("\n--- Fused Projections ---");
    test_one(64, 192, 64, 1, 1, "fused QKV [64→192, seq=64]");
    test_one(64, 128, 64, 1, 1, "wider proj [64→128, seq=64]");

    println!("\n--- Conv1xS (spatial attention-like) ---");
    test_one(64, 64, 64, 1, 3, "conv1x3 (local attention)");
    test_one(64, 64, 64, 1, 8, "conv1x8 (window attn)");
    test_one(64, 64, 64, 1, 16, "conv1x16 [64, seq=64]");
    test_one(64, 64, 64, 1, 32, "conv1x32 [64, seq=64]");
    test_one(64, 64, 64, 1, 64, "conv1x64 FULL SEQ!");
    test_one(32, 32, 32, 1, 32, "conv1x32 [32, seq=32]");
    test_one(16, 16, 16, 1, 16, "conv1x16 [16, seq=16]");

    println!("\n--- Conv2D ---");
    test_one(64, 64, 64, 3, 3, "conv3x3 [64, seq=64]");

    Ok(())
}
