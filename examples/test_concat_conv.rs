//! Test: Does concat(same_size, same_size) → conv1x1 work on ANE?
use half::f16;
fn build_blob(w: &[f32]) -> Vec<u8> {
    let ws = w.len() * 2;
    let t = 128 + ws;
    let mut b = vec![0u8; t];
    b[0] = 1;
    b[4] = 2;
    b[64] = 0xEF;
    b[65] = 0xBE;
    b[66] = 0xAD;
    b[67] = 0xDE;
    b[68] = 1;
    b[72..76].copy_from_slice(&(ws as u32).to_le_bytes());
    b[80..84].copy_from_slice(&128u32.to_le_bytes());
    for (i, &v) in w.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        b[128 + i * 2] = (h & 0xFF) as u8;
        b[128 + i * 2 + 1] = (h >> 8) as u8;
    }
    b
}
fn to_fp16(d: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; d.len() * 2];
    for (i, &v) in d.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        b[i * 2] = (h & 0xFF) as u8;
        b[i * 2 + 1] = (h >> 8) as u8;
    }
    b
}
fn from_fp16(r: &[u8]) -> Vec<f32> {
    let mut o = vec![0.0f32; r.len() / 2];
    for i in 0..o.len() {
        let h = (r[i * 2] as u16) | ((r[i * 2 + 1] as u16) << 8);
        o[i] = f16::from_bits(h).to_f32();
    }
    o
}
fn rand_m(r: usize, c: usize, s: f32, seed: u64) -> Vec<f32> {
    (0..r * c)
        .map(|i| {
            let x = ((i as u64 * 2654435761)
                .wrapping_add(seed)
                .wrapping_mul(0x9E3779B97F4A7C15)
                >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * s
        })
        .collect()
}

fn main() {
    let d = 768;
    let inter = d * 4;
    let sp = 256;
    println!(
        "Test: concat([inter,sp], [inter,sp]) -> conv1x1([d, 2*inter]) at D={}, inter={}, sp={}",
        d, inter, sp
    );

    let wg = rand_m(inter, d, 0.02, 42);
    let wu = rand_m(inter, d, 0.02, 100);
    let gate = rand_m(inter, sp, 0.5, 300);
    let dup = rand_m(inter, sp, 0.5, 400);

    let mut wgt = vec![0.0f32; d * inter];
    for r in 0..d {
        for c in 0..inter {
            wgt[r * inter + c] = wg[c * d + r];
        }
    }
    let mut wut = vec![0.0f32; d * inter];
    for r in 0..d {
        for c in 0..inter {
            wut[r * inter + c] = wu[c * d + r];
        }
    }
    let mut rw = vec![0.0f32; d * 2 * inter];
    for r in 0..d {
        for c in 0..inter {
            rw[r * (2 * inter) + c] = wgt[r * inter + c];
        }
    }
    for r in 0..d {
        for c in 0..inter {
            rw[r * (2 * inter) + inter + c] = wut[r * inter + c];
        }
    }

    let mut m = String::new();
    m.push_str("program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}})]\n{\n");
    m.push_str(&format!("    func main<ios16>(tensor<fp16, [1, {}, 1, {}]> dgate, tensor<fp16, [1, {}, 1, {}]> dup) {{\n", inter, sp, inter, sp));
    m.push_str(&format!("        tensor<fp16, [{}, {}, 1, 1]> RW = const()[name = tensor<string, []>(\"RW\"), val = tensor<fp16, [{}, {}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/RW.bin\"), offset = tensor<uint64, []>(64)))]  ;\n", d, 2*inter, d, 2*inter));
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> cat = concat(values = (dgate, dup), axis = ax, interleave = ci);\n", 2*inter, sp));
    m.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> dx = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = RW, x = cat);\n", d, sp));
    m.push_str("    } -> (dx);\n}\n");

    println!("MIL program size: {} bytes", m.len());

    let names = vec!["@model_path/weights/RW.bin"];
    let blobs = vec![build_blob(&rw)];
    let lens = vec![blobs[0].len()];
    let brefs = vec![blobs[0].as_slice()];
    let bpe = 2usize;

    println!("Compiling...");
    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        &m,
        &names,
        &brefs,
        &lens,
        &[inter * sp * bpe, inter * sp * bpe],
        &[d * sp * bpe],
    ) {
        Ok(e) => {
            println!("Compiled OK");
            e
        }
        Err(err) => {
            println!("COMPILE FAILED: {:?}", err);
            return;
        }
    };

    println!("Writing inputs...");
    exec.write_input(0, &to_fp16(&gate[..])).expect("w");
    exec.write_input(1, &to_fp16(&dup[..])).expect("w");
    println!("Evaluating...");
    exec.eval().expect("e");
    println!("Reading output...");
    let _ = exec.read_output_vec(0).expect("r");
    println!("SUCCESS!");
}
