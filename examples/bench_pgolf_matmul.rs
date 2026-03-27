//! Sweep ANE vs CPU matmul across different matrix sizes to find the crossover point.

use rustane::wrapper::{ANECompiler, ANETensor};
use std::time::Instant;

const TILE_OC: usize = 128;
const ROW: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;
const ITERS: usize = 50;
const WARMUP: usize = 3;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        beta: f32,
        b: *const f32,
        ldb: i32,
        c: *mut f32,
        ldc: i32,
    );
}

fn cpu_sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], iters: usize) -> f64 {
    let mut c = vec![0.0f32; m * n];
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            cblas_sgemm(
                ROW,
                NO_TRANS,
                NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                0.0,
                b.as_ptr(),
                n as i32,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }
    start.elapsed().as_secs_f64() / iters as f64 * 1000.0
}

fn build_tiled_mil(ic: usize, oc: usize, seq: usize) -> String {
    let sp_total = seq + oc;
    let mut mil = String::new();
    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
    mil.push_str("{\n");
    mil.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        ic, sp_total
    ));
    mil.push_str(
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
    );
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", ic, sp_total));
    mil.push_str("        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n");
    mil.push_str(&format!("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", ic, seq));
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> act = slice_by_size(x = xh, begin = ba, size = sa)[name = string(\"act\")];\n", ic, seq));
    mil.push_str(&format!("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,{}])];\n", seq));
    mil.push_str(&format!("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", ic, oc));
    mil.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> wt = slice_by_size(x = xh, begin = bw, size = sw)[name = string(\"wt\")];\n", ic, oc));
    mil.push_str(&format!("        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,{},{}])];\n", ic, seq));
    mil.push_str(&format!("        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape = ra, x = act)[name = string(\"a2\")];\n", ic, seq));
    mil.push_str("        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n");
    mil.push_str(&format!("        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm = pm, x = a2)[name = string(\"a3\")];\n", seq, ic));
    mil.push_str(&format!("        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,{},{}])];\n", ic, oc));
    mil.push_str(&format!("        tensor<fp16, [1,1,{},{}]> W = reshape(shape = rw, x = wt)[name = string(\"W\")];\n", ic, oc));
    mil.push_str("        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n");
    mil.push_str(&format!("        tensor<fp16, [1,1,{},{}]> yh = matmul(transpose_x = bF, transpose_y = bF, x = a3, y = W)[name = string(\"mm\")];\n", seq, oc));
    mil.push_str(&format!("        tensor<fp16, [1,1,{},{}]> yt = transpose(perm = pm, x = yh)[name = string(\"yt\")];\n", oc, seq));
    mil.push_str(&format!("        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,{},1,{}])];\n", oc, seq));
    mil.push_str(&format!("        tensor<fp16, [1,{},1,{}]> yr = reshape(shape = ro, x = yt)[name = string(\"yr\")];\n", oc, seq));
    mil.push_str(
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
    );
    mil.push_str(&format!("        tensor<fp32, [1,{},1,{}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", oc, seq));
    mil.push_str("    } -> (y);\n");
    mil.push_str("}\n");
    mil
}

fn pack_tiled(ic: usize, oc: usize, seq: usize, weights: &[f32], act: &[f32]) -> Vec<f32> {
    let mut packed = vec![0.0f32; ic * (seq + oc)];
    for i in 0..ic {
        let dst = i * (seq + oc);
        packed[dst..dst + seq].copy_from_slice(&act[i * seq..(i + 1) * seq]);
        for o in 0..oc {
            packed[dst + seq + o] = weights[o * ic + i];
        }
    }
    packed
}

fn make_tile(
    ic: usize,
    oc: usize,
    seq: usize,
    full: &[f32],
    off: usize,
    this_oc: usize,
) -> Vec<f32> {
    let mut tile = vec![0.0f32; ic * (seq + this_oc)];
    for i in 0..ic {
        let src = i * (seq + oc);
        let dst = i * (seq + this_oc);
        tile[dst..dst + seq].copy_from_slice(&full[src..src + seq]);
        tile[dst + seq..dst + seq + this_oc]
            .copy_from_slice(&full[src + seq + off..src + seq + off + this_oc]);
    }
    tile
}

fn bench_ane(
    ic: usize,
    oc: usize,
    seq: usize,
    packed: &[f32],
) -> Result<(f64, usize), Box<dyn std::error::Error>> {
    let mut total_ms = 0.0f64;
    let mut num_tiles = 0usize;

    for tile_off in (0..oc).step_by(TILE_OC) {
        let this_oc = (oc - tile_off).min(TILE_OC);
        let mil = build_tiled_mil(ic, this_oc, seq);
        let tile = make_tile(ic, oc, seq, packed, tile_off, this_oc);
        let input_bytes = ic * (seq + this_oc) * 4;
        let output_bytes = this_oc * seq * 4;

        let mut compiler = ANECompiler::new();
        let mut exec = compiler.compile_single(&mil, None, &[input_bytes], &[output_bytes])?;
        let tensor = ANETensor::from_fp32(tile, vec![1, ic, 1, seq + this_oc])?;

        for _ in 0..WARMUP {
            exec.write_input(0, tensor.as_bytes())?;
            exec.eval()?;
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            exec.write_input(0, tensor.as_bytes())?;
            exec.eval()?;
        }
        total_ms += start.elapsed().as_secs_f64() * 1000.0;
        num_tiles += 1;
    }

    Ok((total_ms / ITERS as f64, num_tiles))
}

fn bench_shape(
    label: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    weights: &[f32],
    act: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let packed = pack_tiled(ic, oc, seq, weights, act);
    let cpu_ms = cpu_sgemm_nn(oc, seq, ic, weights, act, ITERS);

    match bench_ane(ic, oc, seq, &packed) {
        Ok((ane_ms, tiles)) => {
            let speedup = cpu_ms / ane_ms;
            let marker = if speedup > 1.0 { " <<<" } else { "" };
            println!(
                "{:<25} {:>8}x{:>8}x{:>6} {:>8.3} {:>8.3} {:>7.1}x{}",
                label, ic, oc, seq, ane_ms, cpu_ms, speedup, marker
            );
        }
        Err(e) => {
            println!(
                "{:<25} {:>8}x{:>8}x{:>6} {:>8} {:>8.3} ANE_ERR: {}",
                label, ic, oc, seq, "FAIL", cpu_ms, e
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;

    println!("=== ANE vs CPU Matmul Sweep ===");
    println!("Iters={}, Warmup={}, TileOC={}", ITERS, WARMUP, TILE_OC);
    println!();
    println!(
        "{:<25} {:>17} {:>8} {:>8} {:>8}",
        "Shape", "ic x oc x seq", "ANE ms", "CPU ms", "Speedup"
    );
    println!("{}", "-".repeat(90));

    let mut rng = 0u64;
    let mut rand_mat = |r: usize, c: usize| -> Vec<f32> {
        (0..r * c)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.02
            })
            .collect()
    };

    // Sweep: fixed seq=256, varying square dims
    println!("\n--- Square matmuls (seq=256) ---");
    for &dim in &[128, 256, 384, 512, 768, 1024] {
        let w = rand_mat(dim, dim);
        let a = rand_mat(dim, 256);
        bench_shape(&format!("square {}x{}", dim, dim), dim, dim, 256, &w, &a)?;
    }

    // Sweep: fixed dim=256, varying seq
    println!("\n--- Varying sequence length (dim=256) ---");
    for &sp in &[64, 128, 256, 512, 1024] {
        let w = rand_mat(256, 256);
        let a = rand_mat(256, sp);
        bench_shape(&format!("dim=256 seq={}", sp), 256, 256, sp, &w, &a)?;
    }

    // Non-square: MLP-like (2x expansion)
    println!("\n--- Non-square: MLP up (oc=2*ic, seq=256) ---");
    for &dim in &[256, 512, 768, 1024] {
        let w = rand_mat(dim * 2, dim);
        let a = rand_mat(dim, 256);
        bench_shape(
            &format!("up {}x{}", dim * 2, dim),
            dim,
            dim * 2,
            256,
            &w,
            &a,
        )?;
    }

    // Non-square: logits-like (vocab >> dim)
    println!("\n--- Non-square: logits (vocab=4*dim, seq=256) ---");
    for &dim in &[256, 512, 768, 1024] {
        let w = rand_mat(dim * 4, dim);
        let a = rand_mat(dim, 256);
        bench_shape(
            &format!("logits {}x{}", dim * 4, dim),
            dim,
            dim * 4,
            256,
            &w,
            &a,
        )?;
    }

    println!("\nOK");
    Ok(())
}
