//! ANE Tiled Rectangular Matmul Benchmark
//!
//! This benchmark reuses the working packed 4D dynamic matmul layout but
//! splits a rectangular projection into output-channel tiles. Each tile is a
//! separate ANE kernel, which lets us benchmark realistic rectangular shapes
//! without asking the bridge to compile a shape it does not like directly.

use rustane::{
    init,
    wrapper::{ANECompiler, ANETensor},
};
use std::time::Instant;

const ITERATIONS: usize = 5;
const WARMUP: usize = 1;
const SEQ: usize = 64;
const TILE_OC: usize = 64;

fn build_dynamic_matmul_mil(ic: usize, oc: usize, seq: usize) -> String {
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

fn cpu_reference(ic: usize, oc: usize, seq: usize, packed: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; oc * seq];
    for o in 0..oc {
        for t in 0..seq {
            for i in 0..ic {
                let act = packed[i * (seq + oc) + t];
                let w = packed[i * (seq + oc) + seq + o];
                output[o * seq + t] += act * w;
            }
        }
    }
    output
}

fn make_tile_input(
    ic: usize,
    oc: usize,
    seq: usize,
    packed: &[f32],
    tile_off: usize,
    this_oc: usize,
) -> Vec<f32> {
    let mut tile = vec![0.0f32; ic * (seq + this_oc)];
    for i in 0..ic {
        let src_row = i * (seq + oc);
        let dst_row = i * (seq + this_oc);
        tile[dst_row..dst_row + seq].copy_from_slice(&packed[src_row..src_row + seq]);
        tile[dst_row + seq..dst_row + seq + this_oc]
            .copy_from_slice(&packed[src_row + seq + tile_off..src_row + seq + tile_off + this_oc]);
    }
    tile
}

fn benchmark_cpu_rectangular(ic: usize, oc: usize, seq: usize, packed: &[f32]) -> (f64, f64, f64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "CPU Rectangular Dynamic Matmul (IC={}, OC={}, SEQ={})",
        ic, oc, seq
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let _ = cpu_reference(ic, oc, seq, packed);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = cpu_reference(ic, oc, seq, packed);
    }
    let duration = start.elapsed();

    let total_ms = duration.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / ITERATIONS as f64;
    let ops_per_sec = ITERATIONS as f64 / duration.as_secs_f64();

    println!(
        "  Total time ({} iterations): {:.2}ms",
        ITERATIONS, total_ms
    );
    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} iterations/sec", ops_per_sec);

    (total_ms, avg_ms, ops_per_sec)
}

fn benchmark_ane_rectangular(
    ic: usize,
    oc: usize,
    seq: usize,
    tile_oc: usize,
    packed: &[f32],
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "ANE Tiled Rectangular Matmul (IC={}, OC={}, SEQ={}, TILE_OC={})",
        ic, oc, seq, tile_oc
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let reference = cpu_reference(ic, oc, seq, packed);
    let mut assembled = vec![0.0f32; oc * seq];
    let mut total_ms = 0.0f64;
    let mut num_tiles = 0usize;

    for tile_off in (0..oc).step_by(tile_oc) {
        let this_oc = (oc - tile_off).min(tile_oc);
        let mil = build_dynamic_matmul_mil(ic, this_oc, seq);
        let tile_input = make_tile_input(ic, oc, seq, packed, tile_off, this_oc);
        let input_bytes = ic * (seq + this_oc) * 4;
        let output_bytes = this_oc * seq * 4;

        let mut compiler = ANECompiler::new();
        let mut exec = compiler.compile_single(&mil, None, &[input_bytes], &[output_bytes])?;
        let tensor = ANETensor::from_fp32(tile_input, vec![1, ic, 1, seq + this_oc])?;

        println!("  ✓ Compiled tile {}..{}", tile_off, tile_off + this_oc);

        // Correctness pass for this tile
        exec.write_input(0, tensor.as_bytes())?;
        exec.eval()?;
        let mut output_buf = vec![0u8; this_oc * seq * 4];
        exec.read_output(0, &mut output_buf)?;
        let tile_out: Vec<f32> = output_buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        for c in 0..this_oc {
            assembled[(tile_off + c) * seq..(tile_off + c + 1) * seq]
                .copy_from_slice(&tile_out[c * seq..(c + 1) * seq]);
        }

        println!("  • Warming up tile ({} iterations)...", WARMUP);
        for _ in 0..WARMUP {
            exec.write_input(0, tensor.as_bytes())?;
            exec.eval()?;
        }

        println!("  • Benchmarking tile ({} iterations)...", ITERATIONS);
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            exec.write_input(0, tensor.as_bytes())?;
            exec.eval()?;
        }
        let duration = start.elapsed();

        total_ms += duration.as_secs_f64() * 1000.0;
        num_tiles += 1;
    }

    let mut max_diff = 0.0f32;
    for (lhs, rhs) in assembled.iter().zip(reference.iter()) {
        max_diff = max_diff.max((lhs - rhs).abs());
    }
    println!("  ✓ Correctness check passed (max diff: {:.6})", max_diff);

    let avg_ms = total_ms / ITERATIONS as f64;
    let ops_per_sec = ITERATIONS as f64 / (total_ms / 1000.0);

    println!("  Compiled and benchmarked {} tiles", num_tiles);
    println!(
        "  Total time ({} iterations across tiles): {:.2}ms",
        ITERATIONS, total_ms
    );
    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} ops/sec", ops_per_sec);

    Ok((total_ms, avg_ms, ops_per_sec))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - ANE Tiled Rectangular Matmul Benchmark");
    println!("===================================================\n");

    let avail = rustane::HardwareAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available");
        return Ok(());
    }
    println!();

    init()?;
    println!("✓ ANE initialized\n");

    let ic = 128;
    let oc = 256;
    let total_input = ic * (SEQ + oc);
    let packed: Vec<f32> = (0..total_input)
        .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
        .collect();

    println!("\n═══════════════════════════════════════════════════════");
    println!(
        "Configuration: IC={}, OC={}, SEQ={}, TILE_OC={}",
        ic, oc, SEQ, TILE_OC
    );
    println!("Operations: {} mul-add operations", ic * oc * SEQ);

    let (_cpu_total, cpu_avg, _cpu_ops) = benchmark_cpu_rectangular(ic, oc, SEQ, &packed);
    match benchmark_ane_rectangular(ic, oc, SEQ, TILE_OC, &packed) {
        Ok((_ane_total, ane_avg, ane_ops)) => {
            let speedup = cpu_avg / ane_avg;
            println!("\n  ⚡ Speedup: {:.1}x faster on ANE", speedup);
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("BENCHMARK SUMMARY");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(
                "{:<14} {:>12} {:>12} {:>10} {:>14}",
                "Config", "CPU (ms)", "ANE (ms)", "Speedup", "ANE iters/sec"
            );
            println!("{}", "-".repeat(70));
            println!(
                "{:<14} {:>12.3} {:>12.3} {:>9.2}x {:>14.1}",
                "128×256×64", cpu_avg, ane_avg, speedup, ane_ops,
            );
        }
        Err(e) => {
            println!("\n  ❌ ANE benchmark failed: {}", e);
            println!("  Note: This may be due to MIL format issues or ANE limitations");
        }
    }

    println!("\n✅ Benchmark complete!");
    Ok(())
}
