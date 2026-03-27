//! ANE Dynamic Matmul Benchmark
//!
//! This benchmark follows the upstream ANE "dynamic matmul" layout:
//! activations and weights are packed into one input tensor, then sliced
//! inside MIL before a matmul. This avoids the rectangular weight-layout
//! issues that the simpler linear benchmark hit.

use rustane::{
    init,
    wrapper::{ANECompiler, ANETensor},
};
use std::time::Instant;

const ITERATIONS: usize = 5;
const WARMUP: usize = 1;
const SEQ: usize = 64;

fn benchmark_cpu_dynamic(ic: usize, oc: usize, seq: usize) -> (f64, f64, f64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CPU Dynamic Matmul (IC={}, OC={}, SEQ={})", ic, oc, seq);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let total_input = ic * (seq + oc);
    let input: Vec<f32> = (0..total_input)
        .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
        .collect();

    for _ in 0..WARMUP {
        let mut output = vec![0.0f32; oc * seq];
        for o in 0..oc {
            for t in 0..seq {
                for i in 0..ic {
                    let act = input[i * (seq + oc) + t];
                    let w = input[i * (seq + oc) + seq + o];
                    output[o * seq + t] += act * w;
                }
            }
        }
        let _ = output;
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut output = vec![0.0f32; oc * seq];
        for o in 0..oc {
            for t in 0..seq {
                for i in 0..ic {
                    let act = input[i * (seq + oc) + t];
                    let w = input[i * (seq + oc) + seq + o];
                    output[o * seq + t] += act * w;
                }
            }
        }
        let _ = output;
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
    println!("  Throughput: {:.1} ops/sec", ops_per_sec);

    (total_ms, avg_ms, ops_per_sec)
}

fn benchmark_ane_dynamic(
    ic: usize,
    oc: usize,
    seq: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ANE Dynamic Matmul (IC={}, OC={}, SEQ={})", ic, oc, seq);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

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

    let total_input = ic * (seq + oc);
    let input_data: Vec<f32> = (0..total_input)
        .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
        .collect();
    let input_bytes = total_input * 4;
    let output_bytes = oc * seq * 4;

    println!("  Input: {} bytes (FP32 packed)", input_bytes);
    println!("  Output: {} bytes (FP32)", output_bytes);

    println!("  Compiling for ANE...");
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(&mil, None, &[input_bytes], &[output_bytes])?;
    println!("  ✓ Compiled");

    let input_tensor = ANETensor::from_fp32(input_data, vec![1, ic, 1, seq + oc])?;

    println!("  Warming up ({} iterations)...", WARMUP);
    for _ in 0..WARMUP {
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;
    }

    println!("  Benchmarking execution ({} iterations)...", ITERATIONS);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;
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
    println!("  Throughput: {:.1} ops/sec", ops_per_sec);

    Ok((total_ms, avg_ms, ops_per_sec))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - ANE Dynamic Matmul Benchmark");
    println!("=========================================\n");

    let avail = rustane::HardwareAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available");
        return Ok(());
    }
    println!();

    init()?;
    println!("✓ ANE initialized\n");

    let configs = vec![(64, 64, SEQ), (128, 128, SEQ)];
    let mut results = Vec::new();

    for (ic, oc, seq) in configs {
        println!("\n");
        println!("═══════════════════════════════════════════════════════");
        println!("Configuration: IC={}, OC={}, SEQ={}", ic, oc, seq);
        println!("Operations: {} mul-add operations", ic * oc * seq);

        let (_cpu_total, cpu_avg, _cpu_ops) = benchmark_cpu_dynamic(ic, oc, seq);
        match benchmark_ane_dynamic(ic, oc, seq) {
            Ok((_ane_total, ane_avg, ane_ops)) => {
                let speedup = cpu_avg / ane_avg;
                results.push((ic, oc, seq, cpu_avg, ane_avg, speedup, ane_ops));
                println!("\n  ⚡ Speedup: {:.1}x faster on ANE", speedup);
            }
            Err(e) => {
                println!("\n  ❌ ANE benchmark failed: {}", e);
                println!("  Note: This may be due to MIL format issues or ANE limitations");
            }
        }
    }

    println!("\n\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:<14} {:>12} {:>12} {:>10} {:>14}",
        "Config", "CPU (ms)", "ANE (ms)", "Speedup", "ANE ops/sec"
    );
    println!("{}", "-".repeat(70));

    for (ic, oc, seq, cpu_avg, ane_avg, speedup, ane_ops) in &results {
        let config = format!("{}×{}×{}", ic, oc, seq);
        println!(
            "{:<14} {:>12.3} {:>12.3} {:>9.2}x {:>14.1}",
            config, cpu_avg, ane_avg, speedup, ane_ops,
        );
    }

    println!("\n✅ Benchmark complete!");
    Ok(())
}
