//! Comprehensive benchmark: maderix/ANE patterns
//! Tests spatial packing, fp16 I/O, mega-kernels, and fusion scaling.
//!
//! Usage:
//!   cargo run --example bench_maderix_patterns -- <test_id> [dim] [seq] [batch]
//!
//! Test IDs:
//!   1  — Spatial packing (maderix-style) vs channel packing (our-style)
//!   2  — fp16 direct I/O vs fp32 cast I/O
//!   3  — Fused QKV projection (3 matmuls in spatial-packed layout)
//!   4  — Mega-kernel: QKV + attention scoring (Q@K^T + scale + softmax + @V)
//!   5  — Fused FFN: W1 + SiLU + gate + W2 + residual
//!   6  — Fusion scaling: 1/2/3/4/5 matmuls per eval
//!   7  — Dimension sweep: D=32,64,128,256,512 with S=64,128,256
//!
//! All tests run in subprocess isolation (ANE SIGSEGV protection).

use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_maderix_patterns <test_id> [dim] [seq]");
        eprintln!("  1=spatial_vs_channel  2=fp16_vs_fp32  3=fused_qkv");
        eprintln!("  4=mega_sdpa  5=fused_ffn  6=fusion_scaling  7=dim_sweep");
        std::process::exit(1);
    }

    let test_id: u32 = args[1].parse().unwrap_or(0);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let seq: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    match test_id {
        1 => test_spatial_vs_channel(dim, seq),
        2 => test_fp16_vs_fp32(dim, seq),
        3 => test_fused_qkv_spatial(dim, seq),
        4 => test_mega_sdpa(dim, seq),
        5 => test_fused_ffn(dim, seq),
        6 => test_fusion_scaling(dim, seq),
        7 => test_dim_sweep(),
        _ => {
            eprintln!("Unknown test_id: {}", test_id);
            std::process::exit(1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Spatial packing (maderix) vs Channel packing (ours)
// ═══════════════════════════════════════════════════════════════════════════
fn test_spatial_vs_channel(dim: usize, seq: usize) {
    println!(
        "=== Test 1: Spatial vs Channel Packing (D={}, S={}) ===",
        dim, seq
    );

    // A) Channel packing: [1, D + D*D, 1, S] — our approach
    let ch_total = dim + dim * dim;
    let ch_in_b = ch_total * seq * 2; // fp16
    let ch_out_b = dim * seq * 2;

    let ch_mil = channel_packed_matmul_mil(dim, seq);
    match compile_and_eval(&ch_mil, &[], &[], &[], ch_in_b, ch_out_b, "channel") {
        Some(us) => {
            let tflops = 2.0 * dim as f64 * dim as f64 * seq as f64 / (us as f64 * 1e-6) / 1e12;
            println!("  Channel packing:   {}us  ({:.2} TFLOPS)", us, tflops);
        }
        None => println!("  Channel packing:   FAIL"),
    }

    // B) Spatial packing: [1, IC, 1, SEQ+OC] — maderix approach
    let sp_in_b = dim * (seq + dim) * 2; // fp16
    let sp_out_b = dim * seq * 2;

    let sp_mil = spatial_packed_matmul_mil(dim, dim, seq);
    match compile_and_eval(&sp_mil, &[], &[], &[], sp_in_b, sp_out_b, "spatial") {
        Some(us) => {
            let tflops = 2.0 * dim as f64 * dim as f64 * seq as f64 / (us as f64 * 1e-6) / 1e12;
            println!("  Spatial packing:   {}us  ({:.1} TFLOPS)", us, tflops);
        }
        None => println!("  Spatial packing:   FAIL"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: fp16 direct I/O vs fp32 cast I/O
// ═══════════════════════════════════════════════════════════════════════════
fn test_fp16_vs_fp32(dim: usize, seq: usize) {
    println!("=== Test 2: fp16 vs fp32 I/O (D={}, S={}) ===", dim, seq);

    // A) fp32 I/O (our current: fp32→cast(fp16)→matmul→cast(fp32))
    let fp32_in_b = dim * (seq + dim) * 4; // fp32 input
    let fp32_out_b = dim * seq * 4;

    let fp32_mil = fp32_matmul_mil(dim, dim, seq);
    match compile_and_eval(&fp32_mil, &[], &[], &[], fp32_in_b, fp32_out_b, "fp32") {
        Some(us) => {
            let tflops = 2.0 * dim as f64 * dim as f64 * seq as f64 / (us as f64 * 1e-6) / 1e12;
            println!("  fp32 I/O (cast):  {}us  ({:.1} TFLOPS)", us, tflops);
        }
        None => println!("  fp32 I/O (cast):  FAIL"),
    }

    // B) fp16 I/O (maderix: direct fp16 input, no cast overhead)
    let fp16_in_b = dim * (seq + dim) * 2; // fp16 input
    let fp16_out_b = dim * seq * 2;

    let fp16_mil = fp16_matmul_mil(dim, dim, seq);
    match compile_and_eval(&fp16_mil, &[], &[], &[], fp16_in_b, fp16_out_b, "fp16") {
        Some(us) => {
            let tflops = 2.0 * dim as f64 * dim as f64 * seq as f64 / (us as f64 * 1e-6) / 1e12;
            println!("  fp16 I/O (direct): {}us  ({:.1} TFLOPS)", us, tflops);
        }
        None => println!("  fp16 I/O (direct): FAIL"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Fused QKV projection with spatial packing
// ═══════════════════════════════════════════════════════════════════════════
fn test_fused_qkv_spatial(dim: usize, seq: usize) {
    let heads = (dim / 64).max(1); // head_dim=64
    let hd = 64;
    let q_dim = heads * hd;
    let kv_dim = heads * hd; // MHA for now (no GQA)
    let hidden = dim * 4; // FFN hidden

    println!(
        "=== Test 3: Fused QKV Spatial (D={}, S={}, H={}, HD={}) ===",
        dim, seq, heads, hd
    );

    // Input: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM]
    //   sp[0:SEQ]                 = xnorm [DIM, SEQ]
    //   sp[SEQ:SEQ+Q_DIM]        = Wq [DIM, Q_DIM]
    //   sp[SEQ+Q_DIM:SEQ+Q_DIM+KV_DIM] = Wk [DIM, KV_DIM]
    //   sp[SEQ+Q_DIM+KV_DIM:...]  = Wv [DIM, KV_DIM]
    let sp_in = seq + q_dim + kv_dim + kv_dim;
    let in_b = dim * sp_in * 2; // fp16
                                // Output: [1, Q_DIM, 1, SEQ] — just Q for now
    let out_b = q_dim * seq * 2;

    let mil = fused_qkv_spatial_mil(dim, q_dim, kv_dim, heads, hd, seq, sp_in);
    match compile_and_eval(&mil, &[], &[], &[], in_b, out_b, "fused_qkv") {
        Some(us) => {
            let total_flops = 3.0 * 2.0 * dim as f64 * q_dim as f64 * seq as f64;
            let tflops = total_flops / (us as f64 * 1e-6) / 1e12;
            let per_matmul = us as f64 / 3.0;
            println!(
                "  Fused QKV (3 matmuls): {}us  ({:.2} TFLOPS, {:.0}us/matmul)",
                us, tflops, per_matmul
            );
        }
        None => println!("  Fused QKV: FAIL"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Mega-kernel SDPA (QKV + attention scoring)
// ═══════════════════════════════════════════════════════════════════════════
fn test_mega_sdpa(dim: usize, seq: usize) {
    let heads = (dim / 64).max(1);
    let hd = 64;
    let q_dim = heads * hd;

    println!(
        "=== Test 4: Mega SDPA (D={}, S={}, H={}, HD={}) ===",
        dim, seq, heads, hd
    );

    // Same input as Test 3 + causal mask as blobfile
    let sp_in = seq + q_dim + q_dim + q_dim;
    let in_b = dim * sp_in * 2;
    // Output: [1, Q_DIM, 1, SEQ] = attention output
    let out_b = q_dim * seq * 2;

    // Build causal mask blobfile
    let mask_data = build_causal_mask_fp16(seq);

    let mil = mega_sdpa_mil(dim, q_dim, heads, hd, seq, sp_in);
    match compile_and_eval(
        &mil,
        &["@model_path/weights/mask.bin"],
        &[&mask_data],
        &[mask_data.len()],
        in_b,
        out_b,
        "mega_sdpa",
    ) {
        Some(us) => {
            let qkv_flops = 3.0 * 2.0 * dim as f64 * q_dim as f64 * seq as f64;
            let attn_flops = 2.0 * q_dim as f64 * seq as f64 * seq as f64; // Q@K^T
            let av_flops = 2.0 * q_dim as f64 * seq as f64 * hd as f64; // attn@V
            let total = qkv_flops + attn_flops + av_flops;
            let tflops = total / (us as f64 * 1e-6) / 1e12;
            println!("  Mega SDPA: {}us  ({:.1} TFLOPS)", us, tflops);
            println!(
                "    QKV={:.0} + Q@K^T={:.0} + Attn@V={:.0} = {:.0} TFLOPS",
                qkv_flops / 1e9,
                attn_flops / 1e9,
                av_flops / 1e9,
                total / 1e9
            );
        }
        None => println!("  Mega SDPA: FAIL"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 5: Fused FFN (SwiGLU: W1 + SiLU + gate + W2 + residual)
// ═══════════════════════════════════════════════════════════════════════════
fn test_fused_ffn(dim: usize, seq: usize) {
    let hidden = dim * 4;

    println!(
        "=== Test 5: Fused FFN (D={}, H={}, S={}) ===",
        dim, hidden, seq
    );

    // Input: [1, DIM, 1, SEQ + SEQ + HIDDEN + HIDDEN + HIDDEN]
    //   sp[0:SEQ]                     = x2norm [DIM, SEQ]
    //   sp[SEQ:2*SEQ]                 = x2 [DIM, SEQ]  (for residual)
    //   sp[2*SEQ:2*SEQ+HIDDEN]       = W1 [DIM, HIDDEN]
    //   sp[2*SEQ+HIDDEN:2*SEQ+2*HIDDEN] = W3 [DIM, HIDDEN]
    //   sp[2*SEQ+2*HIDDEN:...]        = W2 [DIM, HIDDEN]
    let sp_in = 2 * seq + 3 * hidden;
    let in_b = dim * sp_in * 2;
    let out_b = dim * seq * 2;

    let mil = fused_ffn_mil(dim, hidden, seq, sp_in);
    match compile_and_eval(&mil, &[], &[], &[], in_b, out_b, "fused_ffn") {
        Some(us) => {
            let w1_flops = 2.0 * dim as f64 * hidden as f64 * seq as f64;
            let w3_flops = w1_flops;
            let w2_flops = 2.0 * hidden as f64 * dim as f64 * seq as f64;
            let total = w1_flops + w3_flops + w2_flops;
            let tflops = total / (us as f64 * 1e-6) / 1e12;
            println!("  Fused FFN: {}us  ({:.1} TFLOPS)", us, tflops);
            println!(
                "    W1={:.0} + W3={:.0} + W2={:.0} = {:.0} TFLOPS",
                w1_flops / 1e9,
                w3_flops / 1e9,
                w2_flops / 1e9,
                total / 1e9
            );
        }
        None => println!("  Fused FFN: FAIL"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 6: Fusion scaling (1 to 5 matmuls per eval)
// ═══════════════════════════════════════════════════════════════════════════
fn test_fusion_scaling(dim: usize, seq: usize) {
    println!("=== Test 6: Fusion Scaling (D={}, S={}) ===", dim, seq);
    println!(
        "  {:>2} matmuls | eval (us) | per-matmul (us) | TFLOPS",
        "N"
    );
    println!("  ----------+-----------+-----------------+--------");

    for n_matmuls in 1..=5u32 {
        let total_ch = dim + n_matmuls as usize * dim * dim;
        let in_b = total_ch * seq * 2;
        let out_b = dim * seq * 2;

        let mil = fusion_scaling_mil(dim, seq, n_matmuls);
        match compile_and_eval(
            &mil,
            &[],
            &[],
            &[],
            in_b,
            out_b,
            &format!("fuse{}", n_matmuls),
        ) {
            Some(us) => {
                let total_flops = n_matmuls as f64 * 2.0 * dim as f64 * dim as f64 * seq as f64;
                let tflops = total_flops / (us as f64 * 1e-6) / 1e12;
                let per_matmul = us as f64 / n_matmuls as f64;
                println!(
                    "  {:>9} | {:>9} | {:>15.0} | {:.2}",
                    n_matmuls, us, per_matmul, tflops
                );
            }
            None => {
                println!("  {:>9} | FAIL", n_matmuls);
                break; // stop if larger fusion fails
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 7: Dimension sweep
// ═══════════════════════════════════════════════════════════════════════════
fn test_dim_sweep() {
    println!("=== Test 7: Dimension Sweep ===");
    println!(
        "  {:>6} | {:>6} | {:>10} | {:>10} | {:>10} | {:>10}",
        "DIM", "SEQ", "1-matmul", "3-matmul", "fp16-I/O", "spatial"
    );
    println!("  -------+--------+------------+------------+------------+------------");

    let dims = [32, 64, 128, 256, 512];
    let seqs = [64, 128, 256];

    for &dim in &dims {
        for &seq in &seqs {
            let mut results = Vec::new();

            // 1 matmul channel-packed
            let ch1_in = (dim + dim * dim) * seq * 2;
            let ch1_out = dim * seq * 2;
            let mil1 = channel_packed_matmul_mil(dim, seq);
            results.push(
                match compile_and_eval(&mil1, &[], &[], &[], ch1_in, ch1_out, "sweep") {
                    Some(us) => format!("{:>10}", us),
                    None => "       FAIL".to_string(),
                },
            );

            // 3 matmuls channel-packed
            let ch3_in = (dim + 3 * dim * dim) * seq * 2;
            let ch3_out = dim * seq * 2;
            let mil3 = fusion_scaling_mil(dim, seq, 3);
            results.push(
                match compile_and_eval(&mil3, &[], &[], &[], ch3_in, ch3_out, "sweep") {
                    Some(us) => format!("{:>10}", us),
                    None => "       FAIL".to_string(),
                },
            );

            // fp16 direct I/O
            let fp16_in = dim * (seq + dim) * 2;
            let fp16_out = dim * seq * 2;
            let mil16 = fp16_matmul_mil(dim, dim, seq);
            results.push(
                match compile_and_eval(&mil16, &[], &[], &[], fp16_in, fp16_out, "sweep") {
                    Some(us) => format!("{:>10}", us),
                    None => "       FAIL".to_string(),
                },
            );

            // spatial packing
            let sp_in = dim * (seq + dim) * 2;
            let sp_out = dim * seq * 2;
            let mil_sp = spatial_packed_matmul_mil(dim, dim, seq);
            results.push(
                match compile_and_eval(&mil_sp, &[], &[], &[], sp_in, sp_out, "sweep") {
                    Some(us) => format!("{:>10}", us),
                    None => "       FAIL".to_string(),
                },
            );

            println!(
                "  {:>6} | {:>6} | {} | {} | {} | {}",
                dim, seq, results[0], results[1], results[2], results[3]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn compile_and_eval(
    mil: &str,
    weight_names: &[&str],
    weight_datas: &[&[u8]],
    weight_lens: &[usize],
    in_b: usize,
    out_b: usize,
    label: &str,
) -> Option<u128> {
    let mut exec = match rustane::wrapper::ANECompiler::new().compile_multi(
        mil,
        weight_names,
        weight_datas,
        weight_lens,
        &[in_b],
        &[out_b],
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  [{}] COMPILE_FAIL: {}", label, e);
            return None;
        }
    };

    // Use fp16 data for fp16-sized inputs (even bytes), fp32 for 4-byte-aligned
    let data = if in_b % 4 == 0 && in_b / 4 < 1_000_000 {
        make_fp32_data(in_b)
    } else {
        make_fp16_data(in_b)
    };
    exec.write_input(0, &data).unwrap();

    // Warm up
    if let Err(e) = exec.eval() {
        eprintln!("  [{}] EVAL_FAIL (warmup): {}", label, e);
        return None;
    }

    // Time 3 evals, take min
    let mut min_us = u128::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        if exec.eval().is_err() {
            eprintln!("  [{}] EVAL_FAIL (timed)", label);
            return None;
        }
        let us = start.elapsed().as_micros();
        min_us = min_us.min(us);
    }

    // Read output to verify non-zero
    let out = exec.read_output_vec(0).unwrap();
    let nz = out.iter().filter(|&&b| b != 0).count();
    if nz == 0 {
        eprintln!("  [{}] WARNING: output is all zeros", label);
    }

    Some(min_us)
}

fn make_fp16_data(size_bytes: usize) -> Vec<u8> {
    // Generate fp16 data: small values to avoid overflow
    let n = size_bytes / 2;
    let mut data = Vec::with_capacity(size_bytes);
    for i in 0..n {
        // Scale: 0.01 * (i % 100 - 50) — range [-0.5, 0.49]
        let val: f32 = 0.01 * ((i % 100) as f32 - 50.0);
        let fp16 = f32_to_fp16(val);
        data.push((fp16 & 0xFF) as u8);
        data.push((fp16 >> 8) as u8);
    }
    data
}

fn make_fp32_data(size_bytes: usize) -> Vec<u8> {
    let n = size_bytes / 4;
    let mut data = Vec::with_capacity(size_bytes);
    for i in 0..n {
        let val: f32 = 0.01 * ((i % 100) as f32 - 50.0);
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

fn f32_to_fp16(f: f32) -> u16 {
    // Use half crate for proper conversion
    use half::f16;
    f16::from_f32(f).to_bits()
}

fn build_causal_mask_fp16(seq: usize) -> Vec<u8> {
    // Causal mask: -65504.0 for masked, 0.0 for unmasked
    // Stored as fp16 blob with 64-byte header
    let n = seq * seq;
    let mut fp16_data = Vec::with_capacity(n);
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                // Masked position: max negative fp16
                fp16_data.push(0x00); // -65504 = 0xFC01 in fp16
                fp16_data.push(0x01);
            } else {
                // Unmasked: 0.0
                fp16_data.push(0x00);
                fp16_data.push(0x00);
            }
        }
    }

    // Build blob with 64-byte header (matching maderix build_blob)
    let data_size = n * 2;
    let total = 128 + data_size; // maderix uses 128-byte header
    let mut blob = vec![0u8; total];
    // Header pattern from maderix: b[0]=1, b[4]=2, b[64..68]=0xDEADBEEF, b[68]=1
    blob[0] = 1;
    blob[4] = 2;
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 1;
    // Data size and offset
    blob[72..76].copy_from_slice(&(data_size as u32).to_le_bytes());
    blob[80..84].copy_from_slice(&128u32.to_le_bytes());
    // Copy fp16 data
    blob[128..128 + data_size].copy_from_slice(&fp16_data);
    blob
}

// ═══════════════════════════════════════════════════════════════════════════
// MIL GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

const MIL_HDR: &str =
    "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, \
{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, \
{\"coremltools-version\", \"9.0\"}})]\n{\n";

/// Channel-packed matmul (our approach): [1, D+D*D, 1, S]
/// Weights in channel dim, activations in first D channels.
fn channel_packed_matmul_mil(dim: usize, seq: usize) -> String {
    let total_ch = dim + dim * dim;
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        total_ch, seq
    ));

    // Slice activations
    m.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=x,begin=b0,size=sa)[name=string(\"act\")];\n", dim, seq));

    // Slice weights: first spatial position
    m.push_str(&format!("        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,{},0,0])];\n", dim));
    m.push_str(&format!("        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim * dim, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> wf = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wf\")];\n", dim * dim, seq));

    // Larger const first (const order matters!)
    m.push_str(&format!("        tensor<int32, [4]> wrs = const()[name=string(\"wrs\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, dim));
    m.push_str(&format!("        tensor<int32, [4]> sw1 = const()[name=string(\"sw1\"), val=tensor<int32, [4]>([1,{},1,1])];\n", dim * dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,1]> wf1 = slice_by_size(x=wf,begin=b0,size=sw1)[name=string(\"wf1\")];\n", dim * dim));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W = reshape(shape=wrs,x=wf1)[name=string(\"W\")];\n",
        dim, dim
    ));

    // Reshape + transpose activations
    m.push_str(&format!("        tensor<int32, [4]> rs = const()[name=string(\"rs\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xr = reshape(shape=rs,x=act)[name=string(\"xr\")];\n",
        dim, seq
    ));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xt = transpose(perm=pm,x=xr)[name=string(\"xt\")];\n",
        seq, dim
    ));

    // Matmul
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> mm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=W)[name=string(\"mm\")];\n", seq, dim));

    // Transpose back + reshape
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> mt = transpose(perm=pm,x=mm)[name=string(\"mt\")];\n",
        dim, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> y = reshape(shape=os,x=mt)[name=string(\"y\")];\n",
        dim, seq
    ));

    m.push_str("    } -> (y);\n}\n");
    m
}

/// Spatial-packed matmul (maderix approach): [1, IC, 1, SEQ+OC]
/// Each channel row: [activation_seq | weight_oc]
fn spatial_packed_matmul_mil(ic: usize, oc: usize, seq: usize) -> String {
    let sp = seq + oc;
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        ic, sp
    ));

    // Slice activations: [1, IC, 1, SEQ]
    m.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", ic, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=x,begin=b0,size=sa)[name=string(\"act\")];\n", ic, seq));

    // Slice weights: [1, IC, 1, OC]
    m.push_str(&format!("        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq));
    m.push_str(&format!("        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", ic, oc));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc));

    // Reshape activations: [1,IC,1,SEQ] → [1,1,IC,SEQ] → [1,1,SEQ,IC]
    m.push_str(&format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", ic, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n",
        ic, seq
    ));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
        seq, ic
    ));

    // Reshape weights: [1,IC,1,OC] → [1,1,IC,OC]
    m.push_str(&format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", ic, oc));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n",
        ic, oc
    ));

    // Matmul: [1,1,SEQ,IC] @ [1,1,IC,OC] → [1,1,SEQ,OC]
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc));

    // Reshape: [1,1,SEQ,OC] → [1,OC,1,SEQ]
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n",
        oc, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", oc, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n",
        oc, seq
    ));

    m.push_str("    } -> (y);\n}\n");
    m
}

/// fp32 I/O matmul (with cast inside MIL): [1, IC, 1, SEQ+OC] fp32
fn fp32_matmul_mil(ic: usize, oc: usize, seq: usize) -> String {
    let sp = seq + oc;
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        ic, sp
    ));
    m.push_str("        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n",
        ic, sp
    ));

    // Slice activations
    m.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", ic, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=xh,begin=b0,size=sa)[name=string(\"act\")];\n", ic, seq));

    // Slice weights
    m.push_str(&format!("        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq));
    m.push_str(&format!("        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", ic, oc));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc));

    // Reshape
    m.push_str(&format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", ic, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n",
        ic, seq
    ));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
        seq, ic
    ));
    m.push_str(&format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", ic, oc));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n",
        ic, oc
    ));

    // Matmul
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n",
        oc, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", oc, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n",
        oc, seq
    ));

    // Cast back to fp32
    m.push_str("        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n");
    m.push_str(&format!(
        "        tensor<fp32, [1,{},1,{}]> y = cast(dtype=to32,x=yr)[name=string(\"cout\")];\n",
        oc, seq
    ));

    m.push_str("    } -> (y);\n}\n");
    m
}

/// fp16 direct I/O matmul: [1, IC, 1, SEQ+OC] fp16 (no cast)
fn fp16_matmul_mil(ic: usize, oc: usize, seq: usize) -> String {
    spatial_packed_matmul_mil(ic, oc, seq)
}

/// Fused QKV with spatial packing (maderix-style)
/// Input: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM]
/// Output: [1, Q_DIM, 1, SEQ]
fn fused_qkv_spatial_mil(
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    heads: usize,
    hd: usize,
    seq: usize,
    sp_in: usize,
) -> String {
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, sp_in
    ));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");

    // Slice xnorm [1,DIM,1,SEQ]
    m.push_str("        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> xn = slice_by_size(x=x,begin=bx,size=sx)[name=string(\"xn\")];\n", dim, seq));

    // Slice Wq [1,DIM,1,Q_DIM]
    m.push_str(&format!("        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq));
    m.push_str(&format!("        tensor<int32, [4]> swq = const()[name=string(\"swq\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, q_dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> Wq = slice_by_size(x=x,begin=bq,size=swq)[name=string(\"Wq\")];\n", dim, q_dim));

    // Slice Wk [1,DIM,1,KV_DIM]
    m.push_str(&format!("        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq + q_dim));
    m.push_str(&format!("        tensor<int32, [4]> swk = const()[name=string(\"swk\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, kv_dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> Wk = slice_by_size(x=x,begin=bk,size=swk)[name=string(\"Wk\")];\n", dim, kv_dim));

    // Slice Wv [1,DIM,1,KV_DIM]
    m.push_str(&format!("        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq + q_dim + kv_dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> Wv = slice_by_size(x=x,begin=bv,size=swk)[name=string(\"Wv\")];\n", dim, kv_dim));

    // Reshape xnorm: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → [1,1,SEQ,DIM]
    m.push_str(&format!("        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xn2 = reshape(shape=r2,x=xn)[name=string(\"xn2\")];\n",
        dim, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n",
        seq, dim
    ));

    // Reshape weights
    m.push_str(&format!("        tensor<int32, [4]> rwq = const()[name=string(\"rwq\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, q_dim));
    m.push_str(&format!("        tensor<int32, [4]> rwk = const()[name=string(\"rwk\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, kv_dim));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> Wq2 = reshape(shape=rwq,x=Wq)[name=string(\"Wq2\")];\n",
        dim, q_dim
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> Wk2 = reshape(shape=rwk,x=Wk)[name=string(\"Wk2\")];\n",
        dim, kv_dim
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> Wv2 = reshape(shape=rwk,x=Wv)[name=string(\"Wv2\")];\n",
        dim, kv_dim
    ));

    // QKV matmul
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", seq, q_dim));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", seq, kv_dim));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", seq, kv_dim));

    // Transpose back: [1,1,SEQ,X] → [1,1,X,SEQ] → [1,X,1,SEQ]
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];\n",
        q_dim, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", q_dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> qf = reshape(shape=qsh,x=qt)[name=string(\"qf\")];\n",
        q_dim, seq
    ));

    m.push_str("    } -> (qf);\n}\n");
    m
}

/// Mega SDPA kernel: QKV projection + attention scoring + attn@V
/// Input: [1, DIM, 1, SEQ + Q_DIM + Q_DIM + Q_DIM]  (MHA: Q_DIM=DIM)
/// Uses blobfile for causal mask
/// Output: [1, Q_DIM, 1, SEQ]
fn mega_sdpa_mil(
    dim: usize,
    q_dim: usize,
    heads: usize,
    hd: usize,
    seq: usize,
    sp_in: usize,
) -> String {
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, sp_in
    ));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    m.push_str("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n");

    // Slice xnorm
    m.push_str("        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> xn = slice_by_size(x=x,begin=bx,size=sx)[name=string(\"xn\")];\n", dim, seq));

    // Slice Wq
    m.push_str(&format!("        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq));
    m.push_str(&format!("        tensor<int32, [4]> swq = const()[name=string(\"swq\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, q_dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> Wq = slice_by_size(x=x,begin=bq,size=swq)[name=string(\"Wq\")];\n", dim, q_dim));

    // Slice Wk
    m.push_str(&format!("        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq + q_dim));
    m.push_str(&format!("        tensor<int32, [4]> swk = const()[name=string(\"swk\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, q_dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> Wk = slice_by_size(x=x,begin=bk,size=swk)[name=string(\"Wk\")];\n", dim, q_dim));

    // Slice Wv
    m.push_str(&format!("        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq + 2 * q_dim));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> Wv = slice_by_size(x=x,begin=bv,size=swk)[name=string(\"Wv\")];\n", dim, q_dim));

    // Reshape xnorm for matmul
    m.push_str(&format!("        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xn2 = reshape(shape=r2,x=xn)[name=string(\"xn2\")];\n",
        dim, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n",
        seq, dim
    ));

    // Reshape weights
    m.push_str(&format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, q_dim));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> Wq2 = reshape(shape=rw,x=Wq)[name=string(\"Wq2\")];\n",
        dim, q_dim
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> Wk2 = reshape(shape=rw,x=Wk)[name=string(\"Wk2\")];\n",
        dim, q_dim
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> Wv2 = reshape(shape=rw,x=Wv)[name=string(\"Wv2\")];\n",
        dim, q_dim
    ));

    // QKV matmul
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", seq, q_dim));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", seq, q_dim));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", seq, q_dim));

    // Transpose back and reshape to [1,X,1,SEQ]
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];\n",
        q_dim, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];\n",
        q_dim, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];\n",
        q_dim, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", q_dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> qf = reshape(shape=qsh,x=qt)[name=string(\"qf\")];\n",
        q_dim, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> kf = reshape(shape=qsh,x=kt)[name=string(\"kf\")];\n",
        q_dim, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> vf = reshape(shape=qsh,x=vt)[name=string(\"vf\")];\n",
        q_dim, seq
    ));

    // Reshape to heads: [1,Q_DIM,1,SEQ] → [1,HEADS,HD,SEQ] → [1,HEADS,SEQ,HD]
    m.push_str(&format!("        tensor<int32, [4]> hsh = const()[name=string(\"hsh\"), val=tensor<int32, [4]>([1,{}, {},{}])];\n", heads, hd, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> q4 = reshape(shape=hsh,x=qf)[name=string(\"rq\")];\n",
        heads, hd, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n",
        heads, seq, hd
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> k4 = reshape(shape=hsh,x=kf)[name=string(\"rk\")];\n",
        heads, hd, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n",
        heads, seq, hd
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> v4 = reshape(shape=hsh,x=vf)[name=string(\"rv\")];\n",
        heads, hd, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n",
        heads, seq, hd
    ));

    // Q@K^T → [1,HEADS,SEQ,SEQ]
    m.push_str(&format!("        tensor<fp16, [1,{},{},{}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", heads, seq, seq));

    // Scale by 1/sqrt(HD)
    let scale = 1.0 / (hd as f64).sqrt();
    m.push_str(&format!(
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({:.6})];\n",
        scale
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n",
        heads, seq, seq
    ));

    // Causal mask (blobfile)
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{},{}]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(128)))];\n", seq, seq, seq, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n",
        heads, seq, seq
    ));

    // Softmax
    m.push_str("        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n");
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n",
        heads, seq, seq
    ));

    // attn@V → [1,HEADS,SEQ,HD]
    m.push_str(&format!("        tensor<fp16, [1,{},{},{}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];\n", heads, seq, hd));

    // Reshape back: [1,HEADS,SEQ,HD] → [1,HEADS,HD,SEQ] → [1,Q_DIM,1,SEQ]
    m.push_str(&format!(
        "        tensor<fp16, [1,{},{},{}]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n",
        heads, hd, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> af = reshape(shape=qsh,x=at)[name=string(\"ra\")];\n",
        q_dim, seq
    ));

    m.push_str("    } -> (af);\n}\n");
    m
}

/// Fused FFN (SwiGLU): x2norm @ W1 → SiLU → ×(x2norm@W3) → @W2 → +x2 → residual
/// Input: [1, DIM, 1, SEQ + SEQ + HIDDEN + HIDDEN + HIDDEN]
/// Output: [1, DIM, 1, SEQ]
fn fused_ffn_mil(dim: usize, hidden: usize, seq: usize, sp_in: usize) -> String {
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        dim, sp_in
    ));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");

    // Slice x2norm [1,DIM,1,SEQ]
    m.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> s_ds = const()[name=string(\"s_ds\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> x2norm = slice_by_size(x=x,begin=b0,size=s_ds)[name=string(\"x2norm\")];\n", dim, seq));

    // Slice x2 [1,DIM,1,SEQ] (for residual)
    m.push_str(&format!("        tensor<int32, [4]> b_x2 = const()[name=string(\"b_x2\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> x2 = slice_by_size(x=x,begin=b_x2,size=s_ds)[name=string(\"x2\")];\n", dim, seq));

    // Slice W1 [1,DIM,1,HIDDEN]
    m.push_str(&format!("        tensor<int32, [4]> b_w1 = const()[name=string(\"b_w1\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", 2 * seq));
    m.push_str(&format!("        tensor<int32, [4]> s_wh = const()[name=string(\"s_wh\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, hidden));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> W1 = slice_by_size(x=x,begin=b_w1,size=s_wh)[name=string(\"W1\")];\n", dim, hidden));

    // Slice W3 [1,DIM,1,HIDDEN]
    m.push_str(&format!("        tensor<int32, [4]> b_w3 = const()[name=string(\"b_w3\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", 2 * seq + hidden));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> W3 = slice_by_size(x=x,begin=b_w3,size=s_wh)[name=string(\"W3\")];\n", dim, hidden));

    // Slice W2 [1,DIM,1,HIDDEN]
    m.push_str(&format!("        tensor<int32, [4]> b_w2 = const()[name=string(\"b_w2\"), val=tensor<int32, [4]>([0,0,0,{}])];\n", 2 * seq + 2 * hidden));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> W2r = slice_by_size(x=x,begin=b_w2,size=s_wh)[name=string(\"W2r\")];\n", dim, hidden));

    // Reshape x2norm for matmul: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → [1,1,SEQ,DIM]
    m.push_str(&format!("        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, seq));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> xn2 = reshape(shape=rd,x=x2norm)[name=string(\"xn2\")];\n", dim, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n",
        seq, dim
    ));

    // Reshape weights: [1,DIM,1,HIDDEN] → [1,1,DIM,HIDDEN]
    m.push_str(&format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", dim, hidden));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W12 = reshape(shape=rw,x=W1)[name=string(\"W12\")];\n",
        dim, hidden
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W32 = reshape(shape=rw,x=W3)[name=string(\"W32\")];\n",
        dim, hidden
    ));

    // h1 = x2norm @ W1, h3 = x2norm @ W3
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];\n", seq, hidden));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];\n", seq, hidden));

    // Reshape back: [1,1,SEQ,HIDDEN] → [1,1,HIDDEN,SEQ] → [1,HIDDEN,1,SEQ]
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];\n",
        hidden, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];\n",
        hidden, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", hidden, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];\n",
        hidden, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];\n",
        hidden, seq
    ));

    // SiLU + gate: silu(h1) * h3
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n",
        hidden, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n",
        hidden, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n",
        hidden, seq
    ));

    // gate @ W2^T: [1,HIDDEN,1,SEQ] → [1,1,HIDDEN,SEQ] → [1,1,SEQ,HIDDEN] @ [1,1,HIDDEN,DIM]
    m.push_str(&format!("        tensor<int32, [4]> rg = const()[name=string(\"rg\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", hidden, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> g2 = reshape(shape=rg,x=gate)[name=string(\"g2\")];\n",
        hidden, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> gt = transpose(perm=pm,x=g2)[name=string(\"gtt\")];\n",
        seq, hidden
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W22 = reshape(shape=rw,x=W2r)[name=string(\"W22\")];\n",
        dim, hidden
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> W2t = transpose(perm=pm,x=W22)[name=string(\"W2t\")];\n",
        hidden, dim
    ));
    m.push_str(&format!("        tensor<fp16, [1,1,{},{}]> fm = matmul(transpose_x=bF,transpose_y=bF,x=gt,y=W2t)[name=string(\"fm\")];\n", seq, dim));

    // Reshape ffn_out: [1,1,SEQ,DIM] → [1,1,DIM,SEQ] → [1,DIM,1,SEQ]
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];\n",
        dim, seq
    ));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> ffn_out = reshape(shape=rd2,x=ft)[name=string(\"ffn_out\")];\n", dim, seq));

    // Residual: x_next = x2 + ffn_out
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> x_next = add(x=x2,y=ffn_out)[name=string(\"x_next\")];\n", dim, seq));

    m.push_str("    } -> (x_next);\n}\n");

    // Fix: need rd2 const before using it
    let rd2_line = format!("        tensor<int32, [4]> rd2 = const()[name=string(\"rd2\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", dim, seq);
    let ffn_out_line = format!("        tensor<fp16, [1,{},1,{}]> ffn_out = reshape(shape=rd2,x=ft)[name=string(\"ffn_out\")];\n", dim, seq);
    // Insert rd2 before the reshape that uses it
    m = m.replacen(&ffn_out_line, &format!("{}\n{}", rd2_line, ffn_out_line), 1);

    m
}

/// Fusion scaling: N matmuls in one program (spatial-packed)
/// Input: [1, IC, 1, SEQ + N*OC]
/// Output: [1, OC, 1, SEQ] (first matmul output)
fn fusion_scaling_mil(ic: usize, seq: usize, n: u32) -> String {
    let oc = ic; // square for simplicity
    let sp_in = seq + n as usize * oc;
    let mut m = String::new();
    m.push_str(MIL_HDR);
    m.push_str(&format!(
        "    func main<ios18>(tensor<fp16, [1, {}, 1, {}]> x) {{\n",
        ic, sp_in
    ));

    // Slice activations
    m.push_str("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    m.push_str(&format!("        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", ic, seq));
    m.push_str(&format!("        tensor<fp16, [1,{},1,{}]> act = slice_by_size(x=x,begin=b0,size=sa)[name=string(\"act\")];\n", ic, seq));

    // Larger const first (ANE const order)
    m.push_str(&format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", ic, oc));
    m.push_str("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    m.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");

    // Reshape activations
    m.push_str(&format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{},{}])];\n", ic, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n",
        ic, seq
    ));
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
        seq, ic
    ));

    // N matmuls
    for i in 0..n {
        let w_off = seq + i as usize * oc;
        m.push_str("        tensor<int32, [4]> bw");
        m.push_str(&i.to_string());
        m.push_str(" = const()[name=string(\"bw");
        m.push_str(&i.to_string());
        m.push_str("\"), val=tensor<int32, [4]>([0,0,0,");
        m.push_str(&w_off.to_string());
        m.push_str(")])];\n");

        m.push_str("        tensor<int32, [4]> sw");
        m.push_str(&i.to_string());
        m.push_str(" = const()[name=string(\"sw");
        m.push_str(&i.to_string());
        m.push_str("\"), val=tensor<int32, [4]>([1,");
        m.push_str(&ic.to_string());
        m.push_str(",1,");
        m.push_str(&oc.to_string());
        m.push_str(")])];\n");

        m.push_str("        tensor<fp16, [1,");
        m.push_str(&ic.to_string());
        m.push_str(",1,");
        m.push_str(&oc.to_string());
        m.push_str("]> wt");
        m.push_str(&i.to_string());
        m.push_str(" = slice_by_size(x=x,begin=bw");
        m.push_str(&i.to_string());
        m.push_str(",size=sw");
        m.push_str(&i.to_string());
        m.push_str(")[name=string(\"wt");
        m.push_str(&i.to_string());
        m.push_str("\")];\n");

        m.push_str("        tensor<fp16, [1,1,");
        m.push_str(&ic.to_string());
        m.push_str(",");
        m.push_str(&oc.to_string());
        m.push_str("]> W");
        m.push_str(&i.to_string());
        m.push_str(" = reshape(shape=rw,x=wt");
        m.push_str(&i.to_string());
        m.push_str(")[name=string(\"W");
        m.push_str(&i.to_string());
        m.push_str("\")];\n");

        m.push_str("        tensor<fp16, [1,1,");
        m.push_str(&seq.to_string());
        m.push_str(",");
        m.push_str(&oc.to_string());
        m.push_str("]> m");
        m.push_str(&i.to_string());
        m.push_str(" = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W");
        m.push_str(&i.to_string());
        m.push_str(")[name=string(\"m");
        m.push_str(&i.to_string());
        m.push_str("\")];\n");
    }

    // Output first matmul result
    m.push_str(&format!(
        "        tensor<fp16, [1,1,{},{}]> yt = transpose(perm=pm,x=m0)[name=string(\"yt\")];\n",
        oc, seq
    ));
    m.push_str(&format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{},1,{}])];\n", oc, seq));
    m.push_str(&format!(
        "        tensor<fp16, [1,{},1,{}]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n",
        oc, seq
    ));

    m.push_str("    } -> (y);\n}\n");
    m
}
