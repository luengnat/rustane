//! Phase 10 Plan 1: CPU Attention Optimization
//!
//! Profiles CPU attention and compares optimized implementations:
//! 1. Baseline: current train_transformer.rs attention (BLAS matmul + loop softmax)
//! 2. Optimized: BLAS matmul + SIMD-friendly softmax + fused operations
//!
//! Usage: cargo run --release --example bench_attention

use std::time::Instant;

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
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn vDSP_vsmsa(n: usize, a: f32, b: *const f32, c: f32, d: *mut f32);
    fn vDSP_vsadd(n: usize, b: *const f32, c: f32, d: *mut f32);
}
const ROW: i32 = 101;
const NT: i32 = 111;
const TR: i32 = 112;

fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            NT,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}
fn mm_at(a: &[f32], k: usize, m: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            TR,
            NT,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}
fn mm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            TR,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
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

fn cpu_attention_baseline(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    sp: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scores = mm_at(q, d, sp, k, sp);
    let scale = 1.0 / (d as f32).sqrt();
    let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();
    let mut attn = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let mut mx = f32::NEG_INFINITY;
        for j in 0..sp {
            mx = mx.max(scaled[i * sp + j]);
        }
        let mut sm = 0.0_f32;
        for j in 0..sp {
            let e = (scaled[i * sp + j] - mx).exp();
            attn[i * sp + j] = e;
            sm += e;
        }
        for j in 0..sp {
            attn[i * sp + j] /= sm;
        }
    }
    let av = mm_abt(&attn, sp, sp, v, d);
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            out[h * sp + i] = av[i * d + h];
        }
    }
    (out, attn)
}

fn cpu_attention_optimized(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    sp: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (d as f32).sqrt();

    let scores = mm_at(q, d, sp, k, sp);

    let mut attn = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let row = &scores[i * sp..(i + 1) * sp];
        let mut mx = f32::NEG_INFINITY;
        for &s in row {
            mx = mx.max(s);
        }
        let mx_scaled = mx * scale;

        let mut sm = 0.0_f32;
        for j in 0..sp {
            let e = (row[j] * scale - mx_scaled).exp();
            attn[i * sp + j] = e;
            sm += e;
        }
        let inv_sm = 1.0 / sm;
        for j in 0..sp {
            attn[i * sp + j] *= inv_sm;
        }
    }

    let av = mm_abt(&attn, sp, sp, v, d);
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            out[h * sp + i] = av[i * d + h];
        }
    }
    (out, attn)
}

fn cpu_attention_flash(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    sp: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (d as f32).sqrt();

    let scores = mm_at(q, d, sp, k, sp);

    let mut attn = vec![0.0f32; sp * sp];
    let mut row_max = vec![f32::NEG_INFINITY; sp];
    let mut row_sum = vec![0.0f32; sp];

    for i in 0..sp {
        let base = i * sp;
        let row = &scores[base..base + sp];
        let mut mx = f32::NEG_INFINITY;
        for &s in row {
            mx = mx.max(s);
        }
        row_max[i] = mx;
        let mx_s = mx * scale;

        let mut sm = 0.0_f32;
        for j in 0..sp {
            let e = (row[j] * scale - mx_s).exp();
            attn[base + j] = e;
            sm += e;
        }
        row_sum[i] = sm;
        let inv_sm = 1.0 / sm;
        for j in 0..sp {
            attn[base + j] *= inv_sm;
        }
    }

    let av = mm_abt(&attn, sp, sp, v, d);
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            out[h * sp + i] = av[i * d + h];
        }
    }
    (out, attn)
}

fn bench_attention(label: &str, d: usize, sp: usize, heads: usize, iters: usize) {
    let head_dim = d / heads;
    let q = rand_m(d, sp, 0.02, 42);
    let k = rand_m(d, sp, 0.02, 43);
    let v = rand_m(d, sp, 0.02, 44);

    let mut times = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for h in 0..heads {
            let qh = &q[h * head_dim * sp..(h + 1) * head_dim * sp];
            let kh = &k[h * head_dim * sp..(h + 1) * head_dim * sp];
            let vh = &v[h * head_dim * sp..(h + 1) * head_dim * sp];
            let _ = cpu_attention_baseline(qh, kh, vh, head_dim, sp);
        }
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let baseline_us: f64 = times.iter().sum::<f64>() / iters as f64;

    let mut times = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for h in 0..heads {
            let qh = &q[h * head_dim * sp..(h + 1) * head_dim * sp];
            let kh = &k[h * head_dim * sp..(h + 1) * head_dim * sp];
            let vh = &v[h * head_dim * sp..(h + 1) * head_dim * sp];
            let _ = cpu_attention_optimized(qh, kh, vh, head_dim, sp);
        }
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let optimized_us: f64 = times.iter().sum::<f64>() / iters as f64;

    let mut times = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for h in 0..heads {
            let qh = &q[h * head_dim * sp..(h + 1) * head_dim * sp];
            let kh = &k[h * head_dim * sp..(h + 1) * head_dim * sp];
            let vh = &v[h * head_dim * sp..(h + 1) * head_dim * sp];
            let _ = cpu_attention_flash(qh, kh, vh, head_dim, sp);
        }
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let flash_us: f64 = times.iter().sum::<f64>() / iters as f64;

    let speedup = baseline_us / optimized_us;
    let flash_speedup = baseline_us / flash_us;
    println!("  {:25} D={:>4} SP={:>3} heads={:>2} hd={:>4} baseline={:>8.0}us opt={:>8.0}us flash={:>8.0}us speedup={:.2}x flash={:.2}x",
        label, d, sp, heads, head_dim, baseline_us, optimized_us, flash_us, speedup, flash_speedup);
}

fn main() {
    println!("============================================================");
    println!("  Phase 10: CPU Attention Optimization Benchmark");
    println!("============================================================\n");

    let iters = 100;
    println!("  Single-head attention (current approach):");
    for &d in &[256, 512, 768, 1024, 2048] {
        for &sp in &[256, 512] {
            bench_attention("single-head", d, sp, 1, iters);
        }
    }

    println!("\n  Multi-head attention (heads=8, 12, 16):");
    for &(d, heads) in &[(768, 12), (1024, 16), (2048, 16)] {
        for &sp in &[256, 512] {
            bench_attention("multi-head", d, sp, heads, iters);
        }
    }

    println!("\n  Impact analysis (12L transformer, D=768, heads=12):");
    let d = 768;
    let sp = 256;
    let heads = 12;
    let head_dim = 64;
    let layers = 12;
    let iters = 50;

    let q = rand_m(d, sp, 0.02, 42);
    let k = rand_m(d, sp, 0.02, 43);
    let v = rand_m(d, sp, 0.02, 44);

    let mut baseline_total = 0.0_f64;
    let mut opt_total = 0.0_f64;
    for _ in 0..iters {
        let t = Instant::now();
        for _ in 0..layers {
            for h in 0..heads {
                let qh = &q[h * head_dim * sp..(h + 1) * head_dim * sp];
                let kh = &k[h * head_dim * sp..(h + 1) * head_dim * sp];
                let vh = &v[h * head_dim * sp..(h + 1) * head_dim * sp];
                let _ = cpu_attention_baseline(qh, kh, vh, head_dim, sp);
            }
        }
        baseline_total += t.elapsed().as_secs_f64() * 1e6;

        let t = Instant::now();
        for _ in 0..layers {
            for h in 0..heads {
                let qh = &q[h * head_dim * sp..(h + 1) * head_dim * sp];
                let kh = &k[h * head_dim * sp..(h + 1) * head_dim * sp];
                let vh = &v[h * head_dim * sp..(h + 1) * head_dim * sp];
                let _ = cpu_attention_optimized(qh, kh, vh, head_dim, sp);
            }
        }
        opt_total += t.elapsed().as_secs_f64() * 1e6;
    }
    let baseline_avg = baseline_total / iters as f64;
    let opt_avg = opt_total / iters as f64;

    println!(
        "    Baseline:  {:.0}us total ({:.1}us/layer, {:.1}us/head)",
        baseline_avg,
        baseline_avg / layers as f64,
        baseline_avg / (layers * heads) as f64
    );
    println!(
        "    Optimized: {:.0}us total ({:.1}us/layer, {:.1}us/head)",
        opt_avg,
        opt_avg / layers as f64,
        opt_avg / (layers * heads) as f64
    );
    println!("    Speedup:   {:.2}x", baseline_avg / opt_avg);
    println!(
        "    Saved:     {:.0}us per step ({:.1}% of typical 100ms step)",
        baseline_avg - opt_avg,
        (baseline_avg - opt_avg) / 100_000.0 * 100.0
    );

    println!("\n============================================================");
    println!("  Phase 10 Complete");
    println!("============================================================");
}
