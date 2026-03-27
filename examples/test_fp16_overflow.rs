//! Phase 9 Plan 1: Characterize fp16 Overflow Boundaries
//!
//! Tests where fp16 overflow occurs during training:
//! 1. Which operations produce values > fp16 max (65504)?
//! 2. At what D does overflow start?
//! 3. Where in the forward/backward pass does it happen?
//!
//! Usage: cargo run --release --example test_fp16_overflow

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

fn stats(data: &[f32], label: &str) {
    if data.is_empty() {
        return;
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0_f64;
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut over_fp16 = 0usize;
    let fp16_max = 65504.0_f32;
    for &v in data {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        if v.is_infinite() {
            inf += 1;
            continue;
        }
        if v.abs() > fp16_max {
            over_fp16 += 1;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }
    let avg = sum / data.len() as f64;
    println!(
        "  {:30} min={:+.4} max={:+.4} avg={:+.6} nan={} inf={} over_fp16={}",
        label, min, max, avg, nan, inf, over_fp16
    );
}

fn analyze_forward(d: usize, sp: usize) {
    let inter = d * 4;
    println!("\n=== Forward Pass D={} SP={} inter={} ===", d, sp, inter);

    let wq = rand_m(d, d, 0.02, 42);
    let wk = rand_m(d, d, 0.02, 43);
    let wv = rand_m(d, d, 0.02, 44);
    let wo = rand_m(d, d, 0.02, 45);
    let wg = rand_m(inter, d, 0.02, 46);
    let wu = rand_m(inter, d, 0.02, 47);
    let wd = rand_m(d, inter, 0.02, 48);

    let x = rand_m(d, sp, 0.1, 9999);

    let q = mm(&wq, d, d, &x, sp);
    let k = mm(&wk, d, d, &x, sp);
    let v = mm(&wv, d, d, &x, sp);

    let scores = mm_at(&q, d, sp, &k, sp);
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

    let av = mm_abt(&attn, sp, sp, &v, d);
    let mut attn_out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            attn_out[h * sp + i] = av[i * d + h];
        }
    }

    let mut y = vec![0.0f32; d * sp];
    let out = mm(&wo, d, d, &attn_out, sp);
    for i in 0..d * sp {
        y[i] = out[i] + x[i];
    }

    let gate = mm(&wg, inter, d, &y, sp);
    let up = mm(&wu, inter, d, &y, sp);
    let mut fused = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        let s = 1.0 / (1.0 + (-gate[i]).exp());
        fused[i] = gate[i] * s * up[i];
    }
    let down = mm(&wd, d, inter, &fused, sp);
    let mut out2 = vec![0.0f32; d * sp];
    for i in 0..out2.len() {
        out2[i] = down[i] + y[i];
    }

    stats(&x, "input x");
    stats(&q, "Q = Wq @ x");
    stats(&k, "K = Wk @ x");
    stats(&v, "V = Wv @ x");
    stats(&scores, "QK^T scores");
    stats(&scaled, "scaled scores");
    stats(&attn, "attention weights");
    stats(&attn_out, "attn_out = attn @ V");
    stats(&y, "y = Wo @ attn_out + x");
    stats(&gate, "gate = Wg @ y");
    stats(&up, "up = Wu @ y");
    stats(&fused, "fused = SiLU(gate) * up");
    stats(&out2, "output = Wd @ fused + y");
}

fn analyze_backward(d: usize, sp: usize) {
    let inter = d * 4;
    println!("\n=== Backward Pass D={} SP={} inter={} ===", d, sp, inter);

    let wq = rand_m(d, d, 0.02, 42);
    let wk = rand_m(d, d, 0.02, 43);
    let wv = rand_m(d, d, 0.02, 44);
    let wo = rand_m(d, d, 0.02, 45);
    let wg = rand_m(inter, d, 0.02, 46);
    let wu = rand_m(inter, d, 0.02, 47);
    let wd = rand_m(d, inter, 0.02, 48);

    let x = rand_m(d, sp, 0.1, 9999);
    let lr = 0.001_f32;

    let q = mm(&wq, d, d, &x, sp);
    let k = mm(&wk, d, d, &x, sp);
    let v = mm(&wv, d, d, &x, sp);
    let scores = mm_at(&q, d, sp, &k, sp);
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
    let av = mm_abt(&attn, sp, sp, &v, d);
    let mut attn_out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            attn_out[h * sp + i] = av[i * d + h];
        }
    }
    let mut y = vec![0.0f32; d * sp];
    let out_proj = mm(&wo, d, d, &attn_out, sp);
    for i in 0..d * sp {
        y[i] = out_proj[i] + x[i];
    }
    let gate = mm(&wg, inter, d, &y, sp);
    let up = mm(&wu, inter, d, &y, sp);
    let mut fused = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        let s = 1.0 / (1.0 + (-gate[i]).exp());
        fused[i] = gate[i] * s * up[i];
    }
    let down = mm(&wd, d, inter, &fused, sp);
    let mut final_out = vec![0.0f32; d * sp];
    for i in 0..final_out.len() {
        final_out[i] = down[i] + y[i];
    }

    let n = final_out.len() as f32;
    let mut dy = vec![0.0f32; final_out.len()];
    for i in 0..final_out.len() {
        dy[i] = 2.0 * final_out[i] / n;
    }

    let dwd = mm_abt(&dy, d, sp, &fused, inter);
    let dfused = mm_at(&wd, d, inter, &dy, sp);

    let mut dup = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        let s = 1.0 / (1.0 + (-gate[i]).exp());
        dup[i] = dfused[i] * gate[i] * s;
    }
    let dwu = mm_abt(&dup, inter, sp, &y, d);
    let mut dgate = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        let g = gate[i];
        let s = 1.0 / (1.0 + (-g).exp());
        dgate[i] = dfused[i] * up[i] * s * (1.0 + g * (1.0 - s));
    }
    let dwg = mm_abt(&dgate, inter, sp, &y, d);

    let dattn_out = mm_at(&wo, d, d, &dy, sp);
    let dwo = mm_abt(&dy, d, sp, &attn_out, d);
    let dattn = mm_at(&dattn_out, d, sp, &v, sp);
    let dv_final = mm(&dattn_out, d, sp, &attn, sp);

    let mut dscaled = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let mut dot = 0.0_f32;
        for j in 0..sp {
            dot += dattn[i * sp + j] * attn[i * sp + j];
        }
        for j in 0..sp {
            dscaled[i * sp + j] = attn[i * sp + j] * (dattn[i * sp + j] - dot) * scale;
        }
    }
    let dq_final = mm_abt(&k, d, sp, &dscaled, sp);
    let dk_final = mm(&q, d, sp, &dscaled, sp);
    let dwq = mm_abt(&dq_final, d, sp, &x, d);
    let dwk = mm_abt(&dk_final, d, sp, &x, d);
    let dwv = mm_abt(&dv_final, d, sp, &x, d);

    stats(&dy, "dL/dy (loss gradient)");
    stats(&dwd, "dL/dWd");
    stats(&dfused, "dL/dfused");
    stats(&dup, "dL/dup");
    stats(&dwu, "dL/dWu");
    stats(&dgate, "dL/dgate");
    stats(&dwg, "dL/dWg");
    stats(&dattn_out, "dL/dattn_out");
    stats(&dwo, "dL/dWo");
    stats(&dattn, "dL/dattn");
    stats(&dv_final, "dL/dV");
    stats(&dscaled, "dL/dscaled");
    stats(&dq_final, "dL/dQ");
    stats(&dk_final, "dL/dK");
    stats(&dwq, "dL/dWq");
    stats(&dwk, "dL/dWk");
    stats(&dwv, "dL/dWv");

    println!("\n  SGD update (lr={}):", lr);
    for (name, w, dw) in [
        ("Wq", &wq[..], &dwq[..]),
        ("Wk", &wk[..], &dwk[..]),
        ("Wv", &wv[..], &dwv[..]),
        ("Wo", &wo[..], &dwo[..]),
        ("Wg", &wg[..], &dwg[..]),
        ("Wu", &wu[..], &dwu[..]),
        ("Wd", &wd[..], &dwd[..]),
    ] {
        let max_update = dw.iter().map(|d| (d * lr).abs()).fold(0.0_f32, f32::max);
        let max_weight = w.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
        println!(
            "    {:4}: max_weight={:.4} max_update={:.8} ratio={:.2e}",
            name,
            max_weight,
            max_update,
            max_update / max_weight.max(1e-10)
        );
    }
}

fn main() {
    println!("============================================================");
    println!("  Phase 9: fp16 Overflow Characterization");
    println!("============================================================");
    println!("  fp16 range: ±65504");
    println!("  fp16 subnormal: < 6e-5 (precision loss)");

    for &d in &[256, 512, 768, 1024, 2048] {
        analyze_forward(d, 256);
        analyze_backward(d, 256);
    }

    println!("\n============================================================");
    println!("  Phase 9 Characterization Complete");
    println!("============================================================");
}
