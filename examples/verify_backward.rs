//! Verify backward correctness using numerical gradient (finite differences).
//!
//! Uses CPU-only forward (exact, no fp16) to compute MSE loss against a target.
//! Compares analytical gradients from backward_cpu against numerical gradients.
//! Loss and numerical gradients are computed in f64 to avoid catastrophic
//! cancellation when subtracting nearly-equal f32 loss values.
//!
//! Usage: cargo run --release --example verify_backward -- [D] [SP] [EPS]

use std::env;

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

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, sp: usize) -> (Vec<f32>, Vec<f32>) {
    let scores = mm_at(q, d, sp, k, sp); // [SP, SP]
    let scale = 1.0 / (d as f32).sqrt();
    let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();
    // Softmax per row
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
    // av = attn @ V^T → [SP, D], then transpose to [D, SP]
    let av = mm_abt(&attn, sp, sp, v, d);
    let mut out = vec![0.0f32; d * sp];
    for h in 0..d {
        for i in 0..sp {
            out[h * sp + i] = av[i * d + h]; // transpose [SP,D] → [D,SP]
        }
    }
    (out, attn)
}

struct TransformerLayer {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    d: usize,
    inter: usize,
    sp: usize,
    // Cached activations for backward
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    attn_out: Vec<f32>,
    attn_in: Vec<f32>,
    ffn_in: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    // Pre-allocated backward buffers
    fused: Vec<f32>,
    dup: Vec<f32>,
    dgate: Vec<f32>,
    dffn_in: Vec<f32>,
    dscaled: Vec<f32>,
    dx: Vec<f32>,
}

impl TransformerLayer {
    fn new(d: usize, inter: usize, sp: usize, seed: u64) -> Self {
        TransformerLayer {
            wq: rand_m(d, d, 0.1, seed),
            wk: rand_m(d, d, 0.1, seed + 1),
            wv: rand_m(d, d, 0.1, seed + 2),
            wo: rand_m(d, d, 0.1, seed + 3),
            wg: rand_m(inter, d, 0.1, seed + 4),
            wu: rand_m(inter, d, 0.1, seed + 5),
            wd: rand_m(d, inter, 0.1, seed + 6),
            d,
            inter,
            sp,
            q: vec![0.0; d * sp],
            k: vec![0.0; d * sp],
            v: vec![0.0; d * sp],
            attn: vec![0.0; sp * sp],
            attn_out: vec![0.0; d * sp],
            attn_in: vec![0.0; d * sp],
            ffn_in: vec![0.0; d * sp],
            gate: vec![0.0; inter * sp],
            up: vec![0.0; inter * sp],
            fused: vec![0.0; inter * sp],
            dup: vec![0.0; inter * sp],
            dgate: vec![0.0; inter * sp],
            dffn_in: vec![0.0; d * sp],
            dscaled: vec![0.0; sp * sp],
            dx: vec![0.0; d * sp],
        }
    }

    /// CPU forward with caching (caches activations for backward)
    fn forward_cached(&mut self, x: &[f32]) -> Vec<f32> {
        let d = self.d;
        let sp = self.sp;
        self.attn_in.copy_from_slice(x);
        let q = mm(&self.wq, d, d, x, sp);
        let k = mm(&self.wk, d, d, x, sp);
        let v = mm(&self.wv, d, d, x, sp);
        self.q.copy_from_slice(&q);
        self.k.copy_from_slice(&k);
        self.v.copy_from_slice(&v);
        let (attn_out, attn_weights) = cpu_attention(&self.q, &self.k, &self.v, d, sp);
        self.attn_out.copy_from_slice(&attn_out);
        self.attn.copy_from_slice(&attn_weights);
        let out = mm(&self.wo, d, d, &self.attn_out, sp);
        let mut y = vec![0.0f32; d * sp];
        for i in 0..d * sp {
            y[i] = out[i] + x[i];
        }
        let gate = mm(&self.wg, self.inter, d, &y, sp);
        let up = mm(&self.wu, self.inter, d, &y, sp);
        self.gate.copy_from_slice(&gate);
        self.up.copy_from_slice(&up);
        self.ffn_in.copy_from_slice(&y);
        let mut fused = vec![0.0f32; self.inter * sp];
        for i in 0..self.inter * sp {
            let s = 1.0 / (1.0 + (-self.gate[i]).exp());
            fused[i] = self.gate[i] * s * self.up[i];
        }
        let down = mm(&self.wd, d, self.inter, &fused, sp);
        let mut out2 = vec![0.0f32; d * sp];
        for i in 0..d * sp {
            out2[i] = down[i] + y[i];
        }
        out2
    }

    /// CPU-only forward (no caching, for numerical gradient)
    fn forward_pure(&self, x: &[f32]) -> Vec<f32> {
        let d = self.d;
        let sp = self.sp;
        let q = mm(&self.wq, d, d, x, sp);
        let k = mm(&self.wk, d, d, x, sp);
        let v = mm(&self.wv, d, d, x, sp);
        let (attn_out, _) = cpu_attention(&q, &k, &v, d, sp);
        let out = mm(&self.wo, d, d, &attn_out, sp);
        let mut y = vec![0.0f32; d * sp];
        for i in 0..d * sp {
            y[i] = out[i] + x[i];
        }
        let gate = mm(&self.wg, self.inter, d, &y, sp);
        let up = mm(&self.wu, self.inter, d, &y, sp);
        let mut fused = vec![0.0f32; self.inter * sp];
        for i in 0..self.inter * sp {
            let s = 1.0 / (1.0 + (-gate[i]).exp());
            fused[i] = gate[i] * s * up[i];
        }
        let down = mm(&self.wd, d, self.inter, &fused, sp);
        let mut out2 = vec![0.0f32; d * sp];
        for i in 0..d * sp {
            out2[i] = down[i] + y[i];
        }
        out2
    }

    /// Backward — identical to train_transformer.rs backward_cpu
    fn backward_cpu(
        &mut self,
        dy: &[f32],
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let d = self.d;
        let sp = self.sp;
        let inter = self.inter;
        // FFN backward
        for i in 0..inter * sp {
            let s = 1.0 / (1.0 + (-self.gate[i]).exp());
            self.fused[i] = self.gate[i] * s * self.up[i];
        }
        let dwd = mm_abt(dy, d, sp, &self.fused, inter);
        let dfused = mm_at(&self.wd, d, inter, dy, sp);
        for i in 0..inter * sp {
            let s = 1.0 / (1.0 + (-self.gate[i]).exp());
            self.dup[i] = dfused[i] * self.gate[i] * s;
        }
        let dwu = mm_abt(&self.dup, inter, sp, &self.ffn_in, d);
        for i in 0..inter * sp {
            let g = self.gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            self.dgate[i] = dfused[i] * self.up[i] * s * (1.0 + g * (1.0 - s));
        }
        let dwg = mm_abt(&self.dgate, inter, sp, &self.ffn_in, d);
        let dx_ffn_g = mm_at(&self.wg, inter, d, &self.dgate, sp);
        let dx_ffn_u = mm_at(&self.wu, inter, d, &self.dup, sp);
        for i in 0..d * sp {
            self.dffn_in[i] = dx_ffn_g[i] + dx_ffn_u[i] + dy[i];
        }
        // Attention backward
        let dattn_out = mm_at(&self.wo, d, d, &self.dffn_in, sp);
        let dwo = mm_abt(&self.dffn_in, d, sp, &self.attn_out, d);
        let dattn = mm_at(&dattn_out, d, sp, &self.v, sp);
        let dv_final = mm(&dattn_out, d, sp, &self.attn, sp);
        let scale = 1.0 / (d as f32).sqrt();
        for i in 0..sp {
            let mut dot = 0.0_f32;
            for j in 0..sp {
                dot += dattn[i * sp + j] * self.attn[i * sp + j];
            }
            for j in 0..sp {
                self.dscaled[i * sp + j] =
                    self.attn[i * sp + j] * (dattn[i * sp + j] - dot) * scale;
            }
        }
        let dq_final = mm_abt(&self.k, d, sp, &self.dscaled, sp);
        let dk_final = mm(&self.q, d, sp, &self.dscaled, sp);
        let dwq = mm_abt(&dq_final, d, sp, &self.attn_in, d);
        let dwk = mm_abt(&dk_final, d, sp, &self.attn_in, d);
        let dwv = mm_abt(&dv_final, d, sp, &self.attn_in, d);
        let dx_q = mm_at(&self.wq, d, d, &dq_final, sp);
        let dx_k = mm_at(&self.wk, d, d, &dk_final, sp);
        let dx_v = mm_at(&self.wv, d, d, &dv_final, sp);
        for i in 0..d * sp {
            self.dx[i] = dx_q[i] + dx_k[i] + dx_v[i] + self.dffn_in[i];
        }
        (self.dx.clone(), dwq, dwk, dwv, dwo, dwg, dwu, dwd)
    }

    fn clone_with_weights(&self, weight_idx: usize, replacement: &[f32]) -> Self {
        let mut s = Self {
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            wg: self.wg.clone(),
            wu: self.wu.clone(),
            wd: self.wd.clone(),
            d: self.d,
            inter: self.inter,
            sp: self.sp,
            q: vec![0.0; 0],
            k: vec![0.0; 0],
            v: vec![0.0; 0],
            attn: vec![0.0; 0],
            attn_out: vec![0.0; 0],
            attn_in: vec![0.0; 0],
            ffn_in: vec![0.0; 0],
            gate: vec![0.0; 0],
            up: vec![0.0; 0],
            fused: vec![0.0; 0],
            dup: vec![0.0; 0],
            dgate: vec![0.0; 0],
            dffn_in: vec![0.0; 0],
            dscaled: vec![0.0; 0],
            dx: vec![0.0; 0],
        };
        match weight_idx {
            0 => s.wq = replacement.to_vec(),
            1 => s.wk = replacement.to_vec(),
            2 => s.wv = replacement.to_vec(),
            3 => s.wo = replacement.to_vec(),
            4 => s.wg = replacement.to_vec(),
            5 => s.wu = replacement.to_vec(),
            6 => s.wd = replacement.to_vec(),
            _ => {}
        }
        s
    }
}

/// MSE loss computed in f64 to avoid catastrophic cancellation in finite differences.
fn mse_loss_f64(y: &[f32], target: &[f32]) -> f64 {
    let n = y.len() as f64;
    let mut sum = 0.0_f64;
    for i in 0..y.len() {
        let diff = y[i] as f64 - target[i] as f64;
        sum += diff * diff;
    }
    sum / n
}

/// Numerical gradient using centered differences with f64 loss accumulation.
/// Returns the gradient as f32.
fn numerical_grad_weight(
    layer: &TransformerLayer,
    x: &[f32],
    target: &[f32],
    weight_idx: usize,
    elem_idx: usize,
    eps: f32,
) -> f32 {
    let weights = [
        &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.wg, &layer.wu, &layer.wd,
    ];
    let w = weights[weight_idx];

    let mut w_plus = w.to_vec();
    w_plus[elem_idx] += eps;
    let layer_p = layer.clone_with_weights(weight_idx, &w_plus);
    let y_plus = layer_p.forward_pure(x);
    let loss_plus = mse_loss_f64(&y_plus, target);

    let mut w_minus = w.to_vec();
    w_minus[elem_idx] -= eps;
    let layer_m = layer.clone_with_weights(weight_idx, &w_minus);
    let y_minus = layer_m.forward_pure(x);
    let loss_minus = mse_loss_f64(&y_minus, target);

    ((loss_plus - loss_minus) / (2.0 * eps as f64)) as f32
}

fn compare_grad(
    name: &str,
    analytical: &[f32],
    numerical: &[f32],
    step: usize,
    rel_threshold: f32,
    abs_floor: f32,
) {
    let mut max_rel = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut fail_count = 0;
    let mut checked = 0usize;

    // Only compare at indices where numerical was actually computed
    for i in (0..analytical.len()).step_by(step) {
        if i >= numerical.len() {
            break;
        }
        checked += 1;
        let a = analytical[i];
        let n = numerical[i];
        let abs_diff = (a - n).abs();
        // Use a floor for the denominator so tiny gradients don't cause huge rel errors
        let denom = a.abs().max(n.abs()).max(abs_floor);
        let rel = abs_diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        if abs_diff > max_abs {
            max_abs = abs_diff;
        }
        if rel > rel_threshold {
            fail_count += 1;
        }
    }

    let status = if fail_count == 0 {
        "✅ PASS"
    } else {
        "❌ FAIL"
    };
    println!(
        "  {:20} max_rel={:.2e} max_abs={:.2e} failures={}/{} {}",
        name, max_rel, max_abs, fail_count, checked, status
    );
    if fail_count > 0 && fail_count <= 5 {
        let mut shown = 0;
        for i in (0..analytical.len()).step_by(step) {
            if shown >= 3 || i >= numerical.len() {
                break;
            }
            let a = analytical[i];
            let n = numerical[i];
            let abs_diff = (a - n).abs();
            let denom = a.abs().max(n.abs()).max(abs_floor);
            let rel = abs_diff / denom;
            if rel > rel_threshold {
                println!(
                    "    [{}] a={:+.8e} n={:+.8e} diff={:.2e} rel={:.2e}",
                    i, a, n, abs_diff, rel
                );
                shown += 1;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let eps: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e-4);
    let inter = d * 4;

    println!("=== Backward Correctness Verification ===");
    println!("  D={} SP={} inter={} eps={:.1e}\n", d, sp, inter, eps);

    let mut layer = TransformerLayer::new(d, inter, sp, 42);
    let x = rand_m(d, sp, 0.5, 9999);
    let target_w = rand_m(d, d, 0.5, 12345);
    let target = mm(&target_w, d, d, &x, sp);

    // Forward + backward to get analytical gradients
    let y = layer.forward_cached(&x);
    let n = y.len() as f32;
    let mut dy = vec![0.0f32; y.len()];
    for i in 0..y.len() {
        dy[i] = 2.0 * (y[i] - target[i]) / n;
    }
    let (dx_a, dwq_a, dwk_a, dwv_a, dwo_a, dwg_a, dwu_a, dwd_a) = layer.backward_cpu(&dy);

    // Numerical gradients for each weight
    println!("Weight gradients (sampled):");
    let sample_n = 50.min(d * d); // Sample 50 elements from each weight

    for (name, analytical, weight_idx) in [
        ("dWq", &dwq_a[..], 0usize),
        ("dWk", &dwk_a[..], 1usize),
        ("dWv", &dwv_a[..], 2usize),
        ("dWo", &dwo_a[..], 3usize),
        ("dWg", &dwg_a[..], 4usize),
        ("dWu", &dwu_a[..], 5usize),
        ("dWd", &dwd_a[..], 6usize),
    ] {
        let weights = [
            &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.wg, &layer.wu, &layer.wd,
        ];
        let weight = weights[weight_idx];
        let step = if sample_n < weight.len() {
            weight.len() / sample_n
        } else {
            1
        };
        let mut numerical = vec![0.0f32; analytical.len()];
        for i in (0..weight.len()).step_by(step) {
            numerical[i] = numerical_grad_weight(&layer, &x, &target, weight_idx, i, eps);
        }
        compare_grad(name, analytical, &numerical, step, 0.01, 1e-3);
        // Show actual values for failing gradients
        if name == "dWg" || name == "dWu" {
            println!(
                "    [info] {} analytical max={:.2e} num max={:.2e}",
                name,
                analytical.iter().map(|&v| v.abs()).fold(0.0_f32, f32::max),
                numerical.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
            );
        }
    }

    // Numerical gradient for input (dx)
    println!("\nInput gradient (dx):");
    let mut dx_numerical = vec![0.0f32; d * sp];
    let dx_sample = 50.min(d * sp);
    let dx_step = if dx_sample < d * sp {
        d * sp / dx_sample
    } else {
        1
    };
    for i in (0..d * sp).step_by(dx_step) {
        let mut x_plus = x.to_vec();
        x_plus[i] += eps;
        let y_plus = layer.forward_pure(&x_plus);
        let loss_plus = mse_loss_f64(&y_plus, &target);

        let mut x_minus = x.to_vec();
        x_minus[i] -= eps;
        let y_minus = layer.forward_pure(&x_minus);
        let loss_minus = mse_loss_f64(&y_minus, &target);

        dx_numerical[i] = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dx", &dx_a, &dx_numerical, dx_step, 0.01, 1e-3);

    test_linear(); // W @ x → MSE
    test_residual(); // W @ x + x → MSE
    test_attention_only(); // Q,K,V projections → attention → MSE
    test_ffn_only(); // FFN with residual → MSE
    test_gradient_descent(); // Verify gradient step reduces loss
}

/// Verify that taking a gradient step with the analytical backward actually reduces loss.
fn test_gradient_descent() {
    println!("\n=== Gradient Descent Sanity Check ===");
    let d: usize = 64;
    let sp: usize = 16;
    let lr: f32 = 0.01;

    let mut layer = TransformerLayer::new(d, d * 4, sp, 42);
    let x = rand_m(d, sp, 0.5, 9999);
    let target_w = rand_m(d, d, 0.5, 12345);
    let target = mm(&target_w, d, d, &x, sp);

    // Compute initial loss
    let y0 = layer.forward_cached(&x);
    let loss0 = mse_loss_f64(&y0, &target);

    // Compute backward
    let n = y0.len() as f32;
    let mut dy = vec![0.0f32; y0.len()];
    for i in 0..y0.len() {
        dy[i] = 2.0 * (y0[i] - target[i]) / n;
    }
    let (_, dwq, dwk, dwv, dwo, dwg, dwu, dwd) = layer.backward_cpu(&dy);

    // Take gradient step
    let wlen = d * d;
    for i in 0..wlen {
        layer.wq[i] -= lr * dwq[i];
    }
    for i in 0..wlen {
        layer.wk[i] -= lr * dwk[i];
    }
    for i in 0..wlen {
        layer.wv[i] -= lr * dwv[i];
    }
    for i in 0..wlen {
        layer.wo[i] -= lr * dwo[i];
    }
    let ilen = d * 4 * d;
    for i in 0..ilen {
        layer.wg[i] -= lr * dwg[i];
    }
    for i in 0..ilen {
        layer.wu[i] -= lr * dwu[i];
    }
    for i in 0..ilen {
        layer.wd[i] -= lr * dwd[i];
    }

    // Compute loss after step
    let y1 = layer.forward_cached(&x);
    let loss1 = mse_loss_f64(&y1, &target);

    let improved = loss1 < loss0;
    let pct = (loss0 - loss1) / loss0 * 100.0;
    println!(
        "  loss_before={:.8e} loss_after={:.8e} change={:+.2e} ({:+.1}%) {}",
        loss0,
        loss1,
        loss1 - loss0,
        pct,
        if improved { "✅ PASS" } else { "❌ FAIL" }
    );
}

/// Test attention backward in isolation: x → Q=Wq@x, K=Wk@x, V=Wv@x → attn → Wo@attn_out → MSE
fn test_attention_only() {
    let d: usize = 32;
    let sp: usize = 8;
    let eps: f32 = 1e-4;
    let sample_n = 30;

    println!("\n=== Attention-only Verification ===");
    println!("  D={} SP={}\n", d, sp);

    let wq = rand_m(d, d, 0.1, 42);
    let wk = rand_m(d, d, 0.1, 43);
    let wv = rand_m(d, d, 0.1, 44);
    let wo = rand_m(d, d, 0.1, 45);
    let x = rand_m(d, sp, 0.5, 9999);
    let target_w = rand_m(d, d, 0.5, 12345);
    let target = mm(&target_w, d, d, &x, sp);

    // Forward
    let q = mm(&wq, d, d, &x, sp);
    let k = mm(&wk, d, d, &x, sp);
    let v = mm(&wv, d, d, &x, sp);
    let (attn_out, attn_weights) = cpu_attention(&q, &k, &v, d, sp);
    let y = mm(&wo, d, d, &attn_out, sp);

    // Loss gradient
    let n = y.len() as f32;
    let mut dy = vec![0.0f32; y.len()];
    for i in 0..y.len() {
        dy[i] = 2.0 * (y[i] - target[i]) / n;
    }

    // Backward (same as full layer, but only attention part)
    // dattn_out = Wo^T @ dy
    let dattn_out = mm_at(&wo, d, d, &dy, sp);
    let dwo_a = mm_abt(&dy, d, sp, &attn_out, d);

    // dattn = dattn_out^T @ V → but V is stored [d, sp], and mm_at(a, k, m, b, n) = a^T @ b
    // dattn_out is [d x sp], we need dattn_out^T @ V where V is [d x sp]
    // Actually: in forward, av = mm(&attn, sp, sp, v, d) = attn[sp x sp] @ V_read_as[sp x d]
    // So dL/dattn = dattn_out_read_as[sp x d]^T @ ... hmm let me just match what the full layer does
    let dattn = mm_at(&dattn_out, d, sp, &v, sp); // [sp x sp]
    let dv_final = mm(&dattn_out, d, sp, &attn_weights, sp); // [d x sp]

    // Softmax backward
    let scale = 1.0 / (d as f32).sqrt();
    let mut dscaled = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let mut dot = 0.0_f32;
        for j in 0..sp {
            dot += dattn[i * sp + j] * attn_weights[i * sp + j];
        }
        for j in 0..sp {
            dscaled[i * sp + j] = attn_weights[i * sp + j] * (dattn[i * sp + j] - dot) * scale;
        }
    }

    // scores = Q^T @ K → dscaled is [sp x sp]
    // dQ = K @ dscaled^T, dK = Q @ dscaled
    let dq_final = mm_abt(&k, d, sp, &dscaled, sp); // [d x sp]
    let dk_final = mm(&q, d, sp, &dscaled, sp); // [d x sp]

    let dwq_a = mm_abt(&dq_final, d, sp, &x, d);
    let dwk_a = mm_abt(&dk_final, d, sp, &x, d);
    let dwv_a = mm_abt(&dv_final, d, sp, &x, d);

    // Numerical gradient for Wq
    let step = (wq.len() / sample_n).max(1);
    let mut dwq_num = vec![0.0f32; wq.len()];
    for i in (0..wq.len()).step_by(step) {
        let mut wp = wq.clone();
        wp[i] += eps;
        let qp = mm(&wp, d, d, &x, sp);
        let (ap, _) = cpu_attention(&qp, &k, &v, d, sp);
        let yp = mm(&wo, d, d, &ap, sp);
        let lp = mse_loss_f64(&yp, &target);

        let mut wm = wq.clone();
        wm[i] -= eps;
        let qm = mm(&wm, d, d, &x, sp);
        let (am, _) = cpu_attention(&qm, &k, &v, d, sp);
        let ym = mm(&wo, d, d, &am, sp);
        let lm = mse_loss_f64(&ym, &target);

        dwq_num[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWq (attn)", &dwq_a, &dwq_num, step, 0.01, 1e-3);

    // Numerical gradient for Wk
    let step = (wk.len() / sample_n).max(1);
    let mut dwk_num = vec![0.0f32; wk.len()];
    for i in (0..wk.len()).step_by(step) {
        let mut wp = wk.clone();
        wp[i] += eps;
        let kp = mm(&wp, d, d, &x, sp);
        let (ap, _) = cpu_attention(&q, &kp, &v, d, sp);
        let yp = mm(&wo, d, d, &ap, sp);
        let lp = mse_loss_f64(&yp, &target);

        let mut wm = wk.clone();
        wm[i] -= eps;
        let km = mm(&wm, d, d, &x, sp);
        let (am, _) = cpu_attention(&q, &km, &v, d, sp);
        let ym = mm(&wo, d, d, &am, sp);
        let lm = mse_loss_f64(&ym, &target);

        dwk_num[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWk (attn)", &dwk_a, &dwk_num, step, 0.01, 1e-3);

    // Numerical gradient for Wv
    let step = (wv.len() / sample_n).max(1);
    let mut dwv_num = vec![0.0f32; wv.len()];
    for i in (0..wv.len()).step_by(step) {
        let mut wp = wv.clone();
        wp[i] += eps;
        let vp = mm(&wp, d, d, &x, sp);
        let (ap, _) = cpu_attention(&q, &k, &vp, d, sp);
        let yp = mm(&wo, d, d, &ap, sp);
        let lp = mse_loss_f64(&yp, &target);

        let mut wm = wv.clone();
        wm[i] -= eps;
        let vm = mm(&wm, d, d, &x, sp);
        let (am, _) = cpu_attention(&q, &k, &vm, d, sp);
        let ym = mm(&wo, d, d, &am, sp);
        let lm = mse_loss_f64(&ym, &target);

        dwv_num[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWv (attn)", &dwv_a, &dwv_num, step, 0.01, 1e-3);

    // Numerical gradient for Wo
    let step = (wo.len() / sample_n).max(1);
    let mut dwo_num = vec![0.0f32; wo.len()];
    for i in (0..wo.len()).step_by(step) {
        let mut wp = wo.clone();
        wp[i] += eps;
        let yp = mm(&wp, d, d, &attn_out, sp);
        let lp = mse_loss_f64(&yp, &target);

        let mut wm = wo.clone();
        wm[i] -= eps;
        let ym = mm(&wm, d, d, &attn_out, sp);
        let lm = mse_loss_f64(&ym, &target);

        dwo_num[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWo (attn)", &dwo_a, &dwo_num, step, 0.01, 1e-6);

    println!();
}

fn test_linear() {
    println!("\n=== Test: Linear W @ x → MSE ===");
    let d = 64;
    let sp = 16;
    let w = rand_m(d, d, 0.1, 42);
    let x = rand_m(d, sp, 0.5, 9999);
    let target = mm(&rand_m(d, d, 0.5, 12345), d, d, &x, sp);

    let out = mm(&w, d, d, &x, sp);
    let n = out.len() as f64;
    let mut dy = vec![0.0f32; out.len()];
    for i in 0..out.len() {
        dy[i] = (2.0 * (out[i] - target[i]) / out.len() as f32) as f32;
    }
    let dw_a = mm_abt(&dy, d, sp, &x, d);

    // Check a single element directly
    let eps: f32 = 1e-4;
    println!(
        "    dy[0]={:.8e} target[0]={:.8e} n={:.0}",
        dy[0], target[0], n
    );
    println!("    dw_a[0]={:.8e} dw_a[1]={:.8e}", dw_a[0], dw_a[1]);

    // Numerical for element 0
    {
        let mut wp = w.clone();
        wp[0] += eps;
        let op = mm(&wp, d, d, &x, sp);
        let lp = mse_loss_f64(&op, &target);
        let mut wm = w.clone();
        wm[0] -= eps;
        let om = mm(&wm, d, d, &x, sp);
        let lm = mse_loss_f64(&om, &target);
        let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
        println!(
            "    dw[0]: analytical={:.8e} numerical={:.8e} rel={:.6e}",
            dw_a[0],
            num,
            (dw_a[0] - num).abs() / dw_a[0].max(num)
        );
    }
    // Numerical for element 1
    {
        let mut wp = w.clone();
        wp[1] += eps;
        let op = mm(&wp, d, d, &x, sp);
        let lp = mse_loss_f64(&op, &target);
        let mut wm = w.clone();
        wm[1] -= eps;
        let om = mm(&wm, d, d, &x, sp);
        let lm = mse_loss_f64(&om, &target);
        let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
        println!(
            "    dw[1]: analytical={:.8e} numerical={:.8e} rel={:.6e}",
            dw_a[1],
            num,
            (dw_a[1] - num).abs() / dw_a[1].max(num)
        );
    }

    // Direct computation of dW[0,0]
    // dW[0,0] = sum_j dy[0*sp+j] * x[0*sp+j]
    let mut dw00_direct = 0.0_f64;
    for j in 0..sp {
        dw00_direct += dy[0 * sp + j] as f64 * x[0 * sp + j] as f64;
    }
    println!(
        "    dw[0,0] direct={:.8e} mm_abt={:.8e}",
        dw00_direct as f32, dw_a[0]
    );

    let sample_n = 50;
    let step = (w.len() / sample_n).max(1);
    let mut dw_num = vec![0.0f32; w.len()];
    for i in (0..w.len()).step_by(step) {
        let mut wp = w.clone();
        wp[i] += eps;
        let op = mm(&wp, d, d, &x, sp);
        let lp = mse_loss_f64(&op, &target);
        let mut wm = w.clone();
        wm[i] -= eps;
        let om = mm(&wm, d, d, &x, sp);
        let lm = mse_loss_f64(&om, &target);
        dw_num[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dW", &dw_a, &dw_num, step, 0.02, 1e-6);
}

fn test_residual() {
    println!("\n=== Test: W @ x + x (residual) → MSE ===");
    let d = 64;
    let sp = 16;
    let w = rand_m(d, d, 0.1, 42);
    let x = rand_m(d, sp, 0.5, 9999);
    let target = mm(&rand_m(d, d, 0.5, 12345), d, d, &x, sp);

    let proj = mm(&w, d, d, &x, sp);
    let mut y = vec![0.0f32; d * sp];
    for i in 0..d * sp {
        y[i] = proj[i] + x[i];
    }
    let n = y.len() as f64;
    let mut dy = vec![0.0f32; y.len()];
    for i in 0..y.len() {
        dy[i] = (2.0 * (y[i] - target[i]) / y.len() as f32) as f32;
    }
    let dw_a = mm_abt(&dy, d, sp, &x, d);

    let eps: f32 = 1e-4;
    let sample_n = 50;
    let step = (w.len() / sample_n).max(1);
    let mut dw_num = vec![0.0f32; w.len()];
    for i in (0..w.len()).step_by(step) {
        let mut wp = w.clone();
        wp[i] += eps;
        let p = mm(&wp, d, d, &x, sp);
        let mut yp = vec![0.0f32; d * sp];
        for j in 0..d * sp {
            yp[j] = p[j] + x[j];
        }
        let lp = mse_loss_f64(&yp, &target);
        let mut wm = w.clone();
        wm[i] -= eps;
        let m = mm(&wm, d, d, &x, sp);
        let mut ym = vec![0.0f32; d * sp];
        for j in 0..d * sp {
            ym[j] = m[j] + x[j];
        }
        let lm = mse_loss_f64(&ym, &target);
        dw_num[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dW", &dw_a, &dw_num, step, 0.02, 1e-6);
}

/// Minimal FFN-only test: verify SwiGLU backward in isolation.
fn test_ffn_only() {
    let d: usize = 64;
    let inter: usize = d * 4;
    let sp: usize = 16;
    let eps: f32 = 1e-4;

    println!("\n=== FFN-only Verification (no attention) ===");
    println!("  D={} inter={} SP={}\n", d, inter, sp);

    let wg = rand_m(inter, d, 0.02, 42);
    let wu = rand_m(inter, d, 0.02, 43);
    let wd = rand_m(d, inter, 0.02, 44);
    let x = rand_m(d, sp, 0.1, 9999);
    let target_w = rand_m(d, d, 0.01, 12345);
    let target = mm(&target_w, d, d, &x, sp);

    // Forward: y = x (skip attention), gate = Wg@x, up = Wu@x, fused = silu(gate)*up, out = Wd@fused
    let gate = mm(&wg, inter, d, &x, sp);
    let up = mm(&wu, inter, d, &x, sp);
    let mut fused = vec![0.0f32; inter * sp];
    let mut sig = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        sig[i] = 1.0 / (1.0 + (-gate[i]).exp());
        fused[i] = gate[i] * sig[i] * up[i];
    }
    let out = mm(&wd, d, inter, &fused, sp);
    let y_out = &out; // Just Wd@fused, no residual

    let n = y_out.len() as f64;
    let loss_base = mse_loss_f64(y_out, &target);
    let mut dy = vec![0.0f32; y_out.len()];
    for i in 0..y_out.len() {
        dy[i] = (2.0 * (y_out[i] - target[i]) / y_out.len() as f32) as f32;
    }

    // Backward
    let dwd_a = mm_abt(&dy, d, sp, &fused, inter);
    let dfused = mm_at(&wd, d, inter, &dy, sp);
    let mut dup = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        dup[i] = dfused[i] * gate[i] * sig[i];
    }
    let dwu_a = mm_abt(&dup, inter, sp, &x, d);
    let mut dgate = vec![0.0f32; inter * sp];
    for i in 0..inter * sp {
        dgate[i] = dfused[i] * up[i] * sig[i] * (1.0 + gate[i] * (1.0 - sig[i]));
    }
    let dwg_a = mm_abt(&dgate, inter, sp, &x, d);

    // Numerical gradient for Wg
    let sample_n = 50;
    let step = (wg.len() / sample_n).max(1);
    let mut dwg_num = vec![0.0f32; wg.len()];
    for i in (0..wg.len()).step_by(step) {
        let mut wg_p = wg.clone();
        wg_p[i] += eps;
        let gate_p = mm(&wg_p, inter, d, &x, sp);
        let mut fused_p = vec![0.0f32; inter * sp];
        for j in 0..inter * sp {
            let s = 1.0 / (1.0 + (-gate_p[j]).exp());
            fused_p[j] = gate_p[j] * s * up[j]; // same up
        }
        let out_p = mm(&wd, d, inter, &fused_p, sp);
        let loss_p = mse_loss_f64(&out_p, &target);

        let mut wg_m = wg.clone();
        wg_m[i] -= eps;
        let gate_m = mm(&wg_m, inter, d, &x, sp);
        let mut fused_m = vec![0.0f32; inter * sp];
        for j in 0..inter * sp {
            let s = 1.0 / (1.0 + (-gate_m[j]).exp());
            fused_m[j] = gate_m[j] * s * up[j];
        }
        let out_m = mm(&wd, d, inter, &fused_m, sp);
        let loss_m = mse_loss_f64(&out_m, &target);

        dwg_num[i] = ((loss_p - loss_m) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWg (FFN)", &dwg_a, &dwg_num, step, 0.01, 1e-6);

    // Numerical gradient for Wu
    let step = (wu.len() / sample_n).max(1);
    let mut dwu_num = vec![0.0f32; wu.len()];
    for i in (0..wu.len()).step_by(step) {
        let mut wu_p = wu.clone();
        wu_p[i] += eps;
        let up_p = mm(&wu_p, inter, d, &x, sp);
        let mut fused_p = vec![0.0f32; inter * sp];
        for j in 0..inter * sp {
            fused_p[j] = gate[j] * sig[j] * up_p[j]; // same gate
        }
        let out_p = mm(&wd, d, inter, &fused_p, sp);
        let loss_p = mse_loss_f64(&out_p, &target);

        let mut wu_m = wu.clone();
        wu_m[i] -= eps;
        let up_m = mm(&wu_m, inter, d, &x, sp);
        let mut fused_m = vec![0.0f32; inter * sp];
        for j in 0..inter * sp {
            fused_m[j] = gate[j] * sig[j] * up_m[j];
        }
        let out_m = mm(&wd, d, inter, &fused_m, sp);
        let loss_m = mse_loss_f64(&out_m, &target);

        dwu_num[i] = ((loss_p - loss_m) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWu (FFN)", &dwu_a, &dwu_num, step, 0.01, 1e-6);

    // Numerical gradient for Wd
    let step = (wd.len() / sample_n).max(1);
    let mut dwd_num = vec![0.0f32; wd.len()];
    for i in (0..wd.len()).step_by(step) {
        let mut wd_p = wd.clone();
        wd_p[i] += eps;
        let out_p = mm(&wd_p, d, inter, &fused, sp);
        let loss_p = mse_loss_f64(&out_p, &target);

        let mut wd_m = wd.clone();
        wd_m[i] -= eps;
        let out_m = mm(&wd_m, d, inter, &fused, sp);
        let loss_m = mse_loss_f64(&out_m, &target);

        dwd_num[i] = ((loss_p - loss_m) / (2.0 * eps as f64)) as f32;
    }
    compare_grad("dWd (FFN)", &dwd_a, &dwd_num, step, 0.01, 1e-6);

    println!();
}
