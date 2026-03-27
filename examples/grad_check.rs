//! Focused gradient check for pgolf_train backward pass components

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        l: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        b: f32,
        C: *mut f32,
        ldc: i32,
    );
}

fn mm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            101,
            111,
            111,
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
fn mm_tn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
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
fn mm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            101,
            112,
            111,
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

fn rng(n: usize, s: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u64 ^ s).wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0
        })
        .collect()
}

fn rms_norm_fwd(x: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>();
    let rms = (ss / n as f32 + eps).sqrt();
    (x.iter().map(|v| v / rms).collect(), rms)
}

fn rms_norm_bwd(dy: &[f32], x: &[f32], rms: f32) -> Vec<f32> {
    let n = x.len();
    let nf = n as f32;
    let dot: f32 = (0..n).map(|i| dy[i] * x[i]).sum::<f32>();
    (0..n)
        .map(|i| (dy[i] - x[i] * dot / (rms * rms * nf)) / rms)
        .collect()
}

fn grad_check(name: &str, anal: &[f32], param: &[f32], f: &dyn Fn(&[f32]) -> f32) {
    let eps = 1e-4f32;
    let mut maxerr = 0.0f32;
    for i in 0..param.len().min(50) {
        let mut p = param.to_vec();
        p[i] += eps;
        let lp = f(&p);
        p[i] -= 2.0 * eps;
        let lm = f(&p);
        let num = (lp - lm) / (2.0 * eps);
        let abs_err = (anal[i] - num).abs();
        let rel = abs_err / (anal[i].abs() + num.abs() + 1e-6);
        let err = if anal[i].abs() + num.abs() < 1e-4 {
            abs_err
        } else {
            rel
        };
        if err > maxerr {
            maxerr = err;
        }
    }
    let status = if maxerr < 0.001 {
        "PASS"
    } else if maxerr < 0.05 {
        "WARN"
    } else {
        "FAIL"
    };
    println!("  [{status}] {name}: max_err={:.6}", maxerr);
    if maxerr > 0.01 {
        for i in 0..param.len().min(10) {
            let mut p = param.to_vec();
            p[i] += eps;
            let lp = f(&p);
            p[i] -= 2.0 * eps;
            let lm = f(&p);
            let num = (lp - lm) / (2.0 * eps);
            let abs_err = (anal[i] - num).abs();
            if abs_err > 0.001 {
                println!(
                    "    [{}] anal={:.6} num={:.6} abs_err={:.6e}",
                    i, anal[i], num, abs_err
                );
            }
        }
    }
}

fn main() {
    println!("=== Gradient Check ===\n");

    // Test 1: Cross-entropy
    println!("1. Cross-entropy");
    {
        let v = 6;
        let sp = 4;
        let logits = rng(sp * v, 1);
        let tokens = vec![0u16, 3, 5, 2, 1];
        let mut dl = vec![0.0f32; sp * v];
        for p in 0..sp - 1 {
            let t = tokens[p + 1] as usize;
            let b = (p + 1) * v;
            let mut mx = f32::NEG_INFINITY;
            for i in 0..v {
                mx = mx.max(logits[b + i]);
            }
            let mut sm = 0.0f32;
            for i in 0..v {
                dl[b + i] = (logits[b + i] - mx).exp();
                sm += dl[b + i];
            }
            for i in 0..v {
                dl[b + i] /= sm;
            }
            dl[b + t] -= 1.0;
            let sc = 1.0 / (sp - 1) as f32;
            for i in 0..v {
                dl[b + i] *= sc;
            }
        }
        grad_check("CE dlogits", &dl, &logits, &|l: &[f32]| {
            let mut loss = 0.0f32;
            for p in 0..sp - 1 {
                let t = tokens[p + 1] as usize;
                let b = (p + 1) * v;
                let mut mx = f32::NEG_INFINITY;
                for i in 0..v {
                    mx = mx.max(l[b + i]);
                }
                let sm: f32 = (0..v).map(|i| (l[b + i] - mx).exp()).sum();
                loss -= l[b + t] - mx - sm.ln();
            }
            loss / (sp - 1) as f32
        });
    }

    // Test 2: RMSNorm
    println!("\n2. RMSNorm");
    {
        let n = 8;
        let x = rng(n, 10);
        let dy = rng(n, 20);
        let (_, rms) = rms_norm_fwd(&x, 1e-6);
        let dx = rms_norm_bwd(&dy, &x, rms);
        grad_check("RMSNorm dx", &dx, &x, &|x: &[f32]| {
            let (y, _) = rms_norm_fwd(x, 1e-6);
            (0..n).map(|i| y[i] * dy[i]).sum::<f32>()
        });
    }

    // Test 3: Linear (y=W@x, L=dot(dy,y))
    println!("\n3. Linear layer");
    {
        let m = 4;
        let k = 3;
        let n = 2;
        let w = rng(m * k, 30);
        let x = rng(k * n, 40);
        let dy = rng(m * n, 50);
        let dw = mm_tn(m, k, n, &dy, &x);
        let dx = mm_nt(k, n, m, &w, &dy);
        println!("  dx analytical: {:?}", &dx[..6.min(dx.len())]);
        // Manual: dx[i,j] = sum_l w[l,k=i] * dy[l,j]
        let mut dx_manual = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                for l in 0..m {
                    dx_manual[i * n + j] += w[l * k + i] * dy[l * n + j];
                }
            }
        }
        println!("  dx manual:     {:?}", dx_manual);
        // Numerical for first element
        let mut xc = x.clone();
        xc[0] += 1e-4;
        let lp = (0..m * n)
            .map(|i| mm(m, n, k, &w, &xc)[i] * dy[i])
            .sum::<f32>();
        xc[0] -= 2e-4;
        let lm = (0..m * n)
            .map(|i| mm(m, n, k, &w, &xc)[i] * dy[i])
            .sum::<f32>();
        println!("  dx numerical[0]: {}", (lp - lm) / (2e-4));
        grad_check("Linear dW", &dw, &w, &|w: &[f32]| {
            let y = mm(m, n, k, w, &x);
            (0..m * n).map(|i| y[i] * dy[i]).sum::<f32>()
        });
        grad_check("Linear dx", &dx, &x, &|x: &[f32]| {
            let y = mm(m, n, k, &w, x);
            (0..m * n).map(|i| y[i] * dy[i]).sum::<f32>()
        });
    }

    // Test 4: Causal attention (most likely buggy)
    println!("\n4. Causal attention (GQA)");
    {
        let sp = 2;
        let hd = 2;
        let heads = 1;
        let kv_heads = 1;
        let dim = hd * heads;
        let q = rng(sp * dim, 60);
        let k = rng(sp * dim, 70);
        let v = rng(sp * dim, 80);
        let scale = 1.0 / (hd as f32).sqrt();

        // Forward
        let mut out = vec![0.0f32; sp * dim];
        let mut aw = vec![0.0f32; heads * sp * sp];
        for h in 0..heads {
            let kv_h = h * kv_heads / heads;
            for i in 0..sp {
                for j in 0..=i {
                    let dot: f32 = (0..hd)
                        .map(|d| q[i * dim + h * hd + d] * k[j * dim + kv_h * hd + d])
                        .sum();
                    aw[h * sp * sp + i * sp + j] = dot * scale;
                }
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    mx = mx.max(aw[h * sp * sp + i * sp + j]);
                }
                let mut sm = 0.0f32;
                for j in 0..=i {
                    aw[h * sp * sp + i * sp + j] = (aw[h * sp * sp + i * sp + j] - mx).exp();
                    sm += aw[h * sp * sp + i * sp + j];
                }
                for j in 0..=i {
                    aw[h * sp * sp + i * sp + j] /= sm;
                }
                for d in 0..hd {
                    let val: f32 = (0..=i)
                        .map(|j| aw[h * sp * sp + i * sp + j] * v[j * dim + kv_h * hd + d])
                        .sum();
                    out[i * dim + h * hd + d] = val;
                }
            }
        }

        // Backward (same as pgolf_train)
        let dy = rng(sp * dim, 90);
        let mut dq = vec![0.0f32; sp * dim];
        let mut dk = vec![0.0f32; sp * dim];
        let mut dv = vec![0.0f32; sp * dim];

        for h in 0..heads {
            let kv_h = h * kv_heads / heads;
            let ab = h * sp * sp;

            // Extract per-head arrays
            let mut dy_h = vec![0.0f32; sp * hd];
            let mut v_h = vec![0.0f32; sp * hd];
            for i in 0..sp {
                for d in 0..hd {
                    dy_h[i * hd + d] = dy[i * dim + h * hd + d];
                    v_h[i * hd + d] = v[i * dim + kv_h * hd + d];
                }
            }

            // d_aw = dy_h @ v_h^T  [sp,sp]
            let mut d_aw = mm_tn(sp, sp, hd, &dy_h, &v_h);
            // Zero out upper triangle (causal mask)
            for i in 0..sp {
                for j in (i + 1)..sp {
                    d_aw[i * sp + j] = 0.0;
                }
            }
            // d_vh = aw^T @ dy_h  [sp,hd]
            let d_vh = mm_nt(sp, hd, sp, &aw[ab..ab + sp * sp], &dy_h);
            for j in 0..sp {
                for d in 0..hd {
                    dv[j * dim + kv_h * hd + d] += d_vh[j * hd + d];
                }
            }

            // softmax backward -> d_scores
            let mut ds = vec![0.0f32; sp * sp];
            for i in 0..sp {
                let dot: f32 = (0..=i)
                    .map(|j| aw[ab + i * sp + j] * d_aw[i * sp + j])
                    .sum();
                for j in 0..=i {
                    ds[i * sp + j] = (d_aw[i * sp + j] - aw[ab + i * sp + j] * dot) * scale;
                }
            }

            // Extract q_h, k_h
            let mut q_h = vec![0.0f32; sp * hd];
            let mut k_h = vec![0.0f32; sp * hd];
            for i in 0..sp {
                for d in 0..hd {
                    q_h[i * hd + d] = q[i * dim + h * hd + d];
                    k_h[i * hd + d] = k[i * dim + kv_h * hd + d];
                }
            }

            // dq_h = d_scores @ k_h
            let dq_h = mm(sp, hd, sp, &ds, &k_h);
            // dk_h = d_scores^T @ q_h
            let dk_h = mm_nt(sp, hd, sp, &ds, &q_h);
            for i in 0..sp {
                for d in 0..hd {
                    dq[i * dim + h * hd + d] += dq_h[i * hd + d];
                    dk[i * dim + kv_h * hd + d] += dk_h[i * hd + d];
                }
            }
        }

        let fwd = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            let mut o = vec![0.0f32; sp * dim];
            for h in 0..heads {
                let kv_h = h * kv_heads / heads;
                for i in 0..sp {
                    let mut scores = vec![0.0f32; sp];
                    for j in 0..=i {
                        scores[j] = (0..hd)
                            .map(|d| q[i * dim + h * hd + d] * k[j * dim + kv_h * hd + d])
                            .sum::<f32>()
                            * scale;
                    }
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..=i {
                        mx = mx.max(scores[j]);
                    }
                    let sm: f32 = (0..=i).map(|j| (scores[j] - mx).exp()).sum::<f32>();
                    for j in 0..=i {
                        let a = (scores[j] - mx).exp() / sm;
                        for d in 0..hd {
                            o[i * dim + h * hd + d] += a * v[j * dim + kv_h * hd + d];
                        }
                    }
                }
            }
            (0..sp * dim).map(|i| o[i] * dy[i]).sum::<f32>()
        };

        let q_c = q.clone();
        let k_c = k.clone();
        let v_c = v.clone();
        grad_check("Attn dq", &dq, &q_c, &|q: &[f32]| fwd(q, &k, &v));
        grad_check("Attn dk", &dk, &k_c, &|k: &[f32]| fwd(&q, k, &v));
        grad_check("Attn dv", &dv, &v_c, &|v: &[f32]| fwd(&q, &k, v));
    }

    // Test 5: relu^2 backward
    println!("\n5. relu^2 MLP activation");
    {
        let n = 8;
        let x = rng(n, 100);
        let dy = rng(n, 110);
        let mut dx = vec![0.0f32; n];
        for i in 0..n {
            if x[i] > 0.0 {
                dx[i] = 2.0 * x[i] * dy[i];
            }
        }
        grad_check("relu2 dx", &dx, &x, &|x: &[f32]| {
            (0..n).map(|i| x[i].max(0.0).powi(2) * dy[i]).sum::<f32>()
        });
    }

    println!("\n=== Done ===");
}
