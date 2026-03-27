fn rng(n: usize, s: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u64 ^ s).wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0
        })
        .collect()
}

fn attn_fwd(q: &[f32], k: &[f32], v: &[f32], sp: usize, hd: usize, dim: usize) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; sp * dim];
    for i in 0..sp {
        let mut scores = vec![0.0f32; sp];
        for j in 0..=i {
            scores[j] = (0..hd)
                .map(|d| q[i * dim + d] * k[j * dim + d])
                .sum::<f32>()
                * scale;
        }
        let mut mx = f32::NEG_INFINITY;
        for j in 0..=i {
            mx = mx.max(scores[j]);
        }
        let sm: f32 = (0..=i).map(|j| (scores[j] - mx).exp()).sum();
        for j in 0..=i {
            for d in 0..hd {
                out[i * dim + d] += (scores[j] - mx).exp() / sm * v[j * dim + d];
            }
        }
    }
    out
}

fn main() {
    let sp = 2;
    let hd = 2;
    let dim = hd;
    let q = rng(sp * dim, 60);
    let k = rng(sp * dim, 70);
    let v = rng(sp * dim, 80);
    let dy = rng(sp * dim, 90);
    let scale = 1.0 / (hd as f32).sqrt();

    // Analytical backward
    let out = attn_fwd(&q, &k, &v, sp, hd, dim);
    let loss_val: f32 = (0..sp * dim).map(|i| out[i] * dy[i]).sum();
    println!("out={:.6?}", out);
    println!("loss_val={:.6}", loss_val);
    let mut dq = vec![0.0f32; sp * dim];
    let mut dk = vec![0.0f32; sp * dim];
    let mut dv = vec![0.0f32; sp * dim];

    // Recompute attention weights for backward
    let mut aw = vec![0.0f32; sp * sp];
    for i in 0..sp {
        for j in 0..=i {
            aw[i * sp + j] = (0..hd)
                .map(|d| q[i * dim + d] * k[j * dim + d])
                .sum::<f32>()
                * scale;
        }
        let mut mx = f32::NEG_INFINITY;
        for j in 0..=i {
            mx = mx.max(aw[i * sp + j]);
        }
        let sm: f32 = (0..=i).map(|j| (aw[i * sp + j] - mx).exp()).sum();
        for j in 0..=i {
            aw[i * sp + j] = (aw[i * sp + j] - mx).exp() / sm;
        }
    }

    // d_aw, dv
    for i in 0..sp {
        for j in 0..=i {
            for d in 0..hd {
                let da = dy[i * dim + d] * v[j * dim + d]; // d_aw[i,j] contribution
                dq[i * dim + d] += 0.0; // placeholder
            }
        }
    }
    // Proper: d_aw[i,j] = sum_d dy[i,d]*v[j,d]
    let mut d_aw = vec![0.0f32; sp * sp];
    for i in 0..sp {
        for j in 0..=i {
            for d in 0..hd {
                d_aw[i * sp + j] += dy[i * dim + d] * v[j * dim + d];
            }
        }
    }
    for j in 0..sp {
        for d in 0..hd {
            for i in j..sp {
                dv[j * dim + d] += aw[i * sp + j] * dy[i * dim + d];
            }
        }
    }

    // softmax backward -> d_scores (unscaled)
    let mut d_s = vec![0.0f32; sp * sp];
    for i in 0..sp {
        let dot: f32 = (0..=i).map(|j| aw[i * sp + j] * d_aw[i * sp + j]).sum();
        for j in 0..=i {
            d_s[i * sp + j] = aw[i * sp + j] * (d_aw[i * sp + j] - dot);
        }
    }

    // dq, dk (apply scale here since scores = QK^T * scale)
    for i in 0..sp {
        for j in 0..=i {
            for d in 0..hd {
                dq[i * dim + d] += d_s[i * sp + j] * scale * k[j * dim + d];
                dk[j * dim + d] += d_s[i * sp + j] * scale * q[i * dim + d];
            }
        }
    }

    // Numerical gradient check
    let eps = 1e-4f32;
    let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
        let o = attn_fwd(q, k, v, sp, hd, dim);
        (0..sp * dim).map(|i| o[i] * dy[i]).sum::<f32>()
    };

    println!("aw={:.4?}", &aw[..sp * sp]);
    println!("d_aw={:.4?}", &d_aw[..sp * sp]);
    println!("d_s={:.4?}", &d_s[..sp * sp]);
    println!("scale={:.4}", scale);
    println!("q={:.6?}", q);
    println!("k={:.6?}", k);
    println!("v={:.6?}", v);
    println!("dy={:.6?}", dy);
    println!("dq={:.6?}", dq);
    println!("dk={:.6?}", dk);
    println!();
    // Detailed check for q[2]
    let idx = 2;
    let mut qc = q.clone();
    qc[idx] += eps;
    let lp = loss(&qc, &k, &v);
    qc[idx] -= 2.0 * eps;
    let lm = loss(&qc, &k, &v);
    let l0 = loss(&q, &k, &v);
    println!(
        "q[2]={:.6}, loss(q)={:.6}, loss(q+eps)={:.6}, loss(q-eps)={:.6}",
        q[idx], l0, lp, lm
    );
    println!(
        "num_grad=({:.6}-{:.6})/2e-4 = {:.6}",
        lp,
        lm,
        (lp - lm) / (2.0 * eps)
    );
    println!("anal_grad={:.6}", dq[idx]);
    for idx in 0..sp * dim {
        let mut qc = q.clone();
        qc[idx] += eps;
        let lp = loss(&qc, &k, &v);
        qc[idx] -= 2.0 * eps;
        let lm = loss(&qc, &k, &v);
        let num = (lp - lm) / (2.0 * eps);
        let err = (dq[idx] - num).abs();
        let rel = err / (dq[idx].abs() + num.abs() + 1e-8);
        if err > 0.001 {
            println!(
                "dq[{}] anal={:.6} num={:.6} abs={:.2e} rel={:.4}",
                idx, dq[idx], num, err, rel
            );
        }
    }
    for idx in 0..sp * dim {
        let mut kc = k.clone();
        kc[idx] += eps;
        let lp = loss(&q, &kc, &v);
        kc[idx] -= 2.0 * eps;
        let lm = loss(&q, &kc, &v);
        let num = (lp - lm) / (2.0 * eps);
        let err = (dk[idx] - num).abs();
        let rel = err / (dk[idx].abs() + num.abs() + 1e-8);
        if err > 0.001 {
            println!(
                "dk[{}] anal={:.6} num={:.6} abs={:.2e} rel={:.4}",
                idx, dk[idx], num, err, rel
            );
        }
    }
    for idx in 0..sp * dim {
        let mut vc = v.clone();
        vc[idx] += eps;
        let lp = loss(&q, &k, &vc);
        vc[idx] -= 2.0 * eps;
        let lm = loss(&q, &k, &vc);
        let num = (lp - lm) / (2.0 * eps);
        let err = (dv[idx] - num).abs();
        let rel = err / (dv[idx].abs() + num.abs() + 1e-8);
        if err > 0.001 {
            println!(
                "dv[{}] anal={:.6} num={:.6} abs={:.2e} rel={:.4}",
                idx, dv[idx], num, err, rel
            );
        }
    }
}
