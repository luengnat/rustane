//! Parameter-Golf Training with Backprop (Rust CPU, BLAS)
//!
//! Usage: cargo run --release --example pgolf_train

use std::env;
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

fn sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
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

fn sgemm_tn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // C[m,n] = A[m,k] @ B^T[k,n]
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
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

fn sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // C[m,n] = A^T[k,m] @ B[k,n]
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            112,
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

struct Param {
    data: Vec<f32>,
    grad: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    muon_buf: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Param {
    fn new(data: Vec<f32>) -> Self {
        let n = data.len();
        Param {
            data,
            grad: vec![0.0f32; n],
            m: vec![0.0f32; n],
            v: vec![0.0f32; n],
            muon_buf: vec![0.0f32; n],
            rows: 0,
            cols: 0,
        }
    }

    fn with_shape(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        let n = data.len();
        Param {
            data,
            grad: vec![0.0f32; n],
            m: vec![0.0f32; n],
            v: vec![0.0f32; n],
            muon_buf: vec![0.0f32; n],
            rows,
            cols,
        }
    }

    fn adam(&mut self, lr: f32, step: usize) {
        let (b1, b2, eps) = (0.9f32, 0.95f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(step as i32 + 1);
        let bc2 = 1.0 - b2.powi(step as i32 + 1);
        for i in 0..self.data.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * self.grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * self.grad[i] * self.grad[i];
            self.data[i] -= lr * self.m[i] / bc1 / ((self.v[i] / bc2).sqrt() + eps);
        }
        self.grad.fill(0.0);
    }

    fn muon(&mut self, lr: f32, step: usize) {
        let n = self.data.len();
        if n == 0 || self.rows == 0 {
            return;
        }

        let momentum = 0.95f32;
        let warmup_steps = 500usize;
        let muon_backend_steps = 5;

        let mom = if warmup_steps > 0 {
            let t = (step as f32 / warmup_steps as f32).min(1.0);
            (1.0 - t) * 0.85 + t * momentum
        } else {
            momentum
        };

        let (a_coeff, b_coeff, c_coeff) = (3.4445f32, -4.7750f32, 2.0315f32);
        let eps = 1e-7f32;

        for i in 0..n {
            self.muon_buf[i] = mom * self.muon_buf[i] + self.grad[i];
        }

        let mut g_eff = vec![0.0f32; n];
        for i in 0..n {
            g_eff[i] = self.grad[i] + mom * self.muon_buf[i];
        }

        let mut x_norm: f32 = g_eff.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if x_norm > eps {
            let inv = 1.0 / x_norm;
            for v in g_eff.iter_mut() {
                *v *= inv;
            }
        }

        let (m, k) = (self.rows, self.cols);
        let transposed = m > k;

        let mut x = if transposed {
            let mut xt = vec![0.0f32; k * m];
            for i in 0..k {
                for j in 0..m {
                    xt[i * m + j] = g_eff[j * k + i];
                }
            }
            xt
        } else {
            g_eff
        };

        let (xr, xc) = if transposed { (k, m) } else { (m, k) };

        for _ in 0..muon_backend_steps {
            let mut a_mat = vec![0.0f32; xr * xr];
            for i in 0..xr {
                for j in 0..xr {
                    a_mat[i * xr + j] = 0.0f32;
                    for p in 0..xc {
                        a_mat[i * xr + j] += x[i * xc + p] * x[j * xc + p];
                    }
                }
            }
            let mut aa = vec![0.0f32; xr * xr];
            for i in 0..xr {
                for j in 0..xr {
                    aa[i * xr + j] = 0.0f32;
                    for p in 0..xr {
                        aa[i * xr + j] += a_mat[i * xr + p] * a_mat[p * xr + j];
                    }
                }
            }
            let mut b_mat = vec![0.0f32; xr * xr];
            for i in 0..xr {
                for j in 0..xr {
                    b_mat[i * xr + j] = b_coeff * a_mat[i * xr + j] + c_coeff * aa[i * xr + j];
                }
            }
            let mut bx = vec![0.0f32; xr * xc];
            for i in 0..xr {
                for j in 0..xc {
                    bx[i * xc + j] = 0.0f32;
                    for p in 0..xr {
                        bx[i * xc + j] += b_mat[i * xr + p] * x[p * xc + j];
                    }
                }
            }
            for i in 0..xr {
                for j in 0..xc {
                    x[i * xc + j] = a_coeff * x[i * xc + j] + bx[i * xc + j];
                }
            }
        }

        let g_ortho = if transposed {
            let mut xt = vec![0.0f32; m * k];
            for i in 0..m {
                for j in 0..k {
                    xt[i * k + j] = x[j * m + i];
                }
            }
            xt
        } else {
            x
        };

        let scale = (m as f32 / k as f32).max(1.0).sqrt();
        for i in 0..n {
            self.data[i] -= lr * g_ortho[i] * scale;
        }
        self.grad.fill(0.0);
    }
}

struct Model {
    wte: Param,
    layers: Vec<LayerParams>,
}

struct LayerParams {
    wq: Param,
    wk: Param,
    wv: Param,
    wo: Param,
    w_up: Param,
    w_down: Param,
}

fn rng_matrix(rows: usize, cols: usize, std: f32, seed: u64) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let x = ((i as u64)
                .wrapping_add(seed)
                .wrapping_mul(0x9E3779B97F4A7C15)
                >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * std
        })
        .collect()
}

struct LayerCache {
    x_pre: Vec<f32>,
    normed: Vec<f32>,
    rms1: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    attn_weights: Vec<f32>,
    vh_heads: Vec<f32>,
    attn_out: Vec<f32>,
    x_after_attn: Vec<f32>,
    normed2: Vec<f32>,
    rms2: f32,
    up: Vec<f32>,
}

struct ForwardCache {
    layer_caches: Vec<LayerCache>,
    x_final: Vec<f32>,
    rms_f: f32,
    logits: Vec<f32>,
}

fn rms_norm_fwd(x: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (ss / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    (x.iter().map(|&v| v * inv).collect(), rms)
}

fn rms_norm_bwd(dy: &[f32], x: &[f32], rms: f32, n: usize) -> Vec<f32> {
    let n_f = n as f32;
    let dot: f32 = (0..n).map(|i| dy[i] * x[i]).sum();
    (0..n)
        .map(|i| (dy[i] - x[i] * dot / (rms * rms * n_f)) / rms)
        .collect()
}

fn rope_fwd(
    q: &mut [f32],
    k: &mut [f32],
    sp: usize,
    hd: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) {
    let half = hd / 2;
    for t in 0..sp {
        for d in 0..half {
            let freq = t as f32 * 10000.0_f32.powf(-2.0 * d as f32 / hd as f32);
            let (c, s) = (freq.cos(), freq.sin());
            for h in 0..heads {
                let b = t * dim + h * hd;
                let (v0, v1) = (q[b + d], q[b + d + half]);
                q[b + d] = v0 * c - v1 * s;
                q[b + d + half] = v0 * s + v1 * c;
            }
            for h in 0..kv_heads {
                let b = t * dim + h * hd;
                let (v0, v1) = (k[b + d], k[b + d + half]);
                k[b + d] = v0 * c - v1 * s;
                k[b + d + half] = v0 * s + v1 * c;
            }
        }
    }
}

fn rope_bwd(
    dq: &mut [f32],
    dk: &mut [f32],
    sp: usize,
    hd: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) {
    let half = hd / 2;
    for t in 0..sp {
        for d in 0..half {
            let freq = t as f32 * 10000.0_f32.powf(-2.0 * d as f32 / hd as f32);
            let (c, s) = (freq.cos(), freq.sin());
            for h in 0..heads {
                let b = t * dim + h * hd;
                let (g0, g1) = (dq[b + d], dq[b + d + half]);
                dq[b + d] = g0 * c + g1 * s;
                dq[b + d + half] = -g0 * s + g1 * c;
            }
            for h in 0..kv_heads {
                let b = t * dim + h * hd;
                let (g0, g1) = (dk[b + d], dk[b + d + half]);
                dk[b + d] = g0 * c + g1 * s;
                dk[b + d + half] = -g0 * s + g1 * c;
            }
        }
    }
}

fn forward(
    tok: &[u16],
    model: &Model,
    dim: usize,
    heads: usize,
    kv_heads: usize,
    mlp_dim: usize,
    vocab: usize,
    sp: usize,
) -> (f32, ForwardCache) {
    let layers = model.layers.len();
    let hd = dim / heads;
    let n = sp * dim;

    let mut x = vec![0.0f32; n];
    for pos in 0..sp {
        let idx = tok[pos] as usize;
        for d in 0..dim {
            x[pos * dim + d] = model.wte.data[idx * dim + d];
        }
    }

    let mut layer_caches = Vec::new();

    for l in 0..layers {
        let lw = &model.layers[l];
        let x_pre = x.clone();

        let (nm, rms1) = rms_norm_fwd(&x, 1e-6);
        let q = sgemm_nn(dim, sp, dim, &lw.wq.data, &nm);
        let k = sgemm_nn(dim, sp, dim, &lw.wk.data, &nm);
        let v = sgemm_nn(dim, sp, dim, &lw.wv.data, &nm);
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        rope_fwd(&mut q_rope, &mut k_rope, sp, hd, heads, kv_heads, dim);

        // causal attention
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = vec![0.0f32; n];
        let mut attn_weights = vec![0.0f32; heads * sp * sp];
        let mut vh_heads = vec![0.0f32; heads * sp * hd];
        for h in 0..heads {
            let kv_h = if kv_heads == heads {
                h
            } else {
                h * kv_heads / heads
            };
            let aw_base = h * sp * sp;
            let vh_base = h * sp * hd;
            for i in 0..sp {
                for d in 0..hd {
                    vh_heads[vh_base + i * hd + d] = v[i * dim + kv_h * hd + d];
                }
            }
            for i in 0..sp {
                let aw_i = aw_base + i * sp;
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    let dot: f32 = (0..hd)
                        .map(|d| q_rope[i * dim + h * hd + d] * k_rope[j * dim + kv_h * hd + d])
                        .sum();
                    attn_weights[aw_i + j] = dot * scale;
                    mx = mx.max(attn_weights[aw_i + j]);
                }
                let mut sm = 0.0f32;
                for j in 0..=i {
                    attn_weights[aw_i + j] = (attn_weights[aw_i + j] - mx).exp();
                    sm += attn_weights[aw_i + j];
                }
                let inv = 1.0 / sm;
                for j in 0..=i {
                    attn_weights[aw_i + j] *= inv;
                }
                for d in 0..hd {
                    let val: f32 = (0..=i)
                        .map(|j| attn_weights[aw_i + j] * vh_heads[vh_base + j * hd + d])
                        .sum();
                    attn_out[i * dim + h * hd + d] = val;
                }
            }
        }

        let proj = sgemm_nn(dim, sp, dim, &lw.wo.data, &attn_out);
        for i in 0..n {
            x[i] += proj[i];
        }
        let x_after_attn = x.clone();

        let (nm2, rms2) = rms_norm_fwd(&x, 1e-6);
        let up = sgemm_nn(mlp_dim, sp, dim, &lw.w_up.data, &nm2);
        let activated: Vec<f32> = up
            .iter()
            .map(|&v| {
                let a = v.max(0.0);
                a * a
            })
            .collect();
        let down = sgemm_nn(dim, sp, mlp_dim, &lw.w_down.data, &activated);
        for i in 0..n {
            x[i] += down[i];
        }

        layer_caches.push(LayerCache {
            x_pre,
            normed: nm,
            rms1,
            q,
            k,
            v,
            q_rope,
            k_rope,
            attn_weights,
            vh_heads,
            attn_out,
            x_after_attn,
            normed2: nm2,
            rms2,
            up,
        });
    }

    let (x_final, rms_f) = rms_norm_fwd(&x, 1e-6);
    let logits = sgemm_nn(vocab, sp, dim, &model.wte.data, &x_final);

    let mut loss = 0.0f32;
    for pos in 0..sp {
        let target = tok[pos + 1] as usize;
        let base = pos * vocab;
        let mut mx = f32::NEG_INFINITY;
        for i in 0..vocab {
            mx = mx.max(logits[base + i]);
        }
        let sm: f32 = (0..vocab).map(|i| (logits[base + i] - mx).exp()).sum();
        loss += -(logits[base + target] - mx - sm.ln());
    }
    (
        loss / sp as f32,
        ForwardCache {
            layer_caches,
            x_final,
            rms_f,
            logits,
        },
    )
}

fn backward(
    tok: &[u16],
    model: &mut Model,
    cache: &ForwardCache,
    dim: usize,
    heads: usize,
    kv_heads: usize,
    mlp_dim: usize,
    vocab: usize,
    sp: usize,
) {
    let layers = model.layers.len();
    let hd = dim / heads;
    let n = sp * dim;

    // dlogits from cross-entropy
    let mut dlogits = vec![0.0f32; sp * vocab];
    for pos in 0..sp {
        let target = tok[pos + 1] as usize;
        let base = pos * vocab;
        let mut mx = f32::NEG_INFINITY;
        for i in 0..vocab {
            mx = mx.max(cache.logits[base + i]);
        }
        let mut sm = 0.0f32;
        for i in 0..vocab {
            dlogits[base + i] = (cache.logits[base + i] - mx).exp();
            sm += dlogits[base + i];
        }
        for i in 0..vocab {
            dlogits[base + i] /= sm;
        }
        dlogits[base + target] -= 1.0;
        let sc = 1.0 / sp as f32;
        for i in 0..vocab {
            dlogits[base + i] *= sc;
        }
    }

    // d_wte from logits = dlogits @ x_final^T
    let dwte = sgemm_tn(vocab, dim, sp, &dlogits, &cache.x_final);
    for i in 0..dwte.len() {
        model.wte.grad[i] += dwte[i];
    }

    // dx from final norm + logits
    let mut dx = sgemm_nt(dim, sp, vocab, &model.wte.data, &dlogits);
    // For the final RMSNorm, x_pre is the x after the last layer
    let last = &cache.layer_caches[layers - 1];
    let wo_last = &model.layers[layers - 1].wo.data;
    let wup_last = &model.layers[layers - 1].w_up.data;
    let wdn_last = &model.layers[layers - 1].w_down.data;
    let mut x_after_last = last.x_pre.clone();
    let proj = sgemm_nn(dim, sp, dim, wo_last, &last.attn_out);
    for i in 0..n {
        x_after_last[i] += proj[i];
    }
    let (nm2, _) = rms_norm_fwd(&x_after_last, 1e-6);
    let up = sgemm_nn(mlp_dim, sp, dim, wup_last, &nm2);
    let activated: Vec<f32> = up
        .iter()
        .map(|&v| {
            let a = v.max(0.0);
            a * a
        })
        .collect();
    let down = sgemm_nn(dim, sp, mlp_dim, wdn_last, &activated);
    for i in 0..n {
        x_after_last[i] += down[i];
    }

    dx = rms_norm_bwd(&dx, &x_after_last, cache.rms_f, n);

    for l in (0..layers).rev() {
        let lc = &cache.layer_caches[l];

        let wq_data = &model.layers[l].wq.data;
        let wk_data = &model.layers[l].wk.data;
        let wv_data = &model.layers[l].wv.data;
        let wo_data = &model.layers[l].wo.data;
        let wup_data = &model.layers[l].w_up.data;
        let wdn_data = &model.layers[l].w_down.data;

        // MLP backward: x_out = x_after_attn + down
        // d(down) = dx (residual), and d(x_after_attn) also gets dx (residual) + dr2 (from MLP)
        let d_down = &dx;
        let d_act = sgemm_nt(mlp_dim, sp, dim, wdn_data, d_down);
        let d_up: Vec<f32> = (0..sp * mlp_dim)
            .map(|i| {
                if lc.up[i] > 0.0 {
                    2.0 * lc.up[i] * d_act[i]
                } else {
                    0.0
                }
            })
            .collect();

        let activated: Vec<f32> = (0..sp * mlp_dim)
            .map(|i| {
                let a = lc.up[i].max(0.0);
                a * a
            })
            .collect();
        let dw_down = sgemm_tn(dim, mlp_dim, sp, d_down, &activated);
        let dw_up = sgemm_tn(mlp_dim, dim, sp, &d_up, &lc.normed2);

        let d_normed2 = sgemm_nt(dim, sp, mlp_dim, wup_data, &d_up);
        let dr2 = rms_norm_bwd(&d_normed2, &lc.x_after_attn, lc.rms2, n);

        // d(x_after_attn) = dx (residual) + dr2 (from MLP path)
        let mut d_x_after_attn = vec![0.0f32; n];
        for i in 0..n {
            d_x_after_attn[i] = dx[i] + dr2[i];
        }

        // Attention backward: x_after_attn = x_pre + proj, where proj = wo @ attn_out
        // d(proj) = d_x_after_attn
        let d_attn_out = sgemm_nt(dim, sp, dim, wo_data, &d_x_after_attn);
        let d_wo = sgemm_tn(dim, dim, sp, &d_x_after_attn, &lc.attn_out);

        let scale = 1.0 / (hd as f32).sqrt();
        let mut dq_rope = vec![0.0f32; n];
        let mut dk_rope = vec![0.0f32; n];
        let mut dv = vec![0.0f32; n];
        for h in 0..heads {
            let kv_h = if kv_heads == heads {
                h
            } else {
                h * kv_heads / heads
            };
            let aw_base = h * sp * sp;
            let vh_base = h * sp * hd;

            let mut d_av_h = vec![0.0f32; sp * hd];
            for i in 0..sp {
                for d in 0..hd {
                    d_av_h[i * hd + d] = d_attn_out[i * dim + h * hd + d];
                }
            }
            let vh_h = &lc.vh_heads[vh_base..vh_base + sp * hd];
            let mut d_aw = sgemm_tn(sp, sp, hd, &d_av_h, vh_h);
            for i in 0..sp {
                for j in (i + 1)..sp {
                    d_aw[i * sp + j] = 0.0;
                }
            }

            let attn_h = &lc.attn_weights[aw_base..aw_base + sp * sp];
            let d_vh_h = sgemm_nt(sp, hd, sp, attn_h, &d_av_h);
            for j in 0..sp {
                for d in 0..hd {
                    dv[j * dim + kv_h * hd + d] += d_vh_h[j * hd + d];
                }
            }

            let mut d_scores = vec![0.0f32; sp * sp];
            for i in 0..sp {
                let dot: f32 = (0..=i)
                    .map(|j| attn_h[i * sp + j] * d_aw[i * sp + j])
                    .sum::<f32>();
                for j in 0..=i {
                    d_scores[i * sp + j] = attn_h[i * sp + j] * (d_aw[i * sp + j] - dot) * scale;
                }
            }

            let mut qh = vec![0.0f32; sp * hd];
            let mut kh = vec![0.0f32; sp * hd];
            for i in 0..sp {
                for d in 0..hd {
                    qh[i * hd + d] = lc.q_rope[i * dim + h * hd + d];
                    kh[i * hd + d] = lc.k_rope[i * dim + kv_h * hd + d];
                }
            }
            let dqh = sgemm_nn(sp, hd, sp, &d_scores, &kh);
            let dkh = sgemm_nt(sp, hd, sp, &d_scores, &qh);
            for i in 0..sp {
                for d in 0..hd {
                    dq_rope[i * dim + h * hd + d] += dqh[i * hd + d];
                    dk_rope[i * dim + kv_h * hd + d] += dkh[i * hd + d];
                }
            }
        }

        let mut dq = dq_rope;
        let mut dk = dk_rope;
        rope_bwd(&mut dq, &mut dk, sp, hd, heads, kv_heads, dim);

        let dwq = sgemm_tn(dim, dim, sp, &dq, &lc.normed);
        let dwk = sgemm_tn(dim, dim, sp, &dk, &lc.normed);
        let dwv = sgemm_tn(dim, dim, sp, &dv, &lc.normed);

        let d_normed = {
            let d1 = sgemm_nt(dim, sp, dim, wq_data, &dq);
            let d2 = sgemm_nt(dim, sp, dim, wk_data, &dk);
            let d3 = sgemm_nt(dim, sp, dim, wv_data, &dv);
            let mut dn = vec![0.0f32; n];
            for i in 0..n {
                dn[i] = d1[i] + d2[i] + d3[i];
            }
            dn
        };

        let dr1 = rms_norm_bwd(&d_normed, &lc.x_pre, lc.rms1, n);

        // d(x_pre) = d_x_after_attn (residual) + dr1 (from attention QKV path through rms_norm_1)
        dx = vec![0.0f32; n];
        for i in 0..n {
            dx[i] = d_x_after_attn[i] + dr1[i];
        }

        {
            let layer = &mut model.layers[l];
            for i in 0..dwq.len() {
                layer.wq.grad[i] += dwq[i];
            }
            for i in 0..dwk.len() {
                layer.wk.grad[i] += dwk[i];
            }
            for i in 0..dwv.len() {
                layer.wv.grad[i] += dwv[i];
            }
            for i in 0..d_wo.len() {
                layer.wo.grad[i] += d_wo[i];
            }
            for i in 0..dw_up.len() {
                layer.w_up.grad[i] += dw_up[i];
            }
            for i in 0..dw_down.len() {
                layer.w_down.grad[i] += dw_down[i];
            }
        }
    }

    for pos in 0..sp {
        let idx = tok[pos] as usize;
        for d in 0..dim {
            model.wte.grad[idx * dim + d] += dx[pos * dim + d];
        }
    }
}

fn load_tokens(data_path: &str, count: usize) -> Vec<u16> {
    use std::fs;
    use std::io::Read;
    let pattern = format!("{}/fineweb_train_*.bin", data_path);
    let paths: Vec<_> = glob::glob(&pattern)
        .unwrap()
        .filter_map(|p| p.ok())
        .collect();
    if paths.is_empty() {
        panic!("No shards at {}", pattern);
    }
    let mut all = Vec::new();
    for path in &paths {
        let mut f = fs::File::open(path).unwrap();
        let mut hdr = [0u8; 1024];
        f.read_exact(&mut hdr).unwrap();
        let n = i32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let mut buf = vec![0u8; n * 2];
        f.read_exact(&mut buf).unwrap();
        for i in 0..n {
            all.push(u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]));
        }
        if all.len() >= count {
            break;
        }
    }
    all.truncate(count);
    all
}

fn main() {
    let data_path = env::var("DATA_PATH").unwrap_or_else(|_| {
        "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024".to_string()
    });

    println!("============================================================");
    println!("  Parameter-Golf Training: Rust CPU (BLAS) + Adam");
    println!("============================================================\n");

    let dim = 256;
    let heads = 4;
    let kv_heads = 2;
    let mlp_mult = 2;
    let vocab = 1024;
    let nlayers = 4;
    let mlp_dim = dim * mlp_mult;
    let sp = 256;
    let steps = 30;
    let batch_tokens = 2048;
    let seqs_per_step = batch_tokens / sp;
    let lr = 0.04;

    println!(
        "  Config: {}L {}D {}H {}KVH vocab={} mlp={}x sp={}",
        nlayers, dim, heads, kv_heads, vocab, mlp_mult, sp
    );
    println!(
        "  Steps: {}, Batch: {} tokens ({} seqs), Optimizer: Muon(lr={}) + Adam(lr=0.05)",
        steps, batch_tokens, seqs_per_step, lr
    );

    println!("\n  Loading FineWeb data...");
    let tokens = load_tokens(&data_path, steps * batch_tokens + 1000);
    println!("  Loaded {} tokens\n", tokens.len());

    let wte = Param::new(rng_matrix(vocab, dim, 0.005, 42));
    let qk_std = 0.02 / (dim as f32).sqrt();
    let layers: Vec<LayerParams> = (0..nlayers)
        .map(|l| {
            let s = (l * 100) as u64;
            LayerParams {
                wq: Param::with_shape(rng_matrix(dim, dim, qk_std, 100 + s), dim, dim),
                wk: Param::with_shape(rng_matrix(dim, dim, qk_std, 200 + s), dim, dim),
                wv: Param::with_shape(rng_matrix(dim, dim, 0.02, 300 + s), dim, dim),
                wo: Param::with_shape(rng_matrix(dim, dim, qk_std, 400 + s), dim, dim),
                w_up: Param::with_shape(rng_matrix(mlp_dim, dim, 0.02, 500 + s), mlp_dim, dim),
                w_down: Param::with_shape(rng_matrix(dim, mlp_dim, 0.02, 600 + s), dim, mlp_dim),
            }
        })
        .collect();
    let mut model = Model { wte, layers };

    println!(
        "  {:>6} {:>10} {:>10} {:>12} {:>10}",
        "step", "loss", "ppl", "ms/step", "tok/s"
    );
    println!("  {}", "-".repeat(52));

    for step in 1..=steps {
        let offset = (step - 1) * batch_tokens;
        let t0 = Instant::now();
        let mut total_loss = 0.0f32;

        for s in 0..seqs_per_step {
            let tok_start = offset + s * sp;
            let seq = &tokens[tok_start..tok_start + sp + 1];
            let (loss, cache) = forward(seq, &model, dim, heads, kv_heads, mlp_dim, vocab, sp);
            total_loss += loss;
            backward(
                seq, &mut model, &cache, dim, heads, kv_heads, mlp_dim, vocab, sp,
            );
        }

        // apply gradients
        let embed_lr = 0.05f32;
        let matrix_lr = 0.04f32;
        let max_grad_norm = 1.0f32;
        let clip = |p: &mut Param| {
            let norm: f32 = p.grad.iter().map(|&g| g * g).sum::<f32>().sqrt();
            if norm > max_grad_norm {
                let s = max_grad_norm / norm;
                for g in p.grad.iter_mut() {
                    *g *= s;
                }
            }
        };
        clip(&mut model.wte);
        model.wte.adam(embed_lr, step);
        for l in 0..nlayers {
            clip(&mut model.layers[l].wq);
            model.layers[l].wq.muon(matrix_lr, step);
            clip(&mut model.layers[l].wk);
            model.layers[l].wk.muon(matrix_lr, step);
            clip(&mut model.layers[l].wv);
            model.layers[l].wv.muon(matrix_lr, step);
            clip(&mut model.layers[l].wo);
            model.layers[l].wo.muon(matrix_lr, step);
            clip(&mut model.layers[l].w_up);
            model.layers[l].w_up.muon(matrix_lr, step);
            clip(&mut model.layers[l].w_down);
            model.layers[l].w_down.muon(matrix_lr, step);
        }

        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let avg_loss = total_loss / seqs_per_step as f32;
        let ppl = avg_loss.exp();
        let tok_s = batch_tokens as f64 / elapsed.as_secs_f64();
        println!(
            "  {:>6} {:>10.4} {:>10.1} {:>10.1} {:>10.0}",
            step, avg_loss, ppl, ms, tok_s
        );
    }

    println!("\n  MLX reference: loss 6.9351 -> 5.4287 in 30 steps (~24.5ms/step, ~84K tok/s)");
    println!("  Both use Muon+Adam with matching hyperparameters.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_data_dir() -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rustane_pgolf_train_{}_{}",
            std::process::id(),
            stamp
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_fineweb_shard(dir: &std::path::Path, tokens: &[u16]) {
        let path = dir.join("fineweb_train_000.bin");
        let mut file = File::create(path).unwrap();
        let mut header = vec![0u8; 1024];
        header[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        file.write_all(&header).unwrap();
        for token in tokens {
            file.write_all(&token.to_le_bytes()).unwrap();
        }
    }

    fn naive_sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (idx, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= tol,
                "mismatch at {}: lhs={} rhs={} tol={}",
                idx,
                lhs,
                rhs,
                tol
            );
        }
    }

    fn embed_tokens(tokens: &[u16], wte: &[f32], dim: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; tokens.len() * dim];
        for (pos, &tok) in tokens.iter().enumerate() {
            let idx = tok as usize;
            for d in 0..dim {
                x[pos * dim + d] = wte[idx * dim + d];
            }
        }
        x
    }

    fn manual_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sp: usize,
        hd: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; sp * dim];
        let mut weights = vec![0.0f32; heads * sp * sp];
        for h in 0..heads {
            let kv_h = if kv_heads == heads {
                h
            } else {
                h * kv_heads / heads
            };
            let aw_base = h * sp * sp;
            for i in 0..sp {
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    let dot: f32 = (0..hd)
                        .map(|d| q[i * dim + h * hd + d] * k[j * dim + kv_h * hd + d])
                        .sum();
                    let score = dot * scale;
                    weights[aw_base + i * sp + j] = score;
                    mx = mx.max(score);
                }
                let mut sum = 0.0f32;
                for j in 0..=i {
                    let idx = aw_base + i * sp + j;
                    weights[idx] = (weights[idx] - mx).exp();
                    sum += weights[idx];
                }
                for j in 0..=i {
                    let idx = aw_base + i * sp + j;
                    weights[idx] /= sum;
                    for d in 0..hd {
                        out[i * dim + h * hd + d] += weights[idx] * v[j * dim + kv_h * hd + d];
                    }
                }
            }
        }
        (out, weights)
    }

    fn fill_small_weights(data: &mut [f32], scale: f32, seed: usize) {
        for (idx, value) in data.iter_mut().enumerate() {
            let raw = ((idx + seed) % 17) as f32 - 8.0;
            *value = raw * scale;
        }
    }

    fn get_param(model: &Model, name: &str, idx: usize) -> f32 {
        match name {
            "wte" => model.wte.data[idx],
            "wq" => model.layers[0].wq.data[idx],
            "wo" => model.layers[0].wo.data[idx],
            "w_down" => model.layers[0].w_down.data[idx],
            _ => unreachable!(),
        }
    }

    fn set_param(model: &mut Model, name: &str, idx: usize, value: f32) {
        match name {
            "wte" => model.wte.data[idx] = value,
            "wq" => model.layers[0].wq.data[idx] = value,
            "wo" => model.layers[0].wo.data[idx] = value,
            "w_down" => model.layers[0].w_down.data[idx] = value,
            _ => unreachable!(),
        }
    }

    fn build_test_model() -> (Model, usize, usize, usize, usize, usize) {
        let dim = 4;
        let heads = 2;
        let kv_heads = 2;
        let mlp_dim = 8;
        let vocab = 6;

        let mut wte = Param::new(rng_matrix(vocab, dim, 0.05, 42));
        fill_small_weights(&mut wte.data, 0.03, 7);

        let mut layer = LayerParams {
            wq: Param::new(rng_matrix(dim, dim, 0.05, 100)),
            wk: Param::new(rng_matrix(dim, dim, 0.05, 200)),
            wv: Param::new(rng_matrix(dim, dim, 0.05, 300)),
            wo: Param::new(rng_matrix(dim, dim, 0.05, 400)),
            w_up: Param::new(rng_matrix(mlp_dim, dim, 0.05, 500)),
            w_down: Param::new(rng_matrix(dim, mlp_dim, 0.05, 600)),
        };
        fill_small_weights(&mut layer.wq.data, 0.02, 1);
        fill_small_weights(&mut layer.wk.data, 0.02, 2);
        fill_small_weights(&mut layer.wv.data, 0.02, 3);
        fill_small_weights(&mut layer.wo.data, 0.02, 4);
        fill_small_weights(&mut layer.w_up.data, 0.02, 5);
        fill_small_weights(&mut layer.w_down.data, 0.02, 6);

        (
            Model {
                wte,
                layers: vec![layer],
            },
            dim,
            heads,
            kv_heads,
            mlp_dim,
            vocab,
        )
    }

    #[test]
    fn test_sgemm_nn_matches_naive_rectangular() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.0, 2.0, 3.0, 1.0, 4.0, 5.0, 2.0, 6.0, 7.0, 3.0, 8.0];
        let actual = sgemm_nn(2, 3, 4, &a, &b);
        let expected = naive_sgemm_nn(2, 3, 4, &a, &b);
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "lhs={} rhs={}", lhs, rhs);
        }
    }

    #[test]
    fn test_rms_norm_step_matches_formula() {
        let x = vec![1.0f32, -2.0, 3.0, -4.0];
        let (y, rms) = rms_norm_fwd(&x, 1e-6);
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let expected_rms = (mean_sq + 1e-6).sqrt();
        assert!((rms - expected_rms).abs() < 1e-6);
        let expected: Vec<f32> = x.iter().map(|&v| v / expected_rms).collect();
        assert_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_rms_norm_backward_matches_finite_difference() {
        let x = vec![0.25f32, -0.5, 0.75, -1.0];
        let dy = vec![0.2f32, -0.3, 0.4, -0.1];
        let (_, rms) = rms_norm_fwd(&x, 1e-6);
        let dx = rms_norm_bwd(&dy, &x, rms, x.len());
        let eps = 1e-4f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += eps;
            let yp = rms_norm_fwd(&xp, 1e-6).0;
            let lp: f32 = yp.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();

            let mut xm = x.clone();
            xm[i] -= eps;
            let ym = rms_norm_fwd(&xm, 1e-6).0;
            let lm: f32 = ym.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();

            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (dx[i] - numeric).abs() < 5e-3,
                "rmsnorm dx mismatch at {}: analytic={} numeric={}",
                i,
                dx[i],
                numeric
            );
        }
    }

    #[test]
    fn test_rope_roundtrip_step() {
        let original_q = vec![1.0f32, 2.0, 3.0, 4.0];
        let original_k = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut q = original_q.clone();
        let mut k = original_k.clone();
        rope_fwd(&mut q, &mut k, 2, 2, 1, 1, 2);
        rope_bwd(&mut q, &mut k, 2, 2, 1, 1, 2);
        assert_close(&q, &original_q, 1e-5);
        assert_close(&k, &original_k, 1e-5);
    }

    #[test]
    fn test_forward_cache_tracks_each_step() {
        let (model, dim, heads, kv_heads, mlp_dim, vocab) = build_test_model();
        let tokens = vec![0u16, 1, 2, 3];
        let sp = tokens.len() - 1;
        let (loss, cache) = forward(&tokens, &model, dim, heads, kv_heads, mlp_dim, vocab, sp);
        assert!(loss.is_finite());

        let layer = &model.layers[0];
        let lc = &cache.layer_caches[0];
        let x_pre = embed_tokens(&tokens[..sp], &model.wte.data, dim);
        assert_close(&lc.x_pre, &x_pre, 1e-6);

        let normed = rms_norm_fwd(&x_pre, 1e-6).0;
        assert_close(&lc.normed, &normed, 1e-6);

        let q = naive_sgemm_nn(dim, sp, dim, &layer.wq.data, &normed);
        let k = naive_sgemm_nn(dim, sp, dim, &layer.wk.data, &normed);
        let v = naive_sgemm_nn(dim, sp, dim, &layer.wv.data, &normed);
        assert_close(&lc.q, &q, 1e-6);
        assert_close(&lc.k, &k, 1e-6);
        assert_close(&lc.v, &v, 1e-6);

        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        rope_fwd(
            &mut q_rope,
            &mut k_rope,
            sp,
            dim / heads,
            heads,
            kv_heads,
            dim,
        );
        assert_close(&lc.q_rope, &q_rope, 1e-6);
        assert_close(&lc.k_rope, &k_rope, 1e-6);

        let (attn_out, attn_weights) =
            manual_attention(&q_rope, &k_rope, &v, sp, dim / heads, heads, kv_heads, dim);
        assert_close(&lc.attn_out, &attn_out, 1e-6);
        assert_close(&lc.attn_weights, &attn_weights, 1e-6);
        for i in 0..sp {
            let row = &lc.attn_weights[i * sp..(i + 1) * sp];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
            for j in (i + 1)..sp {
                assert_eq!(row[j], 0.0);
            }
        }

        let proj = naive_sgemm_nn(dim, sp, dim, &layer.wo.data, &attn_out);
        let mut x_after_attn = x_pre.clone();
        for i in 0..x_after_attn.len() {
            x_after_attn[i] += proj[i];
        }
        assert_close(&lc.x_after_attn, &x_after_attn, 1e-6);

        let normed2 = rms_norm_fwd(&x_after_attn, 1e-6).0;
        assert_close(&lc.normed2, &normed2, 1e-6);

        let up = naive_sgemm_nn(mlp_dim, sp, dim, &layer.w_up.data, &normed2);
        assert_close(&lc.up, &up, 1e-6);
    }

    #[test]
    fn test_load_tokens_reads_expected_count() {
        let dir = temp_data_dir();
        let tokens = vec![11u16, 22, 33, 44, 55];
        write_fineweb_shard(&dir, &tokens);

        let loaded = load_tokens(dir.to_str().unwrap(), 3);
        assert_eq!(loaded, tokens[..3].to_vec());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_forward_backward_gradients_match_finite_difference() {
        let (mut model, dim, heads, kv_heads, mlp_dim, vocab) = build_test_model();
        let tokens = vec![0u16, 1, 2, 3];
        let sp = tokens.len() - 1;

        let (loss, cache) = forward(&tokens, &model, dim, heads, kv_heads, mlp_dim, vocab, sp);
        assert!(loss.is_finite());

        backward(
            &tokens, &mut model, &cache, dim, heads, kv_heads, mlp_dim, vocab, sp,
        );

        let eps = 1e-3f32;
        let checks = [
            ("wte", 0usize, model.wte.grad[0]),
            ("wq", 0usize, model.layers[0].wq.grad[0]),
            ("wo", 0usize, model.layers[0].wo.grad[0]),
            ("w_down", 0usize, model.layers[0].w_down.grad[0]),
        ];

        for (name, idx, analytic) in checks {
            let original = get_param(&model, name, idx);
            set_param(&mut model, name, idx, original + eps);
            let plus = forward(&tokens, &model, dim, heads, kv_heads, mlp_dim, vocab, sp).0;
            set_param(&mut model, name, idx, original - eps);
            let minus = forward(&tokens, &model, dim, heads, kv_heads, mlp_dim, vocab, sp).0;
            set_param(&mut model, name, idx, original);

            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (analytic - numeric).abs() < 5e-2,
                "{} gradient mismatch: analytic={} numeric={}",
                name,
                analytic,
                numeric
            );
        }
    }
}
