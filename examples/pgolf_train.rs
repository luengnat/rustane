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

        let (a_c, b_c, c_c) = (3.4445f32, -4.7750f32, 2.0315f32);
        let eps = 1e-7f32;

        for i in 0..n {
            self.muon_buf[i] = mom * self.muon_buf[i] + self.grad[i];
        }

        let mut g_eff = vec![0.0f32; n];
        for i in 0..n {
            g_eff[i] = self.grad[i] + mom * self.muon_buf[i];
        }

        let x_norm: f32 = g_eff.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if x_norm > eps {
            let inv = 1.0 / x_norm;
            for v in g_eff.iter_mut() {
                *v *= inv;
            }
        }

        let (r, c) = (self.rows, self.cols);
        let transposed = r > c;
        let (mr, mc) = if transposed { (c, r) } else { (r, c) };

        let mut x = if transposed {
            let mut xt = vec![0.0f32; c * r];
            for i in 0..c {
                for j in 0..r {
                    xt[i * r + j] = g_eff[j * c + i];
                }
            }
            xt
        } else {
            g_eff
        };

        for _ in 0..muon_backend_steps {
            // a_mat = x @ x^T  [mr, mr]
            let a_mat = sgemm_tn(mr, mr, mc, &x, &x);
            // aa = a_mat @ a_mat  [mr, mr]
            let aa = sgemm_nn(mr, mr, mr, &a_mat, &a_mat);
            // b_mat = b * a_mat + c * aa  [mr, mr]
            let mut b_mat = vec![0.0f32; mr * mr];
            for i in 0..mr * mr {
                b_mat[i] = b_c * a_mat[i] + c_c * aa[i];
            }
            // bx = b_mat @ x  [mr, mc]
            let bx = sgemm_nn(mr, mc, mr, &b_mat, &x);
            // x = a * x + bx  [mr, mc]
            for i in 0..mr * mc {
                x[i] = a_c * x[i] + bx[i];
            }
        }

        let g_ortho = if transposed {
            let mut xt = vec![0.0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    xt[i * c + j] = x[j * r + i];
                }
            }
            xt
        } else {
            x
        };

        let scale = (r as f32 / c as f32).max(1.0).sqrt();
        for i in 0..n {
            self.data[i] -= lr * g_ortho[i] * scale;
        }
        self.grad.fill(0.0);
    }
}

struct Model {
    wte: Param,
    skip_weights: Param,
    n_encoder: usize,
    n_decoder: usize,
    layers: Vec<LayerParams>,
}

struct LayerParams {
    wq: Param,
    wk: Param,
    wv: Param,
    wo: Param,
    w_up: Param,
    w_down: Param,
    q_gain: Param,
    attn_scale: Param,
    mlp_scale: Param,
    resid_mix: Param,
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
    x0: Vec<f32>,
    x_mixed: Vec<f32>,
    normed: Vec<f32>,
    rms1: f32,
    q_pre_rope: Vec<f32>,
    k_pre_rope: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    attn_weights: Vec<f32>,
    vh_heads: Vec<f32>,
    attn_out: Vec<f32>,
    x_after_attn: Vec<f32>,
    normed2: Vec<f32>,
    rms2: f32,
    up: Vec<f32>,
    skip_added: Vec<f32>,
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
    logit_softcap: f32,
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

    let (x_normed, _) = rms_norm_fwd(&x, 1e-6);
    x = x_normed;
    let x0 = x.clone();

    let mut skips: Vec<Vec<f32>> = Vec::new();
    let mut layer_caches = Vec::new();

    for l in 0..layers {
        let lw = &model.layers[l];
        let x_pre = x.clone();

        // resid_mix: x = mix[0] * x + mix[1] * x0
        let mut x_mixed = vec![0.0f32; n];
        for i in 0..n {
            x_mixed[i] =
                lw.resid_mix.data[i % dim] * x[i] + lw.resid_mix.data[dim + i % dim] * x0[i];
        }

        // Encoder layers: save skip before block
        let mut skip_added = vec![0.0f32; n];
        if l < model.n_encoder {
            // No skip added in encoder phase
        } else {
            // Decoder: add skip if available
            let skip_idx = l - model.n_encoder;
            if skip_idx < skips.len() {
                let skip = skips[skips.len() - 1 - skip_idx].clone();
                for i in 0..n {
                    x_mixed[i] += model.skip_weights.data[skip_idx * dim + i % dim] * skip[i];
                    skip_added[i] = skip[i];
                }
            }
        }

        // Attention: rms_norm, then Q/K with RMSNorm before RoPE
        let (nm, rms1) = rms_norm_fwd(&x_mixed, 1e-6);
        let q_raw = sgemm_nn(dim, sp, dim, &lw.wq.data, &nm);
        let k_raw = sgemm_nn(dim, sp, dim, &lw.wk.data, &nm);
        let v = sgemm_nn(dim, sp, dim, &lw.wv.data, &nm);

        // RMSNorm Q and K per-head before RoPE (MLX does rms_norm(q), rms_norm(k))
        let mut q_pre_rope = vec![0.0f32; n];
        let mut k_pre_rope = vec![0.0f32; n];
        for h in 0..heads {
            for t in 0..sp {
                let q_base = t * dim + h * hd;
                let (qn, _) = rms_norm_fwd(&q_raw[q_base..q_base + hd], 1e-6);
                q_pre_rope[q_base..q_base + hd].copy_from_slice(&qn);
            }
        }
        for h in 0..kv_heads {
            for t in 0..sp {
                let k_base = t * dim + h * hd;
                let (kn, _) = rms_norm_fwd(&k_raw[k_base..k_base + hd], 1e-6);
                k_pre_rope[k_base..k_base + hd].copy_from_slice(&kn);
            }
        }

        // Apply q_gain per head
        for h in 0..heads {
            let gain = lw.q_gain.data[h];
            for t in 0..sp {
                let base = t * dim + h * hd;
                for d in 0..hd {
                    q_pre_rope[base + d] *= gain;
                }
            }
        }

        let mut q_rope = q_pre_rope.clone();
        let mut k_rope = k_pre_rope.clone();
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

        // proj = wo @ attn_out (wo starts at zero, so this is zero initially)
        let proj = sgemm_nn(dim, sp, dim, &lw.wo.data, &attn_out);

        // x = x_mixed + attn_scale * attn_out + proj
        // Since wo starts at zero, attn_out is scaled by attn_scale
        for i in 0..n {
            let scaled_attn = lw.attn_scale.data[i % dim] * attn_out[i] + proj[i];
            x[i] = x_mixed[i] + scaled_attn;
        }
        let x_after_attn = x.clone();

        // MLP
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
            x[i] += lw.mlp_scale.data[i % dim] * down[i];
        }

        if l < model.n_encoder {
            skips.push(x.clone());
        }

        layer_caches.push(LayerCache {
            x_pre,
            x0: x0.clone(),
            x_mixed,
            normed: nm,
            rms1,
            q_pre_rope,
            k_pre_rope,
            q_rope,
            k_rope,
            attn_weights,
            vh_heads,
            attn_out,
            x_after_attn,
            normed2: nm2,
            rms2,
            up,
            skip_added,
        });
    }

    let (x_final, rms_f) = rms_norm_fwd(&x, 1e-6);
    let logits_raw = sgemm_nn(vocab, sp, dim, &model.wte.data, &x_final);

    // logit softcap: softcap * tanh(logits / softcap)
    let logits: Vec<f32> = logits_raw
        .iter()
        .map(|&l| logit_softcap * (l / logit_softcap).tanh())
        .collect();

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
    logit_softcap: f32,
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

    // Through logit softcap: logits = c * tanh(logits_raw / c)
    // d(logits_raw) = dlogits * (1 - tanh^2(logits_raw / c))
    let mut dlogits_raw = vec![0.0f32; sp * vocab];
    for i in 0..dlogits.len() {
        let t = (cache.logits[i] / logit_softcap).tanh();
        dlogits_raw[i] = dlogits[i] * (1.0 - t * t);
    }

    // d_wte from logits = dlogits_raw @ x_final^T
    let dwte = sgemm_tn(vocab, dim, sp, &dlogits_raw, &cache.x_final);
    for i in 0..dwte.len() {
        model.wte.grad[i] += dwte[i];
    }

    // dx from final norm + logits
    let mut dx = sgemm_nt(dim, sp, vocab, &model.wte.data, &dlogits_raw);
    dx = rms_norm_bwd(
        &dx,
        &cache.layer_caches[layers - 1].x_after_attn,
        cache.rms_f,
        n,
    );

    // Accumulate gradient for x0 (the initial post-embed-norm x)
    // x0 flows through every layer's resid_mix, so we need to sum contributions
    let mut dx0 = vec![0.0f32; n];

    for l in (0..layers).rev() {
        let lc = &cache.layer_caches[l];

        let wq_data = model.layers[l].wq.data.clone();
        let wk_data = model.layers[l].wk.data.clone();
        let wv_data = model.layers[l].wv.data.clone();
        let wo_data = model.layers[l].wo.data.clone();
        let wup_data = model.layers[l].w_up.data.clone();
        let wdn_data = model.layers[l].w_down.data.clone();
        let attn_scale = model.layers[l].attn_scale.data.clone();
        let mlp_scale = model.layers[l].mlp_scale.data.clone();
        let resid_mix = model.layers[l].resid_mix.data.clone();

        // MLP backward: x_out = x_after_attn + mlp_scale * down
        // d(down) = dx, d(x_after_attn) = dx + dr2
        let d_down = &dx;
        let d_act = sgemm_nt(mlp_dim, sp, dim, &wdn_data, d_down);

        // d(mlp_scale): sum over sp positions for each dim
        // down has shape [sp, dim], d_down has shape [sp, dim]
        // We need to reconstruct "down" (the activated projected values)
        let activated: Vec<f32> = (0..sp * mlp_dim)
            .map(|i| {
                let a = lc.up[i].max(0.0);
                a * a
            })
            .collect();
        let down_values = sgemm_nn(dim, sp, mlp_dim, &wdn_data, &activated);

        for pos in 0..sp {
            for d in 0..dim {
                model.layers[l].mlp_scale.grad[d] += dx[pos * dim + d] * down_values[pos * dim + d];
            }
        }

        let d_up: Vec<f32> = (0..sp * mlp_dim)
            .map(|i| {
                if lc.up[i] > 0.0 {
                    mlp_scale[i % dim] * 2.0 * lc.up[i] * d_act[i]
                } else {
                    0.0
                }
            })
            .collect();

        let dw_down = sgemm_tn(dim, mlp_dim, sp, d_down, &activated);
        let dw_up = sgemm_tn(mlp_dim, dim, sp, &d_up, &lc.normed2);

        let d_normed2 = sgemm_nt(dim, sp, mlp_dim, &wup_data, &d_up);
        let dr2 = rms_norm_bwd(&d_normed2, &lc.x_after_attn, lc.rms2, n);

        // d(x_after_attn) = dx (residual) + dr2 (from MLP path)
        let mut d_x_after_attn = vec![0.0f32; n];
        for i in 0..n {
            d_x_after_attn[i] = dx[i] + dr2[i];
        }

        // Attention backward:
        // x_after_attn = x_mixed + attn_scale * attn_out + proj
        // where proj = wo @ attn_out
        // d(attn_out) = attn_scale * d_x_after_attn + wo^T @ d_x_after_attn
        // d(wo) = d_x_after_attn @ attn_out^T

        // d(attn_scale)
        for pos in 0..sp {
            for d in 0..dim {
                model.layers[l].attn_scale.grad[d] +=
                    d_x_after_attn[pos * dim + d] * lc.attn_out[pos * dim + d];
            }
        }

        // Combined d(attn_out) = diag(attn_scale) @ d_x_after_attn + wo^T @ d_x_after_attn
        let mut d_attn_out = sgemm_nt(dim, sp, dim, &wo_data, &d_x_after_attn);
        for i in 0..n {
            d_attn_out[i] += attn_scale[i % dim] * d_x_after_attn[i];
        }
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

        // Inverse RoPE
        let mut dq = dq_rope;
        let mut dk = dk_rope;
        rope_bwd(&mut dq, &mut dk, sp, hd, heads, kv_heads, dim);

        // q_gain backward: dq_raw += d(q_pre_rope) where q_pre_rope = q_gain * rms_norm(q_raw)
        // d(q_gain[h]) = sum over (t,d) of dq_pre_rope[t, h, d] * rms_norm_q[t, h, d]
        // But we need to first undo the q_gain scaling
        for h in 0..heads {
            let gain = model.layers[l].q_gain.data[h];
            let mut d_gain = 0.0f32;
            for t in 0..sp {
                for d in 0..hd {
                    let idx = t * dim + h * hd + d;
                    d_gain += dq[idx] * lc.q_pre_rope[idx];
                    dq[idx] *= gain;
                }
            }
            model.layers[l].q_gain.grad[h] += d_gain;
        }

        // Now dq and dk are w.r.t. q_pre_rope and k_pre_rope (post rms_norm)
        // We need to backprop through per-head rms_norm for Q and K
        // q_pre_rope[t, h, :] = rms_norm(q_raw[t, h, :])
        // This is per-(t,h) rms_norm over hd dimensions
        let mut dq_raw = vec![0.0f32; n];
        let mut dk_raw = vec![0.0f32; n];
        for h in 0..heads {
            for t in 0..sp {
                let base = t * dim + h * hd;
                let dq_slice = &dq[base..base + hd];
                let q_slice = &lc.q_pre_rope[base..base + hd];
                let (_, rms_q) = rms_norm_fwd(q_slice, 1e-6);
                let dr = rms_norm_bwd(dq_slice, q_slice, rms_q, hd);
                dq_raw[base..base + hd].copy_from_slice(&dr);
            }
        }
        for h in 0..kv_heads {
            for t in 0..sp {
                let base = t * dim + h * hd;
                let dk_slice = &dk[base..base + hd];
                let k_slice = &lc.k_pre_rope[base..base + hd];
                let (_, rms_k) = rms_norm_fwd(k_slice, 1e-6);
                let dr = rms_norm_bwd(dk_slice, k_slice, rms_k, hd);
                dk_raw[base..base + hd].copy_from_slice(&dr);
            }
        }

        let dwq = sgemm_tn(dim, dim, sp, &dq_raw, &lc.normed);
        let dwk = sgemm_tn(dim, dim, sp, &dk_raw, &lc.normed);
        let dwv = sgemm_tn(dim, dim, sp, &dv, &lc.normed);

        let d_normed = {
            let d1 = sgemm_nt(dim, sp, dim, &wq_data, &dq_raw);
            let d2 = sgemm_nt(dim, sp, dim, &wk_data, &dk_raw);
            let d3 = sgemm_nt(dim, sp, dim, &wv_data, &dv);
            let mut dn = vec![0.0f32; n];
            for i in 0..n {
                dn[i] = d1[i] + d2[i] + d3[i];
            }
            dn
        };

        let dr1 = rms_norm_bwd(&d_normed, &lc.x_mixed, lc.rms1, n);

        // d(x_mixed) = d_x_after_attn + dr1
        let mut d_x_mixed = vec![0.0f32; n];
        for i in 0..n {
            d_x_mixed[i] = d_x_after_attn[i] + dr1[i];
        }

        // resid_mix backward: x_mixed = mix[0]*x + mix[1]*x0
        // d(mix[0][d]) += sum over sp of d_x_mixed[pos, d] * x_pre[pos, d]
        // d(mix[1][d]) += sum over sp of d_x_mixed[pos, d] * x0[pos, d]
        // d(x_pre) += mix[0] * d_x_mixed
        // d(x0) += mix[1] * d_x_mixed
        let mut dx_out = vec![0.0f32; n];
        for pos in 0..sp {
            for d in 0..dim {
                let idx = pos * dim + d;
                let grad = d_x_mixed[idx];
                model.layers[l].resid_mix.grad[d] += grad * lc.x_pre[idx];
                model.layers[l].resid_mix.grad[dim + d] += grad * lc.x0[idx];
                dx_out[idx] = resid_mix[d] * grad;
                dx0[idx] += resid_mix[dim + d] * grad;
            }
        }

        // Skip connection backward (decoder layers only)
        let skip_idx = l.wrapping_sub(model.n_encoder);
        if l >= model.n_encoder && skip_idx < model.n_encoder {
            let sw_base = skip_idx * dim;
            for pos in 0..sp {
                for d in 0..dim {
                    let idx = pos * dim + d;
                    model.skip_weights.grad[sw_base + d] += d_x_mixed[idx] * lc.skip_added[idx];
                }
            }
        }

        // d(x_pre) flows to previous layer's x
        // d(x0) accumulates across all layers
        // x_pre for layer l is the output of layer l-1 (or x0 for layer 0)
        // We need: dx_for_prev = dx_out + (residual from MLP/attn)
        // But actually the residual connection is handled by the dx that flows through.
        // The key insight: dx carries the gradient from the output of this block back to
        // the input. The block output = x_mixed + attn_scaled + mlp_scaled.
        // d(x_mixed) is already computed as d_x_after_attn + dr1.
        // d(x_pre) = mix[0] * d(x_mixed) is what goes to the next layer.
        // d(x0) = mix[1] * d(x_mixed) accumulates.
        // For layer 0: x_pre = x0, so d(x0) also gets dx_out.
        dx = dx_out;

        // We need to accumulate dx0 contributions. For now just set dx = dx_out.
        // dx0 will be handled separately through the x0 chain.

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

    // Total gradient for x0: dx (from layer 0's x_pre path) + dx0 (from all layers' resid_mix[1])
    let mut dx0_total = vec![0.0f32; n];
    for i in 0..n {
        dx0_total[i] = dx[i] + dx0[i];
    }

    // Backprop through initial RMSNorm: x0 = rms_norm(wte_embed)
    let (x_embed, rms_embed) = {
        let mut xe = vec![0.0f32; n];
        for pos in 0..sp {
            let idx = tok[pos] as usize;
            for d in 0..dim {
                xe[pos * dim + d] = model.wte.data[idx * dim + d];
            }
        }
        let (_, rms) = rms_norm_fwd(&xe, 1e-6);
        (xe, rms)
    };
    let dx_embed = rms_norm_bwd(&dx0_total, &x_embed, rms_embed, n);
    for pos in 0..sp {
        let idx = tok[pos] as usize;
        for d in 0..dim {
            model.wte.grad[idx * dim + d] += dx_embed[pos * dim + d];
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
    println!("  Parameter-Golf Training: Rust CPU (BLAS) — MLX Architecture");
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
    let logit_softcap = 30.0f32;
    let qk_gain_init = 1.5f32;
    let tied_embed_init_std = 0.005f32;
    let embed_lr = 0.05f32;
    let matrix_lr = 0.04f32;
    let scalar_lr = 0.04f32;

    let n_encoder = nlayers / 2;
    let n_decoder = nlayers - n_encoder;
    let n_skip = n_encoder.min(n_decoder);

    println!(
        "  Config: {}L ({}enc+{}dec) {}D {}H {}KVH vocab={} mlp={}x sp={}",
        nlayers, n_encoder, n_decoder, dim, heads, kv_heads, vocab, mlp_mult, sp
    );
    println!(
        "  Steps: {}, Batch: {} tokens ({} seqs), softcap={}",
        steps, batch_tokens, seqs_per_step, logit_softcap
    );
    println!(
        "  LR: embed={}, matrix={}, scalar={}",
        embed_lr, matrix_lr, scalar_lr
    );

    println!("\n  Loading FineWeb data...");
    let tokens = load_tokens(&data_path, steps * batch_tokens + 1000);
    println!("  Loaded {} tokens\n", tokens.len());

    let wte = Param::new(rng_matrix(vocab, dim, tied_embed_init_std, 42));

    let skip_weights = Param::new(vec![1.0f32; n_skip * dim]);

    let qk_std = 0.02 / (dim as f32).sqrt();
    let layers: Vec<LayerParams> = (0..nlayers)
        .map(|l| {
            let s = (l * 100) as u64;
            let mut resid_mix = vec![0.0f32; 2 * dim];
            for d in 0..dim {
                resid_mix[d] = 1.0;
                resid_mix[dim + d] = 0.0;
            }
            LayerParams {
                wq: Param::with_shape(rng_matrix(dim, dim, qk_std, 100 + s), dim, dim),
                wk: Param::with_shape(rng_matrix(dim, dim, qk_std, 200 + s), dim, dim),
                wv: Param::with_shape(rng_matrix(dim, dim, 0.02, 300 + s), dim, dim),
                wo: Param::with_shape(vec![0.0f32; dim * dim], dim, dim),
                w_up: Param::with_shape(rng_matrix(mlp_dim, dim, 0.02, 500 + s), mlp_dim, dim),
                w_down: Param::with_shape(vec![0.0f32; dim * mlp_dim], dim, mlp_dim),
                q_gain: Param::new(vec![qk_gain_init; heads]),
                attn_scale: Param::new(vec![1.0f32; dim]),
                mlp_scale: Param::new(vec![1.0f32; dim]),
                resid_mix: Param::new(resid_mix),
            }
        })
        .collect();
    let mut model = Model {
        wte,
        skip_weights,
        n_encoder,
        n_decoder,
        layers,
    };

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
            let (loss, cache) = forward(
                seq,
                &model,
                dim,
                heads,
                kv_heads,
                mlp_dim,
                vocab,
                sp,
                logit_softcap,
            );
            total_loss += loss;
            backward(
                seq,
                &mut model,
                &cache,
                dim,
                heads,
                kv_heads,
                mlp_dim,
                vocab,
                sp,
                logit_softcap,
            );
        }

        // Apply gradients: Adam for embeddings, Muon for 2D matrices, Adam for scalars
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

        clip(&mut model.skip_weights);
        model.skip_weights.adam(scalar_lr, step);

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

            // Scalar params: Adam
            model.layers[l].q_gain.adam(scalar_lr, step);
            model.layers[l].attn_scale.adam(scalar_lr, step);
            model.layers[l].mlp_scale.adam(scalar_lr, step);
            model.layers[l].resid_mix.adam(scalar_lr, step);
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
    println!("  Matching MLX architecture: encoder-decoder, skip connections, zeroed wo/w_down,");
    println!("  learned scales (attn_scale, mlp_scale, resid_mix), q_gain, logit softcap.");
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
            "wk" => model.layers[0].wk.data[idx],
            "wv" => model.layers[0].wv.data[idx],
            "wo" => model.layers[0].wo.data[idx],
            "w_down" => model.layers[0].w_down.data[idx],
            "w_up" => model.layers[0].w_up.data[idx],
            "skip_weights" => model.skip_weights.data[idx],
            "attn_scale" => model.layers[0].attn_scale.data[idx],
            "mlp_scale" => model.layers[0].mlp_scale.data[idx],
            "resid_mix" => model.layers[0].resid_mix.data[idx],
            "q_gain" => model.layers[0].q_gain.data[idx],
            _ => unreachable!(),
        }
    }

    fn set_param(model: &mut Model, name: &str, idx: usize, value: f32) {
        match name {
            "wte" => model.wte.data[idx] = value,
            "wq" => model.layers[0].wq.data[idx] = value,
            "wk" => model.layers[0].wk.data[idx] = value,
            "wv" => model.layers[0].wv.data[idx] = value,
            "wo" => model.layers[0].wo.data[idx] = value,
            "w_down" => model.layers[0].w_down.data[idx] = value,
            "w_up" => model.layers[0].w_up.data[idx] = value,
            "skip_weights" => model.skip_weights.data[idx] = value,
            "attn_scale" => model.layers[0].attn_scale.data[idx] = value,
            "mlp_scale" => model.layers[0].mlp_scale.data[idx] = value,
            "resid_mix" => model.layers[0].resid_mix.data[idx] = value,
            "q_gain" => model.layers[0].q_gain.data[idx] = value,
            _ => unreachable!(),
        }
    }

    fn build_test_model() -> (Model, usize, usize, usize, usize, usize) {
        let dim = 4;
        let heads = 2;
        let kv_heads = 2;
        let mlp_dim = 8;
        let vocab = 6;
        let nlayers = 2;
        let n_encoder = nlayers / 2;
        let n_decoder = nlayers - n_encoder;
        let n_skip = n_encoder.min(n_decoder);

        let mut wte = Param::new(rng_matrix(vocab, dim, 0.05, 42));
        fill_small_weights(&mut wte.data, 0.03, 7);

        let skip_weights = Param::new(vec![1.0f32; n_skip * dim]);

        let mut make_layer = |seed: u64| {
            let mut resid_mix = vec![0.0f32; 2 * dim];
            for d in 0..dim {
                resid_mix[d] = 1.0;
                resid_mix[dim + d] = 0.0;
            }
            LayerParams {
                wq: Param::with_shape(rng_matrix(dim, dim, 0.05, 100 + seed), dim, dim),
                wk: Param::with_shape(rng_matrix(dim, dim, 0.05, 200 + seed), dim, dim),
                wv: Param::with_shape(rng_matrix(dim, dim, 0.05, 300 + seed), dim, dim),
                wo: Param::with_shape(vec![0.0f32; dim * dim], dim, dim),
                w_up: Param::with_shape(rng_matrix(mlp_dim, dim, 0.05, 500 + seed), mlp_dim, dim),
                w_down: Param::with_shape(vec![0.0f32; dim * mlp_dim], dim, mlp_dim),
                q_gain: Param::new(vec![1.5f32; heads]),
                attn_scale: Param::new(vec![1.0f32; dim]),
                mlp_scale: Param::new(vec![1.0f32; dim]),
                resid_mix: Param::new(resid_mix),
            }
        };

        let mut layer0 = make_layer(0);
        fill_small_weights(&mut layer0.wq.data, 0.02, 1);
        fill_small_weights(&mut layer0.wk.data, 0.02, 2);
        fill_small_weights(&mut layer0.wv.data, 0.02, 3);

        let mut layer1 = make_layer(100);
        fill_small_weights(&mut layer1.wq.data, 0.02, 4);
        fill_small_weights(&mut layer1.wk.data, 0.02, 5);
        fill_small_weights(&mut layer1.wv.data, 0.02, 6);

        (
            Model {
                wte,
                skip_weights,
                n_encoder,
                n_decoder,
                layers: vec![layer0, layer1],
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
        let (loss, cache) = forward(
            &tokens, &model, dim, heads, kv_heads, mlp_dim, vocab, sp, 30.0,
        );
        assert!(loss.is_finite());

        let layer = &model.layers[0];
        let lc = &cache.layer_caches[0];
        let x_embed = embed_tokens(&tokens[..sp], &model.wte.data, dim);
        let (x_normed, _) = rms_norm_fwd(&x_embed, 1e-6);

        // x_mixed = mix[0]*x + mix[1]*x0, where x=x0 initially, mix[0]=1, mix[1]=0
        assert_close(&lc.x_mixed, &x_normed, 1e-6);

        let normed = rms_norm_fwd(&lc.x_mixed, 1e-6).0;
        assert_close(&lc.normed, &normed, 1e-6);

        let q = naive_sgemm_nn(dim, sp, dim, &layer.wq.data, &normed);
        let k = naive_sgemm_nn(dim, sp, dim, &layer.wk.data, &normed);
        let v = naive_sgemm_nn(dim, sp, dim, &layer.wv.data, &normed);

        // Verify Q/K RMSNorm per head
        let hd = dim / heads;
        for h in 0..heads {
            for t in 0..sp {
                let base = t * dim + h * hd;
                let (qn, _) = rms_norm_fwd(&q[base..base + hd], 1e-6);
                assert_close(&lc.q_pre_rope[base..base + hd], &qn, 1e-6);
            }
        }
        for h in 0..kv_heads {
            for t in 0..sp {
                let base = t * dim + h * hd;
                let (kn, _) = rms_norm_fwd(&k[base..base + hd], 1e-6);
                assert_close(&lc.k_pre_rope[base..base + hd], &kn, 1e-6);
            }
        }

        let mut q_rope = lc.q_pre_rope.clone();
        let mut k_rope = lc.k_pre_rope.clone();
        rope_fwd(&mut q_rope, &mut k_rope, sp, hd, heads, kv_heads, dim);
        assert_close(&lc.q_rope, &q_rope, 1e-6);
        assert_close(&lc.k_rope, &k_rope, 1e-6);

        let (attn_out, attn_weights) =
            manual_attention(&q_rope, &k_rope, &v, sp, hd, heads, kv_heads, dim);
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

        // wo is zero, so proj is zero. x_after_attn = x_mixed + attn_scale * attn_out
        let mut x_after_attn = lc.x_mixed.clone();
        for i in 0..x_after_attn.len() {
            x_after_attn[i] += lc.attn_out[i];
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
        let logit_softcap = 30.0f32;

        let (loss, cache) = forward(
            &tokens,
            &model,
            dim,
            heads,
            kv_heads,
            mlp_dim,
            vocab,
            sp,
            logit_softcap,
        );
        assert!(loss.is_finite());

        backward(
            &tokens,
            &mut model,
            &cache,
            dim,
            heads,
            kv_heads,
            mlp_dim,
            vocab,
            sp,
            logit_softcap,
        );
        assert!(loss.is_finite());

        let eps = 1e-3f32;
        let checks = [
            ("wq", 0usize, model.layers[0].wq.grad[0]),
            ("wv", 0usize, model.layers[0].wv.grad[0]),
            ("skip_weights", 0usize, model.skip_weights.grad[0]),
            ("attn_scale", 0usize, model.layers[0].attn_scale.grad[0]),
            ("mlp_scale", 0usize, model.layers[0].mlp_scale.grad[0]),
            ("resid_mix", 0usize, model.layers[0].resid_mix.grad[0]),
            ("q_gain", 0usize, model.layers[0].q_gain.grad[0]),
        ];

        for (name, idx, analytic) in checks {
            let original = get_param(&model, name, idx);
            set_param(&mut model, name, idx, original + eps);
            let plus = forward(
                &tokens,
                &model,
                dim,
                heads,
                kv_heads,
                mlp_dim,
                vocab,
                sp,
                logit_softcap,
            )
            .0;
            set_param(&mut model, name, idx, original - eps);
            let minus = forward(
                &tokens,
                &model,
                dim,
                heads,
                kv_heads,
                mlp_dim,
                vocab,
                sp,
                logit_softcap,
            )
            .0;
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

        // wte check separately (tied embedding has two gradient paths)
        {
            let wte_idx = 0;
            let analytic = model.wte.grad[wte_idx];
            let original = model.wte.data[wte_idx];
            model.wte.data[wte_idx] = original + eps;
            let plus = forward(
                &tokens,
                &model,
                dim,
                heads,
                kv_heads,
                mlp_dim,
                vocab,
                sp,
                logit_softcap,
            )
            .0;
            model.wte.data[wte_idx] = original - eps;
            let minus = forward(
                &tokens,
                &model,
                dim,
                heads,
                kv_heads,
                mlp_dim,
                vocab,
                sp,
                logit_softcap,
            )
            .0;
            model.wte.data[wte_idx] = original;
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (analytic - numeric).abs() < 5e-2,
                "wte gradient mismatch: analytic={} numeric={}",
                analytic,
                numeric
            );
        }
    }
}
