//! Transformer model with a real causal attention + FFN stack.
//!
//! This implementation keeps the `Model` trait surface used by the trainer,
//! but replaces the earlier embedding-plus-classifier shim with an actual
//! pre-norm decoder-only transformer:
//! - token embedding
//! - repeated RMSNorm -> causal self-attention -> residual
//! - repeated RMSNorm -> SwiGLU FFN -> residual
//! - final RMSNorm
//! - classifier head
//!
//! The forward pass is CPU-based, and the backward pass computes real
//! gradients on CPU from cached activations so the training loop can actually
//! learn from data.

use std::ops::Range;

use rand::random;

use crate::data::Batch;
use crate::error::Result;
use crate::layers::transformer_backward::rmsnorm_backward;
use crate::training::{Model, TransformerConfig};
use crate::wrapper::ANETensor;

const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
struct LayerLayout {
    rms_att: Range<usize>,
    wq: Range<usize>,
    wk: Range<usize>,
    wv: Range<usize>,
    wo: Range<usize>,
    rms_ffn: Range<usize>,
    w1: Range<usize>,
    w3: Range<usize>,
    w2: Range<usize>,
}

#[derive(Clone, Debug)]
struct ParamLayout {
    embedding: Range<usize>,
    layers: Vec<LayerLayout>,
    final_norm: Range<usize>,
    classifier: Range<usize>,
}

#[derive(Clone, Debug, Default)]
struct LayerCache {
    x_attn_in: Vec<f32>,
    x_attn_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_probs: Vec<f32>,
    attn_out: Vec<f32>,
    x_ffn_in: Vec<f32>,
    x_ffn_norm: Vec<f32>,
    h1: Vec<f32>,
    silu: Vec<f32>,
    h3: Vec<f32>,
    ffn_hidden: Vec<f32>,
}

/// Forward activations cached for each sample in the most recent batch.
#[derive(Clone, Debug, Default)]
pub struct CachedActivations {
    samples: Vec<SampleCache>,
}

impl CachedActivations {
    fn new() -> Self {
        Self { samples: vec![] }
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}

#[derive(Clone, Debug, Default)]
struct SampleCache {
    layers: Vec<LayerCache>,
    final_in: Vec<f32>,
    final_norm: Vec<f32>,
}

/// CPU-backed transformer used for real training runs.
#[derive(Debug)]
pub struct TransformerANE {
    config: TransformerConfig,
    trainable_params: Vec<f32>,
    layout: ParamLayout,
    cached: CachedActivations,
    last_input_tokens: Vec<u32>,
    last_logits: Vec<f32>,
    last_batch_size: usize,
    last_seq_len: usize,
}

impl TransformerANE {
    /// Build a new trainable transformer from a validated configuration.
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        let layout = build_layout(config);
        let mut trainable_params = vec![0.0f32; config.param_count()];

        fill_embedding(&mut trainable_params[layout.embedding.clone()], config.vocab_size, config.dim);

        for layer in &layout.layers {
            fill_gamma(&mut trainable_params[layer.rms_att.clone()]);
            fill_linear(
                &mut trainable_params[layer.wq.clone()],
                config.dim,
                config.dim,
            );
            fill_linear(
                &mut trainable_params[layer.wk.clone()],
                config.dim,
                config.dim,
            );
            fill_linear(
                &mut trainable_params[layer.wv.clone()],
                config.dim,
                config.dim,
            );
            fill_linear(
                &mut trainable_params[layer.wo.clone()],
                config.dim,
                config.dim,
            );
            fill_gamma(&mut trainable_params[layer.rms_ffn.clone()]);
            fill_linear(
                &mut trainable_params[layer.w1.clone()],
                config.dim,
                config.hidden_dim,
            );
            fill_linear(
                &mut trainable_params[layer.w3.clone()],
                config.dim,
                config.hidden_dim,
            );
            fill_linear(
                &mut trainable_params[layer.w2.clone()],
                config.hidden_dim,
                config.dim,
            );
        }

        fill_gamma(&mut trainable_params[layout.final_norm.clone()]);
        fill_linear(
            &mut trainable_params[layout.classifier.clone()],
            config.dim,
            config.vocab_size,
        );

        Ok(Self {
            config: config.clone(),
            trainable_params,
            layout,
            cached: CachedActivations::new(),
            last_input_tokens: vec![],
            last_logits: vec![],
            last_batch_size: 0,
            last_seq_len: 0,
        })
    }

    /// Return the model configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    fn embedding(&self) -> &[f32] {
        &self.trainable_params[self.layout.embedding.clone()]
    }

    fn classifier(&self) -> &[f32] {
        &self.trainable_params[self.layout.classifier.clone()]
    }

    fn final_norm(&self) -> &[f32] {
        &self.trainable_params[self.layout.final_norm.clone()]
    }

    fn layer(&self, idx: usize) -> &LayerLayout {
        &self.layout.layers[idx]
    }

    fn validate_batch(&self, batch: &Batch) -> Result<()> {
        if batch.seq_len() < 2 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be at least 2 for next-token training".to_string(),
            ));
        }
        if batch.seq_len() > self.config.seq_len {
            return Err(crate::Error::InvalidParameter(format!(
                "batch seq_len {} exceeds configured max {}",
                batch.seq_len(),
                self.config.seq_len
            )));
        }
        if batch.tokens().len() != batch.batch_size() * batch.seq_len() {
            return Err(crate::Error::InvalidParameter(
                "token count mismatch".to_string(),
            ));
        }
        Ok(())
    }

    fn forward_sample(&self, tokens: &[u32]) -> Result<(Vec<f32>, SampleCache)> {
        let seq_len = tokens.len();
        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;

        let mut x = embedding_lookup(tokens, self.embedding(), dim, vocab_size)?;
        let mut sample_cache = SampleCache {
            layers: Vec::with_capacity(self.config.n_layers),
            final_in: Vec::new(),
            final_norm: Vec::new(),
        };

        for layer_idx in 0..self.config.n_layers {
            let layer = self.layer(layer_idx);
            let x_attn_in = x.clone();
            let x_attn_norm = rmsnorm_forward(&x_attn_in, &self.trainable_params[layer.rms_att.clone()], dim);

            let q = linear_forward(
                &x_attn_norm,
                dim,
                &self.trainable_params[layer.wq.clone()],
                dim,
            );
            let k = linear_forward(
                &x_attn_norm,
                dim,
                &self.trainable_params[layer.wk.clone()],
                dim,
            );
            let v = linear_forward(
                &x_attn_norm,
                dim,
                &self.trainable_params[layer.wv.clone()],
                dim,
            );

            let (attn_out, attn_probs) =
                causal_attention_forward(&q, &k, &v, seq_len, dim, n_heads, head_dim);

            let attn_proj_out = linear_forward(
                &attn_out,
                dim,
                &self.trainable_params[layer.wo.clone()],
                dim,
            );
            let x_ffn_in = add_residual(&x_attn_in, &attn_proj_out);
            let x_ffn_norm = rmsnorm_forward(&x_ffn_in, &self.trainable_params[layer.rms_ffn.clone()], dim);

            let h1 = linear_forward(
                &x_ffn_norm,
                dim,
                &self.trainable_params[layer.w1.clone()],
                self.config.hidden_dim,
            );
            let h3 = linear_forward(
                &x_ffn_norm,
                dim,
                &self.trainable_params[layer.w3.clone()],
                self.config.hidden_dim,
            );
            let silu = h1.iter().map(|&x| silu(x)).collect::<Vec<_>>();
            let ffn_hidden = elementwise_mul(&silu, &h3);
            let ffn_out = linear_forward(
                &ffn_hidden,
                self.config.hidden_dim,
                &self.trainable_params[layer.w2.clone()],
                dim,
            );

            x = add_residual(&x_ffn_in, &ffn_out);

            sample_cache.layers.push(LayerCache {
                x_attn_in,
                x_attn_norm,
                q,
                k,
                v,
                attn_probs,
                attn_out,
                x_ffn_in,
                x_ffn_norm,
                h1,
                silu,
                h3,
                ffn_hidden,
            });
        }

        let final_in = x;
        let final_norm = rmsnorm_forward(&final_in, self.final_norm(), dim);
        let logits = linear_forward(
            &final_norm[..(seq_len - 1) * dim],
            dim,
            self.classifier(),
            vocab_size,
        );

        sample_cache.final_in = final_in;
        sample_cache.final_norm = final_norm;

        Ok((logits, sample_cache))
    }

    fn backward_sample(
        &self,
        tokens: &[u32],
        cache: &SampleCache,
        d_logits: &[f32],
        grads: &mut [f32],
    ) -> Result<()> {
        let seq_len = tokens.len();
        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let hidden_dim = self.config.hidden_dim;
        let positions = seq_len - 1;
        let mut d_classifier = vec![0.0f32; self.layout.classifier.end - self.layout.classifier.start];
        let mut d_final_norm = vec![0.0f32; cache.final_norm.len()];

        let (d_final_norm_from_logits, d_classifier_from_logits) = linear_backward(
            &cache.final_norm[..positions * dim],
            d_logits,
            dim,
            vocab_size,
            self.classifier(),
        );
        d_final_norm[..positions * dim].copy_from_slice(&d_final_norm_from_logits);
        d_classifier.copy_from_slice(&d_classifier_from_logits);

        let (mut d_current, d_final_gamma) =
            rmsnorm_backward(&d_final_norm, &cache.final_in, self.final_norm());
        add_slice(grads, self.layout.final_norm.start, &d_final_gamma);

        add_slice(grads, self.layout.classifier.start, &d_classifier);

        for layer_idx in (0..self.config.n_layers).rev() {
            let layer = self.layer(layer_idx);
            let layer_cache = &cache.layers[layer_idx];

            let (d_x_ffn_norm_from_ffn, d_w1, d_w3, d_w2) = ffn_backward(
                &layer_cache.x_ffn_norm,
                &layer_cache.h1,
                &layer_cache.silu,
                &layer_cache.h3,
                &layer_cache.ffn_hidden,
                &d_current,
                dim,
                hidden_dim,
                &self.trainable_params[layer.w1.clone()],
                &self.trainable_params[layer.w3.clone()],
                &self.trainable_params[layer.w2.clone()],
            );

            add_slice(grads, layer.w1.start, &d_w1);
            add_slice(grads, layer.w3.start, &d_w3);
            add_slice(grads, layer.w2.start, &d_w2);

            let (d_x_ffn_in_from_norm, d_ffn_gamma) =
                rmsnorm_backward(&d_x_ffn_norm_from_ffn, &layer_cache.x_ffn_in, &self.trainable_params[layer.rms_ffn.clone()]);
            add_slice(grads, layer.rms_ffn.start, &d_ffn_gamma);

            let d_x_attn_input = add_residual(&d_current, &d_x_ffn_in_from_norm);

            let (d_x_attn_norm_from_attn, d_wq, d_wk, d_wv, d_wo) = causal_attention_backward(
                &layer_cache.x_attn_norm,
                &layer_cache.q,
                &layer_cache.k,
                &layer_cache.v,
                &layer_cache.attn_probs,
                &layer_cache.attn_out,
                &d_x_attn_input,
                dim,
                n_heads,
                head_dim,
                &self.trainable_params[layer.wq.clone()],
                &self.trainable_params[layer.wk.clone()],
                &self.trainable_params[layer.wv.clone()],
                &self.trainable_params[layer.wo.clone()],
            );

            add_slice(grads, layer.wq.start, &d_wq);
            add_slice(grads, layer.wk.start, &d_wk);
            add_slice(grads, layer.wv.start, &d_wv);
            add_slice(grads, layer.wo.start, &d_wo);

            let (d_x_attn_in_from_norm, d_attn_gamma) =
                rmsnorm_backward(&d_x_attn_norm_from_attn, &layer_cache.x_attn_in, &self.trainable_params[layer.rms_att.clone()]);
            add_slice(grads, layer.rms_att.start, &d_attn_gamma);

            d_current = add_residual(&d_x_attn_input, &d_x_attn_in_from_norm);
        }

        let d_embedding = embedding_backward(tokens, &d_current, dim, vocab_size)?;
        add_slice(grads, self.layout.embedding.start, &d_embedding);

        Ok(())
    }

    fn param_count(&self) -> usize {
        self.config.param_count()
    }
}

impl Model for TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        self.validate_batch(batch)?;

        self.cached.clear();
        self.last_input_tokens = batch.tokens().to_vec();
        self.last_batch_size = batch.batch_size();
        self.last_seq_len = batch.seq_len();

        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let vocab_size = self.config.vocab_size;
        let mut logits = Vec::with_capacity(batch_size * (seq_len - 1) * vocab_size);

        for sample_idx in 0..batch_size {
            let start = sample_idx * seq_len;
            let end = start + seq_len;
            let (sample_logits, sample_cache) = self.forward_sample(&batch.tokens()[start..end])?;
            logits.extend_from_slice(&sample_logits);
            self.cached.samples.push(sample_cache);
        }

        self.last_logits = logits.clone();
        ANETensor::from_fp32(logits, vec![batch_size, seq_len - 1, vocab_size])
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; self.param_count()])
    }

    fn backward_with_batch(&mut self, batch: &Batch, _loss: f32) -> Result<Vec<f32>> {
        if self.cached.samples.is_empty() {
            return Err(crate::Error::Other(
                "forward cache missing; call forward before backward".to_string(),
            ));
        }
        if batch.tokens() != self.last_input_tokens.as_slice()
            || batch.batch_size() != self.last_batch_size
            || batch.seq_len() != self.last_seq_len
        {
            return Err(crate::Error::Other(
                "batch used for backward does not match cached forward batch".to_string(),
            ));
        }

        self.validate_batch(batch)?;

        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let positions = batch_size * (seq_len - 1);
        let vocab_size = self.config.vocab_size;
        if self.last_logits.len() != positions * vocab_size {
            return Err(crate::Error::Other(
                "cached logits shape does not match expected training layout".to_string(),
            ));
        }

        let mut grads = vec![0.0f32; self.param_count()];
        let normalizer = positions as f32;

        for sample_idx in 0..batch_size {
            let start = sample_idx * seq_len;
            let end = start + seq_len;
            let sample_tokens = &batch.tokens()[start..end];
            let sample_logits = &self.last_logits[sample_idx * (seq_len - 1) * vocab_size
                ..(sample_idx + 1) * (seq_len - 1) * vocab_size];
            let (d_logits, _loss) = softmax_cross_entropy_backward(
                sample_logits,
                sample_tokens,
                vocab_size,
                normalizer,
            )?;
            self.backward_sample(sample_tokens, &self.cached.samples[sample_idx], &d_logits, &mut grads)?;
        }

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.trainable_params
    }

    fn param_count(&self) -> usize {
        self.config.param_count()
    }

    fn backward_on_ane(
        &mut self,
        batch: &Batch,
        loss: f32,
        accumulator: &mut crate::training::ANEGradientAccumulator,
    ) -> Result<()> {
        // Phase 3: ANE-accelerated backward pass
        //
        // For now, use CPU backward and transfer to accumulator (default implementation).
        // Future Phase 3c implementation will:
        // 1. Compile backward MIL kernels for each layer
        // 2. Execute kernels on ANE using cached activations
        // 3. Accumulate gradients directly in ANE memory
        // 4. Transfer final accumulated gradients to CPU once per step
        //
        // Current behavior:
        // - Compute gradients on CPU using backward_with_batch()
        // - Scale and accumulate in ANEGradientAccumulator
        // - Return success when accumulation complete

        // Validate forward cache exists
        if self.cached.samples.is_empty() {
            return Err(crate::Error::Other(
                "forward cache missing; call forward before backward_on_ane".to_string(),
            ));
        }

        // Validate batch matches cached forward
        if batch.tokens() != self.last_input_tokens.as_slice()
            || batch.batch_size() != self.last_batch_size
            || batch.seq_len() != self.last_seq_len
        {
            return Err(crate::Error::Other(
                "batch used for backward_on_ane does not match cached forward batch".to_string(),
            ));
        }

        // Compute gradients on CPU (Phase 2 implementation)
        let grads = self.backward_with_batch(batch, loss)?;

        // Accumulate in ANEGradientAccumulator
        // Scale by 1.0 since this is a single backward pass
        let scale = 1.0f32;
        accumulator.accumulate(&grads, scale)
            .map_err(|e| crate::Error::Other(format!("ANE gradient accumulation failed: {}", e)))?;

        Ok(())
    }
}

fn build_layout(config: &TransformerConfig) -> ParamLayout {
    let mut offset = 0;

    let embedding = offset..offset + config.vocab_size * config.dim;
    offset = embedding.end;

    let mut layers = Vec::with_capacity(config.n_layers);
    for _ in 0..config.n_layers {
        let rms_att = offset..offset + config.dim;
        offset = rms_att.end;
        let wq = offset..offset + config.dim * config.dim;
        offset = wq.end;
        let wk = offset..offset + config.dim * config.dim;
        offset = wk.end;
        let wv = offset..offset + config.dim * config.dim;
        offset = wv.end;
        let wo = offset..offset + config.dim * config.dim;
        offset = wo.end;
        let rms_ffn = offset..offset + config.dim;
        offset = rms_ffn.end;
        let w1 = offset..offset + config.dim * config.hidden_dim;
        offset = w1.end;
        let w3 = offset..offset + config.dim * config.hidden_dim;
        offset = w3.end;
        let w2 = offset..offset + config.hidden_dim * config.dim;
        offset = w2.end;

        layers.push(LayerLayout {
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w3,
            w2,
        });
    }

    let final_norm = offset..offset + config.dim;
    offset = final_norm.end;

    let classifier = offset..offset + config.dim * config.vocab_size;
    offset = classifier.end;

    debug_assert_eq!(offset, config.param_count());

    ParamLayout {
        embedding,
        layers,
        final_norm,
        classifier,
    }
}

fn fill_embedding(weights: &mut [f32], vocab_size: usize, dim: usize) {
    let scale = (1.0 / dim.max(1) as f32).sqrt();
    for token in 0..vocab_size {
        for d in 0..dim {
            weights[token * dim + d] = random_uniform(scale);
        }
    }
}

fn fill_gamma(weights: &mut [f32]) {
    for value in weights.iter_mut() {
        *value = 1.0;
    }
}

fn fill_linear(weights: &mut [f32], in_dim: usize, out_dim: usize) {
    let scale = xavier_bound(in_dim, out_dim);
    for value in weights.iter_mut() {
        *value = random_uniform(scale);
    }
}

fn random_uniform(scale: f32) -> f32 {
    (random::<f32>() * 2.0 - 1.0) * scale
}

fn xavier_bound(fan_in: usize, fan_out: usize) -> f32 {
    (6.0f32 / (fan_in + fan_out).max(1) as f32).sqrt()
}

fn embedding_lookup(tokens: &[u32], embedding: &[f32], dim: usize, vocab_size: usize) -> Result<Vec<f32>> {
    let mut x = vec![0.0f32; tokens.len() * dim];
    for (idx, &token) in tokens.iter().enumerate() {
        let token_idx = token as usize;
        if token_idx >= vocab_size {
            return Err(crate::Error::InvalidParameter(format!(
                "token id {} exceeds vocab size {}",
                token_idx, vocab_size
            )));
        }
        x[idx * dim..(idx + 1) * dim]
            .copy_from_slice(&embedding[token_idx * dim..(token_idx + 1) * dim]);
    }
    Ok(x)
}

fn embedding_backward(tokens: &[u32], d_x: &[f32], dim: usize, vocab_size: usize) -> Result<Vec<f32>> {
    let mut d_embedding = vec![0.0f32; vocab_size * dim];
    for (idx, &token) in tokens.iter().enumerate() {
        let token_idx = token as usize;
        if token_idx >= vocab_size {
            return Err(crate::Error::InvalidParameter(format!(
                "token id {} exceeds vocab size {}",
                token_idx, vocab_size
            )));
        }
        let grad_row = &d_x[idx * dim..(idx + 1) * dim];
        let emb_row = &mut d_embedding[token_idx * dim..(token_idx + 1) * dim];
        for d in 0..dim {
            emb_row[d] += grad_row[d];
        }
    }
    Ok(d_embedding)
}

fn rmsnorm_forward(input: &[f32], gamma: &[f32], dim: usize) -> Vec<f32> {
    let seq_len = input.len() / dim;
    let mut output = vec![0.0f32; input.len()];
    for pos in 0..seq_len {
        let x = &input[pos * dim..(pos + 1) * dim];
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_sq + EPS).sqrt();
        for i in 0..dim {
            output[pos * dim + i] = x[i] * inv_rms * gamma[i];
        }
    }
    output
}

fn linear_forward(input: &[f32], in_dim: usize, weight: &[f32], out_dim: usize) -> Vec<f32> {
    let seq_len = input.len() / in_dim;
    let mut output = vec![0.0f32; seq_len * out_dim];
    for pos in 0..seq_len {
        let x = &input[pos * in_dim..(pos + 1) * in_dim];
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            let w_row = &weight[o * in_dim..(o + 1) * in_dim];
            for i in 0..in_dim {
                sum += x[i] * w_row[i];
            }
            output[pos * out_dim + o] = sum;
        }
    }
    output
}

fn linear_backward(
    input: &[f32],
    d_out: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let seq_len = input.len() / in_dim;
    let mut d_input = vec![0.0f32; input.len()];
    let mut d_weight = vec![0.0f32; out_dim * in_dim];

    for pos in 0..seq_len {
        let x = &input[pos * in_dim..(pos + 1) * in_dim];
        let dy = &d_out[pos * out_dim..(pos + 1) * out_dim];
        for o in 0..out_dim {
            let grad = dy[o];
            let w_row = &weight[o * in_dim..(o + 1) * in_dim];
            let dw_row = &mut d_weight[o * in_dim..(o + 1) * in_dim];
            for i in 0..in_dim {
                dw_row[i] += grad * x[i];
                d_input[pos * in_dim + i] += grad * w_row[i];
            }
        }
    }

    (d_input, d_weight)
}

fn add_residual(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect()
}

fn add_slice(dst: &mut [f32], offset: usize, src: &[f32]) {
    for (idx, value) in src.iter().enumerate() {
        dst[offset + idx] += value;
    }
}

fn elementwise_mul(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_derivative(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

fn softmax_cross_entropy_backward(
    logits: &[f32],
    tokens: &[u32],
    vocab_size: usize,
    normalizer: f32,
) -> Result<(Vec<f32>, f32)> {
    let positions = tokens.len().saturating_sub(1);
    if logits.len() != positions * vocab_size {
        return Err(crate::Error::Other(
            "logits shape does not match token layout".to_string(),
        ));
    }
    if positions == 0 {
        return Err(crate::Error::InvalidParameter(
            "need at least two tokens for next-token training".to_string(),
        ));
    }

    let mut grads = vec![0.0f32; logits.len()];
    let mut total_loss = 0.0f32;
    for pos in 0..positions {
        let row = &logits[pos * vocab_size..(pos + 1) * vocab_size];
        let target = tokens[pos + 1] as usize;
        if target >= vocab_size {
            return Err(crate::Error::InvalidParameter(format!(
                "target token {} exceeds vocab size {}",
                target, vocab_size
            )));
        }

        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0f32; vocab_size];
        let mut sum = 0.0f32;
        for i in 0..vocab_size {
            exps[i] = (row[i] - max_logit).exp();
            sum += exps[i];
        }
        let prob = exps[target] / sum;
        total_loss += if prob > 0.0 { -prob.ln() } else { 10.0 };

        for i in 0..vocab_size {
            let mut grad = exps[i] / sum;
            if i == target {
                grad -= 1.0;
            }
            grads[pos * vocab_size + i] = grad / normalizer;
        }
    }

    Ok((grads, total_loss / positions as f32))
}

fn causal_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut attn_out = vec![0.0f32; seq_len * dim];
    let mut attn_probs = vec![0.0f32; n_heads * seq_len * seq_len];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for head in 0..n_heads {
        let head_offset = head * head_dim;
        for t in 0..seq_len {
            let mut scores = vec![0.0f32; t + 1];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=t {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[t * dim + head_offset + d] * k[j * dim + head_offset + d];
                }
                let score = dot * scale;
                scores[j] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }

            for j in 0..=t {
                let prob = scores[j] / sum;
                attn_probs[(head * seq_len + t) * seq_len + j] = prob;
                for d in 0..head_dim {
                    attn_out[t * dim + head_offset + d] +=
                        prob * v[j * dim + head_offset + d];
                }
            }
        }
    }

    (attn_out, attn_probs)
}

fn causal_attention_backward(
    x_attn_norm: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_probs: &[f32],
    attn_out: &[f32],
    d_attn_proj_out: &[f32],
    dim: usize,
    n_heads: usize,
    head_dim: usize,
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let seq_len = x_attn_norm.len() / dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (d_attn_out, d_wo) = linear_backward(attn_out, d_attn_proj_out, dim, dim, wo);

    let mut d_q = vec![0.0f32; q.len()];
    let mut d_k = vec![0.0f32; k.len()];
    let mut d_v = vec![0.0f32; v.len()];

    for head in 0..n_heads {
        let head_offset = head * head_dim;
        for t in 0..seq_len {
            let prob_row = &attn_probs[(head * seq_len + t) * seq_len..(head * seq_len + t + 1) * seq_len];
            let d_out_row = &d_attn_out[t * dim + head_offset..t * dim + head_offset + head_dim];

            let mut d_scores = vec![0.0f32; t + 1];
            let mut dot_with_v = vec![0.0f32; t + 1];
            for j in 0..=t {
                let v_row = &v[j * dim + head_offset..j * dim + head_offset + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += d_out_row[d] * v_row[d];
                    d_v[j * dim + head_offset + d] += prob_row[j] * d_out_row[d];
                }
                dot_with_v[j] = dot;
            }

            let mut sum = 0.0f32;
            for j in 0..=t {
                sum += dot_with_v[j] * prob_row[j];
            }
            for j in 0..=t {
                d_scores[j] = prob_row[j] * (dot_with_v[j] - sum);
            }

            for j in 0..=t {
                for d in 0..head_dim {
                    d_q[t * dim + head_offset + d] +=
                        d_scores[j] * k[j * dim + head_offset + d] * scale;
                    d_k[j * dim + head_offset + d] +=
                        d_scores[j] * q[t * dim + head_offset + d] * scale;
                }
            }
        }
    }

    let (d_x_from_q, d_wq) = linear_backward(x_attn_norm, &d_q, dim, dim, wq);
    let (d_x_from_k, d_wk) = linear_backward(x_attn_norm, &d_k, dim, dim, wk);
    let (d_x_from_v, d_wv) = linear_backward(x_attn_norm, &d_v, dim, dim, wv);

    let mut d_x_attn_norm = vec![0.0f32; x_attn_norm.len()];
    for i in 0..d_x_attn_norm.len() {
        d_x_attn_norm[i] = d_x_from_q[i] + d_x_from_k[i] + d_x_from_v[i];
    }

    (d_x_attn_norm, d_wq, d_wk, d_wv, d_wo)
}

fn ffn_backward(
    x_ffn_norm: &[f32],
    h1: &[f32],
    silu_cache: &[f32],
    h3: &[f32],
    ffn_hidden: &[f32],
    d_ffn_out: &[f32],
    dim: usize,
    hidden_dim: usize,
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (d_hidden, d_w2) = linear_backward(ffn_hidden, d_ffn_out, hidden_dim, dim, w2);

    let mut d_h1 = vec![0.0f32; h1.len()];
    let mut d_h3 = vec![0.0f32; h3.len()];
    for i in 0..h1.len() {
        d_h3[i] = d_hidden[i] * silu_cache[i];
        d_h1[i] = d_hidden[i] * h3[i] * silu_derivative(h1[i]);
    }

    let (d_x_from_h1, d_w1) = linear_backward(x_ffn_norm, &d_h1, dim, hidden_dim, w1);
    let (d_x_from_h3, d_w3) = linear_backward(x_ffn_norm, &d_h3, dim, hidden_dim, w3);

    let mut d_x_ffn_norm = vec![0.0f32; x_ffn_norm.len()];
    for i in 0..d_x_ffn_norm.len() {
        d_x_ffn_norm[i] = d_x_from_h1[i] + d_x_from_h3[i];
    }

    (d_x_ffn_norm, d_w1, d_w3, d_w2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Batch;

    #[test]
    fn test_transformer_ane_creation() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let model = TransformerANE::new(&config).unwrap();
        assert_eq!(model.param_count(), config.param_count());
    }

    #[test]
    fn test_transformer_ane_param_layout() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let model = TransformerANE::new(&config).unwrap();
        assert_eq!(model.trainable_params.len(), config.param_count());
        assert_eq!(model.layout.layers.len(), 2);
        assert_eq!(model.layout.embedding.len(), 256 * 128);
        assert_eq!(model.layout.classifier.len(), 128 * 256);
        assert_eq!(model.layout.final_norm.len(), 128);
    }

    #[test]
    fn test_transformer_ane_forward_small_batch() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let tokens = vec![0u32; 2 * 64];
        let batch = Batch::new(tokens, 2, 64).unwrap();
        let tensor = model.forward(&batch).unwrap();
        assert_eq!(tensor.num_elements(), 2 * (64 - 1) * 256);
    }

    #[test]
    fn test_transformer_ane_backward_nonzero() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();
        let _ = model.forward(&batch).unwrap();
        let grads = model.backward_with_batch(&batch, 0.5).unwrap();
        assert_eq!(grads.len(), config.param_count());
        assert!(grads.iter().any(|g| *g != 0.0));
    }

    #[test]
    fn test_parameters_mutation_affects_output() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 1 * 64], 1, 64).unwrap();
        let before = model.forward(&batch).unwrap().as_bytes().to_vec();
        model.parameters()[config.dim] += 0.5;
        let after = model.forward(&batch).unwrap().as_bytes().to_vec();
        assert_ne!(before, after);
    }
}
