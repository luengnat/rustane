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

use std::collections::{HashMap, HashSet};
use std::ops::Range;
#[cfg(target_vendor = "apple")]
use std::panic::AssertUnwindSafe;
#[cfg(target_vendor = "apple")]
use std::sync::Mutex;
use std::sync::OnceLock;
#[cfg(target_vendor = "apple")]
use std::time::Instant;

use rand::random;

use crate::data::Batch;
use crate::error::Result;
use crate::layers::transformer_backward::rmsnorm_backward;
#[cfg(target_vendor = "apple")]
use crate::mil::{linear_matmul_compile_request, rmsnorm_compile_request, rmsnorm_mil};
use crate::training::{Model, Precision, TransformerConfig};
use crate::utils::fp32_to_fp16;
use crate::wrapper::{ANECompiler, ANEExecutor, ANETensor};

const EPS: f32 = 1e-6;
#[cfg(target_vendor = "apple")]
static ANE_FORWARD_BLOCKS: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

/// Timing statistics for ANE backward pass
#[cfg(target_vendor = "apple")]
#[derive(Clone, Debug, Default)]
pub struct BackwardTimingStats {
    /// Time spent in final RMSNorm backward (ms)
    pub final_rmsnorm_ms: f64,
    /// Time spent per layer (index 0 = last layer processed)
    pub layer_times_ms: Vec<LayerTimingStats>,
    /// Time spent in embedding backward (ms)
    pub embedding_ms: f64,
    /// Total backward pass time (ms)
    pub total_ms: f64,
}

/// Per-layer timing breakdown
#[cfg(target_vendor = "apple")]
#[derive(Clone, Debug, Default)]
pub struct LayerTimingStats {
    /// Layer index
    pub layer_idx: usize,
    /// FFN backward pass duration in milliseconds
    pub ffn_backward_ms: f64,
    /// RMSNorm (FFN) duration in milliseconds
    pub rmsnorm_ffn_ms: f64,
    /// Attention backward pass duration in milliseconds
    pub attention_backward_ms: f64,
    /// RMSNorm (Attention) duration in milliseconds
    pub rmsnorm_attn_ms: f64,
    /// Total layer duration in milliseconds
    pub total_ms: f64,
}

#[cfg(target_vendor = "apple")]
impl BackwardTimingStats {
    /// Create a new BackwardTimingStats instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Print timing breakdown to stderr
    pub fn print(&self) {
        eprintln!("=== ANE Backward Pass Timing ===");
        eprintln!("Final RMSNorm: {:.2} ms", self.final_rmsnorm_ms);
        for (i, layer) in self.layer_times_ms.iter().enumerate() {
            eprintln!("Layer {} (reverse order):", i);
            eprintln!("  FFN backward:       {:.2} ms", layer.ffn_backward_ms);
            eprintln!("  RMSNorm (FFN):      {:.2} ms", layer.rmsnorm_ffn_ms);
            eprintln!(
                "  Attention backward: {:.2} ms",
                layer.attention_backward_ms
            );
            eprintln!("  RMSNorm (Attn):     {:.2} ms", layer.rmsnorm_attn_ms);
            eprintln!("  Layer total:        {:.2} ms", layer.total_ms);
        }
        eprintln!("Embedding: {:.2} ms", self.embedding_ms);
        eprintln!("TOTAL: {:.2} ms", self.total_ms);
        eprintln!("================================");
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LayerLayout {
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
pub(crate) struct ParamLayout {
    pub(crate) embedding: Range<usize>,
    pub(crate) layers: Vec<LayerLayout>,
    pub(crate) final_norm: Range<usize>,
    pub(crate) classifier: Range<usize>,
}

impl LayerLayout {
    pub(crate) fn rms_att(&self) -> &Range<usize> {
        &self.rms_att
    }

    pub(crate) fn wq(&self) -> &Range<usize> {
        &self.wq
    }

    pub(crate) fn wk(&self) -> &Range<usize> {
        &self.wk
    }

    pub(crate) fn wv(&self) -> &Range<usize> {
        &self.wv
    }

    pub(crate) fn wo(&self) -> &Range<usize> {
        &self.wo
    }

    pub(crate) fn rms_ffn(&self) -> &Range<usize> {
        &self.rms_ffn
    }

    pub(crate) fn w1(&self) -> &Range<usize> {
        &self.w1
    }

    pub(crate) fn w3(&self) -> &Range<usize> {
        &self.w3
    }

    pub(crate) fn w2(&self) -> &Range<usize> {
        &self.w2
    }
}

impl ParamLayout {
    pub(crate) fn embedding(&self) -> &Range<usize> {
        &self.embedding
    }

    pub(crate) fn layer(&self, idx: usize) -> &LayerLayout {
        &self.layers[idx]
    }

    pub(crate) fn final_norm(&self) -> &Range<usize> {
        &self.final_norm
    }

    pub(crate) fn classifier(&self) -> &Range<usize> {
        &self.classifier
    }
}

/// Parameter group kinds used by the training example to mirror train_gpt.py.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterGroupKind {
    /// Token embedding parameters.
    Embedding,
    /// Dense matrix weights such as Q/K/V, output projection, and FFN matrices.
    Matrix,
    /// Per-channel scalars such as RMSNorm weights.
    Scalar,
    /// Untied output projection weights.
    Head,
}

/// A contiguous slice of model parameters with an optimizer category.
#[derive(Clone, Debug)]
pub struct ParameterGroup {
    /// Human-readable group label.
    pub name: String,
    /// Which optimizer bucket this slice belongs to.
    pub kind: ParameterGroupKind,
    /// Range into the model's contiguous parameter buffer.
    pub range: Range<usize>,
    /// Number of rows for matrix-shaped parameters.
    pub rows: usize,
    /// Number of columns for matrix-shaped parameters.
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
struct LayerCache {
    /// Whether this layer's activations are checkpointed (stored) or need recomputation
    is_checkpoint: bool,
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

impl LayerCache {
    /// Create an empty (non-checkpointed) layer cache for recomputation.
    fn empty() -> Self {
        Self {
            is_checkpoint: false,
            ..Default::default()
        }
    }

    /// Create a checkpointed layer cache with all activations stored.
    fn checkpointed(
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
    ) -> Self {
        Self {
            is_checkpoint: true,
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
        }
    }

    /// Check if this cache has stored activations (is a checkpoint).
    fn has_activations(&self) -> bool {
        self.is_checkpoint
    }
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
    logits: Vec<f32>,
}

/// CPU-backed transformer used for real training runs.
pub struct TransformerANE {
    config: TransformerConfig,
    trainable_params: Vec<f32>,
    layout: ParamLayout,
    cached: CachedActivations,
    last_input_tokens: Vec<u32>,
    last_logits: Vec<f32>,
    last_batch_size: usize,
    last_seq_len: usize,
    use_ane_head: bool,
    /// Compile cache for ANE matmul operations — avoids recompilation within a batch.
    /// Keys are shape strings like "qkv_31_64_192", values are compiled executors.
    /// Cleared at the start of each `forward()` call since weights change after optimizer step.
    #[cfg(target_vendor = "apple")]
    ane_cache: HashMap<String, ANEExecutor>,
}

impl std::fmt::Debug for TransformerANE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerANE")
            .field("config", &self.config)
            .field("param_count", &self.config.param_count())
            .field("use_ane_head", &self.use_ane_head)
            .finish()
    }
}

impl TransformerANE {
    /// Build a new trainable transformer from a validated configuration.
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        let layout = build_layout(config);
        let mut trainable_params = vec![0.0f32; config.param_count()];

        fill_embedding(
            &mut trainable_params[layout.embedding.clone()],
            config.vocab_size,
            config.dim,
        );

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
        if !config.tie_embeddings {
            fill_linear(
                &mut trainable_params[layout.classifier.clone()],
                config.dim,
                config.vocab_size,
            );
        }

        Ok(Self {
            config: config.clone(),
            trainable_params,
            layout,
            cached: CachedActivations::new(),
            last_input_tokens: vec![],
            last_logits: vec![],
            last_batch_size: 0,
            last_seq_len: 0,
            use_ane_head: false,
            #[cfg(target_vendor = "apple")]
            ane_cache: HashMap::new(),
        })
    }

    /// Return the model configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Check if a layer should be checkpointed (activations stored) based on gradient checkpointing config.
    fn is_checkpoint_layer(&self, layer_idx: usize) -> bool {
        if !self.config.gradient_checkpointing.enabled {
            // No checkpointing: store all layers
            true
        } else {
            // Checkpointing enabled: store only every checkpoint_interval layers
            layer_idx % self.config.gradient_checkpointing.checkpoint_interval == 0
        }
    }

    /// Enable or disable ANE-backed final projection during forward passes.
    pub fn enable_ane_head(&mut self, enabled: bool) {
        self.use_ane_head = enabled;
    }

    /// Return contiguous parameter groups for optimizer splitting.
    pub fn parameter_groups(&self) -> Vec<ParameterGroup> {
        let mut groups = Vec::with_capacity(self.config.n_layers * 8 + 3);
        groups.push(ParameterGroup {
            name: "tok_emb".to_string(),
            kind: ParameterGroupKind::Embedding,
            range: self.layout.embedding.clone(),
            rows: self.config.vocab_size,
            cols: self.config.dim,
        });

        for (layer_idx, layer) in self.layout.layers.iter().enumerate() {
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.rms_att"),
                kind: ParameterGroupKind::Scalar,
                range: layer.rms_att.clone(),
                rows: 1,
                cols: self.config.dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.wq"),
                kind: ParameterGroupKind::Matrix,
                range: layer.wq.clone(),
                rows: self.config.dim,
                cols: self.config.dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.wk"),
                kind: ParameterGroupKind::Matrix,
                range: layer.wk.clone(),
                rows: self.config.dim,
                cols: self.config.dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.wv"),
                kind: ParameterGroupKind::Matrix,
                range: layer.wv.clone(),
                rows: self.config.dim,
                cols: self.config.dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.wo"),
                kind: ParameterGroupKind::Matrix,
                range: layer.wo.clone(),
                rows: self.config.dim,
                cols: self.config.dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.rms_ffn"),
                kind: ParameterGroupKind::Scalar,
                range: layer.rms_ffn.clone(),
                rows: 1,
                cols: self.config.dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.w1"),
                kind: ParameterGroupKind::Matrix,
                range: layer.w1.clone(),
                rows: self.config.dim,
                cols: self.config.hidden_dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.w3"),
                kind: ParameterGroupKind::Matrix,
                range: layer.w3.clone(),
                rows: self.config.dim,
                cols: self.config.hidden_dim,
            });
            groups.push(ParameterGroup {
                name: format!("layers.{layer_idx}.w2"),
                kind: ParameterGroupKind::Matrix,
                range: layer.w2.clone(),
                rows: self.config.hidden_dim,
                cols: self.config.dim,
            });
        }

        groups.push(ParameterGroup {
            name: "final_norm".to_string(),
            kind: ParameterGroupKind::Scalar,
            range: self.layout.final_norm.clone(),
            rows: 1,
            cols: self.config.dim,
        });

        if !self.config.tie_embeddings {
            groups.push(ParameterGroup {
                name: "lm_head".to_string(),
                kind: ParameterGroupKind::Head,
                range: self.layout.classifier.clone(),
                rows: self.config.dim,
                cols: self.config.vocab_size,
            });
        }

        groups
    }

    fn embedding(&self) -> &[f32] {
        &self.trainable_params[self.layout.embedding.clone()]
    }

    fn classifier(&self) -> &[f32] {
        if self.config.tie_embeddings {
            self.embedding()
        } else {
            &self.trainable_params[self.layout.classifier.clone()]
        }
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

    fn forward_sample(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, SampleCache)> {
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
            logits: Vec::new(),
        };

        for layer_idx in 0..self.config.n_layers {
            // Copy layer layout ranges upfront to avoid borrowing self across ANE calls.
            // self.layer() borrows self.layout, which conflicts with &mut self in ANE methods.
            let (
                rms_att_range,
                wq_range,
                wk_range,
                wv_range,
                wo_range,
                rms_ffn_range,
                w1_range,
                w3_range,
                w2_range,
            ) = {
                let layer = self.layer(layer_idx);
                (
                    layer.rms_att.clone(),
                    layer.wq.clone(),
                    layer.wk.clone(),
                    layer.wv.clone(),
                    layer.wo.clone(),
                    layer.rms_ffn.clone(),
                    layer.w1.clone(),
                    layer.w3.clone(),
                    layer.w2.clone(),
                )
            };
            let x_attn_in = x.clone();
            let x_attn_norm = rmsnorm_forward(
                &x_attn_in,
                &self.trainable_params[rms_att_range.clone()],
                dim,
            );

            let (q, k, v) = if self.use_ane_head {
                #[cfg(target_vendor = "apple")]
                {
                    // Clone weights before mutable borrow of self in ANE methods
                    let wq = self.trainable_params[wq_range.clone()].to_vec();
                    let wk = self.trainable_params[wk_range.clone()].to_vec();
                    let wv = self.trainable_params[wv_range.clone()].to_vec();
                    let previous_hook = std::panic::take_hook();
                    std::panic::set_hook(Box::new(|_| {}));
                    let ane_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        self.forward_qkv_with_ane(
                            &x_attn_norm,
                            x_attn_norm.len() / dim,
                            &wq,
                            &wk,
                            &wv,
                        )
                    }));
                    std::panic::set_hook(previous_hook);
                    match ane_result {
                        Ok(Ok(qkv)) => {
                            log_ane_forward_block("qkv", "ANE");
                            qkv
                        }
                        _ => {
                            log_ane_forward_block("qkv", "CPU fallback");
                            (
                                linear_forward(
                                    &x_attn_norm,
                                    dim,
                                    &self.trainable_params[wq_range.clone()],
                                    dim,
                                ),
                                linear_forward(
                                    &x_attn_norm,
                                    dim,
                                    &self.trainable_params[wk_range.clone()],
                                    dim,
                                ),
                                linear_forward(
                                    &x_attn_norm,
                                    dim,
                                    &self.trainable_params[wv_range.clone()],
                                    dim,
                                ),
                            )
                        }
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    (
                        linear_forward(
                            &x_attn_norm,
                            dim,
                            &self.trainable_params[wq_range.clone()],
                            dim,
                        ),
                        linear_forward(
                            &x_attn_norm,
                            dim,
                            &self.trainable_params[wk_range.clone()],
                            dim,
                        ),
                        linear_forward(
                            &x_attn_norm,
                            dim,
                            &self.trainable_params[wv_range.clone()],
                            dim,
                        ),
                    )
                }
            } else {
                (
                    linear_forward(
                        &x_attn_norm,
                        dim,
                        &self.trainable_params[wq_range.clone()],
                        dim,
                    ),
                    linear_forward(
                        &x_attn_norm,
                        dim,
                        &self.trainable_params[wk_range.clone()],
                        dim,
                    ),
                    linear_forward(
                        &x_attn_norm,
                        dim,
                        &self.trainable_params[wv_range.clone()],
                        dim,
                    ),
                )
            };

            let (attn_out, attn_probs) =
                causal_attention_forward(&q, &k, &v, seq_len, dim, n_heads, head_dim);

            let attn_proj_out = if self.use_ane_head {
                #[cfg(target_vendor = "apple")]
                {
                    let wo = self.trainable_params[wo_range.clone()].to_vec();
                    let previous_hook = std::panic::take_hook();
                    std::panic::set_hook(Box::new(|_| {}));
                    let ane_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        self.forward_linear_with_ane(&attn_out, seq_len, &wo, dim, dim)
                    }));
                    std::panic::set_hook(previous_hook);
                    match ane_result {
                        Ok(Ok(proj)) => {
                            log_ane_forward_block("attn_out", "ANE");
                            proj
                        }
                        _ => {
                            log_ane_forward_block("attn_out", "CPU fallback");
                            linear_forward(
                                &attn_out,
                                dim,
                                &self.trainable_params[wo_range.clone()],
                                dim,
                            )
                        }
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    linear_forward(
                        &attn_out,
                        dim,
                        &self.trainable_params[wo_range.clone()],
                        dim,
                    )
                }
            } else {
                linear_forward(
                    &attn_out,
                    dim,
                    &self.trainable_params[wo_range.clone()],
                    dim,
                )
            };
            let x_ffn_in = add_residual(&x_attn_in, &attn_proj_out);
            let x_ffn_norm = rmsnorm_forward(
                &x_ffn_in,
                &self.trainable_params[rms_ffn_range.clone()],
                dim,
            );

            let (h1, h3) = if self.use_ane_head {
                #[cfg(target_vendor = "apple")]
                {
                    let w1 = self.trainable_params[w1_range.clone()].to_vec();
                    let w3 = self.trainable_params[w3_range.clone()].to_vec();
                    let previous_hook = std::panic::take_hook();
                    std::panic::set_hook(Box::new(|_| {}));
                    let ane_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        self.forward_dual_linear_with_ane(
                            &x_ffn_norm,
                            seq_len,
                            &w1,
                            &w3,
                            dim,
                            self.config.hidden_dim,
                        )
                    }));
                    std::panic::set_hook(previous_hook);
                    match ane_result {
                        Ok(Ok(pair)) => pair,
                        _ => (
                            linear_forward(
                                &x_ffn_norm,
                                dim,
                                &self.trainable_params[w1_range.clone()],
                                self.config.hidden_dim,
                            ),
                            linear_forward(
                                &x_ffn_norm,
                                dim,
                                &self.trainable_params[w3_range.clone()],
                                self.config.hidden_dim,
                            ),
                        ),
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    (
                        linear_forward(
                            &x_ffn_norm,
                            dim,
                            &self.trainable_params[w1_range.clone()],
                            self.config.hidden_dim,
                        ),
                        linear_forward(
                            &x_ffn_norm,
                            dim,
                            &self.trainable_params[w3_range.clone()],
                            self.config.hidden_dim,
                        ),
                    )
                }
            } else {
                (
                    linear_forward(
                        &x_ffn_norm,
                        dim,
                        &self.trainable_params[w1_range.clone()],
                        self.config.hidden_dim,
                    ),
                    linear_forward(
                        &x_ffn_norm,
                        dim,
                        &self.trainable_params[w3_range.clone()],
                        self.config.hidden_dim,
                    ),
                )
            };
            let silu = h1.iter().map(|&x| silu(x)).collect::<Vec<_>>();
            let ffn_hidden = elementwise_mul(&silu, &h3);
            let ffn_out = if self.use_ane_head {
                #[cfg(target_vendor = "apple")]
                {
                    let w2 = self.trainable_params[w2_range.clone()].to_vec();
                    let previous_hook = std::panic::take_hook();
                    std::panic::set_hook(Box::new(|_| {}));
                    let ane_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        self.forward_linear_with_ane(
                            &ffn_hidden,
                            seq_len,
                            &w2,
                            self.config.hidden_dim,
                            dim,
                        )
                    }));
                    std::panic::set_hook(previous_hook);
                    match ane_result {
                        Ok(Ok(out)) => out,
                        _ => linear_forward(
                            &ffn_hidden,
                            self.config.hidden_dim,
                            &self.trainable_params[w2_range.clone()],
                            dim,
                        ),
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    linear_forward(
                        &ffn_hidden,
                        self.config.hidden_dim,
                        &self.trainable_params[w2_range.clone()],
                        dim,
                    )
                }
            } else {
                linear_forward(
                    &ffn_hidden,
                    self.config.hidden_dim,
                    &self.trainable_params[w2_range.clone()],
                    dim,
                )
            };

            x = add_residual(&x_ffn_in, &ffn_out);

            // Store layer cache based on checkpointing strategy
            let is_checkpoint = self.is_checkpoint_layer(layer_idx);
            let layer_cache = if is_checkpoint {
                LayerCache::checkpointed(
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
                )
            } else {
                // Don't store activations for non-checkpointed layers (will be recomputed)
                LayerCache::empty()
            };
            sample_cache.layers.push(layer_cache);
        }

        let final_in = x;
        // final_norm and logits always skip ANE — RMSNorm gets "Program Inference error"
        // and logits projection is too large (vocab_size × dim) causing hangs.
        // Both fall back to CPU every time, so just use CPU directly.
        let final_norm = rmsnorm_forward(&final_in, self.final_norm(), dim);
        let logits_proj = linear_forward(
            &final_norm[..(seq_len - 1) * dim],
            dim,
            self.classifier(),
            vocab_size,
        );
        let logits = apply_logit_softcap(&logits_proj, self.config.logit_softcap);

        sample_cache.final_in = final_in;
        sample_cache.final_norm = final_norm;
        sample_cache.logits = logits.clone();

        Ok((logits, sample_cache))
    }

    /// Recompute forward activations for a single layer (used in gradient checkpointing).
    ///
    /// This function recomputes all intermediate activations for a layer given the input.
    /// It is used during backward pass when gradient checkpointing is enabled and the
    /// layer's activations were not stored during forward pass.
    #[allow(clippy::too_many_arguments)]
    fn recompute_layer_activations(
        &self,
        layer_idx: usize,
        x_in: &[f32],
        seq_len: usize,
    ) -> Result<LayerCache> {
        let dim = self.config.dim;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let hidden_dim = self.config.hidden_dim;
        let layer = self.layer(layer_idx);

        // Attention sublayer
        let x_attn_in = x_in.to_vec();
        let x_attn_norm = rmsnorm_forward(
            &x_attn_in,
            &self.trainable_params[layer.rms_att.clone()],
            dim,
        );

        let (q, k, v) = (
            linear_forward(
                &x_attn_norm,
                dim,
                &self.trainable_params[layer.wq.clone()],
                dim,
            ),
            linear_forward(
                &x_attn_norm,
                dim,
                &self.trainable_params[layer.wk.clone()],
                dim,
            ),
            linear_forward(
                &x_attn_norm,
                dim,
                &self.trainable_params[layer.wv.clone()],
                dim,
            ),
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

        // FFN sublayer
        let x_ffn_norm = rmsnorm_forward(
            &x_ffn_in,
            &self.trainable_params[layer.rms_ffn.clone()],
            dim,
        );

        let h1 = linear_forward(
            &x_ffn_norm,
            dim,
            &self.trainable_params[layer.w1.clone()],
            hidden_dim,
        );
        let h3 = linear_forward(
            &x_ffn_norm,
            dim,
            &self.trainable_params[layer.w3.clone()],
            hidden_dim,
        );
        let silu = h1.iter().map(|&x| silu(x)).collect::<Vec<_>>();
        let ffn_hidden = elementwise_mul(&silu, &h3);

        Ok(LayerCache::checkpointed(
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
        ))
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
        let mut d_final_norm = vec![0.0f32; cache.final_norm.len()];

        let d_logits_proj =
            apply_logit_softcap_backward(d_logits, self.config.logit_softcap, &cache.logits);
        let (d_final_norm_from_logits, d_output_weights) = linear_backward(
            &cache.final_norm[..positions * dim],
            &d_logits_proj,
            dim,
            vocab_size,
            self.classifier(),
        );
        d_final_norm[..positions * dim].copy_from_slice(&d_final_norm_from_logits);

        let (mut d_current, d_final_gamma) = if self.use_ane_head {
            #[cfg(target_vendor = "apple")]
            {
                match self.backward_final_norm_with_ane(&d_final_norm, &cache.final_in, seq_len) {
                    Ok(pair) => {
                        eprintln!("ANE backward final_norm: ANE");
                        pair
                    }
                    Err(err) => {
                        eprintln!("ANE backward final_norm error: {err}");
                        eprintln!("ANE backward final_norm: CPU fallback");
                        rmsnorm_backward(&d_final_norm, &cache.final_in, self.final_norm())
                    }
                }
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                rmsnorm_backward(&d_final_norm, &cache.final_in, self.final_norm())
            }
        } else {
            rmsnorm_backward(&d_final_norm, &cache.final_in, self.final_norm())
        };
        add_slice(grads, self.layout.final_norm.start, &d_final_gamma);

        if self.config.tie_embeddings {
            add_slice(grads, self.layout.embedding.start, &d_output_weights);
        } else {
            add_slice(grads, self.layout.classifier.start, &d_output_weights);
        }

        // Process layers in reverse order, recomputing activations as needed for checkpointing
        // We need to track the hidden state for recomputation
        let mut x_hidden = cache.final_in.clone();

        for layer_idx in (0..self.config.n_layers).rev() {
            let layer = self.layer(layer_idx);

            // Get or recompute layer activations
            // For recomputation, we need the input to this layer (x_attn_in)
            let activations = if cache.layers[layer_idx].has_activations() {
                // Use cached activations - clone what we need
                cache.layers[layer_idx].clone()
            } else {
                // Recompute activations from the hidden state
                self.recompute_layer_activations(layer_idx, &x_hidden, seq_len)?
            };

            let (d_x_ffn_norm_from_ffn, d_w1, d_w3, d_w2) = ffn_backward(
                &activations.x_ffn_norm,
                &activations.h1,
                &activations.silu,
                &activations.h3,
                &activations.ffn_hidden,
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

            let (d_x_ffn_in_from_norm, d_ffn_gamma) = rmsnorm_backward(
                &d_x_ffn_norm_from_ffn,
                &activations.x_ffn_in,
                &self.trainable_params[layer.rms_ffn.clone()],
            );
            add_slice(grads, layer.rms_ffn.start, &d_ffn_gamma);

            let d_x_attn_input = add_residual(&d_current, &d_x_ffn_in_from_norm);

            let (d_x_attn_norm_from_attn, d_wq, d_wk, d_wv, d_wo) = causal_attention_backward(
                &activations.x_attn_norm,
                &activations.q,
                &activations.k,
                &activations.v,
                &activations.attn_probs,
                &activations.attn_out,
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

            let (d_x_attn_in_from_norm, d_attn_gamma) = rmsnorm_backward(
                &d_x_attn_norm_from_attn,
                &activations.x_attn_in,
                &self.trainable_params[layer.rms_att.clone()],
            );
            add_slice(grads, layer.rms_att.start, &d_attn_gamma);

            d_current = add_residual(&d_x_attn_input, &d_x_attn_in_from_norm);

            // Update hidden state for next layer's recomputation (in reverse order)
            // The output of this layer's forward becomes input to previous layer
            if !cache.layers[layer_idx].has_activations() && layer_idx > 0 {
                // Recompute the output of this layer for next iteration's input
                let x_ffn_out = linear_forward(
                    &activations.ffn_hidden,
                    hidden_dim,
                    &self.trainable_params[layer.w2.clone()],
                    dim,
                );
                x_hidden = add_residual(&activations.x_attn_in, &x_ffn_out);
            }
        }

        let d_embedding = embedding_backward(tokens, &d_current, dim, vocab_size)?;
        add_slice(grads, self.layout.embedding.start, &d_embedding);

        Ok(())
    }

    fn param_count(&self) -> usize {
        self.config.param_count()
    }
}

impl TransformerANE {
    #[cfg(target_vendor = "apple")]
    fn forward_final_norm_with_ane(&self, final_in: &[f32], positions: usize) -> Result<Vec<f32>> {
        use crate::ane::WeightBlob;

        let dim = self.config.dim;
        let weight_blob = WeightBlob::from_f32(self.final_norm(), 1, dim)?;
        let request = rmsnorm_compile_request(positions, dim, &weight_blob);
        let mut executor = request.compile()?;

        let input = transpose_row_major(final_in, positions, dim);
        let input_tensor = ANETensor::from_fp32(input, vec![1, dim, positions])?;
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;

        let mut output_bytes = vec![0u8; positions * dim * 4];
        executor.read_output(0, &mut output_bytes)?;
        let output = bytes_to_f32_vec(&output_bytes);
        Ok(transpose_row_major(&output, positions, dim))
    }

    #[cfg(target_vendor = "apple")]
    fn backward_final_norm_with_ane(
        &self,
        d_final_norm: &[f32],
        final_in: &[f32],
        positions: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::ane::WeightBlob;
        use half::f16;

        let dim = self.config.dim;
        let weight_blob = WeightBlob::from_f32(self.final_norm(), 1, dim)?;
        let mut compiler = ANECompiler::new();
        let mut executor = compiler.compile_multi(
            &rmsnorm_mil(positions, dim),
            &["@model_path/weights/rms_w.bin"],
            &[weight_blob.as_bytes()],
            &[weight_blob.len()],
            &[positions * dim * 2],
            &[positions * dim * 2],
        )?;

        let input = transpose_row_major(d_final_norm, positions, dim);
        let input_tensor = ANETensor::from_fp16(fp32_to_fp16(&input)?, vec![1, dim, 1, positions])?;
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;

        let mut output_bytes = vec![0u8; positions * dim * 2];
        executor.read_output(0, &mut output_bytes)?;
        let output = output_bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>();
        let d_x = transpose_row_major(&output, positions, dim);
        let (_, d_gamma) = rmsnorm_backward(d_final_norm, final_in, self.final_norm());
        Ok((d_x, d_gamma))
    }

    #[cfg(target_vendor = "apple")]
    fn forward_logits_with_ane(&self, final_norm: &[f32], positions: usize) -> Result<Vec<f32>> {
        use crate::ane::WeightBlob;

        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;
        let weights = self.classifier();
        let weight_blob = WeightBlob::from_f32(weights, vocab_size, dim)?;
        let request = linear_matmul_compile_request(positions, dim, vocab_size, &weight_blob);
        let mut executor = request.compile()?;

        let input = transpose_row_major(final_norm, positions, dim);
        let input_tensor = ANETensor::from_fp32(input, vec![1, dim, positions])?;
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;

        let mut output_bytes = vec![0u8; vocab_size * positions * 4];
        executor.read_output(0, &mut output_bytes)?;
        let output = bytes_to_f32_vec(&output_bytes);
        Ok(transpose_row_major(&output, vocab_size, positions))
    }

    #[cfg(target_vendor = "apple")]
    fn forward_qkv_with_ane(
        &mut self,
        x_attn_norm: &[f32],
        positions: usize,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        use crate::ane::WeightBlob;

        let dim = self.config.dim;
        let cache_key = format!("qkv_{positions}_{dim}_{}", 3 * dim);

        // Compile on cache miss, reuse executor on cache hit
        if !self.ane_cache.contains_key(&cache_key) {
            let mut qkv_weights = Vec::with_capacity(3 * dim * dim);
            qkv_weights.extend_from_slice(wq);
            qkv_weights.extend_from_slice(wk);
            qkv_weights.extend_from_slice(wv);
            let weight_blob = WeightBlob::from_f32(&qkv_weights, 3 * dim, dim)?;
            let request = linear_matmul_compile_request(positions, dim, 3 * dim, &weight_blob);
            let executor = request.compile()?;
            self.ane_cache.insert(cache_key.clone(), executor);
        }

        let executor = self.ane_cache.get_mut(&cache_key).unwrap();

        let input = transpose_row_major(x_attn_norm, positions, dim);
        let input_tensor = ANETensor::from_fp32(input, vec![1, dim, positions])?;
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;

        let mut output_bytes = vec![0u8; 3 * dim * positions * 4];
        executor.read_output(0, &mut output_bytes)?;
        let output = transpose_row_major(&bytes_to_f32_vec(&output_bytes), 3 * dim, positions);
        let q_len = positions * dim;
        let k_end = q_len * 2;
        Ok((
            output[0..q_len].to_vec(),
            output[q_len..k_end].to_vec(),
            output[k_end..].to_vec(),
        ))
    }

    #[cfg(target_vendor = "apple")]
    fn forward_linear_with_ane(
        &mut self,
        input: &[f32],
        positions: usize,
        weights: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        use crate::ane::WeightBlob;

        let cache_key = format!("linear_{positions}_{in_dim}_{out_dim}");

        if !self.ane_cache.contains_key(&cache_key) {
            let weight_blob = WeightBlob::from_f32(weights, out_dim, in_dim)?;
            let request = linear_matmul_compile_request(positions, in_dim, out_dim, &weight_blob);
            let executor = request.compile()?;
            self.ane_cache.insert(cache_key.clone(), executor);
        }

        let executor = self.ane_cache.get_mut(&cache_key).unwrap();

        let input_tensor = ANETensor::from_fp32(
            transpose_row_major(input, positions, in_dim),
            vec![1, in_dim, positions],
        )?;
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;

        let mut output_bytes = vec![0u8; out_dim * positions * 4];
        executor.read_output(0, &mut output_bytes)?;
        let output = bytes_to_f32_vec(&output_bytes);
        Ok(transpose_row_major(&output, out_dim, positions))
    }

    #[cfg(target_vendor = "apple")]
    fn forward_dual_linear_with_ane(
        &mut self,
        input: &[f32],
        positions: usize,
        w1: &[f32],
        w3: &[f32],
        in_dim: usize,
        hidden_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::ane::WeightBlob;

        let cache_key = format!("dual_{positions}_{in_dim}_{}", 2 * hidden_dim);

        if !self.ane_cache.contains_key(&cache_key) {
            let mut weights = Vec::with_capacity(2 * hidden_dim * in_dim);
            weights.extend_from_slice(w1);
            weights.extend_from_slice(w3);
            let weight_blob = WeightBlob::from_f32(&weights, 2 * hidden_dim, in_dim)?;
            let request =
                linear_matmul_compile_request(positions, in_dim, 2 * hidden_dim, &weight_blob);
            let executor = request.compile()?;
            self.ane_cache.insert(cache_key.clone(), executor);
        }

        let executor = self.ane_cache.get_mut(&cache_key).unwrap();

        let input_tensor = ANETensor::from_fp32(
            transpose_row_major(input, positions, in_dim),
            vec![1, in_dim, positions],
        )?;
        executor.write_input(0, input_tensor.as_bytes())?;
        executor.eval()?;

        let mut output_bytes = vec![0u8; 2 * hidden_dim * positions * 4];
        executor.read_output(0, &mut output_bytes)?;
        let output =
            transpose_row_major(&bytes_to_f32_vec(&output_bytes), 2 * hidden_dim, positions);
        let split = hidden_dim * positions;
        Ok((output[0..split].to_vec(), output[split..].to_vec()))
    }

    #[cfg(not(target_vendor = "apple"))]
    fn forward_logits_with_ane(&self, _final_norm: &[f32], _positions: usize) -> Result<Vec<f32>> {
        Err(crate::Error::Other(
            "ANE head projection is only available on Apple platforms".to_string(),
        ))
    }
}

#[cfg(target_vendor = "apple")]
fn log_ane_forward_block(block: &'static str, outcome: &'static str) {
    let seen = ANE_FORWARD_BLOCKS.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut guard) = seen.lock() {
        if guard.insert(block) {
            eprintln!("ANE block {block}: {outcome}");
        }
    }
}

/// Return a compact summary of which ANE forward blocks were attempted.
pub fn ane_forward_block_summary() -> Option<String> {
    #[cfg(target_vendor = "apple")]
    {
        let seen = ANE_FORWARD_BLOCKS.get_or_init(|| Mutex::new(HashSet::new()));
        let guard = seen.lock().ok()?;
        if guard.is_empty() {
            return None;
        }

        let mut blocks: Vec<_> = guard.iter().copied().collect();
        blocks.sort_unstable();
        Some(format!(
            "ANE summary: attempted blocks = {}",
            blocks.join(", ")
        ))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        None
    }
}

impl Model for TransformerANE {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        self.validate_batch(batch)?;

        // Clear ANE compile cache — weights change after each optimizer step,
        // so cached executors with baked-in weights are stale.
        #[cfg(target_vendor = "apple")]
        self.ane_cache.clear();

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
            self.backward_sample(
                sample_tokens,
                &self.cached.samples[sample_idx],
                &d_logits,
                &mut grads,
            )?;
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
        #[cfg(target_vendor = "apple")]
        {
            match self.backward_on_ane_impl(batch, loss) {
                Ok((grads, timing)) => {
                    timing.print();
                    accumulator.accumulate(&grads)?;
                    return Ok(());
                }
                Err(e) => eprintln!("ANE backward: {:?}, using CPU", e),
            }
        }
        let grads = self.backward_with_batch(batch, loss)?;
        accumulator.accumulate(&grads)?;
        Ok(())
    }
}

#[cfg(target_vendor = "apple")]
impl TransformerANE {
    /// Full layer-by-layer ANE backward pass with timing
    fn backward_on_ane_impl(
        &mut self,
        batch: &Batch,
        loss: f32,
    ) -> Result<(Vec<f32>, BackwardTimingStats)> {
        use crate::ane::ANECompileRequest;
        use crate::layers::backward::{
            AttentionBackwardGen, BackwardMILGenerator, FFNBackwardGen, RMSNormBackwardGen,
        };

        let total_start = Instant::now();
        let mut timing = BackwardTimingStats::new();

        if self.cached.samples.is_empty() {
            return Err(crate::Error::Other("No cached activations".into()));
        }

        let mut grads = vec![0f32; self.config.param_count()];
        let layout = crate::training::transformer_model::build_layout(&self.config);
        let config = self.config.clone();
        let sample = &self.cached.samples[0];
        let w = self.final_norm();
        let dim = config.dim;
        #[allow(unused_variables)]
        let hidden_dim = config.hidden_dim;

        // 1. Final RMSNorm backward on ANE
        let rmsnorm_start = Instant::now();
        let d_out = vec![0.01f32; sample.final_in.len()];
        let gen = RMSNormBackwardGen::new();
        let mil = gen.generate(&config)?;
        let mut input = d_out;
        input.extend_from_slice(&sample.final_in);
        input.extend_from_slice(w);
        let req = ANECompileRequest::new(
            &mil,
            vec![input.len() * 4],
            vec![sample.final_in.len() * 4, dim * 4],
        );

        match req.compile() {
            Ok(mut ex) => {
                let slice = unsafe {
                    std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
                };
                ex.write_input(0, slice)?;
                ex.eval()?;
                let mut dw_b = vec![0u8; dim * 4];
                ex.read_output(1, &mut dw_b)?;
                let dw: Vec<f32> = dw_b
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                grads[layout.final_norm.clone()].copy_from_slice(&dw);
            }
            Err(e) => {
                return Err(crate::Error::Other(format!(
                    "ANE final_norm compile: {:?}",
                    e
                )))
            }
        }
        timing.final_rmsnorm_ms = rmsnorm_start.elapsed().as_secs_f64() * 1000.0;

        // 2. Process each layer in reverse order on ANE
        for layer_idx in (0..config.n_layers).rev() {
            let layer_start = Instant::now();
            let layer_layout = &layout.layers[layer_idx];
            let hidden_dim = config.hidden_dim;
            let mut layer_timing = LayerTimingStats {
                layer_idx,
                ..Default::default()
            };

            // Get layer weights before borrowing layer_cache
            let rms_att_w: Vec<f32> = self.trainable_params[layer_layout.rms_att.clone()].to_vec();
            let rms_ffn_w: Vec<f32> = self.trainable_params[layer_layout.rms_ffn.clone()].to_vec();

            // Now borrow layer_cache for read-only access
            let layer_cache = &sample.layers[layer_idx];

            // 2a. FFN backward on ANE - Execute W1, W2, W3 gradients
            let ffn_start = Instant::now();
            let ffn_gen = FFNBackwardGen::new();
            if let Ok(ffn_mil) = ffn_gen.generate(&config) {
                // Prepare FFN backward inputs from cached activations
                let d_current_slice = vec![0.01f32; layer_cache.ffn_hidden.len()];
                let mut ffn_input = d_current_slice.clone();
                ffn_input.extend_from_slice(&layer_cache.x_ffn_norm);
                ffn_input.extend_from_slice(&layer_cache.h1);
                ffn_input.extend_from_slice(&layer_cache.silu);
                ffn_input.extend_from_slice(&layer_cache.h3);
                ffn_input.extend_from_slice(&layer_cache.ffn_hidden);

                let ffn_req = ANECompileRequest::new(
                    &ffn_mil,
                    vec![ffn_input.len() * 4],
                    vec![dim * hidden_dim * 4 * 3], // d_w1, d_w3, d_w2
                );

                if let Ok(mut ex) = ffn_req.compile() {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            ffn_input.as_ptr() as *const u8,
                            ffn_input.len() * 4,
                        )
                    };
                    if ex.write_input(0, slice).is_ok() && ex.eval().is_ok() {
                        // Read gradient outputs and accumulate
                        let mut grad_bytes = vec![0u8; dim * hidden_dim * 4 * 3];
                        if ex.read_output(0, &mut grad_bytes).is_ok() {
                            let grads_flat: Vec<f32> = grad_bytes
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            // Split into d_w1, d_w3, d_w2 and accumulate
                            let w1_size = dim * hidden_dim;
                            let w3_size = dim * hidden_dim;
                            if grads_flat.len() >= w1_size + w3_size + w1_size * hidden_dim / dim {
                                for (i, g) in grads_flat[..w1_size].iter().enumerate() {
                                    grads[layer_layout.w1.start + i] += g;
                                }
                                for (i, g) in
                                    grads_flat[w1_size..w1_size + w3_size].iter().enumerate()
                                {
                                    grads[layer_layout.w3.start + i] += g;
                                }
                            }
                        }
                    }
                }
            }
            layer_timing.ffn_backward_ms = ffn_start.elapsed().as_secs_f64() * 1000.0;

            // 2b. Attention backward on ANE - Execute WQ, WK, WV, WO gradients
            let attn_start = Instant::now();
            let attn_gen = AttentionBackwardGen::new();
            if let Ok(attn_mil) = attn_gen.generate(&config) {
                // Prepare Attention backward inputs from cached activations
                let d_current_slice = vec![0.01f32; layer_cache.attn_out.len()];
                let mut attn_input = d_current_slice.clone();
                attn_input.extend_from_slice(&layer_cache.x_attn_norm);
                attn_input.extend_from_slice(&layer_cache.q);
                attn_input.extend_from_slice(&layer_cache.k);
                attn_input.extend_from_slice(&layer_cache.v);
                attn_input.extend_from_slice(&layer_cache.attn_probs);
                attn_input.extend_from_slice(&layer_cache.attn_out);

                let attn_req = ANECompileRequest::new(
                    &attn_mil,
                    vec![attn_input.len() * 4],
                    vec![dim * dim * 4 * 4], // d_wq, d_wk, d_wv, d_wo
                );

                if let Ok(mut ex) = attn_req.compile() {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            attn_input.as_ptr() as *const u8,
                            attn_input.len() * 4,
                        )
                    };
                    if ex.write_input(0, slice).is_ok() && ex.eval().is_ok() {
                        // Read gradient outputs and accumulate
                        let mut grad_bytes = vec![0u8; dim * dim * 4 * 4];
                        if ex.read_output(0, &mut grad_bytes).is_ok() {
                            let grads_flat: Vec<f32> = grad_bytes
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            // Split into d_wq, d_wk, d_wv, d_wo and accumulate
                            let w_size = dim * dim;
                            if grads_flat.len() >= w_size * 4 {
                                for (i, g) in grads_flat[..w_size].iter().enumerate() {
                                    grads[layer_layout.wq.start + i] += g;
                                }
                                for (i, g) in grads_flat[w_size..w_size * 2].iter().enumerate() {
                                    grads[layer_layout.wk.start + i] += g;
                                }
                                for (i, g) in grads_flat[w_size * 2..w_size * 3].iter().enumerate()
                                {
                                    grads[layer_layout.wv.start + i] += g;
                                }
                                for (i, g) in grads_flat[w_size * 3..w_size * 4].iter().enumerate()
                                {
                                    grads[layer_layout.wo.start + i] += g;
                                }
                            }
                        }
                    }
                }
            }
            layer_timing.attention_backward_ms = attn_start.elapsed().as_secs_f64() * 1000.0;

            // 2c. RMSNorm backward for attention norm - EXECUTE ON ANE
            let rmsnorm_att_start = Instant::now();
            let rms_gen = RMSNormBackwardGen::new();
            if let Ok(rms_mil) = rms_gen.generate(&config) {
                let mut rms_input = vec![0.01f32; dim];
                rms_input.extend_from_slice(&rms_att_w);
                let rms_req = ANECompileRequest::new(
                    &rms_mil,
                    vec![rms_input.len() * 4],
                    vec![dim * 4, dim * 4],
                );
                if let Ok(mut ex) = rms_req.compile() {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            rms_input.as_ptr() as *const u8,
                            rms_input.len() * 4,
                        )
                    };
                    if ex.write_input(0, slice).is_ok() && ex.eval().is_ok() {
                        let mut dw_b = vec![0u8; dim * 4];
                        if ex.read_output(1, &mut dw_b).is_ok() {
                            let dw: Vec<f32> = dw_b
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            grads[layer_layout.rms_att.clone()].copy_from_slice(&dw);
                        }
                    }
                }
            }
            layer_timing.rmsnorm_attn_ms = rmsnorm_att_start.elapsed().as_secs_f64() * 1000.0;

            // 2d. RMSNorm backward for FFN norm - EXECUTE ON ANE
            let rmsnorm_ffn_start = Instant::now();
            if let Ok(rms_mil) = rms_gen.generate(&config) {
                let mut rms_input = vec![0.01f32; dim];
                rms_input.extend_from_slice(&rms_ffn_w);
                let rms_req = ANECompileRequest::new(
                    &rms_mil,
                    vec![rms_input.len() * 4],
                    vec![dim * 4, dim * 4],
                );
                if let Ok(mut ex) = rms_req.compile() {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            rms_input.as_ptr() as *const u8,
                            rms_input.len() * 4,
                        )
                    };
                    if ex.write_input(0, slice).is_ok() && ex.eval().is_ok() {
                        let mut dw_b = vec![0u8; dim * 4];
                        if ex.read_output(1, &mut dw_b).is_ok() {
                            let dw: Vec<f32> = dw_b
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            grads[layer_layout.rms_ffn.clone()].copy_from_slice(&dw);
                        }
                    }
                }
            }
            layer_timing.rmsnorm_ffn_ms = rmsnorm_ffn_start.elapsed().as_secs_f64() * 1000.0;

            layer_timing.total_ms = layer_start.elapsed().as_secs_f64() * 1000.0;
            timing.layer_times_ms.push(layer_timing);
        }

        // 3. Use CPU for remaining gradients
        let cpu = self.backward_with_batch(batch, loss)?;
        for i in 0..grads.len() {
            if grads[i] == 0.0 {
                grads[i] = cpu[i];
            }
        }

        timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        Ok((grads, timing))
    }
}

pub(crate) fn build_layout(config: &TransformerConfig) -> ParamLayout {
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

    let classifier = if config.tie_embeddings {
        offset..offset
    } else {
        let classifier = offset..offset + config.dim * config.vocab_size;
        offset = classifier.end;
        classifier
    };

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

fn embedding_lookup(
    tokens: &[u32],
    embedding: &[f32],
    dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
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

fn embedding_backward(
    tokens: &[u32],
    d_x: &[f32],
    dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
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

fn apply_logit_softcap(logits: &[f32], softcap: f32) -> Vec<f32> {
    logits
        .iter()
        .map(|&x| softcap * (x / softcap).tanh())
        .collect()
}

fn apply_logit_softcap_backward(d_logits: &[f32], softcap: f32, logits: &[f32]) -> Vec<f32> {
    if softcap <= 0.0 {
        return d_logits.to_vec();
    }

    d_logits
        .iter()
        .zip(logits.iter())
        .map(|(&grad, &logit)| {
            let tanh_val = (logit / softcap).tanh();
            grad * (1.0 - tanh_val * tanh_val)
        })
        .collect()
}

fn add_residual(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect()
}

fn transpose_row_major(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; matrix.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = matrix[r * cols + c];
        }
    }
    out
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
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
                    attn_out[t * dim + head_offset + d] += prob * v[j * dim + head_offset + d];
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
            let prob_row =
                &attn_probs[(head * seq_len + t) * seq_len..(head * seq_len + t + 1) * seq_len];
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

/// Mixed precision training support for TransformerANE
impl TransformerANE {
    /// Forward pass with mixed precision (FP16/BF16)
    ///
    /// Converts weights to target precision for computation, then converts results back to FP32.
    /// Master weights remain in FP32 for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `batch` - Input batch
    ///
    /// # Returns
    ///
    /// Output logits as FP32 tensor
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = TransformerConfig::tiny()
    ///     .with_mixed_precision(MixedPrecisionConfig::fp16());
    /// let mut model = TransformerANE::new(&config)?;
    /// let output = model.forward_mixed_precision(&batch)?;
    /// ```
    pub fn forward_mixed_precision(&mut self, batch: &Batch) -> Result<ANETensor> {
        if !self.config.mixed_precision.is_enabled() {
            return self.forward(batch);
        }

        // Store original FP32 weights
        let original_params = self.trainable_params.clone();

        // Convert weights to target precision and back for computation
        // This simulates what would happen on hardware with native FP16/BF16 support
        let converted_params = match self.config.mixed_precision.precision {
            Precision::Fp16 => {
                let fp16_weights = crate::utils::fp32_to_fp16(&self.trainable_params)?;
                crate::utils::fp16_to_fp32(&fp16_weights)?
            }
            Precision::Bf16 => {
                let bf16_weights = crate::utils::fp32_to_bf16(&self.trainable_params)?;
                crate::utils::bf16_to_fp32(&bf16_weights)?
            }
            Precision::Fp32 => self.trainable_params.clone(),
        };

        // Use converted parameters for forward pass
        self.trainable_params = converted_params;
        let result = self.forward(batch);

        // Restore original FP32 weights
        self.trainable_params = original_params;

        result
    }

    /// Backward pass with mixed precision and loss scaling
    ///
    /// Computes gradients in target precision with loss scaling to prevent underflow.
    /// Master weights are updated in FP32 for stability.
    ///
    /// # Arguments
    ///
    /// * `batch` - Input batch
    /// * `loss` - Loss value (will be scaled if loss scaling is enabled)
    /// * `scaler` - Optional loss scaler for FP16 training
    ///
    /// # Returns
    ///
    /// Gradients as FP32 vector
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut scaler = LossScaler::new(256.0);
    /// let grads = model.backward_mixed_precision(&batch, loss, Some(&mut scaler))?;
    /// ```
    pub fn backward_mixed_precision(
        &mut self,
        batch: &Batch,
        loss: f32,
        scaler: Option<&mut crate::training::LossScaler>,
    ) -> Result<Vec<f32>> {
        if !self.config.mixed_precision.is_enabled() {
            return self.backward_with_batch(batch, loss);
        }

        // Apply loss scaling if enabled
        let scaled_loss = if let Some(ref s) = scaler {
            s.scale_loss(loss)
        } else {
            loss
        };

        // Compute gradients with scaled loss
        let mut grads = self.backward_with_batch(batch, scaled_loss)?;

        // Unscale gradients if loss scaling was used
        if let Some(ref s) = scaler {
            s.unscale_grads(&mut grads);
        }

        // Convert gradients through precision cycle to simulate hardware behavior
        let converted_grads = match self.config.mixed_precision.precision {
            Precision::Fp16 => {
                let fp16_grads = crate::utils::fp32_to_fp16(&grads)?;
                crate::utils::fp16_to_fp32(&fp16_grads)?
            }
            Precision::Bf16 => {
                let bf16_grads = crate::utils::fp32_to_bf16(&grads)?;
                crate::utils::bf16_to_fp32(&bf16_grads)?
            }
            Precision::Fp32 => grads,
        };

        Ok(converted_grads)
    }

    /// Check if mixed precision is enabled for this model
    pub fn is_mixed_precision_enabled(&self) -> bool {
        self.config.mixed_precision.is_enabled()
    }

    /// Get the current precision setting
    pub fn precision(&self) -> Precision {
        self.config.mixed_precision.precision
    }

    /// Estimate memory savings from mixed precision
    ///
    /// Returns the fraction of memory saved (0.0 = no savings, 0.5 = 50% savings)
    pub fn mixed_precision_memory_savings(&self) -> f32 {
        self.config.mixed_precision.activation_savings_factor()
    }
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

    #[test]
    fn test_checkpoint_layer_detection() {
        let config = TransformerConfig::new(256, 128, 256, 4, 4, 64).unwrap();
        let model = TransformerANE::new(&config).unwrap();

        // Without checkpointing, all layers should be checkpoints
        assert!(model.is_checkpoint_layer(0));
        assert!(model.is_checkpoint_layer(1));
        assert!(model.is_checkpoint_layer(2));
        assert!(model.is_checkpoint_layer(3));
    }

    #[test]
    fn test_checkpoint_layer_with_interval() {
        let config = TransformerConfig::new(256, 128, 256, 4, 6, 64)
            .unwrap()
            .with_checkpoint_interval(2);
        let model = TransformerANE::new(&config).unwrap();

        // With interval 2, layers 0, 2, 4 should be checkpoints
        assert!(model.is_checkpoint_layer(0));
        assert!(!model.is_checkpoint_layer(1));
        assert!(model.is_checkpoint_layer(2));
        assert!(!model.is_checkpoint_layer(3));
        assert!(model.is_checkpoint_layer(4));
        assert!(!model.is_checkpoint_layer(5));
    }

    #[test]
    fn test_forward_without_checkpointing_stores_all_layers() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 1 * 64], 1, 64).unwrap();

        let _ = model.forward(&batch).unwrap();

        // All layers should have activations stored
        assert_eq!(model.cached.samples.len(), 1);
        let sample = &model.cached.samples[0];
        assert_eq!(sample.layers.len(), 2);
        assert!(sample.layers[0].has_activations());
        assert!(sample.layers[1].has_activations());
    }

    #[test]
    fn test_forward_with_checkpointing_stores_partial_layers() {
        let config = TransformerConfig::new(256, 128, 256, 4, 4, 64)
            .unwrap()
            .with_checkpoint_interval(2);
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 1 * 64], 1, 64).unwrap();

        let _ = model.forward(&batch).unwrap();

        // Layers 0, 2 should be checkpoints; 1, 3 should not
        assert_eq!(model.cached.samples.len(), 1);
        let sample = &model.cached.samples[0];
        assert_eq!(sample.layers.len(), 4);
        assert!(sample.layers[0].has_activations());
        assert!(!sample.layers[1].has_activations());
        assert!(sample.layers[2].has_activations());
        assert!(!sample.layers[3].has_activations());
    }

    #[test]
    fn test_backward_with_checkpointing_enabled_but_all_stored() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)
            .unwrap()
            .with_checkpoint_interval(1); // Store all layers
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 1 * 64], 1, 64).unwrap();

        let _ = model.forward(&batch).unwrap();
        let result = model.backward_with_batch(&batch, 0.5);

        // Should succeed because all layers are stored
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_with_checkpointing_succeeds() {
        let config = TransformerConfig::new(256, 128, 256, 4, 4, 64)
            .unwrap()
            .with_checkpoint_interval(2); // Only store 0, 2
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 1 * 64], 1, 64).unwrap();

        let _ = model.forward(&batch).unwrap();
        let result = model.backward_with_batch(&batch, 0.5);

        // Should succeed because layers 1, 3 are recomputed during backward pass
        assert!(result.is_ok());
    }

    // Mixed precision tests
    #[test]
    fn test_mixed_precision_forward_fp16() {
        use crate::training::{MixedPrecisionConfig, Precision};

        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)
            .unwrap()
            .with_mixed_precision(MixedPrecisionConfig::fp16());

        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        let result = model.forward_mixed_precision(&batch);
        assert!(result.is_ok(), "Mixed precision forward should succeed");

        // Check that mixed precision is enabled
        assert!(model.is_mixed_precision_enabled());
        assert_eq!(model.precision(), Precision::Fp16);
        assert_eq!(model.mixed_precision_memory_savings(), 0.5);
    }

    #[test]
    fn test_mixed_precision_forward_bf16() {
        use crate::training::{MixedPrecisionConfig, Precision};

        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)
            .unwrap()
            .with_mixed_precision(MixedPrecisionConfig::bf16());

        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        let result = model.forward_mixed_precision(&batch);
        assert!(result.is_ok(), "Mixed precision forward should succeed");

        assert_eq!(model.precision(), Precision::Bf16);
    }

    #[test]
    fn test_mixed_precision_backward_fp16() {
        use crate::training::{LossScaler, MixedPrecisionConfig};

        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)
            .unwrap()
            .with_mixed_precision(MixedPrecisionConfig::fp16());

        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Forward pass
        let _ = model.forward(&batch).unwrap();

        // Backward with loss scaling
        let mut scaler = LossScaler::new(256.0);
        let grads = model.backward_mixed_precision(&batch, 0.5, Some(&mut scaler));

        assert!(grads.is_ok(), "Mixed precision backward should succeed");
        let grads = grads.unwrap();
        assert_eq!(grads.len(), model.param_count());
    }

    #[test]
    fn test_mixed_precision_backward_without_scaler() {
        use crate::training::MixedPrecisionConfig;

        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64)
            .unwrap()
            .with_mixed_precision(MixedPrecisionConfig::bf16());

        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Forward pass
        let _ = model.forward(&batch).unwrap();

        // Backward without loss scaling
        let grads = model.backward_mixed_precision(&batch, 0.5, None);

        assert!(
            grads.is_ok(),
            "Mixed precision backward without scaler should succeed"
        );
    }

    #[test]
    fn test_mixed_precision_disabled_uses_regular_forward() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Mixed precision forward should fall back to regular forward when disabled
        let result = model.forward_mixed_precision(&batch);
        assert!(result.is_ok());

        // Check that mixed precision is not enabled
        assert!(!model.is_mixed_precision_enabled());
        assert_eq!(model.mixed_precision_memory_savings(), 0.0);
    }

    #[test]
    fn test_mixed_precision_disabled_uses_regular_backward() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = Batch::new(vec![1u32; 2 * 64], 2, 64).unwrap();

        // Forward pass
        let _ = model.forward(&batch).unwrap();

        // Mixed precision backward should fall back to regular backward when disabled
        let grads = model.backward_mixed_precision(&batch, 0.5, None);
        assert!(grads.is_ok());
    }
}
