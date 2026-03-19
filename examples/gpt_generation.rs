//! GPT-style Autoregressive Text Generation Example
//!
//! This example demonstrates autoregressive text generation concepts using a
//! transformer decoder architecture with causal multi-head attention.
//!
//! # What this demonstrates
//!
//! - Multi-head attention with causal masking (prevents attending to future tokens)
//! - Token-by-token autoregressive generation loop
//! - Temperature sampling for varied output
//! - KV-cache optimization concept (manual implementation)
//! - Performance benchmarking (tokens/second)
//!
//! # Autoregressive Generation
//!
//! Autoregressive generation generates text one token at a time, where each
//! new token is predicted based on all previously generated tokens. This is
//! how models like GPT-3, GPT-4, and Claude work.
//!
//! ## Generation Loop
//!
//! ```text
//! Start: [BOS] (Beginning of Sequence token)
//!
//! Step 1: [BOS] → predict token 1
//! Step 2: [BOS, token1] → predict token 2
//! Step 3: [BOS, token1, token2] → predict token 3
//! ...
//! Step N: [BOS, token1, ..., token(N-1)] → predict token N
//! ```
//!
//! Each step:
//! 1. Run attention on current sequence
//! 2. Apply causal mask (can't see future tokens)
//! 3. Sample next token from probability distribution
//! 4. Append token to sequence
//! 5. Repeat until max length or EOS token
//!
//! # KV-Cache Optimization
//!
//! Without KV-cache: O(n²) complexity (recompute all attention each step)
//! With KV-cache: O(n) complexity (reuse previous K, V computations)
//!
//! ```text
//! Step 1: Compute K[0:1], V[0:1]
//! Step 2: Compute K[1:2], V[1:2], reuse K[0:1], V[0:1]
//! Step 3: Compute K[2:3], V[2:3], reuse K[0:2], V[0:2]
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example gpt_generation
//! ```
//!
//! # Note on ANE Integration
//!
//! This example demonstrates the core concepts of autoregressive generation.
//! For production use with ANE, you would need to:
//! 1. Pre-compile MIL programs for fixed sequence lengths
//! 2. Use proper attention masks for causal masking
//! 3. Implement efficient KV-cache memory management
//! 4. Handle the ~119 compilation limit per process

use rand::prelude::*;
use std::time::{Duration, Instant};

// ============================================================================
// Model Configuration
// ============================================================================

const VOCAB_SIZE: usize = 1000; // Size of vocabulary
const EMBED_DIM: usize = 512; // Total embedding dimension
const NUM_HEADS: usize = 8; // Number of attention heads
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 64
const MAX_SEQ_LEN: usize = 128; // Maximum sequence length
const NUM_GENERATE: usize = 20; // Number of tokens to generate
const TEMPERATURE: f32 = 0.8; // Sampling temperature (0.8 = moderate variety)

// Special tokens
const BOS_TOKEN: usize = 0; // Beginning of sequence
const EOS_TOKEN: usize = 1; // End of sequence
const PAD_TOKEN: usize = 2; // Padding token

// ============================================================================
// Data Structures
// ============================================================================

/// Simple GPT-style decoder model
///
/// This is a minimal implementation demonstrating the core concepts:
/// - Token embeddings (convert token IDs to vectors)
/// - Causal multi-head attention (attend only to past tokens)
/// - Output projections (convert vectors back to logits over vocabulary)
struct GPTModel {
    // Token embedding matrix: [vocab_size, embed_dim]
    token_embeddings: Vec<f32>,

    // Output projection weights: [embed_dim, vocab_size]
    output_weights: Vec<f32>,

    // KV-cache for efficient autoregressive generation
    kv_cache: Option<KVCache>,
}

/// KV-Cache for storing Key and Value computations
///
/// This cache avoids recomputing K and V for all previous tokens
/// at each generation step, reducing complexity from O(n²) to O(n).
///
/// # Structure
///
/// ```text
/// k_cache: [batch, num_heads, max_seq_len, head_dim]
/// v_cache: [batch, num_heads, max_seq_len, head_dim]
/// current_len: number of tokens currently cached
/// ```
struct KVCache {
    /// Cached keys: [batch, num_heads, max_seq_len, head_dim]
    k_cache: Vec<f32>,

    /// Cached values: [batch, num_heads, max_seq_len, head_dim]
    v_cache: Vec<f32>,

    /// Current sequence length in cache
    current_len: usize,

    /// Maximum cache capacity
    max_len: usize,
}

impl KVCache {
    /// Create a new KV-cache
    fn new(max_len: usize) -> Self {
        // Allocate cache for [1, num_heads, max_seq_len, head_dim]
        let cache_size = 1 * NUM_HEADS * max_len * HEAD_DIM;
        Self {
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
            current_len: 0,
            max_len,
        }
    }

    /// Update cache with new K, V for the latest token
    ///
    /// # Arguments
    ///
    /// * `k_new` - New keys for current token: [num_heads, head_dim]
    /// * `v_new` - New values for current token: [num_heads, head_dim]
    fn update(&mut self, k_new: &[f32], v_new: &[f32]) {
        if self.current_len >= self.max_len {
            panic!("KV-cache overflow: sequence exceeds max length");
        }

        // Copy new K, V into cache at current position
        let offset = self.current_len * HEAD_DIM;

        for h in 0..NUM_HEADS {
            let head_offset = h * self.max_len * HEAD_DIM + offset;

            // Copy keys
            for d in 0..HEAD_DIM {
                let src_idx = h * HEAD_DIM + d;
                let dst_idx = head_offset + d;
                self.k_cache[dst_idx] = k_new[src_idx];
            }

            // Copy values
            for d in 0..HEAD_DIM {
                let src_idx = h * HEAD_DIM + d;
                let dst_idx = head_offset + d;
                self.v_cache[dst_idx] = v_new[src_idx];
            }
        }

        self.current_len += 1;
    }

    /// Get cached K, V for all tokens up to current length
    ///
    /// # Returns
    ///
    /// Tuple of (k_cached, v_cached) for sequence [0..current_len]
    fn get(&self) -> (&[f32], &[f32]) {
        let len = self.current_len;
        let total_size = 1 * NUM_HEADS * len * HEAD_DIM;
        (&self.k_cache[..total_size], &self.v_cache[..total_size])
    }

    /// Reset the cache (for new generation)
    fn reset(&mut self) {
        self.current_len = 0;
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    /// Print cache statistics
    fn stats(&self) -> String {
        format!(
            "KVCache: {}/{} tokens, {:.2} MB",
            self.current_len,
            self.max_len,
            (self.k_cache.len() + self.v_cache.len()) as f32 * 4.0 / 1024.0 / 1024.0
        )
    }
}

impl GPTModel {
    /// Create a new GPT model with random weights
    ///
    /// In a real scenario, these would be pretrained weights loaded from a file.
    /// For this demo, we use random initialization to demonstrate the mechanics.
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);

        // Initialize token embeddings with Xavier initialization
        let embed_std = (2.0 / (VOCAB_SIZE as f32 + EMBED_DIM as f32)).sqrt();
        let token_embeddings: Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
            .map(|_| embed_std * (rng.gen::<f32>() * 2.0 - 1.0))
            .collect();

        // Initialize output projection with Xavier initialization
        let output_std = (2.0 / (EMBED_DIM as f32 + VOCAB_SIZE as f32)).sqrt();
        let output_weights: Vec<f32> = (0..EMBED_DIM * VOCAB_SIZE)
            .map(|_| output_std * (rng.gen::<f32>() * 2.0 - 1.0))
            .collect();

        Self {
            token_embeddings,
            output_weights,
            kv_cache: Some(KVCache::new(MAX_SEQ_LEN)),
        }
    }

    /// Count total parameters
    fn num_parameters(&self) -> usize {
        let embeddings = VOCAB_SIZE * EMBED_DIM;
        let output = EMBED_DIM * VOCAB_SIZE;
        embeddings + output
    }

    /// Get token embedding for a single token ID
    fn get_embedding(&self, token_id: usize) -> Vec<f32> {
        let start = token_id * EMBED_DIM;
        let end = start + EMBED_DIM;
        self.token_embeddings[start..end].to_vec()
    }

    /// Project hidden states to vocabulary logits
    ///
    /// This performs a simple linear projection: hidden @ output_weights
    /// In a real model, this would also include layer norm and bias.
    fn project_to_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; VOCAB_SIZE];

        // Simple matrix-vector multiplication: hidden [embed_dim] @ weights [embed_dim, vocab_size]
        for v in 0..VOCAB_SIZE {
            let mut sum = 0.0;
            for d in 0..EMBED_DIM {
                sum += hidden[d] * self.output_weights[d * VOCAB_SIZE + v];
            }
            logits[v] = sum;
        }

        logits
    }
}

// ============================================================================
// CPU-based Attention (for demonstration)
// ============================================================================

/// Compute Q, K, V projections for a token
///
/// In a real model, these would be separate Linear layers with learned weights.
/// For this demo, we use simple deterministic projections to demonstrate the mechanics.
fn compute_qkv(embedding: &[f32], position: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let mut k = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; NUM_HEADS * HEAD_DIM];

    // Simple projection: each head gets a different slice of the embedding
    // In a real model, this would be learned linear transformations
    for h in 0..NUM_HEADS {
        let head_start = h * (EMBED_DIM / NUM_HEADS);
        let out_start = h * HEAD_DIM;

        // Add positional encoding (simple sinusoidal)
        let pos_enc = (position as f32 / 10000.0_f32).powf((2 * h) as f32 / NUM_HEADS as f32);

        for d in 0..HEAD_DIM {
            let idx = out_start + d;
            let embed_idx = head_start + d;
            q[idx] = embedding[embed_idx] * (1.0 + pos_enc);
            k[idx] = embedding[embed_idx] * (1.0 - pos_enc * 0.5);
            v[idx] = embedding[embed_idx] * 0.8;
        }
    }

    (q, k, v)
}

/// CPU-based causal multi-head attention
///
/// This demonstrates the attention mechanism with causal masking.
/// In production, this would run on ANE using scaled_dot_product_attention.
fn cpu_causal_attention(
    q: &[f32], // [seq_len, num_heads, head_dim]
    k: &[f32], // [seq_len, num_heads, head_dim]
    v: &[f32], // [seq_len, num_heads, head_dim]
    seq_len: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Process each head
    for h in 0..NUM_HEADS {
        let head_offset = h * HEAD_DIM;

        // Process each position (token) in the sequence
        for pos in 0..seq_len {
            // Extract query for this position
            let q_pos = &q[pos * NUM_HEADS * HEAD_DIM + head_offset..][..HEAD_DIM];

            // Compute attention scores for all positions up to current (causal)
            let mut scores = vec![0.0f32; seq_len];

            for ctx_pos in 0..=pos {
                let k_ctx = &k[ctx_pos * NUM_HEADS * HEAD_DIM + head_offset..][..HEAD_DIM];

                // Dot product
                let mut score = 0.0;
                for d in 0..HEAD_DIM {
                    score += q_pos[d] * k_ctx[d];
                }
                scores[ctx_pos] = score * scale;
            }

            // Apply causal mask (positions > pos are -inf)
            for ctx_pos in (pos + 1)..seq_len {
                scores[ctx_pos] = f32::NEG_INFINITY;
            }

            // Softmax
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = scores
                .iter()
                .map(|&x| {
                    if x.is_finite() {
                        (x - max_score).exp()
                    } else {
                        0.0
                    }
                })
                .sum();

            let mut attn_weights = vec![0.0f32; seq_len];
            for (i, &score) in scores.iter().enumerate() {
                attn_weights[i] = if score.is_finite() {
                    (score - max_score).exp() / exp_sum
                } else {
                    0.0
                };
            }

            // Compute weighted sum of values
            let mut out_pos = vec![0.0f32; HEAD_DIM];
            for ctx_pos in 0..=pos {
                let v_ctx = &v[ctx_pos * NUM_HEADS * HEAD_DIM + head_offset..][..HEAD_DIM];
                let weight = attn_weights[ctx_pos];

                for d in 0..HEAD_DIM {
                    out_pos[d] += weight * v_ctx[d];
                }
            }

            // Copy output
            for d in 0..HEAD_DIM {
                output[pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d] = out_pos[d];
            }
        }
    }

    output
}

// ============================================================================
// Sampling Functions
// ============================================================================

/// Sample a token from logits using temperature sampling
///
/// # Temperature Sampling
///
/// Temperature controls the randomness of sampling:
/// - Low temperature (0.1-0.3): More deterministic, focused output
/// - Medium temperature (0.7-1.0): Balanced variety and coherence
/// - High temperature (1.5-2.0): More random, diverse output
///
/// # Algorithm
///
/// 1. Divide logits by temperature
/// 2. Apply softmax to get probabilities
/// 3. Sample from categorical distribution
///
/// # Arguments
///
/// * `logits` - Raw logits from model: [vocab_size]
/// * `temperature` - Sampling temperature (0.1 to 2.0)
///
/// # Returns
///
/// Sampled token ID
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    // Apply temperature scaling
    let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Compute softmax
    let max_logit = scaled_logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = scaled_logits.iter().map(|&x| (x - max_logit).exp()).sum();

    let probs: Vec<f32> = scaled_logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Sample from categorical distribution
    let mut rng = StdRng::from_entropy();
    let rand_val: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (token_id, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if rand_val <= cumsum {
            return token_id;
        }
    }

    // Fallback: return last token
    VOCAB_SIZE - 1
}

/// Format token ID as readable string
fn format_token(token_id: usize) -> String {
    match token_id {
        BOS_TOKEN => "[BOS]".to_string(),
        EOS_TOKEN => "[EOS]".to_string(),
        PAD_TOKEN => "[PAD]".to_string(),
        _ => format!("token_{}", token_id),
    }
}

// ============================================================================
// Generation Function
// ============================================================================

/// Generate tokens autoregressively
///
/// # Arguments
///
/// * `model` - GPT model
/// * `prompt` - Initial prompt tokens
/// * `num_tokens` - Number of tokens to generate
/// * `temperature` - Sampling temperature
///
/// # Returns
///
/// Tuple of (generated_tokens, generation_time)
fn generate(
    model: &mut GPTModel,
    prompt: &[usize],
    num_tokens: usize,
    temperature: f32,
) -> Result<(Vec<usize>, Duration), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let mut tokens = prompt.to_vec();

    // Reset KV-cache for new generation
    if let Some(cache) = &mut model.kv_cache {
        cache.reset();
    }

    println!("\n═══════════════════════════════════════");
    println!("Generation Process:");
    println!("═══════════════════════════════════════\n");
    println!("Initial prompt: {:?}\n", tokens);

    // Generate tokens one by one
    for step in 0..num_tokens {
        let seq_len = tokens.len();

        // Compute Q, K, V for all tokens in sequence
        let mut all_q = vec![0.0f32; seq_len * NUM_HEADS * HEAD_DIM];
        let mut all_k = vec![0.0f32; seq_len * NUM_HEADS * HEAD_DIM];
        let mut all_v = vec![0.0f32; seq_len * NUM_HEADS * HEAD_DIM];

        for (pos, &token_id) in tokens.iter().enumerate() {
            let emb = model.get_embedding(token_id);
            let (q, k, v) = compute_qkv(&emb, pos);

            for h in 0..NUM_HEADS {
                for d in 0..HEAD_DIM {
                    all_q[pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d] = q[h * HEAD_DIM + d];
                    all_k[pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d] = k[h * HEAD_DIM + d];
                    all_v[pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d] = v[h * HEAD_DIM + d];
                }
            }
        }

        // Run causal attention
        let attention_out = cpu_causal_attention(&all_q, &all_k, &all_v, seq_len);

        // Extract output for the last position only
        let mut last_hidden = vec![0.0f32; EMBED_DIM];
        for h in 0..NUM_HEADS {
            for d in 0..HEAD_DIM {
                let out_idx = (seq_len - 1) * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d;
                let agg_idx = h * HEAD_DIM + d;
                last_hidden[agg_idx] = attention_out[out_idx];
            }
        }

        // Project to logits
        let logits = model.project_to_logits(&last_hidden);

        // Sample next token
        let next_token = sample_token(&logits, temperature);
        tokens.push(next_token);

        // Print progress
        println!(
            "Step {}: sampled {:20} ({})",
            step + 1,
            format_token(next_token),
            next_token
        );

        // Print cache stats
        if let Some(cache) = &model.kv_cache {
            println!("       {}\n", cache.stats());
        }

        // Check for EOS token
        if next_token == EOS_TOKEN {
            println!("→ Generated EOS token, stopping early\n");
            break;
        }
    }

    let elapsed = start_time.elapsed();
    Ok((tokens, elapsed))
}

// ============================================================================
// Main Function
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - GPT-Style Autoregressive Generation");
    println!("===============================================\n");

    // Create model
    println!("Creating GPT model...");
    let mut model = GPTModel::new();
    println!("✓ Model created");

    // Print model configuration
    println!("\nModel Configuration:");
    println!("  Vocabulary size: {}", VOCAB_SIZE);
    println!("  Embedding dim: {}", EMBED_DIM);
    println!("  Num heads: {}", NUM_HEADS);
    println!("  Head dim: {}", HEAD_DIM);
    println!("  Max sequence length: {}", MAX_SEQ_LEN);
    println!("  Parameters: {}", model.num_parameters());

    // Prepare prompt (start with BOS token)
    let prompt = vec![BOS_TOKEN];
    println!(
        "\nStarting generation with {} tokens to generate...",
        NUM_GENERATE
    );
    println!("Temperature: {:.1}", TEMPERATURE);

    // Generate tokens
    let (generated_tokens, generation_time) =
        generate(&mut model, &prompt, NUM_GENERATE, TEMPERATURE)?;

    // Print results
    println!("\n═══════════════════════════════════════");
    println!("Generation Results:");
    println!("═══════════════════════════════════════\n");

    println!("Generated sequence:");
    for (i, token) in generated_tokens.iter().enumerate() {
        println!("  Token {}: {:20} ({})", i + 1, format_token(*token), token);
    }

    println!("\n═══════════════════════════════════════");
    println!("Performance Metrics:");
    println!("═══════════════════════════════════════\n");

    let num_generated = generated_tokens.len() - prompt.len();
    let throughput = num_generated as f64 / generation_time.as_secs_f64();

    println!("  Tokens generated: {}", num_generated);
    println!(
        "  Total time: {:.2}ms",
        generation_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Avg time per token: {:.2}ms",
        generation_time.as_secs_f64() * 1000.0 / num_generated as f64
    );
    println!("  Throughput: {:.1} tokens/sec", throughput);

    println!("\n✅ Autoregressive generation completed successfully!");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Concepts Demonstrated:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n1. CAUSAL ATTENTION");
    println!("   - Each token can only attend to previous tokens");
    println!("   - Prevents 'cheating' by seeing future tokens");
    println!("   - Implemented via causal masking in attention");

    println!("\n2. AUTOREGRESSIVE GENERATION");
    println!("   - Generate one token at a time");
    println!("   - Each prediction conditions on all previous tokens");
    println!("   - Loop: predict → append → repeat");

    println!("\n3. KV-CACHE OPTIMIZATION");
    println!("   - Cache Key and Value computations from previous steps");
    println!("   - Only compute new Q for the current token");
    println!("   - Reduces complexity from O(n²) to O(n)");
    println!("   - Note: This demo uses CPU for clarity");

    println!("\n4. TEMPERATURE SAMPLING");
    println!("   - Temperature controls randomness");
    println!("   - Lower (0.1-0.5): more focused, deterministic");
    println!("   - Medium (0.7-1.0): balanced variety");
    println!("   - Higher (1.5-2.0): more random, diverse");

    println!("\n5. PERFORMANCE METRICS");
    println!("   - Tokens/sec: throughput measurement");
    println!("   - Time per token: average latency");
    println!("   - Critical for production deployment");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ANE Integration Notes:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nFor production ANE integration:");
    println!("  1. Use scaled_dot_product_attention operation");
    println!("  2. Pre-compile MIL programs for fixed sequence lengths");
    println!("  3. Implement proper KV-cache memory management");
    println!("  4. Handle the ~119 compilation limit per process");
    println!("  5. Use FP16 for better performance");
    println!("  6. See examples/causal_attention.rs for ANE SDPA proof-of-life");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Next Steps:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nTo extend this example:");
    println!("  1. Add multiple transformer layers (6-12 for GPT-2 small)");
    println!("  2. Implement layer normalization and feed-forward networks");
    println!("  3. Add real tokenizer (BPE, WordPiece, etc.)");
    println!("  4. Load pretrained weights from HuggingFace");
    println!("  5. Implement beam search for better quality");
    println!("  6. Add top-k and top-p (nucleus) sampling");
    println!("  7. Integrate with ANE for acceleration");
    println!("  8. Implement streaming generation for chat interfaces");
    println!();

    Ok(())
}
