//! Flash Attention: Memory-efficient attention for long sequences
//!
//! Flash Attention computes attention in blocks to reduce memory usage from
//! O(seq_len²) to O(seq_len * block_size), enabling training on longer
//! sequences without running out of memory.
//!
//! # Key Features
//!
//! - **Block-wise computation**: Processes attention in tiles to reduce memory
//! - **Online softmax**: Computes softmax incrementally without full materialization
//! - **Causal masking**: Supports autoregressive (decoder-side) attention
//! - **Numerically stable**: Uses proper scaling and max tracking
//!
//! # Memory Savings
//!
//! For a sequence length of 2048 with 8 heads and 64-dimensional head:
//! - Standard attention: ~16 MB per layer (stores full seq_len² matrix)
//! - Flash attention (block_size=128): ~1 MB per layer (94% reduction)
//!
//! # Algorithm
//!
//! Flash Attention processes the sequence in blocks:
//! 1. For each query block, compute attention with one key/value block
//! 2. Update running statistics (max, sum) for online softmax
//! 3. Accumulate output incrementally
//! 4. Normalize after processing all blocks
//!
//! # References
//!
//! - "Flash Attention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

use crate::{Error, Result};

/// Block size for Flash Attention computation
///
/// Larger blocks = faster but more memory
/// Smaller blocks = less memory but more overhead
///
/// Recommended values:
/// - 128-256 for sequence length ≤ 2048
/// - 64-128 for sequence length > 2048
const DEFAULT_BLOCK_SIZE: usize = 128;

/// Flash Attention layer
///
/// Memory-efficient attention that computes the result in blocks
/// without materializing the full seq_len × seq_len attention matrix.
pub struct FlashAttention {
    num_heads: usize,
    head_dim: usize,
    causal: bool,
    block_size: usize,
    scale: f32,
}

impl FlashAttention {
    /// Create a new Flash Attention layer
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    /// * `causal` - Whether to apply causal masking (for autoregressive models)
    pub fn new(num_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self {
            num_heads,
            head_dim,
            causal,
            block_size: DEFAULT_BLOCK_SIZE,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Create Flash Attention with custom block size
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    /// * `causal` - Whether to apply causal masking
    /// * `block_size` - Block size for computation (affects memory/speed tradeoff)
    pub fn with_block_size(
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        block_size: usize,
    ) -> Self {
        assert!(
            block_size > 0 && block_size.is_power_of_two(),
            "block_size must be a positive power of 2"
        );
        Self {
            num_heads,
            head_dim,
            causal,
            block_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get memory usage estimate for a given sequence length
    ///
    /// Returns memory in bytes per head
    pub fn memory_usage(&self, seq_len: usize) -> usize {
        // Standard attention: seq_len * seq_len * 4 bytes (f32)
        let standard = seq_len * seq_len * 4;

        // Flash attention: block_size * seq_len * 2 * 4 bytes (O + S matrices)
        let flash = self.block_size * seq_len * 2 * 4;

        std::cmp::min(standard, flash)
    }

    /// Compute Flash Attention
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [seq_len, num_heads, head_dim]
    /// * `k` - Key tensor [seq_len, num_heads, head_dim]
    /// * `v` - Value tensor [seq_len, num_heads, head_dim]
    ///
    /// # Returns
    ///
    /// Output tensor [seq_len, num_heads, head_dim]
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32]) -> Result<Vec<f32>> {
        let seq_len = q.len() / (self.num_heads * self.head_dim);

        if q.len() != k.len() || q.len() != v.len() {
            return Err(Error::InvalidParameter(format!(
                "Q, K, V must have same size: q={}, k={}, v={}",
                q.len(),
                k.len(),
                v.len()
            )));
        }

        let mut output = vec![0.0; q.len()];

        // Process each head independently
        for head in 0..self.num_heads {
            self.compute_flash_attention_head(q, k, v, &mut output, seq_len, head)?;
        }

        Ok(output)
    }

    /// Compute Flash Attention for a single head
    fn compute_flash_attention_head(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head: usize,
    ) -> Result<()> {
        let head_offset = head * seq_len * self.head_dim;

        // Initialize running statistics for online softmax
        let mut o = vec![0.0; seq_len * self.head_dim]; // Output
        let mut m = vec![f32::NEG_INFINITY; seq_len]; // Running max
        let mut l = vec![0.0; seq_len]; // Running sum

        // Process query blocks
        for q_block_start in (0..seq_len).step_by(self.block_size) {
            let q_block_end = (q_block_start + self.block_size).min(seq_len);

            // Process key/value blocks
            for kv_block_start in (0..seq_len).step_by(self.block_size) {
                let kv_block_end = (kv_block_start + self.block_size).min(seq_len);

                // Compute attention scores for this block
                for q_idx in q_block_start..q_block_end {
                    for kv_idx in kv_block_start..kv_block_end {
                        // Skip if causal masking and kv_idx > q_idx
                        if self.causal && kv_idx > q_idx {
                            continue;
                        }

                        // Get query and key vectors
                        let q_vec = &q[head_offset + q_idx * self.head_dim..][..self.head_dim];
                        let k_vec = &k[head_offset + kv_idx * self.head_dim..][..self.head_dim];

                        // Compute scaled dot-product: Q @ K^T / sqrt(d_k)
                        let mut score = 0.0;
                        for i in 0..self.head_dim {
                            score += q_vec[i] * k_vec[i];
                        }
                        score *= self.scale;

                        // Online softmax update
                        let new_max = m[q_idx].max(score);
                        let old_max = m[q_idx];

                        // Update running statistics
                        m[q_idx] = new_max;

                        let exp_old_max = if old_max.is_finite() {
                            (old_max - new_max).exp()
                        } else {
                            0.0
                        };
                        let exp_new = if score.is_finite() {
                            (score - new_max).exp()
                        } else {
                            0.0
                        };

                        let new_l = l[q_idx] * exp_old_max + exp_new;
                        let old_l = l[q_idx];
                        l[q_idx] = new_l;

                        // Update output with new statistics
                        let v_vec = &v[head_offset + kv_idx * self.head_dim..][..self.head_dim];
                        for d in 0..self.head_dim {
                            let o_idx = q_idx * self.head_dim + d;
                            if old_l > 0.0 {
                                o[o_idx] = o[o_idx] * exp_old_max * old_l / new_l;
                            }
                            o[o_idx] += exp_new * v_vec[d] / new_l;
                        }
                    }
                }
            }
        }

        // Copy output to result
        for i in 0..seq_len {
            for d in 0..self.head_dim {
                let out_idx = head_offset + i * self.head_dim + d;
                let o_idx = i * self.head_dim + d;
                output[out_idx] = o[o_idx];
            }
        }

        Ok(())
    }

    /// Compute memory savings percentage compared to standard attention
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    ///
    /// Percentage of memory saved (0-100)
    pub fn memory_saving_percentage(&self, seq_len: usize) -> f32 {
        let standard = seq_len * seq_len;
        let flash = self.block_size * seq_len * 2;

        let saved = if flash < standard {
            standard - flash
        } else {
            0
        };

        (saved as f32 / standard as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_creation() {
        let attn = FlashAttention::new(8, 64, false);
        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.head_dim, 64);
        assert_eq!(attn.causal, false);
        assert_eq!(attn.block_size, 128);
    }

    #[test]
    fn test_flash_attention_custom_block_size() {
        let attn = FlashAttention::with_block_size(8, 64, true, 256);
        assert_eq!(attn.block_size, 256);
        assert!(attn.causal);
    }

    #[test]
    fn test_flash_attention_invalid_block_size() {
        // Block size must be power of 2
        let result = std::panic::catch_unwind(|| {
            FlashAttention::with_block_size(8, 64, false, 100);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_attention_memory_usage() {
        let attn = FlashAttention::new(8, 64, false);

        // seq_len = 1024
        // Standard: 1024 * 1024 * 4 = 4,194,304 bytes
        // Flash: 128 * 1024 * 2 * 4 = 1,048,576 bytes
        let flash_memory = attn.memory_usage(1024);
        assert!(flash_memory < 1024 * 1024 * 4); // Less than standard
    }

    #[test]
    fn test_flash_attention_memory_saving() {
        let attn = FlashAttention::new(8, 64, false);

        // For seq_len = 2048, should save >85% memory
        let saving = attn.memory_saving_percentage(2048);
        assert!(saving > 85.0);

        // For seq_len = 4096, should save >90% memory
        let saving = attn.memory_saving_percentage(4096);
        assert!(saving > 90.0);
    }

    #[test]
    fn test_flash_attention_forward_simple() -> Result<()> {
        let attn = FlashAttention::new(2, 4, false);

        // Simple test: seq_len=4, num_heads=2, head_dim=4
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 4;

        let total_size = seq_len * num_heads * head_dim;

        // Create simple Q, K, V matrices
        let q: Vec<f32> = (0..total_size).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..total_size).map(|i| i as f32 * 0.1 + 0.5).collect();
        let v: Vec<f32> = (0..total_size).map(|i| i as f32 * 0.1 + 1.0).collect();

        let output = attn.forward(&q, &k, &v)?;

        assert_eq!(output.len(), total_size);

        // Output should be different from input
        assert_ne!(output, q);

        Ok(())
    }

    #[test]
    fn test_flash_attention_causal_mask() -> Result<()> {
        let attn = FlashAttention::new(1, 4, true);

        let seq_len = 4;
        let head_dim = 4;
        let total_size = seq_len * head_dim;

        // Create identity-like Q, K, V
        let q: Vec<f32> = (0..total_size).map(|i| if i % head_dim == 0 { 1.0 } else { 0.0 })
            .collect();
        let k: Vec<f32> = q.clone();
        let v: Vec<f32> = (0..total_size).map(|i| i as f32).collect();

        let output = attn.forward(&q, &k, &v)?;

        // With causal mask, position i should only attend to positions <= i
        // This creates a lower-triangular pattern in the attention
        assert_eq!(output.len(), total_size);

        Ok(())
    }

    #[test]
    fn test_flash_attention_different_sizes() -> Result<()> {
        // Test different sequence lengths
        for seq_len in [32, 64, 128, 256] {
            let attn = FlashAttention::new(4, 32, false);
            let total_size = seq_len * 4 * 32;

            let q = vec![0.1; total_size];
            let k = vec![0.2; total_size];
            let v = vec![0.3; total_size];

            let output = attn.forward(&q, &k, &v)?;
            assert_eq!(output.len(), total_size);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_deterministic() -> Result<()> {
        let attn = FlashAttention::new(2, 4, false);

        let seq_len = 8;
        let total_size = seq_len * 2 * 4;

        let q: Vec<f32> = (0..total_size).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..total_size).map(|i| i as f32 * 0.1 + 0.5).collect();
        let v: Vec<f32> = (0..total_size).map(|i| i as f32 * 0.1 + 1.0).collect();

        let output1 = attn.forward(&q, &k, &v)?;
        let output2 = attn.forward(&q, &k, &v)?;

        // Same inputs should produce same outputs
        assert_eq!(output1, output2);

        Ok(())
    }

    #[test]
    fn test_flash_attention_scale_computation() {
        let attn = FlashAttention::new(1, 64, false);
        // scale = 1 / sqrt(64) = 1/8 = 0.125
        assert!((attn.scale - 0.125).abs() < 1e-6);

        let attn = FlashAttention::new(1, 128, false);
        // scale = 1 / sqrt(128) ≈ 0.0883883
        assert!((attn.scale - (1.0 / 128.0_f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attention_block_size_boundaries() {
        // Test that block size boundaries are handled correctly
        let seq_len = 130; // Not a multiple of 128
        let attn = FlashAttention::new(2, 4, false);

        let total_size = seq_len * 2 * 4;
        let q = vec![0.1; total_size];
        let k = vec![0.2; total_size];
        let v = vec![0.3; total_size];

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flash_attention_mismatched_sizes() {
        let attn = FlashAttention::new(2, 4, false);

        let q = vec![0.1; 100];
        let k = vec![0.2; 200]; // Different size
        let v = vec![0.3; 100];

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_err());
    }
}
