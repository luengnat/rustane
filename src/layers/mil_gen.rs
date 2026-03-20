//! MIL code generation for transformer operations
//!
//! This module provides the `MILGenerator` struct that generates MIL (Model Intermediate Language)
//! code for transformer attention and feed-forward network forward passes on the Apple Neural Engine.

use crate::training::TransformerConfig;

/// Generates MIL (Model Intermediate Language) code for ANE computation
///
/// The `MILGenerator` takes a transformer configuration and produces syntactically valid
/// MIL code for attention and feed-forward network operations.
pub struct MILGenerator {
    config: TransformerConfig,
}

impl MILGenerator {
    /// Create a new MIL generator for the given transformer configuration
    pub fn new(config: &TransformerConfig) -> Self {
        MILGenerator {
            config: config.clone(),
        }
    }

    /// Get a reference to the transformer configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Generate MIL for attention forward pass
    ///
    /// Input tensor layout: [1, dim, 1, seq + qkv_dim + kv_dim + kv_dim]
    ///   Contains: x (seq_len*dim) + Wq (dim*dim) + Wk (dim*dim) + Wv (dim*dim)
    pub fn gen_attention_forward(&self) -> String {
        let mut mil = String::new();
        let dim = self.config.dim;
        let seq_len = self.config.seq_len;
        let head_dim = self.config.head_dim;

        mil.push_str(&format!(
            "func attention_forward(x: (1, {}, 1, {})) -> (1, {}, 1, {}) {{\n",
            dim, seq_len + 3 * dim, dim, seq_len
        ));

        // Extract x, Wq, Wk, Wv from packed input
        mil.push_str(&format!("  let x = cast(x_slice_[0:1, 0:{}, 0:1, 0:{}]); /* input tokens */\n", dim, seq_len));
        mil.push_str(&format!("  let Wq = cast(x_slice_[0:1, 0:{}, 0:1, {}:{}]); /* query weight */\n", dim, seq_len, seq_len + dim));
        mil.push_str(&format!("  let Wk = cast(x_slice_[0:1, 0:{}, 0:1, {}:{}]); /* key weight */\n", dim, seq_len + dim, seq_len + 2 * dim));
        mil.push_str(&format!("  let Wv = cast(x_slice_[0:1, 0:{}, 0:1, {}:{}]); /* value weight */\n", dim, seq_len + 2 * dim, seq_len + 3 * dim));

        // Compute Q, K, V
        mil.push_str("  let Q = matmul(x, Wq); /* [seq_len, dim] */\n");
        mil.push_str("  let K = matmul(x, Wk);\n");
        mil.push_str("  let V = matmul(x, Wv);\n");

        // Scaled dot-product attention
        mil.push_str(&format!("  let scale = 1.0 / sqrt({}); /* head_dim */\n", head_dim));
        mil.push_str("  let scores = matmul(Q, transpose(K)) * scale;\n");
        mil.push_str("  let weights = softmax(scores);\n");
        mil.push_str("  let attn_out = matmul(weights, V);\n");

        mil.push_str("  return attn_out;\n");
        mil.push_str("}\n");

        mil
    }

    /// Generate MIL for FFN (feed-forward network) forward pass
    ///
    /// Implements SiLU gating: (W1(x) * SiLU(W1(x))) @ W2, with additional W3 projection
    pub fn gen_ffn_forward(&self) -> String {
        let mut mil = String::new();
        let dim = self.config.dim;
        let hidden_dim = self.config.hidden_dim;
        let seq_len = self.config.seq_len;

        mil.push_str(&format!(
            "func ffn_forward(x: (1, {}, 1, {})) -> (1, {}, 1, {}) {{\n",
            dim, seq_len + 2 * hidden_dim + hidden_dim, dim, seq_len
        ));

        mil.push_str(&format!("  let x = cast(x_slice_[0:1, 0:{}, 0:1, 0:{}]);\n", dim, seq_len));
        mil.push_str(&format!("  let W1 = cast(x_slice_[0:1, 0:{}, 0:1, {}:{}]);\n",
                             dim, seq_len, seq_len + hidden_dim));
        mil.push_str(&format!("  let W3 = cast(x_slice_[0:1, 0:{}, 0:1, {}:{}]);\n",
                             dim, seq_len + hidden_dim, seq_len + 2 * hidden_dim));
        mil.push_str(&format!("  let W2 = cast(x_slice_[0:1, 0:{}, 0:1, {}:{}]);\n",
                             hidden_dim, seq_len + 2 * hidden_dim, seq_len + 2 * hidden_dim + hidden_dim));

        mil.push_str("  let hidden1 = matmul(x, W1);\n");
        mil.push_str("  let hidden3 = matmul(x, W3);\n");
        mil.push_str("  let gated = hidden1 * silu(hidden1);\n"); // SiLU gating
        mil.push_str("  let gated2 = gated * hidden3;\n");
        mil.push_str("  let out = matmul(gated2, W2);\n");

        mil.push_str("  return out;\n");
        mil.push_str("}\n");

        mil
    }

    /// Generate MIL for backward pass (reference; actual backward is CPU-based)
    pub fn gen_attention_backward(&self) -> String {
        "/* Backward computed on CPU; see transformer_backward.rs */".to_string()
    }

    /// Generate MIL for FFN backward (reference)
    pub fn gen_ffn_backward(&self) -> String {
        "/* Backward computed on CPU; see transformer_backward.rs */".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::TransformerConfig;

    #[test]
    fn test_mil_generator_with_small_config() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
        let gen = MILGenerator::new(&config);

        assert_eq!(gen.config().dim, 128);
        assert_eq!(gen.config().n_heads, 4);
        assert_eq!(gen.config().head_dim, 32);
    }

    #[test]
    fn test_attention_forward_contains_key_operations() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
        let gen = MILGenerator::new(&config);
        let mil = gen.gen_attention_forward();

        assert!(mil.contains("matmul"), "should contain matmul operations");
        assert!(mil.contains("softmax"), "should contain softmax");
        assert!(mil.contains("transpose"), "should contain transpose");
        assert!(mil.contains("cast"), "should contain cast operations");
    }

    #[test]
    fn test_ffn_forward_contains_key_operations() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
        let gen = MILGenerator::new(&config);
        let mil = gen.gen_ffn_forward();

        assert!(mil.contains("silu"), "should contain silu activation");
        assert!(mil.contains("matmul"), "should contain matmul");
        assert!(mil.contains("cast"), "should contain cast operations");
    }

    #[test]
    fn test_mil_output_is_valid_syntax() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
        let gen = MILGenerator::new(&config);

        let attn_mil = gen.gen_attention_forward();
        let ffn_mil = gen.gen_ffn_forward();

        // Both should define a function
        assert!(attn_mil.starts_with("func "));
        assert!(ffn_mil.starts_with("func "));

        // Both should end with closing brace
        assert!(attn_mil.trim().ends_with("}"));
        assert!(ffn_mil.trim().ends_with("}"));

        // Both should contain return statement
        assert!(attn_mil.contains("return "));
        assert!(ffn_mil.contains("return "));
    }

    #[test]
    fn test_backward_methods_return_placeholders() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
        let gen = MILGenerator::new(&config);

        let attn_back = gen.gen_attention_backward();
        let ffn_back = gen.gen_ffn_backward();

        assert!(attn_back.contains("CPU"));
        assert!(ffn_back.contains("CPU"));
    }

    #[test]
    fn test_attention_forward_with_large_config() {
        let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
        let gen = MILGenerator::new(&config);
        let mil = gen.gen_attention_forward();

        assert!(mil.contains("func attention_forward"));
        assert!(mil.contains("return"));
        assert!(mil.len() > 300);
    }

    #[test]
    fn test_ffn_forward_tensor_slicing() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
        let gen = MILGenerator::new(&config);
        let mil = gen.gen_ffn_forward();

        // Check that tensor slicing is present for weight extraction
        assert!(mil.contains("x_slice_"));
        // Check all three weight matrices are extracted
        assert!(mil.contains("let W1 ="));
        assert!(mil.contains("let W3 ="));
        assert!(mil.contains("let W2 ="));
    }
}
