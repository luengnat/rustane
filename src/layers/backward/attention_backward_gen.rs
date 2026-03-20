//! Multi-head attention backward pass MIL generator
//!
//! Generates MIL code for scaled dot-product attention backward pass.
//!
//! # ANE Limitation
//!
//! The generated MIL uses multiple inputs which ANE doesn't support.
//! ANE requires single input with embedded BLOBFILE weights.
//! This MIL format is provided for reference but won't compile on ANE.

use super::{validate_mil_structure, BackwardMILGenerator};
use crate::ane::Result;
use crate::training::TransformerConfig;

/// MIL generator for multi-head attention backward pass
#[derive(Debug)]
pub struct AttentionBackwardGen;

impl AttentionBackwardGen {
    /// Create new attention backward MIL generator
    pub fn new() -> Self {
        AttentionBackwardGen
    }

    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let dim = config.dim;
        let seq_len = config.seq_len;

        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
        mil.push_str("{\n");
        mil.push_str(&format!(
            "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> d_out, tensor<fp32, [1, {}, 1, {}]> q, tensor<fp32, [1, {}, 1, {}]> k, tensor<fp32, [1, {}, 1, {}]> v) {{\n",
            dim, seq_len,  // d_out
            dim, seq_len,  // q
            dim, seq_len,  // k
            dim, seq_len,  // v
        ));

        // Simplified attention backward pass
        // d_Wv = x^T @ d_out
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_out_T = transpose(x=d_out, axes=[0, 3, 2, 1])[name = string(\"d_out_T\")];\n", seq_len, dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> v_T = transpose(x=v, axes=[0, 3, 2, 1])[name = string(\"v_T\")];\n", seq_len, dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_wv = matmul(x=v_T, y=d_out_T)[name = string(\"d_wv\")];\n", dim, dim));

        // d_Wq = x^T @ d_out (simplified)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> q_T = transpose(x=q, axes=[0, 3, 2, 1])[name = string(\"q_T\")];\n", seq_len, dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_wq = matmul(x=q_T, y=d_out_T)[name = string(\"d_wq\")];\n", dim, dim));

        // d_Wk = x^T @ d_out (simplified)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> k_T = transpose(x=k, axes=[0, 3, 2, 1])[name = string(\"k_T\")];\n", seq_len, dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_wk = matmul(x=k_T, y=d_out_T)[name = string(\"d_wk\")];\n", dim, dim));

        // d_Wo = output @ d_out (simplified)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_wo = matmul(x=v, y=d_out)[name = string(\"d_wo\")];\n", dim, dim));

        mil.push_str("    } -> (d_wq, d_wk, d_wv, d_wo);\n");
        mil.push_str("}\n");

        mil
    }

    /// Helper function to calculate input sizes for multiple inputs
    pub fn input_sizes(&self, config: &TransformerConfig) -> Vec<usize> {
        let dim = config.dim;
        let seq_len = config.seq_len;

        vec![
            dim * seq_len * 4, // d_out
            dim * seq_len * 4, // q
            dim * seq_len * 4, // k
            dim * seq_len * 4, // v
        ]
    }

    /// Helper function to calculate output sizes in bytes
    pub fn output_sizes(&self, config: &TransformerConfig) -> Vec<usize> {
        let dim = config.dim;

        vec![
            dim * dim * 4, // d_wq
            dim * dim * 4, // d_wk
            dim * dim * 4, // d_wv
            dim * dim * 4, // d_wo
        ]
    }
}

impl Default for AttentionBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for AttentionBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let mil_code = self.generate(config)?;
        validate_mil_structure(
            &mil_code,
            "attention_backward",
            &["d_out", "q", "k", "v"],
            &["d_wq", "d_wk", "d_wv", "d_wo"],
        )?;
        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "attention_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_backward_gen_creation() {
        let gen = AttentionBackwardGen::new();
        assert_eq!(gen.operation_name(), "attention_backward");
    }

    #[test]
    fn test_attention_backward_generate_mil() {
        let gen = AttentionBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        assert!(mil_code.contains("program(1.3)"));
        assert!(mil_code.contains("ios18"));
    }

    #[test]
    fn test_attention_backward_input_sizes() {
        let gen = AttentionBackwardGen::new();
        let config = TransformerConfig::tiny(); // dim=128, seq_len=64
        let sizes = gen.input_sizes(&config);

        // Each input: 128*64*4 = 32768
        assert_eq!(sizes.len(), 4);
        assert_eq!(sizes[0], 32768);
    }

    #[test]
    fn test_attention_backward_output_sizes() {
        let gen = AttentionBackwardGen::new();
        let config = TransformerConfig::tiny(); // dim=128
        let sizes = gen.output_sizes(&config);

        // Each output: 128*128*4 = 65536
        assert_eq!(sizes.len(), 4);
        assert_eq!(sizes[0], 65536);
    }
}
