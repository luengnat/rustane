//! Feed-forward network backward pass MIL generator
//!
//! Generates MIL code for FFN (SwiGLU) backward pass.
//!
//! # ANE Limitation
//!
//! The generated MIL uses multiple inputs which ANE doesn't support.
//! ANE requires single input with embedded BLOBFILE weights.
//! This MIL format is provided for reference but won't compile on ANE.

use super::{validate_mil_structure, BackwardMILGenerator};
use crate::ane::Result;
use crate::training::TransformerConfig;

/// MIL generator for feed-forward network backward pass
#[derive(Debug)]
pub struct FFNBackwardGen;

impl FFNBackwardGen {
    /// Create new FFN backward MIL generator
    pub fn new() -> Self {
        FFNBackwardGen
    }

    /// Generate MIL code for FFN backward operation
    ///
    /// Implements backward pass for SwiGLU FFN:
    /// Forward: output = SiLU(W1 @ x) * (W3 @ x) @ W2
    /// Backward: Computes gradients for W1, W2, W3
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let seq_len = config.seq_len;

        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremlc-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
        mil.push_str("{\n");
        mil.push_str(&format!(
            "    func main<ios18>(tensor<fp32, [1, {}, 1, {}]> d_out, tensor<fp32, [1, {}, 1, {}]> x, tensor<fp32, [1, {}, 1, {}]> w1_out, tensor<fp32, [1, {}, 1, {}]> w3_out, tensor<fp32, [1, {}, 1, {}]> silu) {{\n",
            dim, seq_len,  // d_out
            dim, seq_len,  // x
            hidden_dim, seq_len,  // w1_out
            hidden_dim, seq_len,  // w3_out
            hidden_dim, seq_len,  // silu
        ));

        // Compute gate = silu * w3_out (intermediate activation)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> gate = mul(x=silu, y=w3_out)[name = string(\"gate\")];\n", hidden_dim, seq_len));

        // d_w2 = gate^T @ d_out (simplified - compute sum over sequence)
        mil.push_str("        tensor<int32, [1]> rax1 = const()[name = string(\"rax1\"), val=tensor<int32, [1]>([1])];\n");
        mil.push_str("        bool kd = const()[name = string(\"kd\"), val=bool(true)];\n");

        // Compute d_w2: transpose(gate) @ d_out
        // For simplicity, use matmul with proper reshaping
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_out_T = transpose(x=d_out, axes=[0, 3, 2, 1])[name = string(\"d_out_T\")];\n", seq_len, dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> gate_T = transpose(x=gate, axes=[0, 3, 2, 1])[name = string(\"gate_T\")];\n", seq_len, hidden_dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w2 = matmul(x=gate_T, y=d_out_T)[name = string(\"d_w2\")];\n", hidden_dim, dim));

        // d_intermediate = d_out @ w2^T (simplified as d_out broadcast)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_gate = mul(x=d_out, y=w3_out)[name = string(\"d_gate\")];\n", hidden_dim, seq_len));

        // d_w3_out = d_intermediate * silu_derivative (simplified as d_gate * 1.0)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w3_out = d_gate[name = string(\"d_w3_out\")];\n", hidden_dim, seq_len));

        // d_w1_out = d_intermediate * w3_out * silu_derivative
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w1_out_term = mul(x=d_gate, y=w3_out)[name = string(\"d_w1_t\")];\n", hidden_dim, seq_len));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w1_out = mul(x=d_w1_out_term, y=silu)[name = string(\"d_w1_out\")];\n", hidden_dim, seq_len));

        // d_w1 = x^T @ d_w1_out
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> x_T = transpose(x=x, axes=[0, 3, 2, 1])[name = string(\"x_T\")];\n", seq_len, dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w1_out_T = transpose(x=d_w1_out, axes=[0, 3, 2, 1])[name = string(\"d_w1_out_T\")];\n", seq_len, hidden_dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w1 = matmul(x=x_T, y=d_w1_out_T)[name = string(\"d_w1\")];\n", dim, hidden_dim));

        // d_w3 = x^T @ d_w3_out
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w3_out_T = transpose(x=d_w3_out, axes=[0, 3, 2, 1])[name = string(\"d_w3_out_T\")];\n", seq_len, hidden_dim));
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_w3 = matmul(x=x_T, y=d_w3_out_T)[name = string(\"d_w3\")];\n", dim, hidden_dim));

        // d_x (simplified gradient for input)
        mil.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> d_x = mul(x=d_out, y=x)[name = string(\"d_x\")];\n", dim, seq_len));

        mil.push_str("    } -> (d_w1, d_w2, d_w3);\n");
        mil.push_str("}\n");

        mil
    }

    /// Helper function to calculate input sizes for multiple inputs
    pub fn input_sizes(&self, config: &TransformerConfig) -> Vec<usize> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let seq_len = config.seq_len;

        vec![
            dim * seq_len * 4,        // d_out
            dim * seq_len * 4,        // x
            hidden_dim * seq_len * 4, // w1_out
            hidden_dim * seq_len * 4, // w3_out
            hidden_dim * seq_len * 4, // silu
        ]
    }

    /// Helper function to calculate output sizes in bytes
    pub fn output_sizes(&self, config: &TransformerConfig) -> Vec<usize> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;

        vec![
            dim * hidden_dim * 4, // d_w1
            hidden_dim * dim * 4, // d_w2
            dim * hidden_dim * 4, // d_w3
        ]
    }
}

impl Default for FFNBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for FFNBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let mil_code = self.generate(config)?;
        validate_mil_structure(
            &mil_code,
            "ffn_backward",
            &["d_out", "x", "w1_out", "w3_out", "silu"],
            &["d_w1", "d_w2", "d_w3"],
        )?;
        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "ffn_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_backward_gen_creation() {
        let gen = FFNBackwardGen::new();
        assert_eq!(gen.operation_name(), "ffn_backward");
    }

    #[test]
    fn test_ffn_backward_generate_mil() {
        let gen = FFNBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        assert!(mil_code.contains("program(1.3)"));
        assert!(mil_code.contains("ios18"));
    }

    #[test]
    fn test_ffn_backward_input_sizes() {
        let gen = FFNBackwardGen::new();
        let config = TransformerConfig::tiny(); // dim=128, hidden_dim=256, seq_len=64
        let sizes = gen.input_sizes(&config);

        // d_out: 128*64*4 = 32768
        // x: 128*64*4 = 32768
        // w1_out: 256*64*4 = 65536
        // w3_out: 256*64*4 = 65536
        // silu: 256*64*4 = 65536
        assert_eq!(sizes[0], 32768);
        assert_eq!(sizes[1], 32768);
        assert_eq!(sizes[2], 65536);
    }

    #[test]
    fn test_ffn_backward_output_sizes() {
        let gen = FFNBackwardGen::new();
        let config = TransformerConfig::tiny(); // dim=128, hidden_dim=256
        let sizes = gen.output_sizes(&config);

        // d_w1: 128*256*4 = 131072
        // d_w2: 256*128*4 = 131072
        // d_w3: 128*256*4 = 131072
        assert_eq!(sizes[0], 131072);
        assert_eq!(sizes[1], 131072);
        assert_eq!(sizes[2], 131072);
    }
}
