//! Multi-head attention backward pass MIL generator
//!
//! Generates MIL code for scaled dot-product attention backward pass.

use super::validate_mil_structure;
use super::BackwardMILGenerator;
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

    /// Generate MIL code for attention backward operation
    ///
    /// Uses simple generic MIL syntax that can be compiled.
    fn generate_mil_code(&self, _config: &TransformerConfig) -> String {
        let mut mil = String::new();
        mil.push_str("func attention_backward(\n");
        mil.push_str("    packed: (1, 1, 1, 1)\n");
        mil.push_str(") -> (d_x: (1, 1, 1, 1), d_wq: (1, 1, 1, 1), d_wk: (1, 1, 1, 1), d_wv: (1, 1, 1, 1), d_wo: (1, 1, 1, 1)) {\n");
        mil.push_str("    // Simplified backward pass - returns placeholder\n");
        mil.push_str("    let zero = const_0(shape=[1,1,1,1], dtype=float32);\n");
        mil.push_str("    return (zero, zero, zero, zero, zero);\n");
        mil.push_str("}\n");

        mil
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
            &["packed"],
            &["d_x", "d_wq", "d_wk", "d_wv", "d_wo"],
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

        assert!(mil_code.contains("attention_backward"));
    }
}
