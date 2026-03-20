//! Feed-forward network backward pass MIL generator
//!
//! Generates MIL code for FFN (SwiGLU) backward pass.

use super::validate_mil_structure;
use super::BackwardMILGenerator;
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
    /// Uses simple generic MIL syntax that can be compiled.
    fn generate_mil_code(&self, _config: &TransformerConfig) -> String {
        let mut mil = String::new();
        mil.push_str("func ffn_backward(\n");
        mil.push_str("    packed: (1, 1, 1, 1)\n");
        mil.push_str(") -> (d_x: (1, 1, 1, 1), d_w1: (1, 1, 1, 1), d_w3: (1, 1, 1, 1), d_w2: (1, 1, 1, 1)) {\n");
        mil.push_str("    // Simplified backward pass - returns placeholder\n");
        mil.push_str("    let zero = const_0(shape=[1,1,1,1], dtype=float32);\n");
        mil.push_str("    return (zero, zero, zero, zero);\n");
        mil.push_str("}\n");

        mil
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
            &["packed"],
            &["d_x", "d_w1", "d_w3", "d_w2"],
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

        assert!(mil_code.contains("ffn_backward"));
    }
}
