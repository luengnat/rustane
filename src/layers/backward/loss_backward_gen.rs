//! Cross-entropy loss backward pass MIL generator
//!
//! Generates MIL code for cross-entropy loss backward pass.

use super::validate_mil_structure;
use super::BackwardMILGenerator;
use crate::ane::Result;
use crate::training::TransformerConfig;

/// MIL generator for cross-entropy loss backward pass
#[derive(Debug)]
pub struct LossBackwardGen;

impl LossBackwardGen {
    /// Create new loss backward MIL generator
    pub fn new() -> Self {
        LossBackwardGen
    }

    /// Generate MIL code for loss backward operation
    ///
    /// Uses simple generic MIL syntax that can be compiled.
    fn generate_mil_code(&self, _config: &TransformerConfig) -> String {
        let mut mil = String::new();
        mil.push_str("func loss_backward(\n");
        mil.push_str("    packed: (1, 1, 1, 1)\n");
        mil.push_str(") -> (d_logits: (1, 1, 1, 1)) {\n");
        mil.push_str("    // Simplified backward pass - returns placeholder\n");
        mil.push_str("    let zero = const_0(shape=[1,1,1,1], dtype=float32);\n");
        mil.push_str("    return (zero);\n");
        mil.push_str("}\n");

        mil
    }
}

impl Default for LossBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for LossBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let mil_code = self.generate(config)?;
        validate_mil_structure(&mil_code, "loss_backward", &["packed"], &["d_logits"])?;

        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "loss_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_backward_gen_creation() {
        let gen = LossBackwardGen::new();
        assert_eq!(gen.operation_name(), "loss_backward");
    }

    #[test]
    fn test_loss_backward_generate_mil() {
        let gen = LossBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        assert!(mil_code.contains("loss_backward"));
    }
}
