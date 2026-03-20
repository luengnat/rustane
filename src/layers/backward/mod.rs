//! # Backward Pass MIL Generators
//!
//! This module implements gradient computation for all transformer operations
//! via ANE-executable MIL code generation.

pub mod attention_backward_gen;
pub mod ffn_backward_gen;
pub mod loss_backward_gen;
pub mod rmsnorm_backward_gen;
pub mod validation;

use crate::ane::Result;
use crate::training::TransformerConfig;

/// Trait for ANE backward MIL generators
pub trait BackwardMILGenerator {
    /// Generate MIL code for this backward operation
    fn generate(&self, config: &TransformerConfig) -> Result<String>;

    /// Validate generated kernel against CPU reference
    fn validate(&self, config: &TransformerConfig) -> Result<()>;

    /// Operation name for debugging and logging
    fn operation_name(&self) -> &'static str;
}

// Re-export generators
pub use attention_backward_gen::AttentionBackwardGen;
pub use ffn_backward_gen::FFNBackwardGen;
pub use loss_backward_gen::LossBackwardGen;
pub use rmsnorm_backward_gen::RMSNormBackwardGen;

// Re-export validation
pub use validation::{BackwardValidationSuite, ValidationReport};

/// Helper function to validate MIL code structure
pub fn validate_mil_structure(
    mil_code: &str,
    func_name: &str,
    expected_inputs: &[&str],
    expected_outputs: &[&str],
) -> Result<()> {
    // Check for IR schema declaration (#!irms6 or program(1.3))
    if !mil_code.contains("#!irms6") && !mil_code.contains("program(1.3)") {
        return Err(crate::ane::ANEError::CompileFailed(format!(
            "MIL for '{}' missing IR schema declaration (#!irms6 or program(1.3))",
            func_name
        ))
        .into());
    }

    // Check function declaration
    // Support both older 'main name(...)' and newer 'func main<ios18>(...)'
    let has_func = mil_code.contains(&format!("main {}", func_name))
        || mil_code.contains(&format!("func {}", func_name))
        || mil_code.contains("func main");

    if !has_func {
        return Err(crate::ane::ANEError::CompileFailed(format!(
            "MIL missing function declaration for '{}'",
            func_name
        ))
        .into());
    }

    // Check inputs
    for &input in expected_inputs {
        if !mil_code.contains(input) {
            return Err(crate::ane::ANEError::CompileFailed(format!(
                "MIL for '{}' missing input '{}'",
                func_name, input
            ))
            .into());
        }
    }

    // Check outputs
    for &output in expected_outputs {
        if !mil_code.contains(output) {
            return Err(crate::ane::ANEError::CompileFailed(format!(
                "MIL for '{}' missing output '{}'",
                func_name, output
            ))
            .into());
        }
    }

    // Check return statement
    if !mil_code.contains("return") && !mil_code.contains("} ->") {
        return Err(crate::ane::ANEError::CompileFailed(format!(
            "MIL for '{}' missing return statement",
            func_name
        ))
        .into());
    }

    Ok(())
}
