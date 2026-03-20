//! # Backward Pass MIL Generators
//!
//! This module implements gradient computation for all transformer operations
//! via ANE-executable MIL code generation.
//!
//! ## Architecture
//!
//! Each backward operation is generated as MIL code:
//! - **RMSNormBackwardGen**: Normalization gradients (scale + bias)
//! - **AttentionBackwardGen**: Attention gradients (Q, K, V, O)
//! - **FFNBackwardGen**: Feed-forward gradients (linear layers + activation)
//! - **LossBackwardGen**: Cross-entropy loss gradients
//!
//! ## Validation
//!
//! All generators are validated once at startup via **BackwardValidationSuite**
//! with 1e-6 relative error tolerance against CPU reference implementations.
//!
//! ## Usage
//!
//! ```ignore
//! use rustane::layers::backward::*;
//!
//! let suite = BackwardValidationSuite::new();
//! let report = suite.validate_all(&config)?;
//! assert!(report.loss_passed);
//! ```

pub mod attention_backward_gen;
pub mod ffn_backward_gen;
pub mod loss_backward_gen;
pub mod rmsnorm_backward_gen;
pub mod validation;

use crate::ane::Result;
use crate::training::TransformerConfig;

/// Trait for ANE backward MIL generators
///
/// All backward operations on ANE must implement this trait to provide:
/// 1. MIL code generation for the backward operation
/// 2. Validation against CPU reference implementations
/// 3. Operation identification for debugging
pub trait BackwardMILGenerator {
    /// Generate MIL code for this backward operation
    ///
    /// # Arguments
    /// - `config`: Transformer configuration (determines dimensions, precision, etc.)
    ///
    /// # Returns
    /// MIL code as a string that can be compiled by ANERuntime
    ///
    /// # Example
    /// ```ignore
    /// let mil_code = generator.generate(&config)?;
    /// assert!(mil_code.contains("mil ="));
    /// ```
    fn generate(&self, config: &TransformerConfig) -> Result<String>;

    /// Validate generated kernel against CPU reference
    ///
    /// # Arguments
    /// - `config`: Transformer configuration
    ///
    /// # Returns
    /// Ok(()) if validation passes (1e-6 relative tolerance)
    ///
    /// # Process
    /// 1. Generate MIL code via `generate()`
    /// 2. Compile to ANE kernel
    /// 3. Run on tiny reference batch (batch_size=2, seq_len=4)
    /// 4. Compare ANE gradients vs CPU reference
    /// 5. Verify relative error < 1e-6
    fn validate(&self, config: &TransformerConfig) -> Result<()>;

    /// Operation name for debugging and logging
    ///
    /// # Returns
    /// Static string identifier (e.g., "rmsnorm_backward", "attention_backward")
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
///
/// Checks that generated MIL code contains required elements:
/// - Function declaration with correct name
/// - All expected inputs
/// - All expected outputs
/// - Return statement
///
/// # Arguments
/// * `mil_code` - The generated MIL code string
/// * `func_name` - Expected function name
/// * `expected_inputs` - List of expected input variable names
/// * `expected_outputs` - List of expected output variable names
///
/// # Returns
/// Ok(()) if all checks pass, error with details otherwise
pub fn validate_mil_structure(
    mil_code: &str,
    func_name: &str,
    expected_inputs: &[&str],
    expected_outputs: &[&str],
) -> Result<()> {
    // Check function declaration
    if !mil_code.contains(&format!("main {}", func_name))
        && !mil_code.contains(&format!("func {}", func_name))
    {
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
    if !mil_code.contains("return") {
        return Err(crate::ane::ANEError::CompileFailed(format!(
            "MIL for '{}' missing return statement",
            func_name
        ))
        .into());
    }

    // Check for IR schema declaration (#!irms6)
    if !mil_code.contains("#!irms6") {
        return Err(crate::ane::ANEError::CompileFailed(format!(
            "MIL for '{}' missing IR schema declaration",
            func_name
        ))
        .into());
    }

    Ok(())
}
