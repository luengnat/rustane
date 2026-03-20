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

pub mod rmsnorm_backward_gen;
pub mod attention_backward_gen;
pub mod ffn_backward_gen;
pub mod loss_backward_gen;
pub mod validation;

use crate::training::TransformerConfig;
use crate::ane::Result;

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
pub use rmsnorm_backward_gen::RMSNormBackwardGen;
pub use attention_backward_gen::AttentionBackwardGen;
pub use ffn_backward_gen::FFNBackwardGen;
pub use loss_backward_gen::LossBackwardGen;

// Re-export validation
pub use validation::{BackwardValidationSuite, ValidationReport};
