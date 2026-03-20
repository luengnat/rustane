//! Backward pass MIL generators for ANE
//!
//! This module provides MIL (Model Intermediate Language) code generators for
//! backward propagation (gradient computation) on Apple Neural Engine (ANE).
//!
//! # Architecture
//!
//! Each backward operation implements the `BackwardMILGenerator` trait, which defines:
//! - MIL code generation for the backward operation
//! - Validation against CPU reference implementations
//! - Operation identification for debugging
//!
//! # Components
//!
//! - **BackwardMILGenerator**: Core trait for all backward MIL generators
//! - **RMSNormBackwardGen**: RMSNorm layer normalization backward
//! - **AttentionBackwardGen**: Multi-head attention backward (dQ, dK, dV, dO)
//! - **FFNBackwardGen**: Feed-forward network backward
//! - **LossBackwardGen**: Cross-entropy loss backward
//!
//! # Usage
//!
//! ```ignore
//! use crate::layers::backward::{BackwardMILGenerator, RMSNormBackwardGen};
//!
//! let config = TransformerConfig::tiny();
//! let generator = RMSNormBackwardGen::new();
//!
//! // Generate MIL code
//! let mil_code = generator.generate(&config)?;
//!
//! // Validate against CPU reference
//! generator.validate(&config)?;
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
