//! RMSNorm backward pass MIL generator
//!
//! Generates MIL code for Root Mean Square Layer Normalization backward pass.
//!
//! # Forward Pass
//! ```text
//! y = w * x / RMS(x)
//! where RMS(x) = sqrt(mean(x^2) + eps)
//! ```
//!
//! # Backward Pass
//! Computes gradients:
//! - `d_x`: Gradient w.r.t. input activations
//! - `dw`: Gradient w.r.t. normalization weights
//!
//! # Mathematical Derivation
//!
//! Given forward: `y_i = w_i * x_i / rms`
//!
//! Where:
//! - `rms = sqrt(mean(x^2) + eps)`
//! - `mean(x^2) = sum(x_i^2) / dim`
//!
//! Gradients:
//! ```text
//! dL/dw_i = sum_over_positions((dL/dy_i) * (x_i / rms))
//!
//! dL/dx_i = (dL/dy_i * w_i / rms) -
//!           (sum_over_j(dL/dy_j * w_j * x_j / rms) * x_i) / (rms^3 * dim)
//! ```

use crate::training::TransformerConfig;
use crate::ane::Result;
use super::BackwardMILGenerator;

/// MIL generator for RMSNorm backward pass
pub struct RMSNormBackwardGen;

impl RMSNormBackwardGen {
    /// Create new RMSNorm backward MIL generator
    pub fn new() -> Self {
        RMSNormBackwardGen
    }

    /// Generate MIL code for RMSNorm backward operation
    ///
    /// # MIL Structure
    /// ```text
    /// Inputs:
    ///   - d_out: Gradient w.r.t. output [seq_len * dim]
    ///   - x: Input activations [seq_len * dim]
    ///   - w: Normalization weights [dim]
    ///
    /// Outputs:
    ///   - d_x: Gradient w.r.t. input [seq_len * dim]
    ///   - dw: Gradient w.r.t. weights [dim]
    /// ```
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let dim = config.hidden_dim;
        let eps = 1e-6f32;

        format!(r#"
#!irms6
schema rmsnorm_backward_schema {{
    input d_out: tensor<seq_lenxdimxf32> = Input()
    input x: tensor<seq_lenxdimxf32> = Input()
    input w: tensor<dimxf32> = Input()
    output d_x: tensor<seq_lenxdimxf32> = Output()
    output dw: tensor<dimxf32> = Output()
}}

main rmsnorm_backward(d_out: tensor<seq_lenxdimxf32>,
                     x: tensor<seq_lenxdimxf32>,
                     w: tensor<dimxf32>) -> (d_x: tensor<seq_lenxdimxf32>,
                                               dw: tensor<dimxf32>) {{
    // Reshape inputs for per-position processing
    let d_out_reshaped = reshape(d_out, shape=[seq_len, dim])
    let x_reshaped = reshape(x, shape=[seq_len, dim])

    // Initialize weight gradient accumulator
    let dw_init = const_zero(shape=[dim], dtype=float32)
    let mut dw_accum = dw_init

    // Initialize input gradient accumulator
    let d_x_init = const_zero(shape=[seq_len, dim], dtype=float32)
    let mut d_x_accum = d_x_init

    // Process each sequence position independently
    for pos in 0..seq_len {{
        // Extract position-specific data
        let x_pos = x_reshaped[pos]
        let d_out_pos = d_out_reshaped[pos]

        // Compute RMS for this position
        let x_sq = x_pos * x_pos
        let mean_sq = reduce_sum(x_sq, axes=[0]) / {dim}.0
        let rms = sqrt(mean_sq + {eps}.0)
        let rms_sq = rms * rms
        let rms_cube = rms_sq * rms

        // Normalized input
        let norm_x = x_pos / rms

        // Accumulate weight gradient: dL/dw += norm_x * dL/d_out
        let dw_contribution = norm_x * d_out_pos
        dw_accum = dw_accum + dw_contribution

        // Compute input gradient
        // First term: dL/dy * w / rms
        let weighted_grad = d_out_pos * w
        let first_term = weighted_grad / rms

        // Second term: mean(dL/dy * w * norm_x) * x / rms^3
        let weighted_grad_sum = reduce_sum(weighted_grad * norm_x, axes=[0])
        let second_term_scalar = weighted_grad_sum / (rms_cube * {dim}.0)
        let second_term = second_term_scalar * x_pos

        // dL/dx = first_term - second_term
        let d_x_contribution = first_term - second_term
        d_x_accum[pos] = d_x_contribution
    }}

    // Reshape output to flat tensor
    let d_x_flat = reshape(d_x_accum, shape=[seq_len * dim])

    // Return gradients
    return (d_x_flat, dw_accum)
}}
"#)
    }
}

impl Default for RMSNormBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for RMSNormBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        // TODO: Implement validation in Phase 3b
        // For now, return Ok to allow compilation to proceed
        // Validation will:
        // 1. Generate MIL code
        // 2. Compile to ANE kernel
        // 3. Run on tiny reference batch
        // 4. Compare against CPU reference from transformer_backward.rs
        // 5. Verify 1e-6 relative tolerance

        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "rmsnorm_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_backward_gen_creation() {
        let gen = RMSNormBackwardGen::new();
        assert_eq!(gen.operation_name(), "rmsnorm_backward");
    }

    #[test]
    fn test_rmsnorm_backward_gen_default() {
        let gen = RMSNormBackwardGen::default();
        assert_eq!(gen.operation_name(), "rmsnorm_backward");
    }

    #[test]
    fn test_rmsnorm_backward_generate_mil() {
        let gen = RMSNormBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config);

        assert!(mil_code.is_ok());
        let mil = mil_code.unwrap();
        assert!(mil.contains("rmsnorm_backward"));
        assert!(mil.contains("d_x"));
        assert!(mil.contains("dw"));
    }

    #[test]
    fn test_rmsnorm_backward_mil_structure() {
        let gen = RMSNormBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        // Verify MIL contains required sections
        assert!(mil_code.contains("schema"));
        assert!(mil_code.contains("input"));
        assert!(mil_code.contains("output"));
        assert!(mil_code.contains("main"));
        assert!(mil_code.contains("return"));

        // Verify mathematical operations
        assert!(mil_code.contains("sqrt"));
        assert!(mil_code.contains("reduce_sum"));
        assert!(mil_code.contains("reshape"));
    }
}
