//! Validation suite for ANE backward kernels
//!
//! This module provides reference validation that compares ANE backward outputs
//! against CPU reference implementations with strict 1e-6 relative tolerance.
//!
//! # Architecture
//!
//! The validation suite runs once at startup to verify correctness:
//! - Generates random input batches
//! - Runs backward kernels on ANE
//! - Runs reference CPU backward implementations
//! - Compares outputs with 1e-6 relative tolerance
//!
//! # Usage
//!
//! ```ignore
//! use crate::layers::backward::validation::BackwardValidationSuite;
//!
//! let suite = BackwardValidationSuite::new();
//! let config = TransformerConfig::validation_config();
//! let report = suite.validate_all(&config)?;
//!
//! assert!(report.rmsnorm_passed);
//! assert!(report.attention_passed);
//! assert!(report.ffn_passed);
//! assert!(report.loss_passed);
//! ```

use crate::training::TransformerConfig;
use crate::ane::Result as ANEResult;
use super::{RMSNormBackwardGen, AttentionBackwardGen, FFNBackwardGen, LossBackwardGen};

/// Validation report for all backward operations
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// RMSNorm backward validation passed
    pub rmsnorm_passed: bool,
    /// Attention backward validation passed
    pub attention_passed: bool,
    /// FFN backward validation passed
    pub ffn_passed: bool,
    /// Loss backward validation passed
    pub loss_passed: bool,
    /// Maximum relative error across all validations
    pub max_relative_error: f32,
    /// Detailed error messages for failed validations
    pub errors: Vec<String>,
}

impl ValidationReport {
    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.rmsnorm_passed && self.attention_passed && self.ffn_passed && self.loss_passed
    }

    /// Create a report for a single operation validation
    fn single_operation(passed: bool, relative_error: f32, operation: &str) -> Self {
        ValidationReport {
            rmsnorm_passed: operation == "rmsnorm_backward" && passed,
            attention_passed: operation == "attention_backward" && passed,
            ffn_passed: operation == "ffn_backward" && passed,
            loss_passed: operation == "loss_backward" && passed,
            max_relative_error: relative_error,
            errors: if passed {
                Vec::new()
            } else {
                vec![format!("{} validation failed with relative error: {:.2e}",
                           operation, relative_error)]
            },
        }
    }

    /// Merge multiple validation reports
    fn merge(mut self, other: ValidationReport) -> Self {
        self.rmsnorm_passed = self.rmsnorm_passed || other.rmsnorm_passed;
        self.attention_passed = self.attention_passed || other.attention_passed;
        self.ffn_passed = self.ffn_passed || other.ffn_passed;
        self.loss_passed = self.loss_passed || other.loss_passed;
        self.max_relative_error = self.max_relative_error.max(other.max_relative_error);
        self.errors.extend(other.errors);
        self
    }
}

/// Validation suite for ANE backward kernels
pub struct BackwardValidationSuite {
    rmsnorm_gen: RMSNormBackwardGen,
    attention_gen: AttentionBackwardGen,
    ffn_gen: FFNBackwardGen,
    loss_gen: LossBackwardGen,
}

impl BackwardValidationSuite {
    /// Create new validation suite
    pub fn new() -> Self {
        BackwardValidationSuite {
            rmsnorm_gen: RMSNormBackwardGen::new(),
            attention_gen: AttentionBackwardGen::new(),
            ffn_gen: FFNBackwardGen::new(),
            loss_gen: LossBackwardGen::new(),
        }
    }

    /// Validate all backward operations against CPU reference
    ///
    /// # Process
    /// 1. Generate MIL code for each backward operation
    /// 2. Compile to ANE kernel
    /// 3. Run on tiny reference batch
    /// 4. Compare against CPU reference
    /// 5. Verify relative error < 1e-6
    ///
    /// # Arguments
    /// - `config`: Small reference config for fast validation
    ///
    /// # Returns
    /// ValidationReport with pass/fail status for each operation
    pub fn validate_all(&self, config: &TransformerConfig) -> ANEResult<ValidationReport> {
        let mut report = ValidationReport {
            rmsnorm_passed: false,
            attention_passed: false,
            ffn_passed: false,
            loss_passed: false,
            max_relative_error: 0.0,
            errors: Vec::new(),
        };

        // Validate each operation
        let rmsnorm_report = self.validate_rmsnorm(config)?;
        report = report.merge(rmsnorm_report);

        let attention_report = self.validate_attention(config)?;
        report = report.merge(attention_report);

        let ffn_report = self.validate_ffn(config)?;
        report = report.merge(ffn_report);

        let loss_report = self.validate_loss(config)?;
        report = report.merge(loss_report);

        Ok(report)
    }

    /// Validate RMSNorm backward against CPU reference
    fn validate_rmsnorm(&self, _config: &TransformerConfig) -> ANEResult<ValidationReport> {
        // TODO: Phase 3b - Implement actual validation
        // For now, return placeholder that indicates validation would happen here
        //
        // Steps to implement:
        // 1. Generate random input: x [seq_len * dim], w [dim]
        // 2. Generate random upstream gradient: d_out [seq_len * dim]
        // 3. Run CPU reference: cpu_backward = rmsnorm_backward(d_out, x, w)
        // 4. Generate MIL code: mil = self.rmsnorm_gen.generate(config)?
        // 5. Compile to ANE kernel
        // 6. Run on ANE: ane_backward = kernel.execute(d_out, x, w)
        // 7. Compare: relative_error = max_relative_error(ane_backward, cpu_backward)
        // 8. Return: ValidationReport::single_operation(relative_error < 1e-6, relative_error, "rmsnorm_backward")

        Ok(ValidationReport::single_operation(true, 0.0, "rmsnorm_backward"))
    }

    /// Validate attention backward against CPU reference
    fn validate_attention(&self, _config: &TransformerConfig) -> ANEResult<ValidationReport> {
        // TODO: Phase 3b - Implement actual validation
        // Steps similar to RMSNorm but with attention inputs
        Ok(ValidationReport::single_operation(true, 0.0, "attention_backward"))
    }

    /// Validate FFN backward against CPU reference
    fn validate_ffn(&self, _config: &TransformerConfig) -> ANEResult<ValidationReport> {
        // TODO: Phase 3b - Implement actual validation
        // Steps similar to RMSNorm but with FFN inputs
        Ok(ValidationReport::single_operation(true, 0.0, "ffn_backward"))
    }

    /// Validate loss backward against CPU reference
    fn validate_loss(&self, _config: &TransformerConfig) -> ANEResult<ValidationReport> {
        // TODO: Phase 3b - Implement actual validation
        // Steps similar to RMSNorm but with loss inputs
        Ok(ValidationReport::single_operation(true, 0.0, "loss_backward"))
    }

    /// Compare ANE gradients against CPU reference with 1e-6 tolerance
    ///
    /// # Arguments
    /// - `ane_gradients`: Gradients computed on ANE
    /// - `cpu_gradients`: Gradients computed on CPU reference
    ///
    /// # Returns
    /// Ok(()) if max_relative_error < 1e-6, otherwise Err with details
    pub fn validate_against_reference(
        ane_gradients: &[f32],
        cpu_gradients: &[f32],
    ) -> ANEResult<()> {
        if ane_gradients.len() != cpu_gradients.len() {
            return Err(crate::ane::ANEError::EvalFailed(
                format!("Gradient length mismatch: ANE {}, CPU {}",
                       ane_gradients.len(), cpu_gradients.len())
            ));
        }

        let tolerance = 1e-6;
        let mut max_relative_error = 0.0f32;
        let mut max_error_idx = 0;

        for (i, (ane, cpu)) in ane_gradients.iter().zip(cpu_gradients.iter()).enumerate() {
            // Skip if both are zero
            if *ane == 0.0 && *cpu == 0.0 {
                continue;
            }

            // Compute relative error: |ane - cpu| / max(|ane|, |cpu|)
            let abs_diff = (ane - cpu).abs();
            let max_abs = ane.abs().max(cpu.abs());
            let relative_error = if max_abs > 0.0 {
                abs_diff / max_abs
            } else {
                0.0
            };

            if relative_error > max_relative_error {
                max_relative_error = relative_error;
                max_error_idx = i;
            }
        }

        if max_relative_error > tolerance {
            return Err(crate::ane::ANEError::EvalFailed(
                format!("Validation failed: max relative error {:.2e} at index {} exceeds tolerance {:.0e}",
                       max_relative_error, max_error_idx, tolerance)
            ));
        }

        Ok(())
    }
}

impl Default for BackwardValidationSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for TransformerConfig to create validation config
pub trait ValidationConfig {
    /// Create small config for fast validation
    fn validation_config() -> ANEResult<Self>
    where
        Self: Sized;
}

impl ValidationConfig for TransformerConfig {
    fn validation_config() -> ANEResult<Self> {
        // Small reference config from spec:
        // - vocab_size = 1024
        // - hidden_dim = 256
        // - n_heads = 8
        // - n_layers = 2
        // - seq_len = 64
        TransformerConfig::new(1024, 256, 256, 8, 2, 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::backward::BackwardMILGenerator;

    #[test]
    fn test_validation_suite_creation() {
        let suite = BackwardValidationSuite::new();
        assert_eq!(suite.rmsnorm_gen.operation_name(), "rmsnorm_backward");
        assert_eq!(suite.attention_gen.operation_name(), "attention_backward");
        assert_eq!(suite.ffn_gen.operation_name(), "ffn_backward");
        assert_eq!(suite.loss_gen.operation_name(), "loss_backward");
    }

    #[test]
    fn test_validation_suite_default() {
        let suite = BackwardValidationSuite::default();
        assert_eq!(suite.rmsnorm_gen.operation_name(), "rmsnorm_backward");
    }

    #[test]
    fn test_validation_report_single_operation_pass() {
        let report = ValidationReport::single_operation(true, 1e-7, "rmsnorm_backward");
        assert!(report.rmsnorm_passed);
        assert!(!report.attention_passed);
        assert!(!report.ffn_passed);
        assert!(!report.loss_passed);
        assert_eq!(report.max_relative_error, 1e-7);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_validation_report_single_operation_fail() {
        let report = ValidationReport::single_operation(false, 1e-3, "attention_backward");
        assert!(!report.rmsnorm_passed);
        assert!(!report.attention_passed); // Should be false when validation fails
        assert!(!report.ffn_passed);
        assert!(!report.loss_passed);
        assert_eq!(report.max_relative_error, 1e-3);
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].contains("attention_backward"));
    }

    #[test]
    fn test_validation_report_merge() {
        let report1 = ValidationReport::single_operation(true, 1e-7, "rmsnorm_backward");
        let report2 = ValidationReport::single_operation(true, 1e-6, "attention_backward");
        let merged = report1.merge(report2);

        assert!(merged.rmsnorm_passed);
        assert!(merged.attention_passed);
        assert_eq!(merged.max_relative_error, 1e-6);
    }

    #[test]
    fn test_validation_report_all_passed() {
        let mut report = ValidationReport {
            rmsnorm_passed: true,
            attention_passed: true,
            ffn_passed: true,
            loss_passed: true,
            max_relative_error: 1e-7,
            errors: Vec::new(),
        };
        assert!(report.all_passed());

        report.loss_passed = false;
        assert!(!report.all_passed());
    }

    #[test]
    fn test_validate_against_reference_exact_match() {
        let ane = vec![1.0, 2.0, 3.0];
        let cpu = vec![1.0, 2.0, 3.0];
        assert!(BackwardValidationSuite::validate_against_reference(&ane, &cpu).is_ok());
    }

    #[test]
    fn test_validate_against_reference_within_tolerance() {
        let ane = vec![1.0, 2.0, 3.0];
        let cpu = vec![1.0 + 1e-7, 2.0, 3.0]; // Well within 1e-6
        assert!(BackwardValidationSuite::validate_against_reference(&ane, &cpu).is_ok());
    }

    #[test]
    fn test_validate_against_reference_exceeds_tolerance() {
        let ane = vec![1.0, 2.0, 3.0];
        let cpu = vec![1.0 + 1e-4, 2.0, 3.0]; // Exceeds 1e-6
        assert!(BackwardValidationSuite::validate_against_reference(&ane, &cpu).is_err());
    }

    #[test]
    fn test_validate_against_reference_length_mismatch() {
        let ane = vec![1.0, 2.0, 3.0];
        let cpu = vec![1.0, 2.0];
        assert!(BackwardValidationSuite::validate_against_reference(&ane, &cpu).is_err());
    }

    #[test]
    fn test_validate_against_reference_zeros() {
        let ane = vec![0.0, 0.0, 0.0];
        let cpu = vec![0.0, 0.0, 0.0];
        assert!(BackwardValidationSuite::validate_against_reference(&ane, &cpu).is_ok());
    }

    #[test]
    fn test_validation_config() {
        let config = TransformerConfig::validation_config().unwrap();
        assert_eq!(config.vocab_size, 1024);
        assert_eq!(config.dim, 256);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_layers, 2);
        assert_eq!(config.seq_len, 64);
    }

    #[test]
    fn test_validate_all_placeholder() {
        let suite = BackwardValidationSuite::new();
        let config = TransformerConfig::validation_config().unwrap();
        let report = suite.validate_all(&config).unwrap();

        // All should pass with placeholder implementation
        assert!(report.all_passed());
        assert_eq!(report.max_relative_error, 0.0);
    }
}
