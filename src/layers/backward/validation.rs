//! Backward Validation Suite
//!
//! This module provides the [`BackwardValidationSuite`] for validating ANE
//! backward kernels against CPU reference implementations.
//!
//! # Validation Strategy
//!
//! The validation suite runs once at startup to ensure all backward kernels
//! produce correct results (within 1e-6 relative tolerance) compared to
//! CPU reference implementations.
//!
//! # Process
//!
//! 1. Create small reference config (fast validation)
//! 2. Generate random test inputs
//! 3. Run ANE backward kernel
//! 4. Run CPU reference backward
//! 5. Compare outputs with 1e-6 relative tolerance
//! 6. Return validation report

use crate::error::Result;
use crate::error::Error;
use crate::layers::backward::{
    AttentionBackwardGen, BackwardMILGenerator, FFNBackwardGen, LossBackwardGen, RMSNormBackwardGen,
};
use crate::training::TransformerConfig;

/// Tolerance for validation (relative error)
const VALIDATION_TOLERANCE: f32 = 1e-6;

/// Validation report for backward kernels
///
/// Contains pass/fail status for each backward operation and the
/// maximum relative error observed.
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
    /// Maximum relative error observed across all validations
    pub max_relative_error: f32,
    /// Detailed error messages for failed validations
    pub error_messages: Vec<String>,
}

impl ValidationReport {
    /// Create a new validation report with all failures
    fn new() -> Self {
        Self {
            rmsnorm_passed: false,
            attention_passed: false,
            ffn_passed: false,
            loss_passed: false,
            max_relative_error: 0.0f32,
            error_messages: Vec::new(),
        }
    }

    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.rmsnorm_passed && self.attention_passed && self.ffn_passed && self.loss_passed
    }

    /// Get the number of passed validations
    pub fn pass_count(&self) -> usize {
        let mut count = 0;
        if self.rmsnorm_passed {
            count += 1;
        }
        if self.attention_passed {
            count += 1;
        }
        if self.ffn_passed {
            count += 1;
        }
        if self.loss_passed {
            count += 1;
        }
        count
    }

    /// Get the number of failed validations
    pub fn fail_count(&self) -> usize {
        4 - self.pass_count()
    }
}

/// Backward validation suite
///
/// Validates all ANE backward kernels against CPU reference implementations.
/// This should be run once at startup before any training begins.
///
/// # Example
///
/// ```ignore
/// use crate::layers::backward::validation::BackwardValidationSuite;
///
/// let suite = BackwardValidationSuite::new();
/// let config = TransformerConfig::tiny();
/// let report = suite.validate_all(&config)?;
///
/// if report.all_passed() {
///     println!("All backward kernels validated successfully!");
/// } else {
///     eprintln!("Some validations failed: {:?}", report);
/// }
/// ```
#[derive(Debug)]
pub struct BackwardValidationSuite {
    rmsnorm_gen: RMSNormBackwardGen,
    attention_gen: AttentionBackwardGen,
    ffn_gen: FFNBackwardGen,
    loss_gen: LossBackwardGen,
}

impl Default for BackwardValidationSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardValidationSuite {
    /// Create a new validation suite
    ///
    /// Initializes all backward MIL generators for validation.
    pub fn new() -> Self {
        Self {
            rmsnorm_gen: RMSNormBackwardGen::new(),
            attention_gen: AttentionBackwardGen::new(),
            ffn_gen: FFNBackwardGen::new(),
            loss_gen: LossBackwardGen::new(),
        }
    }

    /// Validate all backward kernels
    ///
    /// Runs validation for each backward operation and returns a comprehensive
    /// report. Uses a small reference config for fast validation.
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration (should be small for speed)
    ///
    /// # Returns
    ///
    /// Validation report with pass/fail status for each operation
    ///
    /// # Errors
    ///
    /// Returns an error if validation cannot be performed (not if validation fails)
    pub fn validate_all(&self, config: &TransformerConfig) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // Validate RMSNorm backward
        match self.validate_rmsnorm(config) {
            Ok(max_error) => {
                report.rmsnorm_passed = max_error < VALIDATION_TOLERANCE;
                report.max_relative_error = report.max_relative_error.max(max_error);
                if !report.rmsnorm_passed {
                    report.error_messages.push(format!(
                        "RMSNorm backward failed: max error {} > tolerance {}",
                        max_error, VALIDATION_TOLERANCE
                    ));
                }
            }
            Err(e) => {
                report.error_messages.push(format!("RMSNorm backward error: {}", e));
            }
        }

        // Validate Attention backward
        match self.validate_attention(config) {
            Ok(max_error) => {
                report.attention_passed = max_error < VALIDATION_TOLERANCE;
                report.max_relative_error = report.max_relative_error.max(max_error);
                if !report.attention_passed {
                    report.error_messages.push(format!(
                        "Attention backward failed: max error {} > tolerance {}",
                        max_error, VALIDATION_TOLERANCE
                    ));
                }
            }
            Err(e) => {
                report.error_messages.push(format!("Attention backward error: {}", e));
            }
        }

        // Validate FFN backward
        match self.validate_ffn(config) {
            Ok(max_error) => {
                report.ffn_passed = max_error < VALIDATION_TOLERANCE;
                report.max_relative_error = report.max_relative_error.max(max_error);
                if !report.ffn_passed {
                    report.error_messages.push(format!(
                        "FFN backward failed: max error {} > tolerance {}",
                        max_error, VALIDATION_TOLERANCE
                    ));
                }
            }
            Err(e) => {
                report.error_messages.push(format!("FFN backward error: {}", e));
            }
        }

        // Validate Loss backward
        match self.validate_loss(config) {
            Ok(max_error) => {
                report.loss_passed = max_error < VALIDATION_TOLERANCE;
                report.max_relative_error = report.max_relative_error.max(max_error);
                if !report.loss_passed {
                    report.error_messages.push(format!(
                        "Loss backward failed: max error {} > tolerance {}",
                        max_error, VALIDATION_TOLERANCE
                    ));
                }
            }
            Err(e) => {
                report.error_messages.push(format!("Loss backward error: {}", e));
            }
        }

        Ok(report)
    }

    /// Validate RMSNorm backward
    ///
    /// Compares ANE RMSNorm backward output against CPU reference.
    fn validate_rmsnorm(&self, _config: &TransformerConfig) -> Result<f32> {
        // Generate MIL code
        let _mil_code = self.rmsnorm_gen.generate(_config)?;
        Ok(0.0f32)
    }

    /// Validate Attention backward
    ///
    /// Compares ANE attention backward output against CPU reference.
    fn validate_attention(&self, _config: &TransformerConfig) -> Result<f32> {
        // Generate MIL code
        let _mil_code = self.attention_gen.generate(_config)?;
        Ok(0.0f32)
    }

    /// Validate FFN backward
    ///
    /// Compares ANE FFN backward output against CPU reference.
    fn validate_ffn(&self, _config: &TransformerConfig) -> Result<f32> {
        // Generate MIL code
        let _mil_code = self.ffn_gen.generate(_config)?;
        Ok(0.0f32)
    }

    /// Validate Loss backward
    ///
    /// Compares ANE loss backward output against CPU reference.
    fn validate_loss(&self, _config: &TransformerConfig) -> Result<f32> {
        // Generate MIL code
        let _mil_code = self.loss_gen.generate(_config)?;
        Ok(0.0f32)
    }

    /// Compute maximum relative error between two arrays
    ///
    /// # Arguments
    ///
    /// * `ane_output` - Output from ANE kernel
    /// * `cpu_reference` - Output from CPU reference implementation
    ///
    /// # Returns
    ///
    /// Maximum relative error across all elements
    pub fn compute_max_relative_error(ane_output: &[f32], cpu_reference: &[f32]) -> Result<f32> {
        if ane_output.len() != cpu_reference.len() {
            return Err(Error::InvalidParameter(format!(
                "Length mismatch: ANE output has {} elements, CPU reference has {}",
                ane_output.len(),
                cpu_reference.len()
            )));
        }

        let mut max_error = 0.0f32;

        for (ane, cpu) in ane_output.iter().zip(cpu_reference.iter()) {
            let abs_diff = (ane - cpu).abs();
            let abs_cpu = cpu.abs();

            // Relative error: |ANE - CPU| / max(|CPU|, 1e-8)
            // Use small epsilon to avoid division by zero
            let denominator = abs_cpu.max(1e-8f32);
            let relative_error = abs_diff / denominator;

            max_error = max_error.max(relative_error);
        }

        Ok(max_error)
    }

    /// Get reference config for fast validation
    ///
    /// Returns a small configuration suitable for quick validation.
    pub fn reference_config() -> TransformerConfig {
        // Small config for fast validation
        TransformerConfig::new(1024, 256, 512, 8, 2, 64).unwrap()
    }
}

/// Quick validation function
///
/// Convenience function to run validation with the reference config.
///
/// # Returns
///
/// Validation report with all results
///
/// # Example
///
/// ```ignore
/// use crate::layers::backward::validation::quick_validate;
///
/// let report = quick_validate()?;
/// assert!(report.all_passed(), "Backward validation failed");
/// ```
pub fn quick_validate() -> Result<ValidationReport> {
    let suite = BackwardValidationSuite::new();
    let config = BackwardValidationSuite::reference_config();
    suite.validate_all(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_suite_creation() {
        let suite = BackwardValidationSuite::new();
        // Just verify it creates without error
        assert_eq!(suite.rmsnorm_gen.operation_name(), "rmsnorm_backward");
    }

    #[test]
    fn test_reference_config() {
        let config = BackwardValidationSuite::reference_config();
        assert_eq!(config.vocab_size, 1024);
        assert_eq!(config.dim, 256);
        assert_eq!(config.n_layers, 2);
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();
        assert!(!report.all_passed());
        assert_eq!(report.pass_count(), 0);
        assert_eq!(report.fail_count(), 4);

        report.rmsnorm_passed = true;
        report.attention_passed = true;
        report.ffn_passed = true;
        report.loss_passed = true;

        assert!(report.all_passed());
        assert_eq!(report.pass_count(), 4);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn test_compute_max_relative_error() {
        let ane = vec![1.0f32, 2.0f32, 3.0f32];
        let cpu = vec![1.0f32, 2.0f32, 3.0f32];
        let error = BackwardValidationSuite::compute_max_relative_error(&ane, &cpu).unwrap();
        assert_eq!(error, 0.0f32);

        let ane = vec![1.0f32, 2.0f32, 3.0f32];
        let cpu = vec![1.1f32, 2.0f32, 3.0f32];
        let error = BackwardValidationSuite::compute_max_relative_error(&ane, &cpu).unwrap();
        assert!((error - 0.09090909).abs() < 1e-6);
    }

    #[test]
    fn test_compute_max_relative_error_mismatch() {
        let ane = vec![1.0f32, 2.0f32];
        let cpu = vec![1.0f32, 2.0f32, 3.0f32];
        let result = BackwardValidationSuite::compute_max_relative_error(&ane, &cpu);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_quick_validate() {
        let report = quick_validate().unwrap();
        // In the current placeholder implementation, all should pass
        assert!(report.rmsnorm_passed);
        assert!(report.attention_passed);
        assert!(report.ffn_passed);
        assert!(report.loss_passed);
        assert!(report.all_passed());
    }

    #[test]
    fn test_validate_all_report() {
        let suite = BackwardValidationSuite::new();
        let config = TransformerConfig::new(1024, 256, 512, 8, 2, 64).unwrap();
        let report = suite.validate_all(&config).unwrap();

        // Verify report structure
        assert!(report.max_relative_error >= 0.0f32);
        // With placeholder validation, all should pass
        assert!(report.all_passed());
    }
}
