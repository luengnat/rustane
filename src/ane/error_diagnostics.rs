//! Detailed ANE Error Diagnostics
//!
//! Provides comprehensive error analysis, categorization, and recovery suggestions
//! for ANE operations during training.

use crate::ane::ANEError;
use crate::training::TransformerConfig;
use std::collections::HashMap;
use std::fmt;

/// Error category for grouping and analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Compilation errors (MIL code generation, kernel compilation)
    Compilation,
    /// Runtime execution errors (ANE evaluation, kernel execution)
    Runtime,
    /// Resource errors (memory, IOSurface, framework availability)
    Resource,
    /// Data errors (shape mismatches, invalid inputs)
    Data,
    /// Configuration errors (invalid model parameters)
    Configuration,
}

impl ErrorCategory {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ErrorCategory::Compilation => "Compilation",
            ErrorCategory::Runtime => "Runtime",
            ErrorCategory::Resource => "Resource",
            ErrorCategory::Data => "Data",
            ErrorCategory::Configuration => "Configuration",
        }
    }

    /// Get severity level (1-5, where 5 is most severe)
    pub fn severity(&self) -> u8 {
        match self {
            ErrorCategory::Configuration => 5, // Cannot proceed without fixing
            ErrorCategory::Resource => 4,      // May require system-level changes
            ErrorCategory::Compilation => 3,   // Usually fixable in code
            ErrorCategory::Data => 2,          // Input preprocessing issue
            ErrorCategory::Runtime => 2,       // May be transient or recoverable
        }
    }

    /// Whether this error category is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            ErrorCategory::Runtime => true,        // Can retry with CPU fallback
            ErrorCategory::Data => true,           // Can fix input data
            ErrorCategory::Compilation => false,   // Need to fix MIL code
            ErrorCategory::Resource => false,      // Need more resources
            ErrorCategory::Configuration => false, // Need to fix config
        }
    }
}

/// Detailed diagnostic information for an ANE error
#[derive(Debug, Clone)]
pub struct ErrorDiagnostic {
    /// The original error
    pub error: ANEError,
    /// Error category
    pub category: ErrorCategory,
    /// Likely root cause
    pub root_cause: String,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<String>,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Whether retry is recommended
    pub retry_recommended: bool,
    /// Suggested batch size reduction factor (if retry recommended)
    pub batch_reduction_factor: Option<f32>,
}

impl ErrorDiagnostic {
    /// Create diagnostic from an ANE error
    pub fn from_error(error: ANEError) -> Self {
        let (category, root_cause, suggestions, retry, reduction) = analyze_error(&error);

        let mut context = HashMap::new();
        context.insert("error_type".to_string(), format!("{:?}", error));
        context.insert(
            "timestamp".to_string(),
            format!("{:?}", std::time::SystemTime::now()),
        );

        Self {
            error,
            category,
            root_cause,
            recovery_suggestions: suggestions,
            context,
            retry_recommended: retry,
            batch_reduction_factor: reduction,
        }
    }

    /// Add context information
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    /// Add operation context
    pub fn with_operation(self, operation: &str) -> Self {
        self.with_context("operation", operation)
    }

    /// Add layer context
    pub fn with_layer(self, layer_idx: usize, layer_type: &str) -> Self {
        self.with_context("layer_idx", &layer_idx.to_string())
            .with_context("layer_type", layer_type)
    }

    /// Add model configuration context
    pub fn with_config(self, config: &TransformerConfig) -> Self {
        self.with_context("model_dim", &config.dim.to_string())
            .with_context("model_layers", &config.n_layers.to_string())
            .with_context("seq_len", &config.seq_len.to_string())
    }

    /// Format as human-readable report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           ANE Error Diagnostic Report                      ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════╝\n\n");

        // Error summary
        report.push_str(&format!("📛 Error: {}\n", self.error));
        report.push_str(&format!(
            "📂 Category: {} (Severity: {}/5)\n",
            self.category.name(),
            self.category.severity()
        ));
        report.push_str(&format!("🔍 Root Cause: {}\n\n", self.root_cause));

        // Context
        if !self.context.is_empty() {
            report.push_str("📋 Context:\n");
            for (key, value) in &self.context {
                report.push_str(&format!("   • {}: {}\n", key, value));
            }
            report.push('\n');
        }

        // Recovery suggestions
        if !self.recovery_suggestions.is_empty() {
            report.push_str("💡 Recovery Suggestions:\n");
            for (i, suggestion) in self.recovery_suggestions.iter().enumerate() {
                report.push_str(&format!("   {}. {}\n", i + 1, suggestion));
            }
            report.push('\n');
        }

        // Retry recommendation
        if self.retry_recommended {
            report.push_str("🔄 Retry Recommended: Yes\n");
            if let Some(factor) = self.batch_reduction_factor {
                report.push_str(&format!(
                    "   → Reduce batch size by {:.0}%\n",
                    (1.0 - factor) * 100.0
                ));
            }
        } else {
            report.push_str("🔄 Retry Recommended: No (manual intervention required)\n");
        }

        report
    }
}

impl fmt::Display for ErrorDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.format_report())
    }
}

/// Analyze an error to determine category, root cause, and recovery options
fn analyze_error(error: &ANEError) -> (ErrorCategory, String, Vec<String>, bool, Option<f32>) {
    match error {
        ANEError::FrameworkNotFound => (
            ErrorCategory::Resource,
            "ANE framework not available on this system".to_string(),
            vec![
                "Verify this is an Apple Silicon Mac (M1/M2/M3 or later)".to_string(),
                "Check macOS version is 11.0 or later".to_string(),
                "Ensure ANE framework is properly installed".to_string(),
                "Consider using CPU-only training as fallback".to_string(),
            ],
            false,
            None,
        ),

        ANEError::CompileFailed(msg) => analyze_compile_error(msg),

        ANEError::EvalFailed(msg) => analyze_eval_error(msg),

        ANEError::IOSurfaceError(msg) => (
            ErrorCategory::Resource,
            "IOSurface operation failed - likely memory or resource issue".to_string(),
            vec![
                format!("IOSurface error: {}", msg),
                "Check available system memory".to_string(),
                "Reduce batch size or sequence length".to_string(),
                "Close other applications using GPU/ANE".to_string(),
                "Consider using CPU fallback for this operation".to_string(),
            ],
            true,
            Some(0.5),
        ),

        ANEError::InvalidShape { expected, got } => (
            ErrorCategory::Data,
            format!("Tensor shape mismatch: expected {}, got {}", expected, got),
            vec![
                "Verify input data preprocessing pipeline".to_string(),
                "Check batch collation parameters".to_string(),
                "Ensure model configuration matches data dimensions".to_string(),
                "Review data loader output shapes".to_string(),
            ],
            true,
            None,
        ),

        ANEError::WeightBlobError(msg) => (
            ErrorCategory::Resource,
            format!("Weight blob construction failed: {}", msg),
            vec![
                "Verify model parameters are within ANE limits".to_string(),
                "Check model size is not too large for available memory".to_string(),
                "Ensure weight tensors are properly initialized".to_string(),
                "Try reducing model size or using gradient checkpointing".to_string(),
            ],
            false,
            None,
        ),

        ANEError::ConfigError(msg) => (
            ErrorCategory::Configuration,
            format!("Invalid model configuration: {}", msg),
            vec![
                "Review TransformerConfig parameters".to_string(),
                "Ensure dim is divisible by n_heads".to_string(),
                "Verify all dimensions are positive".to_string(),
                "Check configuration constraints in documentation".to_string(),
            ],
            false,
            None,
        ),

        ANEError::HWXNotFound(msg) => (
            ErrorCategory::Resource,
            format!("HWX file not found: {}", msg),
            vec![
                "Verify HWX files exist in search paths".to_string(),
                "Run PyTorch→CoreML conversion script".to_string(),
                "Check file permissions".to_string(),
                "Use MIL compilation as fallback".to_string(),
            ],
            false,
            None,
        ),

        ANEError::InvalidHWX(msg) => (
            ErrorCategory::Data,
            format!("Invalid HWX file format: {}", msg),
            vec![
                "Verify HWX file is not corrupted".to_string(),
                "Check HWX matches target architecture (H16G for M4)".to_string(),
                "Re-extract HWX from CoreML model".to_string(),
                "Use MIL compilation as fallback".to_string(),
            ],
            false,
            None,
        ),

        ANEError::IOError(msg) => (
            ErrorCategory::Resource,
            format!("I/O error: {}", msg),
            vec![
                "Check file permissions".to_string(),
                "Verify disk space available".to_string(),
                "Check file path is valid".to_string(),
                "Retry operation".to_string(),
            ],
            true,
            None,
        ),
    }
}

/// Analyze compilation errors
fn analyze_compile_error(msg: &str) -> (ErrorCategory, String, Vec<String>, bool, Option<f32>) {
    let (root_cause, suggestions) = if msg.contains("missing") || msg.contains("expected") {
        (
            "MIL code generation error - incomplete or incorrect MIL structure".to_string(),
            vec![
                "Review MIL generator implementation for the failing layer".to_string(),
                "Check all required inputs and outputs are defined".to_string(),
                "Verify MIL syntax is correct (irms6, program, etc.)".to_string(),
                "Run validation suite before ANE execution".to_string(),
            ],
        )
    } else if msg.contains("shape") || msg.contains("dimension") {
        (
            "Tensor shape error in MIL code".to_string(),
            vec![
                "Verify tensor shapes in MIL match runtime dimensions".to_string(),
                "Check reshape operations have correct dimensions".to_string(),
                "Review sequence length and hidden dimension parameters".to_string(),
            ],
        )
    } else if msg.contains("memory") || msg.contains("buffer") {
        (
            "Memory allocation failure during compilation".to_string(),
            vec![
                "Reduce model size or batch dimensions".to_string(),
                "Free up system memory".to_string(),
                "Check for memory leaks in previous operations".to_string(),
            ],
        )
    } else {
        (
            format!("Compilation error: {}", msg),
            vec![
                "Review full error message for specific issue".to_string(),
                "Check MIL code generation logic".to_string(),
                "Verify all tensor operations are supported".to_string(),
            ],
        )
    };

    (
        ErrorCategory::Compilation,
        root_cause,
        suggestions,
        false,
        None,
    )
}

/// Analyze evaluation errors
fn analyze_eval_error(msg: &str) -> (ErrorCategory, String, Vec<String>, bool, Option<f32>) {
    let (root_cause, suggestions, retry, reduction) =
        if msg.contains("memory") || msg.contains("OOM") {
            (
                "Out of memory during ANE execution".to_string(),
                vec![
                    "Reduce batch size significantly".to_string(),
                    "Enable gradient accumulation to maintain effective batch size".to_string(),
                    "Use mixed precision training (FP16) to reduce memory".to_string(),
                    "Implement gradient checkpointing".to_string(),
                    "Fall back to CPU for this batch".to_string(),
                ],
                true,
                Some(0.5),
            )
        } else if msg.contains("timeout") || msg.contains("hang") {
            (
                "ANE operation timeout - kernel took too long".to_string(),
                vec![
                    "Reduce sequence length or model dimensions".to_string(),
                    "Check for infinite loops in MIL code".to_string(),
                    "Break operation into smaller chunks".to_string(),
                    "Use CPU fallback for this operation".to_string(),
                ],
                true,
                Some(0.75),
            )
        } else if msg.contains("shape") || msg.contains("dimension") {
            (
                "Runtime shape mismatch".to_string(),
                vec![
                    "Verify input shapes match expected dimensions".to_string(),
                    "Check batch preprocessing pipeline".to_string(),
                    "Review model configuration consistency".to_string(),
                ],
                true,
                None,
            )
        } else {
            (
                format!("ANE execution error: {}", msg),
                vec![
                    "Check ANE is available and not in use by another process".to_string(),
                    "Verify input data is valid and properly formatted".to_string(),
                    "Try CPU fallback to isolate the issue".to_string(),
                    "Review system logs for ANE-related errors".to_string(),
                ],
                true,
                Some(0.5),
            )
        };

    (
        ErrorCategory::Runtime,
        root_cause,
        suggestions,
        retry,
        reduction,
    )
}

/// Error aggregator for collecting and analyzing multiple errors
#[derive(Debug, Clone, Default)]
pub struct ErrorAggregator {
    errors: Vec<ErrorDiagnostic>,
    by_category: HashMap<ErrorCategory, usize>,
}

impl ErrorAggregator {
    /// Create new error aggregator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error to the aggregator
    pub fn add_error(&mut self, error: ANEError) {
        let diagnostic = ErrorDiagnostic::from_error(error);
        *self.by_category.entry(diagnostic.category).or_insert(0) += 1;
        self.errors.push(diagnostic);
    }

    /// Add error with context
    pub fn add_error_with_context(
        &mut self,
        error: ANEError,
        operation: &str,
        layer_idx: Option<usize>,
        layer_type: Option<&str>,
    ) {
        let mut diagnostic = ErrorDiagnostic::from_error(error).with_operation(operation);
        if let Some(idx) = layer_idx {
            diagnostic = diagnostic.with_layer(idx, layer_type.unwrap_or("unknown"));
        }
        *self.by_category.entry(diagnostic.category).or_insert(0) += 1;
        self.errors.push(diagnostic);
    }

    /// Get total error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get error count by category
    pub fn errors_by_category(&self, category: ErrorCategory) -> usize {
        *self.by_category.get(&category).unwrap_or(&0)
    }

    /// Get most common error category
    pub fn most_common_category(&self) -> Option<ErrorCategory> {
        self.by_category
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(cat, _)| *cat)
    }

    /// Check if errors are recoverable
    pub fn are_errors_recoverable(&self) -> bool {
        self.errors.iter().all(|e| e.category.is_recoverable())
    }

    /// Get summary report
    pub fn summary(&self) -> String {
        if self.errors.is_empty() {
            return "No errors recorded".to_string();
        }

        let mut summary = String::new();
        summary.push_str("╔════════════════════════════════════════════════════════════╗\n");
        summary.push_str("║              Error Aggregation Summary                    ║\n");
        summary.push_str("╚════════════════════════════════════════════════════════════╝\n\n");

        summary.push_str(&format!("📊 Total Errors: {}\n\n", self.errors.len()));

        // By category
        summary.push_str("📂 Errors by Category:\n");
        for (category, count) in self.by_category.iter() {
            let percentage = (*count as f32 / self.errors.len() as f32) * 100.0;
            summary.push_str(&format!(
                "   • {}: {} ({:.1}%)\n",
                category.name(),
                count,
                percentage
            ));
        }
        summary.push('\n');

        // Most common
        if let Some(cat) = self.most_common_category() {
            summary.push_str(&format!("🎯 Most Common Category: {}\n", cat.name()));
        }

        // Recoverability
        if self.are_errors_recoverable() {
            summary.push_str("✅ All errors are recoverable with retry or fallback\n");
        } else {
            summary.push_str("⚠️  Some errors require manual intervention\n");
        }

        summary
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[ErrorDiagnostic] {
        &self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_diagnostic_creation() {
        let error = ANEError::CompileFailed("test error".to_string());
        let diagnostic = ErrorDiagnostic::from_error(error);

        assert_eq!(diagnostic.category, ErrorCategory::Compilation);
        assert!(!diagnostic.recovery_suggestions.is_empty());
    }

    #[test]
    fn test_error_with_context() {
        let error = ANEError::EvalFailed("OOM".to_string());
        let diagnostic = ErrorDiagnostic::from_error(error)
            .with_operation("backward")
            .with_layer(0, "attention");

        assert_eq!(
            diagnostic.context.get("operation"),
            Some(&"backward".to_string())
        );
        assert_eq!(diagnostic.context.get("layer_idx"), Some(&"0".to_string()));
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new();

        aggregator.add_error(ANEError::CompileFailed("error1".to_string()));
        aggregator.add_error(ANEError::CompileFailed("error2".to_string()));
        aggregator.add_error(ANEError::EvalFailed("error3".to_string()));

        assert_eq!(aggregator.error_count(), 3);
        assert_eq!(aggregator.errors_by_category(ErrorCategory::Compilation), 2);
        assert_eq!(aggregator.errors_by_category(ErrorCategory::Runtime), 1);
    }

    #[test]
    fn test_error_category_severity() {
        assert_eq!(ErrorCategory::Configuration.severity(), 5);
        assert_eq!(ErrorCategory::Resource.severity(), 4);
        assert_eq!(ErrorCategory::Compilation.severity(), 3);
        assert_eq!(ErrorCategory::Data.severity(), 2);
        assert_eq!(ErrorCategory::Runtime.severity(), 2);
    }

    #[test]
    fn test_recoverability() {
        assert!(!ErrorCategory::Configuration.is_recoverable());
        assert!(!ErrorCategory::Resource.is_recoverable());
        assert!(!ErrorCategory::Compilation.is_recoverable());
        assert!(ErrorCategory::Data.is_recoverable());
        assert!(ErrorCategory::Runtime.is_recoverable());
    }
}
