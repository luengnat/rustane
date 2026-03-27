//! ANE Error Handling Tests
//!
//! Tests for ANEError types, error diagnostics, and error aggregator.

use rustane::ane::{ANEError, ErrorAggregator, ErrorCategory, ErrorDiagnostic};

// ============================================================================
// TEST 1: ANEError Creation and Variants
// ============================================================================

#[test]
fn test_ane_error_framework_not_found() {
    let err = ANEError::FrameworkNotFound;
    assert!(err.to_string().contains("framework"));
}

#[test]
fn test_ane_error_compile_failed() {
    let err = ANEError::CompileFailed("MIL syntax error".to_string());
    assert!(err.to_string().contains("compilation"));
    assert!(err.to_string().contains("MIL syntax error"));
}

#[test]
fn test_ane_error_eval_failed() {
    let err = ANEError::EvalFailed("Runtime execution error".to_string());
    assert!(err.to_string().contains("eval"));
    assert!(err.to_string().contains("Runtime execution error"));
}

#[test]
fn test_ane_error_io_surface_error() {
    let err = ANEError::IOSurfaceError("Surface creation failed".to_string());
    assert!(err.to_string().contains("IOSurface"));
}

#[test]
fn test_ane_error_invalid_shape() {
    let err = ANEError::InvalidShape {
        expected: "128x256".to_string(),
        got: "64x128".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("128"));
    assert!(msg.contains("256"));
    assert!(msg.contains("64"));
    assert!(msg.contains("Shape mismatch"));
}

#[test]
fn test_ane_error_config_error() {
    let err = ANEError::ConfigError("Invalid configuration".to_string());
    assert!(err.to_string().contains("Config"));
}

#[test]
fn test_ane_error_weight_blob_error() {
    let err = ANEError::WeightBlobError("Quantization failed".to_string());
    assert!(err.to_string().contains("Weight blob"));
}

// ============================================================================
// TEST 2: ErrorCategory Classification
// ============================================================================

#[test]
fn test_error_category_variants() {
    let _categories = [
        ErrorCategory::Compilation,
        ErrorCategory::Runtime,
        ErrorCategory::Resource,
        ErrorCategory::Data,
        ErrorCategory::Configuration,
    ];
}

#[test]
fn test_error_category_name() {
    assert_eq!(ErrorCategory::Compilation.name(), "Compilation");
    assert_eq!(ErrorCategory::Runtime.name(), "Runtime");
    assert_eq!(ErrorCategory::Resource.name(), "Resource");
    assert_eq!(ErrorCategory::Data.name(), "Data");
    assert_eq!(ErrorCategory::Configuration.name(), "Configuration");
}

#[test]
fn test_error_category_severity() {
    // Configuration errors have highest severity
    assert_eq!(ErrorCategory::Configuration.severity(), 5);
    // Resource errors are high severity
    assert_eq!(ErrorCategory::Resource.severity(), 4);
    // Compilation is medium
    assert_eq!(ErrorCategory::Compilation.severity(), 3);
}

#[test]
fn test_error_category_is_recoverable() {
    // Runtime errors are recoverable
    assert!(ErrorCategory::Runtime.is_recoverable());
    assert!(ErrorCategory::Data.is_recoverable());

    // Compilation and resource errors are not recoverable
    assert!(!ErrorCategory::Compilation.is_recoverable());
    assert!(!ErrorCategory::Resource.is_recoverable());
    assert!(!ErrorCategory::Configuration.is_recoverable());
}

#[test]
fn test_error_category_from_ane_error() {
    let compilation_err = ANEError::CompileFailed("test".to_string());
    let diag = ErrorDiagnostic::from_error(compilation_err);
    assert_eq!(diag.category, ErrorCategory::Compilation);

    let exec_err = ANEError::EvalFailed("test".to_string());
    let diag = ErrorDiagnostic::from_error(exec_err);
    assert_eq!(diag.category, ErrorCategory::Runtime);

    let io_err = ANEError::IOSurfaceError("test".to_string());
    let diag = ErrorDiagnostic::from_error(io_err);
    assert_eq!(diag.category, ErrorCategory::Resource);

    let shape_err = ANEError::InvalidShape {
        expected: "128".to_string(),
        got: "64".to_string(),
    };
    let diag = ErrorDiagnostic::from_error(shape_err);
    assert_eq!(diag.category, ErrorCategory::Data);

    let config_err = ANEError::ConfigError("test".to_string());
    let diag = ErrorDiagnostic::from_error(config_err);
    assert_eq!(diag.category, ErrorCategory::Configuration);
}

// ============================================================================
// TEST 3: ErrorDiagnostic
// ============================================================================

#[test]
fn test_error_diagnostic_from_error() {
    let err = ANEError::CompileFailed("MIL parse error".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert_eq!(diagnostic.category, ErrorCategory::Compilation);
    assert!(!diagnostic.recovery_suggestions.is_empty());
    assert!(diagnostic.context.contains_key("error_type"));
}

#[test]
fn test_error_diagnostic_retry_recommended() {
    // Runtime errors should recommend retry
    let err = ANEError::EvalFailed("test".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err);
    assert!(diagnostic.retry_recommended);

    // Compilation errors should not recommend retry
    let err = ANEError::CompileFailed("test".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err);
    assert!(!diagnostic.retry_recommended);
}

#[test]
fn test_error_diagnostic_batch_reduction() {
    // Runtime errors may suggest batch reduction
    let err = ANEError::EvalFailed("test".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err);
    // Should have some reduction factor
    assert!(diagnostic.batch_reduction_factor.is_some());
}

#[test]
fn test_error_diagnostic_root_cause() {
    let err = ANEError::FrameworkNotFound;
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert!(!diagnostic.root_cause.is_empty());
    assert!(!diagnostic.recovery_suggestions.is_empty());
}

#[test]
fn test_error_diagnostic_context() {
    let err = ANEError::InvalidShape {
        expected: "128".to_string(),
        got: "64".to_string(),
    };
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert!(diagnostic.context.contains_key("error_type"));
    assert!(diagnostic.context.contains_key("timestamp"));
}

#[test]
fn test_error_diagnostic_with_operation() {
    let err = ANEError::EvalFailed("test".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err).with_operation("forward_pass");

    assert!(diagnostic.context.contains_key("operation"));
    assert_eq!(
        diagnostic.context.get("operation"),
        Some(&"forward_pass".to_string())
    );
}

#[test]
fn test_error_diagnostic_with_layer() {
    let err = ANEError::CompileFailed("test".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err)
        .with_operation("attention")
        .with_layer(3, "MultiHeadAttention");

    assert!(diagnostic.context.contains_key("layer_idx"));
    assert!(diagnostic.context.contains_key("layer_type"));
}

// ============================================================================
// TEST 4: ErrorAggregator
// ============================================================================

#[test]
fn test_error_aggregator_creation() {
    let aggregator = ErrorAggregator::new();
    assert_eq!(aggregator.error_count(), 0);
}

#[test]
fn test_error_aggregator_add_error() {
    let mut aggregator = ErrorAggregator::new();

    aggregator.add_error(ANEError::CompileFailed("test".to_string()));
    assert_eq!(aggregator.error_count(), 1);
}

#[test]
fn test_error_aggregator_multiple_errors() {
    let mut aggregator = ErrorAggregator::new();

    aggregator.add_error(ANEError::CompileFailed("test1".to_string()));
    aggregator.add_error(ANEError::CompileFailed("test2".to_string()));
    aggregator.add_error(ANEError::EvalFailed("test3".to_string()));

    assert_eq!(aggregator.error_count(), 3);

    let summary = aggregator.summary();
    assert!(summary.contains("3"));
}

#[test]
fn test_error_aggregator_by_category() {
    let mut aggregator = ErrorAggregator::new();

    aggregator.add_error(ANEError::CompileFailed("test".to_string()));
    aggregator.add_error(ANEError::CompileFailed("test".to_string()));
    aggregator.add_error(ANEError::EvalFailed("test".to_string()));

    let compilation_count = aggregator.errors_by_category(ErrorCategory::Compilation);
    let runtime_count = aggregator.errors_by_category(ErrorCategory::Runtime);

    assert_eq!(compilation_count, 2);
    assert_eq!(runtime_count, 1);
}

#[test]
fn test_error_aggregator_most_common_category() {
    let mut aggregator = ErrorAggregator::new();

    aggregator.add_error(ANEError::CompileFailed("test1".to_string()));
    aggregator.add_error(ANEError::EvalFailed("test2".to_string()));
    aggregator.add_error(ANEError::EvalFailed("test3".to_string()));

    let most_common = aggregator.most_common_category();
    assert_eq!(most_common, Some(ErrorCategory::Runtime));
}

#[test]
fn test_error_aggregator_are_errors_recoverable() {
    let mut aggregator = ErrorAggregator::new();

    // All recoverable
    aggregator.add_error(ANEError::EvalFailed("test1".to_string()));
    assert!(aggregator.are_errors_recoverable());

    // Add non-recoverable error
    aggregator.add_error(ANEError::CompileFailed("test2".to_string()));
    assert!(!aggregator.are_errors_recoverable());
}

#[test]
fn test_error_aggregator_summary() {
    let mut aggregator = ErrorAggregator::new();

    aggregator.add_error(ANEError::CompileFailed("test1".to_string()));
    aggregator.add_error(ANEError::EvalFailed("test2".to_string()));

    let summary = aggregator.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("Total Errors: 2"));
}

#[test]
fn test_error_aggregator_add_with_context() {
    let mut aggregator = ErrorAggregator::new();

    aggregator.add_error_with_context(
        ANEError::EvalFailed("test".to_string()),
        "forward_pass",
        Some(3),
        Some("Attention"),
    );

    assert_eq!(aggregator.error_count(), 1);
    // The summary should contain error count info
    let summary = aggregator.summary();
    assert!(summary.contains("Total Errors: 1"));
}

#[test]
fn test_error_aggregator_clear() {
    // Note: ErrorAggregator doesn't have a clear() method
    // Test that a new aggregator starts empty
    let new_aggregator = ErrorAggregator::new();
    assert_eq!(new_aggregator.error_count(), 0);
}

// ============================================================================
// TEST 5: Error Recovery Analysis
// ============================================================================

#[test]
fn test_error_category_recovery_steps() {
    // Verify that different error types get categorized correctly
    let compilation_err = ANEError::CompileFailed("MIL parse error".to_string());
    let diag_comp = ErrorDiagnostic::from_error(compilation_err);
    assert_eq!(diag_comp.category, ErrorCategory::Compilation);
    assert!(!diag_comp.retry_recommended);

    let exec_err = ANEError::EvalFailed("Kernel execution timeout".to_string());
    let diag_exec = ErrorDiagnostic::from_error(exec_err);
    assert_eq!(diag_exec.category, ErrorCategory::Runtime);
    assert!(diag_exec.retry_recommended);
}

#[test]
fn test_shape_mismatch_diagnostic() {
    let err = ANEError::InvalidShape {
        expected: "1024x2048".to_string(),
        got: "512x1024".to_string(),
    };
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert_eq!(diagnostic.category, ErrorCategory::Data);
    assert!(!diagnostic.recovery_suggestions.is_empty());
}

#[test]
fn test_config_error_diagnostic() {
    let err = ANEError::ConfigError("dim must be divisible by 128".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert_eq!(diagnostic.category, ErrorCategory::Configuration);
    assert!(!diagnostic.retry_recommended);
}

// ============================================================================
// TEST 6: Error Display and Formatting
// ============================================================================

#[test]
fn test_error_display_contains_context() {
    let err = ANEError::InvalidShape {
        expected: "1024x2048".to_string(),
        got: "512x1024".to_string(),
    };
    let msg = err.to_string();

    // Message should contain shape information
    assert!(msg.contains("1024"));
    assert!(msg.contains("2048"));
    assert!(msg.contains("512"));
}

#[test]
fn test_error_debug_format() {
    let err = ANEError::ConfigError("test config".to_string());
    let debug_msg = format!("{:?}", err);

    // Debug format should contain error type
    assert!(debug_msg.contains("ConfigError"));
}

#[test]
fn test_compile_error_display() {
    let err = ANEError::CompileFailed("unexpected token".to_string());
    let msg = err.to_string();

    assert!(msg.contains("compilation"));
    assert!(msg.contains("unexpected token"));
}

#[test]
fn test_eval_error_display() {
    let err = ANEError::EvalFailed("timeout after 30s".to_string());
    let msg = err.to_string();

    assert!(msg.contains("eval"));
    assert!(msg.contains("timeout"));
}

#[test]
fn test_framework_not_found_display() {
    let err = ANEError::FrameworkNotFound;
    let msg = err.to_string();

    assert!(msg.contains("ANE"));
    assert!(msg.contains("framework"));
}
