//! Error Handling and Recovery Example
//!
//! Demonstrates the comprehensive error handling, retry logic,
//! and graceful degradation strategies introduced in Phase 4.

use rustane::ane::{
    ANEError, ErrorDiagnostic, ErrorLog, ErrorReporter, ErrorSeverity,
    FallbackExecutor, FallbackStrategy, RetryConfig, RetryPolicy,
};

/// Example 1: Basic Error Diagnostics
///
/// Demonstrates how to analyze ANE errors and get recovery suggestions
fn example_error_diagnostics() {
    println!("=== Example 1: Error Diagnostics ===\n");

    // Simulate different types of errors
    let errors = vec![
        ANEError::EvalFailed("Out of memory".to_string()),
        ANEError::CompileFailed("Shape mismatch in MIL code".to_string()),
        ANEError::IOSurfaceError("Surface creation failed".to_string()),
        ANEError::ConfigError("dim not divisible by n_heads".to_string()),
    ];

    for error in errors {
        let diagnostic = ErrorDiagnostic::from_error(error.clone())
            .with_operation("backward_pass")
            .with_layer(0, "attention");

        println!("Error: {}", error);
        println!("Category: {:?}", diagnostic.category);
        println!("Severity: {:?}", ErrorSeverity::from_category(diagnostic.category));
        println!("Root Cause: {}", diagnostic.root_cause);
        println!("Recoverable: {}", diagnostic.retry_recommended);

        if let Some(factor) = diagnostic.batch_reduction_factor {
            println!("Suggested Batch Reduction: {:.0}%", (1.0 - factor) * 100.0);
        }

        println!("\nRecovery Suggestions:");
        for (i, suggestion) in diagnostic.recovery_suggestions.iter().enumerate() {
            println!("  {}. {}", i + 1, suggestion);
        }
        println!();
    }
}

/// Example 2: Automatic Retry with Batch Reduction
///
/// Demonstrates how retry policy automatically reduces batch size on failures
fn example_retry_with_batch_reduction() {
    println!("=== Example 2: Retry with Batch Reduction ===\n");

    let config = RetryConfig {
        max_attempts: 3,
        enable_batch_reduction: true,
        batch_reduction_factor: 0.5,
        min_batch_fraction: 0.125,
        ..Default::default()
    };

    let policy = RetryPolicy::with_config(config).unwrap();

    // Simulate an operation that fails twice then succeeds
    let mut attempt = 0;
    let result = policy.execute(|batch_fraction| -> Result<Vec<f32>, ANEError> {
        attempt += 1;
        println!("Attempt {} with batch fraction {:.2}", attempt, batch_fraction);

        if attempt < 3 {
            Err(ANEError::EvalFailed("Simulated failure".to_string()))
        } else {
            println!("Success!");
            Ok(vec![1.0, 2.0, 3.0])
        }
    });

    match result {
        rustane::ane::RetryResult::Success { result, attempts } => {
            println!("\nOperation succeeded after {} attempts", attempts);
            println!("Result: {:?}", result);
        }
        rustane::ane::RetryResult::Failure { total_attempts, .. } => {
            println!("\nOperation failed after {} attempts", total_attempts);
        }
    }
}

/// Example 3: Graceful CPU Fallback
///
/// Demonstrates automatic fallback from ANE to CPU when ANE fails
fn example_cpu_fallback() {
    println!("=== Example 3: Graceful CPU Fallback ===\n");

    let mut executor = FallbackExecutor::with_strategy(FallbackStrategy::ANEWithCPUFallback);

    // Simulate ANE failure with CPU fallback
    let result = executor.execute(
        || -> Result<Vec<f32>, ANEError> {
            println!("Attempting ANE execution...");
            Err(ANEError::EvalFailed("ANE unavailable".to_string()))
        },
        || -> Result<Vec<f32>, ANEError> {
            println!("Falling back to CPU execution...");
            Ok(vec![1.0, 2.0, 3.0, 4.0])
        },
        "matmul_backward",
    );

    println!("\nFallback Statistics:");
    println!("Total attempts: {}", executor.stats().total_attempts);
    println!("ANE successes: {}", executor.stats().ane_successes);
    println!("CPU fallbacks: {}", executor.stats().cpu_fallbacks);
    println!("Fallback rate: {:.1}%", executor.stats().fallback_rate());

    match result {
        rustane::ane::FallbackResult::ANESuccess(v) => {
            println!("\nResult from ANE: {:?}", v);
        }
        rustane::ane::FallbackResult::CPUFallback(v) => {
            println!("\nResult from CPU fallback: {:?}", v);
        }
        rustane::ane::FallbackResult::CompleteFailure(e) => {
            println!("\nComplete failure: {}", e);
        }
    }
}

/// Example 4: Per-Layer Failure Tracking
///
/// Demonstrates how to track failures per layer and disable ANE for problematic layers
fn example_per_layer_failure_tracking() {
    println!("=== Example 4: Per-Layer Failure Tracking ===\n");

    let mut reporter = ErrorReporter::new(true);

    // Simulate errors on different layers
    for layer_idx in 0..4 {
        let error = if layer_idx == 2 {
            ANEError::EvalFailed("Persistent failure on layer 2".to_string())
        } else {
            ANEError::EvalFailed(format!("Transient error on layer {}", layer_idx))
        };

        let diagnostic = ErrorDiagnostic::from_error(error)
            .with_layer(layer_idx, "attention")
            .with_operation("backward");

        reporter.report(&diagnostic, Some("layer_backward"));
    }

    println!("\nError Log Summary:");
    reporter.print_summary();
}

/// Example 5: Global Error Reporting
///
/// Demonstrates the global error reporter for application-wide tracking
fn example_global_error_reporting() {
    println!("=== Example 5: Global Error Reporting ===\n");

    // Create a local error log
    let mut error_log = ErrorLog::new();

    // Simulate errors from different parts of the application
    let errors = vec![
        ("forward_pass", ANEError::EvalFailed("OOM during forward".to_string())),
        ("backward_pass", ANEError::EvalFailed("Timeout during backward".to_string())),
        ("gradient_accum", ANEError::IOSurfaceError("Surface transfer failed".to_string())),
    ];

    for (operation, error) in errors {
        let diagnostic = ErrorDiagnostic::from_error(error).with_operation(operation);
        error_log.log_diagnostic(&diagnostic, Some(operation));
    }

    // Print summary
    println!("\nError Log Summary:");
    println!("{}", error_log.format_summary());
}

/// Example 6: Production Training Loop with Error Handling
///
/// Demonstrates a complete training loop with all error handling features
fn example_production_training_loop() {
    println!("=== Example 6: Production Training Loop ===\n");

    // This would be your actual model and data
    // let config = TransformerConfig::tiny();
    // let mut model = TransformerANE::new(&config).unwrap();
    // let dataloader = ...;

    println!("Training configuration:");
    println!("  - Retry policy: 3 attempts with batch reduction");
    println!("  - Fallback strategy: ANE → CPU");
    println!("  - Error logging: Enabled with verbose output");
    println!("  - Per-layer tracking: Disable ANE after 3 consecutive failures");
    println!();

    // Simulate training steps with error handling
    for step in 0..5 {
        println!("Step {}:", step);

        // In production, you would:
        // 1. Create retry policy for this step
        // 2. Execute forward pass with fallback
        // 3. Execute backward pass with fallback
        // 4. Report any errors
        // 5. Update per-layer failure statistics

        if step == 2 {
            println!("  ⚠️  ANE failed, using CPU fallback");
        } else {
            println!("  ✅ ANE execution successful");
        }
    }

    println!("\nTraining completed with fallback statistics:");
    println!("  - ANE success rate: 80%");
    println!("  - CPU fallback rate: 20%");
    println!("  - Total errors handled: 5");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ANE Error Handling and Recovery Examples\n");
    println!("This example demonstrates the production-ready error handling\n");
    println!("features introduced in Phase 4.\n");
    println!("===========================================================\n");

    example_error_diagnostics();
    println!("\n");

    example_retry_with_batch_reduction();
    println!("\n");

    example_cpu_fallback();
    println!("\n");

    example_per_layer_failure_tracking();
    println!("\n");

    example_global_error_reporting();
    println!("\n");

    example_production_training_loop();

    Ok(())
}
