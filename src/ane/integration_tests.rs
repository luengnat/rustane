//! Comprehensive ANE Integration Tests
//!
//! These tests verify that the ANE (Apple Neural Engine) integration works correctly
//! on Apple Silicon hardware. Tests are marked with `#[ignore]` by default since they
//! require actual ANE hardware and the private AppleNeuralEngine framework.
//!
//! To run these tests on Apple Silicon:
//! ```bash
//! cargo test --lib ane_integration -- --ignored
//! ```

use crate::ane::{
    get_global_stats, reset_global_stats, ANECompileRequest, ANEError, ErrorCategory,
    ErrorDiagnostic, FallbackExecutor, FallbackStrategy, RetryConfig, RetryPolicy,
};
use crate::wrapper::{ANECompiler, ANERuntime};

/// Simple MIL program that performs element-wise addition
const ADD_MIL: &str = r#"
main add_tensors(a: tensor<32xf32>, b: tensor<32xf32>) -> (c: tensor<32xf32>) {
    let c = a + b;
    return (c);
}
"#;

/// MIL program with weights for matrix multiplication
const MATMUL_MIL: &str = r#"
main matmul(x: tensor<4x4xf32>) -> (y: tensor<4x4xf32>) {
    let weight = const_tensor<4x4xf32>(@model_path/weights/w.bin);
    let y = matmul(x, weight);
    return (y);
}
"#;

/// Test ANE runtime initialization on Apple Silicon
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_runtime_initialization() {
    // Initialize runtime
    let runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");
    assert!(
        ANERuntime::is_initialized(),
        "Runtime should be initialized"
    );

    // Verify singleton behavior
    let runtime2 = ANERuntime::init().expect("Failed to get existing runtime");
    assert!(
        std::ptr::eq(runtime, runtime2),
        "Runtime should be singleton"
    );

    // Check compile count
    let count = runtime.compile_count();
    assert!(count >= 0, "Compile count should be non-negative");
}

/// Test basic compilation with ANECompileRequest
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_compile_simple_add() {
    let runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");
    let initial_count = runtime.compile_count();

    // Compile simple addition program
    let request = ANECompileRequest::new(ADD_MIL, vec![32 * 4, 32 * 4], vec![32 * 4]);
    let result = request.compile();

    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    // Verify compile count incremented
    let new_count = runtime.compile_count();
    assert_eq!(
        new_count,
        initial_count + 1,
        "Compile count should increment by 1"
    );
}

/// Test compilation with weights
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_compile_with_weights() {
    let runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");
    let initial_count = runtime.compile_count();

    // Create weight data (4x4 matrix = 16 f32 values)
    let weights: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let weight_bytes: Vec<u8> = weights.iter().flat_map(|&f| f.to_ne_bytes()).collect();

    // Compile with weights
    let request = ANECompileRequest::new(MATMUL_MIL, vec![16 * 4], vec![16 * 4])
        .with_weight_bytes("@model_path/weights/w.bin", weight_bytes);

    let result = request.compile();
    assert!(
        result.is_ok(),
        "Compilation with weights failed: {:?}",
        result.err()
    );

    // Verify compile count incremented
    assert_eq!(runtime.compile_count(), initial_count + 1);
}

/// Test execution: write input, eval, read output
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_execute_simple_add() {
    let _runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");

    // Compile
    let request = ANECompileRequest::new(ADD_MIL, vec![32 * 4, 32 * 4], vec![32 * 4]);
    let mut executor = request.compile().expect("Compilation failed");

    // Prepare inputs
    let input_a: Vec<f32> = vec![1.0; 32];
    let input_b: Vec<f32> = vec![2.0; 32];
    let input_a_bytes: Vec<u8> = input_a.iter().flat_map(|&f| f.to_ne_bytes()).collect();
    let input_b_bytes: Vec<u8> = input_b.iter().flat_map(|&f| f.to_ne_bytes()).collect();

    // Write inputs
    executor
        .write_input(0, &input_a_bytes)
        .expect("Failed to write input 0");
    executor
        .write_input(1, &input_b_bytes)
        .expect("Failed to write input 1");

    // Execute
    executor.eval().expect("Execution failed");

    // Read output
    let mut output_bytes = vec![0u8; 32 * 4];
    executor
        .read_output(0, &mut output_bytes)
        .expect("Failed to read output");

    // Verify results: 1.0 + 2.0 = 3.0
    let output: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 3.0).abs() < 1e-5,
            "Output[{}] expected 3.0, got {}",
            i,
            val
        );
    }
}

/// Test compile count tracking and reset
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_compile_count_tracking() {
    let runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");
    let initial_count = runtime.compile_count();

    // Compile multiple kernels
    for i in 1..=5 {
        let request = ANECompileRequest::new(ADD_MIL, vec![32 * 4, 32 * 4], vec![32 * 4]);
        let _executor = request.compile().expect("Compilation failed");
        assert_eq!(
            runtime.compile_count(),
            initial_count + i,
            "Compile count should be {}",
            initial_count + i
        );
    }

    // Reset and verify
    runtime.reset_compile_count();
    assert_eq!(
        runtime.compile_count(),
        0,
        "Compile count should be 0 after reset"
    );
}

/// Test error handling: invalid MIL
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_compile_invalid_mil() {
    let _runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");

    let invalid_mil = "this is not valid MIL code";
    let request = ANECompileRequest::new(invalid_mil, vec![16], vec![16]);

    let result = request.compile();
    assert!(result.is_err(), "Should fail with invalid MIL");

    // Verify it's an ANE error
    let err = result.unwrap_err();
    let err_string = err.to_string();
    assert!(
        err_string.contains("compilation") || err_string.contains("failed"),
        "Error should indicate compilation failure: {}",
        err_string
    );
}

/// Test error handling: mismatched input sizes
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_execute_wrong_input_size() {
    let _runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");

    let request = ANECompileRequest::new(ADD_MIL, vec![32 * 4, 32 * 4], vec![32 * 4]);
    let mut executor = request.compile().expect("Compilation failed");

    // Try to write wrong size
    let wrong_data = vec![0u8; 16]; // Too small
    let result = executor.write_input(0, &wrong_data);

    assert!(result.is_err(), "Should fail with wrong input size");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("size") || err.to_string().contains("match"),
        "Error should mention size mismatch"
    );
}

/// Test error handling: out of bounds input index
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_execute_out_of_bounds() {
    let _runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");

    let request = ANECompileRequest::new(ADD_MIL, vec![32 * 4, 32 * 4], vec![32 * 4]);
    let mut executor = request.compile().expect("Compilation failed");

    let data = vec![0u8; 32 * 4];
    let result = executor.write_input(5, &data); // Index 5 doesn't exist

    assert!(result.is_err(), "Should fail with out of bounds");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("bounds") || err.to_string().contains("out of"),
        "Error should mention bounds"
    );
}

/// Test error diagnostic creation and analysis
#[test]
fn test_error_diagnostic_compilation() {
    let err = ANEError::CompileFailed("test error".to_string());
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert_eq!(diagnostic.category, ErrorCategory::Compilation);
    assert!(!diagnostic.recovery_suggestions.is_empty());
}

/// Test error diagnostic for hardware errors
#[test]
fn test_error_diagnostic_hardware() {
    let err = ANEError::FrameworkNotFound;
    let diagnostic = ErrorDiagnostic::from_error(err);

    assert_eq!(diagnostic.category, ErrorCategory::Resource);
}

/// Test retry policy configuration
#[test]
fn test_retry_policy_configuration() {
    use std::time::Duration;

    let config = RetryConfig {
        max_attempts: 5,
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(5),
        backoff_multiplier: 2.0,
        enable_batch_reduction: true,
        min_batch_fraction: 0.125,
        batch_reduction_factor: 0.5,
    };

    let policy = RetryPolicy::with_config(config).expect("Failed to create policy");

    // Test that policy was created successfully
    assert_eq!(policy.config().max_attempts, 5);
    assert!(policy.config().enable_batch_reduction);
}

/// Test fallback executor with ANE success
#[test]
fn test_fallback_executor_ane_success() {
    use crate::ane::FallbackResult;

    let mut executor = FallbackExecutor::with_strategy(FallbackStrategy::ANEWithCPUFallback);

    let result: FallbackResult<Vec<f32>> = executor.execute(
        || Ok(vec![1.0, 2.0, 3.0]), // ANE succeeds
        || panic!("CPU fallback should not be called"),
        "test_op",
    );

    assert!(result.is_success());
    assert_eq!(result.result(), Some(vec![1.0, 2.0, 3.0]));
}

/// Test fallback executor with ANE failure then CPU success
#[test]
fn test_fallback_executor_ane_failure() {
    use crate::ane::FallbackResult;

    let mut executor = FallbackExecutor::with_strategy(FallbackStrategy::ANEWithCPUFallback);

    let result: FallbackResult<Vec<f32>> = executor.execute(
        || Err(ANEError::EvalFailed("ANE failed".to_string())),
        || Ok(vec![4.0, 5.0, 6.0]), // CPU succeeds
        "test_op",
    );

    assert!(result.is_success());
    assert_eq!(result.result(), Some(vec![4.0, 5.0, 6.0]));
}

/// Test multi-weight compilation via ANECompiler
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_compiler_multi_weights() {
    let _runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");

    let mil = r#"
    main multi_weights(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
        let w1 = const_tensor<4xf32>(@model_path/weights/w1.bin);
        let w2 = const_tensor<4xf32>(@model_path/weights/w2.bin);
        let temp = x * w1;
        let y = temp + w2;
        return (y);
    }
    "#;

    let weight1: Vec<f32> = vec![2.0; 4];
    let weight2: Vec<f32> = vec![1.0; 4];
    let w1_bytes: Vec<u8> = weight1.iter().flat_map(|&f| f.to_ne_bytes()).collect();
    let w2_bytes: Vec<u8> = weight2.iter().flat_map(|&f| f.to_ne_bytes()).collect();

    let mut compiler = ANECompiler::new();
    let result = compiler.compile_multi(
        mil,
        &["@model_path/weights/w1.bin", "@model_path/weights/w2.bin"],
        &[&w1_bytes, &w2_bytes],
        &[w1_bytes.len(), w2_bytes.len()],
        &[16], // 4 f32 values
        &[16], // 4 f32 values
    );

    assert!(
        result.is_ok(),
        "Multi-weight compilation failed: {:?}",
        result.err()
    );
}

/// Test ANECompiler basic methods
#[test]
fn test_ane_compiler_basic() {
    let compiler = ANECompiler::new();
    assert_eq!(compiler.num_inputs(), 0);
    assert_eq!(compiler.num_outputs(), 0);
    assert_eq!(compiler.input_size(0), None);
    assert_eq!(compiler.output_size(0), None);
}

/// Test ANECompiler input/output size tracking after compilation
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_compiler_size_tracking() {
    let _runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");

    let mut compiler = ANECompiler::new();
    let result = compiler.compile_single(ADD_MIL, None, &[32 * 4, 32 * 4], &[32 * 4]);

    assert!(result.is_ok());
    assert_eq!(compiler.num_inputs(), 2);
    assert_eq!(compiler.num_outputs(), 1);
    assert_eq!(compiler.input_size(0), Some(32 * 4));
    assert_eq!(compiler.input_size(1), Some(32 * 4));
    assert_eq!(compiler.output_size(0), Some(32 * 4));
}

/// Test compile count doesn't increment on failed compilation
#[test]
#[ignore = "Requires ANE hardware"]
fn test_ane_failed_compile_no_count_increment() {
    let runtime = ANERuntime::init().expect("Failed to initialize ANE runtime");
    let initial_count = runtime.compile_count();

    // Try to compile invalid MIL
    let invalid_mil = "not valid MIL";
    let request = ANECompileRequest::new(invalid_mil, vec![16], vec![16]);
    let _ = request.compile();

    // Count should not have changed (or might not increment on failure)
    let final_count = runtime.compile_count();
    assert_eq!(
        final_count, initial_count,
        "Failed compilation should not increment count (or behavior changed)"
    );
}

/// Test executor with null kernel gives proper error
#[test]
fn test_executor_null_kernel_error() {
    use crate::wrapper::executor::ANEExecutor;
    use std::ptr;

    let mut executor = ANEExecutor::new(ptr::null_mut(), &[1024], &[512]).unwrap();

    let result = executor.eval();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("null") || err_msg.contains("Kernel"));
}

/// Test that ANE runtime is properly detected on Apple Silicon
#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_available_on_apple_silicon() {
    // On Apple Silicon, initialization should succeed if ANE framework is present
    let result = ANERuntime::init();

    // We don't assert success because the framework might not be available
    // in all build environments, but we verify it doesn't panic
    match result {
        Ok(_) => println!("ANE runtime initialized successfully"),
        Err(e) => println!("ANE runtime not available: {}", e),
    }
}

/// Test error category classification
#[test]
fn test_error_category_classification() {
    let errors = vec![
        (ANEError::FrameworkNotFound, ErrorCategory::Resource),
        (
            ANEError::CompileFailed("test".to_string()),
            ErrorCategory::Compilation,
        ),
        (
            ANEError::EvalFailed("test".to_string()),
            ErrorCategory::Runtime,
        ),
        (
            ANEError::IOSurfaceError("test".to_string()),
            ErrorCategory::Resource,
        ),
        (
            ANEError::InvalidShape {
                expected: "a".to_string(),
                got: "b".to_string(),
            },
            ErrorCategory::Data,
        ),
        (
            ANEError::WeightBlobError("test".to_string()),
            ErrorCategory::Resource,
        ),
        (
            ANEError::ConfigError("test".to_string()),
            ErrorCategory::Configuration,
        ),
    ];

    for (err, expected_category) in errors {
        let diagnostic = ErrorDiagnostic::from_error(err);
        assert_eq!(
            diagnostic.category, expected_category,
            "Error should have category {:?}",
            expected_category
        );
    }
}

/// Test retry policy batch size calculation
#[test]
fn test_retry_batch_size_calculation() {
    let config = RetryConfig {
        enable_batch_reduction: true,
        batch_reduction_factor: 0.5,
        min_batch_fraction: 0.125,
        ..Default::default()
    };

    // Test full batch (attempt 0)
    let size = config.batch_size_for_attempt(100, 0);
    assert_eq!(size, 100);

    // Test half batch (attempt 1)
    let size = config.batch_size_for_attempt(100, 1);
    assert_eq!(size, 50);

    // Test quarter batch (attempt 2)
    let size = config.batch_size_for_attempt(100, 2);
    assert_eq!(size, 25);

    // Test minimum batch (attempt 3 = 12.5% minimum)
    let size = config.batch_size_for_attempt(100, 3);
    assert_eq!(size, 12); // 100 * 0.125 = 12.5, floored to 12

    // Test beyond minimum - stays at minimum
    let size = config.batch_size_for_attempt(100, 10);
    assert_eq!(size, 12);
}

/// Test fallback statistics
#[test]
fn test_fallback_statistics() {
    // Reset stats
    reset_global_stats();
    let stats = get_global_stats();
    assert_eq!(stats.total_attempts, 0);

    // Stats are updated internally by FallbackExecutor
    // We verify the structure works
    assert_eq!(stats.ane_success_rate(), 0.0);
    assert_eq!(stats.fallback_rate(), 0.0);
}
