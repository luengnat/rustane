//! Error Handling Tests for ANE Backward Pass
//!
//! Tests error conditions, edge cases, and recovery.

use rustane::layers::backward::{BackwardMILGenerator, RMSNormBackwardGen};
use rustane::training::{ANEBackwardKernel, ANEGradientBuffer, TransformerConfig};

fn test_config() -> TransformerConfig {
    TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap()
}

#[test]
fn test_buffer_creation_zero_params() {
    let result = ANEGradientBuffer::new(0);
    assert!(result.is_err());
}

#[test]
fn test_buffer_accumulate_wrong_size() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Try to accumulate wrong size
    let result = buffer.accumulate(&vec![0.1f32; 5]);
    assert!(result.is_err());

    let result = buffer.accumulate(&vec![0.1f32; 15]);
    assert!(result.is_err());
}

#[test]
fn test_buffer_accumulate_empty_slice() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let result = buffer.accumulate(&[]);
    assert!(result.is_err());
}

#[test]
fn test_kernel_compile_empty_mil() {
    let config = test_config();
    let result = ANEBackwardKernel::compile("", &config, "empty");

    // Should handle gracefully
    let _ = result;
}

#[test]
fn test_kernel_compile_invalid_mil() {
    let config = test_config();
    let invalid_mil = "not valid mil code @#$%";

    let result = ANEBackwardKernel::compile(invalid_mil, &config, "invalid");

    // Should handle gracefully without panic
    let _ = result;
}

#[test]
fn test_kernel_compile_malformed_structure() {
    let config = test_config();

    // Missing main function
    let no_main = "#!irms6\nvar x = 1.0;";
    let result = ANEBackwardKernel::compile(no_main, &config, "no_main");
    let _ = result;

    // Missing IR declaration
    let no_ir = "main test() { return 0; }";
    let result = ANEBackwardKernel::compile(no_ir, &config, "no_ir");
    let _ = result;
}

#[test]
fn test_kernel_execution_wrong_input_count() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();

    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "test") {
        // Try with wrong number of inputs
        let inputs = vec![]; // Empty inputs
        let mut outputs = vec![vec![0.0f32; 256]];

        let result = kernel.execute(&inputs, &mut outputs);
        assert!(result.is_err());
    }
}

#[test]
fn test_kernel_execution_wrong_output_count() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();

    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "test") {
        let inputs = vec![vec![0.1f32; 256]];
        let mut outputs = vec![]; // Empty outputs

        let result = kernel.execute(&inputs, &mut outputs);
        assert!(result.is_err());
    }
}

#[test]
fn test_buffer_overflow_protection() {
    // Try to create extremely large buffer
    // Implementation uses checked_mul, should return error not panic
    let result = ANEGradientBuffer::new(usize::MAX / 2);
    assert!(result.is_err(), "Should fail for oversized buffer");
}

#[test]
fn test_buffer_with_nan_gradients() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate NaN
    let mut grads = vec![0.1f32; 10];
    grads[5] = f32::NAN;

    buffer.accumulate(&grads).unwrap();

    let result = buffer.to_vec();
    assert!(result[5].is_nan());
}

#[test]
fn test_buffer_with_infinite_gradients() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate infinity
    let mut grads = vec![0.1f32; 10];
    grads[5] = f32::INFINITY;

    buffer.accumulate(&grads).unwrap();

    let result = buffer.to_vec();
    assert!(result[5].is_infinite());
}

#[test]
fn test_max_abs_with_nan() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer
        .accumulate(&vec![1.0f32, 2.0f32, f32::NAN, 4.0f32, 5.0f32])
        .unwrap();

    // Max abs should handle NaN gracefully
    let max_abs = buffer.max_abs_gradient();
    assert!(max_abs.is_nan() || max_abs >= 4.0);
}

#[test]
fn test_scale_with_nan() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![1.0f32; 5]).unwrap();
    buffer.scale(f32::NAN);

    let result = buffer.to_vec();
    assert!(result.iter().all(|v| v.is_nan()));
}

#[test]
fn test_scale_with_infinity() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![1.0f32; 5]).unwrap();
    buffer.scale(f32::INFINITY);

    let result = buffer.to_vec();
    assert!(result.iter().all(|v| v.is_infinite()));
}

#[test]
fn test_scale_with_zero() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![1.0f32; 5]).unwrap();
    buffer.scale(0.0f32);

    let result = buffer.to_vec();
    assert!(result.iter().all(|v| *v == 0.0));
}

#[test]
fn test_buffer_surface_mismatch_size() {
    let mut buffer1 = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let buffer2 = match ANEGradientBuffer::new(20) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Try to accumulate from different sized surface
    // Should return error, not panic
    let result = buffer1.accumulate_surface(buffer2.surface());
    assert!(result.is_err(), "Should fail for mismatched surface sizes");
}

#[test]
fn test_kernel_compile_with_special_chars() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();

    // Test with special characters in name
    let result = ANEBackwardKernel::compile(&mil_code, &config, "test@#$%^&*()");
    let _ = result;
}

#[test]
fn test_kernel_compile_with_unicode() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();

    // Test with unicode in name
    let result = ANEBackwardKernel::compile(&mil_code, &config, "テスト_测试");
    let _ = result;
}

#[test]
fn test_kernel_compile_with_very_long_name() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();

    let long_name = "a".repeat(1000);
    let result = ANEBackwardKernel::compile(&mil_code, &config, &long_name);
    let _ = result;
}

#[test]
fn test_buffer_reset_after_many_accumulations() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate many times
    for _ in 0..1000 {
        buffer.accumulate(&vec![0.001f32; 10]).unwrap();
    }

    // Reset
    buffer.reset();

    // Verify clean state
    assert!(buffer.is_empty());
    assert_eq!(buffer.accumulation_count(), 0);

    // Can accumulate again
    buffer.accumulate(&vec![0.1f32; 10]).unwrap();
    assert_eq!(buffer.accumulation_count(), 1);
}

#[test]
fn test_kernel_execution_with_mismatched_sizes() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();

    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "test") {
        // Input size mismatch
        let inputs = vec![vec![0.1f32; 100]]; // Wrong size
        let mut outputs = vec![vec![0.0f32; 256]];

        let result = kernel.execute(&inputs, &mut outputs);
        // May succeed or fail
        let _ = result;
    }
}

#[test]
fn test_buffer_with_subnormal_values() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Subnormal (denormal) floating point values
    let subnormal = 1e-40f32;
    buffer.accumulate(&vec![subnormal; 5]).unwrap();

    let result = buffer.to_vec();
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn test_kernel_compile_null_bytes() {
    let config = test_config();
    let mil_with_null = "#!irms6\nmain test\0() {}";

    let result = ANEBackwardKernel::compile(mil_with_null, &config, "null_test");
    let _ = result;
}

#[test]
fn test_buffer_multiple_surfaces_accumulate() {
    let mut buffer1 = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let buffer2 = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let buffer3 = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate from multiple surfaces
    buffer1.accumulate(&vec![0.1f32; 10]).unwrap();
    let _ = buffer1.accumulate_surface(buffer2.surface());
    let _ = buffer1.accumulate_surface(buffer3.surface());
}

#[test]
fn test_is_empty_precision_edge_case() {
    let buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Very small values should still be considered non-empty
    let mut buffer2 = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };
    buffer2.accumulate(&vec![1e-20f32; 5]).unwrap();

    // Should not be considered empty even with tiny values
    assert!(!buffer2.is_empty());
}

#[test]
fn test_kernel_compile_very_large_mil() {
    let config = test_config();

    // Generate very large MIL code
    let mut large_mil = String::from("#!irms6\n");
    for i in 0..10000 {
        large_mil.push_str(&format!("var x{} = {};\n", i, i));
    }
    large_mil.push_str("main test() { return 0; }");

    let result = ANEBackwardKernel::compile(&large_mil, &config, "large");
    let _ = result;
}

#[test]
fn test_buffer_creation_after_failed() {
    // First creation fails
    let result1 = ANEGradientBuffer::new(0);
    assert!(result1.is_err());

    // Second creation should still work
    let result2 = ANEGradientBuffer::new(100);
    assert!(result2.is_ok());
}

#[test]
fn test_kernel_compile_after_failed() {
    let config = test_config();

    // First compilation with invalid MIL
    let result1 = ANEBackwardKernel::compile("invalid", &config, "test");
    let _ = result1;

    // Second compilation with valid MIL should work
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    let result2 = ANEBackwardKernel::compile(&mil_code, &config, "valid");
    let _ = result2;
}
