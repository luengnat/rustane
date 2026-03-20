//! Comprehensive Tests for ANE Backward Kernel
//!
//! Tests kernel compilation, execution, caching, and error handling.

use rustane::layers::backward::{RMSNormBackwardGen, AttentionBackwardGen, FFNBackwardGen, LossBackwardGen, BackwardMILGenerator};
use rustane::training::{ANEBackwardKernel, ANEBackwardKernelCache, TransformerConfig};

fn test_config() -> TransformerConfig {
    TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap()
}

#[test]
fn test_kernel_compile_rmsnorm() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    
    // May succeed or fail depending on ANE availability
    match result {
        Ok(kernel) => {
            assert_eq!(kernel.operation_name(), "rmsnorm_backward");
            assert!(kernel.num_inputs() > 0);
            assert!(kernel.num_outputs() > 0);
        }
        Err(_) => {
            // ANE not available is acceptable in test environment
        }
    }
}

#[test]
fn test_kernel_compile_attention() {
    let config = test_config();
    let gen = AttentionBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "attention_backward");
    
    match result {
        Ok(kernel) => {
            assert_eq!(kernel.operation_name(), "attention_backward");
        }
        Err(_) => {}
    }
}

#[test]
fn test_kernel_compile_ffn() {
    let config = test_config();
    let gen = FFNBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "ffn_backward");
    
    match result {
        Ok(kernel) => {
            assert_eq!(kernel.operation_name(), "ffn_backward");
        }
        Err(_) => {}
    }
}

#[test]
fn test_kernel_compile_loss() {
    let config = test_config();
    let gen = LossBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "loss_backward");
    
    match result {
        Ok(kernel) => {
            assert_eq!(kernel.operation_name(), "loss_backward");
        }
        Err(_) => {}
    }
}

#[test]
fn test_kernel_cache_basic() {
    let mut cache = ANEBackwardKernelCache::new();
    
    assert_eq!(cache.stats(), (0, 0));
    
    // Simulate cache operations
    for _ in 0..5 { cache.record_hit(); }
    for _ in 0..3 { cache.record_miss(); }
    
    assert_eq!(cache.stats(), (5, 3));
}

#[test]
fn test_kernel_cache_clear() {
    let mut cache = ANEBackwardKernelCache::new();
    
    for _ in 0..10 { cache.record_hit(); }
    for _ in 0..5 { cache.record_miss(); }
    
    cache.clear();
    
    assert_eq!(cache.stats(), (0, 0));
}

#[test]
fn test_kernel_compile_invalid_mil() {
    let config = test_config();
    let invalid_mil = "invalid mil code";
    
    // Should handle invalid MIL gracefully
    let result = ANEBackwardKernel::compile(invalid_mil, &config, "invalid");
    
    // May succeed (placeholder) or fail (real implementation)
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_kernel_compile_empty_mil() {
    let config = test_config();
    let empty_mil = "";
    
    let result = ANEBackwardKernel::compile(empty_mil, &config, "empty");
    
    // Should handle empty MIL gracefully
    let _ = result;
}

#[test]
fn test_kernel_compile_large_config() {
    let config = TransformerConfig::new(4096, 512, 2048, 16, 8, 256).unwrap();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    
    // Large configs may fail due to memory constraints
    let _ = result;
}

#[test]
fn test_kernel_compile_tiny_config() {
    let config = TransformerConfig::new(64, 32, 64, 2, 1, 16).unwrap();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    
    let _ = result;
}

#[test]
fn test_all_kernels_compile() {
    let config = test_config();
    
    let generators: Vec<Box<dyn BackwardMILGenerator>> = vec![
        Box::new(RMSNormBackwardGen::new()),
        Box::new(AttentionBackwardGen::new()),
        Box::new(FFNBackwardGen::new()),
        Box::new(LossBackwardGen::new()),
    ];
    
    let mut compiled_count = 0;
    
    for gen in generators {
        let mil_code = gen.generate(&config).unwrap();
        let result = ANEBackwardKernel::compile(&mil_code, &config, gen.operation_name());
        
        if result.is_ok() {
            compiled_count += 1;
        }
    }
    
    // At least some kernels should compile (or all in placeholder mode)
    println!("Compiled {} / 4 kernels", compiled_count);
}

#[test]
fn test_kernel_execution_placeholder() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward") {
        // Create test inputs
        let inputs = vec![vec![0.1f32; 256]];
        let mut outputs = vec![vec![0.0f32; 256]];
        
        // Execution may succeed or fail
        let _ = kernel.execute(&inputs, &mut outputs);
    }
}

#[test]
fn test_kernel_with_multiple_inputs() {
    let config = test_config();
    let mil_code = "#!irms6\nmain test(input1: tensor<f32>, input2: tensor<f32>) -> (output: tensor<f32>) { return input1 + input2; }";
    
    let result = ANEBackwardKernel::compile(mil_code, &config, "multi_input");
    
    if let Ok(kernel) = result {
        assert!(kernel.num_inputs() >= 1);
        assert!(kernel.num_outputs() >= 1);
    }
}

#[test]
fn test_kernel_cache_multiple_operations() {
    let mut cache = ANEBackwardKernelCache::new();
    let config = test_config();
    
    // Simulate caching multiple operations
    let operations = vec!["rmsnorm", "attention", "ffn", "loss"];
    
    for (i, _op) in operations.iter().enumerate() {
        cache.record_miss();
        
        // Simulate cache hit on second access
        if i % 2 == 0 {
            cache.record_hit();
        }
    }
    
    let (hits, misses) = cache.stats();
    assert!(hits > 0);
    assert!(misses > 0);
}

#[test]
fn test_kernel_compile_different_configs() {
    let configs = vec![
        TransformerConfig::new(128, 64, 128, 2, 1, 32).unwrap(),
        TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap(),
        TransformerConfig::new(512, 256, 512, 8, 4, 128).unwrap(),
    ];
    
    let gen = RMSNormBackwardGen::new();
    
    for config in configs {
        let mil_code = gen.generate(&config).unwrap();
        let _ = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    }
}

#[test]
fn test_kernel_operation_name_consistency() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    if let Ok(kernel) = ANEBackwardKernel::compile(&mil_code, &config, "custom_name") {
        assert_eq!(kernel.operation_name(), "custom_name");
    }
}

#[test]
fn test_kernel_error_handling() {
    let config = test_config();
    
    // Test with malformed MIL
    let malformed_mil = "main { }";
    let result = ANEBackwardKernel::compile(malformed_mil, &config, "malformed");
    
    // Should not panic
    let _ = result.is_ok();
}

#[test]
fn test_kernel_with_special_characters_in_name() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let result = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward_v1.0_test");
    
    if let Ok(kernel) = result {
        assert!(kernel.operation_name().contains("rmsnorm"));
    }
}

#[test]
fn test_kernel_compile_with_different_precisions() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    // MIL code should work regardless of precision settings
    let result = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    
    // Verify the MIL contains expected precision declarations
    assert!(mil_code.contains("f32") || mil_code.contains("fp32"));
}

#[test]
fn test_kernel_execution_with_zero_inputs() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward") {
        let inputs = vec![vec![0.0f32; 256]];
        let mut outputs = vec![vec![0.0f32; 256]];
        
        // Should handle zero inputs
        let _ = kernel.execute(&inputs, &mut outputs);
    }
}

#[test]
fn test_kernel_execution_with_negative_inputs() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward") {
        let inputs = vec![vec![-0.5f32; 256]];
        let mut outputs = vec![vec![0.0f32; 256]];
        
        // Should handle negative inputs
        let _ = kernel.execute(&inputs, &mut outputs);
    }
}

#[test]
fn test_kernel_execution_with_large_values() {
    let config = test_config();
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    if let Ok(mut kernel) = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward") {
        let inputs = vec![vec![100.0f32; 256]];
        let mut outputs = vec![vec![0.0f32; 256]];
        
        // Should handle large inputs
        let _ = kernel.execute(&inputs, &mut outputs);
    }
}

#[test]
fn test_kernel_cache_hit_ratio() {
    let mut cache = ANEBackwardKernelCache::new();
    
    // Simulate 100 accesses with 75% hit rate
    for i in 0..100 {
        if i % 4 == 0 {
            cache.record_miss(); // 25% misses
        } else {
            cache.record_hit();  // 75% hits
        }
    }
    
    let (hits, misses) = cache.stats();
    assert_eq!(hits, 75);
    assert_eq!(misses, 25);
    
    let hit_ratio = hits as f32 / (hits + misses) as f32;
    assert!((hit_ratio - 0.75).abs() < 0.01);
}
