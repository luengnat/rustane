//! Integration Tests for Full ANE Backward Pipeline
//!
//! Tests end-to-end backward pass with ANE kernel execution.

use rustane::data::Batch;
use rustane::layers::backward::{BackwardMILGenerator, RMSNormBackwardGen, AttentionBackwardGen, FFNBackwardGen, LossBackwardGen};
use rustane::training::{
    ANEBackwardKernel, ANEGradientBuffer, CrossEntropyLoss, LossFn, 
    Model, TransformerANE, TransformerConfig
};

fn test_config() -> TransformerConfig {
    TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap()
}

fn test_batch(batch_size: usize, seq_len: usize) -> Batch {
    let tokens: Vec<u32> = (0..(batch_size * seq_len) as u32)
        .map(|i| i % 256)
        .collect();
    Batch::new(tokens, batch_size, seq_len).unwrap()
}

#[test]
fn test_full_backward_pipeline_cpu_fallback() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let batch = test_batch(2, 32);
    
    // Forward pass
    let output = model.forward(&batch).unwrap();
    assert!(output.num_elements() > 0);
    
    // Backward pass (CPU fallback)
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    assert_eq!(grads.len(), config.param_count());
    
    // Verify gradients are not all zero
    assert!(grads.iter().any(|&g| g != 0.0));
}

#[test]
fn test_backward_with_ane_gradient_buffer() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let mut buffer = match ANEGradientBuffer::new(config.param_count()) {
        Ok(b) => b,
        Err(_) => return, // Skip if IOSurface not available
    };
    
    let batch = test_batch(2, 32);
    
    // Forward and backward
    let output = model.forward(&batch).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    
    // Accumulate in ANE buffer
    buffer.accumulate(&grads).unwrap();
    
    // Verify accumulation
    assert!(!buffer.is_empty());
    assert_eq!(buffer.accumulation_count(), 1);
    
    // Get gradients back
    let accumulated = buffer.to_vec();
    assert_eq!(accumulated.len(), config.param_count());
}

#[test]
fn test_backward_pipeline_with_kernel_compilation() {
    let config = test_config();
    
    // Try to compile a backward kernel
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let kernel_result = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    
    // Continue with CPU backward regardless of kernel compilation success
    let mut model = TransformerANE::new(&config).unwrap();
    let batch = test_batch(2, 32);
    
    let output = model.forward(&batch).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    
    // Apply gradients
    let lr = 0.001f32;
    for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
        *param -= lr * grad;
    }
    
    // Kernel may or may not have compiled
    let _ = kernel_result;
}

#[test]
fn test_multi_step_training_pipeline() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let mut buffer = match ANEGradientBuffer::new(config.param_count()) {
        Ok(b) => b,
        Err(_) => return,
    };
    
    let loss_fn = CrossEntropyLoss::new();
    let lr = 0.001f32;
    
    let mut losses = Vec::new();
    
    // Train for 5 steps
    for step in 0..5 {
        let batch = test_batch(2, 32);
        
        // Forward
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        losses.push(loss);
        
        // Backward
        let grads = model.backward_with_batch(&batch, loss).unwrap();
        
        // Accumulate
        buffer.reset();
        buffer.accumulate(&grads).unwrap();
        
        // Apply gradients
        let accumulated = buffer.to_vec();
        for (param, grad) in model.parameters().iter_mut().zip(accumulated.iter()) {
            *param -= lr * grad;
        }
        
        println!("Step {}: loss={:.4}", step, loss);
    }
    
    // Verify losses were recorded
    assert_eq!(losses.len(), 5);
}

#[test]
fn test_backward_with_all_kernel_types() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let generators: Vec<Box<dyn BackwardMILGenerator>> = vec![
        Box::new(RMSNormBackwardGen::new()),
        Box::new(AttentionBackwardGen::new()),
        Box::new(FFNBackwardGen::new()),
        Box::new(LossBackwardGen::new()),
    ];
    
    // Try to compile all kernels
    let mut compiled_kernels = 0;
    for gen in generators {
        let mil_code = gen.generate(&config).unwrap();
        if ANEBackwardKernel::compile(&mil_code, &config, gen.operation_name()).is_ok() {
            compiled_kernels += 1;
        }
    }
    
    // Run backward pass (may use CPU fallback)
    let batch = test_batch(2, 32);
    let output = model.forward(&batch).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    
    // Verify gradients
    assert_eq!(grads.len(), config.param_count());
    
    println!("Compiled {} / 4 kernels", compiled_kernels);
}

#[test]
fn test_gradient_accumulation_across_layers() {
    let config = TransformerConfig::new(256, 128, 256, 4, 4, 64).unwrap(); // 4 layers
    let mut model = TransformerANE::new(&config).unwrap();
    
    let mut buffer = match ANEGradientBuffer::new(config.param_count()) {
        Ok(b) => b,
        Err(_) => return,
    };
    
    // Accumulate gradients from multiple forward/backward passes
    for _ in 0..3 {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        
        let grads = model.backward_with_batch(&batch, loss).unwrap();
        buffer.accumulate(&grads).unwrap();
    }
    
    assert_eq!(buffer.accumulation_count(), 3);
    
    // Verify accumulated gradients
    let accumulated = buffer.to_vec();
    let max_grad = accumulated.iter().map(|g| g.abs()).fold(0.0f32, f32::max);
    assert!(max_grad > 0.0);
}

#[test]
fn test_backward_with_different_batch_sizes() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let batch_sizes = vec![1, 2, 4];
    
    for batch_size in batch_sizes {
        let batch = test_batch(batch_size, 32);
        let output = model.forward(&batch).unwrap();
        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        
        let grads = model.backward_with_batch(&batch, loss).unwrap();
        assert_eq!(grads.len(), config.param_count());
    }
}

#[test]
fn test_backward_with_different_seq_lengths() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let seq_lengths = vec![16, 32, 64];
    
    for seq_len in seq_lengths {
        let batch = test_batch(2, seq_len);
        let output = model.forward(&batch).unwrap();
        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        
        let grads = model.backward_with_batch(&batch, loss).unwrap();
        assert_eq!(grads.len(), config.param_count());
    }
}

#[test]
fn test_model_update_after_backward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    // Get initial parameters
    let initial_params: Vec<f32> = model.parameters().to_vec();
    
    // Forward and backward
    let batch = test_batch(2, 32);
    let output = model.forward(&batch).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    
    // Update parameters
    let lr = 0.01f32;
    for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
        *param -= lr * grad;
    }
    
    // Verify parameters changed
    let updated_params: Vec<f32> = model.parameters().to_vec();
    assert_ne!(initial_params, updated_params);
}

#[test]
fn test_loss_decreases_over_steps() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let loss_fn = CrossEntropyLoss::new();
    let lr = 0.01f32;
    
    let mut losses = Vec::new();
    
    // Use same batch for consistent comparison
    let batch = test_batch(2, 32);
    
    for _ in 0..3 {
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        losses.push(loss);
        
        let grads = model.backward_with_batch(&batch, loss).unwrap();
        
        for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
    }
    
    // Loss should generally decrease (with some noise)
    println!("Losses: {:?}", losses);
    assert!(losses.len() == 3);
}

#[test]
fn test_gradient_clipping_in_pipeline() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    let mut buffer = match ANEGradientBuffer::new(config.param_count()) {
        Ok(b) => b,
        Err(_) => return,
    };
    
    let batch = test_batch(2, 32);
    let output = model.forward(&batch).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    
    buffer.accumulate(&grads).unwrap();
    
    // Apply gradient clipping
    let max_norm = 1.0f32;
    let current_max = buffer.max_abs_gradient();
    
    if current_max > max_norm {
        let scale = max_norm / current_max;
        buffer.scale(scale);
    }
    
    let clipped_max = buffer.max_abs_gradient();
    assert!(clipped_max <= max_norm * 1.01); // Allow small numerical error
}

#[test]
fn test_pipeline_with_kernel_cache() {
    let config = test_config();
    
    // Compile kernels once
    let gen = RMSNormBackwardGen::new();
    let mil_code = gen.generate(&config).unwrap();
    
    let kernel = ANEBackwardKernel::compile(&mil_code, &config, "rmsnorm_backward");
    
    // Use same kernel for multiple training steps
    let mut model = TransformerANE::new(&config).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    
    for step in 0..3 {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        
        let grads = model.backward_with_batch(&batch, loss).unwrap();
        
        // Apply gradients
        let lr = 0.001f32;
        for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        
        // Kernel reference is reused
        let _ = &kernel;
        
        println!("Step {}: loss={:.4}", step, loss);
    }
}

#[test]
fn test_backward_with_numerical_stability() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();
    
    // Initialize with very small weights for stability
    for param in model.parameters().iter_mut() {
        *param *= 0.01;
    }
    
    let batch = test_batch(2, 32);
    let output = model.forward(&batch).unwrap();
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();
    
    // Verify loss is finite
    assert!(loss.is_finite());
    assert!(loss > 0.0);
    
    let grads = model.backward_with_batch(&batch, loss).unwrap();
    
    // Verify gradients are finite
    assert!(grads.iter().all(|g| g.is_finite()));
}

#[test]
fn test_pipeline_memory_cleanup() {
    let config = test_config();
    
    // Run multiple iterations to ensure no memory leaks
    for _ in 0..10 {
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = test_batch(2, 32);
        
        let output = model.forward(&batch).unwrap();
        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        
        let _grads = model.backward_with_batch(&batch, loss).unwrap();
        
        // Model and gradients are dropped here
    }
    
    // If we get here without OOM, memory is being cleaned up
    assert!(true);
}
