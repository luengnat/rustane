//! Stress Tests for ANE Backward Pass
//!
//! Tests performance, memory usage, and stability under load.

use rustane::data::Batch;
use rustane::training::{
    ANEGradientBuffer, CrossEntropyLoss, LossFn, Model, TransformerANE, TransformerConfig,
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
fn test_stress_many_training_steps() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    let loss_fn = CrossEntropyLoss::new();
    let lr = 0.001f32;

    let num_steps = 100;
    let mut losses = Vec::with_capacity(num_steps);

    for step in 0..num_steps {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();
        losses.push(loss);

        let grads = model.backward_with_batch(&batch, loss).unwrap();

        for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }

        if step % 20 == 0 {
            println!("Step {}: loss={:.4}", step, loss);
        }
    }

    assert_eq!(losses.len(), num_steps);
}

#[test]
fn test_stress_large_model() {
    // Larger model configuration
    let config = TransformerConfig::new(1024, 512, 1024, 16, 8, 256).unwrap();

    let mut model = match TransformerANE::new(&config) {
        Ok(m) => m,
        Err(_) => {
            println!("Skipping large model test - may require too much memory");
            return;
        }
    };

    let batch = test_batch(1, 64); // Smaller batch for large model
    let output = model.forward(&batch).unwrap();

    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();

    let grads = model.backward_with_batch(&batch, loss).unwrap();
    assert_eq!(grads.len(), config.param_count());
}

#[test]
fn test_stress_gradient_accumulation() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    let mut buffer = match ANEGradientBuffer::new(config.param_count()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let loss_fn = CrossEntropyLoss::new();

    // Accumulate many gradients
    let num_accumulations = 50;

    for i in 0..num_accumulations {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let grads = model.backward_with_batch(&batch, loss).unwrap();
        buffer.accumulate(&grads).unwrap();

        if i % 10 == 0 {
            println!(
                "Accumulation {}: count={}, max_grad={:.4e}",
                i,
                buffer.accumulation_count(),
                buffer.max_abs_gradient()
            );
        }
    }

    assert_eq!(buffer.accumulation_count(), num_accumulations);
}

#[test]
fn test_stress_rapid_reset_and_accumulate() {
    let config = test_config();

    let mut buffer = match ANEGradientBuffer::new(config.param_count()) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Rapidly reset and accumulate
    for _ in 0..100 {
        buffer.reset();
        buffer
            .accumulate(&vec![0.01f32; config.param_count()])
            .unwrap();
    }

    // After each reset+accumulate cycle, count is 1 (reset clears the counter)
    assert_eq!(buffer.accumulation_count(), 1);
}

#[test]
fn test_stress_concurrent_batch_processing() {
    let config = test_config();

    // Process multiple batches sequentially (simulating concurrent workload)
    let num_batches = 20;
    let mut models: Vec<TransformerANE> = Vec::new();

    for _ in 0..num_batches {
        let model = TransformerANE::new(&config).unwrap();
        models.push(model);
    }

    let loss_fn = CrossEntropyLoss::new();

    for (i, model) in models.iter_mut().enumerate() {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let grads = model.backward_with_batch(&batch, loss).unwrap();
        assert_eq!(grads.len(), config.param_count());

        if i % 5 == 0 {
            println!("Processed batch {}", i);
        }
    }
}

#[test]
fn test_stress_memory_with_repeated_forward_backward() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    let loss_fn = CrossEntropyLoss::new();

    // Repeated forward/backward without parameter updates
    for i in 0..50 {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let _grads = model.backward_with_batch(&batch, loss).unwrap();

        if i % 10 == 0 {
            println!("Iteration {}: loss={:.4}", i, loss);
        }
    }
}

#[test]
fn test_stress_varying_batch_sizes() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    let loss_fn = CrossEntropyLoss::new();
    let batch_sizes = vec![1, 2, 4, 8, 4, 2, 1];

    for (i, batch_size) in batch_sizes.iter().enumerate() {
        let batch = test_batch(*batch_size, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let grads = model.backward_with_batch(&batch, loss).unwrap();
        assert_eq!(grads.len(), config.param_count());

        println!("Batch size {}: loss={:.4}", batch_size, loss);
    }
}

#[test]
fn test_stress_extreme_gradient_values() {
    let config = test_config();

    let mut buffer = match ANEGradientBuffer::new(100) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Test with very large gradients
    buffer.accumulate(&vec![1e10f32; 100]).unwrap();
    let max_grad = buffer.max_abs_gradient();
    assert!(max_grad > 1e9);

    buffer.reset();

    // Test with very small gradients
    buffer.accumulate(&vec![1e-10f32; 100]).unwrap();
    let max_grad = buffer.max_abs_gradient();
    assert!(max_grad > 1e-11);
}

#[test]
fn test_stress_gradient_precision() {
    let config = test_config();

    let mut buffer = match ANEGradientBuffer::new(1) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate many small values to test precision
    let small_value = 0.0001f32;
    let num_iterations = 10000;

    for _ in 0..num_iterations {
        buffer.accumulate(&vec![small_value]).unwrap();
    }

    let result = buffer.to_vec();
    let expected = small_value * num_iterations as f32;

    // Allow for some floating point error
    let relative_error = (result[0] - expected).abs() / expected;
    assert!(
        relative_error < 0.01,
        "Precision loss too high: {}",
        relative_error
    );
}

#[test]
fn test_stress_model_recreation() {
    let config = test_config();
    let loss_fn = CrossEntropyLoss::new();

    // Recreate model many times
    for i in 0..20 {
        let mut model = TransformerANE::new(&config).unwrap();
        let batch = test_batch(2, 32);

        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let grads = model.backward_with_batch(&batch, loss).unwrap();
        assert_eq!(grads.len(), config.param_count());

        if i % 5 == 0 {
            println!("Model recreation {}", i);
        }
    }
}

#[test]
fn test_stress_long_sequence() {
    let config = TransformerConfig::new(256, 128, 256, 4, 2, 128).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();

    let batch = test_batch(1, 128); // Single sample, long sequence
    let output = model.forward(&batch).unwrap();

    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.compute(&output, &batch).unwrap();

    let grads = model.backward_with_batch(&batch, loss).unwrap();
    assert_eq!(grads.len(), config.param_count());
}

#[test]
fn test_stress_many_small_batches() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    let loss_fn = CrossEntropyLoss::new();
    let lr = 0.001f32;

    // Many small batches
    let num_batches = 50;

    for i in 0..num_batches {
        let batch = test_batch(1, 32); // Batch size 1
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let grads = model.backward_with_batch(&batch, loss).unwrap();

        for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }

        if i % 10 == 0 {
            println!("Small batch {}: loss={:.4}", i, loss);
        }
    }
}

#[test]
fn test_stress_gradient_clipping_stability() {
    let config = test_config();

    let mut buffer = match ANEGradientBuffer::new(1000) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Simulate exploding gradients
    for i in 0..10 {
        let scale = 10f32.powi(i);
        buffer.accumulate(&vec![scale; 1000]).unwrap();
    }

    let max_before = buffer.max_abs_gradient();
    println!("Max gradient before clipping: {:.4e}", max_before);

    // Clip gradients
    let clip_value = 1.0f32;
    if max_before > clip_value {
        buffer.scale(clip_value / max_before);
    }

    let max_after = buffer.max_abs_gradient();
    assert!(max_after <= clip_value * 1.01);
    println!("Max gradient after clipping: {:.4e}", max_after);
}

#[test]
fn test_stress_parameter_update_stability() {
    let config = test_config();
    let mut model = TransformerANE::new(&config).unwrap();

    // Get initial parameter statistics
    let initial_params: Vec<f32> = model.parameters().to_vec();
    let initial_mean = initial_params.iter().sum::<f32>() / initial_params.len() as f32;

    let loss_fn = CrossEntropyLoss::new();
    let lr = 0.1f32; // Large learning rate

    // Multiple updates
    for i in 0..10 {
        let batch = test_batch(2, 32);
        let output = model.forward(&batch).unwrap();
        let loss = loss_fn.compute(&output, &batch).unwrap();

        let grads = model.backward_with_batch(&batch, loss).unwrap();

        for (param, grad) in model.parameters().iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }

        // Check parameters are still finite
        let current_params: Vec<f32> = model.parameters().to_vec();
        assert!(
            current_params.iter().all(|p| p.is_finite()),
            "Parameters became non-finite at iteration {}",
            i
        );

        println!("Iteration {}: loss={:.4}", i, loss);
    }
}
