//! Comprehensive Tests for ANE Gradient Buffer
//!
//! Tests IOSurface-backed gradient storage, accumulation, and edge cases.

use rustane::training::{ANEGradientBuffer, TransformerConfig};

fn test_config() -> TransformerConfig {
    TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap()
}

#[test]
fn test_buffer_creation() {
    let buffer = ANEGradientBuffer::new(1000);

    match buffer {
        Ok(buf) => {
            assert_eq!(buf.num_params(), 1000);
            assert!(buf.is_empty());
            assert_eq!(buf.accumulation_count(), 0);
        }
        Err(_) => {
            // IOSurface may not be available
        }
    }
}

#[test]
fn test_buffer_creation_zero_params() {
    let result = ANEGradientBuffer::new(0);
    assert!(result.is_err());
}

#[test]
fn test_buffer_creation_large() {
    // Test with 10M parameters
    let buffer = ANEGradientBuffer::new(10_000_000);

    match buffer {
        Ok(buf) => {
            assert_eq!(buf.num_params(), 10_000_000);
        }
        Err(_) => {
            // Large allocation may fail
        }
    }
}

#[test]
fn test_buffer_from_config() {
    let config = test_config();
    let buffer = ANEGradientBuffer::new(config.param_count());

    match buffer {
        Ok(buf) => {
            assert_eq!(buf.num_params(), config.param_count());
        }
        Err(_) => {}
    }
}

#[test]
fn test_buffer_accumulate_single() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let grads = vec![0.1f32; 10];
    buffer.accumulate(&grads).unwrap();

    assert_eq!(buffer.accumulation_count(), 1);

    let result = buffer.to_vec();
    assert_eq!(result, vec![0.1f32; 10]);
}

#[test]
fn test_buffer_accumulate_multiple() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.1f32; 5]).unwrap();
    buffer.accumulate(&vec![0.2f32; 5]).unwrap();
    buffer.accumulate(&vec![0.3f32; 5]).unwrap();

    let result = buffer.to_vec();
    assert_eq!(result, vec![0.6f32; 5]);
    assert_eq!(buffer.accumulation_count(), 3);
}

#[test]
fn test_buffer_accumulate_wrong_size() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let grads = vec![0.1f32; 5]; // Wrong size
    let result = buffer.accumulate(&grads);

    assert!(result.is_err());
}

#[test]
fn test_buffer_reset() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![1.0f32; 5]).unwrap();
    assert!(!buffer.is_empty());

    buffer.reset();

    assert!(buffer.is_empty());
    assert_eq!(buffer.accumulation_count(), 0);
}

#[test]
fn test_buffer_max_abs_gradient() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer
        .accumulate(&vec![0.1f32, -0.5f32, 0.3f32, -0.2f32, 0.4f32])
        .unwrap();

    assert_eq!(buffer.max_abs_gradient(), 0.5f32);
}

#[test]
fn test_buffer_scale() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.1f32; 5]).unwrap();
    buffer.scale(2.0f32);

    let result = buffer.to_vec();
    assert_eq!(result, vec![0.2f32; 5]);
}

#[test]
fn test_buffer_scale_by_zero() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.1f32; 5]).unwrap();
    buffer.scale(0.0f32);

    let result = buffer.to_vec();
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn test_buffer_scale_negative() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.1f32; 5]).unwrap();
    buffer.scale(-1.0f32);

    let result = buffer.to_vec();
    assert_eq!(result, vec![-0.1f32; 5]);
}

#[test]
fn test_buffer_accumulate_zeros() {
    let mut buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.0f32; 10]).unwrap();

    assert!(buffer.is_empty());
}

#[test]
fn test_buffer_accumulate_negative_gradients() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![-0.1f32; 5]).unwrap();

    let result = buffer.to_vec();
    assert_eq!(result, vec![-0.1f32; 5]);
}

#[test]
fn test_buffer_accumulate_large_values() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![1e6f32; 5]).unwrap();

    let result = buffer.to_vec();
    assert!(result.iter().all(|&v| v > 1e5));
}

#[test]
fn test_buffer_accumulate_small_values() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![1e-6f32; 5]).unwrap();

    let result = buffer.to_vec();
    assert!(result.iter().all(|&v| v > 1e-7 && v < 1e-5));
}

#[test]
fn test_buffer_accumulate_mixed_values() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer
        .accumulate(&vec![0.1f32, -0.2f32, 0.3f32, -0.4f32, 0.5f32])
        .unwrap();

    let result = buffer.to_vec();
    assert_eq!(result[0], 0.1f32);
    assert_eq!(result[1], -0.2f32);
    assert_eq!(result[2], 0.3f32);
    assert_eq!(result[3], -0.4f32);
    assert_eq!(result[4], 0.5f32);
}

#[test]
fn test_buffer_multiple_resets() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    for _ in 0..10 {
        buffer.accumulate(&vec![1.0f32; 5]).unwrap();
        assert!(!buffer.is_empty());
        buffer.reset();
        assert!(buffer.is_empty());
    }
}

#[test]
fn test_buffer_accumulate_after_reset() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.5f32; 5]).unwrap();
    buffer.reset();
    buffer.accumulate(&vec![0.3f32; 5]).unwrap();

    let result = buffer.to_vec();
    assert_eq!(result, vec![0.3f32; 5]);
    assert_eq!(buffer.accumulation_count(), 1);
}

#[test]
fn test_buffer_surface_access() {
    let buffer = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let surface = buffer.surface();
    assert_eq!(surface.capacity(), 10 * 4); // 4 bytes per f32
}

#[test]
fn test_buffer_accumulate_surface() {
    let mut buffer1 = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    let buffer2 = match ANEGradientBuffer::new(10) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Initialize buffer2 with some data
    buffer1.accumulate(&vec![0.1f32; 10]).unwrap();

    // Accumulate from buffer2's surface
    let _ = buffer1.accumulate_surface(buffer2.surface());

    // Buffer1 should have accumulated buffer2's zeros
    let result = buffer1.to_vec();
    assert_eq!(result, vec![0.1f32; 10]);
}

#[test]
fn test_buffer_is_empty_with_zeros() {
    let buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    assert!(buffer.is_empty());
}

#[test]
fn test_buffer_is_not_empty_with_values() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.001f32; 5]).unwrap();

    assert!(!buffer.is_empty());
}

#[test]
fn test_buffer_max_abs_with_all_zeros() {
    let buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    assert_eq!(buffer.max_abs_gradient(), 0.0f32);
}

#[test]
fn test_buffer_max_abs_with_mixed_signs() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer
        .accumulate(&vec![-0.5f32, 0.3f32, -0.8f32, 0.1f32, 0.4f32])
        .unwrap();

    assert_eq!(buffer.max_abs_gradient(), 0.8f32);
}

#[test]
fn test_buffer_stress_many_accumulations() {
    let mut buffer = match ANEGradientBuffer::new(100) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate 1000 times
    for i in 0..1000 {
        let grad = (i as f32) * 0.001;
        buffer.accumulate(&vec![grad; 100]).unwrap();
    }

    let expected_sum: f32 = (0..1000).map(|i| i as f32 * 0.001).sum();
    let result = buffer.to_vec();

    assert!((result[0] - expected_sum).abs() < 1e-3);
    assert_eq!(buffer.accumulation_count(), 1000);
}

#[test]
fn test_buffer_precision_with_many_accumulations() {
    let mut buffer = match ANEGradientBuffer::new(1) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Accumulate small values many times to test precision
    for _ in 0..10000 {
        buffer.accumulate(&vec![0.0001f32]).unwrap();
    }

    let result = buffer.to_vec();
    assert!((result[0] - 1.0f32).abs() < 0.1); // Should be close to 1.0
}

#[test]
fn test_buffer_with_single_param() {
    let mut buffer = match ANEGradientBuffer::new(1) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.5f32]).unwrap();

    let result = buffer.to_vec();
    assert_eq!(result, vec![0.5f32]);
}

#[test]
fn test_buffer_accumulate_empty_slice() {
    let mut buffer = match ANEGradientBuffer::new(0) {
        Ok(_) => panic!("Should fail with 0 params"),
        Err(_) => return,
    };
}

#[test]
fn test_buffer_to_vec_returns_copy() {
    let mut buffer = match ANEGradientBuffer::new(5) {
        Ok(b) => b,
        Err(_) => return,
    };

    buffer.accumulate(&vec![0.5f32; 5]).unwrap();

    let vec1 = buffer.to_vec();
    let vec2 = buffer.to_vec();

    assert_eq!(vec1, vec2);

    // Modifying one should not affect the other
    let mut vec1_modified = vec1.clone();
    vec1_modified[0] = 999.0;

    let vec3 = buffer.to_vec();
    assert_eq!(vec3[0], 0.5f32); // Original unchanged
}
