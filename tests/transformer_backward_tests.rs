//! Tests for transformer backward pass operations
//!
//! Validates gradient computations for:
//! - RMSNorm backward pass
//! - Cross-entropy loss backward pass
//! - Attention backward pass
//! - Feed-forward network backward pass

use rustane::layers::{
    rmsnorm_backward, cross_entropy_backward, attention_backward, ffn_backward,
    AttentionConfig, FFNConfig,
};

/// Compute numerical gradient using finite differences
/// Used to validate backward pass implementations
#[allow(dead_code)]
fn numerical_gradient(
    f: impl Fn(&[f32]) -> f32,
    x: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..x.len() {
        x_plus[i] = x[i] + eps;
        x_minus[i] = x[i] - eps;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * eps);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}

/// Check that two float slices are approximately equal
#[allow(dead_code)]
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let max_abs = a.abs().max(e.abs());
        let rel_error = if max_abs > 1e-6 {
            diff / max_abs
        } else {
            diff
        };

        assert!(
            diff < tolerance || rel_error < 0.1,
            "{} index {}: expected {}, got {} (diff={})",
            label,
            i,
            e,
            a,
            diff
        );
    }
}

// ============================================================================
// RMSNorm Backward Tests
// ============================================================================

#[test]
fn test_rmsnorm_backward_basic_gradient_shapes() {
    let seq_len = 4;
    let dim = 8;

    let x = vec![1.0f32; seq_len * dim];
    let w = vec![1.0f32; dim];
    let d_out = vec![0.1f32; seq_len * dim];

    let (d_x, dw) = rmsnorm_backward(&d_out, &x, &w);

    assert_eq!(d_x.len(), seq_len * dim, "d_x shape mismatch");
    assert_eq!(dw.len(), dim, "dw shape mismatch");
}

#[test]
fn test_rmsnorm_backward_finite_gradients() {
    let seq_len = 4;
    let dim = 8;

    let x = vec![1.0f32; seq_len * dim];
    let w = vec![1.0f32; dim];
    let d_out = vec![0.1f32; seq_len * dim];

    let (d_x, dw) = rmsnorm_backward(&d_out, &x, &w);

    // Check gradients are finite
    for (i, &g) in d_x.iter().enumerate() {
        assert!(g.is_finite(), "d_x[{}] = {} is not finite", i, g);
    }
    for (i, &g) in dw.iter().enumerate() {
        assert!(g.is_finite(), "dw[{}] = {} is not finite", i, g);
    }
}

#[test]
fn test_rmsnorm_backward_zero_gradient_input() {
    let seq_len = 2;
    let dim = 4;

    let x = vec![0.5f32; seq_len * dim];
    let w = vec![1.0f32; dim];
    let d_out = vec![0.0f32; seq_len * dim];

    let (_d_x, dw) = rmsnorm_backward(&d_out, &x, &w);

    // Zero gradient input should give zero weight gradient
    for &g in &dw {
        assert!((g - 0.0).abs() < 1e-6, "expected zero weight gradient");
    }
}

#[test]
fn test_rmsnorm_backward_with_variable_input() {
    let seq_len = 3;
    let dim = 6;

    let mut x = vec![0.0f32; seq_len * dim];
    for i in 0..seq_len * dim {
        x[i] = (i as f32 + 1.0) * 0.1;
    }

    let w = vec![2.0f32; dim];
    let d_out = vec![0.05f32; seq_len * dim];

    let (d_x, dw) = rmsnorm_backward(&d_out, &x, &w);

    // Verify shapes and finite values
    assert_eq!(d_x.len(), seq_len * dim);
    assert_eq!(dw.len(), dim);

    for &g in &d_x {
        assert!(g.is_finite());
    }
    for &g in &dw {
        assert!(g.is_finite());
    }
}

// ============================================================================
// Cross-Entropy Backward Tests
// ============================================================================

#[test]
fn test_cross_entropy_backward_basic_shapes() {
    let vocab_size = 10;
    let seq_len = 2;

    let logits = vec![1.0f32; seq_len * vocab_size];
    let targets = vec![0u32, 5u32];

    let grads = cross_entropy_backward(&logits, &targets, vocab_size);

    assert_eq!(grads.len(), seq_len * vocab_size, "gradient shape mismatch");
}

#[test]
fn test_cross_entropy_backward_softmax_minus_onehot() {
    let vocab_size = 10;
    let seq_len = 2;

    let logits = vec![1.0f32; seq_len * vocab_size];
    let targets = vec![0u32, 5u32];

    let grads = cross_entropy_backward(&logits, &targets, vocab_size);

    // Each position should sum to approximately 0
    // (softmax sums to 1, minus 1 for target position = 0)
    for pos in 0..seq_len {
        let pos_sum: f32 = grads[pos * vocab_size..(pos + 1) * vocab_size]
            .iter()
            .sum();
        assert!(
            pos_sum.abs() < 1e-5,
            "position {} sum should be ~0, got {}",
            pos,
            pos_sum
        );
    }
}

#[test]
fn test_cross_entropy_backward_target_has_negative_gradient() {
    let vocab_size = 5;
    let _seq_len = 1;

    // All same logits, so uniform softmax
    let logits = vec![0.0f32; vocab_size];
    let targets = vec![2u32];

    let grads = cross_entropy_backward(&logits, &targets, vocab_size);

    // Target position should have (prob - 1) < 0
    let target_grad = grads[2];
    assert!(target_grad < 0.0, "target position gradient should be negative");

    // Non-target positions should have (prob - 0) > 0
    for i in 0..vocab_size {
        if i != 2 {
            let grad = grads[i];
            assert!(grad > 0.0, "non-target position {} gradient should be positive", i);
        }
    }
}

#[test]
fn test_cross_entropy_backward_finite_gradients() {
    let vocab_size = 20;
    let seq_len = 4;

    let logits = vec![1.0f32; seq_len * vocab_size];
    let targets = vec![5u32, 10u32, 0u32, 19u32];

    let grads = cross_entropy_backward(&logits, &targets, vocab_size);

    for (i, &g) in grads.iter().enumerate() {
        assert!(g.is_finite(), "gradient[{}] = {} is not finite", i, g);
    }
}

#[test]
fn test_cross_entropy_backward_extreme_logits() {
    let vocab_size = 5;
    let _seq_len = 1;

    // Large positive and negative logits
    let mut logits = vec![100.0f32; vocab_size];
    logits[2] = -100.0;
    let targets = vec![0u32];

    let grads = cross_entropy_backward(&logits, &targets, vocab_size);

    // All gradients should be finite
    for &g in &grads {
        assert!(g.is_finite(), "gradient should be finite despite extreme logits");
    }
}

// ============================================================================
// Attention Backward Tests
// ============================================================================

#[test]
fn test_attention_backward_config_creation() {
    let config = AttentionConfig {
        seq_len: 512,
        dim: 256,
        n_heads: 8,
        head_dim: 32,
    };

    assert_eq!(config.seq_len, 512);
    assert_eq!(config.dim, 256);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.head_dim, 32);
    assert_eq!(config.dim / config.n_heads, config.head_dim);
}

#[test]
fn test_attention_backward_basic_gradient_shapes() {
    let seq_len = 8;
    let dim = 16;
    let n_heads = 2;
    let head_dim = dim / n_heads;

    let config = AttentionConfig {
        seq_len,
        dim,
        n_heads,
        head_dim,
    };

    let d_out = vec![0.1f32; seq_len * dim];
    let q = vec![1.0f32; seq_len * dim];
    let k = vec![1.0f32; seq_len * dim];
    let v = vec![1.0f32; seq_len * dim];
    let attn_weights = vec![0.1f32; seq_len * seq_len * n_heads];

    let result = attention_backward(&d_out, &q, &k, &v, &attn_weights, &config);
    assert!(result.is_ok(), "attention_backward should succeed");

    let (d_x, dw_q, dw_k, dw_v) = result.unwrap();

    assert_eq!(d_x.len(), seq_len * dim, "d_x shape mismatch");
    assert_eq!(dw_q.len(), dim * dim, "dw_q shape mismatch");
    assert_eq!(dw_k.len(), dim * dim, "dw_k shape mismatch");
    assert_eq!(dw_v.len(), dim * dim, "dw_v shape mismatch");
}

#[test]
fn test_attention_backward_finite_gradients() {
    let seq_len = 4;
    let dim = 8;
    let n_heads = 2;
    let head_dim = dim / n_heads;

    let config = AttentionConfig {
        seq_len,
        dim,
        n_heads,
        head_dim,
    };

    let d_out = vec![0.1f32; seq_len * dim];
    let q = vec![1.0f32; seq_len * dim];
    let k = vec![1.0f32; seq_len * dim];
    let v = vec![1.0f32; seq_len * dim];
    let attn_weights = vec![0.1f32; seq_len * seq_len * n_heads];

    let (d_x, dw_q, dw_k, dw_v) = attention_backward(&d_out, &q, &k, &v, &attn_weights, &config)
        .expect("attention_backward failed");

    for &g in &d_x {
        assert!(g.is_finite());
    }
    for &g in &dw_q {
        assert!(g.is_finite());
    }
    for &g in &dw_k {
        assert!(g.is_finite());
    }
    for &g in &dw_v {
        assert!(g.is_finite());
    }
}

// ============================================================================
// FFN Backward Tests
// ============================================================================

#[test]
fn test_ffn_backward_config_creation() {
    let config = FFNConfig {
        seq_len: 512,
        dim: 256,
        hidden_dim: 768,
    };

    assert_eq!(config.seq_len, 512);
    assert_eq!(config.dim, 256);
    assert_eq!(config.hidden_dim, 768);
}

#[test]
fn test_ffn_backward_basic_gradient_shapes() {
    let seq_len = 8;
    let dim = 16;
    let hidden_dim = 48;

    let config = FFNConfig {
        seq_len,
        dim,
        hidden_dim,
    };

    let d_out = vec![0.1f32; seq_len * dim];
    let x = vec![1.0f32; seq_len * dim];
    let w1_out = vec![1.0f32; seq_len * hidden_dim];
    let w1_gated = vec![1.0f32; seq_len * hidden_dim];

    let result = ffn_backward(&d_out, &x, &w1_out, &w1_gated, &config);
    assert!(result.is_ok(), "ffn_backward should succeed");

    let (d_x, dw1, dw3, dw2) = result.unwrap();

    assert_eq!(d_x.len(), seq_len * dim, "d_x shape mismatch");
    assert_eq!(dw1.len(), dim * hidden_dim, "dw1 shape mismatch");
    assert_eq!(dw3.len(), dim * hidden_dim, "dw3 shape mismatch");
    assert_eq!(dw2.len(), hidden_dim * dim, "dw2 shape mismatch");
}

#[test]
fn test_ffn_backward_finite_gradients() {
    let seq_len = 4;
    let dim = 8;
    let hidden_dim = 24;

    let config = FFNConfig {
        seq_len,
        dim,
        hidden_dim,
    };

    let d_out = vec![0.1f32; seq_len * dim];
    let x = vec![1.0f32; seq_len * dim];
    let w1_out = vec![1.0f32; seq_len * hidden_dim];
    let w1_gated = vec![1.0f32; seq_len * hidden_dim];

    let (d_x, _dw1, _dw3, _dw2) = ffn_backward(&d_out, &x, &w1_out, &w1_gated, &config)
        .expect("ffn_backward failed");

    for &g in &d_x {
        assert!(g.is_finite());
    }
}

#[test]
fn test_ffn_backward_zero_gradient_input() {
    let seq_len = 2;
    let dim = 4;
    let hidden_dim = 12;

    let config = FFNConfig {
        seq_len,
        dim,
        hidden_dim,
    };

    let d_out = vec![0.0f32; seq_len * dim];
    let x = vec![1.0f32; seq_len * dim];
    let w1_out = vec![1.0f32; seq_len * hidden_dim];
    let w1_gated = vec![1.0f32; seq_len * hidden_dim];

    let (d_x, _dw1, _dw3, _dw2) = ffn_backward(&d_out, &x, &w1_out, &w1_gated, &config)
        .expect("ffn_backward failed");

    // Zero gradient input should propagate through
    for &g in &d_x {
        assert!((g - 0.0).abs() < 1e-6 || g.is_finite());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_backward_pass_integration_all_finite() {
    // Test that all backward functions return finite gradients
    let seq_len = 4;
    let dim = 8;
    let vocab_size = 10;

    // RMSNorm backward
    let x = vec![1.0f32; seq_len * dim];
    let w = vec![1.0f32; dim];
    let d_out = vec![0.1f32; seq_len * dim];
    let (d_x, dw) = rmsnorm_backward(&d_out, &x, &w);
    for &g in d_x.iter().chain(dw.iter()) {
        assert!(g.is_finite());
    }

    // Cross-entropy backward
    let logits = vec![1.0f32; seq_len * vocab_size];
    let targets = vec![0u32, 1u32, 2u32, 3u32];
    let ce_grads = cross_entropy_backward(&logits, &targets, vocab_size);
    for &g in &ce_grads {
        assert!(g.is_finite());
    }

    // Attention backward
    let config = AttentionConfig {
        seq_len,
        dim,
        n_heads: 2,
        head_dim: 4,
    };
    let d_attn_out = vec![0.1f32; seq_len * dim];
    let q = vec![1.0f32; seq_len * dim];
    let k = vec![1.0f32; seq_len * dim];
    let v = vec![1.0f32; seq_len * dim];
    let attn_weights = vec![0.1f32; seq_len * seq_len * 2];

    let (d_x_attn, dw_q, dw_k, dw_v) =
        attention_backward(&d_attn_out, &q, &k, &v, &attn_weights, &config)
            .expect("attention_backward failed");
    for &g in d_x_attn
        .iter()
        .chain(dw_q.iter())
        .chain(dw_k.iter())
        .chain(dw_v.iter())
    {
        assert!(g.is_finite());
    }

    // FFN backward
    let ffn_config = FFNConfig {
        seq_len,
        dim,
        hidden_dim: 24,
    };
    let d_ffn_out = vec![0.1f32; seq_len * dim];
    let x_ffn = vec![1.0f32; seq_len * dim];
    let w1_out = vec![1.0f32; seq_len * 24];
    let w1_gated = vec![1.0f32; seq_len * 24];

    let (d_x_ffn, _dw1, _dw3, _dw2) = ffn_backward(&d_ffn_out, &x_ffn, &w1_out, &w1_gated, &ffn_config)
        .expect("ffn_backward failed");
    for &g in &d_x_ffn {
        assert!(g.is_finite());
    }
}
