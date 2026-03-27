//! ANE Retry Policy Tests
//!
//! Tests for RetryPolicy, RetryConfig, and automatic batch size reduction.

use rustane::ane::{ANEError, Result, RetryConfig, RetryPolicy, RetryResult};
use std::time::Duration;

// ============================================================================
// TEST 1: RetryConfig Creation and Validation
// ============================================================================

#[test]
fn test_retry_config_default() {
    let config = RetryConfig::default();
    assert_eq!(config.max_attempts, 3);
    assert_eq!(config.initial_delay, Duration::from_millis(100));
    assert_eq!(config.max_delay, Duration::from_secs(5));
    assert_eq!(config.backoff_multiplier, 2.0);
    assert!(config.enable_batch_reduction);
    assert_eq!(config.min_batch_fraction, 0.125);
    assert_eq!(config.batch_reduction_factor, 0.5);
    assert!(config.validate().is_ok());
}

#[test]
fn test_retry_config_conservative() {
    let config = RetryConfig::conservative();
    assert_eq!(config.max_attempts, 5);
    assert_eq!(config.batch_reduction_factor, 0.75);
    assert_eq!(config.min_batch_fraction, 0.25);
    assert!(config.validate().is_ok());
}

#[test]
fn test_retry_config_aggressive() {
    let config = RetryConfig::aggressive();
    assert_eq!(config.max_attempts, 2);
    assert_eq!(config.batch_reduction_factor, 0.25);
    assert_eq!(config.min_batch_fraction, 0.0625);
    assert!(config.validate().is_ok());
}

#[test]
fn test_retry_config_builder_pattern() {
    let config = RetryConfig::default()
        .with_max_attempts(10)
        .without_batch_reduction();

    assert_eq!(config.max_attempts, 10);
    assert!(!config.enable_batch_reduction);
    assert!(config.validate().is_ok());
}

// ============================================================================
// TEST 2: RetryConfig Calculations
// ============================================================================

#[test]
fn test_delay_for_attempt() {
    let config = RetryConfig::default();

    // Attempt 0: 100ms
    assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));

    // Attempt 1: 100ms * 2 = 200ms
    assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));

    // Attempt 2: 100ms * 4 = 400ms
    assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
}

#[test]
fn test_delay_respects_max() {
    let config = RetryConfig::default();

    // Attempt 10 would be 100ms * 2^10 = 102400ms, but capped at max_delay (5s)
    let delay = config.delay_for_attempt(10);
    assert!(delay <= config.max_delay);
}

#[test]
fn test_batch_size_for_attempt() {
    let config = RetryConfig::default();
    let original_size = 1000;

    // Attempt 0: full size
    assert_eq!(config.batch_size_for_attempt(original_size, 0), 1000);

    // Attempt 1: 50% of original
    assert_eq!(config.batch_size_for_attempt(original_size, 1), 500);

    // Attempt 2: 25% of original
    assert_eq!(config.batch_size_for_attempt(original_size, 2), 250);
}

#[test]
fn test_batch_size_respects_minimum() {
    let config = RetryConfig::default();
    let original_size = 1000;

    // Even at attempt 10, should not go below min_batch_fraction (12.5%)
    let batch_size = config.batch_size_for_attempt(original_size, 10);
    let min_size = (original_size as f32 * config.min_batch_fraction) as usize;
    assert!(batch_size >= min_size);
}

#[test]
fn test_batch_size_reduction_disabled() {
    let config = RetryConfig::default().without_batch_reduction();
    let original_size = 1000;

    // Should always return original size
    assert_eq!(config.batch_size_for_attempt(original_size, 0), 1000);
    assert_eq!(config.batch_size_for_attempt(original_size, 5), 1000);
    assert_eq!(config.batch_size_for_attempt(original_size, 10), 1000);
}

// ============================================================================
// TEST 3: RetryPolicy Creation
// ============================================================================

#[test]
fn test_retry_policy_default() {
    // RetryPolicy::new() returns Self directly
    let policy = RetryPolicy::default();
    let _ = policy;
}

#[test]
fn test_retry_policy_with_config() {
    let config = RetryConfig::default();
    let policy = RetryPolicy::with_config(config);
    assert!(policy.is_ok());
}

#[test]
fn test_retry_policy_with_custom_config() {
    let config = RetryConfig::conservative();
    let policy = RetryPolicy::with_config(config);
    assert!(policy.is_ok());
}

#[test]
fn test_retry_policy_invalid_zero_attempts() {
    let config = RetryConfig::default().with_max_attempts(0);
    let policy = RetryPolicy::with_config(config);
    // Should fail validation
    assert!(policy.is_err());
}

// ============================================================================
// TEST 4: RetryPolicy Execution - Success Cases
// ============================================================================

#[test]
fn test_retry_policy_immediate_success() {
    let policy = RetryPolicy::default();

    let result = policy.execute(|_batch_fraction| -> Result<Vec<f32>> { Ok(vec![1.0, 2.0, 3.0]) });

    assert!(result.is_success());
    assert_eq!(result.attempts(), 1);
    assert_eq!(result.result(), Some(vec![1.0, 2.0, 3.0]));
}

#[test]
fn test_retry_policy_success_after_retry() {
    let policy = RetryPolicy::default();
    use std::sync::{Arc, Mutex};

    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let result = policy.execute(move |_batch_fraction| -> Result<Vec<f32>> {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        if *count < 3 {
            Err(ANEError::EvalFailed("transient error".to_string()))
        } else {
            Ok(vec![42.0])
        }
    });

    assert!(result.is_success());
    assert_eq!(result.attempts(), 3);
    assert_eq!(result.result(), Some(vec![42.0]));
}

// ============================================================================
// TEST 5: RetryPolicy Execution - Failure Cases
// ============================================================================

#[test]
fn test_retry_policy_exhausted_all_attempts() {
    let config = RetryConfig {
        max_attempts: 3,
        ..Default::default()
    };
    let policy = RetryPolicy::with_config_direct(config);

    let result = policy.execute(|_batch_fraction| -> Result<Vec<f32>> {
        Err(ANEError::EvalFailed("always fails".to_string()))
    });

    match result {
        RetryResult::Failure {
            last_error,
            total_attempts,
            ..
        } => {
            assert!(last_error.to_string().contains("always fails"));
            assert_eq!(total_attempts, 3);
        }
        RetryResult::Success { .. } => panic!("Should have failed"),
    }
}

#[test]
fn test_retry_policy_tracks_all_errors() {
    let config = RetryConfig {
        max_attempts: 3,
        ..Default::default()
    };
    let policy = RetryPolicy::with_config_direct(config);

    let result = policy.execute(|_batch_fraction| -> Result<Vec<f32>> {
        Err(ANEError::EvalFailed("fail".to_string()))
    });

    match result {
        RetryResult::Failure { errors, .. } => {
            assert_eq!(errors.len(), 3);
        }
        RetryResult::Success { .. } => panic!("Should have failed"),
    }
}

// ============================================================================
// TEST 6: Batch Fraction Tracking
// ============================================================================

#[test]
fn test_batch_fraction_decreases_on_retry() {
    let config = RetryConfig {
        max_attempts: 5,
        enable_batch_reduction: true,
        batch_reduction_factor: 0.5,
        min_batch_fraction: 0.125,
        ..Default::default()
    };
    let policy = RetryPolicy::with_config_direct(config);

    use std::sync::{Arc, Mutex};
    let fractions = Arc::new(Mutex::new(Vec::new()));
    let fractions_clone = Arc::clone(&fractions);

    let _ = policy.execute(move |batch_fraction| -> Result<Vec<f32>> {
        fractions_clone.lock().unwrap().push(batch_fraction);
        Err(ANEError::EvalFailed("retry".to_string()))
    });

    let fractions = fractions.lock().unwrap();
    assert!(fractions.len() >= 2);

    // Fractions should decrease (or stay at minimum)
    for i in 1..fractions.len() {
        assert!(fractions[i] <= fractions[i - 1]);
    }
}

#[test]
fn test_batch_fraction_never_below_minimum() {
    let config = RetryConfig {
        max_attempts: 10,
        enable_batch_reduction: true,
        batch_reduction_factor: 0.5,
        min_batch_fraction: 0.2,
        ..Default::default()
    };
    let policy = RetryPolicy::with_config_direct(config);

    use std::sync::{Arc, Mutex};
    let fractions = Arc::new(Mutex::new(Vec::new()));
    let fractions_clone = Arc::clone(&fractions);

    let _ = policy.execute(move |batch_fraction| -> Result<Vec<f32>> {
        fractions_clone.lock().unwrap().push(batch_fraction);
        Err(ANEError::EvalFailed("retry".to_string()))
    });

    let fractions = fractions.lock().unwrap();
    for &f in fractions.iter() {
        assert!(f >= 0.2, "Batch fraction {} below minimum 0.2", f);
    }
}

// ============================================================================
// TEST 7: RetryResult Handling
// ============================================================================

#[test]
fn test_retry_result_is_success() {
    let result: RetryResult<Vec<f32>> = RetryResult::Success {
        result: vec![1.0],
        attempts: 1,
    };
    assert!(result.is_success());
}

#[test]
fn test_retry_result_is_not_success_for_failure() {
    let result: RetryResult<Vec<f32>> = RetryResult::Failure {
        last_error: ANEError::EvalFailed("failed".to_string()),
        total_attempts: 3,
        errors: vec![],
    };
    assert!(!result.is_success());
}

#[test]
fn test_retry_result_result_success() {
    let result: RetryResult<Vec<f32>> = RetryResult::Success {
        result: vec![42.0],
        attempts: 2,
    };
    let data = result.result();
    assert_eq!(data, Some(vec![42.0]));
}

#[test]
fn test_retry_result_result_failure() {
    let result: RetryResult<Vec<f32>> = RetryResult::Failure {
        last_error: ANEError::EvalFailed("failed".to_string()),
        total_attempts: 3,
        errors: vec![],
    };
    let data = result.result();
    assert_eq!(data, None);
}

#[test]
fn test_retry_result_attempts() {
    let success: RetryResult<Vec<f32>> = RetryResult::Success {
        result: vec![],
        attempts: 5,
    };
    assert_eq!(success.attempts(), 5);

    let failure: RetryResult<Vec<f32>> = RetryResult::Failure {
        last_error: ANEError::EvalFailed("failed".to_string()),
        total_attempts: 3,
        errors: vec![],
    };
    assert_eq!(failure.attempts(), 3);
}

// ============================================================================
// TEST 8: Error Type Handling
// ============================================================================

#[test]
fn test_retry_with_compile_failed() {
    let policy = RetryPolicy::default();

    let result = policy.execute(|_| -> Result<Vec<f32>> {
        Err(ANEError::CompileFailed("compile error".to_string()))
    });

    assert!(!result.is_success());
}

#[test]
fn test_retry_with_iosurface_error() {
    let policy = RetryPolicy::default();

    let result = policy.execute(|_| -> Result<Vec<f32>> {
        Err(ANEError::IOSurfaceError("surface error".to_string()))
    });

    assert!(!result.is_success());
}

#[test]
fn test_retry_with_config_error() {
    let policy = RetryPolicy::default();

    let result = policy.execute(|_| -> Result<Vec<f32>> {
        Err(ANEError::ConfigError("config error".to_string()))
    });

    assert!(!result.is_success());
}

#[test]
fn test_retry_preserves_error_context() {
    let policy = RetryPolicy::default();

    let specific_error = ANEError::EvalFailed("GPU timeout after 30s".to_string());
    let result = policy.execute(|_| -> Result<Vec<f32>> { Err(specific_error.clone()) });

    match result {
        RetryResult::Failure { last_error, .. } => {
            assert!(last_error.to_string().contains("GPU timeout"));
        }
        RetryResult::Success { .. } => panic!("Should have failed"),
    }
}

// ============================================================================
// TEST 9: Edge Cases
// ============================================================================

#[test]
fn test_retry_with_single_attempt() {
    let config = RetryConfig::default().with_max_attempts(1);
    let policy = RetryPolicy::with_config(config).unwrap();

    let result =
        policy.execute(|_| -> Result<Vec<f32>> { Err(ANEError::EvalFailed("fails".to_string())) });

    match result {
        RetryResult::Failure { total_attempts, .. } => {
            assert_eq!(total_attempts, 1);
        }
        RetryResult::Success { .. } => panic!("Should have failed"),
    }
}

#[test]
fn test_retry_with_many_attempts() {
    let config = RetryConfig::default().with_max_attempts(10);
    let policy = RetryPolicy::with_config(config).unwrap();

    let mut call_count = 0;
    let _ = policy.execute(|_| -> Result<Vec<f32>> {
        call_count += 1;
        Err(ANEError::EvalFailed("fail".to_string()))
    });

    assert_eq!(call_count, 10);
}

#[test]
fn test_closure_receives_batch_fraction() {
    let policy = RetryPolicy::default();

    let mut received_fraction = false;
    let _ = policy.execute(|batch_fraction| -> Result<Vec<f32>> {
        // batch_fraction should be between 0 and 1
        assert!(batch_fraction > 0.0);
        assert!(batch_fraction <= 1.0);
        received_fraction = true;
        Ok(vec![1.0])
    });

    assert!(received_fraction);
}

// ============================================================================
// TEST 10: Concurrency Safety
// ============================================================================

#[test]
fn test_retry_policy_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<RetryPolicy>();
}

#[test]
fn test_retry_config_is_clone() {
    let config1 = RetryConfig::default();
    let config2 = config1.clone();

    assert_eq!(config1.max_attempts, config2.max_attempts);
    assert_eq!(
        config1.batch_reduction_factor,
        config2.batch_reduction_factor
    );
}
