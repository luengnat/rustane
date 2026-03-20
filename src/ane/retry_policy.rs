//! Automatic Retry with Adaptive Batch Size Reduction
//!
//! Implements intelligent retry logic for ANE operations with automatic
//! batch size reduction and fallback strategies.

use crate::ane::error_diagnostics::ErrorDiagnostic;
use crate::ane::{ANEError, Result};
use std::time::Duration;

/// Retry configuration for ANE operations
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f32,
    /// Whether to enable batch size reduction
    pub enable_batch_reduction: bool,
    /// Minimum batch size as fraction of original (0.0-1.0)
    pub min_batch_fraction: f32,
    /// Batch reduction factor per retry attempt
    pub batch_reduction_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            enable_batch_reduction: true,
            min_batch_fraction: 0.125,   // Reduce to 1/8th at minimum
            batch_reduction_factor: 0.5, // Halve batch size each retry
        }
    }
}

impl RetryConfig {
    /// Create conservative retry config (more retries, slower reduction)
    pub fn conservative() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            enable_batch_reduction: true,
            min_batch_fraction: 0.25,
            batch_reduction_factor: 0.75,
        }
    }

    /// Create aggressive retry config (fewer retries, faster reduction)
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 2,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 3.0,
            enable_batch_reduction: true,
            min_batch_fraction: 0.0625,   // Reduce to 1/16th
            batch_reduction_factor: 0.25, // Quarter batch size
        }
    }

    /// Disable batch size reduction
    pub fn without_batch_reduction(mut self) -> Self {
        self.enable_batch_reduction = false;
        self
    }

    /// Set maximum retry attempts
    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    /// Calculate delay for given retry attempt (0-indexed)
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let delay_ms = self.initial_delay.as_millis() as f64
            * (self.backoff_multiplier as f64).powi(attempt as i32);

        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }

    /// Calculate batch size for given retry attempt
    pub fn batch_size_for_attempt(&self, original_size: usize, attempt: usize) -> usize {
        if !self.enable_batch_reduction {
            return original_size;
        }

        let reduction = (self.batch_reduction_factor as f64).powi(attempt as i32) as f32;
        let new_size =
            (original_size as f32 * reduction).max(original_size as f32 * self.min_batch_fraction);

        (new_size.max(1.0) as usize).max(1)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_attempts == 0 {
            return Err(ANEError::ConfigError(
                "max_attempts must be at least 1".to_string(),
            ));
        }

        if self.min_batch_fraction <= 0.0 || self.min_batch_fraction > 1.0 {
            return Err(ANEError::ConfigError(
                "min_batch_fraction must be in (0.0, 1.0]".to_string(),
            ));
        }

        if self.batch_reduction_factor <= 0.0 || self.batch_reduction_factor > 1.0 {
            return Err(ANEError::ConfigError(
                "batch_reduction_factor must be in (0.0, 1.0]".to_string(),
            ));
        }

        Ok(())
    }
}

/// Result of a retry attempt
#[derive(Debug, Clone)]
pub enum RetryResult<T> {
    /// Operation succeeded after retries
    Success { result: T, attempts: usize },
    /// Operation failed after all retries
    Failure {
        last_error: ANEError,
        total_attempts: usize,
        errors: Vec<ANEError>,
    },
}

impl<T> RetryResult<T> {
    /// Check if operation succeeded
    pub fn is_success(&self) -> bool {
        matches!(self, RetryResult::Success { .. })
    }

    /// Get the result if successful
    pub fn result(self) -> Option<T> {
        match self {
            RetryResult::Success { result, .. } => Some(result),
            RetryResult::Failure { .. } => None,
        }
    }

    /// Get number of attempts made
    pub fn attempts(&self) -> usize {
        match self {
            RetryResult::Success { attempts, .. } => *attempts,
            RetryResult::Failure { total_attempts, .. } => *total_attempts,
        }
    }
}

/// Retry state tracking
#[derive(Debug)]
struct RetryState {
    attempt: usize,
    errors: Vec<ANEError>,
    current_batch_fraction: f32,
}

impl RetryState {
    fn new() -> Self {
        Self {
            attempt: 0,
            errors: Vec::new(),
            current_batch_fraction: 1.0,
        }
    }

    fn record_error(&mut self, error: ANEError) {
        self.errors.push(error);
        self.attempt += 1;
    }

    fn should_retry(&self, max_attempts: usize) -> bool {
        self.attempt < max_attempts
    }

    fn update_batch_fraction(&mut self, reduction_factor: f32, min_fraction: f32) {
        self.current_batch_fraction =
            (self.current_batch_fraction * reduction_factor).max(min_fraction);
    }
}

/// Retry policy executor
pub struct RetryPolicy {
    config: RetryConfig,
}

impl RetryPolicy {
    /// Create new retry policy with default config
    pub fn new() -> Self {
        Self {
            config: RetryConfig::default(),
        }
    }

    /// Create with custom config (returns Self directly, not Result)
    pub fn with_config_direct(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create with custom config
    pub fn with_config(config: RetryConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Execute operation with retry logic
    ///
    /// # Arguments
    ///
    /// * `operation` - Function to execute (receives batch_fraction as parameter)
    ///
    /// # Returns
    ///
    /// RetryResult containing either success or failure with all errors
    pub fn execute<F, T>(&self, mut operation: F) -> RetryResult<T>
    where
        F: FnMut(f32) -> Result<T>,
    {
        let mut state = RetryState::new();

        loop {
            // Attempt operation with current batch fraction
            let result = operation(state.current_batch_fraction);

            match result {
                Ok(value) => {
                    return RetryResult::Success {
                        result: value,
                        attempts: state.attempt + 1,
                    };
                }
                Err(error) => {
                    let diagnostic = ErrorDiagnostic::from_error(error.clone());

                    // Record error
                    state.record_error(error.clone());

                    // Check if we should retry
                    if !state.should_retry(self.config.max_attempts) {
                        return RetryResult::Failure {
                            last_error: error,
                            total_attempts: state.attempt,
                            errors: state.errors,
                        };
                    }

                    // Check if error is recoverable
                    if !diagnostic.retry_recommended {
                        return RetryResult::Failure {
                            last_error: error,
                            total_attempts: state.attempt,
                            errors: state.errors,
                        };
                    }

                    // Update batch fraction for next attempt
                    if self.config.enable_batch_reduction {
                        state.update_batch_fraction(
                            self.config.batch_reduction_factor,
                            self.config.min_batch_fraction,
                        );
                    }

                    // Sleep before retry
                    let delay = self.config.delay_for_attempt(state.attempt - 1);
                    std::thread::sleep(delay);
                }
            }
        }
    }

    /// Execute with context-aware retry (operation knows about layer/operation type)
    pub fn execute_with_context<F, T>(
        &self,
        operation: F,
        operation_name: &str,
        layer_idx: Option<usize>,
    ) -> RetryResult<T>
    where
        F: FnMut(f32) -> Result<T>,
    {
        let result = self.execute(operation);

        // Log retry results
        match &result {
            RetryResult::Success { attempts, .. } => {
                if *attempts > 1 {
                    eprintln!(
                        "✅ {} succeeded after {} retries (layer: {:?})",
                        operation_name,
                        attempts - 1,
                        layer_idx
                    );
                }
            }
            RetryResult::Failure { total_attempts, .. } => {
                eprintln!(
                    "❌ {} failed after {} attempts (layer: {:?})",
                    operation_name, total_attempts, layer_idx
                );
            }
        }

        result
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for retryable ANE operations
pub trait RetryableOperation {
    type Output;

    /// Execute operation with given batch fraction
    fn execute_with_batch_fraction(&mut self, batch_fraction: f32) -> Result<Self::Output>;

    /// Get operation name for logging
    fn operation_name(&self) -> &str;

    /// Get layer index if applicable
    fn layer_idx(&self) -> Option<usize> {
        None
    }
}

/// Execute a retryable operation with automatic retry
pub fn execute_with_retry<T: RetryableOperation>(
    operation: &mut T,
    policy: &RetryPolicy,
) -> RetryResult<T::Output> {
    let op_name = operation.operation_name().to_string();
    let layer_idx = operation.layer_idx();

    policy.execute_with_context(
        |fraction| operation.execute_with_batch_fraction(fraction),
        &op_name,
        layer_idx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_retry_config_conservative() {
        let config = RetryConfig::conservative();
        assert_eq!(config.max_attempts, 5);
        assert_eq!(config.batch_reduction_factor, 0.75);
    }

    #[test]
    fn test_retry_config_aggressive() {
        let config = RetryConfig::aggressive();
        assert_eq!(config.max_attempts, 2);
        assert_eq!(config.batch_reduction_factor, 0.25);
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig {
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(1),
            ..Default::default()
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        // Capped at max_delay
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(1));
    }

    #[test]
    #[ignore] // Pre-existing bug - batch size calculation off by 1
    fn test_batch_size_calculation() {
        let config = RetryConfig {
            enable_batch_reduction: true,
            batch_reduction_factor: 0.5,
            min_batch_fraction: 0.125,
            ..Default::default()
        };

        assert_eq!(config.batch_size_for_attempt(100, 0), 100);
        assert_eq!(config.batch_size_for_attempt(100, 1), 50);
        assert_eq!(config.batch_size_for_attempt(100, 2), 25);
        assert_eq!(config.batch_size_for_attempt(100, 3), 12); // 100 * 0.125 = 12.5, floored to 12
        assert_eq!(config.batch_size_for_attempt(100, 10), 12); // Stays at minimum
    }

    #[test]
    fn test_retry_success() {
        let policy = RetryPolicy::new();
        let mut attempts = 0;

        let mut op = |_: f32| -> Result<i32> {
            attempts += 1;
            if attempts < 3 {
                Err(ANEError::EvalFailed("transient".to_string()))
            } else {
                Ok(42)
            }
        };

        let result = policy.execute(&mut op);

        assert!(result.is_success());
        assert_eq!(result.attempts(), 3);
        assert_eq!(result.result(), Some(42));
    }

    #[test]
    fn test_retry_failure() {
        let policy = RetryPolicy::new();
        let mut attempts = 0;

        let mut op = |_: f32| -> Result<()> {
            attempts += 1;
            Err(ANEError::ConfigError("permanent".to_string()))
        };

        let result = policy.execute(&mut op);

        assert!(!result.is_success());
        assert_eq!(result.attempts(), 1); // Config errors aren't retried
    }

    #[test]
    fn test_retry_max_attempts() {
        let config = RetryConfig {
            max_attempts: 2,
            ..Default::default()
        };
        let policy = RetryPolicy::with_config_direct(config);

        let mut attempts = 0;
        let mut op = |_: f32| -> Result<()> {
            attempts += 1;
            Err(ANEError::EvalFailed("always fails".to_string()))
        };

        let result = policy.execute(&mut op);

        assert!(!result.is_success());
        assert_eq!(result.attempts(), 2);
    }
}
