//! Graceful Degradation Strategies
//!
//! Provides fallback mechanisms for ANE operations when they fail,
//! allowing training to continue with CPU execution.

use crate::ane::{ANEError, ErrorDiagnostic, Result};
use std::sync::LazyLock;

/// Statistics about fallback usage
#[derive(Debug, Clone, Default)]
pub struct FallbackStats {
    /// Total number of operations attempted
    pub total_attempts: usize,
    /// Number of times ANE succeeded
    pub ane_successes: usize,
    /// Number of times CPU fallback was used
    pub cpu_fallbacks: usize,
    /// Number of complete failures (both ANE and CPU failed)
    pub complete_failures: usize,
}

impl FallbackStats {
    /// Calculate ANE success rate
    pub fn ane_success_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            return 0.0;
        }
        (self.ane_successes as f64 / self.total_attempts as f64) * 100.0
    }

    /// Calculate fallback rate
    pub fn fallback_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            return 0.0;
        }
        (self.cpu_fallbacks as f64 / self.total_attempts as f64) * 100.0
    }

    /// Format statistics as report
    pub fn format_report(&self) -> String {
        format!(
            "Fallback Statistics:\n\
             • Total attempts: {}\n\
             • ANE successes: {} ({:.1}%)\n\
             • CPU fallbacks: {} ({:.1}%)\n\
             • Complete failures: {} ({:.1}%)",
            self.total_attempts,
            self.ane_successes,
            self.ane_success_rate(),
            self.cpu_fallbacks,
            self.fallback_rate(),
            self.complete_failures,
            if self.total_attempts > 0 {
                (self.complete_failures as f64 / self.total_attempts as f64) * 100.0
            } else {
                0.0
            }
        )
    }
}

/// Global fallback statistics
static GLOBAL_STATS: LazyLock<std::sync::RwLock<FallbackStats>> =
    LazyLock::new(|| std::sync::RwLock::new(FallbackStats::default()));

/// Update global statistics
fn update_stats(ane_success: bool, cpu_fallback: bool, complete_failure: bool) {
    if let Ok(mut stats) = GLOBAL_STATS.write() {
        stats.total_attempts += 1;
        if ane_success {
            stats.ane_successes += 1;
        }
        if cpu_fallback {
            stats.cpu_fallbacks += 1;
        }
        if complete_failure {
            stats.complete_failures += 1;
        }
    }
}

/// Get global fallback statistics
pub fn get_global_stats() -> FallbackStats {
    GLOBAL_STATS.read().map(|s| s.clone()).unwrap_or_default()
}

/// Reset global statistics
pub fn reset_global_stats() {
    if let Ok(mut stats) = GLOBAL_STATS.write() {
        *stats = FallbackStats::default();
    }
}

/// Fallback strategy for ANE operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Try ANE only, fail if it doesn't work
    ANENoFallback,
    /// Try ANE, fall back to CPU on error
    ANEWithCPUFallback,
    /// Try ANE, then CPU, then fail
    ANEThenCPUThenFail,
    /// CPU only (skip ANE entirely)
    CPUOnly,
}

impl Default for FallbackStrategy {
    fn default() -> Self {
        Self::ANEWithCPUFallback
    }
}

impl FallbackStrategy {
    /// Check if this strategy includes ANE execution
    pub fn uses_ane(&self) -> bool {
        matches!(
            self,
            Self::ANENoFallback | Self::ANEWithCPUFallback | Self::ANEThenCPUThenFail
        )
    }

    /// Check if this strategy includes CPU fallback
    pub fn has_cpu_fallback(&self) -> bool {
        matches!(
            self,
            Self::ANEWithCPUFallback | Self::ANEThenCPUThenFail | Self::CPUOnly
        )
    }
}

/// Result of an operation with fallback
#[derive(Debug, Clone)]
pub enum FallbackResult<T> {
    /// Operation succeeded on ANE
    ANESuccess(T),
    /// Operation succeeded on CPU after ANE failed
    CPUFallback(T),
    /// Operation failed on both ANE and CPU
    CompleteFailure(ANEError),
}

impl<T> FallbackResult<T> {
    /// Check if operation succeeded
    pub fn is_success(&self) -> bool {
        matches!(self, Self::ANESuccess(_) | Self::CPUFallback(_))
    }

    /// Get the result value
    pub fn result(self) -> Option<T> {
        match self {
            Self::ANESuccess(v) | Self::CPUFallback(v) => Some(v),
            Self::CompleteFailure(_) => None,
        }
    }

    /// Check if result came from ANE
    pub fn is_from_ane(&self) -> bool {
        matches!(self, Self::ANESuccess(_))
    }

    /// Check if result came from CPU fallback
    pub fn is_from_cpu_fallback(&self) -> bool {
        matches!(self, Self::CPUFallback(_))
    }

    /// Convert to standard Result
    pub fn to_result(self) -> Result<T> {
        match self {
            Self::ANESuccess(v) | Self::CPUFallback(v) => Ok(v),
            Self::CompleteFailure(e) => Err(e),
        }
    }
}

/// Fallback executor for ANE operations with automatic CPU degradation
pub struct FallbackExecutor {
    strategy: FallbackStrategy,
    stats: FallbackStats,
}

impl FallbackExecutor {
    /// Create new fallback executor with default strategy
    pub fn new() -> Self {
        Self {
            strategy: FallbackStrategy::default(),
            stats: FallbackStats::default(),
        }
    }

    /// Create with custom strategy
    pub fn with_strategy(strategy: FallbackStrategy) -> Self {
        Self {
            strategy,
            stats: FallbackStats::default(),
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &FallbackStats {
        &self.stats
    }

    /// Execute operation with fallback according to strategy
    ///
    /// # Arguments
    ///
    /// * `ane_op` - ANE operation to execute
    /// * `cpu_op` - CPU fallback operation
    /// * `operation_name` - Name for logging
    ///
    /// # Returns
    ///
    /// FallbackResult indicating execution path
    pub fn execute<F, G, T>(
        &mut self,
        ane_op: F,
        cpu_op: G,
        operation_name: &str,
    ) -> FallbackResult<T>
    where
        F: FnOnce() -> Result<T>,
        G: FnOnce() -> Result<T>,
    {
        self.stats.total_attempts += 1;

        match self.strategy {
            FallbackStrategy::ANENoFallback => match ane_op() {
                Ok(result) => {
                    self.stats.ane_successes += 1;
                    update_stats(true, false, false);
                    FallbackResult::ANESuccess(result)
                }
                Err(e) => {
                    self.stats.complete_failures += 1;
                    update_stats(false, false, true);

                    let diagnostic =
                        ErrorDiagnostic::from_error(e.clone()).with_operation(operation_name);

                    eprintln!("❌ {} failed (no fallback): {}", operation_name, e);
                    eprintln!("{}", diagnostic.format_report());

                    FallbackResult::CompleteFailure(e)
                }
            },

            FallbackStrategy::ANEWithCPUFallback | FallbackStrategy::ANEThenCPUThenFail => {
                // Try ANE first
                match ane_op() {
                    Ok(result) => {
                        self.stats.ane_successes += 1;
                        update_stats(true, false, false);
                        FallbackResult::ANESuccess(result)
                    }
                    Err(ane_error) => {
                        let diagnostic = ErrorDiagnostic::from_error(ane_error.clone())
                            .with_operation(operation_name);

                        // Check if error is recoverable
                        if !diagnostic.retry_recommended {
                            eprintln!("❌ {} failed with non-recoverable error", operation_name);
                            eprintln!("{}", diagnostic.format_report());

                            self.stats.complete_failures += 1;
                            update_stats(false, false, true);
                            return FallbackResult::CompleteFailure(ane_error);
                        }

                        eprintln!(
                            "⚠️  {} failed on ANE, trying CPU fallback...",
                            operation_name
                        );
                        eprintln!("   ANE error: {}", ane_error);

                        // Try CPU fallback
                        match cpu_op() {
                            Ok(result) => {
                                self.stats.cpu_fallbacks += 1;
                                update_stats(false, true, false);
                                eprintln!("✅ {} succeeded on CPU fallback", operation_name);
                                FallbackResult::CPUFallback(result)
                            }
                            Err(cpu_error) => {
                                self.stats.complete_failures += 1;
                                update_stats(false, false, true);

                                eprintln!("❌ {} failed on both ANE and CPU", operation_name);
                                eprintln!("   ANE error: {}", ane_error);
                                eprintln!("   CPU error: {}", cpu_error);

                                FallbackResult::CompleteFailure(ane_error)
                            }
                        }
                    }
                }
            }

            FallbackStrategy::CPUOnly => match cpu_op() {
                Ok(result) => {
                    update_stats(false, true, false);
                    FallbackResult::CPUFallback(result)
                }
                Err(e) => {
                    self.stats.complete_failures += 1;
                    update_stats(false, false, true);
                    FallbackResult::CompleteFailure(e)
                }
            },
        }
    }

    /// Execute and convert to standard Result
    pub fn execute_to_result<F, G, T>(
        &mut self,
        ane_op: F,
        cpu_op: G,
        operation_name: &str,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
        G: FnOnce() -> Result<T>,
    {
        self.execute(ane_op, cpu_op, operation_name).to_result()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FallbackStats::default();
    }
}

impl Default for FallbackExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper macro for executing operations with fallback
#[macro_export]
macro_rules! with_fallback {
    ($executor:expr, $op_name:expr, $ane_op:expr, $cpu_op:expr) => {
        $executor.execute($ane_op, $cpu_op, $op_name)
    };
}

/// Layer-specific fallback configuration
#[derive(Debug, Clone)]
pub struct LayerFallbackConfig {
    /// Whether ANE is enabled for this layer
    pub ane_enabled: bool,
    /// Number of consecutive failures before disabling ANE for this layer
    pub max_consecutive_failures: usize,
    /// Current consecutive failure count
    consecutive_failures: usize,
}

impl LayerFallbackConfig {
    /// Create new layer fallback config
    pub fn new(ane_enabled: bool, max_consecutive_failures: usize) -> Self {
        Self {
            ane_enabled,
            max_consecutive_failures,
            consecutive_failures: 0,
        }
    }

    /// Check if ANE should be attempted for this layer
    pub fn should_try_ane(&self) -> bool {
        self.ane_enabled && self.consecutive_failures < self.max_consecutive_failures
    }

    /// Record ANE success
    pub fn record_ane_success(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Record ANE failure
    pub fn record_ane_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    /// Reset failure count
    pub fn reset_failures(&mut self) {
        self.consecutive_failures = 0;
    }
}

/// Per-layer fallback manager
#[derive(Debug)]
pub struct LayerFallbackManager {
    configs: Vec<LayerFallbackConfig>,
}

impl LayerFallbackManager {
    /// Create new manager for given number of layers
    pub fn new(num_layers: usize, ane_enabled: bool) -> Self {
        let configs = (0..num_layers)
            .map(|_| LayerFallbackConfig::new(ane_enabled, 3))
            .collect();

        Self { configs }
    }

    /// Get config for specific layer
    pub fn layer_config(&self, layer_idx: usize) -> Option<&LayerFallbackConfig> {
        self.configs.get(layer_idx)
    }

    /// Get mutable config for specific layer
    pub fn layer_config_mut(&mut self, layer_idx: usize) -> Option<&mut LayerFallbackConfig> {
        self.configs.get_mut(layer_idx)
    }

    /// Record success for layer
    pub fn record_success(&mut self, layer_idx: usize) {
        if let Some(config) = self.configs.get_mut(layer_idx) {
            config.record_ane_success();
        }
    }

    /// Record failure for layer
    pub fn record_failure(&mut self, layer_idx: usize) {
        if let Some(config) = self.configs.get_mut(layer_idx) {
            config.record_ane_failure();
        }
    }

    /// Reset all layer failure counts
    pub fn reset_all(&mut self) {
        for config in &mut self.configs {
            config.reset_failures();
        }
    }
}

// Legacy compatibility - re-export simple types
pub use crate::ane::fallback::FallbackResult as SimpleFallbackResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_stats() {
        let mut stats = FallbackStats::default();
        stats.total_attempts = 100;
        stats.ane_successes = 70;
        stats.cpu_fallbacks = 25;
        stats.complete_failures = 5;

        assert_eq!(stats.ane_success_rate(), 70.0);
        assert_eq!(stats.fallback_rate(), 25.0);
    }

    #[test]
    fn test_fallback_strategy() {
        assert!(FallbackStrategy::ANEWithCPUFallback.uses_ane());
        assert!(FallbackStrategy::ANEWithCPUFallback.has_cpu_fallback());
        assert!(!FallbackStrategy::ANENoFallback.has_cpu_fallback());
        assert!(!FallbackStrategy::CPUOnly.uses_ane());
    }

    #[test]
    fn test_fallback_result() {
        let result: FallbackResult<i32> = FallbackResult::ANESuccess(42);
        assert!(result.is_success());
        assert!(result.is_from_ane());
        assert!(!result.is_from_cpu_fallback());
        assert_eq!(result.result(), Some(42));
    }

    #[test]
    fn test_fallback_executor_ane_success() {
        let mut executor = FallbackExecutor::new();
        let result = executor.execute(|| Ok(42), || Ok(0), "test_op");

        assert!(result.is_from_ane());
        assert_eq!(result.result(), Some(42));
        assert_eq!(executor.stats().ane_successes, 1);
    }

    #[test]
    fn test_fallback_executor_cpu_fallback() {
        let mut executor = FallbackExecutor::new();
        let result = executor.execute(
            || Err(ANEError::EvalFailed("test".to_string())),
            || Ok(42),
            "test_op",
        );

        assert!(result.is_from_cpu_fallback());
        assert_eq!(result.result(), Some(42));
        assert_eq!(executor.stats().cpu_fallbacks, 1);
    }

    #[test]
    fn test_layer_fallback_config() {
        let mut config = LayerFallbackConfig::new(true, 3);
        assert!(config.should_try_ane());

        config.record_ane_failure();
        config.record_ane_failure();
        assert!(config.should_try_ane());

        config.record_ane_failure();
        assert!(!config.should_try_ane()); // Exceeded max failures

        config.record_ane_success();
        assert!(config.should_try_ane()); // Reset after success
    }
}
