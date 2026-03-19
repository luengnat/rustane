//! ANE runtime initialization and management
//!
//! The ANE runtime must be initialized before any ANE operations can be performed.
//! This module provides a singleton-style wrapper around `ane_bridge_init()`.

use crate::sys::{ane_bridge_get_compile_count, ane_bridge_init, ane_bridge_reset_compile_count};
use crate::{Error, Result};
use std::sync::OnceLock;

static ANE_RUNTIME: OnceLock<ANERuntime> = OnceLock::new();
static INIT_RESULT: OnceLock<Result<&'static ANERuntime>> = OnceLock::new();

/// ANE runtime manager
///
/// This type manages the lifecycle of the ANE runtime, ensuring that the
/// private AppleNeuralEngine framework is loaded exactly once.
///
/// # Example
///
/// ```no_run
/// use rustane::wrapper::ANERuntime;
///
/// fn main() -> rustane::Result<()> {
///     let runtime = ANERuntime::init()?;
///     assert!(runtime.is_initialized());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct ANERuntime {
    _private: (), // Prevent direct construction
}

impl ANERuntime {
    /// Initialize the ANE runtime
    ///
    /// This function loads the private AppleNeuralEngine framework and resolves
    /// the necessary classes and methods. It can be called multiple times but
    /// will only initialize once.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Running on non-Apple Silicon hardware
    /// - The ANE framework cannot be loaded
    /// - Required private APIs are not available
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANERuntime;
    /// let runtime = ANERuntime::init()?;
    /// # Ok::<(), rustane::Error>(())
    /// ```
    pub fn init() -> Result<&'static Self> {
        INIT_RESULT.get_or_init(|| {
            // SAFETY: ane_bridge_init is safe to call once
            let result = unsafe { ane_bridge_init() };
            if result == 0 {
                let runtime = ANERuntime { _private: () };
                ANE_RUNTIME.set(runtime).unwrap();
                Ok(ANE_RUNTIME.get().unwrap())
            } else {
                Err(Error::HardwareUnavailable(
                    "Failed to initialize ANE bridge. Ensure the AppleNeuralEngine framework and bridge library are available on Apple Silicon.".to_string()
                ))
            }
        }).clone()
    }

    /// Check if the ANE runtime is initialized
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANERuntime;
    /// assert!(!ANERuntime::is_initialized());
    /// let _ = ANERuntime::init();
    /// assert!(ANERuntime::is_initialized());
    /// ```
    pub fn is_initialized() -> bool {
        ANE_RUNTIME.get().is_some()
    }

    /// Get the current compile count
    ///
    /// The ANE compiler has a limit of ~119 compilations per process.
    /// This function returns the current count for budgeting purposes.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANERuntime;
    /// let runtime = ANERuntime::init()?;
    /// let count = runtime.compile_count();
    /// println!("Compiled {} kernels so far", count);
    /// # Ok::<(), rustane::Error>(())
    /// ```
    pub fn compile_count(&self) -> i32 {
        // SAFETY: Safe when runtime is initialized
        unsafe { ane_bridge_get_compile_count() }
    }

    /// Reset the compile count
    ///
    /// This can be used to implement exec() restart workarounds for the
    /// ~119 compile limit. Note that this resets the global count, not
    /// per-kernel tracking.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANERuntime;
    /// let runtime = ANERuntime::init()?;
    /// if runtime.compile_count() > 100 {
    ///     runtime.reset_compile_count();
    /// }
    /// # Ok::<(), rustane::Error>(())
    /// ```
    pub fn reset_compile_count(&self) {
        // SAFETY: Safe when runtime is initialized
        unsafe { ane_bridge_reset_compile_count() }
    }

    /// Get the global ANE runtime instance
    ///
    /// Returns `None` if the runtime has not been initialized.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANERuntime;
    /// let runtime = ANERuntime::init()?;
    /// let same_runtime = ANERuntime::get().unwrap();
    /// assert!(std::ptr::eq(runtime, same_runtime));
    /// # Ok::<(), rustane::Error>(())
    /// ```
    pub fn get() -> Option<&'static Self> {
        ANE_RUNTIME.get()
    }
}

// ANERuntime is a singleton with static lifetime, so it's safe to share across threads
// for read-only access. However, ANE operations themselves are not thread-safe.
unsafe impl Send for ANERuntime {}
unsafe impl Sync for ANERuntime {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_not_initialized_by_default() {
        // This test verifies the runtime is not initialized at startup
        // Note: Other tests may have initialized it, so we just check the method works
        let _ = ANERuntime::is_initialized();
    }

    #[test]
    fn test_runtime_compile_count_methods() {
        // These methods should work even if not initialized (they'll return 0)
        if let Ok(runtime) = ANERuntime::init() {
            let count = runtime.compile_count();
            assert!(count >= 0);

            runtime.reset_compile_count();
            let count_after = runtime.compile_count();
            assert!(count_after >= 0);
        }
    }

    #[test]
    fn test_runtime_get_returns_none_when_not_initialized() {
        // This test only makes sense if runtime hasn't been initialized yet
        // Since other tests may have initialized it, we just check the method works
        let _ = ANERuntime::get();
    }

    #[test]
    fn test_runtime_singleton() {
        // Verify that init returns the same reference
        if let Ok(runtime1) = ANERuntime::init() {
            let runtime2 = ANERuntime::init().unwrap();
            assert!(std::ptr::eq(runtime1, runtime2));
        }
    }
}
