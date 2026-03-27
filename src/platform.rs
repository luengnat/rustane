//! Platform and hardware accelerator availability detection
//!
//! This module provides utilities for detecting ANE and SME availability
//! and platform compatibility.

use std::env;

/// Hardware accelerator availability information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareAvailability {
    /// Running on Apple Silicon
    pub is_apple_silicon: bool,
    /// macOS version is 15+
    pub is_macos_15_plus: bool,
    /// ANE framework is available
    pub is_ane_available: bool,
    /// SME (Scalable Matrix Extension) is available
    pub is_sme_available: bool,
    /// SME vector length (0 if not available)
    pub sme_vector_length: usize,
    /// Human-readable availability description
    pub description: String,
}

/// Deprecated: Use HardwareAvailability instead
#[deprecated(since = "0.2.0", note = "Use HardwareAvailability instead")]
pub type ANEAvailability = HardwareAvailability;

#[allow(deprecated)]
impl ANEAvailability {
    /// Check ANE availability at runtime
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::platform::ANEAvailability;
    /// let avail = ANEAvailability::check();
    /// if avail.is_available() {
    ///     println!("ANE is available!");
    /// } else {
    ///     println!("ANE not available: {}", avail.description);
    /// }
    /// ```
    pub fn check() -> Self {
        let is_apple_silicon = Self::is_apple_silicon();
        let is_macos_15_plus = Self::is_macos_15_plus();
        let is_ane_available = is_apple_silicon && is_macos_15_plus;

        let mut description = if is_apple_silicon {
            format!("Apple Silicon detected")
        } else {
            format!("Not running on Apple Silicon")
        };

        if is_apple_silicon {
            if is_macos_15_plus {
                description.push_str(", macOS 15+ detected");
            } else {
                description.push_str(", but macOS < 15");
            }
        }

        if is_ane_available {
            description.push_str(", ANE should be available");
        } else {
            description.push_str(", ANE not available");
        }

        HardwareAvailability {
            is_apple_silicon,
            is_macos_15_plus,
            is_ane_available,
            is_sme_available: is_apple_silicon && cfg!(target_arch = "aarch64"),
            sme_vector_length: if is_apple_silicon { 512 } else { 0 },
            description,
        }
    }

    /// Check if ANE is available for use
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::platform::ANEAvailability;
    /// assert!(ANEAvailability::check().is_available() == ANEAvailability::check().is_ane_available);
    /// ```
    pub fn is_available(&self) -> bool {
        self.is_ane_available
    }

    /// Get a human-readable description
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::platform::ANEAvailability;
    /// let avail = ANEAvailability::check();
    /// println!("{}", avail.description);
    /// ```
    pub fn describe(&self) -> &str {
        &self.description
    }

    /// Check if running on Apple Silicon
    fn is_apple_silicon() -> bool {
        // Check architecture
        #[cfg(target_arch = "aarch64")]
        {
            // On aarch64, we're likely on Apple Silicon
            // But we should verify we're actually on macOS
            if cfg!(target_os = "macos") {
                return true;
            }
        }

        false
    }

    /// Check if running on macOS 15+
    fn is_macos_15_plus() -> bool {
        if !cfg!(target_os = "macos") {
            return false;
        }

        // Try to get macOS version
        if let Ok(version) = env::var("MACOS_VERSION_DEPLOYMENT_TARGET") {
            // From build environment
            if let Ok(v) = version.parse::<f32>() {
                return v >= 15.0;
            }
        }

        // Runtime check - this will only work if we can actually detect it
        // For now, we'll assume we're on a supported version if we're on Apple Silicon
        // and the ANE framework loads successfully
        true
    }
}

/// Macro to skip tests if ANE is not available
///
/// # Example
///
/// /// ```rust
/// /// use rustane::require_ane;
/// /// require_ane!();
/// /// fn test_something() {
/// ///     // This test will only run if ANE is available
/// /// }
/// /// ```
#[macro_export]
macro_rules! require_ane {
    () => {
        if !$crate::platform::ANEAvailability::check().is_available() {
            eprintln!("Skipping test: ANE not available");
            return;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability_check() {
        let avail = HardwareAvailability::check();
        // Just verify it runs without panic
        let _ = avail.is_available();
        let _ = avail.describe();
    }

    #[test]
    fn test_availability_fields() {
        let avail = HardwareAvailability::check();
        // Fields should be consistent
        if avail.is_apple_silicon && avail.is_macos_15_plus {
            assert!(avail.is_ane_available);
        }
    }
}
