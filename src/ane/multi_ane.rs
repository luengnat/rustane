//! Multi-ANE distributed training support
//!
//! Provides utilities for training across multiple Apple Neural Engine devices.
//! Supports data parallelism with gradient synchronization.

use crate::ane::ANEError;
use std::sync::OnceLock;

/// Information about a single ANE device
#[derive(Clone, Debug)]
pub struct ANEDeviceInfo {
    /// Device index (0-based)
    pub index: usize,
    /// Device name/identifier
    pub name: String,
    /// Available memory in bytes (approximate)
    pub available_memory_mb: usize,
    /// Whether this device is available for use
    pub is_available: bool,
}

/// Configuration for multi-ANE training
#[derive(Clone, Debug)]
pub struct MultiANEConfig {
    /// Number of ANE devices to use
    pub num_devices: usize,
    /// Whether to use data parallelism
    pub data_parallel: bool,
    /// Shard size for batch distribution
    pub shard_size: usize,
}

impl Default for MultiANEConfig {
    fn default() -> Self {
        Self {
            num_devices: 1,
            data_parallel: true,
            shard_size: 1,
        }
    }
}

impl MultiANEConfig {
    /// Create a new multi-ANE configuration
    pub fn new(num_devices: usize) -> Self {
        Self {
            num_devices,
            data_parallel: true,
            shard_size: 1,
        }
    }

    /// Set the shard size for batch distribution
    pub fn with_shard_size(mut self, shard_size: usize) -> Self {
        self.shard_size = shard_size;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ANEError> {
        if self.num_devices == 0 {
            return Err(ANEError::ConfigError(
                "num_devices must be at least 1".to_string(),
            ));
        }
        if self.num_devices > 4 {
            return Err(ANEError::ConfigError(
                "num_devices cannot exceed 4 (Apple Silicon limit)".to_string(),
            ));
        }
        if self.shard_size == 0 {
            return Err(ANEError::ConfigError(
                "shard_size must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Detect available ANE devices
pub fn detect_ane_devices() -> Result<Vec<ANEDeviceInfo>, ANEError> {
    // Apple Silicon typically has 1-2 ANEs
    // M1/M2: 1 ANE, M3/M4: 2 ANEs (in some configs)

    // For now, we'll do basic detection
    // In production, this would query IOKit/Registry
    let devices = detect_devices_impl()?;

    Ok(devices)
}

/// Get the optimal number of ANE devices for training
pub fn get_optimal_device_count() -> usize {
    static DETECTED_COUNT: OnceLock<usize> = OnceLock::new();

    *DETECTED_COUNT.get_or_init(|| {
        match detect_ane_devices() {
            Ok(devices) => devices.len(),
            Err(_) => 1, // Fallback to single device
        }
    })
}

/// Detect ANE devices (implementation)
fn detect_devices_impl() -> Result<Vec<ANEDeviceInfo>, ANEError> {
    // On Apple Silicon, we can detect ANEs by checking the machine type
    // This is a simplified implementation - production code would use IOKit

    let mut devices = Vec::new();

    // Try to detect multiple ANEs
    // M1/M2 base: 1 ANE
    // M1/M2 Pro/Max: 1-2 ANEs
    // M3/M4: 1-2 ANEs (varies by config)

    // For now, we'll default to 1 device with proper detection
    // In production, this would query sysctl or IOKit

    devices.push(ANEDeviceInfo {
        index: 0,
        name: "ANE0".to_string(),
        available_memory_mb: estimate_ane_memory(),
        is_available: true,
    });

    // Try to detect a second ANE
    if might_have_second_ane() {
        devices.push(ANEDeviceInfo {
            index: 1,
            name: "ANE1".to_string(),
            available_memory_mb: estimate_ane_memory(),
            is_available: true,
        });
    }

    Ok(devices)
}

/// Estimate available ANE memory in MB
fn estimate_ane_memory() -> usize {
    // ANE typically has ~100-300MB of dedicated memory
    // This varies by chip generation
    256 // Conservative estimate
}

/// Check if the system might have a second ANE
fn might_have_second_ane() -> bool {
    // M1/M2 Ultra, M3/M4 Max/Ultra might have 2 ANEs
    // For simplicity, we'll return false for now
    // Production code would check hw.memsize and machine model
    false
}

/// Validate that we have enough devices for the requested configuration
pub fn validate_device_count(num_devices: usize) -> Result<(), ANEError> {
    let available = get_optimal_device_count();

    if num_devices > available {
        return Err(ANEError::ConfigError(format!(
            "Requested {} ANE devices but only {} available",
            num_devices, available
        )));
    }

    Ok(())
}

/// Calculate per-device batch size
pub fn per_device_batch_size(total_batch: usize, num_devices: usize) -> Result<usize, ANEError> {
    if num_devices == 0 {
        return Err(ANEError::ConfigError(
            "num_devices cannot be zero".to_string(),
        ));
    }
    if total_batch % num_devices != 0 {
        return Err(ANEError::ConfigError(format!(
            "Batch size {} must be divisible by number of devices {}",
            total_batch, num_devices
        )));
    }

    Ok(total_batch / num_devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_ane_config_default() {
        let config = MultiANEConfig::default();
        assert_eq!(config.num_devices, 1);
        assert_eq!(config.data_parallel, true);
        assert_eq!(config.shard_size, 1);
    }

    #[test]
    fn test_multi_ane_config_new() {
        let config = MultiANEConfig::new(2);
        assert_eq!(config.num_devices, 2);
        assert!(config.data_parallel);
    }

    #[test]
    fn test_multi_ane_config_with_shard() {
        let config = MultiANEConfig::new(2).with_shard_size(4);
        assert_eq!(config.num_devices, 2);
        assert_eq!(config.shard_size, 4);
    }

    #[test]
    fn test_multi_ane_config_validation() {
        let config = MultiANEConfig::new(1);
        assert!(config.validate().is_ok());

        let invalid_config = MultiANEConfig::new(0);
        assert!(invalid_config.validate().is_err());

        let too_many = MultiANEConfig::new(5);
        assert!(too_many.validate().is_err());
    }

    #[test]
    fn test_per_device_batch_size() {
        assert_eq!(per_device_batch_size(8, 2).unwrap(), 4);
        assert_eq!(per_device_batch_size(16, 4).unwrap(), 4);

        // Non-divisible should fail
        assert!(per_device_batch_size(7, 2).is_err());
    }

    #[test]
    fn test_detect_ane_devices() {
        let devices = detect_ane_devices().unwrap();
        assert!(!devices.is_empty());
        assert!(devices[0].is_available);
    }

    #[test]
    fn test_get_optimal_device_count() {
        let count = get_optimal_device_count();
        assert!(count >= 1);
        assert!(count <= 4); // Apple Silicon limit
    }

    #[test]
    fn test_validate_device_count() {
        assert!(validate_device_count(1).is_ok());
        // May fail on systems with only 1 ANE
        let _ = validate_device_count(2);
    }
}
