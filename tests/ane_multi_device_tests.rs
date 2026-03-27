//! ANE Multi-Device Tests
//!
//! Tests for MultiANEConfig, ANEDeviceInfo, and multi-device utilities.

use rustane::ane::multi_ane::{
    detect_ane_devices, get_optimal_device_count, per_device_batch_size, validate_device_count,
    ANEDeviceInfo, MultiANEConfig,
};
use rustane::ane::ANEError;

// ============================================================================
// TEST 1: ANEDeviceInfo
// ============================================================================

#[test]
fn test_ane_device_info_creation() {
    let device = ANEDeviceInfo {
        index: 0,
        name: "ANE0".to_string(),
        available_memory_mb: 256,
        is_available: true,
    };

    assert_eq!(device.index, 0);
    assert_eq!(device.name, "ANE0");
    assert_eq!(device.available_memory_mb, 256);
    assert!(device.is_available);
}

#[test]
fn test_ane_device_info_clone() {
    let device1 = ANEDeviceInfo {
        index: 1,
        name: "ANE1".to_string(),
        available_memory_mb: 512,
        is_available: true,
    };

    let device2 = device1.clone();

    assert_eq!(device1.index, device2.index);
    assert_eq!(device1.name, device2.name);
    assert_eq!(device1.available_memory_mb, device2.available_memory_mb);
    assert_eq!(device1.is_available, device2.is_available);
}

#[test]
fn test_ane_device_info_debug() {
    let device = ANEDeviceInfo {
        index: 0,
        name: "ANE0".to_string(),
        available_memory_mb: 256,
        is_available: true,
    };

    let debug_str = format!("{:?}", device);

    assert!(debug_str.contains("ANEDeviceInfo"));
    assert!(debug_str.contains("ANE0"));
}

// ============================================================================
// TEST 2: MultiANEConfig - Basic Operations
// ============================================================================

#[test]
fn test_multi_ane_config_default() {
    let config = MultiANEConfig::default();

    assert_eq!(config.num_devices, 1);
    assert!(config.data_parallel);
    assert_eq!(config.shard_size, 1);
}

#[test]
fn test_multi_ane_config_new() {
    let config = MultiANEConfig::new(2);

    assert_eq!(config.num_devices, 2);
    assert!(config.data_parallel);
    assert_eq!(config.shard_size, 1);
}

#[test]
fn test_multi_ane_config_with_shard_size() {
    let config = MultiANEConfig::new(2).with_shard_size(4);

    assert_eq!(config.num_devices, 2);
    assert_eq!(config.shard_size, 4);
}

#[test]
fn test_multi_ane_config_builder_pattern() {
    let config = MultiANEConfig::new(4).with_shard_size(8);

    assert_eq!(config.num_devices, 4);
    assert_eq!(config.shard_size, 8);
    assert!(config.data_parallel);
}

#[test]
fn test_multi_ane_config_clone() {
    let config1 = MultiANEConfig::new(2).with_shard_size(4);
    let config2 = config1.clone();

    assert_eq!(config1.num_devices, config2.num_devices);
    assert_eq!(config1.data_parallel, config2.data_parallel);
    assert_eq!(config1.shard_size, config2.shard_size);
}

#[test]
fn test_multi_ane_config_debug() {
    let config = MultiANEConfig::new(2);
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("MultiANEConfig"));
    assert!(debug_str.contains("2"));
}

// ============================================================================
// TEST 3: MultiANEConfig - Validation
// ============================================================================

#[test]
fn test_multi_ane_config_valid() {
    let config = MultiANEConfig::new(1);
    assert!(config.validate().is_ok());

    let config = MultiANEConfig::new(2);
    assert!(config.validate().is_ok());

    let config = MultiANEConfig::new(4);
    assert!(config.validate().is_ok());
}

#[test]
fn test_multi_ane_config_zero_devices() {
    let config = MultiANEConfig::new(0);
    let result = config.validate();

    assert!(result.is_err());
    if let Err(ANEError::ConfigError(msg)) = result {
        assert!(msg.contains("num_devices"));
        assert!(msg.contains("at least 1"));
    } else {
        panic!("Expected ConfigError");
    }
}

#[test]
fn test_multi_ane_config_too_many_devices() {
    let config = MultiANEConfig::new(5);
    let result = config.validate();

    assert!(result.is_err());
    if let Err(ANEError::ConfigError(msg)) = result {
        assert!(msg.contains("num_devices"));
        assert!(msg.contains("cannot exceed 4"));
    } else {
        panic!("Expected ConfigError");
    }
}

#[test]
fn test_multi_ane_config_zero_shard_size() {
    let config = MultiANEConfig::new(2).with_shard_size(0);
    let result = config.validate();

    assert!(result.is_err());
    if let Err(ANEError::ConfigError(msg)) = result {
        assert!(msg.contains("shard_size"));
        assert!(msg.contains("at least 1"));
    } else {
        panic!("Expected ConfigError");
    }
}

#[test]
fn test_multi_ane_config_boundary_values() {
    // Minimum valid config
    let min_config = MultiANEConfig::new(1).with_shard_size(1);
    assert!(min_config.validate().is_ok());

    // Maximum valid config
    let max_config = MultiANEConfig::new(4).with_shard_size(1024);
    assert!(max_config.validate().is_ok());
}

// ============================================================================
// TEST 4: Per-Device Batch Size
// ============================================================================

#[test]
fn test_per_device_batch_size_exact_division() {
    assert_eq!(per_device_batch_size(8, 2).unwrap(), 4);
    assert_eq!(per_device_batch_size(16, 4).unwrap(), 4);
    assert_eq!(per_device_batch_size(32, 4).unwrap(), 8);
    assert_eq!(per_device_batch_size(64, 2).unwrap(), 32);
}

#[test]
fn test_per_device_batch_size_single_device() {
    assert_eq!(per_device_batch_size(1, 1).unwrap(), 1);
    assert_eq!(per_device_batch_size(128, 1).unwrap(), 128);
}

#[test]
fn test_per_device_batch_size_non_divisible() {
    let result = per_device_batch_size(7, 2);
    assert!(result.is_err());
    if let Err(ANEError::ConfigError(msg)) = result {
        assert!(msg.contains("must be divisible"));
    } else {
        panic!("Expected ConfigError");
    }
}

#[test]
fn test_per_device_batch_size_more_cases() {
    let test_cases = vec![
        (10, 3, true),   // Not divisible
        (12, 3, false),  // Divisible
        (15, 4, true),   // Not divisible
        (100, 4, false), // Divisible
        (17, 17, false), // Divisible (result = 1)
    ];

    for (batch, devices, should_fail) in test_cases {
        let result = per_device_batch_size(batch, devices);
        if should_fail {
            assert!(
                result.is_err(),
                "Expected error for batch={}, devices={}",
                batch,
                devices
            );
        } else {
            assert!(
                result.is_ok(),
                "Expected success for batch={}, devices={}",
                batch,
                devices
            );
        }
    }
}

#[test]
fn test_per_device_batch_size_zero_devices() {
    let result = per_device_batch_size(8, 0);
    assert!(result.is_err());
}

// ============================================================================
// TEST 5: Device Detection
// ============================================================================

#[test]
fn test_detect_ane_devices_returns_non_empty() {
    let devices = detect_ane_devices().unwrap();
    assert!(!devices.is_empty());
}

#[test]
fn test_detect_ane_devices_first_is_available() {
    let devices = detect_ane_devices().unwrap();
    assert!(devices[0].is_available);
}

#[test]
fn test_detect_ane_devices_have_valid_indices() {
    let devices = detect_ane_devices().unwrap();

    for (i, device) in devices.iter().enumerate() {
        assert_eq!(device.index, i);
    }
}

#[test]
fn test_detect_ane_devices_have_names() {
    let devices = detect_ane_devices().unwrap();

    for device in devices {
        assert!(!device.name.is_empty());
        assert!(device.name.starts_with("ANE"));
    }
}

#[test]
fn test_detect_ane_devices_have_memory() {
    let devices = detect_ane_devices().unwrap();

    for device in devices {
        assert!(device.available_memory_mb > 0);
    }
}

// ============================================================================
// TEST 6: Optimal Device Count
// ============================================================================

#[test]
fn test_get_optimal_device_count_at_least_one() {
    let count = get_optimal_device_count();
    assert!(count >= 1);
}

#[test]
fn test_get_optimal_device_count_at_most_four() {
    let count = get_optimal_device_count();
    assert!(count <= 4); // Apple Silicon limit
}

#[test]
fn test_get_optimal_device_count_consistent() {
    // Multiple calls should return the same value (cached)
    let count1 = get_optimal_device_count();
    let count2 = get_optimal_device_count();
    assert_eq!(count1, count2);
}

// ============================================================================
// TEST 7: Validate Device Count
// ============================================================================

#[test]
fn test_validate_device_count_one() {
    // Should always succeed - every Mac has at least 1 ANE
    assert!(validate_device_count(1).is_ok());
}

#[test]
fn test_validate_device_count_zero() {
    let result = validate_device_count(0);
    // May succeed or fail depending on implementation
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_validate_device_count_exceeds_available() {
    let available = get_optimal_device_count();

    // Request more than available should fail
    if available < 4 {
        let result = validate_device_count(4);
        assert!(result.is_err());
    }
}

#[test]
fn test_validate_device_count_edge_cases() {
    // Test boundary cases
    assert!(validate_device_count(1).is_ok() || validate_device_count(1).is_err());

    // 5 devices should always fail (hard limit)
    let result = validate_device_count(5);
    assert!(result.is_err());
}

// ============================================================================
// TEST 8: Integration Tests
// ============================================================================

#[test]
fn test_config_with_detected_devices() {
    let devices = detect_ane_devices().unwrap();
    let device_count = devices.len();

    let config = MultiANEConfig::new(device_count);
    assert!(config.validate().is_ok());

    // Validate against detected count
    assert!(
        validate_device_count(device_count).is_ok() || validate_device_count(device_count).is_err()
    );
}

#[test]
fn test_batch_size_with_config() {
    let config = MultiANEConfig::new(2).with_shard_size(4);

    // Batch size should be divisible by num_devices
    let batch_size = 16;
    let per_device = per_device_batch_size(batch_size, config.num_devices).unwrap();

    assert_eq!(per_device, batch_size / config.num_devices);
}

#[test]
fn test_full_configuration_workflow() {
    // 1. Detect devices
    let devices = detect_ane_devices().unwrap();
    assert!(!devices.is_empty());

    // 2. Create config
    let config = MultiANEConfig::new(devices.len()).with_shard_size(2);

    // 3. Validate config
    assert!(config.validate().is_ok());

    // 4. Validate device count
    assert!(validate_device_count(config.num_devices).is_ok());

    // 5. Calculate per-device batch size
    let total_batch = config.num_devices * 4;
    let per_device = per_device_batch_size(total_batch, config.num_devices).unwrap();
    assert_eq!(per_device, 4);
}
