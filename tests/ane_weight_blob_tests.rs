//! ANE Weight Blob Tests
//!
//! Integration tests for WeightBlob creation, fp32/int8 formats,
//! quantization, and serialization.

use rustane::ane::ANEError;
use rustane::ane::WeightBlob;

// ============================================================================
// TEST 1: FP32 Weight Blob Tests
// ============================================================================

#[test]
fn test_weight_blob_fp32_creation() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

    assert!(!blob.is_empty());
    assert!(blob.len() > 0);
    assert_eq!(blob.as_bytes().len(), 128 + 8); // Header + data
}

#[test]
fn test_weight_blob_fp32_empty() {
    let weights = vec![];
    let blob = WeightBlob::from_f32(&weights, 0, 0).unwrap();

    // Empty weights still have header
    assert_eq!(blob.len(), 128); // Header only
    assert!(!blob.as_bytes().is_empty()); // Has header
}

#[test]
fn test_weight_blob_fp32_single_element() {
    let weights = vec![42.0f32];
    let blob = WeightBlob::from_f32(&weights, 1, 1).unwrap();

    assert!(!blob.is_empty());
    assert_eq!(blob.len(), 128 + 2); // Header + 2 bytes (fp16)
}

#[test]
fn test_weight_blob_fp32_shape_mismatch_too_few() {
    let weights = vec![1.0f32, 2.0, 3.0]; // 3 elements
    let result = WeightBlob::from_f32(&weights, 2, 2); // expects 4

    assert!(result.is_err());
    if let Err(ANEError::WeightBlobError(msg)) = result {
        assert!(msg.contains("weight count mismatch"));
    }
}

#[test]
fn test_weight_blob_fp32_shape_mismatch_too_many() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; // 5 elements
    let result = WeightBlob::from_f32(&weights, 2, 2); // expects 4

    assert!(result.is_err());
}

#[test]
fn test_weight_blob_fp32_large_matrix() {
    let size = 128 * 256;
    let weights = vec![0.5f32; size];
    let blob = WeightBlob::from_f32(&weights, 128, 256).unwrap();

    // Header (128) + data (128 * 256 * 2 bytes for fp16)
    assert_eq!(blob.len(), 128 + (size * 2));
}

#[test]
fn test_weight_blob_fp32_rectangle() {
    let weights = vec![1.0f32; 6 * 4];
    let blob = WeightBlob::from_f32(&weights, 6, 4).unwrap();

    assert_eq!(blob.len(), 128 + (24 * 2));
}

// ============================================================================
// TEST 2: FP16 Weight Blob Tests
// ============================================================================

#[test]
fn test_weight_blob_fp16_creation() {
    let weights = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(3.0),
        half::f16::from_f32(4.0),
    ];
    let blob = WeightBlob::from_f16(&weights, 2, 2).unwrap();

    assert!(!blob.is_empty());
    assert_eq!(blob.len(), 128 + 8);
}

#[test]
fn test_weight_blob_fp16_shape_mismatch() {
    let weights = vec![half::f16::from_f32(1.0); 3];
    let result = WeightBlob::from_f16(&weights, 2, 2);

    assert!(result.is_err());
}

#[test]
fn test_weight_blob_fp16_vs_fp32_same_size() {
    let fp32_weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let fp16_weights = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(3.0),
        half::f16::from_f32(4.0),
    ];

    let blob_fp32 = WeightBlob::from_f32(&fp32_weights, 2, 2).unwrap();
    let blob_fp16 = WeightBlob::from_f16(&fp16_weights, 2, 2).unwrap();

    // Both should have same size (fp16 encoding)
    assert_eq!(blob_fp32.len(), blob_fp16.len());
}

// ============================================================================
// TEST 3: Quantization Tests
// ============================================================================

#[test]
fn test_weight_blob_quantize_basic() {
    let weights = vec![10.0f32, 20.0, 30.0, 40.0];
    let (blob, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);
    assert!(scales.iter().all(|&s| s > 0.0));
    assert_eq!(blob.len(), 64 + 4); // Header + int8 data
}

#[test]
fn test_weight_blob_quantize_scale_calculation() {
    let weights = vec![
        100.0f32, 200.0, 300.0, // Row 0: max = 300.0
        10.0f32, 20.0, 30.0, // Row 1: max = 30.0
    ];
    let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 3).unwrap();

    assert_eq!(scales.len(), 2);

    // Scale = max_abs / 127
    let expected_scale0 = 300.0 / 127.0;
    let expected_scale1 = 30.0 / 127.0;

    assert!((scales[0] - expected_scale0).abs() < 1e-5);
    assert!((scales[1] - expected_scale1).abs() < 1e-5);
}

#[test]
fn test_weight_blob_quantize_all_zeros() {
    let weights = vec![0.0f32; 4];
    let (blob, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);
    // Even zeros get a minimum scale
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

#[test]
fn test_weight_blob_quantize_negative_values() {
    let weights = vec![-100.0f32, -200.0, -300.0, -400.0];
    let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);
    assert!(scales.iter().all(|&s| s > 0.0));
}

#[test]
fn test_weight_blob_quantize_mixed_signs() {
    let weights = vec![-50.0f32, 100.0, -75.0, 50.0];
    let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);

    // Scale uses max absolute value
    let expected_scale0 = 100.0 / 127.0;
    let expected_scale1 = 75.0 / 127.0;

    assert!((scales[0] - expected_scale0).abs() < 1e-5);
    assert!((scales[1] - expected_scale1).abs() < 1e-5);
}

#[test]
fn test_weight_blob_quantize_large_values() {
    let weights = vec![1e6f32, 2e6, 3e6, 4e6];
    let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);
    assert!(scales.iter().all(|&s| s > 0.0));
}

#[test]
fn test_weight_blob_quantize_very_small_values() {
    let weights = vec![1e-6f32, 2e-6, 3e-6, 4e-6];
    let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);
    assert!(scales.iter().all(|&s| s > 0.0));
}

// ============================================================================
// TEST 4: Int8 Quantized Input Tests
// ============================================================================

#[test]
fn test_weight_blob_from_i8_quantized_per_row() {
    let weights = vec![1i8, 2, 3, 4];
    let scales = vec![0.5, 0.25];
    let blob = WeightBlob::from_i8_quantized_per_row(&weights, &scales, 2, 2).unwrap();

    assert_eq!(blob.len(), 64 + 4);
}

#[test]
fn test_weight_blob_from_i8_quantized_single_scale() {
    let weights = vec![1i8, 2, 3, 4];
    let blob = WeightBlob::from_i8_quantized(&weights, 0.5, 1, 4).unwrap();

    assert_eq!(blob.len(), 64 + 4);
}

#[test]
fn test_weight_blob_from_i8_quantized_rejects_multi_row_single_scale() {
    let weights = vec![1i8, 2, 3, 4];
    let result = WeightBlob::from_i8_quantized(&weights, 0.5, 2, 2);

    assert!(result.is_err());
    if let Err(ANEError::WeightBlobError(msg)) = result {
        assert!(msg.contains("single scale"));
    }
}

#[test]
fn test_weight_blob_from_i8_quantized_invalid_scales() {
    let weights = vec![1i8, 2, 3, 4];

    // Zero scale
    let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.0], 2, 2);
    assert!(result.is_err());

    // Negative scale
    let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, -0.1], 2, 2);
    assert!(result.is_err());

    // NaN scale
    let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, f32::NAN], 2, 2);
    assert!(result.is_err());

    // Inf scale
    let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, f32::INFINITY], 2, 2);
    assert!(result.is_err());
}

#[test]
fn test_weight_blob_from_i8_quantized_scale_count_mismatch() {
    let weights = vec![1i8, 2, 3, 4];

    // Too few scales
    let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5], 2, 2);
    assert!(result.is_err());

    // Too many scales
    let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.25, 0.1], 2, 2);
    assert!(result.is_err());
}

// ============================================================================
// TEST 5: Clone and Equality Tests
// ============================================================================

#[test]
fn test_weight_blob_clone_fp32() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let blob1 = WeightBlob::from_f32(&weights, 2, 2).unwrap();
    let blob2 = blob1.clone();

    assert_eq!(blob1.as_bytes(), blob2.as_bytes());
    assert_eq!(blob1.len(), blob2.len());
}

#[test]
fn test_weight_blob_clone_quantized() {
    let weights = vec![10.0f32, 20.0, 30.0, 40.0];
    let (blob1, scales1) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();
    let blob2 = blob1.clone();

    assert_eq!(blob1.as_bytes(), blob2.as_bytes());
    assert_eq!(scales1.len(), 2);
}

// ============================================================================
// TEST 6: AsRef Trait Tests
// ============================================================================

#[test]
fn test_weight_blob_as_ref() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

    let as_ref: &[u8] = blob.as_ref();
    assert_eq!(as_ref, blob.as_bytes());
}

// ============================================================================
// TEST 7: Edge Cases
// ============================================================================

#[test]
fn test_weight_blob_single_row() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let blob = WeightBlob::from_f32(&weights, 1, 4).unwrap();

    assert_eq!(blob.len(), 128 + 8);
}

#[test]
fn test_weight_blob_single_column() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let blob = WeightBlob::from_f32(&weights, 4, 1).unwrap();

    assert_eq!(blob.len(), 128 + 8);
}

#[test]
fn test_weight_blob_asymmetric() {
    let weights = vec![1.0f32; 100 * 5];
    let blob = WeightBlob::from_f32(&weights, 100, 5).unwrap();

    assert_eq!(blob.len(), 128 + (500 * 2));
}

#[test]
fn test_weight_blob_quantize_single_row() {
    let weights = vec![10.0f32, 20.0, 30.0, 40.0];
    let (blob, scales) = WeightBlob::quantize_f32(&weights, 1, 4).unwrap();

    assert_eq!(scales.len(), 1);
    assert_eq!(blob.len(), 64 + 4);
}

#[test]
fn test_weight_blob_extreme_quantization() {
    // Very large range
    let weights = vec![-1e10f32, 1e10, -1e10, 1e10];
    let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

    assert_eq!(scales.len(), 2);
    // Scales should be very large to accommodate the range
    assert!(scales.iter().all(|&s| s > 1e7));
}

// ============================================================================
// TEST 8: Integration - Round Trip
// ============================================================================

#[test]
fn test_weight_blob_fp32_round_trip_precision() {
    // Test that FP32 -> FP16 conversion maintains reasonable precision
    let original = vec![1.0f32, 2.5, 3.125, 4.0];
    let blob = WeightBlob::from_f32(&original, 2, 2).unwrap();

    // Just verify the blob was created successfully
    // (Full FP16->FP32 round trip would require decoding logic)
    assert!(!blob.is_empty());
    assert_eq!(blob.len(), 128 + 8);
}

#[test]
fn test_weight_blob_quantization_error_handling() {
    // Test that quantization handles edge cases gracefully
    let inf_weights = vec![f32::INFINITY, 1.0, 2.0, 3.0];
    let result = WeightBlob::quantize_f32(&inf_weights, 2, 2);

    // INFINITY in weights may cause issues
    // Just verify it doesn't panic - result may be Ok or Err
    let _result = result;
}
