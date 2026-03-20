//! Weight blob format tests
//!
//! Tests for creating ANE-compatible weight blobs from various input formats.

#[cfg(test)]
mod tests {
    use rustane::ane::WeightBlob;

    #[test]
    fn test_weight_blob_from_f32() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        // Blob should have header + data
        let bytes = blob.as_ref();
        assert!(bytes.len() >= 128); // At least header size
        assert_eq!(bytes.len(), 128 + 8); // FP16 payload
    }

    #[test]
    fn test_weight_blob_from_f32_large() {
        let weights = vec![1.5f32; 256 * 512];
        let blob = WeightBlob::from_f32(&weights, 256, 512).unwrap();

        let bytes = blob.as_ref();
        let expected_size = 128 + (256 * 512 * 2); // header + FP16 data
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_weight_blob_from_f32_invalid_shape() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        // Claiming wrong dimensions
        let result = WeightBlob::from_f32(&weights, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_f16() {
        use half::f16;
        let weights = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let blob = WeightBlob::from_f16(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert!(bytes.len() >= 128);
        // FP16 data: 2x2 = 4 values * 2 bytes each = 8 bytes
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_from_f16_invalid_shape() {
        use half::f16;
        let weights = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let result = WeightBlob::from_f16(&weights, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_quantize_f32() {
        let weights = vec![10.0f32, 20.0, 30.0, 40.0];
        let result = WeightBlob::quantize_f32(&weights, 2, 2);
        assert!(result.is_ok());

        let (blob, scales) = result.unwrap();
        // One scale per row (2 rows)
        assert_eq!(scales.len(), 2);

        let bytes = blob.as_ref();
        assert!(bytes.len() > 0);
        assert_eq!(bytes.len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_quantize_f32_scales() {
        let weights = vec![
            100.0f32, 200.0, 300.0, // Row 0: max = 300.0
            10.0f32, 20.0, 30.0,   // Row 1: max = 30.0
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
    fn test_weight_blob_quantize_f32_invalid_shape() {
        let weights = vec![10.0f32, 20.0, 30.0];
        let result = WeightBlob::quantize_f32(&weights, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized() {
        let weights = vec![1i8, 2, 3, 4];
        let scales = vec![0.5f32, 0.25];
        let blob = WeightBlob::from_i8_quantized_per_row(&weights, &scales, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_invalid_shape() {
        let weights = vec![1i8, 2, 3];
        let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.25], 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_single_row() {
        let weights = vec![1i8, 2, 3, 4];
        let blob = WeightBlob::from_i8_quantized(&weights, 0.5, 1, 4).unwrap();

        assert_eq!(blob.as_ref().len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_multi_row_requires_per_row_scales() {
        let weights = vec![1i8, 2, 3, 4];
        let result = WeightBlob::from_i8_quantized(&weights, 0.5, 2, 2);

        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_invalid_scales() {
        let weights = vec![1i8, 2, 3, 4];
        let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.0], 2, 2);

        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_zero_values() {
        let weights = vec![0.0f32, 0.0, 0.0, 0.0];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_negative_values() {
        let weights = vec![-1.0f32, -2.0, -3.0, -4.0];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_mixed_signs() {
        let weights = vec![-1.5f32, 2.5, -3.5, 4.5];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_single_row() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = WeightBlob::from_f32(&weights, 1, 4).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_single_column() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = WeightBlob::from_f32(&weights, 4, 1).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_quantize_symmetric() {
        // Test symmetric quantization around zero
        let weights = vec![-50.0f32, 50.0];
        let (_, scales) = WeightBlob::quantize_f32(&weights, 1, 2).unwrap();

        assert_eq!(scales.len(), 1);
        let expected_scale = 50.0 / 127.0;
        assert!((scales[0] - expected_scale).abs() < 1e-5);
    }

    #[test]
    fn test_weight_blob_quantize_all_zeros() {
        let weights = vec![0.0f32, 0.0, 0.0, 0.0];
        let (blob, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

        assert_eq!(scales.len(), 2);
        // Even zeros get minimum scale of 1e-6 / 127
        assert!(scales[0] > 0.0);
        assert!(scales[1] > 0.0);

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_very_small_values() {
        let weights = vec![1e-6f32, 2e-6, 3e-6, 4e-6];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_very_large_values() {
        let weights = vec![1e6f32, 2e6, 3e6, 4e6];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 8);
    }
}
