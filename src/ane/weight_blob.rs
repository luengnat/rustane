//! Weight blob builders for ANE-compatible formats
//!
//! Provides builders for creating ANE-compatible weight blobs in various formats:
//! FP32, FP16, int8, and quantized formats.

use crate::ane::{ANEError, Result};

/// ANE-formatted weight blob
///
/// Provides memory layout compatible with ANE kernel operations.
/// Format: [global_header (64 bytes)][chunk_header (64 bytes)][data]
#[derive(Clone)]
pub struct WeightBlob(Vec<u8>);

impl WeightBlob {
    /// Build blob from FP32 weights
    ///
    /// Creates a weight blob with proper ANE layout from FP32 data.
    ///
    /// # Arguments
    ///
    /// * `weights` - FP32 weight values (must contain exactly rows × cols elements)
    /// * `rows` - Number of rows in the weight matrix
    /// * `cols` - Number of columns in the weight matrix
    ///
    /// # Returns
    ///
    /// An error if the weight count doesn't match rows × cols
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::ane::WeightBlob;
    /// let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let blob = WeightBlob::from_f32(&weights, 2, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_f32(weights: &[f32], rows: usize, cols: usize) -> Result<Self> {
        if weights.len() != rows * cols {
            return Err(ANEError::WeightBlobError(format!(
                "weight count mismatch: expected {}, got {}",
                rows * cols,
                weights.len()
            )));
        }

        let mut blob = Vec::new();

        // Global header (64 bytes, zeros for now)
        blob.extend_from_slice(&[0u8; 64]);

        // Chunk header (64 bytes, zeros for now)
        blob.extend_from_slice(&[0u8; 64]);

        // FP32 data
        for &w in weights {
            blob.extend_from_slice(&w.to_le_bytes());
        }

        Ok(WeightBlob(blob))
    }

    /// Build blob from FP16 weights
    ///
    /// Creates a weight blob with proper ANE layout from FP16 data.
    ///
    /// # Arguments
    ///
    /// * `weights` - FP16 weight values (must contain exactly rows × cols elements)
    /// * `rows` - Number of rows in the weight matrix
    /// * `cols` - Number of columns in the weight matrix
    ///
    /// # Returns
    ///
    /// An error if the weight count doesn't match rows × cols
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::ane::WeightBlob;
    /// # use half::f16;
    /// let weights = vec![f16::from_f32(1.0); 4];
    /// let blob = WeightBlob::from_f16(&weights, 2, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_f16(weights: &[half::f16], rows: usize, cols: usize) -> Result<Self> {
        if weights.len() != rows * cols {
            return Err(ANEError::WeightBlobError(format!(
                "weight count mismatch: expected {}, got {}",
                rows * cols,
                weights.len()
            )));
        }

        let mut blob = Vec::new();

        // Global header (64 bytes, zeros for now)
        blob.extend_from_slice(&[0u8; 64]);

        // Chunk header (64 bytes, zeros for now)
        blob.extend_from_slice(&[0u8; 64]);

        // FP16 data
        for &w in weights {
            blob.extend_from_slice(&w.to_le_bytes());
        }

        Ok(WeightBlob(blob))
    }

    /// Quantize FP32 weights to int8 and build blob
    ///
    /// Performs per-row quantization from FP32 to int8, storing quantization
    /// scales for later dequantization.
    ///
    /// # Arguments
    ///
    /// * `weights` - FP32 weight values to quantize
    /// * `rows` - Number of rows in the weight matrix
    /// * `cols` - Number of columns in the weight matrix
    ///
    /// # Returns
    ///
    /// A tuple of (blob, scales) where scales is one scale per row
    ///
    /// # Algorithm
    ///
    /// For each row:
    /// 1. Find max absolute value
    /// 2. Scale = max_abs / 127.0 (to fit signed int8)
    /// 3. Quantize each weight as q = (w / scale) rounded to i8
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::ane::WeightBlob;
    /// let weights = vec![10.0f32, 20.0, 30.0, 40.0];
    /// let (blob, scales) = WeightBlob::quantize_f32(&weights, 2, 2)?;
    /// assert_eq!(scales.len(), 2); // One scale per row
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn quantize_f32(
        weights: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Vec<f32>)> {
        if weights.len() != rows * cols {
            return Err(ANEError::WeightBlobError(format!(
                "weight count mismatch: expected {}, got {}",
                rows * cols,
                weights.len()
            )));
        }

        let mut scales = Vec::with_capacity(rows);
        let mut quantized = Vec::with_capacity(rows * cols);

        // Per-row quantization
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let row_weights = &weights[row_start..row_end];

            // Find max absolute value
            let max_abs = row_weights
                .iter()
                .map(|w| w.abs())
                .fold(0.0f32, f32::max)
                .max(1e-6); // Avoid division by zero

            let scale = max_abs / 127.0;
            scales.push(scale);

            // Quantize to int8
            for &w in row_weights {
                let q = ((w / scale).round() as i8) as u8;
                quantized.push(q);
            }
        }

        let mut blob = Vec::new();
        blob.extend_from_slice(&[0u8; 64]); // Global header
        blob.extend_from_slice(&[0u8; 64]); // Chunk header
        blob.extend_from_slice(&quantized);

        Ok((WeightBlob(blob), scales))
    }

    /// Build blob from quantized int8 weights with provided scale
    ///
    /// Creates a weight blob from pre-quantized int8 weights and their scale factor.
    ///
    /// # Arguments
    ///
    /// * `weights` - Pre-quantized int8 weight values
    /// * `scale` - Quantization scale (dequantized value = q * scale)
    /// * `rows` - Number of rows in the weight matrix
    /// * `cols` - Number of columns in the weight matrix
    ///
    /// # Returns
    ///
    /// An error if the weight count doesn't match rows × cols
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::ane::WeightBlob;
    /// let weights = vec![1i8, 2, 3, 4];
    /// let blob = WeightBlob::from_i8_quantized(&weights, 0.5, 2, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_i8_quantized(
        weights: &[i8],
        _scale: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if weights.len() != rows * cols {
            return Err(ANEError::WeightBlobError(format!(
                "weight count mismatch: expected {}, got {}",
                rows * cols,
                weights.len()
            )));
        }

        let mut blob = Vec::new();
        blob.extend_from_slice(&[0u8; 64]); // Global header
        blob.extend_from_slice(&[0u8; 64]); // Chunk header

        for &w in weights {
            blob.push(w as u8);
        }

        Ok(WeightBlob(blob))
    }

    /// Get the blob data as a byte slice
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Get the blob length in bytes
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the blob is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AsRef<[u8]> for WeightBlob {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_blob_from_f32_basic() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        // Header is 128 bytes (64 + 64), data is 16 bytes (4 * 4)
        assert_eq!(bytes.len(), 128 + 16);
    }

    #[test]
    fn test_weight_blob_from_f32_shape_mismatch() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = WeightBlob::from_f32(&weights, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_f16_basic() {
        let weights = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ];
        let blob = WeightBlob::from_f16(&weights, 2, 2).unwrap();

        let bytes = blob.as_ref();
        // Header is 128 bytes, data is 8 bytes (4 * 2)
        assert_eq!(bytes.len(), 128 + 8);
    }

    #[test]
    fn test_weight_blob_from_f16_shape_mismatch() {
        let weights = vec![half::f16::from_f32(1.0); 3];
        let result = WeightBlob::from_f16(&weights, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_quantize_basic() {
        let weights = vec![10.0f32, 20.0, 30.0, 40.0];
        let (blob, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

        assert_eq!(scales.len(), 2);
        assert!(scales[0] > 0.0);
        assert!(scales[1] > 0.0);

        let bytes = blob.as_ref();
        // Header is 128 bytes, data is 4 bytes (4 * 1)
        assert_eq!(bytes.len(), 128 + 4);
    }

    #[test]
    fn test_weight_blob_quantize_scales() {
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
    fn test_weight_blob_quantize_shape_mismatch() {
        let weights = vec![10.0f32, 20.0, 30.0];
        let result = WeightBlob::quantize_f32(&weights, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_basic() {
        let weights = vec![1i8, 2, 3, 4];
        let blob = WeightBlob::from_i8_quantized(&weights, 0.5, 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 4);
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_shape_mismatch() {
        let weights = vec![1i8, 2, 3];
        let result = WeightBlob::from_i8_quantized(&weights, 0.5, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_asref() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = WeightBlob::from_f32(&weights, 2, 2).unwrap();

        let as_ref: &[u8] = blob.as_ref();
        assert_eq!(as_ref.len(), blob.len());
    }

    #[test]
    fn test_weight_blob_clone() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob1 = WeightBlob::from_f32(&weights, 2, 2).unwrap();
        let blob2 = blob1.clone();

        assert_eq!(blob1.as_bytes(), blob2.as_bytes());
    }

    #[test]
    fn test_weight_blob_large_matrix() {
        let weights = vec![1.5f32; 256 * 512];
        let blob = WeightBlob::from_f32(&weights, 256, 512).unwrap();

        let bytes = blob.as_ref();
        let expected_size = 128 + (256 * 512 * 4);
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_weight_blob_quantize_all_zeros() {
        let weights = vec![0.0f32; 4];
        let (blob, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

        assert_eq!(scales.len(), 2);
        // Even zeros get minimum scale of 1e-6 / 127
        assert!(scales[0] > 0.0);
        assert!(scales[1] > 0.0);

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 128 + 4);
    }

    #[test]
    fn test_weight_blob_quantize_negative_values() {
        let weights = vec![-100.0f32, -200.0, -300.0, -400.0];
        let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

        assert_eq!(scales.len(), 2);
        // Both scales should be positive
        assert!(scales[0] > 0.0);
        assert!(scales[1] > 0.0);
    }

    #[test]
    fn test_weight_blob_quantize_mixed_signs() {
        let weights = vec![-50.0f32, 100.0, -75.0, 50.0];
        let (_, scales) = WeightBlob::quantize_f32(&weights, 2, 2).unwrap();

        assert_eq!(scales.len(), 2);
        // Scale 0: max(|-50|, |100|) = 100, scale = 100/127
        // Scale 1: max(|-75|, |50|) = 75, scale = 75/127
        let expected_scale0 = 100.0 / 127.0;
        let expected_scale1 = 75.0 / 127.0;

        assert!((scales[0] - expected_scale0).abs() < 1e-5);
        assert!((scales[1] - expected_scale1).abs() < 1e-5);
    }
}
