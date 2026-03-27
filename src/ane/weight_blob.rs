//! Weight blob builders for ANE-compatible formats.
//!
//! These wrappers expose the same blob encodings used by the crate's internal
//! ANE bridge code so the public API and runtime share a single source of
//! truth.

use crate::ane::blobs::{
    encode_fp16_blob_from_f16, encode_fp16_blob_from_f32, encode_int8_blob, quantize_f32_per_row,
};
use crate::ane::{ANEError, Result};

/// ANE-formatted weight blob
///
/// Provides memory layout compatible with ANE kernel operations.
#[derive(Clone)]
pub struct WeightBlob(Vec<u8>);

impl WeightBlob {
    fn validate_shape(actual: usize, rows: usize, cols: usize) -> Result<()> {
        let expected = rows.saturating_mul(cols);
        if actual != expected {
            return Err(ANEError::WeightBlobError(format!(
                "weight count mismatch: expected {}, got {}",
                expected, actual
            )));
        }
        Ok(())
    }

    fn validate_scale(scale: f32) -> Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(ANEError::WeightBlobError(format!(
                "quantization scale must be finite and positive, got {}",
                scale
            )));
        }
        Ok(())
    }

    fn validate_per_row_scales(scales: &[f32], rows: usize) -> Result<()> {
        if scales.len() != rows {
            return Err(ANEError::WeightBlobError(format!(
                "scale count mismatch: expected {}, got {}",
                rows,
                scales.len()
            )));
        }

        for &scale in scales {
            Self::validate_scale(scale)?;
        }

        Ok(())
    }

    /// Build blob from FP32 weights
    ///
    /// Encodes the weights into ANE's fp16 blob format.
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
        Self::validate_shape(weights.len(), rows, cols)?;
        Ok(WeightBlob(encode_fp16_blob_from_f32(weights, rows, cols)))
    }

    /// Build blob from FP16 weights
    ///
    /// Encodes the weights into ANE's fp16 blob format.
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
        Self::validate_shape(weights.len(), rows, cols)?;
        Ok(WeightBlob(encode_fp16_blob_from_f16(weights, rows, cols)))
    }

    /// Quantize FP32 weights to int8 and build blob
    ///
    /// Performs per-row quantization from FP32 to int8 and returns the ANE int8
    /// blob alongside one scale per row.
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
    pub fn quantize_f32(weights: &[f32], rows: usize, cols: usize) -> Result<(Self, Vec<f32>)> {
        Self::validate_shape(weights.len(), rows, cols)?;
        let (quantized, scales) = quantize_f32_per_row(weights, rows, cols);
        Ok((WeightBlob(encode_int8_blob(&quantized, rows, cols)), scales))
    }

    /// Build blob from quantized int8 weights with per-row scales.
    ///
    /// The blob stores only the quantized payload, so callers must keep the
    /// returned scales alongside the blob for any later dequantization.
    ///
    /// # Arguments
    ///
    /// * `weights` - Pre-quantized int8 weight values
    /// * `scales` - One positive quantization scale per row
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
    /// let scales = vec![0.5, 0.25];
    /// let blob = WeightBlob::from_i8_quantized_per_row(&weights, &scales, 2, 2)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_i8_quantized_per_row(
        weights: &[i8],
        scales: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Self::validate_shape(weights.len(), rows, cols)?;
        Self::validate_per_row_scales(scales, rows)?;
        Ok(WeightBlob(encode_int8_blob(weights, rows, cols)))
    }

    /// Build blob from quantized int8 weights with a single scale.
    ///
    /// This helper is only valid for single-row tensors. Multi-row quantized
    /// tensors require one scale per row and should use
    /// [`WeightBlob::from_i8_quantized_per_row`].
    pub fn from_i8_quantized(weights: &[i8], scale: f32, rows: usize, cols: usize) -> Result<Self> {
        if rows != 1 {
            return Err(ANEError::WeightBlobError(
                "from_i8_quantized accepts a single scale and is only valid for single-row tensors; use from_i8_quantized_per_row for multi-row weights".into(),
            ));
        }
        Self::validate_scale(scale)?;
        Self::from_i8_quantized_per_row(weights, &[scale], rows, cols)
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

    /// Build a multi-layer weight blob for deep conv graphs
    ///
    /// This creates a weight blob with multiple layers, each with its own 128-byte header.
    /// Used for achieving peak ANE throughput by chaining conv operations.
    ///
    /// Format (compatible with our ANE bridge):
    /// - 128 bytes: global header (matches fill_fp16_header format)
    /// - For each layer: 128 bytes chunk header + channels*channels*2 bytes FP16 weights
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of channels (in/out channels are the same for conv 1x1)
    /// * `depth` - Number of convolution layers
    /// * `weights_fn` - Function that generates weights for each layer (layer_index, out_channels, in_channels) -> Vec<f32>
    ///
    /// # Returns
    ///
    /// A WeightBlob containing all layers
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::ane::WeightBlob;
    /// let blob = WeightBlob::multi_layer_conv(512, 32, |i, out_ch, in_ch| {
    ///     vec![0.1f32; out_ch * in_ch]
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn multi_layer_conv<F>(channels: usize, depth: usize, mut weights_fn: F) -> Result<Self>
    where
        F: FnMut(usize, usize, usize) -> Vec<f32>,
    {
        const HEADER_SIZE: usize = 128; // Our ANE bridge header size
        let wsize = channels * channels * 2; // FP16 weights per layer
        let chunk_size = HEADER_SIZE + wsize; // 128-byte header + weights
        let total = HEADER_SIZE + chunk_size * depth; // global header + all chunks

        let mut buf = vec![0u8; total];

        // Global header (128 bytes) - matches fill_fp16_header format
        buf[0] = 0x01;
        buf[4] = 0x02;
        buf[64] = 0xEF;
        buf[65] = 0xBE;
        buf[66] = 0xAD;
        buf[67] = 0xDE;
        buf[68] = 0x01;
        buf[80..84].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());

        // Each layer chunk - 128-byte header matching fill_fp16_header format
        for i in 0..depth {
            let chunk_offset = HEADER_SIZE + i * chunk_size;
            let payload_size = wsize; // FP16 weight data size

            // Chunk header - exact copy of fill_fp16_header
            buf[chunk_offset] = 0x01;
            buf[chunk_offset + 4] = 0x02;
            buf[chunk_offset + 64] = 0xEF;
            buf[chunk_offset + 65] = 0xBE;
            buf[chunk_offset + 66] = 0xAD;
            buf[chunk_offset + 67] = 0xDE;
            buf[chunk_offset + 68] = 0x01;
            buf[chunk_offset + 72..chunk_offset + 76]
                .copy_from_slice(&(payload_size as u32).to_le_bytes());
            buf[chunk_offset + 80..chunk_offset + 84]
                .copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());

            // Generate and encode FP16 weights
            let weights = weights_fn(i, channels, channels);
            let weight_offset = chunk_offset + HEADER_SIZE;
            for (j, &w) in weights.iter().enumerate() {
                let h = half::f16::from_f32(w).to_bits();
                buf[weight_offset + j * 2] = (h & 0xFF) as u8;
                buf[weight_offset + j * 2 + 1] = (h >> 8) as u8;
            }
        }

        Ok(WeightBlob(buf))
    }

    /// Build a multi-layer weight blob with custom FP16 bit patterns
    ///
    /// This allows direct control over FP16 bit patterns, matching the reference
    /// implementation exactly: fp16[j] = (arc4random()&0x03FF)|0x2000
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of channels
    /// * `depth` - Number of layers
    /// * `pattern_fn` - Function that generates FP16 bits for each weight (layer, index) -> u16
    ///
    /// # Returns
    ///
    /// A WeightBlob containing all layers
    pub fn multi_layer_fp16_bits<F>(
        channels: usize,
        depth: usize,
        mut pattern_fn: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, usize) -> u16,
    {
        let wsize = channels * channels * 2; // FP16 weights per layer
        let chunk_size = 64 + wsize;
        let total = 64 + chunk_size * depth;
        let mut buf = vec![0u8; total];

        // Global header
        buf[0] = 0x01;
        buf[4] = 0x02;

        for i in 0..depth {
            let chunk_offset = 64 + i * chunk_size;

            // Chunk header
            buf[chunk_offset] = 0xEF;
            buf[chunk_offset + 1] = 0xBE;
            buf[chunk_offset + 2] = 0xAD;
            buf[chunk_offset + 3] = 0xDE;
            buf[chunk_offset + 4] = 0x01;
            buf[chunk_offset + 10] = 0x08;

            // Generate FP16 bits
            let weight_offset = chunk_offset + 64;
            let num_weights = channels * channels;
            for j in 0..num_weights {
                let fp16_bits = pattern_fn(i, j);
                buf[weight_offset + j * 2] = (fp16_bits & 0xFF) as u8;
                buf[weight_offset + j * 2 + 1] = (fp16_bits >> 8) as u8;
            }
        }

        Ok(WeightBlob(buf))
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
        assert_eq!(bytes.len(), 128 + 8);
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
        assert_eq!(bytes.len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_quantize_scales() {
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
    fn test_weight_blob_quantize_shape_mismatch() {
        let weights = vec![10.0f32, 20.0, 30.0];
        let result = WeightBlob::quantize_f32(&weights, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_basic() {
        let weights = vec![1i8, 2, 3, 4];
        let blob = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.25], 2, 2).unwrap();

        let bytes = blob.as_ref();
        assert_eq!(bytes.len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_shape_mismatch() {
        let weights = vec![1i8, 2, 3];
        let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.25], 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_single_row() {
        let weights = vec![1i8, 2, 3, 4];
        let blob = WeightBlob::from_i8_quantized(&weights, 0.5, 1, 4).unwrap();

        assert_eq!(blob.len(), 64 + 4);
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_rejects_multi_row_single_scale() {
        let weights = vec![1i8, 2, 3, 4];
        let result = WeightBlob::from_i8_quantized(&weights, 0.5, 2, 2);

        assert!(result.is_err());
    }

    #[test]
    fn test_weight_blob_from_i8_quantized_rejects_invalid_scales() {
        let weights = vec![1i8, 2, 3, 4];
        let result = WeightBlob::from_i8_quantized_per_row(&weights, &[0.5, 0.0], 2, 2);

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
        let expected_size = 128 + (256 * 512 * 2);
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
        assert_eq!(bytes.len(), 64 + 4);
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

    #[test]
    fn test_weight_blob_multi_layer_conv_basic() {
        let blob =
            WeightBlob::multi_layer_conv(64, 2, |_i, out_ch, in_ch| vec![1.0f32; out_ch * in_ch])
                .unwrap();

        let bytes = blob.as_ref();
        // Global header (128) + 2 * (chunk header (128) + weights (64*64*2))
        let expected_size = 128 + 2 * (128 + 64 * 64 * 2);
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_weight_blob_multi_layer_conv_header_format() {
        let blob =
            WeightBlob::multi_layer_conv(8, 3, |_i, out_ch, in_ch| vec![0.5f32; out_ch * in_ch])
                .unwrap();

        let bytes = blob.as_ref();

        // Check global header
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[4], 0x02);

        // Check chunk headers
        let wsize = 8 * 8 * 2; // FP16 weights
        let chunk_size = 128 + wsize;

        for i in 0..3 {
            let chunk_offset = 128 + i * chunk_size;
            assert_eq!(bytes[chunk_offset], 0x01);
            assert_eq!(bytes[chunk_offset + 4], 0x02);
            assert_eq!(bytes[chunk_offset + 64], 0xEF);
            assert_eq!(bytes[chunk_offset + 65], 0xBE);
            assert_eq!(bytes[chunk_offset + 66], 0xAD);
            assert_eq!(bytes[chunk_offset + 67], 0xDE);
            assert_eq!(bytes[chunk_offset + 68], 0x01);
            assert_eq!(
                u32::from_le_bytes([
                    bytes[chunk_offset + 72],
                    bytes[chunk_offset + 73],
                    bytes[chunk_offset + 74],
                    bytes[chunk_offset + 75],
                ]),
                wsize as u32
            );
            assert_eq!(
                u32::from_le_bytes([
                    bytes[chunk_offset + 80],
                    bytes[chunk_offset + 81],
                    bytes[chunk_offset + 82],
                    bytes[chunk_offset + 83],
                ]),
                128
            );
        }
    }

    #[test]
    fn test_weight_blob_multi_layer_conv_data_integrity() {
        let test_values: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();

        let blob =
            WeightBlob::multi_layer_conv(8, 1, |_i, _out_ch, _in_ch| test_values.clone()).unwrap();

        let bytes = blob.as_ref();

        // Verify we can read back the FP16 values (with some precision loss)
        let weight_offset = 128 + 128; // global header + chunk header
        for (i, &expected) in test_values.iter().enumerate() {
            let h = u16::from_le_bytes([
                bytes[weight_offset + i * 2],
                bytes[weight_offset + i * 2 + 1],
            ]);
            let actual = half::f16::from_bits(h).to_f32();
            assert!(
                (actual - expected).abs() < 0.01,
                "Value mismatch at index {}",
                i
            );
        }
    }
}
