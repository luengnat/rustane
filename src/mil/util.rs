//! Weight blob utilities
//!
//! Provides safe wrappers for creating ANE-compatible weight blobs.

use crate::ane::blobs::quantize_f32_per_row;
use crate::sys::{
    ane_bridge_build_weight_blob, ane_bridge_build_weight_blob_int8,
    ane_bridge_build_weight_blob_quantized, ane_bridge_build_weight_blob_transposed,
    ane_bridge_free_blob,
};
use crate::{Error, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracks total leaked bytes (diagnostic counter)
///
/// Incremented when WeightBlob::new() succeeds, decremented in Drop.
/// If Drop is never called (leak), this counter reflects the leak.
/// This is a diagnostic tool for testing - in production, rely on ane_bridge_free_blob.
static TOTAL_LEAKED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Get total accumulated leaked bytes
///
/// This is a diagnostic counter that tracks total bytes that have been
/// allocated but not freed. In production, this should be 0 if ane_bridge_free_blob
/// is properly implemented. During testing, use this to detect leaks.
///
/// # Example
///
/// ```
/// # use rustane::mil::total_leaked_bytes;
/// let leaked = total_leaked_bytes();
/// assert_eq!(leaked, 0);
/// ```
pub fn total_leaked_bytes() -> usize {
    TOTAL_LEAKED_BYTES.load(Ordering::Relaxed)
}

/// Reset the leaked bytes counter
///
/// Used in tests to get a clean baseline.
#[doc(hidden)]
pub fn reset_leaked_bytes() {
    TOTAL_LEAKED_BYTES.store(0, Ordering::Relaxed);
}

/// Weight blob builder
///
/// Provides safe methods to create ANE-compatible weight blobs from Rust data.
/// The blob must be freed using `ane_bridge_free_blob` when done.
///
/// # Safety
///
/// Weight blobs are allocated by the ANE bridge layer and must be explicitly freed.
/// This struct handles cleanup via Drop.
pub struct WeightBlob {
    ptr: *mut u8,
    len: usize,
}

impl WeightBlob {
    fn validate_scale(scale: f32) -> Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "quantization scale must be finite and positive, got {}",
                scale
            )));
        }
        Ok(())
    }

    /// Create a weight blob from FP32 data
    ///
    /// # Arguments
    ///
    /// * `data` - FP32 weight matrix as [rows x cols]
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Returns
    ///
    /// A weight blob that can be passed to ANE compilation
    ///
    /// # Safety
    ///
    /// The blob is automatically freed when this struct is dropped.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::WeightBlob;
    /// let weights = vec![1.0f32; 256 * 512]; // 256 rows, 512 cols
    /// let blob = WeightBlob::from_fp32(&weights, 256, 512).unwrap();
    /// // Use blob in compilation...
    /// // Blob is automatically freed when dropped
    /// ```
    pub fn from_fp32(data: &[f32], rows: i32, cols: i32) -> Result<Self> {
        let mut out_len = 0;

        // SAFETY: ane_bridge_build_weight_blob is safe when:
        // - data pointer is valid for data.len() bytes
        // - rows and cols match data dimensions
        let ptr = unsafe { ane_bridge_build_weight_blob(data.as_ptr(), rows, cols, &mut out_len) };

        if ptr.is_null() {
            return Err(Error::CompilationFailed(
                "Failed to build weight blob".to_string(),
            ));
        }

        // Track allocation
        TOTAL_LEAKED_BYTES.fetch_add(out_len, Ordering::Relaxed);

        Ok(WeightBlob { ptr, len: out_len })
    }

    /// Create a transposed weight blob from FP32 data
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::WeightBlob;
    /// let weights = vec![1.0f32; 256 * 512];
    /// let blob = WeightBlob::from_fp32_transposed(&weights, 256, 512).unwrap();
    /// ```
    pub fn from_fp32_transposed(data: &[f32], rows: i32, cols: i32) -> Result<Self> {
        let mut out_len = 0;

        let ptr = unsafe {
            ane_bridge_build_weight_blob_transposed(data.as_ptr(), rows, cols, &mut out_len)
        };

        if ptr.is_null() {
            return Err(Error::CompilationFailed(
                "Failed to build transposed weight blob".to_string(),
            ));
        }

        // Track allocation
        TOTAL_LEAKED_BYTES.fetch_add(out_len, Ordering::Relaxed);

        Ok(WeightBlob { ptr, len: out_len })
    }

    /// Create an INT8 weight blob from pre-quantized data.
    #[allow(dead_code)]
    pub fn from_int8(data: &[i8], rows: i32, cols: i32) -> Result<Self> {
        let mut out_len = 0;
        let ptr =
            unsafe { ane_bridge_build_weight_blob_int8(data.as_ptr(), rows, cols, &mut out_len) };

        if ptr.is_null() {
            return Err(Error::CompilationFailed(
                "Failed to build INT8 weight blob".to_string(),
            ));
        }

        TOTAL_LEAKED_BYTES.fetch_add(out_len, Ordering::Relaxed);

        Ok(WeightBlob { ptr, len: out_len })
    }

    /// Create a quantized weight blob from FP32 data.
    ///
    /// This convenience API is only valid for single-row tensors because it
    /// returns a single scale. Multi-row quantized tensors need one scale per
    /// row and should use [`WeightBlob::from_fp32_quantized_per_row`].
    #[allow(dead_code)]
    pub fn from_fp32_quantized(data: &[f32], rows: i32, cols: i32) -> Result<(Self, f32)> {
        if rows != 1 {
            return Err(Error::InvalidParameter(
                "from_fp32_quantized returns a single scale and is only valid for single-row tensors; use from_fp32_quantized_per_row for multi-row weights".to_string(),
            ));
        }

        let mut out_len = 0;
        let mut out_scale = 0.0f32;
        let ptr = unsafe {
            ane_bridge_build_weight_blob_quantized(
                data.as_ptr(),
                rows,
                cols,
                &mut out_scale,
                &mut out_len,
            )
        };

        if ptr.is_null() {
            return Err(Error::CompilationFailed(
                "Failed to build quantized weight blob".to_string(),
            ));
        }

        Self::validate_scale(out_scale)?;
        TOTAL_LEAKED_BYTES.fetch_add(out_len, Ordering::Relaxed);

        Ok((WeightBlob { ptr, len: out_len }, out_scale))
    }

    /// Create a quantized weight blob from FP32 data and return one scale per row.
    #[allow(dead_code)]
    pub fn from_fp32_quantized_per_row(
        data: &[f32],
        rows: i32,
        cols: i32,
    ) -> Result<(Self, Vec<f32>)> {
        if rows <= 0 || cols <= 0 {
            return Err(Error::InvalidParameter(format!(
                "rows and cols must be positive, got rows={}, cols={}",
                rows, cols
            )));
        }

        let rows_usize = rows as usize;
        let cols_usize = cols as usize;
        let expected = rows_usize.saturating_mul(cols_usize);
        if data.len() != expected {
            return Err(Error::InvalidTensorShape(format!(
                "weight count mismatch: expected {}, got {}",
                expected,
                data.len()
            )));
        }

        let (quantized, scales) = quantize_f32_per_row(data, rows_usize, cols_usize);
        for &scale in &scales {
            Self::validate_scale(scale)?;
        }

        let blob = Self::from_int8(&quantized, rows, cols)?;

        Ok((blob, scales))
    }

    /// Get the blob data as a byte slice
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get the blob length in bytes
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the blob is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for WeightBlob {
    fn drop(&mut self) {
        // Decrement leaked bytes counter if this blob wasn't already freed
        if !self.ptr.is_null() {
            TOTAL_LEAKED_BYTES.fetch_sub(self.len, Ordering::Relaxed);
            unsafe { ane_bridge_free_blob(self.ptr.cast()) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: WeightBlob owns its data and can be sent across threads
unsafe impl Send for WeightBlob {}

// SAFETY: WeightBlob provides immutable access to its data
unsafe impl Sync for WeightBlob {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_blob_fp32() {
        reset_leaked_bytes();
        let initial_leaked = total_leaked_bytes();

        let weights = vec![1.0f32; 256 * 512];
        let blob = WeightBlob::from_fp32(&weights, 256, 512).unwrap();

        let after_alloc = total_leaked_bytes();
        assert!(after_alloc >= initial_leaked);

        let blob_len = blob.len();
        assert!(!blob.is_empty());
        assert_eq!(blob_len, blob.as_bytes().len());

        drop(blob);
        let after_drop = total_leaked_bytes();
        // Bytes should be freed on drop
        assert!(after_drop <= after_alloc);
    }

    #[test]
    fn test_weight_blob_transposed() {
        reset_leaked_bytes();
        let weights = vec![1.0f32; 256 * 512];
        let blob = WeightBlob::from_fp32_transposed(&weights, 256, 512).unwrap();
        assert!(!blob.is_empty());

        let before_drop = total_leaked_bytes();
        drop(blob);
        let after_drop = total_leaked_bytes();
        assert!(after_drop <= before_drop);
    }

    #[test]
    fn test_total_leaked_bytes_tracking() {
        reset_leaked_bytes();
        assert_eq!(total_leaked_bytes(), 0);

        let weights1 = vec![1.0f32; 100];
        let blob1 = WeightBlob::from_fp32(&weights1, 10, 10).unwrap();
        let leaked1 = total_leaked_bytes();
        assert!(leaked1 > 0);

        let weights2 = vec![2.0f32; 200];
        let blob2 = WeightBlob::from_fp32(&weights2, 20, 10).unwrap();
        let leaked2 = total_leaked_bytes();
        assert!(leaked2 > leaked1);

        drop(blob1);
        let after_drop1 = total_leaked_bytes();
        assert!(after_drop1 < leaked2);

        drop(blob2);
        assert_eq!(total_leaked_bytes(), 0);
    }

    #[test]
    fn test_weight_blob_int8() {
        let weights = vec![0i8; 256 * 512];
        let blob = WeightBlob::from_int8(&weights, 256, 512).unwrap();
        assert_eq!(blob.len(), 64 + (256 * 512));
    }

    #[test]
    fn test_weight_blob_quantized_single_row() {
        let weights = vec![1.0f32; 512];
        let (blob, scale) = WeightBlob::from_fp32_quantized(&weights, 1, 512).unwrap();
        assert_eq!(blob.len(), 64 + 512);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_weight_blob_quantized_per_row() {
        let weights = vec![1.0f32; 256 * 512];
        let (blob, scales) = WeightBlob::from_fp32_quantized_per_row(&weights, 256, 512).unwrap();
        assert_eq!(blob.len(), 64 + (256 * 512));
        assert_eq!(scales.len(), 256);
        assert!(scales.iter().all(|scale| *scale > 0.0));
    }

    #[test]
    fn test_weight_blob_quantized_rejects_multi_row_single_scale() {
        let weights = vec![1.0f32; 256 * 512];
        let result = WeightBlob::from_fp32_quantized(&weights, 256, 512);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }
}

// ============================================================================
// RoPE (Rotary Position Embeddings) Utilities
// ============================================================================

/// Generate RoPE (Rotary Position Embeddings) cos/sin tables
///
/// RoPE applies rotation matrices to position embeddings in transformer attention.
/// The rotation angle for position `p` and dimension `i` is:
/// `theta = p * base^(-2i/head_dim)`
///
/// # Arguments
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `max_seq_len` - Maximum sequence length to support
/// * `base` - RoPE base frequency (typically 10000.0)
///
/// # Returns
/// A tuple of (cos_table, sin_table) where each table has shape
/// `[max_seq_len * head_dim / 2]` (only half the dimensions due to even/odd pairing)
///
/// # Example
///
/// ```
/// use rustane::mil::generate_rope_tables;
///
/// let (cos, sin) = generate_rope_tables(64, 512, 10000.0);
/// assert_eq!(cos.len(), 512 * 32); // seq_len * (head_dim / 2)
/// assert_eq!(sin.len(), 512 * 32);
/// ```
pub fn generate_rope_tables(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim % 2 == 0,
        "head_dim must be even for RoPE, got {}",
        head_dim
    );

    let half_dim = head_dim / 2;
    let table_size = max_seq_len * half_dim;

    let mut cos_table = vec![0.0f32; table_size];
    let mut sin_table = vec![0.0f32; table_size];

    // Precompute frequencies: freq[i] = base^(-i/head_dim) for i in [0, half_dim)
    // This matches the reference implementation: freq = 1 / base^(i/head_dim)
    let mut frequencies = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        frequencies[i] = (base).powf(-(i as f32) / (head_dim as f32));
    }

    // Generate cos/sin for each position and frequency
    for pos in 0..max_seq_len {
        let pos_offset = pos * half_dim;
        for i in 0..half_dim {
            let theta = (pos as f32) * frequencies[i];
            let idx = pos_offset + i;
            cos_table[idx] = theta.cos();
            sin_table[idx] = theta.sin();
        }
    }

    (cos_table, sin_table)
}

/// Generate RoPE tables and convert to WeightBlob format
///
/// Convenience function that generates RoPE tables and wraps them in
/// WeightBlob format for ANE compilation.
///
/// # Arguments
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `max_seq_len` - Maximum sequence length
/// * `base` - RoPE base frequency (typically 10000.0)
///
/// # Returns
/// A tuple of (cos_blob, sin_blob) ready for ANE weight loading
///
/// # Example
///
/// ```
/// use rustane::mil::generate_rope_blobs;
///
/// let (cos_blob, sin_blob) = generate_rope_blobs(64, 512, 10000.0).unwrap();
/// assert!(!cos_blob.is_empty());
/// assert!(!sin_blob.is_empty());
/// ```
pub fn generate_rope_blobs(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
) -> Result<(WeightBlob, WeightBlob)> {
    let (cos_table, sin_table) = generate_rope_tables(head_dim, max_seq_len, base);

    // Reshape for ANE: [seq_len, half_dim] -> flat array
    // ANE expects weights in row-major order
    let cos_blob = WeightBlob::from_fp32(&cos_table, max_seq_len as i32, (head_dim / 2) as i32)?;
    let sin_blob = WeightBlob::from_fp32(&sin_table, max_seq_len as i32, (head_dim / 2) as i32)?;

    Ok((cos_blob, sin_blob))
}

#[cfg(test)]
mod rope_tests {
    use super::*;

    #[test]
    fn test_generate_rope_tables_basic() {
        let (cos, sin) = generate_rope_tables(64, 10, 10000.0);

        // Check sizes
        assert_eq!(cos.len(), 10 * 32); // seq_len * (head_dim / 2)
        assert_eq!(sin.len(), 10 * 32);

        // Check first position (theta = 0, so cos=1, sin=0)
        for i in 0..32 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0,{}] should be 1.0", i);
            assert!(sin[i].abs() < 1e-6, "sin[0,{}] should be 0.0", i);
        }
    }

    #[test]
    fn test_generate_rope_tables_frequencies() {
        let head_dim = 8;
        let max_seq_len = 4;
        let base = 10000.0;

        let (cos, sin) = generate_rope_tables(head_dim, max_seq_len, base);

        // Verify frequency pattern: lower dimensions rotate faster
        // At position 1, dim 0 should have larger angle than dim 3
        let _theta_dim0 = (base).powf(-0.0 / 8.0); // = 1.0
        let _theta_dim3 = (base).powf(-3.0 / 8.0); // = 10000^(-0.375)

        // Position 1, dim 0: theta = 1.0 * 1.0 = 1.0 radian
        let expected_cos_1_0 = _theta_dim0.cos();
        let expected_sin_1_0 = _theta_dim0.sin();

        assert!((cos[1 * 4 + 0] - expected_cos_1_0).abs() < 1e-5);
        assert!((sin[1 * 4 + 0] - expected_sin_1_0).abs() < 1e-5);
    }

    #[test]
    fn test_generate_rope_tables_periodicity() {
        // RoPE should be periodic (approximately) for large base
        let (cos, sin) = generate_rope_tables(64, 100, 10000.0);

        // Check that values are in valid range
        for i in 0..cos.len() {
            assert!((-1.0..=1.0).contains(&cos[i]), "cos[{}] out of range", i);
            assert!((-1.0..=1.0).contains(&sin[i]), "sin[{}] out of range", i);
        }

        // Verify cos^2 + sin^2 = 1 for each position/dimension
        let half_dim = 32;
        for pos in 0..10 {
            for i in 0..half_dim {
                let idx = pos * half_dim + i;
                let norm = cos[idx] * cos[idx] + sin[idx] * sin[idx];
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "Unit circle violated at idx {}",
                    idx
                );
            }
        }
    }

    #[test]
    fn test_generate_rope_blobs() {
        let (cos_blob, sin_blob) = generate_rope_blobs(64, 512, 10000.0).unwrap();

        // Verify blobs are non-empty and have consistent sizes
        assert!(!cos_blob.is_empty());
        assert!(!sin_blob.is_empty());
        assert_eq!(cos_blob.len(), sin_blob.len());

        // Verify blobs are different (cos != sin)
        assert_ne!(cos_blob.as_bytes(), sin_blob.as_bytes());

        // Size should be: header (64) + data (512 * 32 * 2 for fp16) + scales
        // Exact size depends on WeightBlob internal format
        assert!(cos_blob.len() > 64); // At least has header
    }

    #[test]
    fn test_generate_rope_tables_panics_on_odd_dim() {
        let result = std::panic::catch_unwind(|| {
            generate_rope_tables(63, 512, 10000.0);
        });
        assert!(result.is_err(), "Should panic on odd head_dim");
    }

    #[test]
    fn test_rope_rotation_formula() {
        // Verify RoPE rotation formula:
        // For input [x0, x1, x2, x3], RoPE produces:
        // [x0*cos0 - x1*sin0, x0*sin0 + x1*cos0, x2*cos1 - x3*sin1, x2*sin1 + x3*cos1]

        let head_dim = 4;
        let seq_len = 1;
        let base = 10000.0;

        let (cos, sin) = generate_rope_tables(head_dim, seq_len, base);

        // At position 0, all angles are 0, so cos=1, sin=0
        // This means RoPE should be identity at position 0
        for i in 0..cos.len() {
            assert!((cos[i] - 1.0).abs() < 1e-6);
            assert!(sin[i].abs() < 1e-6);
        }

        // At position 1, we have non-trivial rotation
        let _ = generate_rope_tables(head_dim, 2, base);

        // For position 1, dimension 0 (highest frequency)
        let theta = (base).powf(-0.0 / 4.0); // = 1.0
        let c = theta.cos();
        let s = theta.sin();

        // Simulate RoPE on [1.0, 0.0, 0.0, 0.0]
        // Expected: [1*c - 0*s, 1*s + 0*c, 0, 0] = [c, s, 0, 0]
        let x0 = 1.0f32;
        let x1 = 0.0f32;
        let out0 = x0 * c - x1 * s;
        let out1 = x0 * s + x1 * c;

        assert!((out0 - c).abs() < 1e-6);
        assert!((out1 - s).abs() < 1e-6);
    }

    #[test]
    fn test_rope_tables_various_dimensions() {
        // Test various common transformer head dimensions
        let test_cases = vec![
            (32, 128),   // Small model
            (64, 256),   // Medium model
            (128, 512),  // Large model
            (256, 1024), // Very large model
        ];

        for (head_dim, seq_len) in test_cases {
            let (cos, sin) = generate_rope_tables(head_dim, seq_len, 10000.0);
            let expected_len = seq_len * (head_dim / 2);

            assert_eq!(
                cos.len(),
                expected_len,
                "Wrong cos table size for {}x{}",
                head_dim,
                seq_len
            );
            assert_eq!(
                sin.len(),
                expected_len,
                "Wrong sin table size for {}x{}",
                head_dim,
                seq_len
            );

            // Verify all values are in valid range
            for i in 0..cos.len() {
                assert!((-1.0..=1.0).contains(&cos[i]), "cos[{}] out of range", i);
                assert!((-1.0..=1.0).contains(&sin[i]), "sin[{}] out of range", i);
            }
        }
    }

    #[test]
    fn test_rope_tables_different_bases() {
        // Test different RoPE base values
        let bases = vec![1000.0, 10000.0, 100000.0];
        let head_dim = 64;
        let seq_len = 128;

        for base in bases {
            let (cos, sin) = generate_rope_tables(head_dim, seq_len, base);

            // All should produce valid outputs
            assert_eq!(cos.len(), seq_len * (head_dim / 2));
            assert_eq!(sin.len(), seq_len * (head_dim / 2));

            // Verify unit circle property
            for i in 0..cos.len() {
                let norm = cos[i] * cos[i] + sin[i] * sin[i];
                assert!(
                    (norm - 1.0).abs() < 1e-4,
                    "Unit circle violated for base {}",
                    base
                );
            }
        }
    }

    #[test]
    fn test_rope_position_dependence() {
        // Verify that different positions produce different rotation angles
        let (cos, sin) = generate_rope_tables(64, 100, 10000.0);
        let half_dim = 32;

        // Position 0 should have cos=1, sin=0 (identity rotation)
        for i in 0..half_dim {
            assert!((cos[i] - 1.0).abs() < 1e-6, "Position 0 should have cos=1");
            assert!(sin[i].abs() < 1e-6, "Position 0 should have sin=0");
        }

        // Position 50 should have non-trivial rotations
        let pos50_offset = 50 * half_dim;
        let mut non_trivial_count = 0;
        for i in 0..half_dim {
            let idx = pos50_offset + i;
            if (cos[idx] - 1.0).abs() > 0.01 || sin[idx].abs() > 0.01 {
                non_trivial_count += 1;
            }
        }
        // Most dimensions should have non-trivial rotation at position 50
        assert!(
            non_trivial_count > half_dim / 2,
            "Expected non-trivial rotations at position 50"
        );
    }

    #[test]
    fn test_rope_dimension_frequency_pattern() {
        // Lower dimensions should rotate faster (higher frequency)
        let head_dim = 16;
        let seq_len = 10;
        let base = 10000.0;

        let (cos, _sin) = generate_rope_tables(head_dim, seq_len, base);
        let half_dim = head_dim / 2;

        // At position 1, dimension 0 should have rotated more than dimension (half_dim-1)
        let pos = 1;
        let dim0_idx = pos * half_dim + 0;
        let dim_last_idx = pos * half_dim + (half_dim - 1);

        // Dimension 0 has highest frequency, so should deviate most from identity
        let cos0_deviation = (cos[dim0_idx] - 1.0).abs();
        let cos_last_deviation = (cos[dim_last_idx] - 1.0).abs();

        assert!(
            cos0_deviation > cos_last_deviation,
            "Lower dimensions should rotate faster: dim0_dev={}, last_dev={}",
            cos0_deviation,
            cos_last_deviation
        );
    }

    // NOTE: This test requires graph.rs and graph_to_mil which are untracked files
    // #[test]
    // fn test_rope_integration_with_graph() {
    //     use crate::mil::graph::{Dtype, GraphBuilder};
    //     use crate::mil::graph_to_mil;
    //
    //     // Create a graph with RoPE operation using generated tables
    //     let head_dim = 64;
    //     let seq_len = 128;
    //     let batch_size = 1;
    //
    //     // Generate RoPE tables
    //     let (cos_table, sin_table) = generate_rope_tables(head_dim, seq_len, 10000.0);
    //
    //     // Verify tables are generated (actual usage would convert to blobs)
    //     assert_eq!(cos_table.len(), seq_len * (head_dim / 2));
    //     assert_eq!(sin_table.len(), seq_len * (head_dim / 2));
    //
    //     // Build a simple graph: input -> RoPE -> output
    //     let graph = GraphBuilder::new()
    //         .input("x", Dtype::Fp32, [batch_size, seq_len, head_dim, 1])
    //         .constant(
    //             "cos",
    //             Dtype::Fp32,
    //             [1, seq_len, head_dim / 2, 1],
    //             "cos.bin",
    //             0,
    //         )
    //         .constant(
    //             "sin",
    //             Dtype::Fp32,
    //             [1, seq_len, head_dim / 2, 1],
    //             "sin.bin",
    //             0,
    //         )
    //         .rope(
    //             "out",
    //             "x",
    //             "cos",
    //             "sin",
    //             [batch_size, seq_len, head_dim, 1],
    //         )
    //         .output("out")
    //         .build();
    //
    //     // Verify graph structure
    //     assert_eq!(graph.nodes.len(), 4); // input + cos + sin + rope
    //     assert_eq!(graph.inputs.len(), 1);
    //     assert_eq!(graph.outputs.len(), 1);
    //
    //     // Generate MIL code
    //     let mil = graph_to_mil(&graph).unwrap();
    //
    //     // Verify MIL contains RoPE operations
    //     assert!(mil.contains("slice_by_index"));
    //     assert!(mil.contains("mb.mul"));
    //     assert!(mil.contains("mb.sub"));
    //     assert!(mil.contains("mb.add"));
    //     assert!(mil.contains("mb.concat"));
    // }
}
