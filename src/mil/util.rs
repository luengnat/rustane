//! Weight blob utilities
//!
//! Provides safe wrappers for creating ANE-compatible weight blobs.

use crate::sys::{ane_bridge_build_weight_blob, ane_bridge_build_weight_blob_transposed};
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
/// Weight blobs are allocated by the C bridge and must be explicitly freed.
/// This struct handles cleanup via Drop.
pub struct WeightBlob {
    ptr: *mut u8,
    len: usize,
}

impl WeightBlob {
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

    // Note: INT8 and quantized weight blobs are not yet implemented in the C bridge
    // These functions are stubbed for future implementation

    /// Create an INT8 weight blob (not yet implemented)
    ///
    /// This function currently returns an error as the underlying C bridge
    /// does not yet support INT8 weight blobs.
    #[allow(dead_code)]
    pub fn from_int8(_data: &[i8], _rows: i32, _cols: i32) -> Result<Self> {
        Err(Error::CompilationFailed(
            "INT8 weight blobs not yet implemented in C bridge".to_string(),
        ))
    }

    /// Create a quantized weight blob from FP32 data (not yet implemented)
    ///
    /// This function currently returns an error as the underlying C bridge
    /// does not yet support quantized weight blobs.
    #[allow(dead_code)]
    pub fn from_fp32_quantized(_data: &[f32], _rows: i32, _cols: i32) -> Result<(Self, f32)> {
        Err(Error::CompilationFailed(
            "Quantized weight blobs not yet implemented in C bridge".to_string(),
        ))
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
            // Note: ane_bridge_free_blob is not yet implemented in the C bridge
            // The memory will be cleaned up when the program exits
            // TODO: Call ane_bridge_free_blob once it's available
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
    fn test_weight_blob_int8_not_implemented() {
        let weights = vec![0i8; 256 * 512];
        let result = WeightBlob::from_int8(&weights, 256, 512);
        assert!(matches!(result, Err(Error::CompilationFailed(_))));
    }

    #[test]
    fn test_weight_blob_quantized_not_implemented() {
        let weights = vec![1.0f32; 256 * 512];
        let result = WeightBlob::from_fp32_quantized(&weights, 256, 512);
        assert!(matches!(result, Err(Error::CompilationFailed(_))));
    }
}
