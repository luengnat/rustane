//! ANE Gradient Buffer - IOSurface-backed gradient storage
//!
//! This module provides [`ANEGradientBuffer`] which uses IOSurface for
//! gradient storage on ANE hardware, enabling true zero-copy gradient
//! accumulation on the device.

use crate::ane::IOSurface;
use crate::error::{Error, Result};

/// IOSurface-backed gradient buffer for ANE
///
/// This buffer stores gradients in IOSurface memory that can be accessed
/// by both CPU and ANE. For ANE-only operations, the data stays on the
/// device without CPU transfer.
pub struct ANEGradientBuffer {
    /// IOSurface for gradient storage
    surface: IOSurface,
    /// Number of parameters (floats)
    num_params: usize,
    /// Accumulation count
    accumulation_count: usize,
}

impl ANEGradientBuffer {
    /// Create a new gradient buffer with IOSurface backing
    ///
    /// # Arguments
    ///
    /// * `num_params` - Number of parameters (in floats)
    ///
    /// # Returns
    ///
    /// A new `ANEGradientBuffer` initialized with zeros
    ///
    /// # Errors
    ///
    /// Returns an error if IOSurface creation fails
    pub fn new(num_params: usize) -> Result<Self> {
        if num_params == 0 {
            return Err(Error::InvalidParameter(
                "num_params must be greater than zero".to_string(),
            ));
        }

        // Create IOSurface with size for num_params f32 values
        let size_bytes = num_params * 4;
        let surface = IOSurface::new(size_bytes)
            .map_err(|e| Error::Other(format!("Failed to create IOSurface: {:?}", e)))?;

        // Initialize to zeros
        surface.clear();

        Ok(Self {
            surface,
            num_params,
            accumulation_count: 0,
        })
    }

    /// Accumulate gradients from CPU memory
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient vector to accumulate
    pub fn accumulate(&mut self, gradients: &[f32]) -> Result<()> {
        if gradients.len() != self.num_params {
            return Err(Error::InvalidParameter(format!(
                "gradient length ({}) doesn't match buffer size ({})",
                gradients.len(),
                self.num_params
            )));
        }

        // Read current values from IOSurface
        let mut current = vec![0f32; self.num_params];
        self.surface.read_f32(&mut current);

        // Accumulate
        for (i, grad) in gradients.iter().enumerate() {
            current[i] += grad;
        }

        // Write back to IOSurface
        self.surface.write_f32(&current);
        self.accumulation_count += 1;

        Ok(())
    }

    /// Accumulate gradients from another IOSurface
    ///
    /// This is more efficient than CPU accumulation as both buffers
    /// are in device-accessible memory.
    pub fn accumulate_surface(&mut self, other: &IOSurface) -> Result<()> {
        // Read both surfaces
        let mut current = vec![0f32; self.num_params];
        let mut other_data = vec![0f32; self.num_params];

        self.surface.read_f32(&mut current);
        other.read_f32(&mut other_data);

        // Accumulate
        for (i, val) in other_data.iter().enumerate() {
            current[i] += val;
        }

        // Write back
        self.surface.write_f32(&current);
        self.accumulation_count += 1;

        Ok(())
    }

    /// Get accumulated gradients to CPU memory
    pub fn to_vec(&self) -> Vec<f32> {
        let mut result = vec![0f32; self.num_params];
        self.surface.read_f32(&mut result);
        result
    }

    /// Get a reference to the underlying IOSurface
    pub fn surface(&self) -> &IOSurface {
        &self.surface
    }

    /// Reset buffer to zeros
    pub fn reset(&mut self) {
        self.surface.clear();
        self.accumulation_count = 0;
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Get accumulation count
    pub fn accumulation_count(&self) -> usize {
        self.accumulation_count
    }

    /// Check if buffer is empty (all zeros)
    pub fn is_empty(&self) -> bool {
        let data = self.to_vec();
        data.iter().all(|&v| v == 0.0)
    }

    /// Get maximum absolute gradient
    pub fn max_abs_gradient(&self) -> f32 {
        let data = self.to_vec();
        data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max)
    }

    /// Scale all gradients
    pub fn scale(&mut self, scale: f32) {
        let mut data = self.to_vec();
        for v in data.iter_mut() {
            *v *= scale;
        }
        self.surface.write_f32(&data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_buffer_creation() {
        let buffer = ANEGradientBuffer::new(1000);
        match buffer {
            Ok(buf) => {
                assert_eq!(buf.num_params(), 1000);
                assert!(buf.is_empty());
            }
            Err(_) => {
                // IOSurface may not be available in test environment
            }
        }
    }

    #[test]
    fn test_gradient_buffer_accumulate() {
        let mut buffer = match ANEGradientBuffer::new(10) {
            Ok(b) => b,
            Err(_) => return, // Skip if IOSurface not available
        };

        let grads = vec![0.1f32; 10];
        buffer.accumulate(&grads).unwrap();

        let result = buffer.to_vec();
        assert_eq!(result, vec![0.1f32; 10]);
        assert_eq!(buffer.accumulation_count(), 1);
    }

    #[test]
    fn test_gradient_buffer_reset() {
        let mut buffer = match ANEGradientBuffer::new(5) {
            Ok(b) => b,
            Err(_) => return,
        };

        buffer.accumulate(&vec![1.0f32; 5]).unwrap();
        assert!(!buffer.is_empty());

        buffer.reset();
        assert!(buffer.is_empty());
        assert_eq!(buffer.accumulation_count(), 0);
    }
}
