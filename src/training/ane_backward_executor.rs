//! ANE backward execution and gradient accumulation
//!
//! This module provides gradient accumulation on ANE memory for efficient training.
//!
//! # Architecture
//!
//! Gradients are accumulated in ANE memory (IOSurface) across multiple chunks,
//! then transferred to CPU once per training step for optimizer updates.
//!
//! # Flow
//!
//! ```text
//! Training Step:
//! ┌─────────────────────────────────────────┐
//! │ Forward Pass (Phase 2)                  │
//! │  Activations cached in IOSurface (FP16) │
//! └──────────────┬──────────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────────┐
//! │ Backward Pass (Phase 3)                 │
//! │  1. Loss backward → dlogits             │
//! │  2. Attention backward → d_attn_params  │
//! │  3. FFN backward → d_ffn_params         │
//! │  4. RMSNorm backward → d_norm_params    │
//! │  All gradients accumulated in IOSurface │
//! └──────────────┬──────────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────────┐
//! │ Transfer to CPU                         │
//! │  accumulated_gradients → Vec<f32>       │
//! └──────────────┬──────────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────────┐
//! │ Optimizer Step                          │
//! │  params -= lr * gradients               │
//! └─────────────────────────────────────────┘
//! ```

use crate::ane::{ANEError, IOSurface, Result as ANEResult};
use std::slice;

/// Gradient accumulator for ANE memory
///
/// Manages gradient accumulation in IOSurface across multiple training chunks.
/// Gradients stay in ANE memory to minimize CPU↔ANE transfers.
pub struct ANEGradientAccumulator {
    /// IOSurface for gradient storage on ANE
    accumulator_surface: IOSurface,

    /// Number of parameters (gradient vector size)
    num_params: usize,

    /// Precision for gradient storage (FP32 for numerical stability)
    precision: Precision,

    /// Number of accumulation steps completed
    steps_completed: u32,
}

/// Precision for gradient storage
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    /// 32-bit floating point (default for gradients)
    FP32,
    /// 16-bit floating point (experimental, not recommended)
    FP16,
}

impl ANEGradientAccumulator {
    /// Create new gradient accumulator
    ///
    /// # Arguments
    /// - `num_params`: Number of parameters (gradient vector size)
    ///
    /// # Returns
    /// New accumulator with zero-initialized IOSurface
    ///
    /// # Example
    /// ```ignore
    /// let accum = ANEGradientAccumulator::new(6_800_000)?;
    /// ```
    pub fn new(num_params: usize) -> ANEResult<Self> {
        let precision = Precision::FP32;
        let bytes_per_param = match precision {
            Precision::FP32 => 4,
            Precision::FP16 => 2,
        };

        let buffer_size = num_params * bytes_per_param;

        // Create IOSurface for gradient storage
        let accumulator_surface = IOSurface::new(buffer_size)
            .map_err(|e| ANEError::IOSurfaceError(e.to_string()))?;

        Ok(ANEGradientAccumulator {
            accumulator_surface,
            num_params,
            precision,
            steps_completed: 0,
        })
    }

    /// Accumulate gradients from one backward pass
    ///
    /// # Arguments
    /// - `gradients`: Gradient vector from model backward pass
    /// - `scale`: Scaling factor (usually 1.0 / accumulation_steps)
    ///
    /// # Process
    /// 1. Transfer gradients from CPU to ANE
    /// 2. Scale gradients by factor
    /// 3. Accumulate into IOSurface
    /// 4. Increment step counter
    pub fn accumulate(&mut self, gradients: &[f32], scale: f32) -> ANEResult<()> {
        if gradients.len() != self.num_params {
            return Err(ANEError::InvalidShape {
                expected: format!("{}", self.num_params),
                got: format!("{}", gradients.len()),
            });
        }

        // Scale gradients
        let scaled_gradients: Vec<f32> = gradients.iter().map(|g| g * scale).collect();

        // Read current accumulated gradients
        let mut accumulated = self.get_accumulated()?;

        // Accumulate
        for (accum, grad) in accumulated.iter_mut().zip(scaled_gradients.iter()) {
            *accum += grad;
        }

        // Convert f32 slice to u8 bytes for IOSurface
        let accumulated_bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                accumulated.as_ptr() as *const u8,
                accumulated.len() * std::mem::size_of::<f32>(),
            )
        };

        // Write back to IOSurface
        self.accumulator_surface.write(accumulated_bytes)
            .map_err(|e| ANEError::IOSurfaceError(e.to_string()))?;

        self.steps_completed += 1;

        Ok(())
    }

    /// Get accumulated gradients (transfer from ANE to CPU)
    ///
    /// # Returns
    /// Copy of accumulated gradients as CPU vector
    ///
    /// # Note
    /// This performs ANE→CPU transfer, so use sparingly (once per training step)
    pub fn get_accumulated(&self) -> ANEResult<Vec<f32>> {
        let bytes = self.accumulator_surface.read()
            .map_err(|e| ANEError::IOSurfaceError(e.to_string()))?;

        // Convert u8 bytes to f32 slice
        let f32_slice: &[f32] = unsafe {
            slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / std::mem::size_of::<f32>(),
            )
        };

        Ok(f32_slice.to_vec())
    }

    /// Reset accumulator for next training step
    ///
    /// # Process
    /// 1. Zero out IOSurface
    /// 2. Reset step counter
    pub fn reset(&mut self) -> ANEResult<()> {
        let zeros_f32 = vec![0.0f32; self.num_params];

        // Convert f32 slice to u8 bytes for IOSurface
        let zeros_bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                zeros_f32.as_ptr() as *const u8,
                zeros_f32.len() * std::mem::size_of::<f32>(),
            )
        };

        self.accumulator_surface.write(zeros_bytes)
            .map_err(|e| ANEError::IOSurfaceError(e.to_string()))?;
        self.steps_completed = 0;
        Ok(())
    }

    /// Get number of accumulation steps completed
    pub fn steps_completed(&self) -> u32 {
        self.steps_completed
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Get precision
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_accumulator_creation() {
        let accum = ANEGradientAccumulator::new(1000).unwrap();
        assert_eq!(accum.num_params(), 1000);
        assert_eq!(accum.steps_completed(), 0);
        assert_eq!(accum.precision(), Precision::FP32);
    }

    #[test]
    fn test_gradient_accumulator_accumulate() {
        let mut accum = ANEGradientAccumulator::new(10).unwrap();

        // First accumulation
        let grads1 = vec![1.0; 10];
        accum.accumulate(&grads1, 0.5).unwrap();

        assert_eq!(accum.steps_completed(), 1);

        let accumulated = accum.get_accumulated().unwrap();
        assert_eq!(accumulated.len(), 10);
        for grad in &accumulated {
            assert_eq!(*grad, 0.5); // 1.0 * 0.5
        }
    }

    #[test]
    fn test_gradient_accumulator_multiple_steps() {
        let mut accum = ANEGradientAccumulator::new(10).unwrap();

        // First accumulation
        let grads1 = vec![1.0; 10];
        accum.accumulate(&grads1, 0.5).unwrap();

        // Second accumulation
        let grads2 = vec![2.0; 10];
        accum.accumulate(&grads2, 0.5).unwrap();

        assert_eq!(accum.steps_completed(), 2);

        let accumulated = accum.get_accumulated().unwrap();
        // First: 1.0 * 0.5 = 0.5
        // Second: 2.0 * 0.5 = 1.0
        // Total: 0.5 + 1.0 = 1.5
        for grad in &accumulated {
            assert_eq!(*grad, 1.5);
        }
    }

    #[test]
    fn test_gradient_accumulator_reset() {
        let mut accum = ANEGradientAccumulator::new(10).unwrap();

        let grads = vec![1.0; 10];
        accum.accumulate(&grads, 0.5).unwrap();

        assert_eq!(accum.steps_completed(), 1);

        accum.reset().unwrap();

        assert_eq!(accum.steps_completed(), 0);

        let accumulated = accum.get_accumulated().unwrap();
        for grad in &accumulated {
            assert_eq!(*grad, 0.0);
        }
    }

    #[test]
    fn test_gradient_accumulator_length_mismatch() {
        let mut accum = ANEGradientAccumulator::new(10).unwrap();

        let grads = vec![1.0; 5]; // Wrong length
        let result = accum.accumulate(&grads, 0.5);

        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_accumulator_large() {
        // Test with realistic model size
        let num_params = 6_800_000;
        let mut accum = ANEGradientAccumulator::new(num_params).unwrap();

        assert_eq!(accum.num_params(), num_params);

        // Verify we can read/write
        let zeros_f32 = vec![0.0f32; num_params];
        let zeros_bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                zeros_f32.as_ptr() as *const u8,
                zeros_f32.len() * std::mem::size_of::<f32>(),
            )
        };
        accum.accumulator_surface.write(zeros_bytes).unwrap();

        let read_bytes = accum.accumulator_surface.read().unwrap();
        assert_eq!(read_bytes.len(), num_params * std::mem::size_of::<f32>());
    }
}
