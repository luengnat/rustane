//! ANE Backward Executor — Gradient Accumulation on ANE
//!
//! This module provides the [`ANEGradientAccumulator`] for managing gradient
//! accumulation entirely on the ANE device during backward propagation.
//!
//! # Architecture
//!
//! The gradient accumulator maintains an IOSurface-backed buffer on the ANE
//! where gradients are accumulated across multiple backward operations. This
//! minimizes CPU↔ANE transfers by keeping gradients in device memory until
//! the end of the training step.
//!
//! # Data Flow
//!
//! ```text
//! Backward Pass:
//! 1. Each backward kernel outputs gradients to ANE memory
//! 2. ANEGradientAccumulator accumulates these gradients
//! 3. After all backward operations complete, transfer to CPU
//! 4. CPU optimizer updates parameters
//! ```
//!
//! # Precision
//!
//! Gradients are accumulated in FP32 for numerical stability, even when
//! activations are stored in FP16.

use crate::error::{Error, Result};
use crate::training::TransformerConfig;
use crate::wrapper::tensor::{ANETensor, TensorDType};

/// Gradient accumulator for ANE backward pass
///
/// Manages gradient accumulation in ANE memory across multiple backward
/// operations. Gradients stay on the ANE device until explicitly transferred
/// to CPU for the optimizer step.
///
/// # Example
///
/// ```ignore
/// use rustane::training::ane_backward_executor::ANEGradientAccumulator;
///
/// let config = TransformerConfig::tiny();
/// let mut accumulator = ANEGradientAccumulator::new(config.param_count())?;
///
/// // Accumulate gradients from backward operations
/// accumulator.accumulate(&gradient_chunk)?;
///
/// // Get final accumulated gradients
/// let gradients = accumulator.get_accumulated()?;
/// ```
#[derive(Debug)]
pub struct ANEGradientAccumulator {
    /// Gradient storage on CPU (ANE surface simulation)
    /// In production, this would be an IOSurface for ANE memory
    accumulator_buffer: Vec<f32>,
    num_params: usize,
    accumulation_count: usize,
}

impl ANEGradientAccumulator {
    /// Create a new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `num_params` - Total number of trainable parameters
    ///
    /// # Returns
    ///
    /// A new `ANEGradientAccumulator` initialized with zeros
    ///
    /// # Errors
    ///
    /// Returns an error if `num_params` is zero
    ///
    /// # Example
    ///
    /// ```ignore
    /// let accumulator = ANEGradientAccumulator::new(1_000_000)?;
    /// ```
    pub fn new(num_params: usize) -> Result<Self> {
        if num_params == 0 {
            return Err(Error::InvalidParameter(
                "num_params must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            accumulator_buffer: vec![0.0f32; num_params],
            num_params,
            accumulation_count: 0,
        })
    }

    /// Create a new gradient accumulator from a TransformerConfig
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration with parameter count
    ///
    /// # Returns
    ///
    /// A new `ANEGradientAccumulator` sized for the config
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = TransformerConfig::tiny();
    /// let accumulator = ANEGradientAccumulator::from_config(&config)?;
    /// ```
    pub fn from_config(config: &TransformerConfig) -> Result<Self> {
        Self::new(config.param_count())
    }

    /// Accumulate gradients into the accumulator
    ///
    /// Adds the provided gradients element-wise to the accumulator buffer.
    /// This is called after each backward operation to accumulate gradients
    /// across all layers.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient vector to accumulate (must match num_params)
    ///
    /// # Errors
    ///
    /// Returns an error if gradient length doesn't match num_params
    ///
    /// # Example
    ///
    /// ```ignore
    /// let gradients = vec![0.1f32; num_params];
    /// accumulator.accumulate(&gradients)?;
    /// ```
    pub fn accumulate(&mut self, gradients: &[f32]) -> Result<()> {
        if gradients.len() != self.num_params {
            return Err(Error::InvalidParameter(format!(
                "gradient length ({}) doesn't match accumulator size ({})",
                gradients.len(),
                self.num_params
            )));
        }

        for (acc, grad) in self.accumulator_buffer.iter_mut().zip(gradients.iter()) {
            *acc += grad;
        }

        self.accumulation_count += 1;
        Ok(())
    }

    /// Accumulate gradients from an ANETensor
    ///
    /// Similar to `accumulate`, but takes an `ANETensor` as input.
    /// This is useful when gradients are output from ANE kernels as tensors.
    ///
    /// # Arguments
    ///
    /// * `tensor` - ANETensor containing gradients (FP32 only)
    ///
    /// # Errors
    ///
    /// Returns an error if tensor size doesn't match or dtype is not FP32
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grad_tensor = ANETensor::from_fp32(gradients, vec![num_params])?;
    /// accumulator.accumulate_tensor(&grad_tensor)?;
    /// ```
    pub fn accumulate_tensor(&mut self, tensor: &ANETensor) -> Result<()> {
        if tensor.dtype() != TensorDType::FP32 {
            return Err(Error::InvalidParameter(
                "gradient tensor must be FP32".to_string(),
            ));
        }

        if tensor.num_elements() != self.num_params {
            return Err(Error::InvalidParameter(format!(
                "tensor elements ({}) doesn't match accumulator size ({})",
                tensor.num_elements(),
                self.num_params
            )));
        }

        // Convert bytes to f32 slice
        let bytes = tensor.as_bytes();
        let num_floats = bytes.len() / 4;
        let gradients: &[f32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, num_floats) };

        self.accumulate(gradients)
    }

    /// Get the accumulated gradients
    ///
    /// Returns a copy of the accumulated gradients. This is called at the
    /// end of the backward pass to transfer gradients to the optimizer.
    ///
    /// # Returns
    ///
    /// Vector of accumulated gradients (length = num_params)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let gradients = accumulator.get_accumulated()?;
    /// optimizer.step(&gradients)?;
    /// ```
    pub fn get_accumulated(&self) -> Result<Vec<f32>> {
        Ok(self.accumulator_buffer.clone())
    }

    /// Get a reference to the accumulated gradients
    ///
    /// More efficient than `get_accumulated` when you don't need ownership.
    ///
    /// # Returns
    ///
    /// Slice reference to the accumulated gradients
    pub fn get_accumulated_ref(&self) -> &[f32] {
        &self.accumulator_buffer
    }

    /// Reset the accumulator to zeros
    ///
    /// Clears all accumulated gradients and resets the accumulation count.
    /// This should be called at the start of each training step.
    ///
    /// # Example
    ///
    /// ```ignore
    /// accumulator.reset()?;
    /// // Start new training step...
    /// ```
    pub fn reset(&mut self) -> Result<()> {
        self.accumulator_buffer.fill(0.0f32);
        self.accumulation_count = 0;
        Ok(())
    }

    /// Get the number of parameters
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Get the number of accumulation operations performed
    ///
    /// This tracks how many times `accumulate()` has been called since
    /// the last reset.
    pub fn accumulation_count(&self) -> usize {
        self.accumulation_count
    }

    /// Check if accumulator is empty (all zeros)
    ///
    /// Returns true if no gradients have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.accumulator_buffer.iter().all(|&v| v == 0.0f32)
    }

    /// Get the maximum absolute gradient value
    ///
    /// Useful for gradient clipping and debugging.
    pub fn max_abs_gradient(&self) -> f32 {
        self.accumulator_buffer
            .iter()
            .map(|&v| v.abs())
            .fold(0.0f32, f32::max)
    }

    /// Scale all accumulated gradients
    ///
    /// Multiplies all gradients by the given scale factor.
    /// Used for gradient clipping and learning rate scheduling.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factor to apply
    pub fn scale(&mut self, scale: f32) {
        for grad in self.accumulator_buffer.iter_mut() {
            *grad *= scale;
        }
    }
}

/// Trait for models that support ANE backward pass
///
/// This trait extends the base `Model` trait with ANE-specific backward
/// functionality. Models implementing this trait can perform backward
/// propagation entirely on the ANE device.
///
/// # Example
///
/// ```ignore
/// impl ANEBackwardModel for TransformerANE {
///     fn backward_on_ane(&mut self, loss: f32) -> Result<Vec<f32>> {
///         // ANE backward implementation
///     }
/// }
/// ```
pub trait ANEBackwardModel {
    /// Execute backward pass on ANE with gradient accumulation
    ///
    /// This method performs the entire backward pass on the ANE device,
    /// accumulating gradients in ANE memory and returning them to CPU
    /// only at the end.
    ///
    /// # Arguments
    ///
    /// * `loss` - Scalar loss value from the forward pass
    ///
    /// # Returns
    ///
    /// Accumulated gradients as a vector of f32 values
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forward cache is missing
    /// - ANE execution fails
    /// - Gradient computation produces NaN/Inf
    fn backward_on_ane(&mut self, loss: f32) -> Result<Vec<f32>>;

    /// Check if ANE backward is available
    ///
    /// Returns true if the model can perform backward on ANE.
    /// This may return false if:
    /// - ANE hardware is not available
    /// - Model configuration doesn't support ANE backward
    fn ane_backward_available(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_creation() {
        let accumulator = ANEGradientAccumulator::new(1000).unwrap();
        assert_eq!(accumulator.num_params(), 1000);
        assert_eq!(accumulator.accumulation_count(), 0);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn test_accumulator_creation_zero_params() {
        let result = ANEGradientAccumulator::new(0);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_accumulator_from_config() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let accumulator = ANEGradientAccumulator::from_config(&config).unwrap();
        assert_eq!(accumulator.num_params(), config.param_count());
    }

    #[test]
    fn test_accumulate_gradients() {
        let mut accumulator = ANEGradientAccumulator::new(10).unwrap();
        let gradients = vec![0.1f32; 10];

        accumulator.accumulate(&gradients).unwrap();
        assert_eq!(accumulator.accumulation_count(), 1);

        let result = accumulator.get_accumulated().unwrap();
        assert_eq!(result, vec![0.1f32; 10]);
    }

    #[test]
    fn test_accumulate_multiple_times() {
        let mut accumulator = ANEGradientAccumulator::new(5).unwrap();

        accumulator.accumulate(&vec![0.1f32; 5]).unwrap();
        accumulator.accumulate(&vec![0.2f32; 5]).unwrap();
        accumulator.accumulate(&vec![0.3f32; 5]).unwrap();

        let result = accumulator.get_accumulated().unwrap();
        assert_eq!(result, vec![0.6f32; 5]);
        assert_eq!(accumulator.accumulation_count(), 3);
    }

    #[test]
    fn test_accumulate_wrong_size() {
        let mut accumulator = ANEGradientAccumulator::new(10).unwrap();
        let gradients = vec![0.1f32; 5]; // Wrong size

        let result = accumulator.accumulate(&gradients);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_reset() {
        let mut accumulator = ANEGradientAccumulator::new(5).unwrap();

        accumulator.accumulate(&vec![0.5f32; 5]).unwrap();
        assert!(!accumulator.is_empty());

        accumulator.reset().unwrap();
        assert!(accumulator.is_empty());
        assert_eq!(accumulator.accumulation_count(), 0);
    }

    #[test]
    fn test_max_abs_gradient() {
        let mut accumulator = ANEGradientAccumulator::new(5).unwrap();

        accumulator
            .accumulate(&vec![0.1f32, -0.5f32, 0.3f32, -0.2f32, 0.4f32])
            .unwrap();

        assert_eq!(accumulator.max_abs_gradient(), 0.5f32);
    }

    #[test]
    fn test_scale() {
        let mut accumulator = ANEGradientAccumulator::new(5).unwrap();

        accumulator.accumulate(&vec![0.1f32; 5]).unwrap();
        accumulator.scale(2.0f32);

        let result = accumulator.get_accumulated().unwrap();
        assert_eq!(result, vec![0.2f32; 5]);
    }

    #[test]
    fn test_accumulate_tensor() {
        let mut accumulator = ANEGradientAccumulator::new(4).unwrap();
        let gradients = vec![0.1f32, 0.2f32, 0.3f32, 0.4f32];
        let tensor = ANETensor::from_fp32(gradients, vec![4]).unwrap();

        accumulator.accumulate_tensor(&tensor).unwrap();

        let result = accumulator.get_accumulated().unwrap();
        assert_eq!(result, vec![0.1f32, 0.2f32, 0.3f32, 0.4f32]);
    }

    #[test]
    fn test_accumulate_tensor_wrong_dtype() {
        let mut accumulator = ANEGradientAccumulator::new(4).unwrap();
        let data = vec![0x3c00u16; 4]; // FP16 data
        let tensor = ANETensor::from_fp16(data, vec![4]).unwrap();

        let result = accumulator.accumulate_tensor(&tensor);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_get_accumulated_ref() {
        let mut accumulator = ANEGradientAccumulator::new(3).unwrap();
        accumulator
            .accumulate(&vec![0.1f32, 0.2f32, 0.3f32])
            .unwrap();

        let reference = accumulator.get_accumulated_ref();
        assert_eq!(reference, &[0.1f32, 0.2f32, 0.3f32]);
    }
}
