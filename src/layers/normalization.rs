//! Normalization layers
//!
//! Provides LayerNorm and RMSNorm implementations commonly used in transformers.

use crate::layers::traits::{Layer, Shape};
use crate::{Error, Result};

/// Layer Normalization
///
/// Applies Layer Normalization over the last dimension of the input tensor.
///
/// Formula: `y = (x - mean) / sqrt(variance + eps) * gamma + beta`
///
/// # Example
///
/// ```rust
/// use rustane::layers::{LayerNormBuilder, Layer};
///
/// let norm = LayerNormBuilder::new(768)
///     .with_epsilon(1e-5)
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LayerNorm {
    name: String,
    normalized_shape: Shape,
    epsilon: f32,
    gamma: Vec<f32>, // Scale
    beta: Vec<f32>,  // Shift
    input_shape: Shape,
    output_shape: Shape,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    pub fn new(normalized_shape: usize) -> Result<Self> {
        Self::with_epsilon(normalized_shape, 1e-5)
    }

    /// Create LayerNorm with custom epsilon
    pub fn with_epsilon(normalized_shape: usize, epsilon: f32) -> Result<Self> {
        if normalized_shape == 0 {
            return Err(Error::InvalidParameter(
                "normalized_shape must be > 0".into(),
            ));
        }

        // Initialize gamma to 1.0 and beta to 0.0 (standard initialization)
        let gamma = vec![1.0f32; normalized_shape];
        let beta = vec![0.0f32; normalized_shape];

        Ok(Self {
            name: "layer_norm".to_string(),
            normalized_shape: vec![normalized_shape],
            epsilon,
            gamma,
            beta,
            input_shape: vec![normalized_shape],
            output_shape: vec![normalized_shape],
        })
    }

    /// Forward pass for LayerNorm
    ///
    /// Input: [batch_size, ..., normalized_shape]
    /// Output: [batch_size, ..., normalized_shape]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_shape.iter().product::<usize>() {
            return Err(Error::InvalidTensorShape(format!(
                "input size {} doesn't match expected {:?}",
                input.len(),
                self.input_shape
            )));
        }

        // For simplicity, assume 2D input: [batch, normalized_shape]
        let batch_size = input.len() / self.normalized_shape[0];
        let normalized_size = self.normalized_shape[0];
        let mut output = vec![0.0f32; input.len()];

        for b in 0..batch_size {
            let offset = b * normalized_size;
            let slice = &input[offset..offset + normalized_size];

            // Compute mean
            let mean: f32 = slice.iter().sum::<f32>() / normalized_size as f32;

            // Compute variance
            let variance: f32 = slice
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / normalized_size as f32;

            // Normalize and apply gamma/beta
            let std_dev = (variance + self.epsilon).sqrt();

            for (i, &x) in slice.iter().enumerate() {
                let normalized = (x - mean) / std_dev;
                output[offset + i] = normalized * self.gamma[i] + self.beta[i];
            }
        }

        Ok(output)
    }

    /// Get epsilon value
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get gamma (scale) parameters
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Get beta (shift) parameters
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }
}

impl Layer for LayerNorm {
    fn forward(
        &self,
        _executor: &mut crate::wrapper::ANEExecutor,
        _input_idx: usize,
        _output_idx: usize,
    ) -> Result<()> {
        // Note: Normalization is CPU-only in this implementation
        // Use the forward() method for CPU-based computation
        Err(Error::ExecutionFailed(
            "LayerNorm forward through ANE not implemented. Use CPU forward() method instead."
                .to_string(),
        ))
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn input_shape(&self) -> &Shape {
        &self.input_shape
    }

    fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn num_parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// RMS Norm
// ============================================================================

/// RMS Normalization (Root Mean Square Normalization)
///
/// A simplified normalization variant used in modern LLMs (LLaMA, PaLM, etc.).
/// More efficient than LayerNorm as it doesn't center the data (no mean subtraction).
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * gamma`
///
/// # Example
///
/// ```rust
/// use rustane::layers::{RMSNormBuilder, Layer};
///
/// let norm = RMSNormBuilder::new(768)
///     .with_epsilon(1e-6)
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RMSNorm {
    name: String,
    normalized_shape: Shape,
    epsilon: f32,
    gamma: Vec<f32>, // Scale only (no beta/shift)
    input_shape: Shape,
    output_shape: Shape,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(normalized_shape: usize) -> Result<Self> {
        Self::with_epsilon(normalized_shape, 1e-6)
    }

    /// Create RMSNorm with custom epsilon
    pub fn with_epsilon(normalized_shape: usize, epsilon: f32) -> Result<Self> {
        if normalized_shape == 0 {
            return Err(Error::InvalidParameter(
                "normalized_shape must be > 0".into(),
            ));
        }

        // Initialize gamma to 1.0
        let gamma = vec![1.0f32; normalized_shape];

        Ok(Self {
            name: "rms_norm".to_string(),
            normalized_shape: vec![normalized_shape],
            epsilon,
            gamma,
            input_shape: vec![normalized_shape],
            output_shape: vec![normalized_shape],
        })
    }

    /// Forward pass for RMSNorm
    ///
    /// Input: [batch_size, ..., normalized_shape]
    /// Output: [batch_size, ..., normalized_shape]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_shape.iter().product::<usize>() {
            return Err(Error::InvalidTensorShape(format!(
                "input size {} doesn't match expected {:?}",
                input.len(),
                self.input_shape
            )));
        }

        // For simplicity, assume 2D input: [batch, normalized_shape]
        let batch_size = input.len() / self.normalized_shape[0];
        let normalized_size = self.normalized_shape[0];
        let mut output = vec![0.0f32; input.len()];

        for b in 0..batch_size {
            let offset = b * normalized_size;
            let slice = &input[offset..offset + normalized_size];

            // Compute mean of squares
            let mean_square: f32 =
                slice.iter().map(|&x| x * x).sum::<f32>() / normalized_size as f32;

            // RMS normalization
            let rms = (mean_square + self.epsilon).sqrt();

            for (i, &x) in slice.iter().enumerate() {
                output[offset + i] = (x / rms) * self.gamma[i];
            }
        }

        Ok(output)
    }

    /// Get epsilon value
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get gamma (scale) parameters
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }
}

impl Layer for RMSNorm {
    fn forward(
        &self,
        _executor: &mut crate::wrapper::ANEExecutor,
        _input_idx: usize,
        _output_idx: usize,
    ) -> Result<()> {
        // Note: Normalization is CPU-only in this implementation
        // Use the forward() method for CPU-based computation
        Err(Error::ExecutionFailed(
            "RMSNorm forward through ANE not implemented. Use CPU forward() method instead."
                .to_string(),
        ))
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn input_shape(&self) -> &Shape {
        &self.input_shape
    }

    fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn num_parameters(&self) -> usize {
        self.gamma.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// Builders
// ============================================================================

/// Builder for LayerNorm
pub struct LayerNormBuilder {
    normalized_shape: usize,
    epsilon: f32,
    name: String,
}

impl LayerNormBuilder {
    /// Create a builder for a LayerNorm with the given normalized size.
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            epsilon: 1e-5,
            name: "layer_norm".to_string(),
        }
    }

    /// Override the epsilon used for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set a custom display name for the layer.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build the configured [`LayerNorm`].
    pub fn build(self) -> Result<LayerNorm> {
        let mut norm = LayerNorm::with_epsilon(self.normalized_shape, self.epsilon)?;
        norm.name = self.name;
        Ok(norm)
    }
}

/// Builder for RMSNorm
pub struct RMSNormBuilder {
    normalized_shape: usize,
    epsilon: f32,
    name: String,
}

impl RMSNormBuilder {
    /// Create a builder for an RMSNorm with the given normalized size.
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            epsilon: 1e-6,
            name: "rms_norm".to_string(),
        }
    }

    /// Override the epsilon used for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set a custom display name for the layer.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build the configured [`RMSNorm`].
    pub fn build(self) -> Result<RMSNorm> {
        let mut norm = RMSNorm::with_epsilon(self.normalized_shape, self.epsilon)?;
        norm.name = self.name;
        Ok(norm)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let norm = LayerNorm::new(768).unwrap();
        assert_eq!(norm.num_parameters(), 768 * 2); // gamma + beta
        assert_eq!(norm.normalized_shape, vec![768]);
    }

    #[test]
    fn test_rms_norm_creation() {
        let norm = RMSNorm::new(768).unwrap();
        assert_eq!(norm.num_parameters(), 768); // gamma only
        assert_eq!(norm.normalized_shape, vec![768]);
    }

    #[test]
    fn test_layer_norm_forward() {
        let norm = LayerNorm::new(4).unwrap();

        // Simple test: input with known statistics
        let input = vec![
            2.0, 4.0, // batch 0: mean=3, var=2
            6.0, 8.0, // batch 1: mean=7, var=2
        ];

        let output = norm.forward(&input).unwrap();

        // Output should be normalized (approximately unit variance)
        // We just check that it doesn't panic and produces output
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_rms_norm_forward() {
        let norm = RMSNorm::new(4).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0]; // single batch

        let output = norm.forward(&input).unwrap();

        // RMS norm should preserve relative magnitudes but normalize scale
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_layer_norm_builder() {
        let norm = LayerNormBuilder::new(256)
            .with_epsilon(1e-6)
            .with_name("my_norm")
            .build()
            .unwrap();

        assert_eq!(norm.name(), "my_norm");
        assert_eq!(norm.epsilon(), 1e-6);
    }

    #[test]
    fn test_rms_norm_builder() {
        let norm = RMSNormBuilder::new(512)
            .with_epsilon(1e-5)
            .with_name("my_rms")
            .build()
            .unwrap();

        assert_eq!(norm.name(), "my_rms");
        assert_eq!(norm.epsilon(), 1e-5);
    }

    #[test]
    fn test_rms_norm_smaller_than_layer_norm() {
        // RMSNorm has fewer parameters (no beta)
        let layer_norm = LayerNorm::new(100).unwrap();
        let rms_norm = RMSNorm::new(100).unwrap();

        assert_eq!(layer_norm.num_parameters(), 200); // gamma + beta
        assert_eq!(rms_norm.num_parameters(), 100); // gamma only
    }
}
