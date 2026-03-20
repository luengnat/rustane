//! SwiGLU Activation Function
//!
//! SwiGLU (Swish-Gated Linear Unit) is a modern activation function used in
//! state-of-the-art language models like LLaMA, PaLM, and GPT-NeoX.
//!
//! Formula: `SwiGLU(x) = Swish(xW) ⊗ (xV)`
//! where:
//!   - Swish(x) = x * sigmoid(x)
//!   - ⊗ denotes element-wise multiplication
//!   - W and V are learned weight matrices
//!
//! This is typically used as: `SwiGLU(x) = (xW_gate * sigmoid(xW_gate)) ⊗ (xW_up)`

use crate::layers::traits::{Layer, Shape};
use crate::{Error, Result};

/// SwiGLU Activation Layer
///
/// Implements the SwiGLU activation function with learned weights.
/// More expressive than standard ReLU/GELU for transformer models.
///
/// # Architecture
///
/// ```text
/// Input → [Split] → Gate Branch (W_gate) → SiLU → [Mul] → Output
///              → Up Branch   (W_up)   ------^
/// ```
///
/// # Example
///
/// ```rust
/// use rustane::layers::{SwiGLUBuilder, Layer};
///
/// let swiglu = SwiGLUBuilder::new(768)
///     .with_multiplier(4)  // 4x expansion like LLaMA
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SwiGLU {
    name: String,
    input_dim: usize,
    multiplier: usize,
    hidden_dim: usize,
    w_gate: Vec<f32>, // Gate weights
    w_up: Vec<f32>,   // Up weights
    w_down: Vec<f32>, // Down projection weights
    input_shape: Shape,
    output_shape: Shape,
}

impl SwiGLU {
    /// Create a new SwiGLU layer
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    ///
    /// Uses default 4x expansion (multiplier = 4)
    pub fn new(input_dim: usize) -> Result<Self> {
        Self::with_multiplier(input_dim, 4)
    }

    /// Create SwiGLU with custom expansion multiplier
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `multiplier` - Expansion factor (typically 2, 4, or 8)
    pub fn with_multiplier(input_dim: usize, multiplier: usize) -> Result<Self> {
        if input_dim == 0 {
            return Err(Error::InvalidParameter("input_dim must be > 0".into()));
        }
        if multiplier == 0 {
            return Err(Error::InvalidParameter("multiplier must be > 0".into()));
        }

        let hidden_dim = input_dim * multiplier;

        // Initialize weights (Xavier/Glorot initialization)
        let std_dev = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let w_gate: Vec<f32> = (0..input_dim * hidden_dim)
            .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * std_dev)
            .collect();
        let w_up: Vec<f32> = (0..input_dim * hidden_dim)
            .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * std_dev)
            .collect();

        // Down projection weights
        let std_dev_down = (2.0 / (hidden_dim + input_dim) as f32).sqrt();
        let w_down: Vec<f32> = (0..hidden_dim * input_dim)
            .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * std_dev_down)
            .collect();

        Ok(Self {
            name: "swiglu".to_string(),
            input_dim,
            multiplier,
            hidden_dim,
            w_gate,
            w_up,
            w_down,
            input_shape: vec![input_dim],
            output_shape: vec![input_dim],
        })
    }

    /// Forward pass for SwiGLU
    ///
    /// # Architecture
    ///
    /// 1. Gate branch: x @ W_gate → SiLU activation
    /// 2. Up branch: x @ W_up
    /// 3. Element-wise multiply: gate * up
    /// 4. Down projection: (gate * up) @ W_down
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size, input_dim]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() % self.input_dim != 0 {
            return Err(Error::InvalidTensorShape(format!(
                "input size {} must be multiple of input_dim {}",
                input.len(),
                self.input_dim
            )));
        }

        let batch_size = input.len() / self.input_dim;
        let mut output = vec![0.0f32; input.len()];

        for b in 0..batch_size {
            let offset = b * self.input_dim;
            let x = &input[offset..offset + self.input_dim];

            // Gate branch: x @ W_gate → SiLU
            let mut gate = vec![0.0f32; self.hidden_dim];
            for i in 0..self.hidden_dim {
                for j in 0..self.input_dim {
                    gate[i] += x[j] * self.w_gate[j * self.hidden_dim + i];
                }
                // Apply SiLU (Swish): x * sigmoid(x)
                gate[i] = gate[i] * (1.0 / (1.0 + (-gate[i]).exp()));
            }

            // Up branch: x @ W_up
            let mut up = vec![0.0f32; self.hidden_dim];
            for i in 0..self.hidden_dim {
                for j in 0..self.input_dim {
                    up[i] += x[j] * self.w_up[j * self.hidden_dim + i];
                }
            }

            // Element-wise multiply
            let mut hidden = vec![0.0f32; self.hidden_dim];
            for i in 0..self.hidden_dim {
                hidden[i] = gate[i] * up[i];
            }

            // Down projection: hidden @ W_down
            let out_offset = b * self.input_dim;
            for i in 0..self.input_dim {
                for j in 0..self.hidden_dim {
                    output[out_offset + i] += hidden[j] * self.w_down[j * self.input_dim + i];
                }
            }
        }

        Ok(output)
    }

    /// Get expansion multiplier
    pub fn multiplier(&self) -> usize {
        self.multiplier
    }

    /// Get hidden dimension size
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get gate weights
    pub fn w_gate(&self) -> &[f32] {
        &self.w_gate
    }

    /// Get up weights
    pub fn w_up(&self) -> &[f32] {
        &self.w_up
    }

    /// Get down weights
    pub fn w_down(&self) -> &[f32] {
        &self.w_down
    }
}

impl Layer for SwiGLU {
    fn forward(
        &self,
        _executor: &mut crate::wrapper::ANEExecutor,
        _input_idx: usize,
        _output_idx: usize,
    ) -> Result<()> {
        // Note: SwiGLU is CPU-only in this implementation
        // Use the forward() method for CPU-based computation
        Err(Error::ExecutionFailed(
            "SwiGLU forward through ANE not implemented. Use CPU forward() method instead."
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
        self.w_gate.len() + self.w_up.len() + self.w_down.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// SiLU (Swish) Activation
// ============================================================================

/// SiLU (Sigmoid Linear Unit) / Swish Activation
///
/// Formula: `SiLU(x) = x * sigmoid(x) = x / (1 + e^{-x})`
///
/// Smooth, non-monotonic activation that performs well in practice.
/// Used as the activation in SwiGLU's gate branch.
#[derive(Clone, Copy)]
pub struct SiLU;

impl SiLU {
    /// Apply the SiLU activation to a single scalar.
    pub fn forward(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    /// Apply the SiLU activation elementwise to a slice.
    pub fn forward_batch(input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| Self::forward(x)).collect()
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for SwiGLU
pub struct SwiGLUBuilder {
    input_dim: usize,
    multiplier: usize,
    name: String,
}

impl SwiGLUBuilder {
    /// Create a builder for a SwiGLU block with the given input dimension.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            multiplier: 4, // Default 4x expansion (LLaMA style)
            name: "swiglu".to_string(),
        }
    }

    /// Set expansion multiplier (default: 4)
    ///
    /// Common values:
    /// - 2: For smaller models
    /// - 4: Standard (LLaMA, GPT-NeoX)
    /// - 8: For very large models
    pub fn with_multiplier(mut self, multiplier: usize) -> Self {
        self.multiplier = multiplier;
        self
    }

    /// Set a custom display name for the layer.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build the configured [`SwiGLU`] layer.
    pub fn build(self) -> Result<SwiGLU> {
        let mut swiglu = SwiGLU::with_multiplier(self.input_dim, self.multiplier)?;
        swiglu.name = self.name;
        Ok(swiglu)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_creation() {
        let swiglu = SwiGLU::new(768).unwrap();
        assert_eq!(swiglu.input_dim, 768);
        assert_eq!(swiglu.multiplier(), 4);
        assert_eq!(swiglu.hidden_dim(), 3072); // 768 * 4
    }

    #[test]
    fn test_swiglu_with_multiplier() {
        let swiglu = SwiGLU::with_multiplier(256, 2).unwrap();
        assert_eq!(swiglu.multiplier(), 2);
        assert_eq!(swiglu.hidden_dim(), 512); // 256 * 2
    }

    #[test]
    fn test_swiglu_forward() {
        let swiglu = SwiGLU::new(4).unwrap();

        // Simple input
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ];

        let output = swiglu.forward(&input).unwrap();

        // Should produce output of same shape
        assert_eq!(output.len(), 8);

        // Output should be different from input (non-identity)
        let all_same = input
            .iter()
            .zip(output.iter())
            .all(|(&in_val, &out_val)| (in_val - out_val).abs() < 1e-6);

        assert!(!all_same, "SwiGLU should transform the input");
    }

    #[test]
    fn test_swiglu_parameter_count() {
        let swiglu = SwiGLU::new(128).unwrap();

        // With multiplier=4 (default):
        // w_gate: 128 * 512 = 65536
        // w_up: 128 * 512 = 65536
        // w_down: 512 * 128 = 65536
        // Total: 196608
        let expected = 128 * 512 * 3;
        assert_eq!(swiglu.num_parameters(), expected);
    }

    #[test]
    fn test_silu_activation() {
        // Test SiLU properties
        assert!(SiLU::forward(0.0) == 0.0); // f(0) = 0
        assert!(SiLU::forward(1.0) > 0.0); // Positive
        assert!(SiLU::forward(-1.0) < 0.0); // Negative

        // Smooth around 0
        let x1 = SiLU::forward(0.1);
        let x2 = SiLU::forward(-0.1);
        assert!((x1 - x2).abs() <= 0.1); // Smooth
    }

    #[test]
    fn test_swiglu_builder() {
        let swiglu = SwiGLUBuilder::new(256)
            .with_multiplier(2)
            .with_name("my_swiglu")
            .build()
            .unwrap();

        assert_eq!(swiglu.name(), "my_swiglu");
        assert_eq!(swiglu.multiplier(), 2);
        assert_eq!(swiglu.hidden_dim(), 512);
    }

    #[test]
    fn test_swiglu_batch_processing() {
        let swiglu = SwiGLU::new(64).unwrap();

        // Process multiple batches
        let batch_size = 10;
        let input: Vec<f32> = (0..batch_size * 64).map(|i| i as f32 / 1000.0).collect();

        let output = swiglu.forward(&input).unwrap();

        assert_eq!(output.len(), input.len());

        // Each batch should be processed independently
        // (we can't easily verify correctness without reference values)
    }
}
