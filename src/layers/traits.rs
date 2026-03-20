//! Core layer traits and interfaces

use crate::wrapper::ANEExecutor;
use crate::Result;

/// Shape of a tensor
pub type Shape = Vec<usize>;

/// Core layer trait
///
/// All neural network layers implement this trait, providing
/// a common interface for forward passes and shape inference.
///
/// # Example
///
/// ```no_run
/// # use rustane::layers::Layer;
/// # use rustane::wrapper::ANEExecutor;
/// struct MyLayer;
///
/// impl Layer for MyLayer {
///     fn forward(&self, executor: &mut ANEExecutor, input_idx: usize, output_idx: usize) -> Result<()> {
///         // Execute layer operation
///         Ok(())
///     }
///
///     fn input_shape(&self) -> &Shape {
///         &vec![1, 256]
///     }
///
///     fn output_shape(&self) -> &Shape {
///         &vec![1, 512]
///     }
/// }
/// ```
pub trait Layer {
    /// Execute the forward pass
    ///
    /// # Arguments
    ///
    /// * `executor` - ANE executor to run the operation
    /// * `input_idx` - Index of input tensor in executor
    /// * `output_idx` - Index of output tensor in executor
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - ANE execution fails
    /// - Tensor shapes don't match
    /// - Layer is not properly configured
    fn forward(
        &self,
        executor: &mut ANEExecutor,
        input_idx: usize,
        output_idx: usize,
    ) -> Result<()>;

    /// Get the expected input shape
    fn input_shape(&self) -> &Shape;

    /// Get the expected output shape
    fn output_shape(&self) -> &Shape;

    /// Get the layer name
    fn name(&self) -> &str;

    /// Get the number of parameters
    fn num_parameters(&self) -> usize;

    /// Return as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Layer builder trait
///
/// Layers that can be configured implement this trait,
/// providing a fluent builder pattern for construction.
pub trait LayerBuilder {
    /// The type of layer this builder creates
    type Layer: Layer;

    /// Build the layer with current configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration is invalid
    /// - Shapes are incompatible
    /// - Parameters are out of bounds
    fn build(self) -> Result<Self::Layer>;
}

/// Helper trait for layers with weights
pub trait WeightsLayer {
    /// Get the weight matrix shape
    fn weight_shape(&self) -> (usize, usize);

    /// Get the total number of weights
    fn num_weights(&self) -> usize {
        let (rows, cols) = self.weight_shape();
        rows * cols
    }
}

/// Helper trait for layers with bias
pub trait BiasLayer {
    /// Get the bias vector size
    fn bias_size(&self) -> usize;

    /// Check if this layer has bias
    fn has_bias(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock layer for testing
    struct MockLayer {
        name: String,
        input: Shape,
        output: Shape,
    }

    impl Layer for MockLayer {
        fn forward(
            &self,
            _executor: &mut ANEExecutor,
            _input_idx: usize,
            _output_idx: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn input_shape(&self) -> &Shape {
            &self.input
        }

        fn output_shape(&self) -> &Shape {
            &self.output
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn num_parameters(&self) -> usize {
            0
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_layer_trait() {
        let layer = MockLayer {
            name: "test".to_string(),
            input: vec![1, 256],
            output: vec![1, 512],
        };

        assert_eq!(layer.name(), "test");
        assert_eq!(layer.input_shape(), &[1, 256]);
        assert_eq!(layer.output_shape(), &[1, 512]);
        assert_eq!(layer.num_parameters(), 0);
    }
}
