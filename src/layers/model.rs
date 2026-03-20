//! Model trait and inspection utilities

use crate::layers::Shape;
use crate::wrapper::ANETensor;
use crate::Result;
use std::fmt;

/// Information about a single layer in a model
#[derive(Clone, Debug)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
    /// Layer type (e.g., "Conv2d", "Linear")
    pub type_name: String,
    /// Input shape
    pub input_shape: Shape,
    /// Output shape
    pub output_shape: Shape,
    /// Number of parameters
    pub num_params: usize,
    /// Whether layer is frozen (no gradient updates)
    pub frozen: bool,
}

impl fmt::Display for LayerInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}): input={:?}, output={:?}, params={}, frozen={}",
            self.name,
            self.type_name,
            self.input_shape,
            self.output_shape,
            self.num_params,
            self.frozen
        )
    }
}

/// Summary of model architecture and parameters
#[derive(Clone, Debug)]
pub struct ModelSummary {
    /// Model name
    pub name: String,
    /// Total number of parameters
    pub total_params: usize,
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Number of frozen parameters
    pub frozen_params: usize,
    /// Information about each layer
    pub layers: Vec<LayerInfo>,
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "Model: {}", self.name)?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f)?;
        writeln!(f, "Total parameters: {}", self.total_params)?;
        writeln!(f, "Trainable parameters: {}", self.trainable_params)?;
        writeln!(f, "Frozen parameters: {}", self.frozen_params)?;
        writeln!(f)?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "Layers:")?;
        writeln!(f, "{}", "=".repeat(60))?;

        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "{}: {}", i + 1, layer)?;
        }

        writeln!(f)?;
        writeln!(f, "{}", "=".repeat(60))?;

        Ok(())
    }
}

/// Core model trait
///
/// All neural network models implement this trait, providing
/// a common interface for forward passes, inspection, and parameter management.
///
/// # Example
///
/// ```no_run
/// # use rustane::layers::Model;
/// # use rustane::layers::Shape;
/// # use rustane::Result;
/// # use rustane::wrapper::{ANEExecutor, ANETensor};
/// struct MyModel {
///     name: String,
///     input_shape: Shape,
///     output_shape: Shape,
/// }
///
/// impl Model for MyModel {
///     fn forward(&mut self, _input: &ANETensor) -> Result<ANETensor> {
///         // Execute forward pass
///         Ok(_input.clone())
///     }
///
///     fn name(&self) -> &str {
///         &self.name
///     }
///
///     fn input_shape(&self) -> Option<&Shape> {
///         Some(&self.input_shape)
///     }
///
///     fn output_shape(&self) -> Option<&Shape> {
///         Some(&self.output_shape)
///     }
///
///     fn num_parameters(&self) -> usize {
///         0
///     }
///
///     fn num_trainable_parameters(&self) -> usize {
///         0
///     }
///
///     fn summary(&self) -> String {
///         self.name().to_string()
///     }
/// }
/// ```
pub trait Model {
    /// Execute the forward pass
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - ANE execution fails
    /// - Tensor shapes don't match
    /// - Model is not properly configured
    fn forward(&mut self, input: &ANETensor) -> Result<ANETensor>;

    /// Get the model name
    fn name(&self) -> &str;

    /// Get the expected input shape
    ///
    /// Returns None if shape is dynamic or unknown
    fn input_shape(&self) -> Option<&Shape>;

    /// Get the expected output shape
    ///
    /// Returns None if shape is dynamic or unknown
    fn output_shape(&self) -> Option<&Shape>;

    /// Get the total number of parameters
    fn num_parameters(&self) -> usize;

    /// Get the number of trainable parameters
    ///
    /// This excludes frozen parameters
    fn num_trainable_parameters(&self) -> usize;

    /// Get a summary of the model architecture
    ///
    /// Returns a formatted string with layer information
    /// and parameter counts
    fn summary(&self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_info_display() {
        let info = LayerInfo {
            name: "conv1".to_string(),
            type_name: "Conv2d".to_string(),
            input_shape: vec![1, 3, 224, 224],
            output_shape: vec![1, 64, 112, 112],
            num_params: 9408,
            frozen: false,
        };

        let display = format!("{}", info);
        assert!(display.contains("conv1"));
        assert!(display.contains("Conv2d"));
        assert!(display.contains("9408"));
    }

    #[test]
    fn test_model_summary_display() {
        let summary = ModelSummary {
            name: "test_model".to_string(),
            total_params: 1000,
            trainable_params: 800,
            frozen_params: 200,
            layers: vec![],
        };

        let display = format!("{}", summary);
        assert!(display.contains("test_model"));
        assert!(display.contains("1000"));
        assert!(display.contains("800"));
        assert!(display.contains("200"));
    }

    #[test]
    fn test_model_summary_empty_layers() {
        let summary = ModelSummary {
            name: "empty_model".to_string(),
            total_params: 0,
            trainable_params: 0,
            frozen_params: 0,
            layers: vec![],
        };

        assert_eq!(summary.total_params, 0);
        assert_eq!(summary.trainable_params, 0);
        assert_eq!(summary.frozen_params, 0);
        assert!(summary.layers.is_empty());
    }
}
