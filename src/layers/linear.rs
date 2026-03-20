//! Linear (fully connected) layer

use crate::layers::traits::{Layer, Shape, WeightsLayer};
use crate::{Error, Result};

/// Linear (fully connected) layer configuration
///
/// # Example
///
/// ```
/// # use rustane::layers::Linear;
/// let layer = Linear::new(256, 512)
///     .with_bias(false)
///     .build()?;
/// ```
#[derive(Clone, Debug)]
pub struct Linear {
    name: String,
    input_features: usize,
    output_features: usize,
    has_bias: bool,
    #[allow(dead_code)]
    weight_data: Option<Vec<f32>>,
    #[allow(dead_code)]
    bias_data: Option<Vec<f32>>,
    input_shape: Shape,
    output_shape: Shape,
}

impl Linear {
    /// Create a new Linear layer builder
    ///
    /// # Arguments
    ///
    /// * `input_features` - Number of input features
    /// * `output_features` - Number of output features
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Linear;
    /// let builder = Linear::new(256, 512);
    /// ```
    pub fn new(input_features: usize, output_features: usize) -> LinearBuilder {
        LinearBuilder {
            name: format!("linear_{}", input_features),
            input_features,
            output_features,
            has_bias: false,
            weight_init: None,
            bias_init: None,
        }
    }
}

impl Layer for Linear {
    fn forward(
        &self,
        _executor: &mut crate::wrapper::ANEExecutor,
        _input_idx: usize,
        _output_idx: usize,
    ) -> Result<()> {
        // Note: In the current architecture, layers don't directly execute
        // They're used during model compilation. This is a placeholder
        // for when we have full model execution support.
        Err(Error::ExecutionFailed(
            "Layer forward pass not yet implemented. Use Model compilation instead.".to_string(),
        ))
    }

    fn input_shape(&self) -> &Shape {
        &self.input_shape
    }

    fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_parameters(&self) -> usize {
        let weights = self.input_features * self.output_features;
        let bias = if self.has_bias {
            self.output_features
        } else {
            0
        };
        weights + bias
    }
}

impl WeightsLayer for Linear {
    fn weight_shape(&self) -> (usize, usize) {
        (self.input_features, self.output_features)
    }
}

/// Builder for Linear layers
pub struct LinearBuilder {
    name: String,
    input_features: usize,
    output_features: usize,
    has_bias: bool,
    weight_init: Option<Vec<f32>>,
    bias_init: Option<Vec<f32>>,
}

impl LinearBuilder {
    /// Set the layer name
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Linear;
    /// let layer = Linear::new(256, 512)
    ///     .with_name("fc1")
    ///     .build();
    /// ```
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Enable or disable bias
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Linear;
    /// let layer = Linear::new(256, 512)
    ///     .with_bias(true)
    ///     .build();
    /// ```
    pub fn with_bias(mut self, has_bias: bool) -> Self {
        self.has_bias = has_bias;
        self
    }

    /// Set custom weight initialization
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Linear;
    /// let weights = vec![1.0; 256 * 512];
    /// let layer = Linear::new(256, 512)
    ///     .with_weights(weights)
    ///     .build();
    /// ```
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weight_init = Some(weights);
        self
    }

    /// Set custom bias initialization
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Linear;
    /// let bias = vec![0.0; 512];
    /// let layer = Linear::new(256, 512)
    ///     .with_bias(true)
    ///     .with_bias_values(bias)
    ///     .build();
    /// ```
    pub fn with_bias_values(mut self, bias: Vec<f32>) -> Self {
        self.bias_init = Some(bias);
        self.has_bias = true;
        self
    }

    /// Build the Linear layer
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Weight dimensions don't match
    /// - Bias dimensions don't match
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Linear;
    /// let layer = Linear::new(256, 512)
    ///     .with_bias(false)
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<Linear> {
        // Validate dimensions
        if self.input_features == 0 {
            return Err(Error::InvalidParameter(
                "input_features must be greater than 0".to_string(),
            ));
        }
        if self.output_features == 0 {
            return Err(Error::InvalidParameter(
                "output_features must be greater than 0".to_string(),
            ));
        }

        // Initialize weights if not provided
        let weight_data = if let Some(weights) = self.weight_init {
            if weights.len() != self.input_features * self.output_features {
                return Err(Error::InvalidParameter(format!(
                    "Weight dimensions: expected {}, got {}",
                    self.input_features * self.output_features,
                    weights.len()
                )));
            }
            weights
        } else {
            // Default: Xavier initialization
            let std_f32 = (2.0 / (self.input_features + self.output_features) as f32).sqrt();
            let mut weights = vec![0.0f32; self.input_features * self.output_features];
            for w in weights.iter_mut() {
                *w = std_f32 * (rand::random::<f32>() * 2.0 - 1.0);
            }
            weights
        };

        // Initialize bias if needed
        let bias_data = if self.has_bias {
            if let Some(bias) = self.bias_init {
                if bias.len() != self.output_features {
                    return Err(Error::InvalidParameter(format!(
                        "Bias dimensions: expected {}, got {}",
                        self.output_features,
                        bias.len()
                    )));
                }
                Some(bias)
            } else {
                // Default: zeros
                Some(vec![0.0f32; self.output_features])
            }
        } else {
            None
        };

        Ok(Linear {
            name: self.name,
            input_features: self.input_features,
            output_features: self.output_features,
            has_bias: self.has_bias,
            weight_data: Some(weight_data),
            bias_data,
            input_shape: vec![1, self.input_features],
            output_shape: vec![1, self.output_features],
        })
    }
}

impl Default for LinearBuilder {
    fn default() -> Self {
        Linear::new(1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_builder() {
        let layer = Linear::new(256, 512).with_bias(false).build().unwrap();

        assert_eq!(layer.input_features, 256);
        assert_eq!(layer.output_features, 512);
        assert!(!layer.has_bias);
        assert_eq!(layer.num_parameters(), 256 * 512);
    }

    #[test]
    fn test_linear_with_bias() {
        let layer = Linear::new(256, 512).with_bias(true).build().unwrap();

        assert!(layer.has_bias);
        assert_eq!(layer.num_parameters(), 256 * 512 + 512);
    }

    #[test]
    fn test_linear_custom_weights() {
        let weights = vec![1.0f32; 256 * 512];
        let layer = Linear::new(256, 512)
            .with_weights(weights.clone())
            .build()
            .unwrap();

        assert!(layer.weight_data.is_some());
        assert_eq!(layer.weight_data.unwrap().len(), 256 * 512);
    }

    #[test]
    fn test_linear_invalid_dimensions() {
        let result = Linear::new(0, 512).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        let result = Linear::new(256, 0).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_linear_weight_shape() {
        let layer = Linear::new(256, 512).build().unwrap();

        assert_eq!(layer.weight_shape(), (256, 512));
    }
}
