//! Convolutional (Conv2d) layer

use crate::layers::traits::{Layer, Shape, WeightsLayer};
use crate::{Error, Result};

/// Convolutional layer configuration
#[derive(Clone, Debug)]
pub struct Conv2d {
    name: String,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    #[allow(dead_code)]
    stride: (usize, usize),
    #[allow(dead_code)]
    padding: (usize, usize, usize, usize),
    has_bias: bool,
    #[allow(dead_code)]
    weight_data: Option<Vec<f32>>,
    #[allow(dead_code)]
    bias_data: Option<Vec<f32>>,
    input_shape: Shape,
    output_shape: Shape,
}

impl Conv2d {
    /// Create a new Conv2d layer builder
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - (height, width) of the convolution kernel
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Conv2d;
    /// let layer = Conv2d::new(3, 64, (7, 7))
    ///     .stride((2, 2))
    ///     .build()?;
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Conv2dBuilder {
        Conv2dBuilder {
            name: format!("conv2d_{}x{}", kernel_size.0, kernel_size.1),
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0, 0, 0), // no padding
            has_bias: false,
            weight_init: None,
            bias_init: None,
        }
    }
}

impl Layer for Conv2d {
    fn forward(
        &self,
        _executor: &mut crate::wrapper::ANEExecutor,
        _input_idx: usize,
        _output_idx: usize,
    ) -> Result<()> {
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
        let weights =
            self.in_channels * self.out_channels * self.kernel_size.0 * self.kernel_size.1;
        let bias = if self.has_bias { self.out_channels } else { 0 };
        weights + bias
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl WeightsLayer for Conv2d {
    fn weight_shape(&self) -> (usize, usize) {
        let total = self.in_channels * self.out_channels * self.kernel_size.0 * self.kernel_size.1;
        (total, 1) // Flattened to 1D
    }
}

/// Builder for Conv2d layers
pub struct Conv2dBuilder {
    name: String,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
    has_bias: bool,
    #[allow(dead_code)]
    weight_init: Option<Vec<f32>>,
    #[allow(dead_code)]
    bias_init: Option<Vec<f32>>,
}

impl Conv2dBuilder {
    /// Set the layer name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set stride
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Conv2d;
    /// let layer = Conv2d::new(3, 64, (7, 7))
    ///     .stride((2, 2))
    ///     .build();
    /// ```
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding (top, bottom, left, right)
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::Conv2d;
    /// let layer = Conv2d::new(3, 64, (7, 7))
    ///     .padding((1, 1, 1, 1))
    ///     .build();
    /// ```
    pub fn padding(mut self, padding: (usize, usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Enable or disable bias
    pub fn with_bias(mut self, has_bias: bool) -> Self {
        self.has_bias = has_bias;
        self
    }

    /// Build the Conv2d layer
    pub fn build(self) -> Result<Conv2d> {
        // Validate parameters
        if self.in_channels == 0 || self.out_channels == 0 {
            return Err(Error::InvalidParameter(
                "Channels must be greater than 0".to_string(),
            ));
        }

        if self.kernel_size.0 == 0 || self.kernel_size.1 == 0 {
            return Err(Error::InvalidParameter(
                "Kernel size must be greater than 0".to_string(),
            ));
        }

        // Calculate output size (valid padding)
        // output = (input - kernel + 2*padding) / stride + 1
        let input_size = 224; // Default input size
        let _output_h =
            (input_size - self.kernel_size.0 + self.padding.0 + self.padding.1) / self.stride.0 + 1;
        let _output_w =
            (input_size - self.kernel_size.1 + self.padding.2 + self.padding.3) / self.stride.1 + 1;

        Ok(Conv2d {
            name: self.name,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            has_bias: self.has_bias,
            weight_data: None,
            bias_data: None,
            input_shape: vec![1, self.in_channels, 224, 224],
            output_shape: vec![1, self.out_channels, 109, 109],
        })
    }
}

impl Default for Conv2dBuilder {
    fn default() -> Self {
        Conv2d::new(1, 1, (1, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_builder() {
        let layer = Conv2d::new(3, 64, (7, 7)).stride((2, 2)).build().unwrap();

        assert_eq!(layer.in_channels, 3);
        assert_eq!(layer.out_channels, 64);
        assert_eq!(layer.kernel_size, (7, 7));
        assert_eq!(layer.stride, (2, 2));
        assert!(!layer.has_bias);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let layer = Conv2d::new(3, 64, (7, 7))
            .padding((1, 1, 1, 1))
            .build()
            .unwrap();

        assert_eq!(layer.padding, (1, 1, 1, 1));
    }

    #[test]
    fn test_conv2d_parameter_count() {
        let layer = Conv2d::new(3, 64, (7, 7)).with_bias(true).build().unwrap();

        let expected = 3 * 64 * 7 * 7 + 64; // weights + bias
        assert_eq!(layer.num_parameters(), expected);
    }

    #[test]
    fn test_conv2d_invalid_channels() {
        let result = Conv2d::new(0, 64, (7, 7)).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        let result = Conv2d::new(3, 0, (7, 7)).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_conv2d_invalid_kernel() {
        let result = Conv2d::new(3, 64, (0, 7)).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        let result = Conv2d::new(3, 64, (7, 0)).build();
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }
}
