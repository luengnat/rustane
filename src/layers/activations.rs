//! Activation function layers

use crate::layers::traits::{Layer, Shape};
use crate::{Error, Result};

/// ReLU activation function
///
/// Applies the rectified linear unit: f(x) = max(0, x)
#[derive(Clone, Debug)]
pub struct ReLU {
    name: String,
    input_shape: Shape,
    output_shape: Shape,
}

impl ReLU {
    /// Create a new ReLU activation
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::ReLU;
    /// let relu = ReLU::new();
    /// ```
    pub fn new() -> Self {
        ReLU {
            name: "relu".to_string(),
            input_shape: vec![1, 256],
            output_shape: vec![1, 256],
        }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
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
        0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// SiLU (Swish) activation function
///
/// Applies the SiLU function: f(x) = x * sigmoid(x)
#[derive(Clone, Debug)]
pub struct SiLU {
    name: String,
    input_shape: Shape,
    output_shape: Shape,
}

impl SiLU {
    /// Create a new SiLU activation
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::SiLU;
    /// let silu = SiLU::new();
    /// ```
    pub fn new() -> Self {
        SiLU {
            name: "silu".to_string(),
            input_shape: vec![1, 256],
            output_shape: vec![1, 256],
        }
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for SiLU {
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
        0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// GELU activation function
///
/// Applies the Gaussian Error Linear Unit
#[derive(Clone, Debug)]
pub struct GELU {
    name: String,
    input_shape: Shape,
    output_shape: Shape,
}

impl GELU {
    /// Create a new GELU activation
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::GELU;
    /// let gelu = GELU::new();
    /// ```
    pub fn new() -> Self {
        GELU {
            name: "gelu".to_string(),
            input_shape: vec![1, 256],
            output_shape: vec![1, 256],
        }
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GELU {
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
        0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        assert_eq!(relu.name(), "relu");
        assert_eq!(relu.num_parameters(), 0);
    }

    #[test]
    fn test_silu() {
        let silu = SiLU::new();
        assert_eq!(silu.name(), "silu");
        assert_eq!(silu.num_parameters(), 0);
    }

    #[test]
    fn test_gelu() {
        let gelu = GELU::new();
        assert_eq!(gelu.name(), "gelu");
        assert_eq!(gelu.num_parameters(), 0);
    }
}
