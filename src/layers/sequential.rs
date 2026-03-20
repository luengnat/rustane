//! Sequential model container

use crate::layers::{Layer, LayerInfo, Model, ModelSummary, Shape};
use crate::wrapper::ANETensor;
use crate::{Error, Result};
use std::sync::Arc;

/// Sequential model container
///
/// Executes layers in order, providing a simple way to build
/// neural networks with a linear architecture.
///
/// # Example
///
/// ```no_run
/// # use rustane::layers::{Sequential, Conv2d, ReLU, Linear};
/// # fn main() -> rustane::Result<()> {
/// let model = Sequential::new("my_model")
///     .add(Box::new(Conv2d::new(3, 64, (7, 7)).stride((2, 2)).build()?))
///     .add(Box::new(ReLU::new()))
///     .add(Box::new(Linear::new(64, 10).build()?));
/// # Ok(())
/// # }
/// ```
pub struct Sequential {
    name: String,
    layers: Vec<Box<dyn Layer>>,
    frozen: Vec<bool>,
    input_shape: Option<Shape>,
    output_shape: Option<Shape>,
}

impl Sequential {
    /// Create a new Sequential builder
    ///
    /// # Arguments
    ///
    /// * `name` - Model name
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::layers::Sequential;
    /// let builder = Sequential::new("my_model");
    /// # let _ = builder;
    /// ```
    pub fn new(name: impl Into<String>) -> SequentialBuilder {
        SequentialBuilder {
            name: name.into(),
            layers: vec![],
            frozen: vec![],
            input_shape: None,
        }
    }

    /// Get the number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the model has no layers
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get a layer by index
    pub fn get(&self, idx: usize) -> Option<&dyn Layer> {
        self.layers.get(idx).map(|layer| layer.as_ref())
    }

    /// Freeze a layer (prevent parameter updates)
    ///
    /// # Arguments
    ///
    /// * `idx` - Layer index
    ///
    /// # Errors
    ///
    /// Returns an error if index is out of bounds
    pub fn freeze_layer(&mut self, idx: usize) -> Result<()> {
        if idx >= self.frozen.len() {
            return Err(Error::InvalidParameter(format!(
                "Layer index {} out of bounds (len={})",
                idx,
                self.frozen.len()
            )));
        }
        self.frozen[idx] = true;
        Ok(())
    }

    /// Unfreeze a layer (allow parameter updates)
    ///
    /// # Arguments
    ///
    /// * `idx` - Layer index
    ///
    /// # Errors
    ///
    /// Returns an error if index is out of bounds
    pub fn unfreeze_layer(&mut self, idx: usize) -> Result<()> {
        if idx >= self.frozen.len() {
            return Err(Error::InvalidParameter(format!(
                "Layer index {} out of bounds (len={})",
                idx,
                self.frozen.len()
            )));
        }
        self.frozen[idx] = false;
        Ok(())
    }

    /// Check if a layer is frozen
    ///
    /// # Arguments
    ///
    /// * `idx` - Layer index
    ///
    /// # Errors
    ///
    /// Returns an error if index is out of bounds
    pub fn is_frozen(&self, idx: usize) -> Result<bool> {
        if idx >= self.frozen.len() {
            return Err(Error::InvalidParameter(format!(
                "Layer index {} out of bounds (len={})",
                idx,
                self.frozen.len()
            )));
        }
        Ok(self.frozen[idx])
    }

    /// Calculate total number of parameters
    fn count_parameters(&self) -> (usize, usize) {
        let mut total = 0;
        let mut trainable = 0;

        for (layer, frozen) in self.layers.iter().zip(self.frozen.iter()) {
            let params = layer.num_parameters();
            total += params;
            if !frozen {
                trainable += params;
            }
        }

        (total, trainable)
    }
}

/// Shared layer wrapper for parameter sharing
///
/// Wraps a layer in an Arc to enable cheap cloning and parameter sharing
/// across multiple models or layers.
///
/// # Example
///
/// ```no_run
/// # use rustane::layers::{SharedLayer, ReLU, Sequential};
/// # fn main() -> rustane::Result<()> {
/// let shared_layer = SharedLayer::new(ReLU::new());
///
/// let model1 = Sequential::new("model1")
///     .add_shared(shared_layer.clone())
///     .build();
///
/// let model2 = Sequential::new("model2")
///     .add_shared(shared_layer)
///     .build();
///
/// // Both models share the same underlying layer
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct SharedLayer {
    inner: Arc<dyn Layer>,
}

impl SharedLayer {
    /// Create a new shared layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer to wrap
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::layers::{SharedLayer, ReLU};
    /// let shared = SharedLayer::new(ReLU::new());
    /// ```
    pub fn new<L: Layer + 'static>(layer: L) -> Self {
        Self {
            inner: Arc::new(layer),
        }
    }

    /// Get a reference to the inner layer
    pub fn inner(&self) -> &dyn Layer {
        self.inner.as_ref()
    }
}

impl Layer for SharedLayer {
    fn forward(
        &self,
        executor: &mut crate::wrapper::ANEExecutor,
        input_idx: usize,
        output_idx: usize,
    ) -> Result<()> {
        self.inner.forward(executor, input_idx, output_idx)
    }

    fn input_shape(&self) -> &Shape {
        self.inner.input_shape()
    }

    fn output_shape(&self) -> &Shape {
        self.inner.output_shape()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

impl Model for Sequential {
    fn forward(&mut self, _input: &ANETensor) -> Result<ANETensor> {
        // Note: This is a placeholder implementation
        // In a full implementation, we would execute each layer sequentially
        // through the ANE executor. For now, we return an error to indicate
        // that this is not yet implemented.
        Err(Error::ExecutionFailed(
            "Sequential forward pass not yet implemented. Use Model compilation instead."
                .to_string(),
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_shape(&self) -> Option<&Shape> {
        self.input_shape.as_ref()
    }

    fn output_shape(&self) -> Option<&Shape> {
        self.output_shape.as_ref()
    }

    fn num_parameters(&self) -> usize {
        self.count_parameters().0
    }

    fn num_trainable_parameters(&self) -> usize {
        self.count_parameters().1
    }

    fn summary(&self) -> String {
        let (total_params, trainable_params) = self.count_parameters();
        let frozen_params = total_params - trainable_params;

        let layers: Vec<LayerInfo> = self
            .layers
            .iter()
            .zip(self.frozen.iter())
            .enumerate()
            .map(|(i, (layer, frozen))| LayerInfo {
                name: format!("layer_{}", i),
                type_name: layer.name().to_string(),
                input_shape: layer.input_shape().clone(),
                output_shape: layer.output_shape().clone(),
                num_params: layer.num_parameters(),
                frozen: *frozen,
            })
            .collect();

        let summary = ModelSummary {
            name: self.name.clone(),
            total_params,
            trainable_params,
            frozen_params,
            layers,
        };

        format!("{}", summary)
    }
}

/// Builder for Sequential models
pub struct SequentialBuilder {
    name: String,
    layers: Vec<Box<dyn Layer>>,
    frozen: Vec<bool>,
    input_shape: Option<Shape>,
}

impl SequentialBuilder {
    /// Add a layer to the model
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer to add
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::layers::{Sequential, ReLU};
    /// let builder = Sequential::new("test").add(Box::new(ReLU::new()));
    /// # let _ = builder;
    /// ```
    pub fn add(mut self, layer: Box<dyn Layer>) -> Self {
        // Validate shape compatibility
        if let Some(ref input_shape) = self.input_shape {
            let layer_input = layer.input_shape();
            // Simple validation: check if shapes match
            // In a full implementation, we would do more sophisticated checking
            if layer_input != input_shape && !layer_input.is_empty() {
                // Shape mismatch - this is OK for now, just warn
                // In production, we might want to return an error
            }
        }

        // Update input/output shapes
        if self.input_shape.is_none() {
            self.input_shape = Some(layer.input_shape().clone());
        }

        self.layers.push(layer);
        self.frozen.push(false);
        self
    }

    /// Add a shared layer to the model
    ///
    /// # Arguments
    ///
    /// * `layer` - Shared layer to add
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::layers::{Sequential, SharedLayer, ReLU};
    /// let shared = SharedLayer::new(ReLU::new());
    /// let builder = Sequential::new("test").add_shared(shared.clone());
    /// # let _ = builder;
    /// ```
    pub fn add_shared(mut self, layer: SharedLayer) -> Self {
        // Validate shape compatibility
        if let Some(ref input_shape) = self.input_shape {
            let layer_input = layer.input_shape();
            if layer_input != input_shape && !layer_input.is_empty() {
                // Shape mismatch
            }
        }

        // Update input/output shapes
        if self.input_shape.is_none() {
            self.input_shape = Some(layer.input_shape().clone());
        }

        self.layers.push(Box::new(layer));
        self.frozen.push(false);
        self
    }

    /// Build the Sequential model
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::layers::{Sequential, ReLU};
    /// let model = Sequential::new("test")
    ///     .add(Box::new(ReLU::new()))
    ///     .build();
    /// # let _ = model;
    /// ```
    pub fn build(self) -> Sequential {
        let output_shape = self.layers.last().map(|layer| layer.output_shape().clone());

        Sequential {
            name: self.name,
            layers: self.layers,
            frozen: self.frozen,
            input_shape: self.input_shape,
            output_shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Linear, ReLU};

    #[test]
    fn test_sequential_empty() {
        let model = Sequential::new("empty").build();
        assert_eq!(model.name(), "empty");
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
        assert_eq!(model.num_parameters(), 0);
        assert_eq!(model.num_trainable_parameters(), 0);
    }

    #[test]
    fn test_sequential_single_layer() {
        let model = Sequential::new("single").add(Box::new(ReLU::new())).build();

        assert_eq!(model.name(), "single");
        assert!(!model.is_empty());
        assert_eq!(model.len(), 1);
        assert_eq!(model.num_parameters(), 0);
    }

    #[test]
    fn test_sequential_multi_layer() {
        let model = Sequential::new("multi")
            .add(Box::new(ReLU::new()))
            .add(Box::new(ReLU::new()))
            .build();

        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_sequential_freeze_unfreeze() {
        let mut model = Sequential::new("test")
            .add(Box::new(ReLU::new()))
            .add(Box::new(ReLU::new()))
            .build();

        // Initially not frozen
        assert!(!model.is_frozen(0).unwrap());
        assert!(!model.is_frozen(1).unwrap());

        // Freeze layer 0
        model.freeze_layer(0).unwrap();
        assert!(model.is_frozen(0).unwrap());
        assert!(!model.is_frozen(1).unwrap());

        // Unfreeze layer 0
        model.unfreeze_layer(0).unwrap();
        assert!(!model.is_frozen(0).unwrap());
    }

    #[test]
    fn test_sequential_freeze_out_of_bounds() {
        let mut model = Sequential::new("test").add(Box::new(ReLU::new())).build();

        assert!(model.freeze_layer(1).is_err());
        assert!(model.is_frozen(1).is_err());
        assert!(model.unfreeze_layer(1).is_err());
    }

    #[test]
    fn test_sequential_get() {
        let model = Sequential::new("test")
            .add(Box::new(ReLU::new()))
            .add(Box::new(ReLU::new()))
            .build();

        assert!(model.get(0).is_some());
        assert!(model.get(1).is_some());
        assert!(model.get(2).is_none());
    }

    #[test]
    fn test_sequential_summary() {
        let model = Sequential::new("test")
            .add(Box::new(ReLU::new()))
            .add(Box::new(ReLU::new()))
            .build();

        let summary = model.summary();
        assert!(summary.contains("test"));
        assert!(summary.contains("Layers:"));
    }

    #[test]
    fn test_sequential_parameter_count() {
        let model = Sequential::new("test")
            .add(Box::new(Linear::new(10, 20).build().unwrap()))
            .build();

        assert_eq!(model.num_parameters(), 10 * 20);
        assert_eq!(model.num_trainable_parameters(), 10 * 20);

        // Freeze and check
        let mut model = model;
        model.freeze_layer(0).unwrap();
        assert_eq!(model.num_parameters(), 10 * 20);
        assert_eq!(model.num_trainable_parameters(), 0);
    }

    #[test]
    fn test_sequential_builder_add() {
        let builder = Sequential::new("test")
            .add(Box::new(ReLU::new()))
            .add(Box::new(ReLU::new()));

        // Builder should not be consumed yet
        let model = builder.build();
        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_sequential_shapes() {
        let model = Sequential::new("test").add(Box::new(ReLU::new())).build();

        // ReLU has input_shape [1, 256]
        assert!(model.input_shape().is_some());
        assert!(model.output_shape().is_some());
    }

    #[test]
    fn test_shared_layer_creation() {
        let shared = SharedLayer::new(ReLU::new());
        assert_eq!(shared.name(), "relu");
        assert_eq!(shared.num_parameters(), 0);
    }

    #[test]
    fn test_shared_layer_clone() {
        let shared1 = SharedLayer::new(ReLU::new());
        let shared2 = shared1.clone();

        // Both should point to the same underlying layer
        assert_eq!(shared1.name(), shared2.name());
    }

    #[test]
    fn test_sequential_add_shared() {
        let shared = SharedLayer::new(ReLU::new());

        let model = Sequential::new("test").add_shared(shared.clone()).build();

        assert_eq!(model.len(), 1);
        assert_eq!(model.get(0).unwrap().name(), "relu");
    }

    #[test]
    fn test_shared_layer_in_multiple_models() {
        let shared = SharedLayer::new(ReLU::new());

        let model1 = Sequential::new("model1").add_shared(shared.clone()).build();

        let model2 = Sequential::new("model2").add_shared(shared).build();

        // Both models should have the same layer
        assert_eq!(model1.get(0).unwrap().name(), model2.get(0).unwrap().name());
    }
}
