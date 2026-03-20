//! Persistent ANE Gradient Buffer
//!
//! Maintains gradient storage on ANE device throughout backward pass
//! to minimize CPU↔ANE transfers.

use crate::error::{Error, Result};
use crate::training::TransformerConfig;

/// Persistent gradient buffer on ANE device
///
/// Maintains IOSurface-backed gradient storage throughout the backward pass,
/// accumulating gradients from all layers directly in ANE memory.
/// Only transfers to CPU once at the end of the backward pass.
///
/// # Memory Flow
///
/// ```text
/// Forward Pass:  activations cached in CPU
/// Backward Pass:
///   - Each backward kernel accumulates into this buffer
///   - Gradients stay on ANE device
///   - Single transfer at end
/// Optimizer Step:  CPU optimizer updates parameters
/// ```
///
/// # Example
///
/// ```ignore
/// let config = TransformerConfig::tiny();
/// let mut buffer = ANEPersistentGradientBuffer::new(&config)?;
///
/// // Accumulate from each layer
/// buffer.accumulate_layer_gradients(layer_idx, &d_wq, &d_wk, &d_wv, &d_wo)?;
///
/// // Final transfer to CPU
/// let all_grads = buffer.transfer_to_cpu()?;
/// ```
pub struct ANEPersistentGradientBuffer {
    /// Gradient storage on ANE device (IOSurface-backed)
    /// In production: Vec<ANETensor> with IOSurface backing
    /// Current: CPU simulation with transfer tracking
    gradients: Vec<f32>,
    num_params: usize,
    transfer_count: usize,
}

impl ANEPersistentGradientBuffer {
    /// Create a new persistent gradient buffer
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration with parameter count
    ///
    /// # Returns
    ///
    /// A new buffer initialized with zeros
    ///
    /// # Errors
    ///
    /// Returns error if parameter count is zero
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        let num_params = config.param_count();
        if num_params == 0 {
            return Err(Error::InvalidParameter(
                "Parameter count must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            gradients: vec![0.0f32; num_params],
            num_params,
            transfer_count: 0,
        })
    }

    /// Accumulate gradients from a single layer
    ///
    /// Adds the provided gradients to the corresponding parameter ranges
    /// in the buffer. Used after each layer's backward pass.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0 = first transformer layer)
    /// * `d_wq` - Gradient for Q projection weights
    /// * `d_wk` - Gradient for K projection weights
    /// * `d_wv` - Gradient for V projection weights
    /// * `d_wo` - Gradient for output projection weights
    /// * `d_w1` - Gradient for FFN W1 weights
    /// * `d_w3` - Gradient for FFN W3 weights
    /// * `d_w2` - Gradient for FFN W2 weights
    /// * `d_rms_att` - Gradient for attention RMSNorm
    /// * `d_rms_ffn` - Gradient for FFN RMSNorm
    ///
    /// # Errors
    ///
    /// Returns error if layer_idx is invalid or gradient sizes don't match
    pub fn accumulate_layer_gradients(
        &mut self,
        config: &TransformerConfig,
        layer_idx: usize,
        d_wq: &[f32],
        d_wk: &[f32],
        d_wv: &[f32],
        d_wo: &[f32],
        d_w1: &[f32],
        d_w3: &[f32],
        d_w2: &[f32],
        d_rms_att: &[f32],
        d_rms_ffn: &[f32],
    ) -> Result<()> {
        if layer_idx >= config.n_layers {
            return Err(Error::InvalidParameter(format!(
                "Layer index {} exceeds number of layers {}",
                layer_idx, config.n_layers
            )));
        }

        let layout = super::transformer_model::build_layout(config);
        let layer_layout = layout.layer(layer_idx);

        // Accumulate attention gradients
        self.accumulate_into_range(layer_layout.wq(), d_wq)?;
        self.accumulate_into_range(layer_layout.wk(), d_wk)?;
        self.accumulate_into_range(layer_layout.wv(), d_wv)?;
        self.accumulate_into_range(layer_layout.wo(), d_wo)?;

        // Accumulate FFN gradients
        self.accumulate_into_range(layer_layout.w1(), d_w1)?;
        self.accumulate_into_range(layer_layout.w3(), d_w3)?;
        self.accumulate_into_range(layer_layout.w2(), d_w2)?;

        // Accumulate RMSNorm gradients
        self.accumulate_into_range(layer_layout.rms_att(), d_rms_att)?;
        self.accumulate_into_range(layer_layout.rms_ffn(), d_rms_ffn)?;

        Ok(())
    }

    /// Accumulate final layer gradients (final_norm, classifier, embedding)
    ///
    /// # Arguments
    ///
    /// * `d_final_norm` - Gradient for final RMSNorm layer
    /// * `d_classifier` - Gradient for classifier weights (optional)
    /// * `d_embedding` - Gradient for embedding layer
    pub fn accumulate_final_gradients(
        &mut self,
        config: &TransformerConfig,
        d_final_norm: &[f32],
        d_classifier: Option<&[f32]>,
        d_embedding: &[f32],
    ) -> Result<()> {
        let layout = super::transformer_model::build_layout(config);

        // Accumulate final norm gradients
        self.accumulate_into_range(layout.final_norm(), d_final_norm)?;

        // Accumulate classifier gradients if provided
        if let Some(d_class) = d_classifier {
            if !layout.classifier().is_empty() {
                self.accumulate_into_range(layout.classifier(), d_class)?;
            }
        }

        // Accumulate embedding gradients
        self.accumulate_into_range(layout.embedding(), d_embedding)?;

        Ok(())
    }

    /// Accumulate gradients into a specific parameter range
    fn accumulate_into_range(
        &mut self,
        range: &std::ops::Range<usize>,
        grads: &[f32],
    ) -> Result<()> {
        if range.end > self.num_params {
            return Err(Error::InvalidParameter(format!(
                "Range end {} exceeds parameter count {}",
                range.end, self.num_params
            )));
        }

        if grads.len() != (range.end - range.start) {
            return Err(Error::InvalidParameter(format!(
                "Gradient length {} doesn't match range length {}",
                grads.len(),
                range.end - range.start
            )));
        }

        // Accumulate element-wise
        for (i, &grad_val) in grads.iter().enumerate() {
            self.gradients[range.start + i] += grad_val;
        }

        Ok(())
    }

    /// Transfer all accumulated gradients to CPU
    ///
    /// This is called once at the end of the backward pass.
    /// In production with real ANE hardware, this would:
    /// 1. Lock the IOSurface buffers
    /// 2. Map the GPU memory
    /// 3. Copy to CPU memory
    ///
    /// # Returns
    ///
    /// A vector containing all accumulated gradients
    pub fn transfer_to_cpu(&mut self) -> Result<Vec<f32>> {
        self.transfer_count += 1;
        Ok(self.gradients.clone())
    }

    /// Get the number of transfers performed
    ///
    /// Useful for tracking optimization efficiency
    pub fn transfer_count(&self) -> usize {
        self.transfer_count
    }

    /// Reset gradients to zero (for next backward pass)
    pub fn reset(&mut self) {
        self.gradients.fill(0.0);
    }

    /// Get reference to gradients (without transfer)
    ///
    /// For in-place optimizer updates
    pub fn get(&self) -> &[f32] {
        &self.gradients
    }

    /// Get mutable reference to gradients
    ///
    /// For direct manipulation without transfer
    pub fn get_mut(&mut self) -> &mut [f32] {
        &mut self.gradients
    }

    /// Total number of parameters
    pub fn num_params(&self) -> usize {
        self.num_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_buffer_creation() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let buffer = ANEPersistentGradientBuffer::new(&config).unwrap();

        assert_eq!(buffer.num_params(), config.param_count());
        assert_eq!(buffer.transfer_count(), 0);
        assert!(buffer.get().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_accumulate_layer_gradients() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mut buffer = ANEPersistentGradientBuffer::new(&config).unwrap();

        let dim = config.dim;
        let hidden = config.hidden_dim;

        // Create dummy gradients
        let d_wq = vec![0.1f32; dim * dim];
        let d_wk = vec![0.2f32; dim * dim];
        let d_wv = vec![0.3f32; dim * dim];
        let d_wo = vec![0.4f32; dim * dim];
        let d_w1 = vec![0.5f32; dim * hidden];
        let d_w3 = vec![0.6f32; dim * hidden];
        let d_w2 = vec![0.7f32; hidden * dim];
        let d_rms_att = vec![0.8f32; dim];
        let d_rms_ffn = vec![0.9f32; dim];

        buffer
            .accumulate_layer_gradients(
                &config, 0, &d_wq, &d_wk, &d_wv, &d_wo, &d_w1, &d_w3, &d_w2, &d_rms_att, &d_rms_ffn,
            )
            .unwrap();

        // Verify gradients were accumulated
        let grads = buffer.get();
        assert!(grads.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_transfer_to_cpu() {
        let config = TransformerConfig::tiny();
        let mut buffer = ANEPersistentGradientBuffer::new(&config).unwrap();

        // Set some gradients
        buffer.get_mut()[0] = 1.0;
        buffer.get_mut()[1] = 2.0;

        // Transfer
        let grads = buffer.transfer_to_cpu().unwrap();

        assert_eq!(grads[0], 1.0);
        assert_eq!(grads[1], 2.0);
        assert_eq!(buffer.transfer_count(), 1);
    }
}
