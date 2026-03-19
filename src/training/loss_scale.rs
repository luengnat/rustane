//! Loss scaling for FP16 training
//!
//! Prevents gradient underflow in reduced-precision training by scaling losses
//! and unscaling gradients appropriately.

/// Loss scaler for FP16 training
///
/// Implements dynamic loss scaling to prevent gradient underflow.
/// Scales loss up before backprop, then unscales gradients afterward.
///
/// # Strategy
///
/// 1. Start with initial scale (e.g., 256.0)
/// 2. At each training step, scale loss by this factor
/// 3. Compute backward pass (gradients scaled up by loss scale)
/// 4. Check if gradients are valid (no inf/nan)
/// 5. If valid: optionally increase scale periodically
/// 6. If invalid: decrease scale by backoff factor
///
/// # Example
///
/// ```
/// # use rustane::training::LossScaler;
/// let mut scaler = LossScaler::new(256.0);
/// let loss = 0.5;
///
/// // Scale loss before backprop
/// let scaled_loss = scaler.scale_loss(loss);
/// assert_eq!(scaled_loss, loss * 256.0);
///
/// // After backprop, unscale gradients
/// let mut grads = vec![1.0f32; 100];
/// scaler.unscale_grads(&mut grads);
/// ```
pub struct LossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    steps_since_growth: u32,
}

impl LossScaler {
    /// Create a new loss scaler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `initial_scale` - Starting loss scale (e.g., 256.0 for FP16)
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::LossScaler;
    /// let scaler = LossScaler::new(256.0);
    /// assert_eq!(scaler.current_scale(), 256.0);
    /// ```
    pub fn new(initial_scale: f32) -> Self {
        LossScaler {
            scale: initial_scale,
            growth_factor: 2.0,    // Double scale on success
            backoff_factor: 0.5,   // Halve scale on overflow
            growth_interval: 2000, // Try growth every 2000 steps
            steps_since_growth: 0,
        }
    }

    /// Create a loss scaler for typical transformer models
    ///
    /// Scales initial loss based on model depth to account for gradient
    /// scaling across layers.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::LossScaler;
    /// let scaler = LossScaler::for_transformer(12);
    /// assert!(scaler.current_scale() >= 256.0);
    /// ```
    pub fn for_transformer(num_layers: usize) -> Self {
        // Scale = 256 * sqrt(num_layers) to handle deeper networks
        let scale = 256.0 * (num_layers as f32).sqrt();
        Self::new(scale)
    }

    /// Scale a loss value before backpropagation
    ///
    /// This prevents gradient underflow by magnifying small gradients.
    ///
    /// # Arguments
    ///
    /// * `loss` - Original loss value
    ///
    /// # Returns
    ///
    /// Scaled loss = loss * current_scale
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::LossScaler;
    /// let scaler = LossScaler::new(256.0);
    /// let scaled = scaler.scale_loss(0.5);
    /// assert_eq!(scaled, 0.5 * 256.0);
    /// ```
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale gradients after backpropagation
    ///
    /// Divides all gradients by the current loss scale.
    /// Should be called after backward() but before optimizer step.
    ///
    /// # Arguments
    ///
    /// * `grads` - Mutable gradient array
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::LossScaler;
    /// let scaler = LossScaler::new(256.0);
    /// let mut grads = vec![256.0f32; 100];
    /// scaler.unscale_grads(&mut grads);
    /// // grads now contain ~1.0
    /// assert!((grads[0] - 1.0).abs() < 0.001);
    /// ```
    pub fn unscale_grads(&self, grads: &mut [f32]) {
        let inv_scale = 1.0 / self.scale;
        for grad in grads {
            *grad *= inv_scale;
        }
    }

    /// Update scale based on gradient health
    ///
    /// Call this after computing gradients to check for inf/nan and
    /// adjust scale accordingly.
    ///
    /// Returns true if gradients are valid (no overflow).
    /// Returns false if any gradient is inf/nan (overflow detected).
    ///
    /// On success:
    /// - Increments step counter
    /// - Optionally grows scale every growth_interval steps
    ///
    /// On failure:
    /// - Applies backoff: scale *= backoff_factor
    /// - Resets step counter
    ///
    /// # Arguments
    ///
    /// * `grads` - Computed gradient array
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::LossScaler;
    /// let mut scaler = LossScaler::new(256.0);
    /// let valid_grads = vec![1.0f32; 100];
    /// assert!(scaler.update(&valid_grads));
    ///
    /// let overflow_grads = vec![f32::INFINITY; 100];
    /// assert!(!scaler.update(&overflow_grads));
    /// ```
    pub fn update(&mut self, grads: &[f32]) -> bool {
        // Check for overflow
        let has_overflow = grads.iter().any(|g| !g.is_finite());

        if has_overflow {
            // Reduce scale
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;
            false
        } else {
            // Increment step counter, try growth
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
            true
        }
    }

    /// Get the current loss scale
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::LossScaler;
    /// let scaler = LossScaler::new(256.0);
    /// assert_eq!(scaler.current_scale(), 256.0);
    /// ```
    pub fn current_scale(&self) -> f32 {
        self.scale
    }

    /// Set custom growth parameters
    ///
    /// # Arguments
    ///
    /// * `growth_factor` - Multiply scale by this on success (default 2.0)
    /// * `backoff_factor` - Multiply scale by this on overflow (default 0.5)
    /// * `growth_interval` - Try to grow every N steps (default 2000)
    pub fn with_growth_params(
        mut self,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: u32,
    ) -> Self {
        self.growth_factor = growth_factor;
        self.backoff_factor = backoff_factor;
        self.growth_interval = growth_interval;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_scaler_creation() {
        let scaler = LossScaler::new(256.0);
        assert_eq!(scaler.current_scale(), 256.0);
    }

    #[test]
    fn test_loss_scaler_for_transformer() {
        let scaler = LossScaler::for_transformer(12);
        let expected = 256.0 * (12.0f32).sqrt();
        assert!((scaler.current_scale() - expected).abs() < 0.01);
    }

    #[test]
    fn test_scale_loss() {
        let scaler = LossScaler::new(256.0);
        let loss = 0.5;
        let scaled = scaler.scale_loss(loss);
        assert_eq!(scaled, 128.0);
    }

    #[test]
    fn test_unscale_grads() {
        let scaler = LossScaler::new(256.0);
        let mut grads = vec![256.0f32; 10];
        scaler.unscale_grads(&mut grads);
        for grad in &grads {
            assert!((grad - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_update_valid_grads() {
        let mut scaler = LossScaler::new(256.0);
        let grads = vec![1.0f32; 100];
        assert!(scaler.update(&grads));
        assert_eq!(scaler.current_scale(), 256.0); // No growth yet
    }

    #[test]
    fn test_update_overflow_grads() {
        let mut scaler = LossScaler::new(256.0);
        let grads = vec![f32::INFINITY; 100];
        assert!(!scaler.update(&grads));
        assert_eq!(scaler.current_scale(), 128.0); // Backoff applied
    }

    #[test]
    fn test_update_nan_grads() {
        let mut scaler = LossScaler::new(256.0);
        let mut grads = vec![1.0f32; 100];
        grads[50] = f32::NAN;
        assert!(!scaler.update(&grads));
        assert_eq!(scaler.current_scale(), 128.0);
    }

    #[test]
    fn test_scale_growth() {
        let mut scaler = LossScaler::new(256.0).with_growth_params(2.0, 0.5, 5);

        let grads = vec![1.0f32; 100];

        // First 4 updates - no growth
        for _ in 0..4 {
            assert!(scaler.update(&grads));
            assert_eq!(scaler.current_scale(), 256.0);
        }

        // 5th update triggers growth
        assert!(scaler.update(&grads));
        assert_eq!(scaler.current_scale(), 512.0);
    }

    #[test]
    fn test_custom_growth_params() {
        let scaler = LossScaler::new(256.0).with_growth_params(1.5, 0.75, 100);
        assert_eq!(scaler.current_scale(), 256.0);
    }
}
