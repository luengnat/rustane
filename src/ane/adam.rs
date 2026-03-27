//! Adam Optimizer for ANE Training
//!
//! Implements the Adam optimization algorithm for updating model weights.
//!
//! Adam maintains per-parameter learning rates and momentum estimates:
//! - m: First moment estimate (exponential moving average of gradients)
//! - v: Second moment estimate (exponential moving average of squared gradients)
//!
//! Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

use crate::ane::ANEError;
use std::time::Instant;

/// Adam optimizer configuration
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Learning rate (alpha)
    pub learning_rate: f32,
    /// Beta1 - exponential decay rate for first moment
    pub beta1: f32,
    /// Beta2 - exponential decay rate for second moment
    pub beta2: f32,
    /// Epsilon - small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Gradient clipping threshold (0 = disabled)
    pub max_grad_norm: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

impl AdamConfig {
    /// Create default AdamW configuration (with weight decay)
    pub fn adamw() -> Self {
        Self {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }

    /// Create configuration for training transformers (common settings)
    pub fn transformer() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
        }
    }
}

/// Adam optimizer state
#[derive(Debug)]
pub struct AdamState {
    /// First moment estimates (momentum)
    pub m: Vec<f32>,
    /// Second moment estimates (velocity)
    pub v: Vec<f32>,
    /// Timestep (number of updates performed)
    pub t: usize,
    /// Configuration
    pub config: AdamConfig,
}

impl AdamState {
    /// Create new Adam state for given number of parameters
    pub fn new(num_params: usize, config: AdamConfig) -> Self {
        Self {
            m: vec![0.0f32; num_params],
            v: vec![0.0f32; num_params],
            t: 0,
            config,
        }
    }

    /// Reset state (for curriculum learning or new training phase)
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }
}

/// Adam optimizer for ANE training
pub struct AdamOptimizer {
    /// State for each parameter group
    states: Vec<AdamState>,
    /// Configuration
    config: AdamConfig,
    /// Timing information
    last_step_time: Option<Instant>,
}

impl AdamOptimizer {
    /// Create new Adam optimizer
    pub fn new(config: AdamConfig) -> Self {
        Self {
            states: Vec::new(),
            config,
            last_step_time: None,
        }
    }

    /// Add a parameter group
    pub fn add_param_group(&mut self, num_params: usize) {
        self.states.push(AdamState::new(num_params, self.config));
    }

    /// Perform Adam update step
    ///
    /// Updates weights in-place using gradients.
    ///
    /// # Arguments
    /// * `group_idx` - Parameter group index
    /// * `weights` - Current weights (updated in-place)
    /// * `gradients` - Gradients (will be modified for clipping)
    pub fn step(
        &mut self,
        group_idx: usize,
        weights: &mut [f32],
        gradients: &mut [f32],
    ) -> Result<(), ANEError> {
        let start = Instant::now();

        let num_states = self.states.len();
        let state = self
            .states
            .get_mut(group_idx)
            .ok_or_else(|| ANEError::InvalidShape {
                expected: format!("valid group index 0-{}", num_states.saturating_sub(1)),
                got: group_idx.to_string(),
            })?;

        if weights.len() != state.m.len() {
            return Err(ANEError::InvalidShape {
                expected: format!("{} weights", state.m.len()),
                got: weights.len().to_string(),
            });
        }

        if gradients.len() != weights.len() {
            return Err(ANEError::InvalidShape {
                expected: format!("{} gradients", weights.len()),
                got: gradients.len().to_string(),
            });
        }

        // Gradient clipping
        if self.config.max_grad_norm > 0.0 {
            clip_gradients(gradients, self.config.max_grad_norm);
        }

        // Increment timestep
        state.t += 1;
        let t = state.t as f32;

        // Precompute bias corrections
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);

        // Update each parameter
        for i in 0..weights.len() {
            let grad = gradients[i];
            let weight = &mut weights[i];

            // Weight decay (AdamW - decoupled from gradient)
            if self.config.weight_decay > 0.0 {
                *weight -= self.config.learning_rate * self.config.weight_decay * *weight;
            }

            // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            state.m[i] = self.config.beta1 * state.m[i] + (1.0 - self.config.beta1) * grad;

            // Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
            state.v[i] = self.config.beta2 * state.v[i] + (1.0 - self.config.beta2) * grad * grad;

            // Compute bias-corrected first moment: m̂_t = m_t / (1 - β1^t)
            let m_hat = state.m[i] / bias_correction1;

            // Compute bias-corrected second moment: v̂_t = v_t / (1 - β2^t)
            let v_hat = state.v[i] / bias_correction2;

            // Update parameters: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
            let update = self.config.learning_rate * m_hat / (v_hat.sqrt() + self.config.epsilon);
            *weight -= update;

            // Sanity check for NaN/Inf
            if !weight.is_finite() {
                return Err(ANEError::InvalidShape {
                    expected: "finite weight".to_string(),
                    got: format!("{} at index {}", *weight, i),
                });
            }
        }

        self.last_step_time = Some(start);
        Ok(())
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// Get current timestep for a group
    pub fn timestep(&self, group_idx: usize) -> Option<usize> {
        self.states.get(group_idx).map(|s| s.t)
    }
}

/// Clip gradients by global norm
fn clip_gradients(gradients: &mut [f32], max_norm: f32) {
    // Compute global norm
    let global_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

    // Clip if necessary
    if global_norm > max_norm {
        let scale = max_norm / global_norm;
        for g in gradients.iter_mut() {
            *g *= scale;
        }
    }
}

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get learning rate for given step
    fn get_lr(&self, step: usize, initial_lr: f32) -> f32;
}

/// Cosine annealing scheduler with warmup
pub struct CosineAnnealingScheduler {
    /// Total training steps
    pub total_steps: usize,
    /// Warmup steps (linear warmup from 0 to initial_lr)
    pub warmup_steps: usize,
    /// Minimum learning rate
    pub min_lr: f32,
}

impl LRScheduler for CosineAnnealingScheduler {
    fn get_lr(&self, step: usize, initial_lr: f32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f32 / self.warmup_steps as f32;
            initial_lr * progress
        } else {
            // Cosine annealing
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let cosine = (progress * std::f32::consts::PI).cos();
            self.min_lr + (initial_lr - self.min_lr) * 0.5 * (1.0 + cosine)
        }
    }
}

/// Linear scheduler with warmup
pub struct LinearScheduler {
    /// Total training steps
    pub total_steps: usize,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Minimum learning rate
    pub min_lr: f32,
}

impl LRScheduler for LinearScheduler {
    fn get_lr(&self, step: usize, initial_lr: f32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f32 / self.warmup_steps as f32;
            initial_lr * progress
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Linear decay
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            initial_lr - (initial_lr - self.min_lr) * progress
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_step() {
        let config = AdamConfig::default();
        let mut optimizer = AdamOptimizer::new(config);
        optimizer.add_param_group(4);

        let mut weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut gradients = vec![0.1f32, 0.2, 0.3, 0.4];

        // Perform update
        optimizer.step(0, &mut weights, &mut gradients).unwrap();

        // Weights should have changed
        assert_ne!(weights[0], 1.0);
        assert!(weights.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_gradient_clipping() {
        let mut grads = vec![10.0f32, 20.0, 30.0];
        clip_gradients(&mut grads, 1.0);

        // Norm should be <= 1.0
        let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(norm <= 1.01); // Allow small floating point error
    }

    #[test]
    fn test_cosine_scheduler() {
        let scheduler = CosineAnnealingScheduler {
            total_steps: 1000,
            warmup_steps: 100,
            min_lr: 1e-6,
        };

        // At step 0, should be 0 (start of warmup)
        let lr_0 = scheduler.get_lr(0, 1e-4);
        assert!(lr_0 < 1e-5);

        // At warmup end, should be full LR
        let lr_warmup = scheduler.get_lr(100, 1e-4);
        assert!((lr_warmup - 1e-4).abs() < 1e-7);

        // At end, should be min_lr
        let lr_end = scheduler.get_lr(1000, 1e-4);
        assert!((lr_end - 1e-6).abs() < 1e-7);
    }
}
