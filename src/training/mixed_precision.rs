//! Mixed Precision Training Utilities
//!
//! Provides comprehensive support for mixed precision training with FP16/BF16
//! computations and FP32 master weights for numerical stability.
//!
//! # Overview
//!
//! Mixed precision training combines:
//! - **FP32 master weights**: For numerical stability during optimization
//! - **FP16/BF16 working weights**: For reduced memory and faster computation
//! - **Loss scaling**: To prevent gradient underflow in reduced precision
//!
//! # Memory Savings
//!
//! | Precision | Bytes/element | Memory for 1B params |
//! |-----------|--------------|---------------------|
//! | FP32 | 4 bytes | 4 GB |
//! | FP16 | 2 bytes | 2 GB |
//! | BF16 | 2 bytes | 2 GB |
//!
//! # Quick Start
//!
//! ```no_run
//! use rustane::training::mixed_precision::{Precision, MasterWeights, MixedPrecisionState};
//!
//! // Create master weights (FP32)
//! let mut master = MasterWeights::new(1000);
//!
//! // Get working weights (FP16 view)
//! let working = master.to_fp16();
//!
//! // After computing gradients in FP16, update master weights
//! let fp16_grads = vec![1.0f16; 1000];
//! master.apply_gradients(&fp16_grads, 0.001);
//! ```
//!
//! # Architecture
//!
//! The mixed precision training flow:
//! 1. Copy FP32 master weights → FP16 working weights
//! 2. Forward pass in FP16/BF16
//! 3. Backward pass in FP16/BF16 (with loss scaling)
//! 4. Unscale gradients
//! 5. Apply gradients to FP32 master weights
//! 6. Repeat

use crate::training::loss_scale::LossScaler;
use half::{bf16, f16};

/// Convert FP32 value to FP16
#[inline]
pub fn f32_to_fp16(value: f32) -> f16 {
    f16::from_f32(value)
}

/// Convert FP16 value to FP32
#[inline]
pub fn fp16_to_f32(value: f16) -> f32 {
    value.to_f32()
}

/// Convert FP32 value to BF16
#[inline]
pub fn f32_to_bf16(value: f32) -> bf16 {
    bf16::from_f32(value)
}

/// Convert BF16 value to FP32
#[inline]
pub fn bf16_to_f32(value: bf16) -> f32 {
    value.to_f32()
}

/// Convert a slice of FP32 values to FP16
pub fn f32_slice_to_fp16(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

/// Convert a slice of FP16 values to FP32
pub fn fp16_slice_to_f32(values: &[f16]) -> Vec<f32> {
    values.iter().map(|&v| v.to_f32()).collect()
}

/// Convert a slice of FP32 values to BF16
pub fn f32_slice_to_bf16(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|&v| bf16::from_f32(v)).collect()
}

/// Convert a slice of BF16 values to FP32
pub fn bf16_slice_to_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|&v| v.to_f32()).collect()
}

/// Master weights storage in FP32
///
/// Maintains full-precision weights for numerical stability during training.
/// Working copies in FP16/BF16 are derived from these master weights.
#[derive(Debug, Clone)]
pub struct MasterWeights {
    /// FP32 weight values
    weights: Vec<f32>,
    /// Optional momentum buffer (FP32)
    momentum: Option<Vec<f32>>,
    /// Optional variance buffer (FP32) for Adam-style optimizers
    variance: Option<Vec<f32>>,
}

impl MasterWeights {
    /// Create new master weights storage
    pub fn new(num_params: usize) -> Self {
        Self {
            weights: vec![0.0f32; num_params],
            momentum: None,
            variance: None,
        }
    }

    /// Create from existing weights
    pub fn from_vec(weights: Vec<f32>) -> Self {
        Self {
            weights,
            momentum: None,
            variance: None,
        }
    }

    /// Get number of parameters
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Get mutable reference to weights
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Get immutable reference to weights
    pub fn as_slice(&self) -> &[f32] {
        &self.weights
    }

    /// Initialize with random values (Xavier/He initialization)
    pub fn xavier_init(&mut self, fan_in: usize, fan_out: usize) {
        use rand::{thread_rng, Rng};
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();

        let mut rng = thread_rng();
        for w in &mut self.weights {
            *w = rng.gen::<f32>() * std;
        }
    }

    /// Initialize with zeros
    pub fn zero(&mut self) {
        self.weights.fill(0.0);
    }

    /// Enable momentum buffer
    pub fn with_momentum(mut self) -> Self {
        self.momentum = Some(vec![0.0f32; self.weights.len()]);
        self
    }

    /// Enable variance buffer (for Adam)
    pub fn with_variance(mut self) -> Self {
        self.variance = Some(vec![0.0f32; self.weights.len()]);
        self
    }

    /// Enable both momentum and variance
    pub fn with_adam_state(mut self) -> Self {
        let n = self.weights.len();
        self.momentum = Some(vec![0.0f32; n]);
        self.variance = Some(vec![0.0f32; n]);
        self
    }

    /// Get momentum buffer
    pub fn momentum(&self) -> Option<&[f32]> {
        self.momentum.as_deref()
    }

    /// Get mutable momentum buffer
    pub fn momentum_mut(&mut self) -> Option<&mut [f32]> {
        self.momentum.as_deref_mut()
    }

    /// Get variance buffer
    pub fn variance(&self) -> Option<&[f32]> {
        self.variance.as_deref()
    }

    /// Get mutable variance buffer
    pub fn variance_mut(&mut self) -> Option<&mut [f32]> {
        self.variance.as_deref_mut()
    }

    /// Apply gradients with SGD update
    pub fn apply_sgd(&mut self, grads: &[f32], lr: f32) {
        for (w, g) in self.weights.iter_mut().zip(grads.iter()) {
            *w -= lr * g;
        }
    }

    /// Apply gradients with momentum SGD
    pub fn apply_sgd_momentum(&mut self, grads: &[f32], lr: f32, momentum: f32) {
        if let Some(m) = &mut self.momentum {
            for (i, (w, g)) in self.weights.iter_mut().zip(grads.iter()).enumerate() {
                m[i] = momentum * m[i] - lr * g;
                *w += m[i];
            }
        } else {
            self.apply_sgd(grads, lr);
        }
    }

    /// Apply gradients with Adam optimizer
    pub fn apply_adam(
        &mut self,
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
    ) {
        if let (Some(m), Some(v)) = (&mut self.momentum, &mut self.variance) {
            let bias_correction1 = 1.0 - beta1.powi(step as i32);
            let bias_correction2 = 1.0 - beta2.powi(step as i32);

            for ((w, g), (m_i, v_i)) in self
                .weights
                .iter_mut()
                .zip(grads.iter())
                .zip(m.iter_mut().zip(v.iter_mut()))
            {
                // Update biased first moment estimate
                *m_i = beta1 * *m_i + (1.0 - beta1) * g;
                // Update biased second raw moment estimate
                *v_i = beta2 * *v_i + (1.0 - beta2) * g * g;

                // Compute bias-corrected first moment estimate
                let m_hat = *m_i / bias_correction1;
                // Compute bias-corrected second raw moment estimate
                let v_hat = *v_i / bias_correction2;

                // Update weights
                *w -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        } else {
            // Fall back to SGD if Adam state not initialized
            self.apply_sgd(grads, lr);
        }
    }
}

/// Working precision weights (FP16 or BF16)
///
/// A view of master weights in reduced precision for forward/backward passes.
/// This is a temporary copy that gets updated from master weights each step.
#[derive(Debug, Clone)]
pub enum WorkingWeights {
    /// FP16 working weights
    Fp16(Vec<f16>),
    /// BF16 working weights
    Bf16(Vec<bf16>),
}

impl WorkingWeights {
    /// Create FP16 working weights from master
    pub fn from_master_fp16(master: &MasterWeights) -> Self {
        let fp16_weights = master.weights.iter().map(|&w| f16::from_f32(w)).collect();
        Self::Fp16(fp16_weights)
    }

    /// Create BF16 working weights from master
    pub fn from_master_bf16(master: &MasterWeights) -> Self {
        let bf16_weights = master.weights.iter().map(|&w| bf16::from_f32(w)).collect();
        Self::Bf16(bf16_weights)
    }

    /// Get number of parameters
    pub fn len(&self) -> usize {
        match self {
            Self::Fp16(w) => w.len(),
            Self::Bf16(w) => w.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get as FP16 slice (panics if BF16)
    pub fn as_fp16(&self) -> &[f16] {
        match self {
            Self::Fp16(w) => w,
            Self::Bf16(_) => panic!("WorkingWeights is BF16, not FP16"),
        }
    }

    /// Get as BF16 slice (panics if FP16)
    pub fn as_bf16(&self) -> &[bf16] {
        match self {
            Self::Bf16(w) => w,
            Self::Fp16(_) => panic!("WorkingWeights is FP16, not BF16"),
        }
    }

    /// Convert to FP32 for gradient application
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::Fp16(w) => w.iter().map(|&v| v.to_f32()).collect(),
            Self::Bf16(w) => w.iter().map(|&v| v.to_f32()).collect(),
        }
    }

    /// Copy from master weights (refresh working copy)
    pub fn sync_from_master(&mut self, master: &MasterWeights) {
        match self {
            Self::Fp16(w) => {
                for (dest, src) in w.iter_mut().zip(master.weights.iter()) {
                    *dest = f16::from_f32(*src);
                }
            }
            Self::Bf16(w) => {
                for (dest, src) in w.iter_mut().zip(master.weights.iter()) {
                    *dest = bf16::from_f32(*src);
                }
            }
        }
    }
}

/// Mixed precision training state
///
/// Manages the full state of mixed precision training including:
/// - Master weights (FP32)
/// - Working weights (FP16/BF16)
/// - Loss scaler for FP16 training
#[derive(Debug)]
pub struct MixedPrecisionState {
    /// Master weights in FP32
    pub master: MasterWeights,
    /// Working weights in reduced precision
    pub working: WorkingWeights,
    /// Loss scaler for FP16 training (None for BF16)
    pub loss_scaler: Option<LossScaler>,
    /// Current training step
    pub step: usize,
}

impl MixedPrecisionState {
    /// Create new mixed precision state with FP16
    pub fn new_fp16(num_params: usize, initial_loss_scale: f32) -> Self {
        let master = MasterWeights::new(num_params);
        let working = WorkingWeights::from_master_fp16(&master);
        let loss_scaler = Some(LossScaler::new(initial_loss_scale));

        Self {
            master,
            working,
            loss_scaler,
            step: 0,
        }
    }

    /// Create new mixed precision state with BF16
    pub fn new_bf16(num_params: usize) -> Self {
        let master = MasterWeights::new(num_params);
        let working = WorkingWeights::from_master_bf16(&master);

        // BF16 doesn't need loss scaling due to better dynamic range
        Self {
            master,
            working,
            loss_scaler: None,
            step: 0,
        }
    }

    /// Create from existing FP32 weights (FP16)
    pub fn from_weights_fp16(weights: Vec<f32>, initial_loss_scale: f32) -> Self {
        let master = MasterWeights::from_vec(weights);
        let working = WorkingWeights::from_master_fp16(&master);
        let loss_scaler = Some(LossScaler::new(initial_loss_scale));

        Self {
            master,
            working,
            loss_scaler,
            step: 0,
        }
    }

    /// Create from existing FP32 weights (BF16)
    pub fn from_weights_bf16(weights: Vec<f32>) -> Self {
        let master = MasterWeights::from_vec(weights);
        let working = WorkingWeights::from_master_bf16(&master);

        Self {
            master,
            working,
            loss_scaler: None,
            step: 0,
        }
    }

    /// Check if using FP16 precision
    pub fn is_fp16(&self) -> bool {
        matches!(self.working, WorkingWeights::Fp16(_))
    }

    /// Check if using BF16 precision
    pub fn is_bf16(&self) -> bool {
        matches!(self.working, WorkingWeights::Bf16(_))
    }

    /// Get loss scaler (only for FP16)
    pub fn loss_scaler(&self) -> Option<&LossScaler> {
        self.loss_scaler.as_ref()
    }

    /// Get mutable loss scaler (only for FP16)
    pub fn loss_scaler_mut(&mut self) -> Option<&mut LossScaler> {
        self.loss_scaler.as_mut()
    }

    /// Scale loss for FP16 training
    pub fn scale_loss(&self, loss: f32) -> f32 {
        match &self.loss_scaler {
            Some(scaler) => scaler.scale_loss(loss),
            None => loss,
        }
    }

    /// Unscale gradients in-place
    pub fn unscale_grads(&self, grads: &mut [f32]) {
        if let Some(scaler) = &self.loss_scaler {
            scaler.unscale_grads(grads);
        }
    }

    /// Apply gradients to master weights
    pub fn apply_gradients(&mut self, grads: &[f32], lr: f32) {
        self.master.apply_sgd(grads, lr);
        self.step += 1;
    }

    /// Apply gradients with Adam optimizer
    pub fn apply_adam(&mut self, grads: &[f32], lr: f32, beta1: f32, beta2: f32, eps: f32) {
        let step = self.step + 1;
        self.master.apply_adam(grads, lr, beta1, beta2, eps, step);
        self.step = step;
    }

    /// Sync working weights from master (call after applying gradients)
    pub fn sync_working_weights(&mut self) {
        self.working.sync_from_master(&self.master);
    }

    /// Complete a training step
    ///
    /// 1. Unscale gradients
    /// 2. Check gradient health (for FP16)
    /// 3. Apply gradients to master
    /// 4. Sync working weights
    pub fn complete_step(&mut self, grads: &mut [f32], lr: f32) -> Result<bool, String> {
        // Unscale gradients
        self.unscale_grads(grads);

        // Check gradient health for FP16
        let valid = if let Some(scaler) = &mut self.loss_scaler {
            scaler.update(grads)
        } else {
            !grads.iter().any(|&g| !g.is_finite())
        };

        if !valid {
            return Ok(false);
        }

        // Apply gradients
        self.apply_gradients(grads, lr);

        // Sync working weights for next step
        self.sync_working_weights();

        Ok(true)
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let master_bytes = self.master.len() * 4; // FP32 = 4 bytes
        let working_bytes = self.working.len() * 2; // FP16/BF16 = 2 bytes
        master_bytes + working_bytes
    }

    /// Get memory savings vs pure FP32 training
    pub fn memory_savings_ratio(&self) -> f64 {
        let fp32_only_bytes = self.master.len() * 4 * 2; // Master + working in FP32
        let mixed_bytes = self.memory_bytes() as f64;
        mixed_bytes / fp32_only_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_conversion() {
        let value: f32 = 3.14159;

        // FP32 -> FP16 -> FP32
        let fp16 = f32_to_fp16(value);
        let back = fp16_to_f32(fp16);
        assert!((back - value).abs() < 0.01);

        // FP32 -> BF16 -> FP32
        let bf16_val = f32_to_bf16(value);
        let back_bf16 = bf16_to_f32(bf16_val);
        assert!((back_bf16 - value).abs() < 0.01);
    }

    #[test]
    fn test_slice_conversion() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];

        let fp16_vals = f32_slice_to_fp16(&values);
        assert_eq!(fp16_vals.len(), 4);

        let back = fp16_slice_to_f32(&fp16_vals);
        for (orig, converted) in values.iter().zip(back.iter()) {
            assert!((orig - converted).abs() < 0.01);
        }
    }

    #[test]
    fn test_master_weights_creation() {
        let master = MasterWeights::new(100);
        assert_eq!(master.len(), 100);
        assert!(master.momentum().is_none());
        assert!(master.variance().is_none());
    }

    #[test]
    fn test_master_weights_with_adam() {
        let mut master = MasterWeights::new(100).with_adam_state();
        assert!(master.momentum().is_some());
        assert!(master.variance().is_some());

        // Apply Adam update
        let grads = vec![0.1f32; 100];
        master.apply_adam(&grads, 0.001, 0.9, 0.999, 1e-8, 1);

        // Weights should have changed
        assert!(master.as_slice().iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_working_weights_sync() {
        let mut master = MasterWeights::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let mut working = WorkingWeights::from_master_fp16(&master);

        // Change master
        master.as_mut_slice()[0] = 10.0;

        // Sync working
        working.sync_from_master(&master);

        // Verify sync
        let back = working.to_f32();
        assert!((back[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_mixed_precision_state_fp16() {
        let state = MixedPrecisionState::new_fp16(100, 256.0);

        assert!(state.is_fp16());
        assert!(state.loss_scaler().is_some());

        // Scale loss
        let scaled = state.scale_loss(1.0);
        assert!(scaled > 1.0);
    }

    #[test]
    fn test_mixed_precision_state_bf16() {
        let state = MixedPrecisionState::new_bf16(100);

        assert!(state.is_bf16());
        assert!(state.loss_scaler().is_none());

        // BF16 doesn't scale loss
        let scaled = state.scale_loss(1.0);
        assert_eq!(scaled, 1.0);
    }

    #[test]
    fn test_memory_savings() {
        let state = MixedPrecisionState::new_fp16(1000, 256.0);

        // Mixed precision should use less memory than pure FP32
        let ratio = state.memory_savings_ratio();
        assert!(ratio < 1.0);
        assert!(ratio > 0.5); // Should be around 0.75 (master FP32 + working FP16)
    }

    #[test]
    fn test_training_step() {
        let mut state = MixedPrecisionState::new_fp16(100, 256.0);

        // Simulate valid gradients
        let grads = vec![0.01f32; 100];

        let success = state.complete_step(&mut grads.clone(), 0.001);
        assert!(success.unwrap());
    }
}
