//! Gradient accumulation for training
//!
//! Accumulates gradients over multiple steps for larger effective batch sizes
//! without allocating huge batches. Supports both FP16 and FP32 gradients.

/// Gradient accumulator for multi-step accumulation
///
/// Accumulates gradients over multiple training steps before updating weights.
/// Enables training with larger effective batch sizes without huge memory usage.
/// Also tracks loss values for monitoring training progress.
///
/// # Strategy
///
/// 1. Initialize accumulator with parameter count and total steps
/// 2. Compute gradients in mini-batches
/// 3. Call accumulate() for each mini-batch with gradients, loss, and scale
/// 4. Check is_ready() to determine when optimizer should step
/// 5. Use accumulated gradients from gradients() in optimizer
/// 6. Call reset() for next accumulation phase
///
/// # Example
///
/// ```
/// # use rustane::training::GradAccumulator;
/// let mut accum = GradAccumulator::new(1000, 4); // 1000 params, 4 steps
///
/// // Mini-batch 1
/// let grads1 = vec![1.0f32; 1000];
/// accum.accumulate(&grads1, 2.0, 0.25).unwrap();
///
/// // Mini-batch 2
/// let grads2 = vec![0.5f32; 1000];
/// accum.accumulate(&grads2, 3.0, 0.25).unwrap();
///
/// assert!(!accum.is_ready());
///
/// // After all steps...
/// accum.accumulate(&grads1, 2.5, 0.25).unwrap();
/// accum.accumulate(&grads2, 3.5, 0.25).unwrap();
/// assert!(accum.is_ready());
///
/// let final_grads = accum.gradients();
/// let avg_loss = accum.average_loss();
/// ```
pub struct GradAccumulator {
    /// Accumulated gradients (flattened)
    accumulated_grads: Vec<f32>,

    /// Number of accumulation steps completed
    steps_completed: usize,

    /// Total steps before optimizer should step
    total_steps: usize,

    /// Running sum of losses (for averaging)
    accumulated_loss: f32,
}

impl GradAccumulator {
    /// Create a new gradient accumulator for multi-step accumulation
    ///
    /// # Arguments
    ///
    /// * `param_count` - Number of parameters (gradient vector size)
    /// * `accumulation_steps` - How many backward passes before optimizer step
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let accum = GradAccumulator::new(5000, 8);
    /// assert_eq!(accum.progress(), (0, 8));
    /// assert!(!accum.is_ready());
    /// ```
    pub fn new(param_count: usize, accumulation_steps: usize) -> Self {
        GradAccumulator {
            accumulated_grads: vec![0.0f32; param_count],
            steps_completed: 0,
            total_steps: accumulation_steps,
            accumulated_loss: 0.0,
        }
    }

    /// Accumulate gradients from one backward pass
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient vector from model.backward()
    /// * `loss` - Loss value from this step
    /// * `scale` - Scaling factor (usually 1.0 / accumulation_steps)
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// # use rustane::Result;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads = vec![0.1f32; 100];
    /// accum.accumulate(&grads, 2.0, 0.5).unwrap();
    /// ```
    pub fn accumulate(&mut self, grads: &[f32], loss: f32, scale: f32) -> crate::Result<()> {
        use crate::Error;
        
        if grads.len() != self.accumulated_grads.len() {
            return Err(Error::Other(
                format!("gradient count mismatch: got {}, expected {}",
                    grads.len(), self.accumulated_grads.len())
            ));
        }

        // Accumulate scaled gradients
        for (accum, grad) in self.accumulated_grads.iter_mut().zip(grads.iter()) {
            *accum += grad * scale;
        }

        // Accumulate scaled loss
        self.accumulated_loss += loss * scale;
        self.steps_completed += 1;

        Ok(())
    }

    /// Check if accumulation is complete
    ///
    /// Returns true when steps_completed >= total_steps
    pub fn is_ready(&self) -> bool {
        self.steps_completed >= self.total_steps
    }

    /// Get accumulated gradients (for optimizer)
    ///
    /// Returns a reference to the accumulated gradient array.
    /// Should be called after is_ready() returns true.
    pub fn gradients(&self) -> &[f32] {
        &self.accumulated_grads
    }

    /// Get average loss across accumulated steps
    ///
    /// Returns the accumulated loss value.
    pub fn average_loss(&self) -> f32 {
        self.accumulated_loss
    }

    /// Reset for next accumulation cycle
    ///
    /// Clears all accumulated gradients and loss, resets step counter.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads = vec![0.5f32; 100];
    /// accum.accumulate(&grads, 1.0, 0.5).unwrap();
    /// accum.reset();
    /// assert_eq!(accum.progress(), (0, 2));
    /// assert!(!accum.is_ready());
    /// ```
    pub fn reset(&mut self) {
        self.accumulated_grads.fill(0.0);
        self.accumulated_loss = 0.0;
        self.steps_completed = 0;
    }

    /// Get progress (completed, total)
    ///
    /// Returns a tuple of (steps_completed, total_steps).
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let accum = GradAccumulator::new(100, 4);
    /// assert_eq!(accum.progress(), (0, 4));
    /// ```
    pub fn progress(&self) -> (usize, usize) {
        (self.steps_completed, self.total_steps)
    }

    /// Get number of steps accumulated so far
    ///
    /// Returns how many accumulation steps have been completed.
    /// This is the primary method to track accumulation progress.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 4);
    /// assert_eq!(accum.accumulated_steps(), 0);
    /// accum.accumulate(&vec![1.0; 100], 1.0, 0.25).unwrap();
    /// assert_eq!(accum.accumulated_steps(), 1);
    /// ```
    pub fn accumulated_steps(&self) -> usize {
        self.steps_completed as usize
    }

    /// Get current step (0-indexed)
    ///
    /// Incremented by each accumulate() call.
    /// Kept for backward compatibility with existing code.
    /// Alias for accumulated_steps().
    pub fn current_step(&self) -> usize {
        self.steps_completed as usize
    }

    /// Get total number of steps to accumulate
    /// Kept for backward compatibility with existing code.
    pub fn total_steps(&self) -> usize {
        self.total_steps as usize
    }

    /// Get remaining steps until completion
    /// Kept for backward compatibility with existing code.
    pub fn remaining_steps(&self) -> usize {
        (self.total_steps as usize).saturating_sub(self.steps_completed as usize)
    }

    /// Get the number of parameters
    /// Kept for backward compatibility with existing code.
    pub fn num_params(&self) -> usize {
        self.accumulated_grads.len()
    }

    /// Get accumulated gradient at index
    /// Kept for backward compatibility with existing code.
    pub fn get(&self, idx: usize) -> Option<f32> {
        self.accumulated_grads.get(idx).copied()
    }

    /// Get accumulated gradients with averaging
    ///
    /// Returns a copy of accumulated gradients divided by current step count.
    /// Useful for getting the average gradient so far.
    /// Kept for backward compatibility with existing code.
    pub fn finalize_averaged(&self) -> Vec<f32> {
        let denom = self.steps_completed.max(1) as f32;
        self.accumulated_grads.iter().map(|g| g / denom).collect()
    }

    /// Check if accumulation is complete (backward compatible name)
    /// Kept for backward compatibility with existing code.
    pub fn is_complete(&self) -> bool {
        self.steps_completed >= self.total_steps
    }

    /// Get the accumulated gradients (backward compatible name)
    /// Kept for backward compatibility with existing code.
    pub fn finalize(&self) -> &[f32] {
        &self.accumulated_grads
    }

    /// Accumulate FP16 gradients
    ///
    /// Converts FP16 gradients to FP32 and adds to accumulator.
    /// FP16 is interpreted as bfloat16-like half-precision floats.
    /// Kept for backward compatibility with existing code.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient array in FP16 (as u16 bit patterns)
    /// * `scale` - Scale factor for this mini-batch
    pub fn accumulate_fp16(&mut self, grads: &[u16], scale: f32) {
        if grads.len() != self.accumulated_grads.len() {
            return; // Silently ignore size mismatch
        }

        self.steps_completed += 1;

        for (i, &grad_fp16) in grads.iter().enumerate() {
            // Convert FP16 (u16 bits) to FP32
            let grad_fp32 = half::f16::from_bits(grad_fp16).to_f32();
            self.accumulated_grads[i] += grad_fp32 * scale;
        }
    }

    /// Accumulate FP32 gradients
    ///
    /// Directly adds FP32 gradients to accumulator.
    /// Kept for backward compatibility with existing code.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient array in FP32
    /// * `scale` - Scale factor for this mini-batch
    pub fn accumulate_fp32(&mut self, grads: &[f32], scale: f32) {
        if grads.len() != self.accumulated_grads.len() {
            return; // Silently ignore size mismatch
        }

        self.steps_completed += 1;

        for (i, &grad) in grads.iter().enumerate() {
            self.accumulated_grads[i] += grad * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== NEW TESTS FOR MULTI-STEP ACCUMULATION =====

    #[test]
    fn test_grad_accum_step_tracking() {
        let mut accum = GradAccumulator::new(256, 4);
        assert_eq!(accum.accumulated_steps(), 0);

        // Add some gradients
        accum.accumulate(&vec![0.1; 256], 1.0, 0.25).unwrap();
        assert_eq!(accum.accumulated_steps(), 1);

        accum.accumulate(&vec![0.1; 256], 1.0, 0.25).unwrap();
        assert_eq!(accum.accumulated_steps(), 2);
    }

    #[test]
    fn test_grad_accum_reset_on_optimizer_step() {
        let mut accum = GradAccumulator::new(256, 4);
        accum.accumulate(&vec![0.1; 256], 1.0, 0.25).unwrap();
        accum.accumulate(&vec![0.1; 256], 1.0, 0.25).unwrap();

        assert_eq!(accum.accumulated_steps(), 2);

        // After optimizer step, should reset
        accum.reset(); // or similar finalize mechanism
        assert_eq!(accum.accumulated_steps(), 0);
    }

    #[test]
    fn test_grad_accum_is_ready() {
        let mut accum = GradAccumulator::new(256, 2);
        assert!(!accum.is_ready()); // Need 2 steps

        accum.accumulate(&vec![0.1; 256], 1.0, 0.5).unwrap();
        assert!(!accum.is_ready()); // Still need 1 more

        accum.accumulate(&vec![0.1; 256], 1.0, 0.5).unwrap();
        assert!(accum.is_ready()); // Now ready for optimizer step
    }

    #[test]
    fn test_grad_accumulator_creation() {
        let accum = GradAccumulator::new(100, 4);
        assert_eq!(accum.progress(), (0, 4));
        assert!(!accum.is_ready());
    }

    #[test]
    fn test_accumulation_scaling() {
        let mut accum = GradAccumulator::new(3, 2);
        let grads = vec![2.0, 4.0, 6.0];
        let scale = 0.5;

        accum.accumulate(&grads, 1.0, scale).unwrap();
        let accumulated = accum.gradients();
        assert!((accumulated[0] - 1.0).abs() < 1e-6);  // 2.0 * 0.5
        assert!((accumulated[1] - 2.0).abs() < 1e-6);  // 4.0 * 0.5
        assert!((accumulated[2] - 3.0).abs() < 1e-6);  // 6.0 * 0.5
    }

    #[test]
    fn test_is_ready_signal() {
        let mut accum = GradAccumulator::new(2, 2);
        assert!(!accum.is_ready());

        accum.accumulate(&vec![1.0, 2.0], 0.5, 0.5).unwrap();
        assert!(!accum.is_ready());

        accum.accumulate(&vec![1.0, 2.0], 0.5, 0.5).unwrap();
        assert!(accum.is_ready());
    }

    #[test]
    fn test_loss_averaging() {
        let mut accum = GradAccumulator::new(2, 2);
        accum.accumulate(&vec![1.0, 2.0], 2.0, 0.5).unwrap();
        accum.accumulate(&vec![1.0, 2.0], 4.0, 0.5).unwrap();
        assert!((accum.average_loss() - 3.0).abs() < 1e-6);  // (2.0 * 0.5) + (4.0 * 0.5)
    }

    #[test]
    fn test_reset() {
        let mut accum = GradAccumulator::new(2, 2);
        accum.accumulate(&vec![1.0, 2.0], 1.0, 0.5).unwrap();
        accum.accumulate(&vec![1.0, 2.0], 1.0, 0.5).unwrap();

        accum.reset();
        assert_eq!(accum.progress(), (0, 2));
        assert!(!accum.is_ready());
    }

    // ===== BACKWARD COMPATIBILITY TESTS =====

    #[test]
    fn test_grad_accum_creation_compat() {
        let accum = GradAccumulator::new(1000, 4);
        assert_eq!(accum.num_params(), 1000);
        assert_eq!(accum.total_steps(), 4);
        assert_eq!(accum.current_step(), 0);
        assert!(!accum.is_complete());
    }

    #[test]
    fn test_accumulate_fp32() {
        let mut accum = GradAccumulator::new(10, 2);
        let grads = vec![1.0f32; 10];

        accum.accumulate_fp32(&grads, 1.0);
        assert_eq!(accum.current_step(), 1);
        assert!(!accum.is_complete());

        accum.accumulate_fp32(&grads, 1.0);
        assert_eq!(accum.current_step(), 2);
        assert!(accum.is_complete());

        let result = accum.finalize();
        for grad in result {
            assert_eq!(*grad, 2.0);
        }
    }

    #[test]
    fn test_accumulate_with_scale() {
        let mut accum = GradAccumulator::new(5, 2);
        let grads = vec![0.5f32; 5];

        accum.accumulate_fp32(&grads, 2.0); // Scale by 2
        let result = accum.finalize();
        for grad in result {
            assert_eq!(*grad, 1.0); // 0.5 * 2.0
        }
    }

    #[test]
    fn test_accumulate_fp16() {
        let mut accum = GradAccumulator::new(5, 1);
        let grads_fp16 = vec![0x3c00u16; 5]; // ~1.0 in FP16

        accum.accumulate_fp16(&grads_fp16, 1.0);
        let result = accum.finalize();

        for grad in result {
            // FP16 conversion may have some precision loss
            assert!((grad - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_reset_compat() {
        let mut accum = GradAccumulator::new(10, 2);
        let grads = vec![1.0f32; 10];

        accum.accumulate_fp32(&grads, 1.0);
        assert_eq!(accum.current_step(), 1);

        accum.reset();
        assert_eq!(accum.current_step(), 0);
        let result = accum.finalize();
        for grad in result {
            assert_eq!(*grad, 0.0);
        }
    }

    #[test]
    fn test_remaining_steps() {
        let accum = GradAccumulator::new(10, 4);
        assert_eq!(accum.remaining_steps(), 4);
    }

    #[test]
    fn test_finalize_averaged() {
        let mut accum = GradAccumulator::new(5, 2);
        let grads1 = vec![2.0f32; 5];
        let grads2 = vec![4.0f32; 5];

        accum.accumulate_fp32(&grads1, 1.0);
        accum.accumulate_fp32(&grads2, 1.0);

        let averaged = accum.finalize_averaged();
        for grad in &averaged {
            assert_eq!(*grad, 3.0); // (2.0 + 4.0) / 2
        }
    }

    #[test]
    fn test_size_mismatch() {
        let mut accum = GradAccumulator::new(10, 1);
        let grads = vec![1.0f32; 5]; // Wrong size

        accum.accumulate_fp32(&grads, 1.0); // Should be ignored
        assert_eq!(accum.current_step(), 0); // Not incremented
    }

    #[test]
    fn test_get() {
        let mut accum = GradAccumulator::new(10, 1);
        let grads = vec![0.5f32; 10];

        accum.accumulate_fp32(&grads, 1.0);
        assert_eq!(accum.get(0), Some(0.5));
        assert_eq!(accum.get(5), Some(0.5));
        assert_eq!(accum.get(100), None);
    }
}
