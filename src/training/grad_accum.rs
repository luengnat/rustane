//! Gradient accumulation for training
//!
//! Accumulates gradients over multiple steps for larger effective batch sizes
//! without allocating huge batches. Supports both FP16 and FP32 gradients.

/// Gradient accumulator
///
/// Accumulates gradients over multiple training steps before updating weights.
/// Enables training with larger effective batch sizes without huge memory usage.
///
/// # Strategy
///
/// 1. Initialize accumulator with parameter count and total steps
/// 2. Compute gradients in mini-batches
/// 3. Call accumulate_fp32/fp16 for each mini-batch
/// 4. After accumulating all steps, call finalize()
/// 5. Use accumulated gradients in optimizer
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
/// accum.accumulate_fp32(&grads1, 1.0);
///
/// // Mini-batch 2
/// let grads2 = vec![0.5f32; 1000];
/// accum.accumulate_fp32(&grads2, 1.0);
///
/// assert!(!accum.is_complete());
///
/// // After all steps...
/// accum.accumulate_fp32(&grads1, 1.0);
/// accum.accumulate_fp32(&grads2, 1.0);
/// assert!(accum.is_complete());
///
/// let final_grads = accum.finalize();
/// ```
pub struct GradAccumulator {
    accum: Vec<f32>,
    count: usize,
    total_steps: usize,
}

impl GradAccumulator {
    /// Create a new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `num_params` - Total number of parameters
    /// * `total_steps` - Number of mini-batch steps to accumulate
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let accum = GradAccumulator::new(5000, 8);
    /// assert_eq!(accum.total_steps(), 8);
    /// assert_eq!(accum.current_step(), 0);
    /// ```
    pub fn new(num_params: usize, total_steps: usize) -> Self {
        GradAccumulator {
            accum: vec![0.0f32; num_params],
            count: 0,
            total_steps,
        }
    }

    /// Accumulate FP16 gradients
    ///
    /// Converts FP16 gradients to FP32 and adds to accumulator.
    /// FP16 is interpreted as bfloat16-like half-precision floats.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient array in FP16 (as u16 bit patterns)
    /// * `scale` - Scale factor for this mini-batch
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads_fp16 = vec![0x3c00u16; 100]; // ~1.0 in FP16
    /// accum.accumulate_fp16(&grads_fp16, 1.0);
    /// ```
    pub fn accumulate_fp16(&mut self, grads: &[u16], scale: f32) {
        if grads.len() != self.accum.len() {
            return; // Silently ignore size mismatch
        }

        self.count += 1;

        for (i, &grad_fp16) in grads.iter().enumerate() {
            // Convert FP16 (u16 bits) to FP32
            let grad_fp32 = half::f16::from_bits(grad_fp16).to_f32();
            self.accum[i] += grad_fp32 * scale;
        }
    }

    /// Accumulate FP32 gradients
    ///
    /// Directly adds FP32 gradients to accumulator.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient array in FP32
    /// * `scale` - Scale factor for this mini-batch
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads_fp32 = vec![0.1f32; 100];
    /// accum.accumulate_fp32(&grads_fp32, 1.0);
    /// assert_eq!(accum.current_step(), 1);
    /// ```
    pub fn accumulate_fp32(&mut self, grads: &[f32], scale: f32) {
        if grads.len() != self.accum.len() {
            return; // Silently ignore size mismatch
        }

        self.count += 1;

        for (i, &grad) in grads.iter().enumerate() {
            self.accum[i] += grad * scale;
        }
    }

    /// Get the accumulated gradients
    ///
    /// Should be called after accumulating all mini-batches
    /// (i.e., when is_complete() returns true).
    ///
    /// Optionally applies averaging: accumulated / num_steps.
    ///
    /// # Returns
    ///
    /// Reference to accumulated gradient array
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads = vec![0.5f32; 100];
    /// accum.accumulate_fp32(&grads, 1.0);
    /// accum.accumulate_fp32(&grads, 1.0);
    /// let result = accum.finalize();
    /// ```
    pub fn finalize(&self) -> &[f32] {
        &self.accum
    }

    /// Reset accumulator for next phase
    ///
    /// Clears accumulated gradients and resets step counter.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads = vec![0.5f32; 100];
    /// accum.accumulate_fp32(&grads, 1.0);
    /// assert_eq!(accum.current_step(), 1);
    /// accum.reset();
    /// assert_eq!(accum.current_step(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.accum.iter_mut().for_each(|g| *g = 0.0);
        self.count = 0;
    }

    /// Check if accumulation is complete
    ///
    /// Returns true when current_step() == total_steps()
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::GradAccumulator;
    /// let mut accum = GradAccumulator::new(100, 2);
    /// let grads = vec![0.5f32; 100];
    /// assert!(!accum.is_complete());
    /// accum.accumulate_fp32(&grads, 1.0);
    /// assert!(!accum.is_complete());
    /// accum.accumulate_fp32(&grads, 1.0);
    /// assert!(accum.is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        self.count >= self.total_steps
    }

    /// Get current step (0-indexed)
    ///
    /// Incremented by each accumulate_* call.
    pub fn current_step(&self) -> usize {
        self.count
    }

    /// Get total number of steps to accumulate
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get remaining steps until completion
    pub fn remaining_steps(&self) -> usize {
        self.total_steps.saturating_sub(self.count)
    }

    /// Get the number of parameters
    pub fn num_params(&self) -> usize {
        self.accum.len()
    }

    /// Get accumulated gradient at index
    pub fn get(&self, idx: usize) -> Option<f32> {
        self.accum.get(idx).copied()
    }

    /// Get accumulated gradients with averaging
    ///
    /// Returns a copy of accumulated gradients divided by current step count.
    /// Useful for getting the average gradient so far.
    pub fn finalize_averaged(&self) -> Vec<f32> {
        let denom = self.count.max(1) as f32;
        self.accum.iter().map(|g| g / denom).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_accum_creation() {
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
    fn test_reset() {
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
