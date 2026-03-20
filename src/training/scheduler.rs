//! Learning rate schedulers for adaptive learning during training

/// Trait for learning rate scheduling
///
/// Schedulers adjust the learning rate during training based on the current step.
/// Common strategies include:
/// - Warmup: linearly increase LR from 0 to target
/// - Cosine annealing: smoothly decay LR using cosine schedule
/// - Constant: maintain fixed LR
pub trait LRScheduler: Send {
    /// Get the learning rate for the current step
    ///
    /// # Arguments
    /// - `step`: Current training step (0-based)
    ///
    /// # Returns
    /// Learning rate to use for this step
    fn get_lr(&self, step: u32) -> f32;
}

/// Constant learning rate (no scheduling)
///
/// Returns the same learning rate for all steps.
///
/// # Example
///
/// ```
/// use rustane::training::{ConstantScheduler, LRScheduler};
///
/// let scheduler = ConstantScheduler::new(0.001);
/// assert_eq!(scheduler.get_lr(0), 0.001);
/// assert_eq!(scheduler.get_lr(1000), 0.001);
/// ```
#[derive(Debug, Clone)]
pub struct ConstantScheduler {
    lr: f32,
}

impl ConstantScheduler {
    /// Create a new constant scheduler
    pub fn new(lr: f32) -> Self {
        ConstantScheduler { lr }
    }
}

impl LRScheduler for ConstantScheduler {
    fn get_lr(&self, _step: u32) -> f32 {
        self.lr
    }
}

/// Warmup followed by linear decay
///
/// Linearly increases learning rate from 0 to peak over warmup steps,
/// then linearly decays to zero over remaining steps.
///
/// Useful for stable training starts and controlled convergence.
///
/// # Example
///
/// ```
/// use rustane::training::{WarmupLinearScheduler, LRScheduler};
///
/// let scheduler = WarmupLinearScheduler::new(
///     peak_lr: 0.001,
///     warmup_steps: 1000,
///     total_steps: 10000,
/// );
/// assert!(scheduler.get_lr(0) < scheduler.get_lr(500)); // Warmup phase
/// assert!(scheduler.get_lr(5000) > scheduler.get_lr(9000)); // Decay phase
/// ```
#[derive(Debug, Clone)]
pub struct WarmupLinearScheduler {
    peak_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
}

impl WarmupLinearScheduler {
    /// Create a new warmup-linear scheduler
    ///
    /// # Arguments
    /// - `peak_lr`: Maximum learning rate (reached at end of warmup)
    /// - `warmup_steps`: Number of steps to linearly increase LR
    /// - `total_steps`: Total number of training steps
    pub fn new(peak_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        WarmupLinearScheduler {
            peak_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LRScheduler for WarmupLinearScheduler {
    fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup: 0 -> peak_lr
            let progress = (step + 1) as f32 / self.warmup_steps as f32;
            (self.peak_lr * progress).min(self.peak_lr)
        } else {
            // Linear decay: peak_lr -> 0
            let remaining = (self.total_steps - step) as f32;
            let decay_steps = (self.total_steps - self.warmup_steps) as f32;
            self.peak_lr * (remaining / decay_steps)
        }
    }
}

/// Warmup followed by cosine annealing decay
///
/// Linearly increases learning rate from 0 to peak over warmup steps,
/// then decays smoothly using cosine annealing to a minimum value.
///
/// Often produces better final results than linear decay.
///
/// # Example
///
/// ```
/// use rustane::training::{WarmupCosineScheduler, LRScheduler};
///
/// let scheduler = WarmupCosineScheduler::new(
///     peak_lr: 0.001,
///     warmup_steps: 500,
///     total_steps: 5000,
///     min_lr: 0.00001,
/// );
/// assert!(scheduler.get_lr(0) < scheduler.get_lr(250)); // Warmup
/// // After warmup, LR follows cosine decay
/// ```
#[derive(Debug, Clone)]
pub struct WarmupCosineScheduler {
    peak_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
    min_lr: f32,
}

impl WarmupCosineScheduler {
    /// Create a new warmup-cosine scheduler
    ///
    /// # Arguments
    /// - `peak_lr`: Maximum learning rate (reached at end of warmup)
    /// - `warmup_steps`: Number of steps to linearly increase LR
    /// - `total_steps`: Total number of training steps
    /// - `min_lr`: Minimum learning rate (floor for cosine decay)
    pub fn new(peak_lr: f32, warmup_steps: u32, total_steps: u32, min_lr: f32) -> Self {
        WarmupCosineScheduler {
            peak_lr,
            warmup_steps,
            total_steps,
            min_lr,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup: 0 -> peak_lr
            let progress = step as f32 / self.warmup_steps as f32;
            self.peak_lr * progress
        } else {
            // Cosine decay: peak_lr -> min_lr
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let progress = progress.min(1.0); // Clamp to [0, 1]

            // Cosine annealing
            let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = ConstantScheduler::new(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }

    #[test]
    fn test_warmup_linear_scheduler_warmup_phase() {
        let scheduler = WarmupLinearScheduler::new(0.001, 100, 1000);

        // At start: LR should be 1/100 of peak (0.001 * 1/100 = 0.00001)
        let lr_start = scheduler.get_lr(0);
        assert!(lr_start > 0.000009 && lr_start < 0.000011);

        // At warmup end (step 99): LR should be 100/100 of peak
        let lr_at_warmup = scheduler.get_lr(99);
        assert!(lr_at_warmup >= 0.000999 && lr_at_warmup <= 0.001001);

        // Proportional to warmup progress (at step 49, should be 50/100)
        let lr_half_warmup = scheduler.get_lr(49);
        assert!(lr_half_warmup > 0.000499 && lr_half_warmup < 0.000501); // Near 0.0005
    }

    #[test]
    fn test_warmup_linear_scheduler_decay_phase() {
        let scheduler = WarmupLinearScheduler::new(0.001, 100, 1000);

        // After warmup: should decay linearly
        let lr_start_decay = scheduler.get_lr(100);
        assert!(lr_start_decay > 0.0008); // Still near peak

        let lr_mid_decay = scheduler.get_lr(550);
        assert!(lr_mid_decay > 0.0004 && lr_mid_decay < 0.0006); // Near 0.0005

        let lr_end = scheduler.get_lr(999);
        assert!(lr_end < 0.0001); // Nearly 0
    }

    #[test]
    fn test_warmup_cosine_scheduler_warmup_phase() {
        let scheduler = WarmupCosineScheduler::new(0.001, 100, 1000, 0.0001);

        // At start: LR should be near 0
        assert!(scheduler.get_lr(0) < 0.0001);

        // At warmup end: LR should be near peak
        let lr_at_warmup = scheduler.get_lr(99);
        assert!(lr_at_warmup > 0.0009); // Very close to 0.001
    }

    #[test]
    fn test_warmup_cosine_scheduler_cosine_phase() {
        let scheduler = WarmupCosineScheduler::new(0.001, 100, 1000, 0.0001);

        // After warmup: should follow cosine decay
        let lr_start = scheduler.get_lr(100);
        assert!(lr_start > 0.0008);

        // Cosine decays smoothly
        let lr_mid = scheduler.get_lr(550);
        assert!(lr_mid > 0.0001 && lr_mid < 0.001);

        // At end: should be near min_lr
        let lr_end = scheduler.get_lr(999);
        assert!(lr_end > 0.0001 && lr_end < 0.0002); // Near min_lr
    }

    #[test]
    fn test_warmup_cosine_scheduler_min_lr_floor() {
        let scheduler = WarmupCosineScheduler::new(0.001, 100, 1000, 0.0001);

        // At the very end, LR should be at or above min_lr
        let lr_end = scheduler.get_lr(1000);
        assert!(lr_end >= 0.0001 - 1e-6); // Allow for small floating point error
    }

    #[test]
    fn test_warmup_steps_zero() {
        let scheduler = WarmupLinearScheduler::new(0.001, 0, 1000);
        // With zero warmup, should immediately start decay
        assert!(scheduler.get_lr(0) > 0.0009);
    }

    #[test]
    fn test_warmup_phase_monotonic() {
        let scheduler = WarmupLinearScheduler::new(0.001, 500, 1000);

        // Warmup phase should be monotonically increasing
        for step in 0..499 {
            let lr = scheduler.get_lr(step);
            let next_lr = scheduler.get_lr(step + 1);
            assert!(
                next_lr >= lr - 1e-9,
                "Step {} -> {}: {} -> {}",
                step,
                step + 1,
                lr,
                next_lr
            );
        }
    }

    #[test]
    fn test_decay_phase_monotonic() {
        let scheduler = WarmupLinearScheduler::new(0.001, 500, 1000);

        // Decay phase should be monotonically decreasing
        for step in 500..999 {
            let lr = scheduler.get_lr(step);
            let next_lr = scheduler.get_lr(step + 1);
            assert!(
                next_lr <= lr + 1e-9,
                "Step {} -> {}: {} -> {}",
                step,
                step + 1,
                lr,
                next_lr
            );
        }
    }
}
