//! Distributed training synchronization primitives
//!
//! Provides all-reduce and other collective operations for multi-ANE training.

use crate::{Error, Result};
use std::sync::{Arc, Mutex};

/// All-reduce operation for gradient synchronization across multiple devices
///
/// Averages gradients from multiple sources (e.g., different ANE devices)
/// to ensure consistent model updates during distributed training.
#[derive(Clone, Debug)]
pub struct AllReduce {
    /// Number of devices participating in the all-reduce
    num_devices: usize,
    /// Reduction mode (average, sum, min, max)
    mode: ReduceMode,
}

/// Reduction mode for all-reduce operations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceMode {
    /// Average gradients across devices (default for data parallelism)
    Average,
    /// Sum gradients across devices
    Sum,
    /// Take minimum across devices
    Min,
    /// Take maximum across devices
    Max,
}

impl Default for AllReduce {
    fn default() -> Self {
        Self::new(2)
    }
}

impl AllReduce {
    /// Create a new all-reduce operation
    ///
    /// # Arguments
    ///
    /// * `num_devices` - Number of devices to synchronize
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::training::distributed::AllReduce;
    /// let all_reduce = AllReduce::new(4);
    /// ```
    pub fn new(num_devices: usize) -> Self {
        Self {
            num_devices,
            mode: ReduceMode::Average,
        }
    }

    /// Set the reduction mode
    ///
    /// # Arguments
    ///
    /// * `mode` - Reduction mode to use
    pub fn with_mode(mut self, mode: ReduceMode) -> Self {
        self.mode = mode;
        self
    }

    /// Perform all-reduce on gradients from multiple devices
    ///
    /// Takes gradients from each device and returns the reduced (averaged) gradients.
    ///
    /// # Arguments
    ///
    /// * `gradients_list` - Slice of gradient arrays, one per device
    ///
    /// # Returns
    ///
    /// Reduced gradients as a single vector
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParameter` if:
    /// - The gradient list is empty
    /// - Gradients have different lengths
    /// - Number of gradients doesn't match `num_devices`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grads_device0 = vec![1.0f32, 2.0, 3.0];
    /// let grads_device1 = vec![2.0f32, 4.0, 6.0];
    /// let all_reduce = AllReduce::new(2);
    /// let averaged = all_reduce.all_reduce(&[grads_device0, grads_device1])?;
    /// // Result: [1.5, 3.0, 4.5]
    /// ```
    pub fn all_reduce(&self, gradients_list: &[Vec<f32>]) -> Result<Vec<f32>> {
        if gradients_list.is_empty() {
            return Err(Error::InvalidParameter(
                "Gradient list cannot be empty".to_string(),
            ));
        }

        if gradients_list.len() != self.num_devices {
            return Err(Error::InvalidParameter(format!(
                "Expected {} gradient arrays, got {}",
                self.num_devices,
                gradients_list.len()
            )));
        }

        let grad_len = gradients_list[0].len();
        if grad_len == 0 {
            return Err(Error::InvalidParameter(
                "Gradients cannot be empty".to_string(),
            ));
        }

        // Verify all gradients have the same length
        for (i, grads) in gradients_list.iter().enumerate() {
            if grads.len() != grad_len {
                return Err(Error::InvalidParameter(format!(
                    "Gradient at index {} has length {}, expected {}",
                    i,
                    grads.len(),
                    grad_len
                )));
            }
        }

        // Perform reduction
        let mut result = vec![0.0f32; grad_len];

        match self.mode {
            ReduceMode::Average => {
                // Sum all gradients, then divide by count
                for grads in gradients_list {
                    for (r, &g) in result.iter_mut().zip(grads.iter()) {
                        *r += g;
                    }
                }
                let inv_count = 1.0 / self.num_devices as f32;
                for r in &mut result {
                    *r *= inv_count;
                }
            }
            ReduceMode::Sum => {
                // Sum all gradients without dividing
                for grads in gradients_list {
                    for (r, &g) in result.iter_mut().zip(grads.iter()) {
                        *r += g;
                    }
                }
            }
            ReduceMode::Min => {
                // Take element-wise minimum
                for (i, grads) in gradients_list.iter().enumerate() {
                    if i == 0 {
                        result.clone_from(grads);
                    } else {
                        for (r, &g) in result.iter_mut().zip(grads.iter()) {
                            *r = (*r).min(g);
                        }
                    }
                }
            }
            ReduceMode::Max => {
                // Take element-wise maximum
                for (i, grads) in gradients_list.iter().enumerate() {
                    if i == 0 {
                        result.clone_from(grads);
                    } else {
                        for (r, &g) in result.iter_mut().zip(grads.iter()) {
                            *r = (*r).max(g);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// In-place all-reduce on a mutable gradient buffer
    ///
    /// More memory-efficient than `all_reduce` as it accumulates directly
    /// into the target buffer.
    ///
    /// # Arguments
    ///
    /// * `target` - Mutable buffer to accumulate results into
    /// * `gradients_list` - Gradients from other devices (excluding target device)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut my_grads = vec![1.0f32, 2.0, 3.0];
    /// let other_grads = vec![2.0f32, 4.0f32, 6.0];
    /// all_reduce.all_reduce_in_place(&mut my_grads, &[other_grads])?;
    /// // my_grads: [1.5, 3.0, 4.5]
    /// ```
    pub fn all_reduce_in_place(
        &self,
        target: &mut [f32],
        gradients_list: &[Vec<f32>],
    ) -> Result<()> {
        if gradients_list.len() != self.num_devices - 1 {
            return Err(Error::InvalidParameter(format!(
                "Expected {} gradient arrays (excluding target), got {}",
                self.num_devices - 1,
                gradients_list.len()
            )));
        }

        let grad_len = target.len();

        // Verify all gradients have the same length
        for (i, grads) in gradients_list.iter().enumerate() {
            if grads.len() != grad_len {
                return Err(Error::InvalidParameter(format!(
                    "Gradient at index {} has length {}, expected {}",
                    i,
                    grads.len(),
                    grad_len
                )));
            }
        }

        match self.mode {
            ReduceMode::Average => {
                // Add all gradients to target
                for grads in gradients_list {
                    for (t, &g) in target.iter_mut().zip(grads.iter()) {
                        *t += g;
                    }
                }
                // Divide by total device count
                let inv_count = 1.0 / self.num_devices as f32;
                for t in target.iter_mut() {
                    *t *= inv_count;
                }
            }
            ReduceMode::Sum => {
                // Add all gradients to target without dividing
                for grads in gradients_list {
                    for (t, &g) in target.iter_mut().zip(grads.iter()) {
                        *t += g;
                    }
                }
            }
            ReduceMode::Min => {
                // Take element-wise minimum
                for grads in gradients_list {
                    for (t, &g) in target.iter_mut().zip(grads.iter()) {
                        *t = (*t).min(g);
                    }
                }
            }
            ReduceMode::Max => {
                // Take element-wise maximum
                for grads in gradients_list {
                    for (t, &g) in target.iter_mut().zip(grads.iter()) {
                        *t = (*t).max(g);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the number of devices
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Get the reduction mode
    pub fn mode(&self) -> ReduceMode {
        self.mode
    }
}

/// Distributed gradient synchronizer
///
/// Manages gradient synchronization across multiple training devices.
pub struct DistributedSynchronizer {
    /// All-reduce operation for gradient averaging
    all_reduce: AllReduce,
    /// Synchronized gradients cache
    synchronized_gradients: Arc<Mutex<Option<Vec<f32>>>>,
}

impl DistributedSynchronizer {
    /// Create a new distributed synchronizer
    ///
    /// # Arguments
    ///
    /// * `num_devices` - Number of devices to synchronize
    pub fn new(num_devices: usize) -> Self {
        Self {
            all_reduce: AllReduce::new(num_devices),
            synchronized_gradients: Arc::new(Mutex::new(None)),
        }
    }

    /// Synchronize gradients from multiple devices
    ///
    /// # Arguments
    ///
    /// * `gradients_per_device` - Gradients from each device
    ///
    /// # Returns
    ///
    /// Synchronized (averaged) gradients
    pub fn synchronize(&self, gradients_per_device: &[Vec<f32>]) -> Result<Vec<f32>> {
        let result = self.all_reduce.all_reduce(gradients_per_device)?;

        // Cache the synchronized gradients
        *self.synchronized_gradients.lock().unwrap() = Some(result.clone());

        Ok(result)
    }

    /// Get the last synchronized gradients
    ///
    /// Returns None if no synchronization has been performed yet.
    pub fn get_synchronized(&self) -> Option<Vec<f32>> {
        self.synchronized_gradients.lock().unwrap().clone()
    }

    /// Clear the cached synchronized gradients
    pub fn clear_cache(&self) {
        *self.synchronized_gradients.lock().unwrap() = None;
    }

    /// Get the all-reduce configuration
    pub fn all_reduce(&self) -> &AllReduce {
        &self.all_reduce
    }

    /// Get mutable reference to all-reduce configuration
    pub fn all_reduce_mut(&mut self) -> &mut AllReduce {
        &mut self.all_reduce
    }
}

/// Distributed optimizer state
///
/// Manages optimizer state (e.g., Adam moments) across multiple devices.
#[derive(Clone, Debug)]
pub struct DistributedOptimizerState {
    /// First moment (m) for Adam optimizer - sharded per device
    pub m: Vec<f32>,
    /// Second moment (v) for Adam optimizer - sharded per device
    pub v: Vec<f32>,
    /// Number of devices sharing this state
    pub num_devices: usize,
    /// Device index for this shard
    pub device_index: usize,
}

impl DistributedOptimizerState {
    /// Create a new distributed optimizer state
    ///
    /// # Arguments
    ///
    /// * `param_count` - Total number of parameters
    /// * `num_devices` - Number of devices for sharding
    /// * `device_index` - Index of this device (0 to num_devices-1)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = DistributedOptimizerState::new(1000000, 4, 0)?;
    /// // Device 0 holds parameters 0-249999
    /// ```
    pub fn new(param_count: usize, num_devices: usize, device_index: usize) -> Result<Self> {
        if num_devices == 0 {
            return Err(Error::InvalidParameter(
                "num_devices must be at least 1".to_string(),
            ));
        }

        if device_index >= num_devices {
            return Err(Error::InvalidParameter(format!(
                "device_index {} must be less than num_devices {}",
                device_index, num_devices
            )));
        }

        // Calculate shard size for this device
        let base_size = param_count / num_devices;
        let remainder = param_count % num_devices;
        let shard_size = if device_index < remainder {
            base_size + 1
        } else {
            base_size
        };

        Ok(Self {
            m: vec![0.0f32; shard_size],
            v: vec![0.0f32; shard_size],
            num_devices,
            device_index,
        })
    }

    /// Get the parameter range for this device's shard
    ///
    /// # Returns
    ///
    /// Tuple of (start_index, end_index) for this device's parameters
    pub fn shard_range(&self, total_params: usize) -> (usize, usize) {
        let base_size = total_params / self.num_devices;
        let remainder = total_params % self.num_devices;

        let start = if self.device_index < remainder {
            self.device_index * (base_size + 1)
        } else {
            remainder * (base_size + 1) + (self.device_index - remainder) * base_size
        };

        let end = start + self.m.len();

        (start, end)
    }

    /// Extract this device's shard from full gradients
    ///
    /// # Arguments
    ///
    /// * `full_gradients` - Complete gradient vector
    ///
    /// # Returns
    ///
    /// Gradient shard for this device
    pub fn extract_shard(&self, full_gradients: &[f32]) -> Vec<f32> {
        let (start, end) = self.shard_range(full_gradients.len());
        full_gradients[start..end].to_vec()
    }

    /// Merge this device's shard into full gradients
    ///
    /// # Arguments
    ///
    /// * `shard_gradients` - Gradient shard from this device
    /// * `full_gradients` - Mutable full gradient vector
    pub fn merge_shard(&self, shard_gradients: &[f32], full_gradients: &mut [f32]) -> Result<()> {
        let (start, end) = self.shard_range(full_gradients.len());

        if shard_gradients.len() != (end - start) {
            return Err(Error::InvalidParameter(format!(
                "Shard size {} doesn't match expected range {}..{}",
                shard_gradients.len(),
                start,
                end
            )));
        }

        full_gradients[start..end].copy_from_slice(shard_gradients);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_reduce_average() {
        let grads0 = vec![1.0f32, 2.0, 3.0];
        let grads1 = vec![3.0f32, 4.0, 5.0];

        let all_reduce = AllReduce::new(2);
        let result = all_reduce.all_reduce(&[grads0, grads1]).unwrap();

        assert_eq!(result, vec![2.0, 3.0, 4.0]); // Average
    }

    #[test]
    fn test_all_reduce_sum() {
        let grads0 = vec![1.0f32, 2.0, 3.0];
        let grads1 = vec![3.0f32, 4.0, 5.0];

        let all_reduce = AllReduce::new(2).with_mode(ReduceMode::Sum);
        let result = all_reduce.all_reduce(&[grads0, grads1]).unwrap();

        assert_eq!(result, vec![4.0, 6.0, 8.0]); // Sum
    }

    #[test]
    fn test_all_reduce_min() {
        let grads0 = vec![1.0f32, 5.0, 3.0];
        let grads1 = vec![3.0f32, 2.0, 5.0];

        let all_reduce = AllReduce::new(2).with_mode(ReduceMode::Min);
        let result = all_reduce.all_reduce(&[grads0, grads1]).unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0]); // Min
    }

    #[test]
    fn test_all_reduce_max() {
        let grads0 = vec![1.0f32, 5.0, 3.0];
        let grads1 = vec![3.0f32, 2.0, 5.0];

        let all_reduce = AllReduce::new(2).with_mode(ReduceMode::Max);
        let result = all_reduce.all_reduce(&[grads0, grads1]).unwrap();

        assert_eq!(result, vec![3.0, 5.0, 5.0]); // Max
    }

    #[test]
    fn test_all_reduce_in_place() {
        let mut my_grads = vec![1.0f32, 2.0, 3.0];
        let other_grads = vec![3.0f32, 4.0, 5.0];

        let all_reduce = AllReduce::new(2);
        all_reduce.all_reduce_in_place(&mut my_grads, &[other_grads]).unwrap();

        assert_eq!(my_grads, vec![2.0, 3.0, 4.0]); // Average
    }

    #[test]
    fn test_all_reduce_empty_list_fails() {
        let all_reduce = AllReduce::new(2);
        let result = all_reduce.all_reduce(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_reduce_mismatched_lengths_fails() {
        let grads0 = vec![1.0f32, 2.0];
        let grads1 = vec![3.0f32, 4.0, 5.0];

        let all_reduce = AllReduce::new(2);
        let result = all_reduce.all_reduce(&[grads0, grads1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_distributed_optimizer_state_shard_range() {
        let state = DistributedOptimizerState::new(100, 4, 0).unwrap();
        let (start, end) = state.shard_range(100);
        assert_eq!(start, 0);
        assert_eq!(end, 25);

        let state = DistributedOptimizerState::new(100, 4, 1).unwrap();
        let (start, end) = state.shard_range(100);
        assert_eq!(start, 25);
        assert_eq!(end, 50);

        let state = DistributedOptimizerState::new(100, 4, 3).unwrap();
        let (start, end) = state.shard_range(100);
        assert_eq!(start, 75);
        assert_eq!(end, 100);
    }

    #[test]
    fn test_distributed_optimizer_state_extract_shard() {
        let state = DistributedOptimizerState::new(100, 4, 1).unwrap();
        let full_grads: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let shard = state.extract_shard(&full_grads);
        assert_eq!(shard.len(), 25);
        assert_eq!(shard[0], 25.0);
        assert_eq!(shard[24], 49.0);
    }

    #[test]
    fn test_distributed_optimizer_state_merge_shard() {
        let state = DistributedOptimizerState::new(100, 4, 1).unwrap();
        let shard: Vec<f32> = (25..50).map(|i| i as f32 * 2.0).collect();
        let mut full_grads = vec![0.0f32; 100];

        state.merge_shard(&shard, &mut full_grads).unwrap();

        // Check shard was merged correctly
        assert_eq!(full_grads[25], 50.0);
        assert_eq!(full_grads[49], 98.0);
    }

    #[test]
    fn test_distributed_synchronizer() {
        let sync = DistributedSynchronizer::new(2);
        let grads0 = vec![1.0f32, 2.0, 3.0];
        let grads1 = vec![3.0f32, 4.0, 5.0];

        let result = sync.synchronize(&[grads0, grads1]).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);

        // Check cache
        let cached = sync.get_synchronized();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_distributed_optimizer_state_invalid_device_index() {
        let result = DistributedOptimizerState::new(100, 4, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_distributed_optimizer_state_zero_devices() {
        let result = DistributedOptimizerState::new(100, 0, 0);
        assert!(result.is_err());
    }
}
