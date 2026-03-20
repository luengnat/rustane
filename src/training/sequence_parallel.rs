//! Sequence Parallelism for Training Very Long Sequences
//!
//! Sequence parallelism splits a long sequence across multiple devices,
//! each processing a local shard. This is particularly useful when:
//! - Sequence length exceeds device memory limits
//! - Model doesn't fit with batch size > 1
//! - Combined with Flash Attention for maximum efficiency
//!
//! # Architecture
//!
//! For a sequence of length N and P devices:
//! - Device i receives sequence shard [i * N/P, (i+1) * N/P)
//! - Each device computes attention for its local shard
//! - Gradient synchronization across device boundaries
//! - Overlapping regions ensure correct attention computation
//!
//! # Memory Savings
//!
//! For sequence_length = 8192, batch_size = 1, num_devices = 4:
//! - Without sequence parallelism: O(8192²) attention memory
//! - With sequence parallelism: O(2048²) per device (4x reduction)

use crate::{Error, Result};
use std::collections::HashMap;

/// Configuration for sequence parallelism
#[derive(Debug, Clone)]
pub struct SequenceParallelConfig {
    /// Number of devices for sequence parallelism
    pub num_devices: usize,
    /// Total sequence length
    pub seq_len: usize,
    /// Overlap size for attention computation (for causal masking)
    pub overlap_size: usize,
}

impl SequenceParallelConfig {
    /// Create a new sequence parallelism configuration
    ///
    /// # Arguments
    ///
    /// * `num_devices` - Number of devices to split across
    /// * `seq_len` - Total sequence length
    /// * `overlap_size` - Overlap between shards (needed for causal attention)
    pub fn new(num_devices: usize, seq_len: usize, overlap_size: usize) -> Result<Self> {
        if num_devices == 0 {
            return Err(Error::InvalidParameter(
                "num_devices must be > 0".to_string(),
            ));
        }

        if seq_len % num_devices != 0 {
            return Err(Error::InvalidParameter(format!(
                "seq_len {} must be divisible by num_devices {}",
                seq_len, num_devices
            )));
        }

        if overlap_size >= seq_len / num_devices {
            return Err(Error::InvalidParameter(
                "overlap_size must be less than shard size".to_string(),
            ));
        }

        Ok(Self {
            num_devices,
            seq_len,
            overlap_size,
        })
    }

    /// Get the shard size for each device
    pub fn shard_size(&self) -> usize {
        self.seq_len / self.num_devices
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.seq_len % self.num_devices != 0 {
            return Err(Error::InvalidParameter(format!(
                "seq_len {} not divisible by num_devices {}",
                self.seq_len, self.num_devices
            )));
        }

        if self.overlap_size >= self.shard_size() {
            return Err(Error::InvalidParameter(
                "overlap_size too large".to_string(),
            ));
        }

        Ok(())
    }
}

/// Shard of a sequence for parallel processing
#[derive(Debug, Clone)]
pub struct SequenceShard {
    /// Device ID (0 to num_devices-1)
    pub device_id: usize,
    /// Local shard data [shard_size, ...]
    pub data: Vec<f32>,
    /// Global position in the full sequence [start, end)
    pub global_range: (usize, usize),
    /// Overlap with previous shard (for attention computation)
    pub left_overlap: Option<Vec<f32>>,
    /// Overlap with next shard (for attention computation)
    pub right_overlap: Option<Vec<f32>>,
}

impl SequenceShard {
    /// Create a new sequence shard
    ///
    /// # Arguments
    ///
    /// * `device_id` - Device ID
    /// * `data` - Local shard data
    /// * `global_range` - Global position (start, end)
    pub fn new(device_id: usize, data: Vec<f32>, global_range: (usize, usize)) -> Self {
        Self {
            device_id,
            data,
            global_range,
            left_overlap: None,
            right_overlap: None,
        }
    }

    /// Get the local shard size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if shard is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Sequence parallelism manager
pub struct SequenceParallelism {
    config: SequenceParallelConfig,
}

impl SequenceParallelism {
    /// Create a new sequence parallelism manager
    pub fn new(config: SequenceParallelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Split a sequence into shards across devices
    ///
    /// # Arguments
    ///
    /// * `sequence` - Full sequence data [seq_len, ...]
    ///
    /// # Returns
    ///
    /// Vector of shards, one per device
    pub fn split_sequence(&self, sequence: &[f32]) -> Result<Vec<SequenceShard>> {
        let seq_len = sequence.len();
        let num_devices = self.config.num_devices;
        let shard_size = self.config.shard_size();

        if seq_len != self.config.seq_len {
            return Err(Error::InvalidParameter(format!(
                "sequence length {} doesn't match config {}",
                seq_len, self.config.seq_len
            )));
        }

        let mut shards = Vec::with_capacity(num_devices);

        for device_id in 0..num_devices {
            let start = device_id * shard_size;
            let end = start + shard_size;

            let data: Vec<f32> = sequence[start..end].to_vec();

            let shard = SequenceShard::new(device_id, data, (start, end));
            shards.push(shard);
        }

        Ok(shards)
    }

    /// Add overlap regions to shards for attention computation
    ///
    /// # Arguments
    ///
    /// * `shards` - Shards to add overlap to
    ///
    /// # Returns
    ///
    /// Shards with overlap regions added
    pub fn add_overlap(&self, shards: &mut [SequenceShard]) -> Result<()> {
        let overlap = self.config.overlap_size;

        if overlap == 0 {
            return Ok(());
        }

        for i in 0..shards.len() {
            // Add left overlap (from previous shard)
            if i > 0 {
                let prev_shard = &shards[i - 1];
                let left_start = prev_shard.data.len() - overlap;
                let left_data: Vec<f32> = prev_shard.data[left_start..].to_vec();
                shards[i].left_overlap = Some(left_data);
            }

            // Add right overlap (from next shard)
            if i < shards.len() - 1 {
                let next_shard = &shards[i + 1];
                let right_data: Vec<f32> = next_shard.data[..overlap].to_vec();
                shards[i].right_overlap = Some(right_data);
            }
        }

        Ok(())
    }

    /// Merge shards back into a single sequence
    ///
    /// # Arguments
    ///
    /// * `shards` - Shards to merge
    ///
    /// # Returns
    ///
    /// Merged sequence data
    pub fn merge_shards(&self, shards: &[SequenceShard]) -> Result<Vec<f32>> {
        if shards.len() != self.config.num_devices {
            return Err(Error::InvalidParameter(format!(
                "Expected {} shards, got {}",
                self.config.num_devices, shards.len()
            )));
        }

        let shard_size = self.config.shard_size();
        let mut merged = Vec::with_capacity(self.config.seq_len);

        for shard in shards {
            if shard.data.len() != shard_size {
                return Err(Error::InvalidParameter(format!(
                    "Shard size mismatch: expected {}, got {}",
                    shard_size, shard.data.len()
                )));
            }
            merged.extend(&shard.data);
        }

        Ok(merged)
    }

    /// Get memory savings per device
    ///
    /// # Returns
    ///
    /// (per_device_memory, standard_memory) in bytes
    pub fn memory_savings(&self, elem_size: usize) -> (usize, usize) {
        let standard_memory = self.config.seq_len * self.config.seq_len * elem_size;
        let per_device_memory = self.config.shard_size() * self.config.shard_size() * elem_size;

        (per_device_memory, standard_memory)
    }

    /// Get memory savings percentage
    ///
    /// # Returns
    ///
    /// Percentage of memory saved per device (0-100)
    pub fn memory_saving_percentage(&self, elem_size: usize) -> f32 {
        let (per_device, standard) = self.memory_savings(elem_size);

        if standard == 0 {
            return 0.0;
        }

        let saved = standard.saturating_sub(per_device);
        (saved as f32 / standard as f32) * 100.0
    }

    /// Validate that all devices have consistent shard sizes
    pub fn validate_shards(&self, shards: &[SequenceShard]) -> Result<()> {
        let expected_size = self.config.shard_size();

        for (i, shard) in shards.iter().enumerate() {
            if shard.data.len() != expected_size {
                return Err(Error::InvalidParameter(format!(
                    "Shard {} has size {}, expected {}",
                    i, shard.data.len(), expected_size
                )));
            }

            let expected_start = i * expected_size;
            let expected_end = expected_start + expected_size;

            if shard.global_range != (expected_start, expected_end) {
                return Err(Error::InvalidParameter(format!(
                    "Shard {} has range {:?}, expected ({}, {})",
                    i, shard.global_range, expected_start, expected_end
                )));
            }
        }

        Ok(())
    }

    /// Get the device assignment for a position
    ///
    /// # Arguments
    ///
    /// * `position` - Position in the sequence
    ///
    /// # Returns
    ///
    /// Device ID that should handle this position
    pub fn get_device_for_position(&self, position: usize) -> Result<usize> {
        if position >= self.config.seq_len {
            return Err(Error::InvalidParameter(format!(
                "Position {} exceeds sequence length {}",
                position, self.config.seq_len
            )));
        }

        Ok(position / self.config.shard_size())
    }

    /// Get all positions for a device
    ///
    /// # Arguments
    ///
    /// * `device_id` - Device ID
    ///
    /// # Returns
    ///
    /// Range of positions (start, end) for this device
    pub fn get_positions_for_device(&self, device_id: usize) -> Result<(usize, usize)> {
        if device_id >= self.config.num_devices {
            return Err(Error::InvalidParameter(format!(
                "Device ID {} exceeds num_devices {}",
                device_id, self.config.num_devices
            )));
        }

        let shard_size = self.config.shard_size();
        let start = device_id * shard_size;
        let end = start + shard_size;

        Ok((start, end))
    }

    /// Create communication plan for gradient synchronization
    ///
    /// Returns which devices need to communicate for each shard
    pub fn communication_plan(&self) -> Vec<DeviceCommunication> {
        let mut plan = Vec::new();

        for device_id in 0..self.config.num_devices {
            let mut send_to = Vec::new();
            let mut recv_from = Vec::new();

            // Need to communicate with adjacent devices for overlap
            if device_id > 0 {
                send_to.push(device_id - 1);
                recv_from.push(device_id - 1);
            }

            if device_id < self.config.num_devices - 1 {
                send_to.push(device_id + 1);
                recv_from.push(device_id + 1);
            }

            plan.push(DeviceCommunication {
                device_id,
                send_to,
                recv_from,
            });
        }

        plan
    }
}

/// Communication pattern for a device
#[derive(Debug, Clone)]
pub struct DeviceCommunication {
    /// Device ID
    pub device_id: usize,
    /// Devices this device sends data to
    pub send_to: Vec<usize>,
    /// Devices this device receives data from
    pub recv_from: Vec<usize>,
}

/// Gradient accumulator for sequence parallel training
pub struct SequenceParallelGradAccumulator {
    config: SequenceParallelConfig,
    gradients: Vec<HashMap<usize, Vec<f32>>>, // per-device gradients
}

impl SequenceParallelGradAccumulator {
    /// Create a new gradient accumulator
    pub fn new(config: SequenceParallelConfig) -> Result<Self> {
        config.validate()?;
        let num_devices = config.num_devices;
        Ok(Self {
            config,
            gradients: (0..num_devices)
                .map(|_| HashMap::new())
                .collect(),
        })
    }

    /// Add gradients from a device
    ///
    /// # Arguments
    ///
    /// * `device_id` - Device that computed these gradients
    /// * `local_gradients` - Gradients for this device's shard
    pub fn add_gradients(&mut self, device_id: usize, local_gradients: Vec<f32>) -> Result<()> {
        if device_id >= self.config.num_devices {
            return Err(Error::InvalidParameter(format!(
                "Invalid device_id {}", device_id
            )));
        }

        self.gradients[device_id].insert(device_id, local_gradients);
        Ok(())
    }

    /// Get aggregated gradients for a device
    ///
    /// # Arguments
    ///
    /// * `device_id` - Device ID
    ///
    /// # Returns
    ///
    /// Map of parameter_id to gradient vector
    pub fn get_gradients(&self, device_id: usize) -> Option<&HashMap<usize, Vec<f32>>> {
        self.gradients.get(device_id)
    }

    /// Synchronize gradients across device boundaries
    ///
    /// This handles the gradient flow for overlapping regions
    pub fn synchronize_boundaries(&mut self) -> Result<()> {
        let num_devices = self.config.num_devices;
        let overlap = self.config.overlap_size;

        if overlap == 0 {
            return Ok(());
        }

        // For each adjacent pair of devices, exchange gradient information
        // for the overlapping regions
        for device_id in 0..num_devices {
            if device_id > 0 {
                // Exchange with previous device
                self.exchange_overlap_gradients(device_id, device_id - 1)?;
            }

            if device_id < num_devices - 1 {
                // Exchange with next device
                self.exchange_overlap_gradients(device_id, device_id + 1)?;
            }
        }

        Ok(())
    }

    /// Exchange overlap gradients between two devices
    fn exchange_overlap_gradients(
        &mut self,
        device_a: usize,
        device_b: usize,
    ) -> Result<()> {
        // In a real distributed setting, this would use actual communication
        // For now, we just validate that both devices have gradient data

        if self.gradients[device_a].is_empty() || self.gradients[device_b].is_empty() {
            return Err(Error::Other(
                "Cannot exchange gradients: one or both devices missing gradients".to_string(),
            ));
        }

        // In practice, this would:
        // 1. Extract gradients for overlap region from device_a
        // 2. Extract gradients for overlap region from device_b
        // 3. Send device_a's overlap gradients to device_b
        // 4. Send device_b's overlap gradients to device_a
        // 5. Accumulate received gradients

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_parallel_config_creation() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        assert_eq!(config.num_devices, 4);
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.shard_size(), 512);
    }

    #[test]
    fn test_sequence_parallel_config_validation() {
        // Valid: seq_len divisible by num_devices
        let config = SequenceParallelConfig::new(4, 2048, 64);
        assert!(config.is_ok());

        // Invalid: not divisible
        let result = SequenceParallelConfig::new(4, 2050, 64);
        assert!(result.is_err());

        // Invalid: overlap too large
        let result = SequenceParallelConfig::new(4, 2048, 600);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_parallelism_creation() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config);
        assert!(sp.is_ok());
    }

    #[test]
    fn test_split_sequence() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        // Create a sequence [0.0, 1.0, 2.0, ..., 2047.0]
        let sequence: Vec<f32> = (0..2048).map(|i| i as f32).collect();

        let shards = sp.split_sequence(&sequence).unwrap();

        assert_eq!(shards.len(), 4);

        // Check first shard
        assert_eq!(shards[0].device_id, 0);
        assert_eq!(shards[0].global_range, (0, 512));
        assert_eq!(shards[0].data.len(), 512);
        assert_eq!(shards[0].data[0], 0.0);
        assert_eq!(shards[0].data[511], 511.0);

        // Check last shard
        assert_eq!(shards[3].device_id, 3);
        assert_eq!(shards[3].global_range, (1536, 2048));
        assert_eq!(shards[3].data[0], 1536.0);
        assert_eq!(shards[3].data[511], 2047.0);
    }

    #[test]
    fn test_merge_shards() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        let sequence: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let shards = sp.split_sequence(&sequence).unwrap();

        let merged = sp.merge_shards(&shards).unwrap();

        assert_eq!(merged.len(), 2048);
        assert_eq!(merged, sequence);
    }

    #[test]
    fn test_add_overlap() {
        let config = SequenceParallelConfig::new(4, 2048, 64).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        let sequence: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let mut shards = sp.split_sequence(&sequence).unwrap();

        sp.add_overlap(&mut shards).unwrap();

        // Check middle shards have both overlaps
        assert!(shards[1].left_overlap.is_some());
        assert!(shards[1].right_overlap.is_some());

        // First shard only has right overlap
        assert!(shards[0].left_overlap.is_none());
        assert!(shards[0].right_overlap.is_some());

        // Last shard only has left overlap
        assert!(shards[3].left_overlap.is_some());
        assert!(shards[3].right_overlap.is_none());
    }

    #[test]
    fn test_memory_savings() {
        let config = SequenceParallelConfig::new(4, 8192, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        // Standard: 8192 * 8192 * 4 bytes = 268 MB
        // Per device: 2048 * 2048 * 4 bytes = 16.8 MB
        let (per_device, standard) = sp.memory_savings(4);

        assert_eq!(standard, 8192 * 8192 * 4);
        assert_eq!(per_device, 2048 * 2048 * 4);
    }

    #[test]
    fn test_memory_saving_percentage() {
        let config = SequenceParallelConfig::new(4, 8192, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        let saving = sp.memory_saving_percentage(4);
        // (8192² - 2048²) / 8192² = 93.75%
        assert!((saving - 93.75).abs() < 0.1);
    }

    #[test]
    fn test_get_device_for_position() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        // Each shard is 512 elements
        assert_eq!(sp.get_device_for_position(0).unwrap(), 0);
        assert_eq!(sp.get_device_for_position(511).unwrap(), 0);
        assert_eq!(sp.get_device_for_position(512).unwrap(), 1);
        assert_eq!(sp.get_device_for_position(1023).unwrap(), 1);
        assert_eq!(sp.get_device_for_position(1536).unwrap(), 3);
        assert_eq!(sp.get_device_for_position(2047).unwrap(), 3);
    }

    #[test]
    fn test_get_positions_for_device() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        assert_eq!(sp.get_positions_for_device(0).unwrap(), (0, 512));
        assert_eq!(sp.get_positions_for_device(1).unwrap(), (512, 1024));
        assert_eq!(sp.get_positions_for_device(2).unwrap(), (1024, 1536));
        assert_eq!(sp.get_positions_for_device(3).unwrap(), (1536, 2048));
    }

    #[test]
    fn test_validate_shards() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        let sequence: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let shards = sp.split_sequence(&sequence).unwrap();

        assert!(sp.validate_shards(&shards).is_ok());
    }

    #[test]
    fn test_communication_plan() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let sp = SequenceParallelism::new(config).unwrap();

        let plan = sp.communication_plan();

        assert_eq!(plan.len(), 4);

        // First device only communicates with second
        assert_eq!(plan[0].device_id, 0);
        assert_eq!(plan[0].send_to, vec![1]);
        assert_eq!(plan[0].recv_from, vec![1]);

        // Middle devices communicate with both neighbors
        assert_eq!(plan[1].device_id, 1);
        assert_eq!(plan[1].send_to, vec![0, 2]);
        assert_eq!(plan[1].recv_from, vec![0, 2]);

        // Last device only communicates with third
        assert_eq!(plan[3].device_id, 3);
        assert_eq!(plan[3].send_to, vec![2]);
        assert_eq!(plan[3].recv_from, vec![2]);
    }

    #[test]
    fn test_sequence_shard() {
        let shard = SequenceShard::new(0, vec![1.0, 2.0, 3.0], (0, 3));

        assert_eq!(shard.device_id, 0);
        assert_eq!(shard.len(), 3);
        assert_eq!(shard.global_range, (0, 3));
        assert!(!shard.is_empty());
    }

    #[test]
    fn test_grad_accumulator_creation() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let accum = SequenceParallelGradAccumulator::new(config);
        assert!(accum.is_ok());
    }

    #[test]
    fn test_grad_accumulator_add_gradients() {
        let config = SequenceParallelConfig::new(4, 2048, 128).unwrap();
        let mut accum = SequenceParallelGradAccumulator::new(config).unwrap();

        let grads = vec![0.1; 512];
        assert!(accum.add_gradients(0, grads).is_ok());

        let retrieved = accum.get_gradients(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);
    }

    #[test]
    fn test_different_sequence_lengths() {
        // Test various sequence lengths and device counts
        for (num_devices, seq_len) in [(2, 1024), (4, 2048), (8, 4096)] {
            let config = SequenceParallelConfig::new(num_devices, seq_len, 64).unwrap();
            let sp = SequenceParallelism::new(config).unwrap();

            let sequence: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
            let shards = sp.split_sequence(&sequence).unwrap();
            let merged = sp.merge_shards(&shards).unwrap();

            assert_eq!(merged.len(), seq_len);
            assert_eq!(merged, sequence);
        }
    }
}
