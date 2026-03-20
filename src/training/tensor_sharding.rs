//! Tensor sharding utilities for distributed training across multiple ANEs
//!
//! Provides tools for partitioning tensors and gradients across multiple devices:
//! - `TensorShard`: Describes a shard's device assignment and data range
//! - `ShardStrategy`: Defines sharding strategy (batch, sequence, hidden, model)
//! - `TensorSharder`: Manages sharding operations across devices
//! - `ShardedMILGenerator`: Generates MIL code for sharded operations

use crate::{Error, Result};
use std::ops::Range;

/// Tensor shard descriptor
///
/// Represents a portion of a tensor assigned to a specific device.
/// Contains the device index, byte range, offset, and size information.
#[derive(Clone, Debug)]
pub struct TensorShard {
    /// Index of the device this shard is assigned to
    pub device_index: usize,
    /// Range of indices in the original tensor
    pub range: Range<usize>,
    /// Byte offset in the original tensor
    pub offset: usize,
    /// Size of this shard in elements
    pub size: usize,
}

impl TensorShard {
    /// Create a new tensor shard descriptor.
    ///
    /// # Arguments
    /// * `device_index` - Index of the device this shard is assigned to
    /// * `range` - Range of indices in the original tensor
    pub fn new(device_index: usize, range: Range<usize>) -> Self {
        let size = range.end - range.start;
        Self {
            device_index,
            offset: range.start,
            size,
            range,
        }
    }

    /// Extract shard data from a tensor.
    ///
    /// # Arguments
    /// * `tensor` - The full tensor to extract from
    pub fn extract_from(&self, tensor: &[f32]) -> Vec<f32> {
        tensor[self.range.clone()].to_vec()
    }

    /// Place shard data into a tensor at the correct offset.
    ///
    /// # Arguments
    /// * `tensor` - The full tensor to place data into
    /// * `shard_data` - The shard data to place
    ///
    /// # Errors
    /// Returns error if shard_data size doesn't match shard size
    pub fn place_into(&self, tensor: &mut [f32], shard_data: &[f32]) -> Result<()> {
        if shard_data.len() != self.size {
            return Err(Error::InvalidParameter(format!(
                "Shard data size {} doesn't match shard size {}",
                shard_data.len(),
                self.size
            )));
        }
        tensor[self.range.clone()].copy_from_slice(shard_data);
        Ok(())
    }
}

/// Tensor sharding strategy
///
/// Defines how tensors are partitioned across devices:
/// - `Batch`: Split along batch dimension
/// - `Sequence`: Split along sequence length dimension
/// - `Hidden`: Split along hidden dimension
/// - `Model`: Split by model layers (model parallelism)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Batch parallelism - split batch dimension across devices
    Batch,
    /// Sequence parallelism - split sequence length across devices
    Sequence,
    /// Hidden dimension parallelism - split hidden dimension across devices
    Hidden,
    /// Model parallelism - split model layers across devices
    Model,
}

/// Tensor sharder for distributed computation
///
/// Manages sharding operations across multiple devices.
/// Supports multiple sharding strategies via `ShardStrategy`.
pub struct TensorSharder {
    num_devices: usize,
    strategy: ShardStrategy,
}

impl TensorSharder {
    /// Create a new tensor sharder.
    ///
    /// # Arguments
    /// * `num_devices` - Number of devices to shard across
    /// * `strategy` - Sharding strategy to use
    pub fn new(num_devices: usize, strategy: ShardStrategy) -> Self {
        Self {
            num_devices,
            strategy,
        }
    }

    /// Create a batch-parallel tensor sharder.
    ///
    /// # Arguments
    /// * `num_devices` - Number of devices to shard across
    pub fn batch_parallelism(num_devices: usize) -> Self {
        Self::new(num_devices, ShardStrategy::Batch)
    }

    /// Calculate shard sizes for even distribution across devices.
    ///
    /// Handles non-divisible sizes by distributing remainder to first devices.
    ///
    /// # Arguments
    /// * `total_size` - Total size of the tensor to shard
    pub fn calculate_shard_sizes(&self, total_size: usize) -> Vec<usize> {
        let base_size = total_size / self.num_devices;
        let remainder = total_size % self.num_devices;

        (0..self.num_devices)
            .map(|i| {
                if i < remainder {
                    base_size + 1
                } else {
                    base_size
                }
            })
            .collect()
    }

    /// Shard a tensor across devices based on the configured strategy.
    ///
    /// # Arguments
    /// * `tensor` - The tensor to shard
    ///
    /// # Returns
    /// Vector of `TensorShard` descriptors, one per device
    pub fn shard_tensor(&self, tensor: &[f32]) -> Result<Vec<TensorShard>> {
        let shard_sizes = self.calculate_shard_sizes(tensor.len());
        let mut shards = Vec::new();
        let mut offset = 0;

        for (device_idx, &size) in shard_sizes.iter().enumerate() {
            let range = offset..(offset + size);
            shards.push(TensorShard::new(device_idx, range));
            offset += size;
        }

        Ok(shards)
    }

    /// Shard a batch of data across devices.
    ///
    /// # Arguments
    /// * `batch_size` - Total batch size (must be divisible by num_devices)
    /// * `seq_len` - Sequence length per sample
    /// * `hidden_dim` - Hidden dimension size
    ///
    /// # Returns
    /// Vector of (offset, per_device_batch) tuples for each device
    ///
    /// # Errors
    /// Returns error if batch_size is not divisible by num_devices
    pub fn shard_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Vec<(usize, usize)>> {
        if batch_size % self.num_devices != 0 {
            return Err(Error::InvalidParameter(format!(
                "Batch size {} must be divisible by num_devices {}",
                batch_size, self.num_devices
            )));
        }

        let per_device_batch = batch_size / self.num_devices;
        let mut shards = Vec::new();

        for i in 0..self.num_devices {
            let offset = i * per_device_batch * seq_len * hidden_dim;
            shards.push((offset, per_device_batch));
        }

        Ok(shards)
    }

    /// Gather shards from multiple devices back into a single tensor.
    ///
    /// # Arguments
    /// * `shards` - Slice of shard data from each device
    /// * `total_size` - Expected total size after gathering
    ///
    /// # Errors
    /// Returns error if number of shards doesn't match num_devices,
    /// or if shards exceed total_size
    pub fn gather_shards(&self, shards: &[Vec<f32>], total_size: usize) -> Result<Vec<f32>> {
        if shards.len() != self.num_devices {
            return Err(Error::InvalidParameter(format!(
                "Expected {} shards, got {}",
                self.num_devices,
                shards.len()
            )));
        }

        let mut result = vec![0.0f32; total_size];
        let mut offset = 0;

        for shard in shards {
            if offset + shard.len() > total_size {
                return Err(Error::InvalidParameter(
                    "Shards exceed total size".to_string(),
                ));
            }
            result[offset..(offset + shard.len())].copy_from_slice(shard);
            offset += shard.len();
        }

        Ok(result)
    }

    /// Get the number of devices.
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Get the configured sharding strategy.
    pub fn strategy(&self) -> ShardStrategy {
        self.strategy
    }
}

/// Sharded MIL code generator
///
/// Generates MIL (Metal Intermediate Language) code for sharded operations
/// across multiple ANE devices.
pub struct ShardedMILGenerator {
    num_devices: usize,
}

impl ShardedMILGenerator {
    /// Create a new sharded MIL generator.
    ///
    /// # Arguments
    /// * `num_devices` - Number of devices to generate sharded code for
    pub fn new(num_devices: usize) -> Self {
        Self { num_devices }
    }

    /// Generate MIL code for a sharded linear layer.
    ///
    /// # Arguments
    /// * `name` - Name of the layer
    /// * `input_dim` - Input dimension (currently unused in template)
    /// * `output_dim` - Output dimension (currently unused in template)
    pub fn sharded_linear_mil(&self, name: &str, _input_dim: usize, _output_dim: usize) -> String {
        format!(
            r#"//! Sharded linear layer for {name}
#!irms6
builtin linear_out = linear{{i4, o4}}(input: tensor<*xi32>, weight: tensor<*xf32>, bias: tensor<*xf32>) -> (output: tensor<*xf32>);
"#
        )
    }

    /// Generate MIL code for a sharded layer normalization.
    ///
    /// # Arguments
    /// * `name` - Name of the normalization layer
    /// * `dim` - Normalization dimension (currently unused in template)
    pub fn sharded_layer_norm_mil(&self, name: &str, _dim: usize) -> String {
        format!(
            r#"//! Sharded layer norm for {name}
#!irms6
builtin layer_norm_out = layer_norm{{i4, epsilon}}(input: tensor<*xi32>, weight: tensor<*xf32>, bias: tensor<*xf32>, epsilon: const<f32>) -> (output: tensor<*xf32>);
"#
        )
    }

    /// Generate MIL code for an all-reduce operation.
    ///
    /// # Arguments
    /// * `name` - Name of the operation
    /// * `size` - Size of the tensor to reduce
    ///
    /// # Note
    /// This is a placeholder - actual all-reduce requires inter-device communication
    pub fn all_reduce_mil(&self, name: &str, size: usize) -> String {
        format!(
            r#"//! All-reduce for {name} (placeholder)
//! Note: Actual all-reduce requires inter-device communication
#!irms6
builtin allreduce_out = identity(x: tensor<{size}xf32>) -> (reduced: tensor<{size}xf32>);
"#
        )
    }

    /// Get the number of devices.
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shard_extract() {
        let shard = TensorShard::new(0, 0..3);
        let tensor = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let extracted = shard.extract_from(&tensor);
        assert_eq!(extracted, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_shard_place() {
        let shard = TensorShard::new(0, 0..3);
        let shard_data = vec![10.0f32, 20.0, 30.0];
        let mut tensor = vec![0.0f32; 6];
        shard.place_into(&mut tensor, &shard_data).unwrap();
        assert_eq!(tensor[0..3], vec![10.0, 20.0, 30.0]);
        assert_eq!(tensor[3..6], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tensor_sharder_calculate_sizes() {
        let sharder = TensorSharder::batch_parallelism(3);
        let sizes = sharder.calculate_shard_sizes(10);
        assert_eq!(sizes, vec![4, 3, 3]);
    }

    #[test]
    fn test_tensor_sharder_shard_tensor() {
        let sharder = TensorSharder::batch_parallelism(2);
        let tensor = vec![1.0f32; 10];
        let shards = sharder.shard_tensor(&tensor).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].size, 5);
        assert_eq!(shards[1].size, 5);
        assert_eq!(shards[0].range, 0..5);
        assert_eq!(shards[1].range, 5..10);
    }

    #[test]
    fn test_tensor_sharder_shard_batch() {
        let sharder = TensorSharder::batch_parallelism(4);
        let shards = sharder.shard_batch(8, 64, 128).unwrap();
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0], (0, 2));
        assert_eq!(shards[1], (16384, 2));
        assert_eq!(shards[2], (32768, 2));
        assert_eq!(shards[3], (49152, 2));
    }

    #[test]
    fn test_tensor_sharder_shard_batch_invalid() {
        let sharder = TensorSharder::batch_parallelism(3);
        let result = sharder.shard_batch(8, 64, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_sharder_gather_shards() {
        let sharder = TensorSharder::batch_parallelism(2);
        let shard0 = vec![1.0f32, 2.0, 3.0];
        let shard1 = vec![4.0f32, 5.0, 6.0];
        let result = sharder.gather_shards(&[shard0, shard1], 6).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sharded_mil_generator() {
        let gen = ShardedMILGenerator::new(4);
        let linear_mil = gen.sharded_linear_mil("test_layer", 128, 256);
        assert!(linear_mil.contains("linear"));
        assert!(linear_mil.contains("test_layer"));
        let norm_mil = gen.sharded_layer_norm_mil("test_norm", 128);
        assert!(norm_mil.contains("layer_norm"));
        let reduce_mil = gen.all_reduce_mil("test_reduce", 1000);
        assert!(reduce_mil.contains("all-reduce"));
    }
}
