//! Model Parallelism for Large-Scale Transformers
//!
//! Model parallelism shards individual model components across multiple devices,
//! enabling training of models that don't fit on a single device.
//!
//! # Types of Model Parallelism
//!
//! - **Layer Parallelism**: Different transformer layers on different devices
//! - **Tensor Parallelism**: Individual linear layers split across devices
//! - **Pipeline Parallelism**: Sequential layer execution with micro-batches
//!
//! # Usage
//!
//! ```ignore
//! use rustane::training::model_parallel::*;
//!
//! // Create a model parallel config
//! let config = ModelParallelConfig::new(
//!     num_devices: 4,
//!     parallelism_type: ParallelismType::Layer,
//!     num_layers: 32,
//! );
//!
//! let mp = ModelParallelism::new(config)?;
//!
//! // Get which device should handle a specific layer
//! let device_id = mp.get_device_for_layer(15)?;
//!
//! // Create communication plan for layer transitions
//! let plan = mp.communication_plan()?;
//! ```

use std::collections::HashMap;

/// Error types for model parallelism operations
#[derive(Debug, Clone, PartialEq)]
pub enum ModelParallelError {
    /// Invalid configuration parameter
    InvalidConfiguration(String),
    /// Layer index out of valid range
    LayerOutOfRange {
        /// The requested layer index
        layer: usize,
        /// Total number of layers
        num_layers: usize
    },
    /// Device index out of valid range
    DeviceOutOfRange {
        /// The requested device index
        device: usize,
        /// Total number of devices
        num_devices: usize
    },
    /// Communication between devices failed
    CommunicationError(String),
    /// Shard computation failed
    ShardError(String),
}

impl std::fmt::Display for ModelParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelParallelError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            ModelParallelError::LayerOutOfRange { layer, num_layers } => {
                write!(f, "Layer {} out of range for {} layers", layer, num_layers)
            }
            ModelParallelError::DeviceOutOfRange { device, num_devices } => {
                write!(f, "Device {} out of range for {} devices", device, num_devices)
            }
            ModelParallelError::CommunicationError(msg) => {
                write!(f, "Communication failed: {}", msg)
            }
            ModelParallelError::ShardError(msg) => {
                write!(f, "Shard computation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ModelParallelError {}

/// Type of model parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParallelismType {
    /// Different layers on different devices
    Layer,

    /// Individual tensors split across devices (e.g., attention heads)
    Tensor,

    /// Pipeline parallelism with micro-batches
    Pipeline,

    /// Hybrid: layer + tensor parallelism
    Hybrid,
}

impl ParallelismType {
    /// Get the memory efficiency factor (0-1, lower is better)
    pub fn memory_efficiency(&self) -> f32 {
        match self {
            ParallelismType::Layer => 0.25, // Each device gets 1/N of layers
            ParallelismType::Tensor => 0.5,  // Tensors split, but some redundancy
            ParallelismType::Pipeline => 0.3, // Pipeline stages
            ParallelismType::Hybrid => 0.2,  // Best of both worlds
        }
    }

    /// Get the communication overhead (0-1, lower is better)
    pub fn communication_overhead(&self) -> f32 {
        match self {
            ParallelismType::Layer => 0.3,     // Inter-layer communication
            ParallelismType::Tensor => 0.5,    // Frequent all-reduce
            ParallelismType::Pipeline => 0.2,  // Only at stage boundaries
            ParallelismType::Hybrid => 0.35,   // Combined overhead
        }
    }
}

/// Configuration for model parallelism
#[derive(Debug, Clone)]
pub struct ModelParallelConfig {
    /// Number of devices to use
    pub num_devices: usize,
    /// Type of parallelism
    pub parallelism_type: ParallelismType,
    /// Total number of transformer layers
    pub num_layers: usize,
    /// For pipeline parallelism: number of micro-batches
    pub num_micro_batches: Option<usize>,
    /// For tensor parallelism: number of attention heads per device
    pub heads_per_device: Option<usize>,
}

impl ModelParallelConfig {
    /// Create a new model parallelism configuration
    pub fn new(
        num_devices: usize,
        parallelism_type: ParallelismType,
        num_layers: usize,
    ) -> Result<Self, ModelParallelError> {
        if num_devices == 0 {
            return Err(ModelParallelError::InvalidConfiguration(
                "num_devices must be > 0".to_string(),
            ));
        }
        if num_layers == 0 {
            return Err(ModelParallelError::InvalidConfiguration(
                "num_layers must be > 0".to_string(),
            ));
        }

        // Validate layer parallelism
        if parallelism_type == ParallelismType::Layer && num_layers < num_devices {
            return Err(ModelParallelError::InvalidConfiguration(
                format!("num_layers ({}) must be >= num_devices ({}) for layer parallelism",
                        num_layers, num_devices)
            ));
        }

        Ok(Self {
            num_devices,
            parallelism_type,
            num_layers,
            num_micro_batches: None,
            heads_per_device: None,
        })
    }

    /// Set number of micro-batches for pipeline parallelism
    pub fn with_micro_batches(mut self, num: usize) -> Self {
        self.num_micro_batches = Some(num);
        self
    }

    /// Set heads per device for tensor parallelism
    pub fn with_heads_per_device(mut self, num: usize) -> Self {
        self.heads_per_device = Some(num);
        self
    }
}

/// A shard of the model assigned to a device
#[derive(Debug, Clone)]
pub struct ModelShard {
    /// Device ID for this shard
    pub device_id: usize,
    /// Layer range [start, end) for this shard
    pub layer_range: (usize, usize),
    /// For tensor parallelism: which attention heads
    pub head_range: Option<(usize, usize)>,
    /// Estimated memory usage in bytes
    pub estimated_memory: usize,
}

impl ModelShard {
    /// Create a new model shard
    pub fn new(
        device_id: usize,
        layer_range: (usize, usize),
        head_range: Option<(usize, usize)>,
        estimated_memory: usize,
    ) -> Self {
        Self {
            device_id,
            layer_range,
            head_range,
            estimated_memory,
        }
    }

    /// Get the number of layers in this shard
    pub fn num_layers(&self) -> usize {
        self.layer_range.1 - self.layer_range.0
    }

    /// Check if a layer is in this shard
    pub fn contains_layer(&self, layer: usize) -> bool {
        layer >= self.layer_range.0 && layer < self.layer_range.1
    }

    /// Check if an attention head is in this shard (tensor parallelism)
    pub fn contains_head(&self, head: usize) -> bool {
        if let Some((start, end)) = self.head_range {
            head >= start && head < end
        } else {
            true // No head partitioning
        }
    }
}

/// Communication pattern between devices
#[derive(Debug, Clone)]
pub struct ModelParallelCommunication {
    /// Source device
    pub from_device: usize,
    /// Destination device
    pub to_device: usize,
    /// Communication type
    pub comm_type: CommunicationType,
    /// Data size in bytes
    pub data_size: usize,
}

/// Type of communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationType {
    /// Send activations forward
    ForwardActivation,
    /// Send gradients backward
    BackwardGradient,
    /// All-reduce for tensor parallelism
    AllReduce,
    /// Broadcast for embeddings
    Broadcast,
}

/// Model parallelism manager
pub struct ModelParallelism {
    config: ModelParallelConfig,
    shards: Vec<ModelShard>,
    layer_to_device: HashMap<usize, usize>,
    head_to_device: HashMap<usize, usize>,
}

impl ModelParallelism {
    /// Create a new model parallelism manager
    pub fn new(config: ModelParallelConfig) -> Result<Self, ModelParallelError> {
        let shards = Self::create_shards(&config)?;
        let layer_to_device = Self::map_layers_to_devices(&shards);
        let head_to_device = Self::map_heads_to_devices(&config, &shards)?;

        Ok(Self {
            config,
            shards,
            layer_to_device,
            head_to_device,
        })
    }

    /// Create model shards based on parallelism type
    fn create_shards(config: &ModelParallelConfig) -> Result<Vec<ModelShard>, ModelParallelError> {
        match config.parallelism_type {
            ParallelismType::Layer => Self::create_layer_shards(config),
            ParallelismType::Tensor => Self::create_tensor_shards(config),
            ParallelismType::Pipeline => Self::create_pipeline_shards(config),
            ParallelismType::Hybrid => Self::create_hybrid_shards(config),
        }
    }

    /// Create layer-parallel shards
    fn create_layer_shards(config: &ModelParallelConfig) -> Result<Vec<ModelShard>, ModelParallelError> {
        let num_devices = config.num_devices;
        let num_layers = config.num_layers;
        let layers_per_device = (num_layers + num_devices - 1) / num_devices;

        let mut shards = Vec::new();

        for device_id in 0..num_devices {
            let start = device_id * layers_per_device;
            let end = std::cmp::min(start + layers_per_device, num_layers);

            if start < num_layers {
                // Estimate memory: layers_per_device * model_dim^2 * 4 bytes
                let estimated_memory = layers_per_device * 1024 * 1024 * 4;

                shards.push(ModelShard::new(
                    device_id,
                    (start, end),
                    None, // No head partitioning in layer parallelism
                    estimated_memory,
                ));
            }
        }

        if shards.is_empty() {
            return Err(ModelParallelError::InvalidConfiguration(
                "Failed to create any shards".to_string(),
            ));
        }

        Ok(shards)
    }

    /// Create tensor-parallel shards
    fn create_tensor_shards(config: &ModelParallelConfig) -> Result<Vec<ModelShard>, ModelParallelError> {
        let num_devices = config.num_devices;
        let num_layers = config.num_layers;
        let heads_per_device = config.heads_per_device.unwrap_or(8);

        let mut shards = Vec::new();

        // In tensor parallelism, all devices have all layers
        // but split the attention heads and intermediate dimensions
        for device_id in 0..num_devices {
            let head_start = device_id * heads_per_device;
            let head_end = head_start + heads_per_device;

            // Each device has all layers
            let estimated_memory = num_layers * 1024 * 1024 * 4 / num_devices;

            shards.push(ModelShard::new(
                device_id,
                (0, num_layers),
                Some((head_start, head_end)),
                estimated_memory,
            ));
        }

        Ok(shards)
    }

    /// Create pipeline-parallel shards
    fn create_pipeline_shards(config: &ModelParallelConfig) -> Result<Vec<ModelShard>, ModelParallelError> {
        // Pipeline parallelism is similar to layer parallelism
        // but with micro-batch streaming
        Self::create_layer_shards(config)
    }

    /// Create hybrid parallel shards
    fn create_hybrid_shards(config: &ModelParallelConfig) -> Result<Vec<ModelShard>, ModelParallelError> {
        let num_devices = config.num_devices;
        let num_layers = config.num_layers;

        // Split devices into groups for layer parallelism
        let layer_groups = (num_devices + 1) / 2; // Half for layers
        let layers_per_group = (num_layers + layer_groups - 1) / layer_groups;

        let mut shards = Vec::new();

        for group_id in 0..layer_groups {
            let layer_start = group_id * layers_per_group;
            let layer_end = std::cmp::min(layer_start + layers_per_group, num_layers);

            // Each group has tensor parallelism within
            let devices_in_group = if group_id == layer_groups - 1 {
                num_devices - group_id * 2
            } else {
                2
            };

            for sub_id in 0..devices_in_group {
                let device_id = group_id * 2 + sub_id;

                // Heads partitioned within group
                let heads_per_device = 8;
                let head_start = sub_id * heads_per_device;
                let head_end = head_start + heads_per_device;

                let estimated_memory = (layer_end - layer_start) * 1024 * 1024 * 4 / devices_in_group;

                shards.push(ModelShard::new(
                    device_id,
                    (layer_start, layer_end),
                    Some((head_start, head_end)),
                    estimated_memory,
                ));
            }
        }

        if shards.is_empty() {
            return Err(ModelParallelError::InvalidConfiguration(
                "Failed to create hybrid shards".to_string(),
            ));
        }

        Ok(shards)
    }

    /// Map layers to devices
    fn map_layers_to_devices(shards: &[ModelShard]) -> HashMap<usize, usize> {
        let mut map = HashMap::new();

        for shard in shards {
            for layer in shard.layer_range.0..shard.layer_range.1 {
                map.insert(layer, shard.device_id);
            }
        }

        map
    }

    /// Map attention heads to devices (for tensor parallelism)
    fn map_heads_to_devices(
        config: &ModelParallelConfig,
        shards: &[ModelShard],
    ) -> Result<HashMap<usize, usize>, ModelParallelError> {
        let mut map = HashMap::new();

        match config.parallelism_type {
            ParallelismType::Tensor | ParallelismType::Hybrid => {
                let _heads_per_device = config.heads_per_device.unwrap_or(8);

                for shard in shards {
                    if let Some((start, end)) = shard.head_range {
                        for head in start..end {
                            map.insert(head, shard.device_id);
                        }
                    }
                }

                if map.is_empty() && config.parallelism_type == ParallelismType::Tensor {
                    return Err(ModelParallelError::InvalidConfiguration(
                        "Tensor parallelism requires head partitioning".to_string(),
                    ));
                }
            }
            _ => {
                // For other types, all heads are on all devices
                let total_heads = config.heads_per_device.unwrap_or(32) * config.num_devices;
                for head in 0..total_heads {
                    // Distribute heads round-robin
                    map.insert(head, head % config.num_devices);
                }
            }
        }

        Ok(map)
    }

    /// Get which device should handle a specific layer
    pub fn get_device_for_layer(&self, layer: usize) -> Result<usize, ModelParallelError> {
        if layer >= self.config.num_layers {
            return Err(ModelParallelError::LayerOutOfRange {
                layer,
                num_layers: self.config.num_layers,
            });
        }

        self.layer_to_device
            .get(&layer)
            .copied()
            .ok_or_else(|| ModelParallelError::InvalidConfiguration(format!("Layer {} not mapped", layer)))
    }

    /// Get which device should handle a specific attention head
    pub fn get_device_for_head(&self, head: usize) -> Result<usize, ModelParallelError> {
        self.head_to_device
            .get(&head)
            .copied()
            .ok_or_else(|| ModelParallelError::InvalidConfiguration(format!("Head {} not mapped", head)))
    }

    /// Get all shards
    pub fn shards(&self) -> &[ModelShard] {
        &self.shards
    }

    /// Get shard for a specific device
    pub fn get_shard_for_device(&self, device_id: usize) -> Option<&ModelShard> {
        self.shards.iter().find(|s| s.device_id == device_id)
    }

    /// Get the parallelism type
    pub fn parallelism_type(&self) -> ParallelismType {
        self.config.parallelism_type
    }

    /// Get number of devices
    pub fn num_devices(&self) -> usize {
        self.config.num_devices
    }

    /// Calculate memory efficiency (0-1, lower is better)
    pub fn memory_efficiency(&self) -> f32 {
        self.config.parallelism_type.memory_efficiency()
    }

    /// Calculate communication overhead (0-1, lower is better)
    pub fn communication_overhead(&self) -> f32 {
        self.config.parallelism_type.communication_overhead()
    }

    /// Generate communication plan for forward/backward passes
    pub fn communication_plan(&self) -> Result<Vec<ModelParallelCommunication>, ModelParallelError> {
        let mut plan = Vec::new();

        match self.config.parallelism_type {
            ParallelismType::Layer => {
                // Communication at layer boundaries
                for i in 0..self.shards.len() - 1 {
                    let from_device = self.shards[i].device_id;
                    let to_device = self.shards[i + 1].device_id;

                    // Forward activation
                    plan.push(ModelParallelCommunication {
                        from_device,
                        to_device,
                        comm_type: CommunicationType::ForwardActivation,
                        data_size: 1024 * 1024 * 4, // Estimated
                    });

                    // Backward gradient
                    plan.push(ModelParallelCommunication {
                        from_device: to_device,
                        to_device: from_device,
                        comm_type: CommunicationType::BackwardGradient,
                        data_size: 1024 * 1024 * 4,
                    });
                }
            }
            ParallelismType::Tensor => {
                // All-reduce for each layer
                for _layer in 0..self.config.num_layers {
                    for i in 0..self.config.num_devices {
                        for j in (i + 1)..self.config.num_devices {
                            plan.push(ModelParallelCommunication {
                                from_device: i,
                                to_device: j,
                                comm_type: CommunicationType::AllReduce,
                                data_size: 1024 * 1024 * 4 / self.config.num_devices,
                            });
                        }
                    }
                }
            }
            ParallelismType::Pipeline => {
                // Similar to layer but with micro-batches
                for i in 0..self.shards.len() - 1 {
                    plan.push(ModelParallelCommunication {
                        from_device: self.shards[i].device_id,
                        to_device: self.shards[i + 1].device_id,
                        comm_type: CommunicationType::ForwardActivation,
                        data_size: 1024 * 1024 * 4,
                    });
                }
            }
            ParallelismType::Hybrid => {
                // Combined layer + tensor communication
                // Layer boundaries
                for i in 0..self.shards.len() - 1 {
                    let current_device = self.shards[i].device_id;
                    let next_device = self.shards[i + 1].device_id;

                    if current_device != next_device {
                        plan.push(ModelParallelCommunication {
                            from_device: current_device,
                            to_device: next_device,
                            comm_type: CommunicationType::ForwardActivation,
                            data_size: 1024 * 1024 * 2,
                        });
                    }
                }

                // All-reduce within groups for tensor parallelism
                let layer_groups = (self.config.num_devices + 1) / 2;
                for group in 0..layer_groups {
                    let group_start = group * 2;
                    let group_end = std::cmp::min(group_start + 2, self.config.num_devices);

                    if group_end - group_start > 1 {
                        for i in group_start..group_end {
                            for j in (i + 1)..group_end {
                                plan.push(ModelParallelCommunication {
                                    from_device: i,
                                    to_device: j,
                                    comm_type: CommunicationType::AllReduce,
                                    data_size: 1024 * 1024 * 2,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(plan)
    }

    /// Calculate total memory savings
    pub fn memory_savings(&self) -> (usize, usize) {
        // Original: all layers on one device
        let original_memory = self.config.num_layers * 1024 * 1024 * 4;

        // With parallelism: max memory across devices
        let max_shard_memory = self
            .shards
            .iter()
            .map(|s| s.estimated_memory)
            .max()
            .unwrap_or(0);

        (max_shard_memory, original_memory)
    }

    /// Calculate memory saving percentage
    pub fn memory_saving_percentage(&self) -> f32 {
        let (per_device, original) = self.memory_savings();
        if original == 0 {
            return 0.0;
        }
        ((original - per_device) as f32 / original as f32) * 100.0
    }
}

/// Gradient accumulator for model parallel training
pub struct ModelParallelGradAccumulator {
    num_devices: usize,
    shards: Vec<Vec<f32>>,
}

impl ModelParallelGradAccumulator {
    /// Create a new gradient accumulator
    pub fn new(num_devices: usize, shard_sizes: &[usize]) -> Self {
        let shards = shard_sizes
            .iter()
            .map(|&size| vec![0.0; size])
            .collect();

        Self {
            num_devices,
            shards,
        }
    }

    /// Add gradients for a specific shard
    pub fn add_shard_gradients(&mut self, device_id: usize, grads: &[f32]) -> Result<(), ModelParallelError> {
        if device_id >= self.num_devices {
            return Err(ModelParallelError::DeviceOutOfRange {
                device: device_id,
                num_devices: self.num_devices,
            });
        }

        if grads.len() != self.shards[device_id].len() {
            return Err(ModelParallelError::ShardError(format!(
                "Gradient size mismatch: expected {}, got {}",
                self.shards[device_id].len(),
                grads.len()
            )));
        }

        for (i, &grad) in grads.iter().enumerate() {
            self.shards[device_id][i] += grad;
        }

        Ok(())
    }

    /// Get accumulated gradients for a shard
    pub fn get_shard_gradients(&self, device_id: usize) -> Result<&[f32], ModelParallelError> {
        if device_id >= self.num_devices {
            return Err(ModelParallelError::DeviceOutOfRange {
                device: device_id,
                num_devices: self.num_devices,
            });
        }

        Ok(&self.shards[device_id])
    }

    /// Reset all accumulated gradients
    pub fn reset(&mut self) {
        for shard in &mut self.shards {
            shard.fill(0.0);
        }
    }

    /// Perform all-reduce across tensor parallel shards
    pub fn all_reduce(&mut self, devices: &[usize]) -> Result<(), ModelParallelError> {
        if devices.is_empty() {
            return Ok(());
        }

        // Validate devices
        for &device_id in devices {
            if device_id >= self.num_devices {
                return Err(ModelParallelError::DeviceOutOfRange {
                    device: device_id,
                    num_devices: self.num_devices,
                });
            }
        }

        // Average gradients across devices
        let num = devices.len() as f32;
        let len = self.shards[devices[0]].len();

        for i in 0..len {
            let sum: f32 = devices.iter().map(|&d| self.shards[d][i]).sum();
            let avg = sum / num;

            for &d in devices {
                self.shards[d][i] = avg;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_parallel_config() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        assert_eq!(config.num_devices, 4);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_invalid_config() {
        let result = ModelParallelConfig::new(0, ParallelismType::Layer, 32);
        assert!(result.is_err());

        let result = ModelParallelConfig::new(4, ParallelismType::Layer, 0);
        assert!(result.is_err());

        let result = ModelParallelConfig::new(4, ParallelismType::Layer, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_parallel_shards() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        assert_eq!(mp.num_devices(), 4);
        assert_eq!(mp.shards().len(), 4);

        // Each device should have 8 layers (32 / 4)
        for shard in mp.shards() {
            assert_eq!(shard.num_layers(), 8);
        }
    }

    #[test]
    fn test_layer_to_device_mapping() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        // First 8 layers on device 0
        assert_eq!(mp.get_device_for_layer(0).unwrap(), 0);
        assert_eq!(mp.get_device_for_layer(7).unwrap(), 0);

        // Next 8 layers on device 1
        assert_eq!(mp.get_device_for_layer(8).unwrap(), 1);
        assert_eq!(mp.get_device_for_layer(15).unwrap(), 1);

        // Last 8 layers on device 3
        assert_eq!(mp.get_device_for_layer(24).unwrap(), 3);
        assert_eq!(mp.get_device_for_layer(31).unwrap(), 3);
    }

    #[test]
    fn test_tensor_parallel_shards() {
        let config = ModelParallelConfig::new(4, ParallelismType::Tensor, 32)
            .unwrap()
            .with_heads_per_device(8);
        let mp = ModelParallelism::new(config).unwrap();

        assert_eq!(mp.num_devices(), 4);
        assert_eq!(mp.shards().len(), 4);

        // All devices should have all layers in tensor parallelism
        for shard in mp.shards() {
            assert_eq!(shard.layer_range, (0, 32));
            assert!(shard.head_range.is_some());
        }
    }

    #[test]
    fn test_head_to_device_mapping() {
        let config = ModelParallelConfig::new(4, ParallelismType::Tensor, 32)
            .unwrap()
            .with_heads_per_device(8);
        let mp = ModelParallelism::new(config).unwrap();

        // First 8 heads on device 0
        assert_eq!(mp.get_device_for_head(0).unwrap(), 0);
        assert_eq!(mp.get_device_for_head(7).unwrap(), 0);

        // Next 8 heads on device 1
        assert_eq!(mp.get_device_for_head(8).unwrap(), 1);
        assert_eq!(mp.get_device_for_head(15).unwrap(), 1);
    }

    #[test]
    fn test_shard_contains_layer() {
        let shard = ModelShard::new(0, (8, 16), None, 1024);

        assert!(!shard.contains_layer(7));
        assert!(shard.contains_layer(8));
        assert!(shard.contains_layer(15));
        assert!(!shard.contains_layer(16));
    }

    #[test]
    fn test_shard_contains_head() {
        let shard_with_heads = ModelShard::new(0, (0, 32), Some((8, 16)), 1024);

        assert!(!shard_with_heads.contains_head(7));
        assert!(shard_with_heads.contains_head(8));
        assert!(shard_with_heads.contains_head(15));
        assert!(!shard_with_heads.contains_head(16));

        let shard_no_heads = ModelShard::new(0, (0, 32), None, 1024);
        assert!(shard_no_heads.contains_head(100)); // No partitioning
    }

    #[test]
    fn test_memory_savings() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        let (per_device, original) = mp.memory_savings();

        // Per device should have 1/4 of layers
        assert!(per_device < original);

        // Should be roughly 4x savings
        let ratio = original as f32 / per_device as f32;
        assert!(ratio >= 3.5 && ratio <= 4.5);
    }

    #[test]
    fn test_memory_saving_percentage() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        let savings = mp.memory_saving_percentage();

        // Should save around 75%
        assert!(savings > 70.0 && savings < 80.0);
    }

    #[test]
    fn test_communication_plan_layer_parallel() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        let plan = mp.communication_plan().unwrap();

        // Should have communication between adjacent devices
        // 4 devices means 3 boundaries, each with forward + backward
        assert_eq!(plan.len(), 6);

        // Check forward activations
        let forward_comms: Vec<_> = plan
            .iter()
            .filter(|c| c.comm_type == CommunicationType::ForwardActivation)
            .collect();

        assert_eq!(forward_comms.len(), 3);
    }

    #[test]
    fn test_communication_plan_tensor_parallel() {
        let config = ModelParallelConfig::new(2, ParallelismType::Tensor, 8)
            .unwrap()
            .with_heads_per_device(4);
        let mp = ModelParallelism::new(config).unwrap();

        let plan = mp.communication_plan().unwrap();

        // Should have all-reduce communications
        let all_reduces: Vec<_> = plan
            .iter()
            .filter(|c| c.comm_type == CommunicationType::AllReduce)
            .collect();

        assert!(!all_reduces.is_empty());
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut accum = ModelParallelGradAccumulator::new(4, &[100, 100, 100, 100]);

        // Add gradients to device 0
        accum.add_shard_gradients(0, &[1.0; 100]).unwrap();

        // Verify
        let grads = accum.get_shard_gradients(0).unwrap();
        assert_eq!(grads.len(), 100);
        assert_eq!(grads[0], 1.0);

        // Reset
        accum.reset();
        let grads = accum.get_shard_gradients(0).unwrap();
        assert_eq!(grads[0], 0.0);
    }

    #[test]
    fn test_gradient_accumulator_invalid_device() {
        let mut accum = ModelParallelGradAccumulator::new(4, &[100, 100, 100, 100]);

        let result = accum.add_shard_gradients(10, &[1.0; 100]);
        assert!(result.is_err());

        let result = accum.get_shard_gradients(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_reduce() {
        let mut accum = ModelParallelGradAccumulator::new(4, &[10, 10, 10, 10]);

        // Add different gradients to each device
        accum.add_shard_gradients(0, &[1.0; 10]).unwrap();
        accum.add_shard_gradients(1, &[3.0; 10]).unwrap();
        accum.add_shard_gradients(2, &[5.0; 10]).unwrap();
        accum.add_shard_gradients(3, &[7.0; 10]).unwrap();

        // All-reduce should average: (1+3+5+7)/4 = 4
        accum.all_reduce(&[0, 1, 2, 3]).unwrap();

        let grads0 = accum.get_shard_gradients(0).unwrap();
        let grads1 = accum.get_shard_gradients(1).unwrap();

        assert!((grads0[0] - 4.0).abs() < 0.001);
        assert!((grads1[0] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_parallel_shards() {
        let config = ModelParallelConfig::new(4, ParallelismType::Pipeline, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        assert_eq!(mp.num_devices(), 4);
        assert_eq!(mp.shards().len(), 4);

        // Pipeline parallelism should distribute layers similar to layer parallelism
        let total_layers: usize = mp.shards().iter().map(|s| s.num_layers()).sum();
        assert_eq!(total_layers, 32);
    }

    #[test]
    fn test_hybrid_parallel_shards() {
        let config = ModelParallelConfig::new(4, ParallelismType::Hybrid, 32)
            .unwrap()
            .with_heads_per_device(4);
        let mp = ModelParallelism::new(config).unwrap();

        assert_eq!(mp.num_devices(), 4);

        // Hybrid should combine layer and head partitioning
        let has_head_partitioning = mp.shards().iter().any(|s| s.head_range.is_some());
        assert!(has_head_partitioning);
    }

    #[test]
    fn test_parallelism_type_properties() {
        assert_eq!(ParallelismType::Layer.memory_efficiency(), 0.25);
        assert_eq!(ParallelismType::Tensor.memory_efficiency(), 0.5);
        assert_eq!(ParallelismType::Pipeline.memory_efficiency(), 0.3);
        assert_eq!(ParallelismType::Hybrid.memory_efficiency(), 0.2);

        assert_eq!(ParallelismType::Layer.communication_overhead(), 0.3);
        assert_eq!(ParallelismType::Tensor.communication_overhead(), 0.5);
        assert_eq!(ParallelismType::Pipeline.communication_overhead(), 0.2);
        assert_eq!(ParallelismType::Hybrid.communication_overhead(), 0.35);
    }

    #[test]
    fn test_get_shard_for_device() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        let shard = mp.get_shard_for_device(0);
        assert!(shard.is_some());
        assert_eq!(shard.unwrap().device_id, 0);

        let shard = mp.get_shard_for_device(10);
        assert!(shard.is_none());
    }

    #[test]
    fn test_layer_out_of_range() {
        let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        let result = mp.get_device_for_layer(100);
        assert!(result.is_err());
    }
}
