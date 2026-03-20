//! Large Model Support (7B+ Parameters)
//!
//! Provides utilities and optimizations for training very large transformer models
//! (7B+ parameters) that don't fit in memory on a single device.
//!
//! # Features
//!
//! - **Memory-efficient initialization**: Initialize layers progressively
//! - **Parameter sharding**: Split model parameters across multiple devices
//! - **Gradient checkpointing**: Reduce activation memory with recomputation
//! - **Mixed precision training**: Use FP16/BF16 to reduce memory by 50%
//! - **Model parallelism**: Distribute layers across devices
//!
//! # Usage
//!
//! ```ignore
//! use rustane::training::large_models::*;
//!
//! // Create a large model configuration
//! let config = LargeModelConfig::new(
//!     num_layers: 32,
//!     hidden_dim: 4096,
//!     num_heads: 32,
//!     intermediate_dim: 16384,
//! );
//!
//! // Calculate memory requirements
//! let memory = config.calculate_memory_requirements();
//!
//! // Create a memory-optimized initializer
//! let initializer = LargeModelInitializer::new(config);
//! ```

use std::collections::HashMap;

/// Error types for large model operations
#[derive(Debug, Clone, PartialEq)]
pub enum LargeModelError {
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Insufficient memory
    InsufficientMemory { required: usize, available: usize },
    /// Layer initialization failed
    InitializationFailed(String),
    /// Parameter sharding error
    ShardingError(String),
}

impl std::fmt::Display for LargeModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LargeModelError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            LargeModelError::InsufficientMemory {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient memory: required {} MB, available {} MB",
                    required / 1024 / 1024,
                    available / 1024 / 1024
                )
            }
            LargeModelError::InitializationFailed(msg) => {
                write!(f, "Initialization failed: {}", msg)
            }
            LargeModelError::ShardingError(msg) => {
                write!(f, "Sharding error: {}", msg)
            }
        }
    }
}

impl std::error::Error for LargeModelError {}

/// Configuration for large models
#[derive(Debug, Clone, PartialEq)]
pub struct LargeModelConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension (model width)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Intermediate dimension for FFN (usually 4x hidden_dim)
    pub intermediate_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Use mixed precision (FP16/BF16)
    pub use_mixed_precision: bool,
    /// Use gradient checkpointing
    pub use_gradient_checkpointing: bool,
    /// Number of devices for model parallelism
    pub num_devices: usize,
}

impl LargeModelConfig {
    /// Create a new large model configuration
    pub fn new(
        num_layers: usize,
        hidden_dim: usize,
        num_heads: usize,
        intermediate_dim: usize,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be > 0");
        assert!(hidden_dim > 0, "hidden_dim must be > 0");
        assert!(num_heads > 0, "num_heads must be > 0");
        assert!(intermediate_dim > 0, "intermediate_dim must be > 0");
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );

        Self {
            num_layers,
            hidden_dim,
            num_heads,
            intermediate_dim,
            vocab_size: 50272, // Default vocab size
            max_seq_len: 2048, // Default sequence length
            use_mixed_precision: true,
            use_gradient_checkpointing: true,
            num_devices: 1,
        }
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Enable/disable mixed precision
    pub fn with_mixed_precision(mut self, use_mixed_precision: bool) -> Self {
        self.use_mixed_precision = use_mixed_precision;
        self
    }

    /// Enable/disable gradient checkpointing
    pub fn with_gradient_checkpointing(mut self, use_gradient_checkpointing: bool) -> Self {
        self.use_gradient_checkpointing = use_gradient_checkpointing;
        self
    }

    /// Set number of devices for model parallelism
    pub fn with_num_devices(mut self, num_devices: usize) -> Self {
        self.num_devices = num_devices;
        self
    }

    /// Calculate parameter count for this model
    pub fn calculate_parameter_count(&self) -> usize {
        // Embedding layer: vocab_size * hidden_dim
        let embedding_params = self.vocab_size * self.hidden_dim;

        // Per transformer layer:
        // - Attention: 4 * hidden_dim * hidden_dim (Q, K, V, O projections)
        // - FFN: 2 * hidden_dim * intermediate_dim (up, down projections)
        // - Layer norms: 2 * hidden_dim (2 layer norms with 2 * hidden_dim params each)
        let layer_params = 4 * self.hidden_dim * self.hidden_dim
            + 2 * self.hidden_dim * self.intermediate_dim
            + 4 * self.hidden_dim;

        let transformer_params = layer_params * self.num_layers;

        // Final layer norm and output projection
        let final_params = self.hidden_dim + self.vocab_size * self.hidden_dim;

        embedding_params + transformer_params + final_params
    }

    /// Calculate memory requirements in bytes
    pub fn calculate_memory_requirements(&self) -> LargeModelMemory {
        let param_count = self.calculate_parameter_count();

        // Parameter memory (FP32 = 4 bytes per param)
        let bytes_per_param = if self.use_mixed_precision { 2 } else { 4 };
        let param_memory = param_count * bytes_per_param;

        // Optimizer state (Adam: 2x parameters for m and v)
        let optimizer_memory = param_memory * 2;

        // Activation memory per layer
        // activations: batch_size * seq_len * hidden_dim * 4 bytes
        let batch_size = 1;
        let activation_memory_per_layer = batch_size * self.max_seq_len * self.hidden_dim * 4;
        let total_activation_memory = activation_memory_per_layer * self.num_layers;

        // With gradient checkpointing, we only keep a fraction of activations
        let effective_activation_memory = if self.use_gradient_checkpointing {
            total_activation_memory / 4 // Keep ~25% with checkpointing
        } else {
            total_activation_memory
        };

        // Gradient memory (same as parameter memory)
        let gradient_memory = param_memory;

        // Total memory
        let total_memory =
            param_memory + optimizer_memory + effective_activation_memory + gradient_memory;

        LargeModelMemory {
            parameter_count: param_count,
            parameter_memory_mb: param_memory / 1024 / 1024,
            optimizer_memory_mb: optimizer_memory / 1024 / 1024,
            activation_memory_mb: effective_activation_memory / 1024 / 1024,
            gradient_memory_mb: gradient_memory / 1024 / 1024,
            total_memory_mb: total_memory / 1024 / 1024,
        }
    }

    /// Get model size category
    pub fn size_category(&self) -> ModelSizeCategory {
        let params_billions = self.calculate_parameter_count() as f64 / 1e9;

        match params_billions {
            x if x < 1.0 => ModelSizeCategory::Small,
            x if x < 7.0 => ModelSizeCategory::Medium,
            x if x < 13.0 => ModelSizeCategory::Large,
            x if x < 70.0 => ModelSizeCategory::XL,
            _ => ModelSizeCategory::XXL,
        }
    }
}

/// Memory breakdown for a large model
#[derive(Debug, Clone)]
pub struct LargeModelMemory {
    /// Total number of parameters
    pub parameter_count: usize,
    /// Memory for parameters (MB)
    pub parameter_memory_mb: usize,
    /// Memory for optimizer state (MB)
    pub optimizer_memory_mb: usize,
    /// Memory for activations (MB)
    pub activation_memory_mb: usize,
    /// Memory for gradients (MB)
    pub gradient_memory_mb: usize,
    /// Total memory required (MB)
    pub total_memory_mb: usize,
}

impl LargeModelMemory {
    /// Check if this model fits in available memory
    pub fn fits_in_memory(&self, available_mb: usize) -> bool {
        self.total_memory_mb <= available_mb
    }

    /// Get memory breakdown as percentages
    pub fn breakdown_percentages(&self) -> (f32, f32, f32, f32) {
        let total = self.total_memory_mb as f32;

        (
            (self.parameter_memory_mb as f32 / total) * 100.0,
            (self.optimizer_memory_mb as f32 / total) * 100.0,
            (self.activation_memory_mb as f32 / total) * 100.0,
            (self.gradient_memory_mb as f32 / total) * 100.0,
        )
    }
}

/// Model size categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSizeCategory {
    /// < 1B parameters
    Small,
    /// 1B - 7B parameters
    Medium,
    /// 7B - 13B parameters
    Large,
    /// 13B - 70B parameters
    XL,
    /// > 70B parameters
    XXL,
}

impl ModelSizeCategory {
    /// Get description of this size category
    pub fn description(&self) -> &'static str {
        match self {
            ModelSizeCategory::Small => "< 1B parameters (fits in consumer GPU)",
            ModelSizeCategory::Medium => "1B - 7B parameters (requires optimization)",
            ModelSizeCategory::Large => "7B - 13B parameters (requires multi-GPU/ANE)",
            ModelSizeCategory::XL => "13B - 70B parameters (requires distributed training)",
            ModelSizeCategory::XXL => "> 70B parameters (requires extensive parallelism)",
        }
    }

    /// Get recommended techniques for this size
    pub fn recommended_techniques(&self) -> Vec<&'static str> {
        match self {
            ModelSizeCategory::Small => vec!["Standard training"],
            ModelSizeCategory::Medium => vec![
                "Gradient checkpointing",
                "Mixed precision training",
                "Gradient accumulation",
            ],
            ModelSizeCategory::Large => vec![
                "Gradient checkpointing",
                "Mixed precision training",
                "Gradient accumulation",
                "Model parallelism",
                "Sequence parallelism",
            ],
            ModelSizeCategory::XL => vec![
                "Gradient checkpointing",
                "Mixed precision training",
                "Gradient accumulation",
                "Model parallelism",
                "Sequence parallelism",
                "Pipeline parallelism",
                "Tensor parallelism",
            ],
            ModelSizeCategory::XXL => vec![
                "All optimization techniques",
                "Distributed data parallel",
                "Hybrid parallelism strategies",
                "ZeRO optimization",
            ],
        }
    }
}

/// Initialization strategy for large models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializationStrategy {
    /// Initialize all layers at once (requires lots of memory)
    AllAtOnce,
    /// Initialize layer by layer (memory efficient)
    LayerByLayer,
    /// Initialize in chunks (balance memory and speed)
    Chunked { chunk_size: usize },
}

/// Large model initializer
pub struct LargeModelInitializer {
    config: LargeModelConfig,
    strategy: InitializationStrategy,
}

impl LargeModelInitializer {
    /// Create a new large model initializer
    pub fn new(config: LargeModelConfig) -> Self {
        Self {
            config,
            strategy: InitializationStrategy::LayerByLayer,
        }
    }

    /// Set initialization strategy
    pub fn with_strategy(mut self, strategy: InitializationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Calculate initialization memory requirements
    pub fn init_memory_requirements(&self) -> usize {
        match self.strategy {
            InitializationStrategy::AllAtOnce => {
                // Need memory for all parameters
                self.config.calculate_parameter_count() * 4
            }
            InitializationStrategy::LayerByLayer => {
                // Need memory for one layer at a time
                let layer_params = 4 * self.config.hidden_dim * self.config.hidden_dim
                    + 2 * self.config.hidden_dim * self.config.intermediate_dim;
                layer_params * 4
            }
            InitializationStrategy::Chunked { chunk_size } => {
                // Need memory for chunk_size layers
                let layer_params = 4 * self.config.hidden_dim * self.config.hidden_dim
                    + 2 * self.config.hidden_dim * self.config.intermediate_dim;
                layer_params * chunk_size * 4
            }
        }
    }

    /// Initialize a single layer
    pub fn initialize_layer(&self, layer_idx: usize) -> Result<Vec<f32>, LargeModelError> {
        if layer_idx >= self.config.num_layers {
            return Err(LargeModelError::InitializationFailed(format!(
                "Layer index {} out of range for {} layers",
                layer_idx, self.config.num_layers
            )));
        }

        // Simulated layer initialization
        // In practice, this would initialize:
        // - Q, K, V, O projections for attention
        // - Up, down projections for FFN
        // - Layer norm parameters
        let layer_params = 4 * self.config.hidden_dim * self.config.hidden_dim
            + 2 * self.config.hidden_dim * self.config.intermediate_dim
            + 4 * self.config.hidden_dim;

        // Use Xavier initialization
        let std = (2.0 / (self.config.hidden_dim + self.config.intermediate_dim) as f32).sqrt();
        let mut params = Vec::with_capacity(layer_params);

        for _ in 0..layer_params {
            params.push(rand::random::<f32>() * std);
        }

        Ok(params)
    }

    /// Initialize embedding layer
    pub fn initialize_embedding(&self) -> Result<Vec<f32>, LargeModelError> {
        let embedding_params = self.config.vocab_size * self.config.hidden_dim;
        let mut params = Vec::with_capacity(embedding_params);

        // Normal initialization for embeddings
        let std = 1.0 / (self.config.hidden_dim as f32).sqrt();
        for _ in 0..embedding_params {
            params.push(rand::random::<f32>() * std);
        }

        Ok(params)
    }

    /// Initialize all layers progressively
    pub fn initialize_progressively(&self) -> Result<ProgressiveInitialization, LargeModelError> {
        let memory = self.config.calculate_memory_requirements();
        let init_memory = self.init_memory_requirements();

        Ok(ProgressiveInitialization {
            config: self.config.clone(),
            current_layer: 0,
            total_layers: self.config.num_layers,
            params_initialized: 0,
            total_params: self.config.calculate_parameter_count(),
            init_memory_mb: init_memory / 1024 / 1024,
            total_memory_mb: memory.total_memory_mb,
        })
    }
}

/// Progressive initialization state
#[derive(Debug, Clone)]
pub struct ProgressiveInitialization {
    config: LargeModelConfig,
    current_layer: usize,
    total_layers: usize,
    params_initialized: usize,
    total_params: usize,
    init_memory_mb: usize,
    total_memory_mb: usize,
}

impl ProgressiveInitialization {
    /// Get initialization progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        self.params_initialized as f32 / self.total_params as f32
    }

    /// Get percentage complete
    pub fn percentage(&self) -> f32 {
        self.progress() * 100.0
    }

    /// Check if initialization is complete
    pub fn is_complete(&self) -> bool {
        self.current_layer >= self.total_layers
    }

    /// Get memory efficiency (0.0 to 1.0, higher is better)
    pub fn memory_efficiency(&self) -> f32 {
        1.0 - (self.init_memory_mb as f32 / self.total_memory_mb as f32)
    }

    /// Get current layer index
    pub fn current_layer(&self) -> usize {
        self.current_layer
    }

    /// Get total number of layers
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Get total parameter count
    pub fn total_params(&self) -> usize {
        self.total_params
    }

    /// Get initialization memory in MB
    pub fn init_memory_mb(&self) -> usize {
        self.init_memory_mb
    }

    /// Get total memory in MB
    pub fn total_memory_mb(&self) -> usize {
        self.total_memory_mb
    }
}

/// Parameter sharding strategy
#[derive(Debug, Clone)]
pub struct ParameterSharding {
    /// Number of shards
    pub num_shards: usize,
    /// Which parameters go to which shard
    shard_assignment: HashMap<String, usize>,
}

impl ParameterSharding {
    /// Create a new parameter sharding plan
    pub fn new(num_shards: usize, config: &LargeModelConfig) -> Self {
        let mut shard_assignment = HashMap::new();

        // Shard embedding
        for i in 0..config.vocab_size {
            let shard = i % num_shards;
            shard_assignment.insert(format!("embedding.{}", i), shard);
        }

        // Shard each layer
        for layer in 0..config.num_layers {
            let layer_shard = layer % num_shards;

            // Attention weights
            shard_assignment.insert(format!("layer_{}.attn.q", layer), layer_shard);
            shard_assignment.insert(format!("layer_{}.attn.k", layer), layer_shard);
            shard_assignment.insert(format!("layer_{}.attn.v", layer), layer_shard);
            shard_assignment.insert(format!("layer_{}.attn.o", layer), layer_shard);

            // FFN weights
            shard_assignment.insert(format!("layer_{}.ffn.up", layer), layer_shard);
            shard_assignment.insert(format!("layer_{}.ffn.down", layer), layer_shard);
        }

        Self {
            num_shards,
            shard_assignment,
        }
    }

    /// Get which shard a parameter belongs to
    pub fn get_shard(&self, param_name: &str) -> Option<usize> {
        self.shard_assignment.get(param_name).copied()
    }

    /// Get all parameters for a specific shard
    pub fn get_shard_params(&self, shard_idx: usize) -> Vec<String> {
        self.shard_assignment
            .iter()
            .filter(|(_, &shard)| shard == shard_idx)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Recommended configuration presets for common model sizes
pub struct ModelPresets;

impl ModelPresets {
    /// 7B parameter model (e.g., LLaMA-7B, Pythia-6.9B)
    pub fn model_7b() -> LargeModelConfig {
        LargeModelConfig::new(32, 4096, 32, 16384)
            .with_vocab_size(50272)
            .with_max_seq_len(2048)
            .with_mixed_precision(true)
            .with_gradient_checkpointing(true)
    }

    /// 13B parameter model (e.g., LLaMA-13B)
    pub fn model_13b() -> LargeModelConfig {
        LargeModelConfig::new(40, 5120, 40, 20480)
            .with_vocab_size(50272)
            .with_max_seq_len(2048)
            .with_mixed_precision(true)
            .with_gradient_checkpointing(true)
            .with_num_devices(2)
    }

    /// 30B parameter model
    pub fn model_30b() -> LargeModelConfig {
        LargeModelConfig::new(60, 6656, 52, 26624)
            .with_vocab_size(50272)
            .with_max_seq_len(2048)
            .with_mixed_precision(true)
            .with_gradient_checkpointing(true)
            .with_num_devices(4)
    }

    /// 70B parameter model (e.g., LLaMA-70B)
    pub fn model_70b() -> LargeModelConfig {
        LargeModelConfig::new(80, 8192, 64, 32768)
            .with_vocab_size(50272)
            .with_max_seq_len(2048)
            .with_mixed_precision(true)
            .with_gradient_checkpointing(true)
            .with_num_devices(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_model_config_creation() {
        let config = LargeModelConfig::new(32, 4096, 32, 16384);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.intermediate_dim, 16384);
    }

    #[test]
    fn test_parameter_count_7b() {
        let config = ModelPresets::model_7b();
        let params = config.calculate_parameter_count();

        // Should be approximately 7B parameters
        let params_billions = params as f64 / 1e9;
        assert!(params_billions >= 6.5 && params_billions <= 7.5);
    }

    #[test]
    fn test_parameter_count_13b() {
        let config = ModelPresets::model_13b();
        let params = config.calculate_parameter_count();

        let params_billions = params as f64 / 1e9;
        assert!(params_billions >= 12.5 && params_billions <= 13.5);
    }

    #[test]
    fn test_size_categories() {
        let small = LargeModelConfig::new(6, 768, 12, 3072);
        assert_eq!(small.size_category(), ModelSizeCategory::Small);

        // ModelPresets::model_7b() is actually 6.85B, which is Medium (1B-7B range)
        let medium = ModelPresets::model_7b();
        assert_eq!(medium.size_category(), ModelSizeCategory::Medium);

        // ModelPresets::model_13b() is 13.10B, which is XL (13B-70B range)
        let xl = ModelPresets::model_13b();
        assert_eq!(xl.size_category(), ModelSizeCategory::XL);

        // ModelPresets::model_30b() should also be XL
        let xl2 = ModelPresets::model_30b();
        assert_eq!(xl2.size_category(), ModelSizeCategory::XL);
    }

    #[test]
    fn test_memory_requirements() {
        let config = ModelPresets::model_7b();
        let memory = config.calculate_memory_requirements();

        assert!(memory.parameter_memory_mb > 0);
        assert!(memory.optimizer_memory_mb > 0);
        assert!(memory.activation_memory_mb > 0);
        assert!(memory.total_memory_mb > 0);

        // Optimizer memory should be approximately 2x parameter memory (Adam)
        // Allow small rounding differences
        let ratio = memory.optimizer_memory_mb as f64 / memory.parameter_memory_mb as f64;
        assert!((ratio - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_mixed_precision_savings() {
        let config_fp32 = ModelPresets::model_7b().with_mixed_precision(false);
        let config_fp16 = ModelPresets::model_7b().with_mixed_precision(true);

        let memory_fp32 = config_fp32.calculate_memory_requirements();
        let memory_fp16 = config_fp16.calculate_memory_requirements();

        // FP16 should use roughly half the parameter memory
        assert!(memory_fp16.parameter_memory_mb < memory_fp32.parameter_memory_mb);
        assert!(memory_fp16.parameter_memory_mb >= memory_fp32.parameter_memory_mb / 2);
    }

    #[test]
    fn test_gradient_checkpointing_savings() {
        let config_no_ckpt = ModelPresets::model_7b().with_gradient_checkpointing(false);
        let config_ckpt = ModelPresets::model_7b().with_gradient_checkpointing(true);

        let memory_no_ckpt = config_no_ckpt.calculate_memory_requirements();
        let memory_ckpt = config_ckpt.calculate_memory_requirements();

        // Checkpointing should reduce activation memory
        assert!(memory_ckpt.activation_memory_mb < memory_no_ckpt.activation_memory_mb);
    }

    #[test]
    fn test_model_fits_in_memory() {
        let config = ModelPresets::model_7b();
        let memory = config.calculate_memory_requirements();

        // Should not fit in 16GB
        assert!(!memory.fits_in_memory(16 * 1024));

        // Should fit in 64GB
        assert!(memory.fits_in_memory(64 * 1024));
    }

    #[test]
    fn test_initialization_strategy() {
        let config = ModelPresets::model_7b();

        // Layer-by-layer should use less memory than all-at-once
        let layer_by_layer = LargeModelInitializer::new(config.clone())
            .with_strategy(InitializationStrategy::LayerByLayer)
            .init_memory_requirements();

        let all_at_once = LargeModelInitializer::new(config)
            .with_strategy(InitializationStrategy::AllAtOnce)
            .init_memory_requirements();

        assert!(layer_by_layer < all_at_once);
    }

    #[test]
    fn test_initialize_layer() {
        let config = ModelPresets::model_7b();
        let initializer = LargeModelInitializer::new(config);

        let layer_params = initializer.initialize_layer(0).unwrap();
        assert!(!layer_params.is_empty());

        // Layer 32 should fail (out of range)
        let result = initializer.initialize_layer(32);
        assert!(result.is_err());
    }

    #[test]
    fn test_initialize_embedding() {
        let config = ModelPresets::model_7b();
        let initializer = LargeModelInitializer::new(config);

        let embedding = initializer.initialize_embedding().unwrap();
        assert!(!embedding.is_empty());
    }

    #[test]
    fn test_progressive_initialization() {
        let config = ModelPresets::model_7b();
        let initializer = LargeModelInitializer::new(config);

        let progress = initializer.initialize_progressively().unwrap();

        assert_eq!(progress.current_layer(), 0);
        assert_eq!(progress.total_layers(), 32);
        assert!(!progress.is_complete());
        assert_eq!(progress.progress(), 0.0);
    }

    #[test]
    fn test_parameter_sharding() {
        let config = ModelPresets::model_7b().with_num_devices(4);
        let sharding = ParameterSharding::new(4, &config);

        assert_eq!(sharding.num_shards, 4);

        // Check that some parameters are assigned
        let shard_0_params = sharding.get_shard_params(0);
        assert!(!shard_0_params.is_empty());

        // Check that embedding is sharded
        let shard = sharding.get_shard("embedding.0").unwrap();
        assert!(shard < 4);
    }

    #[test]
    fn test_memory_breakdown_percentages() {
        let config = ModelPresets::model_7b();
        let memory = config.calculate_memory_requirements();

        let (param_pct, opt_pct, act_pct, grad_pct) = memory.breakdown_percentages();

        // All percentages should be positive
        assert!(param_pct > 0.0);
        assert!(opt_pct > 0.0);
        assert!(act_pct > 0.0);
        assert!(grad_pct > 0.0);

        // Sum should be approximately 100%
        let total = param_pct + opt_pct + act_pct + grad_pct;
        assert!((total - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_size_category_descriptions() {
        assert!(!ModelSizeCategory::Small.description().is_empty());
        assert!(!ModelSizeCategory::Large.description().is_empty());
    }

    #[test]
    fn test_recommended_techniques() {
        let small_techs = ModelSizeCategory::Small.recommended_techniques();
        assert!(!small_techs.is_empty());

        let large_techs = ModelSizeCategory::Large.recommended_techniques();
        assert!(large_techs.len() > small_techs.len());
    }
}
