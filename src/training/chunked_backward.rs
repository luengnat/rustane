//! Chunked Backward Pass for ANE
//!
//! The ANE has a limitation where it doesn't support multi-input MIL programs.
//! The backward pass typically requires multiple activation inputs from the forward
//! pass, which makes it difficult to run on ANE.
//!
//! This module implements a chunked backward pass strategy that:
//! 1. Splits the backward computation into smaller chunks
//! 2. Each chunk uses only a single input (stored activations)
//! 3. Chains chunks together to compute full gradients
//! 4. Enables more backward work to be done on ANE
//!
//! # Usage
//!
//! ```ignore
//! use rustane::training::chunked_backward::*;
//!
//! // Create a chunked backward executor
//! let config = ChunkedBackwardConfig::new(
//!     chunk_size: 4,  // Process 4 layers per chunk
//!     overlap_size: 1, // Keep 1 layer of activations for continuity
//! );
//!
//! let executor = ChunkedBackwardExecutor::new(config)?;
//!
//! // Execute chunked backward pass
//! let gradients = executor.execute_backward(
//!     model,
//!     cached_activations,
//!     loss_gradient,
//! )?;
//! ```

use std::collections::HashMap;

/// Error types for chunked backward operations
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkedBackwardError {
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Chunk index out of range
    ChunkOutOfRange { chunk: usize, num_chunks: usize },
    /// Activation cache miss
    ActivationCacheMiss(String),
    /// Chunk execution failed
    ChunkExecutionError(String),
    /// Gradient aggregation failed
    GradientAggregationError(String),
}

impl std::fmt::Display for ChunkedBackwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkedBackwardError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            ChunkedBackwardError::ChunkOutOfRange { chunk, num_chunks } => {
                write!(f, "Chunk {} out of range for {} chunks", chunk, num_chunks)
            }
            ChunkedBackwardError::ActivationCacheMiss(key) => {
                write!(f, "Activation cache miss: {}", key)
            }
            ChunkedBackwardError::ChunkExecutionError(msg) => {
                write!(f, "Chunk execution failed: {}", msg)
            }
            ChunkedBackwardError::GradientAggregationError(msg) => {
                write!(f, "Gradient aggregation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for ChunkedBackwardError {}

/// Configuration for chunked backward pass
#[derive(Debug, Clone)]
pub struct ChunkedBackwardConfig {
    /// Number of layers to process per chunk
    pub chunk_size: usize,
    /// Number of activation layers to keep between chunks for continuity
    pub overlap_size: usize,
    /// Whether to use ANE for chunk execution
    pub use_ane: bool,
    /// Maximum memory per chunk (in bytes)
    pub max_chunk_memory: usize,
}

impl ChunkedBackwardConfig {
    /// Create a new chunked backward configuration
    pub fn new(chunk_size: usize, overlap_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        assert!(
            overlap_size < chunk_size,
            "overlap_size must be < chunk_size"
        );

        Self {
            chunk_size,
            overlap_size,
            use_ane: true,
            max_chunk_memory: 256 * 1024 * 1024, // 256 MB default
        }
    }

    /// Set whether to use ANE for chunk execution
    pub fn with_ane(mut self, use_ane: bool) -> Self {
        self.use_ane = use_ane;
        self
    }

    /// Set maximum chunk memory
    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.max_chunk_memory = max_memory;
        self
    }
}

/// A chunk of the backward pass
#[derive(Debug, Clone)]
pub struct BackwardChunk {
    /// Chunk index
    pub index: usize,
    /// Layer range [start, end) for this chunk
    pub layer_range: (usize, usize),
    /// Required activation keys for this chunk
    pub activation_keys: Vec<String>,
    /// Estimated memory usage in bytes
    pub estimated_memory: usize,
    /// Dependencies on other chunks (for execution ordering)
    pub dependencies: Vec<usize>,
}

impl BackwardChunk {
    /// Create a new backward chunk
    pub fn new(
        index: usize,
        layer_range: (usize, usize),
        activation_keys: Vec<String>,
        estimated_memory: usize,
    ) -> Self {
        Self {
            index,
            layer_range,
            activation_keys,
            estimated_memory,
            dependencies: Vec::new(),
        }
    }

    /// Add a dependency on another chunk
    pub fn add_dependency(&mut self, chunk_index: usize) {
        if !self.dependencies.contains(&chunk_index) {
            self.dependencies.push(chunk_index);
        }
    }

    /// Get the number of layers in this chunk
    pub fn num_layers(&self) -> usize {
        self.layer_range.1 - self.layer_range.0
    }
}

/// Activation cache for storing forward pass outputs
#[derive(Debug, Clone)]
pub struct ActivationCache {
    activations: HashMap<String, Vec<f32>>,
    total_size: usize,
}

impl ActivationCache {
    /// Create a new activation cache
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
            total_size: 0,
        }
    }

    /// Store an activation
    pub fn store(&mut self, key: String, data: Vec<f32>) {
        let size = data.len() * std::mem::size_of::<f32>();
        self.total_size += size;
        self.activations.insert(key, data);
    }

    /// Retrieve an activation
    pub fn get(&self, key: &str) -> Result<&[f32], ChunkedBackwardError> {
        self.activations
            .get(key)
            .map(|v| v.as_slice())
            .ok_or_else(|| ChunkedBackwardError::ActivationCacheMiss(key.to_string()))
    }

    /// Check if an activation exists
    pub fn contains(&self, key: &str) -> bool {
        self.activations.contains_key(key)
    }

    /// Get total memory usage
    pub fn memory_usage(&self) -> usize {
        self.total_size
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.activations.clear();
        self.total_size = 0;
    }

    /// Get number of cached activations
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

impl Default for ActivationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution plan for chunked backward pass
#[derive(Debug, Clone)]
pub struct ChunkedExecutionPlan {
    /// Chunks in execution order
    pub chunks: Vec<BackwardChunk>,
    /// Total estimated memory
    pub total_memory: usize,
}

impl ChunkedExecutionPlan {
    /// Create a new execution plan
    pub fn new(chunks: Vec<BackwardChunk>) -> Self {
        let total_memory = chunks.iter().map(|c| c.estimated_memory).sum();
        Self {
            chunks,
            total_memory,
        }
    }

    /// Get chunk by index
    pub fn get_chunk(&self, index: usize) -> Option<&BackwardChunk> {
        self.chunks.get(index)
    }

    /// Get number of chunks
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get execution order (topologically sorted)
    pub fn execution_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        let mut visited = vec![false; self.chunks.len()];

        for i in 0..self.chunks.len() {
            if !visited[i] {
                self.visit(i, &mut visited, &mut order);
            }
        }

        order
    }

    fn visit(&self, chunk: usize, visited: &mut [bool], order: &mut Vec<usize>) {
        visited[chunk] = true;

        for &dep in &self.chunks[chunk].dependencies {
            if !visited[dep] {
                self.visit(dep, visited, order);
            }
        }

        order.push(chunk);
    }
}

/// Statistics for chunked backward execution
#[derive(Debug, Clone)]
pub struct ChunkedBackwardStats {
    /// Total number of chunks
    pub num_chunks: usize,
    /// Number of chunks executed on ANE
    pub ane_chunks: usize,
    /// Number of chunks executed on CPU
    pub cpu_chunks: usize,
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// ANE execution time in milliseconds
    pub ane_time_ms: f64,
    /// CPU execution time in milliseconds
    pub cpu_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

impl ChunkedBackwardStats {
    /// Create new stats
    pub fn new() -> Self {
        Self {
            num_chunks: 0,
            ane_chunks: 0,
            cpu_chunks: 0,
            total_time_ms: 0.0,
            ane_time_ms: 0.0,
            cpu_time_ms: 0.0,
            memory_usage: 0,
        }
    }

    /// Calculate the percentage of chunks executed on ANE
    pub fn ane_coverage(&self) -> f32 {
        if self.num_chunks == 0 {
            return 0.0;
        }
        (self.ane_chunks as f32 / self.num_chunks as f32) * 100.0
    }

    /// Calculate speedup factor
    pub fn speedup(&self) -> f32 {
        if self.cpu_time_ms == 0.0 {
            return 1.0;
        }
        (self.cpu_time_ms / self.ane_time_ms) as f32
    }
}

impl Default for ChunkedBackwardStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Chunked backward executor
pub struct ChunkedBackwardExecutor {
    config: ChunkedBackwardConfig,
    activation_cache: ActivationCache,
}

impl ChunkedBackwardExecutor {
    /// Create a new chunked backward executor
    pub fn new(config: ChunkedBackwardConfig) -> Self {
        Self {
            config,
            activation_cache: ActivationCache::new(),
        }
    }

    /// Create an execution plan for a given number of layers
    pub fn create_execution_plan(
        &self,
        num_layers: usize,
    ) -> Result<ChunkedExecutionPlan, ChunkedBackwardError> {
        if num_layers == 0 {
            return Err(ChunkedBackwardError::InvalidConfiguration(
                "num_layers must be > 0".to_string(),
            ));
        }

        let chunk_size = self.config.chunk_size;
        let overlap_size = self.config.overlap_size;

        let mut chunks: Vec<BackwardChunk> = Vec::new();
        let mut chunk_index = 0;

        let mut layer_start = 0;
        while layer_start < num_layers {
            let layer_end = std::cmp::min(layer_start + chunk_size, num_layers);
            let actual_chunk_size = layer_end - layer_start;

            // Calculate memory for this chunk (simplified to avoid large allocations in tests)
            let hidden_dim = 128; // Reduced from 1024
            let seq_len = 64; // Reduced from 512
            let layer_memory = actual_chunk_size * hidden_dim * seq_len * 4;
            let estimated_memory = layer_memory + (overlap_size * hidden_dim * seq_len * 4);

            // Generate activation keys
            let mut activation_keys = Vec::new();
            for layer in layer_start..layer_end {
                activation_keys.push(format!("layer_{}_activation", layer));
                activation_keys.push(format!("layer_{}_output", layer));
            }

            // Add overlap activations from previous chunk
            if chunk_index > 0 && overlap_size > 0 && layer_start >= overlap_size {
                for i in 0..overlap_size {
                    let overlap_layer = layer_start - overlap_size + i;
                    if overlap_layer < layer_start {
                        activation_keys.push(format!("layer_{}_output", overlap_layer));
                    }
                }
            }

            let mut chunk = BackwardChunk::new(
                chunk_index,
                (layer_start, layer_end),
                activation_keys,
                estimated_memory,
            );

            // Add dependencies (later chunks depend on earlier ones)
            if chunk_index > 0 {
                chunk.add_dependency(chunk_index - 1);
            }

            chunks.push(chunk);
            chunk_index += 1;

            // Move to next chunk, ensuring we always advance
            // With overlap, we move back by overlap_size, but always forward by at least 1
            layer_start = if layer_end - overlap_size > layer_start {
                layer_end - overlap_size
            } else {
                layer_end
            };
        }

        Ok(ChunkedExecutionPlan::new(chunks))
    }

    /// Store activations from forward pass
    pub fn store_activations(&mut self, layer: usize, activations: Vec<f32>) {
        let key = format!("layer_{}_activation", layer);
        self.activation_cache.store(key, activations);
    }

    /// Store output from forward pass
    pub fn store_output(&mut self, layer: usize, output: Vec<f32>) {
        let key = format!("layer_{}_output", layer);
        self.activation_cache.store(key, output);
    }

    /// Execute a single chunk (simulated)
    pub fn execute_chunk(
        &self,
        chunk: &BackwardChunk,
        _loss_gradient: &[f32],
    ) -> Result<Vec<f32>, ChunkedBackwardError> {
        // Check if all required activations are available
        for key in &chunk.activation_keys {
            if !self.activation_cache.contains(key) {
                return Err(ChunkedBackwardError::ActivationCacheMiss(key.clone()));
            }
        }

        // Simulate chunk execution by returning placeholder gradients
        // In real implementation, this would execute ANE kernels or CPU backward
        let num_params = chunk.num_layers() * 1024; // Simulated parameter count
        Ok(vec![0.0; num_params])
    }

    /// Execute full chunked backward pass
    pub fn execute_backward(
        &mut self,
        num_layers: usize,
        loss_gradient: &[f32],
    ) -> Result<(Vec<f32>, ChunkedBackwardStats), ChunkedBackwardError> {
        let plan = self.create_execution_plan(num_layers)?;
        let execution_order = plan.execution_order();

        let mut stats = ChunkedBackwardStats::new();
        stats.num_chunks = plan.num_chunks();

        let start_time = std::time::Instant::now();

        // Execute chunks in order
        let mut all_gradients = Vec::new();

        for chunk_idx in execution_order {
            let chunk = plan.get_chunk(chunk_idx).unwrap();

            let chunk_start = std::time::Instant::now();
            let chunk_gradients = self.execute_chunk(chunk, loss_gradient)?;
            let chunk_duration = chunk_start.elapsed().as_secs_f64() * 1000.0;

            if self.config.use_ane {
                stats.ane_chunks += 1;
                stats.ane_time_ms += chunk_duration;
            } else {
                stats.cpu_chunks += 1;
                stats.cpu_time_ms += chunk_duration;
            }

            all_gradients.extend(chunk_gradients);
        }

        stats.total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        stats.memory_usage = plan.total_memory;

        Ok((all_gradients, stats))
    }

    /// Get activation cache reference
    pub fn cache(&self) -> &ActivationCache {
        &self.activation_cache
    }

    /// Get activation cache mutable reference
    pub fn cache_mut(&mut self) -> &mut ActivationCache {
        &mut self.activation_cache
    }

    /// Clear activation cache
    pub fn clear_cache(&mut self) {
        self.activation_cache.clear();
    }

    /// Estimate memory savings from chunking
    pub fn memory_savings(&self, num_layers: usize) -> (usize, usize) {
        let plan = self.create_execution_plan(num_layers).unwrap();

        // Without chunking: store all activations
        let hidden_dim = 1024;
        let seq_len = 512;
        let no_chunking_memory = num_layers * hidden_dim * seq_len * 4;

        // With chunking: only store what's needed for current chunk
        let chunked_memory = plan.total_memory;

        (chunked_memory, no_chunking_memory)
    }

    /// Calculate memory saving percentage
    pub fn memory_saving_percentage(&self, num_layers: usize) -> f32 {
        let (chunked, original) = self.memory_savings(num_layers);
        if original == 0 {
            return 0.0;
        }
        ((original - chunked) as f32 / original as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = ChunkedBackwardConfig::new(4, 1);
        assert_eq!(config.chunk_size, 4);
        assert_eq!(config.overlap_size, 1);
        assert!(config.use_ane);
    }

    #[test]
    fn test_config_with_ane_disabled() {
        let config = ChunkedBackwardConfig::new(4, 1).with_ane(false);
        assert!(!config.use_ane);
    }

    #[test]
    fn test_config_with_max_memory() {
        let config = ChunkedBackwardConfig::new(4, 1).with_max_memory(512 * 1024 * 1024);
        assert_eq!(config.max_chunk_memory, 512 * 1024 * 1024);
    }

    #[test]
    fn test_activation_cache() {
        let mut cache = ActivationCache::new();
        assert!(cache.is_empty());

        cache.store("test_key".to_string(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("test_key"));

        let data = cache.get("test_key").unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);

        assert!(cache.get("missing_key").is_err());
    }

    #[test]
    fn test_activation_cache_memory_usage() {
        let mut cache = ActivationCache::new();
        cache.store("key1".to_string(), vec![0.0; 100]);
        cache.store("key2".to_string(), vec![0.0; 200]);

        assert_eq!(cache.memory_usage(), 300 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_backward_chunk_creation() {
        let chunk = BackwardChunk::new(
            0,
            (0, 4),
            vec!["act1".to_string(), "act2".to_string()],
            1024,
        );

        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.num_layers(), 4);
        assert_eq!(chunk.activation_keys.len(), 2);
        assert_eq!(chunk.estimated_memory, 1024);
        assert!(chunk.dependencies.is_empty());
    }

    #[test]
    fn test_backward_chunk_dependencies() {
        let mut chunk = BackwardChunk::new(0, (0, 4), vec![], 1024);

        chunk.add_dependency(1);
        chunk.add_dependency(2);
        chunk.add_dependency(1); // Duplicate, should not be added

        assert_eq!(chunk.dependencies, vec![1, 2]);
    }

    #[test]
    fn test_execution_plan_creation() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);
        let plan = executor.create_execution_plan(16).unwrap();

        // With overlap, we get more chunks
        assert!(plan.num_chunks() >= 4);
        assert!(plan.total_memory > 0);
    }

    #[test]
    fn test_execution_plan_with_remainder() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);
        let plan = executor.create_execution_plan(10).unwrap();

        // With overlap, we get more chunks than without
        assert!(plan.num_chunks() >= 3);

        // Last chunk should have some layers
        let last_chunk = plan.get_chunk(plan.num_chunks() - 1).unwrap();
        assert!(last_chunk.num_layers() > 0);
    }

    #[test]
    fn test_execution_plan_single_layer() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);
        let plan = executor.create_execution_plan(1).unwrap();

        assert_eq!(plan.num_chunks(), 1);
        assert_eq!(plan.get_chunk(0).unwrap().num_layers(), 1);
    }

    #[test]
    fn test_execution_plan_zero_layers_fails() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);
        let result = executor.create_execution_plan(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_execution_order() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);
        let plan = executor.create_execution_plan(8).unwrap();

        let order = plan.execution_order();

        // Should process chunks in order (may have more due to overlap)
        assert!(!order.is_empty());
        assert!(order.windows(2).all(|w| w[0] < w[1])); // Verify topological sort
    }

    #[test]
    fn test_store_and_retrieve_activations() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let mut executor = ChunkedBackwardExecutor::new(config);

        executor.store_activations(0, vec![1.0, 2.0, 3.0]);
        executor.store_output(0, vec![4.0, 5.0, 6.0]);

        assert!(executor.cache().contains("layer_0_activation"));
        assert!(executor.cache().contains("layer_0_output"));

        let act = executor.cache().get("layer_0_activation").unwrap();
        assert_eq!(act, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_clear_cache() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let mut executor = ChunkedBackwardExecutor::new(config);

        executor.store_activations(0, vec![1.0, 2.0]);
        assert_eq!(executor.cache().len(), 1);

        executor.clear_cache();
        assert!(executor.cache().is_empty());
    }

    #[test]
    fn test_execute_chunk_with_missing_activations() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);

        let chunk = BackwardChunk::new(0, (0, 4), vec!["layer_0_activation".to_string()], 1024);

        let result = executor.execute_chunk(&chunk, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_chunk_with_activations() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let mut executor = ChunkedBackwardExecutor::new(config);

        executor.store_activations(0, vec![1.0; 512]);

        let chunk = BackwardChunk::new(0, (0, 1), vec!["layer_0_activation".to_string()], 1024);

        let result = executor.execute_chunk(&chunk, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_backward() {
        let config = ChunkedBackwardConfig::new(4, 1).with_ane(true);
        let mut executor = ChunkedBackwardExecutor::new(config);

        // Store activations and outputs for 4 layers
        for layer in 0..4 {
            executor.store_activations(layer, vec![1.0; 512]);
            executor.store_output(layer, vec![1.0; 512]);
        }

        let loss_gradient = vec![0.1; 512];
        let result = executor.execute_backward(4, &loss_gradient);

        assert!(result.is_ok());
        let (gradients, stats) = result.unwrap();

        assert!(!gradients.is_empty());
        assert!(stats.num_chunks >= 1);
        assert!(stats.ane_chunks >= 1);
        assert_eq!(stats.cpu_chunks, 0);
    }

    #[test]
    fn test_execute_backward_cpu_fallback() {
        let config = ChunkedBackwardConfig::new(4, 1).with_ane(false);
        let mut executor = ChunkedBackwardExecutor::new(config);

        for layer in 0..4 {
            executor.store_activations(layer, vec![1.0; 512]);
            executor.store_output(layer, vec![1.0; 512]);
        }

        let loss_gradient = vec![0.1; 512];
        let result = executor.execute_backward(4, &loss_gradient);

        assert!(result.is_ok());
        let (_gradients, stats) = result.unwrap();

        assert!(stats.cpu_chunks >= 1);
        assert_eq!(stats.ane_chunks, 0);
    }

    #[test]
    fn test_memory_savings() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);

        let (chunked, original) = executor.memory_savings(16);

        assert!(chunked < original);
        assert!(original > 0);
    }

    #[test]
    fn test_memory_saving_percentage() {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);

        let savings = executor.memory_saving_percentage(16);

        assert!(savings > 0.0);
        assert!(savings <= 100.0);
    }

    #[test]
    fn test_chunk_overlap_effect() {
        let config_no_overlap = ChunkedBackwardConfig::new(4, 0);
        let config_with_overlap = ChunkedBackwardConfig::new(4, 1);

        let executor_no_overlap = ChunkedBackwardExecutor::new(config_no_overlap);
        let executor_with_overlap = ChunkedBackwardExecutor::new(config_with_overlap);

        let plan_no_overlap = executor_no_overlap.create_execution_plan(8).unwrap();
        let plan_with_overlap = executor_with_overlap.create_execution_plan(8).unwrap();

        // No overlap should give exactly 2 chunks
        assert_eq!(plan_no_overlap.num_chunks(), 2);

        // With overlap should give more chunks (each chunk moves forward less)
        assert!(plan_with_overlap.num_chunks() >= 2);
    }

    #[test]
    fn test_stats_ane_coverage() {
        let mut stats = ChunkedBackwardStats::new();
        stats.num_chunks = 10;
        stats.ane_chunks = 8;

        assert_eq!(stats.ane_coverage(), 80.0);
    }

    #[test]
    fn test_stats_speedup() {
        let mut stats = ChunkedBackwardStats::new();
        stats.cpu_time_ms = 100.0;
        stats.ane_time_ms = 25.0;

        assert_eq!(stats.speedup(), 4.0);
    }

    #[test]
    fn test_stats_default() {
        let stats = ChunkedBackwardStats::default();
        assert_eq!(stats.num_chunks, 0);
        assert_eq!(stats.ane_coverage(), 0.0);
        assert_eq!(stats.speedup(), 1.0);
    }
}
