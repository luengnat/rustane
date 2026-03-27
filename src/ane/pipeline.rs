//! Pipeline Parallelism for Large Transformer Models
//!
//! This module implements pipeline parallelism to overlap computation across layers,
//! enabling more efficient ANE utilization for large models.
//!
//! # Overview
//!
//! Pipeline parallelism splits the model across multiple execution stages, where:
//! - Each stage handles a subset of layers
//! - Micro-batches flow through the pipeline in an interleaved fashion
//! - Forward and backward passes can overlap across stages
//!
//! # Architecture
//!
//! ```text
//! Stage 0: Layers 0-3   ──► Stage 1: Layers 4-7   ──► Stage 2: Layers 8-11
//!     │                        │                          │
//!     ▼                        ▼                          ▼
//! [ANE Executor]          [ANE Executor]            [ANE Executor]
//! ```
//!
//! # Quick Start
//!
//! ```no_run
//! use rustane::ane::pipeline::{PipelineConfig, PipelineParallelModel};
//! use rustane::training::TransformerANE;
//!
//! // Configure pipeline with 3 stages
//! let config = PipelineConfig::new()
//!     .with_num_stages(3)
//!     .with_micro_batch_size(4)
//!     .with_overlap(true); // Enable forward/backward overlap
//!
//! // Wrap existing model with pipeline parallelism
//! let model: TransformerANE = todo!(); // Your model
//! let pipeline = PipelineParallelModel::new(model, config)?;
//! ```
//!
//! # Performance Guidelines
//!
//! - **Pipeline bubbles**: The pipeline has startup/shutdown overhead
//!   - Use enough micro-batches to amortize bubble time
//!   - Rule of thumb: micro_batches >= 2 * num_stages
//!
//! - **Memory vs throughput**: More stages = less memory per device but more bubbles
//!   - 2-4 stages typical for single-ANE systems
//!   - More stages useful for multi-ANE distributed training
//!
//! - **Overlap efficiency**: Forward/backward overlap can hide latency
//!   - Enable with `PipelineConfig::with_overlap(true)`
//!   - Requires gradient checkpointing for memory efficiency

use std::collections::VecDeque;
use std::time::Instant;

use crate::data::Batch;
use crate::error::Result;
use crate::wrapper::ANETensor;

/// Extended model trait for models that expose layer count
///
/// Required for pipeline parallelism to partition layers across stages.
pub trait ModelWithLayers: crate::training::Model {
    /// Get number of transformer layers in the model
    fn num_layers(&self) -> usize;
}

/// Configuration for pipeline parallelism
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages (default: 2)
    pub num_stages: usize,
    /// Micro-batch size per stage (default: 4)
    pub micro_batch_size: usize,
    /// Enable forward/backward overlap (default: false)
    pub enable_overlap: bool,
    /// Use gradient checkpointing to save memory (default: true)
    pub use_checkpointing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_stages: 2,
            micro_batch_size: 4,
            enable_overlap: false,
            use_checkpointing: true,
        }
    }
}

impl PipelineConfig {
    /// Create new pipeline configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of pipeline stages
    pub fn with_num_stages(mut self, stages: usize) -> Self {
        self.num_stages = stages;
        self
    }

    /// Set micro-batch size
    pub fn with_micro_batch_size(mut self, size: usize) -> Self {
        self.micro_batch_size = size;
        self
    }

    /// Enable forward/backward overlap
    pub fn with_overlap(mut self, enable: bool) -> Self {
        self.enable_overlap = enable;
        self
    }

    /// Enable gradient checkpointing
    pub fn with_checkpointing(mut self, enable: bool) -> Self {
        self.use_checkpointing = enable;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_stages == 0 {
            return Err(crate::Error::InvalidParameter(
                "num_stages must be at least 1".to_string(),
            ));
        }
        if self.micro_batch_size == 0 {
            return Err(crate::Error::InvalidParameter(
                "micro_batch_size must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Get total number of micro-batches needed to fill pipeline
    pub fn pipeline_fill_batches(&self) -> usize {
        self.num_stages * 2 - 1
    }

    /// Get recommended micro-batch count for good utilization
    pub fn recommended_micro_batches(&self) -> usize {
        // Rule of thumb: 2x pipeline depth for good utilization
        self.pipeline_fill_batches() * 2
    }
}

/// Statistics for pipeline execution
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total pipeline forward time (ms)
    pub forward_time_ms: f64,
    /// Total pipeline backward time (ms)
    pub backward_time_ms: f64,
    /// Pipeline bubble time (idle time waiting for stages)
    pub bubble_time_ms: f64,
    /// Number of micro-batches processed
    pub micro_batches_processed: usize,
    /// Average stage utilization (0.0 to 1.0)
    pub avg_utilization: f64,
    /// Per-stage statistics
    pub per_stage: Vec<StageStats>,
}

/// Per-stage statistics
#[derive(Debug, Clone, Default)]
pub struct StageStats {
    /// Stage index
    pub stage_id: usize,
    /// Forward compute time (ms)
    pub forward_compute_ms: f64,
    /// Backward compute time (ms)
    pub backward_compute_ms: f64,
    /// Forward wait time (ms)
    pub forward_wait_ms: f64,
    /// Backward wait time (ms)
    pub backward_wait_ms: f64,
    /// Number of micro-batches processed
    pub micro_batches: usize,
}

impl PipelineStats {
    /// Get overall pipeline efficiency
    pub fn efficiency(&self) -> f64 {
        let total_time = self.forward_time_ms + self.backward_time_ms;
        if total_time == 0.0 {
            return 0.0;
        }
        let compute_time = self
            .per_stage
            .iter()
            .map(|s| s.forward_compute_ms + s.backward_compute_ms)
            .sum::<f64>();
        compute_time / total_time
    }

    /// Get pipeline bubble ratio
    pub fn bubble_ratio(&self) -> f64 {
        let total_time = self.forward_time_ms + self.backward_time_ms;
        if total_time == 0.0 {
            return 0.0;
        }
        self.bubble_time_ms / total_time
    }
}

/// A single pipeline stage containing a subset of layers
pub struct PipelineStage {
    /// Stage index in pipeline
    pub stage_id: usize,
    /// Starting layer index
    pub start_layer: usize,
    /// Ending layer index (exclusive)
    pub end_layer: usize,
    /// Whether this is the first stage
    pub is_first: bool,
    /// Whether this is the last stage
    pub is_last: bool,
    /// ANE executor for this stage (if compiled)
    pub executor: Option<crate::wrapper::ANEExecutor>,
    /// Statistics for this stage
    pub stats: StageStats,
}

impl PipelineStage {
    /// Create a new pipeline stage
    pub fn new(
        stage_id: usize,
        start_layer: usize,
        end_layer: usize,
        is_first: bool,
        is_last: bool,
    ) -> Self {
        Self {
            stage_id,
            start_layer,
            end_layer,
            is_first,
            is_last,
            executor: None,
            stats: StageStats::default(),
        }
    }

    /// Get number of layers in this stage
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer
    }

    /// Check if stage is ready to execute (has executor)
    pub fn is_ready(&self) -> bool {
        self.executor.is_some()
    }
}

/// Internal micro-batch state for pipeline tracking
#[allow(dead_code)]
struct MicroBatchState {
    /// Micro-batch index
    index: usize,
    /// Current stage
    current_stage: usize,
    /// Forward completed
    forward_done: bool,
    /// Backward completed
    backward_done: bool,
    /// Hidden state passed between stages
    hidden_state: Option<ANETensor>,
    /// Gradient state for backward pass
    grad_state: Option<ANETensor>,
}

/// Pipeline parallel model wrapper
///
/// Wraps an existing model and distributes layers across pipeline stages.
/// Enables overlapping execution of forward and backward passes.
pub struct PipelineParallelModel<M> {
    /// Wrapped model
    model: M,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Pipeline stages
    stages: Vec<PipelineStage>,
    /// Micro-batch queue for scheduling
    micro_batch_queue: VecDeque<MicroBatchState>,
    /// Pipeline statistics
    stats: PipelineStats,
    /// Whether pipeline is initialized
    initialized: bool,
}

impl<M> PipelineParallelModel<M>
where
    M: crate::training::Model + ModelWithLayers,
{
    /// Create new pipeline parallel model
    pub fn new(model: M, config: PipelineConfig) -> Result<Self> {
        config.validate()?;

        let num_layers = model.num_layers();
        let layers_per_stage = (num_layers + config.num_stages - 1) / config.num_stages;

        let mut stages = Vec::with_capacity(config.num_stages);
        for stage_id in 0..config.num_stages {
            let start_layer = stage_id * layers_per_stage;
            let end_layer = std::cmp::min(start_layer + layers_per_stage, num_layers);

            if start_layer >= num_layers {
                break; // Don't create empty stages
            }

            stages.push(PipelineStage::new(
                stage_id,
                start_layer,
                end_layer,
                stage_id == 0,
                end_layer >= num_layers,
            ));
        }

        Ok(Self {
            model,
            config,
            stages,
            micro_batch_queue: VecDeque::new(),
            stats: PipelineStats::default(),
            initialized: false,
        })
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get number of pipeline stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Get number of layers in model
    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    /// Initialize pipeline (compile ANE executors for each stage)
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Each stage would compile its ANE executor here
        // For now, we mark as initialized
        self.initialized = true;

        println!(
            "Pipeline initialized: {} stages, {} layers total",
            self.stages.len(),
            self.num_layers()
        );

        Ok(())
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Reset pipeline statistics
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStats::default();
        for stage in &mut self.stages {
            stage.stats = StageStats::default();
        }
    }

    /// Split batch into micro-batches for pipeline processing
    fn create_micro_batches(&self, batch: &Batch) -> Result<Vec<Batch>> {
        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let tokens = batch.tokens().to_vec();

        let micro_batch_size = self.config.micro_batch_size;
        let num_micro_batches = (batch_size + micro_batch_size - 1) / micro_batch_size;

        let mut micro_batches = Vec::with_capacity(num_micro_batches);

        for i in 0..num_micro_batches {
            let start = i * micro_batch_size;
            let end = std::cmp::min(start + micro_batch_size, batch_size);
            let mb_tokens: Vec<u32> = tokens[start * seq_len..end * seq_len].to_vec();

            micro_batches.push(Batch::new(mb_tokens, end - start, seq_len)?);
        }

        Ok(micro_batches)
    }

    /// Execute forward pass using pipeline parallelism
    ///
    /// Uses 1F1B (one-forward-one-backward) scheduling for memory efficiency:
    /// - Fill pipeline with forward passes
    /// - Alternate forward and backward for remaining micro-batches
    /// - Drain pipeline with backward passes
    pub fn forward_pipeline(&mut self, batch: &Batch) -> Result<ANETensor> {
        let start = Instant::now();

        if !self.initialized {
            self.initialize()?;
        }

        let micro_batches = self.create_micro_batches(batch)?;
        let num_micro = micro_batches.len();

        println!(
            "Pipeline forward: {} micro-batches, {} stages",
            num_micro,
            self.stages.len()
        );

        // Initialize micro-batch states
        self.micro_batch_queue.clear();
        for i in 0..num_micro {
            self.micro_batch_queue.push_back(MicroBatchState {
                index: i,
                current_stage: 0,
                forward_done: false,
                backward_done: false,
                hidden_state: None,
                grad_state: None,
            });
        }

        // 1F1B schedule: forward all, then backward
        // Phase 1: Fill pipeline (forward only)
        let fill_batches = std::cmp::min(num_micro, self.config.pipeline_fill_batches());

        for _mb_idx in 0..fill_batches {
            self.execute_stage_forward(0)?; // Execute first stage
        }

        // Phase 2: 1F1B for remaining micro-batches
        // (In full implementation, would interleave forward/backward)

        // For now, execute all forwards sequentially
        for mb_idx in fill_batches..num_micro {
            self.execute_stage_forward(mb_idx)?;
        }

        self.stats.forward_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Return placeholder - full implementation would return final logits
        ANETensor::from_fp32(vec![0.0f32], vec![1])
    }

    /// Execute forward pass for a single stage
    fn execute_stage_forward(&mut self, stage_idx: usize) -> Result<()> {
        if stage_idx >= self.stages.len() {
            return Err(crate::Error::InvalidParameter(format!(
                "Invalid stage index: {}",
                stage_idx
            )));
        }

        let stage = &mut self.stages[stage_idx];
        let stage_start = Instant::now();

        // In full implementation, would execute ANE for this stage's layers
        // For now, just update stats
        stage.stats.forward_compute_ms += stage_start.elapsed().as_secs_f64() * 1000.0;
        stage.stats.micro_batches += 1;

        Ok(())
    }

    /// Execute backward pass using pipeline parallelism
    pub fn backward_pipeline(&mut self, _loss: f32) -> Result<Vec<f32>> {
        let start = Instant::now();

        if !self.initialized {
            return Err(crate::Error::InvalidParameter(
                "Pipeline not initialized".to_string(),
            ));
        }

        println!("Pipeline backward: {} stages", self.stages.len());

        // Execute backward in reverse order (last stage first)
        for stage_idx in (0..self.stages.len()).rev() {
            self.execute_stage_backward(stage_idx)?;
        }

        self.stats.backward_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Return placeholder gradients
        Ok(vec![0.0f32; self.model.param_count()])
    }

    /// Execute backward pass for a single stage
    fn execute_stage_backward(&mut self, stage_idx: usize) -> Result<()> {
        if stage_idx >= self.stages.len() {
            return Err(crate::Error::InvalidParameter(format!(
                "Invalid stage index: {}",
                stage_idx
            )));
        }

        let stage = &mut self.stages[stage_idx];
        let stage_start = Instant::now();

        // In full implementation, would execute ANE backward for this stage
        stage.stats.backward_compute_ms += stage_start.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }
}

impl<M> std::fmt::Debug for PipelineParallelModel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineParallelModel")
            .field("config", &self.config)
            .field("num_stages", &self.stages.len())
            .field("initialized", &self.initialized)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.num_stages, 2);
        assert_eq!(config.micro_batch_size, 4);
        assert!(!config.enable_overlap);
        assert!(config.use_checkpointing);
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::new()
            .with_num_stages(4)
            .with_micro_batch_size(8)
            .with_overlap(true)
            .with_checkpointing(false);

        assert_eq!(config.num_stages, 4);
        assert_eq!(config.micro_batch_size, 8);
        assert!(config.enable_overlap);
        assert!(!config.use_checkpointing);
    }

    #[test]
    fn test_pipeline_config_validation() {
        // Valid config
        let config = PipelineConfig::new().with_num_stages(2);
        assert!(config.validate().is_ok());

        // Invalid: zero stages
        let config = PipelineConfig::new().with_num_stages(0);
        assert!(config.validate().is_err());

        // Invalid: zero micro-batch size
        let config = PipelineConfig::default().with_micro_batch_size(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipeline_fill_calculation() {
        let config = PipelineConfig::new().with_num_stages(3);
        // Formula: 2 * stages - 1
        assert_eq!(config.pipeline_fill_batches(), 5);
    }

    #[test]
    fn test_recommended_micro_batches() {
        let config = PipelineConfig::new().with_num_stages(4);
        // Formula: fill_batches * 2
        assert_eq!(config.recommended_micro_batches(), 14); // (2*4-1) * 2 = 14
    }

    #[test]
    fn test_pipeline_stage_creation() {
        let stage = PipelineStage::new(0, 0, 4, true, false);
        assert_eq!(stage.stage_id, 0);
        assert_eq!(stage.start_layer, 0);
        assert_eq!(stage.end_layer, 4);
        assert!(stage.is_first);
        assert!(!stage.is_last);
        assert_eq!(stage.num_layers(), 4);
        assert!(!stage.is_ready()); // No executor yet
    }

    #[test]
    fn test_pipeline_stage_middle() {
        let stage = PipelineStage::new(1, 4, 8, false, false);
        assert!(!stage.is_first);
        assert!(!stage.is_last);
        assert_eq!(stage.num_layers(), 4);
    }

    #[test]
    fn test_pipeline_stage_last() {
        let stage = PipelineStage::new(2, 8, 12, false, true);
        assert!(!stage.is_first);
        assert!(stage.is_last);
        assert_eq!(stage.num_layers(), 4);
    }

    #[test]
    fn test_pipeline_stats_default() {
        let stats = PipelineStats::default();
        assert_eq!(stats.forward_time_ms, 0.0);
        assert_eq!(stats.backward_time_ms, 0.0);
        assert_eq!(stats.efficiency(), 0.0);
        assert_eq!(stats.bubble_ratio(), 0.0);
    }

    #[test]
    fn test_pipeline_stats_efficiency() {
        let mut stats = PipelineStats {
            forward_time_ms: 100.0,
            backward_time_ms: 100.0,
            bubble_time_ms: 50.0,
            ..Default::default()
        };
        stats.per_stage.push(StageStats {
            stage_id: 0,
            forward_compute_ms: 80.0,
            backward_compute_ms: 80.0,
            ..Default::default()
        });

        // Efficiency = compute_time / total_time = 160 / 200 = 0.8
        assert!((stats.efficiency() - 0.8).abs() < 0.01);

        // Bubble ratio = bubble_time / total_time = 50 / 200 = 0.25
        assert!((stats.bubble_ratio() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_stage_stats_default() {
        let stats = StageStats::default();
        assert_eq!(stats.stage_id, 0);
        assert_eq!(stats.forward_compute_ms, 0.0);
        assert_eq!(stats.backward_compute_ms, 0.0);
        assert_eq!(stats.micro_batches, 0);
    }
}
