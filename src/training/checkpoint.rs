//! Model checkpointing for training persistence
//!
//! Provides save/load functionality for model checkpoints during training.
//! Checkpoints include model weights, optimizer state, and training metadata.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Checkpoint data containing model state and training information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model weights in FP32
    pub weights: Vec<f32>,
    /// Current training step
    pub step: usize,
    /// Training loss at checkpoint time
    pub loss: f32,
    /// Learning rate at checkpoint time
    pub learning_rate: f32,
    /// Checkpoint timestamp (Unix time)
    pub timestamp: i64,
    /// Optional: optimizer state (e.g., Adam moments)
    pub optimizer_state: Option<OptimizerState>,
    /// Optional: loss scaler state
    pub loss_scaler_state: Option<LossScalerState>,
    /// Model configuration (for validation)
    pub config: ModelConfig,
}

/// Optimizer state for checkpointing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerState {
    /// First moment estimate (Adam m)
    pub m: Option<Vec<f32>>,
    /// Second moment estimate (Adam v)
    pub v: Option<Vec<f32>>,
    /// Beta1 value
    pub beta1: f32,
    /// Beta2 value
    pub beta2: f32,
}

/// Loss scaler state for checkpointing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossScalerState {
    /// Current loss scale factor
    pub scale: f32,
    /// Steps since last growth
    pub steps_since_growth: u32,
}

/// Model configuration for validation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub dim: usize,
    /// Feed-forward dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub seq_len: usize,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(
        weights: Vec<f32>,
        step: usize,
        loss: f32,
        learning_rate: f32,
        config: ModelConfig,
    ) -> Self {
        Self {
            weights,
            step,
            loss,
            learning_rate,
            timestamp: chrono_timestamp(),
            optimizer_state: None,
            loss_scaler_state: None,
            config,
        }
    }

    /// Add optimizer state to checkpoint
    pub fn with_optimizer_state(mut self, state: OptimizerState) -> Self {
        self.optimizer_state = Some(state);
        self
    }

    /// Add loss scaler state to checkpoint
    pub fn with_loss_scaler_state(mut self, state: LossScalerState) -> Self {
        self.loss_scaler_state = Some(state);
        self
    }

    /// Save checkpoint to file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                crate::Error::Io(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        let file = File::create(path)
            .map_err(|e| crate::Error::Io(format!("Failed to create checkpoint file: {}", e)))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| crate::Error::Other(format!("Failed to serialize checkpoint: {}", e)))?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| crate::Error::Io(format!("Failed to open checkpoint file: {}", e)))?;
        let reader = BufReader::new(file);
        let checkpoint: Checkpoint = serde_json::from_reader(reader)
            .map_err(|e| crate::Error::Other(format!("Failed to deserialize checkpoint: {}", e)))?;
        Ok(checkpoint)
    }

    /// Validate checkpoint weights against expected parameter count
    pub fn validate(&self, expected_params: usize) -> Result<()> {
        if self.weights.len() != expected_params {
            return Err(crate::Error::InvalidParameter(format!(
                "Checkpoint has {} weights but expected {}",
                self.weights.len(),
                expected_params
            )));
        }
        Ok(())
    }
}

/// Get current Unix timestamp
fn chrono_timestamp() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Generate checkpoint filename from step number
pub fn checkpoint_filename(step: usize) -> String {
    format!("checkpoint_{:05}.json", step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let config = ModelConfig {
            vocab_size: 512,
            dim: 128,
            hidden_dim: 256,
            n_heads: 4,
            n_layers: 2,
            seq_len: 64,
        };

        let weights = vec![0.0f32; 1000];
        let checkpoint = Checkpoint::new(weights.clone(), 100, 2.5, 0.001, config);

        assert_eq!(checkpoint.weights.len(), 1000);
        assert_eq!(checkpoint.step, 100);
        assert_eq!(checkpoint.loss, 2.5);
        assert!(checkpoint.optimizer_state.is_none());
    }

    #[test]
    fn test_checkpoint_with_optimizer_state() {
        let config = ModelConfig {
            vocab_size: 512,
            dim: 128,
            hidden_dim: 256,
            n_heads: 4,
            n_layers: 2,
            seq_len: 64,
        };

        let weights = vec![0.0f32; 100];
        let opt_state = OptimizerState {
            m: Some(vec![0.0f32; 100]),
            v: Some(vec![0.0f32; 100]),
            beta1: 0.9,
            beta2: 0.999,
        };

        let checkpoint =
            Checkpoint::new(weights, 10, 1.0, 0.01, config).with_optimizer_state(opt_state);

        assert!(checkpoint.optimizer_state.is_some());
        let state = checkpoint.optimizer_state.as_ref().unwrap();
        assert_eq!(state.beta1, 0.9);
        assert_eq!(state.beta2, 0.999);
    }

    #[test]
    fn test_checkpoint_validation() {
        let config = ModelConfig {
            vocab_size: 512,
            dim: 128,
            hidden_dim: 256,
            n_heads: 4,
            n_layers: 2,
            seq_len: 64,
        };

        let weights = vec![0.0f32; 100];
        let checkpoint = Checkpoint::new(weights, 10, 1.0, 0.01, config);

        // Correct size
        assert!(checkpoint.validate(100).is_ok());

        // Wrong size
        assert!(checkpoint.validate(200).is_err());
    }

    #[test]
    fn test_checkpoint_filename() {
        assert_eq!(checkpoint_filename(1), "checkpoint_00001.json");
        assert_eq!(checkpoint_filename(100), "checkpoint_00100.json");
        assert_eq!(checkpoint_filename(10000), "checkpoint_10000.json");
    }
}
