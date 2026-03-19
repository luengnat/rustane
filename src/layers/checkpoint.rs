//! Model checkpoint save/load functionality

use crate::layers::{Layer, Model, Sequential};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Model checkpoint containing weights and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Layer weights
    pub layers: Vec<LayerWeights>,
}

/// Metadata about a model checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Rustane version
    pub rustane_version: String,
    /// Timestamp
    pub timestamp: String,
}

/// Weights for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: String,
    /// Weight data (flattened)
    pub weights: Option<Vec<f32>>,
    /// Bias data
    pub bias: Option<Vec<f32>>,
}

impl Checkpoint {
    /// Create a new checkpoint from a Sequential model
    pub fn from_model(model: &Sequential) -> Result<Self> {
        let layers: Vec<LayerWeights> = (0..model.len())
            .map(|i| {
                let layer = model.get(i).unwrap();
                LayerWeights {
                    name: format!("layer_{}", i),
                    layer_type: layer.name().to_string(),
                    weights: None, // TODO: Extract weights from layer
                    bias: None,    // TODO: Extract bias from layer
                }
            })
            .collect();

        let metadata = ModelMetadata {
            name: model.name().to_string(),
            version: "0.1.0".to_string(),
            rustane_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono_timestamp().unwrap_or_else(|| "unknown".to_string()),
        };

        Ok(Self { metadata, layers })
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::Io(format!("Failed to create checkpoint file: {}", e)))?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self)
            .map_err(|e| Error::Io(format!("Failed to serialize checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Io(format!("Failed to open checkpoint file: {}", e)))?;

        let reader = BufReader::new(file);
        let checkpoint: Checkpoint = serde_json::from_reader(reader)
            .map_err(|e| Error::Io(format!("Failed to deserialize checkpoint: {}", e)))?;

        Ok(checkpoint)
    }

    /// Get checkpoint metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get layer weights
    pub fn layers(&self) -> &[LayerWeights] {
        &self.layers
    }
}

impl Sequential {
    /// Save model weights to checkpoint file
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        let checkpoint = Checkpoint::from_model(self)?;
        checkpoint.save(path)
    }

    /// Load model weights from checkpoint file
    pub fn load_checkpoint(&self, _path: &Path) -> Result<()> {
        // TODO: Implement loading weights into layers
        // This requires layers to have set_weights() methods
        Err(Error::ExecutionFailed(
            "Loading checkpoints not yet implemented".to_string(),
        ))
    }
}

/// Get current timestamp as ISO 8601 string
fn chrono_timestamp() -> Option<String> {
    // Simple timestamp implementation without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;

    let secs = duration.as_secs();
    Some(format!("{}", secs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ReLU;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_creation() {
        let model = Sequential::new("test").add(Box::new(ReLU::new())).build();

        let checkpoint = Checkpoint::from_model(&model).unwrap();
        assert_eq!(checkpoint.metadata.name, "test");
        assert_eq!(checkpoint.layers.len(), 1);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let model = Sequential::new("test").add(Box::new(ReLU::new())).build();

        let checkpoint = Checkpoint::from_model(&model).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");

        checkpoint.save(&path).unwrap();

        assert!(path.exists());

        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.metadata.name, checkpoint.metadata.name);
        assert_eq!(loaded.layers.len(), checkpoint.layers.len());
    }

    #[test]
    fn test_checkpoint_metadata() {
        let model = Sequential::new("test_model")
            .add(Box::new(ReLU::new()))
            .build();

        let checkpoint = Checkpoint::from_model(&model).unwrap();
        let metadata = checkpoint.metadata();

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, "0.1.0");
        assert!(!metadata.rustane_version.is_empty());
    }
}
