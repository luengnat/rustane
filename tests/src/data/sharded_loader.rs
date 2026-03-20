//! Tests for sharded data loading types
//!
//! These tests verify the creation and structure of ShardConfig, ShardMetadata,
//! and ShardBatch types used in distributed training scenarios.

#[cfg(test)]
mod tests {
    use rustane::data::ShardConfig;
    use std::path::PathBuf;

    /// Test basic ShardConfig creation
    #[test]
    fn test_shard_config_creation() {
        let config = ShardConfig {
            shard_pattern: "data/shards/shard_*.jsonl".to_string(),
            vocab_size: 50000,
            shard_metadata: None,
        };

        assert_eq!(config.vocab_size, 50000);
        assert_eq!(
            config.shard_pattern,
            "data/shards/shard_*.jsonl".to_string()
        );
        assert!(config.shard_metadata.is_none());
    }

    /// Test ShardMetadata creation
    #[test]
    fn test_shard_metadata_creation() {
        use rustane::data::ShardMetadata;

        let metadata = ShardMetadata {
            shard_idx: 5,
            token_count: 1_000_000,
            path: "data/shards/shard_5.jsonl".to_string(),
        };

        assert_eq!(metadata.shard_idx, 5);
        assert_eq!(metadata.token_count, 1_000_000);
        assert_eq!(metadata.path, "data/shards/shard_5.jsonl");
    }

    /// Test ShardBatch creation with metadata
    #[test]
    fn test_shard_batch_creation() {
        use rustane::data::{ShardBatch, ShardMetadata};

        let metadata = ShardMetadata {
            shard_idx: 0,
            token_count: 500_000,
            path: "data/shards/shard_0.jsonl".to_string(),
        };

        let shard_batch = ShardBatch {
            shard_idx: metadata.shard_idx,
            shard_path: PathBuf::from(&metadata.path),
            token_count: metadata.token_count,
        };

        assert_eq!(shard_batch.shard_idx, 0);
        assert_eq!(shard_batch.token_count, 500_000);
        assert_eq!(
            shard_batch.shard_path,
            PathBuf::from("data/shards/shard_0.jsonl")
        );
    }
}
