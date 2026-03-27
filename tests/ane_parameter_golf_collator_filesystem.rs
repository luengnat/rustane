//! ANE Parameter Golf - Collator and Filesystem Dataset Tests
//!
//! Additional tests for:
//! - Collation strategies (PadCollator, TruncateCollator)
//! - Filesystem datasets (JsonlDataset, TextDataset)
//! - Integration scenarios

use rustane::data::{
    Batch, BatchConfig, Collator, Dataset, JsonlDataset, PadCollator, SequentialDataset,
    TextDataset, TruncateCollator,
};

// ============================================================================
// TEST 1: PadCollator Additional Tests
// ============================================================================

#[test]
fn test_pad_collator_single_sample() {
    let collator = PadCollator::new(10, 0);
    let samples = vec![vec![1, 2, 3]];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (1, 10));
    assert_eq!(batch.tokens(), &[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_pad_collator_many_samples() {
    let collator = PadCollator::new(5, 0);
    let samples = vec![
        vec![1],
        vec![2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
    ];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (5, 5));

    // Verify each row
    assert_eq!(batch.get(0, 0), Some(1));
    assert_eq!(batch.get(0, 4), Some(0)); // Padded
    assert_eq!(batch.get(4, 4), Some(15)); // Full
}

#[test]
fn test_pad_collator_all_same_length() {
    let collator = PadCollator::new(4, 99);
    let samples = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (2, 4));
    assert_eq!(batch.tokens(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    // No padding needed, pad_token not used
}

#[test]
fn test_pad_collator_multiple_exceed_max() {
    let collator = PadCollator::new(3, 0);
    let samples = vec![
        vec![1, 2],       // OK
        vec![3, 4, 5, 6], // Exceeds
        vec![7],          // OK
    ];
    let result = collator.collate(samples);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds max_len"));
}

#[test]
fn test_pad_collator_first_exceeds() {
    let collator = PadCollator::new(2, 0);
    let samples = vec![vec![1, 2, 3]];
    let result = collator.collate(samples);
    assert!(result.is_err());
}

#[test]
fn test_pad_collator_large_batch() {
    let collator = PadCollator::new(128, 0);
    let samples: Vec<Vec<u32>> = (0..32)
        .map(|i| vec![i as u32; 64]) // 32 samples, each 64 tokens
        .collect();

    let batch = collator.collate(samples).unwrap();
    assert_eq!(batch.shape(), (32, 128));
    assert_eq!(batch.tokens().len(), 32 * 128);
}

// ============================================================================
// TEST 2: TruncateCollator Additional Tests
// ============================================================================

#[test]
fn test_truncate_collator_single_sample() {
    let collator = TruncateCollator::new(5, 0);
    let samples = vec![vec![1, 2, 3, 4, 5, 6, 7]];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (1, 5));
    assert_eq!(batch.tokens(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_truncate_collator_mixed_lengths() {
    let collator = TruncateCollator::new(4, 0);
    let samples = vec![
        vec![1],              // Padded
        vec![2, 3, 4, 5],     // Exact
        vec![6, 7, 8, 9, 10], // Truncated
    ];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (3, 4));

    // First: [1, 0, 0, 0]
    assert_eq!(batch.get(0, 0), Some(1));
    assert_eq!(batch.get(0, 1), Some(0));

    // Second: [2, 3, 4, 5]
    assert_eq!(batch.get(1, 3), Some(5));

    // Third: [6, 7, 8, 9] (10 truncated)
    assert_eq!(batch.get(2, 3), Some(9));
}

#[test]
fn test_truncate_collator_all_shorter() {
    let collator = TruncateCollator::new(10, 255);
    let samples = vec![vec![1, 2], vec![3, 4, 5]];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (2, 10));

    // Verify padding with custom token
    assert_eq!(batch.get(0, 2), Some(255));
    assert_eq!(batch.get(0, 9), Some(255));
    assert_eq!(batch.get(1, 3), Some(255));
}

#[test]
fn test_truncate_collator_all_longer() {
    let collator = TruncateCollator::new(3, 0);
    let samples = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (2, 3));
    assert_eq!(batch.tokens(), &[1, 2, 3, 6, 7, 8]);
}

#[test]
fn test_truncate_collator_empty() {
    let collator = TruncateCollator::new(5, 0);
    let result = collator.collate(vec![]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty samples"));
}

#[test]
fn test_truncate_collator_truncate_to_one() {
    let collator = TruncateCollator::new(1, 0);
    let samples = vec![vec![1, 2, 3], vec![4]];
    let batch = collator.collate(samples).unwrap();

    assert_eq!(batch.shape(), (2, 1));
    assert_eq!(batch.tokens(), &[1, 4]);
}

#[test]
fn test_truncate_collator_large_batch() {
    let collator = TruncateCollator::new(512, 0);
    let samples: Vec<Vec<u32>> = (0..16)
        .map(|i| vec![(i * 100) as u32; 1000]) // 16 samples, each 1000 tokens
        .collect();

    let batch = collator.collate(samples).unwrap();
    assert_eq!(batch.shape(), (16, 512));

    // Verify truncation happened
    for i in 0..16 {
        assert_eq!(batch.get(i, 0), Some((i * 100) as u32));
        assert_eq!(batch.get(i, 511), Some((i * 100) as u32));
    }
}

// ============================================================================
// TEST 3: Collator Integration Tests
// ============================================================================

#[test]
fn test_pad_vs_truncate_collator() {
    let samples = vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8]];

    // PadCollator rejects samples exceeding max_len
    let pad_collator = PadCollator::new(4, 0);
    let pad_result = pad_collator.collate(samples.clone());
    assert!(pad_result.is_err());

    // TruncateCollator handles them
    let truncate_collator = TruncateCollator::new(4, 0);
    let truncate_batch = truncate_collator.collate(samples).unwrap();
    assert_eq!(truncate_batch.shape(), (2, 4));
}

#[test]
fn test_collator_with_dataloader() {
    use rustane::data::{DataLoader, SequentialSampler};

    // Create dataset with fixed-length sequences (DataLoader requires uniform lengths)
    let samples = vec![
        vec![1u32, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
        vec![13, 14, 15, 16],
    ];
    let dataset = SequentialDataset::new(samples);
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

    // Collate each batch
    let _collator = PadCollator::new(4, 0);

    for batch_result in dataloader.iter() {
        let batch = batch_result.unwrap();
        // Verify batch structure
        assert_eq!(batch.seq_len(), 4);
        assert_eq!(batch.batch_size(), 2);
    }
}

// ============================================================================
// TEST 4: JsonlDataset Additional Tests
// ============================================================================

#[test]
fn test_jsonl_dataset_single_token_per_line() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "[0]").unwrap();
    writeln!(file, "[1]").unwrap();
    writeln!(file, "[2]").unwrap();

    let dataset = JsonlDataset::load(file.path()).unwrap();
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(0).unwrap(), vec![0]);
    assert_eq!(dataset.get(2).unwrap(), vec![2]);
}

#[test]
fn test_jsonl_dataset_large_tokens() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    // Test with large token IDs
    writeln!(file, "[1000000, 2000000, 3000000]").unwrap();

    let dataset = JsonlDataset::load(file.path()).unwrap();
    assert_eq!(dataset.len(), 1);
    assert_eq!(dataset.get(0).unwrap(), vec![1000000, 2000000, 3000000]);
}

#[test]
fn test_jsonl_dataset_many_lines() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    for i in 0..100 {
        writeln!(file, "[{}]", i).unwrap();
    }

    let dataset = JsonlDataset::load(file.path()).unwrap();
    assert_eq!(dataset.len(), 100);
    assert_eq!(dataset.get(50).unwrap(), vec![50]);
}

#[test]
fn test_jsonl_dataset_invalid_json() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "[0, 1, 2]").unwrap();
    writeln!(file, "not valid json").unwrap();

    let result = JsonlDataset::load(file.path());
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Failed to parse"));
}

#[test]
fn test_jsonl_dataset_empty_array() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "[]").unwrap();
    writeln!(file, "[1, 2]").unwrap();

    let dataset = JsonlDataset::load(file.path()).unwrap();
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.get(0).unwrap(), Vec::<u32>::new());
}

#[test]
fn test_jsonl_dataset_negative_tokens() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "[-1, 0, 1]").unwrap();

    // u32 cannot represent negative numbers
    let result = JsonlDataset::load(file.path());
    assert!(result.is_err());
}

// ============================================================================
// TEST 5: TextDataset Additional Tests
// ============================================================================

#[test]
fn test_text_dataset_empty_file() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "   ").unwrap(); // Whitespace only

    let result = TextDataset::load_space_separated(file.path());
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("no valid samples"));
}

#[test]
fn test_text_dataset_single_token_lines() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "1").unwrap();
    writeln!(file, "2").unwrap();
    writeln!(file, "3").unwrap();

    let dataset = TextDataset::load_space_separated(file.path()).unwrap();
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(0).unwrap(), vec![1]);
}

#[test]
fn test_text_dataset_mixed_whitespace() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "1  2   3").unwrap(); // Multiple spaces
    writeln!(file, "4\t5\t6").unwrap(); // Tabs

    let dataset = TextDataset::load_space_separated(file.path()).unwrap();
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.get(0).unwrap(), vec![1, 2, 3]);
    assert_eq!(dataset.get(1).unwrap(), vec![4, 5, 6]);
}

#[test]
fn test_text_dataset_comma_with_spaces() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "1, 2, 3").unwrap();
    writeln!(file, "4 ,5 ,6").unwrap();
    writeln!(file, "7 , 8 , 9").unwrap();

    let dataset = TextDataset::load_comma_separated(file.path()).unwrap();
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(0).unwrap(), vec![1, 2, 3]);
    assert_eq!(dataset.get(1).unwrap(), vec![4, 5, 6]);
}

#[test]
fn test_text_dataset_invalid_token_comma() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "1,2,abc,3").unwrap();

    let result = TextDataset::load_comma_separated(file.path());
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("cannot parse"));
}

#[test]
fn test_text_dataset_negative_token() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "-1 0 1").unwrap();

    let result = TextDataset::load_space_separated(file.path());
    assert!(result.is_err());
}

#[test]
fn test_text_dataset_overflow_token() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "99999999999999999999").unwrap(); // Overflow u32

    let result = TextDataset::load_space_separated(file.path());
    assert!(result.is_err());
}

// ============================================================================
// TEST 6: Parameter Golf Integration Scenarios
// ============================================================================

#[test]
fn test_prepare_batch_for_parameter_golf() {
    // Simulate preparing a batch for parameter-golf training
    let raw_tokens: Vec<u16> = (0..2048).collect();

    // Convert to u32 for Batch
    let tokens_u32: Vec<u32> = raw_tokens.iter().map(|&t| t as u32).collect();

    // Create batch with seq_len=1024
    let batch = Batch::new(tokens_u32, 2, 1024).unwrap();
    assert_eq!(batch.shape(), (2, 1024));

    // Split for gradient accumulation
    // Chunks must be multiples of seq_len, so 512 rounds up to 1024
    let chunks = batch.into_chunks(512).unwrap();
    assert_eq!(chunks.len(), 2); // 2048 / 1024 = 2 chunks

    for chunk in &chunks {
        assert_eq!(chunk.seq_len(), 1024);
        assert_eq!(chunk.tokens.len(), 1024);
    }
}

#[test]
fn test_collate_for_variable_length_sequences() {
    // Parameter-golf data may have variable-length sequences
    let raw_sequences = vec![
        vec![1u32, 2, 3, 4, 5],       // 5 tokens
        vec![6, 7, 8],                // 3 tokens
        vec![9, 10, 11, 12],          // 4 tokens
        vec![13, 14, 15, 16, 17, 18], // 6 tokens
    ];

    // Use truncate collator to normalize to fixed length
    let collator = TruncateCollator::new(5, 0);
    let batch = collator.collate(raw_sequences).unwrap();

    assert_eq!(batch.shape(), (4, 5));

    // Verify last sequence was truncated
    assert_eq!(batch.get(3, 0), Some(13));
    assert_eq!(batch.get(3, 4), Some(17)); // 18 was truncated
}

#[test]
fn test_batch_config_for_multi_gpu() {
    // Simulate multi-GPU parameter-golf training setup
    let world_size = 4;
    let global_batch_tokens = 262_144;
    let seq_len = 1024;
    let grad_accum_steps = 8;

    let configs: Vec<BatchConfig> = (0..world_size)
        .map(|rank| {
            BatchConfig::new(
                global_batch_tokens,
                seq_len,
                grad_accum_steps,
                world_size,
                rank,
            )
        })
        .collect();

    // Each rank should have same tokens_per_rank_accum
    for config in &configs {
        assert_eq!(config.tokens_per_rank_accum(), 8_192);
        assert!(config.validate().is_ok());
    }

    // Different ranks should have different rank IDs
    let ranks: Vec<usize> = configs.iter().map(|c| c.rank).collect();
    assert_eq!(ranks, vec![0, 1, 2, 3]);
}

#[test]
fn test_end_to_end_data_pipeline() {
    use rustane::data::{DataLoader, SequentialSampler};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Step 1: Create a text file with token data
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "0 1 2 3 4").unwrap();
    writeln!(file, "5 6 7 8 9").unwrap();
    writeln!(file, "10 11 12 13 14").unwrap();
    writeln!(file, "15 16 17 18 19").unwrap();

    // Step 2: Load dataset
    let dataset = TextDataset::load_space_separated(file.path()).unwrap();
    assert_eq!(dataset.len(), 4);

    // Step 3: Create dataloader
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

    // Step 4: Iterate and verify
    let mut batch_count = 0;
    for batch_result in dataloader.iter() {
        let batch = batch_result.unwrap();
        assert_eq!(batch.seq_len(), 5);
        batch_count += 1;
    }

    assert_eq!(batch_count, 2); // 4 samples / 2 batch_size = 2 batches
}

// ============================================================================
// TEST 7: Boundary and Edge Cases
// ============================================================================

#[test]
fn test_collator_max_len_zero() {
    let collator = PadCollator::new(0, 0);
    let samples = vec![vec![]];
    let batch = collator.collate(samples).unwrap();
    assert_eq!(batch.shape(), (1, 0));
}

#[test]
fn test_collator_very_large_max_len() {
    let collator = PadCollator::new(10000, 0);
    let samples = vec![vec![1, 2, 3]];
    let batch = collator.collate(samples).unwrap();
    assert_eq!(batch.shape(), (1, 10000));
    assert_eq!(batch.tokens().len(), 10000);
}

#[test]
fn test_text_dataset_file_not_found() {
    let result = TextDataset::load_space_separated("/nonexistent/path.txt");
    assert!(result.is_err());
}

#[test]
fn test_jsonl_dataset_file_not_found() {
    let result = JsonlDataset::load("/nonexistent/path.jsonl");
    assert!(result.is_err());
}

#[test]
fn test_batch_from_collator_chunks() {
    let collator = TruncateCollator::new(64, 0);
    let samples: Vec<Vec<u32>> = (0..8).map(|i| vec![i as u32; 32]).collect();

    let batch = collator.collate(samples).unwrap();
    assert_eq!(batch.shape(), (8, 64));

    // Chunk for gradient accumulation
    // Chunks must be multiples of seq_len=64, so 32 rounds up to 64
    let chunks = batch.into_chunks(32).unwrap();
    assert_eq!(chunks.len(), 8); // 8*64 / 64 = 8 chunks of 64 tokens each
}
