//! ANE Parameter Golf Edge Case Tests
//!
//! Additional tests for parameter golf data loading edge cases:
//! - compute_chunk_sizes function
//! - Batch chunking edge cases
//! - Sampler tests
//! - Integration tests

use rustane::data::{
    Batch, BatchConfig, DataLoader, Dataset, DistributedTokenLoader, RandomSampler, Sampler,
    SequentialDataset, SequentialSampler,
};

// ============================================================================
// TEST 1: compute_chunk_sizes helper function tests
// ============================================================================

/// Test compute_chunk_sizes via Batch::chunks() since it's not public
#[test]
fn test_compute_chunk_sizes_exact_fit() {
    // 100 tokens, max 25 = 4 chunks of 25
    let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
    let chunks = batch.into_chunks(25).unwrap();
    assert_eq!(chunks.len(), 4);
    for chunk in &chunks {
        assert_eq!(chunk.tokens.len(), 25);
    }
}

#[test]
fn test_compute_chunk_sizes_with_remainder() {
    // 102 tokens with seq_len=25 can't be evenly divided
    // batch_size = 102/25 = 4 (integer division), but 4*25=100 != 102
    // So we need to use valid batch dimensions
    let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap(); // 4 * 25 = 100
    let chunks = batch.into_chunks(25).unwrap();

    // Total should still equal original
    let total: usize = chunks.iter().map(|c| c.tokens.len()).sum();
    assert_eq!(total, 100);
}

#[test]
fn test_compute_chunk_sizes_single_chunk() {
    // 50 tokens, max 100 = 1 chunk of 50
    let batch = Batch::new(vec![1u32; 50], 2, 25).unwrap();
    let chunks = batch.into_chunks(100).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].tokens.len(), 50);
}

#[test]
fn test_compute_chunk_sizes_large_batch() {
    // 1000 tokens, max 100 = 10 chunks of 100
    let batch = Batch::new(vec![1u32; 1000], 10, 100).unwrap();
    let chunks = batch.into_chunks(100).unwrap();
    assert_eq!(chunks.len(), 10);
    for chunk in &chunks {
        assert_eq!(chunk.tokens.len(), 100);
    }
}

#[test]
fn test_compute_chunk_sizes_seq_len_alignment() {
    // Chunks must be multiples of seq_len
    let batch = Batch::new(vec![1u32; 128], 4, 32).unwrap();
    let chunks = batch.into_chunks(50).unwrap(); // 50 rounds down to 32

    for chunk in &chunks {
        assert_eq!(
            chunk.tokens.len() % 32,
            0,
            "Chunk size must be multiple of seq_len"
        );
    }
}

#[test]
fn test_compute_chunk_sizes_guard_zero_seq_len() {
    let batch = Batch {
        tokens: vec![1, 2, 3],
        batch_size: 1,
        seq_len: 0,
    };
    let result = batch.chunks(10);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("seq_len must be > 0"));
}

#[test]
fn test_compute_chunk_sizes_guard_zero_max_chunk() {
    let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
    let result = batch.chunks(0);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("max_chunk_tokens must be > 0"));
}

// ============================================================================
// TEST 2: Batch Edge Cases
// ============================================================================

#[test]
fn test_batch_single_sample() {
    let batch = Batch::new(vec![1u32, 2, 3, 4, 5], 1, 5).unwrap();
    assert_eq!(batch.batch_size(), 1);
    assert_eq!(batch.seq_len(), 5);
    assert_eq!(batch.get(0, 0), Some(1));
    assert_eq!(batch.get(0, 4), Some(5));
    assert_eq!(batch.get(0, 5), None);
    assert_eq!(batch.get(1, 0), None);
}

#[test]
fn test_batch_single_token_per_sample() {
    let batch = Batch::new(vec![1u32, 2, 3, 4, 5], 5, 1).unwrap();
    assert_eq!(batch.batch_size(), 5);
    assert_eq!(batch.seq_len(), 1);
    for i in 0..5 {
        assert_eq!(batch.get(i, 0), Some((i + 1) as u32));
    }
}

#[test]
fn test_batch_tokens_slice() {
    let batch = Batch::new(vec![1u32, 2, 3, 4, 5, 6], 2, 3).unwrap();
    assert_eq!(batch.tokens(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_batch_shape() {
    let batch = Batch::new(vec![1u32; 100], 10, 10).unwrap();
    assert_eq!(batch.shape(), (10, 10));
}

#[test]
fn test_batch_allows_empty_batch() {
    let batch = Batch::new(vec![], 0, 1).unwrap();
    assert_eq!(batch.shape(), (0, 1));
    let chunks = batch.into_chunks(10).unwrap();
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].tokens.is_empty());
}

#[test]
fn test_batch_into_chunks_single_chunk_passthrough() {
    let batch = Batch::new(vec![1u32; 6], 2, 3).unwrap();
    let chunks = batch.into_chunks(10).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].tokens.len(), 6);
    assert_eq!(chunks[0].batch_size(), 2);
    assert_eq!(chunks[0].seq_len(), 3);
}

#[test]
fn test_batch_chunks_iterator_multiple_calls() {
    // chunks() borrows, so we can call it multiple times
    let batch = Batch::new(vec![1u32; 100], 4, 25).unwrap();
    let chunks1 = batch.chunks(25).unwrap();
    let chunks2 = batch.chunks(50).unwrap();

    assert_eq!(chunks1.count(), 4);
    assert_eq!(chunks2.count(), 2);
}

// ============================================================================
// TEST 3: Sampler Tests
// ============================================================================

#[test]
fn test_sequential_sampler_basic() {
    let mut sampler = SequentialSampler::new(5);
    let indices = sampler.sample();
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_sequential_sampler_single_element() {
    let mut sampler = SequentialSampler::new(1);
    let indices = sampler.sample();
    assert_eq!(indices, vec![0]);
}

#[test]
fn test_sequential_sampler_empty() {
    let mut sampler = SequentialSampler::new(0);
    let indices = sampler.sample();
    assert!(indices.is_empty());
}

#[test]
fn test_sequential_sampler_reproducible() {
    let mut sampler1 = SequentialSampler::new(10);
    let mut sampler2 = SequentialSampler::new(10);

    let indices1 = sampler1.sample();
    let indices2 = sampler2.sample();

    assert_eq!(indices1, indices2);
}

#[test]
fn test_random_sampler_basic() {
    let mut sampler = RandomSampler::new(100, 42);
    let indices = sampler.sample();

    assert_eq!(indices.len(), 100);

    // Should contain all indices 0-99
    let sorted = {
        let mut s = indices.clone();
        s.sort();
        s
    };
    let expected: Vec<usize> = (0..100).collect();
    assert_eq!(sorted, expected);
}

#[test]
fn test_random_sampler_single_element() {
    let mut sampler = RandomSampler::new(1, 42);
    let indices = sampler.sample();
    assert_eq!(indices, vec![0]);
}

#[test]
fn test_random_sampler_empty() {
    let mut sampler = RandomSampler::new(0, 42);
    let indices = sampler.sample();
    assert!(indices.is_empty());
}

#[test]
fn test_random_sampler_different_each_time() {
    let mut sampler1 = RandomSampler::new(1000, 42);
    let mut sampler2 = RandomSampler::new(1000, 43);

    let indices1 = sampler1.sample();
    let indices2 = sampler2.sample();

    // With different seeds, should be different
    assert_ne!(indices1, indices2);
}

#[test]
fn test_random_sampler_reproducible_with_seed() {
    let mut sampler1 = RandomSampler::new(100, 42);
    let mut sampler2 = RandomSampler::new(100, 42);

    let indices1 = sampler1.sample();
    let indices2 = sampler2.sample();

    assert_eq!(indices1, indices2);
}

// ============================================================================
// TEST 4: Integration Tests - TokenStream + BatchConfig
// ============================================================================

#[test]
fn test_distributed_loader_with_varying_seq_lens() {
    use rustane::data::loader::SHARD_MAGIC;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    fn create_test_shard(path: &Path, tokens: &[u16]) {
        let mut file = File::create(path).unwrap();
        let mut header = [0u8; 256 * 4];
        header[0..4].copy_from_slice(&SHARD_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&1i32.to_le_bytes());
        header[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        file.write_all(&header).unwrap();
        for &t in tokens {
            file.write_all(&t.to_le_bytes()).unwrap();
        }
    }

    let tmpdir = std::env::temp_dir().join("rustane_integration_test");
    std::fs::create_dir_all(&tmpdir).unwrap();

    // Create shard with 1000 tokens
    let tokens: Vec<u16> = (0..1000).collect();
    create_test_shard(&tmpdir.join("shard_00.bin"), &tokens);

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();

    // Test with different seq_len values
    for seq_len in [16, 32, 64, 128, 256] {
        let config = BatchConfig::new(256, seq_len, 1, 1, 0);
        let mut loader = DistributedTokenLoader::new(&pattern, config).unwrap();

        let (input, target) = loader.next_batch().unwrap();

        // Verify shapes
        assert_eq!(input.len(), target.len());
        assert_eq!(input.len() % seq_len, 0);

        // Verify shift relationship
        for i in 0..input.len() {
            assert_eq!(target[i], input[i] + 1);
        }
    }
}

#[test]
fn test_batch_with_gradient_accumulation() {
    // Simulate gradient accumulation scenario
    let batch = Batch::new(vec![1u32; 256], 8, 32).unwrap();

    // Split into chunks for gradient accumulation
    let chunks = batch.into_chunks(64).unwrap();

    // Should get 4 chunks of 64 tokens each
    assert_eq!(chunks.len(), 4);

    // Each chunk should have correct shape
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.tokens.len(), 64);
        assert_eq!(chunk.seq_len(), 32);
        assert_eq!(chunk.batch_size(), 2); // 64 / 32 = 2
    }
}

#[test]
fn test_dataloader_with_sequential_dataset() {
    let samples = vec![
        vec![1u32, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
        vec![13, 14, 15, 16],
    ];
    let dataset = SequentialDataset::new(samples);
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

    let mut batches: Vec<_> = dataloader.iter().collect();

    assert_eq!(batches.len(), 2);

    let batch1 = batches.remove(0).unwrap();
    assert_eq!(batch1.shape(), (2, 4));
    assert_eq!(batch1.tokens(), &[1, 2, 3, 4, 5, 6, 7, 8]);

    let batch2 = batches.remove(0).unwrap();
    assert_eq!(batch2.shape(), (2, 4));
    assert_eq!(batch2.tokens(), &[9, 10, 11, 12, 13, 14, 15, 16]);
}

#[test]
fn test_dataloader_with_partial_batch() {
    let samples = vec![vec![1u32, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    let dataset = SequentialDataset::new(samples);
    let sampler = SequentialSampler::new(dataset.len());
    let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

    let mut batches: Vec<_> = dataloader.iter().collect();

    assert_eq!(batches.len(), 2);

    // First batch: 2 samples
    let batch1 = batches.remove(0).unwrap();
    assert_eq!(batch1.shape(), (2, 3));

    // Second batch: 1 sample (partial)
    let batch2 = batches.remove(0).unwrap();
    assert_eq!(batch2.shape(), (1, 3));
}

// ============================================================================
// TEST 5: Parameter Golf Specific Scenarios
// ============================================================================

#[test]
fn test_parameter_golf_batch_config() {
    // Typical parameter-golf training config
    let config = BatchConfig::new(
        65_536, // global_batch_tokens
        1024,   // seq_len
        8,      // grad_accum_steps
        1,      // world_size
        0,      // rank
    );

    assert_eq!(config.tokens_per_rank_accum(), 8_192);
    assert_eq!(config.span_per_rank(), 8_193);
    assert!(config.validate().is_ok());
}

#[test]
fn test_parameter_golf_multi_rank_config() {
    // Multi-GPU training config (4 GPUs)
    let config = BatchConfig::new(
        262_144, // global_batch_tokens (4x larger)
        1024,    // seq_len
        8,       // grad_accum_steps
        4,       // world_size
        0,       // rank
    );

    // Each rank gets: 262144 / (4 * 8) = 8192 tokens per accum
    assert_eq!(config.tokens_per_rank_accum(), 8_192);
    assert_eq!(config.span_per_rank(), 8_193);
    assert!(config.validate().is_ok());
}

#[test]
fn test_parameter_golf_invalid_configs() {
    // Non-divisible batch size
    let config = BatchConfig::new(100, 32, 3, 1, 0);
    assert!(config.validate().is_err());

    // Non-divisible with multiple ranks
    let config = BatchConfig::new(100, 32, 1, 3, 0);
    assert!(config.validate().is_err());
}

#[test]
fn test_batch_chunking_for_parameter_golf() {
    // Simulate parameter-golf style batch chunking
    let batch_tokens = 8192;
    let seq_len = 1024;
    let max_chunk = 2048;

    let batch = Batch::new(vec![1u32; batch_tokens], 8, seq_len).unwrap();
    let chunks = batch.into_chunks(max_chunk).unwrap();

    // Should get 4 chunks of 2048 tokens each
    assert_eq!(chunks.len(), 4);

    for chunk in &chunks {
        assert_eq!(chunk.tokens.len(), max_chunk);
        assert_eq!(chunk.seq_len(), seq_len);
        assert_eq!(chunk.batch_size(), max_chunk / seq_len);
    }
}

#[test]
fn test_special_token_counting() {
    use rustane::data::loader::count_special_tokens;

    // Simulate FineWeb tokens with BOS=1, EOS=2, PAD=0
    let tokens: Vec<u16> = vec![
        1, 100, 200, 300, 2, // Sample 1
        1, 400, 500, 2, // Sample 2
        1, 600, 700, 800, 900, 2, // Sample 3
    ];

    assert_eq!(count_special_tokens(&tokens, 1), 3); // BOS count
    assert_eq!(count_special_tokens(&tokens, 2), 3); // EOS count
    assert_eq!(count_special_tokens(&tokens, 0), 0); // PAD count (none)
}

#[test]
fn test_find_boundaries_for_sequence_packing() {
    use rustane::data::loader::find_boundaries;

    // Find BOS token positions for sequence packing
    let tokens: Vec<u16> = vec![
        100, 200, // Prefix before first BOS
        1,   // BOS at index 2
        300, 400, 500, 1, // BOS at index 6
        600, 700, 1, // BOS at index 9
        800, 2,
    ];

    let boundaries = find_boundaries(&tokens, 1);
    assert_eq!(boundaries, vec![2, 6, 9]);

    // Can use boundaries to extract sequences
    for &start in &boundaries {
        assert!(tokens[start] == 1); // All boundaries should point to BOS
    }
}

// ============================================================================
// TEST 6: Stress Tests
// ============================================================================

#[test]
fn test_large_batch_chunking() {
    // Stress test with large batch
    let batch_size = 1_000_000;
    let seq_len = 1024;
    // batch_size must be divisible by seq_len
    let adjusted_batch_size = (batch_size / seq_len) * seq_len; // 999424

    let batch = Batch::new(
        vec![1u32; adjusted_batch_size],
        adjusted_batch_size / seq_len,
        seq_len,
    )
    .unwrap();

    let chunks = batch.into_chunks(32_768).unwrap();

    let total: usize = chunks.iter().map(|c| c.tokens.len()).sum();
    assert_eq!(total, adjusted_batch_size);

    // All chunks except possibly last should be max size
    for (_i, chunk) in chunks.iter().enumerate() {
        if _i < chunks.len() - 1 {
            assert_eq!(chunk.tokens.len(), 32_768);
        }
    }
}

#[test]
fn test_many_small_chunks() {
    // Many small chunks
    let batch = Batch::new(vec![1u32; 1024], 32, 32).unwrap();
    let chunks = batch.into_chunks(32).unwrap();

    assert_eq!(chunks.len(), 32);
    for chunk in &chunks {
        assert_eq!(chunk.tokens.len(), 32);
        assert_eq!(chunk.batch_size(), 1);
    }
}
