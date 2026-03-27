//! ANE Parameter Golf Data Loader Tests
//!
//! Comprehensive tests for parameter golf data loading infrastructure:
//! - ShardHeader validation
//! - Token streaming
//! - Batch generation
//! - Distributed loading

use rustane::data::loader::{
    count_special_tokens, find_boundaries, find_shards, load_shard, load_shard_range, BatchConfig,
    DataLoaderError, DistributedTokenLoader, ShardHeader, TokenStream, HEADER_BYTES, SHARD_MAGIC,
    SHARD_VERSION,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ============================================================================
// TEST HELPERS
// ============================================================================

/// Create a test shard file with the given tokens
fn create_test_shard(path: &Path, tokens: &[u16]) -> Result<(), DataLoaderError> {
    let mut file = File::create(path).unwrap();

    // Write header
    let mut header = [0u8; HEADER_BYTES as usize];
    header[0..4].copy_from_slice(&SHARD_MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&SHARD_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
    file.write_all(&header).unwrap();

    // Write tokens
    let mut token_bytes = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        token_bytes.extend_from_slice(&t.to_le_bytes());
    }
    file.write_all(&token_bytes).unwrap();

    Ok(())
}

/// Create a test shard with custom header values (for error testing)
fn create_test_shard_with_header(
    path: &Path,
    magic: i32,
    version: i32,
    num_tokens: i32,
    tokens: &[u16],
) -> Result<(), DataLoaderError> {
    let mut file = File::create(path).unwrap();

    let mut header = [0u8; HEADER_BYTES as usize];
    header[0..4].copy_from_slice(&magic.to_le_bytes());
    header[4..8].copy_from_slice(&version.to_le_bytes());
    header[8..12].copy_from_slice(&num_tokens.to_le_bytes());
    file.write_all(&header).unwrap();

    let mut token_bytes = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        token_bytes.extend_from_slice(&t.to_le_bytes());
    }
    file.write_all(&token_bytes).unwrap();

    Ok(())
}

// ============================================================================
// TEST 1: ShardHeader Tests
// ============================================================================

#[test]
fn test_shard_header_valid() {
    let tmpdir = std::env::temp_dir().join("rustane_shard_header_test");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let tokens = vec![1u16, 2, 3, 4, 5];
    create_test_shard(&path, &tokens).unwrap();

    let header = ShardHeader::from_file(&path).unwrap();
    assert_eq!(header.magic, SHARD_MAGIC);
    assert_eq!(header.version, SHARD_VERSION);
    assert_eq!(header.num_tokens as usize, tokens.len());
}

#[test]
fn test_shard_header_invalid_magic() {
    let tmpdir = std::env::temp_dir().join("rustane_invalid_magic");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let tokens = vec![1u16, 2, 3];
    create_test_shard_with_header(&path, 99999, SHARD_VERSION, 3, &tokens).unwrap();

    let result = ShardHeader::from_file(&path);
    assert!(result.is_err());

    if let Err(DataLoaderError::InvalidMagic {
        expected,
        found,
        file,
    }) = result
    {
        assert_eq!(expected, SHARD_MAGIC);
        assert_eq!(found, 99999);
        assert!(file.contains("test.bin"));
    } else {
        panic!("Expected InvalidMagic error");
    }
}

#[test]
fn test_shard_header_invalid_version() {
    let tmpdir = std::env::temp_dir().join("rustane_invalid_version");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let tokens = vec![1u16, 2, 3];
    create_test_shard_with_header(&path, SHARD_MAGIC, 999, 3, &tokens).unwrap();

    let result = ShardHeader::from_file(&path);
    assert!(result.is_err());

    if let Err(DataLoaderError::InvalidVersion {
        expected,
        found,
        file,
    }) = result
    {
        assert_eq!(expected, SHARD_VERSION);
        assert_eq!(found, 999);
        assert!(file.contains("test.bin"));
    } else {
        panic!("Expected InvalidVersion error");
    }
}

#[test]
fn test_shard_header_size_mismatch() {
    let tmpdir = std::env::temp_dir().join("rustane_size_mismatch");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    // Header says 100 tokens, but file only has 5
    let tokens = vec![1u16, 2, 3, 4, 5];
    create_test_shard_with_header(&path, SHARD_MAGIC, SHARD_VERSION, 100, &tokens).unwrap();

    let header = ShardHeader::from_file(&path).unwrap();
    let result = header.validate_size(&path);

    assert!(result.is_err());
    if let Err(DataLoaderError::SizeMismatch {
        expected,
        found,
        file,
    }) = result
    {
        assert!(expected > found); // Expected more bytes
        assert!(file.contains("test.bin"));
    } else {
        panic!("Expected SizeMismatch error");
    }
}

#[test]
fn test_shard_header_file_not_found() {
    let path = PathBuf::from("/nonexistent/path/to/shard.bin");
    let result = ShardHeader::from_file(&path);
    assert!(result.is_err());
}

#[test]
fn test_shard_header_clone() {
    let tmpdir = std::env::temp_dir().join("rustane_clone_test");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let tokens = vec![1u16, 2, 3];
    create_test_shard(&path, &tokens).unwrap();

    let header1 = ShardHeader::from_file(&path).unwrap();
    let header2 = header1.clone();

    assert_eq!(header1.magic, header2.magic);
    assert_eq!(header1.version, header2.version);
    assert_eq!(header1.num_tokens, header2.num_tokens);
}

#[test]
fn test_shard_header_debug() {
    let tmpdir = std::env::temp_dir().join("rustane_debug_test");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let tokens = vec![1u16, 2, 3];
    create_test_shard(&path, &tokens).unwrap();

    let header = ShardHeader::from_file(&path).unwrap();
    let debug_str = format!("{:?}", header);

    assert!(debug_str.contains("ShardHeader"));
    assert!(debug_str.contains(&SHARD_MAGIC.to_string()));
}

// ============================================================================
// TEST 2: load_shard Tests
// ============================================================================

#[test]
fn test_load_shard_basic() {
    let tmpdir = std::env::temp_dir().join("rustane_load_shard");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original = vec![100u16, 200, 300, 400, 500, 600];
    create_test_shard(&path, &original).unwrap();

    let loaded = load_shard(&path).unwrap();
    assert_eq!(loaded, original);
}

#[test]
fn test_load_shard_empty() {
    let tmpdir = std::env::temp_dir().join("rustane_empty_shard");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original = vec![];
    create_test_shard(&path, &original).unwrap();

    let loaded = load_shard(&path).unwrap();
    assert!(loaded.is_empty());
    assert_eq!(loaded.len(), 0);
}

#[test]
fn test_load_shard_large() {
    let tmpdir = std::env::temp_dir().join("rustane_large_shard");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    // Create a larger shard (10K tokens)
    let original: Vec<u16> = (0..10_000).map(|i| (i % 1024) as u16).collect();
    create_test_shard(&path, &original).unwrap();

    let loaded = load_shard(&path).unwrap();
    assert_eq!(loaded.len(), 10_000);
    assert_eq!(loaded, original);
}

#[test]
fn test_load_shard_file_not_found() {
    let path = PathBuf::from("/nonexistent/path/to/shard.bin");
    let result = load_shard(&path);
    assert!(result.is_err());
}

#[test]
fn test_load_shard_all_token_ranges() {
    let tmpdir = std::env::temp_dir().join("rustane_ranges");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    // Test with tokens at various ranges
    let tokens = vec![0u16, 1, 2, 511, 512, 1022, 1023];
    create_test_shard(&path, &tokens).unwrap();

    let loaded = load_shard(&path).unwrap();
    assert_eq!(loaded, tokens);

    // Verify all tokens are in valid vocab range
    for &t in &loaded {
        assert!(t < 1024, "Token {} out of vocab range", t);
    }
}

// ============================================================================
// TEST 3: load_shard_range Tests
// ============================================================================

#[test]
fn test_load_shard_range_basic() {
    let tmpdir = std::env::temp_dir().join("rustane_range_basic");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original: Vec<u16> = (0..100).map(|i| i as u16).collect();
    create_test_shard(&path, &original).unwrap();

    // Load middle range
    let loaded = load_shard_range(&path, 25, 50).unwrap();
    assert_eq!(loaded.len(), 50);
    assert_eq!(loaded[0], 25);
    assert_eq!(loaded[49], 74);
}

#[test]
fn test_load_shard_range_from_start() {
    let tmpdir = std::env::temp_dir().join("rustane_range_start");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original: Vec<u16> = (100..200).map(|i| i as u16).collect();
    create_test_shard(&path, &original).unwrap();

    let loaded = load_shard_range(&path, 0, 10).unwrap();
    assert_eq!(loaded.len(), 10);
    assert_eq!(loaded, (100..110).map(|i| i as u16).collect::<Vec<_>>());
}

#[test]
fn test_load_shard_range_to_end() {
    let tmpdir = std::env::temp_dir().join("rustane_range_end");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original: Vec<u16> = (0..50).map(|i| i as u16).collect();
    create_test_shard(&path, &original).unwrap();

    let loaded = load_shard_range(&path, 40, 10).unwrap();
    assert_eq!(loaded.len(), 10);
    assert_eq!(loaded, (40..50).map(|i| i as u16).collect::<Vec<_>>());
}

#[test]
fn test_load_shard_range_out_of_bounds() {
    let tmpdir = std::env::temp_dir().join("rustane_range_oob");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original: Vec<u16> = (0..100).map(|i| i as u16).collect();
    create_test_shard(&path, &original).unwrap();

    // Start beyond file
    let result = load_shard_range(&path, 200, 10);
    assert!(result.is_err());

    // Range extends beyond file
    let result = load_shard_range(&path, 90, 20);
    assert!(result.is_err());
}

#[test]
fn test_load_shard_range_zero_count() {
    let tmpdir = std::env::temp_dir().join("rustane_range_zero");
    std::fs::create_dir_all(&tmpdir).unwrap();
    let path = tmpdir.join("test.bin");

    let original: Vec<u16> = (0..100).map(|i| i as u16).collect();
    create_test_shard(&path, &original).unwrap();

    let loaded = load_shard_range(&path, 50, 0).unwrap();
    assert!(loaded.is_empty());
}

// ============================================================================
// TEST 4: find_shards Tests
// ============================================================================

#[test]
fn test_find_shards_basic() {
    let tmpdir = std::env::temp_dir().join("rustane_find_shards");
    std::fs::create_dir_all(&tmpdir).unwrap();

    // Create test files
    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2, 3]).unwrap();
    create_test_shard(&tmpdir.join("shard_01.bin"), &[4, 5, 6]).unwrap();
    create_test_shard(&tmpdir.join("shard_02.bin"), &[7, 8, 9]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let shards = find_shards(&pattern).unwrap();

    assert_eq!(shards.len(), 3);
    assert!(shards[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_00"));
    assert!(shards[1]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_01"));
    assert!(shards[2]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_02"));
}

#[test]
fn test_find_shards_sorted() {
    let tmpdir = std::env::temp_dir().join("rustane_find_sorted");
    std::fs::create_dir_all(&tmpdir).unwrap();

    // Create files in non-sorted order
    create_test_shard(&tmpdir.join("shard_05.bin"), &[]).unwrap();
    create_test_shard(&tmpdir.join("shard_01.bin"), &[]).unwrap();
    create_test_shard(&tmpdir.join("shard_10.bin"), &[]).unwrap();
    create_test_shard(&tmpdir.join("shard_02.bin"), &[]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let shards = find_shards(&pattern).unwrap();

    // Should be sorted
    assert!(shards[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_01"));
    assert!(shards[1]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_02"));
    assert!(shards[2]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_05"));
    assert!(shards[3]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("shard_10"));
}

#[test]
fn test_find_shards_empty_pattern() {
    let tmpdir = std::env::temp_dir().join("rustane_find_empty");
    std::fs::create_dir_all(&tmpdir).unwrap();

    let pattern = tmpdir
        .join("nonexistent_*.bin")
        .to_string_lossy()
        .to_string();
    let result = find_shards(&pattern);

    assert!(result.is_err());
    if let Err(DataLoaderError::NotFound(msg)) = result {
        assert!(msg.contains("nonexistent"));
    } else {
        panic!("Expected NotFound error");
    }
}

#[test]
fn test_find_shards_invalid_pattern() {
    let result = find_shards("[invalid");
    assert!(result.is_err());
}

// ============================================================================
// TEST 5: TokenStream Tests
// ============================================================================

#[test]
fn test_token_stream_creation() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_create");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2, 3, 4, 5]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let stream = TokenStream::new(&pattern).unwrap();

    assert_eq!(stream.num_shards(), 1);
    assert_eq!(stream.position(), (0, 0));
}

#[test]
fn test_token_stream_take_basic() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_take");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[10, 20, 30, 40, 50]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let mut stream = TokenStream::new(&pattern).unwrap();

    let tokens = stream.take(3).unwrap();
    assert_eq!(tokens, vec![10, 20, 30]);
    assert_eq!(stream.position(), (0, 3));
}

#[test]
fn test_token_stream_take_all() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_all");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2, 3, 4, 5]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let mut stream = TokenStream::new(&pattern).unwrap();

    let tokens = stream.take(5).unwrap();
    assert_eq!(tokens, vec![1, 2, 3, 4, 5]);
    assert_eq!(stream.position(), (0, 5));
}

#[test]
fn test_token_stream_multi_shard() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_multi");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2, 3]).unwrap();
    create_test_shard(&tmpdir.join("shard_01.bin"), &[4, 5, 6]).unwrap();
    create_test_shard(&tmpdir.join("shard_02.bin"), &[7, 8, 9]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let mut stream = TokenStream::new(&pattern).unwrap();

    // Take across shard boundaries
    let tokens = stream.take(7).unwrap();
    assert_eq!(tokens, vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(stream.position(), (2, 1)); // shard 2, offset 1
}

#[test]
fn test_token_stream_wrap_around() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_wrap");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2]).unwrap();
    create_test_shard(&tmpdir.join("shard_01.bin"), &[3, 4]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let mut stream = TokenStream::new(&pattern).unwrap();

    // Take more tokens than available - should wrap around
    let tokens = stream.take(6).unwrap();
    assert_eq!(tokens, vec![1, 2, 3, 4, 1, 2]);
}

#[test]
fn test_token_stream_reset() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_reset");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[10, 20, 30]).unwrap();
    create_test_shard(&tmpdir.join("shard_01.bin"), &[40, 50, 60]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let mut stream = TokenStream::new(&pattern).unwrap();

    // Advance stream
    let _ = stream.take(4).unwrap();
    assert_eq!(stream.position(), (1, 1));

    // Reset
    stream.reset().unwrap();
    assert_eq!(stream.position(), (0, 0));

    // Should start from beginning
    let tokens = stream.take(3).unwrap();
    assert_eq!(tokens, vec![10, 20, 30]);
}

#[test]
fn test_token_stream_total_tokens() {
    let tmpdir = std::env::temp_dir().join("rustane_stream_total");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2, 3, 4, 5]).unwrap();
    create_test_shard(&tmpdir.join("shard_01.bin"), &[6, 7, 8, 9, 10]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let stream = TokenStream::new(&pattern).unwrap();

    let total = stream.total_tokens().unwrap();
    assert_eq!(total, 10);
}

// ============================================================================
// TEST 6: BatchConfig Tests
// ============================================================================

#[test]
fn test_batch_config_basic() {
    let config = BatchConfig::new(524_288, 1024, 8, 1, 0);

    assert_eq!(config.global_batch_tokens, 524_288);
    assert_eq!(config.seq_len, 1024);
    assert_eq!(config.grad_accum_steps, 8);
    assert_eq!(config.world_size, 1);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_batch_config_tokens_per_rank_accum() {
    // tokens_per_rank_accum = global_batch_tokens / (world_size * grad_accum_steps)
    let config = BatchConfig::new(65_536, 1024, 8, 1, 0);
    assert_eq!(config.tokens_per_rank_accum(), 8_192); // 65536 / (1 * 8) = 8192

    let config2 = BatchConfig::new(65_536, 1024, 8, 4, 0);
    assert_eq!(config2.tokens_per_rank_accum(), 2_048); // 65536 / (4 * 8) = 2048

    // Test with smaller values
    let config3 = BatchConfig::new(8192, 1024, 1, 1, 0);
    assert_eq!(config3.tokens_per_rank_accum(), 8192); // 8192 / (1 * 1) = 8192
}

#[test]
fn test_batch_config_span_per_rank() {
    let config = BatchConfig::new(65_536, 1024, 8, 1, 0);
    // span = tokens_per_rank_accum + 1 (for x,y shift)
    // tokens_per_rank_accum = 65536 / 8 = 8192, so span = 8193
    assert_eq!(config.span_per_rank(), 8_193);

    // Test with smaller values
    let config2 = BatchConfig::new(8192, 1024, 1, 1, 0);
    assert_eq!(config2.span_per_rank(), 8193);
}

#[test]
fn test_batch_config_valid() {
    let config = BatchConfig::new(524_288, 1024, 8, 1, 0);
    assert!(config.validate().is_ok());

    let config2 = BatchConfig::new(32_768, 512, 4, 2, 0);
    assert!(config2.validate().is_ok());
}

#[test]
fn test_batch_config_invalid_not_divisible() {
    // 100 is not divisible by 3 * 2 = 6
    let config = BatchConfig::new(100, 10, 2, 3, 0);
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_batch_config_single_rank() {
    let config = BatchConfig::new(8192, 1024, 1, 1, 0);
    assert_eq!(config.tokens_per_rank_accum(), 8192);
    assert_eq!(config.span_per_rank(), 8193);
    assert!(config.validate().is_ok());
}

#[test]
fn test_batch_config_multi_rank() {
    let config = BatchConfig::new(32_768, 1024, 8, 4, 0);

    // Each rank gets: 32768 / (4 * 8) = 1024 tokens per accum
    assert_eq!(config.tokens_per_rank_accum(), 1024);
    assert_eq!(config.span_per_rank(), 1025);
    assert!(config.validate().is_ok());
}

#[test]
fn test_batch_config_clone() {
    let config1 = BatchConfig::new(16_384, 512, 4, 2, 1);
    let config2 = config1.clone();

    assert_eq!(config1.global_batch_tokens, config2.global_batch_tokens);
    assert_eq!(config1.rank, config2.rank);
}

#[test]
fn test_batch_config_debug() {
    let config = BatchConfig::new(8192, 1024, 8, 1, 0);
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("BatchConfig"));
    assert!(debug_str.contains("8192"));
}

// ============================================================================
// TEST 7: DistributedTokenLoader Tests
// ============================================================================

#[test]
fn test_distributed_loader_creation() {
    let tmpdir = std::env::temp_dir().join("rustane_dist_loader");
    std::fs::create_dir_all(&tmpdir).unwrap();

    create_test_shard(&tmpdir.join("shard_00.bin"), &[1, 2, 3, 4, 5]).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let config = BatchConfig::new(4, 2, 1, 1, 0);

    let loader = DistributedTokenLoader::new(&pattern, config).unwrap();
    assert!(loader.total_tokens().unwrap() > 0);
}

#[test]
fn test_distributed_loader_single_batch() {
    let tmpdir = std::env::temp_dir().join("rustane_dist_batch");
    std::fs::create_dir_all(&tmpdir).unwrap();

    // Create shard with enough tokens for a batch
    let tokens: Vec<u16> = (0..100).collect();
    create_test_shard(&tmpdir.join("shard_00.bin"), &tokens).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let config = BatchConfig::new(16, 2, 1, 1, 0); // 16 tokens total, seq_len=2, 1 accum, 1 rank

    let mut loader = DistributedTokenLoader::new(&pattern, config).unwrap();
    let (input, target) = loader.next_batch().unwrap();

    // input should have num_seqs * seq_len tokens where num_seqs = (span-1) / seq_len
    // span = 16 + 1 = 17, num_seqs = 16 / 2 = 8, so input has 8 * 2 = 16 tokens
    assert_eq!(input.len(), 16);
    assert_eq!(target.len(), 16);

    // Verify shift relationship: target[i] = input[i] + 1
    for i in 0..input.len() {
        assert_eq!(target[i], input[i] + 1);
    }
}

#[test]
fn test_distributed_loader_multi_rank_simulation() {
    let tmpdir = std::env::temp_dir().join("rustane_dist_multi");
    std::fs::create_dir_all(&tmpdir).unwrap();

    // Create large shard for multi-rank test
    let tokens: Vec<u16> = (0..1000).collect();
    create_test_shard(&tmpdir.join("shard_00.bin"), &tokens).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();

    // Simulate 4 ranks - batch size 32, 4 ranks, 1 accum = 8 tokens per rank + 1 = span 9
    let config = BatchConfig::new(32, 4, 1, 4, 0);

    let mut loader_rank0 = DistributedTokenLoader::new(&pattern, config.clone()).unwrap();
    let config_rank1 = BatchConfig::new(32, 4, 1, 4, 1);
    let mut loader_rank1 = DistributedTokenLoader::new(&pattern, config_rank1).unwrap();

    let (input0, _target0) = loader_rank0.next_batch().unwrap();
    let (input1, _target1) = loader_rank1.next_batch().unwrap();

    // Each rank should get different tokens
    // num_seqs = 8 / 4 = 2 seqs per rank, each seq is 4 tokens = 8 tokens total per rank
    assert_eq!(input0.len(), 8);
    assert_eq!(input1.len(), 8);

    // Ranks should get disjoint slices
    assert_ne!(input0[0], input1[0]);
}

#[test]
fn test_distributed_loader_reset() {
    let tmpdir = std::env::temp_dir().join("rustane_dist_reset");
    std::fs::create_dir_all(&tmpdir).unwrap();

    let tokens: Vec<u16> = (0..50).collect();
    create_test_shard(&tmpdir.join("shard_00.bin"), &tokens).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let config = BatchConfig::new(8, 2, 1, 1, 0);

    let mut loader = DistributedTokenLoader::new(&pattern, config).unwrap();

    // Get first batch
    let (input1, _) = loader.next_batch().unwrap();

    // Reset
    loader.reset().unwrap();

    // Get first batch again - should be the same
    let (input2, _) = loader.next_batch().unwrap();
    assert_eq!(input1, input2);
}

#[test]
fn test_distributed_loader_sequential_batches() {
    let tmpdir = std::env::temp_dir().join("rustane_dist_seq");
    std::fs::create_dir_all(&tmpdir).unwrap();

    let tokens: Vec<u16> = (0..100).collect();
    create_test_shard(&tmpdir.join("shard_00.bin"), &tokens).unwrap();

    let pattern = tmpdir.join("shard_*.bin").to_string_lossy().to_string();
    let config = BatchConfig::new(12, 2, 1, 1, 0);

    let mut loader = DistributedTokenLoader::new(&pattern, config).unwrap();

    let (batch1_input, batch1_target) = loader.next_batch().unwrap();
    let (batch2_input, _batch2_target) = loader.next_batch().unwrap();
    let (batch3_input, _batch3_target) = loader.next_batch().unwrap();

    // Each batch gets span-1 tokens / seq_len sequences * seq_len
    // span = 12 + 1 = 13, num_seqs = 12/2 = 6, each batch = 6*2 = 12 tokens
    assert_eq!(batch1_input.len(), 12);
    assert_eq!(batch2_input.len(), 12);
    assert_eq!(batch3_input.len(), 12);

    // Batches should be sequential
    // Batch 1: chunk=[0-12], input=[0-11], target=[1-12]
    // Batch 2: chunk=[13-25], input=[13-24], target=[14-25]
    // Batch 3: chunk=[26-38], input=[26-37], target=[27-38]
    assert_eq!(&batch1_input[..4], &[0, 1, 2, 3]);
    assert_eq!(&batch2_input[..4], &[13, 14, 15, 16]);
    assert_eq!(&batch3_input[..4], &[26, 27, 28, 29]);

    // Targets should be shifted by 1
    assert_eq!(&batch1_target[..4], &[1, 2, 3, 4]);
}

// ============================================================================
// TEST 8: Utility Function Tests
// ============================================================================

#[test]
fn test_count_special_tokens() {
    let tokens = vec![1u16, 2, 3, 1, 4, 5, 1, 2, 2];

    // BOS = 1
    assert_eq!(count_special_tokens(&tokens, 1), 3);
    // EOS = 2
    assert_eq!(count_special_tokens(&tokens, 2), 3);
    // PAD = 0 (not present)
    assert_eq!(count_special_tokens(&tokens, 0), 0);
    // Non-existent token
    assert_eq!(count_special_tokens(&tokens, 999), 0);
}

#[test]
fn test_count_special_tokens_empty() {
    let tokens: Vec<u16> = vec![];
    assert_eq!(count_special_tokens(&tokens, 1), 0);
}

#[test]
fn test_count_special_tokens_all_same() {
    let tokens = vec![1u16, 1, 1, 1, 1];
    assert_eq!(count_special_tokens(&tokens, 1), 5);
}

#[test]
fn test_find_boundaries() {
    let tokens = vec![5u16, 10, 1, 20, 30, 1, 40, 50, 60];

    let boundaries = find_boundaries(&tokens, 1);
    assert_eq!(boundaries, vec![2, 5]);
}

#[test]
fn test_find_boundaries_no_matches() {
    let tokens = vec![5u16, 10, 15, 20, 25];

    let boundaries = find_boundaries(&tokens, 999);
    assert!(boundaries.is_empty());
}

#[test]
fn test_find_boundaries_at_start() {
    let tokens = vec![1u16, 10, 15, 1, 25, 30];

    let boundaries = find_boundaries(&tokens, 1);
    assert_eq!(boundaries, vec![0, 3]);
}

#[test]
fn test_find_boundaries_at_end() {
    let tokens = vec![5u16, 10, 15, 20, 25, 1];

    let boundaries = find_boundaries(&tokens, 1);
    assert_eq!(boundaries, vec![5]);
}

#[test]
fn test_find_boundaries_empty() {
    let tokens: Vec<u16> = vec![];

    let boundaries = find_boundaries(&tokens, 1);
    assert!(boundaries.is_empty());
}

// ============================================================================
// TEST 9: Error Handling Edge Cases
// ============================================================================

#[test]
fn test_invalid_magic_error_display() {
    let err = DataLoaderError::InvalidMagic {
        expected: SHARD_MAGIC,
        found: 99999,
        file: "test.bin".to_string(),
    };

    let msg = err.to_string();
    assert!(msg.contains("Invalid shard magic"));
    assert!(msg.contains("expected"));
    assert!(msg.contains("99999"));
    assert!(msg.contains("test.bin"));
}

#[test]
fn test_invalid_version_error_display() {
    let err = DataLoaderError::InvalidVersion {
        expected: SHARD_VERSION,
        found: 999,
        file: "test.bin".to_string(),
    };

    let msg = err.to_string();
    assert!(msg.contains("Invalid shard version"));
}

#[test]
fn test_size_mismatch_error_display() {
    let err = DataLoaderError::SizeMismatch {
        expected: 1000,
        found: 500,
        file: "test.bin".to_string(),
    };

    let msg = err.to_string();
    assert!(msg.contains("Size mismatch"));
    assert!(msg.contains("1000"));
    assert!(msg.contains("500"));
}

#[test]
fn test_short_read_error_display() {
    let err = DataLoaderError::ShortRead {
        expected: 100,
        got: 50,
        file: "test.bin".to_string(),
    };

    let msg = err.to_string();
    assert!(msg.contains("Short read"));
}

#[test]
fn test_io_error_display() {
    let err = DataLoaderError::IoError("connection failed".to_string());
    assert_eq!(err.to_string(), "IO error: connection failed");
}

#[test]
fn test_not_found_error_display() {
    let err = DataLoaderError::NotFound("*.bin".to_string());
    assert!(err.to_string().contains("No files found"));
}

#[test]
fn test_data_loader_error_is_variant() {
    // Verify all error variants are distinct
    let magic_err = DataLoaderError::InvalidMagic {
        expected: 1,
        found: 2,
        file: "a.bin".to_string(),
    };
    let version_err = DataLoaderError::InvalidVersion {
        expected: 1,
        found: 2,
        file: "b.bin".to_string(),
    };
    let size_err = DataLoaderError::SizeMismatch {
        expected: 100,
        found: 50,
        file: "c.bin".to_string(),
    };
    let short_err = DataLoaderError::ShortRead {
        expected: 10,
        got: 5,
        file: "d.bin".to_string(),
    };
    let io_err = DataLoaderError::IoError("test".to_string());
    let not_found_err = DataLoaderError::NotFound("*.bin".to_string());

    assert_ne!(format!("{:?}", magic_err), format!("{:?}", version_err));
    assert_ne!(format!("{:?}", size_err), format!("{:?}", short_err));
    assert_ne!(format!("{:?}", io_err), format!("{:?}", not_found_err));
}
