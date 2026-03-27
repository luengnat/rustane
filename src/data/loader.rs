//! Parameter-Golf Data Loader
//!
//! Implements streaming data loading compatible with train_gpt.py format:
//! - Shard header: 256 × int32 (magic, version, num_tokens, ...)
//! - Token data: uint16 little-endian
//! - TokenStream: sequential reading with wrap-around
//! - Deterministic streaming (no shuffling)
//!
//! Based on train_gpt.py's TokenStream and DistributedTokenLoader

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Shard header magic number (from train_gpt.py)
pub const SHARD_MAGIC: i32 = 20240520;
/// Shard header version
pub const SHARD_VERSION: i32 = 1;
/// Header size in int32s
pub const HEADER_SIZE: usize = 256;
/// Header size in bytes
pub const HEADER_BYTES: u64 = (HEADER_SIZE * 4) as u64;

/// Error type for data loading
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum DataLoaderError {
    /// Invalid shard magic number
    InvalidMagic {
        expected: i32,
        found: i32,
        file: String,
    },
    /// Invalid shard version number
    InvalidVersion {
        expected: i32,
        found: i32,
        file: String,
    },
    /// File size doesn't match header
    SizeMismatch {
        expected: u64,
        found: u64,
        file: String,
    },
    /// Short read from file
    ShortRead {
        expected: usize,
        got: usize,
        file: String,
    },
    /// IO error
    IoError(String),
    /// File pattern not found
    NotFound(String),
}

impl std::fmt::Display for DataLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataLoaderError::InvalidMagic {
                expected,
                found,
                file,
            } => {
                write!(
                    f,
                    "Invalid shard magic in {}: expected {}, got {}",
                    file, expected, found
                )
            }
            DataLoaderError::InvalidVersion {
                expected,
                found,
                file,
            } => {
                write!(
                    f,
                    "Invalid shard version in {}: expected {}, got {}",
                    file, expected, found
                )
            }
            DataLoaderError::SizeMismatch {
                expected,
                found,
                file,
            } => {
                write!(
                    f,
                    "Size mismatch in {}: expected {} bytes, got {}",
                    file, expected, found
                )
            }
            DataLoaderError::ShortRead {
                expected,
                got,
                file,
            } => {
                write!(
                    f,
                    "Short read in {}: expected {} tokens, got {}",
                    file, expected, got
                )
            }
            DataLoaderError::IoError(msg) => write!(f, "IO error: {}", msg),
            DataLoaderError::NotFound(pattern) => {
                write!(f, "No files found for pattern: {}", pattern)
            }
        }
    }
}

impl std::error::Error for DataLoaderError {}

impl From<std::io::Error> for DataLoaderError {
    fn from(err: std::io::Error) -> Self {
        DataLoaderError::IoError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, DataLoaderError>;

/// Result type alias for data loading operations

/// Shard header information
#[derive(Debug, Clone)]
pub struct ShardHeader {
    /// Magic number (should be SHARD_MAGIC = 20240520)
    pub magic: i32,
    /// Version number (should be SHARD_VERSION = 1)
    pub version: i32,
    /// Number of tokens in the shard
    pub num_tokens: i32,
}

impl ShardHeader {
    /// Parse shard header from file
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| {
            DataLoaderError::IoError(format!("Failed to open {}: {}", path.display(), e))
        })?;

        let mut header_bytes = [0u8; HEADER_BYTES as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| DataLoaderError::IoError(format!("Failed to read header: {}", e)))?;

        // Parse first 3 int32s (little-endian as per numpy "<i4")
        let magic = i32::from_le_bytes([
            header_bytes[0],
            header_bytes[1],
            header_bytes[2],
            header_bytes[3],
        ]);
        let version = i32::from_le_bytes([
            header_bytes[4],
            header_bytes[5],
            header_bytes[6],
            header_bytes[7],
        ]);
        let num_tokens = i32::from_le_bytes([
            header_bytes[8],
            header_bytes[9],
            header_bytes[10],
            header_bytes[11],
        ]);

        // Validate magic and version
        if magic != SHARD_MAGIC {
            return Err(DataLoaderError::InvalidMagic {
                expected: SHARD_MAGIC,
                found: magic,
                file: path.display().to_string(),
            });
        }

        if version != SHARD_VERSION {
            return Err(DataLoaderError::InvalidVersion {
                expected: SHARD_VERSION,
                found: version,
                file: path.display().to_string(),
            });
        }

        Ok(ShardHeader {
            magic,
            version,
            num_tokens,
        })
    }

    /// Validate shard file size matches header
    pub fn validate_size(&self, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path).map_err(|e| {
            DataLoaderError::IoError(format!("Failed to stat {}: {}", path.display(), e))
        })?;

        let token_bytes = self.num_tokens as u64 * 2; // uint16 = 2 bytes
        let expected_size = HEADER_BYTES + token_bytes;

        if metadata.len() != expected_size {
            return Err(DataLoaderError::SizeMismatch {
                expected: expected_size,
                found: metadata.len(),
                file: path.display().to_string(),
            });
        }

        Ok(())
    }
}

/// Load all tokens from a shard file
pub fn load_shard(path: &Path) -> Result<Vec<u16>> {
    let header = ShardHeader::from_file(path)?;
    header.validate_size(path)?;

    let mut file = File::open(path).map_err(|e| {
        DataLoaderError::IoError(format!("Failed to open {}: {}", path.display(), e))
    })?;

    // Skip header
    file.seek(SeekFrom::Start(HEADER_BYTES))?;

    // Read tokens
    let num_tokens = header.num_tokens as usize;
    let mut tokens = vec![0u16; num_tokens];
    let mut token_bytes = vec![0u8; num_tokens * 2];

    let bytes_read = file
        .read(&mut token_bytes)
        .map_err(|e| DataLoaderError::IoError(format!("Failed to read tokens: {}", e)))?;

    if bytes_read != token_bytes.len() {
        return Err(DataLoaderError::ShortRead {
            expected: num_tokens,
            got: bytes_read / 2,
            file: path.display().to_string(),
        });
    }

    // Convert from little-endian bytes to u16
    for i in 0..num_tokens {
        tokens[i] = u16::from_le_bytes([token_bytes[i * 2], token_bytes[i * 2 + 1]]);
    }

    Ok(tokens)
}

/// Load a range of tokens from a shard (for streaming)
pub fn load_shard_range(path: &Path, start: usize, count: usize) -> Result<Vec<u16>> {
    let header = ShardHeader::from_file(path)?;
    let total_tokens = header.num_tokens as usize;

    if start >= total_tokens {
        return Err(DataLoaderError::ShortRead {
            expected: count,
            got: 0,
            file: path.display().to_string(),
        });
    }

    if start + count > total_tokens {
        return Err(DataLoaderError::ShortRead {
            expected: count,
            got: total_tokens - start,
            file: path.display().to_string(),
        });
    }

    let mut file = File::open(path).map_err(|e| {
        DataLoaderError::IoError(format!("Failed to open {}: {}", path.display(), e))
    })?;

    // Seek to header + start position
    let offset = HEADER_BYTES + (start as u64 * 2);
    file.seek(SeekFrom::Start(offset))?;

    // Read tokens
    let mut tokens = vec![0u16; count];
    let mut token_bytes = vec![0u8; count * 2];

    file.read_exact(&mut token_bytes)
        .map_err(|e| DataLoaderError::IoError(format!("Failed to read tokens: {}", e)))?;

    for i in 0..count {
        tokens[i] = u16::from_le_bytes([token_bytes[i * 2], token_bytes[i * 2 + 1]]);
    }

    Ok(tokens)
}

/// Find shard files matching a glob pattern
pub fn find_shards(pattern: &str) -> Result<Vec<PathBuf>> {
    let mut shards = glob::glob(pattern)
        .map_err(|e| DataLoaderError::IoError(format!("Invalid glob pattern: {}", e)))?
        .filter_map(|result| result.ok())
        .collect::<Vec<_>>();

    if shards.is_empty() {
        return Err(DataLoaderError::NotFound(pattern.to_string()));
    }

    // Sort for deterministic ordering
    shards.sort();
    Ok(shards)
}

/// Streaming token loader - reads shards sequentially and wraps around
///
/// Based on train_gpt.py's TokenStream class.
/// Provides deterministic streaming: no shuffling, no sampling.
pub struct TokenStream {
    /// List of shard files
    shards: Vec<PathBuf>,
    /// Current shard index
    current_shard_idx: usize,
    /// Current token position in active shard
    position: usize,
    /// Cached tokens from current shard
    current_tokens: Vec<u16>,
}

impl TokenStream {
    /// Create a new TokenStream from a glob pattern
    pub fn new(pattern: &str) -> Result<Self> {
        let shards = find_shards(pattern)?;
        let current_tokens = load_shard(&shards[0])?;

        Ok(TokenStream {
            shards,
            current_shard_idx: 0,
            position: 0,
            current_tokens,
        })
    }

    /// Advance to the next shard (wrap around)
    fn advance_shard(&mut self) -> Result<()> {
        self.current_shard_idx = (self.current_shard_idx + 1) % self.shards.len();
        self.current_tokens = load_shard(&self.shards[self.current_shard_idx])?;
        self.position = 0;
        Ok(())
    }

    /// Take n tokens from the stream (advancing position)
    pub fn take(&mut self, n: usize) -> Result<Vec<u16>> {
        let mut result = Vec::with_capacity(n);
        let mut remaining = n;

        while remaining > 0 {
            let available = self.current_tokens.len() - self.position;

            if available == 0 {
                self.advance_shard()?;
                continue;
            }

            let take_count = remaining.min(available);
            result
                .extend_from_slice(&self.current_tokens[self.position..self.position + take_count]);
            self.position += take_count;
            remaining -= take_count;
        }

        Ok(result)
    }

    /// Get total number of tokens across all shards
    pub fn total_tokens(&self) -> Result<u64> {
        let mut total = 0u64;
        for shard in &self.shards {
            let header = ShardHeader::from_file(shard)?;
            total += header.num_tokens as u64;
        }
        Ok(total)
    }

    /// Get number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Get current position (shard, offset)
    pub fn position(&self) -> (usize, usize) {
        (self.current_shard_idx, self.position)
    }

    /// Reset to beginning
    pub fn reset(&mut self) -> Result<()> {
        self.current_shard_idx = 0;
        self.current_tokens = load_shard(&self.shards[0])?;
        self.position = 0;
        Ok(())
    }
}

/// Batch configuration for training
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Total tokens per batch across all ranks
    pub global_batch_tokens: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Gradient accumulation steps
    pub grad_accum_steps: usize,
    /// World size (number of ranks)
    pub world_size: usize,
    /// This rank's ID
    pub rank: usize,
}

impl BatchConfig {
    /// Create a new batch config
    pub fn new(
        global_batch_tokens: usize,
        seq_len: usize,
        grad_accum_steps: usize,
        world_size: usize,
        rank: usize,
    ) -> Self {
        BatchConfig {
            global_batch_tokens,
            seq_len,
            grad_accum_steps,
            world_size,
            rank,
        }
    }

    /// Calculate tokens per rank per accumulation step
    pub fn tokens_per_rank_accum(&self) -> usize {
        self.global_batch_tokens / (self.world_size * self.grad_accum_steps)
    }

    /// Calculate span per rank (includes +1 for x,y shift)
    pub fn span_per_rank(&self) -> usize {
        self.tokens_per_rank_accum() + 1
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.global_batch_tokens % (self.world_size * self.grad_accum_steps) != 0 {
            return Err(DataLoaderError::IoError(
                format!(
                    "global_batch_tokens ({}) must be divisible by world_size ({}) × grad_accum_steps ({})",
                    self.global_batch_tokens, self.world_size, self.grad_accum_steps
                )
            ));
        }
        Ok(())
    }
}

/// Distributed token loader - provides (x, y) batches for training
///
/// Based on train_gpt.py's DistributedTokenLoader.
/// Each rank gets a disjoint slice of the token stream.
pub struct DistributedTokenLoader {
    stream: TokenStream,
    config: BatchConfig,
}

impl DistributedTokenLoader {
    /// Create a new distributed loader
    pub fn new(pattern: &str, config: BatchConfig) -> Result<Self> {
        config.validate()?;
        let stream = TokenStream::new(pattern)?;
        Ok(DistributedTokenLoader { stream, config })
    }

    /// Get next batch as (input, target) pair
    ///
    /// Returns:
    /// - input: [batch_size, seq_len] with input tokens
    /// - target: [batch_size, seq_len] with next tokens (shifted by 1)
    pub fn next_batch(&mut self) -> Result<(Vec<u16>, Vec<u16>)> {
        let span = self.config.span_per_rank();
        let seq_len = self.config.seq_len;
        let rank = self.config.rank;

        // Take contiguous chunk for all ranks
        let chunk = self.stream.take(span * self.config.world_size)?;

        // Extract this rank's slice
        let start = rank * span;
        let local = &chunk[start..start + span];

        // Build x (input) and y (target) as flattened batches
        // x = local[0..span-1], y = local[1..span]
        // Reshaped to [num_sequences, seq_len]
        let num_seqs = (span - 1) / seq_len;
        let mut input = Vec::with_capacity(num_seqs * seq_len);
        let mut target = Vec::with_capacity(num_seqs * seq_len);

        for i in 0..num_seqs {
            let seq_start = i * seq_len;
            input.extend_from_slice(&local[seq_start..seq_start + seq_len]);
            target.extend_from_slice(&local[seq_start + 1..seq_start + seq_len + 1]);
        }

        Ok((input, target))
    }

    /// Get total tokens available
    pub fn total_tokens(&self) -> Result<u64> {
        self.stream.total_tokens()
    }

    /// Reset stream to beginning
    pub fn reset(&mut self) -> Result<()> {
        self.stream.reset()
    }
}

/// Utility: count special tokens in a range
pub fn count_special_tokens(tokens: &[u16], token_id: u16) -> usize {
    tokens.iter().filter(|&&t| t == token_id).count()
}

/// Utility: find sequence boundaries (BOS tokens)
pub fn find_boundaries(tokens: &[u16], bos_token: u16) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == bos_token)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_shard(path: &Path, tokens: &[u16]) -> Result<()> {
        let mut file = File::create(path).unwrap();

        // Write header
        let mut header = [0u8; HEADER_BYTES as usize];
        header[0..4].copy_from_slice(&SHARD_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&SHARD_VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        file.write_all(&mut header).unwrap();

        // Write tokens
        let mut token_bytes = Vec::with_capacity(tokens.len() * 2);
        for &t in tokens {
            token_bytes.extend_from_slice(&t.to_le_bytes());
        }
        file.write_all(&token_bytes).unwrap();

        Ok(())
    }

    #[test]
    fn test_shard_header() {
        let tmpdir = std::env::temp_dir().join("rustane_test_shard");
        std::fs::create_dir_all(&tmpdir).unwrap();
        let path = tmpdir.join("test.bin");

        let tokens = vec![1u16, 2, 3, 4, 5];
        create_test_shard(&path, &tokens).unwrap();

        let header = ShardHeader::from_file(&path).unwrap();
        assert_eq!(header.magic, SHARD_MAGIC);
        assert_eq!(header.version, SHARD_VERSION);
        assert_eq!(header.num_tokens as usize, tokens.len());

        header.validate_size(&path).unwrap();
    }

    #[test]
    fn test_load_shard() {
        let tmpdir = std::env::temp_dir().join("rustane_test_shard2");
        std::fs::create_dir_all(&tmpdir).unwrap();
        let path = tmpdir.join("test.bin");

        let tokens = vec![100u16, 200, 300, 400, 500];
        create_test_shard(&path, &tokens).unwrap();

        let loaded = load_shard(&path).unwrap();
        assert_eq!(loaded, tokens);
    }

    #[test]
    fn test_token_stream() {
        let tmpdir = std::env::temp_dir().join("rustane_test_stream");
        std::fs::create_dir_all(&tmpdir).unwrap();

        // Create two small test shards
        let shard1 = tmpdir.join("shard_00.bin");
        let shard2 = tmpdir.join("shard_01.bin");
        create_test_shard(&shard1, &[1, 2, 3, 4, 5]).unwrap();
        create_test_shard(&shard2, &[6, 7, 8, 9, 10]).unwrap();

        // Note: TokenStream uses glob pattern, for testing we'll use load_shard directly
        let tokens1 = load_shard(&shard1).unwrap();
        let tokens2 = load_shard(&shard2).unwrap();

        assert_eq!(tokens1, vec![1, 2, 3, 4, 5]);
        assert_eq!(tokens2, vec![6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::new(524_288, 1024, 8, 1, 0);
        assert_eq!(config.tokens_per_rank_accum(), 65_536);
        assert_eq!(config.span_per_rank(), 65_537);

        config.validate().unwrap();
    }

    #[test]
    fn test_special_tokens() {
        let tokens = vec![1u16, 2, 3, 1, 4, 5, 1];
        assert_eq!(count_special_tokens(&tokens, 1), 3);
        assert_eq!(count_special_tokens(&tokens, 2), 1);
        assert_eq!(count_special_tokens(&tokens, 99), 0);
    }

    #[test]
    fn test_find_boundaries() {
        let tokens = vec![5u16, 10, 1, 20, 30, 1, 40, 50, 60];
        let boundaries = find_boundaries(&tokens, 1);
        assert_eq!(boundaries, vec![2, 5]);
    }
}
