//! File-based dataset implementations for loading data from disk

use crate::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::loader::load_shard;
use super::Dataset;

/// Dataset that loads token sequences from a JSONL file
///
/// Each line should be a JSON array of u32 tokens.
/// Example JSONL content:
/// ```json
/// [0, 1, 2, 3]
/// [4, 5, 6, 7]
/// ```
///
/// Loads entire file into memory. For memory-mapped access, use FileSystemDataset.
///
/// # Example
///
/// ```ignore
/// use rustane::data::{JsonlDataset, Dataset};
///
/// let dataset = JsonlDataset::load("data/tokens.jsonl")?;
/// println!("Loaded {} samples", dataset.len());
/// let sample = dataset.get(0)?;
/// ```
pub struct JsonlDataset {
    samples: Vec<Vec<u32>>,
}

impl JsonlDataset {
    /// Load a JSONL file containing token sequences
    ///
    /// Each line is parsed as a JSON array of u32 integers.
    ///
    /// # Arguments
    /// - `path`: Path to JSONL file
    ///
    /// # Errors
    /// Returns an error if:
    /// - File cannot be opened
    /// - Any line is not valid JSON
    /// - Any JSON array contains non-integer values
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| crate::Error::Io(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                crate::Error::Io(format!("Failed to read line {}: {}", line_num, e))
            })?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let sample: Vec<u32> = serde_json::from_str(&line).map_err(|e| {
                crate::Error::Other(format!("Failed to parse line {}: {}", line_num, e))
            })?;

            samples.push(sample);
        }

        if samples.is_empty() {
            return Err(crate::Error::Other(
                "JSONL file contains no valid samples".to_string(),
            ));
        }

        Ok(JsonlDataset { samples })
    }

    /// Get the inner samples (for testing/debugging)
    pub fn inner(&self) -> &[Vec<u32>] {
        &self.samples
    }
}

impl Dataset for JsonlDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        self.samples.get(idx).cloned().ok_or_else(|| {
            crate::Error::InvalidParameter(format!(
                "dataset index out of bounds: {} >= {}",
                idx,
                self.samples.len()
            ))
        })
    }
}

/// A simple text file dataset where each line is space/comma-separated tokens
///
/// Example format (space-separated):
/// ```text
/// 0 1 2 3 4
/// 5 6 7 8 9
/// ```
///
/// Or comma-separated:
/// ```text
/// 0,1,2,3,4
/// 5,6,7,8,9
/// ```
pub struct TextDataset {
    samples: Vec<Vec<u32>>,
}

impl TextDataset {
    /// Load a text file with space-separated token IDs
    ///
    /// Each line contains token IDs separated by whitespace.
    ///
    /// # Arguments
    /// - `path`: Path to text file
    ///
    /// # Errors
    /// Returns an error if:
    /// - File cannot be opened
    /// - Any token cannot be parsed as u32
    pub fn load_space_separated<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| crate::Error::Io(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                crate::Error::Io(format!("Failed to read line {}: {}", line_num, e))
            })?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let sample: Result<Vec<u32>> = line
                .split_whitespace()
                .map(|token| {
                    token.parse::<u32>().map_err(|_| {
                        crate::Error::InvalidParameter(format!(
                            "Line {}: cannot parse '{}' as u32",
                            line_num, token
                        ))
                    })
                })
                .collect();

            samples.push(sample?);
        }

        if samples.is_empty() {
            return Err(crate::Error::Other(
                "Text file contains no valid samples".to_string(),
            ));
        }

        Ok(TextDataset { samples })
    }

    /// Load a text file with comma-separated token IDs
    ///
    /// Each line contains token IDs separated by commas.
    pub fn load_comma_separated<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| crate::Error::Io(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                crate::Error::Io(format!("Failed to read line {}: {}", line_num, e))
            })?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let sample: Result<Vec<u32>> = line
                .split(',')
                .map(|token| {
                    token.trim().parse::<u32>().map_err(|_| {
                        crate::Error::InvalidParameter(format!(
                            "Line {}: cannot parse '{}' as u32",
                            line_num, token
                        ))
                    })
                })
                .collect();

            samples.push(sample?);
        }

        if samples.is_empty() {
            return Err(crate::Error::Other(
                "Text file contains no valid samples".to_string(),
            ));
        }

        Ok(TextDataset { samples })
    }

    /// Get the inner samples (for testing/debugging)
    pub fn inner(&self) -> &[Vec<u32>] {
        &self.samples
    }
}

impl Dataset for TextDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        self.samples.get(idx).cloned().ok_or_else(|| {
            crate::Error::InvalidParameter(format!(
                "dataset index out of bounds: {} >= {}",
                idx,
                self.samples.len()
            ))
        })
    }
}

/// Dataset backed by parameter-golf binary shard files (.bin format).
///
/// Loads tokens from one or more FineWeb-style shard files and slices them into
/// fixed-length samples suitable for next-token prediction training.
///
/// The shard format matches train_gpt.py / parameter-golf:
/// - Header: 256 × int32 (little-endian). First 3 values: magic=20240520, version=1, num_tokens.
/// - Body: num_tokens × uint16 (little-endian) — token IDs.
///
/// # Example
///
/// ```ignore
/// use rustane::data::{ShardTokenDataset, Dataset};
///
/// let dataset = ShardTokenDataset::from_pattern(
///     "~/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
///     256,  // seq_len
///     Some(2),  // max_shards (for quick testing)
/// )?;
/// println!("Loaded {} samples, {} tokens", dataset.len(), dataset.total_tokens());
/// let sample = dataset.get(0)?;  // Vec<u32> of length seq_len
/// ```
pub struct ShardTokenDataset {
    /// Flattened tokens (u32) from all loaded shards
    tokens: Vec<u32>,
    /// Sequence length for slicing
    seq_len: usize,
    /// Number of samples available (tokens.len() / seq_len)
    num_samples: usize,
    /// Shard file paths that were loaded
    shard_paths: Vec<PathBuf>,
    /// Total tokens across all loaded shards (before slicing)
    total_tokens: usize,
}

impl ShardTokenDataset {
    /// Create a ShardTokenDataset from a glob pattern matching shard files.
    ///
    /// # Arguments
    /// - `pattern`: Glob pattern for shard files (e.g., ".../fineweb_train_*.bin")
    /// - `seq_len`: Sequence length for slicing tokens into samples
    /// - `max_shards`: Optional limit on number of shards to load (None = all)
    ///
    /// # Errors
    /// Returns an error if no shards match the pattern or loading fails.
    pub fn from_pattern(pattern: &str, seq_len: usize, max_shards: Option<usize>) -> Result<Self> {
        if seq_len == 0 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be > 0".to_string(),
            ));
        }

        // Expand ~ in pattern
        let expanded = if pattern.starts_with("~/") {
            let home = std::env::var("HOME")
                .map_err(|e| crate::Error::Other(format!("HOME not set: {}", e)))?;
            format!("{}{}", home, &pattern[1..])
        } else {
            pattern.to_string()
        };

        let mut shard_paths: Vec<PathBuf> = glob::glob(&expanded)
            .map_err(|e| crate::Error::Other(format!("Invalid glob pattern: {}", e)))?
            .filter_map(|entry| entry.ok())
            .collect();

        if shard_paths.is_empty() {
            return Err(crate::Error::Other(format!(
                "No shard files found matching pattern: {}",
                pattern
            )));
        }

        shard_paths.sort();

        if let Some(max) = max_shards {
            shard_paths.truncate(max);
        }

        // Load tokens from all shards
        let mut all_tokens = Vec::new();
        for path in &shard_paths {
            let shard_tokens = load_shard(path).map_err(|e| {
                crate::Error::Other(format!("Failed to load shard {}: {}", path.display(), e))
            })?;
            all_tokens.reserve(shard_tokens.len());
            for &t in &shard_tokens {
                all_tokens.push(t as u32);
            }
        }

        let total_tokens = all_tokens.len();
        let num_samples = total_tokens / seq_len;

        if num_samples == 0 {
            return Err(crate::Error::Other(format!(
                "Not enough tokens ({}) for seq_len ({}). Need at least {} tokens.",
                total_tokens, seq_len, seq_len
            )));
        }

        // Truncate to exact multiple of seq_len
        all_tokens.truncate(num_samples * seq_len);

        Ok(ShardTokenDataset {
            tokens: all_tokens,
            seq_len,
            num_samples,
            shard_paths,
            total_tokens,
        })
    }

    /// Create from a list of explicit shard file paths.
    pub fn from_paths(paths: &[PathBuf], seq_len: usize) -> Result<Self> {
        if seq_len == 0 {
            return Err(crate::Error::InvalidParameter(
                "seq_len must be > 0".to_string(),
            ));
        }
        if paths.is_empty() {
            return Err(crate::Error::Other("No shard paths provided".to_string()));
        }

        let mut all_tokens = Vec::new();
        for path in paths {
            let shard_tokens = load_shard(path).map_err(|e| {
                crate::Error::Other(format!("Failed to load shard {}: {}", path.display(), e))
            })?;
            for &t in &shard_tokens {
                all_tokens.push(t as u32);
            }
        }

        let total_tokens = all_tokens.len();
        let num_samples = total_tokens / seq_len;

        if num_samples == 0 {
            return Err(crate::Error::Other(format!(
                "Not enough tokens ({}) for seq_len ({}).",
                total_tokens, seq_len
            )));
        }

        all_tokens.truncate(num_samples * seq_len);

        Ok(ShardTokenDataset {
            tokens: all_tokens,
            seq_len,
            num_samples,
            shard_paths: paths.to_vec(),
            total_tokens,
        })
    }

    /// Get the number of shards that were loaded.
    pub fn shard_count(&self) -> usize {
        self.shard_paths.len()
    }

    /// Get total tokens loaded (before seq_len slicing).
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get the sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get the shard paths.
    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    /// Get raw token slice for a specific sample (zero-copy).
    pub fn get_slice(&self, idx: usize) -> Option<&[u32]> {
        if idx >= self.num_samples {
            return None;
        }
        let start = idx * self.seq_len;
        Some(&self.tokens[start..start + self.seq_len])
    }
}

impl Dataset for ShardTokenDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        self.get_slice(idx).map(|s| s.to_vec()).ok_or_else(|| {
            crate::Error::InvalidParameter(format!(
                "dataset index out of bounds: {} >= {}",
                idx, self.num_samples
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_jsonl_dataset_basic() -> Result<()> {
        // Create a temporary JSONL file
        let mut file = NamedTempFile::new()
            .map_err(|e| crate::Error::Io(format!("Failed to create temp file: {}", e)))?;

        writeln!(file, "[0, 1, 2]")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;
        writeln!(file, "[3, 4, 5]")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;

        let dataset = JsonlDataset::load(file.path())?;
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0)?, vec![0, 1, 2]);
        assert_eq!(dataset.get(1)?, vec![3, 4, 5]);

        Ok(())
    }

    #[test]
    fn test_jsonl_dataset_empty_lines() -> Result<()> {
        let mut file = NamedTempFile::new()
            .map_err(|e| crate::Error::Io(format!("Failed to create temp file: {}", e)))?;

        writeln!(file, "[1, 2]")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;
        writeln!(file).map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?; // Empty line
        writeln!(file, "[3, 4]")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;

        let dataset = JsonlDataset::load(file.path())?;
        assert_eq!(dataset.len(), 2); // Empty line skipped
        assert_eq!(dataset.get(0)?, vec![1, 2]);
        assert_eq!(dataset.get(1)?, vec![3, 4]);

        Ok(())
    }

    #[test]
    fn test_text_dataset_space_separated() -> Result<()> {
        let mut file = NamedTempFile::new()
            .map_err(|e| crate::Error::Io(format!("Failed to create temp file: {}", e)))?;

        writeln!(file, "0 1 2 3")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;
        writeln!(file, "4 5 6").map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;

        let dataset = TextDataset::load_space_separated(file.path())?;
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0)?, vec![0, 1, 2, 3]);
        assert_eq!(dataset.get(1)?, vec![4, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_text_dataset_comma_separated() -> Result<()> {
        let mut file = NamedTempFile::new()
            .map_err(|e| crate::Error::Io(format!("Failed to create temp file: {}", e)))?;

        writeln!(file, "0,1,2,3")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;
        writeln!(file, "4, 5, 6")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?; // With spaces

        let dataset = TextDataset::load_comma_separated(file.path())?;
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0)?, vec![0, 1, 2, 3]);
        assert_eq!(dataset.get(1)?, vec![4, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_text_dataset_space_with_extra_whitespace() -> Result<()> {
        let mut file = NamedTempFile::new()
            .map_err(|e| crate::Error::Io(format!("Failed to create temp file: {}", e)))?;

        writeln!(file, "  0   1  2  ")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;

        let dataset = TextDataset::load_space_separated(file.path())?;
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.get(0)?, vec![0, 1, 2]);

        Ok(())
    }

    #[test]
    fn test_text_dataset_invalid_token() -> Result<()> {
        let mut file = NamedTempFile::new()
            .map_err(|e| crate::Error::Io(format!("Failed to create temp file: {}", e)))?;

        writeln!(file, "0 1 invalid 3")
            .map_err(|e| crate::Error::Io(format!("Failed to write: {}", e)))?;

        let result = TextDataset::load_space_separated(file.path());
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_jsonl_dataset_not_found() {
        let result = JsonlDataset::load("/nonexistent/path.jsonl");
        assert!(result.is_err());
    }

    #[test]
    fn test_shard_token_dataset_from_paths() -> Result<()> {
        use std::io::Write;

        // Create a temporary shard file in parameter-golf format
        let tmpdir = std::env::temp_dir().join("rustane_test_shard_ds");
        std::fs::create_dir_all(&tmpdir).map_err(|e| crate::Error::Io(e.to_string()))?;

        let shard_path = tmpdir.join("test_shard_000000.bin");
        let mut file = File::create(&shard_path).map_err(|e| crate::Error::Io(e.to_string()))?;

        // Write header: 256 int32s (magic, version, num_tokens, zeros...)
        let num_tokens: i32 = 20;
        let mut header = vec![0i32; 256];
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = num_tokens;
        let header_bytes: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&header_bytes)
            .map_err(|e| crate::Error::Io(e.to_string()))?;

        // Write 20 u16 tokens: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ...]
        let tokens: Vec<u16> = (0..20u16).map(|i| (i + 1) * 10).collect();
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&token_bytes)
            .map_err(|e| crate::Error::Io(e.to_string()))?;
        drop(file);

        // Load with seq_len=5 → should get 4 samples
        let dataset = ShardTokenDataset::from_paths(&[shard_path.clone()], 5)?;
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.total_tokens(), 20);
        assert_eq!(dataset.seq_len(), 5);
        assert_eq!(dataset.shard_count(), 1);

        // Verify first sample
        let sample0 = dataset.get(0)?;
        assert_eq!(sample0, vec![10, 20, 30, 40, 50]);

        // Verify last sample
        let sample3 = dataset.get(3)?;
        assert_eq!(sample3, vec![160, 170, 180, 190, 200]);

        // Out of bounds
        assert!(dataset.get(4).is_err());

        // Verify get_slice (zero-copy)
        let slice = dataset.get_slice(1).unwrap();
        assert_eq!(slice, &[60, 70, 80, 90, 100]);

        // Cleanup
        std::fs::remove_dir_all(&tmpdir).ok();

        Ok(())
    }

    #[test]
    fn test_shard_token_dataset_zero_seq_len() {
        let result = ShardTokenDataset::from_paths(&[PathBuf::from("/nonexistent.bin")], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_shard_token_dataset_empty_paths() {
        let result = ShardTokenDataset::from_paths(&[], 128);
        assert!(result.is_err());
    }
}
