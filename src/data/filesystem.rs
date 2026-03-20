//! File-based dataset implementations for loading data from disk

use crate::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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
}
