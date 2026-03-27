//! HWX File Loader for ANE Runtime
//!
//! Loads pre-compiled ANE HWX files (Mach-O binaries with ANE operations)
//! extracted from CoreML models. This bypasses the MIL compilation step
//! and the ~119 compile limit.
//!
//! Based on tinygrad's reverse-engineered HWX format:
//! - HWX files are Mach-O binaries
//! - ANE operations start at offset 0x4000
//! - Operations are 0x300 bytes each (register configurations)
//!
//! Reference: tinygrad/extra/accel/ane/

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::ane::error::ANEError;
use crate::ane::kernel::ANEKernel;

/// HWX file header magic numbers
const HWX_MAGIC_MH_CIGAM_64: &[u8] = b"\xcf\xfa\xed\xfe"; // Little-endian 64-bit Mach-O
const HWX_MAGIC_MH_MAGIC_64: &[u8] = b"\xfe\xed\xfa\xcf"; // Big-endian 64-bit Mach-O

/// ANE operations start at this offset in HWX files
const ANE_OPS_OFFSET: usize = 0x4000;

/// Size of each ANE operation
const ANE_OP_SIZE: usize = 0x300;

/// Loaded HWX program
#[derive(Debug, Clone)]
pub struct HWXProgram {
    /// Program name
    pub name: String,
    /// Raw HWX data
    pub data: Vec<u8>,
    /// Extracted ANE operations
    pub operations: Vec<ANEOperation>,
    /// Metadata from Mach-O header
    pub metadata: HWXMetadata,
}

/// HWX file metadata
#[derive(Debug, Clone, Default)]
pub struct HWXMetadata {
    /// Target architecture (h11, h13, etc.)
    pub architecture: String,
    /// ANE hardware version
    pub ane_version: String,
    /// Number of operations
    pub num_operations: usize,
}

/// Single ANE operation (0x300 bytes of register config)
#[derive(Debug, Clone)]
pub struct ANEOperation {
    /// Operation index
    pub index: usize,
    /// Raw operation bytes
    pub data: Vec<u8>,
    /// Operation type (inferred from register values)
    pub op_type: String,
}

/// HWX file loader
pub struct HWXLoader {
    /// Cache of loaded programs
    cache: HashMap<String, HWXProgram>,
    /// Search paths for HWX files
    search_paths: Vec<PathBuf>,
}

impl HWXLoader {
    /// Create a new HWX loader
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            search_paths: vec![
                PathBuf::from("./models/hwx"),
                PathBuf::from("./models/coreml/compiled"),
                PathBuf::from("/usr/local/share/rustane/hwx"),
            ],
        }
    }

    /// Add a search path
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Load a HWX file
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<HWXProgram, ANEError> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Check cache
        if let Some(program) = self.cache.get(&name) {
            return Ok(program.clone());
        }

        // Find the file
        let full_path = if path.exists() {
            path.to_path_buf()
        } else {
            self.find_file(&name)?
        };

        // Read file
        let data = fs::read(&full_path)
            .map_err(|e| ANEError::IOError(format!("Failed to read HWX file: {}", e)))?;

        // Parse HWX
        let program = self.parse_hwx(name, data)?;

        // Cache it
        self.cache.insert(program.name.clone(), program.clone());

        Ok(program)
    }

    /// Find a HWX file in search paths
    fn find_file(&self, name: &str) -> Result<PathBuf, ANEError> {
        for path in &self.search_paths {
            let hwx_path = path.join(format!("{}.hwx", name));
            if hwx_path.exists() {
                return Ok(hwx_path);
            }
        }

        Err(ANEError::HWXNotFound(format!(
            "HWX file '{}' not found in search paths: {:?}",
            name, self.search_paths
        )))
    }

    /// Parse HWX file data
    fn parse_hwx(&self, name: String, data: Vec<u8>) -> Result<HWXProgram, ANEError> {
        // Check magic
        if data.len() < 4 {
            return Err(ANEError::InvalidHWX("File too small".to_string()));
        }

        let is_valid_mach_o =
            &data[0..4] == HWX_MAGIC_MH_CIGAM_64 || &data[0..4] == HWX_MAGIC_MH_MAGIC_64;

        if !is_valid_mach_o {
            // Try to fix byte order (tinygrad does this)
            // Some HWX files have corrupted headers
            return self.parse_hwx_with_fix(name, data);
        }

        // Extract ANE operations
        let operations = self.extract_operations(&data)?;

        // Parse metadata
        let metadata = self.parse_metadata(&data)?;

        Ok(HWXProgram {
            name,
            data,
            operations,
            metadata,
        })
    }

    /// Try to parse HWX with header fix
    fn parse_hwx_with_fix(&self, name: String, mut data: Vec<u8>) -> Result<HWXProgram, ANEError> {
        // tinygrad's fix: replace first 4 bytes with MH_CIGAM_64 magic
        if data.len() >= 4 {
            data[0..4].copy_from_slice(HWX_MAGIC_MH_CIGAM_64);
        }

        // Now try parsing again
        self.parse_hwx(name, data)
    }

    /// Extract ANE operations from HWX data
    fn extract_operations(&self, data: &[u8]) -> Result<Vec<ANEOperation>, ANEError> {
        let mut operations = Vec::new();

        if data.len() < ANE_OPS_OFFSET + ANE_OP_SIZE {
            return Err(ANEError::InvalidHWX(
                "HWX file too small for ANE operations".to_string(),
            ));
        }

        // Operations start at 0x4000
        let ops_data = &data[ANE_OPS_OFFSET..];

        // Parse operations (each is 0x300 bytes)
        for (i, chunk) in ops_data.chunks(ANE_OP_SIZE).enumerate() {
            if chunk.len() < ANE_OP_SIZE {
                // Last chunk might be partial
                break;
            }

            // Infer operation type from register values
            let op_type = self.infer_operation_type(chunk);

            operations.push(ANEOperation {
                index: i,
                data: chunk.to_vec(),
                op_type,
            });
        }

        Ok(operations)
    }

    /// Infer operation type from register values
    fn infer_operation_type(&self, op_data: &[u8]) -> String {
        // Check specific register values to determine op type
        // Based on tinygrad's aneregs.json

        if op_data.len() < 0x250 {
            return "unknown".to_string();
        }

        // Check NeuronType at offset 0x246
        let neuron_type = u16::from_le_bytes([op_data[0x246], op_data[0x247]]);

        match neuron_type {
            0x10 => "copy".to_string(),
            0x11 => "relu".to_string(),
            0x12 => "custom_neuron".to_string(),
            _ => {
                // Check if it's a convolution
                let has_kernel = op_data[0x34] != 0 || op_data[0x35] != 0;
                if has_kernel {
                    "conv".to_string()
                } else {
                    "unknown".to_string()
                }
            }
        }
    }

    /// Parse Mach-O metadata
    fn parse_metadata(&self, data: &[u8]) -> Result<HWXMetadata, ANEError> {
        // This is a simplified parser - full Mach-O parsing is complex
        // For now, extract basic info

        let mut metadata = HWXMetadata::default();

        // Try to find architecture string in the binary
        if let Ok(text) = String::from_utf8(data.clone().to_vec()) {
            if text.contains("h13") {
                metadata.architecture = "h13".to_string();
            } else if text.contains("h11") {
                metadata.architecture = "h11".to_string();
            }
        }

        Ok(metadata)
    }

    /// Convert HWX program to ANE kernel
    pub fn to_kernel(&self, _program: &HWXProgram) -> Result<ANEKernel, ANEError> {
        // TODO: Implement HWX to kernel conversion
        Err(ANEError::ConfigError(
            "HWX to kernel conversion not yet implemented".to_string(),
        ))
    }

    /// Load all HWX files from a directory
    pub fn load_directory<P: AsRef<Path>>(&mut self, dir: P) -> Result<Vec<HWXProgram>, ANEError> {
        let dir = dir.as_ref();
        let mut programs = Vec::new();

        if !dir.exists() {
            return Ok(programs);
        }

        for entry in fs::read_dir(dir)
            .map_err(|e| ANEError::IOError(format!("Failed to read directory: {}", e)))?
        {
            let entry =
                entry.map_err(|e| ANEError::IOError(format!("Failed to read entry: {}", e)))?;

            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("hwx") {
                match self.load(&path) {
                    Ok(program) => programs.push(program),
                    Err(e) => eprintln!("Warning: Failed to load {}: {}", path.display(), e),
                }
            }
        }

        Ok(programs)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (
            self.cache.len(),
            self.cache.values().map(|p| p.data.len()).sum(),
        )
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for HWXLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hwx_loader_new() {
        let loader = HWXLoader::new();
        assert_eq!(loader.search_paths.len(), 3);
    }

    #[test]
    fn test_find_file_not_found() {
        let loader = HWXLoader::new();
        assert!(loader.find_file("nonexistent").is_err());
    }
}
