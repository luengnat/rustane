//! ANE Kernel Cache for Compiled Programs
//!
//! Provides disk-based caching of compiled ANE kernels to:
//! - Avoid recompilation overhead (~4-5 seconds per kernel)
//! - Stay within the ~119 compilation budget per process
//! - Enable faster startup for repeated workloads
//!
//! # Cache Key Generation
//!
//! Cache keys are generated using SHA-256 hashes of:
//! - MIL source code
//! - Weight data (all blobs concatenated)
//! - Input/output tensor shapes
//! - Compilation options
//!
//! # Quick Start
//!
//! ```no_run
//! use rustane::ane::kernel_cache::{KernelCache, CacheConfig};
//!
//! // Create cache with default config
//! let cache = KernelCache::new("/tmp/ane_cache")?;
//!
//! // Generate cache key for MIL program
//! let key = cache.generate_key(mil_code, &weights, &input_sizes, &output_sizes);
//!
//! // Try to load from cache
//! if let Some(program) = cache.load(&key)? {
//!     // Use cached program
//! } else {
//!     // Compile and store
//!     let program = compile_mil(mil_code)?;
//!     cache.store(&key, &program)?;
//! }
//! ```
//!
//! # Cache Eviction
//!
//! The cache uses LRU (Least Recently Used) eviction when:
//! - Cache size exceeds `max_cache_size_bytes`
//! - Individual file age exceeds `max_file_age_days`
//!
//! Eviction runs automatically on cache creation and periodically during operation.

use crate::error::Result;
use crate::Error;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default cache directory name (relative to project temp dir)
#[allow(dead_code)]
const DEFAULT_CACHE_DIR: &str = "ane_kernel_cache";

/// Cache file extension
const CACHE_FILE_EXT: &str = "hwxcache";

/// Metadata file extension
const METADATA_FILE_EXT: &str = "meta";

/// Maximum cache entry age (7 days by default)
const DEFAULT_MAX_FILE_AGE_DAYS: u64 = 7;

/// Maximum total cache size (1 GB by default)
const DEFAULT_MAX_CACHE_SIZE_BYTES: u64 = 1024 * 1024 * 1024;

/// Magic number for cache file format validation
const CACHE_MAGIC: u32 = 0x414E4543; // "ANEC" in little endian

/// Cache file format version
const CACHE_VERSION: u16 = 1;

/// Configuration for kernel cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum total cache size in bytes
    pub max_cache_size_bytes: u64,
    /// Maximum age of cache entries in days
    pub max_file_age_days: u64,
    /// Enable cache statistics tracking
    pub enable_stats: bool,
    /// Enable automatic eviction on cache creation
    pub auto_evict: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size_bytes: DEFAULT_MAX_CACHE_SIZE_BYTES,
            max_file_age_days: DEFAULT_MAX_FILE_AGE_DAYS,
            enable_stats: true,
            auto_evict: true,
        }
    }
}

impl CacheConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum cache size in megabytes
    pub fn with_max_size_mb(mut self, mb: u64) -> Self {
        self.max_cache_size_bytes = mb * 1024 * 1024;
        self
    }

    /// Set maximum file age in days
    pub fn with_max_age_days(mut self, days: u64) -> Self {
        self.max_file_age_days = days;
        self
    }

    /// Enable or disable statistics tracking
    pub fn with_stats(mut self, enable: bool) -> Self {
        self.enable_stats = enable;
        self
    }

    /// Enable or disable automatic eviction
    pub fn with_auto_evict(mut self, enable: bool) -> Self {
        self.auto_evict = enable;
        self
    }
}

/// Metadata for a cached kernel
#[derive(Debug, Clone)]
pub struct CacheEntryMetadata {
    /// Cache key (SHA-256 hex)
    pub key: String,
    /// Size of cached data in bytes
    pub size_bytes: u64,
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,
    /// Last access timestamp (Unix epoch seconds)
    pub last_accessed: u64,
    /// Access count (for LRU+LFU hybrid)
    pub access_count: u32,
    /// MIL source hash (for validation)
    pub mil_hash: String,
    /// Weights hash (for validation)
    pub weights_hash: String,
    /// Compilation options hash
    pub options_hash: String,
}

impl CacheEntryMetadata {
    /// Create new metadata with current timestamp
    pub fn new(
        key: String,
        size_bytes: u64,
        mil_hash: String,
        weights_hash: String,
        options_hash: String,
    ) -> Self {
        let now = current_timestamp();
        Self {
            key,
            size_bytes,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            mil_hash,
            weights_hash,
            options_hash,
        }
    }

    /// Update access time and count
    pub fn touch(&mut self) {
        self.last_accessed = current_timestamp();
        self.access_count += 1;
    }

    /// Serialize metadata to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Write magic and version
        buf.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
        buf.extend_from_slice(&CACHE_VERSION.to_le_bytes());

        // Write metadata fields
        write_string(&mut buf, &self.key);
        write_string(&mut buf, &self.mil_hash);
        write_string(&mut buf, &self.weights_hash);
        write_string(&mut buf, &self.options_hash);
        buf.extend_from_slice(&self.size_bytes.to_le_bytes());
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.last_accessed.to_le_bytes());
        buf.extend_from_slice(&self.access_count.to_le_bytes());

        buf
    }

    /// Deserialize metadata from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;

        // Read and validate magic
        if bytes.len() < 4 {
            return Err(Error::Io("Metadata too short for magic".to_string()));
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != CACHE_MAGIC {
            return Err(Error::Io(format!("Invalid cache magic: 0x{:08X}", magic)));
        }
        offset += 4;

        // Read version
        if bytes.len() < offset + 2 {
            return Err(Error::Io("Metadata too short for version".to_string()));
        }
        let _version = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        offset += 2;

        // Read strings
        let (key, len) = read_string(&bytes[offset..])?;
        offset += len;

        let (mil_hash, len) = read_string(&bytes[offset..])?;
        offset += len;

        let (weights_hash, len) = read_string(&bytes[offset..])?;
        offset += len;

        let (options_hash, len) = read_string(&bytes[offset..])?;
        offset += len;

        // Read numeric fields
        if bytes.len() < offset + 24 {
            return Err(Error::Io(
                "Metadata too short for numeric fields".to_string(),
            ));
        }
        let size_bytes = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        let created_at = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        let last_accessed = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        let access_count = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);

        Ok(Self {
            key,
            size_bytes,
            created_at,
            last_accessed,
            access_count,
            mil_hash,
            weights_hash,
            options_hash,
        })
    }
}

/// Statistics for cache operations
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache writes
    pub writes: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current cache size in bytes
    pub current_size_bytes: u64,
    /// Number of cached entries
    pub entry_count: u64,
    /// Total bytes saved by cache hits
    pub bytes_saved: u64,
}

impl CacheStats {
    /// Get hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get cache efficiency (bytes_saved / total_bytes_read)
    pub fn efficiency(&self) -> f64 {
        let total_read = self.bytes_saved + self.current_size_bytes;
        if total_read == 0 {
            0.0
        } else {
            self.bytes_saved as f64 / total_read as f64
        }
    }
}

/// ANE Kernel Cache for storing compiled HWX programs
pub struct KernelCache {
    /// Cache directory path
    cache_dir: PathBuf,
    /// Cache configuration
    config: CacheConfig,
    /// In-memory metadata index
    metadata_index: HashMap<String, CacheEntryMetadata>,
    /// Cache statistics
    stats: CacheStats,
}

impl KernelCache {
    /// Create a new kernel cache at the specified directory
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        Self::with_config(cache_dir, CacheConfig::default())
    }

    /// Create a new kernel cache with custom configuration
    pub fn with_config<P: AsRef<Path>>(cache_dir: P, config: CacheConfig) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::Io(format!("Failed to create cache directory: {}", e)))?;

        let mut cache = Self {
            cache_dir,
            config: config.clone(),
            metadata_index: HashMap::new(),
            stats: CacheStats::default(),
        };

        // Load existing metadata index
        cache.load_metadata_index()?;

        // Run eviction if enabled
        if config.auto_evict {
            cache.evict_old_entries()?;
        }

        // Update stats
        cache.update_size_stats();

        Ok(cache)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Generate a cache key from MIL code, weights, and options
    pub fn generate_key(
        &self,
        mil_code: &str,
        weights: &HashMap<String, Vec<u8>>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> String {
        let mut hasher = Sha256::new();

        // Hash MIL code
        hasher.update(mil_code.as_bytes());
        let mil_hash = hasher.finalize_reset();

        // Hash weights (sorted by name for determinism)
        let mut weight_names: Vec<_> = weights.keys().collect();
        weight_names.sort();
        for name in weight_names {
            hasher.update(name.as_bytes());
            hasher.update(weights.get(name).unwrap());
        }
        let weights_hash = hasher.finalize_reset();

        // Hash tensor shapes
        for &size in input_sizes {
            hasher.update(size.to_le_bytes());
        }
        for &size in output_sizes {
            hasher.update(size.to_le_bytes());
        }
        let options_hash = hasher.finalize();

        // Combine hashes into final key
        let mut final_hasher = Sha256::new();
        final_hasher.update(mil_hash);
        final_hasher.update(weights_hash);
        final_hasher.update(options_hash);

        hex::encode(final_hasher.finalize())
    }

    /// Generate component hashes separately (for metadata)
    fn generate_component_hashes(
        &self,
        mil_code: &str,
        weights: &HashMap<String, Vec<u8>>,
    ) -> (String, String) {
        let mil_hash = hex::encode(Sha256::digest(mil_code.as_bytes()));

        let mut weights_hasher = Sha256::new();
        let mut weight_names: Vec<_> = weights.keys().collect();
        weight_names.sort();
        for name in weight_names {
            weights_hasher.update(name.as_bytes());
            weights_hasher.update(weights.get(name).unwrap());
        }
        let weights_hash = hex::encode(weights_hasher.finalize());

        (mil_hash, weights_hash)
    }

    /// Try to load a cached program
    pub fn load(&mut self, key: &str) -> Result<Option<Vec<u8>>> {
        let cache_file = self.cache_file_path(key);

        if !cache_file.exists() {
            self.stats.misses += 1;
            return Ok(None);
        }

        // Read cached data efficiently
        let data = fs::read(&cache_file)
            .map_err(|e| Error::Io(format!("Failed to read cache file: {}", e)))?;

        // Update access time in metadata (lazy update - only touch in-memory)
        if let Some(metadata) = self.metadata_index.get_mut(key) {
            metadata.touch();
            // Defer disk write to avoid I/O on every cache hit
        }

        self.stats.hits += 1;
        self.stats.bytes_saved += data.len() as u64;

        Ok(Some(data))
    }

    /// Store a compiled program in the cache
    pub fn store(
        &mut self,
        key: &str,
        program_data: &[u8],
        mil_code: &str,
        weights: &HashMap<String, Vec<u8>>,
    ) -> Result<()> {
        // Run eviction if needed before storing
        if self.stats.current_size_bytes + program_data.len() as u64
            > self.config.max_cache_size_bytes
        {
            self.evict_old_entries()?;
        }

        let cache_file = self.cache_file_path(key);
        let meta_file = self.metadata_file_path(key);

        // Write cache file
        let mut file = BufWriter::new(
            File::create(&cache_file)
                .map_err(|e| Error::Io(format!("Failed to create cache file: {}", e)))?,
        );
        file.write_all(program_data)
            .map_err(|e| Error::Io(format!("Failed to write cache data: {}", e)))?;
        file.flush()
            .map_err(|e| Error::Io(format!("Failed to flush cache data: {}", e)))?;

        // Generate component hashes
        let (mil_hash, weights_hash) = self.generate_component_hashes(mil_code, weights);

        // Create and save metadata
        let metadata = CacheEntryMetadata::new(
            key.to_string(),
            program_data.len() as u64,
            mil_hash,
            weights_hash,
            String::new(), // options_hash
        );

        self.save_metadata(&meta_file, &metadata)?;
        self.metadata_index.insert(key.to_string(), metadata);

        self.stats.writes += 1;
        self.update_size_stats();

        Ok(())
    }

    /// Check if a key exists in the cache
    pub fn contains(&self, key: &str) -> bool {
        self.metadata_index.contains_key(key) && self.cache_file_path(key).exists()
    }

    /// Remove an entry from the cache
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        let cache_file = self.cache_file_path(key);
        let meta_file = self.metadata_file_path(key);

        let mut removed = false;

        if cache_file.exists() {
            fs::remove_file(&cache_file)
                .map_err(|e| Error::Io(format!("Failed to remove cache file: {}", e)))?;
            removed = true;
        }

        if meta_file.exists() {
            fs::remove_file(&meta_file)
                .map_err(|e| Error::Io(format!("Failed to remove metadata file: {}", e)))?;
        }

        if self.metadata_index.remove(key).is_some() {
            self.stats.evictions += 1;
        }

        self.update_size_stats();

        Ok(removed)
    }

    /// Clear all cache entries
    pub fn clear(&mut self) -> Result<()> {
        let keys: Vec<String> = self.metadata_index.keys().cloned().collect();
        for key in keys {
            let _ = self.remove(&key);
        }
        self.metadata_index.clear();
        self.stats = CacheStats::default();
        Ok(())
    }

    /// Flush all pending metadata updates to disk
    /// Call this periodically or before shutdown to persist access times
    pub fn flush_metadata(&mut self) -> Result<()> {
        for key in self.metadata_index.keys().cloned().collect::<Vec<_>>() {
            self.save_metadata_entry(&key)?;
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get number of cached entries
    pub fn entry_count(&self) -> usize {
        self.metadata_index.len()
    }

    /// Get current cache size in bytes
    pub fn size_bytes(&self) -> u64 {
        self.stats.current_size_bytes
    }

    /// Get the cache directory path
    pub fn dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Evict old entries based on age and LRU
    pub fn evict_old_entries(&mut self) -> Result<()> {
        let now = current_timestamp();
        let max_age_seconds = self.config.max_file_age_days * 24 * 60 * 60;

        // Find entries to evict
        let mut to_evict: Vec<(String, u64)> = Vec::new(); // (key, score)

        for (key, meta) in &self.metadata_index {
            let age = now - meta.last_accessed;
            let score = age * (meta.access_count as u64 + 1); // Higher = evict first

            // Always evict if too old
            if age > max_age_seconds {
                to_evict.push((key.clone(), u64::MAX));
            } else {
                to_evict.push((key.clone(), score));
            }
        }

        // Sort by score (highest first)
        to_evict.sort_by(|a, b| b.1.cmp(&a.1));

        // Evict until under size limit
        while self.stats.current_size_bytes > self.config.max_cache_size_bytes {
            if let Some((key, _)) = to_evict.pop() {
                self.remove(&key)?;
            } else {
                break;
            }
        }

        Ok(())
    }

    // ========== Private Methods ==========

    fn cache_file_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.{}", key, CACHE_FILE_EXT))
    }

    fn metadata_file_path(&self, key: &str) -> PathBuf {
        self.cache_dir
            .join(format!("{}.{}", key, METADATA_FILE_EXT))
    }

    fn load_metadata_index(&mut self) -> Result<()> {
        let _pattern = format!("*.{}", METADATA_FILE_EXT);
        let entries = fs::read_dir(&self.cache_dir)
            .map_err(|e| Error::Io(format!("Failed to read cache dir: {}", e)))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .map_or(false, |ext| ext == METADATA_FILE_EXT)
            {
                if let Some(key) = path.file_stem().and_then(|s| s.to_str()) {
                    match self.load_metadata_entry(key) {
                        Ok(Some(meta)) => {
                            self.metadata_index.insert(key.to_string(), meta);
                        }
                        Ok(None) => {}
                        Err(_) => {
                            // Ignore corrupted metadata
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn load_metadata_entry(&self, key: &str) -> Result<Option<CacheEntryMetadata>> {
        let meta_file = self.metadata_file_path(key);
        if !meta_file.exists() {
            return Ok(None);
        }

        let mut file = BufReader::new(
            File::open(&meta_file)
                .map_err(|e| Error::Io(format!("Failed to open metadata file: {}", e)))?,
        );
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| Error::Io(format!("Failed to read metadata file: {}", e)))?;

        let metadata = CacheEntryMetadata::from_bytes(&bytes)?;
        Ok(Some(metadata))
    }

    fn save_metadata(&self, path: &Path, metadata: &CacheEntryMetadata) -> Result<()> {
        let mut file = BufWriter::new(
            File::create(path)
                .map_err(|e| Error::Io(format!("Failed to create metadata file: {}", e)))?,
        );
        file.write_all(&metadata.to_bytes())
            .map_err(|e| Error::Io(format!("Failed to write metadata: {}", e)))?;
        file.flush()
            .map_err(|e| Error::Io(format!("Failed to flush metadata: {}", e)))?;
        Ok(())
    }

    fn save_metadata_entry(&self, key: &str) -> Result<()> {
        if let Some(metadata) = self.metadata_index.get(key) {
            let meta_file = self.metadata_file_path(key);
            self.save_metadata(&meta_file, metadata)?;
        }
        Ok(())
    }

    fn update_size_stats(&mut self) {
        self.stats.current_size_bytes = 0;
        self.stats.entry_count = 0;

        for (_, meta) in &self.metadata_index {
            self.stats.current_size_bytes += meta.size_bytes;
            self.stats.entry_count += 1;
        }
    }
}

impl Drop for KernelCache {
    fn drop(&mut self) {
        // Save final stats
        if self.config.enable_stats {
            let stats_file = self.cache_dir.join("cache_stats.json");
            let stats_json = format!(
                r#"{{"hits":{},"misses":{},"writes":{},"evictions":{},"size_bytes":{},"entries":{},"hit_rate":{:.3}}}"#,
                self.stats.hits,
                self.stats.misses,
                self.stats.writes,
                self.stats.evictions,
                self.stats.current_size_bytes,
                self.stats.entry_count,
                self.stats.hit_rate()
            );
            let _ = fs::write(stats_file, stats_json);
        }
    }
}

// ========== Helper Functions ==========

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_string(bytes: &[u8]) -> Result<(String, usize)> {
    if bytes.len() < 2 {
        return Err(Error::Io("String too short".to_string()));
    }
    let len = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    if bytes.len() < 2 + len {
        return Err(Error::Io("String data incomplete".to_string()));
    }
    let s = String::from_utf8_lossy(&bytes[2..2 + len]).to_string();
    Ok((s, 2 + len))
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_cache() -> (KernelCache, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig::default()
            .with_max_size_mb(1)
            .with_max_age_days(1)
            .with_auto_evict(true);
        let cache = KernelCache::with_config(temp_dir.path(), config).unwrap();
        (cache, temp_dir)
    }

    #[test]
    fn test_cache_creation() {
        let (_cache, _temp_dir) = create_test_cache();
        // Cache should be created successfully
    }

    #[test]
    fn test_cache_key_generation() {
        let (cache, _temp_dir) = create_test_cache();

        let mil_code = "func main() -> () { }";
        let weights: HashMap<String, Vec<u8>> = HashMap::new();
        let input_sizes = vec![16];
        let output_sizes = vec![8];

        let key1 = cache.generate_key(mil_code, &weights, &input_sizes, &output_sizes);
        let key2 = cache.generate_key(mil_code, &weights, &input_sizes, &output_sizes);

        // Same inputs should produce same key
        assert_eq!(key1, key2);

        // Different MIL should produce different key
        let key3 = cache.generate_key("different", &weights, &input_sizes, &output_sizes);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_store_and_load() {
        let (mut cache, _temp_dir) = create_test_cache();

        let key = "test_key_12345678901234567890123456789012";
        let program_data = vec![1u8, 2, 3, 4, 5];
        let mil_code = "func test() {}";
        let weights: HashMap<String, Vec<u8>> = HashMap::new();

        // Store
        cache.store(key, &program_data, mil_code, &weights).unwrap();

        // Load
        let loaded = cache.load(key).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap(), program_data);

        // Check stats
        assert_eq!(cache.stats().writes, 1);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_miss() {
        let (mut cache, _temp_dir) = create_test_cache();

        let result = cache.load("nonexistent_key").unwrap();
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_contains() {
        let (mut cache, _temp_dir) = create_test_cache();

        let key = "test_key_contains";
        let program_data = vec![1u8, 2, 3];
        let mil_code = "func test() {}";
        let weights: HashMap<String, Vec<u8>> = HashMap::new();

        assert!(!cache.contains(key));

        cache.store(key, &program_data, mil_code, &weights).unwrap();

        assert!(cache.contains(key));
    }

    #[test]
    fn test_cache_remove() {
        let (mut cache, _temp_dir) = create_test_cache();

        let key = "test_key_remove";
        let program_data = vec![1u8, 2, 3];
        let mil_code = "func test() {}";
        let weights: HashMap<String, Vec<u8>> = HashMap::new();

        cache.store(key, &program_data, mil_code, &weights).unwrap();
        assert!(cache.contains(key));

        cache.remove(key).unwrap();
        assert!(!cache.contains(key));
    }

    #[test]
    fn test_cache_clear() {
        let (mut cache, _temp_dir) = create_test_cache();

        for i in 0..5 {
            let key = format!("test_key_{}", i);
            let program_data = vec![i as u8; 10];
            let mil_code = "func test() {}";
            let weights: HashMap<String, Vec<u8>> = HashMap::new();
            cache
                .store(&key, &program_data, mil_code, &weights)
                .unwrap();
        }

        assert_eq!(cache.entry_count(), 5);

        cache.clear().unwrap();

        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn test_cache_stats_tracking() {
        let (mut cache, _temp_dir) = create_test_cache();

        let key = "test_key_stats";
        let program_data = vec![1u8, 2, 3, 4, 5];
        let mil_code = "func test() {}";
        let weights: HashMap<String, Vec<u8>> = HashMap::new();

        cache.store(key, &program_data, mil_code, &weights).unwrap();

        // Load multiple times
        cache.load(key).unwrap();
        cache.load(key).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.hits, 2);
        assert!(stats.hit_rate() > 0.5);
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = CacheEntryMetadata::new(
            "test_key".to_string(),
            1024,
            "mil_hash_123".to_string(),
            "weights_hash_456".to_string(),
            "options_hash_789".to_string(),
        );

        let bytes = metadata.to_bytes();
        let restored = CacheEntryMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(metadata.key, restored.key);
        assert_eq!(metadata.size_bytes, restored.size_bytes);
        assert_eq!(metadata.mil_hash, restored.mil_hash);
    }
}
