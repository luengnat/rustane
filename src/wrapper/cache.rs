//! Kernel compilation cache with LRU eviction
//!
//! Reduces recompilation overhead during training by caching compiled kernels.
//! Prevents memory exhaustion by respecting the ~119 kernel limit with LRU eviction.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::wrapper::{ANECompiler, ANEExecutor};
use crate::Result;

/// Single cache entry holding both compiler and executor
///
/// We store the compiler to keep the kernel handle alive.
/// The executor holds a dangling pointer to the kernel, so we
/// must keep the compiler around for the lifetime of the executor.
struct CacheEntry {
    _compiler: ANECompiler, // Owns kernel; must not be dropped
    executor: ANEExecutor,  // Borrows from compiler
    hits: u64,              // Statistics
}

/// Kernel compilation cache with LRU eviction
///
/// Caches compiled ANE kernels by MIL text + optional weight data.
/// Keeps both ANECompiler and ANEExecutor together to prevent dangling pointers.
///
/// # Architecture
///
/// ```text
/// HashMap<hash> -> CacheEntry
///   |_ ANECompiler (owns kernel handle)
///   |_ ANEExecutor (uses kernel handle)
/// VecDeque<hash> -> LRU order (oldest first)
/// ```
///
/// When cache is full (>= max_entries), evicts the oldest entry.
///
/// # Example
///
/// ```no_run
/// # use rustane::wrapper::KernelCache;
/// let mut cache = KernelCache::with_default_limit();
/// let mil = "program(1.3) { ... }".to_string();
/// let executor = cache.get_or_compile(&mil, None, &[256], &[512])?;
/// # Ok::<(), rustane::Error>(())
/// ```
pub struct KernelCache {
    entries: HashMap<u64, CacheEntry>,
    lru: VecDeque<u64>,
    max_entries: usize,
}

impl KernelCache {
    /// Create a new cache with specified max entries
    ///
    /// # Arguments
    ///
    /// * `max_entries` - Maximum number of cached kernels before LRU eviction
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::KernelCache;
    /// let cache = KernelCache::new(80);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn new(max_entries: usize) -> Self {
        KernelCache {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            max_entries,
        }
    }

    /// Create a cache with the recommended default limit
    ///
    /// The default is 80, leaving ~39 kernels of headroom below the ~119 limit
    /// to prevent mysterious crashes near the limit.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::KernelCache;
    /// let cache = KernelCache::with_default_limit();
    /// assert_eq!(cache.max_entries(), 80);
    /// ```
    pub fn with_default_limit() -> Self {
        Self::new(80)
    }

    /// Get the max number of entries
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Calculate a cache key from MIL text and optional weight data
    ///
    /// Uses SHA-like hashing to fingerprint the compilation inputs.
    /// Collisions are extremely unlikely with real MIL + weights.
    ///
    /// # Arguments
    ///
    /// * `mil_text` - MIL program source
    /// * `weight_data` - Optional pre-quantized weight blob
    ///
    /// # Returns
    ///
    /// u64 hash suitable for cache lookup
    pub fn cache_key(mil_text: &str, weight_data: Option<&[u8]>) -> u64 {
        let mut hasher = DefaultHasher::new();
        mil_text.hash(&mut hasher);
        if let Some(weights) = weight_data {
            weights.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Compile a kernel if not already cached, or return cached executor
    ///
    /// On cache miss:
    /// 1. Compile the MIL program via ANECompiler::compile_single
    /// 2. Store both compiler and executor together
    /// 3. Update LRU order
    /// 4. Evict oldest if at max_entries
    ///
    /// On cache hit:
    /// 1. Increment hit counter for stats
    /// 2. Move to back of LRU (most recent)
    /// 3. Return mutable ref to executor
    ///
    /// # Arguments
    ///
    /// * `mil_text` - MIL program source
    /// * `weight_data` - Optional pre-quantized weight blob
    /// * `input_sizes` - Input buffer sizes
    /// * `output_sizes` - Output buffer sizes
    ///
    /// # Returns
    ///
    /// Mutable reference to cached executor, or error on compilation failure
    pub fn get_or_compile(
        &mut self,
        mil_text: &str,
        weight_data: Option<&[u8]>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<&mut ANEExecutor> {
        let key = Self::cache_key(mil_text, weight_data);

        // Cache hit: move to back (most recent), increment counter
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hits += 1;
            // Move to back of LRU queue
            self.lru.retain(|&k| k != key);
            self.lru.push_back(key);
            // SAFETY: We just ensured the key exists
            return Ok(&mut self.entries.get_mut(&key).unwrap().executor);
        }

        // Cache miss: compile and insert
        // Evict oldest if needed
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        // Compile
        let mut compiler = ANECompiler::new();
        let executor = compiler.compile_single(mil_text, weight_data, input_sizes, output_sizes)?;

        // Insert into cache
        self.entries.insert(
            key,
            CacheEntry {
                _compiler: compiler,
                executor,
                hits: 1,
            },
        );
        self.lru.push_back(key);

        // SAFETY: We just inserted the key
        Ok(&mut self.entries.get_mut(&key).unwrap().executor)
    }

    /// Compile a multi-weight kernel if not cached, or return cached executor.
    ///
    /// Like `get_or_compile` but uses `compile_multi` for kernels that reference
    /// multiple named weight files (e.g. `@model_path/weights/rms_w.bin` in MIL).
    pub fn get_or_compile_multi(
        &mut self,
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        weight_lens: &[usize],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> crate::Result<&mut ANEExecutor> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        mil_text.hash(&mut hasher);
        for wd in weight_datas {
            wd.hash(&mut hasher);
        }
        let key = hasher.finish();

        if self.entries.contains_key(&key) {
            self.entries.get_mut(&key).unwrap().hits += 1;
            self.lru.retain(|&k| k != key);
            self.lru.push_back(key);
            return Ok(&mut self.entries.get_mut(&key).unwrap().executor);
        }

        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let mut compiler = ANECompiler::new();
        let executor = compiler.compile_multi(
            mil_text,
            weight_names,
            weight_datas,
            weight_lens,
            input_sizes,
            output_sizes,
        )?;

        self.entries.insert(
            key,
            CacheEntry {
                _compiler: compiler,
                executor,
                hits: 1,
            },
        );
        self.lru.push_back(key);
        Ok(&mut self.entries.get_mut(&key).unwrap().executor)
    }

    /// Evict the least-recently-used entry
    fn evict_lru(&mut self) {
        if let Some(old_key) = self.lru.pop_front() {
            self.entries.remove(&old_key);
        }
    }

    /// Number of cached kernels
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::KernelCache;
    /// let cache = KernelCache::new(10);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache hit rate as a fraction [0.0, 1.0]
    ///
    /// Returns 0.0 if no entries or no hits yet.
    pub fn hit_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }

        let total_hits: u64 = self.entries.values().map(|e| e.hits).sum();
        let total_entries = self.entries.len() as u64;

        // Hit rate = hits / (hits + misses)
        // We track hits but not initial miss, so approximate as:
        // hit_rate ≈ total_hits / (total_hits + number_of_entries)
        if total_hits == 0 {
            0.0
        } else {
            total_hits as f64 / (total_hits as f64 + total_entries as f64)
        }
    }

    /// Get cache statistics as (size, capacity, hit_rate)
    pub fn stats(&self) -> (usize, usize, f64) {
        (self.len(), self.max_entries, self.hit_rate())
    }

    /// Clear all cached entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = KernelCache::new(50);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_entries(), 50);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_default_limit() {
        let cache = KernelCache::with_default_limit();
        assert_eq!(cache.max_entries(), 80);
    }

    #[test]
    fn test_cache_key_consistency() {
        let mil = "program(1.3) {}";
        let key1 = KernelCache::cache_key(mil, None);
        let key2 = KernelCache::cache_key(mil, None);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_differs_with_weights() {
        let mil = "program(1.3) {}";
        let weights1 = [1u8; 100];
        let weights2 = [2u8; 100];

        let key_no_weights = KernelCache::cache_key(mil, None);
        let key_with_weights1 = KernelCache::cache_key(mil, Some(&weights1));
        let key_with_weights2 = KernelCache::cache_key(mil, Some(&weights2));

        assert_ne!(key_no_weights, key_with_weights1);
        assert_ne!(key_with_weights1, key_with_weights2);
    }

    #[test]
    fn test_cache_hit_rate_empty() {
        let cache = KernelCache::new(10);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = KernelCache::new(100);
        let (size, cap, rate) = cache.stats();
        assert_eq!(size, 0);
        assert_eq!(cap, 100);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = KernelCache::new(10);
        // We don't add entries in test - just test clear on empty cache
        assert!(cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }
}
