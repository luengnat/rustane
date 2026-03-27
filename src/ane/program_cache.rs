//! ANE Program Cache
//!
//! Inspired by Orion's T084: Program cache for weight swapping.
//!
//! Cache keyed by (kernel_name, layer_idx, weights_id, config_hash).
//! Cache owns all stored programs — callers must NOT release them.
//!
//! # Usage Pattern
//!
//! ```no_run
//! use rustane::ane::{ANEProgramCache, ANEExecutor};
//!
//! let mut cache = ANEProgramCache::new();
//!
//! // Try lookup
//! if let Some(executor) = cache.lookup("attn_linear", 0, "step_0001", "cfg_v1") {
//!     // Use cached executor
//! } else {
//!     // Compile new executor
//!     let executor = compile_kernel(...)?;
//!     cache.store("attn_linear", 0, "step_0001", "cfg_v1", executor);
//! }
//! ```

use crate::wrapper::ANEExecutor;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Cache key: (kernel_name, layer_idx, weights_id, config_hash)
type CacheKey = (String, usize, String, String);

/// Cache entry holding an executor with metadata
struct CacheEntry {
    executor: Arc<Mutex<ANEExecutor>>,
    weights_id: String,
    last_used: Instant,
    use_count: u64,
}

/// Thread-safe program cache for ANE kernels with LRU eviction
///
/// Implements Orion's T084: Program cache for weight swapping.
/// Reduces compilation overhead by caching compiled kernels.
///
/// Key improvements:
/// - Non-destructive lookup (returns shared reference, doesn't remove)
/// - LRU eviction when cache is full
/// - Shared ownership via Arc<Mutex<>> for concurrent access
pub struct ANEProgramCache {
    cache: Mutex<HashMap<CacheKey, CacheEntry>>,
    lru_queue: Mutex<VecDeque<CacheKey>>, // Tracks usage order for LRU
    max_size: usize,
    stats: Mutex<CacheStats>,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Number of entries in cache
    pub entries: usize,
    /// Total evictions
    pub evictions: usize,
}

impl ANEProgramCache {
    /// Create a new empty cache with default size (100 entries)
    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    /// Create a new cache with specified capacity
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::with_capacity(max_size)),
            lru_queue: Mutex::new(VecDeque::with_capacity(max_size)),
            max_size,
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Lookup a cached executor (non-destructive)
    ///
    /// Returns a shared reference to the executor if found.
    /// Updates LRU tracking on hit.
    ///
    /// # Arguments
    /// * `kernel_name` - Name of the kernel (e.g., "attn_qkv")
    /// * `layer_idx` - Layer index
    /// * `weights_id` - Weights version ID (e.g., "step_0001")
    /// * `config_hash` - Configuration hash for cache key
    ///
    /// Returns Some(Arc<Mutex<ANEExecutor>>) if found, None otherwise
    pub fn lookup(
        &self,
        kernel_name: &str,
        layer_idx: usize,
        weights_id: &str,
        config_hash: &str,
    ) -> Option<Arc<Mutex<ANEExecutor>>> {
        let key = (
            kernel_name.to_string(),
            layer_idx,
            weights_id.to_string(),
            config_hash.to_string(),
        );

        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = cache.get_mut(&key) {
            // Cache hit - update metadata
            stats.hits += 1;
            entry.last_used = Instant::now();
            entry.use_count += 1;

            // Update LRU order
            drop(cache); // Release cache lock before acquiring lru_queue lock
            self.update_lru(&key);

            // Return shared reference
            cache = self.cache.lock().unwrap();
            Some(cache.get(&key).unwrap().executor.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Store an executor in the cache
    ///
    /// If cache is at capacity, evicts least recently used entry.
    ///
    /// # Arguments
    /// * `kernel_name` - Name of the kernel
    /// * `layer_idx` - Layer index
    /// * `weights_id` - Weights version ID
    /// * `config_hash` - Configuration hash
    /// * `executor` - Compiled executor to cache
    pub fn store(
        &self,
        kernel_name: &str,
        layer_idx: usize,
        weights_id: &str,
        config_hash: &str,
        executor: ANEExecutor,
    ) {
        let key = (
            kernel_name.to_string(),
            layer_idx,
            weights_id.to_string(),
            config_hash.to_string(),
        );

        // Evict if at capacity
        self.evict_if_needed();

        let mut cache = self.cache.lock().unwrap();
        let mut lru_queue = self.lru_queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let entry = CacheEntry {
            executor: Arc::new(Mutex::new(executor)),
            weights_id: weights_id.to_string(),
            last_used: Instant::now(),
            use_count: 0,
        };

        cache.insert(key.clone(), entry);
        lru_queue.push_back(key);
        stats.entries = cache.len();
    }

    /// Evict least recently used entry if at capacity
    fn evict_if_needed(&self) {
        let cache = self.cache.lock().unwrap();
        if cache.len() < self.max_size {
            return;
        }
        drop(cache);

        let mut lru_queue = self.lru_queue.lock().unwrap();
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Find and remove oldest entry that's still in cache
        while let Some(key) = lru_queue.pop_front() {
            if cache.remove(&key).is_some() {
                stats.evictions += 1;
                stats.entries = cache.len();
                break;
            }
        }
    }

    /// Update LRU order for a key (move to back = most recently used)
    fn update_lru(&self, key: &CacheKey) {
        let mut lru_queue = self.lru_queue.lock().unwrap();
        // Remove if present and re-add at back
        if let Some(pos) = lru_queue.iter().position(|k| k == key) {
            lru_queue.remove(pos);
            lru_queue.push_back(key.clone());
        }
    }

    /// Evict all entries with the given weights_id
    ///
    /// Call this when weights are updated to invalidate stale cache entries.
    pub fn evict_by_weights_id(&self, weights_id: &str) {
        let mut cache = self.cache.lock().unwrap();
        let mut lru_queue = self.lru_queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let keys_to_remove: Vec<CacheKey> = cache
            .iter()
            .filter(|(_, entry)| entry.weights_id == weights_id)
            .map(|(key, _)| key.clone())
            .collect();

        stats.evictions += keys_to_remove.len();

        for key in &keys_to_remove {
            cache.remove(key);
            if let Some(pos) = lru_queue.iter().position(|k| k == key) {
                lru_queue.remove(pos);
            }
        }

        stats.entries = cache.len();
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        let mut lru_queue = self.lru_queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.evictions += cache.len();
        cache.clear();
        lru_queue.clear();
        stats.entries = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get current cache size
    pub fn size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get maximum cache size
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }
}

impl Default for ANEProgramCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Step-based weights ID generator (Orion T086)
///
/// Generates unique weights IDs for each training step.
/// Used for cache key generation and invalidation.
pub struct WeightsIdGenerator {
    counter: Mutex<usize>,
}

impl WeightsIdGenerator {
    /// Create a new generator starting at step 0
    pub fn new() -> Self {
        Self {
            counter: Mutex::new(0),
        }
    }

    /// Get the next weights ID
    ///
    /// Returns IDs in the format "step_00000001"
    pub fn next_id(&self) -> String {
        let mut counter = self.counter.lock().unwrap();
        *counter += 1;
        format!("step_{:08}", *counter)
    }

    /// Get current step number
    pub fn current_step(&self) -> usize {
        *self.counter.lock().unwrap()
    }

    /// Reset to a specific step
    pub fn reset(&self, step: usize) {
        *self.counter.lock().unwrap() = step;
    }
}

impl Default for WeightsIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wrapper::ANECompiler;

    /// Try to create a mock executor (returns None if ANE unavailable).
    fn try_mock_executor() -> Option<ANEExecutor> {
        let mut compiler = ANECompiler::new();
        compiler
            .compile_single(r#"program(1.3) { }"#, None, &[16], &[16])
            .ok()
    }

    /// Tests that don't need ANE hardware

    #[test]
    fn test_weights_id_generator() {
        let gen = WeightsIdGenerator::new();
        assert_eq!(gen.current_step(), 0);
        assert_eq!(gen.next_id(), "step_00000001");
        assert_eq!(gen.next_id(), "step_00000002");
        assert_eq!(gen.current_step(), 2);
        gen.reset(100);
        assert_eq!(gen.current_step(), 100);
        assert_eq!(gen.next_id(), "step_00000101");
    }

    #[test]
    fn test_cache_stats() {
        let cache = ANEProgramCache::new();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_cache_hit_rate_no_lookups() {
        let cache = ANEProgramCache::new();
        assert_eq!(cache.hit_rate(), 0.0);
    }

    /// Tests that require ANE hardware (ignored gracefully on other machines)
    // These tests create real ANE executors via compile_single() which
    // dlopen's the ANE framework. This segfaults on machines where the
    // pre-existing IOSurface signature mismatch hasn't been fixed.

    #[test]
    #[ignore] // Requires ANE hardware
    fn test_cache_store_lookup() {
        let Some(executor) = try_mock_executor() else {
            return;
        };
        let cache = ANEProgramCache::new();
        cache.store("attn", 0, "step_0001", "cfg_v1", executor);
        assert_eq!(cache.size(), 1);
        let found = cache.lookup("attn", 0, "step_0001", "cfg_v1");
        assert!(found.is_some());
        assert_eq!(cache.size(), 1); // non-destructive
    }

    #[test]
    #[ignore] // Requires ANE hardware
    fn test_cache_multiple_lookups() {
        let Some(executor) = try_mock_executor() else {
            return;
        };
        let cache = ANEProgramCache::new();
        cache.store("attn", 0, "step_0001", "cfg_v1", executor);
        for i in 0..10 {
            assert!(
                cache.lookup("attn", 0, "step_0001", "cfg_v1").is_some(),
                "Lookup {} failed",
                i
            );
        }
        assert_eq!(cache.size(), 1);
        assert_eq!(cache.stats().hits, 10);
    }

    #[test]
    #[ignore] // Requires ANE hardware
    fn test_cache_lru_eviction() {
        let Some(exec1) = try_mock_executor() else {
            return;
        };
        let Some(exec2) = try_mock_executor() else {
            return;
        };
        let Some(exec3) = try_mock_executor() else {
            return;
        };
        let cache = ANEProgramCache::with_capacity(2);
        cache.store("k1", 0, "w1", "cfg", exec1);
        cache.store("k2", 0, "w2", "cfg", exec2);
        cache.store("k3", 0, "w3", "cfg", exec3);
        assert_eq!(cache.size(), 2);
        assert!(cache.lookup("k1", 0, "w1", "cfg").is_none());
        assert!(cache.lookup("k2", 0, "w2", "cfg").is_some());
        assert!(cache.lookup("k3", 0, "w3", "cfg").is_some());
    }

    #[test]
    #[ignore] // Requires ANE hardware
    fn test_cache_lru_updates_on_access() {
        let Some(exec1) = try_mock_executor() else {
            return;
        };
        let Some(exec2) = try_mock_executor() else {
            return;
        };
        let Some(exec3) = try_mock_executor() else {
            return;
        };
        let cache = ANEProgramCache::with_capacity(2);
        cache.store("k1", 0, "w1", "cfg", exec1);
        cache.store("k2", 0, "w2", "cfg", exec2);
        let _ = cache.lookup("k1", 0, "w1", "cfg"); // touch k1
        cache.store("k3", 0, "w3", "cfg", exec3);
        assert_eq!(cache.size(), 2);
        assert!(cache.lookup("k1", 0, "w1", "cfg").is_some());
        assert!(cache.lookup("k2", 0, "w2", "cfg").is_none()); // evicted
    }

    #[test]
    #[ignore] // Requires ANE hardware
    fn test_cache_hit_rate() {
        let Some(executor) = try_mock_executor() else {
            return;
        };
        let cache = ANEProgramCache::new();
        cache.store("attn", 0, "s1", "c", executor);
        let _ = cache.lookup("attn", 0, "s2", "c"); // miss
        assert_eq!(cache.hit_rate(), 0.0);
        let _ = cache.lookup("attn", 0, "s1", "c"); // hit
        assert_eq!(cache.hit_rate(), 0.5);
    }

    #[test]
    #[ignore] // Requires ANE hardware
    fn test_evict_by_weights_id() {
        let Some(exec1) = try_mock_executor() else {
            return;
        };
        let Some(exec2) = try_mock_executor() else {
            return;
        };
        let cache = ANEProgramCache::new();
        cache.store("k1", 0, "s1", "c", exec1);
        cache.store("k2", 0, "s2", "c", exec2);
        cache.evict_by_weights_id("s1");
        assert_eq!(cache.size(), 1);
        assert!(cache.lookup("k1", 0, "s1", "c").is_none());
        assert!(cache.lookup("k2", 0, "s2", "c").is_some());
    }
}
