//! ANE Memory Pool for Efficient IOSurface Allocation
//!
//! Provides a memory pooling mechanism for ANE IOSurface allocations to:
//! - Reduce allocation overhead (IOSurface creation is expensive)
//! - Enable buffer reuse across kernel invocations
//! - Minimize memory fragmentation
//! - Track memory usage statistics
//!
//! # Architecture
//!
//! The memory pool uses a size-class based allocation strategy:
//! - Buffers are grouped into size classes (e.g., 1KB, 4KB, 16KB, 64KB, 256KB, 1MB)
//! - Each size class maintains a pool of reusable buffers
//! - Allocations round up to the nearest size class
//! - Freed buffers return to their size class pool for reuse
//!
//! # Quick Start
//!
//! ```no_run
//! use rustane::ane::memory_pool::{MemoryPool, PoolConfig};
//!
//! // Create pool with default configuration
//! let mut pool = MemoryPool::new();
//!
//! // Or custom configuration
//! let config = PoolConfig::default()
//!     .with_max_pool_size(16)  // Keep up to 16 buffers per size class
//!     .with_max_memory_mb(512); // 512MB total limit
//! let mut pool = MemoryPool::with_config(config);
//!
//! // Allocate a buffer
//! let mut handle = pool.allocate(10_000)?; // Rounds up to 16KB class
//!
//! // Use the buffer
//! handle.write(&data)?;
//! let ptr = handle.lock_read()?;
//! // ... use pointer ...
//! handle.unlock_read()?;
//!
//! // Buffer automatically returns to pool when handle is dropped
//! ```
//!
//! # Size Classes
//!
//! Default size classes (powers of 2):
//! - 1 KB (1,024 bytes)
//! - 4 KB (4,096 bytes)
//! - 16 KB (16,384 bytes)
//! - 64 KB (65,536 bytes)
//! - 256 KB (262,144 bytes)
//! - 1 MB (1,048,576 bytes)
//! - 4 MB (4,194,304 bytes)
//! - 16 MB (16,777,216 bytes)

use crate::ane::IOSurface;
use crate::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Default size classes for buffer allocation (in bytes)
const DEFAULT_SIZE_CLASSES: [usize; 8] = [
    1024,     // 1 KB
    4096,     // 4 KB
    16384,    // 16 KB
    65536,    // 64 KB
    262144,   // 256 KB
    1048576,  // 1 MB
    4194304,  // 4 MB
    16777216, // 16 MB
];

/// Configuration for memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers to keep per size class
    pub max_pool_size_per_class: usize,
    /// Maximum total memory usage in bytes (0 = unlimited)
    pub max_total_memory: usize,
    /// Enable statistics tracking
    pub enable_stats: bool,
    /// Size classes to use (empty = use defaults)
    pub size_classes: Vec<usize>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size_per_class: 8,
            max_total_memory: 256 * 1024 * 1024, // 256 MB
            enable_stats: true,
            size_classes: DEFAULT_SIZE_CLASSES.to_vec(),
        }
    }
}

impl PoolConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum pool size per size class
    pub fn with_max_pool_size(mut self, size: usize) -> Self {
        self.max_pool_size_per_class = size;
        self
    }

    /// Set maximum total memory in megabytes
    pub fn with_max_memory_mb(mut self, mb: usize) -> Self {
        self.max_total_memory = mb * 1024 * 1024;
        self
    }

    /// Enable or disable statistics tracking
    pub fn with_stats(mut self, enable: bool) -> Self {
        self.enable_stats = enable;
        self
    }

    /// Set custom size classes
    pub fn with_size_classes(mut self, classes: Vec<usize>) -> Self {
        self.size_classes = classes;
        self
    }
}

/// Statistics for a single size class
#[derive(Debug, Clone, Default)]
pub struct SizeClassStats {
    /// Size class in bytes
    pub size_class: usize,
    /// Number of buffers currently in pool
    pub pooled_buffers: usize,
    /// Number of buffers currently allocated (checked out)
    pub allocated_buffers: usize,
    /// Total allocations since pool creation
    pub total_allocations: u64,
    /// Total frees since pool creation
    pub total_frees: u64,
}

/// Pool-wide statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total memory currently allocated
    pub current_memory_bytes: usize,
    /// Total memory currently pooled
    pub pooled_memory_bytes: usize,
    /// Total memory ever allocated (peak)
    pub peak_memory_bytes: usize,
    /// Total number of allocations
    pub total_allocations: u64,
    /// Total number of frees
    pub total_frees: u64,
    /// Number of cache hits (reused from pool)
    pub cache_hits: u64,
    /// Number of cache misses (new allocation required)
    pub cache_misses: u64,
    /// Statistics per size class
    pub per_class: Vec<SizeClassStats>,
}

impl PoolStats {
    /// Get cache hit rate (0.0 to 1.0)
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Get memory efficiency (pooled / current)
    pub fn memory_efficiency(&self) -> f64 {
        if self.current_memory_bytes == 0 {
            0.0
        } else {
            self.pooled_memory_bytes as f64 / self.current_memory_bytes as f64
        }
    }
}

/// Internal buffer pool for a single size class
struct BufferPool {
    /// Size class in bytes
    size_class: usize,
    /// Pool of reusable buffers
    buffers: Vec<IOSurface>,
    /// Number of buffers currently allocated
    allocated: usize,
    /// Total allocations
    total_allocations: u64,
    /// Total frees
    total_frees: u64,
}

impl BufferPool {
    fn new(size_class: usize) -> Self {
        Self {
            size_class,
            buffers: Vec::new(),
            allocated: 0,
            total_allocations: 0,
            total_frees: 0,
        }
    }

    fn stats(&self) -> SizeClassStats {
        SizeClassStats {
            size_class: self.size_class,
            pooled_buffers: self.buffers.len(),
            allocated_buffers: self.allocated,
            total_allocations: self.total_allocations,
            total_frees: self.total_frees,
        }
    }
}

/// Memory pool for ANE IOSurface allocations
pub struct MemoryPool {
    /// Pool configuration
    config: PoolConfig,
    /// Buffer pools per size class (keyed by size class)
    pools: HashMap<usize, BufferPool>,
    /// Current total memory usage
    current_memory: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// Total cache hits
    cache_hits: u64,
    /// Total cache misses
    cache_misses: u64,
}

impl MemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(PoolConfig::default())
    }

    /// Create a new memory pool with custom configuration
    pub fn with_config(config: PoolConfig) -> Result<Self> {
        let mut pools = HashMap::new();

        for &size_class in &config.size_classes {
            pools.insert(size_class, BufferPool::new(size_class));
        }

        Ok(Self {
            config,
            pools,
            current_memory: 0,
            peak_memory: 0,
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Find the smallest size class that fits the requested size
    fn find_size_class(&self, size: usize) -> Option<usize> {
        self.config
            .size_classes
            .iter()
            .find(|&&sc| sc >= size)
            .copied()
    }

    /// Allocate a buffer from the pool
    ///
    /// Returns a PooledBuffer handle that automatically returns
    /// the buffer to the pool when dropped.
    pub fn allocate(&mut self, size: usize) -> Result<PooledBuffer> {
        let size_class = self.find_size_class(size).ok_or_else(|| {
            crate::Error::Io(format!(
                "Requested size {} exceeds maximum pool size class {}",
                size,
                self.config.size_classes.last().copied().unwrap_or(0)
            ))
        })?;

        // Check memory limit
        if self.config.max_total_memory > 0
            && self.current_memory + size_class > self.config.max_total_memory
        {
            return Err(crate::Error::Io(format!(
                "Memory pool limit exceeded: would use {} bytes (limit: {})",
                self.current_memory + size_class,
                self.config.max_total_memory
            )));
        }

        // Try to get a buffer from the pool
        let surface = {
            let pool = self.pools.get_mut(&size_class).unwrap();

            if let Some(surface) = pool.buffers.pop() {
                pool.allocated += 1;
                pool.total_allocations += 1;
                self.cache_hits += 1;
                surface
            } else {
                // Need to allocate a new buffer
                let surface = IOSurface::new(size_class)?;
                pool.allocated += 1;
                pool.total_allocations += 1;
                self.cache_misses += 1;
                surface
            }
        };

        self.current_memory += size_class;
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }

        Ok(PooledBuffer {
            surface,
            size_class,
            pool: self as *mut Self,
            returned: false,
        })
    }

    /// Return a buffer to the pool
    fn return_buffer(&mut self, surface: IOSurface, size_class: usize) {
        if let Some(pool) = self.pools.get_mut(&size_class) {
            pool.allocated -= 1;
            pool.total_frees += 1;

            // Only keep if under pool size limit
            if pool.buffers.len() < self.config.max_pool_size_per_class {
                pool.buffers.push(surface);
            }
            // Otherwise, let the buffer drop (IOSurface cleanup happens automatically)
        }

        self.current_memory -= size_class;
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        let pooled_memory: usize = self
            .pools
            .values()
            .map(|p| p.buffers.len() * p.size_class)
            .sum();

        let per_class: Vec<SizeClassStats> = self.pools.values().map(|p| p.stats()).collect();

        PoolStats {
            current_memory_bytes: self.current_memory,
            pooled_memory_bytes: pooled_memory,
            peak_memory_bytes: self.peak_memory,
            total_allocations: self.pools.values().map(|p| p.total_allocations).sum(),
            total_frees: self.pools.values().map(|p| p.total_frees).sum(),
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            per_class,
        }
    }

    /// Clear all pooled buffers (but keep allocated ones)
    pub fn clear(&mut self) {
        for pool in self.pools.values_mut() {
            pool.buffers.clear();
        }
    }

    /// Get the number of size classes
    pub fn num_size_classes(&self) -> usize {
        self.config.size_classes.len()
    }

    /// Check if pool is at capacity for a given size
    pub fn can_allocate(&self, size: usize) -> bool {
        self.find_size_class(size)
            .map(|sc| {
                self.config.max_total_memory == 0
                    || self.current_memory + sc <= self.config.max_total_memory
            })
            .unwrap_or(false)
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new().expect("Failed to create default memory pool")
    }
}

/// Handle to a pooled buffer
///
/// Automatically returns the buffer to the pool when dropped.
/// Use `into_surface()` to take ownership and prevent return.
pub struct PooledBuffer {
    /// The underlying IOSurface
    surface: IOSurface,
    /// Size class of this buffer
    size_class: usize,
    /// Pointer to parent pool (for returning)
    pool: *mut MemoryPool,
    /// Whether buffer has been returned to pool
    returned: bool,
}

impl PooledBuffer {
    /// Get the capacity of this buffer in bytes
    pub fn capacity(&self) -> usize {
        self.size_class
    }

    /// Get the underlying IOSurface
    pub fn surface(&self) -> &IOSurface {
        &self.surface
    }

    /// Get mutable reference to IOSurface
    pub fn surface_mut(&mut self) -> &mut IOSurface {
        &mut self.surface
    }

    /// Write data to the buffer
    pub fn write(&self, data: &[u8]) -> Result<()> {
        self.surface.write(data)
    }

    /// Read data from the buffer
    pub fn read(&self, dest: &mut [u8]) -> Result<()> {
        self.surface.read(dest)
    }

    /// Lock the buffer for reading
    pub fn lock_read(&self) -> Result<*const u8> {
        self.surface.lock_read()
    }

    /// Unlock the buffer after reading
    pub fn unlock_read(&self) -> Result<()> {
        self.surface.unlock_read()
    }

    /// Lock the buffer for writing
    pub fn lock_write(&self) -> Result<*mut u8> {
        self.surface.lock_write()
    }

    /// Unlock the buffer after writing
    pub fn unlock_write(&self) -> Result<()> {
        self.surface.unlock_write()
    }

    /// Take ownership of the buffer, preventing return to pool
    pub fn into_surface(mut self) -> IOSurface {
        self.returned = true;
        // Use std::mem::replace to extract the surface
        // We need a dummy surface to replace with
        let dummy = IOSurface::new(1024).unwrap_or_else(|_| {
            // Fallback: create minimal surface
            IOSurface::new(1024).expect("Failed to create dummy surface")
        });
        std::mem::replace(&mut self.surface, dummy)
    }

    /// Explicitly return buffer to pool early
    pub fn return_to_pool(&mut self) {
        if !self.returned && !self.pool.is_null() {
            unsafe {
                // We need to extract the surface without dropping it
                let dummy = IOSurface::new(1024).ok();
                if let Some(dummy_surface) = dummy {
                    let surface = std::mem::replace(&mut self.surface, dummy_surface);
                    (*self.pool).return_buffer(surface, self.size_class);
                    self.returned = true;
                }
            }
        }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if !self.returned && !self.pool.is_null() {
            // Extract surface using std::mem::replace
            let dummy = IOSurface::new(1024).ok();
            if let Some(dummy_surface) = dummy {
                let surface = std::mem::replace(&mut self.surface, dummy_surface);
                unsafe {
                    (*self.pool).return_buffer(surface, self.size_class);
                }
            }
        }
    }
}

// SAFETY: PooledBuffer wraps IOSurface which is thread-safe
unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

/// Thread-safe memory pool wrapper
pub struct SharedMemoryPool {
    inner: Arc<Mutex<MemoryPool>>,
}

impl SharedMemoryPool {
    /// Create a new shared memory pool
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(MemoryPool::new()?)),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: PoolConfig) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(MemoryPool::with_config(config)?)),
        })
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&self, size: usize) -> Result<PooledBuffer> {
        self.inner.lock().unwrap().allocate(size)
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.inner.lock().unwrap().stats()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }
}

impl Default for SharedMemoryPool {
    fn default() -> Self {
        Self::new().expect("Failed to create shared memory pool")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = MemoryPool::new().unwrap();
        assert_eq!(pool.num_size_classes(), 8);
    }

    #[test]
    fn test_pool_custom_config() {
        let config = PoolConfig::default()
            .with_max_pool_size(4)
            .with_max_memory_mb(128)
            .with_stats(false);

        let pool = MemoryPool::with_config(config).unwrap();
        assert_eq!(pool.config.max_pool_size_per_class, 4);
        assert_eq!(pool.config.max_total_memory, 128 * 1024 * 1024);
        assert!(!pool.config.enable_stats);
    }

    #[test]
    fn test_allocate_and_return() {
        let mut pool = MemoryPool::new().unwrap();

        // Allocate buffer
        let buffer = pool.allocate(1000).unwrap(); // Should use 1KB class
        assert!(buffer.capacity() >= 1000);

        // Buffer should be returned when dropped
        drop(buffer);

        // Check stats
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_frees, 1);
    }

    #[test]
    fn test_size_class_selection() {
        let pool = MemoryPool::new().unwrap();

        // Small allocation should use smallest class
        assert_eq!(pool.find_size_class(100), Some(1024));
        assert_eq!(pool.find_size_class(1024), Some(1024));

        // Larger allocations should round up
        assert_eq!(pool.find_size_class(1025), Some(4096));
        assert_eq!(pool.find_size_class(5000), Some(16384));

        // Too large should return None
        assert_eq!(pool.find_size_class(100_000_000), None);
    }

    #[test]
    fn test_buffer_reuse() {
        let mut pool = MemoryPool::new().unwrap();

        // First allocation (cache miss)
        let buffer1 = pool.allocate(1000).unwrap();
        let stats1 = pool.stats();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);

        // Return to pool
        drop(buffer1);

        // Second allocation (should be cache hit)
        let _buffer2 = pool.allocate(1000).unwrap();
        let stats2 = pool.stats();
        assert!(stats2.cache_hits >= 1);
    }

    #[test]
    fn test_memory_limit() {
        let config = PoolConfig::default().with_max_memory_mb(1); // 1 MB limit
        let mut pool = MemoryPool::with_config(config).unwrap();

        // Allocate several buffers
        let mut allocated = Vec::new();
        while let Ok(buffer) = pool.allocate(65536) {
            // 64 KB each
            allocated.push(buffer);
        }

        // Should have allocated some buffers
        assert!(!allocated.is_empty());

        // Eventually should hit the limit
        let can_alloc_more = pool.can_allocate(65536);
        assert!(!can_alloc_more);
    }

    #[test]
    fn test_stats_tracking() {
        let mut pool = MemoryPool::new().unwrap();

        // Allocate and free several buffers
        for _ in 0..5 {
            let buffer = pool.allocate(1000).unwrap();
            drop(buffer);
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 5);
        assert_eq!(stats.total_frees, 5);
        assert!(stats.cache_hit_rate() > 0.0);
    }
}
