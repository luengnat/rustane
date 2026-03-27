# ANE Kernel Cache Guide

## Overview

The ANE Kernel Cache provides disk-based persistence of compiled ANE kernels to:

- **Avoid recompilation overhead** (~4-5 seconds per kernel)
- **Stay within the ~119 compilation budget** per process
- **Enable faster startup** for repeated workloads
- **Share compiled kernels** across sessions

## Architecture

### Cache Key Generation

Cache keys are generated using SHA-256 hashes of:

```
key = SHA256(
    SHA256(mil_code) ||
    SHA256(sorted_weight_names || concatenated_weights) ||
    SHA256(input_shapes || output_shapes)
)
```

This ensures:
- Same MIL code + same weights = same key
- Different weights = different key
- Deterministic across sessions

### Cache Storage

```
cache_dir/
├── <hash>.hwxcache    # Compiled HWX program data
├── <hash>.meta        # Metadata (size, timestamps, hashes)
└── cache_stats.json   # Statistics (written on drop)
```

### Eviction Policy

LRU (Least Recently Used) with age limits:

1. **Age-based eviction**: Entries older than `max_file_age_days` are always evicted
2. **Size-based eviction**: When cache exceeds `max_cache_size_bytes`, least-used entries are removed
3. **Hybrid scoring**: `score = age * (access_count + 1)` - older, less-used entries evicted first

## Quick Start

### Basic Usage

```rust
use rustane::ane::kernel_cache::{KernelCache, CacheConfig};

// Create cache with default config
let mut cache = KernelCache::new("/tmp/ane_cache")?;

// Generate cache key
let key = cache.generate_key(mil_code, &weights, &input_sizes, &output_sizes);

// Try to load from cache
if let Some(program_data) = cache.load(&key)? {
    // Use cached program - load HWX directly
    let executor = load_hwx_program(&program_data)?;
} else {
    // Compile new program
    let executor = compile_mil(mil_code, &weights)?;

    // Store in cache
    cache.store(&key, &program_data, mil_code, &weights)?;
}
```

### Custom Configuration

```rust
let config = CacheConfig::default()
    .with_max_size_mb(512)    // 512 MB cache limit
    .with_max_age_days(14)     // 2 week max age
    .with_stats(true)          // Enable statistics
    .with_auto_evict(true);    // Auto-evict on creation

let cache = KernelCache::with_config("/path/to/cache", config)?;
```

### Integration with Training Loop

```rust
use rustane::ane::{KernelCache, ANECompileRequest};

struct TrainingSession {
    kernel_cache: KernelCache,
    // ...
}

impl TrainingSession {
    fn get_or_compile_kernel(
        &mut self,
        layer_name: &str,
        mil_code: &str,
        weights: &HashMap<String, Vec<u8>>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<ANEExecutor> {
        let key = self.kernel_cache.generate_key(
            mil_code,
            weights,
            input_sizes,
            output_sizes,
        );

        // Try cache first
        if let Some(program_data) = self.kernel_cache.load(&key)? {
            return load_hwx_program(&program_data);
        }

        // Compile and cache
        let request = ANECompileRequest::new(mil_code, input_sizes, output_sizes)
            .with_weights(weights.iter().map(|(k, v)| (k.clone(), v.clone())));
        let executor = request.compile()?;

        // Store compiled program
        let program_data = executor.to_hwx_data()?;
        self.kernel_cache.store(&key, &program_data, mil_code, weights)?;

        Ok(executor)
    }
}
```

## API Reference

### KernelCache

```rust
pub struct KernelCache {
    // ...
}

impl KernelCache {
    /// Create cache at specified directory
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self>;

    /// Create cache with custom config
    pub fn with_config<P: AsRef<Path>>(cache_dir: P, config: CacheConfig) -> Result<Self>;

    /// Generate cache key from MIL code and weights
    pub fn generate_key(
        &self,
        mil_code: &str,
        weights: &HashMap<String, Vec<u8>>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> String;

    /// Load cached program
    pub fn load(&mut self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Store compiled program
    pub fn store(
        &mut self,
        key: &str,
        program_data: &[u8],
        mil_code: &str,
        weights: &HashMap<String, Vec<u8>>,
    ) -> Result<()>;

    /// Check if key exists in cache
    pub fn contains(&self, key: &str) -> bool;

    /// Remove entry from cache
    pub fn remove(&mut self, key: &str) -> Result<bool>;

    /// Clear all cache entries
    pub fn clear(&mut self) -> Result<()>;

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats;

    /// Get number of cached entries
    pub fn entry_count(&self) -> usize;

    /// Get current cache size in bytes
    pub fn size_bytes(&self) -> u64;
}
```

### CacheConfig

```rust
pub struct CacheConfig {
    pub max_cache_size_bytes: u64,
    pub max_file_age_days: u64,
    pub enable_stats: bool,
    pub auto_evict: bool,
}

impl CacheConfig {
    pub fn new() -> Self;
    pub fn with_max_size_mb(self, mb: u64) -> Self;
    pub fn with_max_age_days(self, days: u64) -> Self;
    pub fn with_stats(self, enable: bool) -> Self;
    pub fn with_auto_evict(self, enable: bool) -> Self;
}
```

### CacheStats

```rust
pub struct CacheStats {
    pub hits: u64,               // Cache hits
    pub misses: u64,             // Cache misses
    pub writes: u64,             // Cache writes
    pub evictions: u64,          // Evicted entries
    pub current_size_bytes: u64, // Current cache size
    pub entry_count: u64,        // Number of entries
    pub bytes_saved: u64,        // Bytes saved by hits
}

impl CacheStats {
    /// Get hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64;

    /// Get cache efficiency
    pub fn efficiency(&self) -> f64;
}
```

## Performance Guidelines

### Cache Hit Rate

| Hit Rate | Assessment | Action |
|----------|------------|--------|
| > 80% | Excellent | Cache is well-sized |
| 50-80% | Good | Consider larger cache |
| < 50% | Poor | Review cache size or allocation patterns |

### Memory Savings

For a typical training run with 50 unique kernels:

| Scenario | Compilation Time | Cache Benefit |
|----------|-----------------|---------------|
| Cold cache | ~225 seconds | Baseline |
| Warm cache (80% hit) | ~45 seconds | 5x faster |
| Hot cache (95% hit) | ~11 seconds | 20x faster |

### Recommended Cache Sizes

| Use Case | Cache Size | Max Age |
|----------|-----------|---------|
| Development | 256 MB | 1 day |
| Testing | 512 MB | 7 days |
| Production | 1-2 GB | 30 days |

## Best Practices

1. **Persistent cache directory**: Use a dedicated directory outside temp folders
2. **Periodic cleanup**: Run eviction manually during maintenance windows
3. **Monitor hit rates**: Track cache statistics in production
4. **Cache warmup**: Pre-populate cache for known kernels before training
5. **Version control**: Include cache version in key for invalidation

## Troubleshooting

### Low Cache Hit Rate

**Symptom**: `stats().hit_rate()` < 50%

**Causes**:
- Weights changing too frequently
- MIL code regenerating with same semantics
- Cache being cleared too often

**Solutions**:
1. Use stable weight IDs (e.g., step-based naming)
2. Normalize MIL code before hashing
3. Increase cache size

### Cache Directory Growing Unbounded

**Symptom**: Cache directory exceeds expected size

**Solutions**:
1. Reduce `max_cache_size_bytes`
2. Reduce `max_file_age_days`
3. Enable `auto_evict`
4. Run manual `evict_old_entries()`

### Cache Key Collisions

**Symptom**: Wrong program loaded from cache

**Causes**:
- Hash collision (extremely rare with SHA-256)
- Incorrect weight ordering

**Solutions**:
1. Verify weight sorting is deterministic
2. Include more context in hash (config options)
3. Add validation metadata

## Integration with ANECompileRequest

Future enhancement: Integrate cache directly into `ANECompileRequest::compile()`:

```rust
impl ANECompileRequest {
    pub fn compile_with_cache(
        self,
        cache: &mut KernelCache,
    ) -> Result<ANEExecutor> {
        let key = self.generate_cache_key();

        if let Some(data) = cache.load(&key)? {
            return load_executor_from_data(&data);
        }

        let executor = self.compile()?;
        cache.store(&key, &executor.to_data()?, ...)?;

        Ok(executor)
    }
}
```

## Related Documentation

- `docs/ANE_PROGRAM_CACHE.md` - Original program cache design
- `docs/ANE_COMPILE_BUDGET.md` - Compilation budget management
- `docs/ANE_MEMORY_POOL.md` - Memory pooling for IOSurface buffers
