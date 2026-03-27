# ANE Memory Pool Guide

## Overview

The ANE Memory Pool provides efficient IOSurface buffer allocation and reuse to:
- **Reduce allocation overhead** - IOSurface creation is expensive (~10-50μs per allocation)
- **Enable buffer reuse** - Avoid repeated allocations for recurring kernel operations
- **Minimize fragmentation** - Size-class based allocation prevents memory fragmentation
- **Track memory usage** - Built-in statistics for monitoring and debugging

## Architecture

### Size-Class Allocation

The memory pool uses a size-class based strategy similar to jemalloc:

```
Size Classes (powers of 2):
├── 1 KB    (1,024 bytes)
├── 4 KB    (4,096 bytes)
├── 16 KB   (16,384 bytes)
├── 64 KB   (65,536 bytes)
├── 256 KB  (262,144 bytes)
├── 1 MB    (1,048,576 bytes)
├── 4 MB    (4,194,304 bytes)
└── 16 MB   (16,777,216 bytes)
```

When allocating:
1. Request is rounded up to nearest size class
2. Pool checks for reusable buffer in that class
3. If available: reuse (cache hit)
4. If not: allocate new IOSurface (cache miss)

### Buffer Lifecycle

```
┌─────────────┐     allocate()    ┌──────────────┐
│  Size Class │ ────────────────> │ PooledBuffer │
│    Pool     │                   │   (checked   │
│             │ <──────────────── │    out)      │
└─────────────┘      drop()       └──────────────┘
       ^
       │ return to pool
       │ (if under limit)
```

## Quick Start

### Basic Usage

```rust
use rustane::ane::memory_pool::{MemoryPool, PoolConfig};

// Create pool with defaults
let mut pool = MemoryPool::new()?;

// Allocate a buffer (automatically rounds up to size class)
let mut buffer = pool.allocate(10_000)?;  // Uses 16 KB class

// Use the buffer
buffer.write(&data)?;
let ptr = buffer.lock_read()?;
// ... use pointer ...
buffer.unlock_read()?;

// Buffer returns to pool automatically when dropped
drop(buffer);
```

### Custom Configuration

```rust
let config = PoolConfig::default()
    .with_max_pool_size(16)      // Keep up to 16 buffers per class
    .with_max_memory_mb(512)     // 512 MB total limit
    .with_stats(true);           // Enable statistics

let mut pool = MemoryPool::with_config(config)?;
```

### Thread-Safe Pool

```rust
use rustane::ane::memory_pool::SharedMemoryPool;

// Create thread-safe pool
let pool = SharedMemoryPool::new()?;

// Can be cloned and shared across threads
let pool_clone = pool.clone();

// Allocate from any thread
let buffer = pool.allocate(4096)?;
```

## API Reference

### MemoryPool

```rust
pub struct MemoryPool {
    // ...
}

impl MemoryPool {
    pub fn new() -> Result<Self>;
    pub fn with_config(config: PoolConfig) -> Result<Self>;

    /// Allocate a buffer (rounds up to size class)
    pub fn allocate(&mut self, size: usize) -> Result<PooledBuffer>;

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats;

    /// Clear all pooled buffers
    pub fn clear(&mut self);

    /// Check if allocation is possible
    pub fn can_allocate(&self, size: usize) -> bool;
}
```

### PoolConfig

```rust
pub struct PoolConfig {
    pub max_pool_size_per_class: usize,  // Default: 8
    pub max_total_memory: usize,          // Default: 256 MB
    pub enable_stats: bool,               // Default: true
    pub size_classes: Vec<usize>,         // Default: 8 classes
}

impl PoolConfig {
    pub fn new() -> Self;
    pub fn with_max_pool_size(self, size: usize) -> Self;
    pub fn with_max_memory_mb(self, mb: usize) -> Self;
    pub fn with_stats(self, enable: bool) -> Self;
    pub fn with_size_classes(self, classes: Vec<usize>) -> Self;
}
```

### PooledBuffer

```rust
pub struct PooledBuffer {
    // Automatically returns to pool when dropped
}

impl PooledBuffer {
    pub fn capacity(&self) -> usize;
    pub fn surface(&self) -> &IOSurface;
    pub fn surface_mut(&mut self) -> &mut IOSurface;

    /// Read/write operations
    pub fn write(&self, data: &[u8]) -> Result<()>;
    pub fn read(&self, dest: &mut [u8]) -> Result<()>;
    pub fn lock_read(&self) -> Result<*const u8>;
    pub fn lock_write(&self) -> Result<*mut u8>;

    /// Take ownership (prevents return to pool)
    pub fn into_surface(self) -> IOSurface;
}
```

### PoolStats

```rust
pub struct PoolStats {
    pub current_memory_bytes: usize,
    pub pooled_memory_bytes: usize,
    pub peak_memory_bytes: usize,
    pub total_allocations: u64,
    pub total_frees: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub per_class: Vec<SizeClassStats>,
}

impl PoolStats {
    /// Get cache hit rate (0.0 to 1.0)
    pub fn cache_hit_rate(&self) -> f64;

    /// Get memory efficiency (pooled / current)
    pub fn memory_efficiency(&self) -> f64;
}
```

## Performance Guidelines

### Cache Hit Rate

| Hit Rate | Assessment | Action |
|----------|------------|--------|
| > 80% | Excellent | Pool is well-sized |
| 50-80% | Good | Consider increasing pool size |
| < 50% | Poor | Increase max_pool_size or review allocation patterns |

### Memory Efficiency

| Efficiency | Assessment | Action |
|------------|------------|--------|
| > 60% | Excellent | Good buffer reuse |
| 30-60% | Acceptable | Normal operation |
| < 30% | Low | Buffers not being reused effectively |

### Size Class Selection

Allocations round up to the nearest size class:

| Request | Size Class | Waste |
|---------|------------|-------|
| 100 bytes | 1 KB | 924 bytes |
| 2 KB | 4 KB | 2 KB |
| 10 KB | 16 KB | 6 KB |
| 100 KB | 256 KB | 156 KB |

For optimal memory usage:
- Size allocations to match size classes
- Use custom size classes for known allocation patterns

## Integration Patterns

### Kernel Input/Output Buffers

```rust
use rustane::ane::{MemoryPool, ANECompileRequest};

struct KernelExecutor {
    pool: MemoryPool,
}

impl KernelExecutor {
    fn execute_kernel(&mut self, kernel: &mut ANEKernel, input: &[f32]) -> Result<Vec<f32>> {
        // Allocate input/output from pool
        let input_size = input.len() * 4; // f32 = 4 bytes
        let output_size = kernel.output_size();

        let mut input_buffer = self.pool.allocate(input_size)?;
        let mut output_buffer = self.pool.allocate(output_size)?;

        // Write input
        input_buffer.write(bytemuck::cast_slice(input))?;

        // Execute kernel
        kernel.write_input(0, &input_buffer)?;
        kernel.eval()?;
        kernel.read_output(0, &output_buffer)?;

        // Read output
        let mut output = vec![0f32; output_size / 4];
        output_buffer.read(bytemuck::cast_slice_mut(&mut output))?;

        Ok(output)
    }
}
```

### Activation Caching

```rust
use rustane::ane::memory_pool::SharedMemoryPool;

struct ActivationCache {
    pool: SharedMemoryPool,
    cache: HashMap<String, PooledBuffer>,
}

impl ActivationCache {
    fn get_or_allocate(&mut self, name: &str, size: usize) -> Result<&mut PooledBuffer> {
        if !self.cache.contains_key(name) {
            let buffer = self.pool.allocate(size)?;
            self.cache.insert(name.to_string(), buffer);
        }
        Ok(self.cache.get_mut(name).unwrap())
    }
}
```

### Training Loop Integration

```rust
struct TrainingSession {
    model: TransformerANE,
    pool: MemoryPool,
    grad_pool: MemoryPool,  // Separate pool for gradients
}

impl TrainingSession {
    fn train_step(&mut self, batch: &Batch) -> Result<f32> {
        // Allocate activation buffers from pool
        let mut activations = Vec::new();
        for _ in 0..self.model.num_layers() {
            activations.push(self.pool.allocate(self.model.activation_size())?);
        }

        // Forward pass with pooled activations
        let output = self.model.forward_pooled(batch, &mut activations)?;

        // Allocate gradient buffer
        let mut grad_buffer = self.grad_pool.allocate(self.model.param_count() * 4)?;

        // Backward pass
        let loss = compute_loss(&output, &batch.targets)?;
        self.model.backward_pooled(&mut grad_buffer)?;

        // Gradients automatically returned to pool
        Ok(loss)
    }
}
```

## Best Practices

1. **Reuse pools across operations**
   - Create pool once, reuse for entire training run
   - Don't create/destroy pools frequently

2. **Size pools appropriately**
   - Set `max_pool_size` based on concurrent buffer needs
   - Set `max_total_memory` to prevent OOM

3. **Monitor statistics**
   - Check `cache_hit_rate()` periodically
   - Alert on low hit rates (< 50%)

4. **Use separate pools for different purposes**
   - One pool for activations
   - One pool for gradients
   - One pool for temporary buffers

5. **Clear pools at epoch boundaries**
   - Call `pool.clear()` between epochs
   - Releases unused buffers back to system

## Troubleshooting

### Low Cache Hit Rate

**Symptom**: `stats().cache_hit_rate()` < 50%

**Causes**:
- Pool size too small
- Allocation sizes vary too much
- Buffers not being returned

**Solutions**:
1. Increase `max_pool_size_per_class`
2. Add more size classes for common sizes
3. Check for buffers held too long

### Memory Limit Exceeded

**Symptom**: `allocate()` returns error about memory limit

**Solutions**:
1. Increase `max_total_memory`
2. Reduce `max_pool_size_per_class`
3. Call `clear()` to release pooled buffers
4. Check for buffer leaks (buffers not dropped)

### High Memory Fragmentation

**Symptom**: Many small allocations, poor memory efficiency

**Solutions**:
1. Add smaller size classes
2. Batch small allocations together
3. Use custom size classes matching allocation pattern

## Related Documentation

- `docs/ANE_PROFILER_GUIDE.md` - Profile memory allocation performance
- `docs/GRADIENT_CHECKPOINTING.md` - Reduce activation memory
- `docs/MIXED_PRECISION_TRAINING.md` - FP16 memory savings
