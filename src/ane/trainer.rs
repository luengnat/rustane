//! ANE Trainer - Hybrid ANE+CPU training with caching and pipelining
//!
//! Implements the architecture from maderix/ANE:
//! - ANE: Forward pass + backward dx (input gradients)
//! - CPU: Backward dW (weight gradients) via cblas + Adam optimizer
//! - Dynamic weights: Pack weights into spatial dimension to avoid recompilation
//! - Kernel caching: Reuse compiled kernels across steps (compile once)
//! - Weight reload: Update weights without recompilation (~494ms vs ~4,200ms)
//! - Pipeline: Overlap ANE execution with CPU gradient computation
//!
//! ## ANE Utilization Strategy
//!
//! The key optimization is the **compile-once, reload-weights** pattern:
//!
//! 1. First step: Compile kernel (expensive: ~4,200ms)
//! 2. Every subsequent step: `reload_weights()` (cheap: ~494ms)
//!
//! This avoids burning the ~119 compile budget and maximizes ANE duty cycle.

use super::mil_generator::{ANEMILOps, ANEMILProgram, ANEShape, ANETensorType};
use super::{ANECompileRequest, ANEError};
use crate::wrapper::ANEExecutor;
use crate::wrapper::KernelCache;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;

/// Compiled ANE kernel with metadata
#[derive(Debug, Clone)]
pub struct ANEKernelTemplate {
    /// Unique identifier for this kernel type
    pub id: String,
    /// MIL program text (for debugging)
    pub mil: String,
    /// Input tensor shape
    pub input_shape: ANEShape,
    /// Output tensor shape
    pub output_shape: ANEShape,
    /// Whether this kernel uses dynamic weights
    pub has_dynamic_weights: bool,
    /// Weight size in bytes (if dynamic)
    pub weight_size: usize,
}

impl ANEKernelTemplate {
    /// Create a simple RMSNorm kernel
    pub fn rmsnorm(channels: usize, seq_len: usize) -> Self {
        let input_shape = ANEShape::seq(channels, seq_len);
        let output_shape = input_shape.clone();
        let invd = 1.0 / (channels as f32);

        let mil = format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {c}, 1, {s}]> x) {{
        tensor<fp16, [1,{c},1,{s}]> sq = mul(x=x,y=x)[name=string("sq")];
        tensor<int32, [1]> rax = const()[name=string("rax"), val=tensor<int32, [1]>([1])];
        bool kd = const()[name=string("kd"), val=bool(true)];
        tensor<fp16, [1,1,1,{s}]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string("ss")];
        fp16 invd = const()[name=string("invd"), val=fp16({invd})];
        tensor<fp16, [1,1,1,{s}]> ss2 = mul(x=ss,y=invd)[name=string("ss2")];
        fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];
        tensor<fp16, [1,1,1,{s}]> ss3 = add(x=ss2,y=eps)[name=string("ss3")];
        fp16 nhalf = const()[name=string("nhalf"), val=fp16(-0.5)];
        tensor<fp16, [1,1,1,{s}]> rrms = pow(x=ss3,y=nhalf)[name=string("rrms")];
        tensor<fp16, [1,{c},1,{s}]> xr = mul(x=x,y=rrms)[name=string("xr")];
        tensor<fp16, [1,{c},1,1]> rw = const()[name=string("rw"), val=tensor<fp16, [1,{c},1,1]>(BLOBFILE(path=string("@model_path/weights/rms_w.bin"), offset=uint64(64)))];
        tensor<fp16, [1,{c},1,{s}]> out = mul(x=xr,y=rw)[name=string("out")];
    }} -> (out);
}}"#,
            c = channels,
            s = seq_len,
            invd = invd
        );

        Self {
            id: format!("rmsnorm_{}_{}", channels, seq_len),
            mil,
            input_shape,
            output_shape,
            has_dynamic_weights: true,
            weight_size: channels * 2, // fp16 weights
        }
    }

    /// Create a dynamic matmul kernel (weights packed in input)
    pub fn dynamic_matmul(in_channels: usize, out_channels: usize, seq_len: usize) -> Self {
        // Input: [1, IC, 1, SEQ + OC] - activations + weights packed
        // Output: [1, OC, 1, SEQ]
        let total_spatial = seq_len + out_channels;
        let input_shape = ANEShape::seq(in_channels, total_spatial);
        let output_shape = ANEShape::seq(out_channels, seq_len);

        let mil = format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {ic}, 1, {sp}]> x) {{
        // Slice activations: [1, IC, 1, SEQ]
        tensor<int32, [4]> ba = const()[name=string("ba"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [4]> sa = const()[name=string("sa"), val=tensor<int32, [4]>([1,{ic},1,{seq}])];
        tensor<fp16, [1,{ic},1,{seq}]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string("act")];
        
        // Slice weights: [1, IC, 1, OC]
        tensor<int32, [4]> bw = const()[name=string("bw"), val=tensor<int32, [4]>([0,0,0,{seq}])];
        tensor<int32, [4]> sw = const()[name=string("sw"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];
        tensor<fp16, [1,{ic},1,{oc}]> W = slice_by_size(x=x,begin=bw,size=sw)[name=string("W")];
        
        // Reshape for matmul: act [1,1,IC,SEQ] -> transpose -> [1,1,SEQ,IC]
        tensor<int32, [4]> ra = const()[name=string("ra"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];
        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra,x=act)[name=string("a2")];
        tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];
        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string("a3")];
        
        // Reshape weights: W [1,IC,1,OC] -> [1,1,IC,OC]
        tensor<int32, [4]> rw = const()[name=string("rw"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];
        tensor<fp16, [1,1,{ic},{oc}]> W2 = reshape(shape=rw,x=W)[name=string("W2")];
        
        // Matmul: [SEQ, IC] @ [IC, OC] = [SEQ, OC]
        bool bF = const()[name=string("bF"), val=bool(false)];
        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W2)[name=string("yh")];
        
        // Reshape output: [1,1,SEQ,OC] -> transpose -> [1,1,OC,SEQ] -> [1,OC,1,SEQ]
        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm,x=yh)[name=string("yt")];
        tensor<int32, [4]> ro = const()[name=string("ro"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];
        tensor<fp16, [1,{oc},1,{seq}]> out = reshape(shape=ro,x=yt)[name=string("out")];
    }} -> (out);
}}"#,
            ic = in_channels,
            oc = out_channels,
            seq = seq_len,
            sp = total_spatial
        );

        Self {
            id: format!("matmul_{}_{}_{}", in_channels, out_channels, seq_len),
            mil,
            input_shape,
            output_shape,
            has_dynamic_weights: true,
            weight_size: in_channels * out_channels * 2, // fp16
        }
    }
}

/// Cache key for kernel lookup
fn cache_key(kernel: &ANEKernelTemplate, input_bytes: usize, output_bytes: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    kernel.id.hash(&mut hasher);
    input_bytes.hash(&mut hasher);
    output_bytes.hash(&mut hasher);
    hasher.finish()
}

/// Cached ANE executor with compiled kernel and LRU tracking
struct CachedEntry {
    executor: ANEExecutor,
    last_used: Instant,
    use_count: u64,
}

/// ANE Trainer with kernel caching and compile-once training
///
/// ## ANE Utilization Optimization
///
/// This trainer maximizes ANE utilization through:
///
/// 1. **Kernel caching**: Compile each unique kernel shape once, reuse across steps
/// 2. **Weight reload**: Use `reload_weights()` instead of recompile (~8.5x faster)
/// 3. **LRU eviction**: Respect the ~119 compile budget by evicting unused kernels
/// 4. **Compile budget tracking**: Warn before hitting ANE memory limits
pub struct ANETrainer {
    /// Kernel cache: hash -> CachedEntry
    kernel_cache: HashMap<u64, CachedEntry>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Total compile count (ANE has ~119 limit)
    compile_count: i32,
    /// Compile budget limit (with safety margin)
    compile_budget: i32,
    /// Whether ANE is available
    ane_available: bool,
    /// Execution stats
    stats: ANETrainerStats,
}

/// Training execution statistics
#[derive(Debug, Default, Clone)]
pub struct ANETrainerStats {
    /// Number of kernels compiled (unique compilations)
    pub kernels_compiled: i32,
    /// Number of cache hits (reused compiled kernels)
    pub cache_hits: usize,
    /// Number of cache misses (had to compile)
    pub cache_misses: usize,
    /// Number of weight reloads (vs recompiles)
    pub weight_reloads: usize,
    /// Number of cache evictions
    pub cache_evictions: usize,
    /// Total ANE execution time (ms)
    pub ane_execution_time_ms: f64,
    /// Total compile time (ms)
    pub compile_time_ms: f64,
    /// Total weight reload time (ms)
    pub reload_time_ms: f64,
}

impl ANETrainerStats {
    /// Get compile savings: how many compiles were avoided
    pub fn compiles_saved(&self) -> usize {
        self.cache_hits
    }

    /// Get estimated time saved by caching (ms)
    pub fn estimated_time_saved_ms(&self) -> f64 {
        // Average compile ~4200ms, average reload ~494ms
        self.compiles_saved() as f64 * (4200.0 - 494.0)
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

impl ANETrainer {
    /// Create a new ANE trainer with specified cache size
    pub fn new(max_cache_size: usize) -> Result<Self, ANEError> {
        let ane_available = crate::wrapper::ANERuntime::init().is_ok();

        if ane_available {
            println!("✅ ANE Trainer initialized (compile-once + weight-reload mode)");
        } else {
            println!("⚠️  ANE not available. Will use CPU fallback.");
        }

        Ok(Self {
            kernel_cache: HashMap::new(),
            max_cache_size,
            compile_count: 0,
            compile_budget: 100, // Leave 19 headroom below ~119
            ane_available,
            stats: ANETrainerStats::default(),
        })
    }

    /// Get a cached executor or compile a new one
    ///
    /// This is the core optimization: kernels are compiled once and cached.
    /// Subsequent calls with the same template + sizes return the cached executor.
    ///
    /// For training loops, use `get_executor()` + `reload_weights()` pattern:
    /// ```no_run
    /// # use rustane::ane::trainer::{ANETrainer, ANEKernelTemplate};
    /// let mut trainer = ANETrainer::new(50).unwrap();
    /// let template = ANEKernelTemplate::rmsnorm(64, 128);
    ///
    /// // First step: compile (expensive ~4200ms)
    /// let exec = trainer.get_executor(&template, 16384, 16384).unwrap();
    ///
    /// // Every step after: reload weights (cheap ~494ms)
    /// exec.reload_weights(&[("rms_w.bin", &new_weights)]).unwrap();
    /// exec.write_input(0, &input_data).unwrap();
    /// exec.eval().unwrap();
    /// let output = exec.read_output_vec(0).unwrap();
    /// ```
    pub fn get_executor(
        &mut self,
        kernel: &ANEKernelTemplate,
        input_bytes: usize,
        output_bytes: usize,
    ) -> Result<&mut ANEExecutor, ANEError> {
        if !self.ane_available {
            return Err(ANEError::FrameworkNotFound);
        }

        let key = cache_key(kernel, input_bytes, output_bytes);

        // Check if already cached
        let is_cached = self.kernel_cache.contains_key(&key);

        if is_cached {
            // Cache hit
            let entry = self.kernel_cache.get_mut(&key).unwrap();
            entry.last_used = Instant::now();
            entry.use_count += 1;
            self.stats.cache_hits += 1;
            // No LRU order update needed - we use timestamp-based eviction
        } else {
            // Cache miss — need to compile
            self.check_compile_budget()?;

            let start = Instant::now();
            let request =
                ANECompileRequest::new(&kernel.mil, vec![input_bytes], vec![output_bytes]);

            let executor = request
                .compile()
                .map_err(|e| ANEError::CompileFailed(e.to_string()))?;
            let compile_time = start.elapsed();

            // Evict if at capacity
            if self.kernel_cache.len() >= self.max_cache_size {
                self.evict_lru();
            }

            // Store in cache
            self.kernel_cache.insert(
                key,
                CachedEntry {
                    executor,
                    last_used: Instant::now(),
                    use_count: 1,
                },
            );

            // Update stats
            self.compile_count += 1;
            self.stats.cache_misses += 1;
            self.stats.kernels_compiled += 1;
            self.stats.compile_time_ms += compile_time.as_secs_f64() * 1000.0;
        }

        // Return mutable ref (borrow checker is happy because we return at end of method)
        Ok(&mut self.kernel_cache.get_mut(&key).unwrap().executor)
    }

    /// Reload weights on a cached executor (training fast path)
    ///
    /// Uses `ANEExecutor::reload_weights()` instead of recompiling.
    /// This is ~8.5x faster than recompilation (~494ms vs ~4,200ms).
    ///
    /// Call this every training step after the initial compile.
    pub fn reload_weights(
        &mut self,
        kernel: &ANEKernelTemplate,
        input_bytes: usize,
        output_bytes: usize,
        weight_files: &[(&str, &[u8])],
    ) -> Result<(), ANEError> {
        let key = cache_key(kernel, input_bytes, output_bytes);

        let entry = self.kernel_cache.get_mut(&key).ok_or_else(|| {
            ANEError::CompileFailed(format!(
                "Kernel {} not in cache — call get_executor() first",
                kernel.id
            ))
        })?;

        let start = Instant::now();
        entry
            .executor
            .reload_weights(weight_files)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        let reload_time = start.elapsed();

        entry.last_used = Instant::now();
        self.stats.weight_reloads += 1;
        self.stats.reload_time_ms += reload_time.as_secs_f64() * 1000.0;

        Ok(())
    }

    /// Evict the least recently used kernel
    ///
    /// Uses timestamp-based eviction: finds the entry with the oldest last_used time.
    /// This is O(n) but eviction is rare (only when cache is full).
    fn evict_lru(&mut self) {
        // Find the key with the oldest last_used timestamp
        let oldest_key = self
            .kernel_cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(&key, _)| key);

        if let Some(key) = oldest_key {
            if self.kernel_cache.remove(&key).is_some() {
                self.stats.cache_evictions += 1;
            }
        }
    }

    /// Check if we have compile budget remaining
    fn check_compile_budget(&self) -> Result<(), ANEError> {
        if self.compile_count >= self.compile_budget {
            return Err(ANEError::CompileFailed(format!(
                "ANE compile budget exhausted ({}/{}). Use cached kernels + weight reload, or checkpoint restart.",
                self.compile_count, self.compile_budget
            )));
        }
        Ok(())
    }

    /// Get current stats
    pub fn get_stats(&self) -> ANETrainerStats {
        self.stats.clone()
    }

    /// Get compile budget status
    pub fn compile_budget_status(&self) -> (i32, i32) {
        (self.compile_count, self.compile_budget)
    }

    /// Clear kernel cache (frees all compiled kernels)
    pub fn clear_cache(&mut self) {
        self.kernel_cache.clear();
    }

    /// Check if compile limit is approaching (>80% used)
    pub fn is_compile_limit_approaching(&self) -> bool {
        self.compile_count as f32 / self.compile_budget as f32 > 0.8
    }

    /// Check if ANE is available
    pub fn is_ane_available(&self) -> bool {
        self.ane_available
    }

    /// Number of cached kernels
    pub fn cache_size(&self) -> usize {
        self.kernel_cache.len()
    }

    /// Execute RMSNorm on ANE (convenience method)
    pub fn rmsnorm_forward(
        &mut self,
        input: &[u8],
        channels: usize,
        seq_len: usize,
    ) -> Result<Vec<u8>, ANEError> {
        let kernel = ANEKernelTemplate::rmsnorm(channels, seq_len);
        let input_bytes = input.len();
        let output_bytes = input_bytes; // RMSNorm preserves size

        let start = Instant::now();
        let executor = self.get_executor(&kernel, input_bytes, output_bytes)?;
        executor
            .write_input(0, input)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        executor
            .eval()
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        let output = executor
            .read_output_vec(0)
            .map_err(|e| ANEError::EvalFailed(e.to_string()))?;
        self.stats.ane_execution_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_template_creation() {
        let kernel = ANEKernelTemplate::rmsnorm(64, 64);
        assert_eq!(kernel.id, "rmsnorm_64_64");
        assert!(kernel.has_dynamic_weights);
        assert_eq!(kernel.weight_size, 64 * 2);
    }

    #[test]
    fn test_dynamic_matmul_kernel() {
        let kernel = ANEKernelTemplate::dynamic_matmul(512, 512, 32);
        assert_eq!(kernel.id, "matmul_512_512_32");
        assert!(kernel.mil.contains("slice_by_size"));
        assert_eq!(kernel.weight_size, 512 * 512 * 2);
    }

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = ANEKernelTemplate::rmsnorm(64, 128);
        let k2 = ANEKernelTemplate::rmsnorm(64, 128);
        let key1 = cache_key(&k1, 1024, 1024);
        let key2 = cache_key(&k2, 1024, 1024);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_differs_by_size() {
        let k = ANEKernelTemplate::rmsnorm(64, 128);
        let key1 = cache_key(&k, 1024, 1024);
        let key2 = cache_key(&k, 2048, 2048);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_stats_default() {
        let stats = ANETrainerStats::default();
        assert_eq!(stats.kernels_compiled, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.weight_reloads, 0);
        assert_eq!(stats.compiles_saved(), 0);
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = ANETrainer::new(50);
        assert!(trainer.is_ok());
        let trainer = trainer.unwrap();
        assert_eq!(trainer.cache_size(), 0);
        // Compile budget should not be approaching
        assert!(!trainer.is_compile_limit_approaching());
    }

    #[test]
    fn test_stats_hit_rate() {
        let mut stats = ANETrainerStats::default();
        stats.cache_hits = 9;
        stats.cache_misses = 1;
        assert!((stats.cache_hit_rate() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_stats_time_saved() {
        let mut stats = ANETrainerStats::default();
        stats.cache_hits = 10;
        // Each hit saves ~3706ms (4200 - 494)
        let saved = stats.estimated_time_saved_ms();
        assert!((saved - 37060.0).abs() < 1.0);
    }

    #[test]
    fn test_lru_eviction_timestamp_based() {
        // Test the timestamp-based eviction logic without requiring ANE hardware
        // The evict_lru method uses timestamp-based eviction to find the oldest entry

        let trainer = ANETrainer::new(2).unwrap();

        // Verify initial state
        assert_eq!(trainer.cache_size(), 0);

        // The actual eviction happens in get_executor when cache is full
        // Since we can't compile without ANE hardware, this test just verifies
        // the trainer can be created and the cache_size method works
        // The actual eviction behavior is tested integration-style in ane_program_cache_tests.rs
    }
}
