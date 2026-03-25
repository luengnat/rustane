//! Apple Neural Engine (ANE) Integration Module
//!
//! Safe Rust bindings to Apple's Neural Engine for efficient deep learning inference and training.
//!
//! # Overview
//!
//! The ANE module provides a complete abstraction over Apple's private ANE framework via `objc2`
//! Rust bindings. It enables high-performance neural network computation on Apple Silicon devices
//! (M1/M2/M3/M4 chips).
//!
//! # Quick Start
//!
//! ```no_run
//! use rustane::ane::{ANECompileRequest, ANEKernel};
//!
//! // Define MIL code for a simple operation
//! let mil_code = r#"
//! #!irms6
//! main add_tensors(a: tensor<32xf32>, b: tensor<32xf32>) -> (c: tensor<32xf32>) {
//!     let c = a + b;
//!     return (c);
//! }
//! "#;
//!
//! // Prepare input data
//! let input_a = vec![1.0f32; 32];
//! let input_b = vec![2.0f32; 32];
//!
//! // Compile and execute
//! let mut kernel = ANECompileRequest::new(mil_code, vec![32*4], vec![32*4])
//!     .compile()
//!     .unwrap();
//!
//! kernel.write_input(0, &input_a).unwrap();
//! kernel.write_input(1, &input_b).unwrap();
//! kernel.eval().unwrap();
//!
//! let mut output_bytes = vec![0u8; 32*4];
//! kernel.read_output(0, &mut output_bytes).unwrap();
//! ```
//!
//! # Architecture
//!
//! The module is organized into core components:
//!
//! - **Error Handling** (`error.rs`): Comprehensive error types
//! - **Diagnostics** (`error_diagnostics.rs`): Error analysis and recovery suggestions
//! - **Retry Policy** (`retry_policy.rs`): Automatic retry with batch reduction
//! - **Fallback** (`fallback.rs`): Graceful CPU degradation
//! - **Logging** (`error_logging.rs`): Structured error logging
//! - **Runtime** (`runtime.rs`): Framework loading and MIL compilation
//! - **Kernel** (`kernel.rs`): ANEKernel lifecycle management
//! - **IOSurface** (`io_surface.rs`): RAII-safe memory management
//! - **Weight Blobs** (`weight_blob.rs`): Weight format abstraction
//! - **Profiler** (`profiler.rs`): Kernel timing and performance analysis
//! - **Operator Fusion** (`operator_fusion.rs`): Fuse operations for performance
//! - **Tiling** (`tiling.rs`): Split large operations into ANE-compatible chunks
//! - **Program Cache** (`program_cache.rs`): Cache compiled kernels for reuse
//! - **Training Architecture** (`training_architecture.rs`): ANE training with compile budget
//! - **Trainer** (`trainer.rs`): Hybrid ANE+CPU training with caching
//!
//! # Error Handling & Recovery
//!
//! ## Automatic Retry with Batch Reduction
//!
//! ```no_run
//! use rustane::ane::{RetryPolicy, RetryConfig, ANEError, Result};
//!
//! let policy = RetryPolicy::with_config(RetryConfig {
//!     max_attempts: 3,
//!     enable_batch_reduction: true,
//!     ..Default::default()
//! }).unwrap();
//!
//! let result = policy.execute(|batch_fraction| -> Result<Vec<f32>> {
//!     // Your ANE operation here
//!     // batch_fraction ranges from 1.0 (full) to 0.125 (1/8th)
//!     Ok(vec![])
//! });
//! ```
//!
//! ## Graceful CPU Fallback
//!
//! ```no_run
//! use rustane::ane::{FallbackExecutor, FallbackStrategy, ANEError, Result};
//!
//! let mut executor = FallbackExecutor::with_strategy(FallbackStrategy::ANEWithCPUFallback);
//!
//! let result = executor.execute(
//!     || { /* ANE operation */ Err(ANEError::EvalFailed("test".to_string())) },
//!     || { /* CPU fallback */ Ok(vec![1.0, 2.0, 3.0]) },
//!     "my_operation"
//! );
//! ```
//!
//! # Design Philosophy
//!
//! **CPU-ANE Coordination**: The module follows a CPU ↔ ANE split design:
//! - Heavy compute (forward/backward passes) → ANE
//! - Data coordination, loading, optimization → CPU
//! - This balance maximizes ANE utilization while keeping the system flexible
//!
//! **Type Safety**: All operations use Rust's type system to prevent:
//! - Shape mismatches (tensor dimensions validated at compile/runtime)
//! - Memory safety (RAII patterns, no manual cleanup)
//! - Silent failures (Result<T> throughout)
//!
//! **Numerical Stability**: ANE operations use:
//! - Per-row quantization with scale tracking
//! - Epsilon guards for division by zero
//! - Numerically stable softmax (max subtraction)
//!
//! # Key Components
//!
//! ## ANEError
//!
//! Comprehensive error enum covering all failure modes:
//! - Framework not found
//! - Compilation failures
//! - Evaluation failures
//! - I/O Surface errors
//! - Shape mismatches
//! - Configuration errors
//!
//! ## ANERuntime & Compilation
//!
//! Loads the private ANE framework and compiles MIL code to ANE kernels:
//! ```ignore
//! let request = ANECompileRequest {
//!     mil_text: "func matmul...".to_string(),
//!     weights: HashMap::new(),
//!     input_sizes: vec![4096],
//!     output_sizes: vec![1024],
//! };
//! let kernel = request.compile()?;
//! ```
//!
//! ## ANEKernel Lifecycle
//!
//! Manages kernel execution with safe I/O:
//! ```ignore
//! let mut kernel = ANECompileRequest::compile()?;
//! kernel.write_input(0, &input_data)?;
//! kernel.eval()?;
//! let output = kernel.read_output(0)?;
//! ```
//!
//! ## IOSurface Wrapper
//!
//! RAII-safe memory management for ANE data transfer:
//! ```ignore
//! let mut surface = IOSurface::new(buffer_size)?;
//! surface.write(&data)?;
//! let result = surface.read()?;
//! // Automatically cleaned up when dropped
//! ```
//!
//! ## Weight Blob Builders
//!
//! Format abstraction for multiple weight precisions:
//! ```ignore
//! // FP32 format
//! let blob_fp32 = WeightBlob::from_f32(&weights, rows, cols);
//!
//! // Quantized int8 with scale
//! let (blob_int8, scales) = WeightBlob::quantize_f32(&weights, rows, cols);
//! ```
//!
//! # Integration with TransformerANE
//!
//! The ANE module integrates with the training system:
//! 1. MIL code generated by layers/mil_gen.rs
//! 2. Weights converted to appropriate format via WeightBlob
//! 3. Kernel compiled via ANERuntime
//! 4. Forward/backward passes executed via ANEKernel
//! 5. Gradients transferred back to CPU for optimization
//!
//! # Platform Requirements
//!
//! - **OS**: macOS 15+ (Sequoia)
//! - **Hardware**: Apple Silicon with ANE (M1/M2/M3/M4)
//! - **Rust**: 1.70+ with 2021 edition
//!
//! # Limitations & Future Work
//!
//! Current Phase 2 implementation:
//! - MIL code generation for attention and FFN forward passes
//! - CPU-based backward passes with cached activations
//! - Numerical gradient validation
//!
//! Phase 3 will add:
//! - Full ANE backward passes for all layers
//! - Advanced optimization (gradient accumulation on ANE, quantized training)
//! - Production checkpointing and experiment tracking

pub(crate) mod blobs;
/// ANE-specific error types.
pub mod error;
/// Detailed error diagnostics and recovery analysis
pub mod error_diagnostics;
/// Error logging and reporting
pub mod error_logging;
/// Graceful degradation strategies
pub mod fallback;
/// IOSurface RAII wrapper for ANE I/O operations.
pub mod io_surface;
/// ANE kernel wrapper for managing compiled models and I/O operations.
pub mod kernel;
/// Memory pool for efficient IOSurface allocation
// pub mod memory_pool; // TODO: uncomment when memory_pool.rs compiles
/// MIL code generator for ANE operations
pub(crate) mod mil_generator;
/// Multi-ANE distributed training support
pub mod multi_ane;
/// Operator fusion for performance optimization
pub mod operator_fusion;
/// ANE profiler for kernel timing and performance analysis
pub mod profiler;
/// ANE program cache for avoiding recompilation
pub mod program_cache;
/// Automatic retry with adaptive batch size reduction
pub mod retry_policy;
/// Low-level runtime and compile/load/eval support for the private ANE APIs.
pub mod runtime;
pub(crate) mod sys;
/// Tiling for large operations
pub mod tiling;
/// ANE trainer
pub mod trainer;
/// ANE training architecture
pub mod training_architecture;
/// Weight blob builders for ANE-compatible formats.
pub mod weight_blob;

pub use error::ANEError;
pub use error_diagnostics::{ErrorAggregator, ErrorCategory, ErrorDiagnostic};
pub use error_logging::{ErrorLog, ErrorLogEntry, ErrorReporter, ErrorSeverity};
pub use fallback::{FallbackExecutor, FallbackResult, FallbackStats, FallbackStrategy};
pub use io_surface::IOSurface;
pub use kernel::ANEKernel;
// pub use memory_pool::{
//     MemoryPool, PoolConfig, PoolStats, PooledBuffer, SharedMemoryPool, SizeClassStats,
// };
pub use multi_ane::{
    detect_ane_devices, get_optimal_device_count, per_device_batch_size, validate_device_count,
    ANEDeviceInfo, MultiANEConfig,
};
pub use operator_fusion::{ActivationType, FusedKernelRegistry, FusedKernelType};
pub use profiler::{
    ANEProfiler, KernelStats, KernelTimer, KernelTiming, ProfilerMetrics, StepProfile,
};
pub use retry_policy::{
    execute_with_retry, RetryConfig, RetryPolicy, RetryResult, RetryableOperation,
};
pub use runtime::ANECompileRequest;
pub use weight_blob::WeightBlob;

/// Result type for ANE operations
pub type Result<T> = std::result::Result<T, ANEError>;
