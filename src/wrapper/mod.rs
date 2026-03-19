//! Safe wrappers for ANE resources
//!
//! This module provides memory-safe, idiomatic Rust wrappers around the raw FFI bindings.
//! All resources are managed using RAII (Resource Acquisition Is Initialization) and
//! are automatically cleaned up when dropped.
//!
//! # Architecture
//!
//! - [`ANERuntime`]: Manages ANE framework initialization
//! - [`ANECompiler`]: Compiles MIL programs into executable kernels
//! - [`ANEExecutor`]: Executes compiled kernels and manages I/O
//! - [`ANETensor`]: Type-safe tensor data wrapper
//!
//! # Thread Safety
//!
//! ANE resources are **not thread-safe** and should not be shared across threads.
//! All wrapper types implement `!Send` and `!Sync` to prevent accidental misuse.

pub mod cache;
pub mod compiler;
pub mod executor;
pub mod runtime;
pub mod tensor;

pub use cache::KernelCache;
pub use compiler::ANECompiler;
pub use executor::ANEExecutor;
pub use runtime::ANERuntime;
pub use tensor::ANETensor;
