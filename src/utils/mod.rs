//! Utility functions for model development
//!
//! This module provides practical utilities for:
//! - Loading and saving model weights
//! - Converting between weight formats (FP32, FP16)
//! - Benchmarking inference performance
//! - Profiling memory and ANE utilization

pub mod benchmark;
pub mod conversion;
pub mod loading;
pub mod profiling;

pub use benchmark::{benchmark_inference, BenchmarkResults};
pub use conversion::{fp16_to_fp32, fp32_to_fp16, transpose_weights};
pub use loading::{load_weights_from_file, save_weights_to_file};
pub use profiling::{ANEProfiler, MemoryProfiler, ProfileReport};
