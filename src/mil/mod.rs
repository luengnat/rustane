//! MIL (Model Intermediate Language) utilities
//!
//! This module provides helpers for constructing MIL programs for the ANE.
//! MIL is Apple's intermediate representation for neural network computations.

pub mod builder;
pub mod programs;
pub mod util;

// NOTE: codegen, graph, lora, passes are untracked files with pre-existing
// compilation errors (Op::RoPE match arm). They are not used by any
// committed code. Re-enable when fixed.

pub use builder::MILBuilder;
pub use programs::{
    conv1x1_compile_request, conv1x1_mil, linear_matmul_compile_request, rmsnorm_compile_request,
    rmsnorm_mil, LinearLayer,
};
pub use util::{generate_rope_blobs, generate_rope_tables, total_leaked_bytes, WeightBlob};
