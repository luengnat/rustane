//! MIL (Model Intermediate Language) utilities
//!
//! This module provides helpers for constructing MIL programs for the ANE.
//! MIL is Apple's intermediate representation for neural network computations.

pub mod builder;
pub mod cpu_fallback;
pub mod programs;
pub mod util;

// NOTE: codegen, graph, lora, passes are untracked files with pre-existing
// compilation errors (Op::RoPE match arm). They are not used by any
// committed code. Re-enable when fixed.

pub use builder::{MILBuilder, SizeValidationResult};
pub use cpu_fallback::{
    embedding_lookup_cpu, gelu_cpu, layer_norm_cpu, reduce_mean_cpu, rms_norm_cpu, rope_cpu,
    should_use_ane, silu_cpu, ExecutionTarget, ANE_MIN_ELEMENTS,
};
pub use programs::{
    bwd_ffn_compile_request, bwd_ffn_mil, bwd_qkv_compile_request, bwd_qkv_mil,
    conv1x1_compile_request, conv1x1_mil, linear_matmul_compile_request, rmsnorm_compile_request,
    rmsnorm_mil, LinearLayer,
};
pub use util::{generate_rope_blobs, generate_rope_tables, total_leaked_bytes, WeightBlob};
