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
    bwd_ffn_dh1_compile_request, bwd_ffn_dh1_mil, bwd_ffn_dh3_compile_request, bwd_ffn_dh3_mil,
    bwd_ffn_dx_compile_request, bwd_ffn_dx_mil, bwd_qkv_compile_request, bwd_qkv_mil,
    bwd_sdpa_bwd1_combined_compile_request, bwd_sdpa_bwd1_combined_mil,
    bwd_sdpa_bwd1_dpf_compile_request, bwd_sdpa_bwd1_dpf_mil, bwd_sdpa_bwd1_dvf_compile_request,
    bwd_sdpa_bwd1_dvf_mil, bwd_sdpa_bwd1_pf_compile_request, bwd_sdpa_bwd1_pf_mil,
    bwd_sdpa_bwd2_dkf_compile_request, bwd_sdpa_bwd2_dkf_mil, bwd_sdpa_bwd2_dqf_compile_request,
    bwd_sdpa_bwd2_dqf_mil, conv1x1_compile_request, conv1x1_mil, dynamic_matmul_rect_input_bytes,
    dynamic_matmul_rect_mil, dynamic_matmul_rect_output_bytes, linear_matmul_compile_request,
    pack_dynamic_matmul_rect_input, pack_rect_weights_into, rmsnorm_compile_request, rmsnorm_mil,
    LinearLayer,
};
pub use util::{generate_rope_blobs, generate_rope_tables, total_leaked_bytes, WeightBlob};
