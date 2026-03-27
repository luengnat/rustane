//! Comprehensive ANE Constraint Tester
//!
//! Tests all 20 constraints from the Orion paper (arXiv:2603.06728).
//! Uses subprocess isolation per test to survive ANE native crashes.
//!
//! Run with: cargo run --example test_ane_constraints

use std::process::Command;
use std::time::Instant;

fn main() {
    eprintln!();
    eprintln!("  ============================================================");
    eprintln!("  ANE Constraint Tester (Orion paper, 20 constraints)");
    eprintln!("  ============================================================");

    // Build the test binary first
    eprintln!("\n  Building test binary...");
    let build_status = Command::new("cargo")
        .args(["build", "--example", "test_ane_constraint"])
        .output()
        .expect("failed to run cargo build");
    if !build_status.status.success() {
        eprintln!(
            "  Build failed: {}",
            String::from_utf8_lossy(&build_status.stderr)
        );
        return;
    }
    eprintln!("  Build OK.");

    // ============================================================
    // MIL IR Restrictions
    // ============================================================
    eprintln!("\n  ════════════════════════════════════════════════════════════");
    eprintln!("  MIL IR Restrictions");
    eprintln!("  ════════════════════════════════════════════════════════════");

    // #1: concat MIL op
    eprintln!("\n── #1: concat MIL op (should fail per paper) ──");
    run_test("concat", "concat_basic");

    // #10: gelu activation
    eprintln!("\n── #10: gelu activation (should fail per paper) ──");
    run_test("gelu", "gelu_basic");

    // #12: matmul transpose flags
    eprintln!("\n── #12: matmul with transpose (should work with named consts) ──");
    run_test("matmul_transpose", "matmul_transpose_named");

    // #13: conv bias
    eprintln!("\n── #13: conv with bias (should fail per paper) ──");
    run_test("conv_bias", "conv_bias_basic");

    // ============================================================
    // Memory and I/O Constraints
    // ============================================================
    eprintln!("\n  ════════════════════════════════════════════════════════════");
    eprintln!("  Memory and I/O Constraints");
    eprintln!("  ════════════════════════════════════════════════════════════");

    // #4: minimum IOSurface ~49KB
    eprintln!("\n── #4: minimum IOSurface size ──");
    run_test("min_surface", "min_surface_tiny");
    run_test("min_surface", "min_surface_small");
    run_test("min_surface", "min_surface_ok");

    // #8: BLOBFILE offset
    eprintln!("\n── #8: BLOBFILE offset (64 vs 128) ──");
    run_test("blobfile_offset", "blobfile_offset_64");
    run_test("blobfile_offset", "blobfile_offset_128");

    // #2: multi-output uniform sizes
    eprintln!("\n── #2: multi-output with uniform sizes ──");
    run_test("multi_output", "multi_output_uniform");
    run_test("multi_output", "multi_output_nonuniform");

    // #3: multi-output alphabetical ordering
    eprintln!("\n── #3: multi-output alphabetical ordering ──");
    run_test("multi_output_order", "multi_output_alpha");
    run_test("multi_output_order", "multi_output_reverse");

    // ============================================================
    // Performance Characteristics
    // ============================================================
    eprintln!("\n  ════════════════════════════════════════════════════════════");
    eprintln!("  Performance Characteristics");
    eprintln!("  ════════════════════════════════════════════════════════════");

    // #16: 32K-channel conv rejection
    eprintln!("\n── #16: large-channel convolutions ──");
    run_test("large_channels", "conv_4k_channels");
    run_test("large_channels", "conv_16k_channels");
    run_test("large_channels", "conv_32k_channels");

    // #17: conv1x1 vs matmul speed
    eprintln!("\n── #17: conv1x1 vs matmul speed comparison ──");
    run_test("conv_vs_matmul", "conv_64x64");
    run_test("conv_vs_matmul", "matmul_64x64");
    run_test("conv_vs_matmul", "conv_128x384");
    run_test("conv_vs_matmul", "matmul_128x384");

    // ============================================================
    // Fused Programs (the important ones)
    // ============================================================
    eprintln!("\n  ════════════════════════════════════════════════════════════");
    eprintln!("  Fused Programs (Orion-style)");
    eprintln!("  ════════════════════════════════════════════════════════════");

    // Fused FFN (conv + SiLU + conv, with concat taps)
    eprintln!("\n── Fused FFN (W1+W3 → SiLU → W2, with taps) ──");
    run_test("fused_ffn", "fused_ffn_small");
    run_test("fused_ffn", "fused_ffn_medium");

    // Fused FFN with concat (save intermediates for backward)
    eprintln!("\n── Fused FFN with concat taps ──");
    run_test("fused_ffn_taps", "fused_ffn_taps_small");

    // Native layer_norm for RMSNorm hack
    eprintln!("\n── Native layer_norm (for ANEMLL RMSNorm trick) ──");
    run_test("layer_norm", "layer_norm_basic");
    run_test("layer_norm", "layer_norm_with_weight");

    // RMSNorm via concat trick (ANEMLL-style)
    eprintln!("\n── RMSNorm via concat([-x, x]) → layer_norm trick ──");
    run_test("rmsnorm_trick", "rmsnorm_trick_basic");
    run_test("rmsnorm_trick", "rmsnorm_trick_with_weight");

    // Manual RMSNorm (mul/reduce_sum/pow — what we had before)
    eprintln!("\n── Manual RMSNorm (mul/reduce_sum/pow) ──");
    run_test("rmsnorm_manual", "rmsnorm_manual_basic");

    // softmax
    eprintln!("\n── softmax op ──");
    run_test("softmax", "softmax_basic");

    // sigmoid
    eprintln!("\n── sigmoid op ──");
    run_test("sigmoid", "sigmoid_basic");

    // ============================================================
    // Multi-weight programs
    // ============================================================
    eprintln!("\n  ════════════════════════════════════════════════════════════");
    eprintln!("  Multi-Weight Programs");
    eprintln!("  ════════════════════════════════════════════════════════════");

    // Dual conv (W1 + W3 in one program)
    eprintln!("\n── Dual conv (W1+W3 parallel, 2 weight files) ──");
    run_test("dual_conv", "dual_conv_basic");

    // QKV fused (Wq + Wk + Wv in one program)
    eprintln!("\n── QKV fused (Wq+Wk+Wv, 3 weight files) ──");
    run_test("qkv_fused", "qkv_fused_basic");

    // Multi-input (two inputs to one program)
    eprintln!("\n── Multi-input (2 IOSurface inputs) ──");
    run_test("multi_input", "multi_input_add");

    eprintln!("\n  Done.");
}

fn run_test(category: &str, name: &str) {
    eprint!("  [{:>18}] {:<30} ... ", category, name);
    let t0 = Instant::now();

    let output = Command::new("./target/debug/examples/test_ane_constraint")
        .env("RUSTANE_TEST_NAME", name)
        .output();

    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    match output {
        Ok(out) => {
            let status = out.status;
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&out.stderr);

            if status.success() && !stdout.is_empty() {
                // Parse "OK detail" or "PASS detail"
                if stdout.starts_with("OK") {
                    let detail = stdout.strip_prefix("OK ").unwrap_or("");
                    eprintln!("✓ {} ({:.0}ms)", detail, elapsed);
                } else {
                    eprintln!("✓ {} ({:.0}ms)", stdout, elapsed);
                }
            } else {
                let code = status.code().unwrap_or(-1);
                let reason = classify_failure(&stderr, code);
                eprintln!("✗ {} ({:.0}ms)", reason, elapsed);
            }
        }
        Err(e) => eprintln!("✗ spawn: {} ({:.0}ms)", e, elapsed),
    }
}

fn classify_failure(stderr: &str, code: i32) -> String {
    if stderr.contains("Program Inference error") {
        "inference err".into()
    } else if stderr.contains("compile") && (stderr.contains("failed") || stderr.contains("error"))
    {
        "compile err".into()
    } else if code == -11 || code == 139 {
        "SIGSEGV".into()
    } else if code == -6 || code == 134 {
        "SIGABRT".into()
    } else if code == 137 {
        "timeout".into()
    } else if code == 0 {
        // Exited OK but no stdout — empty output
        if stderr.contains("all zeros") {
            "all zeros".into()
        } else {
            let line = stderr.lines().next().unwrap_or("");
            format!("exit 0, stderr: {}", &line[..line.len().min(80)])
        }
    } else {
        let line = stderr.lines().next().unwrap_or("");
        format!("exit {code}: {}", &line[..line.len().min(60)])
    }
}
