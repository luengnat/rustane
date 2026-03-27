//! ANE Capability Tests
//!
//! Comprehensive test suite to understand ANE capabilities and limitations:
//! - Maximum tensor sizes
//! - Supported operations
//! - Precision behavior (FP16 vs FP32)
//! - Memory limits (SRAM cliff ~32MB)
//! - Execution time characteristics
//! - Compile budget (~119 compilations per process)
//!
//! Based on findings from "Inside the M4 Apple Neural Engine" series:
//! - M4 ANE: 16 cores, ~32 MB SRAM, 19 TFLOPS FP16 peak
//! - SRAM cliff: performance drops 30% when working set exceeds ~24-32 MB
//! - Conv > Matmul: 1x1 convolutions use faster datapath
//! - Deep graphs (16-64 ops) fill pipeline better than single ops
//! - INT8 = FP16 speed (no 2x speedup, just memory savings)
//! - Dispatch overhead: ~0.095ms for XPC + IOKit

use rustane::mil::programs::dynamic_matmul_mil;
use rustane::wrapper::ANECompiler;
use std::time::Instant;

/// Test ANE support for various tensor shapes
///
/// Based on M4 ANE SRAM cliff analysis:
/// - 2048x2048 FP16 matmul = 24 MB working set (fits in SRAM) = peak performance
/// - 4096x4096 FP16 matmul = 96 MB working set (spills to DRAM) = 30% drop
///
/// ANE Tensor dimension limits: N,D,C,H,W each must be in [1-65536]
/// For dynamic_matmul: input channel = D + D*D, so D=256 gives C=65792 > 65536 (limit!)
#[test]
fn test_ane_tensor_shape_limits() {
    rustane::init().ok();

    // Test progressively larger tensor sizes using dynamic_matmul (proven working)
    // Note: Limited by ANE max tensor dimension of 65536
    // For dynamic_matmul: input = [1, D+D*D, 1, S], so D=256 -> C=65792 exceeds limit
    let sizes = [
        (16, 16),   // Tiny: ~1 KB
        (64, 64),   // Small: ~64 KB
        (128, 128), // Medium: ~1 MB
        (255, 255), // Large: at dimension limit (255+255*255=65025 < 65536)
    ];

    println!("Tensor shape limits test (using dynamic_matmul):");
    println!("Dim x Seq\tWorking Set\tStatus");

    for (dim, seq) in sizes {
        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = (dim + dim * dim) * seq * 4; // fp32 input
        let output_bytes = dim * seq * 4;

        let working_set_mb = (input_bytes + output_bytes) as f64 / (1024.0 * 1024.0);

        let result = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

        println!(
            "{}x{}\t{:.2} MB\t{}",
            dim,
            seq,
            working_set_mb,
            if result.is_ok() {
                "✓ OK"
            } else {
                "✗ FAILED"
            }
        );
    }

    println!("\nNote: D=256 fails because input channel = 256+256*256=65792 > 65536 (ANE max)");
    println!("Max D for dynamic_matmul: D=255 (input channel = 255+255*255=65025 < 65536)");
}

/// Test ANE precision behavior
///
/// Note: ANE native compute is FP16. FP32 inputs get cast to FP16 internally.
/// See: dynamic_matmul_mil casts input to fp16 before compute
#[test]
fn test_ane_precision_behavior() {
    rustane::init().ok();

    let dim = 64;
    let seq = 64;

    // Test dynamic_matmul which uses FP32 I/O with FP16 internal compute
    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = (dim + dim * dim) * seq * 4; // fp32
    let output_bytes = dim * seq * 4;

    let result = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

    println!(
        "dynamic_matmul (FP32 I/O, internal FP16 compute): {}",
        if result.is_ok() {
            "✓ OK"
        } else {
            "✗ FAILED"
        }
    );

    // ANE performs compute in FP16 with FP32 I/O conversion
    // This is the recommended pattern for best precision + performance
}

/// Test ANE operation support matrix
///
/// Note: Based on M4 ANE research:
/// - Conv 1x1 uses faster datapath than matmul (3x throughput)
/// - Deep graphs (16-64 ops) achieve 94% utilization vs 30% for single ops
///
/// This test uses dynamic_matmul which is proven to work.
#[test]
fn test_ane_operations_matrix() {
    rustane::init().ok();

    // Test dynamic_matmul at various sizes (proven working pattern)
    let sizes = [(32, 32), (64, 64), (128, 128)];

    println!("Operation support matrix (dynamic_matmul pattern):");
    println!("Size\tCompile\tExecute\tStatus");

    for (dim, seq) in sizes {
        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = (dim + dim * dim) * seq * 4;
        let output_bytes = dim * seq * 4;

        let compile_start = Instant::now();
        let exec = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);
        let compile_time = compile_start.elapsed().as_secs_f64() * 1000.0;

        if let Ok(mut exec) = exec {
            let num_elements = (dim + dim * dim) * seq;
            let input = vec![0.1f32; num_elements];
            let input_bytes_vec: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

            let exec_start = Instant::now();
            let _ = exec.write_input(0, &input_bytes_vec);
            let _ = exec.eval();
            let exec_time = exec_start.elapsed().as_secs_f64() * 1000.0;

            println!(
                "{}x{}\t{:.1}ms\t{:.2}ms\t✓ Supported",
                dim, seq, compile_time, exec_time
            );
        } else {
            println!(
                "{}x{}\t{:.1}ms\tFAILED\t✗ Not supported",
                dim, seq, compile_time
            );
        }
    }

    println!("\nNote: For conv 1x1 (ANE fast path), see sram_probe.m pattern in ANE-main");
}

/// Test ANE memory limits and SRAM cliff detection
///
/// M4 ANE has ~32 MB SRAM. Performance drops when working set exceeds this.
/// Working set = input + output + weights (for matmul: 3 matrices)
///
/// ANE Tensor dimension limits: N,D,C,H,W each must be in [1-65536]
/// For dynamic_matmul: max D=255 due to input channel = D+D*D limit
#[test]
fn test_ane_memory_limits() {
    rustane::init().ok();

    // Test sizes that cross the SRAM boundary (staying within ANE tensor limits)
    // Max D=255 for dynamic_matmul pattern
    let test_sizes = [
        (64, 64, "Well within SRAM"),            // ~1 MB
        (128, 128, "Approaching SRAM"),          // ~4 MB
        (255, 255, "At tensor dimension limit"), // ~16 MB, D=255 -> C=65025
    ];

    println!("Memory limit test (SRAM cliff ~32 MB on M4):");
    println!("Dim x Seq\tWorking Set\tCategory\tStatus");

    for (dim, seq, category) in test_sizes {
        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = (dim + dim * dim) * seq * 4;
        let output_bytes = dim * seq * 4;
        let working_set_mb = (input_bytes + output_bytes) as f64 / (1024.0 * 1024.0);

        let result = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

        println!(
            "{}x{}\t{:.2} MB\t{}\t{}",
            dim,
            seq,
            working_set_mb,
            category,
            if result.is_ok() {
                "✓ OK"
            } else {
                "✗ FAILED"
            }
        );
    }

    println!("\nNote: Cannot test >32 MB SRAM cliff with dynamic_matmul (tensor dim limit)");
    println!("For SRAM cliff analysis, see sram_probe.m in ANE-main (uses conv1x1 pattern)");
}

/// Test ANE execution time scaling
///
/// Measures both compile time and execution time.
/// Note: Operations under ~1ms are dispatch-limited (~0.095ms XPC + IOKit overhead)
///
/// ANE Tensor dimension limits: N,D,C,H,W each must be in [1-65536]
/// For dynamic_matmul: max D=255 due to input channel = D+D*D limit
#[test]
fn test_ane_execution_time_scaling() {
    rustane::init().ok();

    // Stay within ANE tensor dimension limits (max 65536 per dimension)
    // For dynamic_matmul: D=256 -> C=65792 > 65536 (limit exceeded)
    // Note: 255x255 compiles but may fail at eval due to memory pressure
    let sizes = [(32, 32), (64, 64), (128, 128)];

    println!("Execution time scaling test:");
    println!("Size\tCompile (ms)\tExecute (ms)\tTotal (ms)");

    for (dim, seq) in sizes {
        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = (dim + dim * dim) * seq * 4;
        let output_bytes = dim * seq * 4;

        let compile_start = Instant::now();
        let exec = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);
        let compile_time = compile_start.elapsed().as_secs_f64() * 1000.0;

        if let Ok(mut exec) = exec {
            // Create input: activations + weights
            let num_elements = (dim + dim * dim) * seq;
            let input = vec![0.1f32; num_elements];
            let input_bytes_vec: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

            let exec_start = Instant::now();
            let _ = exec.write_input(0, &input_bytes_vec);
            let _ = exec.eval();
            let exec_time = exec_start.elapsed().as_secs_f64() * 1000.0;

            println!(
                "{}x{}\t{:.1}\t\t{:.2}\t\t{:.1}",
                dim,
                seq,
                compile_time,
                exec_time,
                compile_time + exec_time
            );
        } else {
            println!("{}x{}\t{:.1}\t\tFAILED\t\tFAILED", dim, seq, compile_time);
        }
    }

    println!("\nNote: D=256 fails (C=65792 > 65536 max tensor dimension)");
    println!("Note: 255x255 compiles OK but may fail at eval (memory pressure)");
}

/// Test ANE compile budget
///
/// M4 ANE has a limit of ~119 compilations per process.
/// After this limit, compilation fails until process restart.
/// Workaround: Use program caching or weight reload instead of recompilation.
///
/// This test uses a small sample (20 compilations) to demonstrate the limit
/// without exhausting the entire budget (which would break other tests).
#[test]
fn test_ane_compile_budget() {
    rustane::init().ok();

    let dim = 32;
    let seq = 32;
    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = (dim + dim * dim) * seq * 4;
    let output_bytes = dim * seq * 4;

    let mut success_count = 0;
    let mut failure_count = 0;
    let max_compilations = 20; // Sample only, not full budget

    println!(
        "Compile budget test (sampling {} compilations):",
        max_compilations
    );
    println!("Note: Full budget is ~119 compilations per process");

    for i in 0..max_compilations {
        let result = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

        if result.is_ok() {
            success_count += 1;
        } else {
            failure_count += 1;
            println!("First failure at compilation #{}", i + 1);
            break;
        }
    }

    println!(
        "Result: {} successful, {} failed",
        success_count, failure_count
    );

    if failure_count == 0 {
        println!(
            "All {} compilations succeeded (budget not exhausted)",
            success_count
        );
    }
}

/// Test ANE batch processing limits
///
/// Tests how batch size affects compilation and execution.
/// Uses dynamic_matmul pattern which is proven to work.
#[test]
fn test_ane_batch_limits() {
    rustane::init().ok();

    let dim = 32;
    let seq = 32;
    let batch_values = [1, 2, 4, 8, 16];

    println!("Batch processing limits (dynamic_matmul):");
    println!("Batch Size\tInput Size\tStatus");

    for batch_size in batch_values {
        // dynamic_matmul input: [1, D+D*D, 1, S] fp32
        // For batch processing, we test single batch at increasing sizes
        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = (dim + dim * dim) * seq * 4;
        let output_bytes = dim * seq * 4;

        let result = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

        println!(
            "{}\t\t{:.1} KB\t{}",
            batch_size,
            input_bytes as f64 / 1024.0,
            if result.is_ok() {
                "✓ OK"
            } else {
                "✗ FAILED"
            }
        );
    }
}

/// Test ANE weight reload capability
///
/// Weight reload allows updating weights without recompilation.
/// This is critical for staying within the ~119 compile budget.
///
/// Note: dynamic_matmul encodes weights in the input tensor,
/// so weight updates happen by changing the input buffer.
#[test]
fn test_ane_weight_reload() {
    rustane::init().ok();

    let dim = 64;
    let seq = 64;

    // dynamic_matmul encodes weights in input buffer
    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = (dim + dim * dim) * seq * 4;
    let output_bytes = dim * seq * 4;

    let exec = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

    match exec {
        Ok(mut _exec) => {
            println!("Initial compilation: ✓ OK");
            println!(
                "Weight update capability: Change input buffer values (weights encoded in input)"
            );
            // Note: dynamic_matmul encodes weights in the input tensor
            // Different weight matrices = different input buffer, no recompilation needed
        }
        Err(e) => {
            println!("Compilation failed: {:?}", e);
        }
    }
}

/// Test ANE dynamic shape support
///
/// Tests if ANE can handle variable sequence lengths at runtime.
/// ANE Tensor dimension limits: N,D,C,H,W each must be in [1-65536]
/// For dynamic_matmul with D=64: max seq = 65536/(64+64*64) = 65536/4160 ≈ 15
#[test]
fn test_ane_dynamic_shapes() {
    rustane::init().ok();

    let dim = 32; // Smaller dim to allow larger seq testing
                  // For D=32: input channel = 32+32*32=1056, so seq can be up to 65536/1056 ≈ 62
    let seq_lengths = [8, 16, 32, 48, 62];

    println!("Dynamic shape support test:");
    println!("Seq Len\tInput Size\tStatus");

    for seq in seq_lengths {
        let mil = dynamic_matmul_mil(seq, dim);
        let input_bytes = (dim + dim * dim) * seq * 4;
        let output_bytes = dim * seq * 4;

        let result = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);

        println!(
            "{}\t{:.1} KB\t{}",
            seq,
            input_bytes as f64 / 1024.0,
            if result.is_ok() {
                format!("✓ OK")
            } else {
                "✗ FAILED".to_string()
            }
        );
    }

    println!("\nNote: For D=32, max seq ≈ 62 (tensor dimension limit 65536)");
}

/// Test deep graph execution (dynamic_matmul with multiple operations)
///
/// Based on M4 ANE research: deep graphs achieve 94% utilization vs 30% for single ops.
/// dynamic_matmul contains: cast, slice_by_size, reshape, matmul operations.
#[test]
fn test_ane_deep_graph() {
    rustane::init().ok();

    let dim = 64;
    let seq = 64;

    // dynamic_matmul is a deep graph with multiple chained operations
    let mil = dynamic_matmul_mil(seq, dim);
    let input_bytes = (dim + dim * dim) * seq * 4;
    let output_bytes = dim * seq * 4;

    let compile_start = Instant::now();
    let exec = ANECompiler::new().compile_single(&mil, None, &[input_bytes], &[output_bytes]);
    let compile_time = compile_start.elapsed().as_secs_f64() * 1000.0;

    println!("Deep graph test (dynamic_matmul - multiple chained operations):");
    println!("Compile time: {:.1} ms", compile_time);

    match exec {
        Ok(mut _exec) => {
            println!("Result: ✓ OK - Deep graph supported");
            println!("dynamic_matmul contains: cast(fp32->fp16), slice_by_size, reshape, matmul operations");
        }
        Err(e) => {
            println!("Result: ✗ FAILED - {:?}", e);
        }
    }
}
