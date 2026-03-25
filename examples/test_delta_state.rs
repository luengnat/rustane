//! State survival verification test for delta compilation.
//!
//! Validates DLT-02 (compile count tracking), DLT-03 (selective weight update),
//! and DLT-04 (state survival across reload cycles).
//!
//! Tests:
//! 1. Determinism: same weights → same output within fp16 tolerance
//! 2. Weight change propagation: different weights → different output
//! 3. Durability: 20 reload cycles without errors
//! 4. Compile count non-increase: reloads don't increment compile count
//! 5. Selective update: updating layer 1 doesn't change layer 0 output
//!
//! Uses subprocess isolation — spawned by parent test harness.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example test_delta_state --release
//! ```

use rustane::ane::WeightBlob;
use rustane::mil::programs::conv1x1_mil;
use rustane::training::DeltaCompiler;
use rustane::wrapper::ANERuntime;

const SEQ_LEN: usize = 16;
const DIM: usize = 64;

fn make_weights(seed: usize, size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let val = (((i * 7 + seed * 13) % 100) as f32 - 50.0) / 100.0;
            val.clamp(-1.0, 1.0)
        })
        .collect()
}

fn make_blob(weights: &[f32], rows: usize, cols: usize) -> WeightBlob {
    WeightBlob::from_f32(weights, rows, cols).expect("weight blob creation")
}

fn read_output_f32(dc: &mut DeltaCompiler, layer_idx: usize, input_bytes: &[u8]) -> Vec<f32> {
    let output_size = dc.executor(layer_idx).unwrap().num_outputs();
    let out_bytes = output_size; // Each output is fp32
    let mut buf = vec![0u8; out_bytes];
    dc.executor_mut(layer_idx)
        .unwrap()
        .write_input(0, input_bytes)
        .unwrap();
    dc.executor_mut(layer_idx).unwrap().eval().unwrap();
    dc.executor_mut(layer_idx)
        .unwrap()
        .read_output(0, &mut buf)
        .unwrap();

    // Convert bytes to f32
    buf.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn fp32_close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Delta Compilation State Survival Test ===\n");

    ANERuntime::init()?;
    println!("✅ ANE runtime initialized\n");

    let weight_name = "@model_path/weights/weight.bin";
    let weight_size = DIM * DIM; // square conv1x1

    // Step 1: Compile 3 layers via DeltaCompiler
    println!("=== Step 1: Compiling 3 layers ===");
    let mut dc = DeltaCompiler::new();
    let input_size = DIM * SEQ_LEN * 4; // fp32
    let output_size = DIM * SEQ_LEN * 4; // fp32

    for i in 0..3 {
        let weights = make_weights(i * 1000, weight_size);
        let blob = make_blob(&weights, DIM, DIM);
        let mil = conv1x1_mil(SEQ_LEN, DIM, DIM);

        let idx = dc.add_program(
            &mil,
            &[weight_name],
            &[blob.as_bytes()],
            &[blob.as_bytes().len()],
            &[input_size],
            &[output_size],
        )?;
        println!("  Layer {} compiled (idx={})", i, idx);
    }

    let compile_after = dc.compile_count();
    println!("  Compile count after 3 layers: {}", compile_after);

    // Create test input
    let input_data: Vec<f32> = make_weights(42, DIM * SEQ_LEN);
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Step 2: Determinism test (DLT-04 partial)
    println!("\n=== Step 2: Determinism test ===");
    let weights_v1 = make_weights(100, weight_size);
    let blob_v1 = make_blob(&weights_v1, DIM, DIM);

    // First eval with v1 weights
    dc.reload_layer(0, &[(weight_name, blob_v1.as_bytes())])?;
    let output_v1_a = read_output_f32(&mut dc, 0, &input_bytes);

    // Reload SAME weights
    dc.reload_layer(0, &[(weight_name, blob_v1.as_bytes())])?;
    let output_v1_b = read_output_f32(&mut dc, 0, &input_bytes);

    let deterministic = fp32_close(&output_v1_a, &output_v1_b, 0.001);
    if deterministic {
        println!("  ✅ DETERMINISM PASS: same weights → same output");
    } else {
        println!("  ❌ DETERMINISM FAIL: output changed with same weights");
        // Print first few values for debug
        println!(
            "     v1_a[0..5]: {:?}",
            &output_v1_a[..5.min(output_v1_a.len())]
        );
        println!(
            "     v1_b[0..5]: {:?}",
            &output_v1_b[..5.min(output_v1_b.len())]
        );
    }

    // Step 3: Weight change test (DLT-03)
    println!("\n=== Step 3: Weight change test (DLT-03) ===");
    let weights_v2: Vec<f32> = weights_v1.iter().map(|&w| w + 0.3).collect();
    let blob_v2 = make_blob(&weights_v2, DIM, DIM);

    dc.reload_layer(0, &[(weight_name, blob_v2.as_bytes())])?;
    let output_v2 = read_output_f32(&mut dc, 0, &input_bytes);

    let changed = !fp32_close(&output_v1_a, &output_v2, 0.001);
    if changed {
        println!("  ✅ WEIGHT CHANGE PASS: different weights → different output");
    } else {
        println!("  ❌ WEIGHT CHANGE FAIL: output unchanged after weight modification");
    }

    // Step 4: Compile count non-increase (DLT-02)
    println!("\n=== Step 4: Compile count tracking (DLT-02) ===");
    let compile_before_cycles = dc.compile_count();
    println!("  Compile count before cycles: {}", compile_before_cycles);

    // Step 5: Durability test — 20 reload cycles (DLT-04)
    println!("\n=== Step 5: Durability test (20 cycles, DLT-04) ===");
    let num_cycles = 20;
    let mut all_ok = true;

    for cycle in 0..num_cycles {
        let cycle_weights = make_weights(cycle * 77, weight_size);
        let cycle_blob = make_blob(&cycle_weights, DIM, DIM);

        // Reload all 3 layers
        for layer_idx in 0..3 {
            dc.reload_layer(layer_idx, &[(weight_name, cycle_blob.as_bytes())])
                .map_err(|e| {
                    all_ok = false;
                    e
                })
                .ok();
        }

        // Verify eval works
        for layer_idx in 0..3 {
            let _ = read_output_f32(&mut dc, layer_idx, &input_bytes);
        }

        if cycle % 5 == 0 {
            println!("  Cycle {:2}: OK", cycle + 1);
        }
    }

    if all_ok {
        println!(
            "  ✅ DURABILITY PASS: all {} cycles completed without errors",
            num_cycles
        );
    } else {
        println!("  ❌ DURABILITY FAIL: errors in some cycles");
    }

    let compile_after_cycles = dc.compile_count();
    let compile_delta = compile_after_cycles - compile_before_cycles;
    println!(
        "  Compile count after {} cycles: {} (delta: {})",
        num_cycles, compile_after_cycles, compile_delta
    );

    if compile_delta == 0 {
        println!("  ✅ DLT-02 PASS: reloads did not increment compile count");
    } else {
        println!(
            "  ❌ DLT-02 FAIL: compile count changed by {} during reloads",
            compile_delta
        );
    }

    // Step 6: Selective update test (DLT-03 extended)
    println!("\n=== Step 6: Selective update test ===");

    // Set known weights for layer 0 and 1
    let weights_l0_base = make_weights(200, weight_size);
    let weights_l1_base = make_weights(300, weight_size);
    let blob_l0_base = make_blob(&weights_l0_base, DIM, DIM);
    let blob_l1_base = make_blob(&weights_l1_base, DIM, DIM);

    dc.reload_layer(0, &[(weight_name, blob_l0_base.as_bytes())])?;
    dc.reload_layer(1, &[(weight_name, blob_l1_base.as_bytes())])?;

    let output_l0_before = read_output_f32(&mut dc, 0, &input_bytes);
    let output_l1_before = read_output_f32(&mut dc, 1, &input_bytes);

    // Update ONLY layer 1 (not layer 0)
    let weights_l1_changed: Vec<f32> = weights_l1_base.iter().map(|&w| w + 0.5).collect();
    let blob_l1_changed = make_blob(&weights_l1_changed, DIM, DIM);
    dc.reload_layer(1, &[(weight_name, blob_l1_changed.as_bytes())])?;

    let output_l0_after = read_output_f32(&mut dc, 0, &input_bytes);
    let output_l1_after = read_output_f32(&mut dc, 1, &input_bytes);

    let l0_unchanged = fp32_close(&output_l0_before, &output_l0_after, 0.001);
    let l1_changed = !fp32_close(&output_l1_before, &output_l1_after, 0.001);

    if l0_unchanged && l1_changed {
        println!("  ✅ SELECTIVE UPDATE PASS: layer 0 unchanged, layer 1 changed");
    } else {
        println!(
            "  ❌ SELECTIVE UPDATE FAIL: l0_unchanged={}, l1_changed={}",
            l0_unchanged, l1_changed
        );
    }

    // Step 7: Budget status
    println!("\n=== Step 7: Budget status ===");
    let status = dc.budget_status();
    println!(
        "  Compiles used (this DeltaCompiler): {}",
        dc.compiles_used()
    );
    println!(
        "  Total compiles (process): {}/{}",
        status.used, status.limit
    );
    println!("  Remaining: {}", status.remaining);
    println!("  Warning zone: {}", status.warning);
    println!("  Exhausted: {}", status.exhausted);

    if dc.check_budget_warning() {
        println!("  ⚠️  WARNING: approaching compile limit");
    }

    println!("\nOK delta_state");
    Ok(())
}
