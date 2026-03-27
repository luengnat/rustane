//! Minimal ANE Test Example
//!
//! This example tests basic ANE functionality on Apple Silicon.
//! It compiles a simple MIL program and executes it on the ANE.
//!
//! Run with:
//! ```bash
//! cargo run --example ane_minimal_test
//! ```

use rustane::ane::ANECompileRequest;
use rustane::wrapper::ANERuntime;
use rustane::Result;

fn main() -> Result<()> {
    println!("ANE Minimal Test Example");
    println!("========================\n");

    // Check if we're on Apple Silicon
    #[cfg(not(target_vendor = "apple"))]
    {
        println!("This example requires Apple Silicon hardware.");
        println!("Current platform does not support ANE.");
        return Ok(());
    }

    #[cfg(target_vendor = "apple")]
    {
        // Try to initialize ANE runtime
        println!("Step 1: Initializing ANE runtime...");
        let runtime = match ANERuntime::init() {
            Ok(rt) => {
                println!("  ✓ ANE runtime initialized successfully");
                rt
            }
            Err(e) => {
                println!("  ✗ Failed to initialize ANE runtime: {}", e);
                println!("\nNote: This example requires:");
                println!("  - Apple Silicon Mac (M1/M2/M3/M4)");
                println!("  - macOS with ANE framework");
                return Ok(());
            }
        };

        // Get initial compile count
        let initial_count = runtime.compile_count();
        println!("  Initial compile count: {}", initial_count);

        // Define simple MIL program
        let mil_code = r#"
        main add_tensors(a: tensor<4xf32>, b: tensor<4xf32>) -> (c: tensor<4xf32>) {
            let c = a + b;
            return (c);
        }
        "#;

        // Compile the program
        println!("\nStep 2: Compiling MIL program...");
        let request = ANECompileRequest::new(mil_code, vec![16, 16], vec![16]);

        let mut executor = match request.compile() {
            Ok(exec) => {
                println!("  ✓ Compilation successful");
                println!(
                    "  Compile count: {} -> {}",
                    initial_count,
                    runtime.compile_count()
                );
                exec
            }
            Err(e) => {
                println!("  ✗ Compilation failed: {}", e);
                let err_str = e.to_string();
                if err_str.contains("ANE") || err_str.contains("framework") {
                    println!("\n  ANE-related error detected.");
                    println!("  This might indicate:");
                    println!("    - Not running on Apple Silicon");
                    println!("    - ANE framework not available");
                }
                return Ok(());
            }
        };

        // Prepare test inputs
        println!("\nStep 3: Preparing test data...");
        let input_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

        let input_a_bytes: Vec<u8> = input_a.iter().flat_map(|&f| f.to_ne_bytes()).collect();
        let input_b_bytes: Vec<u8> = input_b.iter().flat_map(|&f| f.to_ne_bytes()).collect();

        println!("  Input A: {:?}", input_a);
        println!("  Input B: {:?}", input_b);

        // Write inputs to ANE
        println!("\nStep 4: Writing inputs to ANE...");
        executor.write_input(0, &input_a_bytes)?;
        executor.write_input(1, &input_b_bytes)?;
        println!("  ✓ Inputs written successfully");

        // Execute
        println!("\nStep 5: Executing on ANE...");
        executor.eval()?;
        println!("  ✓ Execution completed");

        // Read output
        println!("\nStep 6: Reading output...");
        let mut output_bytes = vec![0u8; 16];
        executor.read_output(0, &mut output_bytes)?;

        let output: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        println!("  Output: {:?}", output);

        // Verify results
        println!("\nStep 7: Verifying results...");
        let expected: Vec<f32> = input_a
            .iter()
            .zip(input_b.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut all_correct = true;
        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            if diff > 1e-5 {
                println!(
                    "  ✗ Output[{}]: expected {}, got {} (diff: {})",
                    i, exp, actual, diff
                );
                all_correct = false;
            }
        }

        if all_correct {
            println!("  ✓ All outputs correct!");
            println!("\n  Expected: {:?}", expected);
            println!("  Got:      {:?}", output);
        }

        // Summary
        println!("\n========================");
        println!("ANE Test Summary:");
        println!("  - Runtime initialized: ✓");
        println!("  - Compilation: ✓");
        println!("  - Execution: ✓");
        println!(
            "  - Results verified: {}",
            if all_correct { "✓" } else { "✗" }
        );
        println!("  - Total compiles: {}", runtime.compile_count());

        if all_correct {
            println!("\n✅ ANE integration test PASSED!");
        } else {
            println!("\n❌ ANE integration test FAILED!");
        }
    }

    Ok(())
}
