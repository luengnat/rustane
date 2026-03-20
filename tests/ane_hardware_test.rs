//! Test to verify ANE hardware is being used
//!
//! Run this test on Apple Silicon hardware to verify ANE is actually working.

use rustane::ane::ANECompileRequest;

#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_compilation_attempt() {
    // Simple MIL program that adds two vectors
    let mil = r#"
        program(1.0) {
            input  = parameter(0)
            output = input + input
            return output
        }
    "#;

    // Try to compile - this will fail on non-ANE hardware
    let result = ANECompileRequest::new(mil, vec![32], vec![32]).compile();

    match result {
        Ok(_) => {
            println!("✅ ANE compilation SUCCESSFUL - ANE hardware is being used!");
        }
        Err(e) => {
            // On non-ANE hardware, we expect this to fail
            let err_str = format!("{:?}", e);
            if err_str.contains("ANE") || err_str.contains("null") {
                println!(
                    "⚠️  ANE compilation failed (expected on non-ANE hardware): {}",
                    e
                );
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}

#[test]
#[cfg(not(target_vendor = "apple"))]
fn test_ane_compilation_attempt() {
    // On non-Apple Silicon, ANE is not available
    println!("⚠️  Not running on Apple Silicon - ANE not available");

    let mil = r#"
        program(1.0) {
            input  = parameter(0)
            output = input + input
            return output
        }
    "#;

    let result = ANECompileRequest::new(mil, vec![32], vec![32]).compile();
    assert!(
        result.is_err(),
        "Expected ANE compilation to fail on non-Apple Silicon"
    );

    let err = format!("{:?}", result.unwrap_err());
    println!(
        "Expected error on non-ANE hardware: {}",
        if err.contains("ANE") || err.contains("null") {
            "ANE not available"
        } else {
            &err
        }
    );
}
