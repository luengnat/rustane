//! ANE Correct Syntax Test - Using ANE Project Format
//!
//! The ANE project uses a specific MIL format:
//! func main<ios18>(tensor<fp32, [1, C, 1, S]> x) { ... }

use rustane::ane::ANECompileRequest;
use rustane::wrapper::ANERuntime;

#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_project_syntax() {
    println!("\n=== ANE Project Syntax Test ===\n");

    let _runtime = ANERuntime::init().expect("ANE init failed");

    // Test 1: Simple identity with ANE format
    let mil1 = r#"func main<ios18>(tensor<fp32, [1, 4, 1, 1]> x) {
    return x
}"#;
    test_compile("ane_identity", mil1, vec![16], vec![16]);

    // Test 2: Matmul with ANE format
    let mil2 = r#"func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {
    let w = const_tensor<fp32, [1, 4, 1, 4]>(@w.bin)
    return x * w
}"#;
    test_compile("ane_matmul", mil2, vec![64], vec![64]);

    // Test 3: Add operation
    let mil3 = r#"func main<ios18>(tensor<fp32, [1, 4, 1, 1]> a, tensor<fp32, [1, 4, 1, 1]> b) {
    return a + b
}"#;
    test_compile("ane_add", mil3, vec![16, 16], vec![16]);

    // Test 4: ReLU
    let mil4 = r#"func main<ios18>(tensor<fp32, [1, 4, 1, 1]> x) {
    return max(x, 0.0)
}"#;
    test_compile("ane_relu", mil4, vec![16], vec![16]);

    // Test 5: Larger size (like real model)
    let mil5 = r#"func main<ios18>(tensor<fp32, [1, 512, 1, 1]> x) {
    return x
}"#;
    test_compile("ane_512", mil5, vec![2048], vec![2048]);

    println!("\n═══════════════════════════════════════");
}

fn test_compile(name: &str, mil: &str, inputs: Vec<usize>, outputs: Vec<usize>) {
    println!("Testing: {}", name);
    println!("  MIL: {}", mil.lines().next().unwrap_or(mil));

    let request = ANECompileRequest::new(mil, inputs, outputs);
    match request.compile() {
        Ok(_) => println!("  ✅ SUCCESS\n"),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("InvalidMILProgram") {
                println!("  ❌ InvalidMILProgram\n");
            } else {
                println!("  ❌ {}\n", &msg[..msg.len().min(50)]);
            }
        }
    }
}
