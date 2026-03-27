//! ANE MIL Syntax Discovery
//!
//! Systematically test different MIL syntax variations to find what ANE supports

use rustane::ane::ANECompileRequest;
use rustane::wrapper::ANERuntime;

fn test_compile(name: &str, mil: &str, inputs: Vec<usize>, outputs: Vec<usize>) -> bool {
    print!("{:30} ", name);
    let request = ANECompileRequest::new(mil, inputs, outputs);
    match request.compile() {
        Ok(_) => {
            println!("✅ PASS");
            true
        }
        Err(e) => {
            let err = e.to_string();
            if err.contains("InvalidMILProgram") {
                println!("❌ InvalidMILProgram");
            } else if err.contains("null") {
                println!("❌ Null kernel");
            } else {
                println!("❌ {}", &err[..err.len().min(40)]);
            }
            false
        }
    }
}

#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_syntax_variations() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║         ANE MIL SYNTAX DISCOVERY                           ║");
    println!("╚════════════════════════════════════════════════════════════\n");

    let _runtime = ANERuntime::init().expect("ANE init failed");

    // ========== BASIC STRUCTURE ==========
    println!("\n--- Basic Structure Variations ---");

    // Single line vs multi-line
    test_compile(
        "single_line",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = x; return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "multi_line",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
    let y = x;
    return (y);
}"#,
        vec![16],
        vec![16],
    );

    // Different main function names
    test_compile(
        "main_add_tensors",
        r#"main add_tensors(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = x; return (y); }"#,
        vec![16],
        vec![16],
    );

    // Different parameter names
    test_compile(
        "param_a_b",
        r#"main f(a: tensor<4xf32>, b: tensor<4xf32>) -> (c: tensor<4xf32>) { let c = a + b; return (c); }"#,
        vec![16, 16],
        vec![16],
    );

    // ========== TENSOR TYPES ==========
    println!("\n--- Tensor Type Variations ---");

    test_compile(
        "f32",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = x; return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "f16",
        r#"main f(x: tensor<4xf16>) -> (y: tensor<4xf16>) { let y = x; return (y); }"#,
        vec![8],
        vec![8],
    );

    test_compile(
        "int32",
        r#"main f(x: tensor<4xi32>) -> (y: tensor<4xi32>) { let y = x; return (y); }"#,
        vec![16],
        vec![16],
    );

    // ========== TENSOR SHAPES ==========
    println!("\n--- Tensor Shape Variations ---");

    test_compile(
        "shape_1d",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = x; return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "shape_2d",
        r#"main f(x: tensor<2x4xf32>) -> (y: tensor<2x4xf32>) { let y = x; return (y); }"#,
        vec![32],
        vec![32],
    );

    test_compile(
        "shape_3d",
        r#"main f(x: tensor<2x2x4xf32>) -> (y: tensor<2x2x4xf32>) { let y = x; return (y); }"#,
        vec![64],
        vec![64],
    );

    // ========== BASIC OPERATIONS ==========
    println!("\n--- Basic Operations ---");

    test_compile(
        "op_add",
        r#"main f(a: tensor<4xf32>, b: tensor<4xf32>) -> (c: tensor<4xf32>) { let c = a + b; return (c); }"#,
        vec![16, 16],
        vec![16],
    );

    test_compile(
        "op_mul",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = x * 2.0; return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "op_sub",
        r#"main f(a: tensor<4xf32>, b: tensor<4xf32>) -> (c: tensor<4xf32>) { let c = a - b; return (c); }"#,
        vec![16, 16],
        vec![16],
    );

    test_compile(
        "op_div",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = x / 2.0; return (y); }"#,
        vec![16],
        vec![16],
    );

    // ========== CONSTANTS ==========
    println!("\n--- Constants ---");

    test_compile(
        "const_float",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let two = 2.0; let y = x * two; return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "const_int",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let two = 2; let y = x * 2.0; return (y); }"#,
        vec![16],
        vec![16],
    );

    // ========== REDUCTION ==========
    println!("\n--- Reduction Operations ---");

    test_compile(
        "reduction_sum",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<1xf32>) { let y = sum(x); return (y); }"#,
        vec![16],
        vec![4],
    );

    test_compile(
        "reduction_mean",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<1xf32>) { let y = mean(x); return (y); }"#,
        vec![16],
        vec![4],
    );

    test_compile(
        "reduction_max",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<1xf32>) { let y = max(x); return (y); }"#,
        vec![16],
        vec![4],
    );

    // ========== ACTIVATIONS ==========
    println!("\n--- Activations ---");

    test_compile(
        "relu_builtin",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = relu(x); return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "relu_max",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let zero = 0.0; let y = max(x, zero); return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "sigmoid",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = sigmoid(x); return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "tanh",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = tanh(x); return (y); }"#,
        vec![16],
        vec![16],
    );

    test_compile(
        "softmax",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { let y = softmax(x); return (y); }"#,
        vec![16],
        vec![16],
    );

    // ========== MATRIX OPERATIONS ==========
    println!("\n--- Matrix Operations ---");

    test_compile(
        "matmul_2d",
        r#"main f(a: tensor<2x4xf32>, b: tensor<4x4xf32>) -> (c: tensor<2x4xf32>) { let c = matmul(a, b); return (c); }"#,
        vec![32, 64],
        vec![32],
    );

    test_compile(
        "transpose",
        r#"main f(x: tensor<2x4xf32>) -> (y: tensor<4x2xf32>) { let y = transpose(x); return (y); }"#,
        vec![32],
        vec![32],
    );

    test_compile(
        "reshape",
        r#"main f(x: tensor<8xf32>) -> (y: tensor<2x4xf32>) { let y = reshape(x, [2, 4]); return (y); }"#,
        vec![32],
        vec![32],
    );

    println!("\n════════════════════════════════════════════════════════════");
}

#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_real_sizes() {
    println!("\n--- Real Model Sizes ---\n");

    let _runtime = ANERuntime::init().expect("ANE init failed");

    // Test with sizes actually used in transformers
    let sizes = vec![64, 128, 256, 512, 768, 1024];

    for size in sizes {
        let mil = format!(
            r#"main f(x: tensor<{}xf32>) -> (y: tensor<{}xf32>) {{ let y = x + x; return (y); }}"#,
            size, size
        );
        test_compile(
            &format!("size_{}", size),
            &mil,
            vec![size * 4],
            vec![size * 4],
        );
    }
}

#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_complex_expressions() {
    println!("\n--- Complex Expressions ---\n");

    let _runtime = ANERuntime::init().expect("ANE init failed");

    // Multiple operations
    test_compile(
        "chain_ops",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { 
            let a = x + x; 
            let b = a * 2.0; 
            let y = b - x; 
            return (y); 
        }"#,
        vec![16],
        vec![16],
    );

    // With intermediate constants
    test_compile(
        "intermediate_const",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { 
            let scale = 0.5; 
            let bias = 1.0; 
            let y = x * scale + bias; 
            return (y); 
        }"#,
        vec![16],
        vec![16],
    );

    // Layer norm simplified
    test_compile(
        "layernorm_simple",
        r#"main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) { 
            let mean = mean(x); 
            let diff = x - mean; 
            let y = diff * diff; 
            return (y); 
        }"#,
        vec![16],
        vec![16],
    );
}
