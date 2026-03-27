//! ANE Debugging Test
//!
//! Tests ANE compilation and execution to identify failure points

use rustane::ane::{ANECompileRequest, ANEError, ErrorDiagnostic};
use rustane::wrapper::ANERuntime;
use rustane::Result;

/// Test ANE with progressively complex operations
#[test]
fn test_ane_simple_ops() {
    println!("\n=== ANE Simple Operations Test ===\n");

    // Initialize ANE
    let runtime = match ANERuntime::init() {
        Ok(rt) => {
            println!("✅ ANE Runtime initialized");
            rt
        }
        Err(e) => {
            println!("❌ ANE init failed: {}", e);
            return;
        }
    };

    let initial_count = runtime.compile_count();
    println!("Initial compile count: {}", initial_count);

    // Test 1: Element-wise multiply
    println!("\nTest 1: Element-wise multiply");
    let mil1 = r#"
    main mul(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
        let two = 2.0;
        let y = x * two;
        return (y);
    }
    "#;

    test_mil_operation(
        "multiply",
        mil1,
        vec![16],
        vec![16],
        vec![1.0, 2.0, 3.0, 4.0],
    );

    // Test 2: Element-wise add
    println!("\nTest 2: Element-wise add");
    let mil2 = r#"
    main add(a: tensor<4xf32>, b: tensor<4xf32>) -> (c: tensor<4xf32>) {
        let c = a + b;
        return (c);
    }
    "#;

    test_mil_operation_2input(
        "add",
        mil2,
        vec![16, 16],
        vec![16],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![10.0, 20.0, 30.0, 40.0],
    );

    // Test 3: Matmul (what transformer uses)
    println!("\nTest 3: Matrix multiplication");
    let mil3 = r#"
    main matmul(x: tensor<2x4xf32>) -> (y: tensor<2x4xf32>) {
        let w = const_tensor<4x4xf32>(@model_path/w.bin);
        let y = matmul(x, w);
        return (y);
    }
    "#;

    let weights: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let weight_bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_ne_bytes()).collect();

    test_mil_with_weights(
        "matmul",
        mil3,
        vec![32],
        vec![32],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "@model_path/w.bin",
        weight_bytes,
    );

    println!("\nFinal compile count: {}", runtime.compile_count());
}

fn test_mil_operation(
    name: &str,
    mil: &str,
    input_sizes: Vec<usize>,
    output_size: Vec<usize>,
    test_input: Vec<f32>,
) {
    let request = ANECompileRequest::new(mil, input_sizes, output_size);

    match request.compile() {
        Ok(mut exec) => {
            println!("  ✅ {} compiled", name);

            let input_bytes: Vec<u8> = test_input.iter().flat_map(|f| f.to_ne_bytes()).collect();

            if let Err(e) = exec.write_input(0, &input_bytes) {
                println!("  ❌ {} write input failed: {}", name, e);
                return;
            }

            if let Err(e) = exec.eval() {
                println!("  ❌ {} eval failed: {}", name, e);
                diagnose_error(&e);
                return;
            }

            let mut output_bytes = vec![0u8; test_input.len() * 4];
            if let Err(e) = exec.read_output(0, &mut output_bytes) {
                println!("  ❌ {} read output failed: {}", name, e);
                return;
            }

            let output: Vec<f32> = output_bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            println!("  ✅ {} executed: {:?} -> {:?}", name, test_input, output);
        }
        Err(e) => {
            println!("  ❌ {} compilation failed: {}", name, e);
            diagnose_error(&e);
        }
    }
}

fn test_mil_operation_2input(
    name: &str,
    mil: &str,
    input_sizes: Vec<usize>,
    output_size: Vec<usize>,
    input1: Vec<f32>,
    input2: Vec<f32>,
) {
    let request = ANECompileRequest::new(mil, input_sizes, output_size);

    match request.compile() {
        Ok(mut exec) => {
            println!("  ✅ {} compiled", name);

            let bytes1: Vec<u8> = input1.iter().flat_map(|f| f.to_ne_bytes()).collect();
            let bytes2: Vec<u8> = input2.iter().flat_map(|f| f.to_ne_bytes()).collect();

            if let Err(e) = exec.write_input(0, &bytes1) {
                println!("  ❌ {} write input 0 failed: {}", name, e);
                return;
            }
            if let Err(e) = exec.write_input(1, &bytes2) {
                println!("  ❌ {} write input 1 failed: {}", name, e);
                return;
            }

            if let Err(e) = exec.eval() {
                println!("  ❌ {} eval failed: {}", name, e);
                diagnose_error(&e);
                return;
            }

            let mut output_bytes = vec![0u8; input1.len() * 4];
            if let Err(e) = exec.read_output(0, &mut output_bytes) {
                println!("  ❌ {} read output failed: {}", name, e);
                return;
            }

            let output: Vec<f32> = output_bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            println!(
                "  ✅ {} executed\n     {:?} + {:?} = {:?}",
                name, input1, input2, output
            );
        }
        Err(e) => {
            println!("  ❌ {} compilation failed: {}", name, e);
            diagnose_error(&e);
        }
    }
}

fn test_mil_with_weights(
    name: &str,
    mil: &str,
    input_size: Vec<usize>,
    output_size: Vec<usize>,
    test_input: Vec<f32>,
    weight_path: &str,
    weight_bytes: Vec<u8>,
) {
    let request = ANECompileRequest::new(mil, input_size, output_size)
        .with_weight_bytes(weight_path, weight_bytes);

    match request.compile() {
        Ok(mut exec) => {
            println!("  ✅ {} compiled", name);

            let input_bytes: Vec<u8> = test_input.iter().flat_map(|f| f.to_ne_bytes()).collect();

            if let Err(e) = exec.write_input(0, &input_bytes) {
                println!("  ❌ {} write input failed: {}", name, e);
                return;
            }

            if let Err(e) = exec.eval() {
                println!("  ❌ {} eval failed: {}", name, e);
                diagnose_error(&e);
                return;
            }

            let mut output_bytes = vec![0u8; test_input.len() * 4];
            if let Err(e) = exec.read_output(0, &mut output_bytes) {
                println!("  ❌ {} read output failed: {}", name, e);
                return;
            }

            let output: Vec<f32> = output_bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            println!("  ✅ {} executed: output = {:?}", name, output);
        }
        Err(e) => {
            println!("  ❌ {} compilation failed: {}", name, e);
            diagnose_error(&e);
        }
    }
}

fn diagnose_error(error: &dyn std::error::Error) {
    let err_str = error.to_string();

    if err_str.contains("ane_bridge") || err_str.contains("ANE") {
        println!("     ANE bridge error detected");
    }

    if err_str.contains("compile") {
        println!("     → MIL compilation error");
    } else if err_str.contains("eval") {
        println!("     → ANE execution error");
    } else if err_str.contains("IOSurface") {
        println!("     → Memory/IOSurface error");
    }
}

/// Test what operations are failing in transformer
#[test]
fn test_ane_transformer_operations() {
    println!("\n=== ANE Transformer Operations Test ===\n");

    // These are the operations used in transformer forward pass
    let operations = vec![
        (
            "rmsnorm",
            r#"
        main rmsnorm(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
            let mean_sq = mean(x * x);
            let rms = sqrt(mean_sq + 1e-6);
            let y = x / rms;
            return (y);
        }
        "#,
        ),
        (
            "softmax",
            r#"
        main softmax(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
            let max_val = max(x);
            let exp_x = exp(x - max_val);
            let sum_exp = sum(exp_x);
            let y = exp_x / sum_exp;
            return (y);
        }
        "#,
        ),
        (
            "gelu",
            r#"
        main gelu(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
            let sqrt_2_over_pi = 0.7978845608;
            let y = 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
            return (y);
        }
        "#,
        ),
    ];

    for (name, mil) in operations {
        println!("Testing {}...", name);
        let request = ANECompileRequest::new(mil, vec![16], vec![16]);

        match request.compile() {
            Ok(_) => println!("  ✅ {} compiles", name),
            Err(e) => {
                println!("  ❌ {} failed: {}", name, e);
            }
        }
    }
}
